"""
Calcite v2 Surprise-Gated UTF-8 Byte Chunker (`calcite_v2/chunker.py`).

Improvements over `simvq_split/chunker.py`:
1. Zero monkey-patching: calls `utf8_span_and_compact_cuda` directly when on CUDA
   (with exact PyTorch fallback on CPU).
2. Scaled intra-chunk byte decoder (`use_scaled_decoder=True`):
   - Explicit causal `byte_offset_emb` (`0..max_utf8_bytes`) so the decoder knows the
     exact byte offset inside the current chunk / UTF-8 character.
   - Lightweight causal 4-byte gated depthwise/pointwise residual refinement (`O(1)` 3-byte ring).
   - Proportional `byte_d_ff` scaling with `d_model` instead of frozen `128`.
3. Standalone router pretraining interface (`router_forward_only`) so the `~150K` 2-bit router
   can be trained and frozen BEFORE step 0 of backbone training.
4. Exact incremental streaming API (`init_stream_state`, `encode_byte_step`, `decode_byte_step`)
   matching batch `encode_to_chunks` / `decode_from_chunks`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels.fast_calcite_swa_chunker import (
    utf8_span_and_compact_cuda,
)
from simvq_split.baselines import RMSNorm, SwiGLU
from simvq_split.chunker import (
    _enforce_utf8_chunk_span_jit,
    _utf8_tensor_flags,
    enforce_max_chunk_span_vec,
)
from simvq_split.quant_lut import PackedLUT2BitLinear, quantize_ternary_ste, quantize_ternary_weights_ls


@dataclass
class ChunkContext:
    b_mask: torch.Tensor
    valid_mask: torch.Tensor
    p_prob: torch.Tensor
    p_chunk: torch.Tensor
    h_byte: torch.Tensor
    c_win8: torch.Tensor
    slot_idx: torch.Tensor
    offset_in_chunk: torch.Tensor
    valid_chunk: torch.Tensor


@dataclass
class ChunkerStreamState:
    ring_char: torch.Tensor        # (B, 7, d_byte)
    ring_rchar: torch.Tensor       # (B, 3, d_rchar)
    prev_k_r: torch.Tensor         # (B, router_dim)
    prev_h_ent: torch.Tensor       # (B,)
    prev_z_hat: torch.Tensor       # (B, d_model)
    cur_z_bar: torch.Tensor        # (B, d_model)
    ring_dec_conv1: torch.Tensor   # (B, 3, d_model) if scaled decoder else (B, 0, 1)
    ring_dec_conv2: torch.Tensor   # (B, 6, d_model) if scaled decoder else (B, 0, 1)
    run_len: torch.Tensor          # (B,) int64
    cp_run: torch.Tensor           # (B,) int64
    cont_streak: torch.Tensor      # (B,) int64
    credits: torch.Tensor          # (B,) int64
    step_idx: int
    has_chunk: bool

    def memory_bytes(self) -> int:
        b = max(1, int(self.ring_char.size(0)))
        total = 0
        for t in (
            self.ring_char,
            self.ring_rchar,
            self.prev_k_r,
            self.prev_h_ent,
            self.prev_z_hat,
            self.cur_z_bar,
            self.ring_dec_conv1,
            self.ring_dec_conv2,
            self.run_len,
            self.cp_run,
            self.cont_streak,
            self.credits,
        ):
            total += int(t.numel() * t.element_size())
        return total // b


class CalciteV2Chunker(nn.Module):
    def __init__(
        self,
        vocab_size: int = 259,
        d_model: int = 256,
        d_byte: int = 32,
        d_rchar: int = 16,
        r_hidden: int = 96,
        byte_d_ff: int = 256,
        target_chunk_bytes: float = 4.6,
        max_chunk_bytes: int = 10,
        pad_token_id: int = 0,
        byte_chunk_mode: str = "utf8",
        max_utf8_bytes: int = 16,
        use_static_m_cap: bool = True,
        use_scaled_decoder: bool = True,
        use_fast_kernels: bool = True,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.d_byte = int(min(d_byte, d_model))
        self.d_rchar = int(d_rchar)
        self.r_hidden = int(r_hidden)
        self.router_dim = 32
        self.target_chunk_bytes = float(target_chunk_bytes)
        self.max_chunk_bytes = int(max_chunk_bytes)
        self.pad_token_id = int(pad_token_id)
        self.byte_chunk_mode = str(byte_chunk_mode)
        self.max_utf8_bytes = int(max(4, max_utf8_bytes))
        self.use_static_m_cap = bool(use_static_m_cap)
        self.use_scaled_decoder = bool(use_scaled_decoder)
        self.use_fast_kernels = bool(use_fast_kernels)

        self.refill_int = 100
        self.cost_int = max(100, int(round(100.0 * 0.72 * self.target_chunk_bytes)))
        self.burst_int = 4 * self.cost_int

        # 8-byte causal encoder & lexical pathway
        self.enc_norm_in = RMSNorm(8 * self.d_byte)
        self.enc_up = nn.Linear(8 * self.d_byte, d_model, bias=False)
        self.enc_norm = RMSNorm(d_model)
        self.enc_mlp = SwiGLU(d_model, byte_d_ff)
        self.lex_norm = RMSNorm(8 * self.d_byte)
        self.lex_up = nn.Linear(8 * self.d_byte, d_model, bias=False)

        # Decoupled ternary-stable UTF-8 boundary router
        self.router_emb = nn.Embedding(self.vocab_size, self.d_rchar)
        nn.init.normal_(self.router_emb.weight, std=0.05)
        self.router_in_norm = RMSNorm(4 * self.d_rchar)
        self.router_in_proj = nn.Linear(4 * self.d_rchar, self.r_hidden, bias=True)
        self.router_q = nn.Linear(self.r_hidden, self.router_dim, bias=False)
        self.router_k = nn.Linear(self.r_hidden, self.router_dim, bias=False)
        self.router_ctx = nn.Linear(self.r_hidden, 1, bias=True)
        self.router_ent_norm = RMSNorm(self.r_hidden)
        self.router_ent_head = nn.Linear(self.r_hidden, self.vocab_size, bias=False)

        init_b = -math.log(max(1.0, self.target_chunk_bytes - 1.0)) - 0.65
        nn.init.constant_(self.router_ctx.bias, init_b)
        self.router_feat_weights = nn.Parameter(
            torch.tensor([1.15, 0.45, 0.75], dtype=torch.float32)
        )
        self.router_locked: bool = False

        # Byte decoder pathway
        self.dechunk_norm = RMSNorm(d_model)
        if self.use_scaled_decoder:
            self.byte_offset_emb = nn.Embedding(self.max_utf8_bytes + 1, d_model)
            nn.init.normal_(self.byte_offset_emb.weight, std=0.02)
            # 2-scale causal gated byte-mixer (`k=4, d=1` + `k=4, d=2`, 10-byte RF, O(1) 9-byte ring)
            self.dec_dw_norm = RMSNorm(d_model)
            self.dec_dw_conv = nn.Conv1d(
                d_model, d_model, kernel_size=4, groups=d_model, bias=True
            )
            nn.init.normal_(self.dec_dw_conv.weight, std=0.02)
            nn.init.zeros_(self.dec_dw_conv.bias)
            self.dec_dw_gate = nn.Linear(d_model, d_model, bias=False)
            self.dec_dw_out = nn.Linear(d_model, d_model, bias=False)

            self.dec_dw2_norm = RMSNorm(d_model)
            self.dec_dw2_conv = nn.Conv1d(
                d_model, d_model, kernel_size=4, dilation=2, groups=d_model, bias=True
            )
            nn.init.normal_(self.dec_dw2_conv.weight, std=0.02)
            nn.init.zeros_(self.dec_dw2_conv.bias)
            self.dec_dw2_gate = nn.Linear(d_model, d_model, bias=False)
            self.dec_dw2_out = nn.Linear(d_model, d_model, bias=False)
        else:
            self.byte_offset_emb = None
            self.dec_dw_conv = None
            self.dec_dw_norm = None
            self.dec_dw_gate = None
            self.dec_dw_out = None
            self.dec_dw2_norm = None
            self.dec_dw2_conv = None
            self.dec_dw2_gate = None
            self.dec_dw2_out = None

        self.dec_fuse_norm = RMSNorm(d_model + 8 * self.d_byte)
        self.dec_fuse = nn.Linear(d_model + 8 * self.d_byte, d_model, bias=False)
        self.dec_norm = RMSNorm(d_model)
        self.dec_mlp = SwiGLU(d_model, byte_d_ff)

        self.last_ratio_loss: torch.Tensor | None = None
        self._last_chunk_ratio_val: float = 1.0
        self._last_chunk_ratio_t: torch.Tensor | None = None
        self.last_m_cap_used: int = 0

    @property
    def last_chunk_ratio(self) -> float:
        if self._last_chunk_ratio_t is not None:
            self._last_chunk_ratio_val = float(self._last_chunk_ratio_t.item())
            self._last_chunk_ratio_t = None
        return self._last_chunk_ratio_val

    def static_m_cap_for_T(self, T: int) -> int:
        if T <= 1:
            return max(1, T)
        if self.byte_chunk_mode != "utf8":
            return T
        k_paid = min(
            T - 1,
            (self.burst_int + self.refill_int * (T - 1)) // self.cost_int,
        )
        s_min = min(self.max_chunk_bytes, max(1, self.max_utf8_bytes - 3))
        m_cap = 1 + k_paid + ((T - 1 - k_paid) // s_min)
        return min(T, m_cap)

    def _r_lin(self, lin: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if isinstance(lin, nn.Linear):
            if self.router_locked:
                b = lin.bias.to(x.dtype) if lin.bias is not None else None
                return F.linear(x, lin.weight.to(x.dtype), b)
            wq = quantize_ternary_ste(lin.weight).to(x.dtype)
            b = lin.bias.to(x.dtype) if lin.bias is not None else None
            return F.linear(x, wq, b)
        return lin(x)

    def _router_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.router_emb,
            self.router_in_norm,
            self.router_in_proj,
            self.router_q,
            self.router_k,
            self.router_ctx,
            self.router_ent_norm,
            self.router_ent_head,
        )

    def is_locked_router_module(self, mod: nn.Module) -> bool:
        if not self.router_locked:
            return False
        return any(
            mod is rmod
            for rmod in (
                self.router_in_proj,
                self.router_q,
                self.router_k,
                self.router_ctx,
                self.router_ent_head,
            )
        )

    def lock_router(self, locked: bool = True) -> None:
        flag = bool(locked)
        router_mods = self._router_modules()
        if flag and not self.router_locked:
            with torch.no_grad():
                for mod in router_mods:
                    if isinstance(mod, nn.Linear):
                        _, trits_i8, scale_1d = quantize_ternary_weights_ls(mod.weight.data)
                        scale_fp16 = scale_1d.to(torch.float16).float().unsqueeze(-1)
                        w_aligned = (
                            (trits_i8.float() * scale_fp16)
                            .reshape(mod.weight.shape)
                            .to(mod.weight.dtype)
                        )
                        mod.weight.data.copy_(w_aligned)
                        if mod.bias is not None:
                            mod.bias.data.copy_(
                                mod.bias.data.to(torch.float16).to(mod.bias.dtype)
                            )
        for mod in router_mods:
            for p in mod.parameters():
                p.requires_grad_(not flag)
        self.router_feat_weights.requires_grad_(not flag)
        self.router_locked = flag

    def compute_state_bytes(self) -> int:
        base = (
            7 * self.d_byte * 2
            + 3 * self.d_rchar * 2
            + self.router_dim * 2
            + 4
            + self.d_model * 4
            + self.d_model * 4  # cur_z_bar
            + 4
        )
        if self.byte_chunk_mode == "utf8":
            base += 3 * 4
        if self.use_scaled_decoder:
            base += 9 * self.d_model * 2  # causal 2-scale dw conv rings (3 + 6)
        return int(base)

    def _causal_win8(self, e_tok: torch.Tensor) -> torch.Tensor:
        c_char = e_tok[..., : self.d_byte]
        B, T, _ = c_char.shape
        c_pad = F.pad(c_char, (0, 0, 7, 0))
        shifts = [c_pad[:, 7 - k : 7 - k + T, :] for k in range(8)]
        return torch.cat(shifts, dim=-1).view(B, T, 8 * self.d_byte)

    def _router_probs(
        self, token_ids: torch.Tensor, work_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = token_ids.shape
        log_v = math.log(max(2, self.vocab_size))
        rc = self.router_emb(token_ids).to(work_dtype)
        rc_pad = F.pad(rc, (0, 0, 3, 0))
        r_win4 = torch.cat(
            [rc_pad[:, 3 - k : 3 - k + T, :] for k in range(4)], dim=-1
        ).view(B, T, 4 * self.d_rchar)
        h_r = F.silu(self._r_lin(self.router_in_proj, self.router_in_norm(r_win4)))

        q_r = F.normalize(self._r_lin(self.router_q, h_r).float(), dim=-1)
        k_r = F.normalize(self._r_lin(self.router_k, h_r).float(), dim=-1)
        d_cos = torch.ones(B, T, device=token_ids.device, dtype=torch.float32)
        if T > 1:
            d_cos[:, 1:] = 0.5 * (1.0 - (q_r[:, 1:] * k_r[:, :-1]).sum(dim=-1))

        loc_logits = self._r_lin(self.router_ent_head, self.router_ent_norm(h_r)).float()
        log_p = F.log_softmax(loc_logits, dim=-1)
        h_ent_norm = (-(log_p.exp() * log_p).sum(dim=-1) / log_v).clamp(0.0, 1.5)
        dh_pos = torch.zeros_like(h_ent_norm)
        if T > 1:
            dh_pos[:, 1:] = F.relu(h_ent_norm[:, 1:] - h_ent_norm[:, :-1])

        wf = self.router_feat_weights.float()
        ctx_logit = self._r_lin(self.router_ctx, h_r).float().squeeze(-1)
        p_prob = torch.sigmoid(
            wf[0] * d_cos + wf[1] * h_ent_norm + wf[2] * dh_pos + ctx_logit
        )
        return p_prob, loc_logits

    def router_forward_only(
        self, token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        """Standalone training loss for pre-freezing the UTF-8 router before step 0."""
        B, T = token_ids.shape
        work_dtype = self.router_emb.weight.dtype
        p_prob, loc_logits = self._router_probs(token_ids, work_dtype)
        valid_mask = token_ids != self.pad_token_id
        valid_mask[:, 0] = True

        _, is_cont, char_len = _utf8_tensor_flags(token_ids)
        is_cont_v = is_cont & valid_mask
        c1 = F.pad(is_cont_v[:, :-1], (1, 0), value=False) if T > 1 else torch.zeros_like(is_cont_v)
        c2 = F.pad(is_cont_v[:, :-2], (2, 0), value=False) if T > 2 else torch.zeros_like(is_cont_v)
        c3 = F.pad(is_cont_v[:, :-3], (3, 0), value=False) if T > 3 else torch.zeros_like(is_cont_v)
        can_fire = valid_mask & (~(is_cont_v & (~(c1 & c2 & c3))))
        is_lead = (valid_mask & (~is_cont)).long()
        raw_cand = (p_prob >= 0.5) & can_fire
        b_mask = _enforce_utf8_chunk_span_jit(
            raw_cand,
            valid_mask,
            can_fire,
            is_lead,
            char_len,
            self.max_chunk_bytes,
            self.max_utf8_bytes,
            self.refill_int,
            self.cost_int,
            self.burst_int,
        )

        ratio_mask = (valid_mask[:, 1:] & (~is_cont[:, 1:])).float()
        denom_r = ratio_mask.sum().clamp(min=1.0)
        f_hard = ((b_mask[:, 1:].float() * ratio_mask).sum() / denom_r).detach()
        g_soft = (p_prob[:, 1:] * ratio_mask).sum() / denom_r
        n_comp = max(1.5, float(self.target_chunk_bytes))
        l_ratio = (n_comp / (n_comp - 1.0)) * (
            (n_comp - 1.0) * f_hard * g_soft + (1.0 - f_hard) * (1.0 - g_soft)
        )
        ent_mask_f = valid_mask[:, 1:].float()
        ent_per_tok = F.cross_entropy(
            loc_logits[:, :-1].reshape(-1, self.vocab_size),
            token_ids[:, 1:].reshape(-1),
            reduction="none",
        ).view(B, T - 1)
        l_ent = (ent_per_tok * ent_mask_f).sum() / ent_mask_f.sum().clamp(min=1.0)
        chunk_ratio = float((1.0 / f_hard.clamp(min=1e-4)).item())
        return l_ratio + 0.35 * l_ent, chunk_ratio

    @staticmethod
    def _compute_offset_in_chunk(b_mask: torch.Tensor, max_offset: int) -> torch.Tensor:
        """Strictly causal byte offset `0..max_offset` since the most recent boundary fire."""
        B, T = b_mask.shape
        t_idx = torch.arange(T, device=b_mask.device, dtype=torch.long).unsqueeze(0).expand(B, T)
        fire_pos = torch.where(b_mask, t_idx, torch.zeros_like(t_idx))
        last_fire, _ = torch.cummax(fire_pos, dim=1)
        return (t_idx - last_fire).clamp(0, max_offset)

    def encode_to_chunks(
        self,
        e_tok: torch.Tensor,
        token_ids: torch.Tensor,
        tok_emb_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, ChunkContext]:
        del tok_emb_weight
        B, T, D = e_tok.shape

        c_win8 = self._causal_win8(e_tok)
        c_norm = self.enc_norm_in(c_win8)
        h_enc0 = e_tok + self.enc_up(c_norm)
        h_byte = h_enc0 + self.enc_mlp(self.enc_norm(h_enc0))

        if self.router_locked:
            with torch.no_grad():
                p_prob, loc_logits = self._router_probs(token_ids, e_tok.dtype)
            p_prob = p_prob.detach()
        else:
            p_prob, loc_logits = self._router_probs(token_ids, e_tok.dtype)

        valid_mask = token_ids != self.pad_token_id
        valid_mask[:, 0] = True

        use_cuda_compact = (
            self.use_fast_kernels
            and self.byte_chunk_mode == "utf8"
            and token_ids.is_cuda
        )
        if use_cuda_compact:
            m_cap_req = self.static_m_cap_for_T(T) if self.use_static_m_cap else T
            b_mask, chunk_counts, slot_idx, gather_pos = utf8_span_and_compact_cuda(
                token_ids,
                p_prob,
                self.pad_token_id,
                self.max_chunk_bytes,
                self.max_utf8_bytes,
                self.refill_int,
                self.cost_int,
                self.burst_int,
                m_cap_req,
            )
            if self.use_static_m_cap:
                M_max = m_cap_req
            else:
                raw_m = int(chunk_counts.max().item())
                M_max = min(T, ((raw_m + 31) // 32) * 32) if T >= 32 else max(1, raw_m)
                gather_pos = gather_pos[:, :M_max]
                slot_idx = slot_idx.clamp(max=max(0, M_max - 1))

            if T > 1:
                if not self.router_locked:
                    _, is_cont, _ = _utf8_tensor_flags(token_ids)
                    ratio_mask_f = (valid_mask[:, 1:] & (~is_cont[:, 1:])).float()
                    r_sum = ratio_mask_f.sum()
                    denom_r = r_sum.clamp(min=1.0)
                    f_hard = ((b_mask[:, 1:].float() * ratio_mask_f).sum() / denom_r).detach()
                    g_soft = (p_prob[:, 1:] * ratio_mask_f).sum() / denom_r
                    n_comp = max(1.5, float(self.target_chunk_bytes))
                    l_ratio = (n_comp / (n_comp - 1.0)) * (
                        (n_comp - 1.0) * f_hard * g_soft + (1.0 - f_hard) * (1.0 - g_soft)
                    )
                    ent_mask_f = valid_mask[:, 1:].float()
                    ent_per_tok = F.cross_entropy(
                        loc_logits[:, :-1].reshape(-1, self.vocab_size),
                        token_ids[:, 1:].reshape(-1),
                        reduction="none",
                    ).view(B, T - 1)
                    l_ent = (ent_per_tok * ent_mask_f).sum() / ent_mask_f.sum().clamp(min=1.0)
                    self.last_ratio_loss = torch.where(
                        r_sum > 0.0, l_ratio + 0.25 * l_ent, 0.0 * p_prob.sum()
                    )
                    self._last_chunk_ratio_t = torch.where(
                        r_sum > 0.0, 1.0 / f_hard.clamp(min=1e-4), torch.ones_like(f_hard)
                    )
                else:
                    self.last_ratio_loss = None
                    valid_non_first = valid_mask[:, 1:].float().sum().clamp(min=1.0)
                    fires_non_first = b_mask[:, 1:].float().sum().clamp(min=1.0)
                    self._last_chunk_ratio_t = valid_non_first / fires_non_first
            else:
                self._last_chunk_ratio_val = 1.0
                self._last_chunk_ratio_t = None
                self.last_ratio_loss = None if self.router_locked else (0.0 * p_prob.sum())
        else:
            if self.byte_chunk_mode == "utf8":
                _, is_cont, char_len = _utf8_tensor_flags(token_ids)
                is_cont_v = is_cont & valid_mask
                c1 = F.pad(is_cont_v[:, :-1], (1, 0), value=False) if T > 1 else torch.zeros_like(is_cont_v)
                c2 = F.pad(is_cont_v[:, :-2], (2, 0), value=False) if T > 2 else torch.zeros_like(is_cont_v)
                c3 = F.pad(is_cont_v[:, :-3], (3, 0), value=False) if T > 3 else torch.zeros_like(is_cont_v)
                can_fire = valid_mask & (~(is_cont_v & (~(c1 & c2 & c3))))
                is_lead = (valid_mask & (~is_cont)).long()
                raw_cand = (p_prob >= 0.5) & can_fire
                b_mask = _enforce_utf8_chunk_span_jit(
                    raw_cand,
                    valid_mask,
                    can_fire,
                    is_lead,
                    char_len,
                    self.max_chunk_bytes,
                    self.max_utf8_bytes,
                    self.refill_int,
                    self.cost_int,
                    self.burst_int,
                )
                ratio_mask = (valid_mask[:, 1:] & (~is_cont[:, 1:])) if T > 1 else valid_mask[:, 1:]
            else:
                raw_b = (p_prob >= 0.5) & valid_mask
                raw_b[:, 0] = True
                b_mask = enforce_max_chunk_span_vec(raw_b, valid_mask, self.max_chunk_bytes)
                ratio_mask = valid_mask[:, 1:]

            if T > 1:
                ratio_mask_f = ratio_mask.float()
                r_sum = ratio_mask_f.sum()
                has_valid = r_sum > 0.0
                denom_r = r_sum.clamp(min=1.0)
                f_hard = ((b_mask[:, 1:].float() * ratio_mask_f).sum() / denom_r).detach()
                self._last_chunk_ratio_t = torch.where(
                    has_valid, 1.0 / f_hard.clamp(min=1e-4), torch.ones_like(f_hard)
                )
                if not self.router_locked:
                    g_soft = (p_prob[:, 1:] * ratio_mask_f).sum() / denom_r
                    n_comp = max(1.5, float(self.target_chunk_bytes))
                    l_ratio = (n_comp / (n_comp - 1.0)) * (
                        (n_comp - 1.0) * f_hard * g_soft + (1.0 - f_hard) * (1.0 - g_soft)
                    )
                    ent_mask_f = valid_mask[:, 1:].float()
                    ent_per_tok = F.cross_entropy(
                        loc_logits[:, :-1].reshape(-1, self.vocab_size),
                        token_ids[:, 1:].reshape(-1),
                        reduction="none",
                    ).view(B, T - 1)
                    l_ent = (ent_per_tok * ent_mask_f).sum() / ent_mask_f.sum().clamp(min=1.0)
                    self.last_ratio_loss = torch.where(
                        has_valid, l_ratio + 0.25 * l_ent, 0.0 * p_prob.sum()
                    )
                else:
                    self.last_ratio_loss = None
            else:
                self._last_chunk_ratio_val = 1.0
                self._last_chunk_ratio_t = None
                self.last_ratio_loss = None if self.router_locked else (0.0 * p_prob.sum())

            chunk_counts = b_mask.long().sum(dim=1)
            if self.use_static_m_cap:
                M_max = self.static_m_cap_for_T(T)
            else:
                raw_m = int(chunk_counts.max().item())
                M_max = min(T, ((raw_m + 31) // 32) * 32) if (self.byte_chunk_mode == "utf8" and T >= 32) else max(1, raw_m)
            slot_idx = (b_mask.long().cumsum(dim=1) - 1).clamp(min=0, max=max(0, M_max - 1))
            t_arange = torch.arange(T, device=e_tok.device, dtype=torch.long).unsqueeze(0)
            rank_key = (~b_mask).long() * (T + 1) + t_arange
            gather_pos = rank_key.argsort(dim=1)[:, :M_max]

        self.last_m_cap_used = int(M_max)
        valid_chunk = (
            torch.arange(M_max, device=e_tok.device, dtype=torch.long).unsqueeze(0)
            < chunk_counts.unsqueeze(1)
        )
        chunk_mask_f = valid_chunk.unsqueeze(-1).to(e_tok.dtype)

        pos_d = gather_pos.unsqueeze(-1).expand(B, M_max, D)
        z_in = torch.gather(h_byte, 1, pos_d) * chunk_mask_f
        pos_w8 = gather_pos.unsqueeze(-1).expand(B, M_max, 8 * self.d_byte)
        c_win8_chunk = torch.gather(c_win8, 1, pos_w8)
        h_lex_chunk = self.lex_up(self.lex_norm(c_win8_chunk)) * chunk_mask_f
        p_chunk = torch.gather(p_prob, 1, gather_pos) * valid_chunk.float()
        offset_in_chunk = self._compute_offset_in_chunk(b_mask, self.max_utf8_bytes)

        ctx = ChunkContext(
            b_mask=b_mask,
            valid_mask=valid_mask,
            p_prob=p_prob,
            p_chunk=p_chunk,
            h_byte=h_byte,
            c_win8=c_norm,
            slot_idx=slot_idx,
            offset_in_chunk=offset_in_chunk,
            valid_chunk=valid_chunk,
        )
        return z_in, h_lex_chunk, ctx

    def decode_from_chunks(
        self,
        z_hat: torch.Tensor,
        ctx: ChunkContext,
    ) -> torch.Tensor:
        B, T, D = ctx.h_byte.shape
        M_max = z_hat.size(1)

        if M_max > 1:
            pc = ctx.p_chunk[:, 1:].unsqueeze(-1).to(z_hat.dtype)
            z_tail = pc * z_hat[:, 1:] + (1.0 - pc) * z_hat[:, :-1]
            z_bar = torch.cat([z_hat[:, :1], z_tail], dim=1)
        else:
            z_bar = z_hat

        z_up = torch.gather(
            z_bar,
            1,
            ctx.slot_idx.unsqueeze(-1).expand(B, T, D),
        )
        c_conf = torch.where(ctx.b_mask, ctx.p_prob, 1.0 - ctx.p_prob)
        ste_c = 0.85 + 0.15 * (c_conf + (1.0 - c_conf).detach())
        z_tilde = self.dechunk_norm(z_up) * ste_c.to(z_up.dtype).unsqueeze(-1)
        if self.use_scaled_decoder and self.byte_offset_emb is not None:
            z_tilde = z_tilde + self.byte_offset_emb(ctx.offset_in_chunk).to(z_tilde.dtype)

        dec_cat = torch.cat([z_tilde, ctx.c_win8.to(z_tilde.dtype)], dim=-1)
        dec_in = ctx.h_byte + self.dec_fuse(self.dec_fuse_norm(dec_cat))
        if (
            self.use_scaled_decoder
            and self.dec_dw_conv is not None
            and self.dec_dw_norm is not None
            and self.dec_dw_gate is not None
            and self.dec_dw_out is not None
            and self.dec_dw2_norm is not None
            and self.dec_dw2_conv is not None
            and self.dec_dw2_gate is not None
            and self.dec_dw2_out is not None
        ):
            xn1 = self.dec_dw_norm(dec_in)
            c1 = self.dec_dw_conv(F.pad(xn1.transpose(1, 2), (3, 0))).transpose(1, 2)
            dec_in = dec_in + self.dec_dw_out(F.silu(c1) * self.dec_dw_gate(xn1))
            xn2 = self.dec_dw2_norm(dec_in)
            c2 = self.dec_dw2_conv(F.pad(xn2.transpose(1, 2), (6, 0))).transpose(1, 2)
            dec_in = dec_in + self.dec_dw2_out(F.silu(c2) * self.dec_dw2_gate(xn2))
        return dec_in + self.dec_mlp(self.dec_norm(dec_in))

    # ------------------------------------------------------------------
    # Strict O(1) Incremental Streaming API (exact parity with batch)
    # ------------------------------------------------------------------

    def init_stream_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> ChunkerStreamState:
        ring_dec1 = (
            torch.zeros(batch_size, 3, self.d_model, device=device, dtype=dtype)
            if self.use_scaled_decoder
            else torch.zeros(batch_size, 0, 1, device=device, dtype=dtype)
        )
        ring_dec2 = (
            torch.zeros(batch_size, 6, self.d_model, device=device, dtype=dtype)
            if self.use_scaled_decoder
            else torch.zeros(batch_size, 0, 1, device=device, dtype=dtype)
        )
        return ChunkerStreamState(
            ring_char=torch.zeros(batch_size, 7, self.d_byte, device=device, dtype=dtype),
            ring_rchar=torch.zeros(batch_size, 3, self.d_rchar, device=device, dtype=dtype),
            prev_k_r=torch.zeros(batch_size, self.router_dim, device=device, dtype=torch.float32),
            prev_h_ent=torch.zeros(batch_size, device=device, dtype=torch.float32),
            prev_z_hat=torch.zeros(batch_size, self.d_model, device=device, dtype=dtype),
            cur_z_bar=torch.zeros(batch_size, self.d_model, device=device, dtype=dtype),
            ring_dec_conv1=ring_dec1,
            ring_dec_conv2=ring_dec2,
            run_len=torch.zeros(batch_size, device=device, dtype=torch.long),
            cp_run=torch.zeros(batch_size, device=device, dtype=torch.long),
            cont_streak=torch.zeros(batch_size, device=device, dtype=torch.long),
            credits=torch.full((batch_size,), self.burst_int, device=device, dtype=torch.long),
            step_idx=0,
            has_chunk=False,
        )

    @torch.no_grad()
    def encode_byte_step(
        self,
        token_t: torch.Tensor,
        e_tok_t: torch.Tensor,
        state: ChunkerStreamState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Processes ONE byte `token_t` `(B,)` and updates `state` in-place.
        Returns `(fire_t, p_t, h_byte_t, c_norm_t, h_lex_t, offset_t, valid_t)`.
        """
        dt = e_tok_t.dtype
        log_v = math.log(max(2, self.vocab_size))

        ct = e_tok_t[..., : self.d_byte]
        c_win8_t = torch.cat(
            [ct] + [state.ring_char[:, 6 - k] for k in range(7)],
            dim=-1,
        )
        rc_t = self.router_emb(token_t).to(dt)
        r_win4_t = torch.cat(
            [rc_t] + [state.ring_rchar[:, 2 - k] for k in range(3)],
            dim=-1,
        )

        # Shift character rings
        state.ring_char[:, :-1] = state.ring_char[:, 1:].clone()
        state.ring_char[:, -1] = ct
        state.ring_rchar[:, :-1] = state.ring_rchar[:, 1:].clone()
        state.ring_rchar[:, -1] = rc_t

        c_norm_t = self.enc_norm_in(c_win8_t)
        h_enc0 = e_tok_t + self.enc_up(c_norm_t)
        h_byte_t = h_enc0 + self.enc_mlp(self.enc_norm(h_enc0))
        h_lex_t = self.lex_up(self.lex_norm(c_win8_t))

        h_rt = F.silu(self._r_lin(self.router_in_proj, self.router_in_norm(r_win4_t)))
        q_rt = F.normalize(self._r_lin(self.router_q, h_rt).float(), dim=-1)
        k_rt = F.normalize(self._r_lin(self.router_k, h_rt).float(), dim=-1)
        if state.step_idx == 0:
            d_cos_t = torch.ones_like(state.prev_h_ent)
        else:
            d_cos_t = 0.5 * (1.0 - (q_rt * state.prev_k_r).sum(dim=-1))

        loc_t = self._r_lin(self.router_ent_head, self.router_ent_norm(h_rt)).float()
        lp_t = F.log_softmax(loc_t, dim=-1)
        h_norm_t = (-(lp_t.exp() * lp_t).sum(dim=-1) / log_v).clamp(0.0, 1.5)
        dh_t = torch.zeros_like(h_norm_t) if state.step_idx == 0 else F.relu(h_norm_t - state.prev_h_ent)

        wf = self.router_feat_weights.float()
        ctx_t = self._r_lin(self.router_ctx, h_rt).float().squeeze(-1)
        p_t = torch.sigmoid(wf[0] * d_cos_t + wf[1] * h_norm_t + wf[2] * dh_t + ctx_t)

        vt = token_t != self.pad_token_id
        if state.step_idx == 0:
            vt = torch.ones_like(vt)

        if self.byte_chunk_mode == "utf8":
            _, is_cont_t, cl_t = _utf8_tensor_flags(token_t)
            if state.step_idx == 0:
                fire_t = vt
            else:
                state.credits = torch.where(
                    vt, torch.clamp(state.credits + self.refill_int, max=self.burst_int), state.credits
                )
                is_cont_vt_pre = is_cont_t & vt
                shielded_t = is_cont_vt_pre & (state.cont_streak < 3)
                can_fire_t = vt & (~shielded_t)
                can_soft_t = (state.credits >= self.cost_int) & can_fire_t & (p_t >= 0.5)
                must_span_t = (state.cp_run >= self.max_chunk_bytes) | (
                    (state.run_len + cl_t) > self.max_utf8_bytes
                )
                fire_t = can_fire_t & (can_soft_t | must_span_t)
                state.credits = torch.where(
                    fire_t, torch.clamp(state.credits - self.cost_int, min=0), state.credits
                )
            is_cont_vt = is_cont_t & vt
            is_lead_t = (vt & (~is_cont_t)).long()
            state.cp_run = torch.where(
                vt, torch.where(fire_t, is_lead_t, state.cp_run + is_lead_t), state.cp_run
            )
            state.cont_streak = torch.where(
                is_cont_vt, state.cont_streak + 1, torch.zeros_like(state.cont_streak)
            )
        else:
            if state.step_idx == 0:
                fire_t = vt
            else:
                fire_t = vt & ((p_t >= 0.5) | (state.run_len >= self.max_chunk_bytes))

        # Offset before updating run_len: 0 when fire_t, else previous run_len
        offset_t = torch.where(fire_t, torch.zeros_like(state.run_len), state.run_len).clamp(
            0, self.max_utf8_bytes
        )
        state.run_len = torch.where(
            vt, torch.where(fire_t, torch.ones_like(state.run_len), state.run_len + 1), state.run_len
        )
        state.prev_k_r = k_rt
        state.prev_h_ent = h_norm_t
        state.step_idx += 1

        return fire_t, p_t, h_byte_t, c_norm_t, h_lex_t, offset_t, vt

    @torch.no_grad()
    def decode_byte_step(
        self,
        z_hat_new: torch.Tensor | None,
        fire_t: torch.Tensor,
        p_t: torch.Tensor,
        h_byte_t: torch.Tensor,
        c_norm_t: torch.Tensor,
        offset_t: torch.Tensor,
        state: ChunkerStreamState,
    ) -> torch.Tensor:
        """Decodes ONE byte `(B, d_model)` using `O(1)` `cur_z_bar` and causal conv ring."""
        dt = h_byte_t.dtype
        if z_hat_new is not None:
            if not state.has_chunk:
                z_bar_new = z_hat_new
                state.has_chunk = True
            else:
                pc = p_t.unsqueeze(-1).to(dt)
                z_bar_new = pc * z_hat_new + (1.0 - pc) * state.prev_z_hat
            f_mask = fire_t.unsqueeze(-1)
            state.cur_z_bar = torch.where(f_mask, z_bar_new, state.cur_z_bar)
            state.prev_z_hat = torch.where(f_mask, z_hat_new, state.prev_z_hat)

        z_up = state.cur_z_bar
        c_conf = torch.where(fire_t, p_t, 1.0 - p_t)
        ste_c = 0.85 + 0.15 * torch.ones_like(c_conf)  # exact eval STE value (c + (1 - c) = 1.0)
        z_tilde = self.dechunk_norm(z_up) * ste_c.to(dt).unsqueeze(-1)
        if self.use_scaled_decoder and self.byte_offset_emb is not None:
            z_tilde = z_tilde + self.byte_offset_emb(offset_t).to(dt)

        dec_cat = torch.cat([z_tilde, c_norm_t.to(dt)], dim=-1)
        dec_in = h_byte_t + self.dec_fuse(self.dec_fuse_norm(dec_cat))
        if (
            self.use_scaled_decoder
            and self.dec_dw_conv is not None
            and self.dec_dw_norm is not None
            and self.dec_dw_gate is not None
            and self.dec_dw_out is not None
            and self.dec_dw2_norm is not None
            and self.dec_dw2_conv is not None
            and self.dec_dw2_gate is not None
            and self.dec_dw2_out is not None
        ):
            xn1 = self.dec_dw_norm(dec_in)  # (B, D)
            win4 = torch.cat([state.ring_dec_conv1, xn1.unsqueeze(1)], dim=1)  # (B, 4, D)
            state.ring_dec_conv1 = win4[:, 1:, :].clone()
            w1 = self.dec_dw_conv.weight.squeeze(1).to(dt)
            b1 = self.dec_dw_conv.bias.to(dt) if self.dec_dw_conv.bias is not None else None
            c1 = (win4.transpose(1, 2) * w1.unsqueeze(0)).sum(dim=-1)
            if b1 is not None:
                c1 = c1 + b1.unsqueeze(0)
            dec_in = dec_in + self.dec_dw_out(F.silu(c1) * self.dec_dw_gate(xn1))

            xn2 = self.dec_dw2_norm(dec_in)  # (B, D)
            win7 = torch.cat([state.ring_dec_conv2, xn2.unsqueeze(1)], dim=1)  # (B, 7, D)
            state.ring_dec_conv2 = win7[:, 1:, :].clone()
            # Dilation=2 taps at indices [0, 2, 4, 6]
            taps2 = win7[:, 0::2, :]  # (B, 4, D)
            w2 = self.dec_dw2_conv.weight.squeeze(1).to(dt)
            b2 = self.dec_dw2_conv.bias.to(dt) if self.dec_dw2_conv.bias is not None else None
            c2 = (taps2.transpose(1, 2) * w2.unsqueeze(0)).sum(dim=-1)
            if b2 is not None:
                c2 = c2 + b2.unsqueeze(0)
            dec_in = dec_in + self.dec_dw2_out(F.silu(c2) * self.dec_dw2_gate(xn2))
        return dec_in + self.dec_mlp(self.dec_norm(dec_in))
