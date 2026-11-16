"""
Surprise-Gated End-to-End Byte Chunker (`SurpriseHNetChunker`) for `Calcite-5` (`SimVQ-Split` Phase B).

Combines an 8-byte causal lexical/encoder path with a decoupled, ternary-stable `BLT + H-Net`
boundary router (`router_emb` -> `r_win4` -> ternary-STE router layers) that locks after warmup
so Crystal (`VoronoiCrystalLayer`), Woodbury (`NeuralWoodburyLayer`), `0-QAT PTQ`, `STE QAT`,
and `2-bit uint8 T-MAC LUT` see 100% stationary chunk boundaries (`O(1)` streaming state `1,256 B`).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .baselines import RMSNorm, SwiGLU
from .quant_lut import quantize_ternary_ste, quantize_ternary_weights_ls


@dataclass
class ByteChunkContext:
    b_mask: torch.Tensor
    valid_mask: torch.Tensor
    p_prob: torch.Tensor
    p_chunk: torch.Tensor
    h_byte: torch.Tensor
    c_win8: torch.Tensor
    slot_idx: torch.Tensor
    valid_chunk: torch.Tensor


@torch.jit.script
def _enforce_max_chunk_span_jit(
    raw_b: torch.Tensor,
    valid_mask: torch.Tensor,
    max_span: int,
) -> torch.Tensor:
    """
    Strictly causal O(1)-state boundary guard enforcing max chunk length <= `max_span`
    on valid non-PAD tokens while keeping padded tail tokens inactive (`False`).
    """
    B, T = raw_b.shape
    out = torch.zeros_like(raw_b)
    run_len = torch.zeros(B, device=raw_b.device, dtype=torch.long)
    for t in range(T):
        vt = valid_mask[:, t]
        if t == 0:
            fire = vt
        else:
            fire = vt & (raw_b[:, t] | (run_len >= max_span))
        out[:, t] = fire
        run_len = torch.where(
            vt,
            torch.where(fire, torch.ones_like(run_len), run_len + 1),
            run_len,
        )
    return out


def enforce_max_chunk_span_vec(
    raw_b: torch.Tensor,
    valid_mask: torch.Tensor,
    max_span: int,
) -> torch.Tensor:
    """
    Vectorized O(1)-span causal boundary guard (`cummax` + modulo), bit-for-bit identical
    to `_enforce_max_chunk_span_jit` with zero Python loop over `T`.
    """
    if max_span <= 1:
        return valid_mask.clone()
    raw_v = (raw_b & valid_mask).clone()
    raw_v[:, 0] = valid_mask[:, 0]
    v_step = (valid_mask.long().cumsum(dim=1) - 1).clamp(min=0)
    fire_v = torch.where(raw_v, v_step, torch.zeros_like(v_step))
    last_fire_v, _ = torch.cummax(fire_v, dim=1)
    span_v = v_step - last_fire_v
    forced = (span_v > 0) & ((span_v % max_span) == 0)
    return valid_mask & (raw_v | forced)


def _utf8_tensor_flags(
    token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure bitwise tensor extraction of UTF-8 continuation & leading-byte lengths."""
    is_byte = token_ids >= 3
    raw_v = (token_ids - 3).clamp(min=0)
    is_cont = is_byte & (torch.bitwise_and(raw_v, 0xC0) == 0x80)
    is_4 = is_byte & (torch.bitwise_and(raw_v, 0xF8) == 0xF0)
    is_3 = is_byte & (torch.bitwise_and(raw_v, 0xF0) == 0xE0)
    is_2 = is_byte & (torch.bitwise_and(raw_v, 0xE0) == 0xC0)
    char_len = torch.ones_like(token_ids, dtype=torch.long)
    char_len = torch.where(is_2, torch.full_like(char_len, 2), char_len)
    char_len = torch.where(is_3, torch.full_like(char_len, 3), char_len)
    char_len = torch.where(is_4, torch.full_like(char_len, 4), char_len)
    return is_byte, is_cont, char_len


@torch.jit.script
def _enforce_utf8_chunk_span_jit(
    raw_cand: torch.Tensor,
    valid_mask: torch.Tensor,
    can_fire_mask: torch.Tensor,
    is_lead: torch.Tensor,
    char_len: torch.Tensor,
    max_cp_span: int,
    max_byte_span: int,
    refill_int: int,
    cost_int: int,
    burst_int: int,
) -> torch.Tensor:
    """
    Device-resident strictly causal O(1)-state UTF-8 boundary guard (exact integer leaky bucket):
    1. Continuation bytes (`cont_streak < 3`) are pre-masked via `can_fire_mask`.
    2. Checks `byte_run + char_len <= max_byte_span` causally at each leading byte.
    3. Bounds chunk rate via integer leaky-bucket credits without host-CPU syncs.
    """
    B, T = raw_cand.shape
    byte_run = torch.zeros(B, device=raw_cand.device, dtype=torch.long)
    cp_run = torch.zeros(B, device=raw_cand.device, dtype=torch.long)
    credits = torch.full((B,), burst_int, device=raw_cand.device, dtype=torch.long)

    rc_u = raw_cand.unbind(1)
    cf_u = can_fire_mask.unbind(1)
    vt_u = valid_mask.unbind(1)
    ld_u = is_lead.unbind(1)
    cl_u = char_len.unbind(1)
    outs: list[torch.Tensor] = []

    for t in range(T):
        vt = vt_u[t]
        ld = ld_u[t]
        if t == 0:
            fire = vt
        else:
            credits = torch.where(
                vt, torch.clamp(credits + refill_int, max=burst_int), credits
            )
            can_soft = (credits >= cost_int) & rc_u[t]
            must_span = (cp_run >= max_cp_span) | ((byte_run + cl_u[t]) > max_byte_span)
            fire = cf_u[t] & (can_soft | must_span)
            credits = torch.where(
                fire, torch.clamp(credits - cost_int, min=0), credits
            )
        outs.append(fire)
        fire_l = fire.long()
        byte_run = torch.where(vt, (1 - fire_l) * byte_run + 1, byte_run)
        cp_run = torch.where(vt, (1 - fire_l) * cp_run + ld, cp_run)
    return torch.stack(outs, dim=1)


def compute_chunker_state_bytes(
    d_model: int,
    d_byte: int = 32,
    d_rchar: int = 16,
    router_dim: int = 32,
    byte_chunk_mode: str = "byte",
) -> int:
    """
    Exact O(1) streaming state footprint (bytes) of `SurpriseHNetChunker` independent of L:
    - `7` previous byte slices `BF16` (`c_win8`): `7 * d_byte * 2` (`448 B`)
    - `3` previous router byte embeddings `BF16` (`r_win4`): `3 * d_rchar * 2` (`96 B`)
    - `prev_k_r` router key `BF16`: `router_dim * 2` (`64 B`)
    - `prev_h_ent` predictive entropy scalar `FP32`: `4` (`4 B`)
    - `prev_z_hat` 2-chunk smoother vector `FP32`: `d_model * 4` (`640 B`)
    - `run_len` causal span counter `int32`: `4` (`4 B`)
    Total in `"byte"` mode = `1,256 B`; in `"utf8"` mode (`+12 B` for `cp_run`, `cont_streak`, `credits` `int32`) = `1,268 B`.
    """
    base = (
        7 * d_byte * 2
        + 3 * d_rchar * 2
        + router_dim * 2
        + 4
        + d_model * 4
        + 4
    )
    return base + (12 if byte_chunk_mode == "utf8" else 0)


class SurpriseHNetChunker(nn.Module):
    """
    Zero-SubLN Stationary-After-Warmup Surprise-Gated Byte Chunker (H-Net Eq. 4-10 + BLT Entropy).
    """

    def __init__(
        self,
        vocab_size: int = 259,
        d_model: int = 160,
        d_byte: int = 32,
        d_rchar: int = 16,
        r_hidden: int = 96,
        byte_d_ff: int = 128,
        router_dim: int = 32,
        target_chunk_bytes: float = 4.9,
        max_chunk_bytes: int = 10,
        pad_token_id: int = 0,
        byte_chunk_mode: str = "byte",
        max_utf8_bytes: int = 16,
        rate_refill: float = 0.30,
        burst_cap: float = 25.0,
        use_static_m_cap: bool = False,
    ):
        super().__init__()
        assert (8 * d_byte) % 32 == 0, "8 * d_byte must divide cleanly into 32-weight LUT groups."
        assert (4 * d_rchar) % 32 == 0, "4 * d_rchar must divide cleanly into 32-weight LUT groups."
        assert r_hidden % 32 == 0, "r_hidden must divide cleanly into 32-weight LUT groups."
        assert d_model % 32 == 0, "d_model must divide cleanly into 32-weight LUT groups."
        assert byte_chunk_mode in ("byte", "utf8"), f"Unsupported byte_chunk_mode: {byte_chunk_mode}"

        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.d_byte = int(d_byte)
        self.d_rchar = int(d_rchar)
        self.r_hidden = int(r_hidden)
        self.router_dim = int(router_dim)
        self.target_chunk_bytes = float(target_chunk_bytes)
        self.max_chunk_bytes = int(max_chunk_bytes)
        self.pad_token_id = int(pad_token_id)
        self.byte_chunk_mode = str(byte_chunk_mode)
        self.max_utf8_bytes = int(max(4, max_utf8_bytes))
        self.rate_refill = float(rate_refill)
        self.burst_cap = float(burst_cap)
        self.refill_int = int(round(self.rate_refill * 10.0))
        self.cost_int = 10
        self.burst_int = int(round(self.burst_cap * 10.0))
        self.use_static_m_cap = bool(use_static_m_cap)
        self.router_locked: bool = False

        # 8-byte causal character slice encoder & word-scale lexical anchor (Zero SubLN)
        self.lex_norm = RMSNorm(8 * d_byte)
        self.lex_up = nn.Linear(8 * d_byte, d_model, bias=False)
        self.enc_norm_in = RMSNorm(8 * d_byte)
        self.enc_up = nn.Linear(8 * d_byte, d_model, bias=False)
        nn.init.normal_(self.enc_up.weight, std=0.008)
        self.enc_norm = RMSNorm(d_model)
        self.enc_mlp = SwiGLU(d_model, byte_d_ff)
        nn.init.normal_(self.enc_mlp.w_down.weight, std=0.012)

        # Decoupled ternary-stable BLT + H-Net boundary router
        self.router_emb = nn.Embedding(self.vocab_size, self.d_rchar)
        nn.init.normal_(self.router_emb.weight, std=0.04)
        self.router_in_norm = RMSNorm(4 * self.d_rchar)
        self.router_in_proj = nn.Linear(4 * self.d_rchar, self.r_hidden, bias=False)
        self.router_q = nn.Linear(self.r_hidden, self.router_dim, bias=False)
        self.router_k = nn.Linear(self.r_hidden, self.router_dim, bias=False)
        self.router_ctx = nn.Linear(self.r_hidden, 1, bias=True)
        nn.init.constant_(self.router_ctx.bias, -2.80)
        self.router_ent_norm = RMSNorm(self.r_hidden)
        self.router_ent_head = nn.Linear(self.r_hidden, self.vocab_size, bias=False)
        self.router_feat_weights = nn.Parameter(torch.tensor([1.6, 2.0, 2.4]))

        # Confidence-weighted causal dechunk & byte spelling decoder (Zero SubLN)
        self.dechunk_norm = RMSNorm(d_model)
        self.dec_fuse_norm = RMSNorm(d_model + 8 * d_byte)
        self.dec_fuse = nn.Linear(d_model + 8 * d_byte, d_model, bias=False)
        self.dec_norm = RMSNorm(d_model)
        self.dec_mlp = SwiGLU(d_model, byte_d_ff)

        self.last_ratio_loss: torch.Tensor | None = None
        self._last_chunk_ratio_val: float = float(target_chunk_bytes)
        self._last_chunk_ratio_t: torch.Tensor | None = None
        self.last_m_cap_used: int = 0

    @property
    def last_chunk_ratio(self) -> float:
        if self._last_chunk_ratio_t is not None:
            self._last_chunk_ratio_val = float(self._last_chunk_ratio_t.item())
            self._last_chunk_ratio_t = None
        return self._last_chunk_ratio_val

    @last_chunk_ratio.setter
    def last_chunk_ratio(self, value: float) -> None:
        self._last_chunk_ratio_t = None
        self._last_chunk_ratio_val = float(value)

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
        """Return True when the router is locked and `mod` is a ternary-frozen router Linear layer."""
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
                        _, trits_i8, scale_1d = quantize_ternary_weights_ls(
                            mod.weight.data
                        )
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

    def encode_to_chunks(
        self,
        e_tok: torch.Tensor,
        token_ids: torch.Tensor,
        tok_emb_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, ByteChunkContext]:
        del tok_emb_weight
        B, T, D = e_tok.shape

        c_win8 = self._causal_win8(e_tok)
        c_norm = self.enc_norm_in(c_win8)
        h_lex_byte = self.lex_up(self.lex_norm(c_win8))
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
        if self.byte_chunk_mode == "utf8":
            _, is_cont, char_len = _utf8_tensor_flags(token_ids)
            is_cont_v = is_cont & valid_mask
            if T > 1:
                c1 = F.pad(is_cont_v[:, :-1], (1, 0), value=False)
            else:
                c1 = torch.zeros_like(is_cont_v)
            if T > 2:
                c2 = F.pad(is_cont_v[:, :-2], (2, 0), value=False)
            else:
                c2 = torch.zeros_like(is_cont_v)
            if T > 3:
                c3 = F.pad(is_cont_v[:, :-3], (3, 0), value=False)
            else:
                c3 = torch.zeros_like(is_cont_v)
            shielded = is_cont_v & (~(c1 & c2 & c3))
            can_fire_mask = valid_mask & (~shielded)
            is_lead = (valid_mask & (~is_cont)).long()
            raw_cand = (p_prob >= 0.5) & can_fire_mask
            b_mask = _enforce_utf8_chunk_span_jit(
                raw_cand,
                valid_mask,
                can_fire_mask,
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
            ratio_t = 1.0 / f_hard.clamp(min=1e-4)
            self._last_chunk_ratio_t = torch.where(
                has_valid, ratio_t, torch.ones_like(ratio_t)
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
            if self.byte_chunk_mode == "utf8" and T >= 32:
                M_max = min(T, ((raw_m + 31) // 32) * 32)
            else:
                M_max = raw_m
        self.last_m_cap_used = int(M_max)
        slot_idx = (b_mask.long().cumsum(dim=1) - 1).clamp(min=0, max=max(0, M_max - 1))

        t_arange = torch.arange(T, device=e_tok.device, dtype=torch.long).unsqueeze(0)
        rank_key = (~b_mask).long() * (T + 1) + t_arange
        gather_pos = rank_key.argsort(dim=1)[:, :M_max]

        valid_chunk = (
            torch.arange(M_max, device=e_tok.device, dtype=torch.long).unsqueeze(0)
            < chunk_counts.unsqueeze(1)
        )
        chunk_mask_f = valid_chunk.unsqueeze(-1).to(e_tok.dtype)

        pos_d = gather_pos.unsqueeze(-1).expand(B, M_max, D)
        z_in = torch.gather(h_byte, 1, pos_d) * chunk_mask_f
        h_lex_chunk = torch.gather(h_lex_byte, 1, pos_d) * chunk_mask_f
        p_chunk = torch.gather(p_prob, 1, gather_pos) * valid_chunk.float()

        ctx = ByteChunkContext(
            b_mask=b_mask,
            valid_mask=valid_mask,
            p_prob=p_prob,
            p_chunk=p_chunk,
            h_byte=h_byte,
            c_win8=c_norm,
            slot_idx=slot_idx,
            valid_chunk=valid_chunk,
        )
        return z_in, h_lex_chunk, ctx

    def decode_from_chunks(
        self,
        z_hat: torch.Tensor,
        ctx: ByteChunkContext,
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

        dec_cat = torch.cat([z_tilde, ctx.c_win8.to(z_tilde.dtype)], dim=-1)
        dec_in = ctx.h_byte + self.dec_fuse(self.dec_fuse_norm(dec_cat))
        return dec_in + self.dec_mlp(self.dec_norm(dec_in))

    @torch.no_grad()
    def forward_streaming_check(
        self,
        e_tok: torch.Tensor,
        token_ids: torch.Tensor,
        tok_emb_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Strict O(1)-state sequential reference allocating ONLY fixed-size ring buffers
        independent of T (`compute_chunker_state_bytes()` = `1,256 B` per batch item),
        returning `(p_prob, b_mask, h_byte, state_bytes_per_item)`.
        """
        del tok_emb_weight
        B, T, D = e_tok.shape
        dev = e_tok.device
        dt = e_tok.dtype
        log_v = math.log(max(2, self.vocab_size))

        ring_char = torch.zeros(B, 7, self.d_byte, device=dev, dtype=torch.bfloat16)
        ring_rchar = torch.zeros(B, 3, self.d_rchar, device=dev, dtype=torch.bfloat16)
        prev_k_r = torch.zeros(B, self.router_dim, device=dev, dtype=torch.bfloat16)
        prev_h_ent = torch.zeros(B, device=dev, dtype=torch.float32)
        prev_z_hat = torch.zeros(B, D, device=dev, dtype=torch.float32)
        run_len = torch.zeros(B, device=dev, dtype=torch.int32)

        state_bytes = int(
            (
                ring_char.numel() * ring_char.element_size()
                + ring_rchar.numel() * ring_rchar.element_size()
                + prev_k_r.numel() * prev_k_r.element_size()
                + prev_h_ent.numel() * prev_h_ent.element_size()
                + prev_z_hat.numel() * prev_z_hat.element_size()
                + run_len.numel() * run_len.element_size()
            )
            // B
        )
        is_utf8 = self.byte_chunk_mode == "utf8"
        if is_utf8:
            cp_run = torch.zeros(B, device=dev, dtype=torch.int32)
            cont_streak = torch.zeros(B, device=dev, dtype=torch.int32)
            credits = torch.full((B,), self.burst_int, device=dev, dtype=torch.int32)
            state_bytes += int(
                (
                    cp_run.numel() * cp_run.element_size()
                    + cont_streak.numel() * cont_streak.element_size()
                    + credits.numel() * credits.element_size()
                )
                // B
            )
        else:
            cp_run = run_len
            cont_streak = run_len
            credits = run_len

        p_list = []
        b_list = []
        hb_list = []
        w_feat = self.router_feat_weights.float()

        for t in range(T):
            et = e_tok[:, t]
            ct = et[..., : self.d_byte]
            c_win8_t = torch.cat(
                [ct] + [ring_char[:, 6 - k].to(dt) for k in range(7)],
                dim=-1,
            )
            tid_t = token_ids[:, t]
            rc_t = self.router_emb(tid_t).to(dt)
            r_win4_t = torch.cat(
                [rc_t] + [ring_rchar[:, 2 - k].to(dt) for k in range(3)],
                dim=-1,
            )

            if t < T - 1:
                ring_char[:, :-1] = ring_char[:, 1:].clone()
                ring_char[:, -1] = ct.to(torch.bfloat16)
                ring_rchar[:, :-1] = ring_rchar[:, 1:].clone()
                ring_rchar[:, -1] = rc_t.to(torch.bfloat16)

            h_enc0 = et + self.enc_up(self.enc_norm_in(c_win8_t))
            h_bt = h_enc0 + self.enc_mlp(self.enc_norm(h_enc0))

            h_rt = F.silu(self._r_lin(self.router_in_proj, self.router_in_norm(r_win4_t)))
            q_rt = F.normalize(self._r_lin(self.router_q, h_rt).float(), dim=-1)
            k_rt = F.normalize(self._r_lin(self.router_k, h_rt).float(), dim=-1)
            if t == 0:
                d_cos_t = torch.ones(B, device=dev, dtype=torch.float32)
            else:
                d_cos_t = 0.5 * (1.0 - (q_rt * prev_k_r.float()).sum(dim=-1))

            loc_t = self._r_lin(self.router_ent_head, self.router_ent_norm(h_rt)).float()
            lp_t = F.log_softmax(loc_t, dim=-1)
            h_norm_t = (-(lp_t.exp() * lp_t).sum(dim=-1) / log_v).clamp(0.0, 1.5)
            dh_t = torch.zeros_like(h_norm_t) if t == 0 else F.relu(h_norm_t - prev_h_ent)

            ctx_t = self._r_lin(self.router_ctx, h_rt).float().squeeze(-1)
            p_t = torch.sigmoid(
                w_feat[0] * d_cos_t + w_feat[1] * h_norm_t + w_feat[2] * dh_t + ctx_t
            )

            vt = tid_t != self.pad_token_id
            if is_utf8:
                _, is_cont_t, cl_t = _utf8_tensor_flags(tid_t)
                if t == 0:
                    vt = torch.ones_like(vt)
                    fire_t = vt
                else:
                    credits = torch.where(
                        vt,
                        torch.clamp(credits + self.refill_int, max=self.burst_int),
                        credits,
                    )
                    is_cont_vt_pre = is_cont_t & vt
                    shielded_t = is_cont_vt_pre & (cont_streak.long() < 3)
                    can_fire_t = vt & (~shielded_t)
                    can_soft_t = (credits >= self.cost_int) & can_fire_t & (p_t >= 0.5)
                    must_span_t = (cp_run.long() >= self.max_chunk_bytes) | (
                        (run_len.long() + cl_t) > self.max_utf8_bytes
                    )
                    fire_t = can_fire_t & (can_soft_t | must_span_t)
                    credits = torch.where(
                        fire_t,
                        torch.clamp(credits - self.cost_int, min=0),
                        credits,
                    )
                is_cont_vt = is_cont_t & vt
                is_lead_t = (vt & (~is_cont_t)).to(cp_run.dtype)
                cp_run = torch.where(
                    vt,
                    torch.where(fire_t, is_lead_t, cp_run + is_lead_t),
                    cp_run,
                )
                cont_streak = torch.where(
                    is_cont_vt,
                    cont_streak + 1,
                    torch.zeros_like(cont_streak),
                )
            else:
                if t == 0:
                    vt = torch.ones_like(vt)
                    fire_t = vt
                else:
                    fire_t = vt & ((p_t >= 0.5) | (run_len.long() >= self.max_chunk_bytes))

            run_len = torch.where(
                vt,
                torch.where(fire_t, torch.ones_like(run_len), run_len + 1),
                run_len,
            )
            prev_k_r = k_rt.to(torch.bfloat16)
            prev_h_ent = h_norm_t
            prev_z_hat = prev_z_hat * 0.0 + h_bt.float()

            p_list.append(p_t)
            b_list.append(fire_t)
            hb_list.append(h_bt)

        return (
            torch.stack(p_list, dim=1),
            torch.stack(b_list, dim=1),
            torch.stack(hb_list, dim=1),
            state_bytes,
        )
