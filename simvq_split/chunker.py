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


def compute_chunker_state_bytes(
    d_model: int,
    d_byte: int = 32,
    d_rchar: int = 16,
    router_dim: int = 32,
) -> int:
    """
    Exact O(1) streaming state footprint (bytes) of `SurpriseHNetChunker` independent of L:
    - `7` previous byte slices `BF16` (`c_win8`): `7 * d_byte * 2` (`448 B`)
    - `3` previous router byte embeddings `BF16` (`r_win4`): `3 * d_rchar * 2` (`96 B`)
    - `prev_k_r` router key `BF16`: `router_dim * 2` (`64 B`)
    - `prev_h_ent` predictive entropy scalar `FP32`: `4` (`4 B`)
    - `prev_z_hat` 2-chunk smoother vector `FP32`: `d_model * 4` (`640 B`)
    - `run_len` causal span counter `int32`: `4` (`4 B`)
    """
    return (
        7 * d_byte * 2
        + 3 * d_rchar * 2
        + router_dim * 2
        + 4
        + d_model * 4
        + 4
    )


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
    ):
        super().__init__()
        assert (8 * d_byte) % 32 == 0, "8 * d_byte must divide cleanly into 32-weight LUT groups."
        assert (4 * d_rchar) % 32 == 0, "4 * d_rchar must divide cleanly into 32-weight LUT groups."
        assert r_hidden % 32 == 0, "r_hidden must divide cleanly into 32-weight LUT groups."
        assert d_model % 32 == 0, "d_model must divide cleanly into 32-weight LUT groups."

        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.d_byte = int(d_byte)
        self.d_rchar = int(d_rchar)
        self.r_hidden = int(r_hidden)
        self.router_dim = int(router_dim)
        self.target_chunk_bytes = float(target_chunk_bytes)
        self.max_chunk_bytes = int(max_chunk_bytes)
        self.pad_token_id = int(pad_token_id)
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
        self.last_chunk_ratio: float = float(target_chunk_bytes)

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
        raw_b = (p_prob >= 0.5) & valid_mask
        raw_b[:, 0] = True
        b_mask = _enforce_max_chunk_span_jit(raw_b, valid_mask, self.max_chunk_bytes)

        if T > 1:
            mask_r = valid_mask[:, 1:]
            if bool(mask_r.any().item()):
                f_hard = b_mask[:, 1:][mask_r].float().mean().detach()
                self.last_chunk_ratio = float(1.0 / max(1e-4, float(f_hard.item())))
                if not self.router_locked:
                    g_soft = p_prob[:, 1:][mask_r].mean()
                    n_comp = max(1.5, float(self.target_chunk_bytes))
                    l_ratio = (n_comp / (n_comp - 1.0)) * (
                        (n_comp - 1.0) * f_hard * g_soft + (1.0 - f_hard) * (1.0 - g_soft)
                    )
                    l_ent = F.cross_entropy(loc_logits[:, :-1][mask_r], token_ids[:, 1:][mask_r])
                    self.last_ratio_loss = l_ratio + 0.25 * l_ent
                else:
                    self.last_ratio_loss = None
            else:
                self.last_chunk_ratio = 1.0
                self.last_ratio_loss = None if self.router_locked else (0.0 * p_prob.sum())
        else:
            self.last_chunk_ratio = 1.0
            self.last_ratio_loss = None if self.router_locked else (0.0 * p_prob.sum())

        chunk_counts = b_mask.long().sum(dim=1)
        M_max = int(chunk_counts.max().item())
        slot_idx = (b_mask.long().cumsum(dim=1) - 1).clamp(min=0, max=M_max - 1)

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
            rc_t = self.router_emb(token_ids[:, t]).to(dt)
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

            vt = token_ids[:, t] != self.pad_token_id
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
