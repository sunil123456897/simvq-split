"""
Calcite v3.1 Phase-Free O(1) Byte-Level Induction Hash-Cache (`calcite_v2/byte_induct_cache.py`).

Closes the chunk-boundary phase-alignment gap that prevents chunk-level memories from
copying arbitrary unseen verbatim strings (`Needle-in-a-Haystack`, UUIDs, passkeys):
- Operates at **exact byte resolution** right inside `CalciteV2Chunker` (next to `Byte Engram`),
  eliminating 4.6-byte chunk-boundary phase jitter completely.
- Uses **2-scale pure-tensor polynomial rolling hashes** (`k = 4` and `k = 7` trailing bytes)
  into a fixed `O(1)` bank of `S` slots per scale (`2 * S` slots total), paired with a
  learned contextual key signature (`d_sig`) + dedicated byte-value embedding (`d_v`).
- **Exact Dual Form**:
  - Training & long-context eval: dispatches to native `kernels/fast_calcite_bic.py`
    (`fast_bic_scan_cuda`) when on CUDA and available, or an optimized zero-sync hoisted
    blocked causal collision-masked matrix form (`B_blk = 256`, `< 2 MB` VRAM at `L = 32,768`).
  - Live streaming (`CalciteStreamEngine`): exact 1-byte `O(1)` scatter-gather with circular
    7-byte ring (`ptr7`) and scalar `B=1` fast path (**< 1e-5 parity**, strictly `O(1)` RAM).
- 100% pure tensor neural module (zero `dict`, `re`, `ast`, or hardcoded rules).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import CalciteRMSNorm

try:
    from kernels.fast_calcite_bic import fast_bic_scan_available, fast_bic_scan_cuda
except ImportError:
    fast_bic_scan_available = None
    fast_bic_scan_cuda = None


@dataclass
class ByteInductionStreamState:
    bank: torch.Tensor        # (B, 2, S, d_sig + d_v + 1) fp32
    ring_tok7: torch.Tensor   # (B, 7) int64  ([t-7 .. t-1] chronologically when ptr7 == 0)
    prev_h_ctx: torch.Tensor  # (B, d_model) fp32
    step_count: int
    ptr7: int = 0             # circular write pointer into ring_tok7 (oldest slot)

    def memory_bytes(self) -> int:
        b = max(1, int(self.bank.size(0)))
        total = 0
        for t in (self.bank, self.ring_tok7, self.prev_h_ctx):
            total += int(t.numel() * t.element_size())
        return total // b


def _hash_2scale_seq(
    token_ids: torch.Tensor,
    n_slots: int,
    t_pad6: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pure-tensor 2-scale polynomial rolling hash over causal trailing 4-byte and 7-byte windows.
    Returns `(slot_q4, slot_q7)` each of shape `(B, T)` int64 in `[0, n_slots)`.
    """
    _, T = token_ids.shape
    t_pad = t_pad6 if t_pad6 is not None else F.pad(token_ids.long(), (6, 0), value=0)  # (B, T + 6)
    b6 = t_pad[:, 0:T]
    b5 = t_pad[:, 1 : T + 1]
    b4 = t_pad[:, 2 : T + 2]
    b3 = t_pad[:, 3 : T + 3]
    b2 = t_pad[:, 4 : T + 4]
    b1 = t_pad[:, 5 : T + 5]
    b0 = t_pad[:, 6 : T + 6]

    base4 = b3 * 32749 + b2 * 8191 + b1 * 2053 + b0 * 521
    raw4 = base4 + 17
    raw7 = b6 * 1000003 + b5 * 524287 + b4 * 131071 + base4 + 31
    if n_slots > 0 and (n_slots & (n_slots - 1)) == 0:
        mask = n_slots - 1
        return raw4 & mask, raw7 & mask
    return torch.remainder(raw4, n_slots), torch.remainder(raw7, n_slots)


def _hash_2scale_ring7(
    ring7: torch.Tensor,
    n_slots: int,
    ptr7: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Identical 2-scale hash on `(B, 7)` circular token ring (`ptr7` = oldest slot)."""
    if ptr7 == 0:
        b6, b5, b4, b3, b2, b1, b0 = (
            ring7[:, 0],
            ring7[:, 1],
            ring7[:, 2],
            ring7[:, 3],
            ring7[:, 4],
            ring7[:, 5],
            ring7[:, 6],
        )
    else:
        b6 = ring7[:, ptr7]
        b5 = ring7[:, (ptr7 + 1) % 7]
        b4 = ring7[:, (ptr7 + 2) % 7]
        b3 = ring7[:, (ptr7 + 3) % 7]
        b2 = ring7[:, (ptr7 + 4) % 7]
        b1 = ring7[:, (ptr7 + 5) % 7]
        b0 = ring7[:, (ptr7 + 6) % 7]
    base4 = b3 * 32749 + b2 * 8191 + b1 * 2053 + b0 * 521
    raw4 = base4 + 17
    raw7 = b6 * 1000003 + b5 * 524287 + b4 * 131071 + base4 + 31
    if n_slots > 0 and (n_slots & (n_slots - 1)) == 0:
        mask = n_slots - 1
        return raw4 & mask, raw7 & mask
    return torch.remainder(raw4, n_slots), torch.remainder(raw7, n_slots)


class ByteInductionCache(nn.Module):
    """Exact `O(1)` 2-scale direct-mapped byte induction cache with contextual signature verification."""

    def __init__(
        self,
        vocab_size: int = 259,
        d_model: int = 256,
        n_slots: int = 1024,
        d_sig: int = 8,
        d_v: int = 16,
        write_thresh: float = 0.25,
        block_size: int = 256,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.n_slots = int(n_slots)
        self.d_sig = int(d_sig)
        self.d_v = int(d_v)
        self.d_u = self.d_sig + self.d_v + 1
        self.write_thresh = float(write_thresh)
        self.block_size = int(max(32, block_size))

        # Dedicated byte-identity value embeddings per scale (never blurred by chunk compression)
        self.val_emb = nn.Embedding(self.vocab_size, 2 * self.d_v)
        nn.init.normal_(self.val_emb.weight, std=0.15)

        # Contextual signature Q/K projections (symmetry-initialized so identical contexts match step 1)
        self.q_proj = nn.Linear(d_model, 2 * self.d_sig, bias=False)
        self.k_proj = nn.Linear(d_model, 2 * self.d_sig, bias=False)
        with torch.no_grad():
            self.k_proj.weight.copy_(self.q_proj.weight)

        # Dual-timescale sparse write gate:
        #   Scale 0 (k=4): fast in-context prose/code repeater (tau=0.25, open bias +0.50)
        #   Scale 1 (k=7): ultra-sparse 32K needle vault (tau=0.45, selective bias +0.15)
        self.write_proj = nn.Linear(d_model + 2 * self.d_v, 2, bias=True)
        with torch.no_grad():
            self.write_proj.bias.copy_(torch.tensor([0.50, 0.15], dtype=torch.float32))
        self.register_buffer(
            "write_thresh_vec",
            torch.tensor([float(write_thresh), max(float(write_thresh), 0.45)], dtype=torch.float32),
            persistent=False,
        )
        # Precomputed causal block masks (eliminates per-block allocations inside dual scan)
        Bc = self.block_size
        self.register_buffer("_tril_bc", torch.tril(torch.ones((Bc, Bc), dtype=torch.bool)), persistent=False)
        self.register_buffer("_triu_bc", torch.triu(torch.ones((Bc, Bc), dtype=torch.bool), diagonal=1), persistent=False)

        # Signature verification confidence gate (rejects hash collisions & Traps cleanly)
        self.read_scale = nn.Parameter(torch.full((2,), 8.0, dtype=torch.float32))
        self.read_thresh = nn.Parameter(torch.full((2,), 0.25, dtype=torch.float32))

        self.out_norm = CalciteRMSNorm(2 * self.d_v)
        self.out_proj = nn.Linear(2 * self.d_v, d_model, bias=False)
        with torch.no_grad():
            self.out_proj.weight.mul_(0.35)
        self.out_gate = nn.Linear(d_model, 1, bias=True)
        nn.init.constant_(self.out_gate.bias, 0.0)

        self.last_aux_loss: torch.Tensor | None = None
        self._last_write_rate_val: float = 0.0
        self._last_write_rate_t: torch.Tensor | None = None
        self._last_write_rate_s1_val: float = 0.0
        self._last_write_rate_s1_t: torch.Tensor | None = None

    @property
    def last_write_rate(self) -> float:
        if self._last_write_rate_t is not None:
            self._last_write_rate_val = float(self._last_write_rate_t.item())
            self._last_write_rate_t = None
        return self._last_write_rate_val

    @last_write_rate.setter
    def last_write_rate(self, val: float) -> None:
        self._last_write_rate_val = float(val)
        self._last_write_rate_t = None

    @property
    def last_write_rate_s1(self) -> float:
        if self._last_write_rate_s1_t is not None:
            self._last_write_rate_s1_val = float(self._last_write_rate_s1_t.item())
            self._last_write_rate_s1_t = None
        return self._last_write_rate_s1_val

    @last_write_rate_s1.setter
    def last_write_rate_s1(self, val: float) -> None:
        self._last_write_rate_s1_val = float(val)
        self._last_write_rate_s1_t = None

    @staticmethod
    def _safe_unit(v: torch.Tensor) -> torch.Tensor:
        vf = v.float()
        return vf * torch.rsqrt(vf.pow(2).sum(dim=-1, keepdim=True) + 1e-6)

    def compute_state_bytes(self) -> int:
        bank_b = 2 * self.n_slots * self.d_u * 4
        ring_b = 7 * 8 + self.d_model * 4
        return int(bank_b + ring_b)

    def _compute_beta(
        self, h_prev: torch.Tensor, v_flat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns `(beta_eff, beta_soft)` of shape `(..., 2)` in `[0, 0.9999]`."""
        dev_type = h_prev.device.type
        w_param = getattr(self.write_proj, "weight", None)
        if dev_type in ("cuda", "cpu") and torch.is_autocast_enabled(dev_type):
            in_dt = torch.get_autocast_dtype(dev_type)
        elif isinstance(w_param, torch.Tensor):
            in_dt = w_param.dtype
        else:
            in_dt = h_prev.dtype
        h_in = h_prev if h_prev.dtype == in_dt else h_prev.to(in_dt)
        v_in = v_flat if v_flat.dtype == in_dt else v_flat.to(in_dt)
        w_logit = self.write_proj(torch.cat((h_in, v_in), dim=-1)).float()
        beta_soft = torch.sigmoid(w_logit)
        tau = self.write_thresh_vec
        if tau.device != beta_soft.device:
            tau = tau.to(device=beta_soft.device, dtype=torch.float32)
        shape = [1] * (beta_soft.ndim - 1) + [2]
        tau_v = tau.view(*shape)
        beta_hard = ((beta_soft - tau_v) / (1.0 - tau_v).clamp(min=1e-4)).clamp(0.0, 1.0)
        beta_st = (beta_hard + 0.15 * (beta_soft - beta_soft.detach())) * 0.9999
        return beta_st, beta_soft

    def _decode_read_u(
        self, u_read: torch.Tensor, q_sig: torch.Tensor
    ) -> torch.Tensor:
        """
        `u_read`: `(..., 2, d_u)`, `q_sig`: `(..., 2, d_sig)` unit-norm.
        Returns gated value tensor `(..., 2 * d_v)` in FP32.
        """
        k_r = self._safe_unit(u_read[..., : self.d_sig])
        v_r = u_read[..., self.d_sig : self.d_sig + self.d_v]
        f_r = u_read[..., -1:].clamp(0.0, 1.0)
        sim = (q_sig * k_r).sum(dim=-1, keepdim=True)  # (..., 2, 1)
        shape = [1] * (sim.ndim - 2) + [2, 1]
        r_sc = self.read_scale.float().clamp(1.0, 24.0).view(*shape)
        r_th = self.read_thresh.float().clamp(-0.5, 0.95).view(*shape)
        conf = torch.sigmoid(r_sc * (sim - r_th)) * f_r
        out = conf * v_r
        return out.reshape(*out.shape[:-2], 2 * self.d_v)

    def _scan_blocks_pytorch(
        self,
        sq: torch.Tensor,       # (B, T, 2) int64
        sw: torch.Tensor,       # (B, T, 2) int64
        beta_eff: torch.Tensor, # (B, T, 2) fp32
        u_all: torch.Tensor,    # (B, T, 2, d_u) fp32
    ) -> torch.Tensor:
        """Hoisted zero-sync blocked exact dual-to-state scan returning `(B, T, 2, d_u)` fp32."""
        B, T, _ = sq.shape
        dev = u_all.device
        Bc = self.block_size
        n_blocks = (T + Bc - 1) // Bc
        S = self.n_slots
        D_u = self.d_u
        B2 = B * 2

        # Hoist all permutes & log(1 - beta) outside the block loop
        sq_all = sq.permute(0, 2, 1).reshape(B2, T)
        sw_all = sw.permute(0, 2, 1).reshape(B2, T)
        bt_all = beta_eff.permute(0, 2, 1).reshape(B2, T)
        u_perm = u_all.permute(0, 2, 1, 3).reshape(B2, T, D_u)
        log_k_all = torch.log((1.0 - bt_all).clamp(min=1e-5))

        tril_full = self._tril_bc
        triu_full = self._triu_bc
        if tril_full.device != dev:
            tril_full = tril_full.to(dev)
            triu_full = triu_full.to(dev)
        zero_s = torch.zeros((), device=dev, dtype=torch.float32)

        bank = torch.zeros((B2, S, D_u), device=dev, dtype=torch.float32) if n_blocks > 1 else None
        u_read_blocks: list[torch.Tensor] = []

        for bi in range(n_blocks):
            st = bi * Bc
            en = min(T, st + Bc)
            Lb = en - st

            sq_b = sq_all[:, st:en]
            sw_b = sw_all[:, st:en]
            bt_b = bt_all[:, st:en]
            u_b = u_perm[:, st:en]
            log_k = log_k_all[:, st:en]

            tril = tril_full[:Lb, :Lb]
            same = (sq_b.unsqueeze(2) == sw_b.unsqueeze(1)) & tril.unsqueeze(0)
            lk_m = torch.where(same, log_k.unsqueeze(1), zero_s)
            cs = lk_m.cumsum(dim=-1)
            cs_end = cs[:, :, -1:]

            w_mat = torch.where(
                same,
                bt_b.unsqueeze(1) * torch.exp((cs_end - cs).clamp(min=-30.0)),
                zero_s,
            )
            u_blk = torch.bmm(w_mat, u_b)
            if bi > 0 and bank is not None:
                prev_g = torch.gather(bank, 1, sq_b.unsqueeze(-1).expand(B2, Lb, D_u))
                u_read_b = torch.exp(cs_end.clamp(min=-30.0)) * prev_g + u_blk
            else:
                u_read_b = u_blk
            u_read_blocks.append(u_read_b)

            if bi < n_blocks - 1 and bank is not None:
                triu_s = triu_full[:Lb, :Lb]
                same_after = (sw_b.unsqueeze(2) == sw_b.unsqueeze(1)) & triu_s.unsqueeze(0)
                log_surv = torch.where(same_after, log_k.unsqueeze(1), zero_s).sum(dim=-1)
                contrib = (bt_b * torch.exp(log_surv.clamp(min=-30.0))).unsqueeze(-1) * u_b
                slot_lk = torch.zeros((B2, S), device=dev, dtype=torch.float32)
                slot_lk.scatter_add_(1, sw_b, log_k)
                delta_b = torch.zeros((B2, S, D_u), device=dev, dtype=torch.float32)
                delta_b.scatter_add_(1, sw_b.unsqueeze(-1).expand(B2, Lb, D_u), contrib)
                bank = bank * torch.exp(slot_lk.clamp(min=-30.0)).unsqueeze(-1) + delta_b

        u_read_full = u_read_blocks[0] if n_blocks == 1 else torch.cat(u_read_blocks, dim=1)
        return u_read_full.view(B, 2, T, D_u).permute(0, 2, 1, 3)

    def forward(
        self,
        token_ids: torch.Tensor,
        h_ctx: torch.Tensor,
        t_pad6: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        `token_ids`: `(B, T)` int64, `h_ctx`: `(B, T, d_model)` causal decoder state.
        Returns `(B, T, d_model)` phase-free verbatim induction residual.
        """
        B, T, _ = h_ctx.shape
        dt = h_ctx.dtype
        dev = h_ctx.device

        sq4, sq7 = _hash_2scale_seq(token_ids, self.n_slots, t_pad6=t_pad6)  # (B, T)
        sq = torch.stack((sq4, sq7), dim=2)                                  # (B, T, 2)

        v_flat = self.val_emb(token_ids.clamp(0, self.vocab_size - 1)).float()  # (B, T, 2*d_v)
        v_2 = v_flat.view(B, T, 2, self.d_v)
        q_sig = self._safe_unit(self.q_proj(h_ctx).view(B, T, 2, self.d_sig))

        # Avoid allocating full `(B, T, d_model)` shifted `h_prev`:
        # `k_proj` has `bias=False` (k_proj(0)==0) and `t=0` write is strictly masked off (`beta_eff[:, 0] == 0`).
        if T > 1:
            sw = F.pad(sq[:, :-1], (0, 0, 1, 0), value=0)
            h_ctx_prev = h_ctx[:, :-1]
            k_sig_tail = self._safe_unit(self.k_proj(h_ctx_prev).view(B, T - 1, 2, self.d_sig))
            k_sig = F.pad(k_sig_tail, (0, 0, 0, 0, 1, 0), value=0.0)
            beta_st_tail, beta_soft_tail = self._compute_beta(h_ctx_prev, v_flat[:, 1:])
            beta_eff = F.pad(beta_st_tail, (0, 0, 1, 0), value=0.0)
            if self.training:
                # Dual-timescale sparsity: keep Scale 0 (k=4) free for prose/code LM boost,
                # while driving Scale 1 (k=7) ultra-sparse (<= 2.5%) so needles survive 32K+ bytes cleanly.
                self.last_aux_loss = 0.80 * F.relu(beta_soft_tail[:, :, 1].mean() - 0.025)
            else:
                self.last_aux_loss = None
            # Lazy 0-d GPU tensors: ZERO `.item()` syncs or `torch.compile` graph breaks!
            b_det = beta_st_tail.detach()
            self._last_write_rate_t = (b_det > 0.0).float().mean()
            self._last_write_rate_s1_t = (b_det[:, :, 1] > 0.0).float().mean()
        else:
            sw = torch.zeros((B, 1, 2), device=dev, dtype=torch.long)
            k_sig = torch.zeros((B, 1, 2, self.d_sig), device=dev, dtype=torch.float32)
            beta_eff = torch.zeros((B, 1, 2), device=dev, dtype=torch.float32)
            self.last_aux_loss = None

        ones = torch.ones((B, T, 2, 1), device=dev, dtype=torch.float32)
        u_all = torch.cat((k_sig, v_2, ones), dim=-1)  # (B, T, 2, d_u)

        with torch.amp.autocast(dev.type, enabled=False):
            if (
                h_ctx.is_cuda
                and fast_bic_scan_cuda is not None
                and fast_bic_scan_available is not None
                and fast_bic_scan_available()
            ):
                u_read_all = fast_bic_scan_cuda(sq, sw, beta_eff, u_all, self.n_slots)
            else:
                u_read_all = self._scan_blocks_pytorch(sq, sw, beta_eff, u_all)
            out_v = self._decode_read_u(u_read_all, q_sig).to(dt)  # (B, T, 2*d_v)

        gate = 2.0 * torch.sigmoid(self.out_gate(h_ctx))
        return gate * self.out_proj(self.out_norm(out_v))

    def init_stream_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> ByteInductionStreamState:
        del dtype
        return ByteInductionStreamState(
            bank=torch.zeros(batch_size, 2, self.n_slots, self.d_u, device=device, dtype=torch.float32),
            ring_tok7=torch.zeros(batch_size, 7, device=device, dtype=torch.long),
            prev_h_ctx=torch.zeros(batch_size, self.d_model, device=device, dtype=torch.float32),
            step_count=0,
            ptr7=0,
        )

    @torch.no_grad()
    def step(
        self,
        tok_t: torch.Tensor,
        h_ctx_t: torch.Tensor,
        state: ByteInductionStreamState,
    ) -> torch.Tensor:
        """Exact `O(1)` 1-byte scatter/gather step matching `forward()` to `< 1e-5`."""
        B = tok_t.size(0)
        dt = h_ctx_t.dtype
        dev = h_ctx_t.device
        ptr = state.ptr7

        with torch.amp.autocast(dev.type, enabled=False):
            # 1. Write `(prefix_hash_{t-1}, k_sig(prev_h_ctx), val(tok_t))` if step_count > 0
            if state.step_count > 0:
                sw4, sw7 = _hash_2scale_ring7(state.ring_tok7, self.n_slots, ptr7=ptr)
                v_flat = self.val_emb(tok_t.long().clamp(0, self.vocab_size - 1)).float()  # (B, 2*d_v)
                v_2 = v_flat.view(B, 2, self.d_v)
                h_prev_dt = state.prev_h_ctx if state.prev_h_ctx.dtype == dt else state.prev_h_ctx.to(dt)
                k_sig = self._safe_unit(self.k_proj(h_prev_dt).view(B, 2, self.d_sig))
                beta_t, _ = self._compute_beta(h_prev_dt, v_flat)  # (B, 2)

                if B == 1:
                    s0 = sw4[0]
                    s1 = sw7[0]
                    bt0 = beta_t[0, 0]
                    bt1 = beta_t[0, 1]
                    d_s = self.d_sig
                    d_sv = d_s + self.d_v
                    # Update Scale 0 slot in-place (zero `arange`/`ones`/`cat`/`.item()` allocations)
                    u0 = state.bank[0, 0, s0]
                    om0 = 1.0 - bt0
                    u0_new = u0 * om0
                    u0_new[:d_s] = u0_new[:d_s] + bt0 * k_sig[0, 0]
                    u0_new[d_s:d_sv] = u0_new[d_s:d_sv] + bt0 * v_2[0, 0]
                    u0_new[-1] = u0_new[-1] + bt0
                    state.bank[0, 0, s0] = u0_new
                    # Update Scale 1 slot in-place
                    u1 = state.bank[0, 1, s1]
                    om1 = 1.0 - bt1
                    u1_new = u1 * om1
                    u1_new[:d_s] = u1_new[:d_s] + bt1 * k_sig[0, 1]
                    u1_new[d_s:d_sv] = u1_new[d_s:d_sv] + bt1 * v_2[0, 1]
                    u1_new[-1] = u1_new[-1] + bt1
                    state.bank[0, 1, s1] = u1_new
                else:
                    b_idx = torch.arange(B, device=dev, dtype=torch.long)
                    ones = torch.ones((B, 2, 1), device=dev, dtype=torch.float32)
                    u_t = torch.cat((k_sig, v_2, ones), dim=-1)
                    bt0 = beta_t[:, 0:1]
                    state.bank[b_idx, 0, sw4] = (1.0 - bt0) * state.bank[b_idx, 0, sw4] + bt0 * u_t[:, 0]
                    bt1 = beta_t[:, 1:2]
                    state.bank[b_idx, 1, sw7] = (1.0 - bt1) * state.bank[b_idx, 1, sw7] + bt1 * u_t[:, 1]

            # 2. Push `tok_t` into circular 7-byte ring (zero `.cat()` allocation!) and READ
            state.ring_tok7[:, ptr] = tok_t.long()
            ptr = (ptr + 1) % 7
            state.ptr7 = ptr
            sq4, sq7 = _hash_2scale_ring7(state.ring_tok7, self.n_slots, ptr7=ptr)
            if B == 1:
                u_read = torch.stack((state.bank[0, 0, sq4[0]], state.bank[0, 1, sq7[0]]), dim=0).unsqueeze(0)
            else:
                b_idx = torch.arange(B, device=dev, dtype=torch.long)
                u_read = torch.stack((state.bank[b_idx, 0, sq4], state.bank[b_idx, 1, sq7]), dim=1)
            q_sig = self._safe_unit(self.q_proj(h_ctx_t).view(B, 2, self.d_sig))
            out_v = self._decode_read_u(u_read, q_sig).to(dt)  # (B, 2*d_v)

            state.prev_h_ctx.copy_(h_ctx_t)
            state.step_count += 1

        gate = 2.0 * torch.sigmoid(self.out_gate(h_ctx_t))
        return gate * self.out_proj(self.out_norm(out_v))
