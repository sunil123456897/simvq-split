"""
Calcite v3 Epistemic L2 Exact Slot Cache (`calcite_v2/epistemic_slots.py`).

Uses `Woodbury`'s own causal epistemic novelty (`q^T P q`) as a zero-heuristic controller
for a fixed-size `O(1)` content-addressed bank (`S` slots, e.g. `32 x 64` = `16 KB` FP32):
- High-novelty chunks (rare identifiers, numbers, passkeys, entity facts) pin losslessly
  into a slot, advancing the circular eviction pointer ONLY when a chunk is truly novel.
- Repeating an existing key (`cos_sim > 0.72`) overwrites its matching slot in-place
  without evicting unrelated pinned slots.
- Boring filler prose does not advance the ring pointer and cannot wash out pinned needles
  even across `32K+` byte contexts.
- Strict `O(1)` RAM, block-checkpointed Float32 AMP-shielded scan, and exact `stream == forward` parity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from simvq_split.baselines import RMSNorm


@dataclass
class EpistemicSlotsStreamState:
    k_bank: torch.Tensor    # (B, S, D) fp32
    v_bank: torch.Tensor    # (B, S, D) fp32
    filled: torch.Tensor    # (B, S) fp32 in {0, 1}
    ring_ptr: torch.Tensor  # (B,) int64

    def memory_bytes(self) -> int:
        b = max(1, int(self.k_bank.size(0)))
        total = 0
        for t in (self.k_bank, self.v_bank, self.filled, self.ring_ptr):
            total += int(t.numel() * t.element_size())
        return total // b


@torch.jit.script
def _slot_block_scan_jit(
    k_bank0: torch.Tensor,    # (B, S, D) fp32
    v_bank0: torch.Tensor,    # (B, S, D) fp32
    filled0: torch.Tensor,    # (B, S) fp32
    ptr0: torch.Tensor,       # (B,) int64
    q_blk: torch.Tensor,      # (L, B, D) fp32 unit-norm
    k_blk: torch.Tensor,      # (L, B, D) fp32 unit-norm
    v_blk: torch.Tensor,      # (L, B, D) fp32
    w_blk: torch.Tensor,      # (L, B, 1) fp32 in (0, 1)
    match_thresh: float,
    novel_thresh: float,
    read_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    L, B, D = q_blk.shape
    S = k_bank0.size(1)
    k_bank = k_bank0
    v_bank = v_bank0
    filled = filled0
    ptr = ptr0
    slot_arange = torch.arange(S, device=k_bank0.device, dtype=torch.long).unsqueeze(0)

    q_u = q_blk.unbind(0)
    k_u = k_blk.unbind(0)
    v_u = v_blk.unbind(0)
    w_u = w_blk.unbind(0)
    outs: list[torch.Tensor] = []

    for t in range(L):
        qt = q_u[t]  # (B, D)
        kt = k_u[t]  # (B, D)
        vt = v_u[t]  # (B, D)
        wt = w_u[t]  # (B, 1)

        # 1. Causal READ strictly BEFORE current write
        has_any = (filled.sum(dim=1, keepdim=True) > 0.5).to(torch.float32)  # (B, 1)
        sim_r = torch.bmm(k_bank, qt.unsqueeze(-1)).squeeze(-1)  # (B, S)
        sim_r_masked = torch.where(filled > 0.5, sim_r, torch.full_like(sim_r, -1e4))
        max_sim_r = sim_r_masked.max(dim=1, keepdim=True).values
        att_r = F.softmax(sim_r_masked * read_scale, dim=-1)  # (B, S)
        conf_r = torch.sigmoid(10.0 * (max_sim_r - 0.50)) * has_any  # (B, 1)
        read_t = conf_r * torch.bmm(att_r.unsqueeze(1), v_bank).squeeze(1)  # (B, D)
        outs.append(read_t)

        # 2. Content-addressed exact OVERWRITE or novelty-gated RING insert
        sim_k = torch.bmm(k_bank, kt.unsqueeze(-1)).squeeze(-1)  # (B, S)
        sim_k_masked = torch.where(filled > 0.5, sim_k, torch.full_like(sim_k, -2.0))
        max_sim_k, match_idx = sim_k_masked.max(dim=1)  # (B,)

        is_match = (max_sim_k > match_thresh) & (filled.sum(dim=1) > 0.5)  # (B,)
        is_novel = (~is_match) & (wt.squeeze(-1) > novel_thresh)           # (B,)
        do_write = is_match | is_novel                                     # (B,)

        target_idx = torch.where(is_match, match_idx, ptr)  # (B,)
        slot_oh = (slot_arange == target_idx.unsqueeze(1)).to(torch.float32)  # (B, S)
        write_mask = (slot_oh * do_write.to(torch.float32).unsqueeze(1)).unsqueeze(-1)  # (B, S, 1)

        # Straight-through exact slot write so pinned keys/values stay sharp while `wt` gets live grads
        w_st = (1.0 + (wt - wt.detach())).unsqueeze(1)  # (B, 1, 1)
        alpha = write_mask * w_st  # (B, S, 1)
        # For overwriting an existing match, replace cleanly; for novel insert, pin cleanly
        k_new = (1.0 - alpha) * k_bank + alpha * kt.unsqueeze(1)
        k_norm = torch.sqrt(k_new.pow(2).sum(dim=-1, keepdim=True) + 1e-8)
        k_bank = torch.where(write_mask > 0.5, k_new / k_norm, k_bank)
        v_bank = (1.0 - alpha) * v_bank + alpha * vt.unsqueeze(1)

        filled = torch.clamp(filled + write_mask.squeeze(-1), 0.0, 1.0)
        ptr = torch.where(is_novel, (ptr + 1) % S, ptr)

    return torch.stack(outs, dim=0), k_bank, v_bank, filled, ptr


class _CheckpointedEpistemicSlotsFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q_seq: torch.Tensor,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        w_seq: torch.Tensor,
        n_slots: int,
        match_thresh: float,
        novel_thresh: float,
        read_scale: float,
        Bc: int,
    ) -> torch.Tensor:
        with torch.amp.autocast(q_seq.device.type, enabled=False):
            q_seq = q_seq.float()
            k_seq = k_seq.float()
            v_seq = v_seq.float()
            w_seq = w_seq.float()
            L, B, D = q_seq.shape
            n_blocks = (L + Bc - 1) // Bc

            k_starts = torch.empty(n_blocks, B, n_slots, D, device=q_seq.device, dtype=torch.float32)
            v_starts = torch.empty(n_blocks, B, n_slots, D, device=q_seq.device, dtype=torch.float32)
            f_starts = torch.empty(n_blocks, B, n_slots, device=q_seq.device, dtype=torch.float32)
            p_starts = torch.empty(n_blocks, B, device=q_seq.device, dtype=torch.long)

            kb = torch.zeros(B, n_slots, D, device=q_seq.device, dtype=torch.float32)
            vb = torch.zeros(B, n_slots, D, device=q_seq.device, dtype=torch.float32)
            fl = torch.zeros(B, n_slots, device=q_seq.device, dtype=torch.float32)
            pt = torch.zeros(B, device=q_seq.device, dtype=torch.long)
            out_all = torch.empty(L, B, D, device=q_seq.device, dtype=torch.float32)

            for bi in range(n_blocks):
                s = bi * Bc
                e = min(L, s + Bc)
                k_starts[bi].copy_(kb)
                v_starts[bi].copy_(vb)
                f_starts[bi].copy_(fl)
                p_starts[bi].copy_(pt)
                o_b, kb, vb, fl, pt = _slot_block_scan_jit(
                    kb, vb, fl, pt,
                    q_seq[s:e], k_seq[s:e], v_seq[s:e], w_seq[s:e],
                    match_thresh, novel_thresh, read_scale,
                )
                out_all[s:e].copy_(o_b)

        ctx.save_for_backward(q_seq, k_seq, v_seq, w_seq, k_starts, v_starts, f_starts, p_starts)
        ctx.match_thresh = match_thresh
        ctx.novel_thresh = novel_thresh
        ctx.read_scale = read_scale
        ctx.Bc = Bc
        return out_all

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q_seq, k_seq, v_seq, w_seq, k_starts, v_starts, f_starts, p_starts = ctx.saved_tensors
        match_thresh = ctx.match_thresh
        novel_thresh = ctx.novel_thresh
        read_scale = ctx.read_scale
        Bc = ctx.Bc
        L, B, D = q_seq.shape
        n_blocks = k_starts.size(0)
        n_slots = k_starts.size(2)

        dq = torch.zeros_like(q_seq)
        dk = torch.zeros_like(k_seq)
        dv = torch.zeros_like(v_seq)
        dw = torch.zeros_like(w_seq)
        g_kb = torch.zeros(B, n_slots, D, device=q_seq.device, dtype=torch.float32)
        g_vb = torch.zeros(B, n_slots, D, device=q_seq.device, dtype=torch.float32)

        with torch.amp.autocast(q_seq.device.type, enabled=False):
            grad_out = grad_out.float()
            for bi in range(n_blocks - 1, -1, -1):
                s = bi * Bc
                e = min(L, s + Bc)
                kb0 = k_starts[bi].detach().requires_grad_(True)
                vb0 = v_starts[bi].detach().requires_grad_(True)
                fl0 = f_starts[bi].detach()
                pt0 = p_starts[bi].detach()
                qb = q_seq[s:e].detach().requires_grad_(True)
                kb = k_seq[s:e].detach().requires_grad_(True)
                vb = v_seq[s:e].detach().requires_grad_(True)
                wb = w_seq[s:e].detach().requires_grad_(True)
                go_b = grad_out[s:e]

                with torch.enable_grad():
                    o_b, kb_end, vb_end, _, _ = _slot_block_scan_jit(
                        kb0, vb0, fl0, pt0,
                        qb, kb, vb, wb,
                        match_thresh, novel_thresh, read_scale,
                    )
                    g_kb, g_vb, g_qb, g_kb_in, g_vb_in, g_wb = torch.autograd.grad(
                        (o_b, kb_end, vb_end),
                        (kb0, vb0, qb, kb, vb, wb),
                        (go_b, g_kb, g_vb),
                    )
                dq[s:e].copy_(g_qb)
                dk[s:e].copy_(g_kb_in)
                dv[s:e].copy_(g_vb_in)
                dw[s:e].copy_(g_wb)

        return dq, dk, dv, dw, None, None, None, None, None


class EpistemicSlotMemory(nn.Module):
    """Fixed-size `O(1)` exact key-value slot bank controlled by `Woodbury` epistemic novelty."""

    def __init__(
        self,
        d_model: int,
        n_slots: int = 32,
        slot_dim: int = 64,
        match_thresh: float = 0.72,
        novel_thresh: float = 0.45,
        read_scale: float = 12.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_slots = int(n_slots)
        self.slot_dim = int(slot_dim)
        self.match_thresh = float(match_thresh)
        self.novel_thresh = float(novel_thresh)
        self.read_scale = float(read_scale)

        self.q_proj = nn.Linear(d_model, slot_dim, bias=False)
        self.k_proj = nn.Linear(d_model, slot_dim, bias=False)
        self.v_proj = nn.Linear(d_model, slot_dim, bias=False)
        self.write_proj = nn.Linear(d_model, 1, bias=True)
        nn.init.constant_(self.write_proj.bias, -0.1)
        self.novelty_scale = nn.Parameter(torch.tensor(2.0, dtype=torch.float32))
        self.novelty_bias = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))

        self.out_norm = RMSNorm(slot_dim)
        self.out_proj = nn.Linear(slot_dim, d_model, bias=False)
        with torch.no_grad():
            self.out_proj.weight.mul_(0.25)

    @staticmethod
    def _safe_unit(v: torch.Tensor) -> torch.Tensor:
        vf = v.float()
        return vf * torch.rsqrt(vf.pow(2).sum(dim=-1, keepdim=True) + 1e-6)

    def compute_state_bytes(self) -> int:
        return int(self.n_slots * (2 * self.slot_dim * 4 + 4) + 8)

    def forward(
        self,
        x: torch.Tensor,
        epistemic_uncert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        `x`: `(B, T, d_model)`, `epistemic_uncert`: `(B, T, H)` from `Woodbury` (optional).
        Returns `(B, T, d_model)` strictly causal exact-slot recall residual.
        """
        B, T, _ = x.shape
        q = self._safe_unit(self.q_proj(x))  # (B, T, D_s)
        k = self._safe_unit(self.k_proj(x))  # (B, T, D_s)
        v = self.v_proj(x).float()           # (B, T, D_s)

        w_logit = self.write_proj(x).float()  # (B, T, 1)
        if epistemic_uncert is not None:
            nov = epistemic_uncert.detach().float().mean(dim=-1, keepdim=True)
            w_logit = w_logit + self.novelty_scale.float() * (nov - self.novelty_bias.float())
        w_gate = torch.sigmoid(w_logit)

        q_seq = q.transpose(0, 1).contiguous()
        k_seq = k.transpose(0, 1).contiguous()
        v_seq = v.transpose(0, 1).contiguous()
        w_seq = w_gate.transpose(0, 1).contiguous()

        out_seq = _CheckpointedEpistemicSlotsFn.apply(
            q_seq,
            k_seq,
            v_seq,
            w_seq,
            self.n_slots,
            self.match_thresh,
            self.novel_thresh,
            self.read_scale,
            32,
        )
        out_bt = out_seq.transpose(0, 1).to(x.dtype)
        return self.out_proj(self.out_norm(out_bt))

    def init_stream_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> EpistemicSlotsStreamState:
        del dtype
        return EpistemicSlotsStreamState(
            k_bank=torch.zeros(batch_size, self.n_slots, self.slot_dim, device=device, dtype=torch.float32),
            v_bank=torch.zeros(batch_size, self.n_slots, self.slot_dim, device=device, dtype=torch.float32),
            filled=torch.zeros(batch_size, self.n_slots, device=device, dtype=torch.float32),
            ring_ptr=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    @torch.no_grad()
    def step(
        self,
        x_t: torch.Tensor,
        epistemic_uncert_t: torch.Tensor | None,
        state: EpistemicSlotsStreamState,
    ) -> torch.Tensor:
        """Executes ONE chunk step `(B, d_model)` with exact parity to `forward()`."""
        dt = x_t.dtype
        q_t = self._safe_unit(self.q_proj(x_t)).unsqueeze(0)  # (1, B, D_s)
        k_t = self._safe_unit(self.k_proj(x_t)).unsqueeze(0)  # (1, B, D_s)
        v_t = self.v_proj(x_t).float().unsqueeze(0)           # (1, B, D_s)

        w_logit = self.write_proj(x_t).float()  # (B, 1)
        if epistemic_uncert_t is not None:
            u = epistemic_uncert_t.float()
            if u.ndim == 3:
                u = u[:, -1]
            nov = u.mean(dim=-1, keepdim=True)
            w_logit = w_logit + self.novelty_scale.float() * (nov - self.novelty_bias.float())
        w_t = torch.sigmoid(w_logit).unsqueeze(0)  # (1, B, 1)

        out_1, state.k_bank, state.v_bank, state.filled, state.ring_ptr = _slot_block_scan_jit(
            state.k_bank,
            state.v_bank,
            state.filled,
            state.ring_ptr,
            q_t,
            k_t,
            v_t,
            w_t,
            self.match_thresh,
            self.novel_thresh,
            self.read_scale,
        )
        return self.out_proj(self.out_norm(out_1.squeeze(0).to(dt)))
