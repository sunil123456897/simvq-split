"""
Calcite v2 Live O(1) Streaming Inference Engine (`calcite_v2/stream_engine.py`).

Runs `CalciteV2LM` (both FP/BF16 and `FastPackedLUT2BitLinear` 2-bit uint8 T-MAC LUT)
with strictly constant `O(1)` memory independent of sequence length `L`:
- **Lazy `h_lex_t`:** `encode_byte_step(..., compute_lex=False)` skips `lex_norm + lex_up`
  on ~78% non-firing bytes and invokes `compute_lex_from_win8` only on chunk fires.
- **Exact Vectorized Chunked `prefill()`:** computes the exact final `CalciteStreamState`
  (all circular conv/KV rings, UTF-8 router FSM registers, Woodbury `(M, P)` + pending
  micro-chunk buffer, Crystal `s_bh` + refractory/pointer rings, EpistemicSlots bank,
  and multi-block `ByteInductionCache` dual-scale bank) in one parallel pass (`< 2e-5`
  parity vs byte-by-byte `step()`, strictly `O(1)` `memory_bytes()`).
- **Safe `B > 1` Unsynced Batch Stepping:** isolates per-item backbone progression so
  diverging UTF-8 chunk boundaries across batch items maintain `< 1e-5` parity.
- **Opt-In GPU Static CUDA Graphs (`use_cuda_graph=True`):** captures FP32-accumulating
  encoder, non-fire/fire decoder, and backbone commit-phase graphs for `B=1` CUDA streaming
  with zero host-side launch overhead and `< 2e-5` parity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

import kernels.fast_calcite_swa_chunker as _swa_chunker_mod
from kernels.fast_calcite_swa_chunker import (
    fast_swa_rope_attn_cuda,
    get_cached_rope_cos_sin,
)
from simvq_split.baselines import apply_rope
from simvq_split.chunker import _utf8_tensor_flags
from simvq_split.crystal import (
    _sphere_norm_jit,
    log_sinkhorn_6,
    project_birkhoff_to_perm_jit,
)

from .byte_induct_cache import ByteInductionCache, ByteInductionStreamState, _hash_2scale_seq
from .chunker import CalciteV2Chunker, ChunkerStreamState, _depthwise_conv4_shift_add
from .crystal import CalciteV2CrystalLayer, CrystalStreamState
from .epistemic_slots import (
    EpistemicSlotMemory,
    EpistemicSlotsStreamState,
    _scan_and_read_block,
)
from .model import CalciteV2LM, LocalSWABlock
from .norm import sliding_window_sdpa
from .woodbury import (
    CalciteV2WoodburyLayer,
    WoodburyStreamState,
    _deltanet2_m_block_scan_jit,
    _kalman_m_block_scan_jit,
)


def _swa_cuda_ext_ready() -> bool:
    loader = getattr(_swa_chunker_mod, "_load_swa_chunker_cuda_ext", None)
    return callable(loader) and (loader() is not None)


@torch.jit.script
def _compute_utf8_credits_jit(
    valid_cpu: torch.Tensor,  # (T,) bool CPU
    fire_cpu: torch.Tensor,   # (T,) bool CPU
    refill_int: int,
    cost_int: int,
    burst_int: int,
) -> int:
    """Exact leaky-bucket router credit register after `T` bytes (`step_idx == 0` skips debit)."""
    credits = burst_int
    T = int(valid_cpu.size(0))
    for t in range(1, T):
        if bool(valid_cpu[t].item()):
            credits = credits + refill_int
            if credits > burst_int:
                credits = burst_int
        if bool(fire_cpu[t].item()):
            credits = credits - cost_int
            if credits < 0:
                credits = 0
    return credits


@torch.jit.script
def _kalman_p_and_dual_with_final_jit(
    qe_m: torch.Tensor,   # (M_len, BH, d_k, C) fp32
    ke_m: torch.Tensor,   # (M_len, BH, d_k, 1) fp32
    k_m: torch.Tensor,    # (M_len, BH, d_k, 1) fp32
    b_m: torch.Tensor,    # (M_len, BH, 1, 1) fp32
    g_m: torch.Tensor,    # (M_len, BH, 1, 1) fp32
    inv_lam: float,
    tern_gate: float,
    kalman_mix: float,
    n_full: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Exact `P_t` scan matching `CalciteV2WoodburyLayer._kalman_p_and_dual_nograd_jit`,
    also returning `P_mat` immediately after `n_full` complete micro-chunk commits.
    """
    M_len, BH, d_k, C = qe_m.shape
    lam_reg = 1.0 / inv_lam
    rho_g = 0.002 * tern_gate
    rho_c = 0.002 * tern_gate
    slope_s = 18.0 + 2.0 * tern_gate

    P0 = (
        torch.eye(d_k, device=ke_m.device, dtype=torch.float32)
        .unsqueeze(0)
        .expand(BH, d_k, d_k)
        * inv_lam
    )
    P = P0.clone()
    P_final = P.clone()
    uncerts = torch.empty(M_len, BH, C, device=ke_m.device, dtype=torch.float32)
    k_dual = torch.empty(M_len, BH, d_k, 1, device=ke_m.device, dtype=torch.float32)

    qe_u = qe_m.unbind(0)
    ke_u = ke_m.unbind(0)
    k_u = k_m.unbind(0)
    b_u = b_m.unbind(0)
    g_u = g_m.unbind(0)

    for m in range(M_len):
        qem = qe_u[m]
        pq_m = torch.bmm(P, qem)
        uncerts[m] = ((qem * pq_m).sum(dim=1) * lam_reg).clamp(0.0, 1.5)

        kt = k_u[m]
        if kalman_mix > 0.0:
            pk_w = torch.bmm(P, kt)
            quad_w = (kt * pk_w).sum(dim=1, keepdim=True)
            norm_pk = torch.sqrt(pk_w.pow(2).sum(dim=1, keepdim=True) + 1e-8)
            denom_w = torch.maximum(quad_w, norm_pk * 0.4) + 1e-5
            k_rls = pk_w / denom_w
            k_dual[m] = (1.0 - kalman_mix) * kt + kalman_mix * k_rls
        else:
            k_dual[m] = kt

        if m < n_full:
            ket = ke_u[m]
            s_gate = torch.sigmoid((b_u[m] - 0.26) * slope_s)
            w_cov = s_gate * s_gate * s_gate
            pk = torch.bmm(P, ket)
            quad = (ket * pk).sum(dim=1, keepdim=True).clamp(min=1e-5)
            nov = w_cov * 0.92
            denom = 1.0 + nov * quad
            P_next = P - (nov / denom) * torch.bmm(pk, pk.transpose(1, 2))
            P_mid = 0.5 * (P_next + P_next.transpose(1, 2))
            if tern_gate > 0.0:
                rho_rec = ((1.0 - g_u[m]) * w_cov * rho_g).clamp(0.0, rho_c)
                P = (1.0 - rho_rec) * P_mid + rho_rec * P0
                P = 0.5 * (P + P.transpose(1, 2))
            else:
                P = P_mid
            if m == n_full - 1:
                P_final = P.clone()

    return uncerts, k_dual, P_final


def _pad_tail_chronological(x: torch.Tensor, length: int, pad_value: float = 0.0) -> torch.Tensor:
    """Extracts last `length` entries along dim 1, left-padding with `pad_value` if `T < length`."""
    T = x.size(1)
    if length == 0:
        return x[:, :0]
    if T >= length:
        return x[:, T - length : T].contiguous()
    pad_n = length - T
    pad_shape = list(x.shape)
    pad_shape[1] = pad_n
    pad_t = torch.full(pad_shape, pad_value, device=x.device, dtype=x.dtype)
    return torch.cat((pad_t, x), dim=1).contiguous()


def _canonicalize_circular_ring(ring: torch.Tensor, ptr: int) -> None:
    """Rotates a `(B, K, ...)` circular ring in-place so chronological order starts at index `0`."""
    K = ring.size(1)
    if K <= 1 or ptr == 0:
        return
    p = ptr % K
    if p == 0:
        return
    ordered = torch.cat((ring[:, p:], ring[:, :p]), dim=1).contiguous()
    ring.copy_(ordered)


def _canonicalize_swa_ring(ring: torch.Tensor, pos: int) -> None:
    """Rotates a `(B, W, H, D)` `pos % W` SWA ring in-place into chronological `[pos-W .. pos-1]` order."""
    W = ring.size(1)
    if W <= 1 or pos < W:
        return
    oldest = pos % W
    if oldest == 0:
        return
    ordered = torch.cat((ring[:, oldest:], ring[:, :oldest]), dim=1).contiguous()
    ring.copy_(ordered)


def _uncanonicalize_swa_ring(ring: torch.Tensor, pos: int) -> None:
    """Inverse of `_canonicalize_swa_ring`: restores `pos % W` circular slot layout for eager `step()`."""
    W = ring.size(1)
    if W <= 1 or pos < W:
        return
    shift = pos % W
    if shift == 0:
        return
    ring.copy_(torch.roll(ring, shifts=shift, dims=1).contiguous())


def _shift_ring_append_graph_safe(ring: torch.Tensor, new_val: torch.Tensor) -> None:
    """Graph-safe non-overlapping chronological shift: `ring[:, :-1] <- ring[:, 1:]`, `ring[:, -1] <- new_val`."""
    K = ring.size(1)
    if K == 0:
        return
    if K > 1:
        tmp = ring[:, 1:].clone()
        ring[:, :-1].copy_(tmp)
    ring[:, -1].copy_(new_val)


def _dw_conv4_graph_step(
    x_t: torch.Tensor,
    ring: torch.Tensor,
    conv: torch.nn.Conv1d,
    dilation: int,
) -> torch.Tensor:
    """Graph-safe chronological 4-tap depthwise conv step (fixed indices, zero Python ptr state)."""
    d = int(dilation)
    dt = x_t.dtype
    w = conv.weight.squeeze(1)
    if w.dtype != dt:
        w = w.to(dt)
    c = (
        ring[:, 0] * w[:, 0]
        + ring[:, d] * w[:, 1]
        + ring[:, 2 * d] * w[:, 2]
        + x_t * w[:, 3]
    )
    if conv.bias is not None:
        b = conv.bias if conv.bias.dtype == dt else conv.bias.to(dt)
        c = c + b
    _shift_ring_append_graph_safe(ring, x_t)
    return c


@dataclass
class CalciteStreamState:
    chunker: ChunkerStreamState
    swa_k_rings: List[torch.Tensor]        # per SWA block: (B, W, n_heads, head_dim)
    swa_v_rings: List[torch.Tensor]        # per SWA block: (B, W, n_heads, head_dim)
    swa_canon_rings: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    woodbury_fast: Optional[WoodburyStreamState]
    woodbury_slow: WoodburyStreamState
    epistemic_slots: Optional[EpistemicSlotsStreamState]
    crystal_1: CrystalStreamState
    crystal_2: CrystalStreamState
    fb_prev_z_hat: Optional[torch.Tensor]  # (B, d_model) if use_latent_feedback
    tr_carry_B: Optional[torch.Tensor]     # (B, d_model) if use_temporal_residual
    chunk_pos: int
    total_bytes: int
    backbone_ticks: int
    # Per-item isolated states for unsynced `B > 1` batches (None for `B == 1`)
    item_states: Optional[List["CalciteStreamState"]] = None

    def memory_bytes(self) -> int:
        """Exact live allocated state bytes per batch item (strictly O(1) vs `total_bytes`)."""
        if self.item_states is not None:
            return sum(s.memory_bytes() for s in self.item_states) // max(1, len(self.item_states))
        b = max(1, int(self.swa_k_rings[0].size(0))) if self.swa_k_rings else 1
        total = self.chunker.memory_bytes()
        for kr, vr in zip(self.swa_k_rings, self.swa_v_rings):
            total += (kr.numel() * kr.element_size() + vr.numel() * vr.element_size()) // b
        if self.swa_canon_rings is not None:
            for ca, cm in self.swa_canon_rings:
                total += (ca.numel() * ca.element_size() + cm.numel() * cm.element_size()) // b
        if self.woodbury_fast is not None:
            total += self.woodbury_fast.memory_bytes()
        total += self.woodbury_slow.memory_bytes()
        if self.epistemic_slots is not None:
            total += self.epistemic_slots.memory_bytes()
        total += self.crystal_1.memory_bytes()
        total += self.crystal_2.memory_bytes()
        if self.fb_prev_z_hat is not None:
            total += (self.fb_prev_z_hat.numel() * self.fb_prev_z_hat.element_size()) // b
        if self.tr_carry_B is not None:
            total += (self.tr_carry_B.numel() * self.tr_carry_B.element_size()) // b
        return int(total)


@dataclass
class _CudaGraphBundle:
    state_id: int
    device: torch.device
    dtype: torch.dtype
    max_byte_pos: int
    max_chunk_pos: int
    byte_cos: Optional[torch.Tensor]
    byte_sin: Optional[torch.Tensor]
    chunk_cos: torch.Tensor
    chunk_sin: torch.Tensor
    # Static device I/O & scalar counters mutated in-place per replay
    tok_in: torch.Tensor           # (1,) int64
    pos_b: torch.Tensor            # (1,) int64
    pos_c: torch.Tensor            # (1,) int64
    c1_sc: torch.Tensor            # (1,) fp32
    c2_sc: torch.Tensor            # (1,) fp32
    wf_sc: Optional[torch.Tensor]  # (1,) fp32
    ws_sc: torch.Tensor            # (1,) fp32
    # Static intermediate tensors shared across graphs
    fire_out: torch.Tensor         # (1,) bool
    p_out: torch.Tensor            # (1,) fp32
    h_byte_out: torch.Tensor       # (1, D)
    c_norm_out: torch.Tensor       # (1, 8*d_byte)
    win8_out: torch.Tensor         # (1, 8*d_byte)
    offset_out: torch.Tensor       # (1,) int64
    z_hat_out: torch.Tensor        # (1, D)
    logits_out: torch.Tensor       # (1, V)
    # Captured CUDA Graphs
    g_enc: torch.cuda.CUDAGraph
    g_dec_nofire: torch.cuda.CUDAGraph
    g_dec_fire: torch.cuda.CUDAGraph
    g_bb_phases: Dict[Tuple[int, int, int, int], torch.cuda.CUDAGraph]


class CalciteStreamEngine:
    """Zero-allocation-growth `O(1)` byte-streaming engine for `CalciteV2LM` (FP or 2-bit LUT)."""

    def __init__(self, model: CalciteV2LM, use_cuda_graph: bool = False):
        self.model = model
        self.cfg = model.cfg
        self.use_cuda_graph = bool(use_cuda_graph)
        self._cg_bundle: Optional[_CudaGraphBundle] = None

    def enable_cuda_graph(
        self, enabled: bool = True, state: Optional[CalciteStreamState] = None
    ) -> None:
        """Toggles opt-in `B=1` GPU static CUDA Graphs (FP32 accumulation, `< 2e-5` parity)."""
        self.use_cuda_graph = bool(enabled)
        if not self.use_cuda_graph:
            if state is not None and self._cg_bundle is not None and self._cg_bundle.state_id == id(state):
                self._restore_rings_for_eager(state)
            self._cg_bundle = None

    @staticmethod
    def _restore_rings_for_eager(state: CalciteStreamState) -> None:
        """Restores `pos % W` circular SWA KV ring layout when exiting CUDA Graph mode."""
        cs = state.chunker
        if cs.ring_byte_k.size(1) > 0:
            _uncanonicalize_swa_ring(cs.ring_byte_k, cs.step_idx)
            _uncanonicalize_swa_ring(cs.ring_byte_v, cs.step_idx)
        for kr, vr in zip(state.swa_k_rings, state.swa_v_rings):
            _uncanonicalize_swa_ring(kr, state.chunk_pos)
            _uncanonicalize_swa_ring(vr, state.chunk_pos)

    def init_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> CalciteStreamState:
        dev = device or self.model.tok_emb.weight.device
        dt = dtype or self.model.tok_emb.weight.dtype
        if batch_size > 1:
            items = [self.init_state(1, dev, dt) for _ in range(batch_size)]
            first = items[0]
            return CalciteStreamState(
                chunker=first.chunker,
                swa_k_rings=first.swa_k_rings,
                swa_v_rings=first.swa_v_rings,
                swa_canon_rings=first.swa_canon_rings,
                woodbury_fast=first.woodbury_fast,
                woodbury_slow=first.woodbury_slow,
                epistemic_slots=first.epistemic_slots,
                crystal_1=first.crystal_1,
                crystal_2=first.crystal_2,
                fb_prev_z_hat=first.fb_prev_z_hat,
                tr_carry_B=first.tr_carry_B,
                chunk_pos=0,
                total_bytes=0,
                backbone_ticks=0,
                item_states=items,
            )

        swa_blocks = self.model._all_swa_blocks()
        W = self.cfg.window_size
        H = self.cfg.n_heads
        Dh = self.cfg.d_model // H
        k_rings = [torch.zeros(1, W, H, Dh, device=dev, dtype=dt) for _ in swa_blocks]
        v_rings = [torch.zeros(1, W, H, Dh, device=dev, dtype=dt) for _ in swa_blocks]
        canon_rings = (
            [
                (
                    torch.zeros(1, 3, self.cfg.d_model, device=dev, dtype=dt),
                    torch.zeros(1, 3, self.cfg.d_model, device=dev, dtype=dt),
                )
                for _ in swa_blocks
            ]
            if self.cfg.use_canon_chunks
            else None
        )

        wf_state = (
            self.model.woodbury_fast.init_stream_state(1, dev, dt)
            if self.model.woodbury_fast is not None
            else None
        )
        ws_state = self.model.woodbury.init_stream_state(1, dev, dt)
        es_state = (
            self.model.epistemic_slots.init_stream_state(1, dev, dt)
            if self.model.epistemic_slots is not None
            else None
        )
        c1_state = self.model.crystal.init_stream_state(1, dev, dt)
        c2_state = self.model.crystal_2.init_stream_state(1, dev, dt)
        fb_prev = (
            torch.zeros(1, self.cfg.d_model, device=dev, dtype=dt)
            if self.cfg.use_latent_feedback
            else None
        )
        tr_carry = (
            torch.zeros(1, self.cfg.d_model, device=dev, dtype=dt)
            if self.cfg.use_temporal_residual
            else None
        )
        return CalciteStreamState(
            chunker=self.model.byte_chunker.init_stream_state(1, dev, dt),
            swa_k_rings=k_rings,
            swa_v_rings=v_rings,
            swa_canon_rings=canon_rings,
            woodbury_fast=wf_state,
            woodbury_slow=ws_state,
            epistemic_slots=es_state,
            crystal_1=c1_state,
            crystal_2=c2_state,
            fb_prev_z_hat=fb_prev,
            tr_carry_B=tr_carry,
            chunk_pos=0,
            total_bytes=0,
            backbone_ticks=0,
            item_states=None,
        )

    # ------------------------------------------------------------------
    # Exact Incremental Single-Byte Step (Lazy h_lex + Safe B > 1)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _step_backbone(
        self,
        z_in_t: torch.Tensor,
        h_lex_t: torch.Tensor,
        state: CalciteStreamState,
    ) -> torch.Tensor:
        """Executes ONE chunk step through the full `SWA + Woodbury + Crystal` backbone."""
        m = self.model
        pos = state.chunk_pos

        if (
            self.cfg.use_latent_feedback
            and m.fb_norm is not None
            and m.fb_proj is not None
            and state.fb_prev_z_hat is not None
        ):
            if pos > 0:
                h = z_in_t + m.fb_proj(m.fb_norm(state.fb_prev_z_hat))
            else:
                h = z_in_t
        else:
            h = z_in_t

        ring_idx = 0
        if self.cfg.topology == "interleaved" and m.swa_stages is not None:
            use_dca = m.dca_mixers is not None
            hist: List[torch.Tensor] = [h] if use_dca else []
            dca_idx = 0

            def _dca_mix_il(h_cur: torch.Tensor) -> torch.Tensor:
                nonlocal dca_idx
                if use_dca and m.dca_mixers is not None and len(hist) >= 2:
                    h_cur = m.dca_mixers[dca_idx](h_cur, hist)
                    dca_idx += 1
                return h_cur

            for blk in m.swa_stages[0]:
                h = _dca_mix_il(h)
                c_rng = state.swa_canon_rings[ring_idx] if state.swa_canon_rings is not None else None
                h, state.swa_k_rings[ring_idx], state.swa_v_rings[ring_idx] = blk.step(
                    h, state.swa_k_rings[ring_idx], state.swa_v_rings[ring_idx], pos, canon_rings=c_rng
                )
                ring_idx += 1
                if use_dca:
                    hist.append(h)
            h = _dca_mix_il(h)
            if (
                m.woodbury_fast is not None
                and m.woodbury_fast_norm is not None
                and m.woodbury_fast_gate is not None
                and state.woodbury_fast is not None
            ):
                wf_in = m.woodbury_fast_norm(h)
                wf_g = 2.0 * torch.sigmoid(m.woodbury_fast_gate(wf_in))
                h = h + wf_g * m.woodbury_fast.step(wf_in, h_lex_t, state.woodbury_fast)
            if use_dca:
                hist.append(h)

            for blk in m.swa_stages[1]:
                h = _dca_mix_il(h)
                c_rng = state.swa_canon_rings[ring_idx] if state.swa_canon_rings is not None else None
                h, state.swa_k_rings[ring_idx], state.swa_v_rings[ring_idx] = blk.step(
                    h, state.swa_k_rings[ring_idx], state.swa_v_rings[ring_idx], pos, canon_rings=c_rng
                )
                ring_idx += 1
                if use_dca:
                    hist.append(h)
            h = _dca_mix_il(h)
            if m.cryst1_norm is not None and m.cryst1_gate is not None:
                c1_in = m.cryst1_norm(h)
                c1_g = 2.0 * torch.sigmoid(m.cryst1_gate(c1_in))
                h = h + c1_g * m.crystal.step(c1_in, h_lex_t, state.crystal_1)
            if use_dca:
                hist.append(h)

            for blk in m.swa_stages[2]:
                h = _dca_mix_il(h)
                c_rng = state.swa_canon_rings[ring_idx] if state.swa_canon_rings is not None else None
                h, state.swa_k_rings[ring_idx], state.swa_v_rings[ring_idx] = blk.step(
                    h, state.swa_k_rings[ring_idx], state.swa_v_rings[ring_idx], pos, canon_rings=c_rng
                )
                ring_idx += 1
                if use_dca:
                    hist.append(h)
            h = _dca_mix_il(h)
            if m.woodbury_slow_gate is not None:
                ws_in = m.split_norm(h)
                ws_g = 2.0 * torch.sigmoid(m.woodbury_slow_gate(ws_in))
                h = h + ws_g * m.woodbury.step(ws_in, h_lex_t, state.woodbury_slow)
                if (
                    m.epistemic_slots is not None
                    and m.epistemic_slots_gate is not None
                    and state.epistemic_slots is not None
                ):
                    es_g = 2.0 * torch.sigmoid(m.epistemic_slots_gate(ws_in))
                    h = h + es_g * m.epistemic_slots.step(
                        ws_in, m.woodbury.last_epistemic_uncertainty, state.epistemic_slots, h_lex_t=h_lex_t
                    )
            if use_dca:
                hist.append(h)
            h = _dca_mix_il(h)
            h = h + m.mid_mlp(m.mid_mlp_norm(h))
            if use_dca:
                hist.append(h)

            h = _dca_mix_il(h)
            c2_in = m.cryst2_norm(h)
            c2_g = 2.0 * torch.sigmoid(m.cryst2_gate(c2_in))
            h = h + c2_g * m.crystal_2.step(c2_in, h_lex_t, state.crystal_2)
            if use_dca:
                hist.append(h)

            for blk in m.swa_stages[3]:
                h = _dca_mix_il(h)
                c_rng = state.swa_canon_rings[ring_idx] if state.swa_canon_rings is not None else None
                h, state.swa_k_rings[ring_idx], state.swa_v_rings[ring_idx] = blk.step(
                    h, state.swa_k_rings[ring_idx], state.swa_v_rings[ring_idx], pos, canon_rings=c_rng
                )
                ring_idx += 1
                if use_dca:
                    hist.append(h)
            h = _dca_mix_il(h)
        else:
            use_ar = m.attnres_mixers is not None
            use_dca = m.dca_mixers is not None
            use_tr = (
                self.cfg.use_temporal_residual
                and state.tr_carry_B is not None
                and m.tr_rA is not None
                and m.tr_rB is not None
                and m.tr_sB is not None
                and m.tr_phi is not None
            )
            b_stream = (
                (m.tr_phi(state.tr_carry_B) if pos > 0 else torch.zeros_like(h))
                if use_tr and state.tr_carry_B is not None and m.tr_phi is not None
                else None
            )
            tr_idx = 0

            def _tr_step_stream(
                a_cur: torch.Tensor, b_cur: Optional[torch.Tensor], y_fn
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
                nonlocal tr_idx
                if (
                    use_tr
                    and b_cur is not None
                    and m.tr_rA is not None
                    and m.tr_rB is not None
                    and m.tr_sB is not None
                ):
                    u = m.tr_rA[tr_idx].to(a_cur.dtype) * a_cur + m.tr_rB[tr_idx].to(a_cur.dtype) * b_cur
                    y = y_fn(u)
                    a_next = a_cur + y
                    b_next = b_cur + m.tr_sB[tr_idx].to(a_cur.dtype) * y
                    tr_idx += 1
                    return a_next, b_next
                return a_cur + y_fn(a_cur), b_cur

            def _step_swa_blk(
                blk: LocalSWABlock, r_i: int, a_cur: torch.Tensor, b_cur: Optional[torch.Tensor]
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
                c_r = state.swa_canon_rings[r_i] if state.swa_canon_rings is not None else None
                if use_tr:
                    def _attn_fn(u: torch.Tensor) -> torch.Tensor:
                        x1 = blk.norm1(u)
                        if blk.canon_attn is not None and c_r is not None:
                            x1, nxt_ca = blk._canon_step(x1, blk.canon_attn, c_r[0])
                            c_r[0].copy_(nxt_ca)
                        ao, kr, vr = blk.attn.step(x1, state.swa_k_rings[r_i], state.swa_v_rings[r_i], pos)
                        state.swa_k_rings[r_i] = kr
                        state.swa_v_rings[r_i] = vr
                        return ao

                    def _mlp_fn(u: torch.Tensor) -> torch.Tensor:
                        x2 = blk.norm2(u)
                        if blk.canon_mlp is not None and c_r is not None:
                            x2, nxt_cm = blk._canon_step(x2, blk.canon_mlp, c_r[1])
                            c_r[1].copy_(nxt_cm)
                        return blk.mlp(x2)

                    a_cur, b_cur = _tr_step_stream(a_cur, b_cur, _attn_fn)
                    a_cur, b_cur = _tr_step_stream(a_cur, b_cur, _mlp_fn)
                    return a_cur, b_cur
                a_out, kr, vr = blk.step(
                    a_cur, state.swa_k_rings[r_i], state.swa_v_rings[r_i], pos, canon_rings=c_r
                )
                state.swa_k_rings[r_i] = kr
                state.swa_v_rings[r_i] = vr
                return a_out, b_cur

            hist = [h] if (use_ar or use_dca) else []
            dca_idx = 0

            def _dca_mix_hg(h_cur: torch.Tensor) -> torch.Tensor:
                nonlocal dca_idx
                if use_dca and m.dca_mixers is not None and len(hist) >= 2:
                    h_cur = m.dca_mixers[dca_idx](h_cur, hist)
                    dca_idx += 1
                return h_cur

            for blk in m.cortex_pre:
                if use_dca:
                    h = _dca_mix_hg(h)
                h, b_stream = _step_swa_blk(blk, ring_idx, h, b_stream)
                ring_idx += 1
                if use_ar or use_dca:
                    hist.append(h)

            def _long_mem_step(u_in: torch.Tensor) -> torch.Tensor:
                h_split = m.split_norm(u_in)
                c_out = m.crystal.step(h_split, h_lex_t, state.crystal_1)
                w_out = m.woodbury.step(h_split, h_lex_t, state.woodbury_slow)
                if m.stream_gate is not None:
                    gates = 2.0 * torch.sigmoid(m.stream_gate(h_split))
                    delta_l = gates[..., 0:1] * c_out + gates[..., 1:2] * w_out
                else:
                    delta_l = c_out + w_out
                if (
                    m.epistemic_slots is not None
                    and m.epistemic_slots_gate is not None
                    and state.epistemic_slots is not None
                ):
                    es_g = 2.0 * torch.sigmoid(m.epistemic_slots_gate(h_split))
                    delta_l = delta_l + es_g * m.epistemic_slots.step(
                        h_split, m.woodbury.last_epistemic_uncertainty, state.epistemic_slots, h_lex_t=h_lex_t
                    )
                return delta_l

            if use_dca:
                h = _dca_mix_hg(h)
                h, b_stream = _tr_step_stream(h, b_stream, _long_mem_step)
                hist.append(h)
            elif use_ar and m.attnres_mixers is not None:
                h_long_in = m.attnres_mixers[0](h, hist)
                if (
                    use_tr
                    and b_stream is not None
                    and m.tr_rA is not None
                    and m.tr_rB is not None
                    and m.tr_sB is not None
                ):
                    u = m.tr_rA[tr_idx].to(h.dtype) * h_long_in + m.tr_rB[tr_idx].to(h.dtype) * b_stream
                    y = _long_mem_step(u)
                    h = h + y
                    b_stream = b_stream + m.tr_sB[tr_idx].to(h.dtype) * y
                    tr_idx += 1
                else:
                    h = h + _long_mem_step(h_long_in)
                hist.append(h)
            else:
                h, b_stream = _tr_step_stream(h, b_stream, _long_mem_step)

            def _mid_mlp_step(u_in: torch.Tensor) -> torch.Tensor:
                return m.mid_mlp(m.mid_mlp_norm(u_in))

            if use_dca:
                h = _dca_mix_hg(h)
                h, b_stream = _tr_step_stream(h, b_stream, _mid_mlp_step)
                hist.append(h)
            elif use_ar and m.attnres_mixers is not None:
                h_mid_in = m.attnres_mixers[1](h, hist)
                if (
                    use_tr
                    and b_stream is not None
                    and m.tr_rA is not None
                    and m.tr_rB is not None
                    and m.tr_sB is not None
                ):
                    u = m.tr_rA[tr_idx].to(h.dtype) * h_mid_in + m.tr_rB[tr_idx].to(h.dtype) * b_stream
                    y = _mid_mlp_step(u)
                    h = h + y
                    b_stream = b_stream + m.tr_sB[tr_idx].to(h.dtype) * y
                    tr_idx += 1
                else:
                    h = h + _mid_mlp_step(h_mid_in)
                hist.append(h)
            else:
                h, b_stream = _tr_step_stream(h, b_stream, _mid_mlp_step)

            def _c2_step(u_in: torch.Tensor) -> torch.Tensor:
                c2_in = m.cryst2_norm(u_in)
                return (2.0 * torch.sigmoid(m.cryst2_gate(c2_in))) * m.crystal_2.step(
                    c2_in, h_lex_t, state.crystal_2
                )

            if use_dca:
                h = _dca_mix_hg(h)
                h, b_stream = _tr_step_stream(h, b_stream, _c2_step)
                hist.append(h)
            elif use_ar and m.attnres_mixers is not None:
                h_c2_in = m.attnres_mixers[2](h, hist)
                if (
                    use_tr
                    and b_stream is not None
                    and m.tr_rA is not None
                    and m.tr_rB is not None
                    and m.tr_sB is not None
                ):
                    u = m.tr_rA[tr_idx].to(h.dtype) * h_c2_in + m.tr_rB[tr_idx].to(h.dtype) * b_stream
                    y = _c2_step(u)
                    h = h + y
                    b_stream = b_stream + m.tr_sB[tr_idx].to(h.dtype) * y
                    tr_idx += 1
                else:
                    h = h + _c2_step(h_c2_in)
                hist.append(h)
            else:
                h, b_stream = _tr_step_stream(h, b_stream, _c2_step)

            for pi, blk in enumerate(m.cortex_post):
                if use_dca:
                    h = _dca_mix_hg(h)
                elif use_ar and m.attnres_mixers is not None:
                    h = m.attnres_mixers[3 + pi](h, hist)
                h, b_stream = _step_swa_blk(blk, ring_idx, h, b_stream)
                ring_idx += 1
                if use_ar or use_dca:
                    hist.append(h)

            if use_dca:
                h = _dca_mix_hg(h)
            elif use_ar and m.attnres_mixers is not None:
                h = m.attnres_mixers[-1](h, hist)
            if use_tr and b_stream is not None:
                state.tr_carry_B = b_stream.clone()

        if state.fb_prev_z_hat is not None:
            state.fb_prev_z_hat = h.clone()
        state.chunk_pos += 1
        state.backbone_ticks += 1
        return h

    @torch.no_grad()
    def step(
        self,
        token_t: torch.Tensor,
        state: CalciteStreamState,
    ) -> Tuple[torch.Tensor, CalciteStreamState]:
        """
        Processes ONE byte `token_t` `(B,)` and returns `(logits_t, state)` with shape `(B, vocab_size)`.
        - Uses lazy `h_lex_t` (`compute_lex=False`), skipping `lex_up` on ~78% non-firing bytes.
        - Isolates per-item backbone states for `B > 1` so unsynced boundaries stay `< 1e-5` exact.
        - Uses opt-in static CUDA Graphs for `B == 1` on CUDA when `use_cuda_graph=True`.
        """
        if token_t.ndim == 0:
            token_t = token_t.unsqueeze(0)

        if state.item_states is not None:
            B = token_t.size(0)
            if B != len(state.item_states):
                raise ValueError(f"Batch size mismatch: token_t has B={B}, state has {len(state.item_states)}")
            lg_items = []
            for b in range(B):
                lg_b, _ = self.step(token_t[b : b + 1], state.item_states[b])
                lg_items.append(lg_b)
            state.total_bytes = state.item_states[0].total_bytes
            state.backbone_ticks = sum(s.backbone_ticks for s in state.item_states)
            state.chunk_pos = max(s.chunk_pos for s in state.item_states)
            return torch.cat(lg_items, dim=0), state

        if (
            self.use_cuda_graph
            and token_t.is_cuda
            and token_t.size(0) == 1
            and self._can_use_cuda_graph(state)
        ):
            return self._step_cuda_graph(token_t, state)

        e_tok_t = self.model.tok_emb(token_t)
        fire_t, p_t, h_byte_t, c_norm_t, win8_t, offset_t, _ = (
            self.model.byte_chunker.encode_byte_step(
                token_t, e_tok_t, state.chunker, compute_lex=False
            )
        )

        # Heavy backbone AND `lex_norm+lex_up` execute ONLY when a chunk boundary fires!
        if bool(fire_t.any().item()):
            h_lex_t = self.model.byte_chunker.compute_lex_from_win8(win8_t)
            z_hat_new = self._step_backbone(h_byte_t, h_lex_t, state)
        else:
            z_hat_new = None

        h_out = self.model.byte_chunker.decode_byte_step(
            z_hat_new, fire_t, p_t, h_byte_t, c_norm_t, offset_t, state.chunker
        )
        logits = self.model.lm_head(self.model.norm_f(h_out))
        state.total_bytes += 1
        return logits, state

    # ------------------------------------------------------------------
    # Exact Vectorized Chunked Prefill (`< 2e-5` Parity, Strict O(1) RAM)
    # ------------------------------------------------------------------

    @staticmethod
    def _swa_block_prefill_with_state(
        blk: LocalSWABlock,
        h: torch.Tensor,               # (1, M, D)
        k_ring: torch.Tensor,          # (1, W, H, Dh)
        v_ring: torch.Tensor,          # (1, W, H, Dh)
        canon_rings: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        M = h.size(1)
        dev = h.device
        x1 = blk.norm1(h)
        if blk.canon_attn is not None:
            if canon_rings is not None:
                canon_rings[0].copy_(_pad_tail_chronological(x1, 3))
            x1 = x1 + blk.canon_attn(F.pad(x1.transpose(1, 2), (3, 0))).transpose(1, 2)

        attn = blk.attn
        H = attn.n_heads
        Dh = attn.head_dim
        W = attn.window_size
        qkv = attn.qkv(x1).view(1, M, 3, H, Dh)
        cos_tab, sin_tab = get_cached_rope_cos_sin(M, dev, qkv.dtype, head_dim=Dh, base=10000.0)

        if (
            attn.use_fast_kernels
            and qkv.is_cuda
            and Dh in (32, 64)
            and 1 <= W <= 32
            and _swa_cuda_ext_ready()
        ):
            y = fast_swa_rope_attn_cuda(qkv, cos_tab, sin_tab, W).view(1, M, attn.d_model)
            t_start = max(0, M - W)
            n_keep = M - t_start
            k_tail = apply_rope(
                qkv[:, t_start:, 1],
                cos_tab[t_start:M].view(1, n_keep, 1, -1),
                sin_tab[t_start:M].view(1, n_keep, 1, -1),
            )
            v_tail = qkv[:, t_start:, 2]
        else:
            q = qkv[:, :, 0].transpose(1, 2)
            k = qkv[:, :, 1].transpose(1, 2)
            v = qkv[:, :, 2].transpose(1, 2)
            cos_4d = cos_tab.view(1, 1, M, -1)
            sin_4d = sin_tab.view(1, 1, M, -1)
            q = apply_rope(q, cos_4d, sin_4d)
            k = apply_rope(k, cos_4d, sin_4d)
            y = sliding_window_sdpa(
                q, k, v, W, attn._swa_mask0, attn._swa_mask_inner, attn._swa_block_q
            ).transpose(1, 2).contiguous().view(1, M, attn.d_model)
            t_start = max(0, M - W)
            k_tail = k[:, :, t_start:].transpose(1, 2)
            v_tail = v[:, :, t_start:].transpose(1, 2)

        slots = torch.arange(t_start, M, device=dev, dtype=torch.long) % W
        k_ring.zero_()
        v_ring.zero_()
        k_ring[:, slots] = k_tail
        v_ring[:, slots] = v_tail

        h = h + attn.out_proj(y)
        x2 = blk.norm2(h)
        if blk.canon_mlp is not None:
            if canon_rings is not None:
                canon_rings[1].copy_(_pad_tail_chronological(x2, 3))
            x2 = x2 + blk.canon_mlp(F.pad(x2.transpose(1, 2), (3, 0))).transpose(1, 2)
        return h + blk.mlp(x2)

    @staticmethod
    def _woodbury_prefill_with_state(
        wood: CalciteV2WoodburyLayer,
        x: torch.Tensor,              # (1, M, D)
        h_lex: Optional[torch.Tensor],
        state: WoodburyStreamState,
    ) -> torch.Tensor:
        """Vectorized `CalciteV2WoodburyLayer` pass populating exact `WoodburyStreamState`."""
        B, T, D_in = x.shape
        dt = x.dtype
        dev = x.device
        C = wood.C

        with torch.amp.autocast(dev.type, enabled=False):
            x_u = wood._safe_unit(x)
            if h_lex is not None:
                w_lex = torch.sigmoid(wood.lex_logit.float())
                e_u = wood._safe_unit(h_lex)
                z = wood._safe_unit(w_lex * e_u + (1.0 - w_lex) * x_u)
            else:
                z = x_u

            steps = torch.arange(1, T + 1, device=dev, dtype=torch.float32).view(1, T, 1)
            c_mean = torch.cumsum(z, dim=1) / (steps + 2.0)
            zc = (z - c_mean).to(dt)

            z_head_rep = zc[:, :1].expand(B, 3, D_in)
            zp = torch.cat((z_head_rep, zc), dim=1)
            z0 = zp[:, 3:]
            z12 = zp[:, 2:-1] + zp[:, 1:-2]
            z3 = zp[:, 0:-3] + 0.25 * zp[:, 1:-2]

            sc_raw = math.sqrt(wood.D)
            f1_r = wood._safe_unit(wood.f1_proj(z0).view(B, T, wood.H_epi, wood.D), sc_raw)
            f2_r = wood._safe_unit(wood.f2_proj(z12).view(B, T, wood.H_epi, wood.D), sc_raw)
            f3_r = wood._safe_unit(wood.f3_proj(z3).view(B, T, wood.H_epi, wood.D), sc_raw)
            phi_epi = wood._safe_unit(
                torch.tanh(f1_r * 1.2) * torch.tanh(f2_r * 1.2) * torch.tanh(f3_r * 1.2)
            )
            phi_lin = wood._safe_unit(f1_r + f2_r + f3_r)
            phi_rw = wood._safe_unit(0.35 * phi_lin + 0.94 * phi_epi)

            K_l = wood.L_latch
            sal = wood.subj_salience(x).float()
            sal_pad = F.pad(sal, (0, 0, K_l - 1, 0), value=-1e4)
            sal_win = sal_pad.unfold(1, K_l, 1)
            att = F.softmax(sal_win + wood.subj_lag_bias.float().view(1, 1, wood.H_epi, K_l), dim=-1)
            phi_both = torch.cat((phi_rw, phi_epi), dim=-1)
            both_win = (
                F.pad(phi_both.reshape(B, T, wood.epi_dim * 2), (0, 0, K_l - 1, 0))
                .unfold(1, K_l, 1)
                .view(B, T, wood.H_epi, 2 * wood.D, K_l)
            )
            both_agg = torch.matmul(both_win, att.unsqueeze(-1)).squeeze(-1)
            rw_agg, epi_agg = both_agg.chunk(2, dim=-1)
            q_rw_t = wood._safe_unit(rw_agg).to(dt)
            q_epi_t = wood._safe_unit(epi_agg).to(dt)

            # Populate circular causal feature rings (chronological, ptr == 0)
            state.z_cumsum.copy_(z.sum(dim=1))
            state.step_count = T
            state.z_ring3.copy_(zp[:, -3:])
            state.z3_ptr = 0
            km1 = max(0, K_l - 1)
            if km1 > 0:
                state.sal_ring.copy_(_pad_tail_chronological(sal, km1, pad_value=-1e4))
                state.rw_ring.copy_(_pad_tail_chronological(phi_rw, km1, pad_value=0.0))
                state.epi_ring.copy_(_pad_tail_chronological(phi_epi, km1, pad_value=0.0))
            state.latch_ptr = 0

            if wood.H_prose > 0 and wood.prose_q_proj is not None and wood.prose_k_proj is not None:
                q_prose = wood._safe_unit(wood.prose_q_proj(x).view(B, T, wood.H_prose, wood.D)).to(dt)
                k_prose = wood._safe_unit(wood.prose_k_proj(x).view(B, T, wood.H_prose, wood.D)).to(dt)
                q_rw_all = torch.cat((q_rw_t, q_prose), dim=2)
                k_src_all = torch.cat((q_rw_t, k_prose), dim=2)
                q_epi_all = torch.cat((q_epi_t, q_epi_t[:, :, : wood.H_prose]), dim=2)
            else:
                q_rw_all = q_rw_t
                k_src_all = q_rw_t
                q_epi_all = q_epi_t

            n_full = T // C
            rem = T % C
            pad = (C - rem) % C
            state.buf_pos = rem
            state.buf_x.zero_()
            state.buf_k.zero_()
            state.buf_k_epi.zero_()
            if rem > 0:
                state.buf_x[:, :rem].copy_(x[:, T - rem : T])
                state.buf_k[:, :rem].copy_(k_src_all[:, T - rem : T])
                state.buf_k_epi[:, :rem].copy_(q_epi_all[:, T - rem : T])

            if pad > 0:
                x_pad = F.pad(x, (0, 0, 0, pad))
                q_rw_pad = F.pad(q_rw_all, (0, 0, 0, 0, 0, pad))
                k_src_pad = F.pad(k_src_all, (0, 0, 0, 0, 0, pad))
                q_epi_pad = F.pad(q_epi_all, (0, 0, 0, 0, 0, pad))
            else:
                x_pad, q_rw_pad, k_src_pad, q_epi_pad = x, q_rw_all, k_src_all, q_epi_all

            M_len = x_pad.size(1) // C
            q_chunks = q_rw_pad.view(B, M_len, C, wood.H, wood.D)
            k_chunks = k_src_pad.view(B, M_len, C, wood.H, wood.D)
            q_epi_chunks = q_epi_pad.view(B, M_len, C, wood.H, wood.D)
            x_chunks = x_pad.view(B, M_len, C, D_in)

            w_scores = F.softmax(wood.write_pool_score(x_chunks).float(), dim=2)
            k_seq = wood._safe_unit((k_chunks.float() * w_scores.unsqueeze(-1)).sum(dim=2)).to(dt)
            k_epi_seq = wood._safe_unit((q_epi_chunks.float() * w_scores.unsqueeze(-1)).sum(dim=2)).to(dt)
            v_tok = wood.v_proj(x_chunks).view(B, M_len, C, wood.H, wood.D)
            v_seq = (v_tok * w_scores.to(dt).unsqueeze(-1)).sum(dim=2)
            z_head = (x_chunks * w_scores.to(dt).mean(dim=-1, keepdim=True)).sum(dim=2)

            if (
                wood.use_deltanet2_gates
                and wood.erase_vec_proj is not None
                and wood.write_vec_proj is not None
                and wood.alpha_vec_proj is not None
            ):
                e_vec = torch.sigmoid(wood.erase_vec_proj(z_head)).view(B, M_len, wood.H, wood.D)
                w_vec = torch.sigmoid(wood.write_vec_proj(z_head)).view(B, M_len, wood.H, wood.D)
                a_vec = torch.sigmoid(wood.alpha_vec_proj(z_head)).view(B, M_len, wood.H, wood.D)
                beta = e_vec.mean(dim=-1, keepdim=True)
                gamma = a_vec.mean(dim=-1, keepdim=True)
            else:
                e_vec = w_vec = a_vec = None
                assert wood.beta_proj is not None and wood.gamma_proj is not None
                beta = torch.sigmoid(wood.beta_proj(z_head)).unsqueeze(-1)
                gamma = torch.sigmoid(wood.gamma_proj(z_head)).unsqueeze(-1)

            t_gate = wood._tern_gate()
            inv_dh = t_gate / 144.0
            kalman_mix = 1.0 if wood.use_kalman_precond else 0.0

            if wood.K_sub > 1 and wood.bank_router is not None:
                K_s = wood.K_sub
                H_eb = wood.H_epi * K_s
                HB = wood.H_banks
                BH = B * HB
                r_write_st, _, _ = wood._route_sub_banks(k_seq[:, :, : wood.H_epi])
                r_read_st, _, _ = wood._route_sub_banks(q_rw_t)

                def _exp_epi(t_all: torch.Tensor, h_dim: int) -> torch.Tensor:
                    t_epi = t_all.narrow(h_dim, 0, wood.H_epi)
                    t_exp = t_epi.unsqueeze(h_dim + 1).expand(
                        *t_epi.shape[: h_dim + 1], K_s, *t_epi.shape[h_dim + 1 :]
                    )
                    new_shape = list(t_epi.shape[:h_dim]) + [H_eb] + list(t_epi.shape[h_dim + 1 :])
                    t_eb = t_exp.reshape(*new_shape)
                    if wood.H_prose > 0:
                        return torch.cat((t_eb, t_all.narrow(h_dim, wood.H_epi, wood.H_prose)), dim=h_dim)
                    return t_eb

                q_c_b = _exp_epi(q_chunks, 3)
                qe_c_b = _exp_epi(q_epi_chunks, 3)
                k_s_b = _exp_epi(k_seq, 2)
                ke_s_b = _exp_epi(k_epi_seq, 2)
                v_s_b = _exp_epi(v_seq, 2)
                b_epi_b = (beta[:, :, : wood.H_epi, 0].float().unsqueeze(-1) * r_write_st).reshape(
                    B, M_len, H_eb, 1
                )
                g_epi_b = (
                    1.0 - r_write_st * (1.0 - gamma[:, :, : wood.H_epi, 0].float().unsqueeze(-1))
                ).reshape(B, M_len, H_eb, 1)
                if wood.H_prose > 0:
                    b_b = torch.cat((b_epi_b, beta[:, :, wood.H_epi :].float()), dim=2)
                    g_b = torch.cat((g_epi_b, gamma[:, :, wood.H_epi :].float()), dim=2)
                else:
                    b_b, g_b = b_epi_b, g_epi_b

                qe_m = qe_c_b.float().permute(1, 0, 3, 4, 2).reshape(M_len, BH, wood.D, C)
                ke_m = ke_s_b.float().permute(1, 0, 2, 3).reshape(M_len, BH, wood.D, 1)
                k_m = k_s_b.float().permute(1, 0, 2, 3).reshape(M_len, BH, wood.D, 1).contiguous()
                v_m = v_s_b.float().permute(1, 0, 2, 3).reshape(M_len, BH, wood.D, 1).contiguous()
                q_m = q_c_b.float().permute(1, 0, 3, 4, 2).reshape(M_len, BH, wood.D, C).contiguous()
                b_m = b_b.float().permute(1, 0, 2, 3).reshape(M_len, BH, 1, 1).contiguous()
                g_m = g_b.float().permute(1, 0, 2, 3).reshape(M_len, BH, 1, 1).contiguous()

                u_m, kd_m, P_final = _kalman_p_and_dual_with_final_jit(
                    qe_m, ke_m, k_m, b_m, g_m, wood.inv_lam, t_gate, kalman_mix, n_full
                )
                M0 = torch.zeros(BH, wood.D, wood.D, device=dev, dtype=torch.float32)
                if wood.use_deltanet2_gates and e_vec is not None and w_vec is not None and a_vec is not None:
                    rw_d = r_write_st.unsqueeze(-1)
                    e_epi_b = (e_vec[:, :, : wood.H_epi].float().unsqueeze(3) * rw_d).reshape(B, M_len, H_eb, wood.D)
                    w_epi_b = (w_vec[:, :, : wood.H_epi].float().unsqueeze(3) * rw_d).reshape(B, M_len, H_eb, wood.D)
                    a_epi_b = (1.0 - rw_d * (1.0 - a_vec[:, :, : wood.H_epi].float().unsqueeze(3))).reshape(
                        B, M_len, H_eb, wood.D
                    )
                    if wood.H_prose > 0:
                        e_b = torch.cat((e_epi_b, e_vec[:, :, wood.H_epi :].float()), dim=2)
                        w_b = torch.cat((w_epi_b, w_vec[:, :, wood.H_epi :].float()), dim=2)
                        a_b = torch.cat((a_epi_b, a_vec[:, :, wood.H_epi :].float()), dim=2)
                    else:
                        e_b, w_b, a_b = e_epi_b, w_epi_b, a_epi_b
                    e_m = e_b.permute(1, 0, 2, 3).reshape(M_len, BH, wood.D, 1).contiguous()
                    w_m = w_b.permute(1, 0, 2, 3).reshape(M_len, BH, wood.D, 1).contiguous()
                    a_m = a_b.permute(1, 0, 2, 3).reshape(M_len, BH, 1, wood.D).contiguous()
                    if n_full > 0:
                        o_full, M_final = _deltanet2_m_block_scan_jit(
                            M0, q_m[:n_full], k_m[:n_full], kd_m[:n_full], v_m[:n_full],
                            e_m[:n_full], w_m[:n_full], a_m[:n_full], inv_dh,
                        )
                    else:
                        o_full, M_final = None, M0
                else:
                    if n_full > 0:
                        o_full, M_final = _kalman_m_block_scan_jit(
                            M0, q_m[:n_full], k_m[:n_full], kd_m[:n_full], v_m[:n_full],
                            b_m[:n_full], g_m[:n_full], inv_dh,
                        )
                    else:
                        o_full, M_final = None, M0

                if rem > 0:
                    o_tail = torch.bmm(M_final, q_m[n_full]).unsqueeze(0)
                    out_m = o_tail if o_full is None else torch.cat((o_full, o_tail), dim=0)
                else:
                    assert o_full is not None
                    out_m = o_full

                state.M_mat.copy_(M_final)
                state.P_mat.copy_(P_final)

                u_bank_seq = (
                    u_m.reshape(M_len, B, HB, C).permute(1, 0, 3, 2).reshape(B, M_len * C, HB)[:, :T]
                )
                read_bank_seq = (
                    out_m.reshape(M_len, B, HB, wood.D, C)
                    .permute(1, 0, 4, 2, 3)
                    .reshape(B, M_len * C, HB, wood.D)[:, :T]
                )
                read_epi = (
                    read_bank_seq[:, :, :H_eb].view(B, T, wood.H_epi, K_s, wood.D)
                    * r_read_st.unsqueeze(-1)
                ).sum(dim=3)
                u_epi_comb = (
                    u_bank_seq[:, :, :H_eb].view(B, T, wood.H_epi, K_s) * r_read_st
                ).sum(dim=3)
                if wood.H_prose > 0:
                    read_all = torch.cat((read_epi, read_bank_seq[:, :, H_eb:]), dim=2).to(dt)
                    uncert_seq = torch.cat((u_epi_comb, u_bank_seq[:, :, H_eb:]), dim=2).to(dt)
                else:
                    read_all = read_epi.to(dt)
                    uncert_seq = u_epi_comb.to(dt)
                read_seq = read_all.reshape(B, T, wood.total_dim)
            else:
                BH = B * wood.H
                qe_m = q_epi_chunks.float().permute(1, 0, 3, 4, 2).reshape(M_len, BH, wood.D, C)
                ke_m = k_epi_seq.float().permute(1, 0, 2, 3).reshape(M_len, BH, wood.D, 1)
                k_m = k_seq.float().permute(1, 0, 2, 3).reshape(M_len, BH, wood.D, 1).contiguous()
                v_m = v_seq.float().permute(1, 0, 2, 3).reshape(M_len, BH, wood.D, 1).contiguous()
                q_m = q_chunks.float().permute(1, 0, 3, 4, 2).reshape(M_len, BH, wood.D, C).contiguous()
                b_m = beta.float().permute(1, 0, 2, 3).reshape(M_len, BH, 1, 1).contiguous()
                g_m = gamma.float().permute(1, 0, 2, 3).reshape(M_len, BH, 1, 1).contiguous()

                u_m, kd_m, P_final = _kalman_p_and_dual_with_final_jit(
                    qe_m, ke_m, k_m, b_m, g_m, wood.inv_lam, t_gate, kalman_mix, n_full
                )
                M0 = torch.zeros(BH, wood.D, wood.D, device=dev, dtype=torch.float32)
                if wood.use_deltanet2_gates and e_vec is not None and w_vec is not None and a_vec is not None:
                    e_m = e_vec.float().permute(1, 0, 2, 3).reshape(M_len, BH, wood.D, 1).contiguous()
                    w_m = w_vec.float().permute(1, 0, 2, 3).reshape(M_len, BH, wood.D, 1).contiguous()
                    a_m = a_vec.float().permute(1, 0, 2, 3).reshape(M_len, BH, 1, wood.D).contiguous()
                    if n_full > 0:
                        o_full, M_final = _deltanet2_m_block_scan_jit(
                            M0, q_m[:n_full], k_m[:n_full], kd_m[:n_full], v_m[:n_full],
                            e_m[:n_full], w_m[:n_full], a_m[:n_full], inv_dh,
                        )
                    else:
                        o_full, M_final = None, M0
                else:
                    if n_full > 0:
                        o_full, M_final = _kalman_m_block_scan_jit(
                            M0, q_m[:n_full], k_m[:n_full], kd_m[:n_full], v_m[:n_full],
                            b_m[:n_full], g_m[:n_full], inv_dh,
                        )
                    else:
                        o_full, M_final = None, M0

                if rem > 0:
                    o_tail = torch.bmm(M_final, q_m[n_full]).unsqueeze(0)
                    out_m = o_tail if o_full is None else torch.cat((o_full, o_tail), dim=0)
                else:
                    assert o_full is not None
                    out_m = o_full

                state.M_mat.copy_(M_final)
                state.P_mat.copy_(P_final)

                uncert_seq = (
                    u_m.reshape(M_len, B, wood.H, C).permute(1, 0, 3, 2).reshape(B, M_len * C, wood.H)[:, :T].to(dt)
                )
                read_seq = (
                    out_m.reshape(M_len, B, wood.H, wood.D, C)
                    .permute(1, 0, 4, 2, 3)
                    .reshape(B, M_len * C, wood.H, wood.D)[:, :T]
                    .to(dt)
                    .reshape(B, T, wood.total_dim)
                )

            wood.last_epistemic_uncertainty = uncert_seq

            u_epi = uncert_seq[:, :, : wood.H_epi].float()
            conf_gate_epi = torch.sigmoid(
                wood.conf_slope.float().view(1, 1, wood.H_epi)
                * (wood.conf_thresh.float().view(1, 1, wood.H_epi) - u_epi)
            )
            if wood.H_prose > 0 and wood.query_intent_proj is not None:
                q_intent = torch.sigmoid(wood.query_intent_proj(x).float())
                prose_gate = (1.0 - 0.88 * q_intent).expand(B, T, wood.H_prose)
                all_gates = torch.cat((conf_gate_epi, prose_gate), dim=-1).unsqueeze(-1)
            else:
                q_intent = None
                all_gates = conf_gate_epi.unsqueeze(-1)

            normed_read = (
                wood.out_norm(read_seq).view(B, T, wood.H, wood.D) * all_gates.to(dt)
            ).reshape(B, T, wood.total_dim)
            raw_u_feats = torch.cat(
                (
                    u_epi - 0.42,
                    torch.tanh((u_epi - 0.42) * 7.0),
                    F.relu(u_epi - 0.36) * 2.5,
                ),
                dim=-1,
            )
            u_feats = raw_u_feats.to(dt) * wood.uncert_scale.to(dt)
            if q_intent is not None:
                u_feats = q_intent.to(dt) * u_feats
            return wood.out_proj(torch.cat((normed_read, u_feats), dim=-1))

    @staticmethod
    def _epistemic_slots_prefill_with_state(
        es: EpistemicSlotMemory,
        x: torch.Tensor,
        epistemic_uncert: Optional[torch.Tensor],
        h_lex: Optional[torch.Tensor],
        state: EpistemicSlotsStreamState,
    ) -> torch.Tensor:
        """Vectorized Hillis-Steele `EpistemicSlotMemory` pass populating exact stream state."""
        B, L, _ = x.shape
        H, S, d_k, d_v = es.n_heads, es.n_slots, es.d_k, es.d_v
        dt = x.dtype
        dev = x.device

        z_qk = es._blend_key_input(x, h_lex)
        q = es._safe_unit(es.q_proj(z_qk).view(B, L, H, d_k))
        k_raw = es._safe_unit(es.k_proj(z_qk).view(B, L, H, d_k))
        v = es.v_proj(x).float().view(B, L, H, d_v)
        if h_lex is not None:
            v = v + es.v_lex_proj(h_lex).float().view(B, L, H, d_v)

        k_pad = F.pad(k_raw, (0, 0, 0, 0, 3, 0))
        lw = F.softmax(es.lag_logits.float(), dim=-1)
        k_lag = es._safe_unit(
            lw[:, 0].view(1, 1, H, 1) * k_pad[:, 2 : 2 + L]
            + lw[:, 1].view(1, 1, H, 1) * k_pad[:, 1 : 1 + L]
            + lw[:, 2].view(1, 1, H, 1) * k_pad[:, 0:L]
        )

        beta_st, _ = es._compute_beta(x, epistemic_uncert)
        if L > 1:
            t_mask = torch.ones((1, L, 1), device=dev, dtype=torch.float32)
            t_mask[:, 0, :] = 0.0
            beta_eff = beta_st * t_mask
        else:
            beta_eff = torch.zeros_like(beta_st)

        r_st, _, _ = es._route_slots(k_lag)
        alpha = beta_eff.unsqueeze(-1) * r_st
        ones = torch.ones((B, L, H, 1), device=dev, dtype=torch.float32)
        u_all = torch.cat((k_lag, v, ones), dim=-1)

        Bc = es.block_size
        n_blocks = (L + Bc - 1) // Bc
        with torch.amp.autocast(dev.type, enabled=False):
            s_cur = torch.zeros((B, H, S, es.d_u), device=dev, dtype=torch.float32)
            r_sc = es.read_scale.float()
            n_lg = es.null_logit.float()
            outs: List[torch.Tensor] = []
            for bi in range(n_blocks):
                st_i = bi * Bc
                en_i = min(L, st_i + Bc)
                o_b, s_cur = _scan_and_read_block(
                    s_cur, q[:, st_i:en_i], alpha[:, st_i:en_i], u_all[:, st_i:en_i], r_sc, n_lg, d_k, d_v
                )
                outs.append(o_b)
            out_flat = (outs[0] if n_blocks == 1 else torch.cat(outs, dim=1)).reshape(B, L, H * d_v).to(dt)

        state.kvf_bank.copy_(s_cur)
        state.k_ring.copy_(_pad_tail_chronological(k_raw, 3))
        state.k_ptr = 0
        state.step_count = L
        return es.out_proj(es.out_norm(out_flat))

    @staticmethod
    def _crystal_prefill_with_state(
        cr: CalciteV2CrystalLayer,
        x: torch.Tensor,              # (1, M, D)
        h_lex: Optional[torch.Tensor],
        state: CrystalStreamState,
    ) -> torch.Tensor:
        """Vectorized `CalciteV2CrystalLayer` pass populating exact `CrystalStreamState`."""
        B, M, D = x.shape
        dt = x.dtype
        dev = x.device
        H, G_h, d_s, C = cr.H, cr.G_h, cr.d_s, cr.C
        BH = B * H
        scale_r = math.sqrt(float(d_s))
        sc_d = math.sqrt(float(D))

        with torch.amp.autocast(dev.type, enabled=False):
            x_u = cr._safe_unit(x)
            if h_lex is not None:
                w_lex = torch.sigmoid(cr.lex_logit.float())
                z = cr._safe_unit(w_lex * cr._safe_unit(h_lex) + (1.0 - w_lex) * x_u)
            else:
                z = x_u

            steps = torch.arange(1, M + 1, device=dev, dtype=torch.float32).view(1, M, 1)
            zc = cr._safe_unit(z - torch.cumsum(z, dim=1) / (steps + 2.0), sc_d).to(dt)
            u_all = cr._safe_unit(cr.slot_proj(zc).view(B, M, H, cr.d_slot))
            sal_all = cr.subj_sal_proj(zc).float()  # (B, M, H)

            # 5-step write-lag key (`kw_vec_tok`)
            sal_w_win = F.pad(sal_all, (0, 0, 4, 0), value=-1e4).unfold(1, 5, 1)  # (B, M, H, 5)
            u_w_win = (
                F.pad(u_all.reshape(B, M, H * cr.d_slot), (0, 0, 4, 0))
                .unfold(1, 5, 1)
                .view(B, M, H, cr.d_slot, 5)
            )
            w_lag_att = F.softmax(sal_w_win + cr.write_lag_logits.float().view(1, 1, H, 5), dim=-1)
            kw_vec_tok = cr._safe_unit(
                torch.matmul(u_w_win, w_lag_att.unsqueeze(-1)).squeeze(-1)
                + 0.30 * cr._safe_unit(cr.w_ctx_proj(x).view(B, M, H, cr.d_slot))
            )

            # 4-step read-lag query (`q_soft`)
            sal_r_win = F.pad(sal_all, (0, 0, 3, 0), value=-1e4).unfold(1, 4, 1)  # (B, M, H, 4)
            u_r_win = (
                F.pad(u_all.reshape(B, M, H * cr.d_slot), (0, 0, 3, 0))
                .unfold(1, 4, 1)
                .view(B, M, H, cr.d_slot, 4)
                .permute(0, 1, 2, 4, 3)  # (B, M, H, 4, d_slot)
            )
            r_lag_att = F.softmax(
                sal_r_win.unsqueeze(3) + cr.read_lag_logits.float().view(1, 1, H, cr.R_read, 4),
                dim=-1,
            )  # (B, M, H, R_read, 4)
            r_u_agg = torch.matmul(r_lag_att, u_r_win).reshape(B, M, cr.H_r, cr.d_slot)
            qr_vec = cr._safe_unit(
                r_u_agg + 0.30 * cr._safe_unit(cr.r_ctx_proj(x).view(B, M, cr.H_r, cr.d_slot))
            )

            k_slots_h = state.cached_k_slots_h
            if k_slots_h is None or k_slots_h.device != dev:
                k_slots_h = cr._safe_unit(cr.K_slots).to(dev)
                state.cached_k_slots_h = k_slots_h

            q_logits = (
                torch.matmul(qr_vec.view(B, M, H, cr.R_read, cr.d_slot), k_slots_h.transpose(-1, -2))
                * float(cr.slot_temp)
            )
            q_soft = F.softmax(q_logits, dim=-1).to(dt)  # (B, M, H, R_read, G_h)

            # Populate token-level rings & counters
            state.z_cumsum.copy_(z.sum(dim=1))
            state.step_count = M
            state.sal_w_ring.copy_(_pad_tail_chronological(sal_all, 4, pad_value=-1e4))
            state.u_w_ring.copy_(_pad_tail_chronological(u_all, 4, pad_value=0.0))
            state.w_ptr = 0
            state.sal_r_ring.copy_(_pad_tail_chronological(sal_all, 3, pad_value=-1e4))
            state.u_r_ring.copy_(_pad_tail_chronological(u_all, 3, pad_value=0.0))
            state.r_ptr = 0

            n_full = M // C
            rem = M % C
            state.buf_pos = rem
            state.buf_x.zero_()
            state.buf_zc.zero_()
            state.buf_u.zero_()
            state.buf_kw.zero_()
            if rem > 0:
                state.buf_x[:, :rem].copy_(x[:, M - rem : M])
                state.buf_zc[:, :rem].copy_(zc[:, M - rem : M])
                state.buf_u[:, :rem].copy_(u_all[:, M - rem : M])
                state.buf_kw[:, :rem].copy_(kw_vec_tok[:, M - rem : M])

            cb = state.cached_cb
            c_norm_t = state.cached_c_norm_t
            if cb is None or c_norm_t is None or cb.device != dev:
                cb, c_norm_t = cr.get_codebook_fp32()
                cb = cb.to(dev)
                c_norm_t = c_norm_t.to(dev)
                state.cached_cb = cb
                state.cached_c_norm_t = c_norm_t

            s_cur = state.s_bh.clone()  # initialized to s0_bh by init_state
            s_before_list: List[torch.Tensor] = []

            if n_full > 0:
                M_full = n_full * C
                x_c = x[:, :M_full].view(B, n_full, C, D)
                zc_f = zc[:, :M_full]
                u_f = u_all[:, :M_full]
                kw_c = kw_vec_tok[:, :M_full].view(B, n_full, C, H, cr.d_slot)

                s_temp = float(cr.slot_temp)
                o_temp = float(cr.op_temp)
                g_tau = float(cr.gate_temp)
                eff_alpha = 1.0 if (not cr.training and cr.snap_alpha > 0.0) else (0.85 * float(cr.snap_alpha))
                is_hard = eff_alpha >= 0.9999

                w_logit_tok = cr.w_eta_proj(x_c).float()  # (B, n_full, C, H)
                w_pool = F.softmax(w_logit_tok * 2.0, dim=2)
                w_logit = (w_logit_tok * w_pool).sum(dim=2)
                kw_vec = cr._safe_unit((kw_c * w_pool.unsqueeze(-1)).sum(dim=2))

                v_tok = cr.w_val_proj(x_c).float().view(B, n_full, C, H, d_s)
                v_pool = F.softmax(cr.pool_score(x_c).float(), dim=2)
                v_unit = cr._safe_unit((v_tok * v_pool.unsqueeze(-1)).sum(dim=2))
                v_logits = torch.matmul(v_unit, c_norm_t) * 9.0
                v_probs = F.softmax(v_logits, dim=-1)
                v_soft = _sphere_norm_jit(torch.matmul(v_probs, cb), scale_r)
                v_hard = cb[v_logits.argmax(dim=-1)]
                if is_hard:
                    v_ste = v_hard
                elif eff_alpha > 0.0:
                    v_ste = _sphere_norm_jit((1.0 - eff_alpha) * v_soft + eff_alpha * v_hard, scale_r)
                else:
                    v_ste = v_soft

                # Sliding pointer window of `W_chunks * C` committed chunks
                w_len = cr.W_chunks * C
                pad_w = (cr.W_chunks - 1) * C
                zc_w_all = (
                    F.pad(zc_f, (0, 0, pad_w, 0))
                    .unfold(1, w_len, C)
                    .permute(0, 1, 3, 2)
                    .contiguous()
                )  # (B, n_full, w_len, D)
                u_w_all = (
                    F.pad(u_f.reshape(B, M_full, H * cr.d_slot), (0, 0, pad_w, 0))
                    .unfold(1, w_len, C)
                    .permute(0, 1, 3, 2)
                    .contiguous()
                    .view(B, n_full, w_len, H, cr.d_slot)
                )
                valid_f = torch.ones((B, M_full), device=dev, dtype=torch.bool)
                win_valid_all = F.pad(valid_f, (pad_w, 0), value=False).unfold(1, w_len, C)  # (B, n_full, w_len)

                state.zc_win.copy_(zc_w_all[:, -1])
                state.u_win.copy_(u_w_all[:, -1])
                state.win_valid.copy_(win_valid_all[:, -1])

                ptr_raw = cr.op_ptr_proj(zc_w_all).float().view(
                    B, n_full, w_len, H, 2
                ) + cr.ptr_pos_bias.float().view(1, 1, w_len, H, 2)
                ptr_raw = ptr_raw.masked_fill(~win_valid_all.unsqueeze(-1).unsqueeze(-1), -1e4)
                att_a = F.softmax(ptr_raw[..., 0], dim=2)
                att_b = F.softmax(ptr_raw[..., 1] - 4.5 * att_a, dim=2)
                ua_vec = cr._safe_unit((u_w_all * att_a.unsqueeze(-1)).sum(dim=2))
                ub_vec = cr._safe_unit((u_w_all * att_b.unsqueeze(-1)).sum(dim=2))

                all_v = torch.stack((kw_vec, ua_vec, ub_vec), dim=3)  # (B, n_full, H, 3, d_slot)
                all_lg = torch.matmul(all_v, k_slots_h.transpose(-1, -2)) * s_temp
                k_logits, psi_a, psi_b = all_lg[:, :, :, 0], all_lg[:, :, :, 1], all_lg[:, :, :, 2]

                k_prob = F.softmax(k_logits, dim=-1)
                w_peak = k_prob / k_prob.max(dim=-1, keepdim=True).values.clamp(min=1e-5)
                w_eta = torch.sigmoid((w_logit + 0.58) * g_tau - 0.58)
                kw_soft = (w_eta.unsqueeze(-1) * w_peak).unsqueeze(-1)
                k_hard = F.one_hot(k_prob.argmax(dim=-1), num_classes=G_h).float()
                w_hard_g = (w_eta >= 0.36).float()
                kw_targ = (w_hard_g.unsqueeze(-1) * k_hard).unsqueeze(-1)
                kw_eff = kw_targ if is_hard else ((1.0 - eff_alpha) * kw_soft + eff_alpha * kw_targ)

                ws = 1.0 if ((not cr.training and cr.snap_alpha > 0.0) or cr.qat_active_mix > 0.0) else 0.0
                _, _, T_swap = cr._compute_ptr_swap(psi_a, psi_b, well_scale=ws)

                x_ch_mean = x_c.float().mean(dim=2).to(dt)  # (B, n_full, D)
                state.z_mean_ring.copy_(_pad_tail_chronological(x_ch_mean, 3))
                state.zmean_ptr = 0
                z_op_win = (
                    F.pad(x_ch_mean, (0, 0, 3, 0))
                    .unfold(1, 4, 1)
                    .permute(0, 1, 3, 2)
                    .contiguous()
                    .view(B, n_full, 4 * D)
                )
                z_op = F.silu(cr.op_ctx_proj(z_op_win))

                B_mats = state.cached_B_mats
                if B_mats is None or state.cached_o_temp != o_temp or B_mats.device != dev:
                    B_mats = log_sinkhorn_6(cr.birkhoff_logits.float().to(dev) * o_temp)
                    state.cached_B_mats = B_mats
                    state.cached_o_temp = o_temp

                m_prob = F.softmax(cr.op_mode_proj(z_op).float().view(B, n_full, H, 1 + cr.M_glob) * o_temp, dim=-1)
                B_flat = B_mats.reshape(H, cr.M_glob, G_h * G_h)
                glob_mix = torch.matmul(m_prob[..., 1:].unsqueeze(-2), B_flat).view(B, n_full, H, G_h, G_h)
                P_cand = m_prob[..., 0:1].unsqueeze(-1) * T_swap + glob_mix

                p_logit_raw = cr.perm_gate_proj(z_op).float()
                p_logit = -4.5 + F.softplus(p_logit_raw + 4.5)
                p_raw = torch.sigmoid((p_logit + 0.49) * g_tau - 0.49)  # (B, n_full, H)

                p_f_m1 = state.p_f_m1.clone()
                p_f_m2 = state.p_f_m2.clone()
                p_eta_list: List[torch.Tensor] = []
                for mc in range(n_full):
                    p_prev = torch.maximum(p_f_m1, p_f_m2)
                    p_eta_mc = p_raw[:, mc] * (1.0 - 0.92 * torch.sigmoid((p_prev - 0.35) * 12.0))
                    p_f_m2 = p_f_m1
                    p_f_m1 = p_eta_mc
                    p_eta_list.append(p_eta_mc)
                state.p_f_m1.copy_(p_f_m1)
                state.p_f_m2.copy_(p_f_m2)
                p_eta = torch.stack(p_eta_list, dim=1)  # (B, n_full, H)

                eye = cr.eye_Gh.float().view(1, 1, 1, G_h, G_h)
                P_soft = eye + p_eta.unsqueeze(-1).unsqueeze(-1) * (P_cand - eye)
                if eff_alpha > 0.0:
                    P_perm_hard = project_birkhoff_to_perm_jit(P_cand.reshape(-1, G_h, G_h)).view(
                        B, n_full, H, G_h, G_h
                    )
                    p_hard_g = (p_eta >= 0.38).float()
                    P_targ = eye + p_hard_g.unsqueeze(-1).unsqueeze(-1) * (P_perm_hard - eye)
                    P_eff = P_targ if is_hard else ((1.0 - eff_alpha) * P_soft + eff_alpha * P_targ)
                else:
                    P_eff = P_soft

                kw_u = kw_eff.reshape(B, n_full, H, G_h, 1).permute(1, 0, 2, 3, 4).reshape(n_full, BH, G_h, 1).unbind(0)
                vs_u = v_ste.reshape(B, n_full, H, 1, d_s).permute(1, 0, 2, 3, 4).reshape(n_full, BH, 1, d_s).unbind(0)
                pe_u = P_eff.reshape(B, n_full, H, G_h, G_h).permute(1, 0, 2, 3, 4).reshape(n_full, BH, G_h, G_h).unbind(0)
                for mc in range(n_full):
                    s_before_list.append(s_cur)
                    s_w = (1.0 - kw_u[mc]) * s_cur + kw_u[mc] * vs_u[mc]
                    s_next = torch.bmm(pe_u[mc], s_w)
                    s_cur = s_next if is_hard else _sphere_norm_jit(s_next, scale_r)
                state.s_bh.copy_(s_cur)

            if rem > 0:
                s_before_list.append(s_cur)

            n_mc = len(s_before_list)
            s_mc = (
                torch.stack(s_before_list, dim=0)
                .view(n_mc, B, H, G_h, d_s)
                .permute(1, 0, 2, 3, 4)
                .to(dt)
            )
            s_tok = s_mc.unsqueeze(2).expand(B, n_mc, C, H, G_h, d_s).reshape(
                B, n_mc * C, H, G_h, d_s
            )[:, :M]
            ret = torch.matmul(q_soft, s_tok).reshape(B, M, cr.H_r * d_s)
            return cr.out_proj(cr.out_norm(ret))

    @staticmethod
    def _bic_prefill_with_state(
        bic: ByteInductionCache,
        token_ids: torch.Tensor,
        h_ctx: torch.Tensor,
        t_pad6: Optional[torch.Tensor],
        state: ByteInductionStreamState,
    ) -> torch.Tensor:
        """Exact blocked dual-form `ByteInductionCache` pass populating final `bank` + `ring_tok7`."""
        B, T, _ = h_ctx.shape
        dt = h_ctx.dtype
        dev = h_ctx.device
        S = bic.n_slots
        D_u = bic.d_u
        Bc = bic.block_size
        B2 = B * 2

        sq4, sq7 = _hash_2scale_seq(token_ids, S, t_pad6=t_pad6)
        sq = torch.stack((sq4, sq7), dim=2)

        v_flat = bic.val_emb(token_ids.clamp(0, bic.vocab_size - 1)).float()
        v_2 = v_flat.view(B, T, 2, bic.d_v)
        q_sig = bic._safe_unit(bic.q_proj(h_ctx).view(B, T, 2, bic.d_sig))

        if T > 1:
            sw = F.pad(sq[:, :-1], (0, 0, 1, 0), value=0)
            h_ctx_prev = h_ctx[:, :-1]
            k_sig_tail = bic._safe_unit(bic.k_proj(h_ctx_prev).view(B, T - 1, 2, bic.d_sig))
            k_sig = F.pad(k_sig_tail, (0, 0, 0, 0, 1, 0), value=0.0)
            beta_st_tail, _ = bic._compute_beta(h_ctx_prev, v_flat[:, 1:])
            beta_eff = F.pad(beta_st_tail, (0, 0, 1, 0), value=0.0)
        else:
            sw = torch.zeros((B, 1, 2), device=dev, dtype=torch.long)
            k_sig = torch.zeros((B, 1, 2, bic.d_sig), device=dev, dtype=torch.float32)
            beta_eff = torch.zeros((B, 1, 2), device=dev, dtype=torch.float32)

        ones = torch.ones((B, T, 2, 1), device=dev, dtype=torch.float32)
        u_all = torch.cat((k_sig, v_2, ones), dim=-1)

        with torch.amp.autocast(dev.type, enabled=False):
            n_blocks = (T + Bc - 1) // Bc
            sq_all = sq.permute(0, 2, 1).reshape(B2, T)
            sw_all = sw.permute(0, 2, 1).reshape(B2, T)
            bt_all = beta_eff.permute(0, 2, 1).reshape(B2, T)
            u_perm = u_all.permute(0, 2, 1, 3).reshape(B2, T, D_u)
            log_k_all = torch.log((1.0 - bt_all).clamp(min=1e-5))

            tril_full = bic._tril_bc if bic._tril_bc.device == dev else bic._tril_bc.to(dev)
            triu_full = bic._triu_bc if bic._triu_bc.device == dev else bic._triu_bc.to(dev)
            zero_s = torch.zeros((), device=dev, dtype=torch.float32)

            bank = torch.zeros((B2, S, D_u), device=dev, dtype=torch.float32)
            u_read_blocks: List[torch.Tensor] = []

            for bi in range(n_blocks):
                st_i = bi * Bc
                en_i = min(T, st_i + Bc)
                Lb = en_i - st_i

                sq_b = sq_all[:, st_i:en_i]
                sw_b = sw_all[:, st_i:en_i]
                bt_b = bt_all[:, st_i:en_i]
                u_b = u_perm[:, st_i:en_i]
                log_k = log_k_all[:, st_i:en_i]

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
                if bi > 0:
                    prev_g = torch.gather(bank, 1, sq_b.unsqueeze(-1).expand(B2, Lb, D_u))
                    u_read_b = torch.exp(cs_end.clamp(min=-30.0)) * prev_g + u_blk
                else:
                    u_read_b = u_blk
                u_read_blocks.append(u_read_b)

                # Update bank on ALL blocks (including final block) so streaming continuation is exact
                triu_s = triu_full[:Lb, :Lb]
                same_after = (sw_b.unsqueeze(2) == sw_b.unsqueeze(1)) & triu_s.unsqueeze(0)
                log_surv = torch.where(same_after, log_k.unsqueeze(1), zero_s).sum(dim=-1)
                contrib = (bt_b * torch.exp(log_surv.clamp(min=-30.0))).unsqueeze(-1) * u_b
                slot_lk = torch.zeros((B2, S), device=dev, dtype=torch.float32)
                slot_lk.scatter_add_(1, sw_b, log_k)
                delta_b = torch.zeros((B2, S, D_u), device=dev, dtype=torch.float32)
                delta_b.scatter_add_(1, sw_b.unsqueeze(-1).expand(B2, Lb, D_u), contrib)
                bank = bank * torch.exp(slot_lk.clamp(min=-30.0)).unsqueeze(-1) + delta_b

            u_read_full = (
                u_read_blocks[0] if n_blocks == 1 else torch.cat(u_read_blocks, dim=1)
            ).view(B, 2, T, D_u).permute(0, 2, 1, 3)
            out_v = bic._decode_read_u(u_read_full, q_sig).to(dt)

        state.bank.copy_(bank.view(B, 2, S, D_u))
        state.ring_tok7.copy_(_pad_tail_chronological(token_ids.long(), 7))
        state.prev_h_ctx.copy_(h_ctx[:, -1].float())
        state.step_count = T
        state.ptr7 = 0

        gate = 2.0 * torch.sigmoid(bic.out_gate(h_ctx))
        return gate * bic.out_proj(bic.out_norm(out_v))

    @torch.no_grad()
    def _prefill_fresh_b1(
        self,
        token_ids: torch.Tensor,  # (1, T)
        state: CalciteStreamState,
        return_all_logits: bool,
    ) -> Tuple[torch.Tensor, CalciteStreamState]:
        """Single-pass exact vectorized prefill for `B == 1` from empty `state`."""
        m = self.model
        ch = m.byte_chunker
        dev = token_ids.device
        _, T = token_ids.shape

        # 1. Byte encoder + router compaction (reuses native CUDA UTF-8 compaction when on CUDA)
        e_tok = m.tok_emb(token_ids)
        dt = e_tok.dtype
        was_training = m.training
        if was_training:
            m.train(False)
        try:
            z_in, h_lex_chunk, ctx = ch.encode_to_chunks(e_tok, token_ids, m.tok_emb.weight)
            M = int(ctx.chunk_counts[0].item()) if ctx.chunk_counts is not None else z_in.size(1)
            z_in = z_in[:, :M]
            h_lex_chunk = h_lex_chunk[:, :M]

            # Populate encoder streaming rings & router FSM state
            cs = state.chunker
            cs.ring_char.copy_(_pad_tail_chronological(e_tok[..., : ch.d_byte], 7))
            cs.char_ptr = 0

            rc_tail_tok = token_ids[:, max(0, T - 4) :]
            rc_tail = ch.router_emb(rc_tail_tok)
            if rc_tail.dtype != dt:
                rc_tail = rc_tail.to(dt)
            cs.ring_rchar.copy_(_pad_tail_chronological(rc_tail, 3))
            cs.rchar_ptr = 0

            # Final router key & normalized entropy from last 4 bytes only
            r_win4_last = _pad_tail_chronological(rc_tail, 4).flip(1).reshape(1, 4 * ch.d_rchar)
            h_r_last = F.silu(ch._r_lin(ch.router_in_proj, ch.router_in_norm(r_win4_last)))
            cs.prev_k_r.copy_(F.normalize(ch._r_lin(ch.router_k, h_r_last).float(), dim=-1))
            loc_last = ch._r_lin(ch.router_ent_head, ch.router_ent_norm(h_r_last)).float()
            lp_last = F.log_softmax(loc_last, dim=-1)
            log_v = math.log(max(2, ch.vocab_size))
            cs.prev_h_ent.copy_((-(lp_last.exp() * lp_last).sum(dim=-1) / log_v).clamp(0.0, 1.5))

            engram_ring_len = cs.ring_tok3.size(1)
            if ch.use_byte_engram and engram_ring_len >= 3:
                cs.ring_tok3.copy_(_pad_tail_chronological(token_ids.long(), engram_ring_len))
                cs.tok3_ptr = 0
            if ch.use_byte_induct_cache:
                cs.last_tok = token_ids[:, -1].clone()

            if (
                ch.use_sym_byte_swa
                and ch.enc_dw2_norm is not None
                and (ch.enc_dw2_gate is not None or ch.enc_dw2_in_gate is not None)
            ):
                # Recompute `enc_dw2` ring input on ONLY the final <= 6 bytes (depends on last <= 13 bytes)
                n_out_tail = min(T, 6)
                t0_tail = max(0, T - 13)
                e_sub = e_tok[:, t0_tail:T]
                c_win8_sub = ch._causal_win8(e_sub)[:, -n_out_tail:]
                e_last = e_tok[:, T - n_out_tail : T]
                h_enc0_sub = e_last + ch.enc_up(ch.enc_norm_in(c_win8_sub))
                h_b0_sub = h_enc0_sub + ch.enc_mlp(ch.enc_norm(h_enc0_sub))
                xe_tail = ch.enc_dw2_norm(h_b0_sub)
                if ch.enc_dw2_in_gate is not None:
                    xc_e_tail, _ = ch._project_dw_in_gate(xe_tail, None, ch.enc_dw2_in_gate)
                else:
                    xc_e_tail = xe_tail
                cs.ring_enc_conv2.copy_(_pad_tail_chronological(xc_e_tail, 6))
                cs.enc2_ptr = 0

            # Exact router FSM registers (`run_len`, `cp_run`, `cont_streak`, `credits`)
            b_mask_0 = ctx.b_mask[0]
            valid_0 = ctx.valid_mask[0]
            fire_idx = b_mask_0.nonzero(as_tuple=False).squeeze(-1)
            t_last_fire = int(fire_idx[-1].item()) if fire_idx.numel() > 0 else 0
            run_len_val = int(valid_0[t_last_fire:].long().sum().item())
            cs.run_len.fill_(run_len_val)
            if ch.byte_chunk_mode == "utf8":
                _, is_cont, _ = _utf8_tensor_flags(token_ids)
                is_cont_0 = is_cont[0]
                is_lead_0 = (valid_0 & (~is_cont_0)).long()
                cs.cp_run.fill_(int(is_lead_0[t_last_fire:].sum().item()))
                is_cont_v0 = is_cont_0 & valid_0
                non_cont = (~is_cont_v0).nonzero(as_tuple=False).squeeze(-1)
                c_streak = T if non_cont.numel() == 0 else (T - 1 - int(non_cont[-1].item()))
                cs.cont_streak.fill_(c_streak)
                cred_val = _compute_utf8_credits_jit(
                    valid_0.cpu(), b_mask_0.cpu(), ch.refill_int, ch.cost_int, ch.burst_int
                )
                cs.credits.fill_(cred_val)
            else:
                cs.cp_run.zero_()
                cs.cont_streak.zero_()
                cs.credits.fill_(ch.burst_int)
            cs.step_idx = T

            # 2. Vectorized chunk backbone pass extracting exact SWA/Woodbury/Crystal/EpistemicSlots states
            h = z_in
            ring_idx = 0
            if self.cfg.topology == "interleaved" and m.swa_stages is not None:
                use_dca = m.dca_mixers is not None
                hist: List[torch.Tensor] = [h] if use_dca else []
                dca_idx = 0

                def _dca_mix_il_pf(h_cur: torch.Tensor) -> torch.Tensor:
                    nonlocal dca_idx
                    if use_dca and m.dca_mixers is not None and len(hist) >= 2:
                        h_cur = m.dca_mixers[dca_idx](h_cur, hist)
                        dca_idx += 1
                    return h_cur

                for blk in m.swa_stages[0]:
                    h = _dca_mix_il_pf(h)
                    c_rng = state.swa_canon_rings[ring_idx] if state.swa_canon_rings is not None else None
                    h = self._swa_block_prefill_with_state(
                        blk, h, state.swa_k_rings[ring_idx], state.swa_v_rings[ring_idx], c_rng
                    )
                    ring_idx += 1
                    if use_dca:
                        hist.append(h)
                h = _dca_mix_il_pf(h)
                if (
                    m.woodbury_fast is not None
                    and m.woodbury_fast_norm is not None
                    and m.woodbury_fast_gate is not None
                    and state.woodbury_fast is not None
                ):
                    wf_in = m.woodbury_fast_norm(h)
                    wf_g = 2.0 * torch.sigmoid(m.woodbury_fast_gate(wf_in))
                    h = h + wf_g * self._woodbury_prefill_with_state(
                        m.woodbury_fast, wf_in, h_lex_chunk, state.woodbury_fast
                    )
                if use_dca:
                    hist.append(h)

                for blk in m.swa_stages[1]:
                    h = _dca_mix_il_pf(h)
                    c_rng = state.swa_canon_rings[ring_idx] if state.swa_canon_rings is not None else None
                    h = self._swa_block_prefill_with_state(
                        blk, h, state.swa_k_rings[ring_idx], state.swa_v_rings[ring_idx], c_rng
                    )
                    ring_idx += 1
                    if use_dca:
                        hist.append(h)
                h = _dca_mix_il_pf(h)
                if m.cryst1_norm is not None and m.cryst1_gate is not None:
                    c1_in = m.cryst1_norm(h)
                    c1_g = 2.0 * torch.sigmoid(m.cryst1_gate(c1_in))
                    h = h + c1_g * self._crystal_prefill_with_state(
                        m.crystal, c1_in, h_lex_chunk, state.crystal_1
                    )
                if use_dca:
                    hist.append(h)

                for blk in m.swa_stages[2]:
                    h = _dca_mix_il_pf(h)
                    c_rng = state.swa_canon_rings[ring_idx] if state.swa_canon_rings is not None else None
                    h = self._swa_block_prefill_with_state(
                        blk, h, state.swa_k_rings[ring_idx], state.swa_v_rings[ring_idx], c_rng
                    )
                    ring_idx += 1
                    if use_dca:
                        hist.append(h)
                h = _dca_mix_il_pf(h)
                if m.woodbury_slow_gate is not None:
                    ws_in = m.split_norm(h)
                    ws_g = 2.0 * torch.sigmoid(m.woodbury_slow_gate(ws_in))
                    h = h + ws_g * self._woodbury_prefill_with_state(
                        m.woodbury, ws_in, h_lex_chunk, state.woodbury_slow
                    )
                    if (
                        m.epistemic_slots is not None
                        and m.epistemic_slots_gate is not None
                        and state.epistemic_slots is not None
                    ):
                        es_g = 2.0 * torch.sigmoid(m.epistemic_slots_gate(ws_in))
                        h = h + es_g * self._epistemic_slots_prefill_with_state(
                            m.epistemic_slots,
                            ws_in,
                            m.woodbury.last_epistemic_uncertainty,
                            h_lex_chunk,
                            state.epistemic_slots,
                        )
                if use_dca:
                    hist.append(h)
                h = _dca_mix_il_pf(h)
                h = h + m.mid_mlp(m.mid_mlp_norm(h))
                if use_dca:
                    hist.append(h)

                h = _dca_mix_il_pf(h)
                c2_in = m.cryst2_norm(h)
                c2_g = 2.0 * torch.sigmoid(m.cryst2_gate(c2_in))
                h = h + c2_g * self._crystal_prefill_with_state(
                    m.crystal_2, c2_in, h_lex_chunk, state.crystal_2
                )
                if use_dca:
                    hist.append(h)

                for blk in m.swa_stages[3]:
                    h = _dca_mix_il_pf(h)
                    c_rng = state.swa_canon_rings[ring_idx] if state.swa_canon_rings is not None else None
                    h = self._swa_block_prefill_with_state(
                        blk, h, state.swa_k_rings[ring_idx], state.swa_v_rings[ring_idx], c_rng
                    )
                    ring_idx += 1
                    if use_dca:
                        hist.append(h)
                h = _dca_mix_il_pf(h)
            else:
                use_ar = m.attnres_mixers is not None
                use_dca = m.dca_mixers is not None
                hist = [h] if (use_ar or use_dca) else []
                dca_idx = 0

                def _dca_mix_hg_pf(h_cur: torch.Tensor) -> torch.Tensor:
                    nonlocal dca_idx
                    if use_dca and m.dca_mixers is not None and len(hist) >= 2:
                        h_cur = m.dca_mixers[dca_idx](h_cur, hist)
                        dca_idx += 1
                    return h_cur

                for blk in m.cortex_pre:
                    if use_dca:
                        h = _dca_mix_hg_pf(h)
                    c_rng = state.swa_canon_rings[ring_idx] if state.swa_canon_rings is not None else None
                    h = self._swa_block_prefill_with_state(
                        blk, h, state.swa_k_rings[ring_idx], state.swa_v_rings[ring_idx], c_rng
                    )
                    ring_idx += 1
                    if use_ar or use_dca:
                        hist.append(h)

                if use_dca:
                    h = _dca_mix_hg_pf(h)
                    h_long_in = h
                elif use_ar and m.attnres_mixers is not None:
                    h_long_in = m.attnres_mixers[0](h, hist)
                else:
                    h_long_in = h
                h_split = m.split_norm(h_long_in)
                c_out = self._crystal_prefill_with_state(m.crystal, h_split, h_lex_chunk, state.crystal_1)
                w_out = self._woodbury_prefill_with_state(m.woodbury, h_split, h_lex_chunk, state.woodbury_slow)
                if m.stream_gate is not None:
                    gates = 2.0 * torch.sigmoid(m.stream_gate(h_split))
                    delta_l = gates[..., 0:1] * c_out + gates[..., 1:2] * w_out
                else:
                    delta_l = c_out + w_out
                if (
                    m.epistemic_slots is not None
                    and m.epistemic_slots_gate is not None
                    and state.epistemic_slots is not None
                ):
                    es_g = 2.0 * torch.sigmoid(m.epistemic_slots_gate(h_split))
                    delta_l = delta_l + es_g * self._epistemic_slots_prefill_with_state(
                        m.epistemic_slots,
                        h_split,
                        m.woodbury.last_epistemic_uncertainty,
                        h_lex_chunk,
                        state.epistemic_slots,
                    )
                h = h + delta_l
                if use_ar or use_dca:
                    hist.append(h)

                if use_dca:
                    h = _dca_mix_hg_pf(h)
                    h_mid_in = h
                elif use_ar and m.attnres_mixers is not None:
                    h_mid_in = m.attnres_mixers[1](h, hist)
                else:
                    h_mid_in = h
                h = h + m.mid_mlp(m.mid_mlp_norm(h_mid_in))
                if use_ar or use_dca:
                    hist.append(h)

                if use_dca:
                    h = _dca_mix_hg_pf(h)
                    h_c2_in = h
                elif use_ar and m.attnres_mixers is not None:
                    h_c2_in = m.attnres_mixers[2](h, hist)
                else:
                    h_c2_in = h
                c2_in = m.cryst2_norm(h_c2_in)
                h = h + (2.0 * torch.sigmoid(m.cryst2_gate(c2_in))) * self._crystal_prefill_with_state(
                    m.crystal_2, c2_in, h_lex_chunk, state.crystal_2
                )
                if use_ar or use_dca:
                    hist.append(h)

                for pi, blk in enumerate(m.cortex_post):
                    if use_dca:
                        h = _dca_mix_hg_pf(h)
                    elif use_ar and m.attnres_mixers is not None:
                        h = m.attnres_mixers[3 + pi](h, hist)
                    c_rng = state.swa_canon_rings[ring_idx] if state.swa_canon_rings is not None else None
                    h = self._swa_block_prefill_with_state(
                        blk, h, state.swa_k_rings[ring_idx], state.swa_v_rings[ring_idx], c_rng
                    )
                    ring_idx += 1
                    if use_ar or use_dca:
                        hist.append(h)

                if use_dca:
                    h = _dca_mix_hg_pf(h)
                elif use_ar and m.attnres_mixers is not None:
                    h = m.attnres_mixers[-1](h, hist)

            z_hat = h  # (1, M, D)
            state.chunk_pos = M
            state.backbone_ticks = M

            # 3. Vectorized byte decoder pass extracting exact decoder conv/KV/BIC rings
            p_chunk = ctx.p_chunk[:, :M]
            if M > 1:
                pc = p_chunk[:, 1:].unsqueeze(-1).to(z_hat.dtype)
                z_tail = pc * z_hat[:, 1:] + (1.0 - pc) * z_hat[:, :-1]
                z_bar = torch.cat((z_hat[:, :1], z_tail), dim=1)
            else:
                z_bar = z_hat

            cs.has_chunk = True
            cs.prev_z_hat.copy_(z_hat[:, -1])
            cs.cur_z_bar.copy_(z_bar[:, -1])

            slot_clamped = ctx.slot_idx.clamp(max=max(0, M - 1))
            z_up = z_bar[0].index_select(0, slot_clamped[0]).unsqueeze(0)
            z_tilde = ch.dechunk_norm(z_up)
            if ch.use_scaled_decoder and ch.byte_offset_emb is not None:
                z_tilde = z_tilde + ch.byte_offset_emb(ctx.offset_in_chunk).to(z_tilde.dtype)

            dec_cat = torch.cat((z_tilde, ctx.c_win8.to(z_tilde.dtype)), dim=-1)
            dec_in = ctx.h_byte + ch.dec_fuse(ch.dec_fuse_norm(dec_cat))

            if (
                ch.use_scaled_decoder
                and ch.dec_dw_conv is not None
                and ch.dec_dw_norm is not None
                and ch.dec_dw_out is not None
                and (ch.dec_dw_gate is not None or ch.dec_dw_in_gate is not None)
                and ch.dec_dw2_norm is not None
                and ch.dec_dw2_conv is not None
                and ch.dec_dw2_out is not None
                and (ch.dec_dw2_gate is not None or ch.dec_dw2_in_gate is not None)
            ):
                xn1 = ch.dec_dw_norm(dec_in)
                xc1, g1 = ch._project_dw_in_gate(xn1, ch.dec_dw_gate, ch.dec_dw_in_gate)
                cs.ring_dec_conv1.copy_(_pad_tail_chronological(xc1, 3))
                cs.dec1_ptr = 0
                c1 = _depthwise_conv4_shift_add(xc1, ch.dec_dw_conv, dilation=1)
                dec_in = dec_in + ch.dec_dw_out(F.silu(c1) * g1)

                xn2 = ch.dec_dw2_norm(dec_in)
                xc2, g2 = ch._project_dw_in_gate(xn2, ch.dec_dw2_gate, ch.dec_dw2_in_gate)
                cs.ring_dec_conv2.copy_(_pad_tail_chronological(xc2, 6))
                cs.dec2_ptr = 0
                c2 = _depthwise_conv4_shift_add(xc2, ch.dec_dw2_conv, dilation=2)
                dec_in = dec_in + ch.dec_dw2_out(F.silu(c2) * g2)

            if (
                ch.use_sym_byte_swa
                and ch.dec_dw3_enabled
                and ch.dec_dw3_norm is not None
                and ch.dec_dw3_conv is not None
                and ch.dec_dw3_out is not None
                and (ch.dec_dw3_gate is not None or ch.dec_dw3_in_gate is not None)
            ):
                xn3 = ch.dec_dw3_norm(dec_in)
                xc3, g3 = ch._project_dw_in_gate(xn3, ch.dec_dw3_gate, ch.dec_dw3_in_gate)
                cs.ring_dec_conv3.copy_(_pad_tail_chronological(xc3, 12))
                cs.dec3_ptr = 0
                c3 = _depthwise_conv4_shift_add(xc3, ch.dec_dw3_conv, dilation=4)
                dec_in = dec_in + ch.dec_dw3_out(F.silu(c3) * g3)

            if (
                ch.use_sym_byte_swa
                and ch.byte_swa_norm is not None
                and ch.byte_swa_qkv is not None
                and ch.byte_swa_out is not None
            ):

                xb_n = ch.byte_swa_norm(dec_in)
                Wb = ch.byte_swa_window
                Hb = ch.byte_swa_heads
                Hdb = ch.byte_swa_head_dim
                qkv_b = ch.byte_swa_qkv(xb_n).view(1, T, 3, Hb, Hdb)
                cos_b_tab, sin_b_tab = get_cached_rope_cos_sin(T, dev, qkv_b.dtype, head_dim=Hdb, base=10000.0)
                if (
                    ch.use_fast_kernels
                    and qkv_b.is_cuda
                    and Hdb in (32, 64)
                    and 1 <= Wb <= 32
                    and _swa_cuda_ext_ready()
                ):
                    yb_h = fast_swa_rope_attn_cuda(qkv_b, cos_b_tab, sin_b_tab, Wb)
                    dec_in = dec_in + ch.byte_swa_out(yb_h.view(1, T, ch.byte_swa_dim))
                    t_st_b = max(0, T - Wb)
                    n_kb = T - t_st_b
                    kb_tail = apply_rope(
                        qkv_b[:, t_st_b:, 1],
                        cos_b_tab[t_st_b:T].view(1, n_kb, 1, -1),
                        sin_b_tab[t_st_b:T].view(1, n_kb, 1, -1),
                    )
                    vb_tail = qkv_b[:, t_st_b:, 2]
                else:
                    qb = qkv_b[:, :, 0].transpose(1, 2)
                    kb = qkv_b[:, :, 1].transpose(1, 2)
                    vb = qkv_b[:, :, 2].transpose(1, 2)
                    cb_4d = cos_b_tab.view(1, 1, T, -1)
                    sb_4d = sin_b_tab.view(1, 1, T, -1)
                    qb = apply_rope(qb, cb_4d, sb_4d)
                    kb = apply_rope(kb, cb_4d, sb_4d)
                    assert ch._bswa_mask0 is not None and ch._bswa_mask_inner is not None
                    yb = sliding_window_sdpa(
                        qb, kb, vb, Wb, ch._bswa_mask0, ch._bswa_mask_inner, ch._bswa_block_q
                    )
                    dec_in = dec_in + ch.byte_swa_out(
                        yb.transpose(1, 2).contiguous().view(1, T, ch.byte_swa_dim)
                    )
                    t_st_b = max(0, T - Wb)
                    kb_tail = kb[:, :, t_st_b:].transpose(1, 2)
                    vb_tail = vb[:, :, t_st_b:].transpose(1, 2)

                slots_b = torch.arange(t_st_b, T, device=dev, dtype=torch.long) % Wb
                cs.ring_byte_k.zero_()
                cs.ring_byte_v.zero_()
                cs.ring_byte_k[:, slots_b] = kb_tail
                cs.ring_byte_v[:, slots_b] = vb_tail

            need_engram = (
                ch.use_byte_engram
                and ch.engram_emb2 is not None
                and ch.engram_emb3 is not None
                and ch.engram_norm is not None
                and ch.engram_proj is not None
                and ch.engram_gate is not None
            )
            need_bic = (
                ch.use_byte_induct_cache
                and ch.byte_induct_cache is not None
                and cs.byte_induct is not None
            )
            t_pad6 = F.pad(token_ids.long(), (6, 0), value=0) if (need_engram or need_bic) else None

            if need_engram and t_pad6 is not None:
                t0 = t_pad6[:, 6 : 6 + T]
                t1 = t_pad6[:, 5 : 5 + T]
                t2 = t_pad6[:, 4 : 4 + T]
                base2 = t1 * 263 + t0 * 1009
                raw3 = t2 * 691 + base2 + 17
                use_4g = ch.engram_use_4gram and ch.engram_emb4 is not None
                raw4 = (t_pad6[:, 3 : 3 + T] * 4099 + raw3 + 31) if use_4g else None
                slots_e = ch.engram_slots
                if slots_e > 0 and (slots_e & (slots_e - 1)) == 0:
                    em = slots_e - 1
                    h2 = base2 & em
                    h3 = raw3 & em
                    h4 = (raw4 & em) if raw4 is not None else None
                else:
                    h2 = torch.remainder(base2, slots_e)
                    h3 = torch.remainder(raw3, slots_e)
                    h4 = torch.remainder(raw4, slots_e) if raw4 is not None else None
                assert ch.engram_emb2 is not None and ch.engram_emb3 is not None
                assert ch.engram_norm is not None and ch.engram_proj is not None and ch.engram_gate is not None
                e_sum = ch.engram_emb2(h2) + ch.engram_emb3(h3)
                if use_4g and ch.engram_emb4 is not None and h4 is not None:
                    e_sum = e_sum + ch.engram_emb4(h4)
                e_eng = e_sum.to(dec_in.dtype)
                eg = torch.sigmoid(ch.engram_gate(dec_in))
                dec_in = dec_in + eg * ch.engram_proj(ch.engram_norm(e_eng))

            if need_bic and ch.byte_induct_cache is not None and cs.byte_induct is not None:
                dec_in = dec_in + self._bic_prefill_with_state(
                    ch.byte_induct_cache, token_ids, dec_in, t_pad6, cs.byte_induct
                )

            h_out = dec_in + ch.dec_mlp(ch.dec_norm(dec_in))
            state.total_bytes = T
            if return_all_logits:
                logits = m.lm_head(m.norm_f(h_out))
            else:
                logits = m.lm_head(m.norm_f(h_out[:, -1]))
            return logits, state
        finally:
            if was_training:
                m.train(True)

    @torch.no_grad()
    def prefill(
        self,
        token_ids: torch.Tensor,
        state: Optional[CalciteStreamState] = None,
        return_all_logits: bool = False,
    ) -> Tuple[torch.Tensor, CalciteStreamState]:
        """
        Streams a prompt `(B, T)` into strict `O(1)` ring state.
        Uses exact vectorized chunked prefill when starting from empty state (for all standard
        `CalciteV2`/`v3`/`v3.1` configs without multi-pass feedback/TR), matching byte-by-byte
        `step()` to `< 2e-5` with 10x–30x faster prompt ingestion.
        """
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        B, T = token_ids.shape
        if state is None:
            state = self.init_state(batch_size=B, device=token_ids.device)
        if T == 0:
            raise ValueError("prefill requires T >= 1")

        # Exact vectorized fresh prefill path (supports both B=1 and unsynced B>1 cleanly)
        can_vec = (
            state.total_bytes == 0
            and T >= 2
            and not self.cfg.use_latent_feedback
            and not self.cfg.use_temporal_residual
        )
        if can_vec:
            if B == 1 and state.item_states is None:
                return self._prefill_fresh_b1(token_ids, state, return_all_logits=return_all_logits)
            if state.item_states is not None and B == len(state.item_states):
                outs = []
                for b in range(B):
                    lg_b, _ = self._prefill_fresh_b1(
                        token_ids[b : b + 1], state.item_states[b], return_all_logits=return_all_logits
                    )
                    outs.append(lg_b)
                state.total_bytes = state.item_states[0].total_bytes
                state.backbone_ticks = sum(s.backbone_ticks for s in state.item_states)
                state.chunk_pos = max(s.chunk_pos for s in state.item_states)
                return torch.cat(outs, dim=0), state

        all_logits = []
        last_logits = None
        for t in range(T):
            lg, state = self.step(token_ids[:, t], state)
            last_logits = lg
            if return_all_logits:
                all_logits.append(lg)
        if return_all_logits:
            return torch.stack(all_logits, dim=1), state
        assert last_logits is not None
        return last_logits, state

    # ------------------------------------------------------------------
    # Opt-In B=1 GPU Static CUDA Graphs (FP32 Accumulation, Zero Drift)
    # ------------------------------------------------------------------

    def _can_use_cuda_graph(self, state: CalciteStreamState) -> bool:
        if state.item_states is not None:
            return False
        if self.cfg.use_latent_feedback or self.cfg.use_temporal_residual or self.cfg.use_canon_chunks:
            return False
        ch = self.model.byte_chunker
        min_bytes = max(1, ch.byte_swa_window if ch.use_sym_byte_swa else 1)
        if state.total_bytes < min_bytes or state.chunk_pos < self.cfg.window_size or not state.chunker.has_chunk:
            return False
        if self._cg_bundle is not None and self._cg_bundle.state_id == id(state):
            if (
                state.total_bytes >= self._cg_bundle.max_byte_pos
                or state.chunk_pos >= self._cg_bundle.max_chunk_pos
            ):
                self._restore_rings_for_eager(state)
                self._cg_bundle = None
                return False
        return True

    @staticmethod
    def _canonicalize_all_rings(state: CalciteStreamState) -> None:
        """Aligns every circular ring to chronological order (`ptr == 0`) once before CUDA Graph capture."""
        cs = state.chunker
        _canonicalize_circular_ring(cs.ring_char, cs.char_ptr)
        cs.char_ptr = 0
        _canonicalize_circular_ring(cs.ring_rchar, cs.rchar_ptr)
        cs.rchar_ptr = 0
        if cs.ring_tok3.size(1) >= 3:
            _canonicalize_circular_ring(cs.ring_tok3, cs.tok3_ptr)
            cs.tok3_ptr = 0
        if cs.ring_enc_conv2.size(1) > 0:
            _canonicalize_circular_ring(cs.ring_enc_conv2, cs.enc2_ptr)
            cs.enc2_ptr = 0
        if cs.ring_dec_conv1.size(1) > 0:
            _canonicalize_circular_ring(cs.ring_dec_conv1, cs.dec1_ptr)
            cs.dec1_ptr = 0
        if cs.ring_dec_conv2.size(1) > 0:
            _canonicalize_circular_ring(cs.ring_dec_conv2, cs.dec2_ptr)
            cs.dec2_ptr = 0
        if cs.ring_dec_conv3.size(1) > 0:
            _canonicalize_circular_ring(cs.ring_dec_conv3, cs.dec3_ptr)
            cs.dec3_ptr = 0
        if cs.ring_byte_k.size(1) > 0:
            _canonicalize_swa_ring(cs.ring_byte_k, cs.step_idx)
            _canonicalize_swa_ring(cs.ring_byte_v, cs.step_idx)
        if cs.byte_induct is not None:
            _canonicalize_circular_ring(cs.byte_induct.ring_tok7, cs.byte_induct.ptr7)
            cs.byte_induct.ptr7 = 0

        for kr, vr in zip(state.swa_k_rings, state.swa_v_rings):
            _canonicalize_swa_ring(kr, state.chunk_pos)
            _canonicalize_swa_ring(vr, state.chunk_pos)

        for wst in (state.woodbury_fast, state.woodbury_slow):
            if wst is not None:
                _canonicalize_circular_ring(wst.z_ring3, wst.z3_ptr)
                wst.z3_ptr = 0
                if wst.sal_ring.size(1) > 0:
                    _canonicalize_circular_ring(wst.sal_ring, wst.latch_ptr)
                    _canonicalize_circular_ring(wst.rw_ring, wst.latch_ptr)
                    _canonicalize_circular_ring(wst.epi_ring, wst.latch_ptr)
                wst.latch_ptr = 0

        for cst in (state.crystal_1, state.crystal_2):
            _canonicalize_circular_ring(cst.sal_w_ring, cst.w_ptr)
            _canonicalize_circular_ring(cst.u_w_ring, cst.w_ptr)
            cst.w_ptr = 0
            _canonicalize_circular_ring(cst.sal_r_ring, cst.r_ptr)
            _canonicalize_circular_ring(cst.u_r_ring, cst.r_ptr)
            cst.r_ptr = 0
            _canonicalize_circular_ring(cst.z_mean_ring, cst.zmean_ptr)
            cst.zmean_ptr = 0

        if state.epistemic_slots is not None:
            _canonicalize_circular_ring(state.epistemic_slots.k_ring, state.epistemic_slots.k_ptr)
            state.epistemic_slots.k_ptr = 0

    @staticmethod
    def _collect_state_tensors(state: CalciteStreamState) -> List[torch.Tensor]:
        cs = state.chunker
        ts: List[torch.Tensor] = [
            cs.ring_char, cs.ring_rchar, cs.prev_k_r, cs.prev_h_ent,
            cs.prev_z_hat, cs.cur_z_bar, cs.ring_dec_conv1, cs.ring_dec_conv2,
            cs.ring_enc_conv2, cs.ring_dec_conv3, cs.ring_byte_k, cs.ring_byte_v,
            cs.ring_tok3, cs.run_len, cs.cp_run, cs.cont_streak, cs.credits,
        ]
        if cs.byte_induct is not None:
            ts.extend([cs.byte_induct.bank, cs.byte_induct.ring_tok7, cs.byte_induct.prev_h_ctx])
        ts.extend(state.swa_k_rings)
        ts.extend(state.swa_v_rings)
        for wst in (state.woodbury_fast, state.woodbury_slow):
            if wst is not None:
                ts.extend([
                    wst.M_mat, wst.P_mat, wst.z_cumsum, wst.z_ring3,
                    wst.sal_ring, wst.rw_ring, wst.epi_ring,
                    wst.buf_x, wst.buf_k, wst.buf_k_epi,
                ])
        for cst in (state.crystal_1, state.crystal_2):
            ts.extend([
                cst.s_bh, cst.z_cumsum, cst.p_f_m1, cst.p_f_m2,
                cst.zc_win, cst.u_win, cst.win_valid,
                cst.sal_w_ring, cst.u_w_ring, cst.sal_r_ring, cst.u_r_ring,
                cst.z_mean_ring, cst.buf_x, cst.buf_zc, cst.buf_u, cst.buf_kw,
            ])
        if state.epistemic_slots is not None:
            ts.extend([state.epistemic_slots.kvf_bank, state.epistemic_slots.k_ring])
        return ts

    @staticmethod
    def _capture_graph_preserving_state(
        fn,
        tensors_to_preserve: List[torch.Tensor],
    ) -> torch.cuda.CUDAGraph:
        """Warms up and captures `fn` into a `CUDAGraph` without altering live `tensors_to_preserve`."""
        snaps = [t.clone() for t in tensors_to_preserve]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                fn()
                for t, sp in zip(tensors_to_preserve, snaps):
                    t.copy_(sp)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        for t, sp in zip(tensors_to_preserve, snaps):
            t.copy_(sp)
        torch.cuda.synchronize()
        return g

    def _graph_encode_body(self, state: CalciteStreamState, b: _CudaGraphBundle) -> None:
        m = self.model
        ch = m.byte_chunker
        cs = state.chunker
        token_t = b.tok_in
        e_tok_t = m.tok_emb(token_t)
        dt = e_tok_t.dtype
        log_v = math.log(max(2, ch.vocab_size))

        ct = e_tok_t[..., : ch.d_byte]
        c_win8_t = torch.cat(
            [ct] + [cs.ring_char[:, 6 - k] for k in range(7)], dim=-1
        )
        _shift_ring_append_graph_safe(cs.ring_char, ct)

        rc_t = ch.router_emb(token_t)
        if rc_t.dtype != dt:
            rc_t = rc_t.to(dt)
        r_win4_t = torch.cat([rc_t, cs.ring_rchar[:, 2], cs.ring_rchar[:, 1], cs.ring_rchar[:, 0]], dim=-1)
        _shift_ring_append_graph_safe(cs.ring_rchar, rc_t)

        if ch.use_byte_engram and cs.ring_tok3.size(1) >= 3:
            _shift_ring_append_graph_safe(cs.ring_tok3, token_t)

        c_norm_t = ch.enc_norm_in(c_win8_t)
        h_enc0 = e_tok_t + ch.enc_up(c_norm_t)
        h_byte_t = h_enc0 + ch.enc_mlp(ch.enc_norm(h_enc0))
        if (
            ch.use_sym_byte_swa
            and ch.enc_dw2_norm is not None
            and ch.enc_dw2_conv is not None
            and ch.enc_dw2_out is not None
            and (ch.enc_dw2_gate is not None or ch.enc_dw2_in_gate is not None)
        ):
            xe = ch.enc_dw2_norm(h_byte_t)
            xc_e, ge = ch._project_dw_in_gate(xe, ch.enc_dw2_gate, ch.enc_dw2_in_gate)
            ce = _dw_conv4_graph_step(xc_e, cs.ring_enc_conv2, ch.enc_dw2_conv, dilation=2)
            h_byte_t = h_byte_t + ch.enc_dw2_out(F.silu(ce) * ge)

        h_rt = F.silu(ch._r_lin(ch.router_in_proj, ch.router_in_norm(r_win4_t)))
        q_rt = F.normalize(ch._r_lin(ch.router_q, h_rt).float(), dim=-1)
        k_rt = F.normalize(ch._r_lin(ch.router_k, h_rt).float(), dim=-1)
        d_cos_t = 0.5 * (1.0 - (q_rt * cs.prev_k_r).sum(dim=-1))

        loc_t = ch._r_lin(ch.router_ent_head, ch.router_ent_norm(h_rt)).float()
        lp_t = F.log_softmax(loc_t, dim=-1)
        h_norm_t = (-(lp_t.exp() * lp_t).sum(dim=-1) / log_v).clamp(0.0, 1.5)
        dh_t = F.relu(h_norm_t - cs.prev_h_ent)

        wf = ch.router_feat_weights.float()
        ctx_t = ch._r_lin(ch.router_ctx, h_rt).float().squeeze(-1)
        p_t = torch.sigmoid(wf[0] * d_cos_t + wf[1] * h_norm_t + wf[2] * dh_t + ctx_t)
        vt = token_t != ch.pad_token_id

        if ch.byte_chunk_mode == "utf8":
            _, is_cont_t, cl_t = _utf8_tensor_flags(token_t)
            cs.credits.copy_(
                torch.where(vt, torch.clamp(cs.credits + ch.refill_int, max=ch.burst_int), cs.credits)
            )
            is_cont_vt = is_cont_t & vt
            shielded_t = is_cont_vt & (cs.cont_streak < 3)
            can_fire_t = vt & (~shielded_t)
            can_soft_t = (cs.credits >= ch.cost_int) & can_fire_t & (p_t >= 0.5)
            must_span_t = (cs.cp_run >= ch.max_chunk_bytes) | ((cs.run_len + cl_t) > ch.max_utf8_bytes)
            fire_t = can_fire_t & (can_soft_t | must_span_t)
            cs.credits.copy_(torch.where(fire_t, torch.clamp(cs.credits - ch.cost_int, min=0), cs.credits))
            is_lead_t = (vt & (~is_cont_t)).long()
            cs.cp_run.copy_(torch.where(vt, torch.where(fire_t, is_lead_t, cs.cp_run + is_lead_t), cs.cp_run))
            cs.cont_streak.copy_(torch.where(is_cont_vt, cs.cont_streak + 1, torch.zeros_like(cs.cont_streak)))
        else:
            fire_t = vt & ((p_t >= 0.5) | (cs.run_len >= ch.max_chunk_bytes))

        offset_t = torch.where(fire_t, torch.zeros_like(cs.run_len), cs.run_len).clamp(0, ch.max_utf8_bytes)
        cs.run_len.copy_(torch.where(vt, torch.where(fire_t, torch.ones_like(cs.run_len), cs.run_len + 1), cs.run_len))
        cs.prev_k_r.copy_(k_rt)
        cs.prev_h_ent.copy_(h_norm_t)

        b.fire_out.copy_(fire_t)
        b.p_out.copy_(p_t)
        b.h_byte_out.copy_(h_byte_t)
        b.c_norm_out.copy_(c_norm_t)
        b.win8_out.copy_(c_win8_t)
        b.offset_out.copy_(offset_t)

    def _graph_decode_body(self, state: CalciteStreamState, b: _CudaGraphBundle, update_chunk: bool) -> None:
        m = self.model
        ch = m.byte_chunker
        cs = state.chunker
        dt = b.h_byte_out.dtype
        if update_chunk:
            pc = b.p_out.unsqueeze(-1).to(dt)
            z_bar_new = pc * b.z_hat_out + (1.0 - pc) * cs.prev_z_hat
            cs.cur_z_bar.copy_(z_bar_new)
            cs.prev_z_hat.copy_(b.z_hat_out)

        z_tilde = ch.dechunk_norm(cs.cur_z_bar)
        if ch.use_scaled_decoder and ch.byte_offset_emb is not None:
            z_tilde = z_tilde + ch.byte_offset_emb(b.offset_out).to(z_tilde.dtype)

        dec_cat = torch.cat((z_tilde, b.c_norm_out.to(z_tilde.dtype)), dim=-1)
        dec_in = b.h_byte_out + ch.dec_fuse(ch.dec_fuse_norm(dec_cat))

        if (
            ch.use_scaled_decoder
            and ch.dec_dw_conv is not None
            and ch.dec_dw_norm is not None
            and ch.dec_dw_out is not None
            and (ch.dec_dw_gate is not None or ch.dec_dw_in_gate is not None)
            and ch.dec_dw2_norm is not None
            and ch.dec_dw2_conv is not None
            and ch.dec_dw2_out is not None
            and (ch.dec_dw2_gate is not None or ch.dec_dw2_in_gate is not None)
        ):
            xn1 = ch.dec_dw_norm(dec_in)
            xc1, g1 = ch._project_dw_in_gate(xn1, ch.dec_dw_gate, ch.dec_dw_in_gate)
            c1 = _dw_conv4_graph_step(xc1, cs.ring_dec_conv1, ch.dec_dw_conv, dilation=1)
            dec_in = dec_in + ch.dec_dw_out(F.silu(c1) * g1)

            xn2 = ch.dec_dw2_norm(dec_in)
            xc2, g2 = ch._project_dw_in_gate(xn2, ch.dec_dw2_gate, ch.dec_dw2_in_gate)
            c2 = _dw_conv4_graph_step(xc2, cs.ring_dec_conv2, ch.dec_dw2_conv, dilation=2)
            dec_in = dec_in + ch.dec_dw2_out(F.silu(c2) * g2)

        if (
            ch.use_sym_byte_swa
            and ch.dec_dw3_enabled
            and ch.dec_dw3_norm is not None
            and ch.dec_dw3_conv is not None
            and ch.dec_dw3_out is not None
            and (ch.dec_dw3_gate is not None or ch.dec_dw3_in_gate is not None)
        ):
            xn3 = ch.dec_dw3_norm(dec_in)
            xc3, g3 = ch._project_dw_in_gate(xn3, ch.dec_dw3_gate, ch.dec_dw3_in_gate)
            c3 = _dw_conv4_graph_step(xc3, cs.ring_dec_conv3, ch.dec_dw3_conv, dilation=4)
            dec_in = dec_in + ch.dec_dw3_out(F.silu(c3) * g3)

        if (
            ch.use_sym_byte_swa
            and ch.byte_swa_norm is not None
            and ch.byte_swa_qkv is not None
            and ch.byte_swa_out is not None
            and b.byte_cos is not None
            and b.byte_sin is not None
        ):

            xb_n = ch.byte_swa_norm(dec_in)
            qkv_b = ch.byte_swa_qkv(xb_n).view(1, 3, ch.byte_swa_heads, ch.byte_swa_head_dim)
            cos_b = b.byte_cos.index_select(0, b.pos_b).view(1, 1, -1)
            sin_b = b.byte_sin.index_select(0, b.pos_b).view(1, 1, -1)
            qb = apply_rope(qkv_b[:, 0], cos_b, sin_b)
            kb = apply_rope(qkv_b[:, 1], cos_b, sin_b)
            vb = qkv_b[:, 2]
            _shift_ring_append_graph_safe(cs.ring_byte_k, kb)
            _shift_ring_append_graph_safe(cs.ring_byte_v, vb)
            sc_b = 1.0 / math.sqrt(float(ch.byte_swa_head_dim))
            scores_b = torch.matmul(qb.float().unsqueeze(2), cs.ring_byte_k.float().permute(0, 2, 3, 1)) * sc_b
            probs_b = F.softmax(scores_b, dim=-1).to(qkv_b.dtype)
            yb = torch.matmul(probs_b, cs.ring_byte_v.permute(0, 2, 1, 3)).reshape(1, ch.byte_swa_dim)
            dec_in = dec_in + ch.byte_swa_out(yb)

        engram_ring_len = cs.ring_tok3.size(1)
        if (
            ch.use_byte_engram
            and ch.engram_emb2 is not None
            and ch.engram_emb3 is not None
            and ch.engram_norm is not None
            and ch.engram_proj is not None
            and ch.engram_gate is not None
            and engram_ring_len >= 3
        ):
            t0 = cs.ring_tok3[:, -1]
            t1 = cs.ring_tok3[:, -2]
            t2 = cs.ring_tok3[:, -3]
            base2 = t1 * 263 + t0 * 1009
            raw3 = t2 * 691 + base2 + 17
            use_4g = ch.engram_use_4gram and ch.engram_emb4 is not None and engram_ring_len >= 4
            raw4 = (cs.ring_tok3[:, -4] * 4099 + raw3 + 31) if use_4g else None
            slots = ch.engram_slots
            if slots > 0 and (slots & (slots - 1)) == 0:
                em = slots - 1
                h2 = base2 & em
                h3 = raw3 & em
                h4 = (raw4 & em) if raw4 is not None else None
            else:
                h2 = torch.remainder(base2, slots)
                h3 = torch.remainder(raw3, slots)
                h4 = torch.remainder(raw4, slots) if raw4 is not None else None
            e_sum = ch.engram_emb2(h2) + ch.engram_emb3(h3)
            if use_4g and ch.engram_emb4 is not None and h4 is not None:
                e_sum = e_sum + ch.engram_emb4(h4)
            e_eng = e_sum.to(dt)
            eg = torch.sigmoid(ch.engram_gate(dec_in))
            dec_in = dec_in + eg * ch.engram_proj(ch.engram_norm(e_eng))

        if (
            ch.use_byte_induct_cache
            and ch.byte_induct_cache is not None
            and cs.byte_induct is not None
        ):
            bic = ch.byte_induct_cache
            bst = cs.byte_induct
            tok_t = b.tok_in
            S = bic.n_slots
            # 1. Write using chronological ring_tok7 (`[t-7 .. t-1]`)
            b6, b5, b4, b3, b2, b1, b0 = (
                bst.ring_tok7[:, 0], bst.ring_tok7[:, 1], bst.ring_tok7[:, 2],
                bst.ring_tok7[:, 3], bst.ring_tok7[:, 4], bst.ring_tok7[:, 5], bst.ring_tok7[:, 6],
            )
            base4_w = b3 * 32749 + b2 * 8191 + b1 * 2053 + b0 * 521
            raw4_w = base4_w + 17
            raw7_w = b6 * 1000003 + b5 * 524287 + b4 * 131071 + base4_w + 31
            if S > 0 and (S & (S - 1)) == 0:
                sm = S - 1
                sw4, sw7 = raw4_w & sm, raw7_w & sm
            else:
                sw4, sw7 = torch.remainder(raw4_w, S), torch.remainder(raw7_w, S)

            v_flat = bic.val_emb(tok_t.long().clamp(0, bic.vocab_size - 1)).float()
            v_2 = v_flat.view(1, 2, bic.d_v)
            h_prev_dt = bst.prev_h_ctx if bst.prev_h_ctx.dtype == dt else bst.prev_h_ctx.to(dt)
            k_sig = bic._safe_unit(bic.k_proj(h_prev_dt).view(1, 2, bic.d_sig))
            beta_t, _ = bic._compute_beta(h_prev_dt, v_flat)
            ones_u = torch.ones((1, 2, 1), device=dec_in.device, dtype=torch.float32)
            u_new_2 = torch.cat((k_sig, v_2, ones_u), dim=-1)  # (1, 2, d_u)
            for sc_i, sw_i in ((0, sw4), (1, sw7)):
                idx_e = sw_i.view(1, 1, 1).expand(1, 1, bic.d_u)
                u_old = bst.bank[:, sc_i].gather(1, idx_e)
                bt_i = beta_t[:, sc_i].view(1, 1, 1)
                u_upd = (1.0 - bt_i) * u_old + bt_i * u_new_2[:, sc_i : sc_i + 1]
                bst.bank[:, sc_i].scatter_(1, idx_e, u_upd)

            # 2. Shift ring_tok7 and hash current query
            _shift_ring_append_graph_safe(bst.ring_tok7, tok_t.long())
            q6, q5, q4, q3, q2, q1, q0 = (
                bst.ring_tok7[:, 0], bst.ring_tok7[:, 1], bst.ring_tok7[:, 2],
                bst.ring_tok7[:, 3], bst.ring_tok7[:, 4], bst.ring_tok7[:, 5], bst.ring_tok7[:, 6],
            )
            base4_q = q3 * 32749 + q2 * 8191 + q1 * 2053 + q0 * 521
            raw4_q = base4_q + 17
            raw7_q = q6 * 1000003 + q5 * 524287 + q4 * 131071 + base4_q + 31
            if S > 0 and (S & (S - 1)) == 0:
                sm = S - 1
                sq4, sq7 = raw4_q & sm, raw7_q & sm
            else:
                sq4, sq7 = torch.remainder(raw4_q, S), torch.remainder(raw7_q, S)

            u_r0 = bst.bank[:, 0].gather(1, sq4.view(1, 1, 1).expand(1, 1, bic.d_u)).squeeze(1)
            u_r1 = bst.bank[:, 1].gather(1, sq7.view(1, 1, 1).expand(1, 1, bic.d_u)).squeeze(1)
            u_read = torch.stack((u_r0, u_r1), dim=1)
            q_sig = bic._safe_unit(bic.q_proj(dec_in).view(1, 2, bic.d_sig))
            out_v = bic._decode_read_u(u_read, q_sig).to(dt)
            bst.prev_h_ctx.copy_(dec_in.float())
            gate = 2.0 * torch.sigmoid(bic.out_gate(dec_in))
            dec_in = dec_in + gate * bic.out_proj(bic.out_norm(out_v))

        h_out = dec_in + ch.dec_mlp(ch.dec_norm(dec_in))
        b.logits_out.copy_(m.lm_head(m.norm_f(h_out)))

    def _graph_swa_step(
        self,
        blk: LocalSWABlock,
        x_t: torch.Tensor,
        k_ring: torch.Tensor,
        v_ring: torch.Tensor,
        b: _CudaGraphBundle,
    ) -> torch.Tensor:
        attn = blk.attn
        x1 = blk.norm1(x_t)
        qkv = attn.qkv(x1).view(1, 3, attn.n_heads, attn.head_dim)
        cos = b.chunk_cos.index_select(0, b.pos_c).view(1, 1, -1)
        sin = b.chunk_sin.index_select(0, b.pos_c).view(1, 1, -1)
        q = apply_rope(qkv[:, 0], cos, sin)
        k = apply_rope(qkv[:, 1], cos, sin)
        v = qkv[:, 2]
        _shift_ring_append_graph_safe(k_ring, k)
        _shift_ring_append_graph_safe(v_ring, v)
        scale = 1.0 / math.sqrt(float(attn.head_dim))
        scores = torch.matmul(q.float().unsqueeze(2), k_ring.float().permute(0, 2, 3, 1)) * scale
        probs = F.softmax(scores, dim=-1).to(qkv.dtype)
        y = torch.matmul(probs, v_ring.permute(0, 2, 1, 3)).reshape(1, attn.d_model)
        h = x_t + attn.out_proj(y)
        return h + blk.mlp(blk.norm2(h))

    @staticmethod
    def _graph_woodbury_step(
        wood: CalciteV2WoodburyLayer,
        x_t: torch.Tensor,
        h_lex_t: torch.Tensor,
        st: WoodburyStreamState,
        sc_t: torch.Tensor,
        buf_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dt = x_t.dtype
        lam_reg = 1.0 / wood.inv_lam
        x_u = wood._safe_unit(x_t)
        w_lex = torch.sigmoid(wood.lex_logit.float())
        z = wood._safe_unit(w_lex * wood._safe_unit(h_lex_t) + (1.0 - w_lex) * x_u)

        sc_t.add_(1.0)
        st.z_cumsum.add_(z)
        zc = (z - st.z_cumsum / (sc_t.view(1, 1) + 2.0)).to(dt)
        z0 = zc
        z12 = st.z_ring3[:, 2] + st.z_ring3[:, 1]
        z3 = st.z_ring3[:, 0] + 0.25 * st.z_ring3[:, 1]
        _shift_ring_append_graph_safe(st.z_ring3, zc)

        sc_raw = math.sqrt(wood.D)
        f1_r = wood._safe_unit(wood.f1_proj(z0).view(1, wood.H_epi, wood.D), sc_raw)
        f2_r = wood._safe_unit(wood.f2_proj(z12).view(1, wood.H_epi, wood.D), sc_raw)
        f3_r = wood._safe_unit(wood.f3_proj(z3).view(1, wood.H_epi, wood.D), sc_raw)
        phi_epi = wood._safe_unit(torch.tanh(f1_r * 1.2) * torch.tanh(f2_r * 1.2) * torch.tanh(f3_r * 1.2))
        phi_lin = wood._safe_unit(f1_r + f2_r + f3_r)
        phi_rw = wood._safe_unit(0.35 * phi_lin + 0.94 * phi_epi)

        sal_t = wood.subj_salience(x_t).float()
        if wood.L_latch > 1:
            sal_win = torch.cat((st.sal_ring, sal_t.unsqueeze(1)), dim=1).transpose(1, 2)
            rw_win = torch.cat((st.rw_ring, phi_rw.unsqueeze(1)), dim=1).permute(0, 2, 3, 1)
            epi_win = torch.cat((st.epi_ring, phi_epi.unsqueeze(1)), dim=1).permute(0, 2, 3, 1)
            _shift_ring_append_graph_safe(st.sal_ring, sal_t)
            _shift_ring_append_graph_safe(st.rw_ring, phi_rw)
            _shift_ring_append_graph_safe(st.epi_ring, phi_epi)
        else:
            sal_win = sal_t.unsqueeze(-1)
            rw_win = phi_rw.unsqueeze(-1)
            epi_win = phi_epi.unsqueeze(-1)

        att = F.softmax(sal_win + wood.subj_lag_bias.float().unsqueeze(0), dim=-1)
        q_rw_t = wood._safe_unit((rw_win * att.unsqueeze(2)).sum(dim=-1)).to(dt)
        q_epi_t = wood._safe_unit((epi_win * att.unsqueeze(2)).sum(dim=-1)).to(dt)

        if wood.H_prose > 0 and wood.prose_q_proj is not None and wood.prose_k_proj is not None:
            q_prose = wood._safe_unit(wood.prose_q_proj(x_t).view(1, wood.H_prose, wood.D)).to(dt)
            k_prose = wood._safe_unit(wood.prose_k_proj(x_t).view(1, wood.H_prose, wood.D)).to(dt)
            q_rw_all = torch.cat((q_rw_t, q_prose), dim=1)
            k_src_all = torch.cat((q_rw_t, k_prose), dim=1)
            q_epi_all = torch.cat((q_epi_t, q_epi_t[:, : wood.H_prose]), dim=1)
        else:
            q_rw_all, k_src_all, q_epi_all = q_rw_t, q_rw_t, q_epi_t

        BH_b = wood.H_banks
        if wood.K_sub > 1 and wood.bank_router is not None:
            K = wood.K_sub
            H_eb = wood.H_epi * K
            r_read_st, _, _ = wood._route_sub_banks(q_rw_t)

            def _exp_t(t_all: torch.Tensor) -> torch.Tensor:
                t_epi = t_all[:, : wood.H_epi].unsqueeze(2).expand(1, wood.H_epi, K, wood.D).reshape(1, H_eb, wood.D)
                return torch.cat((t_epi, t_all[:, wood.H_epi :]), dim=1) if wood.H_prose > 0 else t_epi

            q_bh = _exp_t(q_rw_all).float().reshape(BH_b, wood.D, 1)
            qe_bh = _exp_t(q_epi_all).float().reshape(BH_b, wood.D, 1)
            read_b = torch.bmm(st.M_mat, q_bh).squeeze(-1).reshape(1, BH_b, wood.D)
            pq_t = torch.bmm(st.P_mat, qe_bh)
            u_b = ((qe_bh * pq_t).sum(dim=1).squeeze(-1) * lam_reg).clamp(0.0, 1.5).reshape(1, BH_b)
            read_epi = (read_b[:, :H_eb].view(1, wood.H_epi, K, wood.D) * r_read_st.unsqueeze(-1)).sum(dim=2)
            u_epi_c = (u_b[:, :H_eb].view(1, wood.H_epi, K) * r_read_st).sum(dim=2)
            if wood.H_prose > 0:
                read_t = torch.cat((read_epi, read_b[:, H_eb:]), dim=1).reshape(1, wood.total_dim).to(dt)
                u_t = torch.cat((u_epi_c, u_b[:, H_eb:]), dim=1).to(dt)
            else:
                read_t = read_epi.reshape(1, wood.total_dim).to(dt)
                u_t = u_epi_c.to(dt)
        else:
            q_bh = q_rw_all.float().reshape(BH_b, wood.D, 1)
            qe_bh = q_epi_all.float().reshape(BH_b, wood.D, 1)
            read_t = torch.bmm(st.M_mat, q_bh).squeeze(-1).reshape(1, wood.total_dim).to(dt)
            pq_t = torch.bmm(st.P_mat, qe_bh)
            u_t = ((qe_bh * pq_t).sum(dim=1).squeeze(-1) * lam_reg).clamp(0.0, 1.5).reshape(1, wood.H).to(dt)

        st.buf_x[:, buf_idx].copy_(x_t)
        st.buf_k[:, buf_idx].copy_(k_src_all)
        st.buf_k_epi[:, buf_idx].copy_(q_epi_all)

        if buf_idx + 1 == wood.C:
            w_scores = F.softmax(wood.write_pool_score(st.buf_x).float(), dim=1)
            kt_h = wood._safe_unit((st.buf_k.float() * w_scores.unsqueeze(-1)).sum(dim=1))
            ket_h = wood._safe_unit((st.buf_k_epi.float() * w_scores.unsqueeze(-1)).sum(dim=1))
            v_tok = wood.v_proj(st.buf_x).view(1, wood.C, wood.H, wood.D)
            vt_h = (v_tok.float() * w_scores.unsqueeze(-1)).sum(dim=1)
            z_head = (st.buf_x * w_scores.to(dt).mean(dim=-1, keepdim=True)).sum(dim=1)

            if (
                wood.use_deltanet2_gates
                and wood.erase_vec_proj is not None
                and wood.write_vec_proj is not None
                and wood.alpha_vec_proj is not None
            ):
                et_h = torch.sigmoid(wood.erase_vec_proj(z_head)).float().view(1, wood.H, wood.D)
                wt_h = torch.sigmoid(wood.write_vec_proj(z_head)).float().view(1, wood.H, wood.D)
                at_h = torch.sigmoid(wood.alpha_vec_proj(z_head)).float().view(1, wood.H, wood.D)
                bt_h = et_h.mean(dim=-1)
                gt_h = at_h.mean(dim=-1)
            else:
                et_h = wt_h = at_h = None
                assert wood.beta_proj is not None and wood.gamma_proj is not None
                bt_h = torch.sigmoid(wood.beta_proj(z_head)).float()
                gt_h = torch.sigmoid(wood.gamma_proj(z_head)).float()

            if wood.K_sub > 1 and wood.bank_router is not None:
                K = wood.K_sub
                H_eb = wood.H_epi * K
                r_write_st, _, _ = wood._route_sub_banks(kt_h[:, : wood.H_epi])

                def _exp_w(t_all: torch.Tensor) -> torch.Tensor:
                    t_epi = t_all[:, : wood.H_epi].unsqueeze(2).expand(1, wood.H_epi, K, wood.D).reshape(1, H_eb, wood.D)
                    return torch.cat((t_epi, t_all[:, wood.H_epi :]), dim=1) if wood.H_prose > 0 else t_epi

                kt = _exp_w(kt_h).reshape(BH_b, wood.D, 1)
                ket = _exp_w(ket_h).reshape(BH_b, wood.D, 1)
                vt = _exp_w(vt_h).reshape(BH_b, wood.D, 1)
                bt_epi = (bt_h[:, : wood.H_epi].unsqueeze(-1) * r_write_st).reshape(1, H_eb)
                gt_epi = (1.0 - r_write_st * (1.0 - gt_h[:, : wood.H_epi].unsqueeze(-1))).reshape(1, H_eb)
                bt = (torch.cat((bt_epi, bt_h[:, wood.H_epi :]), dim=1) if wood.H_prose > 0 else bt_epi).reshape(BH_b, 1, 1)
                gt = (torch.cat((gt_epi, gt_h[:, wood.H_epi :]), dim=1) if wood.H_prose > 0 else gt_epi).reshape(BH_b, 1, 1)
                if wood.use_deltanet2_gates and et_h is not None and wt_h is not None and at_h is not None:
                    rw_d = r_write_st.unsqueeze(-1)
                    et_epi = (et_h[:, : wood.H_epi].unsqueeze(2) * rw_d).reshape(1, H_eb, wood.D)
                    wt_epi = (wt_h[:, : wood.H_epi].unsqueeze(2) * rw_d).reshape(1, H_eb, wood.D)
                    at_epi = (1.0 - rw_d * (1.0 - at_h[:, : wood.H_epi].unsqueeze(2))).reshape(1, H_eb, wood.D)
                    et = (torch.cat((et_epi, et_h[:, wood.H_epi :]), dim=1) if wood.H_prose > 0 else et_epi).reshape(BH_b, wood.D, 1)
                    wt = (torch.cat((wt_epi, wt_h[:, wood.H_epi :]), dim=1) if wood.H_prose > 0 else wt_epi).reshape(BH_b, wood.D, 1)
                    at = (torch.cat((at_epi, at_h[:, wood.H_epi :]), dim=1) if wood.H_prose > 0 else at_epi).reshape(BH_b, 1, wood.D)
                else:
                    et = wt = at = None
            else:
                kt = kt_h.reshape(BH_b, wood.D, 1)
                ket = ket_h.reshape(BH_b, wood.D, 1)
                vt = vt_h.reshape(BH_b, wood.D, 1)
                bt = bt_h.reshape(BH_b, 1, 1)
                gt = gt_h.reshape(BH_b, 1, 1)
                if wood.use_deltanet2_gates and et_h is not None and wt_h is not None and at_h is not None:
                    et = et_h.reshape(BH_b, wood.D, 1)
                    wt = wt_h.reshape(BH_b, wood.D, 1)
                    at = at_h.reshape(BH_b, 1, wood.D)
                else:
                    et = wt = at = None

            t_gate = wood._tern_gate()
            inv_dh = t_gate / 144.0
            if wood.use_kalman_precond:
                pk_w = torch.bmm(st.P_mat, kt)
                quad_w = (kt * pk_w).sum(dim=1, keepdim=True)
                norm_pk = torch.sqrt(pk_w.pow(2).sum(dim=1, keepdim=True) + 1e-8)
                kdt = pk_w / (torch.maximum(quad_w, norm_pk * 0.4) + 1e-5)
            else:
                kdt = kt

            if wood.use_deltanet2_gates and et is not None and wt is not None and at is not None:
                M_dec = st.M_mat * at
                pred = torch.bmm(M_dec, kt)
                wv = wt * vt
                ek = et * kdt
                if inv_dh > 0.0:
                    r_sq = (wv - pred).pow(2).sum(dim=1, keepdim=True) * inv_dh
                    sc = torch.rsqrt(torch.sqrt(1.0 + r_sq * r_sq))
                    wv = wv * sc
                    pred = pred * sc
                st.M_mat.copy_(
                    M_dec + 1.35 * (torch.bmm(wv, kdt.transpose(1, 2)) - torch.bmm(pred, ek.transpose(1, 2)))
                )
            else:
                err = vt - gt * torch.bmm(st.M_mat, kt)
                if inv_dh > 0.0:
                    r_sq = err.pow(2).sum(dim=1, keepdim=True) * inv_dh
                    err = err * torch.rsqrt(torch.sqrt(1.0 + r_sq * r_sq))
                st.M_mat.copy_(gt * st.M_mat + (1.35 * bt) * torch.bmm(err, kdt.transpose(1, 2)))

            slope_s = 18.0 + 2.0 * t_gate
            s_gate = torch.sigmoid((bt - 0.26) * slope_s)
            w_cov = s_gate * s_gate * s_gate
            pk = torch.bmm(st.P_mat, ket)
            quad = (ket * pk).sum(dim=1, keepdim=True).clamp(min=1e-5)
            nov = w_cov * 0.92
            P_next = st.P_mat - (nov / (1.0 + nov * quad)) * torch.bmm(pk, pk.transpose(1, 2))
            P_mid = 0.5 * (P_next + P_next.transpose(1, 2))
            if t_gate > 0.0:
                rho_rec = ((1.0 - gt) * w_cov * 0.002 * t_gate).clamp(0.0, 0.002 * t_gate)
                P_new = (1.0 - rho_rec) * P_mid
                P_new.diagonal(dim1=-2, dim2=-1).add_(rho_rec.squeeze(-1) * wood.inv_lam)
                st.P_mat.copy_(0.5 * (P_new + P_new.transpose(1, 2)))
            else:
                st.P_mat.copy_(P_mid)

        u_epi = u_t[:, : wood.H_epi].float()
        conf_gate_epi = torch.sigmoid(
            wood.conf_slope.float().unsqueeze(0) * (wood.conf_thresh.float().unsqueeze(0) - u_epi)
        )
        if wood.H_prose > 0 and wood.query_intent_proj is not None:
            q_intent = torch.sigmoid(wood.query_intent_proj(x_t).float())
            prose_gate = (1.0 - 0.88 * q_intent).expand(1, wood.H_prose)
            all_gates = torch.cat((conf_gate_epi, prose_gate), dim=-1).unsqueeze(-1)
        else:
            q_intent = None
            all_gates = conf_gate_epi.unsqueeze(-1)

        normed_read = (
            wood.out_norm(read_t).view(1, wood.H, wood.D) * all_gates.to(dt)
        ).reshape(1, wood.total_dim)
        raw_u_feats = torch.cat(
            (u_epi - 0.42, torch.tanh((u_epi - 0.42) * 7.0), F.relu(u_epi - 0.36) * 2.5), dim=-1
        )
        u_feats = raw_u_feats.to(dt) * wood.uncert_scale.to(dt)
        if q_intent is not None:
            u_feats = q_intent.to(dt) * u_feats
        return wood.out_proj(torch.cat((normed_read, u_feats), dim=-1)), u_t

    @staticmethod
    def _graph_crystal_step(
        cr: CalciteV2CrystalLayer,
        x_t: torch.Tensor,
        h_lex_t: torch.Tensor,
        st: CrystalStreamState,
        sc_t: torch.Tensor,
        buf_idx: int,
    ) -> torch.Tensor:
        D = x_t.size(1)
        dt = x_t.dtype
        BH = cr.H
        scale_r = math.sqrt(float(cr.d_s))
        sc_d = math.sqrt(float(D))

        x_u = cr._safe_unit(x_t)
        w_lex = torch.sigmoid(cr.lex_logit.float())
        z = cr._safe_unit(w_lex * cr._safe_unit(h_lex_t) + (1.0 - w_lex) * x_u)

        sc_t.add_(1.0)
        st.z_cumsum.add_(z)
        zc = cr._safe_unit(z - st.z_cumsum / (sc_t.view(1, 1) + 2.0), sc_d).to(dt)
        u_t = cr._safe_unit(cr.slot_proj(zc).view(1, cr.H, cr.d_slot))
        sal_t = cr.subj_sal_proj(zc).float()

        sal_w_win = torch.cat((st.sal_w_ring, sal_t.unsqueeze(1)), dim=1).transpose(1, 2)
        u_w_win = torch.cat((st.u_w_ring, u_t.unsqueeze(1)), dim=1).permute(0, 2, 3, 1)
        w_lag_att = F.softmax(sal_w_win + cr.write_lag_logits.float().unsqueeze(0), dim=-1)
        kw_vec_t = cr._safe_unit(
            torch.matmul(u_w_win, w_lag_att.unsqueeze(-1)).squeeze(-1)
            + 0.30 * cr._safe_unit(cr.w_ctx_proj(x_t).view(1, cr.H, cr.d_slot))
        )
        _shift_ring_append_graph_safe(st.sal_w_ring, sal_t)
        _shift_ring_append_graph_safe(st.u_w_ring, u_t)

        k_slots_h = st.cached_k_slots_h
        assert k_slots_h is not None
        sal_r_win = torch.cat((st.sal_r_ring, sal_t.unsqueeze(1)), dim=1)
        u_r_win = torch.cat((st.u_r_ring, u_t.unsqueeze(1)), dim=1)
        _shift_ring_append_graph_safe(st.sal_r_ring, sal_t)
        _shift_ring_append_graph_safe(st.u_r_ring, u_t)

        r_lag_att = F.softmax(
            sal_r_win.transpose(1, 2).unsqueeze(2) + cr.read_lag_logits.float().view(1, cr.H, cr.R_read, 4),
            dim=-1,
        )
        r_u_agg = torch.matmul(r_lag_att, u_r_win.permute(0, 2, 1, 3)).reshape(1, cr.H_r, cr.d_slot)
        qr_vec = cr._safe_unit(
            r_u_agg + 0.30 * cr._safe_unit(cr.r_ctx_proj(x_t).view(1, cr.H_r, cr.d_slot))
        )
        q_logits = (
            torch.matmul(qr_vec.view(1, cr.H, cr.R_read, cr.d_slot), k_slots_h.transpose(-1, -2))
            * float(cr.slot_temp)
        )
        q_soft = F.softmax(q_logits, dim=-1).to(dt)
        s_h = st.s_bh.view(1, cr.H, cr.G_h, cr.d_s).to(dt)
        ret = torch.matmul(q_soft, s_h).reshape(1, cr.H_r * cr.d_s)
        out = cr.out_proj(cr.out_norm(ret))

        st.buf_x[:, buf_idx].copy_(x_t)
        st.buf_zc[:, buf_idx].copy_(zc)
        st.buf_u[:, buf_idx].copy_(u_t)
        st.buf_kw[:, buf_idx].copy_(kw_vec_t)

        if buf_idx + 1 == cr.C:
            if cr.W_chunks >= 2:
                tmp_zc = st.zc_win[:, cr.C :].clone()
                tmp_u = st.u_win[:, cr.C :].clone()
                tmp_v = st.win_valid[:, cr.C :].clone()
                st.zc_win[:, : -cr.C].copy_(tmp_zc)
                st.u_win[:, : -cr.C].copy_(tmp_u)
                st.win_valid[:, : -cr.C].copy_(tmp_v)
            st.zc_win[:, -cr.C :].copy_(st.buf_zc)
            st.u_win[:, -cr.C :].copy_(st.buf_u)
            st.win_valid[:, -cr.C :].fill_(True)

            s_temp = float(cr.slot_temp)
            o_temp = float(cr.op_temp)
            g_tau = float(cr.gate_temp)
            eff_alpha = 1.0 if (not cr.training and cr.snap_alpha > 0.0) else (0.85 * float(cr.snap_alpha))
            is_hard = eff_alpha >= 0.9999
            cb = st.cached_cb
            c_norm_t = st.cached_c_norm_t
            B_mats = st.cached_B_mats
            assert cb is not None and c_norm_t is not None and B_mats is not None

            w_logit_tok = cr.w_eta_proj(st.buf_x).float()
            w_pool = F.softmax(w_logit_tok * 2.0, dim=1)
            w_logit = (w_logit_tok * w_pool).sum(dim=1)
            kw_vec = cr._safe_unit((st.buf_kw * w_pool.unsqueeze(-1)).sum(dim=1))

            v_tok = cr.w_val_proj(st.buf_x).float().view(1, cr.C, cr.H, cr.d_s)
            v_pool = F.softmax(cr.pool_score(st.buf_x).float(), dim=1)
            v_unit = cr._safe_unit((v_tok * v_pool.unsqueeze(-1)).sum(dim=1))
            v_logits = torch.matmul(v_unit, c_norm_t) * 9.0
            v_probs = F.softmax(v_logits, dim=-1)
            v_soft = _sphere_norm_jit(torch.matmul(v_probs, cb), scale_r)
            v_hard = cb[v_logits.argmax(dim=-1)]
            v_ste = v_hard if is_hard else (
                _sphere_norm_jit((1.0 - eff_alpha) * v_soft + eff_alpha * v_hard, scale_r)
                if eff_alpha > 0.0 else v_soft
            )

            ptr_raw = cr.op_ptr_proj(st.zc_win).float().view(
                1, cr.W_chunks * cr.C, cr.H, 2
            ) + cr.ptr_pos_bias.float().unsqueeze(0)
            ptr_raw = ptr_raw.masked_fill(~st.win_valid.unsqueeze(-1).unsqueeze(-1), -1e4)
            att_a = F.softmax(ptr_raw[..., 0], dim=1)
            att_b = F.softmax(ptr_raw[..., 1] - 4.5 * att_a, dim=1)
            ua_vec = cr._safe_unit((st.u_win * att_a.unsqueeze(-1)).sum(dim=1))
            ub_vec = cr._safe_unit((st.u_win * att_b.unsqueeze(-1)).sum(dim=1))

            all_v = torch.stack((kw_vec, ua_vec, ub_vec), dim=2)
            all_lg = torch.matmul(all_v, k_slots_h.transpose(-1, -2)) * s_temp
            k_logits, psi_a, psi_b = all_lg[:, :, 0], all_lg[:, :, 1], all_lg[:, :, 2]

            k_prob = F.softmax(k_logits, dim=-1)
            w_peak = k_prob / k_prob.max(dim=-1, keepdim=True).values.clamp(min=1e-5)
            w_eta = torch.sigmoid((w_logit + 0.58) * g_tau - 0.58)
            kw_soft = (w_eta.unsqueeze(-1) * w_peak).unsqueeze(-1)
            k_hard = F.one_hot(k_prob.argmax(dim=-1), num_classes=cr.G_h).float()
            w_hard_g = (w_eta >= 0.36).float()
            kw_targ = (w_hard_g.unsqueeze(-1) * k_hard).unsqueeze(-1)
            kw_eff = kw_targ if is_hard else ((1.0 - eff_alpha) * kw_soft + eff_alpha * kw_targ)

            ws = 1.0 if ((not cr.training and cr.snap_alpha > 0.0) or cr.qat_active_mix > 0.0) else 0.0
            _, _, T_swap = cr._compute_ptr_swap(psi_a.unsqueeze(1), psi_b.unsqueeze(1), well_scale=ws)
            T_swap = T_swap.squeeze(1)

            x_ch_mean = st.buf_x.float().mean(dim=1).to(dt)
            z_op_in = torch.cat((st.z_mean_ring, x_ch_mean.unsqueeze(1)), dim=1).reshape(1, 4 * cr.d_model)
            _shift_ring_append_graph_safe(st.z_mean_ring, x_ch_mean)
            z_op = F.silu(cr.op_ctx_proj(z_op_in))

            m_prob = F.softmax(cr.op_mode_proj(z_op).float().view(1, cr.H, 1 + cr.M_glob) * o_temp, dim=-1)
            B_flat = B_mats.reshape(cr.H, cr.M_glob, cr.G_h * cr.G_h)
            glob_mix = torch.matmul(m_prob[..., 1:].unsqueeze(-2), B_flat).view(1, cr.H, cr.G_h, cr.G_h)
            P_cand = m_prob[..., 0:1].unsqueeze(-1) * T_swap + glob_mix

            p_logit_raw = cr.perm_gate_proj(z_op).float()
            p_logit = -4.5 + F.softplus(p_logit_raw + 4.5)
            p_raw = torch.sigmoid((p_logit + 0.49) * g_tau - 0.49)
            p_prev = torch.maximum(st.p_f_m1, st.p_f_m2)
            p_eta = p_raw * (1.0 - 0.92 * torch.sigmoid((p_prev - 0.35) * 12.0))
            st.p_f_m2.copy_(st.p_f_m1)
            st.p_f_m1.copy_(p_eta)

            eye = cr.eye_Gh.float().view(1, 1, cr.G_h, cr.G_h)
            P_soft = eye + p_eta.unsqueeze(-1).unsqueeze(-1) * (P_cand - eye)
            if eff_alpha > 0.0:
                P_perm_hard = project_birkhoff_to_perm_jit(P_cand.reshape(-1, cr.G_h, cr.G_h)).view(
                    1, cr.H, cr.G_h, cr.G_h
                )
                p_hard_g = (p_eta >= 0.38).float()
                P_targ = eye + p_hard_g.unsqueeze(-1).unsqueeze(-1) * (P_perm_hard - eye)
                P_eff = P_targ if is_hard else ((1.0 - eff_alpha) * P_soft + eff_alpha * P_targ)
            else:
                P_eff = P_soft

            kw_bh = kw_eff.reshape(BH, cr.G_h, 1)
            vs_bh = v_ste.reshape(BH, 1, cr.d_s)
            Pe_bh = P_eff.reshape(BH, cr.G_h, cr.G_h)
            s_w = (1.0 - kw_bh) * st.s_bh + kw_bh * vs_bh
            s_next = torch.bmm(Pe_bh, s_w)
            st.s_bh.copy_(s_next if is_hard else _sphere_norm_jit(s_next, scale_r))

        return out

    @staticmethod
    def _graph_epistemic_slots_step(
        es: EpistemicSlotMemory,
        x_t: torch.Tensor,
        u_unc_t: torch.Tensor,
        h_lex_t: torch.Tensor,
        st: EpistemicSlotsStreamState,
    ) -> torch.Tensor:
        H, S, d_k, d_v = es.n_heads, es.n_slots, es.d_k, es.d_v
        dt = x_t.dtype
        z_qk = es._blend_key_input(x_t, h_lex_t)
        q_t = es._safe_unit(es.q_proj(z_qk).view(1, H, d_k))
        k_raw_t = es._safe_unit(es.k_proj(z_qk).view(1, H, d_k))
        v_t = es.v_proj(x_t).float().view(1, H, d_v) + es.v_lex_proj(h_lex_t).float().view(1, H, d_v)

        k_b = st.kvf_bank[..., :d_k]
        v_b = st.kvf_bank[..., d_k : d_k + d_v]
        f_b = st.kvf_bank[..., -1].clamp(0.0, 1.0)
        k_hat = k_b * torch.rsqrt(k_b.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
        r_sc = es.read_scale.float().clamp(2.0, 32.0).view(1, H, 1)
        sim = torch.matmul(k_hat, q_t.unsqueeze(-1)).squeeze(-1) * r_sc + (1.0 - f_b) * (-20.0)
        null_col = es.null_logit.float().view(1, H, 1)
        probs = F.softmax(torch.cat((sim, null_col), dim=-1), dim=-1)[..., :S]
        out_t = torch.matmul(probs.unsqueeze(-2), v_b).squeeze(-2).reshape(1, H * d_v).to(dt)

        lw = F.softmax(es.lag_logits.float(), dim=-1)
        k_lag_t = es._safe_unit(
            lw[:, 0].view(1, H, 1) * st.k_ring[:, 2]
            + lw[:, 1].view(1, H, 1) * st.k_ring[:, 1]
            + lw[:, 2].view(1, H, 1) * st.k_ring[:, 0]
        )
        beta_t, _ = es._compute_beta(x_t, u_unc_t)
        anchors = es._get_unit_anchors()
        r_logits = torch.matmul(k_lag_t.unsqueeze(-2), anchors.transpose(-1, -2)).squeeze(-2) * 6.0
        top_idx = r_logits.argmax(dim=-1)  # (1, H)
        ones_u = torch.ones((1, H, 1), device=x_t.device, dtype=torch.float32)
        u_new = torch.cat((k_lag_t, v_t, ones_u), dim=-1)  # (1, H, d_u)
        idx_e = top_idx.unsqueeze(-1).unsqueeze(-1).expand(1, H, 1, es.d_u)
        old_u = st.kvf_bank.gather(2, idx_e)
        bt = beta_t.unsqueeze(-1).unsqueeze(-1)
        upd_u = (1.0 - bt) * old_u + bt * u_new.unsqueeze(2)
        st.kvf_bank.scatter_(2, idx_e, upd_u)
        _shift_ring_append_graph_safe(st.k_ring, k_raw_t)
        return es.out_proj(es.out_norm(out_t))

    def _graph_backbone_body(
        self,
        state: CalciteStreamState,
        b: _CudaGraphBundle,
        phase: Tuple[int, int, int, int],
    ) -> None:
        m = self.model
        c1_p, c2_p, wf_p, ws_p = phase
        h = b.h_byte_out
        h_lex_t = m.byte_chunker.compute_lex_from_win8(b.win8_out)
        ring_idx = 0

        if self.cfg.topology == "interleaved" and m.swa_stages is not None:
            use_dca = m.dca_mixers is not None
            hist: List[torch.Tensor] = [h] if use_dca else []
            dca_idx = 0

            def _dca_mix_il_cg(h_cur: torch.Tensor) -> torch.Tensor:
                nonlocal dca_idx
                if use_dca and m.dca_mixers is not None and len(hist) >= 2:
                    h_cur = m.dca_mixers[dca_idx](h_cur, hist)
                    dca_idx += 1
                return h_cur

            for blk in m.swa_stages[0]:
                h = _dca_mix_il_cg(h)
                h = self._graph_swa_step(blk, h, state.swa_k_rings[ring_idx], state.swa_v_rings[ring_idx], b)
                ring_idx += 1
                if use_dca:
                    hist.append(h)
            h = _dca_mix_il_cg(h)
            if (
                m.woodbury_fast is not None
                and m.woodbury_fast_norm is not None
                and m.woodbury_fast_gate is not None
                and state.woodbury_fast is not None
                and b.wf_sc is not None
            ):
                wf_in = m.woodbury_fast_norm(h)
                wf_g = 2.0 * torch.sigmoid(m.woodbury_fast_gate(wf_in))
                wf_o, _ = self._graph_woodbury_step(m.woodbury_fast, wf_in, h_lex_t, state.woodbury_fast, b.wf_sc, wf_p)
                h = h + wf_g * wf_o
            if use_dca:
                hist.append(h)

            for blk in m.swa_stages[1]:
                h = _dca_mix_il_cg(h)
                h = self._graph_swa_step(blk, h, state.swa_k_rings[ring_idx], state.swa_v_rings[ring_idx], b)
                ring_idx += 1
                if use_dca:
                    hist.append(h)
            h = _dca_mix_il_cg(h)
            if m.cryst1_norm is not None and m.cryst1_gate is not None:
                c1_in = m.cryst1_norm(h)
                c1_g = 2.0 * torch.sigmoid(m.cryst1_gate(c1_in))
                h = h + c1_g * self._graph_crystal_step(m.crystal, c1_in, h_lex_t, state.crystal_1, b.c1_sc, c1_p)
            if use_dca:
                hist.append(h)

            for blk in m.swa_stages[2]:
                h = _dca_mix_il_cg(h)
                h = self._graph_swa_step(blk, h, state.swa_k_rings[ring_idx], state.swa_v_rings[ring_idx], b)
                ring_idx += 1
                if use_dca:
                    hist.append(h)
            h = _dca_mix_il_cg(h)
            if m.woodbury_slow_gate is not None:
                ws_in = m.split_norm(h)
                ws_g = 2.0 * torch.sigmoid(m.woodbury_slow_gate(ws_in))
                ws_o, u_ws = self._graph_woodbury_step(m.woodbury, ws_in, h_lex_t, state.woodbury_slow, b.ws_sc, ws_p)
                h = h + ws_g * ws_o
                if m.epistemic_slots is not None and m.epistemic_slots_gate is not None and state.epistemic_slots is not None:
                    es_g = 2.0 * torch.sigmoid(m.epistemic_slots_gate(ws_in))
                    h = h + es_g * self._graph_epistemic_slots_step(
                        m.epistemic_slots, ws_in, u_ws, h_lex_t, state.epistemic_slots
                    )
            if use_dca:
                hist.append(h)
            h = _dca_mix_il_cg(h)
            h = h + m.mid_mlp(m.mid_mlp_norm(h))
            if use_dca:
                hist.append(h)

            h = _dca_mix_il_cg(h)
            c2_in = m.cryst2_norm(h)
            c2_g = 2.0 * torch.sigmoid(m.cryst2_gate(c2_in))
            h = h + c2_g * self._graph_crystal_step(m.crystal_2, c2_in, h_lex_t, state.crystal_2, b.c2_sc, c2_p)
            if use_dca:
                hist.append(h)

            for blk in m.swa_stages[3]:
                h = _dca_mix_il_cg(h)
                h = self._graph_swa_step(blk, h, state.swa_k_rings[ring_idx], state.swa_v_rings[ring_idx], b)
                ring_idx += 1
                if use_dca:
                    hist.append(h)
            h = _dca_mix_il_cg(h)
        else:
            use_ar = m.attnres_mixers is not None
            use_dca = m.dca_mixers is not None
            hist = [h] if (use_ar or use_dca) else []
            dca_idx = 0

            def _dca_mix_hg_cg(h_cur: torch.Tensor) -> torch.Tensor:
                nonlocal dca_idx
                if use_dca and m.dca_mixers is not None and len(hist) >= 2:
                    h_cur = m.dca_mixers[dca_idx](h_cur, hist)
                    dca_idx += 1
                return h_cur

            for blk in m.cortex_pre:
                if use_dca:
                    h = _dca_mix_hg_cg(h)
                h = self._graph_swa_step(blk, h, state.swa_k_rings[ring_idx], state.swa_v_rings[ring_idx], b)
                ring_idx += 1
                if use_ar or use_dca:
                    hist.append(h)

            if use_dca:
                h = _dca_mix_hg_cg(h)
                h_long_in = h
            elif use_ar and m.attnres_mixers is not None:
                h_long_in = m.attnres_mixers[0](h, hist)
            else:
                h_long_in = h
            h_split = m.split_norm(h_long_in)
            c_out = self._graph_crystal_step(m.crystal, h_split, h_lex_t, state.crystal_1, b.c1_sc, c1_p)
            w_out, u_ws = self._graph_woodbury_step(m.woodbury, h_split, h_lex_t, state.woodbury_slow, b.ws_sc, ws_p)
            if m.stream_gate is not None:
                gates = 2.0 * torch.sigmoid(m.stream_gate(h_split))
                delta_l = gates[..., 0:1] * c_out + gates[..., 1:2] * w_out
            else:
                delta_l = c_out + w_out
            if m.epistemic_slots is not None and m.epistemic_slots_gate is not None and state.epistemic_slots is not None:
                es_g = 2.0 * torch.sigmoid(m.epistemic_slots_gate(h_split))
                delta_l = delta_l + es_g * self._graph_epistemic_slots_step(
                    m.epistemic_slots, h_split, u_ws, h_lex_t, state.epistemic_slots
                )
            h = h + delta_l
            if use_ar or use_dca:
                hist.append(h)

            if use_dca:
                h = _dca_mix_hg_cg(h)
                h_mid_in = h
            elif use_ar and m.attnres_mixers is not None:
                h_mid_in = m.attnres_mixers[1](h, hist)
            else:
                h_mid_in = h
            h = h + m.mid_mlp(m.mid_mlp_norm(h_mid_in))
            if use_ar or use_dca:
                hist.append(h)

            if use_dca:
                h = _dca_mix_hg_cg(h)
                h_c2_in = h
            elif use_ar and m.attnres_mixers is not None:
                h_c2_in = m.attnres_mixers[2](h, hist)
            else:
                h_c2_in = h
            c2_in = m.cryst2_norm(h_c2_in)
            h = h + (2.0 * torch.sigmoid(m.cryst2_gate(c2_in))) * self._graph_crystal_step(
                m.crystal_2, c2_in, h_lex_t, state.crystal_2, b.c2_sc, c2_p
            )
            if use_ar or use_dca:
                hist.append(h)

            for pi, blk in enumerate(m.cortex_post):
                if use_dca:
                    h = _dca_mix_hg_cg(h)
                elif use_ar and m.attnres_mixers is not None:
                    h = m.attnres_mixers[3 + pi](h, hist)
                h = self._graph_swa_step(blk, h, state.swa_k_rings[ring_idx], state.swa_v_rings[ring_idx], b)
                ring_idx += 1
                if use_ar or use_dca:
                    hist.append(h)

            if use_dca:
                h = _dca_mix_hg_cg(h)
            elif use_ar and m.attnres_mixers is not None:
                h = m.attnres_mixers[-1](h, hist)

        b.z_hat_out.copy_(h)

    def _init_cuda_graph_bundle(self, state: CalciteStreamState, dt: torch.dtype, dev: torch.device) -> _CudaGraphBundle:
        self._canonicalize_all_rings(state)
        ch = self.model.byte_chunker
        max_b = max(65536, state.total_bytes + 16384)
        max_c = max(16384, state.chunk_pos + 4096)
        Dh_c = self.cfg.d_model // self.cfg.n_heads
        chunk_cos, chunk_sin = get_cached_rope_cos_sin(max_c, dev, dt, head_dim=Dh_c, base=10000.0)
        if ch.use_sym_byte_swa:
            byte_cos, byte_sin = get_cached_rope_cos_sin(
                max_b, dev, dt, head_dim=ch.byte_swa_head_dim, base=10000.0
            )
        else:
            byte_cos = byte_sin = None

        # Pre-warm Crystal cached constants before capture
        for cr, cst in ((self.model.crystal, state.crystal_1), (self.model.crystal_2, state.crystal_2)):
            if cst.cached_cb is None or cst.cached_cb.device != dev:
                cb, cn = cr.get_codebook_fp32()
                cst.cached_cb, cst.cached_c_norm_t = cb.to(dev), cn.to(dev)
            if cst.cached_k_slots_h is None or cst.cached_k_slots_h.device != dev:
                cst.cached_k_slots_h = cr._safe_unit(cr.K_slots).to(dev)
            o_temp = float(cr.op_temp)
            if cst.cached_B_mats is None or cst.cached_o_temp != o_temp or cst.cached_B_mats.device != dev:
                cst.cached_B_mats = log_sinkhorn_6(cr.birkhoff_logits.float().to(dev) * o_temp)
                cst.cached_o_temp = o_temp

        bundle = _CudaGraphBundle(
            state_id=id(state),
            device=dev,
            dtype=dt,
            max_byte_pos=max_b,
            max_chunk_pos=max_c,
            byte_cos=byte_cos,
            byte_sin=byte_sin,
            chunk_cos=chunk_cos,
            chunk_sin=chunk_sin,
            tok_in=torch.zeros((1,), device=dev, dtype=torch.long),
            pos_b=torch.tensor([state.total_bytes], device=dev, dtype=torch.long),
            pos_c=torch.tensor([state.chunk_pos], device=dev, dtype=torch.long),
            c1_sc=torch.tensor([float(state.crystal_1.step_count)], device=dev, dtype=torch.float32),
            c2_sc=torch.tensor([float(state.crystal_2.step_count)], device=dev, dtype=torch.float32),
            wf_sc=(
                torch.tensor([float(state.woodbury_fast.step_count)], device=dev, dtype=torch.float32)
                if state.woodbury_fast is not None
                else None
            ),
            ws_sc=torch.tensor([float(state.woodbury_slow.step_count)], device=dev, dtype=torch.float32),
            fire_out=torch.zeros((1,), device=dev, dtype=torch.bool),
            p_out=torch.zeros((1,), device=dev, dtype=torch.float32),
            h_byte_out=torch.zeros((1, self.cfg.d_model), device=dev, dtype=dt),
            c_norm_out=torch.zeros((1, 8 * ch.d_byte), device=dev, dtype=dt),
            win8_out=torch.zeros((1, 8 * ch.d_byte), device=dev, dtype=dt),
            offset_out=torch.zeros((1,), device=dev, dtype=torch.long),
            z_hat_out=torch.zeros((1, self.cfg.d_model), device=dev, dtype=dt),
            logits_out=torch.zeros((1, self.cfg.vocab_size), device=dev, dtype=dt),
            g_enc=torch.cuda.CUDAGraph(),
            g_dec_nofire=torch.cuda.CUDAGraph(),
            g_dec_fire=torch.cuda.CUDAGraph(),
            g_bb_phases={},
        )

        # Warm `_tern_gate` & LUT caches once outside capture
        self.model.woodbury._tern_gate()
        if self.model.woodbury_fast is not None:
            self.model.woodbury_fast._tern_gate()
        if self.model.epistemic_slots is not None:
            self.model.epistemic_slots._get_unit_anchors()

        preserve = self._collect_state_tensors(state) + [
            bundle.c1_sc, bundle.c2_sc, bundle.ws_sc,
            bundle.fire_out, bundle.p_out, bundle.h_byte_out,
            bundle.c_norm_out, bundle.win8_out, bundle.offset_out,
            bundle.z_hat_out, bundle.logits_out,
        ]
        if bundle.wf_sc is not None:
            preserve.append(bundle.wf_sc)

        bundle.g_enc = self._capture_graph_preserving_state(
            lambda: self._graph_encode_body(state, bundle), preserve
        )
        bundle.g_dec_nofire = self._capture_graph_preserving_state(
            lambda: self._graph_decode_body(state, bundle, update_chunk=False), preserve
        )
        bundle.g_dec_fire = self._capture_graph_preserving_state(
            lambda: self._graph_decode_body(state, bundle, update_chunk=True), preserve
        )
        self._cg_bundle = bundle
        return bundle

    @torch.no_grad()
    def _step_cuda_graph(
        self,
        token_t: torch.Tensor,
        state: CalciteStreamState,
    ) -> Tuple[torch.Tensor, CalciteStreamState]:
        dev = token_t.device
        dt = state.chunker.cur_z_bar.dtype
        bundle = self._cg_bundle
        if bundle is None or bundle.state_id != id(state) or bundle.device != dev or bundle.dtype != dt:
            bundle = self._init_cuda_graph_bundle(state, dt, dev)

        bundle.tok_in.copy_(token_t)
        bundle.pos_b.fill_(state.total_bytes)
        bundle.g_enc.replay()
        state.chunker.step_idx += 1
        if self.model.byte_chunker.use_byte_induct_cache:
            state.chunker.last_tok = token_t

        fired = bool(bundle.fire_out.item())
        if fired:
            wf_p = state.woodbury_fast.buf_pos if state.woodbury_fast is not None else 0
            phase = (state.crystal_1.buf_pos, state.crystal_2.buf_pos, wf_p, state.woodbury_slow.buf_pos)
            g_bb = bundle.g_bb_phases.get(phase)
            if g_bb is None:
                preserve = self._collect_state_tensors(state) + [
                    bundle.c1_sc, bundle.c2_sc, bundle.ws_sc, bundle.z_hat_out,
                ]
                if bundle.wf_sc is not None:
                    preserve.append(bundle.wf_sc)
                g_bb = self._capture_graph_preserving_state(
                    lambda: self._graph_backbone_body(state, bundle, phase), preserve
                )
                bundle.g_bb_phases[phase] = g_bb
            bundle.pos_c.fill_(state.chunk_pos)
            g_bb.replay()

            # Sync Python integer phase counters to match exact in-place GPU state
            state.chunk_pos += 1
            state.backbone_ticks += 1
            state.crystal_1.step_count += 1
            state.crystal_1.buf_pos = (state.crystal_1.buf_pos + 1) % self.model.crystal.C
            state.crystal_2.step_count += 1
            state.crystal_2.buf_pos = (state.crystal_2.buf_pos + 1) % self.model.crystal_2.C
            state.woodbury_slow.step_count += 1
            state.woodbury_slow.buf_pos = (state.woodbury_slow.buf_pos + 1) % self.model.woodbury.C
            if state.woodbury_fast is not None and self.model.woodbury_fast is not None:
                state.woodbury_fast.step_count += 1
                state.woodbury_fast.buf_pos = (state.woodbury_fast.buf_pos + 1) % self.model.woodbury_fast.C
            if state.epistemic_slots is not None:
                state.epistemic_slots.step_count += 1

            bundle.g_dec_fire.replay()
        else:
            bundle.g_dec_nofire.replay()

        if state.chunker.byte_induct is not None:
            state.chunker.byte_induct.step_count += 1
        state.total_bytes += 1
        return bundle.logits_out.clone(), state

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_bytes: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = 2,
    ) -> Tuple[torch.Tensor, CalciteStreamState]:
        """Autoregressive byte generation with strictly constant `O(1)` RAM."""
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        logits, state = self.prefill(prompt_ids, return_all_logits=False)
        gen: List[torch.Tensor] = [prompt_ids]
        for _ in range(max_new_bytes):
            if temperature <= 1e-5:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits.float() / float(temperature), dim=-1)
                if 0.0 < top_p < 1.0:
                    sorted_p, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                    cdf = torch.cumsum(sorted_p, dim=-1)
                    mask = (cdf - sorted_p) > top_p
                    sorted_p = sorted_p.masked_fill(mask, 0.0)
                    sorted_p = sorted_p / sorted_p.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                    pick = torch.multinomial(sorted_p, num_samples=1)
                    nxt = sorted_idx.gather(-1, pick)
                else:
                    nxt = torch.multinomial(probs, num_samples=1)
            gen.append(nxt)
            if eos_token_id is not None and bool((nxt == eos_token_id).all().item()):
                break
            logits, state = self.step(nxt.squeeze(-1), state)
        return torch.cat(gen, dim=1), state
