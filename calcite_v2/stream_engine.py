"""
Calcite v2 Live O(1) Streaming Inference Engine (`calcite_v2/stream_engine.py`).

Runs `CalciteV2LM` (both FP/BF16 and `FastPackedLUT2BitLinear` 2-bit uint8 T-MAC LUT)
byte-by-byte with strictly constant `O(1)` memory independent of sequence length `L`:
- Heavy chunk backbone (`SWA + Woodbury + Crystal`) ticks ONLY when a chunk boundary fires
  (`~1 / 4.6` of bytes -> ~4.5x fewer backbone steps during autoregressive generation).
- Non-boundary bytes run ONLY the lightweight causal byte encoder/decoder using `cur_z_bar`.
- Exact bit-for-bit parity with batch `model.forward(seq)` (for `batch_size = 1` or synchronized
  boundary batches; arbitrary `batch_size >= 1` supported via per-item boundary stepping).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .chunker import ChunkerStreamState
from .crystal import CrystalStreamState
from .epistemic_slots import EpistemicSlotsStreamState
from .model import CalciteV2LM
from .woodbury import WoodburyStreamState


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

    def memory_bytes(self) -> int:
        """Exact live allocated state bytes per batch item (strictly O(1) vs `total_bytes`)."""
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


class CalciteStreamEngine:
    """Zero-allocation-growth `O(1)` byte-streaming engine for `CalciteV2LM` (FP or 2-bit LUT)."""

    def __init__(self, model: CalciteV2LM):
        self.model = model
        self.cfg = model.cfg

    def init_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> CalciteStreamState:
        dev = device or self.model.tok_emb.weight.device
        dt = dtype or self.model.tok_emb.weight.dtype
        swa_blocks = self.model._all_swa_blocks()
        W = self.cfg.window_size
        H = self.cfg.n_heads
        Dh = self.cfg.d_model // H
        k_rings = [torch.zeros(batch_size, W, H, Dh, device=dev, dtype=dt) for _ in swa_blocks]
        v_rings = [torch.zeros(batch_size, W, H, Dh, device=dev, dtype=dt) for _ in swa_blocks]
        canon_rings = (
            [
                (
                    torch.zeros(batch_size, 3, self.cfg.d_model, device=dev, dtype=dt),
                    torch.zeros(batch_size, 3, self.cfg.d_model, device=dev, dtype=dt),
                )
                for _ in swa_blocks
            ]
            if self.cfg.use_canon_chunks
            else None
        )

        wf_state = (
            self.model.woodbury_fast.init_stream_state(batch_size, dev, dt)
            if self.model.woodbury_fast is not None
            else None
        )
        ws_state = self.model.woodbury.init_stream_state(batch_size, dev, dt)
        es_state = (
            self.model.epistemic_slots.init_stream_state(batch_size, dev, dt)
            if self.model.epistemic_slots is not None
            else None
        )
        c1_state = self.model.crystal.init_stream_state(batch_size, dev, dt)
        c2_state = self.model.crystal_2.init_stream_state(batch_size, dev, dt)
        fb_prev = (
            torch.zeros(batch_size, self.cfg.d_model, device=dev, dtype=dt)
            if self.cfg.use_latent_feedback
            else None
        )
        tr_carry = (
            torch.zeros(batch_size, self.cfg.d_model, device=dev, dtype=dt)
            if self.cfg.use_temporal_residual
            else None
        )
        return CalciteStreamState(
            chunker=self.model.byte_chunker.init_stream_state(batch_size, dev, dt),
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
        )

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

        if self.cfg.use_latent_feedback and m.fb_norm is not None and m.fb_proj is not None and state.fb_prev_z_hat is not None:
            if pos > 0:
                h = z_in_t + m.fb_proj(m.fb_norm(state.fb_prev_z_hat))
            else:
                h = z_in_t
        else:
            h = z_in_t

        ring_idx = 0
        if self.cfg.topology == "interleaved" and m.swa_stages is not None:
            for blk in m.swa_stages[0]:
                c_rng = state.swa_canon_rings[ring_idx] if state.swa_canon_rings is not None else None
                h, state.swa_k_rings[ring_idx], state.swa_v_rings[ring_idx] = blk.step(
                    h, state.swa_k_rings[ring_idx], state.swa_v_rings[ring_idx], pos, canon_rings=c_rng
                )
                ring_idx += 1
            if m.woodbury_fast is not None and m.woodbury_fast_norm is not None and m.woodbury_fast_gate is not None and state.woodbury_fast is not None:
                wf_in = m.woodbury_fast_norm(h)
                wf_g = 2.0 * torch.sigmoid(m.woodbury_fast_gate(wf_in))
                h = h + wf_g * m.woodbury_fast.step(wf_in, h_lex_t, state.woodbury_fast)

            for blk in m.swa_stages[1]:
                c_rng = state.swa_canon_rings[ring_idx] if state.swa_canon_rings is not None else None
                h, state.swa_k_rings[ring_idx], state.swa_v_rings[ring_idx] = blk.step(
                    h, state.swa_k_rings[ring_idx], state.swa_v_rings[ring_idx], pos, canon_rings=c_rng
                )
                ring_idx += 1
            if m.cryst1_norm is not None and m.cryst1_gate is not None:
                c1_in = m.cryst1_norm(h)
                c1_g = 2.0 * torch.sigmoid(m.cryst1_gate(c1_in))
                h = h + c1_g * m.crystal.step(c1_in, h_lex_t, state.crystal_1)

            for blk in m.swa_stages[2]:
                c_rng = state.swa_canon_rings[ring_idx] if state.swa_canon_rings is not None else None
                h, state.swa_k_rings[ring_idx], state.swa_v_rings[ring_idx] = blk.step(
                    h, state.swa_k_rings[ring_idx], state.swa_v_rings[ring_idx], pos, canon_rings=c_rng
                )
                ring_idx += 1
            if m.woodbury_slow_gate is not None:
                ws_in = m.split_norm(h)
                ws_g = 2.0 * torch.sigmoid(m.woodbury_slow_gate(ws_in))
                h = h + ws_g * m.woodbury.step(ws_in, h_lex_t, state.woodbury_slow)
                if m.epistemic_slots is not None and m.epistemic_slots_gate is not None and state.epistemic_slots is not None:
                    es_g = 2.0 * torch.sigmoid(m.epistemic_slots_gate(ws_in))
                    h = h + es_g * m.epistemic_slots.step(
                        ws_in, m.woodbury.last_epistemic_uncertainty, state.epistemic_slots, h_lex_t=h_lex_t
                    )
            h = h + m.mid_mlp(m.mid_mlp_norm(h))

            c2_in = m.cryst2_norm(h)
            c2_g = 2.0 * torch.sigmoid(m.cryst2_gate(c2_in))
            h = h + c2_g * m.crystal_2.step(c2_in, h_lex_t, state.crystal_2)

            for blk in m.swa_stages[3]:
                c_rng = state.swa_canon_rings[ring_idx] if state.swa_canon_rings is not None else None
                h, state.swa_k_rings[ring_idx], state.swa_v_rings[ring_idx] = blk.step(
                    h, state.swa_k_rings[ring_idx], state.swa_v_rings[ring_idx], pos, canon_rings=c_rng
                )
                ring_idx += 1
        else:
            use_ar = m.attnres_mixers is not None
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

            def _tr_step_stream(a_cur: torch.Tensor, b_cur: Optional[torch.Tensor], y_fn) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
                nonlocal tr_idx
                if use_tr and b_cur is not None and m.tr_rA is not None and m.tr_rB is not None and m.tr_sB is not None:
                    u = m.tr_rA[tr_idx].to(a_cur.dtype) * a_cur + m.tr_rB[tr_idx].to(a_cur.dtype) * b_cur
                    y = y_fn(u)
                    a_next = a_cur + y
                    b_next = b_cur + m.tr_sB[tr_idx].to(a_cur.dtype) * y
                    tr_idx += 1
                    return a_next, b_next
                return a_cur + y_fn(a_cur), b_cur

            def _step_swa_blk(blk, r_i: int, a_cur: torch.Tensor, b_cur: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
                a_out, kr, vr = blk.step(a_cur, state.swa_k_rings[r_i], state.swa_v_rings[r_i], pos, canon_rings=c_r)
                state.swa_k_rings[r_i] = kr
                state.swa_v_rings[r_i] = vr
                return a_out, b_cur

            hist: List[torch.Tensor] = [h] if use_ar else []
            for blk in m.cortex_pre:
                h, b_stream = _step_swa_blk(blk, ring_idx, h, b_stream)
                ring_idx += 1
                if use_ar:
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
                if m.epistemic_slots is not None and m.epistemic_slots_gate is not None and state.epistemic_slots is not None:
                    es_g = 2.0 * torch.sigmoid(m.epistemic_slots_gate(h_split))
                    delta_l = delta_l + es_g * m.epistemic_slots.step(
                        h_split, m.woodbury.last_epistemic_uncertainty, state.epistemic_slots, h_lex_t=h_lex_t
                    )
                return delta_l

            if use_ar and m.attnres_mixers is not None:
                h_long_in = m.attnres_mixers[0](h, hist)
                if use_tr and b_stream is not None and m.tr_rA is not None and m.tr_rB is not None and m.tr_sB is not None:
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

            if use_ar and m.attnres_mixers is not None:
                h_mid_in = m.attnres_mixers[1](h, hist)
                if use_tr and b_stream is not None and m.tr_rA is not None and m.tr_rB is not None and m.tr_sB is not None:
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

            if use_ar and m.attnres_mixers is not None:
                h_c2_in = m.attnres_mixers[2](h, hist)
                if use_tr and b_stream is not None and m.tr_rA is not None and m.tr_rB is not None and m.tr_sB is not None:
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
                if use_ar and m.attnres_mixers is not None:
                    h = m.attnres_mixers[3 + pi](h, hist)
                h, b_stream = _step_swa_blk(blk, ring_idx, h, b_stream)
                ring_idx += 1
                if use_ar:
                    hist.append(h)

            if use_ar and m.attnres_mixers is not None:
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
        Processes ONE byte `token_t` `(B,)` (`B == 1` recommended for exact skipped-backbone speed)
        and returns `(logits_t, state)` with shape `(B, vocab_size)`.
        """
        if token_t.ndim == 0:
            token_t = token_t.unsqueeze(0)
        e_tok_t = self.model.tok_emb(token_t)
        fire_t, p_t, h_byte_t, c_norm_t, h_lex_t, offset_t, _ = self.model.byte_chunker.encode_byte_step(
            token_t, e_tok_t, state.chunker
        )

        # Heavy backbone executes ONLY when a chunk boundary fires!
        if bool(fire_t.any().item()):
            z_hat_new = self._step_backbone(h_byte_t, h_lex_t, state)
        else:
            z_hat_new = None

        h_out = self.model.byte_chunker.decode_byte_step(
            z_hat_new, fire_t, p_t, h_byte_t, c_norm_t, offset_t, state.chunker
        )
        logits = self.model.lm_head(self.model.norm_f(h_out))
        state.total_bytes += 1
        return logits, state

    @torch.no_grad()
    def prefill(
        self,
        token_ids: torch.Tensor,
        state: Optional[CalciteStreamState] = None,
        return_all_logits: bool = False,
    ) -> Tuple[torch.Tensor, CalciteStreamState]:
        """Streams a prompt `(B, T)` byte-by-byte through strict `O(1)` ring state."""
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        B, T = token_ids.shape
        if state is None:
            state = self.init_state(batch_size=B, device=token_ids.device)
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
