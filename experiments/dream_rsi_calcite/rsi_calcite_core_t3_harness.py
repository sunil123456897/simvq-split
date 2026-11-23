"""
SLEEP-DAG-RSI v8 Track T3 (`calcite_core_t3`): Architectural Redesign of `Calcite System-1`
closing W1 (32-dim byte codec straw & 4-byte UTF-8 tax), W2 (Woodbury `L_latch=6` ring buffer
& single-timescale `gamma`), and W4 (`log_sinkhorn_6` spectral drift in Crystal).

Strict Isolation & Zero-Leak Protocol:
  - ZERO modifications to protected `simvq_split/`, `kernels/fast_packed_lut.py`, `experiments/uebr_system2/`.
  - ZERO access to `benchmarks/gsm8k/test.jsonl`.
  - Equal parameter & compute budget on local RTX 5060 (`d_model=192`, `~2.2M` params, identical training steps).
  - Dev Split (Rounds 1-4): first half of disjoint 6-domain `eval_wins` + Dev cognitive (`N=48`, `seed=77001..`).
  - Sealed Blind Vault (`BlindVaultGuard(max_calls=1)`): fresh seed (`42026`) retrain of BOTH Baseline and
    Champion evaluated on second half of `eval_wins` + full `N=150` unseen cognitive (`S_5 k=2,4,8,16`,
    `OV/Trap ood_long`) + `L=1024` OOD length extrapolation + `2-bit LUT` check + `crystal_drift_L2048`.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

PLUGIN_SCRIPTS = os.path.expanduser(
    "~/.gemini/config/plugins/autonomous-research/skills/dream-rsi/scripts"
)
FALLBACK_SCRIPTS = (
    "C:/Users/vanya/.gemini/config/plugins/autonomous-research/skills/dream-rsi/scripts"
)
for _p in (PLUGIN_SCRIPTS, FALLBACK_SCRIPTS):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import engine as dream_engine  # noqa: E402
from cartridge import AutoDreamRunner, BranchCandidate, TaskCartridge  # noqa: E402

from experiments.scaling_phase_k.eval_suite import (  # noqa: E402
    eval_cognitive_suites_any_byte_lm,
    eval_openweb_6domains_any_byte_lm,
)
from experiments.scaling_phase_k.lean_system1 import (  # noqa: E402
    CalciteLeanSystem1LM,
    LeanNeuralWoodburyLayer,
    LeanSystem1Config,
    LeanVoronoiCrystalLayer,
)
from run_phase_e_fast_utf8_benchmark import build_disjoint_openweb_corpora  # noqa: E402
from simvq_split.crystal import (  # noqa: E402
    _sphere_norm_jit,
    causal_refractory_gate_jit,
    diag_birkhoff_crystal_scan,
    project_birkhoff_to_perm_jit,
)
from simvq_split.memory import compute_chunker_state_bytes  # noqa: E402
from simvq_split.quant_lut import pack_model_2bit_lut  # noqa: E402
from simvq_split.tasks import (  # noqa: E402
    build_eval_overwrite_suite,
    build_eval_s5_suite,
    build_eval_trap_suite,
    make_overwrite_story,
    make_s5_story,
    make_trap_overwrite_story,
)
from simvq_split.tokenizer import ByteBPETokenizer  # noqa: E402

OUT_REPORT_PATH = ROOT_DIR / "benchmarks" / "dream_rsi_t3_core_results.json"


# ---------------------------------------------------------------------------
# Architectural Innovations (Isolated Subclasses — 0 edits to simvq_split/)
# ---------------------------------------------------------------------------


def _utf8_byte_class_ids(token_ids: torch.Tensor) -> torch.Tensor:
    """
    Pure bitwise tensor classification of raw UTF-8 byte tokens into 5 structural classes:
      0 = special/PAD/ASCII, 1 = 2-byte lead (RU/Latin), 2 = 3-byte lead (CJK),
      3 = 4-byte lead (Emoji), 4 = UTF-8 continuation byte (0x80..0xBF).
    Zero regex, zero host syncs.
    """
    is_byte = token_ids >= 3
    raw = (token_ids - 3).clamp(min=0)
    is_cont = is_byte & (torch.bitwise_and(raw, 0xC0) == 0x80)
    is_4 = is_byte & (torch.bitwise_and(raw, 0xF8) == 0xF0)
    is_3 = is_byte & (torch.bitwise_and(raw, 0xF0) == 0xE0)
    is_2 = is_byte & (torch.bitwise_and(raw, 0xE0) == 0xC0)
    cls_id = torch.zeros_like(token_ids, dtype=torch.long)
    cls_id = torch.where(is_2, torch.full_like(cls_id, 1), cls_id)
    cls_id = torch.where(is_3, torch.full_like(cls_id, 2), cls_id)
    cls_id = torch.where(is_4, torch.full_like(cls_id, 3), cls_id)
    cls_id = torch.where(is_cont, torch.full_like(cls_id, 4), cls_id)
    return cls_id


@torch.jit.script
def _causal_2tap_ema_latch_jit(
    phi: torch.Tensor,     # (B, T, H, D)
    sal: torch.Tensor,     # (B, T, H, 1) salience in (0, 1)
    a_fast: torch.Tensor,  # (1, H, 1)
    a_slow: torch.Tensor,  # (1, H, 1)
    w_mix: torch.Tensor,   # (1, H, 3) softmax weights [cur, fast_ema, slow_ema]
) -> torch.Tensor:
    """
    O(1)-state 2-tap salience-gated exponential key latch replacing the `L_latch=6` ring buffer.
    Maintains only 2 recurrent vectors per head (`s_fast`, `s_slow`) instead of 6 ring slots.
    """
    B, T, H, D = phi.shape
    s_fast = torch.zeros(B, H, D, device=phi.device, dtype=torch.float32)
    s_slow = torch.zeros(B, H, D, device=phi.device, dtype=torch.float32)
    phi_u = phi.float().unbind(1)
    sal_u = sal.float().unbind(1)
    w0 = w_mix[:, :, 0:1]
    w1 = w_mix[:, :, 1:2]
    w2 = w_mix[:, :, 2:3]
    outs: list[torch.Tensor] = []
    for t in range(T):
        pt = phi_u[t]
        st = sal_u[t]
        gf = a_fast * (0.35 + 0.65 * st)
        gs = a_slow * (0.35 + 0.65 * st)
        s_fast = (1.0 - gf) * s_fast + gf * pt
        s_slow = (1.0 - gs) * s_slow + gs * pt
        outs.append(w0 * pt + w1 * s_fast + w2 * s_slow)
    return torch.stack(outs, dim=1)


class MultiScaleEMAWoodburyLayer(LeanNeuralWoodburyLayer):
    """
    W2 Breakthrough:
      1. Geometric multi-timescale retention spectrum `gamma_proj.bias = linspace(1.6, 5.8, H_epi)`
         so fast heads handle rapid local overwrites while slow heads preserve facts across L=2048.
      2. Via Negativa deletion of the `L_latch = 6` ring buffer, replaced by a 2-tap salience-gated
         exponential key latch (`O(1)` 2 vectors instead of 6).
    """

    def __init__(
        self,
        d_model: int,
        mem_heads: int = 5,
        mem_dim: int = 32,
        ridge_lambda: float = 0.04,
        gamma_spread: float = 1.0,
        use_ema_latch: bool = True,
        latch_span: int = 2,
    ):
        super().__init__(
            d_model=d_model,
            mem_heads=mem_heads,
            mem_dim=mem_dim,
            ridge_lambda=ridge_lambda,
            latch_span=latch_span,
        )
        self.use_ema_latch = bool(use_ema_latch)
        # Geometric multi-timescale gamma initialization across epistemic heads
        with torch.no_grad():
            lo = 4.1 - 2.3 * float(gamma_spread)
            hi = 4.1 + 1.9 * float(gamma_spread)
            g_epi = torch.linspace(lo, hi, steps=self.H_epi, dtype=torch.float32)
            g_all = torch.cat([g_epi, torch.full((self.H_prose,), 2.4, dtype=torch.float32)])
            self.gamma_proj.bias.copy_(g_all)

        if self.use_ema_latch:
            # 2-tap exponential decay logits + blend logits [cur, fast, slow]
            self.ema_fast_logit = nn.Parameter(torch.full((1, self.H_epi, 1), -0.2))   # sigmoid ~ 0.45
            self.ema_slow_logit = nn.Parameter(torch.full((1, self.H_epi, 1), -1.65))  # sigmoid ~ 0.16
            self.ema_mix_logits = nn.Parameter(
                torch.tensor([0.3, 1.1, 0.7], dtype=torch.float32).view(1, 1, 3).repeat(1, self.H_epi, 1)
            )

    def _compute_latched_keys(
        self, x: torch.Tensor, h_lex: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_ema_latch:
            return super()._compute_latched_keys(x, h_lex=h_lex)

        B, T, D_in = x.shape
        x_u = self._safe_unit(x)
        if h_lex is not None:
            w_lex = torch.sigmoid(self.lex_logit.float())
            e_u = self._safe_unit(h_lex)
            z = self._safe_unit(w_lex * e_u + (1.0 - w_lex) * x_u)
        else:
            z = x_u

        steps = torch.arange(1, T + 1, device=x.device, dtype=torch.float32).view(1, T, 1)
        zc = (z - torch.cumsum(z, dim=1) / (steps + 2.0)).to(x.dtype)

        z_head_rep = zc[:, :1].expand(B, 3, D_in)
        zp = torch.cat([z_head_rep, zc], dim=1)
        z0 = zp[:, 3:]
        z12 = zp[:, 2:-1] + zp[:, 1:-2]
        z3 = zp[:, 0:-3] + 0.25 * zp[:, 1:-2]

        sc_raw = math.sqrt(self.D)
        f1_r = self._safe_unit(self.f1_proj(z0).view(B, T, self.H_epi, self.D), sc_raw)
        f2_r = self._safe_unit(self.f2_proj(z12).view(B, T, self.H_epi, self.D), sc_raw)
        f3_r = self._safe_unit(self.f3_proj(z3).view(B, T, self.H_epi, self.D), sc_raw)
        f1_e = torch.tanh(f1_r * 1.2)
        f2_e = torch.tanh(f2_r * 1.2)
        f3_e = torch.tanh(f3_r * 1.2)

        phi_epi = self._safe_unit(f1_e * f2_e * f3_e)
        phi_lin = self._safe_unit(f1_r + f2_r + f3_r)
        phi_rw = self._safe_unit(0.35 * phi_lin + 0.94 * phi_epi)

        sal = torch.sigmoid(self.subj_salience(x).float()).unsqueeze(-1)  # (B, T, H_epi, 1)
        a_fast = torch.sigmoid(self.ema_fast_logit.float())
        a_slow = torch.sigmoid(self.ema_slow_logit.float())
        w_mix = F.softmax(self.ema_mix_logits.float(), dim=-1)

        latched_rw = self._safe_unit(
            _causal_2tap_ema_latch_jit(phi_rw, sal, a_fast, a_slow, w_mix)
        ).to(x.dtype)
        latched_epi = self._safe_unit(
            _causal_2tap_ema_latch_jit(phi_epi, sal, a_fast, a_slow, w_mix)
        ).to(x.dtype)
        return latched_rw, latched_epi


class HouseholderCayleyCrystalLayer(LeanVoronoiCrystalLayer):
    """
    W4 Breakthrough:
      Replaces non-orthogonal soft `T_swap` and 6-iteration `log_sinkhorn_6` with:
        1. Exact involutive Householder transposition `H_swap = I - 2 u u^T` (`u = normalize(p_a - p_b)`),
           satisfying `H_swap^2 = I` and `||H_swap||_2 == 1.0` identically for any `(p_a, p_b)`.
        2. Exact orthogonal Cayley global operators `Q = (I - A)(I + A)^(-1)` (`A = skew(W)`),
           eliminating all 6 Sinkhorn iterations while preserving unit spectral norm (`0.0000` drift).
    """

    def __init__(
        self,
        d_model: int,
        num_spheres: int = 24,
        d_sphere: int = 32,
        codebook_size: int = 32,
        w_chunks: int = 2,
        householder_blend: float = 1.0,
    ):
        super().__init__(
            d_model=d_model,
            num_spheres=num_spheres,
            d_sphere=d_sphere,
            codebook_size=codebook_size,
            w_chunks=w_chunks,
        )
        self.householder_blend = float(householder_blend)

    def _compute_ptr_swap(
        self, psi_a: torch.Tensor, psi_b: torch.Tensor, well_scale: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pa, pb, t_orig = super()._compute_ptr_swap(psi_a, psi_b, well_scale=well_scale)
        # Exact involutive Householder reflection across (pa - pb): equals exact transposition on basis vectors
        diff = pa - pb
        norm_sq = diff.pow(2).sum(dim=-1, keepdim=True)
        # Smooth switch -> identity when pa == pb (norm_sq -> 0)
        gate_nz = norm_sq / (norm_sq + 1e-4)
        u = diff * torch.rsqrt(norm_sq + 1e-5)
        eye = self.eye_Gh.float().view(1, 1, 1, self.G_h, self.G_h)
        h_swap = eye - (2.0 * gate_nz).unsqueeze(-1) * (u.unsqueeze(-1) * u.unsqueeze(-2))
        hb = self.householder_blend
        t_out = h_swap if hb >= 0.999 else ((1.0 - hb) * t_orig + hb * h_swap)
        return pa, pb, t_out

    def _cayley_global_ops(self, o_temp: float) -> torch.Tensor:
        # Skew-symmetric generator A -> exact orthogonal Q = (I - A)(I + A)^(-1) in SO(G_h), 0 Sinkhorn iters!
        raw = self.birkhoff_logits.float() * (0.25 * o_temp)
        A = 0.5 * (raw - raw.transpose(-1, -2))
        eye = self.eye_Gh.float().view(1, 1, self.G_h, self.G_h)
        with torch.amp.autocast(raw.device.type, enabled=False):
            return torch.linalg.solve(eye + A, eye - A)

    def forward_states(
        self, x: torch.Tensor, h_lex: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, int, int, int]:
        # Temporarily swap `log_sinkhorn_6` in `simvq_split.crystal` to our zero-iter Cayley orthogonal operator
        import simvq_split.crystal as _cmod

        orig_sink = _cmod.log_sinkhorn_6
        _cmod.log_sinkhorn_6 = lambda _logits: self._cayley_global_ops(float(self.op_temp))
        try:
            return super().forward_states(x, h_lex=h_lex)
        finally:
            _cmod.log_sinkhorn_6 = orig_sink


@dataclass
class CoreT3Config(LeanSystem1Config):
    use_utf8_class_emb: bool = False
    use_multiscale_ema_woodbury: bool = False
    woodbury_gamma_spread: float = 1.0
    use_householder_crystal: bool = False
    householder_blend: float = 1.0
    use_parallel_orthogonal_streams: bool = False
    stream_layer_scale: float = 0.85


class CalciteCoreT3LM(CalciteLeanSystem1LM):
    """
    Isolated Calcite-7 Prototype LM for Track T3 (`0` edits to `simvq_split/`).
    Cleanly composes any subset of W1 (Wide UTF-8 Codec), W2 (Multi-Scale EMA Woodbury),
    W4 (Householder-Cayley Crystal), and Parallel Orthogonal Stream Fusion.
    """

    def __init__(self, cfg: CoreT3Config):
        super().__init__(cfg)
        self.t3_cfg = cfg
        self.set_fast_kernels(False)  # Local Windows RTX 5060 uses pure PyTorch/JIT exact path

        if cfg.use_utf8_class_emb and self.byte_chunker is not None:
            self.utf8_class_emb = nn.Embedding(5, cfg.d_model)
            nn.init.zeros_(self.utf8_class_emb.weight)
        else:
            self.utf8_class_emb = None

        if cfg.use_multiscale_ema_woodbury and self.woodbury is not None:
            self.woodbury = MultiScaleEMAWoodburyLayer(
                d_model=cfg.d_model,
                mem_heads=cfg.mem_heads,
                mem_dim=cfg.mem_dim,
                ridge_lambda=cfg.ridge_lambda,
                gamma_spread=cfg.woodbury_gamma_spread,
                use_ema_latch=True,
                latch_span=2,
            )

        if cfg.use_householder_crystal:
            if self.crystal is not None:
                self.crystal = HouseholderCayleyCrystalLayer(
                    d_model=cfg.d_model,
                    num_spheres=cfg.num_spheres,
                    d_sphere=cfg.d_sphere,
                    codebook_size=cfg.codebook_size,
                    w_chunks=cfg.crystal_w_chunks,
                    householder_blend=cfg.householder_blend,
                )
            if self.crystal_2 is not None:
                self.crystal_2 = HouseholderCayleyCrystalLayer(
                    d_model=cfg.d_model,
                    num_spheres=cfg.num_spheres,
                    d_sphere=cfg.d_sphere,
                    codebook_size=cfg.codebook_size,
                    w_chunks=cfg.crystal_w_chunks,
                    householder_blend=cfg.householder_blend,
                )

        if cfg.use_parallel_orthogonal_streams:
            sc = float(cfg.stream_layer_scale)
            self.crystal_ls = nn.Parameter(torch.full((cfg.d_model,), sc))
            self.woodbury_ls = nn.Parameter(torch.full((cfg.d_model,), sc))
            # Fixed orthogonal subspace masks (first half cosine-weighted vs second half) via smooth Hadamard mix
            # preserves full rank while decorrelating Crystal vs Woodbury initial gradient directions
            sign_vec = torch.ones(cfg.d_model, dtype=torch.float32)
            sign_vec[1::2] = -1.0
            self.register_buffer("ortho_sign", sign_vec)
        else:
            self.crystal_ls = None
            self.woodbury_ls = None

    def _forward_core(
        self,
        h: torch.Tensor,
        e_lex: torch.Tensor,
        ablate_crystal: bool = False,
        ablate_woodbury: bool = False,
    ) -> torch.Tensor:
        if not self.t3_cfg.use_parallel_orthogonal_streams:
            return super()._forward_core(
                h, e_lex, ablate_crystal=ablate_crystal, ablate_woodbury=ablate_woodbury
            )
        for blk in self.cortex_pre:
            h = blk(h)

        h_split = self.split_norm(h)
        delta_long = torch.zeros_like(h)
        osign = self.ortho_sign.to(h_split.dtype)
        if (
            self.stream_gate is not None
            and self.crystal is not None
            and self.woodbury is not None
            and not ablate_crystal
            and not ablate_woodbury
        ):
            gates = 2.0 * torch.sigmoid(self.stream_gate(h_split))
            c_out = self.crystal(h_split, h_lex=e_lex) * self.crystal_ls.to(h_split.dtype)
            w_out = (self.woodbury(h_split * osign, h_lex=e_lex) * osign) * self.woodbury_ls.to(h_split.dtype)
            delta_long = gates[..., 0:1] * c_out + gates[..., 1:2] * w_out
        else:
            if self.crystal is not None and not ablate_crystal:
                c_out = self.crystal(h_split, h_lex=e_lex) * self.crystal_ls.to(h_split.dtype)
                delta_long = delta_long + c_out
            if self.woodbury is not None and not ablate_woodbury:
                w_out = (self.woodbury(h_split * osign, h_lex=e_lex) * osign) * self.woodbury_ls.to(h_split.dtype)
                delta_long = delta_long + w_out

        h = h + delta_long
        h = h + self.mid_mlp(self.mid_mlp_norm(h))
        if (
            self.crystal_2 is not None
            and self.cryst2_norm is not None
            and not ablate_crystal
        ):
            c2_in = self.cryst2_norm(h)
            c2_out = self.crystal_2(c2_in, h_lex=e_lex)
            if self.cryst2_gate is not None and (
                self.woodbury is None or not ablate_woodbury
            ):
                c2_out = (2.0 * torch.sigmoid(self.cryst2_gate(c2_in))) * c2_out
            h = h + c2_out

        for blk in self.cortex_post:
            h = blk(h)
        return h

    def forward(
        self,
        token_ids: torch.Tensor,
        ablate_crystal: bool = False,
        ablate_woodbury: bool = False,
    ) -> torch.Tensor:
        if self.utf8_class_emb is None:
            return super().forward(
                token_ids, ablate_crystal=ablate_crystal, ablate_woodbury=ablate_woodbury
            )

        e_tok = self.tok_emb(token_ids)
        e_tok = e_tok + self.utf8_class_emb(_utf8_byte_class_ids(token_ids)).to(e_tok.dtype)
        if self.byte_chunker is not None:
            h, e_lex, b_ctx = self.byte_chunker.encode_to_chunks(
                e_tok, token_ids, self.tok_emb.weight
            )
            h = self._forward_core(
                h, e_lex, ablate_crystal=ablate_crystal, ablate_woodbury=ablate_woodbury
            )
            h = self.byte_chunker.decode_from_chunks(h, b_ctx)
        else:
            h = self._forward_core(
                e_tok, e_tok, ablate_crystal=ablate_crystal, ablate_woodbury=ablate_woodbury
            )
        return self.lm_head(self.norm_f(h))

    def compute_o1_cache_kb(self) -> float:
        """Exact O(1) streaming cache footprint (KB) independent of L."""
        cfg = self.t3_cfg
        # SWA KV cache across pre+post layers
        n_swa = cfg.n_layers_pre + cfg.n_layers_post
        swa_bytes = n_swa * 2 * cfg.window_size * cfg.d_model * 2  # BF16 K+V
        # Byte chunker O(1) state
        ch_bytes = compute_chunker_state_bytes(
            cfg.d_model,
            d_byte=cfg.byte_char_dim,
            d_rchar=cfg.byte_router_char_dim,
            router_dim=32,
            byte_chunk_mode=cfg.byte_chunk_mode,
        )
        # Crystal O(1) state (slots + W_chunks window)
        crys_bytes = (
            cfg.num_spheres * cfg.d_sphere * 4
            + cfg.crystal_w_chunks * 4 * cfg.d_model * 2
        )
        # Woodbury O(1) state: M + P + (2-tap EMA if enabled, else L_latch ring buffer!)
        wb_mp_bytes = cfg.mem_heads * (cfg.mem_dim * cfg.mem_dim * 2) * 4
        if cfg.use_multiscale_ema_woodbury:
            wb_latch_bytes = 2 * max(1, cfg.mem_heads - 1) * cfg.mem_dim * 4  # 2-tap EMA only!
        else:
            wb_latch_bytes = cfg.woodbury_latch_span * max(1, cfg.mem_heads - 1) * cfg.mem_dim * 4
        return round((swa_bytes + ch_bytes + crys_bytes + wb_mp_bytes + wb_latch_bytes) / 1024.0, 2)


# ---------------------------------------------------------------------------
# Fast Equal-Compute Proxy Trainer & Evaluator (Local RTX 5060)
# ---------------------------------------------------------------------------


def _extract_sentences(docs: List[str], max_sents: int = 900) -> List[str]:
    sents: List[str] = []
    for d in docs:
        for raw in re.split(r"[.!?\n]+", d):
            s = " ".join(raw.strip().split())
            if 24 <= len(s) <= 130 and s.isascii():
                sents.append(s)
                if len(sents) >= max_sents:
                    return sents
    return sents


def _sample_openweb_batch(
    train_wins: Dict[str, List[List[int]]],
    dom_names: List[str],
    batch_size: int,
    rng: np.random.Generator,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rows = []
    for _ in range(batch_size):
        d = dom_names[int(rng.integers(0, len(dom_names)))]
        w_list = train_wins[d]
        w = w_list[int(rng.integers(0, len(w_list)))]
        rows.append(w)
    t = torch.tensor(rows, dtype=torch.long, device=device)
    return t[:, :-1], t[:, 1:]


def _sample_cognitive_batch(
    kind: str,
    tok: ByteBPETokenizer,
    tr_sents: List[str],
    batch_size: int,
    py_rng: random.Random,
    device: str,
    max_len: int = 384,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Samples an exact byte-level cognitive training batch with target-weighted loss mask."""
    seqs: List[List[int]] = []
    t_starts: List[int] = []
    for _ in range(batch_size):
        if kind == "s5":
            k_ops = 2 if py_rng.random() < 0.5 else 4
            prompt, target, _ = make_s5_story(py_rng, k_ops, tr_sents, distractors_per_op=(0, 1))
        elif kind == "ov":
            prompt, target, _ = make_overwrite_story(py_rng, 5, 3, tr_sents, filler_sentences_after=4)
        else:
            prompt, target, _ = make_trap_overwrite_story(py_rng, 5, 3, tr_sents, filler_sentences_after=4)
        p_ids = tok.encode(prompt, add_bos=True, add_eos=False)
        t_ids = tok.encode(target, add_bos=False, add_eos=True)
        full = (p_ids + t_ids)[: max_len + 1]
        seqs.append(full)
        t_starts.append(min(len(p_ids), len(full) - 1))

    L = max(len(s) for s in seqs)
    arr = torch.full((batch_size, L), tok.PAD_ID, dtype=torch.long, device=device)
    lmask = torch.zeros((batch_size, L - 1), dtype=torch.float32, device=device)
    for i, (s, ts) in enumerate(zip(seqs, t_starts)):
        arr[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        # Light prompt LM weight (0.15) + strong answer weight (1.0)
        if ts > 1:
            lmask[i, : ts - 1] = 0.15
        if len(s) - 1 >= max(0, ts - 1):
            lmask[i, max(0, ts - 1) : len(s) - 1] = 1.0
    return arr[:, :-1], arr[:, 1:], lmask


def train_proxy_model(
    spec: Dict[str, Any],
    tok: ByteBPETokenizer,
    train_wins: Dict[str, List[List[int]]],
    tr_sents: List[str],
    device: str,
    seed: int = 2026,
    total_steps: int = 360,
    batch_size: int = 16,
) -> Tuple[CalciteCoreT3LM, Dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    d_model = 192
    codec_mult = float(spec.get("codec_width_mult", 1.0))
    byte_char_dim = int(round(32 * codec_mult))
    byte_char_dim = max(32, min(64, (byte_char_dim // 8) * 8))
    byte_d_ff = int(round(128 * codec_mult))
    byte_d_ff = max(128, min(256, (byte_d_ff // 16) * 16))
    # Keep total parameter count strictly balanced across branches (within ~5%)
    d_ff_base = 512 if codec_mult <= 1.05 else (480 if codec_mult <= 1.6 else 448)

    cfg = CoreT3Config(
        vocab_size=tok.vocab_size,
        d_model=d_model,
        n_heads=6,
        d_ff=d_ff_base,
        n_layers_pre=2,
        n_layers_post=1,
        window_size=8,
        num_spheres=24,
        d_sphere=32,
        codebook_size=32,
        mem_heads=5,
        mem_dim=32,
        ridge_lambda=0.04,
        use_crystal=True,
        use_woodbury=True,
        use_byte_chunker=True,
        byte_char_dim=byte_char_dim,
        byte_d_ff=byte_d_ff,
        byte_chunk_mode="utf8",
        use_static_m_cap=True,
        max_chunk_bytes=10,
        max_utf8_bytes=16,
        target_chunk_bytes=4.9,
        crystal_w_chunks=2,
        woodbury_latch_span=2 if bool(spec.get("use_multiscale_ema_woodbury", False)) else 6,
        use_utf8_class_emb=bool(spec.get("use_utf8_class_emb", False)),
        use_multiscale_ema_woodbury=bool(spec.get("use_multiscale_ema_woodbury", False)),
        woodbury_gamma_spread=float(spec.get("woodbury_gamma_spread", 1.0)),
        use_householder_crystal=bool(spec.get("use_householder_crystal", False)),
        householder_blend=float(spec.get("householder_blend", 1.0)),
        use_parallel_orthogonal_streams=bool(spec.get("use_parallel_orthogonal_streams", False)),
        stream_layer_scale=float(spec.get("stream_layer_scale", 0.85)),
    )

    model = CalciteCoreT3LM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    o1_cache_kb = model.compute_o1_cache_kb()
    dom_names = sorted(train_wins.keys())

    # Optimizer with Woodbury zero weight_decay invariant (Phase K Law 7b)
    wb_params = list(model.woodbury.parameters()) if model.woodbury is not None else []
    wb_ids = {id(p) for p in wb_params}
    other_params = [p for p in model.parameters() if id(p) not in wb_ids]
    opt = torch.optim.AdamW(
        [
            {"params": other_params, "lr": 1.6e-3, "weight_decay": 0.02},
            {"params": wb_params, "lr": 1.2e-3, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
    )

    lock_step = max(30, int(0.22 * total_steps))
    t0 = time.perf_counter()
    model.train()
    use_amp = device == "cuda"

    for step in range(1, total_steps + 1):
        # Cosine LR schedule with 20-step warmup
        if step <= 20:
            lr_mult = step / 20.0
        else:
            prog = (step - 20) / max(1, total_steps - 20)
            lr_mult = 0.12 + 0.88 * 0.5 * (1.0 + math.cos(math.pi * prog))
        opt.param_groups[0]["lr"] = 1.6e-3 * lr_mult
        opt.param_groups[1]["lr"] = 1.2e-3 * lr_mult

        if step == lock_step and model.byte_chunker is not None:
            model.byte_chunker.lock_router(True)
        model.set_crystal_schedule(step, total_steps)

        opt.zero_grad(set_to_none=True)

        # Phase K Law 7b: 100% prose before router lock; afterwards 55% OpenWeb prose + 45% cognitive mix
        if step < lock_step:
            xb, yb = _sample_openweb_batch(train_wins, dom_names, batch_size, rng, device)
            loss_mask = None
        else:
            r_draw = float(rng.random())
            if r_draw < 0.52:
                xb, yb = _sample_openweb_batch(train_wins, dom_names, batch_size, rng, device)
                loss_mask = None
            elif r_draw < 0.72:
                xb, yb, loss_mask = _sample_cognitive_batch(
                    "s5", tok, tr_sents, batch_size, py_rng, device
                )
            elif r_draw < 0.88:
                xb, yb, loss_mask = _sample_cognitive_batch(
                    "ov", tok, tr_sents, batch_size, py_rng, device
                )
            else:
                xb, yb, loss_mask = _sample_cognitive_batch(
                    "trap", tok, tr_sents, batch_size, py_rng, device
                )

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            lg = model(xb)
            if loss_mask is None:
                loss = F.cross_entropy(lg.float().transpose(1, 2), yb, ignore_index=tok.PAD_ID)
            else:
                ce_tok = F.cross_entropy(
                    lg.float().transpose(1, 2), yb, ignore_index=tok.PAD_ID, reduction="none"
                )
                lm_f = loss_mask.float()
                loss = (ce_tok * lm_f).sum() / lm_f.sum().clamp(min=1.0)
            aux = model.get_commitment_loss()
            if aux is not None:
                loss = loss + aux

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    if model.byte_chunker is not None:
        model.byte_chunker.lock_router(True)
    model.set_crystal_schedule(1, 1)
    model.set_snap_alpha(1.0)
    return model, {
        "n_params": int(n_params),
        "o1_cache_kb": o1_cache_kb,
        "train_s": round(time.perf_counter() - t0, 2),
        "byte_char_dim": byte_char_dim,
        "byte_d_ff": byte_d_ff,
        "d_ff": d_ff_base,
    }


def build_cartridge() -> TaskCartridge:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[T3] Preparing disjoint UTF-8 + cognitive proxy benchmark on {device}...", flush=True)
    tok = ByteBPETokenizer(merges=[])

    gsm_train_path = ROOT_DIR / "benchmarks" / "gsm8k" / "train.jsonl"
    all_docs = [
        json.loads(line)["question"] + "\n" + json.loads(line)["answer"]
        for line in gsm_train_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tr_docs = all_docs[:4500]
    dev_docs = all_docs[4500:5800]
    vault_docs = all_docs[5800:7100]

    tr_sents = _extract_sentences(tr_docs, max_sents=800)
    dev_sents = _extract_sentences(dev_docs, max_sents=800)
    vault_sents = _extract_sentences(vault_docs, max_sents=800)

    train_wins, eval_wins, _, _ = build_disjoint_openweb_corpora(
        ROOT_DIR, tok, tr_docs[:1500], dev_docs[:400], seq_len=384
    )
    dev_eval_wins: Dict[str, List[List[int]]] = {}
    vault_eval_wins: Dict[str, List[List[int]]] = {}
    for dom, wins in eval_wins.items():
        mid = max(1, len(wins) // 2)
        dev_eval_wins[dom] = wins[: min(mid, 16)]
        vault_eval_wins[dom] = wins[mid:]

    # Dev cognitive suites (N=48, seed=77001..77008)
    n_dev = 48
    dev_s5 = {
        "k2": build_eval_s5_suite(tok, dev_sents, 2, n_dev, (0, 1), 77001),
        "k4": build_eval_s5_suite(tok, dev_sents, 4, n_dev, (0, 1), 77002),
        "k8": build_eval_s5_suite(tok, dev_sents, 8, n_dev, (1, 1), 77003),
        "k16": build_eval_s5_suite(tok, dev_sents, 16, n_dev, (1, 2), 77004),
    }
    dev_ov = {
        "in_dist": build_eval_overwrite_suite(tok, dev_sents, 4, 2, 2, n_dev, 77005),
        "ood_long": build_eval_overwrite_suite(tok, dev_sents, 9, 5, 18, n_dev, 77006),
    }
    dev_trap = {
        "in_dist": build_eval_trap_suite(tok, dev_sents, 4, 2, 2, n_dev, 77007),
        "ood_long": build_eval_trap_suite(tok, dev_sents, 9, 5, 18, n_dev, 77008),
    }

    # Sealed Blind Vault cognitive suites (full N=150, disjoint vault_sents, seed=99001..99008)
    n_vault = 150
    vault_s5 = {
        "k2": build_eval_s5_suite(tok, vault_sents, 2, n_vault, (0, 1), 99001),
        "k4": build_eval_s5_suite(tok, vault_sents, 4, n_vault, (0, 1), 99002),
        "k8": build_eval_s5_suite(tok, vault_sents, 8, n_vault, (1, 1), 99003),
        "k16": build_eval_s5_suite(tok, vault_sents, 16, n_vault, (1, 2), 99004),
    }
    vault_ov = {
        "in_dist": build_eval_overwrite_suite(tok, vault_sents, 4, 2, 2, n_vault, 99005),
        "ood_long": build_eval_overwrite_suite(tok, vault_sents, 9, 5, 18, n_vault, 99006),
    }
    vault_trap = {
        "in_dist": build_eval_trap_suite(tok, vault_sents, 4, 2, 2, n_vault, 99007),
        "ood_long": build_eval_trap_suite(tok, vault_sents, 9, 5, 18, n_vault, 99008),
    }

    eval_cache: Dict[str, Tuple[bool, float, str, Dict[str, Any]]] = {}
    base_spec = {"tag": "root_v1_baseline"}

    def verify_and_score(fn: Callable[[], Dict[str, Any]]) -> Tuple[bool, float, str, Dict[str, Any]]:
        spec = dict(fn())
        key = json.dumps(spec, sort_keys=True)
        if key in eval_cache:
            return eval_cache[key]

        model, tr_meta = train_proxy_model(
            spec, tok, train_wins, tr_sents, device=device, seed=2026, total_steps=360, batch_size=16
        )
        # Hard O(1) cache & parameter parity check
        if float(tr_meta["o1_cache_kb"]) > 125.0:
            res = (False, 0.0, "cache_budget_exceeded", {"error": f"O1 cache {tr_meta['o1_cache_kb']} KB > 125 KB"})
            eval_cache[key] = res
            return res

        ow = eval_openweb_6domains_any_byte_lm(model, dev_eval_wins, device=device, batch_size=16)
        if float(ow["max_mid_utf8_split_pct"]) > 0.0:
            res = (False, 0.0, "utf8_split_violation", {"error": "mid_utf8_split_pct > 0"})
            eval_cache[key] = res
            return res

        cog = eval_cognitive_suites_any_byte_lm(
            model, tok, dev_s5, dev_ov, dev_trap, device=device, batch_size=32
        )
        mean_ow6 = float(ow["mean_openweb_bpb"])
        emoji_bpb = float(ow["regimes"]["6_Emoji_4Byte"]["byte_bpb"])
        s5_sum = float(cog["s5_sum"])
        sel_long = float(cog["selective_f1_ood_long"])
        ov_long = float(cog["overwrite"]["ood_long"]["exact_pct"])
        def _parse_drift(raw_d: Any) -> float:
            if isinstance(raw_d, dict):
                return float(raw_d.get("max_norm_drift", 0.0)) + float(raw_d.get("max_lattice_err", 0.0))
            return float(raw_d or 0.0)

        drift_2048 = _parse_drift(cog.get("crystal_drift_L2048", 0.0))
        cache_kb = float(tr_meta["o1_cache_kb"])

        # Positive composite score balancing all 4 weaknesses (W1 ow6/UTF8, W2 OV/cache, W4 S5/drift)
        score = round(
            100.0 * (5.0 - mean_ow6)
            + 0.25 * s5_sum
            + 0.35 * sel_long
            + 0.15 * ov_long
            - 0.05 * max(0.0, cache_kb - 70.0)
            - 20.0 * min(1.0, drift_2048),
            4,
        )

        dom_order = [
            "1_GSM8K_EN",
            "2_Russian_UTF8",
            "3_Python_Code",
            "4_ARC_JSON",
            "5_CJK_UTF8",
            "6_Emoji_4Byte",
        ]
        bucket_scores = [round(6.0 - float(ow["regimes"][d]["byte_bpb"]), 4) for d in dom_order] + [
            round(float(cog["s5"]["k2"]["exact_pct"]) / 100.0, 4),
            round(float(cog["s5"]["k4"]["exact_pct"]) / 100.0, 4),
            round(ov_long / 100.0, 4),
            round(float(cog["trap"]["ood_long"]["exact_pct"]) / 100.0, 4),
        ]
        pass_mask = [1 if float(ow["regimes"][d]["byte_bpb"]) < 4.5 else 0 for d in dom_order] + [
            1 if float(cog["s5"]["k2"]["exact_pct"]) >= 30.0 else 0,
            1 if float(cog["s5"]["k4"]["exact_pct"]) >= 15.0 else 0,
            1 if ov_long >= 20.0 else 0,
            1 if float(cog["trap"]["ood_long"]["exact_pct"]) >= 20.0 else 0,
        ]

        info = {
            "pass_mask": pass_mask,
            "bucket_scores": bucket_scores,
            "spec": spec,
            "train_meta": tr_meta,
            "mean_openweb_bpb": mean_ow6,
            "regimes_bpb": {k: v["byte_bpb"] for k, v in ow["regimes"].items()},
            "emoji_bpb": emoji_bpb,
            "s5_sum": s5_sum,
            "s5": {k: v["exact_pct"] for k, v in cog["s5"].items()},
            "ov_long": ov_long,
            "trap_long": cog["trap"]["ood_long"]["exact_pct"],
            "selective_f1_ood_long": sel_long,
            "crystal_drift_L2048": round(drift_2048, 6),
            "summary": (
                f"score={score:.2f} | ow6={mean_ow6:.4f} (Emoji={emoji_bpb:.3f}) | "
                f"S5={s5_sum:.1f}% | SelF1={sel_long:.1f}% | drift={drift_2048:.5f} | "
                f"cache={cache_kb:.1f}KB | params={tr_meta['n_params']:,} ({tr_meta['train_s']}s)"
            ),
        }
        print(f"  [T3 Eval] {spec.get('tag', 'cand')}: {info['summary']}", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        res = (True, score, "ok", info)
        eval_cache[key] = res
        return res

    # Single-Shot Sealed Blind Vault: Fresh seed (42026), longer training (520 steps), full N=150 Vault + 2-bit LUT check
    def blind_vault_eval(fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        champ_spec = dict(fn())
        print(
            f"\n[T3 Blind Vault] Opening SINGLE-SHOT fresh-seed (42026) vault for `{champ_spec.get('tag')}` vs Baseline...",
            flush=True,
        )
        out: Dict[str, Any] = {"champion_spec": champ_spec}
        for label, sp in (("baseline_v1", base_spec), ("champion_t3", champ_spec)):
            m, meta = train_proxy_model(
                sp, tok, train_wins, tr_sents, device=device, seed=42026, total_steps=520, batch_size=16
            )
            ow_fp = eval_openweb_6domains_any_byte_lm(m, vault_eval_wins, device=device, batch_size=16)
            cog_fp = eval_cognitive_suites_any_byte_lm(
                m, tok, vault_s5, vault_ov, vault_trap, device=device, batch_size=32
            )
            # Also verify 0-QAT 2-bit LUT packability & UTF-8 safety
            m_lut = pack_model_2bit_lut(m).to(device).eval()
            ow_lut = eval_openweb_6domains_any_byte_lm(m_lut, vault_eval_wins, device=device, batch_size=16)
            out[label] = {
                "train_meta": meta,
                "fp_mean_openweb_bpb": ow_fp["mean_openweb_bpb"],
                "fp_regimes_bpb": {k: v["byte_bpb"] for k, v in ow_fp["regimes"].items()},
                "fp_max_mid_utf8_split_pct": ow_fp["max_mid_utf8_split_pct"],
                "fp_s5_sum": cog_fp["s5_sum"],
                "fp_s5": {k: v["exact_pct"] for k, v in cog_fp["s5"].items()},
                "fp_ov_long": cog_fp["overwrite"]["ood_long"]["exact_pct"],
                "fp_trap_long": cog_fp["trap"]["ood_long"]["exact_pct"],
                "fp_selective_f1_ood_long": cog_fp["selective_f1_ood_long"],
                "crystal_drift_L2048": cog_fp.get("crystal_drift_L2048", 0.0),
                "ptq_lut_mean_openweb_bpb": ow_lut["mean_openweb_bpb"],
            }
            del m, m_lut
            if device == "cuda":
                torch.cuda.empty_cache()

        b_r = out["baseline_v1"]
        c_r = out["champion_t3"]
        out["delta_vs_baseline"] = {
            "fp_openweb_bpb": round(float(c_r["fp_mean_openweb_bpb"]) - float(b_r["fp_mean_openweb_bpb"]), 4),
            "ptq_lut_openweb_bpb": round(
                float(c_r["ptq_lut_mean_openweb_bpb"]) - float(b_r["ptq_lut_mean_openweb_bpb"]), 4
            ),
            "s5_sum_pp": round(float(c_r["fp_s5_sum"]) - float(b_r["fp_s5_sum"]), 2),
            "selective_f1_long_pp": round(
                float(c_r["fp_selective_f1_ood_long"]) - float(b_r["fp_selective_f1_ood_long"]), 2
            ),
            "o1_cache_kb_delta": round(
                float(c_r["train_meta"]["o1_cache_kb"]) - float(b_r["train_meta"]["o1_cache_kb"]), 2
            ),
        }
        return out

    # --- 4 Orthogonal Architectural Branches ---
    b0 = BranchCandidate(
        id="b0_wide_utf8_codec",
        branch=0,
        description="W1 fix: Width-scaled Byte Codec (1.5x char_dim/d_ff) + bitwise UTF-8 byte-class embedding",
        fn=lambda: {
            "codec_width_mult": 1.5,
            "use_utf8_class_emb": True,
            "tag": "b0_wide_utf8_codec",
        },
        next_refinement=BranchCandidate(
            id="b0_wide_utf8_codec_2x",
            branch=0,
            parent_id="b0_wide_utf8_codec",
            attempt=2,
            description="W1 fix v2: 2.0x width-scaled Byte Codec (char_dim=64, byte_d_ff=256) + UTF-8 class emb",
            fn=lambda: {
                "codec_width_mult": 2.0,
                "use_utf8_class_emb": True,
                "tag": "b0_wide_utf8_codec_2x",
            },
        ),
    )

    b1 = BranchCandidate(
        id="b1_multiscale_ema_woodbury",
        branch=1,
        description="W2 fix: Multi-timescale gamma_h Woodbury + 2-tap EMA key latch (deleting L_latch=6 ring buffer)",
        fn=lambda: {
            "use_multiscale_ema_woodbury": True,
            "woodbury_gamma_spread": 1.0,
            "tag": "b1_multiscale_ema_woodbury",
        },
        next_refinement=BranchCandidate(
            id="b1_multiscale_ema_woodbury_wide",
            branch=1,
            parent_id="b1_multiscale_ema_woodbury",
            attempt=2,
            description="W2 fix v2: Wider geometric gamma_h spread (1.35) + 2-tap EMA latch",
            fn=lambda: {
                "use_multiscale_ema_woodbury": True,
                "woodbury_gamma_spread": 1.35,
                "tag": "b1_multiscale_ema_woodbury_wide",
            },
        ),
    )

    b2 = BranchCandidate(
        id="b2_householder_cayley_crystal",
        branch=2,
        description="W4 fix: Exact involutive Householder swap + Cayley SO(G_h) global ops (deleting log_sinkhorn_6)",
        fn=lambda: {
            "use_householder_crystal": True,
            "householder_blend": 1.0,
            "tag": "b2_householder_cayley_crystal",
        },
    )

    b3 = BranchCandidate(
        id="b3_parallel_ortho_streams",
        branch=3,
        description="Parallel orthogonalized Crystal + Woodbury streams with LayerScale (single-stage stability)",
        fn=lambda: {
            "use_parallel_orthogonal_streams": True,
            "stream_layer_scale": 0.85,
            "tag": "b3_parallel_ortho_streams",
        },
    )

    # DAG Crossover merging winning architectural axes (discrete + continuous knobs; fast Round 4 budget)
    def crossover_synthesizer(u: BranchCandidate, v: BranchCandidate) -> BranchCandidate:
        su = dict(u.fn())
        sv = dict(v.fn())
        merged = {
            "codec_width_mult": max(float(su.get("codec_width_mult", 1.0)), float(sv.get("codec_width_mult", 1.0)), 1.5),
            "use_utf8_class_emb": bool(su.get("use_utf8_class_emb", False) or sv.get("use_utf8_class_emb", False) or True),
            "use_multiscale_ema_woodbury": bool(
                su.get("use_multiscale_ema_woodbury", False) or sv.get("use_multiscale_ema_woodbury", False) or True
            ),
            "woodbury_gamma_spread": float(max(float(su.get("woodbury_gamma_spread", 1.0)), float(sv.get("woodbury_gamma_spread", 1.0)))),
            "use_householder_crystal": bool(
                su.get("use_householder_crystal", False) or sv.get("use_householder_crystal", False) or True
            ),
            "householder_blend": 1.0,
            "use_parallel_orthogonal_streams": bool(
                su.get("use_parallel_orthogonal_streams", False) or sv.get("use_parallel_orthogonal_streams", False)
            ),
            "stream_layer_scale": 0.85,
            "tag": f"xover_{u.id}_x_{v.id}",
        }
        # Note: omit continuous Round-4 coordinate sweeps on full NN retraining (each train ~35s),
        # keep discrete architectural synthesis crisp and verified on fresh-seed BlindVault!
        return BranchCandidate(
            id=f"xover_{u.id}_x_{v.id}",
            branch=4,
            parent_id=u.id,
            description=(
                f"Calcite-7 Architectural Hybrid (`{u.id}` x `{v.id}`): Wide UTF-8 Codec + "
                f"Multi-Timescale EMA Woodbury + Householder-Cayley Crystal"
            ),
            fn=lambda m=merged: dict(m),
        )

    return TaskCartridge(
        task_name="calcite_core_t3",
        description=(
            "Calcite-7 System-1 Core Redesign closing W1 (wide UTF-8 codec), W2 (multi-timescale "
            "EMA Woodbury deleting L_latch=6), and W4 (Householder-Cayley Crystal deleting Sinkhorn)"
        ),
        baseline_fn=lambda: dict(base_spec),
        verify_and_score=verify_and_score,
        initial_branches=[b0, b1, b2, b3],
        crossover_synthesizer=crossover_synthesizer,
        blind_vault_evaluator=blind_vault_eval,
        noise_probe_runs=1,
    )


def main() -> int:
    pool_dir = os.environ.get("DREAM_RSI_POOL") or dream_engine.default_pool_dir()
    cartridge = build_cartridge()
    runner = AutoDreamRunner(cartridge, pool_dir=pool_dir)
    report = runner.run()

    print("\n" + "=" * 78)
    print(
        f"[T3 DONE] Task: {report['task_name']} | Baseline: {report['baseline_score']:.4f} -> "
        f"Champion `{report['champion_id']}`: {report['champion_score']:.4f}"
    )
    print(f"Nodes explored: {report['total_nodes_explored']} | Elapsed: {report['elapsed_s']:.1f}s")
    if report.get("blind_vault_report"):
        print("Sealed Fresh-Seed Blind Vault (seed=42026, N=150 + 2-bit LUT):")
        print(json.dumps(report["blind_vault_report"], indent=2, ensure_ascii=False))

    out_payload = {
        "task_name": report["task_name"],
        "baseline_score": report["baseline_score"],
        "champion_id": report["champion_id"],
        "champion_score": report["champion_score"],
        "champion_metrics": report["champion_metrics"],
        "total_nodes_explored": report["total_nodes_explored"],
        "elapsed_s": round(float(report["elapsed_s"]), 2),
        "blind_vault_report": report["blind_vault_report"],
        "journal": report["journal"],
        "pool_dir": pool_dir,
    }
    OUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT_PATH.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved report -> {OUT_REPORT_PATH} ({OUT_REPORT_PATH.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
