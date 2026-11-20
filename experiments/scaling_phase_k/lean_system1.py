"""
Pure Neural Lean `Calcite System-1` Architecture (`CalciteLeanSystem1LM`) for Phase K Scaling Laws.

Implements `Via Negativa` ring-buffer pruning cleanly outside protected `simvq_split/` via
configurable `window_size`, `crystal_w_chunks`, and `woodbury_latch_span` without adding any
new abstraction layers or host heuristics (`Zero-Cheat` AST pure).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import torch
import torch.nn as nn

import simvq_split.chunker as _chunker_mod
import simvq_split.crystal as _crystal_mod
from kernels.fast_calcite_scans import (
    configure_fast_ternary_qat,
    fast_causal_refractory_gate,
    fast_crystal_forward,
    fast_diag_birkhoff_crystal_scan,
    fast_project_birkhoff_to_perm,
    fast_utf8_chunk_span,
    fast_woodbury_forward,
)
from kernels.fast_calcite_swa_chunker import (
    patch_fast_chunker,
    patch_fast_swa_attention,
)
from simvq_split.crystal import VoronoiCrystalLayer
from simvq_split.memory import compute_chunker_state_bytes
from simvq_split.models import SimVQSplitConfig, SimVQSplitLM
from simvq_split.woodbury import NeuralWoodburyLayer

_ORIG_CRYSTAL_SCAN = _crystal_mod.diag_birkhoff_crystal_scan


def _amp_shielded_crystal_scan(
    s0: torch.Tensor,
    kw_eff_seq: torch.Tensor,
    kw_vg_seq: torch.Tensor,
    v_ste_seq: torch.Tensor,
    P_eff_seq: torch.Tensor,
    is_hard: bool,
    scale_r: float,
    Bc: int = 32,
) -> torch.Tensor:
    with torch.amp.autocast(s0.device.type, enabled=False):
        return _ORIG_CRYSTAL_SCAN(
            s0.float(),
            kw_eff_seq.float(),
            kw_vg_seq.float(),
            v_ste_seq.float(),
            P_eff_seq.float(),
            is_hard,
            scale_r,
            Bc=Bc,
        )


@dataclass
class LeanSystem1Config(SimVQSplitConfig):
    crystal_w_chunks: int = 2
    woodbury_latch_span: int = 3


class LeanVoronoiCrystalLayer(VoronoiCrystalLayer):
    """`VoronoiCrystalLayer` with AMP `Float32` checkpoint shield and `W_chunks` pruning."""

    @staticmethod
    def _configure_w_chunks(layer: VoronoiCrystalLayer, w_chunks: int) -> None:
        w_chunks_i = int(max(1, w_chunks))
        if w_chunks_i != layer.W_chunks:
            layer.W_chunks = w_chunks_i
            w_len = layer.W_chunks * layer.C
            p_a = (
                torch.linspace(0.6, -0.6, steps=w_len)
                .view(w_len, 1, 1)
                .expand(w_len, layer.H, 1)
            )
            p_b = (
                torch.linspace(-0.6, 0.6, steps=w_len)
                .view(w_len, 1, 1)
                .expand(w_len, layer.H, 1)
            )
            layer.ptr_pos_bias = nn.Parameter(torch.cat((p_a, p_b), dim=-1).clone())

    @classmethod
    def adopt(cls, base: VoronoiCrystalLayer, w_chunks: int) -> LeanVoronoiCrystalLayer:
        base.__class__ = cls
        cls._configure_w_chunks(base, w_chunks)
        return base  # type: ignore[return-value]

    def __init__(
        self,
        d_model: int,
        num_spheres: int = 24,
        d_sphere: int = 32,
        codebook_size: int = 32,
        w_chunks: int = 2,
    ):
        super().__init__(
            d_model=d_model,
            num_spheres=num_spheres,
            d_sphere=d_sphere,
            codebook_size=codebook_size,
        )
        self._configure_w_chunks(self, w_chunks)

    def forward_states(
        self, x: torch.Tensor, h_lex: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, int, int, int]:
        use_fast = bool(getattr(self, "use_fast_kernels", True))
        prev_scan = _crystal_mod.diag_birkhoff_crystal_scan
        prev_ref = _crystal_mod.causal_refractory_gate_jit
        prev_proj = _crystal_mod.project_birkhoff_to_perm_jit
        if use_fast:
            _crystal_mod.diag_birkhoff_crystal_scan = fast_diag_birkhoff_crystal_scan
            _crystal_mod.causal_refractory_gate_jit = fast_causal_refractory_gate
            _crystal_mod.project_birkhoff_to_perm_jit = fast_project_birkhoff_to_perm
        else:
            _crystal_mod.diag_birkhoff_crystal_scan = _amp_shielded_crystal_scan
        try:
            return super().forward_states(x, h_lex=h_lex)
        finally:
            _crystal_mod.diag_birkhoff_crystal_scan = prev_scan
            _crystal_mod.causal_refractory_gate_jit = prev_ref
            _crystal_mod.project_birkhoff_to_perm_jit = prev_proj

    def forward(
        self, x: torch.Tensor, h_lex: torch.Tensor | None = None
    ) -> torch.Tensor:
        if bool(getattr(self, "use_fast_kernels", True)):
            return fast_crystal_forward(self, x, h_lex=h_lex)
        return super().forward(x, h_lex=h_lex)


class LeanNeuralWoodburyLayer(NeuralWoodburyLayer):
    """`NeuralWoodburyLayer` with `Via Negativa`-pruned subject latch span (`L_latch` tokens)."""

    @staticmethod
    def _configure_latch_span(layer: NeuralWoodburyLayer, latch_span: int) -> None:
        l_span = int(max(1, latch_span))
        if l_span != layer.L_latch:
            layer.L_latch = l_span
            if l_span == 3:
                init_lag = torch.tensor([-0.4, 1.4, -0.8], dtype=torch.float32)
            elif l_span == 4:
                init_lag = torch.tensor([-0.6, 1.5, 0.2, -1.0], dtype=torch.float32)
            elif l_span in (5, 6):
                base_lag = torch.tensor([-1.5, -0.5, 0.2, 1.9, -0.5, -1.5, -2.0], dtype=torch.float32)
                init_lag = base_lag[-l_span:].clone()
            else:
                pos = torch.linspace(-1.0, 1.0, steps=l_span, dtype=torch.float32)
                init_lag = 1.6 * torch.exp(-pos * pos * 2.0) - 0.6
            layer.subj_lag_bias = nn.Parameter(
                init_lag.view(1, l_span).repeat(layer.H_epi, 1).clone()
            )

    @classmethod
    def adopt(cls, base: NeuralWoodburyLayer, latch_span: int) -> LeanNeuralWoodburyLayer:
        base.__class__ = cls
        cls._configure_latch_span(base, latch_span)
        return base  # type: ignore[return-value]

    def __init__(
        self,
        d_model: int,
        mem_heads: int = 5,
        mem_dim: int = 32,
        ridge_lambda: float = 0.04,
        latch_span: int = 3,
    ):
        super().__init__(
            d_model=d_model,
            mem_heads=mem_heads,
            mem_dim=mem_dim,
            ridge_lambda=ridge_lambda,
            latch_span=7,
        )
        self._configure_latch_span(self, latch_span)

    def forward(
        self, x: torch.Tensor, h_lex: torch.Tensor | None = None
    ) -> torch.Tensor:
        if bool(getattr(self, "use_fast_kernels", True)):
            return fast_woodbury_forward(self, x, h_lex=h_lex)
        return super().forward(x, h_lex=h_lex)


class CalciteLeanSystem1LM(SimVQSplitLM):
    """
    Pure `Calcite System-1` model (`vocab_size = 259` raw UTF-8) supporting both legacy
    (`W=12, W_chunks=5, L_latch=7`) and `Via Negativa` pruned ring buffers.
    """

    def __init__(self, cfg: LeanSystem1Config):
        super().__init__(cfg)
        self.lean_cfg = cfg
        self.use_fast_kernels: bool = True
        if self.crystal is not None:
            self.crystal = LeanVoronoiCrystalLayer.adopt(self.crystal, cfg.crystal_w_chunks)
        if self.crystal_2 is not None:
            self.crystal_2 = LeanVoronoiCrystalLayer.adopt(self.crystal_2, cfg.crystal_w_chunks)
        if self.woodbury is not None:
            self.woodbury = LeanNeuralWoodburyLayer.adopt(self.woodbury, cfg.woodbury_latch_span)
        self.set_fast_kernels(True)

    def set_fast_kernels(self, enabled: bool) -> None:
        flag = bool(enabled)
        self.use_fast_kernels = flag
        if self.crystal is not None:
            self.crystal.use_fast_kernels = flag
        if self.crystal_2 is not None:
            self.crystal_2.use_fast_kernels = flag
        if self.woodbury is not None:
            self.woodbury.use_fast_kernels = flag
        patch_fast_swa_attention(self, enabled=flag)
        patch_fast_chunker(self, enabled=flag)
        if self.ternary_qat_enabled:
            self.set_ternary_qat(True)

    def set_ternary_qat(self, enabled: bool) -> None:
        flag = bool(enabled)
        self.ternary_qat_enabled = flag
        if self.use_fast_kernels:
            configure_fast_ternary_qat(self, flag)
            act = float(self.qat_mix) if flag else 0.0
            if self.woodbury is not None:
                self.woodbury.qat_active_mix = act
            if self.crystal is not None:
                self.crystal.qat_active_mix = act
            if self.crystal_2 is not None:
                self.crystal_2.qat_active_mix = act
        else:
            configure_fast_ternary_qat(self, False)
            super().set_ternary_qat(flag)

    def forward(
        self,
        token_ids: torch.Tensor,
        ablate_crystal: bool = False,
        ablate_woodbury: bool = False,
    ) -> torch.Tensor:
        if not self.use_fast_kernels or self.byte_chunker is None:
            return super().forward(
                token_ids,
                ablate_crystal=ablate_crystal,
                ablate_woodbury=ablate_woodbury,
            )
        prev_utf8 = _chunker_mod._enforce_utf8_chunk_span_jit
        _chunker_mod._enforce_utf8_chunk_span_jit = fast_utf8_chunk_span
        try:
            return super().forward(
                token_ids,
                ablate_crystal=ablate_crystal,
                ablate_woodbury=ablate_woodbury,
            )
        finally:
            _chunker_mod._enforce_utf8_chunk_span_jit = prev_utf8


def compute_lean_state_cache_kb(cfg: LeanSystem1Config, seq_len: int = 2048) -> float:
    """
    Exact strictly `O(1)` streaming KV + Crystal + Woodbury + Chunker state footprint in KB.
    Matches `compute_state_cache_kb` bit-for-bit when `crystal_w_chunks=5` and
    `woodbury_latch_span=7`, and shrinks proportionally under `Via Negativa` pruning.
    """
    del seq_len  # State is invariant across arbitrary L
    n_loc = cfg.n_layers_pre + cfg.n_layers_post
    local_kv = n_loc * 2 * cfg.window_size * cfg.d_model * 2

    if cfg.use_crystal:
        h_cryst = 2 if (cfg.num_spheres >= 4 and cfg.num_spheres % 2 == 0) else 1
        cryst_lag = max(0, 2 * int(cfg.crystal_w_chunks) - 1)
        cryst_stage = (
            cfg.num_spheres * cfg.d_sphere * 4
            + (cfg.d_model + 1 + 2 * h_cryst) * 4
            + 4 * h_cryst * 2
            + cryst_lag * (cfg.d_model + h_cryst * cfg.d_sphere) * 2
            + (3 + 1) * cfg.d_model * 2
        )
        cryst_b = 2 * cryst_stage
    else:
        cryst_b = 0

    if cfg.use_woodbury:
        h_epi = max(1, cfg.mem_heads - 1)
        wood_mat = cfg.mem_heads * 2 * cfg.mem_dim * cfg.mem_dim * 4
        w_latch_lag = max(0, int(cfg.woodbury_latch_span) - 1)
        wood_ring = (
            (2 * cfg.d_model + 1) * 4
            + (3 * cfg.d_model * 2)
            + (w_latch_lag * (2 * h_epi * cfg.mem_dim + h_epi) * 2)
            + (3 * (cfg.d_model + (2 * cfg.mem_heads + h_epi) * cfg.mem_dim) * 2)
        )
        wood_b = wood_mat + wood_ring
    else:
        wood_b = 0

    chunker_b = (
        compute_chunker_state_bytes(
            cfg.d_model,
            cfg.byte_char_dim,
            cfg.byte_router_char_dim,
            byte_chunk_mode=cfg.byte_chunk_mode,
        )
        if cfg.use_byte_chunker
        else 0
    )
    return (local_kv + cryst_b + wood_b + chunker_b) / 1024.0


def build_tier_lean_config(
    tier_name: str,
    window_size: int = 8,
    crystal_w_chunks: int = 2,
    woodbury_latch_span: int = 3,
) -> LeanSystem1Config:
    """Returns canonical Phase K `CalciteLeanSystem1LM` config for `S1_5M`, `S2_25M`, or `S3_85M`."""
    specs = {
        "S1_5M": (256, 4, 2, 2, 5, 32),
        "S2_25M": (448, 7, 4, 4, 7, 32),
        "S3_85M": (768, 12, 5, 5, 8, 48),
    }
    if tier_name not in specs:
        raise ValueError(f"Unknown tier `{tier_name}`: expected one of {tuple(specs.keys())}")
    d_model, n_heads, n_pre, n_post, mem_heads, mem_dim = specs[tier_name]
    return LeanSystem1Config(
        vocab_size=259,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=None,
        n_layers_pre=n_pre,
        n_layers_post=n_post,
        window_size=int(window_size),
        num_spheres=24,
        d_sphere=32,
        codebook_size=32,
        mem_heads=mem_heads,
        mem_dim=mem_dim,
        ridge_lambda=0.04,
        use_crystal=True,
        use_woodbury=True,
        use_byte_chunker=True,
        byte_char_dim=32,
        byte_d_ff=128,
        byte_router_char_dim=16,
        byte_router_hidden=96,
        byte_commit_scale=0.45,
        router_lock_frac=0.135,
        target_chunk_bytes=4.6,
        max_chunk_bytes=10,
        byte_chunk_mode="utf8",
        max_utf8_bytes=16,
        use_static_m_cap=True,
        ratio_loss_scale=1.5,
        pad_token_id=0,
        crystal_w_chunks=int(crystal_w_chunks),
        woodbury_latch_span=int(woodbury_latch_span),
    )


def get_scaling_tier_specs(
    window_size: int = 8,
    crystal_w_chunks: int = 2,
    woodbury_latch_span: int = 3,
) -> dict[str, dict[str, object]]:
    """Returns serializable summary specifications across `S1_5M`, `S2_25M`, and `S3_85M`."""
    out: dict[str, dict[str, object]] = {}
    for tier in ("S1_5M", "S2_25M", "S3_85M"):
        cfg = build_tier_lean_config(
            tier,
            window_size=window_size,
            crystal_w_chunks=crystal_w_chunks,
            woodbury_latch_span=woodbury_latch_span,
        )
        m = CalciteLeanSystem1LM(cfg)
        n_params = int(sum(p.numel() for p in m.parameters()))
        cfg_leg = build_tier_lean_config(tier, window_size=12, crystal_w_chunks=5, woodbury_latch_span=7)
        cache_lean = compute_lean_state_cache_kb(cfg, 2048)
        cache_leg = compute_lean_state_cache_kb(cfg_leg, 2048)
        out[tier] = {
            "config": asdict(cfg),
            "params": n_params,
            "cache_kb_L2048": round(cache_lean, 2),
            "legacy_cache_kb_L2048": round(cache_leg, 2),
            "cache_reduction_pct": round(100.0 * (1.0 - cache_lean / cache_leg), 2),
        }
    return out
