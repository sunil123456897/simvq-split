"""
Post-Transformer State & Memory Primitives for SimVQ-Split v5 (`Calcite-5 System-1 Core`):
Re-export facade for modularized submodules:
1. `simvq_split.crystal` (`VoronoiCrystalLayer` & Birkhoff-Sinkhorn TorchScript operators)
2. `simvq_split.woodbury` (`NeuralWoodburyLayer`, TorchScript scans & SPD diagnostics)
3. `simvq_split.quant_lut` (`quantize_ternary_weights_ls`, `quantize_ternary_ste`, `PackedLUT2BitLinear`)
"""

from __future__ import annotations

from .baselines import RMSNorm
from .crystal import (
    VoronoiCrystalLayer,
    _sphere_norm_jit,
    causal_refractory_gate_jit,
    diag_birkhoff_crystal_scan_jit,
    log_sinkhorn_6,
    project_birkhoff_to_perm_jit,
)
from .quant_lut import (
    PackedLUT2BitLinear,
    compute_2bit_lut_footprint,
    pack_model_2bit_lut,
    quantize_ternary_ste,
    quantize_ternary_weights_ls,
)
from .woodbury import (
    NeuralWoodburyLayer,
    _neural_woodbury_chunk_scan_fp_jit,
    _neural_woodbury_chunk_scan_tern_jit,
    inspect_woodbury_spd_spectrum,
    neural_woodbury_chunk_scan_jit,
)

__all__ = [
    "RMSNorm",
    "quantize_ternary_weights_ls",
    "quantize_ternary_ste",
    "PackedLUT2BitLinear",
    "pack_model_2bit_lut",
    "compute_2bit_lut_footprint",
    "_sphere_norm_jit",
    "project_birkhoff_to_perm_jit",
    "log_sinkhorn_6",
    "causal_refractory_gate_jit",
    "diag_birkhoff_crystal_scan_jit",
    "_neural_woodbury_chunk_scan_fp_jit",
    "_neural_woodbury_chunk_scan_tern_jit",
    "neural_woodbury_chunk_scan_jit",
    "inspect_woodbury_spd_spectrum",
    "VoronoiCrystalLayer",
    "NeuralWoodburyLayer",
]
