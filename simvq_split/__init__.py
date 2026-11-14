"""
SimVQ-Split: Clean Post-Transformer Architecture Package.

Splits hidden representation into:
1. Continuous Cortex (Local Sliding-Window RoPE Attention + SwiGLU)
2. Voronoi Crystal (Factored Spherical Attractor Product with Householder Rotation-Trick STE)
3. Neural Woodbury Memory (Self-addressed Second-Order Dual-Key Fast Memory)

Zero host heuristics, zero Python session dicts, 100% raw token_ids end-to-end.
"""

from .baselines import (
    BaselineConfig,
    HouseholderDeltaNetLM,
    LoopedTransformerLM,
    RoPETransformerLM,
    SlidingWindowTransformerLM,
    match_baseline_configs,
)
from .models import SimVQSplitConfig, SimVQSplitLM
from .tokenizer import ByteBPETokenizer

__all__ = [
    "ByteBPETokenizer",
    "SimVQSplitConfig",
    "SimVQSplitLM",
    "BaselineConfig",
    "RoPETransformerLM",
    "SlidingWindowTransformerLM",
    "LoopedTransformerLM",
    "HouseholderDeltaNetLM",
    "match_baseline_configs",
]
