"""
Calcite v2 (`calcite_v2`) — Clean, Zero-Monkey-Patch Post-Transformer Architecture.

Upgrades `Calcite-5 System-1` along four cleanly ablatable axes while preserving
strict `O(1)` streaming state, `0.00%` mid-UTF8 boundary splits, native `2-bit uint8 T-MAC LUT`,
and `Zero-Cheat` pure-neural AST invariants:
1. Scaled intra-chunk byte decoder with explicit causal byte-offset embeddings + pre-frozen UTF-8 router.
2. Interleaved multi-timescale stack (fast/slow Woodbury + Crystal, removing the single middle bottleneck
   and legacy `-1.8` gate brakes).
3. True 2nd-order biorthogonal Kalman-Woodbury write preconditioning using stop-grad SPD `P_{t-1}`
   plus multi-band retention horizons for `L = 32K–64K` contexts.
4. Optional 1-chunk causal Latent Feedback (zero extra KV cache, `+1` vector `d_model` in `O(1)` stream).
"""

from .chunker import CalciteV2Chunker, ChunkContext, ChunkerStreamState
from .crystal import CalciteV2CrystalLayer, CrystalStreamState
from .model import (
    CalciteV2Config,
    CalciteV2LM,
    LocalSWABlock,
    build_calcite_v2_tier_config,
    compute_v2_state_cache_kb,
)
from .stream_engine import CalciteStreamEngine, CalciteStreamState
from .woodbury import CalciteV2WoodburyLayer, WoodburyStreamState

__all__ = [
    "CalciteV2Chunker",
    "ChunkContext",
    "ChunkerStreamState",
    "CalciteV2CrystalLayer",
    "CrystalStreamState",
    "CalciteV2WoodburyLayer",
    "WoodburyStreamState",
    "CalciteV2Config",
    "CalciteV2LM",
    "LocalSWABlock",
    "build_calcite_v2_tier_config",
    "compute_v2_state_cache_kb",
    "CalciteStreamEngine",
    "CalciteStreamState",
]
