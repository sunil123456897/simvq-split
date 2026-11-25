"""
Fused Autocast-Preserving RMSNorm and O(T * W) Block-Strided Sliding-Window Attention
helpers for `calcite_v2/` (`calcite_v2/norm.py`).

- `CalciteRMSNorm` subclasses `simvq_split.baselines.RMSNorm` so every downstream
  `isinstance(mod, RMSNorm)` parameter splitter (`gain_q`) and checkpoint `state_dict`
  remains 100% compatible while keeping `simvq_split/baselines.py` at 0-byte diff.
- Calls native fused `F.rms_norm(...)` and preserves the active autocast dtype (`bf16`/`fp16`)
  so all 38 pre-norms avoid promoting activations back to `fp32` (eliminating 79/107 FP32
  QAT cache bloat and halving post-norm VRAM/HBM bandwidth).
- `build_swa_block_masks` + `sliding_window_sdpa` implement exact strictly causal
  `O(T * (Bq + W))` sliding-window attention using at most 2 `F.scaled_dot_product_attention`
  calls (zero Python loop over blocks, zero `W`-replicated FP32 unfold materialization).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from simvq_split.baselines import RMSNorm


class CalciteRMSNorm(RMSNorm):
    """Fused `F.rms_norm` subclass of `RMSNorm` with autocast-dtype preservation."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__(dim=dim, eps=eps)
        self.dim = int(dim)
        self._norm_shape = (int(dim),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dev_type = x.device.type
        if dev_type in ("cuda", "cpu") and torch.is_autocast_enabled(dev_type):
            dt = torch.get_autocast_dtype(dev_type)
        else:
            dt = x.dtype

        if x.dtype == dt:
            w = self.weight if self.weight.dtype == dt else self.weight.to(dt)
            return F.rms_norm(x, self._norm_shape, w, self.eps)

        # Master-residual input (e.g. FP32) under BF16/FP16 autocast:
        # compute exact RMS in x.dtype and emit autocast dtype for downstream Linear/Conv.
        w = self.weight if self.weight.dtype == x.dtype else self.weight.to(x.dtype)
        return F.rms_norm(x, self._norm_shape, w, self.eps).to(dt)


def build_swa_block_masks(
    window_size: int,
    block_q: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Precomputes static boolean attention masks for zero-loop block-strided SDPA:
    - `mask0`: `(1, 1, Bq, Bq)` causal band mask for block 0 (`0 <= r - c < W`)
    - `mask_inner`: `(1, 1, Bq, Bq + W - 1)` exact band mask for blocks `bi >= 1`
      (`r <= c <= r + W - 1`, where every query row has exactly `W` valid keys).
    """
    W = max(1, int(window_size))
    Bq = max(64, W) if block_q is None else max(int(block_q), W)
    r = torch.arange(Bq, dtype=torch.long).unsqueeze(1)
    c0 = torch.arange(Bq, dtype=torch.long).unsqueeze(0)
    mask0 = ((r >= c0) & ((r - c0) < W)).view(1, 1, Bq, Bq)

    k_len = Bq + W - 1
    ci = torch.arange(k_len, dtype=torch.long).unsqueeze(0)
    mask_inner = ((ci >= r) & (ci <= (r + W - 1))).view(1, 1, Bq, k_len)
    return mask0, mask_inner, Bq


def sliding_window_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    mask0: torch.Tensor,
    mask_inner: torch.Tensor,
    block_q: int,
) -> torch.Tensor:
    """
    Exact strictly causal `O(T * (Bq + W))` sliding-window attention using native
    `F.scaled_dot_product_attention` (at most 2 SDPA calls for any `T`, zero loops).

    Args:
        q, k, v: `(B, H, T, D)` RoPE-rotated tensors.
        window_size: Causal sliding window span `W >= 1`.
        mask0: `(1, 1, Bq, Bq)` bool causal band mask.
        mask_inner: `(1, 1, Bq, Bq + W - 1)` bool inner-block band mask.
        block_q: Block length `Bq >= W`.
    """
    B, H, T, D = q.shape
    W = int(window_size)
    if T <= W:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    Bq = int(block_q)
    if T <= Bq:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask0[:, :, :T, :T])

    # Block 0 (`t = 0 .. Bq - 1`) attends causally within `[0, Bq)`
    y0 = F.scaled_dot_product_attention(
        q[:, :, :Bq, :],
        k[:, :, :Bq, :],
        v[:, :, :Bq, :],
        attn_mask=mask0,
    )

    # Blocks `1 .. n_blk - 1` batched into ONE vectorized SDPA call
    n_blk = (T + Bq - 1) // Bq
    n_rest = n_blk - 1
    T_pad = n_blk * Bq
    pad_r = T_pad - T
    if pad_r > 0:
        q_p = F.pad(q, (0, 0, 0, pad_r))
        k_p = F.pad(k, (0, 0, 0, pad_r))
        v_p = F.pad(v, (0, 0, 0, pad_r))
    else:
        q_p, k_p, v_p = q, k, v

    k_len = Bq + W - 1
    q_rest = (
        q_p[:, :, Bq:, :]
        .view(B, H, n_rest, Bq, D)
        .transpose(1, 2)
        .reshape(B * n_rest, H, Bq, D)
    )
    k_src = k_p[:, :, Bq - (W - 1) :, :]
    v_src = v_p[:, :, Bq - (W - 1) :, :]
    k_rest = (
        k_src.unfold(2, k_len, Bq)
        .permute(0, 2, 1, 4, 3)
        .reshape(B * n_rest, H, k_len, D)
    )
    v_rest = (
        v_src.unfold(2, k_len, Bq)
        .permute(0, 2, 1, 4, 3)
        .reshape(B * n_rest, H, k_len, D)
    )
    y_rest = (
        F.scaled_dot_product_attention(q_rest, k_rest, v_rest, attn_mask=mask_inner)
        .view(B, n_rest, H, Bq, D)
        .transpose(1, 2)
        .reshape(B, H, n_rest * Bq, D)
    )
    y_all = torch.cat((y0, y_rest), dim=2)
    return y_all[:, :, :T, :] if pad_r > 0 else y_all


__all__ = [
    "CalciteRMSNorm",
    "build_swa_block_masks",
    "sliding_window_sdpa",
]
