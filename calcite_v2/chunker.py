"""
Calcite v2 Surprise-Gated UTF-8 Byte Chunker (`calcite_v2/chunker.py`).

Improvements over `simvq_split/chunker.py`:
1. Zero monkey-patching: calls `utf8_span_compact_and_offset_cuda` / `utf8_span_and_compact_cuda`
   directly when on CUDA (with O(T) zero-`argsort` PyTorch fallback on CPU).
2. Scaled intra-chunk byte decoder (`use_scaled_decoder=True`):
   - Explicit causal `byte_offset_emb` (`0..max_utf8_bytes`).
   - Coalesced `(B, T, D)` 4-tap shift-add depthwise mixers (`k=4, d=1,2,4`) avoiding `transpose+Conv1d`.
   - Micro Byte-SWA (`W=24, D=32`) wired to native CUDA `fast_swa_rope_attn_cuda` + cached RoPE
     and zero-loop `O(T * (Bq + W))` block-strided SDPA fallback.
3. Tight exact `M_max` during eval (`not self.training`), frozen-router `precomputed_router` reuse
   for zero-duplication QAT, dead `ste_c` skip when `router_locked`, and shared `t_pad6` + bitwise
   power-of-2 hashing across `ByteEngram` and `ByteInductionCache`.
4. Exact incremental streaming API (`init_stream_state`, `encode_byte_step`, `compute_lex_from_win8`,
   `decode_byte_step`) with circular ring pointers (zero `.clone()`) and lazy `compute_lex=False`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import kernels.fast_calcite_swa_chunker as _swa_chunker_mod
from kernels.fast_calcite_swa_chunker import (
    _compact_b_mask_zero_argsort_pytorch,
    fast_swa_rope_attn_cuda,
    get_cached_rope_cos_sin,
    utf8_span_and_compact_cuda,
)
from simvq_split.baselines import SwiGLU, apply_rope
from simvq_split.chunker import (
    _enforce_utf8_chunk_span_jit,
    _utf8_tensor_flags,
    enforce_max_chunk_span_vec,
)
from simvq_split.quant_lut import quantize_ternary_ste, quantize_ternary_weights_ls

from .byte_induct_cache import ByteInductionCache, ByteInductionStreamState
from .norm import CalciteRMSNorm, build_swa_block_masks, sliding_window_sdpa


def _swa_cuda_ext_ready() -> bool:
    loader = getattr(_swa_chunker_mod, "_load_swa_chunker_cuda_ext", None)
    return callable(loader) and (loader() is not None)


def resolve_dw_dim(d_model: int, dw_bottleneck_ratio: float = 1.0) -> int:
    """Resolves inner width `d_dw` for full-byte gated depthwise mixers (`1.0` = full `d_model`)."""
    r = float(dw_bottleneck_ratio)
    if r >= 0.999:
        return int(d_model)
    if r <= 0.0:
        raise ValueError(f"dw_bottleneck_ratio must be > 0, got {dw_bottleneck_ratio}")
    raw = (int(round(float(d_model) * r)) // 16) * 16
    return int(min(int(d_model), max(32, raw)))


def _depthwise_conv4_shift_add(
    x: torch.Tensor, conv: nn.Conv1d, dilation: int
) -> torch.Tensor:
    """
    Exact coalesced `(B, T, D)` 4-tap causal depthwise conv matching
    `conv(F.pad(x.transpose(1, 2), (3 * d, 0))).transpose(1, 2)` to `< 1e-6`
    without non-contiguous transposes or cuDNN depthwise 1D launches.
    """
    T = x.size(1)
    d = int(dilation)
    xp = F.pad(x, (0, 0, 3 * d, 0))
    w = conv.weight.squeeze(1)
    if w.dtype != x.dtype:
        w = w.to(x.dtype)
    out = (
        xp[:, 0:T] * w[:, 0]
        + xp[:, d : d + T] * w[:, 1]
        + xp[:, 2 * d : 2 * d + T] * w[:, 2]
        + xp[:, 3 * d : 3 * d + T] * w[:, 3]
    )
    if conv.bias is not None:
        b = conv.bias if conv.bias.dtype == x.dtype else conv.bias.to(x.dtype)
        out = out + b
    return out


def _dw_conv4_step_circular(
    x_t: torch.Tensor,
    ring: torch.Tensor,
    ptr: int,
    conv: nn.Conv1d,
    dilation: int,
) -> tuple[torch.Tensor, int]:
    """Exact 1-byte circular-ring 4-tap depthwise conv step (zero `torch.cat`/`.clone()`)."""
    ring_len = 3 * int(dilation)
    d = int(dilation)
    dt = x_t.dtype
    w = conv.weight.squeeze(1)
    if w.dtype != dt:
        w = w.to(dt)
    c = (
        ring[:, ptr] * w[:, 0]
        + ring[:, (ptr + d) % ring_len] * w[:, 1]
        + ring[:, (ptr + 2 * d) % ring_len] * w[:, 2]
        + x_t * w[:, 3]
    )
    if conv.bias is not None:
        b = conv.bias if conv.bias.dtype == dt else conv.bias.to(dt)
        c = c + b
    ring[:, ptr] = x_t
    return c, (ptr + 1) % ring_len


@dataclass
class ChunkContext:
    b_mask: torch.Tensor
    valid_mask: torch.Tensor
    p_prob: torch.Tensor
    p_chunk: torch.Tensor
    h_byte: torch.Tensor
    c_win8: torch.Tensor
    slot_idx: torch.Tensor
    offset_in_chunk: torch.Tensor
    valid_chunk: torch.Tensor
    token_ids: torch.Tensor
    chunk_counts: torch.Tensor | None = None
    gather_pos: torch.Tensor | None = None


@dataclass
class ChunkerStreamState:
    ring_char: torch.Tensor        # (B, 7, d_byte)
    ring_rchar: torch.Tensor       # (B, 3, d_rchar)
    prev_k_r: torch.Tensor         # (B, router_dim)
    prev_h_ent: torch.Tensor       # (B,)
    prev_z_hat: torch.Tensor       # (B, d_model)
    cur_z_bar: torch.Tensor        # (B, d_model)
    ring_dec_conv1: torch.Tensor   # (B, 3, d_dw) if scaled decoder else (B, 0, 1)
    ring_dec_conv2: torch.Tensor   # (B, 6, d_dw) if scaled decoder else (B, 0, 1)
    ring_enc_conv2: torch.Tensor   # (B, 6, d_dw) if sym_byte_swa else (B, 0, 1)
    ring_dec_conv3: torch.Tensor   # (B, 12, d_dw) if (sym_byte_swa and dec_dw3_enabled) else (B, 0, 1)
    ring_byte_k: torch.Tensor      # (B, W_byte, byte_swa_heads, d_b_hd) if sym_byte_swa else (B, 0, 1, 1)
    ring_byte_v: torch.Tensor      # (B, W_byte, byte_swa_heads, d_b_hd) if sym_byte_swa else (B, 0, 1, 1)
    ring_tok3: torch.Tensor        # (B, 4 if engram_use_4gram else 3) int64 if byte_engram else (B, 0)
    run_len: torch.Tensor          # (B,) int64
    cp_run: torch.Tensor           # (B,) int64
    cont_streak: torch.Tensor      # (B,) int64
    credits: torch.Tensor          # (B,) int64
    step_idx: int
    has_chunk: bool
    last_tok: torch.Tensor | None = None
    byte_induct: ByteInductionStreamState | None = None
    # Circular ring pointers (chronological order when == 0)
    char_ptr: int = 0
    rchar_ptr: int = 0
    tok3_ptr: int = 0
    enc2_ptr: int = 0
    dec1_ptr: int = 0
    dec2_ptr: int = 0
    dec3_ptr: int = 0

    def memory_bytes(self) -> int:
        b = max(1, int(self.ring_char.size(0)))
        total = 0
        for t in (
            self.ring_char,
            self.ring_rchar,
            self.prev_k_r,
            self.prev_h_ent,
            self.prev_z_hat,
            self.cur_z_bar,
            self.ring_dec_conv1,
            self.ring_dec_conv2,
            self.ring_enc_conv2,
            self.ring_dec_conv3,
            self.ring_byte_k,
            self.ring_byte_v,
            self.ring_tok3,
            self.run_len,
            self.cp_run,
            self.cont_streak,
            self.credits,
        ):
            total += int(t.numel() * t.element_size())
        out = total // b
        if self.byte_induct is not None:
            out += self.byte_induct.memory_bytes()
        return out


class CalciteV2Chunker(nn.Module):
    def __init__(
        self,
        vocab_size: int = 259,
        d_model: int = 256,
        d_byte: int = 32,
        d_rchar: int = 16,
        r_hidden: int = 96,
        byte_d_ff: int = 256,
        target_chunk_bytes: float = 4.6,
        max_chunk_bytes: int = 10,
        pad_token_id: int = 0,
        byte_chunk_mode: str = "utf8",
        max_utf8_bytes: int = 16,
        use_static_m_cap: bool = True,
        use_scaled_decoder: bool = True,
        use_sym_byte_swa: bool = False,
        byte_swa_window: int = 24,
        byte_swa_dim: int = 64,
        use_byte_engram: bool = False,
        engram_slots: int = 4096,
        engram_dim: int = 32,
        engram_use_4gram: bool = False,
        use_byte_induct_cache: bool = False,
        byte_induct_slots: int = 1024,
        byte_induct_dim: int = 16,
        dec_dw3_enabled: bool = True,
        dw_bottleneck_ratio: float = 1.0,
        use_fast_kernels: bool = True,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.d_byte = int(min(d_byte, d_model))
        self.d_rchar = int(d_rchar)
        self.r_hidden = int(r_hidden)
        self.router_dim = 32
        self.target_chunk_bytes = float(target_chunk_bytes)
        self.max_chunk_bytes = int(max_chunk_bytes)
        self.pad_token_id = int(pad_token_id)
        self.byte_chunk_mode = str(byte_chunk_mode)
        self.max_utf8_bytes = int(max(4, max_utf8_bytes))
        self.use_static_m_cap = bool(use_static_m_cap)
        self.use_scaled_decoder = bool(use_scaled_decoder)
        self.use_sym_byte_swa = bool(use_sym_byte_swa)
        self.byte_swa_window = int(byte_swa_window)
        self.byte_swa_dim = int(byte_swa_dim)
        self.byte_swa_heads = (
            max(1, self.byte_swa_dim // 32) if (self.byte_swa_dim % 32 == 0) else 2
        )
        self.byte_swa_head_dim = self.byte_swa_dim // self.byte_swa_heads
        self.use_byte_engram = bool(use_byte_engram)
        self.engram_slots = int(engram_slots)
        self.engram_dim = int(engram_dim)
        self.engram_use_4gram = bool(engram_use_4gram)
        self.engram_ring_len = (
            (4 if self.engram_use_4gram else 3) if self.use_byte_engram else 0
        )
        self.use_byte_induct_cache = bool(use_byte_induct_cache)
        self.byte_induct_slots = int(byte_induct_slots)
        self.byte_induct_dim = int(byte_induct_dim)
        self.dec_dw3_enabled = bool(dec_dw3_enabled)
        self.dw_bottleneck_ratio = float(dw_bottleneck_ratio)
        self.d_dw = resolve_dw_dim(self.d_model, self.dw_bottleneck_ratio)
        self.use_dw_bottleneck = self.d_dw < self.d_model
        self.use_fast_kernels = bool(use_fast_kernels)

        self.refill_int = 100
        self.cost_int = max(100, int(round(100.0 * 0.72 * self.target_chunk_bytes)))
        self.burst_int = 4 * self.cost_int

        # 8-byte causal encoder & lexical pathway
        self.enc_norm_in = CalciteRMSNorm(8 * self.d_byte)
        self.enc_up = nn.Linear(8 * self.d_byte, d_model, bias=False)
        self.enc_norm = CalciteRMSNorm(d_model)
        self.enc_mlp = SwiGLU(d_model, byte_d_ff)
        self.lex_norm = CalciteRMSNorm(8 * self.d_byte)
        self.lex_up = nn.Linear(8 * self.d_byte, d_model, bias=False)

        # Decoupled ternary-stable UTF-8 boundary router
        self.router_emb = nn.Embedding(self.vocab_size, self.d_rchar)
        nn.init.normal_(self.router_emb.weight, std=0.05)
        self.router_in_norm = CalciteRMSNorm(4 * self.d_rchar)
        self.router_in_proj = nn.Linear(4 * self.d_rchar, self.r_hidden, bias=True)
        self.router_q = nn.Linear(self.r_hidden, self.router_dim, bias=False)
        self.router_k = nn.Linear(self.r_hidden, self.router_dim, bias=False)
        self.router_ctx = nn.Linear(self.r_hidden, 1, bias=True)
        self.router_ent_norm = CalciteRMSNorm(self.r_hidden)
        self.router_ent_head = nn.Linear(self.r_hidden, self.vocab_size, bias=False)

        init_b = -math.log(max(1.0, self.target_chunk_bytes - 1.0)) - 0.65
        nn.init.constant_(self.router_ctx.bias, init_b)
        self.router_feat_weights = nn.Parameter(
            torch.tensor([1.15, 0.45, 0.75], dtype=torch.float32)
        )
        self.router_locked: bool = False

        # Byte decoder pathway
        self.dechunk_norm = CalciteRMSNorm(d_model)
        d_dw = self.d_dw
        if self.use_scaled_decoder:
            self.byte_offset_emb = nn.Embedding(self.max_utf8_bytes + 1, d_model)
            nn.init.normal_(self.byte_offset_emb.weight, std=0.02)
            # 2-scale causal gated byte-mixer (`k=4, d=1` + `k=4, d=2`, 10-byte RF, O(1) 9-byte ring)
            self.dec_dw_norm = CalciteRMSNorm(d_model)
            self.dec_dw_conv = nn.Conv1d(
                d_dw, d_dw, kernel_size=4, groups=d_dw, bias=True
            )
            nn.init.normal_(self.dec_dw_conv.weight, std=0.02)
            nn.init.zeros_(self.dec_dw_conv.bias)
            if self.use_dw_bottleneck:
                self.dec_dw_in_gate = nn.Linear(d_model, 2 * d_dw, bias=False)
                self.dec_dw_gate = None
            else:
                self.dec_dw_in_gate = None
                self.dec_dw_gate = nn.Linear(d_model, d_model, bias=False)
            self.dec_dw_out = nn.Linear(d_dw, d_model, bias=False)

            self.dec_dw2_norm = CalciteRMSNorm(d_model)
            self.dec_dw2_conv = nn.Conv1d(
                d_dw, d_dw, kernel_size=4, dilation=2, groups=d_dw, bias=True
            )
            nn.init.normal_(self.dec_dw2_conv.weight, std=0.02)
            nn.init.zeros_(self.dec_dw2_conv.bias)
            if self.use_dw_bottleneck:
                self.dec_dw2_in_gate = nn.Linear(d_model, 2 * d_dw, bias=False)
                self.dec_dw2_gate = None
            else:
                self.dec_dw2_in_gate = None
                self.dec_dw2_gate = nn.Linear(d_model, d_model, bias=False)
            self.dec_dw2_out = nn.Linear(d_dw, d_model, bias=False)
        else:
            self.byte_offset_emb = None
            self.dec_dw_conv = None
            self.dec_dw_norm = None
            self.dec_dw_in_gate = None
            self.dec_dw_gate = None
            self.dec_dw_out = None
            self.dec_dw2_norm = None
            self.dec_dw2_conv = None
            self.dec_dw2_in_gate = None
            self.dec_dw2_gate = None
            self.dec_dw2_out = None

        if self.use_sym_byte_swa:
            # Symmetric encoder dilated mixer (k=4, d=2 -> 14-byte encoder RF)
            self.enc_dw2_norm = CalciteRMSNorm(d_model)
            self.enc_dw2_conv = nn.Conv1d(
                d_dw, d_dw, kernel_size=4, dilation=2, groups=d_dw, bias=True
            )
            nn.init.normal_(self.enc_dw2_conv.weight, std=0.02)
            nn.init.zeros_(self.enc_dw2_conv.bias)
            if self.use_dw_bottleneck:
                self.enc_dw2_in_gate = nn.Linear(d_model, 2 * d_dw, bias=False)
                self.enc_dw2_gate = None
            else:
                self.enc_dw2_in_gate = None
                self.enc_dw2_gate = nn.Linear(d_model, d_model, bias=False)
            self.enc_dw2_out = nn.Linear(d_dw, d_model, bias=False)

            # Optional 3rd-scale decoder dilated mixer (k=4, d=4 -> 22-byte RF; ablatable via dec_dw3_enabled)
            if self.dec_dw3_enabled:
                self.dec_dw3_norm = CalciteRMSNorm(d_model)
                self.dec_dw3_conv = nn.Conv1d(
                    d_dw, d_dw, kernel_size=4, dilation=4, groups=d_dw, bias=True
                )
                nn.init.normal_(self.dec_dw3_conv.weight, std=0.02)
                nn.init.zeros_(self.dec_dw3_conv.bias)
                if self.use_dw_bottleneck:
                    self.dec_dw3_in_gate = nn.Linear(d_model, 2 * d_dw, bias=False)
                    self.dec_dw3_gate = None
                else:
                    self.dec_dw3_in_gate = None
                    self.dec_dw3_gate = nn.Linear(d_model, d_model, bias=False)
                self.dec_dw3_out = nn.Linear(d_dw, d_model, bias=False)
            else:
                self.dec_dw3_norm = None
                self.dec_dw3_conv = None
                self.dec_dw3_in_gate = None
                self.dec_dw3_gate = None
                self.dec_dw3_out = None

            # Micro Byte-SWA (`W_byte = 24`, `d = 64`, `2` heads -> 6 KB O(1) KV ring)
            self.byte_swa_norm = CalciteRMSNorm(d_model)
            self.byte_swa_qkv = nn.Linear(d_model, 3 * self.byte_swa_dim, bias=False)
            self.byte_swa_out = nn.Linear(self.byte_swa_dim, d_model, bias=False)
            with torch.no_grad():
                self.byte_swa_out.weight.mul_(0.35)
            b_inv_freq = 1.0 / (
                10000.0
                ** (
                    torch.arange(0, self.byte_swa_head_dim, 2, dtype=torch.float32)
                    / self.byte_swa_head_dim
                )
            )
            self.register_buffer("byte_swa_inv_freq", b_inv_freq, persistent=False)
            m0, mi, bq = build_swa_block_masks(self.byte_swa_window)
            self.register_buffer("_bswa_mask0", m0, persistent=False)
            self.register_buffer("_bswa_mask_inner", mi, persistent=False)
            self._bswa_block_q = bq
        else:
            self.enc_dw2_norm = None
            self.enc_dw2_conv = None
            self.enc_dw2_in_gate = None
            self.enc_dw2_gate = None
            self.enc_dw2_out = None
            self.dec_dw3_norm = None
            self.dec_dw3_conv = None
            self.dec_dw3_in_gate = None
            self.dec_dw3_gate = None
            self.dec_dw3_out = None
            self.byte_swa_norm = None
            self.byte_swa_qkv = None
            self.byte_swa_out = None
            self.byte_swa_inv_freq = None
            self._bswa_mask0 = None
            self._bswa_mask_inner = None
            self._bswa_block_q = 64

        if self.use_byte_engram:
            # Pure-tensor hashed 2/3(/4)-byte lexical n-gram memory (zero dynamic state!)
            self.engram_emb2 = nn.Embedding(self.engram_slots, self.engram_dim)
            self.engram_emb3 = nn.Embedding(self.engram_slots, self.engram_dim)
            nn.init.normal_(self.engram_emb2.weight, std=0.02)
            nn.init.normal_(self.engram_emb3.weight, std=0.02)
            if self.engram_use_4gram:
                self.engram_emb4 = nn.Embedding(self.engram_slots, self.engram_dim)
                nn.init.normal_(self.engram_emb4.weight, std=0.02)
            else:
                self.engram_emb4 = None
            self.engram_norm = CalciteRMSNorm(self.engram_dim)
            self.engram_proj = nn.Linear(self.engram_dim, d_model, bias=False)
            with torch.no_grad():
                self.engram_proj.weight.mul_(0.25)
            self.engram_gate = nn.Linear(d_model, 1, bias=True)
            nn.init.constant_(self.engram_gate.bias, -1.0)
        else:
            self.engram_emb2 = None
            self.engram_emb3 = None
            self.engram_emb4 = None
            self.engram_norm = None
            self.engram_proj = None
            self.engram_gate = None

        if self.use_byte_induct_cache:
            self.byte_induct_cache = ByteInductionCache(
                vocab_size=self.vocab_size,
                d_model=d_model,
                n_slots=self.byte_induct_slots,
                d_sig=8,
                d_v=self.byte_induct_dim,
            )
        else:
            self.byte_induct_cache = None
        self._bic_active: bool = True

        self.dec_fuse_norm = CalciteRMSNorm(d_model + 8 * self.d_byte)
        self.dec_fuse = nn.Linear(d_model + 8 * self.d_byte, d_model, bias=False)
        self.dec_norm = CalciteRMSNorm(d_model)
        self.dec_mlp = SwiGLU(d_model, byte_d_ff)

        self.last_ratio_loss: torch.Tensor | None = None
        self._last_chunk_ratio_val: float = 1.0
        self._last_chunk_ratio_t: torch.Tensor | None = None
        self.last_m_cap_used: int = 0

    @property
    def last_chunk_ratio(self) -> float:
        if self._last_chunk_ratio_t is not None:
            self._last_chunk_ratio_val = float(self._last_chunk_ratio_t.item())
            self._last_chunk_ratio_t = None
        return self._last_chunk_ratio_val

    def static_m_cap_for_T(self, T: int) -> int:
        if T <= 1:
            return max(1, T)
        if self.byte_chunk_mode != "utf8":
            return T
        k_paid = min(
            T - 1,
            (self.burst_int + self.refill_int * (T - 1)) // self.cost_int,
        )
        s_min = min(self.max_chunk_bytes, max(1, self.max_utf8_bytes - 3))
        m_cap = 1 + k_paid + ((T - 1 - k_paid) // s_min)
        return min(T, m_cap)

    def _r_lin(self, lin: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if isinstance(lin, nn.Linear):
            dt = x.dtype
            if self.router_locked:
                if lin.weight.dtype == dt and (lin.bias is None or lin.bias.dtype == dt):
                    return F.linear(x, lin.weight, lin.bias)
                # Cache dtype-matched frozen router weights keyed by `(data_ptr, dtype, device)`
                key = (int(lin.weight.data_ptr()), dt, x.device)
                cached = lin.__dict__.get("_locked_r_cache")
                if cached is not None and cached[0] == key and not cached[1].is_inference():
                    w_c, b_c = cached[1], cached[2]
                else:
                    with torch.inference_mode(False), torch.no_grad():
                        w_c = lin.weight.detach().to(device=x.device, dtype=dt).clone()
                        b_c = (
                            lin.bias.detach().to(device=x.device, dtype=dt).clone()
                            if lin.bias is not None
                            else None
                        )
                    lin.__dict__["_locked_r_cache"] = (key, w_c, b_c)
                return F.linear(x, w_c, b_c)
            wq = quantize_ternary_ste(lin.weight).to(dt)
            b = lin.bias.to(dt) if lin.bias is not None else None
            return F.linear(x, wq, b)
        return lin(x)

    def _router_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.router_emb,
            self.router_in_norm,
            self.router_in_proj,
            self.router_q,
            self.router_k,
            self.router_ctx,
            self.router_ent_norm,
            self.router_ent_head,
        )

    def is_locked_router_module(self, mod: nn.Module) -> bool:
        if not self.router_locked:
            return False
        return any(
            mod is rmod
            for rmod in (
                self.router_in_proj,
                self.router_q,
                self.router_k,
                self.router_ctx,
                self.router_ent_head,
            )
        )

    def lock_router(self, locked: bool = True) -> None:
        flag = bool(locked)
        router_mods = self._router_modules()
        if flag and not self.router_locked:
            with torch.no_grad():
                for mod in router_mods:
                    if isinstance(mod, nn.Linear):
                        _, trits_i8, scale_1d = quantize_ternary_weights_ls(mod.weight.data)
                        scale_fp16 = scale_1d.to(torch.float16).float().unsqueeze(-1)
                        w_aligned = (
                            (trits_i8.float() * scale_fp16)
                            .reshape(mod.weight.shape)
                            .to(mod.weight.dtype)
                        )
                        mod.weight.data.copy_(w_aligned)
                        if mod.bias is not None:
                            mod.bias.data.copy_(
                                mod.bias.data.to(torch.float16).to(mod.bias.dtype)
                            )
        for mod in router_mods:
            mod.__dict__.pop("_locked_r_cache", None)
            for p in mod.parameters():
                p.requires_grad_(not flag)
        self.router_feat_weights.requires_grad_(not flag)
        self.router_locked = flag

    def compute_state_bytes(self) -> int:
        base = (
            7 * self.d_byte * 2
            + 3 * self.d_rchar * 2
            + self.router_dim * 2
            + 4
            + self.d_model * 4
            + self.d_model * 4  # cur_z_bar
            + 4
        )
        if self.byte_chunk_mode == "utf8":
            base += 3 * 4
        if self.use_scaled_decoder:
            base += 9 * self.d_dw * 2  # causal 2-scale dw conv rings (3 + 6)
        if self.use_sym_byte_swa:
            base += 6 * self.d_dw * 2  # enc_dw2 (6)
            if self.dec_dw3_enabled:
                base += 12 * self.d_dw * 2  # dec_dw3 (12)
            base += (
                2 * self.byte_swa_window * (self.byte_swa_heads * self.byte_swa_head_dim) * 2
            )  # micro byte-SWA K+V ring
        if self.use_byte_engram:
            base += (4 if self.engram_use_4gram else 3) * 8  # ring_tok3 int64
        return int(base)

    @staticmethod
    def _project_dw_in_gate(
        xn: torch.Tensor,
        gate_proj: nn.Module | None,
        in_gate_proj: nn.Module | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Projects normalized byte state into `(x_conv, gate)` of width `d_dw`."""
        if in_gate_proj is not None:
            xc, g = in_gate_proj(xn).chunk(2, dim=-1)
            return xc, g
        assert gate_proj is not None
        return xn, gate_proj(xn)

    @staticmethod
    def _apply_dw_conv4_batch(
        x: torch.Tensor,
        norm: nn.Module,
        conv: nn.Conv1d,
        out_proj: nn.Module,
        dilation: int,
        gate_proj: nn.Module | None,
        in_gate_proj: nn.Module | None,
    ) -> torch.Tensor:
        xn = norm(x)
        xc, g = CalciteV2Chunker._project_dw_in_gate(xn, gate_proj, in_gate_proj)
        c = _depthwise_conv4_shift_add(xc, conv, dilation=dilation)
        return x + out_proj(F.silu(c) * g)

    @staticmethod
    def _apply_dw_conv4_step(
        x_t: torch.Tensor,
        ring: torch.Tensor,
        ptr: int,
        norm: nn.Module,
        conv: nn.Conv1d,
        out_proj: nn.Module,
        dilation: int,
        gate_proj: nn.Module | None,
        in_gate_proj: nn.Module | None,
    ) -> tuple[torch.Tensor, int]:
        xn = norm(x_t)
        xc, g = CalciteV2Chunker._project_dw_in_gate(xn, gate_proj, in_gate_proj)
        c, next_ptr = _dw_conv4_step_circular(xc, ring, ptr, conv, dilation=dilation)
        return x_t + out_proj(F.silu(c) * g), next_ptr

    def _causal_win8(self, e_tok: torch.Tensor) -> torch.Tensor:
        c_char = e_tok[..., : self.d_byte]
        B, T, _ = c_char.shape
        c_pad = F.pad(c_char, (0, 0, 7, 0))
        shifts = [c_pad[:, 7 - k : 7 - k + T, :] for k in range(8)]
        return torch.cat(shifts, dim=-1).view(B, T, 8 * self.d_byte)

    def _router_probs(
        self, token_ids: torch.Tensor, work_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = token_ids.shape
        log_v = math.log(max(2, self.vocab_size))
        rc = self.router_emb(token_ids)
        if rc.dtype != work_dtype:
            rc = rc.to(work_dtype)
        rc_pad = F.pad(rc, (0, 0, 3, 0))
        r_win4 = torch.cat(
            [rc_pad[:, 3 - k : 3 - k + T, :] for k in range(4)], dim=-1
        ).view(B, T, 4 * self.d_rchar)
        h_r = F.silu(self._r_lin(self.router_in_proj, self.router_in_norm(r_win4)))

        q_r = F.normalize(self._r_lin(self.router_q, h_r).float(), dim=-1)
        k_r = F.normalize(self._r_lin(self.router_k, h_r).float(), dim=-1)
        d_cos = torch.ones(B, T, device=token_ids.device, dtype=torch.float32)
        if T > 1:
            d_cos[:, 1:] = 0.5 * (1.0 - (q_r[:, 1:] * k_r[:, :-1]).sum(dim=-1))

        loc_logits = self._r_lin(self.router_ent_head, self.router_ent_norm(h_r)).float()
        log_p = F.log_softmax(loc_logits, dim=-1)
        h_ent_norm = (-(log_p.exp() * log_p).sum(dim=-1) / log_v).clamp(0.0, 1.5)
        dh_pos = torch.zeros_like(h_ent_norm)
        if T > 1:
            dh_pos[:, 1:] = F.relu(h_ent_norm[:, 1:] - h_ent_norm[:, :-1])

        wf = self.router_feat_weights.float()
        ctx_logit = self._r_lin(self.router_ctx, h_r).float().squeeze(-1)
        p_prob = torch.sigmoid(
            wf[0] * d_cos + wf[1] * h_ent_norm + wf[2] * dh_pos + ctx_logit
        )
        return p_prob, loc_logits

    def router_forward_only(
        self, token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        """Standalone training loss for pre-freezing the UTF-8 router before step 0."""
        B, T = token_ids.shape
        work_dtype = self.router_emb.weight.dtype
        p_prob, loc_logits = self._router_probs(token_ids, work_dtype)
        valid_mask = token_ids != self.pad_token_id
        valid_mask[:, 0] = True

        _, is_cont, char_len = _utf8_tensor_flags(token_ids)
        is_cont_v = is_cont & valid_mask
        c1 = F.pad(is_cont_v[:, :-1], (1, 0), value=False) if T > 1 else torch.zeros_like(is_cont_v)
        c2 = F.pad(is_cont_v[:, :-2], (2, 0), value=False) if T > 2 else torch.zeros_like(is_cont_v)
        c3 = F.pad(is_cont_v[:, :-3], (3, 0), value=False) if T > 3 else torch.zeros_like(is_cont_v)
        can_fire = valid_mask & (~(is_cont_v & (~(c1 & c2 & c3))))
        is_lead = (valid_mask & (~is_cont)).long()
        raw_cand = (p_prob >= 0.5) & can_fire
        b_mask = _enforce_utf8_chunk_span_jit(
            raw_cand,
            valid_mask,
            can_fire,
            is_lead,
            char_len,
            self.max_chunk_bytes,
            self.max_utf8_bytes,
            self.refill_int,
            self.cost_int,
            self.burst_int,
        )

        ratio_mask = (valid_mask[:, 1:] & (~is_cont[:, 1:])).float()
        denom_r = ratio_mask.sum().clamp(min=1.0)
        f_hard = ((b_mask[:, 1:].float() * ratio_mask).sum() / denom_r).detach()
        g_soft = (p_prob[:, 1:] * ratio_mask).sum() / denom_r
        n_comp = max(1.5, float(self.target_chunk_bytes))
        l_ratio = (n_comp / (n_comp - 1.0)) * (
            (n_comp - 1.0) * f_hard * g_soft + (1.0 - f_hard) * (1.0 - g_soft)
        )
        ent_mask_f = valid_mask[:, 1:].float()
        ent_per_tok = F.cross_entropy(
            loc_logits[:, :-1].reshape(-1, self.vocab_size),
            token_ids[:, 1:].reshape(-1),
            reduction="none",
        ).view(B, T - 1)
        l_ent = (ent_per_tok * ent_mask_f).sum() / ent_mask_f.sum().clamp(min=1.0)
        chunk_ratio = float((1.0 / f_hard.clamp(min=1e-4)).item())
        return l_ratio + 0.35 * l_ent, chunk_ratio

    @staticmethod
    def _compute_offset_in_chunk(b_mask: torch.Tensor, max_offset: int) -> torch.Tensor:
        """Strictly causal byte offset `0..max_offset` since the most recent boundary fire."""
        B, T = b_mask.shape
        t_idx = torch.arange(T, device=b_mask.device, dtype=torch.long).unsqueeze(0).expand(B, T)
        fire_pos = torch.where(b_mask, t_idx, torch.zeros_like(t_idx))
        last_fire, _ = torch.cummax(fire_pos, dim=1)
        return (t_idx - last_fire).clamp(0, max_offset)

    @staticmethod
    def _exit_inference(t: torch.Tensor) -> torch.Tensor:
        """Converts an `inference_mode()` tensor into a normal detached tensor safe for autograd."""
        if t.is_inference():
            with torch.inference_mode(False):
                return t.clone()
        return t

    @staticmethod
    def extract_router_chunks(ctx: ChunkContext) -> dict[str, object]:
        """Extracts detached frozen-router & compaction tensors safe for autograd QAT reuse."""
        exit_inf = CalciteV2Chunker._exit_inference
        with torch.inference_mode(False):
            return {
                "b_mask": exit_inf(ctx.b_mask),
                "valid_mask": exit_inf(ctx.valid_mask),
                "p_prob": exit_inf(ctx.p_prob),
                "chunk_counts": exit_inf(ctx.chunk_counts) if ctx.chunk_counts is not None else None,
                "slot_idx": exit_inf(ctx.slot_idx),
                "gather_pos": exit_inf(ctx.gather_pos) if ctx.gather_pos is not None else None,
                "offset_in_chunk": exit_inf(ctx.offset_in_chunk),
                "valid_chunk": exit_inf(ctx.valid_chunk),
                "M_max": int(ctx.valid_chunk.size(1)),
            }

    @staticmethod
    def slice_precomputed_chunks(
        chunks: dict[str, object], start: int, end: int
    ) -> dict[str, object]:
        """Slices a batched frozen-router chunk dict along dim 0 (inference-tensor safe)."""
        with torch.inference_mode(False):
            out: dict[str, object] = {}
            for k, v in chunks.items():
                if isinstance(v, torch.Tensor):
                    sl = v[start:end]
                    out[k] = sl.clone() if sl.is_inference() else sl
                else:
                    out[k] = v
            gp = out.get("gather_pos")
            if isinstance(gp, torch.Tensor):
                out["M_max"] = int(gp.size(1))
            return out

    def encode_to_chunks(
        self,
        e_tok: torch.Tensor,
        token_ids: torch.Tensor,
        tok_emb_weight: torch.Tensor | None = None,
        precomputed_router: dict[str, object] | None = None,
        force_static_m_cap: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, ChunkContext]:
        del tok_emb_weight
        B, T, D = e_tok.shape

        c_win8 = self._causal_win8(e_tok)
        c_norm = self.enc_norm_in(c_win8)
        h_enc0 = e_tok + self.enc_up(c_norm)
        h_byte = h_enc0 + self.enc_mlp(self.enc_norm(h_enc0))
        if (
            self.use_sym_byte_swa
            and self.enc_dw2_norm is not None
            and self.enc_dw2_conv is not None
            and self.enc_dw2_out is not None
            and (self.enc_dw2_gate is not None or self.enc_dw2_in_gate is not None)
        ):
            h_byte = self._apply_dw_conv4_batch(
                h_byte,
                self.enc_dw2_norm,
                self.enc_dw2_conv,
                self.enc_dw2_out,
                dilation=2,
                gate_proj=self.enc_dw2_gate,
                in_gate_proj=self.enc_dw2_in_gate,
            )

        offset_in_chunk: torch.Tensor | None = None
        if precomputed_router is not None and self.router_locked:
            b_mask = precomputed_router["b_mask"]
            valid_mask = precomputed_router["valid_mask"]
            p_prob = precomputed_router["p_prob"]
            chunk_counts = precomputed_router["chunk_counts"]
            slot_idx = precomputed_router["slot_idx"]
            gather_pos = precomputed_router["gather_pos"]
            offset_in_chunk = precomputed_router["offset_in_chunk"]
            valid_chunk = precomputed_router["valid_chunk"]
            assert isinstance(b_mask, torch.Tensor)
            assert isinstance(valid_mask, torch.Tensor)
            assert isinstance(p_prob, torch.Tensor)
            assert isinstance(chunk_counts, torch.Tensor)
            assert isinstance(slot_idx, torch.Tensor)
            assert isinstance(gather_pos, torch.Tensor)
            assert isinstance(offset_in_chunk, torch.Tensor)
            assert isinstance(valid_chunk, torch.Tensor)
            if torch.is_grad_enabled():
                exit_inf = self._exit_inference
                b_mask = exit_inf(b_mask)
                valid_mask = exit_inf(valid_mask)
                p_prob = exit_inf(p_prob)
                chunk_counts = exit_inf(chunk_counts)
                slot_idx = exit_inf(slot_idx)
                gather_pos = exit_inf(gather_pos)
                offset_in_chunk = exit_inf(offset_in_chunk)
                valid_chunk = exit_inf(valid_chunk)
            M_max = int(gather_pos.size(1))
            self.last_ratio_loss = None
            self.last_m_cap_used = M_max
        else:
            if self.router_locked:
                with torch.no_grad():
                    p_prob, loc_logits = self._router_probs(token_ids, e_tok.dtype)
                p_prob = p_prob.detach()
            else:
                p_prob, loc_logits = self._router_probs(token_ids, e_tok.dtype)

            valid_mask = token_ids != self.pad_token_id
            valid_mask[:, 0] = True

            use_cuda_compact = (
                self.use_fast_kernels
                and self.byte_chunk_mode == "utf8"
                and token_ids.is_cuda
            )
            # Tight exact M_max in eval (0.0 logit diff, cuts ~42% padded backbone waste at L=2K..32K),
            # except when `force_static_m_cap=True` (e.g. QAT teacher `return_chunks=True` for static compiled shapes).
            tight_eval = (not self.training) and (not force_static_m_cap)

            if use_cuda_compact:
                m_cap_req = self.static_m_cap_for_T(T) if (self.use_static_m_cap or tight_eval) else T
                fused_offset_fn = getattr(
                    _swa_chunker_mod, "utf8_span_compact_and_offset_cuda", None
                )
                if callable(fused_offset_fn):
                    b_mask, chunk_counts, slot_idx, gather_pos, offset_in_chunk = fused_offset_fn(
                        token_ids,
                        p_prob,
                        self.pad_token_id,
                        self.max_chunk_bytes,
                        self.max_utf8_bytes,
                        self.refill_int,
                        self.cost_int,
                        self.burst_int,
                        m_cap_req,
                    )
                    offset_in_chunk = offset_in_chunk.clamp(0, self.max_utf8_bytes)
                else:
                    b_mask, chunk_counts, slot_idx, gather_pos = utf8_span_and_compact_cuda(
                        token_ids,
                        p_prob,
                        self.pad_token_id,
                        self.max_chunk_bytes,
                        self.max_utf8_bytes,
                        self.refill_int,
                        self.cost_int,
                        self.burst_int,
                        m_cap_req,
                    )
                if tight_eval:
                    raw_m = max(1, int(chunk_counts.max().item())) if B > 0 else 1
                    M_max = min(T, raw_m)
                    if M_max <= gather_pos.size(1):
                        gather_pos = gather_pos[:, :M_max]
                        slot_idx = slot_idx.clamp(max=max(0, M_max - 1))
                    else:
                        chunk_counts, slot_idx, gather_pos = _compact_b_mask_zero_argsort_pytorch(
                            b_mask, M_max
                        )
                elif self.use_static_m_cap:
                    M_max = m_cap_req
                else:
                    raw_m = int(chunk_counts.max().item())
                    M_max = min(T, ((raw_m + 31) // 32) * 32) if T >= 32 else max(1, raw_m)
                    gather_pos = gather_pos[:, :M_max]
                    slot_idx = slot_idx.clamp(max=max(0, M_max - 1))

                if T > 1:
                    if not self.router_locked:
                        _, is_cont, _ = _utf8_tensor_flags(token_ids)
                        ratio_mask_f = (valid_mask[:, 1:] & (~is_cont[:, 1:])).float()
                        r_sum = ratio_mask_f.sum()
                        denom_r = r_sum.clamp(min=1.0)
                        f_hard = ((b_mask[:, 1:].float() * ratio_mask_f).sum() / denom_r).detach()
                        g_soft = (p_prob[:, 1:] * ratio_mask_f).sum() / denom_r
                        n_comp = max(1.5, float(self.target_chunk_bytes))
                        l_ratio = (n_comp / (n_comp - 1.0)) * (
                            (n_comp - 1.0) * f_hard * g_soft + (1.0 - f_hard) * (1.0 - g_soft)
                        )
                        ent_mask_f = valid_mask[:, 1:].float()
                        ent_per_tok = F.cross_entropy(
                            loc_logits[:, :-1].reshape(-1, self.vocab_size),
                            token_ids[:, 1:].reshape(-1),
                            reduction="none",
                        ).view(B, T - 1)
                        l_ent = (ent_per_tok * ent_mask_f).sum() / ent_mask_f.sum().clamp(min=1.0)
                        self.last_ratio_loss = torch.where(
                            r_sum > 0.0, l_ratio + 0.25 * l_ent, 0.0 * p_prob.sum()
                        )
                        self._last_chunk_ratio_t = torch.where(
                            r_sum > 0.0, 1.0 / f_hard.clamp(min=1e-4), torch.ones_like(f_hard)
                        )
                    else:
                        self.last_ratio_loss = None
                        valid_non_first = valid_mask[:, 1:].float().sum().clamp(min=1.0)
                        fires_non_first = b_mask[:, 1:].float().sum().clamp(min=1.0)
                        self._last_chunk_ratio_t = valid_non_first / fires_non_first
                else:
                    self._last_chunk_ratio_val = 1.0
                    self._last_chunk_ratio_t = None
                    self.last_ratio_loss = None if self.router_locked else (0.0 * p_prob.sum())
            else:
                if self.byte_chunk_mode == "utf8":
                    _, is_cont, char_len = _utf8_tensor_flags(token_ids)
                    is_cont_v = is_cont & valid_mask
                    c1 = F.pad(is_cont_v[:, :-1], (1, 0), value=False) if T > 1 else torch.zeros_like(is_cont_v)
                    c2 = F.pad(is_cont_v[:, :-2], (2, 0), value=False) if T > 2 else torch.zeros_like(is_cont_v)
                    c3 = F.pad(is_cont_v[:, :-3], (3, 0), value=False) if T > 3 else torch.zeros_like(is_cont_v)
                    can_fire = valid_mask & (~(is_cont_v & (~(c1 & c2 & c3))))
                    is_lead = (valid_mask & (~is_cont)).long()
                    raw_cand = (p_prob >= 0.5) & can_fire
                    b_mask = _enforce_utf8_chunk_span_jit(
                        raw_cand,
                        valid_mask,
                        can_fire,
                        is_lead,
                        char_len,
                        self.max_chunk_bytes,
                        self.max_utf8_bytes,
                        self.refill_int,
                        self.cost_int,
                        self.burst_int,
                    )
                    ratio_mask = (valid_mask[:, 1:] & (~is_cont[:, 1:])) if T > 1 else valid_mask[:, 1:]
                else:
                    raw_b = (p_prob >= 0.5) & valid_mask
                    raw_b[:, 0] = True
                    b_mask = enforce_max_chunk_span_vec(raw_b, valid_mask, self.max_chunk_bytes)
                    ratio_mask = valid_mask[:, 1:]

                if T > 1:
                    ratio_mask_f = ratio_mask.float()
                    r_sum = ratio_mask_f.sum()
                    has_valid = r_sum > 0.0
                    denom_r = r_sum.clamp(min=1.0)
                    f_hard = ((b_mask[:, 1:].float() * ratio_mask_f).sum() / denom_r).detach()
                    self._last_chunk_ratio_t = torch.where(
                        has_valid, 1.0 / f_hard.clamp(min=1e-4), torch.ones_like(f_hard)
                    )
                    if not self.router_locked:
                        g_soft = (p_prob[:, 1:] * ratio_mask_f).sum() / denom_r
                        n_comp = max(1.5, float(self.target_chunk_bytes))
                        l_ratio = (n_comp / (n_comp - 1.0)) * (
                            (n_comp - 1.0) * f_hard * g_soft + (1.0 - f_hard) * (1.0 - g_soft)
                        )
                        ent_mask_f = valid_mask[:, 1:].float()
                        ent_per_tok = F.cross_entropy(
                            loc_logits[:, :-1].reshape(-1, self.vocab_size),
                            token_ids[:, 1:].reshape(-1),
                            reduction="none",
                        ).view(B, T - 1)
                        l_ent = (ent_per_tok * ent_mask_f).sum() / ent_mask_f.sum().clamp(min=1.0)
                        self.last_ratio_loss = torch.where(
                            has_valid, l_ratio + 0.25 * l_ent, 0.0 * p_prob.sum()
                        )
                    else:
                        self.last_ratio_loss = None
                else:
                    self._last_chunk_ratio_val = 1.0
                    self._last_chunk_ratio_t = None
                    self.last_ratio_loss = None if self.router_locked else (0.0 * p_prob.sum())

                chunk_counts_raw = b_mask.long().sum(dim=1)
                if tight_eval:
                    raw_m = max(1, int(chunk_counts_raw.max().item())) if B > 0 else 1
                    M_max = min(T, raw_m)
                elif self.use_static_m_cap:
                    M_max = self.static_m_cap_for_T(T)
                else:
                    raw_m = int(chunk_counts_raw.max().item())
                    M_max = (
                        min(T, ((raw_m + 31) // 32) * 32)
                        if (self.byte_chunk_mode == "utf8" and T >= 32)
                        else max(1, raw_m)
                    )
                # O(B * T) zero-`argsort` exact compaction
                chunk_counts, slot_idx, gather_pos = _compact_b_mask_zero_argsort_pytorch(b_mask, M_max)

            self.last_m_cap_used = int(M_max)
            valid_chunk = (
                torch.arange(M_max, device=e_tok.device, dtype=torch.long).unsqueeze(0)
                < chunk_counts.unsqueeze(1)
            )
            if offset_in_chunk is None:
                offset_in_chunk = self._compute_offset_in_chunk(b_mask, self.max_utf8_bytes)

        chunk_mask_f = valid_chunk.unsqueeze(-1).to(e_tok.dtype)

        # 2D `index_select` on flattened `(B * T, D)` (faster than 3D expanded-index `torch.gather`)
        row_off = (torch.arange(B, device=e_tok.device, dtype=torch.long) * T).unsqueeze(1)
        flat_pos = (gather_pos + row_off).reshape(-1)
        z_in = (
            h_byte.contiguous().view(B * T, D).index_select(0, flat_pos).view(B, M_max, D)
            * chunk_mask_f
        )
        c_win8_dim = 8 * self.d_byte
        c_win8_chunk = (
            c_win8.contiguous()
            .view(B * T, c_win8_dim)
            .index_select(0, flat_pos)
            .view(B, M_max, c_win8_dim)
        )
        h_lex_chunk = self.lex_up(self.lex_norm(c_win8_chunk)) * chunk_mask_f
        p_chunk = torch.gather(p_prob, 1, gather_pos) * valid_chunk.float()

        ctx = ChunkContext(
            b_mask=b_mask,
            valid_mask=valid_mask,
            p_prob=p_prob,
            p_chunk=p_chunk,
            h_byte=h_byte,
            c_win8=c_norm,
            slot_idx=slot_idx,
            offset_in_chunk=offset_in_chunk,
            valid_chunk=valid_chunk,
            token_ids=token_ids,
            chunk_counts=chunk_counts,
            gather_pos=gather_pos,
        )
        return z_in, h_lex_chunk, ctx

    def decode_from_chunks(
        self,
        z_hat: torch.Tensor,
        ctx: ChunkContext,
    ) -> torch.Tensor:
        B, T, D = ctx.h_byte.shape
        M_max = z_hat.size(1)

        if M_max > 1:
            pc = ctx.p_chunk[:, 1:].unsqueeze(-1).to(z_hat.dtype)
            z_tail = pc * z_hat[:, 1:] + (1.0 - pc) * z_hat[:, :-1]
            z_bar = torch.cat((z_hat[:, :1], z_tail), dim=1)
        else:
            z_bar = z_hat

        row_off = (torch.arange(B, device=z_hat.device, dtype=torch.long) * M_max).unsqueeze(1)
        flat_slot = (ctx.slot_idx + row_off).reshape(-1)
        z_up = z_bar.contiguous().view(B * M_max, D).index_select(0, flat_slot).view(B, T, D)

        # When `router_locked` (or no_grad), `c_conf` is detached so `ste_c === 1.0` identically
        if self.router_locked or not torch.is_grad_enabled():
            z_tilde = self.dechunk_norm(z_up)
        else:
            c_conf = torch.where(ctx.b_mask, ctx.p_prob, 1.0 - ctx.p_prob)
            ste_c = 0.85 + 0.15 * (c_conf + (1.0 - c_conf).detach())
            z_tilde = self.dechunk_norm(z_up) * ste_c.to(z_up.dtype).unsqueeze(-1)
        if self.use_scaled_decoder and self.byte_offset_emb is not None:
            z_tilde = z_tilde + self.byte_offset_emb(ctx.offset_in_chunk).to(z_tilde.dtype)

        dec_cat = torch.cat((z_tilde, ctx.c_win8.to(z_tilde.dtype)), dim=-1)
        dec_in = ctx.h_byte + self.dec_fuse(self.dec_fuse_norm(dec_cat))
        if (
            self.use_scaled_decoder
            and self.dec_dw_conv is not None
            and self.dec_dw_norm is not None
            and self.dec_dw_out is not None
            and (self.dec_dw_gate is not None or self.dec_dw_in_gate is not None)
            and self.dec_dw2_norm is not None
            and self.dec_dw2_conv is not None
            and self.dec_dw2_out is not None
            and (self.dec_dw2_gate is not None or self.dec_dw2_in_gate is not None)
        ):
            dec_in = self._apply_dw_conv4_batch(
                dec_in,
                self.dec_dw_norm,
                self.dec_dw_conv,
                self.dec_dw_out,
                dilation=1,
                gate_proj=self.dec_dw_gate,
                in_gate_proj=self.dec_dw_in_gate,
            )
            dec_in = self._apply_dw_conv4_batch(
                dec_in,
                self.dec_dw2_norm,
                self.dec_dw2_conv,
                self.dec_dw2_out,
                dilation=2,
                gate_proj=self.dec_dw2_gate,
                in_gate_proj=self.dec_dw2_in_gate,
            )
        if (
            self.use_sym_byte_swa
            and self.dec_dw3_enabled
            and self.dec_dw3_norm is not None
            and self.dec_dw3_conv is not None
            and self.dec_dw3_out is not None
            and (self.dec_dw3_gate is not None or self.dec_dw3_in_gate is not None)
        ):
            dec_in = self._apply_dw_conv4_batch(
                dec_in,
                self.dec_dw3_norm,
                self.dec_dw3_conv,
                self.dec_dw3_out,
                dilation=4,
                gate_proj=self.dec_dw3_gate,
                in_gate_proj=self.dec_dw3_in_gate,
            )
        if (
            self.use_sym_byte_swa
            and self.byte_swa_norm is not None
            and self.byte_swa_qkv is not None
            and self.byte_swa_out is not None
            and self.byte_swa_inv_freq is not None
        ):
            # Micro Byte-SWA: native CUDA `fast_swa_rope_attn_cuda` or zero-loop block-strided SDPA
            xb_n = self.byte_swa_norm(dec_in)
            qkv_b = self.byte_swa_qkv(xb_n).view(B, T, 3, self.byte_swa_heads, self.byte_swa_head_dim)
            cos_tab, sin_tab = get_cached_rope_cos_sin(
                T, qkv_b.device, qkv_b.dtype, head_dim=self.byte_swa_head_dim, base=10000.0
            )
            if (
                self.use_fast_kernels
                and qkv_b.is_cuda
                and self.byte_swa_head_dim in (32, 64)
                and 1 <= self.byte_swa_window <= 32
                and _swa_cuda_ext_ready()
            ):
                yb_h = fast_swa_rope_attn_cuda(qkv_b, cos_tab, sin_tab, self.byte_swa_window)
                dec_in = dec_in + self.byte_swa_out(yb_h.view(B, T, self.byte_swa_dim))
            else:
                qb = qkv_b[:, :, 0].transpose(1, 2)
                kb = qkv_b[:, :, 1].transpose(1, 2)
                vb = qkv_b[:, :, 2].transpose(1, 2)
                cos_b = cos_tab.view(1, 1, T, -1)
                sin_b = sin_tab.view(1, 1, T, -1)
                qb = apply_rope(qb, cos_b, sin_b)
                kb = apply_rope(kb, cos_b, sin_b)
                assert self._bswa_mask0 is not None and self._bswa_mask_inner is not None
                yb = sliding_window_sdpa(
                    qb,
                    kb,
                    vb,
                    self.byte_swa_window,
                    self._bswa_mask0,
                    self._bswa_mask_inner,
                    self._bswa_block_q,
                )
                dec_in = dec_in + self.byte_swa_out(
                    yb.transpose(1, 2).contiguous().view(B, T, self.byte_swa_dim)
                )

        # Share single `F.pad(token_ids.long(), (6, 0))` across `ByteEngram` and `ByteInductionCache`
        need_engram = (
            self.use_byte_engram
            and self.engram_emb2 is not None
            and self.engram_emb3 is not None
            and self.engram_norm is not None
            and self.engram_proj is not None
            and self.engram_gate is not None
        )
        need_bic = (
            self.use_byte_induct_cache
            and self.byte_induct_cache is not None
            and getattr(self, "_bic_active", True)
        )
        t_pad6 = F.pad(ctx.token_ids.long(), (6, 0), value=0) if (need_engram or need_bic) else None

        if need_engram and t_pad6 is not None:
            t0 = t_pad6[:, 6 : 6 + T]
            t1 = t_pad6[:, 5 : 5 + T]
            t2 = t_pad6[:, 4 : 4 + T]
            base2 = t1 * 263 + t0 * 1009
            raw3 = t2 * 691 + base2 + 17
            use_4g = self.engram_use_4gram and self.engram_emb4 is not None
            raw4 = (t_pad6[:, 3 : 3 + T] * 4099 + raw3 + 31) if use_4g else None
            slots = self.engram_slots
            if slots > 0 and (slots & (slots - 1)) == 0:
                mask = slots - 1
                h2 = base2 & mask
                h3 = raw3 & mask
                h4 = (raw4 & mask) if raw4 is not None else None
            else:
                h2 = torch.remainder(base2, slots)
                h3 = torch.remainder(raw3, slots)
                h4 = torch.remainder(raw4, slots) if raw4 is not None else None
            assert self.engram_emb2 is not None and self.engram_emb3 is not None
            assert self.engram_norm is not None and self.engram_proj is not None and self.engram_gate is not None
            e_sum = self.engram_emb2(h2) + self.engram_emb3(h3)
            if use_4g and self.engram_emb4 is not None and h4 is not None:
                e_sum = e_sum + self.engram_emb4(h4)
            e_eng = e_sum.to(dec_in.dtype)
            eg = torch.sigmoid(self.engram_gate(dec_in))
            dec_in = dec_in + eg * self.engram_proj(self.engram_norm(e_eng))
        if need_bic and self.byte_induct_cache is not None:
            dec_in = dec_in + self.byte_induct_cache(ctx.token_ids, dec_in, t_pad6=t_pad6)
        return dec_in + self.dec_mlp(self.dec_norm(dec_in))

    # ------------------------------------------------------------------
    # Strict O(1) Incremental Streaming API (exact parity with batch)
    # ------------------------------------------------------------------

    def init_stream_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> ChunkerStreamState:
        ring_dec1 = (
            torch.zeros(batch_size, 3, self.d_dw, device=device, dtype=dtype)
            if self.use_scaled_decoder
            else torch.zeros(batch_size, 0, 1, device=device, dtype=dtype)
        )
        ring_dec2 = (
            torch.zeros(batch_size, 6, self.d_dw, device=device, dtype=dtype)
            if self.use_scaled_decoder
            else torch.zeros(batch_size, 0, 1, device=device, dtype=dtype)
        )
        ring_enc2 = (
            torch.zeros(batch_size, 6, self.d_dw, device=device, dtype=dtype)
            if self.use_sym_byte_swa
            else torch.zeros(batch_size, 0, 1, device=device, dtype=dtype)
        )
        ring_dec3 = (
            torch.zeros(batch_size, 12, self.d_dw, device=device, dtype=dtype)
            if (self.use_sym_byte_swa and self.dec_dw3_enabled)
            else torch.zeros(batch_size, 0, 1, device=device, dtype=dtype)
        )
        ring_bk = (
            torch.zeros(
                batch_size,
                self.byte_swa_window,
                self.byte_swa_heads,
                self.byte_swa_head_dim,
                device=device,
                dtype=dtype,
            )
            if self.use_sym_byte_swa
            else torch.zeros(batch_size, 0, 1, 1, device=device, dtype=dtype)
        )
        ring_bv = (
            torch.zeros(
                batch_size,
                self.byte_swa_window,
                self.byte_swa_heads,
                self.byte_swa_head_dim,
                device=device,
                dtype=dtype,
            )
            if self.use_sym_byte_swa
            else torch.zeros(batch_size, 0, 1, 1, device=device, dtype=dtype)
        )
        ring_tok3 = (
            torch.zeros(
                batch_size,
                4 if self.engram_use_4gram else 3,
                device=device,
                dtype=torch.long,
            )
            if self.use_byte_engram
            else torch.zeros(batch_size, 0, device=device, dtype=torch.long)
        )
        return ChunkerStreamState(
            ring_char=torch.zeros(batch_size, 7, self.d_byte, device=device, dtype=dtype),
            ring_rchar=torch.zeros(batch_size, 3, self.d_rchar, device=device, dtype=dtype),
            prev_k_r=torch.zeros(batch_size, self.router_dim, device=device, dtype=torch.float32),
            prev_h_ent=torch.zeros(batch_size, device=device, dtype=torch.float32),
            prev_z_hat=torch.zeros(batch_size, self.d_model, device=device, dtype=dtype),
            cur_z_bar=torch.zeros(batch_size, self.d_model, device=device, dtype=dtype),
            ring_dec_conv1=ring_dec1,
            ring_dec_conv2=ring_dec2,
            ring_enc_conv2=ring_enc2,
            ring_dec_conv3=ring_dec3,
            ring_byte_k=ring_bk,
            ring_byte_v=ring_bv,
            ring_tok3=ring_tok3,
            run_len=torch.zeros(batch_size, device=device, dtype=torch.long),
            cp_run=torch.zeros(batch_size, device=device, dtype=torch.long),
            cont_streak=torch.zeros(batch_size, device=device, dtype=torch.long),
            credits=torch.full((batch_size,), self.burst_int, device=device, dtype=torch.long),
            step_idx=0,
            has_chunk=False,
            last_tok=torch.zeros(batch_size, device=device, dtype=torch.long),
            byte_induct=(
                self.byte_induct_cache.init_stream_state(batch_size, device, dtype)
                if (self.use_byte_induct_cache and self.byte_induct_cache is not None)
                else None
            ),
            char_ptr=0,
            rchar_ptr=0,
            tok3_ptr=0,
            enc2_ptr=0,
            dec1_ptr=0,
            dec2_ptr=0,
            dec3_ptr=0,
        )

    @torch.no_grad()
    def compute_lex_from_win8(self, c_win8_t: torch.Tensor) -> torch.Tensor:
        """Computes lexical chunk embedding `h_lex_t` on firing bytes only (lazy streaming helper)."""
        return self.lex_up(self.lex_norm(c_win8_t))

    @torch.no_grad()
    def encode_byte_step(
        self,
        token_t: torch.Tensor,
        e_tok_t: torch.Tensor,
        state: ChunkerStreamState,
        compute_lex: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Processes ONE byte `token_t` `(B,)` and updates `state` in-place via circular rings.
        Returns `(fire_t, p_t, h_byte_t, c_norm_t, h_lex_or_win8_t, offset_t, valid_t)`.
        When `compute_lex=False`, slot 4 returns raw `c_win8_t` (skipping `lex_norm+lex_up`
        on non-firing bytes); callers pass it to `compute_lex_from_win8()` only when `fire_t` fires.
        """
        dt = e_tok_t.dtype
        log_v = math.log(max(2, self.vocab_size))

        ct = e_tok_t[..., : self.d_byte]
        cp = state.char_ptr
        c_win8_t = torch.cat(
            [ct] + [state.ring_char[:, (cp - 1 - k) % 7] for k in range(7)],
            dim=-1,
        )
        state.ring_char[:, cp] = ct
        state.char_ptr = (cp + 1) % 7

        rc_t = self.router_emb(token_t)
        if rc_t.dtype != dt:
            rc_t = rc_t.to(dt)
        rcp = state.rchar_ptr
        r_win4_t = torch.cat(
            [rc_t] + [state.ring_rchar[:, (rcp - 1 - k) % 3] for k in range(3)],
            dim=-1,
        )
        state.ring_rchar[:, rcp] = rc_t
        state.rchar_ptr = (rcp + 1) % 3

        ring_len = state.ring_tok3.size(1)
        if self.use_byte_engram and ring_len >= 3:
            tp = state.tok3_ptr
            state.ring_tok3[:, tp] = token_t
            state.tok3_ptr = (tp + 1) % ring_len
        if self.use_byte_induct_cache:
            state.last_tok = token_t

        c_norm_t = self.enc_norm_in(c_win8_t)
        h_enc0 = e_tok_t + self.enc_up(c_norm_t)
        h_byte_t = h_enc0 + self.enc_mlp(self.enc_norm(h_enc0))
        if (
            self.use_sym_byte_swa
            and self.enc_dw2_norm is not None
            and self.enc_dw2_conv is not None
            and self.enc_dw2_out is not None
            and (self.enc_dw2_gate is not None or self.enc_dw2_in_gate is not None)
        ):
            h_byte_t, state.enc2_ptr = self._apply_dw_conv4_step(
                h_byte_t,
                state.ring_enc_conv2,
                state.enc2_ptr,
                self.enc_dw2_norm,
                self.enc_dw2_conv,
                self.enc_dw2_out,
                dilation=2,
                gate_proj=self.enc_dw2_gate,
                in_gate_proj=self.enc_dw2_in_gate,
            )
        h_lex_or_win8 = self.compute_lex_from_win8(c_win8_t) if compute_lex else c_win8_t

        h_rt = F.silu(self._r_lin(self.router_in_proj, self.router_in_norm(r_win4_t)))
        q_rt = F.normalize(self._r_lin(self.router_q, h_rt).float(), dim=-1)
        k_rt = F.normalize(self._r_lin(self.router_k, h_rt).float(), dim=-1)
        if state.step_idx == 0:
            d_cos_t = torch.ones_like(state.prev_h_ent)
        else:
            d_cos_t = 0.5 * (1.0 - (q_rt * state.prev_k_r).sum(dim=-1))

        loc_t = self._r_lin(self.router_ent_head, self.router_ent_norm(h_rt)).float()
        lp_t = F.log_softmax(loc_t, dim=-1)
        h_norm_t = (-(lp_t.exp() * lp_t).sum(dim=-1) / log_v).clamp(0.0, 1.5)
        dh_t = torch.zeros_like(h_norm_t) if state.step_idx == 0 else F.relu(h_norm_t - state.prev_h_ent)

        wf = self.router_feat_weights.float()
        ctx_t = self._r_lin(self.router_ctx, h_rt).float().squeeze(-1)
        p_t = torch.sigmoid(wf[0] * d_cos_t + wf[1] * h_norm_t + wf[2] * dh_t + ctx_t)

        vt = token_t != self.pad_token_id
        if state.step_idx == 0:
            vt = torch.ones_like(vt)

        if self.byte_chunk_mode == "utf8":
            _, is_cont_t, cl_t = _utf8_tensor_flags(token_t)
            if state.step_idx == 0:
                fire_t = vt
            else:
                state.credits = torch.where(
                    vt, torch.clamp(state.credits + self.refill_int, max=self.burst_int), state.credits
                )
                is_cont_vt_pre = is_cont_t & vt
                shielded_t = is_cont_vt_pre & (state.cont_streak < 3)
                can_fire_t = vt & (~shielded_t)
                can_soft_t = (state.credits >= self.cost_int) & can_fire_t & (p_t >= 0.5)
                must_span_t = (state.cp_run >= self.max_chunk_bytes) | (
                    (state.run_len + cl_t) > self.max_utf8_bytes
                )
                fire_t = can_fire_t & (can_soft_t | must_span_t)
                state.credits = torch.where(
                    fire_t, torch.clamp(state.credits - self.cost_int, min=0), state.credits
                )
            is_cont_vt = is_cont_t & vt
            is_lead_t = (vt & (~is_cont_t)).long()
            state.cp_run = torch.where(
                vt, torch.where(fire_t, is_lead_t, state.cp_run + is_lead_t), state.cp_run
            )
            state.cont_streak = torch.where(
                is_cont_vt, state.cont_streak + 1, torch.zeros_like(state.cont_streak)
            )
        else:
            if state.step_idx == 0:
                fire_t = vt
            else:
                fire_t = vt & ((p_t >= 0.5) | (state.run_len >= self.max_chunk_bytes))

        # Offset before updating run_len: 0 when fire_t, else previous run_len
        offset_t = torch.where(fire_t, torch.zeros_like(state.run_len), state.run_len).clamp(
            0, self.max_utf8_bytes
        )
        state.run_len = torch.where(
            vt, torch.where(fire_t, torch.ones_like(state.run_len), state.run_len + 1), state.run_len
        )
        state.prev_k_r = k_rt
        state.prev_h_ent = h_norm_t
        state.step_idx += 1

        return fire_t, p_t, h_byte_t, c_norm_t, h_lex_or_win8, offset_t, vt

    @torch.no_grad()
    def decode_byte_step(
        self,
        z_hat_new: torch.Tensor | None,
        fire_t: torch.Tensor,
        p_t: torch.Tensor,
        h_byte_t: torch.Tensor,
        c_norm_t: torch.Tensor,
        offset_t: torch.Tensor,
        state: ChunkerStreamState,
    ) -> torch.Tensor:
        """Decodes ONE byte `(B, d_model)` using `O(1)` `cur_z_bar` and circular conv/KV rings."""
        dt = h_byte_t.dtype
        if z_hat_new is not None:
            if not state.has_chunk:
                z_bar_new = z_hat_new
                state.has_chunk = True
            else:
                pc = p_t.unsqueeze(-1).to(dt)
                z_bar_new = pc * z_hat_new + (1.0 - pc) * state.prev_z_hat
            f_mask = fire_t.unsqueeze(-1)
            state.cur_z_bar = torch.where(f_mask, z_bar_new, state.cur_z_bar)
            state.prev_z_hat = torch.where(f_mask, z_hat_new, state.prev_z_hat)

        # Eval STE factor is identically 1.0 (skip dead `where` + `ones_like` + mul)
        z_tilde = self.dechunk_norm(state.cur_z_bar)
        if self.use_scaled_decoder and self.byte_offset_emb is not None:
            z_tilde = z_tilde + self.byte_offset_emb(offset_t).to(z_tilde.dtype)

        dec_cat = torch.cat((z_tilde, c_norm_t.to(z_tilde.dtype)), dim=-1)
        dec_in = h_byte_t + self.dec_fuse(self.dec_fuse_norm(dec_cat))
        if (
            self.use_scaled_decoder
            and self.dec_dw_conv is not None
            and self.dec_dw_norm is not None
            and self.dec_dw_out is not None
            and (self.dec_dw_gate is not None or self.dec_dw_in_gate is not None)
            and self.dec_dw2_norm is not None
            and self.dec_dw2_conv is not None
            and self.dec_dw2_out is not None
            and (self.dec_dw2_gate is not None or self.dec_dw2_in_gate is not None)
        ):
            dec_in, state.dec1_ptr = self._apply_dw_conv4_step(
                dec_in,
                state.ring_dec_conv1,
                state.dec1_ptr,
                self.dec_dw_norm,
                self.dec_dw_conv,
                self.dec_dw_out,
                dilation=1,
                gate_proj=self.dec_dw_gate,
                in_gate_proj=self.dec_dw_in_gate,
            )
            dec_in, state.dec2_ptr = self._apply_dw_conv4_step(
                dec_in,
                state.ring_dec_conv2,
                state.dec2_ptr,
                self.dec_dw2_norm,
                self.dec_dw2_conv,
                self.dec_dw2_out,
                dilation=2,
                gate_proj=self.dec_dw2_gate,
                in_gate_proj=self.dec_dw2_in_gate,
            )
        if (
            self.use_sym_byte_swa
            and self.dec_dw3_enabled
            and self.dec_dw3_norm is not None
            and self.dec_dw3_conv is not None
            and self.dec_dw3_out is not None
            and (self.dec_dw3_gate is not None or self.dec_dw3_in_gate is not None)
        ):
            dec_in, state.dec3_ptr = self._apply_dw_conv4_step(
                dec_in,
                state.ring_dec_conv3,
                state.dec3_ptr,
                self.dec_dw3_norm,
                self.dec_dw3_conv,
                self.dec_dw3_out,
                dilation=4,
                gate_proj=self.dec_dw3_gate,
                in_gate_proj=self.dec_dw3_in_gate,
            )
        if (
            self.use_sym_byte_swa
            and self.byte_swa_norm is not None
            and self.byte_swa_qkv is not None
            and self.byte_swa_out is not None
            and self.byte_swa_inv_freq is not None
        ):
            # Micro Byte-SWA streaming step: cached RoPE + circular `pos_b % Wb` KV ring (zero `.clone()`)
            B_cur = dec_in.size(0)
            pos_b = max(0, state.step_idx - 1)
            Wb = self.byte_swa_window
            xb_n = self.byte_swa_norm(dec_in)
            qkv_b = self.byte_swa_qkv(xb_n).view(B_cur, 3, self.byte_swa_heads, self.byte_swa_head_dim)
            qb = qkv_b[:, 0]
            kb = qkv_b[:, 1]
            vb = qkv_b[:, 2]
            cos_tab, sin_tab = get_cached_rope_cos_sin(
                pos_b + 1, dec_in.device, qkv_b.dtype, head_dim=self.byte_swa_head_dim, base=10000.0
            )
            cos_b = cos_tab[pos_b : pos_b + 1].view(1, 1, -1)
            sin_b = sin_tab[pos_b : pos_b + 1].view(1, 1, -1)
            qb = apply_rope(qb, cos_b, sin_b)
            kb = apply_rope(kb, cos_b, sin_b)
            slot = pos_b % Wb
            state.ring_byte_k[:, slot] = kb
            state.ring_byte_v[:, slot] = vb
            if pos_b < Wb:
                k_use = state.ring_byte_k[:, : pos_b + 1]
                v_use = state.ring_byte_v[:, : pos_b + 1]
            else:
                k_use = state.ring_byte_k
                v_use = state.ring_byte_v
            sc_b = 1.0 / math.sqrt(float(self.byte_swa_head_dim))
            scores_b = (
                torch.matmul(qb.float().unsqueeze(2), k_use.float().permute(0, 2, 3, 1)) * sc_b
            )
            probs_b = F.softmax(scores_b, dim=-1).to(qkv_b.dtype)
            yb = torch.matmul(probs_b, v_use.permute(0, 2, 1, 3)).reshape(B_cur, self.byte_swa_dim)
            dec_in = dec_in + self.byte_swa_out(yb)
        ring_len = state.ring_tok3.size(1)
        if (
            self.use_byte_engram
            and self.engram_emb2 is not None
            and self.engram_emb3 is not None
            and self.engram_norm is not None
            and self.engram_proj is not None
            and self.engram_gate is not None
            and ring_len >= 3
        ):
            tp = state.tok3_ptr
            t0 = state.ring_tok3[:, (tp - 1) % ring_len]
            t1 = state.ring_tok3[:, (tp - 2) % ring_len]
            t2 = state.ring_tok3[:, (tp - 3) % ring_len]
            base2 = t1 * 263 + t0 * 1009
            raw3 = t2 * 691 + base2 + 17
            use_4g = self.engram_use_4gram and self.engram_emb4 is not None and ring_len >= 4
            raw4 = (
                (state.ring_tok3[:, (tp - 4) % ring_len] * 4099 + raw3 + 31)
                if use_4g
                else None
            )
            slots = self.engram_slots
            if slots > 0 and (slots & (slots - 1)) == 0:
                mask = slots - 1
                h2 = base2 & mask
                h3 = raw3 & mask
                h4 = (raw4 & mask) if raw4 is not None else None
            else:
                h2 = torch.remainder(base2, slots)
                h3 = torch.remainder(raw3, slots)
                h4 = torch.remainder(raw4, slots) if raw4 is not None else None
            e_sum = self.engram_emb2(h2) + self.engram_emb3(h3)
            if use_4g and self.engram_emb4 is not None and h4 is not None:
                e_sum = e_sum + self.engram_emb4(h4)
            e_eng = e_sum.to(dt)
            eg = torch.sigmoid(self.engram_gate(dec_in))
            dec_in = dec_in + eg * self.engram_proj(self.engram_norm(e_eng))
        if (
            self.use_byte_induct_cache
            and self.byte_induct_cache is not None
            and getattr(self, "_bic_active", True)
            and state.byte_induct is not None
            and state.last_tok is not None
        ):
            dec_in = dec_in + self.byte_induct_cache.step(state.last_tok, dec_in, state.byte_induct)
        return dec_in + self.dec_mlp(self.dec_norm(dec_in))
