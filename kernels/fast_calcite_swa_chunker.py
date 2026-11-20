"""
fast_calcite_swa_chunker.py — Native CUDA kernels (`load_inline`, `nvcc`) + clean CPU/PyTorch
fallbacks for the two remaining cortex & byte-chunker bottlenecks in `Calcite System-1`:

1. **Fused Local Sliding-Window RoPE Attention (`W <= 16`, `head_dim == 64`, `BF16/FP16/FP32`):**
   - `get_cached_rope_cos_sin`: zero-copy cached RoPE `(cos, sin)` slices per `(T, device, dtype)`.
   - `fast_swa_rope_attn_cuda`: single-pass fused RoPE + `O(T * W)` SMEM sliding-window attention
     forward + zero-atomic deterministic analytical backward (`torch.library.custom_op` with
     `register_fake` + `register_autograd` for zero-break `torch.compile(..., dynamic=True)`).
   - `patch_fast_swa_attention`: drop-in helper accelerating all local `CausalRoPEAttention`
     layers in `cortex_pre` / `cortex_post` with zero `simvq_split/` edits or `state_dict()` drift.

2. **Zero-`argsort` UTF-8 Chunker + Locked-Router Fast Path:**
   - `utf8_span_and_compact_cuda`: single-pass UTF-8 continuation-shield + leaky-bucket guard +
     compaction returning `(b_mask, chunk_counts, slot_idx, gather_pos)` with ZERO `argsort`.
   - `fast_encode_to_chunks`: skips redundant `quantize_ternary_ste` on the 5 frozen router
     linears when `chunker.router_locked` is `True` (`router_max_abs_diff == 0.0`), eliminates
     `argsort` + intermediate UTF-8 mask launches, and gathers `c_win8` before `lex_norm`/`lex_up`.
   - `patch_fast_chunker`: drop-in helper wiring `fast_encode_to_chunks` onto `model.byte_chunker`.
"""

from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
import shutil
import sysconfig
import tempfile
import types
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F

from simvq_split.baselines import CausalRoPEAttention
from simvq_split.chunker import (
    ByteChunkContext,
    SurpriseHNetChunker,
    _enforce_utf8_chunk_span_jit,
    _utf8_tensor_flags,
    enforce_max_chunk_span_vec,
)

# ---------------------------------------------------------------------------
# CUDA C++ Extension Source (`load_inline`, `nvcc`)
# ---------------------------------------------------------------------------

_CUDA_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> swa_rope_attn_fwd_cuda_impl(
    torch::Tensor qkv,
    torch::Tensor cos,
    torch::Tensor sin,
    int64_t window_size
);

torch::Tensor swa_rope_attn_bwd_cuda_impl(
    torch::Tensor qkv,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor out,
    torch::Tensor lse,
    torch::Tensor grad_out,
    int64_t window_size
);

std::vector<torch::Tensor> utf8_span_and_compact_cuda_impl(
    torch::Tensor token_ids,
    torch::Tensor p_prob,
    int64_t pad_token_id,
    int64_t max_cp_span,
    int64_t max_byte_span,
    int64_t refill_int,
    int64_t cost_int,
    int64_t burst_int,
    int64_t m_cap,
    bool return_masks
);

std::vector<torch::Tensor> swa_rope_attn_fwd_cuda(
    torch::Tensor qkv,
    torch::Tensor cos,
    torch::Tensor sin,
    int64_t window_size
) {
    return swa_rope_attn_fwd_cuda_impl(qkv, cos, sin, window_size);
}

torch::Tensor swa_rope_attn_bwd_cuda(
    torch::Tensor qkv,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor out,
    torch::Tensor lse,
    torch::Tensor grad_out,
    int64_t window_size
) {
    return swa_rope_attn_bwd_cuda_impl(qkv, cos, sin, out, lse, grad_out, window_size);
}

std::vector<torch::Tensor> utf8_span_and_compact_cuda(
    torch::Tensor token_ids,
    torch::Tensor p_prob,
    int64_t pad_token_id,
    int64_t max_cp_span,
    int64_t max_byte_span,
    int64_t refill_int,
    int64_t cost_int,
    int64_t burst_int,
    int64_t m_cap,
    bool return_masks
) {
    return utf8_span_and_compact_cuda_impl(
        token_ids, p_prob, pad_token_id,
        max_cp_span, max_byte_span, refill_int, cost_int, burst_int,
        m_cap, return_masks
    );
}
"""

_CUDA_KERNEL_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include <vector>

namespace {

__device__ __forceinline__ float warp_sum32(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    }
    return v;
}

template <typename T>
__device__ __forceinline__ void load_pair_f32(const T* ptr, int lane, float& e, float& o) {
    e = static_cast<float>(ptr[2 * lane]);
    o = static_cast<float>(ptr[2 * lane + 1]);
}

template <>
__device__ __forceinline__ void load_pair_f32<float>(const float* ptr, int lane, float& e, float& o) {
    const float2 v = reinterpret_cast<const float2*>(ptr)[lane];
    e = v.x;
    o = v.y;
}

template <>
__device__ __forceinline__ void load_pair_f32<at::BFloat16>(
    const at::BFloat16* ptr, int lane, float& e, float& o
) {
    const uint32_t raw = reinterpret_cast<const uint32_t*>(ptr)[lane];
    e = __uint_as_float((raw & 0xFFFFu) << 16);
    o = __uint_as_float(raw & 0xFFFF0000u);
}

template <>
__device__ __forceinline__ void load_pair_f32<at::Half>(
    const at::Half* ptr, int lane, float& e, float& o
) {
    const __half2 v = reinterpret_cast<const __half2*>(ptr)[lane];
    const float2 f = __half22float2(v);
    e = f.x;
    o = f.y;
}

template <typename T>
__device__ __forceinline__ void store_pair_f32(T* ptr, int lane, float e, float o) {
    ptr[2 * lane] = static_cast<T>(e);
    ptr[2 * lane + 1] = static_cast<T>(o);
}

template <>
__device__ __forceinline__ void store_pair_f32<float>(float* ptr, int lane, float e, float o) {
    reinterpret_cast<float2*>(ptr)[lane] = make_float2(e, o);
}

template <>
__device__ __forceinline__ void store_pair_f32<at::BFloat16>(
    at::BFloat16* ptr, int lane, float e, float o
) {
    const uint32_t ue = __float_as_uint(e);
    const uint32_t uo = __float_as_uint(o);
    const uint32_t be = ((ue & 0x7FFFFFFFu) > 0x7F800000u)
        ? ((ue >> 16) | 0x0040u)
        : ((ue + 0x7FFFu + ((ue >> 16) & 1u)) >> 16);
    const uint32_t bo = ((uo & 0x7FFFFFFFu) > 0x7F800000u)
        ? ((uo >> 16) | 0x0040u)
        : ((uo + 0x7FFFu + ((uo >> 16) & 1u)) >> 16);
    reinterpret_cast<uint32_t*>(ptr)[lane] = (be & 0xFFFFu) | (bo << 16);
}

template <>
__device__ __forceinline__ void store_pair_f32<at::Half>(at::Half* ptr, int lane, float e, float o) {
    reinterpret_cast<__half2*>(ptr)[lane] = __float22half2_rn(make_float2(e, o));
}

// ---------------------------------------------------------------------------
// 1a. Fused Local Sliding-Window RoPE Attention Forward (`D == 64, W <= 16`)
// ---------------------------------------------------------------------------
template <typename scalar_t, int MAX_W>
__global__ void swa_rope_attn_fwd_kernel(
    const scalar_t* __restrict__ qkv,      // (B, T, 3, H, 64)
    const scalar_t* __restrict__ cos_ptr,  // (T, 32)
    const scalar_t* __restrict__ sin_ptr,  // (T, 32)
    scalar_t* __restrict__ out,            // (B, T, H, 64)
    float* __restrict__ lse,               // (B, H, T)
    int64_t T,
    int64_t H,
    int W
) {
    constexpr int BT = 32;
    constexpr int N_WARPS = 8;
    constexpr int BKV_MAX = BT + MAX_W;
    constexpr float scale = 0.125f;

    __shared__ float sh_Ke[BKV_MAX][32];
    __shared__ float sh_Ko[BKV_MAX][32];
    __shared__ float sh_Ve[BKV_MAX][32];
    __shared__ float sh_Vo[BKV_MAX][32];

    const int lane = static_cast<int>(threadIdx.x);
    const int warp_id = static_cast<int>(threadIdx.y);
    const int64_t t0 = static_cast<int64_t>(blockIdx.x) * BT;
    const int64_t h = static_cast<int64_t>(blockIdx.y);
    const int64_t b = static_cast<int64_t>(blockIdx.z);
    const int64_t t_end = (t0 + BT < T) ? (t0 + BT) : T;
    const int64_t kv_start = (t0 - (W - 1) > 0) ? (t0 - (W - 1)) : 0;
    const int kv_count = static_cast<int>(t_end - kv_start);

    for (int row = warp_id; row < kv_count; row += N_WARPS) {
        const int64_t t_kv = kv_start + row;
        const int64_t k_off = (((b * T + t_kv) * 3 + 1) * H + h) * 64;
        const int64_t v_off = (((b * T + t_kv) * 3 + 2) * H + h) * 64;
        float ke, ko, ve, vo;
        load_pair_f32<scalar_t>(qkv + k_off, lane, ke, ko);
        load_pair_f32<scalar_t>(qkv + v_off, lane, ve, vo);
        const int64_t cs_off = t_kv * 32 + lane;
        const float c = static_cast<float>(cos_ptr[cs_off]);
        const float s = static_cast<float>(sin_ptr[cs_off]);
        sh_Ke[row][lane] = ke * c - ko * s;
        sh_Ko[row][lane] = ke * s + ko * c;
        sh_Ve[row][lane] = ve;
        sh_Vo[row][lane] = vo;
    }
    __syncthreads();

    for (int r = warp_id; r < BT; r += N_WARPS) {
        const int64_t t = t0 + r;
        if (t >= T) break;

        const int64_t q_off = (((b * T + t) * 3 + 0) * H + h) * 64;
        float qe, qo;
        load_pair_f32<scalar_t>(qkv + q_off, lane, qe, qo);
        const int64_t cs_off = t * 32 + lane;
        const float c = static_cast<float>(cos_ptr[cs_off]);
        const float s = static_cast<float>(sin_ptr[cs_off]);
        const float qre = qe * c - qo * s;
        const float qro = qe * s + qo * c;

        const int64_t win_start = (t - (W - 1) > 0) ? (t - (W - 1)) : 0;
        const int count = static_cast<int>(t - win_start + 1);
        const int base_row = static_cast<int>(win_start - kv_start);

        float scores[MAX_W];
        float m_max = -1e38f;
        for (int idx = 0; idx < count; ++idx) {
            const int kv_row = base_row + idx;
            const float prod = qre * sh_Ke[kv_row][lane] + qro * sh_Ko[kv_row][lane];
            const float dot = warp_sum32(prod) * scale;
            scores[idx] = dot;
            if (idx == 0 || dot > m_max) m_max = dot;
        }

        float denom = 0.0f;
        for (int idx = 0; idx < count; ++idx) {
            const float p = expf(scores[idx] - m_max);
            scores[idx] = p;
            denom += p;
        }
        const float inv_denom = 1.0f / denom;

        float acc_e = 0.0f;
        float acc_o = 0.0f;
        for (int idx = 0; idx < count; ++idx) {
            const float p = scores[idx] * inv_denom;
            const int kv_row = base_row + idx;
            acc_e += p * sh_Ve[kv_row][lane];
            acc_o += p * sh_Vo[kv_row][lane];
        }

        const int64_t out_off = ((b * T + t) * H + h) * 64;
        store_pair_f32<scalar_t>(out + out_off, lane, acc_e, acc_o);
        if (lane == 0) {
            lse[(b * H + h) * T + t] = m_max + logf(denom);
        }
    }
}

// ---------------------------------------------------------------------------
// 1b. Exact Zero-Atomic Analytical Backward (`D == 64, W <= 16`)
// ---------------------------------------------------------------------------
template <typename scalar_t, int MAX_W>
__global__ void swa_rope_attn_bwd_kernel(
    const scalar_t* __restrict__ qkv,      // (B, T, 3, H, 64)
    const scalar_t* __restrict__ cos_ptr,  // (T, 32)
    const scalar_t* __restrict__ sin_ptr,  // (T, 32)
    const scalar_t* __restrict__ out,      // (B, T, H, 64)
    const float* __restrict__ lse,         // (B, H, T)
    const scalar_t* __restrict__ grad_out, // (B, T, H, 64)
    scalar_t* __restrict__ grad_qkv,       // (B, T, 3, H, 64)
    int64_t T,
    int64_t H,
    int W
) {
    constexpr int BT = 32;
    constexpr int N_WARPS = 8;
    constexpr int R_PER_WARP = BT / N_WARPS;
    constexpr int BKV_MAX = BT + MAX_W;
    constexpr float scale = 0.125f;

    __shared__ float sh_A_e[BKV_MAX][32];
    __shared__ float sh_A_o[BKV_MAX][32];
    __shared__ float sh_B_e[BKV_MAX][32];
    __shared__ float sh_B_o[BKV_MAX][32];
    __shared__ float sh_lse[BKV_MAX];
    __shared__ float sh_D[BKV_MAX];

    const int lane = static_cast<int>(threadIdx.x);
    const int warp_id = static_cast<int>(threadIdx.y);
    const int64_t t0 = static_cast<int64_t>(blockIdx.x) * BT;
    const int64_t h = static_cast<int64_t>(blockIdx.y);
    const int64_t b = static_cast<int64_t>(blockIdx.z);
    const int64_t t_end = (t0 + BT < T) ? (t0 + BT) : T;
    const int64_t kv_start = (t0 - (W - 1) > 0) ? (t0 - (W - 1)) : 0;
    const int kv_count = static_cast<int>(t_end - kv_start);

    // Load K_rot and V for [kv_start, t_end)
    for (int row = warp_id; row < kv_count; row += N_WARPS) {
        const int64_t t_kv = kv_start + row;
        const int64_t k_off = (((b * T + t_kv) * 3 + 1) * H + h) * 64;
        const int64_t v_off = (((b * T + t_kv) * 3 + 2) * H + h) * 64;
        float ke, ko, ve, vo;
        load_pair_f32<scalar_t>(qkv + k_off, lane, ke, ko);
        load_pair_f32<scalar_t>(qkv + v_off, lane, ve, vo);
        const int64_t cs_off = t_kv * 32 + lane;
        const float c = static_cast<float>(cos_ptr[cs_off]);
        const float s = static_cast<float>(sin_ptr[cs_off]);
        sh_A_e[row][lane] = ke * c - ko * s;
        sh_A_o[row][lane] = ke * s + ko * c;
        sh_B_e[row][lane] = ve;
        sh_B_o[row][lane] = vo;
    }
    __syncthreads();

    float my_ke[R_PER_WARP], my_ko[R_PER_WARP];
    float my_ve[R_PER_WARP], my_vo[R_PER_WARP];
    float my_qe[R_PER_WARP], my_qo[R_PER_WARP];
    float my_doe[R_PER_WARP], my_doo[R_PER_WARP];

    // Phase 1: Compute dQ for [t0, t_end) and cache local K_rot, V, Q_rot, dO
    for (int ri = 0; ri < R_PER_WARP; ++ri) {
        const int local_t = warp_id + ri * N_WARPS;
        const int64_t t = t0 + local_t;
        if (t >= T) break;

        const int self_kv_row = static_cast<int>(t - kv_start);
        my_ke[ri] = sh_A_e[self_kv_row][lane];
        my_ko[ri] = sh_A_o[self_kv_row][lane];
        my_ve[ri] = sh_B_e[self_kv_row][lane];
        my_vo[ri] = sh_B_o[self_kv_row][lane];

        const int64_t q_off = (((b * T + t) * 3 + 0) * H + h) * 64;
        const int64_t o_off = ((b * T + t) * H + h) * 64;
        float qe, qo, oe, oo, doe, doo;
        load_pair_f32<scalar_t>(qkv + q_off, lane, qe, qo);
        load_pair_f32<scalar_t>(out + o_off, lane, oe, oo);
        load_pair_f32<scalar_t>(grad_out + o_off, lane, doe, doo);

        const int64_t cs_off = t * 32 + lane;
        const float c = static_cast<float>(cos_ptr[cs_off]);
        const float s = static_cast<float>(sin_ptr[cs_off]);
        const float qre = qe * c - qo * s;
        const float qro = qe * s + qo * c;

        my_qe[ri] = qre;
        my_qo[ri] = qro;
        my_doe[ri] = doe;
        my_doo[ri] = doo;

        const float D_t = warp_sum32(doe * oe + doo * oo);
        const float lse_t = lse[(b * H + h) * T + t];
        if (lane == 0) {
            sh_lse[local_t] = lse_t;
            sh_D[local_t] = D_t;
        }

        const int64_t win_start = (t - (W - 1) > 0) ? (t - (W - 1)) : 0;
        const int count = static_cast<int>(t - win_start + 1);
        const int base_row = static_cast<int>(win_start - kv_start);

        float dq_re = 0.0f;
        float dq_ro = 0.0f;
        for (int idx = 0; idx < count; ++idx) {
            const int kv_row = base_row + idx;
            const float kr_e = sh_A_e[kv_row][lane];
            const float kr_o = sh_A_o[kv_row][lane];
            const float dot = warp_sum32(qre * kr_e + qro * kr_o) * scale;
            const float p = expf(dot - lse_t);
            const float dp = warp_sum32(doe * sh_B_e[kv_row][lane] + doo * sh_B_o[kv_row][lane]);
            const float ds = p * (dp - D_t);
            dq_re += ds * kr_e;
            dq_ro += ds * kr_o;
        }
        dq_re *= scale;
        dq_ro *= scale;
        const float dq_e = dq_re * c + dq_ro * s;
        const float dq_o = -dq_re * s + dq_ro * c;
        store_pair_f32<scalar_t>(grad_qkv + q_off, lane, dq_e, dq_o);
    }
    __syncthreads();

    // Repopulate sh_A = Q_rot, sh_B = dO for [t0, min(T, t0 + BT + W - 1))
    for (int ri = 0; ri < R_PER_WARP; ++ri) {
        const int local_t = warp_id + ri * N_WARPS;
        if (t0 + local_t >= T) break;
        sh_A_e[local_t][lane] = my_qe[ri];
        sh_A_o[local_t][lane] = my_qo[ri];
        sh_B_e[local_t][lane] = my_doe[ri];
        sh_B_o[local_t][lane] = my_doo[ri];
    }
    const int64_t q_end = (t0 + BT + W - 1 < T) ? (t0 + BT + W - 1) : T;
    const int halo_start = static_cast<int>(t_end - t0);
    const int halo_end = static_cast<int>(q_end - t0);
    for (int local_t = halo_start + warp_id; local_t < halo_end; local_t += N_WARPS) {
        const int64_t t = t0 + local_t;
        const int64_t q_off = (((b * T + t) * 3 + 0) * H + h) * 64;
        const int64_t o_off = ((b * T + t) * H + h) * 64;
        float qe, qo, oe, oo, doe, doo;
        load_pair_f32<scalar_t>(qkv + q_off, lane, qe, qo);
        load_pair_f32<scalar_t>(out + o_off, lane, oe, oo);
        load_pair_f32<scalar_t>(grad_out + o_off, lane, doe, doo);
        const int64_t cs_off = t * 32 + lane;
        const float c = static_cast<float>(cos_ptr[cs_off]);
        const float s = static_cast<float>(sin_ptr[cs_off]);
        sh_A_e[local_t][lane] = qe * c - qo * s;
        sh_A_o[local_t][lane] = qe * s + qo * c;
        sh_B_e[local_t][lane] = doe;
        sh_B_o[local_t][lane] = doo;
        const float D_t = warp_sum32(doe * oe + doo * oo);
        if (lane == 0) {
            sh_lse[local_t] = lse[(b * H + h) * T + t];
            sh_D[local_t] = D_t;
        }
    }
    __syncthreads();

    // Phase 2: Compute dK and dV for j in [t0, t_end) (dual causal window j <= t <= min(T-1, j+W-1))
    for (int ri = 0; ri < R_PER_WARP; ++ri) {
        const int local_j = warp_id + ri * N_WARPS;
        const int64_t j = t0 + local_j;
        if (j >= T) break;

        const float kr_e = my_ke[ri];
        const float kr_o = my_ko[ri];
        const float v_e = my_ve[ri];
        const float v_o = my_vo[ri];

        const int64_t t_hi = (j + W - 1 < T - 1) ? (j + W - 1) : (T - 1);
        const int count = static_cast<int>(t_hi - j + 1);

        float dk_re = 0.0f;
        float dk_ro = 0.0f;
        float dv_e = 0.0f;
        float dv_o = 0.0f;
        for (int idx = 0; idx < count; ++idx) {
            const int q_row = local_j + idx;
            const float qr_e = sh_A_e[q_row][lane];
            const float qr_o = sh_A_o[q_row][lane];
            const float do_e = sh_B_e[q_row][lane];
            const float do_o = sh_B_o[q_row][lane];
            const float dot = warp_sum32(qr_e * kr_e + qr_o * kr_o) * scale;
            const float p = expf(dot - sh_lse[q_row]);
            const float dp = warp_sum32(do_e * v_e + do_o * v_o);
            const float ds = p * (dp - sh_D[q_row]);
            dk_re += ds * qr_e;
            dk_ro += ds * qr_o;
            dv_e += p * do_e;
            dv_o += p * do_o;
        }
        dk_re *= scale;
        dk_ro *= scale;
        const int64_t cs_off = j * 32 + lane;
        const float c = static_cast<float>(cos_ptr[cs_off]);
        const float s = static_cast<float>(sin_ptr[cs_off]);
        const float dk_e = dk_re * c + dk_ro * s;
        const float dk_o = -dk_re * s + dk_ro * c;

        const int64_t k_off = (((b * T + j) * 3 + 1) * H + h) * 64;
        const int64_t v_off = (((b * T + j) * 3 + 2) * H + h) * 64;
        store_pair_f32<scalar_t>(grad_qkv + k_off, lane, dk_e, dk_o);
        store_pair_f32<scalar_t>(grad_qkv + v_off, lane, dv_e, dv_o);
    }
}

// ---------------------------------------------------------------------------
// 2. Single-Pass Zero-`argsort` UTF-8 Boundary Guard & Compaction Kernel
// ---------------------------------------------------------------------------
__global__ void utf8_span_and_compact_kernel(
    const int64_t* __restrict__ token_ids,
    const float* __restrict__ p_prob,
    bool* __restrict__ b_mask,
    int64_t* __restrict__ chunk_counts,
    int64_t* __restrict__ slot_idx,
    int64_t* __restrict__ gather_pos,
    bool* __restrict__ valid_mask_out,
    bool* __restrict__ is_lead_out,
    int64_t B,
    int64_t T,
    int64_t M_max,
    int64_t pad_token_id,
    int64_t max_cp_span,
    int64_t max_byte_span,
    int64_t refill_int,
    int64_t cost_int,
    int64_t burst_int
) {
    const int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= B) return;

    int64_t byte_run = 0;
    int64_t cp_run = 0;
    int64_t credits = burst_int;
    int64_t cont_streak = 0;
    int64_t count = 0;
    const int64_t max_slot = (M_max > 0) ? (M_max - 1) : 0;
    const int64_t base = b * T;
    const int64_t g_base = b * M_max;

    for (int64_t t = 0; t < T; ++t) {
        const int64_t idx = base + t;
        const int64_t tok = token_ids[idx];
        const bool vt = (t == 0) || (tok != pad_token_id);
        const bool is_byte = (tok >= 3);
        const int64_t raw_v = is_byte ? (tok - 3) : 0;
        const bool is_cont = is_byte && ((raw_v & 0xC0) == 0x80);
        const bool is_cont_v = is_cont && vt;

        const bool shielded = is_cont_v && (cont_streak < 3);
        const bool can_fire = vt && (!shielded);
        const int64_t ld = (vt && (!is_cont)) ? 1 : 0;
        int64_t cl = 1;
        if (is_byte) {
            if ((raw_v & 0xF8) == 0xF0) {
                cl = 4;
            } else if ((raw_v & 0xF0) == 0xE0) {
                cl = 3;
            } else if ((raw_v & 0xE0) == 0xC0) {
                cl = 2;
            }
        }
        cont_streak = is_cont_v ? (cont_streak + 1) : 0;

        if (valid_mask_out != nullptr) {
            valid_mask_out[idx] = vt;
        }
        if (is_lead_out != nullptr) {
            is_lead_out[idx] = (ld != 0);
        }

        bool fire = false;
        if (t == 0) {
            fire = vt;
        } else {
            if (vt) {
                int64_t cr = credits + refill_int;
                credits = (cr > burst_int) ? burst_int : cr;
            }
            const bool rc = (p_prob[idx] >= 0.5f) && can_fire;
            const bool can_soft = (credits >= cost_int) && rc;
            const bool must_span = (cp_run >= max_cp_span) || ((byte_run + cl) > max_byte_span);
            fire = can_fire && (can_soft || must_span);
            if (fire) {
                int64_t cr = credits - cost_int;
                credits = (cr < 0) ? 0 : cr;
            }
        }

        b_mask[idx] = fire;
        if (fire) {
            if (count < M_max) {
                gather_pos[g_base + count] = t;
            }
            ++count;
        }
        int64_t s = count - 1;
        if (s < 0) s = 0;
        if (s > max_slot) s = max_slot;
        slot_idx[idx] = s;

        if (vt) {
            byte_run = fire ? 1 : (byte_run + 1);
            cp_run = fire ? ld : (cp_run + ld);
        }
    }

    chunk_counts[b] = count;

    if (count < M_max) {
        int64_t c = count;
        for (int64_t t = 0; t < T && c < M_max; ++t) {
            if (!b_mask[base + t]) {
                gather_pos[g_base + c] = t;
                ++c;
            }
        }
        while (c < M_max) {
            gather_pos[g_base + c] = 0;
            ++c;
        }
    }
}

template <typename scalar_t>
void launch_swa_fwd(
    const torch::Tensor& qkv_c,
    const torch::Tensor& cos_c,
    const torch::Tensor& sin_c,
    torch::Tensor& out,
    torch::Tensor& lse,
    int64_t B,
    int64_t T,
    int64_t H,
    int W
) {
    dim3 block(32, 8);
    dim3 grid(static_cast<unsigned int>((T + 31) / 32), static_cast<unsigned int>(H), static_cast<unsigned int>(B));
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (W <= 8) {
        swa_rope_attn_fwd_kernel<scalar_t, 8><<<grid, block, 0, stream>>>(
            qkv_c.data_ptr<scalar_t>(), cos_c.data_ptr<scalar_t>(), sin_c.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(), lse.data_ptr<float>(), T, H, W
        );
    } else {
        swa_rope_attn_fwd_kernel<scalar_t, 16><<<grid, block, 0, stream>>>(
            qkv_c.data_ptr<scalar_t>(), cos_c.data_ptr<scalar_t>(), sin_c.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(), lse.data_ptr<float>(), T, H, W
        );
    }
}

template <typename scalar_t>
void launch_swa_bwd(
    const torch::Tensor& qkv_c,
    const torch::Tensor& cos_c,
    const torch::Tensor& sin_c,
    const torch::Tensor& out_c,
    const torch::Tensor& lse_c,
    const torch::Tensor& go_c,
    torch::Tensor& dqkv,
    int64_t B,
    int64_t T,
    int64_t H,
    int W
) {
    dim3 block(32, 8);
    dim3 grid(static_cast<unsigned int>((T + 31) / 32), static_cast<unsigned int>(H), static_cast<unsigned int>(B));
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (W <= 8) {
        swa_rope_attn_bwd_kernel<scalar_t, 8><<<grid, block, 0, stream>>>(
            qkv_c.data_ptr<scalar_t>(), cos_c.data_ptr<scalar_t>(), sin_c.data_ptr<scalar_t>(),
            out_c.data_ptr<scalar_t>(), lse_c.data_ptr<float>(), go_c.data_ptr<scalar_t>(),
            dqkv.data_ptr<scalar_t>(), T, H, W
        );
    } else {
        swa_rope_attn_bwd_kernel<scalar_t, 16><<<grid, block, 0, stream>>>(
            qkv_c.data_ptr<scalar_t>(), cos_c.data_ptr<scalar_t>(), sin_c.data_ptr<scalar_t>(),
            out_c.data_ptr<scalar_t>(), lse_c.data_ptr<float>(), go_c.data_ptr<scalar_t>(),
            dqkv.data_ptr<scalar_t>(), T, H, W
        );
    }
}

}  // namespace

std::vector<torch::Tensor> swa_rope_attn_fwd_cuda_impl(
    torch::Tensor qkv,
    torch::Tensor cos,
    torch::Tensor sin,
    int64_t window_size
) {
    c10::cuda::CUDAGuard device_guard(qkv.device());
    auto qkv_c = qkv.contiguous();
    auto cos_c = cos.to(qkv_c.scalar_type()).reshape({-1, 32}).contiguous();
    auto sin_c = sin.to(qkv_c.scalar_type()).reshape({-1, 32}).contiguous();
    const int64_t B = qkv_c.size(0);
    const int64_t T = qkv_c.size(1);
    const int64_t H = qkv_c.size(3);
    const int64_t D = qkv_c.size(4);
    const int W = static_cast<int>(window_size);
    TORCH_CHECK(qkv_c.dim() == 5 && qkv_c.size(2) == 3 && D == 64, "swa_rope_attn_fwd_cuda requires (B, T, 3, H, 64)");
    TORCH_CHECK(W >= 1 && W <= 16, "swa_rope_attn_fwd_cuda requires 1 <= window_size <= 16");
    TORCH_CHECK(cos_c.size(0) >= T && sin_c.size(0) >= T, "cos/sin length must be >= T");

    auto out = torch::empty({B, T, H, D}, qkv_c.options());
    auto lse = torch::empty({B, H, T}, qkv_c.options().dtype(torch::kFloat32));
    if (B == 0 || T == 0 || H == 0) return {out, lse};

    if (qkv_c.scalar_type() == torch::kBFloat16) {
        launch_swa_fwd<at::BFloat16>(qkv_c, cos_c, sin_c, out, lse, B, T, H, W);
    } else if (qkv_c.scalar_type() == torch::kFloat32) {
        launch_swa_fwd<float>(qkv_c, cos_c, sin_c, out, lse, B, T, H, W);
    } else if (qkv_c.scalar_type() == torch::kFloat16) {
        launch_swa_fwd<at::Half>(qkv_c, cos_c, sin_c, out, lse, B, T, H, W);
    } else {
        TORCH_CHECK(false, "Unsupported dtype for swa_rope_attn_fwd_cuda");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, lse};
}

torch::Tensor swa_rope_attn_bwd_cuda_impl(
    torch::Tensor qkv,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor out,
    torch::Tensor lse,
    torch::Tensor grad_out,
    int64_t window_size
) {
    c10::cuda::CUDAGuard device_guard(qkv.device());
    auto qkv_c = qkv.contiguous();
    auto cos_c = cos.to(qkv_c.scalar_type()).reshape({-1, 32}).contiguous();
    auto sin_c = sin.to(qkv_c.scalar_type()).reshape({-1, 32}).contiguous();
    auto out_c = out.to(qkv_c.scalar_type()).contiguous();
    auto lse_c = lse.to(torch::kFloat32).contiguous();
    auto go_c = grad_out.to(qkv_c.scalar_type()).contiguous();

    const int64_t B = qkv_c.size(0);
    const int64_t T = qkv_c.size(1);
    const int64_t H = qkv_c.size(3);
    const int64_t D = qkv_c.size(4);
    const int W = static_cast<int>(window_size);
    TORCH_CHECK(qkv_c.dim() == 5 && qkv_c.size(2) == 3 && D == 64, "swa_rope_attn_bwd_cuda requires (B, T, 3, H, 64)");
    TORCH_CHECK(W >= 1 && W <= 16, "swa_rope_attn_bwd_cuda requires 1 <= window_size <= 16");

    auto dqkv = torch::empty_like(qkv_c);
    if (B == 0 || T == 0 || H == 0) return dqkv;

    if (qkv_c.scalar_type() == torch::kBFloat16) {
        launch_swa_bwd<at::BFloat16>(qkv_c, cos_c, sin_c, out_c, lse_c, go_c, dqkv, B, T, H, W);
    } else if (qkv_c.scalar_type() == torch::kFloat32) {
        launch_swa_bwd<float>(qkv_c, cos_c, sin_c, out_c, lse_c, go_c, dqkv, B, T, H, W);
    } else if (qkv_c.scalar_type() == torch::kFloat16) {
        launch_swa_bwd<at::Half>(qkv_c, cos_c, sin_c, out_c, lse_c, go_c, dqkv, B, T, H, W);
    } else {
        TORCH_CHECK(false, "Unsupported dtype for swa_rope_attn_bwd_cuda");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dqkv;
}

std::vector<torch::Tensor> utf8_span_and_compact_cuda_impl(
    torch::Tensor token_ids,
    torch::Tensor p_prob,
    int64_t pad_token_id,
    int64_t max_cp_span,
    int64_t max_byte_span,
    int64_t refill_int,
    int64_t cost_int,
    int64_t burst_int,
    int64_t m_cap,
    bool return_masks
) {
    c10::cuda::CUDAGuard device_guard(token_ids.device());
    auto ids_c = token_ids.to(torch::kInt64).contiguous();
    auto prob_c = p_prob.to(torch::kFloat32).contiguous();
    const int64_t B = ids_c.size(0);
    const int64_t T = ids_c.size(1);
    const int64_t M_max = (m_cap > 0) ? m_cap : T;

    auto bool_opts = ids_c.options().dtype(torch::kBool);
    auto long_opts = ids_c.options().dtype(torch::kInt64);
    auto b_mask = torch::empty({B, T}, bool_opts);
    auto chunk_counts = torch::zeros({B}, long_opts);
    auto slot_idx = torch::empty({B, T}, long_opts);
    auto gather_pos = torch::zeros({B, M_max}, long_opts);
    torch::Tensor valid_mask;
    torch::Tensor is_lead;
    if (return_masks) {
        valid_mask = torch::empty({B, T}, bool_opts);
        is_lead = torch::empty({B, T}, bool_opts);
    }

    if (B > 0 && T > 0) {
        const int threads = 32;
        const int blocks = static_cast<int>((B + threads - 1) / threads);
        utf8_span_and_compact_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
            ids_c.data_ptr<int64_t>(),
            prob_c.data_ptr<float>(),
            b_mask.data_ptr<bool>(),
            chunk_counts.data_ptr<int64_t>(),
            slot_idx.data_ptr<int64_t>(),
            gather_pos.data_ptr<int64_t>(),
            return_masks ? valid_mask.data_ptr<bool>() : nullptr,
            return_masks ? is_lead.data_ptr<bool>() : nullptr,
            B, T, M_max,
            pad_token_id, max_cp_span, max_byte_span, refill_int, cost_int, burst_int
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    if (return_masks) {
        return {b_mask, chunk_counts, slot_idx, gather_pos, valid_mask, is_lead};
    }
    return {b_mask, chunk_counts, slot_idx, gather_pos};
}
"""

_CUDA_BUILD_ERROR: str | None = None


def _has_python_headers() -> bool:
    inc = sysconfig.get_path("include")
    return bool(inc) and (Path(inc) / "Python.h").is_file()


@lru_cache(maxsize=1)
def _load_swa_chunker_cuda_ext():
    global _CUDA_BUILD_ERROR
    if not torch.cuda.is_available():
        _CUDA_BUILD_ERROR = "CUDA device unavailable"
        return None
    if shutil.which("nvcc") is None:
        _CUDA_BUILD_ERROR = "nvcc not found in PATH"
        return None
    if not _has_python_headers():
        _CUDA_BUILD_ERROR = "Python.h dev headers missing"
        return None
    try:
        from torch.utils.cpp_extension import load_inline

        build_dir = Path(tempfile.gettempdir()) / "calcite5_fast_swa_chunker_cuda_v1"
        build_dir.mkdir(parents=True, exist_ok=True)
        mod = load_inline(
            name="calcite5_fast_swa_chunker_cuda_ext_v1",
            cpp_sources=[_CUDA_CPP_SOURCE],
            cuda_sources=[_CUDA_KERNEL_SOURCE],
            functions=[
                "swa_rope_attn_fwd_cuda",
                "swa_rope_attn_bwd_cuda",
                "utf8_span_and_compact_cuda",
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            build_directory=str(build_dir),
            with_cuda=True,
            verbose=False,
        )
        _CUDA_BUILD_ERROR = None
        return mod
    except Exception as exc:
        _CUDA_BUILD_ERROR = f"{type(exc).__name__}: {exc}"
        return None


def _is_compiling() -> bool:
    fn = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    if callable(fn):
        return bool(fn())
    return False


# ---------------------------------------------------------------------------
# 1. Cached RoPE cos/sin + Fused Local Sliding-Window RoPE Attention
# ---------------------------------------------------------------------------

_ROPE_TABLE_CACHE: dict[
    tuple[str, int | None, torch.dtype, int, float], tuple[torch.Tensor, torch.Tensor]
] = {}
_CUSTOM_ROPE_CACHE: dict[
    tuple[int, str, int | None, torch.dtype],
    tuple[weakref.ref, torch.Tensor, torch.Tensor],
] = {}


def _next_rope_cap(seq_len: int) -> int:
    cap = 2048
    while cap < seq_len:
        cap *= 2
    return cap


def _compute_rope_tables(
    length: int,
    device: torch.device,
    dtype: torch.dtype,
    head_dim: int,
    base: float,
    inv_freq: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        if inv_freq is None:
            inv_f = 1.0 / (
                float(base)
                ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / float(head_dim))
            )
        else:
            inv_f = inv_freq.detach().to(device=device, dtype=torch.float32)
        t = torch.arange(length, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_f)
        cos = torch.cos(freqs).to(dtype).contiguous()
        sin = torch.sin(freqs).to(dtype).contiguous()
        return cos, sin


def get_cached_rope_cos_sin(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    head_dim: int = 64,
    base: float = 10000.0,
    inv_freq: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns cached `(cos, sin)` of shape `(seq_len, head_dim // 2)` on `(device, dtype)`
    via zero-copy prefix slices from a pre-allocated table (`>= 2048`).
    `torch.compile(..., dynamic=True)` safe: never caches `FakeTensor`s while tracing.
    """
    compiling = _is_compiling()
    if inv_freq is None:
        key = (device.type, device.index, dtype, int(head_dim), float(base))
        cached = _ROPE_TABLE_CACHE.get(key)
        if cached is not None and cached[0].size(0) >= seq_len:
            return cached[0][:seq_len], cached[1][:seq_len]
        if compiling:
            return _compute_rope_tables(seq_len, device, dtype, int(head_dim), float(base), None)
        cap = _next_rope_cap(int(seq_len))
        cos_t, sin_t = _compute_rope_tables(cap, device, dtype, int(head_dim), float(base), None)
        _ROPE_TABLE_CACHE[key] = (cos_t, sin_t)
        return cos_t[:seq_len], sin_t[:seq_len]

    ckey = (id(inv_freq), device.type, device.index, dtype)
    ccached = _CUSTOM_ROPE_CACHE.get(ckey)
    if ccached is not None and ccached[0]() is inv_freq and ccached[1].size(0) >= seq_len:
        return ccached[1][:seq_len], ccached[2][:seq_len]
    if compiling:
        return _compute_rope_tables(seq_len, device, dtype, int(inv_freq.numel() * 2), float(base), inv_freq)
    cap = _next_rope_cap(int(seq_len))
    cos_t, sin_t = _compute_rope_tables(cap, device, dtype, int(inv_freq.numel() * 2), float(base), inv_freq)
    _CUSTOM_ROPE_CACHE[ckey] = (weakref.ref(inv_freq), cos_t, sin_t)
    return cos_t[:seq_len], sin_t[:seq_len]


def _apply_rope_interleaved_f32(
    x: torch.Tensor, cos_f: torch.Tensor, sin_f: torch.Tensor
) -> torch.Tensor:
    # x: (B, T, H, D) float32, cos_f/sin_f: (1, T, 1, D//2) float32
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out1 = x1 * cos_f - x2 * sin_f
    out2 = x1 * sin_f + x2 * cos_f
    return torch.stack((out1, out2), dim=-1).flatten(-2)


def _apply_rope_interleaved_inv_f32(
    dx: torch.Tensor, cos_f: torch.Tensor, sin_f: torch.Tensor
) -> torch.Tensor:
    # Exact adjoint R_t^T of interleaved RoPE: [c, s; -s, c]
    dx1 = dx[..., 0::2]
    dx2 = dx[..., 1::2]
    out1 = dx1 * cos_f + dx2 * sin_f
    out2 = -dx1 * sin_f + dx2 * cos_f
    return torch.stack((out1, out2), dim=-1).flatten(-2)


def _swa_rope_attn_fwd_pytorch(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Exact O(B * T * H * W * D) sliding-window RoPE attention forward (CPU / no-nvcc fallback).
    Matches the fused CUDA kernel equations identically and returns `(out, lse)`.
    """
    B, T, _, H, D = qkv.shape
    W = int(window_size)
    if B == 0 or T == 0 or H == 0:
        return (
            torch.empty((B, T, H, D), device=qkv.device, dtype=qkv.dtype),
            torch.empty((B, H, T), device=qkv.device, dtype=torch.float32),
        )

    cs_dtype = qkv.dtype
    cos_f = cos.to(cs_dtype).reshape(-1, D // 2)[:T].float().view(1, T, 1, D // 2)
    sin_f = sin.to(cs_dtype).reshape(-1, D // 2)[:T].float().view(1, T, 1, D // 2)

    q_rot = _apply_rope_interleaved_f32(qkv[:, :, 0].float(), cos_f, sin_f)
    k_rot = _apply_rope_interleaved_f32(qkv[:, :, 1].float(), cos_f, sin_f)
    v_f = qkv[:, :, 2].float()

    scale = 1.0 / math.sqrt(float(D))
    k_pad = F.pad(k_rot, (0, 0, 0, 0, W - 1, 0))
    v_pad = F.pad(v_f, (0, 0, 0, 0, W - 1, 0))
    # unfold on dim=1 -> (B, T, H, D, W)
    k_win = k_pad.unfold(1, W, 1)
    v_win = v_pad.unfold(1, W, 1)

    scores = (q_rot.unsqueeze(-1) * k_win).sum(dim=-2) * scale  # (B, T, H, W)
    t_idx = torch.arange(T, device=qkv.device).unsqueeze(1)
    w_idx = torch.arange(W, device=qkv.device).unsqueeze(0)
    valid_w = (t_idx + w_idx) >= (W - 1)  # (T, W)
    scores = scores.masked_fill(~valid_w.view(1, T, 1, W), -1e38)

    lse_th = torch.logsumexp(scores, dim=-1)  # (B, T, H)
    probs = torch.exp(scores - lse_th.unsqueeze(-1))
    out_f = (probs.unsqueeze(-2) * v_win).sum(dim=-1)  # (B, T, H, D)
    return out_f.to(qkv.dtype).contiguous(), lse_th.transpose(1, 2).contiguous()


def _swa_rope_attn_bwd_pytorch(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    grad_out: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """
    Exact analytical O(B * T * H * W * D) backward matching `swa_rope_attn_bwd_kernel`
    equation-for-equation (Flash-style `D_t = <do, o>`, dual causal window for `dK/dV`,
    exact adjoint `R_t^T` inverse RoPE).
    """
    B, T, _, H, D = qkv.shape
    W = int(window_size)
    if B == 0 or T == 0 or H == 0:
        return torch.empty_like(qkv)

    cs_dtype = qkv.dtype
    cos_f = cos.to(cs_dtype).reshape(-1, D // 2)[:T].float().view(1, T, 1, D // 2)
    sin_f = sin.to(cs_dtype).reshape(-1, D // 2)[:T].float().view(1, T, 1, D // 2)

    q_rot = _apply_rope_interleaved_f32(qkv[:, :, 0].float(), cos_f, sin_f)
    k_rot = _apply_rope_interleaved_f32(qkv[:, :, 1].float(), cos_f, sin_f)
    v_f = qkv[:, :, 2].float()
    o_f = out.to(cs_dtype).float()
    do_f = grad_out.to(cs_dtype).float()
    lse_th = lse.float().transpose(1, 2)  # (B, T, H)
    D_t = (do_f * o_f).sum(dim=-1)  # (B, T, H)
    scale = 1.0 / math.sqrt(float(D))

    # Phase 1: dQ over past causal window j in [max(0, t-W+1), t]
    k_pad = F.pad(k_rot, (0, 0, 0, 0, W - 1, 0))
    v_pad = F.pad(v_f, (0, 0, 0, 0, W - 1, 0))
    k_win = k_pad.unfold(1, W, 1)  # (B, T, H, D, W)
    v_win = v_pad.unfold(1, W, 1)  # (B, T, H, D, W)

    t_idx = torch.arange(T, device=qkv.device).unsqueeze(1)
    w_idx = torch.arange(W, device=qkv.device).unsqueeze(0)
    valid_past = ((t_idx + w_idx) >= (W - 1)).view(1, T, 1, W)

    scores = (q_rot.unsqueeze(-1) * k_win).sum(dim=-2) * scale
    probs = torch.where(valid_past, torch.exp(scores - lse_th.unsqueeze(-1)), torch.zeros_like(scores))
    dp = (do_f.unsqueeze(-1) * v_win).sum(dim=-2)
    ds = probs * (dp - D_t.unsqueeze(-1))
    dq_rot = ((ds.unsqueeze(-2) * k_win).sum(dim=-1)) * scale
    dq = _apply_rope_interleaved_inv_f32(dq_rot, cos_f, sin_f)

    # Phase 2: dK, dV over dual future window t in [j, min(T-1, j+W-1)]
    q_pad = F.pad(q_rot, (0, 0, 0, 0, 0, W - 1))
    do_pad = F.pad(do_f, (0, 0, 0, 0, 0, W - 1))
    lse_pad = F.pad(lse_th, (0, 0, 0, W - 1), value=1e38)
    D_pad = F.pad(D_t, (0, 0, 0, W - 1), value=0.0)

    q_win = q_pad.unfold(1, W, 1)      # (B, T, H, D, W)
    do_win = do_pad.unfold(1, W, 1)    # (B, T, H, D, W)
    lse_win = lse_pad.unfold(1, W, 1)  # (B, T, H, W)
    D_win = D_pad.unfold(1, W, 1)      # (B, T, H, W)

    valid_fut = ((t_idx + w_idx) < T).view(1, T, 1, W)
    scores_f = (q_win * k_rot.unsqueeze(-1)).sum(dim=-2) * scale
    probs_f = torch.where(valid_fut, torch.exp(scores_f - lse_win), torch.zeros_like(scores_f))
    dp_f = (do_win * v_f.unsqueeze(-1)).sum(dim=-2)
    ds_f = probs_f * (dp_f - D_win)

    dk_rot = ((ds_f.unsqueeze(-2) * q_win).sum(dim=-1)) * scale
    dk = _apply_rope_interleaved_inv_f32(dk_rot, cos_f, sin_f)
    dv = (probs_f.unsqueeze(-2) * do_win).sum(dim=-1)

    return torch.stack((dq, dk, dv), dim=2).to(qkv.dtype).contiguous()


def _swa_fwd_dispatch(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    W = int(window_size)
    if (
        qkv.is_cuda
        and qkv.dim() == 5
        and qkv.size(2) == 3
        and qkv.size(4) == 64
        and 1 <= W <= 16
        and qkv.dtype in (torch.bfloat16, torch.float32, torch.float16)
    ):
        ext = _load_swa_chunker_cuda_ext()
        if ext is not None:
            out, lse = ext.swa_rope_attn_fwd_cuda(qkv, cos, sin, W)
            return out, lse
    return _swa_rope_attn_fwd_pytorch(qkv, cos, sin, W)


def _swa_bwd_dispatch(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    grad_out: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    W = int(window_size)
    if (
        qkv.is_cuda
        and qkv.dim() == 5
        and qkv.size(2) == 3
        and qkv.size(4) == 64
        and 1 <= W <= 16
        and qkv.dtype in (torch.bfloat16, torch.float32, torch.float16)
    ):
        ext = _load_swa_chunker_cuda_ext()
        if ext is not None:
            return ext.swa_rope_attn_bwd_cuda(qkv, cos, sin, out, lse, grad_out, W)
    return _swa_rope_attn_bwd_pytorch(qkv, cos, sin, out, lse, grad_out, W)


_HAS_CUSTOM_OP = hasattr(torch, "library") and hasattr(torch.library, "custom_op")

if _HAS_CUSTOM_OP:

    @torch.library.custom_op("calcite_swa::swa_rope_attn_fwd", mutates_args=())
    def _swa_rope_attn_fwd_op(
        qkv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        window_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _swa_fwd_dispatch(qkv, cos, sin, window_size)

    @_swa_rope_attn_fwd_op.register_fake
    def _swa_rope_attn_fwd_fake(
        qkv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        window_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cos, sin, window_size
        B, T, _, H, D = qkv.shape
        return (
            torch.empty((B, T, H, D), device=qkv.device, dtype=qkv.dtype),
            torch.empty((B, H, T), device=qkv.device, dtype=torch.float32),
        )

    @torch.library.custom_op("calcite_swa::swa_rope_attn_bwd", mutates_args=())
    def _swa_rope_attn_bwd_op(
        qkv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor,
        grad_out: torch.Tensor,
        window_size: int,
    ) -> torch.Tensor:
        return _swa_bwd_dispatch(qkv, cos, sin, out, lse, grad_out, window_size)

    @_swa_rope_attn_bwd_op.register_fake
    def _swa_rope_attn_bwd_fake(
        qkv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor,
        grad_out: torch.Tensor,
        window_size: int,
    ) -> torch.Tensor:
        del cos, sin, out, lse, grad_out, window_size
        return torch.empty_like(qkv)

    def _swa_setup_context(ctx, inputs, output):
        qkv, cos, sin, window_size = inputs
        out, lse = output
        ctx.save_for_backward(qkv, cos, sin, out, lse)
        ctx.window_size = int(window_size)

    def _swa_backward_bridge(ctx, grad_out: torch.Tensor | None, grad_lse: torch.Tensor | None):
        del grad_lse
        if grad_out is None:
            return None, None, None, None
        qkv, cos, sin, out, lse = ctx.saved_tensors
        dqkv = _swa_rope_attn_bwd_op(
            qkv, cos, sin, out, lse, grad_out.contiguous(), ctx.window_size
        )
        return dqkv, None, None, None

    _swa_rope_attn_fwd_op.register_autograd(
        _swa_backward_bridge, setup_context=_swa_setup_context
    )


class _FastSWARoPEAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        window_size: int,
    ) -> torch.Tensor:
        out, lse = _swa_fwd_dispatch(qkv, cos, sin, int(window_size))
        ctx.save_for_backward(qkv, cos, sin, out, lse)
        ctx.window_size = int(window_size)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        qkv, cos, sin, out, lse = ctx.saved_tensors
        dqkv = _swa_bwd_dispatch(
            qkv, cos, sin, out, lse, grad_out.contiguous(), ctx.window_size
        )
        return dqkv, None, None, None


def fast_swa_rope_attn_cuda(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """
    Fused local sliding-window RoPE attention (`O(T * W)` SMEM CUDA kernel for `W <= 16`,
    `head_dim == 64`, `BF16/FP16/FP32`, with exact analytical backward and CPU/PyTorch fallback).

    Args:
        qkv: `(B, T, 3, H, D)` packed Q/K/V tensor (e.g. `self.qkv(x).view(B, T, 3, H, D)`).
        cos: `(T, D//2)` or `(1, 1, T, D//2)` cosine RoPE table.
        sin: `(T, D//2)` or `(1, 1, T, D//2)` sine RoPE table.
        window_size: Causal sliding window span `W >= 1` (`1 <= W <= 16` uses fused CUDA kernel).

    Returns:
        `out`: `(B, T, H, D)` attention output in `qkv.dtype`.
    """
    if _HAS_CUSTOM_OP:
        out, _ = _swa_rope_attn_fwd_op(qkv, cos, sin, int(window_size))
        return out
    return _FastSWARoPEAttnFn.apply(qkv, cos, sin, int(window_size))


def _is_standard_rope_base_10000(attn: CausalRoPEAttention) -> bool:
    inv = getattr(getattr(attn, "rope", None), "inv_freq", None)
    if inv is None:
        return False
    d = int(attn.head_dim)
    ref = 1.0 / (10000.0 ** (torch.arange(0, d, 2, device=inv.device, dtype=torch.float32) / float(d)))
    return bool(inv.numel() == ref.numel() and torch.allclose(inv.float(), ref, rtol=0.0, atol=1e-7))


def _fast_swa_attn_forward(self_attn: CausalRoPEAttention, x: torch.Tensor) -> torch.Tensor:
    B, T, C = x.shape
    qkv = self_attn.qkv(x).view(B, T, 3, self_attn.n_heads, self_attn.head_dim)
    is_std = bool(getattr(self_attn, "_fast_swa_rope_std", False))
    cos, sin = get_cached_rope_cos_sin(
        T,
        qkv.device,
        qkv.dtype,
        head_dim=self_attn.head_dim,
        base=10000.0,
        inv_freq=None if is_std else self_attn.rope.inv_freq,
    )
    y = fast_swa_rope_attn_cuda(qkv, cos, sin, int(self_attn.window_size))
    return self_attn.out_proj(y.view(B, T, C))


def patch_fast_swa_attention(model: nn.Module, enabled: bool = True) -> int:
    """
    Drop-in helper that patches all local-window `CausalRoPEAttention` layers in `model`
    (such as `cortex_pre` and `cortex_post`) to use `fast_swa_rope_attn_cuda` + cached RoPE
    without modifying `simvq_split/` or changing `state_dict()` / `named_parameters()`.
    Restores original methods cleanly when `enabled=False`.
    """
    patched = getattr(model, "_fast_swa_patched_attns", None)
    if isinstance(patched, list):
        for mod in patched:
            if "_orig_forward_before_fast_swa" in mod.__dict__:
                del mod.__dict__["forward"]
                del mod.__dict__["_orig_forward_before_fast_swa"]
            mod.__dict__.pop("_fast_swa_rope_std", None)
        patched.clear()
    else:
        patched = []
        setattr(model, "_fast_swa_patched_attns", patched)

    if not enabled:
        return 0

    count = 0
    warmed: set[tuple[torch.device, torch.dtype, int]] = set()
    for _, mod in model.named_modules():
        if isinstance(mod, CausalRoPEAttention) and mod.window_size is not None and int(mod.window_size) >= 1:
            is_std = _is_standard_rope_base_10000(mod)
            mod.__dict__["_fast_swa_rope_std"] = is_std
            mod.__dict__["_orig_forward_before_fast_swa"] = True
            mod.forward = types.MethodType(_fast_swa_attn_forward, mod)
            patched.append(mod)
            count += 1
            # Pre-warm RoPE cache outside torch.compile tracing (supports nn.Linear & PackedLUT2BitLinear)
            ref_t = getattr(mod.qkv, "weight", None)
            if ref_t is None:
                ref_t = getattr(mod.qkv, "packed_u8", None)
            if ref_t is None:
                ref_t = next(mod.buffers(), None)
            if ref_t is not None:
                dev = ref_t.device
                base_dt = ref_t.dtype if ref_t.is_floating_point() else torch.float32
                dts = (base_dt, torch.bfloat16) if dev.type == "cuda" else (base_dt,)
                for dt in dts:
                    wkey = (dev, dt, int(mod.head_dim))
                    if is_std and wkey not in warmed:
                        get_cached_rope_cos_sin(2048, dev, dt, head_dim=int(mod.head_dim), base=10000.0)
                        warmed.add(wkey)
    return count


# ---------------------------------------------------------------------------
# 2. Zero-`argsort` UTF-8 Span + Compaction & Locked-Router Fast Encode
# ---------------------------------------------------------------------------


def _compute_utf8_static_m_cap(
    T: int,
    max_cp_span: int,
    max_byte_span: int,
    refill_int: int,
    cost_int: int,
    burst_int: int,
) -> int:
    if T <= 1:
        return max(1, int(T))
    k_paid = min(T - 1, (int(burst_int) + int(refill_int) * (T - 1)) // max(1, int(cost_int)))
    s_min = min(int(max_cp_span), max(1, int(max_byte_span) - 3))
    m_cap = 1 + k_paid + ((T - 1 - k_paid) // max(1, s_min))
    return min(int(T), int(m_cap))


def _compact_b_mask_zero_argsort_pytorch(
    b_mask: torch.Tensor, m_cap: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Exact O(B * T) zero-`argsort` compaction in pure PyTorch matching
    `rank_key.argsort(dim=1)[:, :m_cap]` bit-for-bit on both valid and padded tail slots.
    """
    B, T = b_mask.shape
    M_max = int(m_cap)
    dev = b_mask.device
    if B == 0 or T == 0:
        return (
            torch.zeros((B,), device=dev, dtype=torch.long),
            torch.empty((B, T), device=dev, dtype=torch.long),
            torch.zeros((B, M_max), device=dev, dtype=torch.long),
        )

    b_l = b_mask.long()
    csum_t = b_l.cumsum(dim=1)
    chunk_counts = csum_t[:, -1]
    slot_idx = (csum_t - 1).clamp(min=0, max=max(0, M_max - 1))

    csum_f = (~b_mask).long().cumsum(dim=1)
    dest = torch.where(b_mask, csum_t - 1, chunk_counts.unsqueeze(1) + (csum_f - 1))
    t_idx = torch.arange(T, device=dev, dtype=torch.long).unsqueeze(0).expand(B, T)
    gather_full = torch.empty((B, T), device=dev, dtype=torch.long)
    gather_full.scatter_(1, dest, t_idx)
    if M_max <= T:
        gather_pos = gather_full[:, :M_max].contiguous()
    else:
        gather_pos = F.pad(gather_full, (0, M_max - T), value=0).contiguous()
    return chunk_counts, slot_idx, gather_pos


def _utf8_span_and_compact_full(
    token_ids: torch.Tensor,
    p_prob: torch.Tensor,
    pad_token_id: int,
    max_cp_span: int,
    max_byte_span: int,
    refill_int: int,
    cost_int: int,
    burst_int: int,
    m_cap: int,
    return_masks: bool,
) -> tuple[torch.Tensor, ...]:
    if token_ids.is_cuda:
        ext = _load_swa_chunker_cuda_ext()
        if ext is not None:
            res = ext.utf8_span_and_compact_cuda(
                token_ids,
                p_prob,
                int(pad_token_id),
                int(max_cp_span),
                int(max_byte_span),
                int(refill_int),
                int(cost_int),
                int(burst_int),
                int(m_cap),
                bool(return_masks),
            )
            return tuple(res)

    # Exact CPU / no-nvcc fallback (zero `argsort`)
    _, T = token_ids.shape
    valid_mask = token_ids != int(pad_token_id)
    if T > 0:
        valid_mask[:, 0] = True
    _, is_cont, char_len = _utf8_tensor_flags(token_ids)
    is_cont_v = is_cont & valid_mask
    c1 = F.pad(is_cont_v[:, :-1], (1, 0), value=False) if T > 1 else torch.zeros_like(is_cont_v)
    c2 = F.pad(is_cont_v[:, :-2], (2, 0), value=False) if T > 2 else torch.zeros_like(is_cont_v)
    c3 = F.pad(is_cont_v[:, :-3], (3, 0), value=False) if T > 3 else torch.zeros_like(is_cont_v)
    shielded = is_cont_v & (~(c1 & c2 & c3))
    can_fire_mask = valid_mask & (~shielded)
    is_lead_b = valid_mask & (~is_cont)
    raw_cand = (p_prob.float() >= 0.5) & can_fire_mask
    b_mask = _enforce_utf8_chunk_span_jit(
        raw_cand,
        valid_mask,
        can_fire_mask,
        is_lead_b.long(),
        char_len,
        int(max_cp_span),
        int(max_byte_span),
        int(refill_int),
        int(cost_int),
        int(burst_int),
    )
    chunk_counts, slot_idx, gather_pos = _compact_b_mask_zero_argsort_pytorch(b_mask, int(m_cap))
    if return_masks:
        return b_mask, chunk_counts, slot_idx, gather_pos, valid_mask, is_lead_b
    return b_mask, chunk_counts, slot_idx, gather_pos


def utf8_span_and_compact_cuda(
    token_ids: torch.Tensor,
    p_prob: torch.Tensor,
    pad_token_id: int = 0,
    max_cp_span: int = 10,
    max_byte_span: int = 16,
    refill_int: int = 3,
    cost_int: int = 10,
    burst_int: int = 250,
    m_cap: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single-pass UTF-8 continuation-byte shield + integer leaky-bucket boundary guard +
    chunk compaction returning `(b_mask, chunk_counts, slot_idx, gather_pos)` with ZERO `argsort`.
    """
    T = int(token_ids.size(1))
    resolved_cap = (
        int(m_cap)
        if m_cap is not None
        else _compute_utf8_static_m_cap(
            T, max_cp_span, max_byte_span, refill_int, cost_int, burst_int
        )
    )
    b_mask, chunk_counts, slot_idx, gather_pos = _utf8_span_and_compact_full(
        token_ids,
        p_prob,
        int(pad_token_id),
        int(max_cp_span),
        int(max_byte_span),
        int(refill_int),
        int(cost_int),
        int(burst_int),
        resolved_cap,
        return_masks=False,
    )
    return b_mask, chunk_counts, slot_idx, gather_pos


def _fast_locked_r_lin(lin: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Executes a frozen router Linear directly using its ternary-aligned weights
    (skipping redundant `quantize_ternary_ste` while preserving `router_max_abs_diff == 0.0`).
    """
    if isinstance(lin, nn.Linear):
        w = lin.weight.to(x.dtype)
        b = lin.bias.to(x.dtype) if lin.bias is not None else None
        return F.linear(x, w, b)
    return lin(x)


def _fast_locked_router_probs(
    chunker: SurpriseHNetChunker,
    token_ids: torch.Tensor,
    work_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T = token_ids.shape
    log_v = math.log(max(2, chunker.vocab_size))
    rc = chunker.router_emb(token_ids).to(work_dtype)
    rc_pad = F.pad(rc, (0, 0, 3, 0))
    r_win4 = torch.cat(
        [rc_pad[:, 3 - k : 3 - k + T, :] for k in range(4)], dim=-1
    ).view(B, T, 4 * chunker.d_rchar)
    h_r = F.silu(_fast_locked_r_lin(chunker.router_in_proj, chunker.router_in_norm(r_win4)))

    q_r = F.normalize(_fast_locked_r_lin(chunker.router_q, h_r).float(), dim=-1)
    k_r = F.normalize(_fast_locked_r_lin(chunker.router_k, h_r).float(), dim=-1)
    d_cos = torch.ones(B, T, device=token_ids.device, dtype=torch.float32)
    if T > 1:
        d_cos[:, 1:] = 0.5 * (1.0 - (q_r[:, 1:] * k_r[:, :-1]).sum(dim=-1))

    loc_logits = _fast_locked_r_lin(chunker.router_ent_head, chunker.router_ent_norm(h_r)).float()
    log_p = F.log_softmax(loc_logits, dim=-1)
    h_ent_norm = (-(log_p.exp() * log_p).sum(dim=-1) / log_v).clamp(0.0, 1.5)
    dh_pos = torch.zeros_like(h_ent_norm)
    if T > 1:
        dh_pos[:, 1:] = F.relu(h_ent_norm[:, 1:] - h_ent_norm[:, :-1])

    wf = chunker.router_feat_weights.float()
    ctx_logit = _fast_locked_r_lin(chunker.router_ctx, h_r).float().squeeze(-1)
    p_prob = torch.sigmoid(
        wf[0] * d_cos + wf[1] * h_ent_norm + wf[2] * dh_pos + ctx_logit
    )
    return p_prob, loc_logits


def fast_encode_to_chunks(
    chunker: SurpriseHNetChunker,
    e_tok: torch.Tensor,
    token_ids: torch.Tensor,
    tok_emb_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, ByteChunkContext]:
    """
    Drop-in accelerated `SurpriseHNetChunker.encode_to_chunks`:
    1. Skips redundant `quantize_ternary_ste` on all 5 router linears when `chunker.router_locked`
       is `True` (`router_max_abs_diff == 0.0`).
    2. Fuses UTF-8 boundary guard + compaction in a single pass with ZERO `argsort`.
    3. Gathers `c_win8` `(B, M_max, 8*d_byte)` BEFORE `lex_norm`/`lex_up` (exact fwd+bwd parity,
       cutting `lex_up` compute/VRAM `2.5x–4x` and eliminating `(B, T, D)` `h_lex_byte`).
    4. Uses 2D `index_select` on flattened `(B*T, D)` instead of 3D expanded-index `torch.gather`.
    """
    del tok_emb_weight
    B, T, D = e_tok.shape

    c_win8 = chunker._causal_win8(e_tok)
    c_norm = chunker.enc_norm_in(c_win8)
    h_enc0 = e_tok + chunker.enc_up(c_norm)
    h_byte = h_enc0 + chunker.enc_mlp(chunker.enc_norm(h_enc0))

    if chunker.router_locked:
        with torch.no_grad():
            p_prob, loc_logits = _fast_locked_router_probs(chunker, token_ids, e_tok.dtype)
        p_prob = p_prob.detach()
    else:
        p_prob, loc_logits = chunker._router_probs(token_ids, e_tok.dtype)

    if chunker.byte_chunk_mode == "utf8":
        static_cap = chunker.static_m_cap_for_T(T)
        alloc_cap = (
            static_cap
            if chunker.use_static_m_cap
            else (min(T, ((static_cap + 31) // 32) * 32) if T >= 32 else static_cap)
        )
        (
            b_mask,
            chunk_counts,
            slot_idx,
            gather_pos,
            valid_mask,
            is_lead_b,
        ) = _utf8_span_and_compact_full(
            token_ids,
            p_prob,
            chunker.pad_token_id,
            chunker.max_chunk_bytes,
            chunker.max_utf8_bytes,
            chunker.refill_int,
            chunker.cost_int,
            chunker.burst_int,
            alloc_cap,
            return_masks=True,
        )
        if chunker.use_static_m_cap:
            M_max = static_cap
        else:
            raw_m = int(chunk_counts.max().item()) if B > 0 else 0
            M_max = min(T, ((raw_m + 31) // 32) * 32) if T >= 32 else raw_m
            if M_max < gather_pos.size(1):
                gather_pos = gather_pos[:, :M_max].contiguous()
                slot_idx = slot_idx.clamp(max=max(0, M_max - 1))
            elif M_max > gather_pos.size(1):
                chunk_counts, slot_idx, gather_pos = _compact_b_mask_zero_argsort_pytorch(
                    b_mask, M_max
                )
        ratio_mask = is_lead_b[:, 1:] if T > 1 else valid_mask[:, 1:]
    else:
        valid_mask = token_ids != chunker.pad_token_id
        if T > 0:
            valid_mask[:, 0] = True
        raw_b = (p_prob >= 0.5) & valid_mask
        if T > 0:
            raw_b[:, 0] = True
        b_mask = enforce_max_chunk_span_vec(raw_b, valid_mask, chunker.max_chunk_bytes)
        ratio_mask = valid_mask[:, 1:]
        if chunker.use_static_m_cap:
            M_max = chunker.static_m_cap_for_T(T)
        else:
            M_max = int(b_mask.long().sum(dim=1).max().item()) if B > 0 else 0
        chunk_counts, slot_idx, gather_pos = _compact_b_mask_zero_argsort_pytorch(b_mask, M_max)

    if T > 1:
        ratio_mask_f = ratio_mask.float()
        r_sum = ratio_mask_f.sum()
        has_valid = r_sum > 0.0
        denom_r = r_sum.clamp(min=1.0)
        f_hard = ((b_mask[:, 1:].float() * ratio_mask_f).sum() / denom_r).detach()
        ratio_t = 1.0 / f_hard.clamp(min=1e-4)
        chunker._last_chunk_ratio_t = torch.where(
            has_valid, ratio_t, torch.ones_like(ratio_t)
        )
        if not chunker.router_locked:
            g_soft = (p_prob[:, 1:] * ratio_mask_f).sum() / denom_r
            n_comp = max(1.5, float(chunker.target_chunk_bytes))
            l_ratio = (n_comp / (n_comp - 1.0)) * (
                (n_comp - 1.0) * f_hard * g_soft + (1.0 - f_hard) * (1.0 - g_soft)
            )
            ent_mask_f = valid_mask[:, 1:].float()
            ent_per_tok = F.cross_entropy(
                loc_logits[:, :-1].reshape(-1, chunker.vocab_size),
                token_ids[:, 1:].reshape(-1),
                reduction="none",
            ).view(B, T - 1)
            l_ent = (ent_per_tok * ent_mask_f).sum() / ent_mask_f.sum().clamp(min=1.0)
            chunker.last_ratio_loss = torch.where(
                has_valid, l_ratio + 0.25 * l_ent, 0.0 * p_prob.sum()
            )
        else:
            chunker.last_ratio_loss = None
    else:
        chunker._last_chunk_ratio_val = 1.0
        chunker._last_chunk_ratio_t = None
        chunker.last_ratio_loss = None if chunker.router_locked else (0.0 * p_prob.sum())

    chunker.last_m_cap_used = int(M_max)

    valid_chunk = (
        torch.arange(M_max, device=e_tok.device, dtype=torch.long).unsqueeze(0)
        < chunk_counts.unsqueeze(1)
    )
    chunk_mask_f = valid_chunk.unsqueeze(-1).to(e_tok.dtype)

    row_off = (torch.arange(B, device=e_tok.device, dtype=torch.long) * T).unsqueeze(1)
    flat_pos = (gather_pos + row_off).reshape(-1)
    z_in = (
        h_byte.contiguous().view(B * T, D).index_select(0, flat_pos).view(B, M_max, D)
        * chunk_mask_f
    )
    c_win8_dim = c_win8.size(-1)
    c_win8_chunk = (
        c_win8.contiguous()
        .view(B * T, c_win8_dim)
        .index_select(0, flat_pos)
        .view(B, M_max, c_win8_dim)
    )
    h_lex_chunk = chunker.lex_up(chunker.lex_norm(c_win8_chunk)) * chunk_mask_f
    p_chunk = torch.gather(p_prob, 1, gather_pos) * valid_chunk.float()

    ctx = ByteChunkContext(
        b_mask=b_mask,
        valid_mask=valid_mask,
        p_prob=p_prob,
        p_chunk=p_chunk,
        h_byte=h_byte,
        c_win8=c_norm,
        slot_idx=slot_idx,
        valid_chunk=valid_chunk,
    )
    return z_in, h_lex_chunk, ctx


def patch_fast_chunker(model: nn.Module, enabled: bool = True) -> int:
    """
    Drop-in helper that binds `fast_encode_to_chunks` onto `model.byte_chunker.encode_to_chunks`
    without modifying `simvq_split/` or `state_dict()`. Restores cleanly when `enabled=False`.
    """
    bc = getattr(model, "byte_chunker", None)
    if bc is None or not isinstance(bc, SurpriseHNetChunker):
        return 0
    if "_orig_encode_before_fast_chunker" in bc.__dict__:
        del bc.__dict__["encode_to_chunks"]
        del bc.__dict__["_orig_encode_before_fast_chunker"]
    if not enabled:
        return 0
    bc.__dict__["_orig_encode_before_fast_chunker"] = True
    bc.encode_to_chunks = types.MethodType(fast_encode_to_chunks, bc)
    return 1


def get_fast_swa_chunker_backend_info() -> dict[str, object]:
    ext = _load_swa_chunker_cuda_ext()
    return {
        "cuda_available": bool(torch.cuda.is_available()),
        "nvcc_found": shutil.which("nvcc") is not None,
        "cuda_ext_available": ext is not None,
        "cuda_build_error": _CUDA_BUILD_ERROR,
        "custom_op_registered": _HAS_CUSTOM_OP,
    }


__all__ = [
    "fast_encode_to_chunks",
    "fast_swa_rope_attn_cuda",
    "get_cached_rope_cos_sin",
    "get_fast_swa_chunker_backend_info",
    "patch_fast_chunker",
    "patch_fast_swa_attention",
    "utf8_span_and_compact_cuda",
]
