"""
fast_packed_lut.py — Ускоренный движок физического 2-битного умножения `PackedLUT2BitLinear`
без умножений весов (`T-MAC base-3 LUT`):

Три ортогональных бэкенда с единым контрактом `FastPackedLUT2BitLinear(PackedLUT2BitLinear)`:
  1. `cuda_ext` (`torch.utils.cpp_extension.load_inline`, `nvcc`, stream-safe для CUDA Graphs):
     - `n_tot > 4`: тайловое SMEM-ядро `BM=32 x BN=64` (`42.6 KB` SMEM) с `uint32_t` загрузками.
     - `n_tot <= 4`: варп-кооперативное GEMV-ядро `tmac_lut_gemv_kernel` (`9.0 KB` SMEM, 32/32
       активных потоков в варпе, сплит `9+9` LUT без целочисленного деления, L2-резидентные веса).
  2. `cpu_ext` (`torch.utils.cpp_extension.load_inline`, C++ AVX-512 / AVX2 / OpenMP):
     - Автопоиск VS2022 `vcvars64.bat`/`cl.exe` на Windows (`/O2 /openmp:llvm /arch:AVX512 /fp:fast`
       с безопасным фоллбеком на `/arch:AVX2` и `/openmp`).
     - In-register 9+9 T-MAC (`_mm512_permutexvar_ps` / `_mm256_permutevar8x32_ps`) над компактным
       L3-резидентным 16-строчным блоком `uint8` (`6.55 MB` для `25M` против `52.4 MB` `int64` кэша),
       нулевые аллокации на вызов (`thread_local` буфер), адаптивный OpenMP-порог без оверподписки `B=1`.
  3. `pytorch_vec` (векторизованный PyTorch без внешнего компилятора, `n_blk=512` для 4x меньшего VRAM).
  4. `fuse_packed_swiglu_for_inference`: opt-in слияние `w_gate + w_up` упакованных `SwiGLU` для инференса.
"""

from __future__ import annotations

import copy
from functools import lru_cache
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F

from simvq_split.quant_lut import (
    PackedLUT2BitLinear,
    compute_2bit_lut_footprint,
    pack_model_2bit_lut,
)


_CUDA_CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor tmac_lut_cuda_forward_impl(
    torch::Tensor x,
    torch::Tensor packed_u8,
    torch::Tensor scales_fp16,
    torch::Tensor byte_scales_fp16,
    torch::Tensor bias_fp16,
    int64_t out_features,
    int64_t in_features,
    int64_t in_pad,
    int64_t k_bytes,
    bool row_aligned,
    bool has_bias
);

torch::Tensor tmac_lut_cuda_forward(
    torch::Tensor x,
    torch::Tensor packed_u8,
    torch::Tensor scales_fp16,
    torch::Tensor byte_scales_fp16,
    torch::Tensor bias_fp16,
    int64_t out_features,
    int64_t in_features,
    int64_t in_pad,
    int64_t k_bytes,
    bool row_aligned,
    bool has_bias
) {
    return tmac_lut_cuda_forward_impl(
        x, packed_u8, scales_fp16, byte_scales_fp16, bias_fp16,
        out_features, in_features, in_pad, k_bytes, row_aligned, has_bias
    );
}
"""

_CUDA_KERNEL_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

__device__ __forceinline__ float warp_sum32(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    }
    return v;
}

__device__ __forceinline__ void fill_lut9(float a, float b, float* dst) {
    const float v0 = -b;
    const float v2 = b;
    dst[0] = v0 - a;
    dst[1] = v0;
    dst[2] = v0 + a;
    dst[3] = -a;
    dst[4] = 0.0f;
    dst[5] = a;
    dst[6] = v2 - a;
    dst[7] = v2;
    dst[8] = v2 + a;
}

// ---------------------------------------------------------------------------
// 1. Batched Tile Kernel (`n_tot > 4`): BM=32 x BN=64, 42.6 KB SMEM
// ---------------------------------------------------------------------------
template <bool ROW_ALIGNED>
__global__ void tmac_lut_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ packed_u8,
    const __half* __restrict__ scales_fp16,
    const __half* __restrict__ byte_scales_fp16,
    const __half* __restrict__ bias_fp16,
    float* __restrict__ out,
    int64_t n_tot,
    int64_t out_features,
    int64_t in_pad,
    int64_t k_bytes,
    int64_t g32,
    bool has_bias
) {
    __shared__ float smem_lut[4][81][32];
    __shared__ float sx[32][17];

    const int tx = static_cast<int>(threadIdx.x);  // 0..31 (token lane in warp)
    const int ty = static_cast<int>(threadIdx.y);  // 0..7  (warp id)
    const int tid = (ty << 5) | tx;                // 0..255

    const int64_t m_base = static_cast<int64_t>(blockIdx.x) * 32;
    const int64_t o_base = static_cast<int64_t>(blockIdx.y) * 64;
    const int64_t m = m_base + tx;

    float acc[8];
    float grp_acc[8];
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        acc[r] = 0.0f;
        grp_acc[r] = 0.0f;
    }

    const int lut_j = ty & 3;
    const int lut_half = ty >> 2;

    for (int64_t b_base = 0; b_base < k_bytes; b_base += 4) {
        const int64_t rem_b = k_bytes - b_base;
        const int nb_bytes = rem_b >= 4 ? 4 : static_cast<int>(rem_b);
        const int k_limit = nb_bytes * 4;
        const int64_t in_col_base = b_base * 4;

        // 1. Coalesced global -> shared load of 32 tokens x 16 activations
        {
            const int m0 = tid >> 4;
            const int k0 = tid & 15;
            const int64_t gm0 = m_base + m0;
            sx[m0][k0] = (gm0 < n_tot && k0 < k_limit)
                ? x[gm0 * in_pad + in_col_base + k0]
                : 0.0f;

            const int m1 = 16 + m0;
            const int64_t gm1 = m_base + m1;
            sx[m1][k0] = (gm1 < n_tot && k0 < k_limit)
                ? x[gm1 * in_pad + in_col_base + k0]
                : 0.0f;
        }
        __syncthreads();

        // 2. Addition/subtraction-only 81-entry base-3 T-MAC LUT synthesis (zero bank conflicts)
        if (lut_j < nb_bytes) {
            const float x0 = sx[tx][lut_j * 4 + 0];
            const float x1 = sx[tx][lut_j * 4 + 1];
            const float x2 = sx[tx][lut_j * 4 + 2];
            const float x3 = sx[tx][lut_j * 4 + 3];

            for (int p32 = lut_half; p32 < 9; p32 += 2) {
                const int d3 = (p32 < 3) ? 0 : ((p32 < 6) ? 1 : 2);
                const int d2 = p32 - 3 * d3;
                const float v32 = (d3 == 0 ? -x3 : (d3 == 2 ? x3 : 0.0f))
                                + (d2 == 0 ? -x2 : (d2 == 2 ? x2 : 0.0f));
                const int base9 = p32 * 9;

                const float v0 = v32 - x1;
                smem_lut[lut_j][base9 + 0][tx] = v0 - x0;
                smem_lut[lut_j][base9 + 1][tx] = v0;
                smem_lut[lut_j][base9 + 2][tx] = v0 + x0;

                smem_lut[lut_j][base9 + 3][tx] = v32 - x0;
                smem_lut[lut_j][base9 + 4][tx] = v32;
                smem_lut[lut_j][base9 + 5][tx] = v32 + x0;

                const float v2 = v32 + x1;
                smem_lut[lut_j][base9 + 6][tx] = v2 - x0;
                smem_lut[lut_j][base9 + 7][tx] = v2;
                smem_lut[lut_j][base9 + 8][tx] = v2 + x0;
            }
        }
        __syncthreads();

        // 3. Gather LUT entries across 64 output channels (uniform per warp -> zero SMEM bank conflicts)
        if (ROW_ALIGNED) {
            const bool flush_group = ((b_base & 4) != 0);
            const int64_t g_idx = b_base >> 3;
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int64_t o = o_base + r * 8 + ty;
                if (o < out_features) {
                    const uint32_t code4 = *reinterpret_cast<const uint32_t*>(
                        packed_u8 + o * k_bytes + b_base
                    );
                    const uint32_t c0 = min(code4 & 0xFFu, 80u);
                    const uint32_t c1 = min((code4 >> 8) & 0xFFu, 80u);
                    const uint32_t c2 = min((code4 >> 16) & 0xFFu, 80u);
                    const uint32_t c3 = min((code4 >> 24) & 0xFFu, 80u);
                    grp_acc[r] += smem_lut[0][c0][tx]
                                + smem_lut[1][c1][tx]
                                + smem_lut[2][c2][tx]
                                + smem_lut[3][c3][tx];
                    if (flush_group) {
                        const float sc = __half2float(scales_fp16[o * g32 + g_idx]);
                        acc[r] += grp_acc[r] * sc;
                        grp_acc[r] = 0.0f;
                    }
                }
            }
        } else {
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int64_t o = o_base + r * 8 + ty;
                if (o < out_features) {
                    const int64_t row_off = o * k_bytes + b_base;
                    float sum_r = 0.0f;
                    for (int j = 0; j < nb_bytes; ++j) {
                        const uint32_t cj = min(static_cast<uint32_t>(packed_u8[row_off + j]), 80u);
                        const float sj = __half2float(byte_scales_fp16[row_off + j]);
                        sum_r += smem_lut[j][cj][tx] * sj;
                    }
                    acc[r] += sum_r;
                }
            }
        }
        __syncthreads();
    }

    if (m < n_tot) {
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int64_t o = o_base + r * 8 + ty;
            if (o < out_features) {
                float v = acc[r];
                if (has_bias) {
                    v += __half2float(bias_fp16[o]);
                }
                out[m * out_features + o] = v;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Warp-Cooperative Low-Batch GEMV Kernel (`n_tot <= 4`): 9.0 KB SMEM,
//    32/32 active lanes per warp, coalesced uint32_t loads, zero idiv.
// ---------------------------------------------------------------------------
template <bool ROW_ALIGNED>
__global__ void tmac_lut_gemv_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ packed_u8,
    const __half* __restrict__ scales_fp16,
    const __half* __restrict__ byte_scales_fp16,
    const __half* __restrict__ bias_fp16,
    float* __restrict__ out,
    int64_t out_features,
    int64_t in_pad,
    int64_t k_bytes,
    int64_t g32,
    bool has_bias
) {
    // 128 bytes/tile = 16 groups of 32 weights = 512 input features per tile.
    // Static SMEM = 2 * 128 * 9 * 4 = 9,216 B (9.0 KB <= 48 KB), stride 9 is odd (0 bank conflicts).
    __shared__ float sh_lut01[128][9];
    __shared__ float sh_lut23[128][9];

    const int tx = static_cast<int>(threadIdx.x);  // 0..31
    const int ty = static_cast<int>(threadIdx.y);  // 0..7 (8 warps = 8 output rows per block)
    const int tid = (ty << 5) | tx;                // 0..255
    const int64_t o = static_cast<int64_t>(blockIdx.x) * 8 + ty;
    const int64_t m = static_cast<int64_t>(blockIdx.y);
    const float* xm = x + m * in_pad;

    float lane_acc = 0.0f;

    for (int64_t b_base = 0; b_base < k_bytes; b_base += 128) {
        const int64_t rem = k_bytes - b_base;
        const int nb = rem >= 128 ? 128 : static_cast<int>(rem);

        if (tid < nb) {
            const float* xp = xm + (b_base + tid) * 4;
            fill_lut9(xp[0], xp[1], sh_lut01[tid]);
            fill_lut9(xp[2], xp[3], sh_lut23[tid]);
        }
        __syncthreads();

        if (o < out_features) {
            if constexpr (ROW_ALIGNED) {
                const int ng = nb >> 3;
                const int64_t g_base = b_base >> 3;
                if (tx < (ng << 1)) {
                    const int g_loc = tx >> 1;
                    const int b_loc = tx << 2;
                    const uint32_t code4 = *reinterpret_cast<const uint32_t*>(
                        packed_u8 + o * k_bytes + b_base + b_loc
                    );
                    float s4 = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const uint32_t cj = min((code4 >> (j << 3)) & 0xFFu, 80u);
                        const uint32_t hi = (cj * 57u) >> 9;
                        const uint32_t lo = cj - hi * 9u;
                        s4 += sh_lut01[b_loc + j][lo] + sh_lut23[b_loc + j][hi];
                    }
                    const float sc = __half2float(scales_fp16[o * g32 + g_base + g_loc]);
                    lane_acc += s4 * sc;
                }
            } else {
                const int64_t row_off = o * k_bytes + b_base;
                for (int j = tx; j < nb; j += 32) {
                    const uint32_t cj = min(static_cast<uint32_t>(packed_u8[row_off + j]), 80u);
                    const uint32_t hi = (cj * 57u) >> 9;
                    const uint32_t lo = cj - hi * 9u;
                    const float sj = __half2float(byte_scales_fp16[row_off + j]);
                    lane_acc += (sh_lut01[j][lo] + sh_lut23[j][hi]) * sj;
                }
            }
        }
        __syncthreads();
    }

    if (o < out_features) {
        float total = warp_sum32(lane_acc);
        if (tx == 0) {
            if (has_bias) {
                total += __half2float(bias_fp16[o]);
            }
            out[m * out_features + o] = total;
        }
    }
}

}  // namespace

torch::Tensor tmac_lut_cuda_forward_impl(
    torch::Tensor x,
    torch::Tensor packed_u8,
    torch::Tensor scales_fp16,
    torch::Tensor byte_scales_fp16,
    torch::Tensor bias_fp16,
    int64_t out_features,
    int64_t in_features,
    int64_t in_pad,
    int64_t k_bytes,
    bool row_aligned,
    bool has_bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == in_pad, "x shape mismatch vs in_pad");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(packed_u8.is_cuda() && packed_u8.device() == x.device(), "packed_u8 device mismatch");
    TORCH_CHECK(packed_u8.scalar_type() == torch::kUInt8 && packed_u8.is_contiguous(), "packed_u8 must be contiguous uint8");
    TORCH_CHECK(packed_u8.numel() >= out_features * k_bytes, "packed_u8 numel too small");
    TORCH_CHECK(scales_fp16.is_cuda() && scales_fp16.device() == x.device(), "scales_fp16 device mismatch");
    TORCH_CHECK(scales_fp16.scalar_type() == torch::kHalf && scales_fp16.is_contiguous(), "scales_fp16 must be contiguous float16");
    TORCH_CHECK(byte_scales_fp16.is_cuda() && byte_scales_fp16.device() == x.device(), "byte_scales_fp16 device mismatch");
    TORCH_CHECK(byte_scales_fp16.scalar_type() == torch::kHalf && byte_scales_fp16.is_contiguous(), "byte_scales_fp16 must be contiguous float16");
    TORCH_CHECK(bias_fp16.is_cuda() && bias_fp16.device() == x.device(), "bias_fp16 device mismatch");
    TORCH_CHECK(bias_fp16.scalar_type() == torch::kHalf && bias_fp16.is_contiguous(), "bias_fp16 must be contiguous float16");
    TORCH_CHECK(!has_bias || bias_fp16.numel() >= out_features, "bias_fp16 numel too small");

    const c10::cuda::CUDAGuard device_guard(x.device());
    const auto stream = c10::cuda::getCurrentCUDAStream();
    const int64_t n_tot = x.size(0);
    auto out = torch::empty({n_tot, out_features}, x.options());
    if (n_tot == 0 || out_features == 0) {
        return out;
    }

    const float* x_ptr = x.data_ptr<float>();
    const uint8_t* p_ptr = packed_u8.data_ptr<uint8_t>();
    const __half* sc_ptr = reinterpret_cast<const __half*>(scales_fp16.data_ptr<at::Half>());
    const __half* bsc_ptr = reinterpret_cast<const __half*>(byte_scales_fp16.data_ptr<at::Half>());
    const __half* bias_ptr = reinterpret_cast<const __half*>(bias_fp16.data_ptr<at::Half>());
    float* out_ptr = out.data_ptr<float>();
    const int64_t g32 = row_aligned ? (in_features / 32) : 0;

    if (n_tot <= 4) {
        const dim3 block(32, 8);
        const dim3 grid(
            static_cast<unsigned int>((out_features + 7) / 8),
            static_cast<unsigned int>(n_tot)
        );
        if (row_aligned) {
            tmac_lut_gemv_kernel<true><<<grid, block, 0, stream>>>(
                x_ptr, p_ptr, sc_ptr, bsc_ptr, bias_ptr, out_ptr,
                out_features, in_pad, k_bytes, g32, has_bias
            );
        } else {
            tmac_lut_gemv_kernel<false><<<grid, block, 0, stream>>>(
                x_ptr, p_ptr, sc_ptr, bsc_ptr, bias_ptr, out_ptr,
                out_features, in_pad, k_bytes, 0, has_bias
            );
        }
    } else {
        const dim3 block(32, 8);
        const dim3 grid(
            static_cast<unsigned int>((n_tot + 31) / 32),
            static_cast<unsigned int>((out_features + 63) / 64)
        );
        if (row_aligned) {
            tmac_lut_kernel<true><<<grid, block, 0, stream>>>(
                x_ptr, p_ptr, sc_ptr, bsc_ptr, bias_ptr, out_ptr,
                n_tot, out_features, in_pad, k_bytes, g32, has_bias
            );
        } else {
            tmac_lut_kernel<false><<<grid, block, 0, stream>>>(
                x_ptr, p_ptr, sc_ptr, bsc_ptr, bias_ptr, out_ptr,
                n_tot, out_features, in_pad, k_bytes, 0, has_bias
            );
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

_CPU_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

namespace {

inline void build_lut9_pair(const float* xp, float* lut01, float* lut23) {
    const float x0 = xp[0];
    const float x1 = xp[1];
    const float x2 = xp[2];
    const float x3 = xp[3];

    lut01[0] = -x1 - x0;
    lut01[1] = -x1;
    lut01[2] = -x1 + x0;
    lut01[3] = -x0;
    lut01[4] = 0.0f;
    lut01[5] = x0;
    lut01[6] = x1 - x0;
    lut01[7] = x1;
    lut01[8] = x1 + x0;

    lut23[0] = -x3 - x2;
    lut23[1] = -x3;
    lut23[2] = -x3 + x2;
    lut23[3] = -x2;
    lut23[4] = 0.0f;
    lut23[5] = x2;
    lut23[6] = x3 - x2;
    lut23[7] = x3;
    lut23[8] = x3 + x2;
}

inline void build_token_luts(const float* xm, int64_t k_bytes, float* lut_buf) {
    // Stride 32 floats per byte: [0..15] = lut01 (lanes 0..8 used, 9..15 zero), [16..31] = lut23
    for (int64_t b = 0; b < k_bytes; ++b) {
        float* dst = lut_buf + b * 32;
        build_lut9_pair(xm + b * 4, dst, dst + 16);
    }
}

#if defined(__AVX512F__)
inline void compute_block16_aligned_avx512(
    const uint8_t* blk_p,
    const float* blk_sc,
    const float* lut_buf,
    int64_t g32,
    float* out16
) {
    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    __m512 acc = _mm512_setzero_ps();
    for (int64_t g = 0; g < g32; ++g) {
        const float* lb = lut_buf + (g * 8) * 32;
        __m512 gsum = _mm512_setzero_ps();
        for (int j = 0; j < 8; ++j) {
            const __m128i c8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk_p + j * 16));
            const __m128i lo8 = _mm_and_si128(c8, mask_lo);
            const __m128i hi8 = _mm_and_si128(_mm_srli_epi16(c8, 4), mask_lo);
            const __m512i lo32 = _mm512_cvtepu8_epi32(lo8);
            const __m512i hi32 = _mm512_cvtepu8_epi32(hi8);
            const __m512 z01 = _mm512_loadu_ps(lb + j * 32);
            const __m512 z23 = _mm512_loadu_ps(lb + j * 32 + 16);
            const __m512 v01 = _mm512_permutexvar_ps(lo32, z01);
            const __m512 v23 = _mm512_permutexvar_ps(hi32, z23);
            gsum = _mm512_add_ps(gsum, _mm512_add_ps(v01, v23));
        }
        const __m512 sc = _mm512_loadu_ps(blk_sc + g * 16);
        acc = _mm512_fmadd_ps(gsum, sc, acc);
        blk_p += 128;
    }
    _mm512_storeu_ps(out16, acc);
}

inline void compute_2blocks32_aligned_avx512(
    const uint8_t* blk_p0,
    const uint8_t* blk_p1,
    const float* blk_sc0,
    const float* blk_sc1,
    const float* lut_buf,
    int64_t g32,
    float* out32
) {
    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    for (int64_t g = 0; g < g32; ++g) {
        const float* lb = lut_buf + (g * 8) * 32;
        __m512 gsum0 = _mm512_setzero_ps();
        __m512 gsum1 = _mm512_setzero_ps();
        for (int j = 0; j < 8; ++j) {
            const __m512 z01 = _mm512_loadu_ps(lb + j * 32);
            const __m512 z23 = _mm512_loadu_ps(lb + j * 32 + 16);

            const __m128i c0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk_p0 + j * 16));
            const __m512i lo0 = _mm512_cvtepu8_epi32(_mm_and_si128(c0, mask_lo));
            const __m512i hi0 = _mm512_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(c0, 4), mask_lo));
            gsum0 = _mm512_add_ps(
                gsum0,
                _mm512_add_ps(_mm512_permutexvar_ps(lo0, z01), _mm512_permutexvar_ps(hi0, z23))
            );

            const __m128i c1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk_p1 + j * 16));
            const __m512i lo1 = _mm512_cvtepu8_epi32(_mm_and_si128(c1, mask_lo));
            const __m512i hi1 = _mm512_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(c1, 4), mask_lo));
            gsum1 = _mm512_add_ps(
                gsum1,
                _mm512_add_ps(_mm512_permutexvar_ps(lo1, z01), _mm512_permutexvar_ps(hi1, z23))
            );
        }
        acc0 = _mm512_fmadd_ps(gsum0, _mm512_loadu_ps(blk_sc0 + g * 16), acc0);
        acc1 = _mm512_fmadd_ps(gsum1, _mm512_loadu_ps(blk_sc1 + g * 16), acc1);
        blk_p0 += 128;
        blk_p1 += 128;
    }
    _mm512_storeu_ps(out32, acc0);
    _mm512_storeu_ps(out32 + 16, acc1);
}

inline void compute_block16_unaligned_avx512(
    const uint8_t* blk_p,
    const float* blk_bsc,
    const float* lut_buf,
    int64_t k_bytes,
    float* out16
) {
    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    __m512 acc = _mm512_setzero_ps();
    for (int64_t b = 0; b < k_bytes; ++b) {
        const float* lb = lut_buf + b * 32;
        const __m128i c8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk_p + b * 16));
        const __m512i lo32 = _mm512_cvtepu8_epi32(_mm_and_si128(c8, mask_lo));
        const __m512i hi32 = _mm512_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(c8, 4), mask_lo));
        const __m512 v = _mm512_add_ps(
            _mm512_permutexvar_ps(lo32, _mm512_loadu_ps(lb)),
            _mm512_permutexvar_ps(hi32, _mm512_loadu_ps(lb + 16))
        );
        acc = _mm512_fmadd_ps(v, _mm512_loadu_ps(blk_bsc + b * 16), acc);
    }
    _mm512_storeu_ps(out16, acc);
}
#elif defined(__AVX2__)
inline __m256 lut9_lookup_avx2(__m256i idx, __m256 lut07, __m256 lut8, __m256i v8) {
    const __m256 p = _mm256_permutevar8x32_ps(lut07, idx);
    const __m256 m8 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(idx, v8));
    return _mm256_blendv_ps(p, lut8, m8);
}

inline void compute_block16_aligned_avx2(
    const uint8_t* blk_p,
    const float* blk_sc,
    const float* lut_buf,
    int64_t g32,
    float* out16
) {
    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    const __m256i v8 = _mm256_set1_epi32(8);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    for (int64_t g = 0; g < g32; ++g) {
        const float* lb = lut_buf + (g * 8) * 32;
        __m256 gsum0 = _mm256_setzero_ps();
        __m256 gsum1 = _mm256_setzero_ps();
        for (int j = 0; j < 8; ++j) {
            const float* l01 = lb + j * 32;
            const float* l23 = l01 + 16;
            const __m256 y01 = _mm256_loadu_ps(l01);
            const __m256 y01_8 = _mm256_broadcast_ss(l01 + 8);
            const __m256 y23 = _mm256_loadu_ps(l23);
            const __m256 y23_8 = _mm256_broadcast_ss(l23 + 8);

            const __m128i c8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk_p + j * 16));
            const __m128i lo8 = _mm_and_si128(c8, mask_lo);
            const __m128i hi8 = _mm_and_si128(_mm_srli_epi16(c8, 4), mask_lo);

            const __m256i lo_a = _mm256_cvtepu8_epi32(lo8);
            const __m256i hi_a = _mm256_cvtepu8_epi32(hi8);
            gsum0 = _mm256_add_ps(
                gsum0,
                _mm256_add_ps(lut9_lookup_avx2(lo_a, y01, y01_8, v8), lut9_lookup_avx2(hi_a, y23, y23_8, v8))
            );

            const __m256i lo_b = _mm256_cvtepu8_epi32(_mm_srli_si128(lo8, 8));
            const __m256i hi_b = _mm256_cvtepu8_epi32(_mm_srli_si128(hi8, 8));
            gsum1 = _mm256_add_ps(
                gsum1,
                _mm256_add_ps(lut9_lookup_avx2(lo_b, y01, y01_8, v8), lut9_lookup_avx2(hi_b, y23, y23_8, v8))
            );
        }
        acc0 = _mm256_fmadd_ps(gsum0, _mm256_loadu_ps(blk_sc + g * 16), acc0);
        acc1 = _mm256_fmadd_ps(gsum1, _mm256_loadu_ps(blk_sc + g * 16 + 8), acc1);
        blk_p += 128;
    }
    _mm256_storeu_ps(out16, acc0);
    _mm256_storeu_ps(out16 + 8, acc1);
}

inline void compute_block16_unaligned_avx2(
    const uint8_t* blk_p,
    const float* blk_bsc,
    const float* lut_buf,
    int64_t k_bytes,
    float* out16
) {
    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    const __m256i v8 = _mm256_set1_epi32(8);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    for (int64_t b = 0; b < k_bytes; ++b) {
        const float* l01 = lut_buf + b * 32;
        const float* l23 = l01 + 16;
        const __m256 y01 = _mm256_loadu_ps(l01);
        const __m256 y01_8 = _mm256_broadcast_ss(l01 + 8);
        const __m256 y23 = _mm256_loadu_ps(l23);
        const __m256 y23_8 = _mm256_broadcast_ss(l23 + 8);

        const __m128i c8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk_p + b * 16));
        const __m128i lo8 = _mm_and_si128(c8, mask_lo);
        const __m128i hi8 = _mm_and_si128(_mm_srli_epi16(c8, 4), mask_lo);

        const __m256 v0 = _mm256_add_ps(
            lut9_lookup_avx2(_mm256_cvtepu8_epi32(lo8), y01, y01_8, v8),
            lut9_lookup_avx2(_mm256_cvtepu8_epi32(hi8), y23, y23_8, v8)
        );
        const __m256 v1 = _mm256_add_ps(
            lut9_lookup_avx2(_mm256_cvtepu8_epi32(_mm_srli_si128(lo8, 8)), y01, y01_8, v8),
            lut9_lookup_avx2(_mm256_cvtepu8_epi32(_mm_srli_si128(hi8, 8)), y23, y23_8, v8)
        );
        acc0 = _mm256_fmadd_ps(v0, _mm256_loadu_ps(blk_bsc + b * 16), acc0);
        acc1 = _mm256_fmadd_ps(v1, _mm256_loadu_ps(blk_bsc + b * 16 + 8), acc1);
    }
    _mm256_storeu_ps(out16, acc0);
    _mm256_storeu_ps(out16 + 8, acc1);
}
#endif

inline void compute_block16_aligned_scalar(
    const uint8_t* blk_p,
    const float* blk_sc,
    const float* lut_buf,
    int64_t g32,
    float* out16
) {
    float acc[16];
    for (int r = 0; r < 16; ++r) acc[r] = 0.0f;
    for (int64_t g = 0; g < g32; ++g) {
        const float* lb = lut_buf + (g * 8) * 32;
        float gs[16];
        for (int r = 0; r < 16; ++r) gs[r] = 0.0f;
        for (int j = 0; j < 8; ++j) {
            const float* l01 = lb + j * 32;
            const float* l23 = l01 + 16;
            const uint8_t* pj = blk_p + j * 16;
            for (int r = 0; r < 16; ++r) {
                const uint8_t nib = pj[r];
                gs[r] += l01[nib & 0x0Fu] + l23[nib >> 4];
            }
        }
        const float* sc = blk_sc + g * 16;
        for (int r = 0; r < 16; ++r) acc[r] += gs[r] * sc[r];
        blk_p += 128;
    }
    for (int r = 0; r < 16; ++r) out16[r] = acc[r];
}

inline void compute_block16_unaligned_scalar(
    const uint8_t* blk_p,
    const float* blk_bsc,
    const float* lut_buf,
    int64_t k_bytes,
    float* out16
) {
    float acc[16];
    for (int r = 0; r < 16; ++r) acc[r] = 0.0f;
    for (int64_t b = 0; b < k_bytes; ++b) {
        const float* l01 = lut_buf + b * 32;
        const float* l23 = l01 + 16;
        const uint8_t* pb = blk_p + b * 16;
        const float* sc = blk_bsc + b * 16;
        for (int r = 0; r < 16; ++r) {
            const uint8_t nib = pb[r];
            acc[r] += (l01[nib & 0x0Fu] + l23[nib >> 4]) * sc[r];
        }
    }
    for (int r = 0; r < 16; ++r) out16[r] = acc[r];
}

inline void store_block16(float* ym, int64_t o, int64_t out_features, const float* tmp, const float* bias_ptr, bool has_bias) {
    const int64_t rem = out_features - o;
    const int lim = rem >= 16 ? 16 : static_cast<int>(rem);
    if (has_bias) {
        for (int r = 0; r < lim; ++r) ym[o + r] = tmp[r] + bias_ptr[o + r];
    } else {
        std::memcpy(ym + o, tmp, static_cast<size_t>(lim) * sizeof(float));
    }
}

inline void run_token_blks(
    const uint8_t* p_blk,
    const float* sc_blk,
    const float* lut_buf,
    const float* bias_ptr,
    float* ym,
    int64_t out_features,
    int64_t k_bytes,
    int64_t g32,
    int n_blks,
    bool row_aligned,
    bool has_bias,
    bool use_omp_blks
) {
    const int64_t p_stride = k_bytes * 16;
    const int64_t sc_stride = (row_aligned ? g32 : k_bytes) * 16;

#if defined(__AVX512F__)
    if (row_aligned) {
        const int64_t n_pairs = n_blks >> 1;
        auto run_pair_range = [&](int64_t pr_begin, int64_t pr_end) {
            float tmp32[32];
            for (int64_t pr = pr_begin; pr < pr_end; ++pr) {
                const int64_t blk = pr << 1;
                compute_2blocks32_aligned_avx512(
                    p_blk + blk * p_stride,
                    p_blk + (blk + 1) * p_stride,
                    sc_blk + blk * sc_stride,
                    sc_blk + (blk + 1) * sc_stride,
                    lut_buf,
                    g32,
                    tmp32
                );
                store_block16(ym, blk * 16, out_features, tmp32, bias_ptr, has_bias);
                store_block16(ym, (blk + 1) * 16, out_features, tmp32 + 16, bias_ptr, has_bias);
            }
        };
        if (use_omp_blks && n_pairs >= 4) {
            at::parallel_for(0, n_pairs, 2, run_pair_range);
        } else {
            run_pair_range(0, n_pairs);
        }
        if (n_blks & 1) {
            const int64_t blk = n_blks - 1;
            float tmp16[16];
            compute_block16_aligned_avx512(
                p_blk + blk * p_stride,
                sc_blk + blk * sc_stride,
                lut_buf,
                g32,
                tmp16
            );
            store_block16(ym, blk * 16, out_features, tmp16, bias_ptr, has_bias);
        }
        return;
    }
#endif

    auto run_blk_range = [&](int64_t b_begin, int64_t b_end) {
        float tmp16[16];
        for (int64_t blk = b_begin; blk < b_end; ++blk) {
            const uint8_t* bp = p_blk + blk * p_stride;
            const float* bs = sc_blk + blk * sc_stride;
            if (row_aligned) {
#if defined(__AVX512F__)
                compute_block16_aligned_avx512(bp, bs, lut_buf, g32, tmp16);
#elif defined(__AVX2__)
                compute_block16_aligned_avx2(bp, bs, lut_buf, g32, tmp16);
#else
                compute_block16_aligned_scalar(bp, bs, lut_buf, g32, tmp16);
#endif
            } else {
#if defined(__AVX512F__)
                compute_block16_unaligned_avx512(bp, bs, lut_buf, k_bytes, tmp16);
#elif defined(__AVX2__)
                compute_block16_unaligned_avx2(bp, bs, lut_buf, k_bytes, tmp16);
#else
                compute_block16_unaligned_scalar(bp, bs, lut_buf, k_bytes, tmp16);
#endif
            }
            store_block16(ym, blk * 16, out_features, tmp16, bias_ptr, has_bias);
        }
    };
    if (use_omp_blks && n_blks >= 8) {
        at::parallel_for(0, static_cast<int64_t>(n_blks), 4, run_blk_range);
    } else {
        run_blk_range(0, static_cast<int64_t>(n_blks));
    }
}

}  // namespace

std::string tmac_cpu_isa_name() {
#if defined(__AVX512F__)
    return "AVX512F";
#elif defined(__AVX2__)
    return "AVX2";
#else
    return "SCALAR";
#endif
}

torch::Tensor tmac_lut_cpu_pack_blk16(
    torch::Tensor packed_u8,
    int64_t out_features,
    int64_t k_bytes
) {
    TORCH_CHECK(packed_u8.device().is_cpu() && packed_u8.scalar_type() == torch::kUInt8, "packed_u8 must be CPU uint8");
    auto p_c = packed_u8.contiguous();
    const int64_t n_blks = (out_features + 15) / 16;
    auto out = torch::empty({n_blks, k_bytes, 16}, p_c.options());
    const uint8_t* src = p_c.data_ptr<uint8_t>();
    uint8_t* dst = out.data_ptr<uint8_t>();
    for (int64_t blk = 0; blk < n_blks; ++blk) {
        const int64_t o_base = blk * 16;
        uint8_t* bd = dst + blk * k_bytes * 16;
        for (int64_t b = 0; b < k_bytes; ++b) {
            uint8_t* row = bd + b * 16;
            for (int r = 0; r < 16; ++r) {
                const int64_t o = o_base + r;
                if (o < out_features) {
                    const uint8_t c = src[o * k_bytes + b];
                    const uint8_t cc = c > 80 ? 80 : c;
                    const uint8_t hi = static_cast<uint8_t>(cc / 9);
                    const uint8_t lo = static_cast<uint8_t>(cc - hi * 9);
                    row[r] = static_cast<uint8_t>((hi << 4) | lo);
                } else {
                    row[r] = 0x44u;  // (4, 4) -> 0.0f + 0.0f
                }
            }
        }
    }
    return out;
}

torch::Tensor tmac_lut_cpu_forward_blk16(
    torch::Tensor x,
    torch::Tensor packed_blk16,
    torch::Tensor scales_blk16_f32,
    torch::Tensor bias_f32,
    int64_t out_features,
    int64_t in_features,
    int64_t in_pad,
    int64_t k_bytes,
    bool row_aligned,
    bool has_bias
) {
    TORCH_CHECK(x.device().is_cpu() && x.scalar_type() == torch::kFloat32 && x.is_contiguous(), "x must be contiguous CPU float32");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == in_pad && k_bytes * 4 == in_pad, "x shape mismatch");
    TORCH_CHECK(packed_blk16.device().is_cpu() && packed_blk16.scalar_type() == torch::kUInt8 && packed_blk16.is_contiguous(), "packed_blk16 invalid");
    TORCH_CHECK(scales_blk16_f32.device().is_cpu() && scales_blk16_f32.scalar_type() == torch::kFloat32 && scales_blk16_f32.is_contiguous(), "scales_blk16_f32 invalid");
    TORCH_CHECK(!has_bias || (bias_f32.device().is_cpu() && bias_f32.scalar_type() == torch::kFloat32 && bias_f32.numel() >= out_features), "bias_f32 invalid");

    const int64_t n_tot = x.size(0);
    auto out = torch::empty({n_tot, out_features}, x.options());
    if (n_tot == 0 || out_features == 0) {
        return out;
    }

    const int64_t g32 = row_aligned ? (in_features / 32) : 0;
    const int n_blks = static_cast<int>((out_features + 15) / 16);
    const float* x_ptr = x.data_ptr<float>();
    const uint8_t* p_blk = packed_blk16.data_ptr<uint8_t>();
    const float* sc_blk = scales_blk16_f32.data_ptr<float>();
    const float* bias_ptr = has_bias ? bias_f32.data_ptr<float>() : nullptr;
    float* out_ptr = out.data_ptr<float>();

    if (n_tot >= 4) {
        at::parallel_for(0, n_tot, 1, [&](int64_t m_begin, int64_t m_end) {
            std::vector<float> luts(static_cast<size_t>(k_bytes) * 32, 0.0f);
            for (int64_t m = m_begin; m < m_end; ++m) {
                build_token_luts(x_ptr + m * in_pad, k_bytes, luts.data());
                run_token_blks(
                    p_blk, sc_blk, luts.data(), bias_ptr,
                    out_ptr + m * out_features,
                    out_features, k_bytes, g32, n_blks, row_aligned, has_bias, false
                );
            }
        });
    } else {
        static thread_local std::vector<float> tl_luts;
        const size_t need = static_cast<size_t>(k_bytes) * 32;
        if (tl_luts.size() < need) {
            tl_luts.assign(need, 0.0f);
        }
        // Fork ATen intra-op thread pool only when packed matvec compute outweighs sync overhead (~512 KB)
        const bool use_omp_blks = (out_features * k_bytes >= 524288) && (n_blks >= 8);
        for (int64_t m = 0; m < n_tot; ++m) {
            build_token_luts(x_ptr + m * in_pad, k_bytes, tl_luts.data());
            run_token_blks(
                p_blk, sc_blk, tl_luts.data(), bias_ptr,
                out_ptr + m * out_features,
                out_features, k_bytes, g32, n_blks, row_aligned, has_bias, use_omp_blks
            );
        }
    }
    return out;
}

torch::Tensor tmac_lut_cpu_forward(
    torch::Tensor x,
    torch::Tensor packed_u8,
    torch::Tensor scales_fp16,
    torch::Tensor byte_scales_fp16,
    torch::Tensor bias_fp16,
    int64_t out_features,
    int64_t in_features,
    int64_t in_pad,
    int64_t k_bytes,
    bool row_aligned,
    bool has_bias
) {
    // Compatibility entry point: packs on-the-fly and delegates to fast blk16 kernel.
    auto p_blk = tmac_lut_cpu_pack_blk16(packed_u8, out_features, k_bytes);
    const int64_t n_blks = (out_features + 15) / 16;
    const int64_t n_cols = row_aligned ? (in_features / 32) : k_bytes;
    auto src_sc = (row_aligned ? scales_fp16 : byte_scales_fp16).to(torch::kFloat32).reshape({out_features, n_cols}).contiguous();
    auto sc_blk = torch::zeros({n_blks, n_cols, 16}, src_sc.options());
    const float* ss = src_sc.data_ptr<float>();
    float* sd = sc_blk.data_ptr<float>();
    for (int64_t blk = 0; blk < n_blks; ++blk) {
        const int64_t o_base = blk * 16;
        for (int64_t c = 0; c < n_cols; ++c) {
            for (int r = 0; r < 16; ++r) {
                const int64_t o = o_base + r;
                sd[(blk * n_cols + c) * 16 + r] = (o < out_features) ? ss[o * n_cols + c] : 0.0f;
            }
        }
    }
    auto bias_f32 = has_bias ? bias_fp16.to(torch::kFloat32).contiguous() : torch::empty({0}, x.options());
    return tmac_lut_cpu_forward_blk16(
        x, p_blk, sc_blk, bias_f32,
        out_features, in_features, in_pad, k_bytes, row_aligned, has_bias
    );
}
"""


_CUDA_BUILD_ERROR: str | None = None
_CPU_BUILD_ERROR: str | None = None
_CPU_ISA_NAME: str | None = None


def _has_python_headers() -> bool:
    inc = sysconfig.get_path("include")
    return bool(inc) and (Path(inc) / "Python.h").is_file()


@lru_cache(maxsize=1)
def _ensure_windows_msvc_env() -> str | None:
    """Auto-discovers VS2022/2019 `vcvars64.bat` on Windows so `cl.exe` & Windows SDK are in PATH."""
    if sys.platform != "win32":
        return shutil.which("cl")
    existing = shutil.which("cl")
    if existing is not None and os.environ.get("INCLUDE") and os.environ.get("LIB"):
        os.environ.setdefault("VSCMD_ARG_TGT_ARCH", "x64")
        os.environ.setdefault("DISTUTILS_USE_SDK", "1")
        os.environ.setdefault("MSSdk", "1")
        return existing

    vcvars_cands: list[Path] = []
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / (
        r"Microsoft Visual Studio\Installer\vswhere.exe"
    )
    if vswhere.is_file():
        try:
            out = subprocess.run(
                [str(vswhere), "-latest", "-products", "*", "-property", "installationPath"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            inst = out.stdout.strip()
            if inst:
                vcvars_cands.append(Path(inst) / r"VC\Auxiliary\Build\vcvars64.bat")
        except Exception:
            pass

    for year in ("2022", "2019"):
        for edition in ("Community", "BuildTools", "Professional", "Enterprise"):
            for pf in (r"C:\Program Files", r"C:\Program Files (x86)"):
                vcvars_cands.append(
                    Path(pf) / "Microsoft Visual Studio" / year / edition / r"VC\Auxiliary\Build\vcvars64.bat"
                )

    vcvars = next((p for p in vcvars_cands if p.is_file()), None)
    if vcvars is None:
        return existing

    try:
        res = subprocess.run(
            f'call "{vcvars}" >nul && set',
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if res.returncode != 0:
            return existing
        for line in res.stdout.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                if k:
                    os.environ[k] = v
        os.environ.setdefault("VSCMD_ARG_TGT_ARCH", "x64")
        os.environ.setdefault("DISTUTILS_USE_SDK", "1")
        os.environ.setdefault("MSSdk", "1")
        return shutil.which("cl")
    except Exception:
        return existing


@lru_cache(maxsize=1)
def _discover_nvcc_and_cuda_paths() -> tuple[str | None, list[str], list[str]]:
    """Finds `nvcc` across PATH, `CUDA_HOME`, pip `nvidia/cuda_nvcc`, and standard Windows CUDA Toolkits."""
    extra_inc: list[str] = []
    extra_lib: list[str] = []
    nvcc_exe = "nvcc.exe" if sys.platform == "win32" else "nvcc"

    # Collect pip nvidia-cuda-* wheel include/lib dirs if present
    site_roots: list[Path] = []
    for key in ("purelib", "platlib"):
        sp = sysconfig.get_path(key)
        if sp:
            p = Path(sp)
            if p not in site_roots:
                site_roots.append(p)
    torch_site = Path(torch.__file__).resolve().parents[1]
    if torch_site not in site_roots:
        site_roots.append(torch_site)

    pip_nvcc: str | None = None
    for sr in site_roots:
        nv_root = sr / "nvidia"
        if not nv_root.is_dir():
            continue
        # Modern unified pip layout (`nvidia-cuda-nvcc==13.x` -> `nvidia/cu13/...`) and legacy `nvidia/cuda_nvcc/...`
        for pkg_sub in ("cu13", "cu12", "cuda_nvcc"):
            cand = nv_root / pkg_sub / "bin" / nvcc_exe
            if cand.is_file() and pip_nvcc is None:
                pip_nvcc = str(cand)
        for sub in ("cu13", "cu12", "cuda_nvcc", "cuda_runtime", "cuda_cccl", "cublas"):
            for inc_sub in ("include", "include/cccl"):
                inc_d = nv_root / sub / inc_sub
                if inc_d.is_dir() and str(inc_d) not in extra_inc:
                    extra_inc.append(str(inc_d))
            for lib_sub in ("lib/x64", "lib"):
                lib_d = nv_root / sub / lib_sub
                if lib_d.is_dir() and str(lib_d) not in extra_lib:
                    extra_lib.append(str(lib_d))

    nvcc_path = shutil.which("nvcc")
    if nvcc_path is None:
        for env_k in ("CUDA_HOME", "CUDA_PATH"):
            ch = os.environ.get(env_k)
            if ch:
                c = Path(ch) / "bin" / nvcc_exe
                if c.is_file():
                    nvcc_path = str(c)
                    break
    if nvcc_path is None and pip_nvcc is not None:
        nvcc_path = pip_nvcc
    if nvcc_path is None and sys.platform == "win32":
        tk_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
        if tk_root.is_dir():
            cands = sorted(tk_root.glob("v*/bin/nvcc.exe"), reverse=True)
            v13 = [c for c in cands if c.parents[1].name.startswith("v13.")]
            pick = v13[0] if v13 else (cands[0] if cands else None)
            if pick is not None:
                nvcc_path = str(pick)

    if nvcc_path is not None:
        nv_bin = str(Path(nvcc_path).parent)
        nv_home = str(Path(nvcc_path).parents[1])
        cur_path = os.environ.get("PATH", "")
        if nv_bin.lower() not in cur_path.lower():
            os.environ["PATH"] = nv_bin + os.pathsep + cur_path
        os.environ["CUDA_HOME"] = nv_home
        os.environ["CUDA_PATH"] = nv_home
        for inc_sub in ("include", "include/cccl"):
            inc_main = Path(nv_home) / inc_sub
            if inc_main.is_dir() and str(inc_main) not in extra_inc:
                extra_inc.append(str(inc_main))
        for lib_sub in ("lib/x64", "lib"):
            lib_main = Path(nv_home) / lib_sub
            if lib_main.is_dir() and str(lib_main) not in extra_lib:
                extra_lib.append(str(lib_main))
        if sys.platform == "win32":
            cur_inc = os.environ.get("INCLUDE", "")
            for inc_p in reversed(extra_inc):
                if inc_p.lower() not in cur_inc.lower():
                    cur_inc = inc_p + (";" + cur_inc if cur_inc else "")
            if cur_inc:
                os.environ["INCLUDE"] = cur_inc
            cur_lib = os.environ.get("LIB", "")
            for lib_p in reversed(extra_lib):
                if lib_p.lower() not in cur_lib.lower():
                    cur_lib = lib_p + (";" + cur_lib if cur_lib else "")
            if cur_lib:
                os.environ["LIB"] = cur_lib
    return nvcc_path, extra_inc, extra_lib


@lru_cache(maxsize=1)
def _load_cuda_ext():
    global _CUDA_BUILD_ERROR
    if not torch.cuda.is_available():
        _CUDA_BUILD_ERROR = "CUDA device unavailable"
        return None
    if sys.platform == "win32":
        _ensure_windows_msvc_env()
    nvcc_path, extra_inc, extra_lib = _discover_nvcc_and_cuda_paths()
    if nvcc_path is None:
        _CUDA_BUILD_ERROR = "nvcc not found in PATH or CUDA_HOME"
        return None
    if not _has_python_headers():
        _CUDA_BUILD_ERROR = "Python.h dev headers missing"
        return None
    try:
        from torch.utils import cpp_extension

        nv_home = str(Path(nvcc_path).parents[1])
        cpp_extension.CUDA_HOME = nv_home
        load_inline = cpp_extension.load_inline

        build_dir = Path(tempfile.gettempdir()) / "calcite5_tmac_cuda_v3"
        build_dir.mkdir(parents=True, exist_ok=True)
        is_win = sys.platform == "win32"
        # CCCL 13.3 on Windows MSVC requires conforming preprocessor `/Zc:preprocessor`
        cflags = ["/O2", "/EHsc", "/Zc:preprocessor"] if is_win else ["-O3"]
        cuda_cflags = (
            ["-O3", "--use_fast_math", "-Xcompiler", "/Zc:preprocessor"]
            if is_win
            else ["-O3", "--use_fast_math"]
        )
        ldflags = [f"/LIBPATH:{d}" for d in extra_lib] if is_win else [f"-L{d}" for d in extra_lib]
        mod = load_inline(
            name="calcite5_tmac_lut_cuda_ext_v3",
            cpp_sources=[_CUDA_CPP_SOURCE],
            cuda_sources=[_CUDA_KERNEL_SOURCE],
            functions=["tmac_lut_cuda_forward"],
            extra_cflags=cflags,
            extra_cuda_cflags=cuda_cflags,
            extra_include_paths=extra_inc or None,
            extra_ldflags=ldflags or None,
            build_directory=str(build_dir),
            with_cuda=True,
            verbose=False,
        )
        _CUDA_BUILD_ERROR = None
        return mod
    except Exception as exc:
        _CUDA_BUILD_ERROR = f"{type(exc).__name__}: {exc}"
        return None


@lru_cache(maxsize=1)
def _load_cpu_ext():
    global _CPU_BUILD_ERROR, _CPU_ISA_NAME
    if sys.platform == "win32":
        _ensure_windows_msvc_env()
    cxx = shutil.which("cl") if sys.platform == "win32" else (shutil.which("g++") or shutil.which("clang++"))
    if cxx is None:
        cxx = shutil.which("g++") or shutil.which("clang++") or shutil.which("cl")
    if cxx is None:
        _CPU_BUILD_ERROR = "No C++ compiler (g++/clang++/cl) in PATH"
        return None
    if not _has_python_headers():
        _CPU_BUILD_ERROR = "Python.h dev headers missing"
        return None

    from torch.utils.cpp_extension import load_inline

    is_msvc = Path(cxx).name.lower().startswith("cl")
    if is_msvc:
        # Uses PyTorch ATen intra-op thread pool (`at::parallel_for`) to avoid dual OpenMP DLL conflicts on Windows.
        flag_attempts = [
            ("v5_avx512", ["/O2", "/EHsc", "/arch:AVX512", "/fp:fast"], []),
            ("v5_avx2", ["/O2", "/EHsc", "/arch:AVX2", "/fp:fast"], []),
            ("v5_scalar", ["/O2", "/EHsc"], []),
        ]
    else:
        flag_attempts = [
            ("v5_native", ["-O3", "-march=native", "-ffast-math"], []),
            ("v5_generic", ["-O3"], []),
        ]

    last_exc: Exception | None = None
    for tag, cflags, ldflags in flag_attempts:
        try:
            build_dir = Path(tempfile.gettempdir()) / f"calcite5_tmac_cpu_{tag}"
            build_dir.mkdir(parents=True, exist_ok=True)
            mod = load_inline(
                name=f"calcite5_tmac_lut_cpu_ext_{tag}",
                cpp_sources=[_CPU_CPP_SOURCE],
                functions=[
                    "tmac_lut_cpu_forward",
                    "tmac_lut_cpu_forward_blk16",
                    "tmac_lut_cpu_pack_blk16",
                    "tmac_cpu_isa_name",
                ],
                extra_cflags=cflags,
                extra_ldflags=ldflags,
                build_directory=str(build_dir),
                verbose=False,
            )
            _CPU_BUILD_ERROR = None
            _CPU_ISA_NAME = str(mod.tmac_cpu_isa_name())
            return mod
        except Exception as exc:
            last_exc = exc
            continue
    _CPU_BUILD_ERROR = f"{type(last_exc).__name__}: {last_exc}" if last_exc else "Unknown CPU build error"
    return None


def get_lut_backend_info() -> dict[str, object]:
    """Сообщает статус нативных C++/CUDA бэкендов и активный фоллбек."""
    cuda_mod = _load_cuda_ext()
    cpu_mod = _load_cpu_ext()
    nvcc_p, _, _ = _discover_nvcc_and_cuda_paths()
    return {
        "cuda_ext_available": cuda_mod is not None,
        "cuda_build_error": _CUDA_BUILD_ERROR,
        "cpu_ext_available": cpu_mod is not None,
        "cpu_isa": _CPU_ISA_NAME,
        "cpu_build_error": _CPU_BUILD_ERROR,
        "nvcc_path": nvcc_p,
        "fallback_backend": "pytorch_vec",
    }


def _compute_unaligned_byte_scales(
    raw_scales_fp16: torch.Tensor,
    out_features: int,
    in_features: int,
    k_bytes: int,
    g_size: int,
) -> torch.Tensor:
    dev = raw_scales_fp16.device
    row_o = torch.arange(out_features, device=dev).unsqueeze(1)
    col_b = torch.arange(k_bytes, device=dev).unsqueeze(0) * 4
    col_c = col_b.clamp(max=max(0, in_features - 1))
    b_idx = ((row_o * in_features + col_c) // g_size).clamp(
        max=raw_scales_fp16.numel() - 1
    )
    return raw_scales_fp16[b_idx].to(torch.float16).contiguous()


class FastPackedLUT2BitLinear(PackedLUT2BitLinear):
    """
    Ускоренный физический 2-bit uint8 T-MAC LUT слой без умножений весов.
    Наследует `PackedLUT2BitLinear`, сохраняет идентичный `state_dict()`,
    `.bias_fp16` (`Law 15`) и подсчёт `compute_2bit_lut_footprint`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        packed_u8: torch.Tensor,
        raw_scales_fp16: torch.Tensor,
        bias_fp16: torch.Tensor | None,
        g_size: int,
        row_aligned: bool,
        backend: str = "auto",
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            packed_u8=packed_u8,
            raw_scales_fp16=raw_scales_fp16,
            bias_fp16=bias_fp16,
            g_size=g_size,
            row_aligned=row_aligned,
        )
        self.backend = str(backend)
        if not self.row_aligned:
            b_scales = _compute_unaligned_byte_scales(
                self.raw_scales_fp16,
                self.out_features,
                self.in_features,
                self.k_bytes,
                self.g_size,
            )
            self.register_buffer("_byte_scales_fp16", b_scales, persistent=False)
        else:
            self.register_buffer("_byte_scales_fp16", None, persistent=False)

        self._vec_idx_cache: torch.Tensor | None = None
        self._vec_scales_cache: torch.Tensor | None = None
        self._vec_cache_dev: torch.device | None = None
        # Compact L3-resident CPU 16-row block cache (same uint8 size as packed_u8, never int64!)
        self._cpu_blk_key: tuple[int, int, int, int] | None = None
        self._cpu_packed_blk16: torch.Tensor | None = None
        self._cpu_scales_blk16_f32: torch.Tensor | None = None
        self._cpu_bias_f32: torch.Tensor | None = None
        if hasattr(self, "register_load_state_dict_post_hook"):
            self.register_load_state_dict_post_hook(
                lambda mod, _inc: mod.refresh_cached_scales()
            )

    def refresh_cached_scales(self) -> None:
        """Пересчитывает производные буферы масштабов после `load_state_dict`."""
        if not self.row_aligned:
            b_scales = _compute_unaligned_byte_scales(
                self.raw_scales_fp16,
                self.out_features,
                self.in_features,
                self.k_bytes,
                self.g_size,
            )
            self.register_buffer("_byte_scales_fp16", b_scales, persistent=False)
        self._vec_idx_cache = None
        self._vec_scales_cache = None
        self._vec_cache_dev = None
        self._cpu_blk_key = None
        self._cpu_packed_blk16 = None
        self._cpu_scales_blk16_f32 = None
        self._cpu_bias_f32 = None

    @classmethod
    def from_packed(
        cls,
        ref: PackedLUT2BitLinear,
        backend: str = "auto",
    ) -> "FastPackedLUT2BitLinear":
        return cls(
            in_features=ref.in_features,
            out_features=ref.out_features,
            packed_u8=ref.packed_u8.detach().clone(),
            raw_scales_fp16=ref.raw_scales_fp16.detach().clone(),
            bias_fp16=ref.bias_fp16.detach().clone() if ref.bias_fp16 is not None else None,
            g_size=ref.g_size,
            row_aligned=ref.row_aligned,
            backend=backend,
        )

    def _resolve_backend(self, device: torch.device) -> str:
        req = self.backend
        if req == "auto":
            if device.type == "cuda" and _load_cuda_ext() is not None:
                return "cuda_ext"
            if device.type == "cpu" and _load_cpu_ext() is not None:
                return "cpu_ext"
            return "pytorch_vec"
        if req == "cuda_ext":
            if device.type != "cuda" or _load_cuda_ext() is None:
                raise RuntimeError(f"cuda_ext requested but unavailable: {_CUDA_BUILD_ERROR}")
            return "cuda_ext"
        if req == "cpu_ext":
            if device.type != "cpu" or _load_cpu_ext() is None:
                raise RuntimeError(f"cpu_ext requested but unavailable: {_CPU_BUILD_ERROR}")
            return "cpu_ext"
        if req in ("pytorch_vec", "reference"):
            return req
        raise ValueError(f"Unsupported backend={req!r}")

    def _ensure_cpu_blk16_cache(self, cpu_mod) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (
            int(self.packed_u8.data_ptr()),
            int(self.packed_u8._version),
            int(self.raw_scales_fp16.data_ptr()),
            int(self.raw_scales_fp16._version),
        )
        if (
            self._cpu_blk_key == key
            and self._cpu_packed_blk16 is not None
            and self._cpu_scales_blk16_f32 is not None
            and self._cpu_bias_f32 is not None
        ):
            return self._cpu_packed_blk16, self._cpu_scales_blk16_f32, self._cpu_bias_f32

        with torch.inference_mode(False), torch.no_grad():
            packed_2d = self.packed_u8.detach().clone().view(self.out_features, self.k_bytes).contiguous()
            p_blk = cpu_mod.tmac_lut_cpu_pack_blk16(packed_2d, self.out_features, self.k_bytes)

            n_blks = (self.out_features + 15) // 16
            out_pad = n_blks * 16
            if self.row_aligned:
                g32 = self.in_features // 32
                src_sc = (
                    self.scales_fp16 if self.scales_fp16 is not None else self.raw_scales_fp16
                ).detach().to(torch.float32).reshape(self.out_features, g32)
                n_cols = g32
            else:
                b_sc = self._byte_scales_fp16
                if b_sc is None:
                    b_sc = _compute_unaligned_byte_scales(
                        self.raw_scales_fp16.detach(),
                        self.out_features,
                        self.in_features,
                        self.k_bytes,
                        self.g_size,
                    )
                    self.register_buffer("_byte_scales_fp16", b_sc, persistent=False)
                src_sc = b_sc.detach().to(torch.float32).reshape(self.out_features, self.k_bytes)
                n_cols = self.k_bytes

            if out_pad > self.out_features:
                src_sc = F.pad(src_sc, (0, 0, 0, out_pad - self.out_features))
            sc_blk = src_sc.view(n_blks, 16, n_cols).permute(0, 2, 1).contiguous()
            bias_f32 = (
                self.bias_fp16.detach().to(torch.float32).contiguous()
                if self.bias_fp16 is not None
                else torch.empty((0,), dtype=torch.float32)
            )

        self._cpu_blk_key = key
        self._cpu_packed_blk16 = p_blk
        self._cpu_scales_blk16_f32 = sc_blk
        self._cpu_bias_f32 = bias_f32
        return p_blk, sc_blk, bias_f32

    def _forward_pytorch_vec(self, x_flat: torch.Tensor) -> torch.Tensor:
        n_tot = x_flat.size(0)
        k_bytes = self.k_bytes
        dev = x_flat.device
        if self._vec_cache_dev != dev or self._vec_idx_cache is None or self._vec_scales_cache is None:
            with torch.inference_mode(False), torch.no_grad():
                if self.row_aligned:
                    self._vec_scales_cache = self.scales_fp16.detach().t().float().unsqueeze(-1).contiguous()
                    self._vec_idx_cache = (
                        self.packed_u8.detach().view(self.out_features, k_bytes).t().long().unsqueeze(-1).contiguous()
                    )
                else:
                    b_sc = self._byte_scales_fp16
                    if b_sc is None:
                        b_sc = _compute_unaligned_byte_scales(
                            self.raw_scales_fp16.detach(),
                            self.out_features,
                            self.in_features,
                            k_bytes,
                            self.g_size,
                        )
                    self._vec_scales_cache = b_sc.detach().t().float().unsqueeze(-1).contiguous()
                    self._vec_idx_cache = self.packed_u8.detach().t().long().unsqueeze(-1).contiguous()
                self._vec_cache_dev = dev

        idx_base = self._vec_idx_cache
        scales_cached = self._vec_scales_cache
        g32 = self.in_features // 32 if self.row_aligned else 0
        # 512-token block cap keeps fallback peak VRAM 4x smaller than 2048 at long contexts
        n_blk = 512
        outs = []
        for start in range(0, n_tot, n_blk):
            xb = x_flat[start : start + n_blk].float()
            nb = xb.size(0)
            x4 = xb.view(nb, k_bytes, 4)
            z = torch.zeros_like(x4[:, :, 0:1])
            t0 = torch.cat([-x4[:, :, 0:1], z, x4[:, :, 0:1]], dim=-1)
            t1 = torch.cat([-x4[:, :, 1:2], z, x4[:, :, 1:2]], dim=-1)
            t2 = torch.cat([-x4[:, :, 2:3], z, x4[:, :, 2:3]], dim=-1)
            t3 = torch.cat([-x4[:, :, 3:4], z, x4[:, :, 3:4]], dim=-1)
            s01 = t1.unsqueeze(-1) + t0.unsqueeze(-2)
            s23 = t3.unsqueeze(-1) + t2.unsqueeze(-2)
            lut = (s23.unsqueeze(-1).unsqueeze(-1) + s01.unsqueeze(2).unsqueeze(2)).reshape(
                nb, k_bytes, 81
            )
            lut_p = lut.permute(1, 2, 0)
            gathered = torch.gather(lut_p, dim=1, index=idx_base.expand(-1, -1, nb))
            if self.row_aligned:
                grp_sum = gathered.view(g32, 8, self.out_features, nb).sum(dim=1)
                y_b = (grp_sum * scales_cached).sum(dim=0).t()
            else:
                y_b = (gathered * scales_cached).sum(dim=0).t()
            outs.append(y_b)

        y = outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
        if self.bias_fp16 is not None:
            y = y + self.bias_fp16.float().unsqueeze(0)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)
        n_tot = x_flat.size(0)
        if n_tot == 0:
            return torch.empty(orig_shape + (self.out_features,), device=x.device, dtype=x.dtype)

        backend = self._resolve_backend(x.device)
        if backend == "reference":
            return super().forward(x)

        if self.in_pad > self.in_features:
            x_flat = F.pad(x_flat, (0, self.in_pad - self.in_features))

        if backend == "pytorch_vec":
            y = self._forward_pytorch_vec(x_flat)
            return y.to(x.dtype).view(orig_shape + (self.out_features,))

        x_f32 = x_flat.float().contiguous()
        has_bias = self.bias_fp16 is not None

        if backend == "cpu_ext":
            cpu_mod = _load_cpu_ext()
            p_blk, sc_blk, bias_f32 = self._ensure_cpu_blk16_cache(cpu_mod)
            y = cpu_mod.tmac_lut_cpu_forward_blk16(
                x_f32,
                p_blk,
                sc_blk,
                bias_f32,
                self.out_features,
                self.in_features,
                self.in_pad,
                self.k_bytes,
                self.row_aligned,
                has_bias,
            )
            return y.to(x.dtype).view(orig_shape + (self.out_features,))

        packed = self.packed_u8.view(self.out_features, self.k_bytes).contiguous()
        sc = (
            self.scales_fp16.to(torch.float16).contiguous()
            if (self.row_aligned and self.scales_fp16 is not None)
            else self.raw_scales_fp16.to(torch.float16).contiguous()
        )
        if self.row_aligned:
            bsc = sc
        else:
            if self._byte_scales_fp16 is None:
                b_sc = _compute_unaligned_byte_scales(
                    self.raw_scales_fp16,
                    self.out_features,
                    self.in_features,
                    self.k_bytes,
                    self.g_size,
                )
                self.register_buffer("_byte_scales_fp16", b_sc, persistent=False)
            bsc = self._byte_scales_fp16.to(torch.float16).contiguous()
        bias_t = self.bias_fp16.to(torch.float16).contiguous() if has_bias else sc

        y = _load_cuda_ext().tmac_lut_cuda_forward(
            x_f32,
            packed,
            sc,
            bsc,
            bias_t,
            self.out_features,
            self.in_features,
            self.in_pad,
            self.k_bytes,
            self.row_aligned,
            has_bias,
        )
        return y.to(x.dtype).view(orig_shape + (self.out_features,))


class FusedPackedSwiGLU(nn.Module):
    """
    Inference-only lossless fusion of packed `SwiGLU` (`w_gate` + `w_up` -> single `w_gate_up`
    `FastPackedLUT2BitLinear`), halving LUT synthesis & kernel launches with exact parity.
    """

    def __init__(self, w_gate_up: FastPackedLUT2BitLinear, w_down: FastPackedLUT2BitLinear):
        super().__init__()
        self.w_gate_up = w_gate_up
        self.w_down = w_down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gu = self.w_gate_up(x)
        g, u = gu.chunk(2, dim=-1)
        return self.w_down(F.silu(g) * u)


def fuse_packed_swiglu_for_inference(
    lut_model: nn.Module,
    inplace: bool = False,
) -> nn.Module:
    """
    Opt-in inference-only fusion of packed `SwiGLU` (`w_gate` + `w_up`) into `FusedPackedSwiGLU`.
    Does NOT modify training models or serialized checkpoints; exact `< 1e-5` parity.
    """
    target = lut_model if inplace else copy.deepcopy(lut_model)
    target.train(False)

    def _fuse(mod: nn.Module) -> None:
        for name, child in list(mod.named_children()):
            w_g = getattr(child, "w_gate", None)
            w_u = getattr(child, "w_up", None)
            w_d = getattr(child, "w_down", None)
            if (
                isinstance(w_g, PackedLUT2BitLinear)
                and isinstance(w_u, PackedLUT2BitLinear)
                and isinstance(w_d, PackedLUT2BitLinear)
                and w_g.row_aligned
                and w_u.row_aligned
                and w_g.in_features == w_u.in_features
                and w_g.out_features == w_u.out_features
                and w_g.g_size == w_u.g_size
                and (w_g.bias_fp16 is None) == (w_u.bias_fp16 is None)
            ):
                backend_g = getattr(w_g, "backend", "auto")
                if not isinstance(w_d, FastPackedLUT2BitLinear):
                    w_d = FastPackedLUT2BitLinear.from_packed(w_d, backend=backend_g)
                packed_cat = torch.cat([w_g.packed_u8, w_u.packed_u8], dim=0).contiguous()
                scales_cat = torch.cat([w_g.raw_scales_fp16, w_u.raw_scales_fp16], dim=0).contiguous()
                bias_cat = (
                    torch.cat([w_g.bias_fp16, w_u.bias_fp16], dim=0).contiguous()
                    if (w_g.bias_fp16 is not None and w_u.bias_fp16 is not None)
                    else None
                )
                merged = FastPackedLUT2BitLinear(
                    in_features=w_g.in_features,
                    out_features=w_g.out_features + w_u.out_features,
                    packed_u8=packed_cat,
                    raw_scales_fp16=scales_cat,
                    bias_fp16=bias_cat,
                    g_size=w_g.g_size,
                    row_aligned=True,
                    backend=backend_g,
                )
                fused_mod = FusedPackedSwiGLU(merged, w_d)
                fused_mod.train(False)
                setattr(mod, name, fused_mod)
            else:
                _fuse(child)

    _fuse(target)
    return target


def accelerate_lut_model(
    lut_model: nn.Module,
    backend: str = "auto",
    inplace: bool = False,
) -> nn.Module:
    """Заменяет `PackedLUT2BitLinear` в модели на `FastPackedLUT2BitLinear` с заданным бэкендом."""
    if isinstance(lut_model, FastPackedLUT2BitLinear):
        target = lut_model if inplace else copy.deepcopy(lut_model)
        target.backend = str(backend)
        target.train(False)
        return target
    if isinstance(lut_model, PackedLUT2BitLinear):
        fast = FastPackedLUT2BitLinear.from_packed(lut_model, backend=backend)
        fast.train(False)
        return fast

    target = lut_model if inplace else copy.deepcopy(lut_model)
    target.train(False)

    def _swap(mod: nn.Module) -> None:
        for name, child in list(mod.named_children()):
            if isinstance(child, FastPackedLUT2BitLinear):
                child.backend = str(backend)
            elif isinstance(child, PackedLUT2BitLinear):
                setattr(mod, name, FastPackedLUT2BitLinear.from_packed(child, backend=backend))
            else:
                _swap(child)

    _swap(target)
    return target


def pack_fast_model_2bit_lut(
    qat_model: nn.Module,
    backend: str = "auto",
) -> nn.Module:
    """Упаковывает QAT-модель сразу в `FastPackedLUT2BitLinear`."""
    packed = pack_model_2bit_lut(qat_model)
    return accelerate_lut_model(packed, backend=backend, inplace=True)


def save_packed_lut_checkpoint(
    path: Path,
    lut_model: nn.Module,
    cfg,
    tok,
    metadata: dict | None = None,
) -> dict[str, float]:
    """Сохраняет компактный физический 2-bit uint8 LUT чекпоинт (~5.5 MB для 17.49M Tier-1)."""
    fp_info = compute_2bit_lut_footprint(lut_model)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "calcite5_packed_2bit_lut_v1",
        "simvq_split_config": dict(cfg.__dict__),
        "tokenizer": tok.to_dict(),
        "lut_state_dict": {k: v.detach().cpu() for k, v in lut_model.state_dict().items()},
        "lut_footprint": fp_info,
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, p)
    return fp_info


def load_packed_lut_checkpoint(
    path: Path,
    device: str = "cpu",
    backend: str = "auto",
):
    """Загружает физический `2-bit uint8 LUT` чекпоинт напрямую в `FastPackedLUT2BitLinear`."""
    from simvq_split.models import SimVQSplitConfig, SimVQSplitLM
    from simvq_split.tokenizer import ByteBPETokenizer

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    cfg = SimVQSplitConfig(**payload["simvq_split_config"])
    tok = ByteBPETokenizer.from_dict(payload["tokenizer"])
    skel = SimVQSplitLM(cfg).eval()
    if skel.byte_chunker is not None:
        skel.byte_chunker.lock_router(True)
        skel.set_crystal_schedule(1, 1)
    else:
        skel.set_snap_alpha(1.0)
    lut_m = pack_fast_model_2bit_lut(skel, backend=backend)
    lut_m.load_state_dict(payload["lut_state_dict"], strict=True)
    lut_m = lut_m.to(device).eval()
    if lut_m.byte_chunker is not None:
        lut_m.set_crystal_schedule(1, 1)
    else:
        lut_m.set_snap_alpha(1.0)
    return lut_m, cfg, tok, payload
