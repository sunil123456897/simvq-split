"""
fast_calcite_scans.py — Нативные CUDA-ядра обучения v3 для `Calcite System-1` (`load_inline`, `nvcc`),
нулевое дублирование `Crystal`/`Woodbury` и однопроходный кэшируемый `FastTernaryQAT`
без правки защищённого ядра `simvq_split/`.

Узлы v3 (`calcite5_fast_scans_cuda_v3`):
  1. `utf8_chunk_span_cuda`: побитово-точный скан `_enforce_utf8_chunk_span_jit` за 1 запуск ядра.
  2. `causal_refractory_gate_cuda`: 1D рефрактерный вентиль кристалла + поэлементный backward.
  3. `birkhoff_greedy_perm_cuda`: жадная проекция `S_G` (`G <= 32`) в регистрах без 12 запусков `argmax`.
  4. `diag_birkhoff_crystal_scan_cuda` (`fwd` + `bwd`): `template <int GH, bool IS_HARD>` (специализация
     `GH=12` с двойным буфером, 1 барьером в `fwd` и 3 барьерами в `bwd` вместо 3/5).
  5. `woodbury_tern_scan_cuda` (`P_uncerts` + `M fwd/bwd`): `template <int DK, int DV, bool IS_TERN>`
     (`d=32, 48` + обобщённый `d<=64`), 8-варповые GEMV, `warp_sum32`-редукции без серийных `tid==0`
     циклов, сжатый SMEM (`10.5/21.6 KB` вместо `38 KB`) и быстрый FP-путь `inv_dh == 0.0f`.
  6. `fast_crystal_forward` и `fast_woodbury_forward`: устранение двойного `_encode_tokens`,
     двойного `_compute_ptr_swap`, 4x расширения `s_read` при чтении и `118 MB` 5D `z_head`,
     плюс полная компиляция pre/post участков в `compile_calcite_cortex`.
  7. `fused_ternary_ste_cuda` + `configure_fast_ternary_qat`: однопроходное warp-ядро 2-шаговой
     тернарной квантизации (`__fdiv_rn`) и микробатч-кэш по `(data_ptr, weight._version, mix, dtype)`
     через `torch.library.custom_op` с сохранением точного STE-градиента на каждом микробатче и
     нулевыми рекомпиляциями в `torch.compile`.

На CPU или при отсутствии `nvcc` прозрачно использует эталонные функции `simvq_split/`.
"""

from __future__ import annotations

from functools import lru_cache
import importlib.util
import itertools
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import types
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F

from simvq_split.chunker import _enforce_utf8_chunk_span_jit
from simvq_split.crystal import (
    VoronoiCrystalLayer,
    causal_refractory_gate_jit,
    diag_birkhoff_crystal_scan as _ref_crystal_scan,
    log_sinkhorn_6,
    project_birkhoff_to_perm_jit,
)
from simvq_split.quant_lut import (
    PackedLUT2BitLinear,
    quantize_ternary_weights_ls,
)
from simvq_split.woodbury import (
    NeuralWoodburyLayer,
    neural_woodbury_chunk_scan_jit as _ref_woodbury_scan,
)

_CUDA_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <vector>

torch::Tensor utf8_chunk_span_cuda_impl(
    torch::Tensor raw_cand,
    torch::Tensor valid_mask,
    torch::Tensor can_fire_mask,
    torch::Tensor is_lead,
    torch::Tensor char_len,
    int64_t max_cp_span,
    int64_t max_byte_span,
    int64_t refill_int,
    int64_t cost_int,
    int64_t burst_int
);

std::vector<torch::Tensor> causal_refractory_gate_cuda_impl(torch::Tensor p_raw);

torch::Tensor birkhoff_greedy_perm_cuda_impl(torch::Tensor scores);

torch::Tensor crystal_scan_fwd_cuda_impl(
    torch::Tensor s0,
    torch::Tensor kw_eff,
    torch::Tensor v_ste,
    torch::Tensor P_eff,
    bool is_hard,
    double scale_r
);

std::vector<torch::Tensor> crystal_scan_bwd_cuda_impl(
    torch::Tensor s_out,
    torch::Tensor kw_eff,
    torch::Tensor kw_vg,
    torch::Tensor v_ste,
    torch::Tensor P_eff,
    torch::Tensor grad_out,
    bool is_hard,
    double scale_r
);

torch::Tensor woodbury_tern_p_uncerts_cuda_impl(
    torch::Tensor qe_m,
    torch::Tensor ke_m,
    torch::Tensor bm_d,
    torch::Tensor gm_d,
    double inv_lam,
    double tern_gate
);

std::vector<torch::Tensor> woodbury_tern_m_fwd_cuda_impl(
    torch::Tensor q_m,
    torch::Tensor k_m,
    torch::Tensor v_m,
    torch::Tensor b_m,
    torch::Tensor g_m,
    double inv_dh,
    int64_t Bc
);

std::vector<torch::Tensor> woodbury_tern_m_bwd_cuda_impl(
    torch::Tensor q_m,
    torch::Tensor k_m,
    torch::Tensor v_m,
    torch::Tensor b_m,
    torch::Tensor g_m,
    torch::Tensor M_starts,
    torch::Tensor grad_out,
    double inv_dh,
    int64_t Bc
);

torch::Tensor fused_ternary_ste_cuda_impl(
    torch::Tensor w,
    double mix,
    int64_t out_dtype_code
);

torch::Tensor utf8_chunk_span_cuda(
    torch::Tensor raw_cand,
    torch::Tensor valid_mask,
    torch::Tensor can_fire_mask,
    torch::Tensor is_lead,
    torch::Tensor char_len,
    int64_t max_cp_span,
    int64_t max_byte_span,
    int64_t refill_int,
    int64_t cost_int,
    int64_t burst_int
) {
    return utf8_chunk_span_cuda_impl(
        raw_cand, valid_mask, can_fire_mask, is_lead, char_len,
        max_cp_span, max_byte_span, refill_int, cost_int, burst_int
    );
}

std::vector<torch::Tensor> causal_refractory_gate_cuda(torch::Tensor p_raw) {
    return causal_refractory_gate_cuda_impl(p_raw);
}

torch::Tensor birkhoff_greedy_perm_cuda(torch::Tensor scores) {
    return birkhoff_greedy_perm_cuda_impl(scores);
}

torch::Tensor crystal_scan_fwd_cuda(
    torch::Tensor s0,
    torch::Tensor kw_eff,
    torch::Tensor v_ste,
    torch::Tensor P_eff,
    bool is_hard,
    double scale_r
) {
    return crystal_scan_fwd_cuda_impl(s0, kw_eff, v_ste, P_eff, is_hard, scale_r);
}

std::vector<torch::Tensor> crystal_scan_bwd_cuda(
    torch::Tensor s_out,
    torch::Tensor kw_eff,
    torch::Tensor kw_vg,
    torch::Tensor v_ste,
    torch::Tensor P_eff,
    torch::Tensor grad_out,
    bool is_hard,
    double scale_r
) {
    return crystal_scan_bwd_cuda_impl(s_out, kw_eff, kw_vg, v_ste, P_eff, grad_out, is_hard, scale_r);
}

torch::Tensor woodbury_tern_p_uncerts_cuda(
    torch::Tensor qe_m,
    torch::Tensor ke_m,
    torch::Tensor bm_d,
    torch::Tensor gm_d,
    double inv_lam,
    double tern_gate
) {
    return woodbury_tern_p_uncerts_cuda_impl(qe_m, ke_m, bm_d, gm_d, inv_lam, tern_gate);
}

std::vector<torch::Tensor> woodbury_tern_m_fwd_cuda(
    torch::Tensor q_m,
    torch::Tensor k_m,
    torch::Tensor v_m,
    torch::Tensor b_m,
    torch::Tensor g_m,
    double inv_dh,
    int64_t Bc
) {
    return woodbury_tern_m_fwd_cuda_impl(q_m, k_m, v_m, b_m, g_m, inv_dh, Bc);
}

std::vector<torch::Tensor> woodbury_tern_m_bwd_cuda(
    torch::Tensor q_m,
    torch::Tensor k_m,
    torch::Tensor v_m,
    torch::Tensor b_m,
    torch::Tensor g_m,
    torch::Tensor M_starts,
    torch::Tensor grad_out,
    double inv_dh,
    int64_t Bc
) {
    return woodbury_tern_m_bwd_cuda_impl(q_m, k_m, v_m, b_m, g_m, M_starts, grad_out, inv_dh, Bc);
}

torch::Tensor fused_ternary_ste_cuda(
    torch::Tensor w,
    double mix,
    int64_t out_dtype_code
) {
    return fused_ternary_ste_cuda_impl(w, mix, out_dtype_code);
}
"""

_CUDA_KERNEL_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <type_traits>
#include <vector>

namespace {

__device__ __forceinline__ float warp_sum32(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    }
    return v;
}

template <int D_CONST>
__device__ __forceinline__ float warp_reduce_smem_vec(const float* vec, int d_runtime, int lane) {
    if constexpr (D_CONST == 32) {
        return warp_sum32(vec[lane]);
    } else if constexpr (D_CONST == 48) {
        float v = vec[lane];
        if (lane < 16) v += vec[lane + 32];
        return warp_sum32(v);
    } else {
        float v = (lane < d_runtime) ? vec[lane] : 0.0f;
        if (lane + 32 < d_runtime) v += vec[lane + 32];
        return warp_sum32(v);
    }
}

// ---------------------------------------------------------------------------
// 1. UTF-8 Leaky-Bucket Boundary Guard (`_enforce_utf8_chunk_span_jit`)
// ---------------------------------------------------------------------------
__global__ void utf8_chunk_span_kernel(
    const bool* __restrict__ raw_cand,
    const bool* __restrict__ valid_mask,
    const bool* __restrict__ can_fire_mask,
    const int64_t* __restrict__ is_lead,
    const int64_t* __restrict__ char_len,
    bool* __restrict__ out,
    int64_t B,
    int64_t T,
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
    const int64_t base = b * T;

    for (int64_t t = 0; t < T; ++t) {
        const int64_t idx = base + t;
        const bool vt = valid_mask[idx];
        const int64_t ld = is_lead[idx];
        bool fire = false;
        if (t == 0) {
            fire = vt;
        } else {
            if (vt) {
                int64_t cr = credits + refill_int;
                credits = cr > burst_int ? burst_int : cr;
            }
            const bool can_soft = (credits >= cost_int) && raw_cand[idx];
            const bool must_span = (cp_run >= max_cp_span) || ((byte_run + char_len[idx]) > max_byte_span);
            fire = can_fire_mask[idx] && (can_soft || must_span);
            if (fire) {
                int64_t cr = credits - cost_int;
                credits = cr < 0 ? 0 : cr;
            }
        }
        out[idx] = fire;
        if (vt) {
            byte_run = fire ? 1 : (byte_run + 1);
            cp_run = fire ? ld : (cp_run + ld);
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Causal Refractory Gate (`causal_refractory_gate_jit`)
// ---------------------------------------------------------------------------
__global__ void causal_refractory_gate_kernel(
    const float* __restrict__ p_raw,
    float* __restrict__ out,
    float* __restrict__ mult_out,
    int64_t B,
    int64_t M,
    int64_t H
) {
    const int64_t bh = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (bh >= B * H) return;
    const int64_t b = bh / H;
    const int64_t h = bh % H;

    float f_m1 = 0.0f;
    float f_m2 = 0.0f;
    for (int64_t m = 0; m < M; ++m) {
        const float prev = f_m1 > f_m2 ? f_m1 : f_m2;
        const float inh = 1.0f / (1.0f + expf(-(prev - 0.35f) * 12.0f));
        const float mult = 1.0f - 0.92f * inh;
        const int64_t off = (b * M + m) * H + h;
        const float pm = p_raw[off] * mult;
        out[off] = pm;
        mult_out[off] = mult;
        f_m2 = f_m1;
        f_m1 = pm;
    }
}

// ---------------------------------------------------------------------------
// 3. Birkhoff Confidence-Ordered Greedy Permutation Projection (`G <= 32`)
// ---------------------------------------------------------------------------
__global__ void birkhoff_greedy_perm_kernel(
    const float* __restrict__ scores,
    float* __restrict__ P_hard,
    int64_t N,
    int G
) {
    const int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float rmax[32];
    int order[32];
    bool col_used[32];
    const float* sc = scores + n * G * G;
    float* out = P_hard + n * G * G;

    for (int r = 0; r < G; ++r) {
        float mx = sc[r * G + 0];
        for (int c = 1; c < G; ++c) {
            float v = sc[r * G + c];
            if (v > mx) mx = v;
        }
        rmax[r] = mx;
        order[r] = r;
        col_used[r] = false;
    }

    for (int i = 1; i < G; ++i) {
        const int kr = order[i];
        const float kv = rmax[kr];
        int j = i - 1;
        while (j >= 0 && rmax[order[j]] < kv) {
            order[j + 1] = order[j];
            --j;
        }
        order[j + 1] = kr;
    }

    for (int step = 0; step < G; ++step) {
        const int r = order[step];
        int best_c = 0;
        float best_v = sc[r * G + 0] - (col_used[0] ? 1e6f : 0.0f);
        for (int c = 1; c < G; ++c) {
            const float v = sc[r * G + c] - (col_used[c] ? 1e6f : 0.0f);
            if (v > best_v) {
                best_v = v;
                best_c = c;
            }
        }
        col_used[best_c] = true;
        out[r * G + best_c] = 1.0f;
    }
}

// ---------------------------------------------------------------------------
// 4. Diagonal-Overwrite Birkhoff-Voronoi Crystal Scan v3 (`d_s == 32, G_h <= 32`)
//    Specialized `GH=12` + double-buffered SMEM (1 barrier fwd, 3 barriers bwd)
// ---------------------------------------------------------------------------
template <int GH, bool IS_HARD>
__global__ void crystal_scan_fwd_kernel(
    const float* __restrict__ s0,      // (BH, G_h, 32)
    const float* __restrict__ kw_eff,  // (BH, M, G_h)
    const float* __restrict__ v_ste,   // (BH, M, 32)
    const float* __restrict__ P_eff,   // (BH, M, G_h, G_h)
    float* __restrict__ s_out,         // (BH, M, G_h, 32)
    int64_t M,
    int G_h_arg,
    float scale_r
) {
    constexpr int MAX_G = (GH > 0) ? GH : 32;
    const int G = (GH > 0) ? GH : G_h_arg;
    __shared__ float sh_sw[2][MAX_G][33];

    const int64_t bh = static_cast<int64_t>(blockIdx.x);
    const int tx = static_cast<int>(threadIdx.x);  // 0..31 (d)
    const int ty = static_cast<int>(threadIdx.y);  // 0..G-1 (g)

    float s_reg = s0[(bh * G + ty) * 32 + tx];
    int buf = 0;

    for (int64_t m = 0; m < M; ++m) {
        const int64_t bm = bh * M + m;
        s_out[(bm * G + ty) * 32 + tx] = s_reg;

        const float kwe = kw_eff[bm * G + ty];
        const float vs_val = v_ste[bm * 32 + tx];
        const float p_lane = (tx < G) ? P_eff[(bm * G + ty) * G + tx] : 0.0f;

        sh_sw[buf][ty][tx] = (1.0f - kwe) * s_reg + kwe * vs_val;
        __syncthreads();

        float s_next = 0.0f;
        if constexpr (GH > 0) {
            #pragma unroll
            for (int j = 0; j < GH; ++j) {
                const float pij = __shfl_sync(0xFFFFFFFFu, p_lane, j);
                s_next += pij * sh_sw[buf][j][tx];
            }
        } else {
            for (int j = 0; j < G; ++j) {
                const float pij = __shfl_sync(0xFFFFFFFFu, p_lane, j);
                s_next += pij * sh_sw[buf][j][tx];
            }
        }

        if constexpr (IS_HARD) {
            s_reg = s_next;
        } else {
            const float sq = warp_sum32(s_next * s_next);
            s_reg = s_next * (rsqrtf(sq + 1e-6f) * scale_r);
        }
        buf ^= 1;
    }
}

template <int GH, bool IS_HARD>
__global__ void crystal_scan_bwd_kernel(
    const float* __restrict__ s_out,     // (BH, M, G_h, 32)
    const float* __restrict__ kw_eff,    // (BH, M, G_h)
    const float* __restrict__ kw_vg,     // (BH, M, G_h)
    const float* __restrict__ v_ste,     // (BH, M, 32)
    const float* __restrict__ P_eff,     // (BH, M, G_h, G_h)
    const float* __restrict__ grad_out,  // (BH, M, G_h, 32)
    float* __restrict__ g_s0,            // (BH, G_h, 32)
    float* __restrict__ g_kwe,           // (BH, M, G_h)
    float* __restrict__ g_vs,            // (BH, M, 32)
    float* __restrict__ g_P,             // (BH, M, G_h, G_h)
    int64_t M,
    int G_h_arg,
    float scale_r
) {
    constexpr int MAX_G = (GH > 0) ? GH : 32;
    const int G = (GH > 0) ? GH : G_h_arg;
    __shared__ float sh_sw[MAX_G][33];
    __shared__ float sh_gsn[MAX_G][33];
    __shared__ float sh_gsw[2][MAX_G][33];
    __shared__ float sh_P[MAX_G][MAX_G + 1];

    const int64_t bh = static_cast<int64_t>(blockIdx.x);
    const int tx = static_cast<int>(threadIdx.x);  // 0..31 (d)
    const int ty = static_cast<int>(threadIdx.y);  // 0..G-1 (g)

    float g_s = 0.0f;
    int buf = 0;

    for (int64_t m = M - 1; m >= 0; --m) {
        const int64_t bm = bh * M + m;
        const float s_prev = s_out[(bm * G + ty) * 32 + tx];
        const float kwe = kw_eff[bm * G + ty];
        const float kwv = kw_vg[bm * G + ty];
        const float vs_val = v_ste[bm * 32 + tx];
        const float p_lane = (tx < G) ? P_eff[(bm * G + ty) * G + tx] : 0.0f;

        if (tx < G) {
            sh_P[ty][tx] = p_lane;
        }
        sh_sw[ty][tx] = (1.0f - kwe) * s_prev + kwe * vs_val;
        __syncthreads();  // Barrier 1: sh_P & sh_sw ready

        float g_sn = g_s;
        if constexpr (!IS_HARD) {
            float s_next = 0.0f;
            if constexpr (GH > 0) {
                #pragma unroll
                for (int j = 0; j < GH; ++j) {
                    const float pij = __shfl_sync(0xFFFFFFFFu, p_lane, j);
                    s_next += pij * sh_sw[j][tx];
                }
            } else {
                for (int j = 0; j < G; ++j) {
                    const float pij = __shfl_sync(0xFFFFFFFFu, p_lane, j);
                    s_next += pij * sh_sw[j][tx];
                }
            }
            const float sq = warp_sum32(s_next * s_next) + 1e-6f;
            const float dot = warp_sum32(g_s * s_next);
            const float inv = rsqrtf(sq);
            g_sn = (inv * scale_r) * (g_s - s_next * (dot * (inv * inv)));
        }
        sh_gsn[ty][tx] = g_sn;

        // Coalesced g_P[bm, ty, 0..G-1] = sum_d g_sn[ty, d] * sh_sw[j, d]
        float my_gp = 0.0f;
        if constexpr (GH > 0) {
            #pragma unroll
            for (int j = 0; j < GH; ++j) {
                const float pd = warp_sum32(g_sn * sh_sw[j][tx]);
                if (tx == j) my_gp = pd;
            }
        } else {
            for (int j = 0; j < G; ++j) {
                const float pd = warp_sum32(g_sn * sh_sw[j][tx]);
                if (tx == j) my_gp = pd;
            }
        }
        if (tx < G) {
            g_P[(bm * G + ty) * G + tx] = my_gp;
        }
        __syncthreads();  // Barrier 2: sh_gsn ready & all sh_sw reads done

        // g_sw[ty, d] = sum_i P[i, ty] * g_sn[i, d]
        float g_sw_val = 0.0f;
        if constexpr (GH > 0) {
            #pragma unroll
            for (int i = 0; i < GH; ++i) {
                g_sw_val += sh_P[i][ty] * sh_gsn[i][tx];
            }
        } else {
            for (int i = 0; i < G; ++i) {
                g_sw_val += sh_P[i][ty] * sh_gsn[i][tx];
            }
        }

        const float dk = warp_sum32(g_sw_val * (vs_val - s_prev));
        if (tx == 0) {
            g_kwe[bm * G + ty] = dk;
        }

        sh_gsw[buf][ty][tx] = g_sw_val * kwv;
        const float go_m = grad_out[(bm * G + ty) * 32 + tx];
        g_s = go_m + g_sw_val * (1.0f - kwe);
        __syncthreads();  // Barrier 3: sh_gsw[buf] ready & all sh_P/sh_gsn reads done

        if (ty == 0) {
            float dv = 0.0f;
            if constexpr (GH > 0) {
                #pragma unroll
                for (int i = 0; i < GH; ++i) {
                    dv += sh_gsw[buf][i][tx];
                }
            } else {
                for (int i = 0; i < G; ++i) {
                    dv += sh_gsw[buf][i][tx];
                }
            }
            g_vs[bm * 32 + tx] = dv;
        }
        buf ^= 1;
    }

    g_s0[(bh * G + ty) * 32 + tx] = g_s;
}

// ---------------------------------------------------------------------------
// 5a. Woodbury SPD Precision & Uncertainty Scan v3 (`no_grad`, `d<=64, C=4`)
//     Specialized `DK=32, 48`, fused 5-vector GEMV, zero serial loops, 3 barriers
// ---------------------------------------------------------------------------
template <int DK>
__global__ void woodbury_tern_p_uncerts_kernel(
    const float* __restrict__ qe_m,   // (M, BH, d_k, 4)
    const float* __restrict__ ke_m,   // (M, BH, d_k)
    const float* __restrict__ bm_d,   // (M, BH)
    const float* __restrict__ gm_d,   // (M, BH)
    float* __restrict__ uncerts,      // (M, BH, 4)
    int64_t M_len,
    int64_t BH,
    int d_k_arg,
    float inv_lam,
    float tern_gate
) {
    constexpr int MAX_D = (DK > 0) ? DK : 64;
    const int d_k = (DK > 0) ? DK : d_k_arg;
    __shared__ float sh_P[MAX_D][MAX_D + 1];
    __shared__ float sh_ke[MAX_D];
    __shared__ float sh_pk[MAX_D];
    __shared__ float sh_qe[4][MAX_D];
    __shared__ float sh_tmp[5][MAX_D];

    const int64_t bh = static_cast<int64_t>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);  // 0..255
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const float lam_reg = 1.0f / inv_lam;
    const float slope = (tern_gate > 0.0f) ? 20.0f : 18.0f;
    const float rho_g = 0.002f * tern_gate;
    const float rho_c = 0.002f * tern_gate;
    const int dd = d_k * d_k;
    const int gemv_v = (d_k > 0) ? (tid / d_k) : 0;
    const int gemv_r = tid - gemv_v * d_k;

    if constexpr (DK == 32) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int i = (warp_id << 2) + r;
            sh_P[i][lane] = (i == lane) ? inv_lam : 0.0f;
        }
    } else {
        for (int idx = tid; idx < dd; idx += 256) {
            const int i = idx / d_k;
            const int j = idx - i * d_k;
            sh_P[i][j] = (i == j) ? inv_lam : 0.0f;
        }
    }
    __syncthreads();

    for (int64_t m = 0; m < M_len; ++m) {
        const int64_t mb = m * BH + bh;
        if (tid < d_k) {
            sh_ke[tid] = ke_m[mb * d_k + tid];
        }
        if (tid < d_k * 4) {
            const int i = tid >> 2;
            const int c = tid & 3;
            sh_qe[c][i] = qe_m[(mb * d_k + i) * 4 + c];
        }
        __syncthreads();

        if constexpr (DK == 32) {
            if (warp_id < 4) {
                float pq = 0.0f;
                #pragma unroll
                for (int j = 0; j < 32; ++j) {
                    pq += sh_P[lane][j] * sh_qe[warp_id][j];
                }
                const float acc = warp_sum32(sh_qe[warp_id][lane] * pq);
                if (lane == 0) {
                    float u = acc * lam_reg;
                    if (u < 0.0f) u = 0.0f;
                    if (u > 1.5f) u = 1.5f;
                    uncerts[mb * 4 + warp_id] = u;
                }
            } else if (warp_id == 4) {
                float pk = 0.0f;
                #pragma unroll
                for (int j = 0; j < 32; ++j) {
                    pk += sh_P[lane][j] * sh_ke[j];
                }
                sh_pk[lane] = pk;
                float quad = warp_sum32(sh_ke[lane] * pk);
                if (lane == 0) {
                    if (quad < 1e-5f) quad = 1e-5f;
                    const float bt = bm_d[mb];
                    const float gt = gm_d[mb];
                    const float s_gate = 1.0f / (1.0f + expf(-(bt - 0.26f) * slope));
                    const float w_cov = s_gate * s_gate * s_gate;
                    const float nov = w_cov * 0.92f;
                    sh_tmp[4][0] = nov / (1.0f + nov * quad);
                    float rr = (rho_g > 0.0f) ? ((1.0f - gt) * w_cov * rho_g) : 0.0f;
                    if (rr < 0.0f) rr = 0.0f;
                    if (rr > rho_c) rr = rho_c;
                    sh_tmp[4][1] = rr;
                }
            }
            __syncthreads();

            const float alpha = sh_tmp[4][0];
            const float rho = sh_tmp[4][1];
            const float one_m_rho = 1.0f - rho;
            const float diag_add = rho * inv_lam;
            const float pk_j = sh_pk[lane];
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int i = (warp_id << 2) + r;
                const float p_mid = sh_P[i][lane] - alpha * (sh_pk[i] * pk_j);
                sh_P[i][lane] = one_m_rho * p_mid + ((i == lane) ? diag_add : 0.0f);
            }
            __syncthreads();
        } else if constexpr (DK == 48) {
            if (tid < 240) {
                if (gemv_v < 4) {
                    float pq = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 48; ++j) {
                        pq += sh_P[gemv_r][j] * sh_qe[gemv_v][j];
                    }
                    sh_tmp[gemv_v][gemv_r] = sh_qe[gemv_v][gemv_r] * pq;
                } else {
                    float pk = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 48; ++j) {
                        pk += sh_P[gemv_r][j] * sh_ke[j];
                    }
                    sh_pk[gemv_r] = pk;
                    sh_tmp[4][gemv_r] = sh_ke[gemv_r] * pk;
                }
            }
            __syncthreads();

            if (warp_id < 4) {
                const float acc = warp_reduce_smem_vec<48>(sh_tmp[warp_id], 48, lane);
                if (lane == 0) {
                    float u = acc * lam_reg;
                    if (u < 0.0f) u = 0.0f;
                    if (u > 1.5f) u = 1.5f;
                    uncerts[mb * 4 + warp_id] = u;
                }
            }
            float quad = warp_reduce_smem_vec<48>(sh_tmp[4], 48, lane);
            if (quad < 1e-5f) quad = 1e-5f;
            const float bt = bm_d[mb];
            const float gt = gm_d[mb];
            const float s_gate = 1.0f / (1.0f + expf(-(bt - 0.26f) * slope));
            const float w_cov = s_gate * s_gate * s_gate;
            const float nov = w_cov * 0.92f;
            const float alpha = nov / (1.0f + nov * quad);
            float rho = (rho_g > 0.0f) ? ((1.0f - gt) * w_cov * rho_g) : 0.0f;
            if (rho < 0.0f) rho = 0.0f;
            if (rho > rho_c) rho = rho_c;

            const float one_m_rho = 1.0f - rho;
            const float diag_add = rho * inv_lam;
            #pragma unroll
            for (int step = 0; step < 9; ++step) {
                const int idx = tid + step * 256;
                const int i = idx / 48;
                const int j = idx - i * 48;
                const float p_mid = sh_P[i][j] - alpha * (sh_pk[i] * sh_pk[j]);
                sh_P[i][j] = one_m_rho * p_mid + ((i == j) ? diag_add : 0.0f);
            }
            __syncthreads();
        } else {
            for (int idx = tid; idx < d_k * 5; idx += 256) {
                const int v = idx / d_k;
                const int r = idx - v * d_k;
                if (v < 4) {
                    float pq = 0.0f;
                    for (int j = 0; j < d_k; ++j) {
                        pq += sh_P[r][j] * sh_qe[v][j];
                    }
                    sh_tmp[v][r] = sh_qe[v][r] * pq;
                } else {
                    float pk = 0.0f;
                    for (int j = 0; j < d_k; ++j) {
                        pk += sh_P[r][j] * sh_ke[j];
                    }
                    sh_pk[r] = pk;
                    sh_tmp[4][r] = sh_ke[r] * pk;
                }
            }
            __syncthreads();

            if (warp_id < 4) {
                const float acc = warp_reduce_smem_vec<0>(sh_tmp[warp_id], d_k, lane);
                if (lane == 0) {
                    float u = acc * lam_reg;
                    if (u < 0.0f) u = 0.0f;
                    if (u > 1.5f) u = 1.5f;
                    uncerts[mb * 4 + warp_id] = u;
                }
            }
            float quad = warp_reduce_smem_vec<0>(sh_tmp[4], d_k, lane);
            if (quad < 1e-5f) quad = 1e-5f;
            const float bt = bm_d[mb];
            const float gt = gm_d[mb];
            const float s_gate = 1.0f / (1.0f + expf(-(bt - 0.26f) * slope));
            const float w_cov = s_gate * s_gate * s_gate;
            const float nov = w_cov * 0.92f;
            const float alpha = nov / (1.0f + nov * quad);
            float rho = (rho_g > 0.0f) ? ((1.0f - gt) * w_cov * rho_g) : 0.0f;
            if (rho < 0.0f) rho = 0.0f;
            if (rho > rho_c) rho = rho_c;

            const float one_m_rho = 1.0f - rho;
            const float diag_add = rho * inv_lam;
            for (int idx = tid; idx < dd; idx += 256) {
                const int i = idx / d_k;
                const int j = idx - i * d_k;
                const float p_mid = sh_P[i][j] - alpha * (sh_pk[i] * sh_pk[j]);
                sh_P[i][j] = one_m_rho * p_mid + ((i == j) ? diag_add : 0.0f);
            }
            __syncthreads();
        }
    }
}

// ---------------------------------------------------------------------------
// 5b. Woodbury M-Matrix Scan Forward & Backward v3 (`d<=64, C=4`)
//     Specialized `<DK, DV, IS_TERN>`: FP skips sqrtf/rsqrtf/coeff, multi-warp GEMV,
//     compact SMEM (`10.5/21.6 KB`), zero serial `tid==0` loops
// ---------------------------------------------------------------------------
template <int DK, int DV, bool IS_TERN>
__global__ void woodbury_tern_m_fwd_kernel(
    const float* __restrict__ q_m,   // (M, BH, d_k, 4)
    const float* __restrict__ k_m,   // (M, BH, d_k)
    const float* __restrict__ v_m,   // (M, BH, d_v)
    const float* __restrict__ b_m,   // (M, BH)
    const float* __restrict__ g_m,   // (M, BH)
    float* __restrict__ out_m,       // (M, BH, d_v, 4)
    float* __restrict__ M_starts,    // (n_blocks, BH, d_v, d_k)
    int64_t M_len,
    int64_t BH,
    int d_k_arg,
    int d_v_arg,
    float inv_dh,
    int64_t Bc
) {
    constexpr int MAX_DK = (DK > 0) ? DK : 64;
    constexpr int MAX_DV = (DV > 0) ? DV : 64;
    const int d_k = (DK > 0) ? DK : d_k_arg;
    const int d_v = (DV > 0) ? DV : d_v_arg;

    __shared__ float sh_M[MAX_DV][MAX_DK + 1];
    __shared__ float sh_q[4][MAX_DK];
    __shared__ float sh_k[MAX_DK];
    __shared__ float sh_v[MAX_DV];
    __shared__ float sh_err[MAX_DV];
    __shared__ float sh_sq[MAX_DV];
    __shared__ float sh_scale;

    const int64_t bh = static_cast<int64_t>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int dv_dk = d_v * d_k;
    const int gemv_v = (d_v > 0) ? (tid / d_v) : 0;
    const int gemv_r = tid - gemv_v * d_v;

    if constexpr (DK == 32 && DV == 32) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            sh_M[(warp_id << 2) + r][lane] = 0.0f;
        }
    } else {
        for (int idx = tid; idx < dv_dk; idx += 256) {
            const int i = idx / d_k;
            sh_M[i][idx - i * d_k] = 0.0f;
        }
    }
    __syncthreads();

    for (int64_t m = 0; m < M_len; ++m) {
        if ((m % Bc) == 0) {
            const int64_t bi = m / Bc;
            float* dst = M_starts + (bi * BH + bh) * dv_dk;
            if constexpr (DK == 32 && DV == 32) {
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const int i = (warp_id << 2) + r;
                    dst[(i << 5) + lane] = sh_M[i][lane];
                }
            } else {
                for (int idx = tid; idx < dv_dk; idx += 256) {
                    const int i = idx / d_k;
                    dst[idx] = sh_M[i][idx - i * d_k];
                }
            }
        }
        const int64_t mb = m * BH + bh;
        if (tid < d_k) sh_k[tid] = k_m[mb * d_k + tid];
        if (tid < d_v) sh_v[tid] = v_m[mb * d_v + tid];
        if (tid < d_k * 4) {
            const int j = tid >> 2;
            const int c = tid & 3;
            sh_q[c][j] = q_m[(mb * d_k + j) * 4 + c];
        }
        __syncthreads();

        const float bt = b_m[mb];
        const float gt = g_m[mb];

        if constexpr (DK == 32 && DV == 32) {
            if (warp_id < 4) {
                float acc = 0.0f;
                #pragma unroll
                for (int j = 0; j < 32; ++j) {
                    acc += sh_M[lane][j] * sh_q[warp_id][j];
                }
                out_m[(mb * 32 + lane) * 4 + warp_id] = acc;
            } else if (warp_id == 4) {
                float pred = 0.0f;
                #pragma unroll
                for (int j = 0; j < 32; ++j) {
                    pred += sh_M[lane][j] * sh_k[j];
                }
                const float err = sh_v[lane] - gt * pred;
                if constexpr (IS_TERN) {
                    sh_err[lane] = err;
                    const float s2 = warp_sum32(err * err);
                    if (lane == 0) {
                        const float r_sq = s2 * inv_dh;
                        sh_scale = rsqrtf(sqrtf(1.0f + r_sq * r_sq));
                    }
                } else {
                    sh_err[lane] = (1.35f * bt) * err;
                }
            }
            __syncthreads();

            const float kj = sh_k[lane];
            if constexpr (IS_TERN) {
                const float b_eff = 1.35f * bt * sh_scale;
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const int i = (warp_id << 2) + r;
                    sh_M[i][lane] = gt * sh_M[i][lane] + (b_eff * sh_err[i]) * kj;
                }
            } else {
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const int i = (warp_id << 2) + r;
                    sh_M[i][lane] = gt * sh_M[i][lane] + sh_err[i] * kj;
                }
            }
            __syncthreads();
        } else if constexpr (DK == 48 && DV == 48) {
            if (tid < 240) {
                if (gemv_v < 4) {
                    float acc = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 48; ++j) {
                        acc += sh_M[gemv_r][j] * sh_q[gemv_v][j];
                    }
                    out_m[(mb * 48 + gemv_r) * 4 + gemv_v] = acc;
                } else {
                    float pred = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 48; ++j) {
                        pred += sh_M[gemv_r][j] * sh_k[j];
                    }
                    const float err = sh_v[gemv_r] - gt * pred;
                    if constexpr (IS_TERN) {
                        sh_err[gemv_r] = err;
                        sh_sq[gemv_r] = err * err;
                    } else {
                        sh_err[gemv_r] = (1.35f * bt) * err;
                    }
                }
            }
            __syncthreads();

            if constexpr (IS_TERN) {
                const float s2 = warp_reduce_smem_vec<48>(sh_sq, 48, lane);
                const float r_sq = s2 * inv_dh;
                const float b_eff = 1.35f * bt * rsqrtf(sqrtf(1.0f + r_sq * r_sq));
                #pragma unroll
                for (int step = 0; step < 9; ++step) {
                    const int idx = tid + step * 256;
                    const int i = idx / 48;
                    const int j = idx - i * 48;
                    sh_M[i][j] = gt * sh_M[i][j] + (b_eff * sh_err[i]) * sh_k[j];
                }
            } else {
                #pragma unroll
                for (int step = 0; step < 9; ++step) {
                    const int idx = tid + step * 256;
                    const int i = idx / 48;
                    const int j = idx - i * 48;
                    sh_M[i][j] = gt * sh_M[i][j] + sh_err[i] * sh_k[j];
                }
            }
            __syncthreads();
        } else {
            for (int idx = tid; idx < d_v * 5; idx += 256) {
                const int v = idx / d_v;
                const int r = idx - v * d_v;
                if (v < 4) {
                    float acc = 0.0f;
                    for (int j = 0; j < d_k; ++j) {
                        acc += sh_M[r][j] * sh_q[v][j];
                    }
                    out_m[(mb * d_v + r) * 4 + v] = acc;
                } else {
                    float pred = 0.0f;
                    for (int j = 0; j < d_k; ++j) {
                        pred += sh_M[r][j] * sh_k[j];
                    }
                    const float err = sh_v[r] - gt * pred;
                    if constexpr (IS_TERN) {
                        sh_err[r] = err;
                        sh_sq[r] = err * err;
                    } else {
                        sh_err[r] = (1.35f * bt) * err;
                    }
                }
            }
            __syncthreads();

            if constexpr (IS_TERN) {
                const float s2 = warp_reduce_smem_vec<0>(sh_sq, d_v, lane);
                const float r_sq = s2 * inv_dh;
                const float b_eff = 1.35f * bt * rsqrtf(sqrtf(1.0f + r_sq * r_sq));
                for (int idx = tid; idx < dv_dk; idx += 256) {
                    const int i = idx / d_k;
                    const int j = idx - i * d_k;
                    sh_M[i][j] = gt * sh_M[i][j] + (b_eff * sh_err[i]) * sh_k[j];
                }
            } else {
                for (int idx = tid; idx < dv_dk; idx += 256) {
                    const int i = idx / d_k;
                    const int j = idx - i * d_k;
                    sh_M[i][j] = gt * sh_M[i][j] + sh_err[i] * sh_k[j];
                }
            }
            __syncthreads();
        }
    }
}

template <int DK, int DV, bool IS_TERN>
__global__ void woodbury_tern_m_bwd_kernel(
    const float* __restrict__ q_m,       // (M, BH, d_k, 4)
    const float* __restrict__ k_m,       // (M, BH, d_k)
    const float* __restrict__ v_m,       // (M, BH, d_v)
    const float* __restrict__ b_m,       // (M, BH)
    const float* __restrict__ g_m,       // (M, BH)
    const float* __restrict__ M_starts,  // (n_blocks, BH, d_v, d_k)
    const float* __restrict__ grad_out,  // (M, BH, d_v, 4)
    float* __restrict__ M_blk_buf,       // (BH, Bc, d_v, d_k)
    float* __restrict__ dq,              // (M, BH, d_k, 4)
    float* __restrict__ dk,              // (M, BH, d_k)
    float* __restrict__ dv,              // (M, BH, d_v)
    float* __restrict__ db,              // (M, BH)
    float* __restrict__ dg,              // (M, BH)
    int64_t M_len,
    int64_t BH,
    int d_k_arg,
    int d_v_arg,
    float inv_dh,
    int64_t Bc,
    int64_t n_blocks
) {
    constexpr int MAX_DK = (DK > 0) ? DK : 64;
    constexpr int MAX_DV = (DV > 0) ? DV : 64;
    const int d_k = (DK > 0) ? DK : d_k_arg;
    const int d_v = (DV > 0) ? DV : d_v_arg;

    __shared__ float sh_M[MAX_DV][MAX_DK + 1];
    __shared__ float sh_gM[MAX_DV][MAX_DK + 1];
    __shared__ float sh_q[4][MAX_DK];
    __shared__ float sh_go[4][MAX_DV];
    __shared__ float sh_k[MAX_DK];
    __shared__ float sh_v[MAX_DV];
    __shared__ float sh_pred[MAX_DV];
    __shared__ float sh_err[MAX_DV];
    __shared__ float sh_sq[MAX_DV];
    __shared__ float sh_dot[MAX_DV];
    __shared__ float sh_gmk[MAX_DV];
    __shared__ float sh_gerr[MAX_DV];
    __shared__ float sh_gpred[MAX_DV];
    __shared__ float sh_erreff[MAX_DV];
    __shared__ float sh_warp_red[8];

    const int64_t bh = static_cast<int64_t>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);  // 0..255
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int dv_dk = d_v * d_k;
    const int gemv_v = (d_k > 0) ? (tid / d_k) : 0;
    const int gemv_r = tid - gemv_v * d_k;
    float* my_blk = M_blk_buf + (bh * Bc) * dv_dk;

    if constexpr (DK == 32 && DV == 32) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            sh_gM[(warp_id << 2) + r][lane] = 0.0f;
        }
    } else {
        for (int idx = tid; idx < dv_dk; idx += 256) {
            const int i = idx / d_k;
            sh_gM[i][idx - i * d_k] = 0.0f;
        }
    }
    __syncthreads();

    for (int64_t bi = n_blocks - 1; bi >= 0; --bi) {
        const int64_t st = bi * Bc;
        const int64_t en = (st + Bc) < M_len ? (st + Bc) : M_len;
        const int64_t blen = en - st;

        // 1. Load block start M and replay forward within [st, en) saving M_blk[0..blen-2]
        const float* src0 = M_starts + (bi * BH + bh) * dv_dk;
        if constexpr (DK == 32 && DV == 32) {
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int i = (warp_id << 2) + r;
                sh_M[i][lane] = src0[(i << 5) + lane];
            }
        } else {
            for (int idx = tid; idx < dv_dk; idx += 256) {
                const int i = idx / d_k;
                sh_M[i][idx - i * d_k] = src0[idx];
            }
        }
        __syncthreads();

        for (int64_t t = 0; t + 1 < blen; ++t) {
            float* dst_t = my_blk + t * dv_dk;
            if constexpr (DK == 32 && DV == 32) {
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const int i = (warp_id << 2) + r;
                    dst_t[(i << 5) + lane] = sh_M[i][lane];
                }
            } else {
                for (int idx = tid; idx < dv_dk; idx += 256) {
                    const int i = idx / d_k;
                    dst_t[idx] = sh_M[i][idx - i * d_k];
                }
            }

            const int64_t m = st + t;
            const int64_t mb = m * BH + bh;
            if (tid < d_k) sh_k[tid] = k_m[mb * d_k + tid];
            if (tid < d_v) sh_v[tid] = v_m[mb * d_v + tid];
            __syncthreads();

            const float bt = b_m[mb];
            const float gt = g_m[mb];

            // All 8 warps compute pred = M @ k in parallel
            if constexpr (DK == 32 && DV == 32) {
                const float kj = sh_k[lane];
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const int i = (warp_id << 2) + r;
                    const float pred = warp_sum32(sh_M[i][lane] * kj);
                    if (lane == 0) {
                        const float err = sh_v[i] - gt * pred;
                        if constexpr (IS_TERN) {
                            sh_err[i] = err;
                            sh_sq[i] = err * err;
                        } else {
                            sh_err[i] = (1.35f * bt) * err;
                        }
                    }
                }
            } else if constexpr (DK == 48 && DV == 48) {
                const float kj0 = sh_k[lane];
                const float kj1 = (lane < 16) ? sh_k[lane + 32] : 0.0f;
                #pragma unroll
                for (int r = 0; r < 6; ++r) {
                    const int i = warp_id * 6 + r;
                    float partial = sh_M[i][lane] * kj0;
                    if (lane < 16) partial += sh_M[i][lane + 32] * kj1;
                    const float pred = warp_sum32(partial);
                    if (lane == 0) {
                        const float err = sh_v[i] - gt * pred;
                        if constexpr (IS_TERN) {
                            sh_err[i] = err;
                            sh_sq[i] = err * err;
                        } else {
                            sh_err[i] = (1.35f * bt) * err;
                        }
                    }
                }
            } else {
                const float kj0 = (lane < d_k) ? sh_k[lane] : 0.0f;
                const float kj1 = (lane + 32 < d_k) ? sh_k[lane + 32] : 0.0f;
                for (int i = warp_id; i < d_v; i += 8) {
                    float partial = (lane < d_k) ? (sh_M[i][lane] * kj0) : 0.0f;
                    if (lane + 32 < d_k) partial += sh_M[i][lane + 32] * kj1;
                    const float pred = warp_sum32(partial);
                    if (lane == 0) {
                        const float err = sh_v[i] - gt * pred;
                        if constexpr (IS_TERN) {
                            sh_err[i] = err;
                            sh_sq[i] = err * err;
                        } else {
                            sh_err[i] = (1.35f * bt) * err;
                        }
                    }
                }
            }
            __syncthreads();

            float b_eff = 1.0f;
            if constexpr (IS_TERN) {
                const float s2 = warp_reduce_smem_vec<DV>(sh_sq, d_v, lane);
                const float r_sq = s2 * inv_dh;
                b_eff = 1.35f * bt * rsqrtf(sqrtf(1.0f + r_sq * r_sq));
            }

            if constexpr (DK == 32 && DV == 32) {
                const float kj = sh_k[lane];
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const int i = (warp_id << 2) + r;
                    const float e_i = IS_TERN ? (b_eff * sh_err[i]) : sh_err[i];
                    sh_M[i][lane] = gt * sh_M[i][lane] + e_i * kj;
                }
            } else if constexpr (DK == 48 && DV == 48) {
                #pragma unroll
                for (int step = 0; step < 9; ++step) {
                    const int idx = tid + step * 256;
                    const int i = idx / 48;
                    const int j = idx - i * 48;
                    const float e_i = IS_TERN ? (b_eff * sh_err[i]) : sh_err[i];
                    sh_M[i][j] = gt * sh_M[i][j] + e_i * sh_k[j];
                }
            } else {
                for (int idx = tid; idx < dv_dk; idx += 256) {
                    const int i = idx / d_k;
                    const int j = idx - i * d_k;
                    const float e_i = IS_TERN ? (b_eff * sh_err[i]) : sh_err[i];
                    sh_M[i][j] = gt * sh_M[i][j] + e_i * sh_k[j];
                }
            }
            __syncthreads();
        }

        // 2. Reverse-mode pass over [en - 1 .. st]; sh_M already holds M[st + blen - 1]
        for (int64_t t = blen - 1; t >= 0; --t) {
            const int64_t m = st + t;
            const int64_t mb = m * BH + bh;
            if (t < blen - 1) {
                const float* src_t = my_blk + t * dv_dk;
                if constexpr (DK == 32 && DV == 32) {
                    #pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        const int i = (warp_id << 2) + r;
                        sh_M[i][lane] = src_t[(i << 5) + lane];
                    }
                } else {
                    for (int idx = tid; idx < dv_dk; idx += 256) {
                        const int i = idx / d_k;
                        sh_M[i][idx - i * d_k] = src_t[idx];
                    }
                }
            }
            if (tid < d_k) sh_k[tid] = k_m[mb * d_k + tid];
            if (tid < d_v) sh_v[tid] = v_m[mb * d_v + tid];
            if (tid < d_k * 4) {
                const int j = tid >> 2;
                const int c = tid & 3;
                sh_q[c][j] = q_m[(mb * d_k + j) * 4 + c];
            }
            if (tid < d_v * 4) {
                const int i = tid >> 2;
                const int c = tid & 3;
                sh_go[c][i] = grad_out[(mb * d_v + i) * 4 + c];
            }
            __syncthreads();  // Barrier R1

            const float bt = b_m[mb];
            const float gt = g_m[mb];

            // Step A: 8-warp parallel GEMVs pred = M @ k and gmk = gM @ k
            if constexpr (DK == 32 && DV == 32) {
                const float kj = sh_k[lane];
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const int i = (warp_id << 2) + r;
                    const float pred = warp_sum32(sh_M[i][lane] * kj);
                    const float gmk = warp_sum32(sh_gM[i][lane] * kj);
                    if (lane == 0) {
                        const float err = sh_v[i] - gt * pred;
                        sh_pred[i] = pred;
                        sh_err[i] = err;
                        sh_gmk[i] = gmk;
                        sh_dot[i] = gmk * err;
                        if constexpr (IS_TERN) {
                            sh_sq[i] = err * err;
                        } else {
                            const float g_err = (1.35f * bt) * gmk;
                            sh_gerr[i] = g_err;
                            sh_gpred[i] = -gt * g_err;
                            sh_erreff[i] = err;
                        }
                    }
                }
            } else if constexpr (DK == 48 && DV == 48) {
                const float kj0 = sh_k[lane];
                const float kj1 = (lane < 16) ? sh_k[lane + 32] : 0.0f;
                #pragma unroll
                for (int r = 0; r < 6; ++r) {
                    const int i = warp_id * 6 + r;
                    float p_part = sh_M[i][lane] * kj0;
                    float g_part = sh_gM[i][lane] * kj0;
                    if (lane < 16) {
                        p_part += sh_M[i][lane + 32] * kj1;
                        g_part += sh_gM[i][lane + 32] * kj1;
                    }
                    const float pred = warp_sum32(p_part);
                    const float gmk = warp_sum32(g_part);
                    if (lane == 0) {
                        const float err = sh_v[i] - gt * pred;
                        sh_pred[i] = pred;
                        sh_err[i] = err;
                        sh_gmk[i] = gmk;
                        sh_dot[i] = gmk * err;
                        if constexpr (IS_TERN) {
                            sh_sq[i] = err * err;
                        } else {
                            const float g_err = (1.35f * bt) * gmk;
                            sh_gerr[i] = g_err;
                            sh_gpred[i] = -gt * g_err;
                            sh_erreff[i] = err;
                        }
                    }
                }
            } else {
                const float kj0 = (lane < d_k) ? sh_k[lane] : 0.0f;
                const float kj1 = (lane + 32 < d_k) ? sh_k[lane + 32] : 0.0f;
                for (int i = warp_id; i < d_v; i += 8) {
                    float p_part = (lane < d_k) ? (sh_M[i][lane] * kj0) : 0.0f;
                    float g_part = (lane < d_k) ? (sh_gM[i][lane] * kj0) : 0.0f;
                    if (lane + 32 < d_k) {
                        p_part += sh_M[i][lane + 32] * kj1;
                        g_part += sh_gM[i][lane + 32] * kj1;
                    }
                    const float pred = warp_sum32(p_part);
                    const float gmk = warp_sum32(g_part);
                    if (lane == 0) {
                        const float err = sh_v[i] - gt * pred;
                        sh_pred[i] = pred;
                        sh_err[i] = err;
                        sh_gmk[i] = gmk;
                        sh_dot[i] = gmk * err;
                        if constexpr (IS_TERN) {
                            sh_sq[i] = err * err;
                        } else {
                            const float g_err = (1.35f * bt) * gmk;
                            sh_gerr[i] = g_err;
                            sh_gpred[i] = -gt * g_err;
                            sh_erreff[i] = err;
                        }
                    }
                }
            }
            __syncthreads();  // Barrier R2

            if constexpr (IS_TERN) {
                const float s2 = warp_reduce_smem_vec<DV>(sh_sq, d_v, lane);
                const float dot_gmk_err = warp_reduce_smem_vec<DV>(sh_dot, d_v, lane);
                const float r_sq = s2 * inv_dh;
                const float a = 1.0f + r_sq * r_sq;
                const float phi = rsqrtf(sqrtf(a));
                const float dot_ge = (1.35f * bt) * dot_gmk_err;
                const float coeff = (r_sq * phi / a) * inv_dh * dot_ge;
                if (tid == 0) {
                    db[mb] = 1.35f * phi * dot_gmk_err;
                }
                if (tid < d_v) {
                    const float g_eff = (1.35f * bt) * sh_gmk[tid];
                    const float g_err = phi * g_eff - coeff * sh_err[tid];
                    sh_gerr[tid] = g_err;
                    sh_gpred[tid] = -gt * g_err;
                    sh_erreff[tid] = sh_err[tid] * phi;
                    dv[mb * d_v + tid] = g_err;
                }
                __syncthreads();  // Barrier R2b
            } else {
                // Coalesced store of dv
                if (tid < d_v) {
                    dv[mb * d_v + tid] = sh_gerr[tid];
                }
            }

            // Step B: Fused transpose GEMVs for dq[0..3] and dk (+ db if !IS_TERN)
            if constexpr (DK == 32 && DV == 32) {
                if (warp_id < 4) {
                    float acc = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < 32; ++i) {
                        acc += sh_M[i][lane] * sh_go[warp_id][i];
                    }
                    dq[(mb * 32 + lane) * 4 + warp_id] = acc;
                } else if (warp_id == 4) {
                    float gk_u = 0.0f;
                    float gk_p = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < 32; ++i) {
                        gk_u += sh_gM[i][lane] * sh_erreff[i];
                        gk_p += sh_M[i][lane] * sh_gpred[i];
                    }
                    dk[mb * 32 + lane] = (1.35f * bt) * gk_u + gk_p;
                } else {
                    if constexpr (!IS_TERN) {
                        if (warp_id == 5) {
                            const float dot_gmk_err = warp_sum32(sh_dot[lane]);
                            if (lane == 0) {
                                db[mb] = 1.35f * dot_gmk_err;
                            }
                        }
                    }
                }
            } else if constexpr (DK == 48 && DV == 48) {
                if (tid < 240) {
                    if (gemv_v < 4) {
                        float acc = 0.0f;
                        #pragma unroll
                        for (int i = 0; i < 48; ++i) {
                            acc += sh_M[i][gemv_r] * sh_go[gemv_v][i];
                        }
                        dq[(mb * 48 + gemv_r) * 4 + gemv_v] = acc;
                    } else {
                        float gk_u = 0.0f;
                        float gk_p = 0.0f;
                        #pragma unroll
                        for (int i = 0; i < 48; ++i) {
                            gk_u += sh_gM[i][gemv_r] * sh_erreff[i];
                            gk_p += sh_M[i][gemv_r] * sh_gpred[i];
                        }
                        dk[mb * 48 + gemv_r] = (1.35f * bt) * gk_u + gk_p;
                    }
                }
                if constexpr (!IS_TERN) {
                    if (warp_id == 7) {
                        const float dot_gmk_err = warp_reduce_smem_vec<48>(sh_dot, 48, lane);
                        if (lane == 0) {
                            db[mb] = 1.35f * dot_gmk_err;
                        }
                    }
                }
            } else {
                for (int idx = tid; idx < d_k * 5; idx += 256) {
                    const int v = idx / d_k;
                    const int j = idx - v * d_k;
                    if (v < 4) {
                        float acc = 0.0f;
                        for (int i = 0; i < d_v; ++i) {
                            acc += sh_M[i][j] * sh_go[v][i];
                        }
                        dq[(mb * d_k + j) * 4 + v] = acc;
                    } else {
                        float gk_u = 0.0f;
                        float gk_p = 0.0f;
                        for (int i = 0; i < d_v; ++i) {
                            gk_u += sh_gM[i][j] * sh_erreff[i];
                            gk_p += sh_M[i][j] * sh_gpred[i];
                        }
                        dk[mb * d_k + j] = (1.35f * bt) * gk_u + gk_p;
                    }
                }
                if constexpr (!IS_TERN) {
                    if (warp_id == 0) {
                        const float dot_gmk_err = warp_reduce_smem_vec<0>(sh_dot, d_v, lane);
                        if (lane == 0) {
                            db[mb] = 1.35f * dot_gmk_err;
                        }
                    }
                }
            }
            __syncthreads();  // Barrier R3: all sh_gM reads finished

            // Step C: Fused <gM, M> accumulation + sh_gM update
            float gm_m_part = 0.0f;
            if constexpr (DK == 32 && DV == 32) {
                const float q0 = sh_q[0][lane];
                const float q1 = sh_q[1][lane];
                const float q2 = sh_q[2][lane];
                const float q3 = sh_q[3][lane];
                const float kj = sh_k[lane];
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const int i = (warp_id << 2) + r;
                    const float cur_gm = sh_gM[i][lane];
                    gm_m_part += cur_gm * sh_M[i][lane];
                    const float gq_term = sh_go[0][i] * q0
                                        + sh_go[1][i] * q1
                                        + sh_go[2][i] * q2
                                        + sh_go[3][i] * q3;
                    sh_gM[i][lane] = gt * cur_gm + sh_gpred[i] * kj + gq_term;
                }
            } else if constexpr (DK == 48 && DV == 48) {
                #pragma unroll
                for (int step = 0; step < 9; ++step) {
                    const int idx = tid + step * 256;
                    const int i = idx / 48;
                    const int j = idx - i * 48;
                    const float cur_gm = sh_gM[i][j];
                    gm_m_part += cur_gm * sh_M[i][j];
                    const float gq_term = sh_go[0][i] * sh_q[0][j]
                                        + sh_go[1][i] * sh_q[1][j]
                                        + sh_go[2][i] * sh_q[2][j]
                                        + sh_go[3][i] * sh_q[3][j];
                    sh_gM[i][j] = gt * cur_gm + sh_gpred[i] * sh_k[j] + gq_term;
                }
            } else {
                for (int idx = tid; idx < dv_dk; idx += 256) {
                    const int i = idx / d_k;
                    const int j = idx - i * d_k;
                    const float cur_gm = sh_gM[i][j];
                    gm_m_part += cur_gm * sh_M[i][j];
                    const float gq_term = sh_go[0][i] * sh_q[0][j]
                                        + sh_go[1][i] * sh_q[1][j]
                                        + sh_go[2][i] * sh_q[2][j]
                                        + sh_go[3][i] * sh_q[3][j];
                    sh_gM[i][j] = gt * cur_gm + sh_gpred[i] * sh_k[j] + gq_term;
                }
            }
            const float w_gm = warp_sum32(gm_m_part);
            if (lane == 0) sh_warp_red[warp_id] = w_gm;
            __syncthreads();  // Barrier R4: sh_gM updated & sh_warp_red ready

            if (warp_id == 0) {
                float v = (lane < 8) ? sh_warp_red[lane] : 0.0f;
                if constexpr (DV == 32) {
                    v -= sh_gerr[lane] * sh_pred[lane];
                } else if constexpr (DV == 48) {
                    v -= sh_gerr[lane] * sh_pred[lane];
                    if (lane < 16) v -= sh_gerr[lane + 32] * sh_pred[lane + 32];
                } else {
                    if (lane < d_v) v -= sh_gerr[lane] * sh_pred[lane];
                    if (lane + 32 < d_v) v -= sh_gerr[lane + 32] * sh_pred[lane + 32];
                }
                const float tot = warp_sum32(v);
                if (lane == 0) dg[mb] = tot;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Single-Pass Warp-Parallel 32-Group 2-Step LS Ternary Quantization (`fused_ternary_ste_cuda`)
// ---------------------------------------------------------------------------
template <typename InT, typename OutT>
__global__ void fused_ternary_ste_kernel(
    const InT* __restrict__ w,
    OutT* __restrict__ out,
    int64_t n_groups,
    float mix
) {
    const int64_t g = (static_cast<int64_t>(blockIdx.x) << 3) + (threadIdx.x >> 5);
    if (g >= n_groups) return;
    const int lane = static_cast<int>(threadIdx.x & 31);
    const int64_t idx = (g << 5) + lane;

    const float wf = static_cast<float>(w[idx]);
    const float aw = fabsf(wf);
    const float sgn = (wf > 0.0f) ? 1.0f : ((wf < 0.0f) ? -1.0f : 0.0f);

    float s0 = warp_sum32(aw) * 0.03125f;
    if (s0 < 1e-6f) s0 = 1e-6f;
    float t = (aw >= 0.5f * s0) ? sgn : 0.0f;

    #pragma unroll
    for (int it = 0; it < 2; ++it) {
        const float denom = warp_sum32(t * t);
        const float num = warp_sum32(wf * t);
        float s_ls = (denom > 0.0f) ? __fdiv_rn(num, denom > 1.0f ? denom : 1.0f) : s0;
        if (s_ls < 1e-6f) s_ls = 1e-6f;
        t = (aw >= 0.5f * s_ls) ? sgn : 0.0f;
    }

    const float denom = warp_sum32(t * t);
    const float num = warp_sum32(wf * t);
    float scale = (denom > 0.0f) ? __fdiv_rn(num, denom > 1.0f ? denom : 1.0f) : s0;
    if (scale < 1e-6f) scale = 1e-6f;

    float wt = t * scale;
    if constexpr (!std::is_same<InT, float>::value) {
        wt = static_cast<float>(static_cast<InT>(wt));
    }
    const float out_f = (mix < 0.9999f) ? ((1.0f - mix) * wf + mix * wt) : wt;
    out[idx] = static_cast<OutT>(out_f);
}

template <typename InT>
void launch_fused_ternary_ste_out(
    const torch::Tensor& wc,
    torch::Tensor& out,
    int64_t n_groups,
    float mix,
    cudaStream_t stream
) {
    const int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>((n_groups + 7) / 8);
    switch (out.scalar_type()) {
        case torch::kFloat32:
            fused_ternary_ste_kernel<InT, float><<<blocks, threads, 0, stream>>>(
                wc.data_ptr<InT>(), out.data_ptr<float>(), n_groups, mix
            );
            break;
        case torch::kBFloat16:
            fused_ternary_ste_kernel<InT, at::BFloat16><<<blocks, threads, 0, stream>>>(
                wc.data_ptr<InT>(), out.data_ptr<at::BFloat16>(), n_groups, mix
            );
            break;
        case torch::kHalf:
            fused_ternary_ste_kernel<InT, at::Half><<<blocks, threads, 0, stream>>>(
                wc.data_ptr<InT>(), out.data_ptr<at::Half>(), n_groups, mix
            );
            break;
        default:
            TORCH_CHECK(false, "fused_ternary_ste_cuda: unsupported output dtype");
    }
}

}  // namespace

torch::Tensor utf8_chunk_span_cuda_impl(
    torch::Tensor raw_cand,
    torch::Tensor valid_mask,
    torch::Tensor can_fire_mask,
    torch::Tensor is_lead,
    torch::Tensor char_len,
    int64_t max_cp_span,
    int64_t max_byte_span,
    int64_t refill_int,
    int64_t cost_int,
    int64_t burst_int
) {
    c10::cuda::CUDAGuard device_guard(raw_cand.device());
    auto rc = raw_cand.to(torch::kBool).contiguous();
    auto vm = valid_mask.to(torch::kBool).contiguous();
    auto cf = can_fire_mask.to(torch::kBool).contiguous();
    auto ld = is_lead.to(torch::kInt64).contiguous();
    auto cl = char_len.to(torch::kInt64).contiguous();
    const int64_t B = rc.size(0);
    const int64_t T = rc.size(1);
    auto out = torch::empty_like(rc);
    if (B == 0 || T == 0) return out;
    const int threads = 32;
    const int blocks = static_cast<int>((B + threads - 1) / threads);
    utf8_chunk_span_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        rc.data_ptr<bool>(), vm.data_ptr<bool>(), cf.data_ptr<bool>(),
        ld.data_ptr<int64_t>(), cl.data_ptr<int64_t>(), out.data_ptr<bool>(),
        B, T, max_cp_span, max_byte_span, refill_int, cost_int, burst_int
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<torch::Tensor> causal_refractory_gate_cuda_impl(torch::Tensor p_raw) {
    c10::cuda::CUDAGuard device_guard(p_raw.device());
    auto pr = p_raw.to(torch::kFloat32).contiguous();
    const int64_t B = pr.size(0);
    const int64_t M = pr.size(1);
    const int64_t H = pr.size(2);
    auto out = torch::empty_like(pr);
    auto mult = torch::empty_like(pr);
    const int64_t total = B * H;
    if (total == 0 || M == 0) return {out, mult};
    const int threads = 64;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    causal_refractory_gate_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        pr.data_ptr<float>(), out.data_ptr<float>(), mult.data_ptr<float>(), B, M, H
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, mult};
}

torch::Tensor birkhoff_greedy_perm_cuda_impl(torch::Tensor scores) {
    c10::cuda::CUDAGuard device_guard(scores.device());
    auto sc = scores.to(torch::kFloat32).contiguous();
    const int64_t N = sc.size(0);
    const int G = static_cast<int>(sc.size(1));
    TORCH_CHECK(G > 0 && G <= 32 && sc.size(2) == G, "birkhoff_greedy_perm_cuda requires 0 < G <= 32");
    auto out = torch::zeros_like(sc);
    if (N == 0) return out;
    const int threads = 128;
    const int blocks = static_cast<int>((N + threads - 1) / threads);
    birkhoff_greedy_perm_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        sc.data_ptr<float>(), out.data_ptr<float>(), N, G
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor crystal_scan_fwd_cuda_impl(
    torch::Tensor s0,
    torch::Tensor kw_eff,
    torch::Tensor v_ste,
    torch::Tensor P_eff,
    bool is_hard,
    double scale_r
) {
    c10::cuda::CUDAGuard device_guard(s0.device());
    auto s0c = s0.to(torch::kFloat32).contiguous();
    auto kwc = kw_eff.to(torch::kFloat32).contiguous();
    auto vsc = v_ste.to(torch::kFloat32).contiguous();
    auto pec = P_eff.to(torch::kFloat32).contiguous();
    const int64_t BH = s0c.size(0);
    const int G_h = static_cast<int>(s0c.size(1));
    const int d_s = static_cast<int>(s0c.size(2));
    const int64_t M = vsc.size(1);
    TORCH_CHECK(d_s == 32 && G_h > 0 && G_h <= 32, "crystal_scan_fwd_cuda requires d_s == 32 and G_h <= 32");
    auto s_out = torch::empty({BH, M, G_h, d_s}, s0c.options());
    if (BH == 0 || M == 0) return s_out;
    dim3 block(32, G_h);
    dim3 grid(static_cast<unsigned int>(BH));
    auto stream = c10::cuda::getCurrentCUDAStream();
    const float sc_f = static_cast<float>(scale_r);
    if (G_h == 12) {
        if (is_hard) {
            crystal_scan_fwd_kernel<12, true><<<grid, block, 0, stream>>>(
                s0c.data_ptr<float>(), kwc.data_ptr<float>(), vsc.data_ptr<float>(),
                pec.data_ptr<float>(), s_out.data_ptr<float>(), M, G_h, sc_f
            );
        } else {
            crystal_scan_fwd_kernel<12, false><<<grid, block, 0, stream>>>(
                s0c.data_ptr<float>(), kwc.data_ptr<float>(), vsc.data_ptr<float>(),
                pec.data_ptr<float>(), s_out.data_ptr<float>(), M, G_h, sc_f
            );
        }
    } else if (G_h == 16) {
        if (is_hard) {
            crystal_scan_fwd_kernel<16, true><<<grid, block, 0, stream>>>(
                s0c.data_ptr<float>(), kwc.data_ptr<float>(), vsc.data_ptr<float>(),
                pec.data_ptr<float>(), s_out.data_ptr<float>(), M, G_h, sc_f
            );
        } else {
            crystal_scan_fwd_kernel<16, false><<<grid, block, 0, stream>>>(
                s0c.data_ptr<float>(), kwc.data_ptr<float>(), vsc.data_ptr<float>(),
                pec.data_ptr<float>(), s_out.data_ptr<float>(), M, G_h, sc_f
            );
        }
    } else {
        if (is_hard) {
            crystal_scan_fwd_kernel<0, true><<<grid, block, 0, stream>>>(
                s0c.data_ptr<float>(), kwc.data_ptr<float>(), vsc.data_ptr<float>(),
                pec.data_ptr<float>(), s_out.data_ptr<float>(), M, G_h, sc_f
            );
        } else {
            crystal_scan_fwd_kernel<0, false><<<grid, block, 0, stream>>>(
                s0c.data_ptr<float>(), kwc.data_ptr<float>(), vsc.data_ptr<float>(),
                pec.data_ptr<float>(), s_out.data_ptr<float>(), M, G_h, sc_f
            );
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return s_out;
}

std::vector<torch::Tensor> crystal_scan_bwd_cuda_impl(
    torch::Tensor s_out,
    torch::Tensor kw_eff,
    torch::Tensor kw_vg,
    torch::Tensor v_ste,
    torch::Tensor P_eff,
    torch::Tensor grad_out,
    bool is_hard,
    double scale_r
) {
    c10::cuda::CUDAGuard device_guard(s_out.device());
    auto soc = s_out.to(torch::kFloat32).contiguous();
    auto kwc = kw_eff.to(torch::kFloat32).contiguous();
    auto kvc = kw_vg.to(torch::kFloat32).contiguous();
    auto vsc = v_ste.to(torch::kFloat32).contiguous();
    auto pec = P_eff.to(torch::kFloat32).contiguous();
    auto goc = grad_out.to(torch::kFloat32).contiguous();
    const int64_t BH = soc.size(0);
    const int64_t M = soc.size(1);
    const int G_h = static_cast<int>(soc.size(2));
    const int d_s = static_cast<int>(soc.size(3));
    TORCH_CHECK(d_s == 32 && G_h > 0 && G_h <= 32, "crystal_scan_bwd_cuda requires d_s == 32 and G_h <= 32");
    auto g_s0 = torch::empty({BH, G_h, d_s}, soc.options());
    auto g_kwe = torch::empty({BH, M, G_h, 1}, soc.options());
    auto g_vs = torch::empty({BH, M, d_s}, soc.options());
    auto g_P = torch::empty({BH, M, G_h, G_h}, soc.options());
    if (BH == 0 || M == 0) {
        g_s0.zero_();
        return {g_s0, g_kwe, g_vs, g_P};
    }
    dim3 block(32, G_h);
    dim3 grid(static_cast<unsigned int>(BH));
    auto stream = c10::cuda::getCurrentCUDAStream();
    const float sc_f = static_cast<float>(scale_r);
    if (G_h == 12) {
        if (is_hard) {
            crystal_scan_bwd_kernel<12, true><<<grid, block, 0, stream>>>(
                soc.data_ptr<float>(), kwc.data_ptr<float>(), kvc.data_ptr<float>(),
                vsc.data_ptr<float>(), pec.data_ptr<float>(), goc.data_ptr<float>(),
                g_s0.data_ptr<float>(), g_kwe.data_ptr<float>(), g_vs.data_ptr<float>(),
                g_P.data_ptr<float>(), M, G_h, sc_f
            );
        } else {
            crystal_scan_bwd_kernel<12, false><<<grid, block, 0, stream>>>(
                soc.data_ptr<float>(), kwc.data_ptr<float>(), kvc.data_ptr<float>(),
                vsc.data_ptr<float>(), pec.data_ptr<float>(), goc.data_ptr<float>(),
                g_s0.data_ptr<float>(), g_kwe.data_ptr<float>(), g_vs.data_ptr<float>(),
                g_P.data_ptr<float>(), M, G_h, sc_f
            );
        }
    } else if (G_h == 16) {
        if (is_hard) {
            crystal_scan_bwd_kernel<16, true><<<grid, block, 0, stream>>>(
                soc.data_ptr<float>(), kwc.data_ptr<float>(), kvc.data_ptr<float>(),
                vsc.data_ptr<float>(), pec.data_ptr<float>(), goc.data_ptr<float>(),
                g_s0.data_ptr<float>(), g_kwe.data_ptr<float>(), g_vs.data_ptr<float>(),
                g_P.data_ptr<float>(), M, G_h, sc_f
            );
        } else {
            crystal_scan_bwd_kernel<16, false><<<grid, block, 0, stream>>>(
                soc.data_ptr<float>(), kwc.data_ptr<float>(), kvc.data_ptr<float>(),
                vsc.data_ptr<float>(), pec.data_ptr<float>(), goc.data_ptr<float>(),
                g_s0.data_ptr<float>(), g_kwe.data_ptr<float>(), g_vs.data_ptr<float>(),
                g_P.data_ptr<float>(), M, G_h, sc_f
            );
        }
    } else {
        if (is_hard) {
            crystal_scan_bwd_kernel<0, true><<<grid, block, 0, stream>>>(
                soc.data_ptr<float>(), kwc.data_ptr<float>(), kvc.data_ptr<float>(),
                vsc.data_ptr<float>(), pec.data_ptr<float>(), goc.data_ptr<float>(),
                g_s0.data_ptr<float>(), g_kwe.data_ptr<float>(), g_vs.data_ptr<float>(),
                g_P.data_ptr<float>(), M, G_h, sc_f
            );
        } else {
            crystal_scan_bwd_kernel<0, false><<<grid, block, 0, stream>>>(
                soc.data_ptr<float>(), kwc.data_ptr<float>(), kvc.data_ptr<float>(),
                vsc.data_ptr<float>(), pec.data_ptr<float>(), goc.data_ptr<float>(),
                g_s0.data_ptr<float>(), g_kwe.data_ptr<float>(), g_vs.data_ptr<float>(),
                g_P.data_ptr<float>(), M, G_h, sc_f
            );
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {g_s0, g_kwe, g_vs, g_P};
}

torch::Tensor woodbury_tern_p_uncerts_cuda_impl(
    torch::Tensor qe_m,
    torch::Tensor ke_m,
    torch::Tensor bm_d,
    torch::Tensor gm_d,
    double inv_lam,
    double tern_gate
) {
    c10::cuda::CUDAGuard device_guard(qe_m.device());
    auto qe = qe_m.to(torch::kFloat32).contiguous();
    auto ke = ke_m.to(torch::kFloat32).contiguous();
    auto bm = bm_d.to(torch::kFloat32).contiguous();
    auto gm = gm_d.to(torch::kFloat32).contiguous();
    const int64_t M_len = qe.size(0);
    const int64_t BH = qe.size(1);
    const int d_k = static_cast<int>(qe.size(2));
    const int C = static_cast<int>(qe.size(3));
    TORCH_CHECK(d_k > 0 && d_k <= 64 && C == 4, "woodbury_tern_p_uncerts_cuda requires d_k <= 64 and C == 4");
    auto out = torch::empty({M_len, BH, C}, qe.options());
    if (M_len == 0 || BH == 0) return out;
    auto stream = c10::cuda::getCurrentCUDAStream();
    const unsigned int blocks = static_cast<unsigned int>(BH);
    const float inv_lam_f = static_cast<float>(inv_lam);
    const float tgate_f = static_cast<float>(tern_gate);
    if (d_k == 32) {
        woodbury_tern_p_uncerts_kernel<32><<<blocks, 256, 0, stream>>>(
            qe.data_ptr<float>(), ke.data_ptr<float>(), bm.data_ptr<float>(), gm.data_ptr<float>(),
            out.data_ptr<float>(), M_len, BH, d_k, inv_lam_f, tgate_f
        );
    } else if (d_k == 48) {
        woodbury_tern_p_uncerts_kernel<48><<<blocks, 256, 0, stream>>>(
            qe.data_ptr<float>(), ke.data_ptr<float>(), bm.data_ptr<float>(), gm.data_ptr<float>(),
            out.data_ptr<float>(), M_len, BH, d_k, inv_lam_f, tgate_f
        );
    } else {
        woodbury_tern_p_uncerts_kernel<0><<<blocks, 256, 0, stream>>>(
            qe.data_ptr<float>(), ke.data_ptr<float>(), bm.data_ptr<float>(), gm.data_ptr<float>(),
            out.data_ptr<float>(), M_len, BH, d_k, inv_lam_f, tgate_f
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<torch::Tensor> woodbury_tern_m_fwd_cuda_impl(
    torch::Tensor q_m,
    torch::Tensor k_m,
    torch::Tensor v_m,
    torch::Tensor b_m,
    torch::Tensor g_m,
    double inv_dh,
    int64_t Bc
) {
    c10::cuda::CUDAGuard device_guard(q_m.device());
    auto qc = q_m.to(torch::kFloat32).contiguous();
    auto kc = k_m.to(torch::kFloat32).contiguous();
    auto vc = v_m.to(torch::kFloat32).contiguous();
    auto bc = b_m.to(torch::kFloat32).contiguous();
    auto gc = g_m.to(torch::kFloat32).contiguous();
    const int64_t M_len = qc.size(0);
    const int64_t BH = qc.size(1);
    const int d_k = static_cast<int>(qc.size(2));
    const int C = static_cast<int>(qc.size(3));
    const int d_v = static_cast<int>(vc.size(2));
    TORCH_CHECK(d_k > 0 && d_k <= 64 && d_v > 0 && d_v <= 64 && C == 4, "woodbury_tern_m_fwd_cuda requires d <= 64 and C == 4");
    const int64_t n_blocks = (M_len + Bc - 1) / Bc;
    auto out_m = torch::empty({M_len, BH, d_v, C}, qc.options());
    auto M_starts = torch::empty({n_blocks, BH, d_v, d_k}, qc.options());
    if (M_len == 0 || BH == 0) return {out_m, M_starts};
    auto stream = c10::cuda::getCurrentCUDAStream();
    const unsigned int blocks = static_cast<unsigned int>(BH);
    const float inv_dh_f = static_cast<float>(inv_dh);
    const bool is_tern = (inv_dh_f > 0.0f);
    if (d_k == 32 && d_v == 32) {
        if (is_tern) {
            woodbury_tern_m_fwd_kernel<32, 32, true><<<blocks, 256, 0, stream>>>(
                qc.data_ptr<float>(), kc.data_ptr<float>(), vc.data_ptr<float>(),
                bc.data_ptr<float>(), gc.data_ptr<float>(), out_m.data_ptr<float>(),
                M_starts.data_ptr<float>(), M_len, BH, d_k, d_v, inv_dh_f, Bc
            );
        } else {
            woodbury_tern_m_fwd_kernel<32, 32, false><<<blocks, 256, 0, stream>>>(
                qc.data_ptr<float>(), kc.data_ptr<float>(), vc.data_ptr<float>(),
                bc.data_ptr<float>(), gc.data_ptr<float>(), out_m.data_ptr<float>(),
                M_starts.data_ptr<float>(), M_len, BH, d_k, d_v, inv_dh_f, Bc
            );
        }
    } else if (d_k == 48 && d_v == 48) {
        if (is_tern) {
            woodbury_tern_m_fwd_kernel<48, 48, true><<<blocks, 256, 0, stream>>>(
                qc.data_ptr<float>(), kc.data_ptr<float>(), vc.data_ptr<float>(),
                bc.data_ptr<float>(), gc.data_ptr<float>(), out_m.data_ptr<float>(),
                M_starts.data_ptr<float>(), M_len, BH, d_k, d_v, inv_dh_f, Bc
            );
        } else {
            woodbury_tern_m_fwd_kernel<48, 48, false><<<blocks, 256, 0, stream>>>(
                qc.data_ptr<float>(), kc.data_ptr<float>(), vc.data_ptr<float>(),
                bc.data_ptr<float>(), gc.data_ptr<float>(), out_m.data_ptr<float>(),
                M_starts.data_ptr<float>(), M_len, BH, d_k, d_v, inv_dh_f, Bc
            );
        }
    } else {
        if (is_tern) {
            woodbury_tern_m_fwd_kernel<0, 0, true><<<blocks, 256, 0, stream>>>(
                qc.data_ptr<float>(), kc.data_ptr<float>(), vc.data_ptr<float>(),
                bc.data_ptr<float>(), gc.data_ptr<float>(), out_m.data_ptr<float>(),
                M_starts.data_ptr<float>(), M_len, BH, d_k, d_v, inv_dh_f, Bc
            );
        } else {
            woodbury_tern_m_fwd_kernel<0, 0, false><<<blocks, 256, 0, stream>>>(
                qc.data_ptr<float>(), kc.data_ptr<float>(), vc.data_ptr<float>(),
                bc.data_ptr<float>(), gc.data_ptr<float>(), out_m.data_ptr<float>(),
                M_starts.data_ptr<float>(), M_len, BH, d_k, d_v, inv_dh_f, Bc
            );
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out_m, M_starts};
}

std::vector<torch::Tensor> woodbury_tern_m_bwd_cuda_impl(
    torch::Tensor q_m,
    torch::Tensor k_m,
    torch::Tensor v_m,
    torch::Tensor b_m,
    torch::Tensor g_m,
    torch::Tensor M_starts,
    torch::Tensor grad_out,
    double inv_dh,
    int64_t Bc
) {
    c10::cuda::CUDAGuard device_guard(q_m.device());
    auto qc = q_m.to(torch::kFloat32).contiguous();
    auto kc = k_m.to(torch::kFloat32).contiguous();
    auto vc = v_m.to(torch::kFloat32).contiguous();
    auto bc = b_m.to(torch::kFloat32).contiguous();
    auto gc = g_m.to(torch::kFloat32).contiguous();
    auto msc = M_starts.to(torch::kFloat32).contiguous();
    auto goc = grad_out.to(torch::kFloat32).contiguous();
    const int64_t M_len = qc.size(0);
    const int64_t BH = qc.size(1);
    const int d_k = static_cast<int>(qc.size(2));
    const int C = static_cast<int>(qc.size(3));
    const int d_v = static_cast<int>(vc.size(2));
    const int64_t n_blocks = msc.size(0);
    TORCH_CHECK(d_k > 0 && d_k <= 64 && d_v > 0 && d_v <= 64 && C == 4, "woodbury_tern_m_bwd_cuda requires d <= 64 and C == 4");
    auto dq = torch::empty_like(qc);
    auto dk = torch::empty_like(kc);
    auto dv = torch::empty_like(vc);
    auto db = torch::empty_like(bc);
    auto dg = torch::empty_like(gc);
    if (M_len == 0 || BH == 0) return {dq, dk, dv, db, dg};
    auto M_blk_buf = torch::empty({BH, Bc, d_v, d_k}, qc.options());
    auto stream = c10::cuda::getCurrentCUDAStream();
    const unsigned int blocks = static_cast<unsigned int>(BH);
    const float inv_dh_f = static_cast<float>(inv_dh);
    const bool is_tern = (inv_dh_f > 0.0f);
    if (d_k == 32 && d_v == 32) {
        if (is_tern) {
            woodbury_tern_m_bwd_kernel<32, 32, true><<<blocks, 256, 0, stream>>>(
                qc.data_ptr<float>(), kc.data_ptr<float>(), vc.data_ptr<float>(),
                bc.data_ptr<float>(), gc.data_ptr<float>(), msc.data_ptr<float>(),
                goc.data_ptr<float>(), M_blk_buf.data_ptr<float>(),
                dq.data_ptr<float>(), dk.data_ptr<float>(), dv.data_ptr<float>(),
                db.data_ptr<float>(), dg.data_ptr<float>(),
                M_len, BH, d_k, d_v, inv_dh_f, Bc, n_blocks
            );
        } else {
            woodbury_tern_m_bwd_kernel<32, 32, false><<<blocks, 256, 0, stream>>>(
                qc.data_ptr<float>(), kc.data_ptr<float>(), vc.data_ptr<float>(),
                bc.data_ptr<float>(), gc.data_ptr<float>(), msc.data_ptr<float>(),
                goc.data_ptr<float>(), M_blk_buf.data_ptr<float>(),
                dq.data_ptr<float>(), dk.data_ptr<float>(), dv.data_ptr<float>(),
                db.data_ptr<float>(), dg.data_ptr<float>(),
                M_len, BH, d_k, d_v, inv_dh_f, Bc, n_blocks
            );
        }
    } else if (d_k == 48 && d_v == 48) {
        if (is_tern) {
            woodbury_tern_m_bwd_kernel<48, 48, true><<<blocks, 256, 0, stream>>>(
                qc.data_ptr<float>(), kc.data_ptr<float>(), vc.data_ptr<float>(),
                bc.data_ptr<float>(), gc.data_ptr<float>(), msc.data_ptr<float>(),
                goc.data_ptr<float>(), M_blk_buf.data_ptr<float>(),
                dq.data_ptr<float>(), dk.data_ptr<float>(), dv.data_ptr<float>(),
                db.data_ptr<float>(), dg.data_ptr<float>(),
                M_len, BH, d_k, d_v, inv_dh_f, Bc, n_blocks
            );
        } else {
            woodbury_tern_m_bwd_kernel<48, 48, false><<<blocks, 256, 0, stream>>>(
                qc.data_ptr<float>(), kc.data_ptr<float>(), vc.data_ptr<float>(),
                bc.data_ptr<float>(), gc.data_ptr<float>(), msc.data_ptr<float>(),
                goc.data_ptr<float>(), M_blk_buf.data_ptr<float>(),
                dq.data_ptr<float>(), dk.data_ptr<float>(), dv.data_ptr<float>(),
                db.data_ptr<float>(), dg.data_ptr<float>(),
                M_len, BH, d_k, d_v, inv_dh_f, Bc, n_blocks
            );
        }
    } else {
        if (is_tern) {
            woodbury_tern_m_bwd_kernel<0, 0, true><<<blocks, 256, 0, stream>>>(
                qc.data_ptr<float>(), kc.data_ptr<float>(), vc.data_ptr<float>(),
                bc.data_ptr<float>(), gc.data_ptr<float>(), msc.data_ptr<float>(),
                goc.data_ptr<float>(), M_blk_buf.data_ptr<float>(),
                dq.data_ptr<float>(), dk.data_ptr<float>(), dv.data_ptr<float>(),
                db.data_ptr<float>(), dg.data_ptr<float>(),
                M_len, BH, d_k, d_v, inv_dh_f, Bc, n_blocks
            );
        } else {
            woodbury_tern_m_bwd_kernel<0, 0, false><<<blocks, 256, 0, stream>>>(
                qc.data_ptr<float>(), kc.data_ptr<float>(), vc.data_ptr<float>(),
                bc.data_ptr<float>(), gc.data_ptr<float>(), msc.data_ptr<float>(),
                goc.data_ptr<float>(), M_blk_buf.data_ptr<float>(),
                dq.data_ptr<float>(), dk.data_ptr<float>(), dv.data_ptr<float>(),
                db.data_ptr<float>(), dg.data_ptr<float>(),
                M_len, BH, d_k, d_v, inv_dh_f, Bc, n_blocks
            );
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dq, dk, dv, db, dg};
}

torch::Tensor fused_ternary_ste_cuda_impl(
    torch::Tensor w,
    double mix,
    int64_t out_dtype_code
) {
    c10::cuda::CUDAGuard device_guard(w.device());
    auto wc = w.contiguous();
    TORCH_CHECK(wc.numel() % 32 == 0, "fused_ternary_ste_cuda requires w.numel() % 32 == 0");
    torch::ScalarType out_st = wc.scalar_type();
    if (out_dtype_code == 0) out_st = torch::kFloat32;
    else if (out_dtype_code == 1) out_st = torch::kBFloat16;
    else if (out_dtype_code == 2) out_st = torch::kHalf;

    auto out = torch::empty(wc.sizes(), wc.options().dtype(out_st));
    const int64_t n_groups = wc.numel() / 32;
    if (n_groups == 0) return out;
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    const float mix_f = static_cast<float>(mix);
    switch (wc.scalar_type()) {
        case torch::kFloat32:
            launch_fused_ternary_ste_out<float>(wc, out, n_groups, mix_f, stream);
            break;
        case torch::kBFloat16:
            launch_fused_ternary_ste_out<at::BFloat16>(wc, out, n_groups, mix_f, stream);
            break;
        case torch::kHalf:
            launch_fused_ternary_ste_out<at::Half>(wc, out, n_groups, mix_f, stream);
            break;
        default:
            TORCH_CHECK(false, "fused_ternary_ste_cuda: unsupported weight dtype");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

_CUDA_BUILD_ERROR: str | None = None


def _has_python_headers() -> bool:
    inc = sysconfig.get_path("include")
    return bool(inc) and (Path(inc) / "Python.h").is_file()


def _prepend_env_path(var_name: str, dirs: list[str]) -> None:
    if not dirs:
        return
    sep = os.pathsep
    cur = os.environ.get(var_name, "")
    existing = [p for p in cur.split(sep) if p]
    existing_norm = {os.path.normcase(os.path.normpath(p)) for p in existing}
    to_add: list[str] = []
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        nd = os.path.normcase(os.path.normpath(d))
        if nd not in existing_norm:
            to_add.append(d)
            existing_norm.add(nd)
    if to_add:
        os.environ[var_name] = sep.join(to_add + existing)


def _find_vs2022_vcvars64() -> Path | None:
    if sys.platform != "win32":
        return None
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(pf86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere.is_file():
        try:
            proc = subprocess.run(
                [
                    str(vswhere),
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            inst = proc.stdout.strip().splitlines()
            if inst:
                cand = Path(inst[0].strip()) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                if cand.is_file():
                    return cand
        except Exception:
            pass

    for root in (
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Microsoft Visual Studio",
        Path(pf86) / "Microsoft Visual Studio",
    ):
        for year in ("2022", "2019"):
            for edition in ("Community", "Professional", "Enterprise", "BuildTools"):
                cand = root / year / edition / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                if cand.is_file():
                    return cand
    return None


@lru_cache(maxsize=1)
def _activate_windows_msvc_env() -> str | None:
    if sys.platform != "win32":
        return shutil.which("g++") or shutil.which("clang++")
    cl_now = shutil.which("cl")
    if cl_now and os.environ.get("INCLUDE") and os.environ.get("LIB"):
        return cl_now
    vcvars = _find_vs2022_vcvars64()
    if vcvars is None:
        return cl_now
    try:
        proc = subprocess.run(
            f'call "{vcvars}" >nul && set',
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout:
            for line in proc.stdout.splitlines():
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if not k:
                    continue
                ku = k.upper()
                if ku == "PATH":
                    _prepend_env_path("PATH", [p for p in v.split(os.pathsep) if p])
                elif ku in ("INCLUDE", "LIB", "LIBPATH"):
                    _prepend_env_path(k, [p for p in v.split(os.pathsep) if p])
                else:
                    os.environ[k] = v
            os.environ["DISTUTILS_USE_SDK"] = "1"
            os.environ["MSSdk"] = "1"
    except Exception:
        pass
    return shutil.which("cl")


def _discover_nvidia_pip_roots() -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()
    cand_dirs: list[Path] = []
    try:
        spec = importlib.util.find_spec("nvidia")
        if spec is not None and spec.submodule_search_locations:
            for loc in spec.submodule_search_locations:
                cand_dirs.append(Path(loc))
    except Exception:
        pass
    for key in ("platlib", "purelib"):
        p = sysconfig.get_path(key)
        if p:
            cand_dirs.append(Path(p) / "nvidia")
    for d in cand_dirs:
        if d.is_dir():
            nd = os.path.normcase(str(d.resolve()))
            if nd not in seen:
                seen.add(nd)
                roots.append(d.resolve())
    return roots


@lru_cache(maxsize=1)
def prepare_cuda_build_env() -> dict[str, object]:
    """
    Auto-discovers MSVC (`vcvars64.bat`/`cl.exe`) on Windows and `nvcc` + CUDA headers/libs
    (supporting standard CUDA Toolkits and pip `nvidia-cuda-*-cu13` wheels). Updates
    `os.environ` and `torch.utils.cpp_extension.CUDA_HOME` idempotently and returns build flags.
    """
    is_win = sys.platform == "win32"
    nvcc_exe = "nvcc.exe" if is_win else "nvcc"
    cxx_path = _activate_windows_msvc_env() if is_win else (shutil.which("g++") or shutil.which("clang++"))

    include_paths: list[str] = []
    lib_paths: list[str] = []
    bin_paths: list[str] = []
    nvcc_path: str | None = shutil.which("nvcc")
    cuda_home: str | None = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")

    # 1. Inspect standard CUDA Toolkit installs (prefer matching torch.version.cuda major.minor)
    torch_cu = str(getattr(torch.version, "cuda", "") or "")
    toolkit_cands: list[Path] = []
    if cuda_home:
        toolkit_cands.append(Path(cuda_home))
    if is_win:
        tk_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
        if tk_root.is_dir():
            found = sorted(tk_root.glob("v*"), reverse=True)
            if torch_cu:
                found.sort(key=lambda p: (not p.name.startswith(f"v{torch_cu}"), p.name), reverse=False)
            toolkit_cands.extend(found)
    else:
        for p_str in ("/usr/local/cuda", "/usr/local/cuda-13.0", "/usr/local/cuda-12.8"):
            p = Path(p_str)
            if p.is_dir():
                toolkit_cands.append(p)

    for tk in toolkit_cands:
        cand_nvcc = tk / "bin" / nvcc_exe
        if cand_nvcc.is_file():
            if nvcc_path is None:
                nvcc_path = str(cand_nvcc)
            if not cuda_home or not (Path(cuda_home) / "bin" / nvcc_exe).is_file():
                cuda_home = str(tk)
            bin_paths.append(str(tk / "bin"))
            if (tk / "include").is_dir():
                include_paths.append(str(tk / "include"))
            for sub in ("lib/x64", "lib64", "lib"):
                lp = tk / sub
                if lp.is_dir():
                    lib_paths.append(str(lp))
            break

    # 2. Inspect pip `nvidia/cuda_*` wheel directories (`nvidia/cu13`, `nvidia-cuda-nvcc-cu13`, `cuda_runtime`, `cuda_cccl`, etc.)
    for nv_root in _discover_nvidia_pip_roots():
        for pkg_name in (
            "cu13",
            "cuda_nvcc",
            "cuda_runtime",
            "cuda_cccl",
            "cublas",
            "cusparse",
            "cusolver",
            "curand",
        ):
            pdir = nv_root / pkg_name
            if not pdir.is_dir():
                continue
            for extra_nvcc in (pdir / "bin" / nvcc_exe, pdir / "bin" / "x86_64" / nvcc_exe):
                if extra_nvcc.is_file():
                    if nvcc_path is None:
                        nvcc_path = str(extra_nvcc)
                    if not cuda_home or not (Path(cuda_home) / "bin" / nvcc_exe).is_file():
                        cuda_home = str(pdir)
                    bin_paths.append(str(extra_nvcc.parent))
            for inc_sub in ("include", "include/cccl"):
                inc_d = pdir / inc_sub
                if inc_d.is_dir():
                    include_paths.append(str(inc_d))
            for bin_sub in ("bin", "bin/x64", "nvvm/bin"):
                bd = pdir / bin_sub
                if bd.is_dir():
                    bin_paths.append(str(bd))
            for lib_sub in ("lib/x64", "lib64", "lib"):
                ld = pdir / lib_sub
                if ld.is_dir():
                    lib_paths.append(str(ld))

    # Deduplicate preserving order
    def _dedup(items: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for it in items:
            if not it or not os.path.isdir(it):
                continue
            norm = os.path.normcase(os.path.normpath(it))
            if norm not in seen:
                seen.add(norm)
                out.append(it)
        return out

    bin_paths = _dedup(bin_paths)
    include_paths = _dedup(include_paths)
    lib_paths = _dedup(lib_paths)

    _prepend_env_path("PATH", bin_paths)
    if is_win:
        _prepend_env_path("INCLUDE", include_paths)
        _prepend_env_path("LIB", lib_paths)
    if nvcc_path is None:
        nvcc_path = shutil.which("nvcc")
    if cuda_home:
        cur_ch = os.environ.get("CUDA_HOME")
        if not cur_ch or not (Path(cur_ch) / "bin" / nvcc_exe).is_file():
            os.environ["CUDA_HOME"] = cuda_home
        cur_cp = os.environ.get("CUDA_PATH")
        if not cur_cp or not (Path(cur_cp) / "bin" / nvcc_exe).is_file():
            os.environ["CUDA_PATH"] = cuda_home
        try:
            import torch.utils.cpp_extension as _ce

            if not _ce.CUDA_HOME or not (Path(_ce.CUDA_HOME) / "bin" / nvcc_exe).is_file():
                _ce.CUDA_HOME = cuda_home
        except Exception:
            pass

    if is_win:
        extra_cflags = ["/O2", "/EHsc", "/Zc:preprocessor", "/wd4267", "/wd4244", "/wd4068"]
        extra_cuda_cflags = [
            "-O3",
            "--use_fast_math",
            "-allow-unsupported-compiler",
            "-Xcompiler",
            "/O2",
            "-Xcompiler",
            "/EHsc",
            "-Xcompiler",
            "/Zc:preprocessor",
        ]
        extra_ldflags = [f"/LIBPATH:{lp}" for lp in lib_paths]
    else:
        extra_cflags = ["-O3"]
        extra_cuda_cflags = ["-O3", "--use_fast_math"]
        extra_ldflags = [f"-L{lp}" for lp in lib_paths]

    return {
        "nvcc_path": nvcc_path,
        "cuda_home": cuda_home,
        "cxx_path": cxx_path,
        "include_paths": include_paths,
        "lib_paths": lib_paths,
        "extra_cflags": extra_cflags,
        "extra_cuda_cflags": extra_cuda_cflags,
        "extra_ldflags": extra_ldflags,
    }


_SCANS_EXT_UNSET = object()
_SCANS_EXT_CACHE: object = _SCANS_EXT_UNSET


def _load_fast_scans_cuda_ext():
    global _CUDA_BUILD_ERROR, _SCANS_EXT_CACHE
    if _SCANS_EXT_CACHE is not _SCANS_EXT_UNSET:
        return _SCANS_EXT_CACHE
    if not torch.cuda.is_available():
        _CUDA_BUILD_ERROR = "CUDA device unavailable"
        _SCANS_EXT_CACHE = None
        return None
    env_info = prepare_cuda_build_env()
    if env_info["nvcc_path"] is None:
        _CUDA_BUILD_ERROR = "nvcc not found in PATH or CUDA/pip directories"
        _SCANS_EXT_CACHE = None
        return None
    if sys.platform == "win32" and env_info["cxx_path"] is None:
        _CUDA_BUILD_ERROR = "MSVC cl.exe not found (Visual Studio Build Tools missing)"
        _SCANS_EXT_CACHE = None
        return None
    if not _has_python_headers():
        _CUDA_BUILD_ERROR = "Python.h dev headers missing"
        _SCANS_EXT_CACHE = None
        return None
    try:
        from torch.utils.cpp_extension import load_inline

        build_dir = Path(tempfile.gettempdir()) / "calcite5_fast_scans_cuda_v3_1"
        build_dir.mkdir(parents=True, exist_ok=True)
        mod = load_inline(
            name="calcite5_fast_scans_cuda_ext_v3_1",
            cpp_sources=[_CUDA_CPP_SOURCE],
            cuda_sources=[_CUDA_KERNEL_SOURCE],
            functions=[
                "utf8_chunk_span_cuda",
                "causal_refractory_gate_cuda",
                "birkhoff_greedy_perm_cuda",
                "crystal_scan_fwd_cuda",
                "crystal_scan_bwd_cuda",
                "woodbury_tern_p_uncerts_cuda",
                "woodbury_tern_m_fwd_cuda",
                "woodbury_tern_m_bwd_cuda",
                "fused_ternary_ste_cuda",
            ],
            extra_cflags=list(env_info["extra_cflags"]),
            extra_cuda_cflags=list(env_info["extra_cuda_cflags"]),
            extra_ldflags=list(env_info["extra_ldflags"]) or None,
            extra_include_paths=list(env_info["include_paths"]) or None,
            build_directory=str(build_dir),
            with_cuda=True,
            verbose=False,
        )
        _CUDA_BUILD_ERROR = None
        _SCANS_EXT_CACHE = mod
        return mod
    except Exception as exc:
        _CUDA_BUILD_ERROR = f"{type(exc).__name__}: {exc}"
        _SCANS_EXT_CACHE = None
        return None


def get_fast_scans_backend_info() -> dict[str, object]:
    env_info = prepare_cuda_build_env()
    ext = _load_fast_scans_cuda_ext()
    return {
        "cuda_available": bool(torch.cuda.is_available()),
        "nvcc_found": env_info["nvcc_path"] is not None,
        "nvcc_path": env_info["nvcc_path"],
        "cxx_path": env_info["cxx_path"],
        "cuda_ext_available": ext is not None,
        "cuda_build_error": _CUDA_BUILD_ERROR,
        "version": "v3",
    }


# ---------------------------------------------------------------------------
# Autograd Functions & Python Dispatch Wrappers
# ---------------------------------------------------------------------------


_HAS_CUSTOM_OP = hasattr(torch, "library") and hasattr(torch.library, "custom_op")

if _HAS_CUSTOM_OP:

    @torch.library.custom_op("calcite5::utf8_chunk_span", mutates_args=())
    def _utf8_chunk_span_op(
        raw_cand: torch.Tensor,
        valid_mask: torch.Tensor,
        can_fire_mask: torch.Tensor,
        is_lead: torch.Tensor,
        char_len: torch.Tensor,
        max_cp_span: int,
        max_byte_span: int,
        refill_int: int,
        cost_int: int,
        burst_int: int,
    ) -> torch.Tensor:
        ext = _load_fast_scans_cuda_ext()
        assert ext is not None
        return ext.utf8_chunk_span_cuda(
            raw_cand,
            valid_mask,
            can_fire_mask,
            is_lead,
            char_len,
            int(max_cp_span),
            int(max_byte_span),
            int(refill_int),
            int(cost_int),
            int(burst_int),
        )

    @_utf8_chunk_span_op.register_fake
    def _utf8_chunk_span_fake(
        raw_cand: torch.Tensor,
        valid_mask: torch.Tensor,
        can_fire_mask: torch.Tensor,
        is_lead: torch.Tensor,
        char_len: torch.Tensor,
        max_cp_span: int,
        max_byte_span: int,
        refill_int: int,
        cost_int: int,
        burst_int: int,
    ) -> torch.Tensor:
        del valid_mask, can_fire_mask, is_lead, char_len
        del max_cp_span, max_byte_span, refill_int, cost_int, burst_int
        return torch.empty_like(raw_cand, dtype=torch.bool)

    @torch.library.custom_op("calcite5::causal_refractory_gate_fwd", mutates_args=())
    def _causal_refractory_gate_fwd_op(
        p_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ext = _load_fast_scans_cuda_ext()
        assert ext is not None
        out, mult = ext.causal_refractory_gate_cuda(p_raw)
        return out.to(p_raw.dtype), mult

    @_causal_refractory_gate_fwd_op.register_fake
    def _causal_refractory_gate_fwd_fake(
        p_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.empty_like(p_raw), torch.empty_like(p_raw, dtype=torch.float32)

    def _refractory_setup_context(ctx, inputs, output) -> None:
        del inputs
        _, mult = output
        ctx.save_for_backward(mult)

    def _refractory_bwd(
        ctx, grad_out: torch.Tensor | None, grad_mult: torch.Tensor | None
    ):
        del grad_mult
        if grad_out is None:
            return None
        (mult,) = ctx.saved_tensors
        return (grad_out.float() * mult).to(grad_out.dtype)

    _causal_refractory_gate_fwd_op.register_autograd(
        _refractory_bwd, setup_context=_refractory_setup_context
    )

    @torch.library.custom_op("calcite5::birkhoff_greedy_perm", mutates_args=())
    def _birkhoff_greedy_perm_op(scores: torch.Tensor) -> torch.Tensor:
        ext = _load_fast_scans_cuda_ext()
        assert ext is not None
        return ext.birkhoff_greedy_perm_cuda(scores).to(scores.dtype)

    @_birkhoff_greedy_perm_op.register_fake
    def _birkhoff_greedy_perm_fake(scores: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(scores)

    @torch.library.custom_op("calcite5::crystal_scan_fwd", mutates_args=())
    def _crystal_scan_fwd_op(
        s0: torch.Tensor,
        kw_eff_seq: torch.Tensor,
        kw_vg_seq: torch.Tensor,
        v_ste_seq: torch.Tensor,
        P_eff_seq: torch.Tensor,
        is_hard: bool,
        scale_r: float,
    ) -> torch.Tensor:
        del kw_vg_seq
        ext = _load_fast_scans_cuda_ext()
        assert ext is not None
        return ext.crystal_scan_fwd_cuda(
            s0, kw_eff_seq, v_ste_seq, P_eff_seq, bool(is_hard), float(scale_r)
        )

    @_crystal_scan_fwd_op.register_fake
    def _crystal_scan_fwd_fake(
        s0: torch.Tensor,
        kw_eff_seq: torch.Tensor,
        kw_vg_seq: torch.Tensor,
        v_ste_seq: torch.Tensor,
        P_eff_seq: torch.Tensor,
        is_hard: bool,
        scale_r: float,
    ) -> torch.Tensor:
        del kw_eff_seq, kw_vg_seq, P_eff_seq, is_hard, scale_r
        BH, G_h, d_s = s0.shape
        M = v_ste_seq.size(1)
        return torch.empty((BH, M, G_h, d_s), device=s0.device, dtype=torch.float32)

    @torch.library.custom_op("calcite5::crystal_scan_bwd", mutates_args=())
    def _crystal_scan_bwd_op(
        s_out: torch.Tensor,
        kw_eff_seq: torch.Tensor,
        kw_vg_seq: torch.Tensor,
        v_ste_seq: torch.Tensor,
        P_eff_seq: torch.Tensor,
        grad_out: torch.Tensor,
        is_hard: bool,
        scale_r: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ext = _load_fast_scans_cuda_ext()
        assert ext is not None
        g_s0, g_kwe, g_vs, g_P = ext.crystal_scan_bwd_cuda(
            s_out,
            kw_eff_seq,
            kw_vg_seq,
            v_ste_seq,
            P_eff_seq,
            grad_out,
            bool(is_hard),
            float(scale_r),
        )
        return g_s0, g_kwe, g_vs, g_P

    @_crystal_scan_bwd_op.register_fake
    def _crystal_scan_bwd_fake(
        s_out: torch.Tensor,
        kw_eff_seq: torch.Tensor,
        kw_vg_seq: torch.Tensor,
        v_ste_seq: torch.Tensor,
        P_eff_seq: torch.Tensor,
        grad_out: torch.Tensor,
        is_hard: bool,
        scale_r: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del kw_vg_seq, grad_out, is_hard, scale_r
        BH, M, G_h, d_s = s_out.shape
        opts = {"device": s_out.device, "dtype": torch.float32}
        return (
            torch.empty((BH, G_h, d_s), **opts),
            torch.empty_like(kw_eff_seq, dtype=torch.float32),
            torch.empty_like(v_ste_seq, dtype=torch.float32),
            torch.empty_like(P_eff_seq, dtype=torch.float32),
        )

    def _crystal_setup_context(ctx, inputs, output) -> None:
        _, kw_eff_seq, kw_vg_seq, v_ste_seq, P_eff_seq, is_hard, scale_r = inputs
        ctx.save_for_backward(output, kw_eff_seq, kw_vg_seq, v_ste_seq, P_eff_seq)
        ctx.is_hard = bool(is_hard)
        ctx.scale_r = float(scale_r)

    def _crystal_bwd_bridge(ctx, grad_out: torch.Tensor | None):
        if grad_out is None:
            return None, None, None, None, None, None, None
        s_out, kw_eff_seq, kw_vg_seq, v_ste_seq, P_eff_seq = ctx.saved_tensors
        g_s0, g_kwe, g_vs, g_P = _crystal_scan_bwd_op(
            s_out,
            kw_eff_seq,
            kw_vg_seq,
            v_ste_seq,
            P_eff_seq,
            grad_out.contiguous(),
            ctx.is_hard,
            ctx.scale_r,
        )
        return (
            g_s0.to(s_out.dtype),
            g_kwe.to(kw_eff_seq.dtype),
            torch.zeros_like(kw_vg_seq),
            g_vs.to(v_ste_seq.dtype),
            g_P.to(P_eff_seq.dtype),
            None,
            None,
        )

    _crystal_scan_fwd_op.register_autograd(
        _crystal_bwd_bridge, setup_context=_crystal_setup_context
    )

    @torch.library.custom_op("calcite5::woodbury_tern_p_uncerts", mutates_args=())
    def _woodbury_tern_p_uncerts_op(
        qe_m: torch.Tensor,
        ke_m: torch.Tensor,
        bm_d: torch.Tensor,
        gm_d: torch.Tensor,
        inv_lam: float,
        tern_gate: float,
    ) -> torch.Tensor:
        ext = _load_fast_scans_cuda_ext()
        assert ext is not None
        return ext.woodbury_tern_p_uncerts_cuda(
            qe_m, ke_m, bm_d, gm_d, float(inv_lam), float(tern_gate)
        )

    @_woodbury_tern_p_uncerts_op.register_fake
    def _woodbury_tern_p_uncerts_fake(
        qe_m: torch.Tensor,
        ke_m: torch.Tensor,
        bm_d: torch.Tensor,
        gm_d: torch.Tensor,
        inv_lam: float,
        tern_gate: float,
    ) -> torch.Tensor:
        del ke_m, bm_d, gm_d, inv_lam, tern_gate
        M_len, BH, _, C = qe_m.shape
        return torch.empty((M_len, BH, C), device=qe_m.device, dtype=torch.float32)

    @torch.library.custom_op("calcite5::woodbury_tern_m_fwd", mutates_args=())
    def _woodbury_tern_m_fwd_op(
        q_m: torch.Tensor,
        k_m: torch.Tensor,
        v_m: torch.Tensor,
        b_m: torch.Tensor,
        g_m: torch.Tensor,
        inv_dh: float,
        Bc: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ext = _load_fast_scans_cuda_ext()
        assert ext is not None
        out_m, M_starts = ext.woodbury_tern_m_fwd_cuda(
            q_m, k_m, v_m, b_m, g_m, float(inv_dh), int(Bc)
        )
        return out_m, M_starts

    @_woodbury_tern_m_fwd_op.register_fake
    def _woodbury_tern_m_fwd_fake(
        q_m: torch.Tensor,
        k_m: torch.Tensor,
        v_m: torch.Tensor,
        b_m: torch.Tensor,
        g_m: torch.Tensor,
        inv_dh: float,
        Bc: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del k_m, b_m, g_m, inv_dh
        M_len, BH, d_k, C = q_m.shape
        d_v = v_m.size(2)
        n_blocks = (M_len + int(Bc) - 1) // int(Bc)
        opts = {"device": q_m.device, "dtype": torch.float32}
        return (
            torch.empty((M_len, BH, d_v, C), **opts),
            torch.empty((n_blocks, BH, d_v, d_k), **opts),
        )

    @torch.library.custom_op("calcite5::woodbury_tern_m_bwd", mutates_args=())
    def _woodbury_tern_m_bwd_op(
        q_m: torch.Tensor,
        k_m: torch.Tensor,
        v_m: torch.Tensor,
        b_m: torch.Tensor,
        g_m: torch.Tensor,
        M_starts: torch.Tensor,
        grad_out: torch.Tensor,
        inv_dh: float,
        Bc: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ext = _load_fast_scans_cuda_ext()
        assert ext is not None
        dq, dk, dv, db, dg = ext.woodbury_tern_m_bwd_cuda(
            q_m, k_m, v_m, b_m, g_m, M_starts, grad_out, float(inv_dh), int(Bc)
        )
        return dq, dk, dv, db, dg

    @_woodbury_tern_m_bwd_op.register_fake
    def _woodbury_tern_m_bwd_fake(
        q_m: torch.Tensor,
        k_m: torch.Tensor,
        v_m: torch.Tensor,
        b_m: torch.Tensor,
        g_m: torch.Tensor,
        M_starts: torch.Tensor,
        grad_out: torch.Tensor,
        inv_dh: float,
        Bc: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del M_starts, grad_out, inv_dh, Bc
        return (
            torch.empty_like(q_m, dtype=torch.float32),
            torch.empty_like(k_m, dtype=torch.float32),
            torch.empty_like(v_m, dtype=torch.float32),
            torch.empty_like(b_m, dtype=torch.float32),
            torch.empty_like(g_m, dtype=torch.float32),
        )

    def _woodbury_m_setup_context(ctx, inputs, output) -> None:
        q_m, k_m, v_m, b_m, g_m, inv_dh, Bc = inputs
        _, M_starts = output
        ctx.save_for_backward(q_m, k_m, v_m, b_m, g_m, M_starts)
        ctx.inv_dh = float(inv_dh)
        ctx.Bc = int(Bc)

    def _woodbury_m_bwd_bridge(
        ctx, grad_out: torch.Tensor | None, grad_m_starts: torch.Tensor | None
    ):
        del grad_m_starts
        if grad_out is None:
            return None, None, None, None, None, None, None
        q_m, k_m, v_m, b_m, g_m, M_starts = ctx.saved_tensors
        dq, dk, dv, db, dg = _woodbury_tern_m_bwd_op(
            q_m, k_m, v_m, b_m, g_m, M_starts, grad_out.contiguous(), ctx.inv_dh, ctx.Bc
        )
        return dq, dk, dv, db, dg, None, None

    _woodbury_tern_m_fwd_op.register_autograd(
        _woodbury_m_bwd_bridge, setup_context=_woodbury_m_setup_context
    )


def fast_utf8_chunk_span(
    raw_cand: torch.Tensor,
    valid_mask: torch.Tensor,
    can_fire_mask: torch.Tensor,
    is_lead: torch.Tensor,
    char_len: torch.Tensor,
    max_cp_span: int,
    max_byte_span: int,
    refill_int: int,
    cost_int: int,
    burst_int: int,
) -> torch.Tensor:
    if raw_cand.is_cuda:
        ext = _load_fast_scans_cuda_ext()
        if ext is not None:
            if _HAS_CUSTOM_OP:
                return _utf8_chunk_span_op(
                    raw_cand,
                    valid_mask,
                    can_fire_mask,
                    is_lead,
                    char_len,
                    int(max_cp_span),
                    int(max_byte_span),
                    int(refill_int),
                    int(cost_int),
                    int(burst_int),
                )
            return ext.utf8_chunk_span_cuda(
                raw_cand,
                valid_mask,
                can_fire_mask,
                is_lead,
                char_len,
                int(max_cp_span),
                int(max_byte_span),
                int(refill_int),
                int(cost_int),
                int(burst_int),
            )
    return _enforce_utf8_chunk_span_jit(
        raw_cand,
        valid_mask,
        can_fire_mask,
        is_lead,
        char_len,
        max_cp_span,
        max_byte_span,
        refill_int,
        cost_int,
        burst_int,
    )


class _FastRefractoryGateFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p_raw: torch.Tensor) -> torch.Tensor:
        ext = _load_fast_scans_cuda_ext()
        out, mult = ext.causal_refractory_gate_cuda(p_raw)
        ctx.save_for_backward(mult)
        return out.to(p_raw.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (mult,) = ctx.saved_tensors
        return (grad_out.float() * mult).to(grad_out.dtype)


def fast_causal_refractory_gate(p_raw: torch.Tensor) -> torch.Tensor:
    if p_raw.is_cuda and p_raw.dim() == 3:
        ext = _load_fast_scans_cuda_ext()
        if ext is not None:
            if _HAS_CUSTOM_OP:
                out, _ = _causal_refractory_gate_fwd_op(p_raw)
                return out
            return _FastRefractoryGateFn.apply(p_raw)
    return causal_refractory_gate_jit(p_raw)


def fast_project_birkhoff_to_perm(scores: torch.Tensor) -> torch.Tensor:
    if scores.is_cuda and scores.dim() == 3 and 0 < scores.size(1) <= 32 and scores.size(1) == scores.size(2):
        ext = _load_fast_scans_cuda_ext()
        if ext is not None:
            if _HAS_CUSTOM_OP:
                return _birkhoff_greedy_perm_op(scores.detach())
            return ext.birkhoff_greedy_perm_cuda(scores).to(scores.dtype)
    return project_birkhoff_to_perm_jit(scores)


class _FastCrystalScanFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        s0: torch.Tensor,
        kw_eff_seq: torch.Tensor,
        kw_vg_seq: torch.Tensor,
        v_ste_seq: torch.Tensor,
        P_eff_seq: torch.Tensor,
        is_hard: bool,
        scale_r: float,
    ) -> torch.Tensor:
        ext = _load_fast_scans_cuda_ext()
        s_out = ext.crystal_scan_fwd_cuda(
            s0, kw_eff_seq, v_ste_seq, P_eff_seq, bool(is_hard), float(scale_r)
        )
        ctx.save_for_backward(s_out, kw_eff_seq, kw_vg_seq, v_ste_seq, P_eff_seq)
        ctx.is_hard = bool(is_hard)
        ctx.scale_r = float(scale_r)
        return s_out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        s_out, kw_eff_seq, kw_vg_seq, v_ste_seq, P_eff_seq = ctx.saved_tensors
        ext = _load_fast_scans_cuda_ext()
        g_s0, g_kwe, g_vs, g_P = ext.crystal_scan_bwd_cuda(
            s_out,
            kw_eff_seq,
            kw_vg_seq,
            v_ste_seq,
            P_eff_seq,
            grad_out,
            ctx.is_hard,
            ctx.scale_r,
        )
        return (
            g_s0.to(s_out.dtype),
            g_kwe.to(kw_eff_seq.dtype),
            torch.zeros_like(kw_vg_seq),
            g_vs.to(v_ste_seq.dtype),
            g_P.to(P_eff_seq.dtype),
            None,
            None,
        )


def fast_diag_birkhoff_crystal_scan(
    s0: torch.Tensor,
    kw_eff_seq: torch.Tensor,
    kw_vg_seq: torch.Tensor,
    v_ste_seq: torch.Tensor,
    P_eff_seq: torch.Tensor,
    is_hard: bool,
    scale_r: float,
    Bc: int = 32,
) -> torch.Tensor:
    if (
        s0.is_cuda
        and s0.size(2) == 32
        and 0 < s0.size(1) <= 32
        and _load_fast_scans_cuda_ext() is not None
    ):
        with torch.amp.autocast(s0.device.type, enabled=False):
            if _HAS_CUSTOM_OP:
                return _crystal_scan_fwd_op(
                    s0.float(),
                    kw_eff_seq.float(),
                    kw_vg_seq.float(),
                    v_ste_seq.float(),
                    P_eff_seq.float(),
                    bool(is_hard),
                    float(scale_r),
                )
            return _FastCrystalScanFn.apply(
                s0.float(),
                kw_eff_seq.float(),
                kw_vg_seq.float(),
                v_ste_seq.float(),
                P_eff_seq.float(),
                bool(is_hard),
                float(scale_r),
            )
    with torch.amp.autocast(s0.device.type, enabled=False):
        return _ref_crystal_scan(
            s0.float(),
            kw_eff_seq.float(),
            kw_vg_seq.float(),
            v_ste_seq.float(),
            P_eff_seq.float(),
            is_hard,
            scale_r,
            Bc=Bc,
        )


class _FastWoodburyTernaryMFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q_m: torch.Tensor,
        k_m: torch.Tensor,
        v_m: torch.Tensor,
        b_m: torch.Tensor,
        g_m: torch.Tensor,
        inv_dh: float,
        Bc: int,
    ) -> torch.Tensor:
        ext = _load_fast_scans_cuda_ext()
        out_m, M_starts = ext.woodbury_tern_m_fwd_cuda(
            q_m, k_m, v_m, b_m, g_m, float(inv_dh), int(Bc)
        )
        ctx.save_for_backward(q_m, k_m, v_m, b_m, g_m, M_starts)
        ctx.inv_dh = float(inv_dh)
        ctx.Bc = int(Bc)
        return out_m

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q_m, k_m, v_m, b_m, g_m, M_starts = ctx.saved_tensors
        ext = _load_fast_scans_cuda_ext()
        dq, dk, dv, db, dg = ext.woodbury_tern_m_bwd_cuda(
            q_m, k_m, v_m, b_m, g_m, M_starts, grad_out, ctx.inv_dh, ctx.Bc
        )
        return dq, dk, dv, db, dg, None, None


def fast_neural_woodbury_chunk_scan(
    q_chunks: torch.Tensor,
    k_seq: torch.Tensor,
    v_seq: torch.Tensor,
    beta_seq: torch.Tensor,
    gamma_seq: torch.Tensor,
    inv_lam: float,
    q_epi_chunks: torch.Tensor | None = None,
    k_epi_seq: torch.Tensor | None = None,
    tern_gate: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        q_chunks.is_cuda
        and q_chunks.size(2) == 4
        and q_chunks.size(4) <= 64
        and v_seq.size(-1) <= 64
        and _load_fast_scans_cuda_ext() is not None
    ):
        ext = _load_fast_scans_cuda_ext()
        B, M_len, C, H, d_k = q_chunks.shape
        d_v = v_seq.size(-1)
        BH = B * H
        t_gate = float(max(0.0, tern_gate))
        inv_dh = t_gate / 144.0

        with torch.amp.autocast(q_chunks.device.type, enabled=False):
            q_m = q_chunks.float().permute(1, 0, 3, 4, 2).reshape(M_len, BH, d_k, C).contiguous()
            k_m = k_seq.float().permute(1, 0, 2, 3).reshape(M_len, BH, d_k, 1).contiguous()
            v_m = v_seq.float().permute(1, 0, 2, 3).reshape(M_len, BH, d_v, 1).contiguous()
            b_m = beta_seq.float().permute(1, 0, 2, 3).reshape(M_len, BH, 1, 1).contiguous()
            g_m = gamma_seq.float().permute(1, 0, 2, 3).reshape(M_len, BH, 1, 1).contiguous()

            with torch.no_grad():
                if q_epi_chunks is None or q_epi_chunks is q_chunks:
                    qe_m = q_m.detach()
                else:
                    qe_m = q_epi_chunks.detach().float().permute(1, 0, 3, 4, 2).reshape(M_len, BH, d_k, C).contiguous()
                if k_epi_seq is None or k_epi_seq is k_seq:
                    ke_m = k_m.detach()
                else:
                    ke_m = k_epi_seq.detach().float().permute(1, 0, 2, 3).reshape(M_len, BH, d_k, 1).contiguous()
                if _HAS_CUSTOM_OP:
                    u_m = _woodbury_tern_p_uncerts_op(
                        qe_m, ke_m, b_m.detach(), g_m.detach(), float(inv_lam), t_gate
                    )
                else:
                    u_m = ext.woodbury_tern_p_uncerts_cuda(
                        qe_m, ke_m, b_m.detach(), g_m.detach(), float(inv_lam), t_gate
                    )
                uncerts = (
                    u_m.reshape(M_len, B, H, C)
                    .permute(1, 0, 3, 2)
                    .reshape(B, M_len * C, H)
                    .to(v_seq.dtype)
                )

            if _HAS_CUSTOM_OP:
                out_m, _ = _woodbury_tern_m_fwd_op(q_m, k_m, v_m, b_m, g_m, inv_dh, 16)
            else:
                out_m = _FastWoodburyTernaryMFn.apply(q_m, k_m, v_m, b_m, g_m, inv_dh, 16)
            out_all = (
                out_m.reshape(M_len, B, H, d_v, C)
                .permute(1, 0, 4, 2, 3)
                .reshape(B, M_len * C, H, d_v)
                .to(v_seq.dtype)
            )
            return out_all, uncerts

    return _ref_woodbury_scan(
        q_chunks,
        k_seq,
        v_seq,
        beta_seq,
        gamma_seq,
        inv_lam,
        q_epi_chunks,
        k_epi_seq,
        tern_gate,
    )


# ---------------------------------------------------------------------------
# 6. Zero-Duplication Crystal & Lean Woodbury Wrappers (TorchInductor-Ready)
# ---------------------------------------------------------------------------


def _sphere_norm_inline(x: torch.Tensor, scale: float) -> torch.Tensor:
    return x * (torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True) + 1e-6) * scale)


def _get_crystal_sched_tensor(
    layer: VoronoiCrystalLayer,
    device: torch.device,
    s_temp: float,
    o_temp: float,
    g_tau: float,
    eff_alpha: float,
    sharp_w: float,
    snap_a: float,
) -> torch.Tensor:
    key = (device, s_temp, o_temp, g_tau, eff_alpha, sharp_w, snap_a)
    cached = getattr(layer, "_fast_sched_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    t = torch.tensor(
        [s_temp, o_temp, g_tau, eff_alpha, sharp_w, snap_a],
        device=device,
        dtype=torch.float32,
    )
    layer.__dict__["_fast_sched_cache"] = (key, t)
    return t


def _crystal_pre_scan_impl(
    layer: VoronoiCrystalLayer,
    x: torch.Tensor,
    h_lex: torch.Tensor | None,
    sched: torch.Tensor,
    is_hard: bool,
    use_snap: bool,
    use_well: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    B, T, D = x.shape
    C, H, Gh, ds = layer.C, layer.H, layer.G_h, layer.d_slot
    R_read, H_r = layer.R_read, layer.H_r
    pad = (C - (T % C)) % C
    T_pad = T + pad
    M_len = T_pad // C
    scale_r = math.sqrt(layer.d_s)

    s_temp = sched[0]
    o_temp = sched[1]
    g_tau = sched[2]
    eff_alpha = sched[3]
    sharp_w = sched[4]
    snap_a = sched[5]

    # Single `_encode_tokens` and single `K_slots` normalization shared by write, op, and read
    zc, u_vec, sal_subj = layer._encode_tokens(x, h_lex)
    k_slots_u = layer._safe_unit(layer.K_slots)

    sal_w_win = F.pad(sal_subj, (0, 0, 4, 0), value=-1e4).unfold(1, 5, 1)
    w_lag_att = F.softmax(
        sal_w_win + layer.write_lag_logits.float().view(1, 1, H, 5), dim=-1
    )
    u_w_win = (
        F.pad(u_vec.reshape(B, T, H * ds), (0, 0, 4, 0))
        .unfold(1, 5, 1)
        .view(B, T, H, ds, 5)
    )
    kw_vec_tok = layer._safe_unit(
        torch.matmul(u_w_win, w_lag_att.unsqueeze(-1)).squeeze(-1)
        + 0.30 * layer._safe_unit(layer.w_ctx_proj(x).view(B, T, H, ds))
    )

    # Readout attention weights `q_soft` computed from slices `[..., 1:]` (zero extra unfold/repeat)
    r_lag_att = F.softmax(
        sal_w_win[..., 1:].unsqueeze(3)
        + layer.read_lag_logits.float().view(1, 1, H, R_read, 4),
        dim=-1,
    )
    r_u_agg = torch.matmul(r_lag_att, u_w_win[..., 1:].transpose(-1, -2)).reshape(
        B, T, H_r, ds
    )
    qr_vec = layer._safe_unit(
        r_u_agg + 0.30 * layer._safe_unit(layer.r_ctx_proj(x).view(B, T, H_r, ds))
    )
    q_logits = (
        torch.einsum("bthrd,hgd->bthrg", qr_vec.view(B, T, H, R_read, ds), k_slots_u)
        * s_temp
    )
    q_soft = F.softmax(q_logits, dim=-1).to(x.dtype)

    if pad > 0:
        x_pad = F.pad(x, (0, 0, 0, pad))
        zc_pad = F.pad(zc, (0, 0, 0, pad))
        u_pad = F.pad(u_vec, (0, 0, 0, 0, 0, pad))
        kw_pad = F.pad(kw_vec_tok, (0, 0, 0, 0, 0, pad))
        valid_1d = (torch.arange(T_pad, device=x.device) < T).view(1, T_pad, 1)
    else:
        x_pad = x
        zc_pad = zc
        u_pad = u_vec
        kw_pad = kw_vec_tok
        valid_1d = torch.ones(1, T_pad, 1, device=x.device, dtype=torch.bool)
    valid_ch = valid_1d.view(1, M_len, C, 1)

    x_ch = x_pad.view(B, M_len, C, D)
    kw_ch = kw_pad.view(B, M_len, C, H, ds)

    cb, c_norm_t = layer.get_codebook_fp32()
    s0_bh = layer._prepare_s0(B, cb, c_norm_t).reshape(B * H, Gh, layer.d_s)

    w_logit_tok = layer.w_eta_proj(x_ch).float()
    w_pool = F.softmax(w_logit_tok.masked_fill(~valid_ch, -1e4) * 2.0, dim=2)
    w_logit = (w_logit_tok * w_pool).sum(dim=2)
    kw_vec = layer._safe_unit((kw_ch * w_pool.unsqueeze(-1)).sum(dim=2))

    v_tok = layer.w_val_proj(x_ch).float().view(B, M_len, C, H, layer.d_s)
    v_pool = F.softmax(
        layer.pool_score(x_ch).float().masked_fill(~valid_ch, -1e4), dim=2
    )
    v_unit = layer._safe_unit((v_tok * v_pool.unsqueeze(-1)).sum(dim=2))
    v_logits = torch.matmul(v_unit, c_norm_t) * 9.0
    v_hard = cb[v_logits.argmax(dim=-1)]

    need_grad = torch.is_grad_enabled()
    need_commit = bool(layer.training)
    if is_hard and (not need_grad) and (not need_commit):
        v_probs = None
        v_ste = v_hard
    else:
        v_probs = F.softmax(v_logits, dim=-1)
        v_soft = _sphere_norm_inline(torch.matmul(v_probs, cb), scale_r)
        if is_hard:
            v_cur = v_hard
        elif use_snap:
            v_cur = _sphere_norm_inline((1.0 - eff_alpha) * v_soft + eff_alpha * v_hard, scale_r)
        else:
            v_cur = v_soft

        if need_grad and (v_soft.requires_grad or v_cur.requires_grad):
            vs_u, vq_u = v_soft / scale_r, v_cur / scale_r
            u_ste = F.normalize(vs_u + vq_u, dim=-1).detach()
            v_enc = (
                v_soft
                - 2.0 * (v_soft * u_ste).sum(-1, keepdim=True) * u_ste
                + 2.0 * (v_soft * vs_u.detach()).sum(-1, keepdim=True) * vq_u.detach()
            )
            v_ste = v_enc + (v_cur - v_enc).detach()
        else:
            v_ste = v_cur

    left_pad = (layer.W_chunks - 1) * C
    w_len = layer.W_chunks * C
    zc_win = F.pad(zc_pad, (0, 0, left_pad, 0)).unfold(1, w_len, C).transpose(2, 3)
    u_win = (
        F.pad(u_pad.reshape(B, T_pad, H * ds), (0, 0, left_pad, 0))
        .unfold(1, w_len, C)
        .transpose(2, 3)
        .view(B, M_len, w_len, H, ds)
    )
    win_valid = (
        F.pad(valid_1d, (0, 0, left_pad, 0))
        .unfold(1, w_len, C)
        .transpose(2, 3)
        .unsqueeze(-1)
    )

    ptr_raw = layer.op_ptr_proj(zc_win).float().view(
        B, M_len, w_len, H, 2
    ) + layer.ptr_pos_bias.float().view(1, 1, w_len, H, 2)
    ptr_raw = ptr_raw.masked_fill(~win_valid, -1e4)
    att_a = F.softmax(ptr_raw[..., 0], dim=2)
    att_b = F.softmax(ptr_raw[..., 1] - 4.5 * att_a.detach(), dim=2)

    ua_vec = layer._safe_unit((u_win * att_a.unsqueeze(-1)).sum(dim=2))
    ub_vec = layer._safe_unit((u_win * att_b.unsqueeze(-1)).sum(dim=2))

    # Single batched slot projection across (kw_vec, ua_vec, ub_vec)
    all_vecs = torch.cat((kw_vec, ua_vec, ub_vec), dim=1)
    all_logits = torch.einsum("bmhd,hgd->bmhg", all_vecs, k_slots_u) * s_temp
    k_logits, psi_a, psi_b = all_logits.chunk(3, dim=1)

    w_eta = torch.sigmoid((w_logit + 0.58) * g_tau - 0.58)
    w_hard_g = (w_eta.detach() >= 0.36).float()
    if is_hard and (not need_grad) and (not need_commit):
        k_prob = None
        k_hard = F.one_hot(k_logits.argmax(dim=-1), num_classes=Gh).float()
        kw_targ = (w_hard_g.unsqueeze(-1) * k_hard).unsqueeze(-1)
        kw_eff = kw_targ
        kw_vg = kw_targ
    else:
        k_prob = F.softmax(k_logits, dim=-1)
        w_peak = k_prob / k_prob.detach().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        kw_soft = (w_eta.unsqueeze(-1) * w_peak).unsqueeze(-1)
        k_hard = F.one_hot(k_prob.argmax(dim=-1), num_classes=Gh).float()
        kw_targ = (w_hard_g.unsqueeze(-1) * k_hard).unsqueeze(-1)

        if is_hard:
            kw_eff = kw_soft + (kw_targ - kw_soft).detach() if need_grad else kw_targ
            kw_vg = torch.maximum(kw_targ, kw_soft).detach()
        elif use_snap:
            kw_cur = (1.0 - eff_alpha) * kw_soft + eff_alpha * kw_targ
            kw_eff = kw_soft + (kw_cur - kw_soft).detach() if need_grad else kw_cur
            kw_vg = torch.maximum(kw_cur, kw_soft).detach()
        else:
            kw_eff = kw_soft
            kw_vg = kw_soft.detach()

    # Single `_compute_ptr_swap` with resolved well_scale (eliminates redundant QAT/eval call)
    _, _, T_swap = layer._compute_ptr_swap(psi_a, psi_b, well_scale=1.0 if use_well else 0.0)

    z_mean = x_ch.float().mean(dim=2).to(x.dtype)
    z_op_in = (
        F.pad(z_mean, (0, 0, 3, 0))
        .unfold(1, 4, 1)
        .permute(0, 1, 3, 2)
        .reshape(B, M_len, 4 * D)
    )
    z_op = F.silu(layer.op_ctx_proj(z_op_in))

    B_mats = log_sinkhorn_6(layer.birkhoff_logits.float() * o_temp)
    m_prob = F.softmax(
        layer.op_mode_proj(z_op).float().view(B, M_len, H, 1 + layer.M_glob) * o_temp,
        dim=-1,
    )
    P_cand = m_prob[..., 0:1].unsqueeze(-1) * T_swap + torch.einsum(
        "bmhk,hkij->bmhij", m_prob[..., 1:], B_mats
    )

    p_logit_raw = layer.perm_gate_proj(z_op).float()
    p_logit = -4.5 + F.softplus(p_logit_raw + 4.5)
    p_raw = torch.sigmoid((p_logit + 0.49) * g_tau - 0.49)

    if need_commit and k_prob is not None and v_probs is not None:
        wd = w_eta.detach()
        w_sum = wd.sum(dim=(0, 1)).clamp(min=1e-3)
        p_slot_bar = (k_prob * wd.unsqueeze(-1)).sum(dim=(0, 1)) / w_sum.unsqueeze(-1)
        cap_slot = F.relu(p_slot_bar - max(0.20, 1.35 / float(Gh))).pow(2).sum() * 4.0
        p_cb_bar = (v_probs * wd.unsqueeze(-1)).sum(dim=(0, 1)) / w_sum.unsqueeze(-1)
        cap_cb = F.relu(p_cb_bar - max(0.20, 1.35 / float(layer.K))).pow(2).sum() * 4.0
        sharp_k = (wd * (1.0 - k_prob.max(dim=-1).values)).mean()
        commit_v = (wd * (1.0 - (v_unit * (v_hard / scale_r).detach()).sum(dim=-1))).mean()
        w_bimod = (w_eta - w_hard_g).pow(2).mean()
        pre_commit_loss = (
            commit_v
            + cap_slot
            + cap_cb
            + sharp_w * sharp_k
            + snap_a * 0.04 * w_bimod
            + 0.008 * w_eta.mean()
        )
    else:
        pre_commit_loss = x.new_zeros((), dtype=torch.float32)

    BH = B * H
    kw_eff_bh = kw_eff.permute(0, 2, 1, 3, 4).reshape(BH, M_len, Gh, 1)
    kw_vg_bh = kw_vg.permute(0, 2, 1, 3, 4).reshape(BH, M_len, Gh, 1)
    v_ste_bh = v_ste.permute(0, 2, 1, 3).reshape(BH, M_len, layer.d_s)
    return s0_bh, kw_eff_bh, kw_vg_bh, v_ste_bh, P_cand, p_raw, q_soft, pre_commit_loss


def _crystal_post_gate_impl(
    layer: VoronoiCrystalLayer,
    p_eta: torch.Tensor,
    P_cand: torch.Tensor,
    P_perm_hard: torch.Tensor | None,
    pre_commit_loss: torch.Tensor,
    sched: torch.Tensor,
    is_hard: bool,
    use_snap: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, M_len, H, Gh, _ = P_cand.shape
    eye = layer.eye_Gh.float().view(1, 1, 1, Gh, Gh)
    eff_alpha = sched[3]
    sharp_w = sched[4]
    snap_a = sched[5]
    need_grad = torch.is_grad_enabled()

    if use_snap and P_perm_hard is not None:
        p_hard_g = (p_eta.detach() >= 0.38).float()
        P_targ = eye + p_hard_g.unsqueeze(-1).unsqueeze(-1) * (P_perm_hard - eye)
        if is_hard and not need_grad:
            P_eff = P_targ
        else:
            P_soft = eye + p_eta.unsqueeze(-1).unsqueeze(-1) * (P_cand - eye)
            if is_hard:
                P_eff = P_soft + (P_targ - P_soft).detach()
            else:
                P_cur = (1.0 - eff_alpha) * P_soft + eff_alpha * P_targ
                P_eff = P_soft + (P_cur - P_soft).detach() if need_grad else P_cur
    else:
        P_eff = eye + p_eta.unsqueeze(-1).unsqueeze(-1) * (P_cand - eye)

    if layer.training:
        pd = p_eta.detach()
        sharp_p = (pd * (1.0 - P_cand.max(dim=-1).values.mean(dim=-1))).mean()
        p_bin_d = (pd >= 0.38).float()
        commit_loss = (
            pre_commit_loss
            + sharp_w * sharp_p
            + snap_a * 0.04 * (p_eta - p_bin_d).pow(2).mean()
            + 0.015 * F.relu(p_eta.mean() - 0.06)
        )
    else:
        commit_loss = pre_commit_loss
    P_eff_bh = P_eff.permute(0, 2, 1, 3, 4).reshape(B * H, M_len, Gh, Gh)
    return P_eff_bh, commit_loss


def _crystal_readout_impl(
    layer: VoronoiCrystalLayer,
    s_bh: torch.Tensor,
    q_soft: torch.Tensor,
    T: int,
    pad: int,
) -> torch.Tensor:
    B, _, H, R_read, Gh = q_soft.shape
    C = layer.C
    T_pad = T + pad
    M_len = T_pad // C
    BH = B * H
    if pad > 0:
        q_soft_pad = F.pad(q_soft, (0, 0, 0, 0, 0, 0, 0, pad))
    else:
        q_soft_pad = q_soft
    q_mat = (
        q_soft_pad.view(B, M_len, C, H, R_read, Gh)
        .permute(0, 3, 1, 2, 4, 5)
        .reshape(BH * M_len, C * R_read, Gh)
    )
    s_mat = s_bh.to(q_soft.dtype).reshape(BH * M_len, Gh, layer.d_s)
    ret = (
        torch.bmm(q_mat, s_mat)
        .view(B, H, M_len, C, R_read, layer.d_s)
        .permute(0, 2, 3, 1, 4, 5)
        .reshape(B, T_pad, layer.H_r * layer.d_s)
    )
    if pad > 0:
        ret = ret[:, :T]
    return layer.out_proj(layer.out_norm(ret))


def fast_crystal_forward(
    layer: VoronoiCrystalLayer,
    x: torch.Tensor,
    h_lex: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Zero-duplication `VoronoiCrystalLayer.forward` (exact `< 1e-5` parity):
    - Calls `_encode_tokens` and `_safe_unit(K_slots)` ONCE (eliminating the 2x call in `crystal.py:692-693`).
    - Eliminates duplicate `_compute_ptr_swap` in QAT/eval and re-pad/re-unfold of read lag windows.
    - Reads compact `s_bh` via zero-copy `torch.bmm` (eliminating `C*R_read=4x` `s_read` expansion).
    - Invokes `fast_causal_refractory_gate`, `fast_project_birkhoff_to_perm`, and
      `fast_diag_birkhoff_crystal_scan` directly without global module monkey-patching.
    """
    B, T, _ = x.shape
    C, H, Gh = layer.C, layer.H, layer.G_h
    pad = (C - (T % C)) % C
    M_len = (T + pad) // C
    eff_alpha = (
        1.0
        if (not layer.training and layer.snap_alpha > 0.0)
        else (0.85 * float(layer.snap_alpha))
    )
    is_hard = eff_alpha >= 0.9999
    use_snap = eff_alpha > 0.0
    use_well = (not layer.training and layer.snap_alpha > 0.0) or layer.qat_active_mix > 0.0
    scale_r = math.sqrt(layer.d_s)

    sched = _get_crystal_sched_tensor(
        layer,
        x.device,
        float(layer.slot_temp),
        float(layer.op_temp),
        float(layer.gate_temp),
        float(eff_alpha),
        float(layer.sharp_weight),
        float(layer.snap_alpha),
    )

    pre_fn = getattr(layer, "_fast_crystal_pre_fn", None)
    if pre_fn is not None:
        s0_bh, kw_eff_bh, kw_vg_bh, v_ste_bh, P_cand, p_raw, q_soft, pre_commit_loss = pre_fn(
            x, h_lex, sched, is_hard, use_snap, use_well
        )
    else:
        s0_bh, kw_eff_bh, kw_vg_bh, v_ste_bh, P_cand, p_raw, q_soft, pre_commit_loss = (
            _crystal_pre_scan_impl(layer, x, h_lex, sched, is_hard, use_snap, use_well)
        )

    p_eta = fast_causal_refractory_gate(p_raw)
    if use_snap:
        P_perm_hard = fast_project_birkhoff_to_perm(
            P_cand.detach().reshape(-1, Gh, Gh)
        ).view(B, M_len, H, Gh, Gh)
    else:
        P_perm_hard = None

    post_fn = getattr(layer, "_fast_crystal_post_fn", None)
    if post_fn is not None:
        P_eff_bh, commit_loss = post_fn(
            p_eta, P_cand, P_perm_hard, pre_commit_loss, sched, is_hard, use_snap
        )
    else:
        P_eff_bh, commit_loss = _crystal_post_gate_impl(
            layer, p_eta, P_cand, P_perm_hard, pre_commit_loss, sched, is_hard, use_snap
        )
    layer.last_commit_loss = commit_loss if layer.training else None

    s_bh = fast_diag_birkhoff_crystal_scan(
        s0_bh, kw_eff_bh, kw_vg_bh, v_ste_bh, P_eff_bh, is_hard, scale_r
    )

    readout_fn = getattr(layer, "_fast_crystal_readout_fn", None)
    if readout_fn is not None:
        return readout_fn(s_bh, q_soft, T, pad)
    return _crystal_readout_impl(layer, s_bh, q_soft, T, pad)


def _woodbury_pre_scan_impl(
    layer: NeuralWoodburyLayer,
    x: torch.Tensor,
    h_lex: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    B, T, D_in = x.shape
    C = layer.C
    q_rw_t, q_epi_t = layer._compute_latched_keys(x, h_lex=h_lex)

    if layer.H_prose > 0:
        q_prose = layer._safe_unit(
            layer.prose_q_proj(x).view(B, T, layer.H_prose, layer.D)
        ).to(x.dtype)
        k_prose = layer._safe_unit(
            layer.prose_k_proj(x).view(B, T, layer.H_prose, layer.D)
        ).to(x.dtype)
        q_rw_all = torch.cat((q_rw_t, q_prose), dim=2)
        k_src_all = torch.cat((q_rw_t, k_prose), dim=2)
        q_epi_all = torch.cat((q_epi_t, q_epi_t[:, :, : layer.H_prose]), dim=2)
    else:
        q_rw_all = q_rw_t
        k_src_all = q_rw_t
        q_epi_all = q_epi_t

    pad = (C - (T % C)) % C
    if pad > 0:
        x_pad = F.pad(x, (0, 0, 0, pad))
        q_rw_pad = F.pad(q_rw_all, (0, 0, 0, 0, 0, pad))
        k_src_pad = F.pad(k_src_all, (0, 0, 0, 0, 0, pad))
        q_epi_pad = F.pad(q_epi_all, (0, 0, 0, 0, 0, pad))
    else:
        x_pad = x
        q_rw_pad = q_rw_all
        k_src_pad = k_src_all
        q_epi_pad = q_epi_all

    T_pad = x_pad.size(1)
    M_len = T_pad // C

    q_chunks = q_rw_pad.view(B, M_len, C, layer.H, layer.D)
    k_chunks = k_src_pad.view(B, M_len, C, layer.H, layer.D)
    q_epi_chunks = q_epi_pad.view(B, M_len, C, layer.H, layer.D)

    x_chunks = x_pad.view(B, M_len, C, D_in)
    w_scores = F.softmax(layer.write_pool_score(x_chunks).float(), dim=2)

    k_seq = layer._safe_unit(
        (k_chunks.float() * w_scores.unsqueeze(-1)).sum(dim=2)
    ).to(x.dtype)
    k_epi_seq = layer._safe_unit(
        (q_epi_chunks.float() * w_scores.unsqueeze(-1)).sum(dim=2)
    ).to(x.dtype)

    w_scores_x = w_scores.to(x.dtype)
    v_tok = layer.v_proj(x_chunks).view(B, M_len, C, layer.H, layer.D)
    v_seq = (v_tok * w_scores_x.unsqueeze(-1)).sum(dim=2)

    # Lean z_head: avoids 5D `(B, M_len, C, H, D_in)` (~118 MB) temporary tensor
    z_head = (x_chunks * w_scores_x.mean(dim=-1, keepdim=True)).sum(dim=2)
    beta = torch.sigmoid(layer.beta_proj(z_head)).unsqueeze(-1)
    gamma = torch.sigmoid(layer.gamma_proj(z_head)).unsqueeze(-1)
    return q_chunks, k_seq, v_seq, beta, gamma, q_epi_chunks, k_epi_seq


def _woodbury_post_scan_impl(
    layer: NeuralWoodburyLayer,
    x: torch.Tensor,
    read_pad: torch.Tensor,
    uncert_seq: torch.Tensor,
    T: int,
) -> torch.Tensor:
    B = x.size(0)
    read_seq = read_pad[:, :T].reshape(B, T, layer.total_dim)
    u_epi = uncert_seq[:, :, : layer.H_epi].float()
    conf_gate_epi = torch.sigmoid(
        layer.conf_slope.float().view(1, 1, layer.H_epi)
        * (layer.conf_thresh.float().view(1, 1, layer.H_epi) - u_epi)
    )

    if layer.H_prose > 0:
        q_intent = torch.sigmoid(layer.query_intent_proj(x).float())
        prose_gate = (1.0 - 0.88 * q_intent).expand(B, T, layer.H_prose)
        all_gates = torch.cat((conf_gate_epi, prose_gate), dim=-1).unsqueeze(-1)
    else:
        q_intent = None
        all_gates = conf_gate_epi.unsqueeze(-1)

    normed_read = (
        layer.out_norm(read_seq).view(B, T, layer.H, layer.D) * all_gates.to(x.dtype)
    ).reshape(B, T, layer.total_dim)

    raw_u_feats = torch.cat(
        (
            u_epi - 0.42,
            torch.tanh((u_epi - 0.42) * 7.0),
            F.relu(u_epi - 0.36) * 2.5,
        ),
        dim=-1,
    )
    if q_intent is not None:
        u_feats = q_intent.to(x.dtype) * (
            raw_u_feats.to(x.dtype) * layer.uncert_scale.to(x.dtype)
        )
    else:
        u_feats = raw_u_feats.to(x.dtype) * layer.uncert_scale.to(x.dtype)
    return layer.out_proj(torch.cat((normed_read, u_feats), dim=-1))


def _fast_woodbury_tern_gate(layer: NeuralWoodburyLayer) -> float:
    if layer.qat_active_mix > 0.0 or isinstance(layer.v_proj, PackedLUT2BitLinear):
        return 1.0
    w = layer.v_proj.weight
    if (
        layer.training
        and layer._tern_cache_key is not None
        and layer._tern_cache_val == 0.0
        and int(w.data_ptr()) == layer._tern_cache_key[0]
    ):
        return 0.0
    return layer._tern_gate()


def fast_woodbury_forward(
    layer: NeuralWoodburyLayer,
    x: torch.Tensor,
    h_lex: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Lean `NeuralWoodburyLayer.forward` (exact `< 1e-5` parity):
    - Eliminates 5D `z_head` `(B, M_len, C, H, D)` allocation (`~118 MB`).
    - Skips per-step `.item()` device-to-host sync in `_tern_gate()` during continuous FP training.
    - Invokes `fast_neural_woodbury_chunk_scan` directly without global module monkey-patching.
    """
    T = x.size(1)
    pre_fn = getattr(layer, "_fast_woodbury_pre_fn", None)
    if pre_fn is not None:
        q_chunks, k_seq, v_seq, beta, gamma, q_epi_chunks, k_epi_seq = pre_fn(x, h_lex)
    else:
        q_chunks, k_seq, v_seq, beta, gamma, q_epi_chunks, k_epi_seq = (
            _woodbury_pre_scan_impl(layer, x, h_lex)
        )

    t_gate = _fast_woodbury_tern_gate(layer)
    read_pad, uncert_pad = fast_neural_woodbury_chunk_scan(
        q_chunks,
        k_seq,
        v_seq,
        beta,
        gamma,
        layer.inv_lam,
        q_epi_chunks,
        k_epi_seq,
        t_gate,
    )
    uncert_seq = uncert_pad[:, :T]
    layer.last_epistemic_uncertainty = uncert_seq
    layer.last_gate_loss = None

    post_fn = getattr(layer, "_fast_woodbury_post_fn", None)
    if post_fn is not None:
        return post_fn(x, read_pad, uncert_seq, T)
    return _woodbury_post_scan_impl(layer, x, read_pad, uncert_seq, T)


# ---------------------------------------------------------------------------
# 7. Fused & Microbatch-Cached Fast Ternary STE QAT (`torch.compile` safe)
# ---------------------------------------------------------------------------

_DTYPE_TO_CODE: dict[torch.dtype, int] = {
    torch.float32: 0,
    torch.bfloat16: 1,
    torch.float16: 2,
}
_CODE_TO_DTYPE: dict[int, torch.dtype] = {
    0: torch.float32,
    1: torch.bfloat16,
    2: torch.float16,
}


def _dtype_to_code(dtype: torch.dtype) -> int:
    return _DTYPE_TO_CODE.get(dtype, -1)


def _compute_ternary_quant_value(
    w: torch.Tensor,
    mix: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    out_code = _dtype_to_code(out_dtype)
    if (
        w.is_cuda
        and w.numel() % 32 == 0
        and out_code >= 0
        and w.dtype in _DTYPE_TO_CODE
    ):
        ext = _load_fast_scans_cuda_ext()
        if ext is not None:
            return ext.fused_ternary_ste_cuda(w.detach(), float(mix), out_code)
    w_tern, _, _ = quantize_ternary_weights_ls(w)
    if mix < 0.9999:
        return ((1.0 - mix) * w.detach() + mix * w_tern).to(out_dtype)
    return w_tern.to(out_dtype)


class _FastSTELinearWeightFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, w_q_detached: torch.Tensor) -> torch.Tensor:
        ctx.weight_dtype = weight.dtype
        return w_q_detached

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        return grad_out.to(ctx.weight_dtype), None


def fast_quantize_ternary_ste(
    w: torch.Tensor,
    mix: float = 1.0,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Single-pass CUDA-fused ternary STE quantization (`fused_ternary_ste_cuda` when CUDA +
    `numel % 32 == 0`, exact PyTorch fallback otherwise), preserving exact STE gradient
    `dL/dw = dL/dw_q` on every call.
    """
    target_dtype = w.dtype if out_dtype is None else out_dtype
    with torch.no_grad():
        w_q = _compute_ternary_quant_value(w, float(mix), target_dtype)
    if torch.is_grad_enabled() and w.requires_grad:
        return _FastSTELinearWeightFn.apply(w, w_q)
    return w_q


class _QATSlotEntry:
    __slots__ = ("parent_ref", "key", "tensor")

    def __init__(self, parent_model: nn.Module):
        self.parent_ref = weakref.ref(parent_model)
        self.key: tuple[int, int, float, int] | None = None
        self.tensor: torch.Tensor | None = None


_QAT_SLOT_COUNTER = itertools.count(1)
_QAT_SLOTS: dict[int, _QATSlotEntry] = {}
_QAT_FINALIZERS: dict[int, weakref.finalize] = {}
_QAT_CACHE_STATS: dict[str, int] = {"hits": 0, "misses": 0}
_QAT_OPT_STEP_HOOK: object = None


def _on_optimizer_step_post_hook(_optimizer: object, _args: object, _kwargs: object) -> None:
    """Invalidates cached ternary STE weights after any `opt.step()` (incl. `fused=True` AdamW which skips `_version`)."""
    if _QAT_SLOTS:
        clear_fast_qat_cache(reset_stats=False)


def _ensure_qat_opt_step_hook() -> None:
    global _QAT_OPT_STEP_HOOK
    if _QAT_OPT_STEP_HOOK is not None:
        return
    try:
        from torch.optim.optimizer import register_optimizer_step_post_hook

        _QAT_OPT_STEP_HOOK = register_optimizer_step_post_hook(_on_optimizer_step_post_hook)
    except Exception:
        _QAT_OPT_STEP_HOOK = None


def _maybe_remove_qat_opt_step_hook() -> None:
    global _QAT_OPT_STEP_HOOK
    if (not _QAT_SLOTS) and _QAT_OPT_STEP_HOOK is not None:
        try:
            _QAT_OPT_STEP_HOOK.remove()
        except Exception:
            pass
        _QAT_OPT_STEP_HOOK = None


def _drop_qat_cache_slot(slot_id: int) -> None:
    _QAT_SLOTS.pop(slot_id, None)
    _QAT_FINALIZERS.pop(slot_id, None)
    _maybe_remove_qat_opt_step_hook()


def get_fast_qat_cache_stats() -> dict[str, int]:
    """Returns global microbatch cache hit/miss counters and active cached slot count."""
    return {
        "hits": int(_QAT_CACHE_STATS["hits"]),
        "misses": int(_QAT_CACHE_STATS["misses"]),
        "active_slots": len(_QAT_SLOTS),
    }


def clear_fast_qat_cache(reset_stats: bool = False) -> None:
    """Clears cached quantized weight tensors across active QAT slots."""
    for entry in _QAT_SLOTS.values():
        entry.key = None
        entry.tensor = None
    if reset_stats:
        _QAT_CACHE_STATS["hits"] = 0
        _QAT_CACHE_STATS["misses"] = 0


def _cached_ternary_ste_core(
    w: torch.Tensor,
    out_dtype_code: int,
    slot_id: int,
) -> torch.Tensor:
    out_dtype = _CODE_TO_DTYPE.get(out_dtype_code, w.dtype)
    entry = _QAT_SLOTS.get(slot_id)
    if entry is None:
        _QAT_CACHE_STATS["misses"] += 1
        with torch.no_grad():
            return _compute_ternary_quant_value(w, 1.0, out_dtype)

    parent = entry.parent_ref()
    mix = float(getattr(parent, "qat_mix", 1.0)) if parent is not None else 1.0
    key = (int(w.data_ptr()), int(w._version), mix, out_dtype_code)
    cached = entry.tensor
    if (
        cached is not None
        and entry.key == key
        and (not cached.is_inference())
    ):
        _QAT_CACHE_STATS["hits"] += 1
        return cached

    _QAT_CACHE_STATS["misses"] += 1
    if w.is_inference():
        with torch.no_grad():
            return _compute_ternary_quant_value(w, mix, out_dtype)
    with torch.inference_mode(False), torch.no_grad():
        w_q = _compute_ternary_quant_value(w, mix, out_dtype)
    entry.key = key
    entry.tensor = w_q
    return w_q


if _HAS_CUSTOM_OP:

    @torch.library.custom_op("calcite5::cached_ternary_ste", mutates_args=())
    def _cached_ternary_ste_op(
        w: torch.Tensor,
        out_dtype_code: int,
        slot_id: int,
    ) -> torch.Tensor:
        return _cached_ternary_ste_core(w, out_dtype_code, slot_id)

    @_cached_ternary_ste_op.register_fake
    def _cached_ternary_ste_fake(
        w: torch.Tensor,
        out_dtype_code: int,
        slot_id: int,
    ) -> torch.Tensor:
        out_dtype = _CODE_TO_DTYPE.get(out_dtype_code, w.dtype)
        return torch.empty_like(w, dtype=out_dtype)

    def _cached_ternary_ste_setup(ctx, inputs, output) -> None:
        w, _, _ = inputs
        ctx.w_dtype = w.dtype

    def _cached_ternary_ste_bwd(ctx, grad_out: torch.Tensor):
        return grad_out.to(ctx.w_dtype), None, None

    _cached_ternary_ste_op.register_autograd(
        _cached_ternary_ste_bwd, setup_context=_cached_ternary_ste_setup
    )
else:
    _cached_ternary_ste_op = None


def _make_single_pass_qat_forward():
    def _fast_qat_linear_forward(self_lin: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        slot_id: int = self_lin._fast_qat_slot_id
        dev_type = x.device.type
        if torch.is_autocast_enabled(dev_type):
            target_dtype = torch.get_autocast_dtype(dev_type)
        else:
            target_dtype = x.dtype
        out_code = _dtype_to_code(target_dtype)
        if _cached_ternary_ste_op is not None and out_code >= 0:
            w_q = _cached_ternary_ste_op(self_lin.weight, out_code, slot_id)
        else:
            w_q_det = _cached_ternary_ste_core(self_lin.weight, out_code, slot_id)
            if out_code < 0:
                w_q_det = w_q_det.to(target_dtype)
            if torch.is_grad_enabled() and self_lin.weight.requires_grad:
                w_q = _FastSTELinearWeightFn.apply(self_lin.weight, w_q_det)
            else:
                w_q = w_q_det
        x_in = x.to(target_dtype) if x.dtype != target_dtype else x
        b = self_lin.bias.to(target_dtype) if self_lin.bias is not None else None
        return F.linear(x_in, w_q, b)

    return _fast_qat_linear_forward


def configure_fast_ternary_qat(model: nn.Module, enabled: bool) -> int:
    """
    Replaces `nn.Linear.forward` with single-pass, fused, microbatch-cached ternary STE QAT
    (removing the 2x redundant `F.linear` from `register_forward_hook` in `simvq_split/models.py`
    and caching quantized weights per `(data_ptr, weight._version, mix, dtype)` while keeping
    exact live STE gradients flowing to `self_lin.weight` on every microbatch).
    Leaves `simvq_split/` untouched and restores original methods + frees cached weights on `enabled=False`.
    """
    handles = getattr(model, "_qat_handles", None)
    if isinstance(handles, list):
        for h in handles:
            h.remove()
        handles.clear()

    patched = getattr(model, "_fast_qat_patched_linears", None)
    if isinstance(patched, list):
        for mod in patched:
            slot_id = mod.__dict__.pop("_fast_qat_slot_id", None)
            if slot_id is not None:
                fin = _QAT_FINALIZERS.pop(int(slot_id), None)
                if fin is not None:
                    fin.detach()
                _drop_qat_cache_slot(int(slot_id))
            if "_orig_forward_before_fast_qat" in mod.__dict__:
                del mod.__dict__["forward"]
                del mod.__dict__["_orig_forward_before_fast_qat"]
        patched.clear()
    else:
        patched = []
        setattr(model, "_fast_qat_patched_linears", patched)

    if not enabled:
        _maybe_remove_qat_opt_step_hook()
        return 0

    _ensure_qat_opt_step_hook()
    fn = _make_single_pass_qat_forward()
    bc = getattr(model, "byte_chunker", None)
    lm_head = getattr(model, "lm_head", None)
    count = 0
    for _, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and mod is not lm_head:
            if bc is not None and bc.is_locked_router_module(mod):
                continue
            slot_id = next(_QAT_SLOT_COUNTER)
            _QAT_SLOTS[slot_id] = _QATSlotEntry(model)
            _QAT_FINALIZERS[slot_id] = weakref.finalize(
                mod, _drop_qat_cache_slot, slot_id
            )
            mod.__dict__["_fast_qat_slot_id"] = slot_id
            mod.__dict__["_orig_forward_before_fast_qat"] = True
            mod.forward = types.MethodType(fn, mod)
            patched.append(mod)
            count += 1
    return count


def _make_compiled_crystal_stages(
    layer: VoronoiCrystalLayer, mode: str, dynamic: bool = True
) -> None:
    def _pre(
        x: torch.Tensor,
        h_lex: torch.Tensor | None,
        sched: torch.Tensor,
        is_hard: bool,
        use_snap: bool,
        use_well: bool,
    ):
        return _crystal_pre_scan_impl(layer, x, h_lex, sched, is_hard, use_snap, use_well)

    def _post(
        p_eta: torch.Tensor,
        P_cand: torch.Tensor,
        P_perm_hard: torch.Tensor | None,
        pre_commit_loss: torch.Tensor,
        sched: torch.Tensor,
        is_hard: bool,
        use_snap: bool,
    ):
        return _crystal_post_gate_impl(
            layer, p_eta, P_cand, P_perm_hard, pre_commit_loss, sched, is_hard, use_snap
        )

    def _readout(s_bh: torch.Tensor, q_soft: torch.Tensor, T: int, pad: int):
        return _crystal_readout_impl(layer, s_bh, q_soft, T, pad)

    layer.__dict__["_fast_crystal_pre_fn"] = torch.compile(_pre, mode=mode, dynamic=dynamic)
    layer.__dict__["_fast_crystal_post_fn"] = torch.compile(_post, mode=mode, dynamic=dynamic)
    layer.__dict__["_fast_crystal_readout_fn"] = torch.compile(_readout, mode=mode, dynamic=dynamic)


def _make_compiled_woodbury_stages(
    layer: NeuralWoodburyLayer, mode: str, dynamic: bool = True
) -> None:
    # Keep `_woodbury_pre_scan_impl` eager: PyTorch 2.11 nightly Inductor corrupts the
    # backward pass of `_compute_latched_keys` on zero-padded tail chunks (`h_lex == 0`),
    # causing 19x–256x gradient explosion in `byte_chunker.lex_up.weight.grad` (#59/#63).
    layer.__dict__.pop("_fast_woodbury_pre_fn", None)

    def _post(x: torch.Tensor, read_pad: torch.Tensor, uncert_seq: torch.Tensor, T: int):
        return _woodbury_post_scan_impl(layer, x, read_pad, uncert_seq, T)

    layer.__dict__["_fast_woodbury_post_fn"] = torch.compile(_post, mode=mode, dynamic=dynamic)


def _is_fast_woodbury_eligible(wb: object) -> bool:
    if wb is None:
        return False
    is_wb = isinstance(wb, NeuralWoodburyLayer) or hasattr(wb, "_compute_latched_keys")
    if not is_wb:
        return False
    if getattr(wb, "use_kalman_precond", False):
        return False
    if getattr(wb, "use_deltanet2_gates", False):
        return False
    if int(getattr(wb, "K_sub", 1)) != 1:
        return False
    return True


def _has_working_inductor_gpu() -> bool:
    if not hasattr(torch, "compile") or not torch.cuda.is_available():
        return False
    if importlib.util.find_spec("triton") is None:
        return False
    if sys.platform == "win32":
        # Triton's first-run `cuda_utils` build requires MSVC `cl.exe` + SDK headers in PATH/INCLUDE/LIB.
        prepare_cuda_build_env()
    try:
        import torch._inductor.utils as _iu

        if hasattr(_iu, "has_triton") and not _iu.has_triton():
            return False
    except Exception:
        return False
    return True


def compile_calcite_cortex(model: nn.Module, mode: str = "default") -> bool:
    """
    Compiles the pure continuous cortex, SwiGLU MLPs, chunker byte decoder/MLPs, and
    full pre-scan/post-scan Crystal & Woodbury pipelines with `torch.compile(..., dynamic=True)`
    while keeping custom CUDA scan ops cleanly zero-graph-break.
    """
    if not hasattr(torch, "compile"):
        return False
    try:
        import torch._dynamo as _dynamo

        _dynamo.config.cache_size_limit = max(64, int(_dynamo.config.cache_size_limit))
    except Exception:
        pass
    if getattr(model, "mid_mlp", None) is not None:
        model.mid_mlp = torch.compile(model.mid_mlp, mode=mode, dynamic=True)
    swa_stages = getattr(model, "swa_stages", None)
    if swa_stages is not None:
        for stg in swa_stages:
            for i in range(len(stg)):
                stg[i] = torch.compile(stg[i], mode=mode, dynamic=True)
    else:
        for attr in ("cortex_pre", "cortex_post"):
            ml = getattr(model, attr, None)
            if ml is not None:
                for i in range(len(ml)):
                    ml[i] = torch.compile(ml[i], mode=mode, dynamic=True)
    bc = getattr(model, "byte_chunker", None)
    if bc is not None:
        if getattr(bc, "enc_mlp", None) is not None:
            bc.enc_mlp = torch.compile(bc.enc_mlp, mode=mode, dynamic=True)
        if getattr(bc, "dec_mlp", None) is not None:
            bc.dec_mlp = torch.compile(bc.dec_mlp, mode=mode, dynamic=True)
        if hasattr(bc, "decode_from_chunks"):
            bc.decode_from_chunks = torch.compile(bc.decode_from_chunks, mode=mode, dynamic=True)
    for wname in ("woodbury", "woodbury_fast"):
        wb = getattr(model, wname, None)
        if _is_fast_woodbury_eligible(wb):
            _make_compiled_woodbury_stages(wb, mode=mode)
    for cname in ("crystal", "crystal_2"):
        cr = getattr(model, cname, None)
        if cr is not None and isinstance(cr, VoronoiCrystalLayer):
            _make_compiled_crystal_stages(cr, mode=mode)
    return True


def compile_calcite_v2(
    model: nn.Module, mode: str = "default", dynamic: bool = False
) -> bool:
    """
    Safe `Calcite v2 / v3 / v3.1` (`CalciteV2LM`) compiler covering all SWA stages,
    `mid_mlp`, `byte_chunker` (`enc_mlp`, `decode_from_chunks` incl. `byte_swa`/`Engram`/`BIC`),
    `Woodbury` (`CalciteV2WoodburyLayer` pre/post), `Crystal 1+2` pre/post/readout, and
    `EpistemicSlotMemory`. Defaults to `dynamic=False` (matching static training shapes with
    `use_static_m_cap=True`, generating faster Triton kernels, and avoiding PyTorch 2.11 nightly
    Inductor `tiling_utils.get_pw_red_splits` crashes on small/BIC/Crystal reductions).
    Returns `False` cleanly if Triton/Inductor GPU backend is unavailable.
    """
    if not _has_working_inductor_gpu():
        return False
    # Pre-warm native CUDA extensions outside Dynamo tracing so readiness flags constant-fold
    _load_fast_scans_cuda_ext()
    try:
        from kernels.fast_calcite_swa_chunker import _load_swa_chunker_cuda_ext

        _load_swa_chunker_cuda_ext()
    except Exception:
        pass
    try:
        from kernels.fast_calcite_bic import fast_bic_scan_available

        fast_bic_scan_available()
    except Exception:
        pass
    try:
        import torch._dynamo as _dynamo

        _dynamo.config.cache_size_limit = max(64, int(_dynamo.config.cache_size_limit))
    except Exception:
        pass

    if getattr(model, "mid_mlp", None) is not None:
        model.mid_mlp = torch.compile(model.mid_mlp, mode=mode, dynamic=dynamic)

    swa_stages = getattr(model, "swa_stages", None)
    if swa_stages is not None:
        for stg in swa_stages:
            for i in range(len(stg)):
                stg[i] = torch.compile(stg[i], mode=mode, dynamic=dynamic)
    else:
        for attr in ("cortex_pre", "cortex_post"):
            ml = getattr(model, attr, None)
            if ml is not None:
                for i in range(len(ml)):
                    ml[i] = torch.compile(ml[i], mode=mode, dynamic=dynamic)

    bc = getattr(model, "byte_chunker", None)
    if bc is not None:
        if getattr(bc, "enc_mlp", None) is not None:
            bc.enc_mlp = torch.compile(bc.enc_mlp, mode=mode, dynamic=dynamic)
        if hasattr(bc, "decode_from_chunks"):
            dec_dynamic = bool(dynamic) and (getattr(bc, "byte_induct_cache", None) is None)
            bc.decode_from_chunks = torch.compile(
                bc.decode_from_chunks, mode=mode, dynamic=dec_dynamic
            )

    for wname in ("woodbury", "woodbury_fast"):
        wb = getattr(model, wname, None)
        if _is_fast_woodbury_eligible(wb):
            _make_compiled_woodbury_stages(wb, mode=mode, dynamic=dynamic)

    for cname in ("crystal", "crystal_2"):
        cr = getattr(model, cname, None)
        if cr is not None and isinstance(cr, VoronoiCrystalLayer):
            _make_compiled_crystal_stages(cr, mode=mode, dynamic=dynamic)

    es = getattr(model, "epistemic_slots", None)
    if es is not None:
        model.epistemic_slots = torch.compile(es, mode=mode, dynamic=dynamic)

    return True


def unwrap_compiled_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Returns a clean CPU state_dict with any `torch.compile` `_orig_mod.` prefixes stripped."""
    clean: dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        clean_k = k.replace("._orig_mod.", ".")
        if clean_k.startswith("_orig_mod."):
            clean_k = clean_k[len("_orig_mod.") :]
        clean[clean_k] = v.detach().cpu().clone()
    return clean


__all__ = [
    "clear_fast_qat_cache",
    "compile_calcite_cortex",
    "compile_calcite_v2",
    "configure_fast_ternary_qat",
    "fast_causal_refractory_gate",
    "fast_crystal_forward",
    "fast_diag_birkhoff_crystal_scan",
    "fast_neural_woodbury_chunk_scan",
    "fast_project_birkhoff_to_perm",
    "fast_quantize_ternary_ste",
    "fast_utf8_chunk_span",
    "fast_woodbury_forward",
    "get_fast_qat_cache_stats",
    "get_fast_scans_backend_info",
    "prepare_cuda_build_env",
    "unwrap_compiled_state_dict",
]
