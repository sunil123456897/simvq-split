"""
fast_packed_lut.py — Ускоренный движок физического 2-битного умножения `PackedLUT2BitLinear`
без умножений весов (`T-MAC 81-entry base-3 LUT`):

Три ортогональных бэкенда с единым контрактом `FastPackedLUT2BitLinear(PackedLUT2BitLinear)`:
  1. `cuda_ext` (`torch.utils.cpp_extension.load_inline`, `nvcc`):
     Тайловое разделяемое ядро `BM=32 x BN=64` с `40.5 KB` `smem_lut[4][81][32]` без единого
     конфликта банков и без умножений на тернарные веса (`row_aligned=True` читает `uint32_t` =
     4 байта = 16 весов одной транзакцией и масштабирует 1 раз на группу из 32 весов;
     `row_aligned=False` поддерживает произвольный `in_features`, например `woodbury.out_proj`).
  2. `cpu_ext` (`torch.utils.cpp_extension.load_inline`, C++ OpenMP):
     L1-резидентная 81-элементная LUT-таблица без умножений для одно- и многопоточного CPU-инференса.
  3. `pytorch_vec` (векторизованный PyTorch без внешнего компилятора):
     Отказ от 4 последовательных `torch.cat` и мелких блоков `n_blk=256`; синтез 81-элементной
     таблицы двумя тензорными сложениями `(t0 + t1) + (t2 + t3)` с кэшированным транспонированным
     индексом на устройстве.
"""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
import shutil
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
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

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
    const int64_t n_tot = x.size(0);
    auto out = torch::empty({n_tot, out_features}, x.options());
    if (n_tot == 0 || out_features == 0) {
        return out;
    }

    const dim3 block(32, 8);
    const dim3 grid(
        static_cast<unsigned int>((n_tot + 31) / 32),
        static_cast<unsigned int>((out_features + 63) / 64)
    );
    const float* x_ptr = x.data_ptr<float>();
    const uint8_t* p_ptr = packed_u8.data_ptr<uint8_t>();
    const __half* sc_ptr = reinterpret_cast<const __half*>(scales_fp16.data_ptr<at::Half>());
    const __half* bsc_ptr = reinterpret_cast<const __half*>(byte_scales_fp16.data_ptr<at::Half>());
    const __half* bias_ptr = reinterpret_cast<const __half*>(bias_fp16.data_ptr<at::Half>());
    float* out_ptr = out.data_ptr<float>();

    if (row_aligned) {
        const int64_t g32 = in_features / 32;
        tmac_lut_kernel<true><<<grid, block>>>(
            x_ptr, p_ptr, sc_ptr, bsc_ptr, bias_ptr, out_ptr,
            n_tot, out_features, in_pad, k_bytes, g32, has_bias
        );
    } else {
        tmac_lut_kernel<false><<<grid, block>>>(
            x_ptr, p_ptr, sc_ptr, bsc_ptr, bias_ptr, out_ptr,
            n_tot, out_features, in_pad, k_bytes, 0, has_bias
        );
    }
    return out;
}
"""

_CPU_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

inline float half_to_float_cpu(const at::Half& h) {
    return static_cast<float>(h);
}

inline void build_byte_lut81(const float* xp, float* lut81) {
    const float x0 = xp[0];
    const float x1 = xp[1];
    const float x2 = xp[2];
    const float x3 = xp[3];
    for (int d3 = 0; d3 < 3; ++d3) {
        const float v3 = (d3 == 0 ? -x3 : (d3 == 2 ? x3 : 0.0f));
        for (int d2 = 0; d2 < 3; ++d2) {
            const float v32 = v3 + (d2 == 0 ? -x2 : (d2 == 2 ? x2 : 0.0f));
            const int base9 = (d3 * 3 + d2) * 9;
            const float v0 = v32 - x1;
            lut81[base9 + 0] = v0 - x0;
            lut81[base9 + 1] = v0;
            lut81[base9 + 2] = v0 + x0;
            lut81[base9 + 3] = v32 - x0;
            lut81[base9 + 4] = v32;
            lut81[base9 + 5] = v32 + x0;
            const float v2 = v32 + x1;
            lut81[base9 + 6] = v2 - x0;
            lut81[base9 + 7] = v2;
            lut81[base9 + 8] = v2 + x0;
        }
    }
}

}  // namespace

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
    TORCH_CHECK(!x.is_cuda(), "x must be a CPU tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(packed_u8.is_contiguous(), "packed_u8 must be contiguous");

    TORCH_CHECK(x.device().is_cpu(), "x must be on CPU");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == in_pad && k_bytes * 4 == in_pad, "x shape mismatch");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(packed_u8.device().is_cpu() && packed_u8.scalar_type() == torch::kUInt8 && packed_u8.is_contiguous(), "packed_u8 must be contiguous CPU uint8");
    TORCH_CHECK(packed_u8.numel() >= out_features * k_bytes, "packed_u8 numel too small");
    TORCH_CHECK(scales_fp16.device().is_cpu() && scales_fp16.scalar_type() == torch::kHalf && scales_fp16.is_contiguous(), "scales_fp16 must be contiguous CPU float16");
    TORCH_CHECK(byte_scales_fp16.device().is_cpu() && byte_scales_fp16.scalar_type() == torch::kHalf && byte_scales_fp16.is_contiguous(), "byte_scales_fp16 must be contiguous CPU float16");
    TORCH_CHECK(bias_fp16.device().is_cpu() && bias_fp16.scalar_type() == torch::kHalf && bias_fp16.is_contiguous(), "bias_fp16 must be contiguous CPU float16");
    TORCH_CHECK(!has_bias || bias_fp16.numel() >= out_features, "bias_fp16 numel too small");

    const int64_t n_tot = x.size(0);
    auto out = torch::empty({n_tot, out_features}, x.options());
    if (n_tot == 0 || out_features == 0) {
        return out;
    }

    const float* x_ptr = x.data_ptr<float>();
    const uint8_t* p_ptr = packed_u8.data_ptr<uint8_t>();
    const at::Half* sc_ptr = scales_fp16.data_ptr<at::Half>();
    const at::Half* bsc_ptr = byte_scales_fp16.data_ptr<at::Half>();
    const at::Half* bias_ptr = bias_fp16.data_ptr<at::Half>();
    float* out_ptr = out.data_ptr<float>();

    std::vector<float> all_luts(static_cast<size_t>(k_bytes) * 81, 0.0f);
    std::vector<float> b_scales;
    const int64_t g32 = row_aligned ? (in_features / 32) : 0;
    if (row_aligned) {
        TORCH_CHECK(k_bytes == g32 * 8, "row_aligned requires k_bytes == g32 * 8");
        TORCH_CHECK(scales_fp16.numel() >= out_features * g32, "scales_fp16 numel too small");
    } else {
        TORCH_CHECK(byte_scales_fp16.numel() >= out_features * k_bytes, "byte_scales_fp16 numel too small");
        b_scales.resize(static_cast<size_t>(out_features) * k_bytes);
        for (int64_t i = 0; i < out_features * k_bytes; ++i) {
            b_scales[i] = half_to_float_cpu(bsc_ptr[i]);
        }
    }

    auto clamp80 = [](uint8_t code) -> int64_t {
        return static_cast<int64_t>(code > 80 ? 80 : code);
    };

    for (int64_t m = 0; m < n_tot; ++m) {
        const float* xm = x_ptr + m * in_pad;
        for (int64_t b = 0; b < k_bytes; ++b) {
            build_byte_lut81(xm + b * 4, all_luts.data() + b * 81);
        }
        float* ym = out_ptr + m * out_features;

        #pragma omp parallel for schedule(static) if(out_features >= 128)
        for (int64_t o = 0; o < out_features; ++o) {
            const uint8_t* po = p_ptr + o * k_bytes;
            float acc = 0.0f;
            if (row_aligned) {
                const at::Half* so = sc_ptr + o * g32;
                for (int64_t g = 0; g < g32; ++g) {
                    const int64_t bb = g * 8;
                    const float* lb = all_luts.data() + bb * 81;
                    float gs = lb[0 * 81 + clamp80(po[bb + 0])]
                             + lb[1 * 81 + clamp80(po[bb + 1])]
                             + lb[2 * 81 + clamp80(po[bb + 2])]
                             + lb[3 * 81 + clamp80(po[bb + 3])]
                             + lb[4 * 81 + clamp80(po[bb + 4])]
                             + lb[5 * 81 + clamp80(po[bb + 5])]
                             + lb[6 * 81 + clamp80(po[bb + 6])]
                             + lb[7 * 81 + clamp80(po[bb + 7])];
                    acc += gs * half_to_float_cpu(so[g]);
                }
            } else {
                const float* bso = b_scales.data() + o * k_bytes;
                for (int64_t b = 0; b < k_bytes; ++b) {
                    acc += all_luts[b * 81 + clamp80(po[b])] * bso[b];
                }
            }
            if (has_bias) {
                acc += half_to_float_cpu(bias_ptr[o]);
            }
            ym[o] = acc;
        }
    }
    return out;
}
"""


_CUDA_BUILD_ERROR: str | None = None
_CPU_BUILD_ERROR: str | None = None


def _has_python_headers() -> bool:
    inc = sysconfig.get_path("include")
    return bool(inc) and (Path(inc) / "Python.h").is_file()


@lru_cache(maxsize=1)
def _load_cuda_ext():
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

        build_dir = Path(tempfile.gettempdir()) / "calcite5_tmac_cuda_v1"
        build_dir.mkdir(parents=True, exist_ok=True)
        mod = load_inline(
            name="calcite5_tmac_lut_cuda_ext",
            cpp_sources=[_CUDA_CPP_SOURCE],
            cuda_sources=[_CUDA_KERNEL_SOURCE],
            functions=["tmac_lut_cuda_forward"],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            build_directory=str(build_dir),
            verbose=False,
        )
        _CUDA_BUILD_ERROR = None
        return mod
    except Exception as exc:
        _CUDA_BUILD_ERROR = f"{type(exc).__name__}: {exc}"
        return None


@lru_cache(maxsize=1)
def _load_cpu_ext():
    global _CPU_BUILD_ERROR
    cxx = shutil.which("g++") or shutil.which("clang++") or shutil.which("cl")
    if cxx is None:
        _CPU_BUILD_ERROR = "No C++ compiler (g++/clang++/cl) in PATH"
        return None
    if not _has_python_headers():
        _CPU_BUILD_ERROR = "Python.h dev headers missing"
        return None
    try:
        from torch.utils.cpp_extension import load_inline

        build_dir = Path(tempfile.gettempdir()) / "calcite5_tmac_cpu_v1"
        build_dir.mkdir(parents=True, exist_ok=True)
        cflags = ["/O2", "/openmp"] if Path(cxx).name.lower().startswith("cl") else ["-O3", "-fopenmp"]
        ldflags = [] if Path(cxx).name.lower().startswith("cl") else ["-fopenmp"]
        mod = load_inline(
            name="calcite5_tmac_lut_cpu_ext",
            cpp_sources=[_CPU_CPP_SOURCE],
            functions=["tmac_lut_cpu_forward"],
            extra_cflags=cflags,
            extra_ldflags=ldflags,
            build_directory=str(build_dir),
            verbose=False,
        )
        _CPU_BUILD_ERROR = None
        return mod
    except Exception as exc:
        _CPU_BUILD_ERROR = f"{type(exc).__name__}: {exc}"
        return None


def get_lut_backend_info() -> dict[str, object]:
    """Сообщает статус нативных C++/CUDA бэкендов и активный фоллбек."""
    cuda_mod = _load_cuda_ext()
    cpu_mod = _load_cpu_ext()
    return {
        "cuda_ext_available": cuda_mod is not None,
        "cuda_build_error": _CUDA_BUILD_ERROR,
        "cpu_ext_available": cpu_mod is not None,
        "cpu_build_error": _CPU_BUILD_ERROR,
        "nvcc_path": shutil.which("nvcc"),
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

    @classmethod
    def from_packed(
        cls,
        ref: PackedLUT2BitLinear,
        backend: str = "auto",
    ) -> "FastPackedLUT2BitLinear":
        return cls(
            in_features=ref.in_features,
            out_features=ref.out_features,
            packed_u8=ref.packed_u8.detach(),
            raw_scales_fp16=ref.raw_scales_fp16.detach(),
            bias_fp16=ref.bias_fp16.detach() if ref.bias_fp16 is not None else None,
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

    def _forward_pytorch_vec(self, x_flat: torch.Tensor) -> torch.Tensor:
        n_tot = x_flat.size(0)
        k_bytes = self.k_bytes
        dev = x_flat.device
        if self._vec_cache_dev != dev or self._vec_idx_cache is None or self._vec_scales_cache is None:
            if self.row_aligned:
                self._vec_scales_cache = self.scales_fp16.t().float().unsqueeze(-1).contiguous()
                self._vec_idx_cache = (
                    self.packed_u8.view(self.out_features, k_bytes).t().long().unsqueeze(-1).contiguous()
                )
            else:
                b_sc = self._byte_scales_fp16
                if b_sc is None:
                    b_sc = _compute_unaligned_byte_scales(
                        self.raw_scales_fp16,
                        self.out_features,
                        self.in_features,
                        k_bytes,
                        self.g_size,
                    )
                self._vec_scales_cache = b_sc.t().float().unsqueeze(-1).contiguous()
                self._vec_idx_cache = self.packed_u8.t().long().unsqueeze(-1).contiguous()
            self._vec_cache_dev = dev

        idx_base = self._vec_idx_cache
        scales_cached = self._vec_scales_cache
        g32 = self.in_features // 32 if self.row_aligned else 0
        n_blk = 2048
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
        has_bias = self.bias_fp16 is not None
        bias_t = self.bias_fp16.to(torch.float16).contiguous() if has_bias else sc

        if backend == "cuda_ext":
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
        else:
            y = _load_cpu_ext().tmac_lut_cpu_forward(
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
