"""
fast_calcite_bic.py — Shard-Parallel SMEM CUDA Kernel for `Calcite v3.1` `ByteInductionCache`
(`calcite_v2/byte_induct_cache.py`), with zero-graph-break `torch.library.custom_op` +
`register_fake` + `register_autograd` and exact PyTorch blocked dual-form fallback.

Algorithmic & CUDA Design (CODING_STANDARDS.md Section 3b compliant):
- Slots `s in [0, n_slots)` in `ByteInductionCache` are 100% mutually independent across time.
- Partition `n_slots` into disjoint 32-slot shards (`SHARD_SLOTS = 32`, `slot >> 5` = shard ID).
- Each CUDA block launches `N_WARPS = 4` warps (`dim3(32, 4)` = 128 threads); each warp exclusively
  owns one 32-slot shard in static SMEM (`sh_bank[4][32][32] = 16 KB` + `256`-step index/gate tiles
  `3 KB` = **`19.0 KB <= 48 KB`** static SMEM, zero SMEM bank conflicts).
- Because each timestep `t` writes one slot `sw[t]` and reads one slot `sq[t]`, the shard check
  `(slot >> 5) == my_shard` is **100% warp-uniform** (no divergent `__syncthreads()` or shuffles)
  and **100% race-free** (zero atomics!).
- Exact `O(B * T * d_u)` linear time and VRAM: eliminates all `O(B * Bc^2)` dense collision
  matrices (`~144 MB` at `B=64, T=512`), skips bank writes when `bt[t] == 0.0f` while STILL
  propagating exact STE `dL/d(bt[t]) = <g_bank[sw[t]], u[t] - u_old[t]>` in backward!
"""

from __future__ import annotations

from pathlib import Path
import sys
import sysconfig
import tempfile

import torch

from kernels.fast_calcite_scans import prepare_cuda_build_env

_CUDA_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> bic_scan_fwd_cuda_impl(
    torch::Tensor sq_i32,
    torch::Tensor sw_i32,
    torch::Tensor bt_f32,
    torch::Tensor u_f32,
    int64_t n_slots,
    bool save_old
);

std::vector<torch::Tensor> bic_scan_bwd_cuda_impl(
    torch::Tensor sq_i32,
    torch::Tensor sw_i32,
    torch::Tensor bt_f32,
    torch::Tensor u_f32,
    torch::Tensor u_old_f32,
    torch::Tensor grad_out_f32,
    int64_t n_slots
);

std::vector<torch::Tensor> bic_scan_fwd_cuda(
    torch::Tensor sq_i32,
    torch::Tensor sw_i32,
    torch::Tensor bt_f32,
    torch::Tensor u_f32,
    int64_t n_slots,
    bool save_old
) {
    return bic_scan_fwd_cuda_impl(sq_i32, sw_i32, bt_f32, u_f32, n_slots, save_old);
}

std::vector<torch::Tensor> bic_scan_bwd_cuda(
    torch::Tensor sq_i32,
    torch::Tensor sw_i32,
    torch::Tensor bt_f32,
    torch::Tensor u_f32,
    torch::Tensor u_old_f32,
    torch::Tensor grad_out_f32,
    int64_t n_slots
) {
    return bic_scan_bwd_cuda_impl(sq_i32, sw_i32, bt_f32, u_f32, u_old_f32, grad_out_f32, n_slots);
}
"""

_CUDA_KERNEL_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace {

constexpr int BIC_N_WARPS = 4;
constexpr int BIC_SHARD_SLOTS = 32;
constexpr int BIC_TILE_T = 256;

__device__ __forceinline__ float bic_warp_sum32(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    }
    return v;
}

__device__ __forceinline__ int32_t clamp_slot(int32_t s, int32_t n_slots) {
    if (s < 0) return 0;
    if (s >= n_slots) return n_slots - 1;
    return s;
}

// ---------------------------------------------------------------------------
// Shard-Parallel SMEM Forward (`d_u <= 32`, Static SMEM = 19.0 KB <= 48 KB)
// Exact write-then-read recurrence:
//   1. u_old[t] = bank[sw[t]] (if save_old)
//   2. bank[sw[t]] = (1 - bt[t]) * bank[sw[t]] + bt[t] * u[t]
//   3. u_read[t] = bank[sq[t]]
// ---------------------------------------------------------------------------
__global__ void bic_scan_fwd_kernel(
    const int32_t* __restrict__ sq_ptr,   // (BK, T)
    const int32_t* __restrict__ sw_ptr,   // (BK, T)
    const float* __restrict__ bt_ptr,     // (BK, T)
    const float* __restrict__ u_ptr,      // (BK, T, d_u)
    float* __restrict__ out_ptr,          // (BK, T, d_u)
    float* __restrict__ u_old_ptr,        // (BK, T, d_u) or nullptr
    int64_t T,
    int d_u,
    int32_t n_slots,
    int32_t n_shards
) {
    __shared__ float sh_bank[BIC_N_WARPS][BIC_SHARD_SLOTS][32];
    __shared__ int32_t sh_sq[BIC_TILE_T];
    __shared__ int32_t sh_sw[BIC_TILE_T];
    __shared__ float sh_bt[BIC_TILE_T];

    const int lane = static_cast<int>(threadIdx.x);     // 0..31
    const int warp_id = static_cast<int>(threadIdx.y);  // 0..3
    const int tid = (warp_id << 5) | lane;              // 0..127
    const int64_t bk = static_cast<int64_t>(blockIdx.x);
    const int32_t my_shard = static_cast<int32_t>(blockIdx.y) * BIC_N_WARPS + warp_id;
    const bool shard_valid = (my_shard < n_shards);
    const bool save_old = (u_old_ptr != nullptr);
    const bool lane_valid = (lane < d_u);

    #pragma unroll
    for (int s = 0; s < BIC_SHARD_SLOTS; ++s) {
        sh_bank[warp_id][s][lane] = 0.0f;
    }

    const int64_t seq_base = bk * T;
    for (int64_t t0 = 0; t0 < T; t0 += BIC_TILE_T) {
        const int tile_len = static_cast<int>((t0 + BIC_TILE_T < T) ? BIC_TILE_T : (T - t0));
        for (int i = tid; i < tile_len; i += (BIC_N_WARPS * 32)) {
            const int64_t gidx = seq_base + t0 + i;
            sh_sq[i] = clamp_slot(sq_ptr[gidx], n_slots);
            sh_sw[i] = clamp_slot(sw_ptr[gidx], n_slots);
            sh_bt[i] = bt_ptr[gidx];
        }
        __syncthreads();

        if (shard_valid) {
            for (int dt = 0; dt < tile_len; ++dt) {
                const int32_t w_slot = sh_sw[dt];
                const int32_t q_slot = sh_sq[dt];
                const int64_t t = t0 + dt;

                if ((w_slot >> 5) == my_shard) {
                    const int wl = w_slot & 31;
                    const float old_v = sh_bank[warp_id][wl][lane];
                    const int64_t off = (seq_base + t) * d_u + lane;
                    if (save_old && lane_valid) {
                        u_old_ptr[off] = old_v;
                    }
                    const float b_val = sh_bt[dt];
                    if (b_val != 0.0f && lane_valid) {
                        sh_bank[warp_id][wl][lane] = (1.0f - b_val) * old_v + b_val * u_ptr[off];
                    }
                }

                if ((q_slot >> 5) == my_shard) {
                    const int ql = q_slot & 31;
                    if (lane_valid) {
                        out_ptr[(seq_base + t) * d_u + lane] = sh_bank[warp_id][ql][lane];
                    }
                }
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Shard-Parallel SMEM Analytical Backward (`d_u <= 32`, Static SMEM = 19.0 KB)
// Exact reverse-mode adjoint (preserving live STE dL/d(bt) even when bt == 0):
//   1. g_bank[sq[t]] += grad_out[t]
//   2. g_u[t]  = bt[t] * g_bank[sw[t]]
//      g_bt[t] = <g_bank[sw[t]], u[t] - u_old[t]>
//      g_bank[sw[t]] *= (1 - bt[t])
// ---------------------------------------------------------------------------
__global__ void bic_scan_bwd_kernel(
    const int32_t* __restrict__ sq_ptr,     // (BK, T)
    const int32_t* __restrict__ sw_ptr,     // (BK, T)
    const float* __restrict__ bt_ptr,       // (BK, T)
    const float* __restrict__ u_ptr,        // (BK, T, d_u)
    const float* __restrict__ u_old_ptr,    // (BK, T, d_u)
    const float* __restrict__ grad_out_ptr, // (BK, T, d_u)
    float* __restrict__ g_bt_ptr,           // (BK, T)
    float* __restrict__ g_u_ptr,            // (BK, T, d_u)
    int64_t T,
    int d_u,
    int32_t n_slots,
    int32_t n_shards
) {
    __shared__ float sh_gbank[BIC_N_WARPS][BIC_SHARD_SLOTS][32];
    __shared__ int32_t sh_sq[BIC_TILE_T];
    __shared__ int32_t sh_sw[BIC_TILE_T];
    __shared__ float sh_bt[BIC_TILE_T];

    const int lane = static_cast<int>(threadIdx.x);     // 0..31
    const int warp_id = static_cast<int>(threadIdx.y);  // 0..3
    const int tid = (warp_id << 5) | lane;              // 0..127
    const int64_t bk = static_cast<int64_t>(blockIdx.x);
    const int32_t my_shard = static_cast<int32_t>(blockIdx.y) * BIC_N_WARPS + warp_id;
    const bool shard_valid = (my_shard < n_shards);
    const bool lane_valid = (lane < d_u);

    #pragma unroll
    for (int s = 0; s < BIC_SHARD_SLOTS; ++s) {
        sh_gbank[warp_id][s][lane] = 0.0f;
    }

    const int64_t seq_base = bk * T;
    const int64_t n_tiles = (T + BIC_TILE_T - 1) / BIC_TILE_T;
    for (int64_t ti = n_tiles - 1; ti >= 0; --ti) {
        const int64_t t0 = ti * BIC_TILE_T;
        const int tile_len = static_cast<int>((t0 + BIC_TILE_T < T) ? BIC_TILE_T : (T - t0));
        for (int i = tid; i < tile_len; i += (BIC_N_WARPS * 32)) {
            const int64_t gidx = seq_base + t0 + i;
            sh_sq[i] = clamp_slot(sq_ptr[gidx], n_slots);
            sh_sw[i] = clamp_slot(sw_ptr[gidx], n_slots);
            sh_bt[i] = bt_ptr[gidx];
        }
        __syncthreads();

        if (shard_valid) {
            for (int dt = tile_len - 1; dt >= 0; --dt) {
                const int32_t q_slot = sh_sq[dt];
                const int32_t w_slot = sh_sw[dt];
                const int64_t t = t0 + dt;

                if ((q_slot >> 5) == my_shard) {
                    const int ql = q_slot & 31;
                    if (lane_valid) {
                        sh_gbank[warp_id][ql][lane] += grad_out_ptr[(seq_base + t) * d_u + lane];
                    }
                }

                if ((w_slot >> 5) == my_shard) {
                    const int wl = w_slot & 31;
                    const float gb = sh_gbank[warp_id][wl][lane];
                    const float b_val = sh_bt[dt];
                    const int64_t off = (seq_base + t) * d_u + lane;
                    float term = 0.0f;
                    if (lane_valid) {
                        g_u_ptr[off] = b_val * gb;
                        term = gb * (u_ptr[off] - u_old_ptr[off]);
                    }
                    // Warp-uniform reduction across all 32 lanes
                    const float dot = bic_warp_sum32(term);
                    if (lane == 0) {
                        g_bt_ptr[seq_base + t] = dot;
                    }
                    if (b_val != 0.0f && lane_valid) {
                        sh_gbank[warp_id][wl][lane] = (1.0f - b_val) * gb;
                    }
                }
            }
        }
        __syncthreads();
    }
}

}  // namespace

std::vector<torch::Tensor> bic_scan_fwd_cuda_impl(
    torch::Tensor sq_i32,
    torch::Tensor sw_i32,
    torch::Tensor bt_f32,
    torch::Tensor u_f32,
    int64_t n_slots,
    bool save_old
) {
    c10::cuda::CUDAGuard device_guard(u_f32.device());
    auto sq_c = sq_i32.to(torch::kInt32).contiguous();
    auto sw_c = sw_i32.to(torch::kInt32).contiguous();
    auto bt_c = bt_f32.to(torch::kFloat32).contiguous();
    auto u_c = u_f32.to(torch::kFloat32).contiguous();

    TORCH_CHECK(sq_c.dim() == 2 && sw_c.dim() == 2 && bt_c.dim() == 2 && u_c.dim() == 3,
                "bic_scan_fwd_cuda expects sq/sw/bt (BK, T) and u (BK, T, d_u)");
    const int64_t BK = u_c.size(0);
    const int64_t T = u_c.size(1);
    const int d_u = static_cast<int>(u_c.size(2));
    TORCH_CHECK(d_u > 0 && d_u <= 32, "bic_scan_fwd_cuda requires 0 < d_u <= 32");
    TORCH_CHECK(n_slots > 0, "bic_scan_fwd_cuda requires n_slots > 0");

    auto out = torch::empty_like(u_c);
    auto u_old = save_old ? torch::empty_like(u_c) : torch::empty({0}, u_c.options());
    if (BK == 0 || T == 0) {
        return {out, u_old};
    }

    const int32_t s_i32 = static_cast<int32_t>(n_slots);
    const int32_t n_shards = (s_i32 + BIC_SHARD_SLOTS - 1) / BIC_SHARD_SLOTS;
    const dim3 block(32, BIC_N_WARPS);
    const dim3 grid(
        static_cast<unsigned int>(BK),
        static_cast<unsigned int>((n_shards + BIC_N_WARPS - 1) / BIC_N_WARPS)
    );
    auto stream = c10::cuda::getCurrentCUDAStream();
    bic_scan_fwd_kernel<<<grid, block, 0, stream>>>(
        sq_c.data_ptr<int32_t>(),
        sw_c.data_ptr<int32_t>(),
        bt_c.data_ptr<float>(),
        u_c.data_ptr<float>(),
        out.data_ptr<float>(),
        save_old ? u_old.data_ptr<float>() : nullptr,
        T,
        d_u,
        s_i32,
        n_shards
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, u_old};
}

std::vector<torch::Tensor> bic_scan_bwd_cuda_impl(
    torch::Tensor sq_i32,
    torch::Tensor sw_i32,
    torch::Tensor bt_f32,
    torch::Tensor u_f32,
    torch::Tensor u_old_f32,
    torch::Tensor grad_out_f32,
    int64_t n_slots
) {
    c10::cuda::CUDAGuard device_guard(u_f32.device());
    auto sq_c = sq_i32.to(torch::kInt32).contiguous();
    auto sw_c = sw_i32.to(torch::kInt32).contiguous();
    auto bt_c = bt_f32.to(torch::kFloat32).contiguous();
    auto u_c = u_f32.to(torch::kFloat32).contiguous();
    auto uo_c = u_old_f32.to(torch::kFloat32).contiguous();
    auto go_c = grad_out_f32.to(torch::kFloat32).contiguous();

    const int64_t BK = u_c.size(0);
    const int64_t T = u_c.size(1);
    const int d_u = static_cast<int>(u_c.size(2));
    TORCH_CHECK(d_u > 0 && d_u <= 32, "bic_scan_bwd_cuda requires 0 < d_u <= 32");
    TORCH_CHECK(n_slots > 0, "bic_scan_bwd_cuda requires n_slots > 0");

    auto g_bt = torch::empty_like(bt_c);
    auto g_u = torch::empty_like(u_c);
    if (BK == 0 || T == 0) {
        return {g_bt, g_u};
    }

    const int32_t s_i32 = static_cast<int32_t>(n_slots);
    const int32_t n_shards = (s_i32 + BIC_SHARD_SLOTS - 1) / BIC_SHARD_SLOTS;
    const dim3 block(32, BIC_N_WARPS);
    const dim3 grid(
        static_cast<unsigned int>(BK),
        static_cast<unsigned int>((n_shards + BIC_N_WARPS - 1) / BIC_N_WARPS)
    );
    auto stream = c10::cuda::getCurrentCUDAStream();
    bic_scan_bwd_kernel<<<grid, block, 0, stream>>>(
        sq_c.data_ptr<int32_t>(),
        sw_c.data_ptr<int32_t>(),
        bt_c.data_ptr<float>(),
        u_c.data_ptr<float>(),
        uo_c.data_ptr<float>(),
        go_c.data_ptr<float>(),
        g_bt.data_ptr<float>(),
        g_u.data_ptr<float>(),
        T,
        d_u,
        s_i32,
        n_shards
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {g_bt, g_u};
}
"""

_CUDA_BUILD_ERROR: str | None = None


def _has_python_headers() -> bool:
    inc = sysconfig.get_path("include")
    return bool(inc) and (Path(inc) / "Python.h").is_file()


_BIC_EXT_UNSET = object()
_BIC_EXT_CACHE: object = _BIC_EXT_UNSET
_BIC_EXT_READY: bool | None = None


def _load_bic_cuda_ext():
    global _CUDA_BUILD_ERROR, _BIC_EXT_CACHE, _BIC_EXT_READY
    if _BIC_EXT_CACHE is not _BIC_EXT_UNSET:
        return _BIC_EXT_CACHE
    if not torch.cuda.is_available():
        _CUDA_BUILD_ERROR = "CUDA device unavailable"
        _BIC_EXT_CACHE = None
        _BIC_EXT_READY = False
        return None
    env_info = prepare_cuda_build_env()
    if env_info["nvcc_path"] is None:
        _CUDA_BUILD_ERROR = "nvcc not found in PATH or CUDA/pip directories"
        _BIC_EXT_CACHE = None
        _BIC_EXT_READY = False
        return None
    if sys.platform == "win32" and env_info["cxx_path"] is None:
        _CUDA_BUILD_ERROR = "MSVC cl.exe not found (Visual Studio Build Tools missing)"
        _BIC_EXT_CACHE = None
        _BIC_EXT_READY = False
        return None
    if not _has_python_headers():
        _CUDA_BUILD_ERROR = "Python.h dev headers missing"
        _BIC_EXT_CACHE = None
        _BIC_EXT_READY = False
        return None
    try:
        from torch.utils.cpp_extension import load_inline

        build_dir = Path(tempfile.gettempdir()) / "calcite5_fast_bic_cuda_v1"
        build_dir.mkdir(parents=True, exist_ok=True)
        mod = load_inline(
            name="calcite5_fast_bic_cuda_ext_v1",
            cpp_sources=[_CUDA_CPP_SOURCE],
            cuda_sources=[_CUDA_KERNEL_SOURCE],
            functions=["bic_scan_fwd_cuda", "bic_scan_bwd_cuda"],
            extra_cflags=list(env_info["extra_cflags"]),
            extra_cuda_cflags=list(env_info["extra_cuda_cflags"]),
            extra_ldflags=list(env_info["extra_ldflags"]) or None,
            extra_include_paths=list(env_info["include_paths"]) or None,
            build_directory=str(build_dir),
            with_cuda=True,
            verbose=False,
        )
        _CUDA_BUILD_ERROR = None
        _BIC_EXT_CACHE = mod
        _BIC_EXT_READY = True
        return mod
    except Exception as exc:
        _CUDA_BUILD_ERROR = f"{type(exc).__name__}: {exc}"
        _BIC_EXT_CACHE = None
        _BIC_EXT_READY = False
        return None


def fast_bic_scan_available() -> bool:
    """Returns True when the native Shard-Parallel SMEM BIC CUDA kernel is loaded."""
    if _BIC_EXT_READY is not None:
        return _BIC_EXT_READY
    return _load_bic_cuda_ext() is not None


def get_fast_bic_backend_info() -> dict[str, object]:
    env_info = prepare_cuda_build_env()
    ext = _load_bic_cuda_ext()
    return {
        "cuda_available": bool(torch.cuda.is_available()),
        "nvcc_found": env_info["nvcc_path"] is not None,
        "nvcc_path": env_info["nvcc_path"],
        "cxx_path": env_info["cxx_path"],
        "cuda_ext_available": ext is not None,
        "cuda_build_error": _CUDA_BUILD_ERROR,
        "custom_op_registered": _HAS_CUSTOM_OP,
    }


def _bic_scan_pytorch_ref(
    sq: torch.Tensor,
    sw: torch.Tensor,
    bt: torch.Tensor,
    u_all: torch.Tensor,
    n_slots: int,
    block_size: int = 256,
) -> torch.Tensor:
    """Exact blocked dual-form PyTorch fallback matching `ByteInductionCache.forward()`."""
    B, T, K_sc, D_u = u_all.shape
    BK = B * K_sc
    S = int(n_slots)
    dev = u_all.device
    Bc = max(32, int(block_size))
    n_blocks = (T + Bc - 1) // Bc

    sq_bk = sq.permute(0, 2, 1).reshape(BK, T).long()
    sw_bk = sw.permute(0, 2, 1).reshape(BK, T).long()
    bt_bk = bt.float().permute(0, 2, 1).reshape(BK, T)
    u_bk = u_all.float().permute(0, 2, 1, 3).reshape(BK, T, D_u)
    log_k_all = torch.log((1.0 - bt_bk).clamp(min=1e-5))

    zero_f = torch.zeros((), device=dev, dtype=torch.float32)
    tril_full = torch.tril(torch.ones((min(Bc, T), min(Bc, T)), device=dev, dtype=torch.bool))
    triu_full = (
        torch.triu(torch.ones((min(Bc, T), min(Bc, T)), device=dev, dtype=torch.bool), diagonal=1)
        if n_blocks > 1
        else None
    )

    with torch.amp.autocast(dev.type, enabled=False):
        bank = torch.zeros((BK, S, D_u), device=dev, dtype=torch.float32)
        outs: list[torch.Tensor] = []
        for bi in range(n_blocks):
            st = bi * Bc
            en = min(T, st + Bc)
            Lb = en - st

            sq_b = sq_bk[:, st:en]
            sw_b = sw_bk[:, st:en]
            bt_b = bt_bk[:, st:en]
            u_b = u_bk[:, st:en]
            log_k = log_k_all[:, st:en]

            tril = tril_full[:Lb, :Lb]
            same = (sq_b.unsqueeze(2) == sw_b.unsqueeze(1)) & tril.unsqueeze(0)
            lk_m = torch.where(same, log_k.unsqueeze(1), zero_f)
            cs = lk_m.cumsum(dim=-1)
            cs_end = cs[:, :, -1:]

            w_mat = torch.where(
                same,
                bt_b.unsqueeze(1) * torch.exp((cs_end - cs).clamp(min=-30.0)),
                zero_f,
            )
            u_blk = torch.bmm(w_mat, u_b)
            if bi > 0:
                prev_g = torch.gather(bank, 1, sq_b.unsqueeze(-1).expand(BK, Lb, D_u))
                u_read_b = torch.exp(cs_end.clamp(min=-30.0)) * prev_g + u_blk
            else:
                u_read_b = u_blk
            outs.append(u_read_b)

            if bi < n_blocks - 1 and triu_full is not None:
                triu_s = triu_full[:Lb, :Lb]
                same_after = (sw_b.unsqueeze(2) == sw_b.unsqueeze(1)) & triu_s.unsqueeze(0)
                log_surv = torch.where(same_after, log_k.unsqueeze(1), zero_f).sum(dim=-1)
                contrib = (bt_b * torch.exp(log_surv.clamp(min=-30.0))).unsqueeze(-1) * u_b
                slot_lk = torch.zeros((BK, S), device=dev, dtype=torch.float32)
                slot_lk.scatter_add_(1, sw_b, log_k)
                delta_b = torch.zeros((BK, S, D_u), device=dev, dtype=torch.float32)
                delta_b.scatter_add_(1, sw_b.unsqueeze(-1).expand(BK, Lb, D_u), contrib)
                bank = bank * torch.exp(slot_lk.clamp(min=-30.0)).unsqueeze(-1) + delta_b

        out_bk = outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)
        return out_bk.view(B, K_sc, T, D_u).permute(0, 2, 1, 3)


_HAS_CUSTOM_OP = hasattr(torch, "library") and hasattr(torch.library, "custom_op")

if _HAS_CUSTOM_OP:

    @torch.library.custom_op("calcite_bic::bic_scan_fwd", mutates_args=())
    def _bic_scan_fwd_op(
        sq_i32: torch.Tensor,
        sw_i32: torch.Tensor,
        bt_f32: torch.Tensor,
        u_f32: torch.Tensor,
        n_slots: int,
        save_old: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ext = _load_bic_cuda_ext()
        assert ext is not None, f"calcite_bic CUDA ext unavailable: {_CUDA_BUILD_ERROR}"
        out, u_old = ext.bic_scan_fwd_cuda(
            sq_i32, sw_i32, bt_f32, u_f32, int(n_slots), bool(save_old)
        )
        return out, u_old

    @_bic_scan_fwd_op.register_fake
    def _bic_scan_fwd_fake(
        sq_i32: torch.Tensor,
        sw_i32: torch.Tensor,
        bt_f32: torch.Tensor,
        u_f32: torch.Tensor,
        n_slots: int,
        save_old: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del sq_i32, sw_i32, bt_f32, n_slots
        u_old = (
            torch.empty_like(u_f32)
            if save_old
            else torch.empty((0,), device=u_f32.device, dtype=u_f32.dtype)
        )
        return torch.empty_like(u_f32), u_old

    @torch.library.custom_op("calcite_bic::bic_scan_bwd", mutates_args=())
    def _bic_scan_bwd_op(
        sq_i32: torch.Tensor,
        sw_i32: torch.Tensor,
        bt_f32: torch.Tensor,
        u_f32: torch.Tensor,
        u_old_f32: torch.Tensor,
        grad_out_f32: torch.Tensor,
        n_slots: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ext = _load_bic_cuda_ext()
        assert ext is not None, f"calcite_bic CUDA ext unavailable: {_CUDA_BUILD_ERROR}"
        g_bt, g_u = ext.bic_scan_bwd_cuda(
            sq_i32, sw_i32, bt_f32, u_f32, u_old_f32, grad_out_f32, int(n_slots)
        )
        return g_bt, g_u

    @_bic_scan_bwd_op.register_fake
    def _bic_scan_bwd_fake(
        sq_i32: torch.Tensor,
        sw_i32: torch.Tensor,
        bt_f32: torch.Tensor,
        u_f32: torch.Tensor,
        u_old_f32: torch.Tensor,
        grad_out_f32: torch.Tensor,
        n_slots: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del sq_i32, sw_i32, u_old_f32, grad_out_f32, n_slots
        return torch.empty_like(bt_f32), torch.empty_like(u_f32)

    def _bic_setup_context(ctx, inputs, output) -> None:
        sq_i32, sw_i32, bt_f32, u_f32, n_slots, _ = inputs
        _, u_old = output
        ctx.save_for_backward(sq_i32, sw_i32, bt_f32, u_f32, u_old)
        ctx.n_slots = int(n_slots)

    def _bic_backward_bridge(
        ctx, grad_out: torch.Tensor | None, grad_u_old: torch.Tensor | None
    ):
        del grad_u_old
        if grad_out is None:
            return None, None, None, None, None, None
        sq_i32, sw_i32, bt_f32, u_f32, u_old = ctx.saved_tensors
        g_bt, g_u = _bic_scan_bwd_op(
            sq_i32, sw_i32, bt_f32, u_f32, u_old, grad_out.float().contiguous(), ctx.n_slots
        )
        return None, None, g_bt.to(bt_f32.dtype), g_u.to(u_f32.dtype), None, None

    _bic_scan_fwd_op.register_autograd(
        _bic_backward_bridge, setup_context=_bic_setup_context
    )


class _FastBICScanFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        sq_i32: torch.Tensor,
        sw_i32: torch.Tensor,
        bt_f32: torch.Tensor,
        u_f32: torch.Tensor,
        n_slots: int,
        save_old: bool,
    ) -> torch.Tensor:
        ext = _load_bic_cuda_ext()
        assert ext is not None
        out, u_old = ext.bic_scan_fwd_cuda(
            sq_i32, sw_i32, bt_f32, u_f32, int(n_slots), bool(save_old)
        )
        ctx.save_for_backward(sq_i32, sw_i32, bt_f32, u_f32, u_old)
        ctx.n_slots = int(n_slots)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        sq_i32, sw_i32, bt_f32, u_f32, u_old = ctx.saved_tensors
        ext = _load_bic_cuda_ext()
        assert ext is not None
        g_bt, g_u = ext.bic_scan_bwd_cuda(
            sq_i32, sw_i32, bt_f32, u_f32, u_old, grad_out.float().contiguous(), ctx.n_slots
        )
        return None, None, g_bt.to(bt_f32.dtype), g_u.to(u_f32.dtype), None, None


def fast_bic_scan_cuda(
    sq: torch.Tensor,
    sw: torch.Tensor,
    bt: torch.Tensor,
    u_all: torch.Tensor,
    n_slots: int,
) -> torch.Tensor:
    """
    Exact Shard-Parallel SMEM CUDA scan for `ByteInductionCache` (`O(B * T * d_u)`, zero `Bc^2`
    matrices, `19.0 KB` static SMEM) with exact PyTorch blocked fallback on CPU / no-nvcc / `d_u > 32`.

    Args:
        sq: `(B, T, K_sc)` int64 query hash slots in `[0, n_slots)`.
        sw: `(B, T, K_sc)` int64 write hash slots in `[0, n_slots)`.
        bt: `(B, T, K_sc)` float write gate `beta_eff` in `[0, 1)`.
        u_all: `(B, T, K_sc, d_u)` float write payloads `[k_sig, v, 1.0]`.
        n_slots: Number of slots per scale `S > 0`.

    Returns:
        `u_read`: `(B, T, K_sc, d_u)` float32 read slot payloads after causal write-then-read.
    """
    B, T, K_sc, D_u = u_all.shape
    S = int(n_slots)
    if (
        u_all.is_cuda
        and 0 < D_u <= 32
        and S > 0
        and _load_bic_cuda_ext() is not None
    ):
        BK = B * K_sc
        save_old = bool(torch.is_grad_enabled() and (bt.requires_grad or u_all.requires_grad))
        with torch.amp.autocast(u_all.device.type, enabled=False):
            sq_c = sq.detach().permute(0, 2, 1).reshape(BK, T).to(torch.int32).contiguous()
            sw_c = sw.detach().permute(0, 2, 1).reshape(BK, T).to(torch.int32).contiguous()
            bt_c = bt.float().permute(0, 2, 1).reshape(BK, T).contiguous()
            u_c = u_all.float().permute(0, 2, 1, 3).reshape(BK, T, D_u).contiguous()
            if _HAS_CUSTOM_OP:
                out_c, _ = _bic_scan_fwd_op(sq_c, sw_c, bt_c, u_c, S, save_old)
            else:
                out_c = _FastBICScanFn.apply(sq_c, sw_c, bt_c, u_c, S, save_old)
            return out_c.view(B, K_sc, T, D_u).permute(0, 2, 1, 3)

    return _bic_scan_pytorch_ref(sq, sw, bt, u_all, S)


__all__ = [
    "fast_bic_scan_available",
    "fast_bic_scan_cuda",
    "get_fast_bic_backend_info",
]
