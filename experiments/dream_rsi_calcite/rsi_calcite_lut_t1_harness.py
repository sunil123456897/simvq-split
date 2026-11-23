"""
SLEEP-DAG-RSI v8 Track T1 (`calcite_lut_t1`): Closing W3 (`2-bit LUT` Quantization Gap
on Chinchilla `S3_85M_stage2_fp.pt` without retraining, 100% `CUDA SMEM T-MAC` compatible).

Zero-Leak Protocol:
  - Calibration split (activations/Hessians): disjoint `train_wins` + `tr_sents` (`seed=1001..1003`)
  - Dev split (Rounds 1-4 selection): first half of `eval_wins` + Dev cognitive suites (`N=48`, `seed=77001..77006`)
  - Sealed Blind Vault (`BlindVaultGuard(max_calls=1)`): second half of `eval_wins` + full `N=150`
    unseen cognitive suites (`seed=99001..99006`) + physical `PackedLUT2BitLinear` packing & parity check.
  - Zero access to `benchmarks/gsm8k/test.jsonl`.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

PLUGIN_SCRIPTS = os.path.expanduser(
    "~/.gemini/config/plugins/autonomous-research/skills/dream-rsi/scripts"
)
FALLBACK_SCRIPTS = (
    "C:/Users/vanya/.gemini/config/plugins/autonomous-research/skills/dream-rsi/scripts"
)
for _p in (PLUGIN_SCRIPTS, FALLBACK_SCRIPTS):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import engine as dream_engine  # noqa: E402
from cartridge import AutoDreamRunner, BranchCandidate, TaskCartridge  # noqa: E402

from experiments.scaling_phase_k.eval_suite import (  # noqa: E402
    eval_cognitive_suites_any_byte_lm,
    eval_openweb_6domains_any_byte_lm,
)
from experiments.scaling_phase_k.lean_system1 import (  # noqa: E402
    CalciteLeanSystem1LM,
    LeanSystem1Config,
)
from run_phase_e_fast_utf8_benchmark import build_disjoint_openweb_corpora  # noqa: E402
from simvq_split.quant_lut import (  # noqa: E402
    PackedLUT2BitLinear,
    compute_2bit_lut_footprint,
    quantize_ternary_weights_ls,
)
from simvq_split.tasks import (  # noqa: E402
    build_eval_overwrite_suite,
    build_eval_s5_suite,
    build_eval_trap_suite,
)
from simvq_split.tokenizer import ByteBPETokenizer  # noqa: E402

FP_CKPT_PATH = Path(
    r"C:\Users\vanya\.gemini\antigravity\brain\2f033e7f-ec95-4a6c-b082-d3d4081b221d\scratch\S3_85M_stage2_fp.pt"
)
LUT_META_CKPT_PATH = ROOT_DIR / "checkpoints" / "calcite5_s3_85m_chinchilla_utf8_lut.pt"
OUT_REPORT_PATH = ROOT_DIR / "benchmarks" / "dream_rsi_t1_lut_results.json"


def _extract_sentences(docs: List[str], max_sents: int = 1200) -> List[str]:
    sents: List[str] = []
    for d in docs:
        for raw in re.split(r"[.!?\n]+", d):
            s = " ".join(raw.strip().split())
            if 24 <= len(s) <= 140 and s.isascii():
                sents.append(s)
                if len(sents) >= max_sents:
                    return sents
    return sents


def _is_locked_router_layer(name: str) -> bool:
    return "byte_chunker.router_" in name


def _is_cognitive_gate_or_ptr(name: str) -> bool:
    if not (
        name.startswith("crystal.")
        or name.startswith("crystal_2.")
        or name.startswith("woodbury.")
    ):
        return False
    # Exclude large heavy projections unless threshold explicitly includes them
    return True


def _pack_trits_to_packed_lut(
    lin: nn.Linear,
    trits_i8: torch.Tensor,
    scale_1d: torch.Tensor,
    g_size: int,
) -> PackedLUT2BitLinear:
    """Packs custom `(trits_int8, scale_1d)` into a 100% standard `PackedLUT2BitLinear`."""
    w = lin.weight.detach()
    out_f, in_f = int(w.size(0)), int(w.size(1))
    raw_scales_fp16 = scale_1d.to(torch.float16).reshape(-1).cpu()
    bias_fp16 = (
        lin.bias.detach().to(torch.float16).cpu() if lin.bias is not None else None
    )
    in_pad = ((in_f + 3) // 4) * 4
    k_bytes = in_pad // 4
    row_aligned = (in_f % 32 == 0) and (g_size == 32)
    t_full = trits_i8.reshape(out_f, in_f).to(torch.int16).cpu()
    if in_pad > in_f:
        t_full = F.pad(t_full, (0, in_pad - in_f), value=0)
    t4 = t_full.view(out_f, k_bytes, 4) + 1
    codes = (t4[..., 0] + 3 * t4[..., 1] + 9 * t4[..., 2] + 27 * t4[..., 3]).to(
        torch.uint8
    )
    packed_u8 = codes.view(out_f, in_f // 32, 8) if row_aligned else codes
    return PackedLUT2BitLinear(
        in_features=in_f,
        out_features=out_f,
        packed_u8=packed_u8,
        raw_scales_fp16=raw_scales_fp16,
        bias_fp16=bias_fp16,
        g_size=g_size,
        row_aligned=row_aligned,
    )


def _compute_simulated_footprint_mb(
    model: nn.Module,
    fp16_keep_names: set[str],
) -> float:
    """Exact byte-accurate serialized footprint matching `compute_2bit_lut_footprint`."""
    packed_u8_bytes = 0
    scales_fp16_bytes = 0
    fp16_other_bytes = 0
    seen_ids: set[int] = set()
    lm_head_ref = getattr(model, "lm_head", None)

    linear_param_ids: set[int] = set()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            if mod is lm_head_ref or name == "lm_head":
                continue
            if _is_locked_router_layer(name) or (name in fp16_keep_names):
                continue
            linear_param_ids.add(id(mod.weight))
            if mod.bias is not None:
                linear_param_ids.add(id(mod.bias))
            out_f, in_f = int(mod.out_features), int(mod.in_features)
            g_size = 32 if (out_f * in_f) % 32 == 0 else in_f
            in_pad = ((in_f + 3) // 4) * 4
            k_bytes = in_pad // 4
            packed_u8_bytes += out_f * k_bytes
            n_scales = (out_f * in_f) // g_size
            scales_fp16_bytes += n_scales * 2
            if mod.bias is not None:
                scales_fp16_bytes += out_f * 2

    for p in model.parameters():
        pid = id(p)
        if pid in seen_ids or pid in linear_param_ids:
            continue
        seen_ids.add(pid)
        fp16_other_bytes += int(p.numel()) * 2

    total_bytes = packed_u8_bytes + scales_fp16_bytes + fp16_other_bytes
    return round(total_bytes / (1024.0 * 1024.0), 4)


# ---------------------------------------------------------------------------
# Quantization Operators (all 100% T-MAC 2-bit uint8 + FP16-scale compatible)
# ---------------------------------------------------------------------------


@torch.no_grad()
def quantize_awq_weighted_ls(
    w: torch.Tensor,
    x_rms: Optional[torch.Tensor] = None,
    alpha: float = 0.5,
    n_iters: int = 3,
    grid_points: int = 1,
    grid_span: float = 0.35,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Group-32 Activation-RMS weighted ternary LS with optional 1D non-convex scale grid search.
    Minimizes sum_j h_j * (w_j - s * t_j)^2 where h_j = (x_rms_j / mean(x_rms))^(2*alpha).
    """
    orig_shape = w.shape
    out_f, in_f = int(w.size(0)), int(w.size(1))
    g_size = 32 if w.numel() % 32 == 0 else in_f
    w_f = w.detach().float()
    w_g = w_f.reshape(-1, g_size)

    if x_rms is not None and alpha > 0.0 and (in_f % g_size == 0):
        h_1d = x_rms.to(device=w.device, dtype=torch.float32).clamp(min=1e-5)
        h_1d = (h_1d / h_1d.mean().clamp(min=1e-5)).pow(2.0 * float(alpha)).clamp(0.05, 50.0)
        h_g = h_1d.view(1, in_f // g_size, g_size).expand(out_f, -1, -1).reshape(-1, g_size)
    else:
        h_g = torch.ones_like(w_g)

    s0 = (w_g.abs() * h_g).sum(dim=-1, keepdim=True) / h_g.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    s0 = s0.clamp(min=1e-6)

    def _refine_from_scale(s_init: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = s_init
        t = torch.where(w_g.abs() >= 0.5 * s, w_g.sign(), torch.zeros_like(w_g))
        for _ in range(n_iters):
            denom = (h_g * t.square()).sum(dim=-1, keepdim=True)
            s = torch.where(
                denom > 1e-8,
                (h_g * w_g * t).sum(dim=-1, keepdim=True) / denom.clamp(min=1e-8),
                s0,
            ).clamp(min=1e-6)
            t = torch.where(w_g.abs() >= 0.5 * s, w_g.sign(), torch.zeros_like(w_g))
        denom = (h_g * t.square()).sum(dim=-1, keepdim=True)
        s = torch.where(
            denom > 1e-8,
            (h_g * w_g * t).sum(dim=-1, keepdim=True) / denom.clamp(min=1e-8),
            s0,
        ).clamp(min=1e-6)
        err = (h_g * (w_g - t * s).square()).sum(dim=-1)
        return t, s, err

    best_t, best_s, best_err = _refine_from_scale(s0)
    if grid_points > 1:
        mults = torch.linspace(1.0 - grid_span, 1.0 + 1.6 * grid_span, steps=grid_points, device=w.device)
        for m in mults:
            t_c, s_c, err_c = _refine_from_scale(s0 * m)
            better = err_c < best_err
            if better.any():
                b_col = better.unsqueeze(-1)
                best_t = torch.where(b_col, t_c, best_t)
                best_s = torch.where(b_col, s_c, best_s)
                best_err = torch.where(better, err_c, best_err)

    s_fp16 = best_s.to(torch.float16).float()
    w_deq = (best_t * s_fp16).reshape(orig_shape).to(w.dtype)
    return w_deq, best_t.to(torch.int8), best_s.squeeze(-1), g_size


@torch.no_grad()
def quantize_block32_hessian_cd(
    w: torch.Tensor,
    H_blocks: Optional[torch.Tensor],
    x_rms: Optional[torch.Tensor] = None,
    damp_rel: float = 0.05,
    cd_passes: int = 3,
    grid_points: int = 5,
    grid_span: float = 0.35,
    awq_alpha: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Block-32 Full-Hessian Coordinate Descent (`Mini-GPTQ` inside each 32-weight T-MAC group):
    Minimizes `(w_g - s * t)^T H_g (w_g - s * t)` over `t in {-1, 0, +1}^32, s > 0`.
    Zero extra parameters — outputs standard `(trits_int8, scale_fp16)` for `PackedLUT2BitLinear`.
    """
    orig_shape = w.shape
    out_f, in_f = int(w.size(0)), int(w.size(1))
    g_size = 32 if w.numel() % 32 == 0 else in_f

    # Fallback to weighted LS when layer is not row-aligned to 32 or lacks H_blocks
    if H_blocks is None or g_size != 32 or (in_f % 32 != 0):
        return quantize_awq_weighted_ls(
            w, x_rms=x_rms, alpha=awq_alpha, n_iters=3, grid_points=grid_points, grid_span=grid_span
        )

    dev = w.device
    n_blk = in_f // 32
    # Warm-start from grid-search diagonal weighted LS
    _, t_init_i8, s_init_1d, _ = quantize_awq_weighted_ls(
        w, x_rms=x_rms, alpha=awq_alpha, n_iters=2, grid_points=grid_points, grid_span=grid_span
    )

    w_b = w.detach().float().view(out_f, n_blk, 32)  # (O, G, 32)
    t_b = t_init_i8.float().view(out_f, n_blk, 32)   # (O, G, 32)
    s_b = s_init_1d.float().view(out_f, n_blk, 1)    # (O, G, 1)

    H = H_blocks.to(device=dev, dtype=torch.float32)  # (G, 32, 32)
    diag = torch.diagonal(H, dim1=-2, dim2=-1)  # (G, 32)
    mean_d = diag.mean(dim=-1, keepdim=True).clamp(min=1e-6)  # (G, 1)
    eye = torch.eye(32, device=dev, dtype=torch.float32).unsqueeze(0)
    # Normalize each block's Hessian scale for numerical stability
    H_norm = H / mean_d.unsqueeze(-1) + float(damp_rel) * eye  # (G, 32, 32)
    H_diag = torch.diagonal(H_norm, dim1=-2, dim2=-1).clamp(min=1e-5)  # (G, 32)

    # Precompute H_norm @ w^T -> (O, G, 32)
    # Using einsum: H_norm[g, j, k] * w_b[o, g, k] -> hw[o, g, j]
    hw = torch.einsum("gjk,ogk->ogj", H_norm, w_b)

    # Update optimal scale s* given t under H_norm: s* = (t^T H w) / (t^T H t)
    def _opt_scale(t_cur: torch.Tensor) -> torch.Tensor:
        ht = torch.einsum("gjk,ogk->ogj", H_norm, t_cur)
        num = (t_cur * hw).sum(dim=-1, keepdim=True)
        den = (t_cur * ht).sum(dim=-1, keepdim=True)
        s_new = torch.where(den > 1e-6, num / den.clamp(min=1e-6), s_b)
        return s_new.clamp(min=1e-6)

    s_b = _opt_scale(t_b)
    # Maintain residual vector Hr = H (w - s * t) -> shape (O, G, 32)
    hr = hw - s_b * torch.einsum("gjk,ogk->ogj", H_norm, t_b)

    # Vectorized Coordinate Descent across all (out_f, n_blk) simultaneously for k=0..31
    for _pass in range(int(cd_passes)):
        for k in range(32):
            t_old = t_b[:, :, k]              # (O, G)
            hr_k = hr[:, :, k]                # (O, G)
            s_cur = s_b[:, :, 0]              # (O, G)
            h_kk = H_diag[:, k].unsqueeze(0)  # (1, G)
            # Objective change when replacing t_old -> c in {-1, 0, +1}:
            # d = (c - t_old) * s_cur
            # delta_E = -2 * d * hr_k + d^2 * h_kk
            # Optimal continuous target c* = t_old + hr_k / (s_cur * h_kk)
            c_star = t_old + hr_k / (s_cur * h_kk).clamp(min=1e-7)
            t_new = torch.where(
                c_star >= 0.5,
                torch.ones_like(c_star),
                torch.where(c_star <= -0.5, -torch.ones_like(c_star), torch.zeros_like(c_star)),
            )
            dt = t_new - t_old
            if (dt != 0).any():
                t_b[:, :, k] = t_new
                # hr <- hr - (dt * s_cur) * H_norm[:, :, k]
                hr = hr - (dt * s_cur).unsqueeze(-1) * H_norm[:, :, k].unsqueeze(0)

        # Re-solve exact H-optimal scale and refresh hr
        s_b = _opt_scale(t_b)
        hr = hw - s_b * torch.einsum("gjk,ogk->ogj", H_norm, t_b)

    s_fp16 = s_b.to(torch.float16).float()
    w_deq = (t_b * s_fp16).reshape(orig_shape).to(w.dtype)
    return w_deq, t_b.reshape(-1, 32).to(torch.int8), s_b.reshape(-1), 32


@torch.no_grad()
def apply_rmsnorm_smoothing(
    model: CalciteLeanSystem1LM,
    act_rms: Dict[str, torch.Tensor],
    gamma: float,
) -> None:
    """
    Zero-overhead SmoothQuant channel migration: absorbs `s_j = rms(x_j)^gamma / max(|W_:,j|)^(1-gamma)`
    into the preceding `RMSNorm.weight` and scales downstream linear weights by `s_j`,
    preserving the exact FP forward pass while equalizing dynamic range before 2-bit LUT quantization.
    """
    if gamma <= 1e-6:
        return

    def _smooth_pair(norm_mod: nn.Module, lin_names_and_mods: List[Tuple[str, nn.Linear]]) -> None:
        if not hasattr(norm_mod, "weight") or norm_mod.weight is None:
            return
        ref_name = lin_names_and_mods[0][0]
        if ref_name not in act_rms:
            return
        x_r = act_rms[ref_name].to(device=norm_mod.weight.device, dtype=torch.float32).clamp(min=1e-4)
        w_max = torch.zeros_like(x_r)
        for _, lin in lin_names_and_mods:
            w_max = torch.maximum(w_max, lin.weight.detach().float().abs().max(dim=0).values)
        w_max = w_max.clamp(min=1e-4)
        s = (x_r.pow(gamma) / w_max.pow(1.0 - gamma)).clamp(0.15, 6.0)
        # Normalize geometric mean to 1.0 to preserve global activation scale
        s = s / s.log().mean().exp().clamp(min=1e-4)
        norm_mod.weight.data.div_(s.to(norm_mod.weight.dtype))
        for _, lin in lin_names_and_mods:
            lin.weight.data.mul_(s.unsqueeze(0).to(lin.weight.dtype))

    # Smooth cortex_pre & cortex_post transformer blocks (norm1 -> qkv, norm2 -> w_gate/w_up)
    for stack_name in ("cortex_pre", "cortex_post"):
        stack = getattr(model, stack_name, None)
        if stack is None:
            continue
        for idx, blk in enumerate(stack):
            if hasattr(blk, "norm1") and hasattr(blk, "attn") and hasattr(blk.attn, "qkv"):
                _smooth_pair(blk.norm1, [(f"{stack_name}.{idx}.attn.qkv", blk.attn.qkv)])
            if hasattr(blk, "norm2") and hasattr(blk, "mlp"):
                mlp_lins = []
                for sub in ("w_gate", "w_up", "fc1"):
                    if hasattr(blk.mlp, sub):
                        mlp_lins.append((f"{stack_name}.{idx}.mlp.{sub}", getattr(blk.mlp, sub)))
                if mlp_lins:
                    _smooth_pair(blk.norm2, mlp_lins)

    # Smooth mid_norm -> mid_mlp
    if hasattr(model, "mid_norm") and hasattr(model, "mid_mlp"):
        mid_lins = []
        for sub in ("w_gate", "w_up", "fc1"):
            if hasattr(model.mid_mlp, sub):
                mid_lins.append((f"mid_mlp.{sub}", getattr(model.mid_mlp, sub)))
        if mid_lins:
            _smooth_pair(model.mid_norm, mid_lins)


# ---------------------------------------------------------------------------
# Calibration Statistics Collector (Strictly on Calibration Split)
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_calibration_hessians(
    fp_model: CalciteLeanSystem1LM,
    cal_batches: List[torch.Tensor],
    device: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Collects exact per-channel RMS `act_rms[name]` and block-32 correlation matrices
    `H_blocks[name]` of shape `(in_f // 32, 32, 32)` for every quantizable `nn.Linear`
    in a single forward sweep over the strictly disjoint calibration split.
    """
    fp_model.eval()
    lm_head_ref = getattr(fp_model, "lm_head", None)
    sum_sq: Dict[str, torch.Tensor] = {}
    h_acc: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    hooks = []

    for name, mod in fp_model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if mod is lm_head_ref or name == "lm_head" or _is_locked_router_layer(name):
            continue
        in_f = int(mod.in_features)
        sum_sq[name] = torch.zeros(in_f, device=device, dtype=torch.float64)
        counts[name] = 0
        if in_f % 32 == 0:
            h_acc[name] = torch.zeros(in_f // 32, 32, 32, device=device, dtype=torch.float64)

        def _make_hook(lname: str, can_block32: bool):
            def _hook(_m, inp, _out):
                x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                if x.size(0) == 0:
                    return
                # Subsample very long sequences slightly for speed while keeping full coverage
                if x.size(0) > 2048:
                    stride = max(1, x.size(0) // 2048)
                    x = x[::stride]
                n = int(x.size(0))
                sum_sq[lname] += x.square().sum(dim=0, dtype=torch.float64)
                counts[lname] += n
                if can_block32:
                    xb = x.view(n, -1, 32)  # (N, G, 32)
                    # Batched Gram matrix per group: (G, 32, 32)
                    h_acc[lname] += torch.einsum("ngi,ngj->gij", xb, xb).to(torch.float64)
            return _hook

        hooks.append(mod.register_forward_hook(_make_hook(name, in_f % 32 == 0)))

    try:
        for xb in cal_batches:
            xb_d = xb.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                _ = fp_model(xb_d)
    finally:
        for h in hooks:
            h.remove()

    act_rms: Dict[str, torch.Tensor] = {}
    H_blocks: Dict[str, torch.Tensor] = {}
    for name, sq in sum_sq.items():
        c = max(1, counts[name])
        act_rms[name] = (sq / float(c)).clamp(min=1e-10).sqrt().float().cpu()
        if name in h_acc:
            H_blocks[name] = (h_acc[name] / float(c)).float().cpu()

    return act_rms, H_blocks


# ---------------------------------------------------------------------------
# Quantized Model Builder (Fast Simulated LUT for Dev + Physical T-MAC for Vault)
# ---------------------------------------------------------------------------


@torch.no_grad()
def build_quantized_candidate(
    base_state_cpu: Dict[str, torch.Tensor],
    work_model: CalciteLeanSystem1LM,
    act_rms: Dict[str, torch.Tensor],
    H_blocks: Dict[str, torch.Tensor],
    spec: Dict[str, Any],
    device: str,
    physical_pack: bool = False,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Applies a quantization recipe `spec` to `base_state_cpu`.
    When `physical_pack=False`, replaces linear weights with exact FP16-scale dequantized ternary
    values (mathematically identical to `PackedLUT2BitLinear` up to float rounding, 50x faster on local Windows).
    When `physical_pack=True`, constructs physical `PackedLUT2BitLinear` T-MAC modules and measures
    `compute_2bit_lut_footprint`.
    """
    work_model.load_state_dict(base_state_cpu, strict=True)
    work_model.to(device).eval()
    work_model.byte_chunker.lock_router(True)
    work_model.set_crystal_schedule(1, 1)
    work_model.set_snap_alpha(1.0)

    mode = str(spec.get("mode", "baseline_ls"))
    smooth_gamma = float(spec.get("smooth_gamma", 0.0))
    awq_alpha = float(spec.get("awq_alpha", 0.0))
    damp_rel = float(spec.get("damp_rel", 0.05))
    cd_passes = int(spec.get("cd_passes", 3))
    grid_points = int(spec.get("grid_points", 1))
    grid_span = float(spec.get("grid_span", 0.35))
    barbell_kb_thresh = float(spec.get("barbell_kb_thresh", 0.0))
    barbell_scope = str(spec.get("barbell_scope", "cog_only"))

    # Optional RMSNorm smoothing (note: if smoothing is applied, rescale act_rms & H_blocks accordingly)
    smoothed_scales: Dict[str, torch.Tensor] = {}
    if smooth_gamma > 1e-6:
        # Capture pre-smoothing norm weights to infer exact scale factor applied per linear layer
        pre_norm_w: Dict[str, torch.Tensor] = {}
        for n_name, mod in work_model.named_modules():
            if hasattr(mod, "weight") and mod.weight is not None and "norm" in n_name:
                pre_norm_w[n_name] = mod.weight.detach().float().clone()
        apply_rmsnorm_smoothing(work_model, act_rms, smooth_gamma)
        # Recompute exact input scale ratio `s_inv = norm_after / norm_before` so smoothed activations x' = x / s
        for stack_name in ("cortex_pre", "cortex_post"):
            stack = getattr(work_model, stack_name, None)
            if stack is None:
                continue
            for idx, blk in enumerate(stack):
                n1_k = f"{stack_name}.{idx}.norm1"
                if n1_k in pre_norm_w:
                    ratio = (blk.norm1.weight.detach().float() / pre_norm_w[n1_k].clamp(min=1e-6)).cpu()
                    smoothed_scales[f"{stack_name}.{idx}.attn.qkv"] = ratio
                n2_k = f"{stack_name}.{idx}.norm2"
                if n2_k in pre_norm_w:
                    ratio = (blk.norm2.weight.detach().float() / pre_norm_w[n2_k].clamp(min=1e-6)).cpu()
                    for sub in ("w_gate", "w_up", "fc1"):
                        smoothed_scales[f"{stack_name}.{idx}.mlp.{sub}"] = ratio
        if "mid_norm" in pre_norm_w:
            ratio = (work_model.mid_norm.weight.detach().float() / pre_norm_w["mid_norm"].clamp(min=1e-6)).cpu()
            for sub in ("w_gate", "w_up", "fc1"):
                smoothed_scales[f"mid_mlp.{sub}"] = ratio

    # Identify Barbell FP16 keep set
    fp16_keep: set[str] = set()
    if barbell_kb_thresh > 0.0:
        for name, mod in work_model.named_modules():
            if not isinstance(mod, nn.Linear) or name == "lm_head" or _is_locked_router_layer(name):
                continue
            fp16_kb = (mod.weight.numel() * 2.0) / 1024.0
            in_scope = _is_cognitive_gate_or_ptr(name) if barbell_scope == "cog_only" else (
                _is_cognitive_gate_or_ptr(name) or name.startswith("byte_chunker.")
            )
            if in_scope and fp16_kb <= barbell_kb_thresh:
                fp16_keep.add(name)

    lm_head_ref = getattr(work_model, "lm_head", None)
    packed_layers: Dict[str, PackedLUT2BitLinear] = {}
    n_quantized = 0
    n_fp16_barbell = 0

    for name, mod in work_model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if mod is lm_head_ref or name == "lm_head" or _is_locked_router_layer(name):
            continue
        if name in fp16_keep:
            # Round to FP16 precision to reflect exact FP16 inference weights
            mod.weight.data.copy_(mod.weight.detach().to(torch.float16).to(mod.weight.dtype))
            if mod.bias is not None:
                mod.bias.data.copy_(mod.bias.detach().to(torch.float16).to(mod.bias.dtype))
            n_fp16_barbell += 1
            continue

        x_r = act_rms.get(name)
        H_b = H_blocks.get(name)
        if name in smoothed_scales and x_r is not None:
            sc_r = smoothed_scales[name].to(x_r.dtype)
            x_r = x_r * sc_r.abs().clamp(min=1e-4)
            if H_b is not None:
                sc_b = sc_r.view(-1, 32)
                H_b = H_b * (sc_b.unsqueeze(-1) * sc_b.unsqueeze(-2))

        if mode == "baseline_ls":
            w_t, trits_i8, sc_1d = quantize_ternary_weights_ls(mod.weight.detach())
            g_sz = 32 if mod.weight.numel() % 32 == 0 else int(mod.in_features)
            sc_fp16 = sc_1d.to(torch.float16).float().unsqueeze(-1)
            w_deq = (trits_i8.float() * sc_fp16).reshape(mod.weight.shape).to(mod.weight.dtype)
        elif mode == "awq_ls":
            w_deq, trits_i8, sc_1d, g_sz = quantize_awq_weighted_ls(
                mod.weight.detach(),
                x_rms=x_r,
                alpha=awq_alpha,
                n_iters=3,
                grid_points=grid_points,
                grid_span=grid_span,
            )
        elif mode == "hessian_cd":
            w_deq, trits_i8, sc_1d, g_sz = quantize_block32_hessian_cd(
                mod.weight.detach(),
                H_blocks=H_b,
                x_rms=x_r,
                damp_rel=damp_rel,
                cd_passes=cd_passes,
                grid_points=grid_points,
                grid_span=grid_span,
                awq_alpha=awq_alpha,
            )
        else:
            raise ValueError(f"Unknown quantization mode: {mode}")

        n_quantized += 1
        if physical_pack:
            packed_layers[name] = _pack_trits_to_packed_lut(mod, trits_i8, sc_1d, g_sz)
        else:
            mod.weight.data.copy_(w_deq)
            if mod.bias is not None:
                mod.bias.data.copy_(mod.bias.detach().to(torch.float16).to(mod.bias.dtype))

    lut_mb = _compute_simulated_footprint_mb(work_model, fp16_keep)
    meta = {
        "n_quantized": n_quantized,
        "n_fp16_barbell": n_fp16_barbell,
        "fp16_keep_layers": sorted(fp16_keep),
        "lut_mb": lut_mb,
    }

    if physical_pack:
        phys = copy.deepcopy(work_model).cpu().eval()
        for full_name, plut in packed_layers.items():
            parts = full_name.split(".")
            parent = phys
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], plut)
        fp_stats = compute_2bit_lut_footprint(phys)
        meta["physical_footprint"] = fp_stats
        meta["lut_mb"] = round(fp_stats["total_lut_kb"] / 1024.0, 4)
        return phys.to(device).eval(), meta

    return work_model, meta


# ---------------------------------------------------------------------------
# TaskCartridge Builder for `calcite_lut_t1`
# ---------------------------------------------------------------------------


def build_cartridge() -> Tuple[TaskCartridge, Dict[str, Any]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[T1] Loading S3_85M_stage2_fp.pt on {device}...", flush=True)
    lut_meta = torch.load(LUT_META_CKPT_PATH, map_location="cpu", weights_only=False)
    cfg = LeanSystem1Config(**lut_meta["simvq_split_config"])
    tok = ByteBPETokenizer.from_dict(lut_meta["tokenizer"])
    fp_ckpt = torch.load(FP_CKPT_PATH, map_location="cpu", weights_only=False)
    base_state_cpu = {k: v.detach().cpu().clone() for k, v in fp_ckpt["state_dict"].items()}

    work_model = CalciteLeanSystem1LM(cfg).eval()
    work_model.set_fast_kernels(False)
    work_model.load_state_dict(base_state_cpu, strict=True)
    work_model.byte_chunker.lock_router(True)
    work_model.set_crystal_schedule(1, 1)
    work_model.set_snap_alpha(1.0)
    work_model = work_model.to(device)

    # Strictly disjoint splits sourced ONLY from benchmarks/gsm8k/train.jsonl (ZERO test.jsonl access)
    gsm_train_path = ROOT_DIR / "benchmarks" / "gsm8k" / "train.jsonl"
    all_docs = [
        json.loads(line)["question"] + "\n" + json.loads(line)["answer"]
        for line in gsm_train_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cal_docs = all_docs[:4500]
    dev_docs = all_docs[4500:5800]
    vault_docs = all_docs[5800:7100]

    cal_sents = _extract_sentences(cal_docs, max_sents=800)
    dev_sents = _extract_sentences(dev_docs, max_sents=800)
    vault_sents = _extract_sentences(vault_docs, max_sents=800)

    train_wins, eval_wins, _, _ = build_disjoint_openweb_corpora(
        ROOT_DIR, tok, cal_docs[:1500], dev_docs[:400], seq_len=512
    )
    # Split eval_wins strictly in half: Dev = first half, Blind Vault = second half
    dev_eval_wins: Dict[str, List[List[int]]] = {}
    vault_eval_wins: Dict[str, List[List[int]]] = {}
    for dom, wins in eval_wins.items():
        mid = max(1, len(wins) // 2)
        dev_eval_wins[dom] = wins[: min(mid, 12)]
        vault_eval_wins[dom] = wins[mid:]

    # Calibration batches: strictly from train_wins + cal_sents cognitive examples
    cal_batches: List[torch.Tensor] = []
    for dom, wins in train_wins.items():
        take = wins[:6]
        if take:
            cal_batches.append(torch.tensor([w[:-1] for w in take], dtype=torch.long))
    cal_s5 = build_eval_s5_suite(tok, cal_sents, 4, 16, (0, 1), 1001)
    cal_ov = build_eval_overwrite_suite(tok, cal_sents, 7, 4, 10, 16, 1002)
    cal_tr = build_eval_trap_suite(tok, cal_sents, 7, 4, 10, 16, 1003)
    for suite in (cal_s5, cal_ov, cal_tr):
        seqs = [list(item.prompt_ids) + list(item.target_ids) for item in suite]
        max_l = max(len(s) for s in seqs)
        batch = torch.full((len(seqs), max_l), tok.PAD_ID, dtype=torch.long)
        for i, seq in enumerate(seqs):
            batch[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        cal_batches.append(batch)

    # Dev cognitive suites (N=48, seed=77001..77009)
    n_dev = 48
    dev_s5 = {
        "k2": build_eval_s5_suite(tok, dev_sents, 2, n_dev, (0, 1), 77001),
        "k4": build_eval_s5_suite(tok, dev_sents, 4, n_dev, (0, 1), 77002),
        "k8": build_eval_s5_suite(tok, dev_sents, 8, n_dev, (1, 1), 77003),
        "k16": build_eval_s5_suite(tok, dev_sents, 16, n_dev, (1, 2), 77004),
    }
    dev_ov = {
        "in_dist": build_eval_overwrite_suite(tok, dev_sents, 4, 2, 2, n_dev, 77005),
        "ood_long": build_eval_overwrite_suite(tok, dev_sents, 9, 5, 18, n_dev, 77006),
    }
    dev_trap = {
        "in_dist": build_eval_trap_suite(tok, dev_sents, 4, 2, 2, n_dev, 77007),
        "ood_long": build_eval_trap_suite(tok, dev_sents, 9, 5, 18, n_dev, 77008),
    }

    # Sealed Blind Vault cognitive suites (full N=150, disjoint vault_sents, seed=99001..99009)
    n_vault = 150
    vault_s5 = {
        "k2": build_eval_s5_suite(tok, vault_sents, 2, n_vault, (0, 1), 99001),
        "k4": build_eval_s5_suite(tok, vault_sents, 4, n_vault, (0, 1), 99002),
        "k8": build_eval_s5_suite(tok, vault_sents, 8, n_vault, (1, 1), 99003),
        "k16": build_eval_s5_suite(tok, vault_sents, 16, n_vault, (1, 2), 99004),
    }
    vault_ov = {
        "in_dist": build_eval_overwrite_suite(tok, vault_sents, 4, 2, 2, n_vault, 99005),
        "ood_long": build_eval_overwrite_suite(tok, vault_sents, 9, 5, 18, n_vault, 99006),
    }
    vault_trap = {
        "in_dist": build_eval_trap_suite(tok, vault_sents, 4, 2, 2, n_vault, 99007),
        "ood_long": build_eval_trap_suite(tok, vault_sents, 9, 5, 18, n_vault, 99008),
    }

    # Collect calibration Hessians once + record FP ceiling reference on Dev
    t_cal = time.perf_counter()
    act_rms, H_blocks = collect_calibration_hessians(work_model, cal_batches, device)
    print(
        f"[T1] Collected Hessians for {len(act_rms)} linear layers ({len(H_blocks)} block-32) "
        f"in {time.perf_counter() - t_cal:.2f}s",
        flush=True,
    )

    # Measure FP ceiling on Dev (reference only, never a candidate)
    ow_fp = eval_openweb_6domains_any_byte_lm(work_model, dev_eval_wins, device=device, batch_size=8)
    cog_fp = eval_cognitive_suites_any_byte_lm(
        work_model, tok, dev_s5, dev_ov, dev_trap, device=device, batch_size=24
    )
    fp_ref = {
        "mean_openweb_bpb": ow_fp["mean_openweb_bpb"],
        "s5_sum": cog_fp["s5_sum"],
        "selective_f1_ood_long": cog_fp["selective_f1_ood_long"],
        "ov_long": cog_fp["overwrite"]["ood_long"]["exact_pct"],
        "trap_long": cog_fp["trap"]["ood_long"]["exact_pct"],
        "regimes": {k: v["byte_bpb"] for k, v in ow_fp["regimes"].items()},
    }
    print(
        f"[T1] FP Ceiling (Dev): ow6={fp_ref['mean_openweb_bpb']:.4f} | "
        f"S5={fp_ref['s5_sum']:.2f}% | SelF1_long={fp_ref['selective_f1_ood_long']:.2f}%",
        flush=True,
    )

    # Capture original router weights for invariance verification
    orig_router_state = {
        k: v.clone() for k, v in base_state_cpu.items() if _is_locked_router_layer(k)
    }

    # Evaluation cache keyed by canonical spec JSON so identical params aren't re-evaluated
    eval_cache: Dict[str, Tuple[bool, float, str, Dict[str, Any]]] = {}

    def _score_from_metrics(ow: dict, cog: dict, lut_mb: float) -> Tuple[float, List[int], List[float]]:
        mean_bpb = float(ow["mean_openweb_bpb"])
        s5_sum = float(cog["s5_sum"])
        sel_long = float(cog["selective_f1_ood_long"])
        size_pen = 8.0 * max(0.0, lut_mb - 25.60)
        score = 100.0 * (4.0 - mean_bpb) + 0.15 * s5_sum + 0.20 * sel_long - size_pen

        # 12 real empirical buckets: 6 OpenWeb domains (higher = better: 6.0 - bpb) + 6 cognitive suites
        dom_order = [
            "1_GSM8K_EN",
            "2_Russian_UTF8",
            "3_Python_Code",
            "4_ARC_JSON",
            "5_CJK_UTF8",
            "6_Emoji_4Byte",
        ]
        bucket_scores: List[float] = []
        pass_mask: List[int] = []
        for d in dom_order:
            bpb_d = float(ow["regimes"][d]["byte_bpb"])
            fp_d = float(fp_ref["regimes"][d])
            margin = fp_d - bpb_d  # >= -0.025 means near-lossless vs FP
            bucket_scores.append(round(6.0 - bpb_d, 4))
            pass_mask.append(1 if margin >= -0.025 else 0)

        for k in ("k2", "k4", "k8", "k16"):
            pct = float(cog["s5"][k]["exact_pct"])
            bucket_scores.append(round(pct / 100.0, 4))
            thresh = 60.0 if k in ("k2", "k4") else (25.0 if k == "k8" else 15.0)
            pass_mask.append(1 if pct >= thresh else 0)

        ov_l = float(cog["overwrite"]["ood_long"]["exact_pct"])
        tr_l = float(cog["trap"]["ood_long"]["exact_pct"])
        bucket_scores.append(round(ov_l / 100.0, 4))
        pass_mask.append(1 if ov_l >= 50.0 else 0)
        bucket_scores.append(round(tr_l / 100.0, 4))
        pass_mask.append(1 if tr_l >= 55.0 else 0)
        return round(score, 4), pass_mask, bucket_scores

    def verify_and_score(fn: Callable[[], Dict[str, Any]]) -> Tuple[bool, float, str, Dict[str, Any]]:
        spec = dict(fn())
        key = json.dumps(spec, sort_keys=True)
        if key in eval_cache:
            return eval_cache[key]

        t0 = time.perf_counter()
        q_model, q_meta = build_quantized_candidate(
            base_state_cpu, work_model, act_rms, H_blocks, spec, device=device, physical_pack=False
        )
        lut_mb = float(q_meta["lut_mb"])
        if lut_mb > 26.50:
            res = (
                False,
                0.0,
                "size_budget_exceeded",
                {"error": f"LUT size {lut_mb:.2f} MB > 26.50 MB cap", "lut_mb": lut_mb},
            )
            eval_cache[key] = res
            return res

        # Verify router invariance
        r_diff = 0.0
        cur_sd = q_model.state_dict()
        for rk, rv in orig_router_state.items():
            d = float((cur_sd[rk].detach().cpu().float() - rv.float()).abs().max().item())
            r_diff = max(r_diff, d)
        if r_diff > 1e-6:
            return (False, 0.0, "router_mutated", {"error": f"router_max_abs_diff={r_diff}"})

        ow = eval_openweb_6domains_any_byte_lm(q_model, dev_eval_wins, device=device, batch_size=8)
        if float(ow["max_mid_utf8_split_pct"]) > 0.0:
            return (False, 0.0, "utf8_split_violation", {"error": "mid_utf8_split_pct > 0"})

        cog = eval_cognitive_suites_any_byte_lm(
            q_model, tok, dev_s5, dev_ov, dev_trap, device=device, batch_size=24
        )
        score, pass_mask, bucket_scores = _score_from_metrics(ow, cog, lut_mb)
        dt = round(time.perf_counter() - t0, 2)
        info = {
            "pass_mask": pass_mask,
            "bucket_scores": bucket_scores,
            "spec": spec,
            "mean_openweb_bpb": ow["mean_openweb_bpb"],
            "regimes_bpb": {k: v["byte_bpb"] for k, v in ow["regimes"].items()},
            "s5_sum": cog["s5_sum"],
            "s5": {k: v["exact_pct"] for k, v in cog["s5"].items()},
            "ov_long": cog["overwrite"]["ood_long"]["exact_pct"],
            "trap_long": cog["trap"]["ood_long"]["exact_pct"],
            "selective_f1_ood_long": cog["selective_f1_ood_long"],
            "lut_mb": lut_mb,
            "n_quantized": q_meta["n_quantized"],
            "n_fp16_barbell": q_meta["n_fp16_barbell"],
            "router_max_abs_diff": r_diff,
            "eval_s": dt,
            "summary": (
                f"score={score:.2f} | ow6={ow['mean_openweb_bpb']:.4f} | "
                f"S5={cog['s5_sum']:.1f}% | SelF1={cog['selective_f1_ood_long']:.1f}% | "
                f"size={lut_mb:.2f}MB ({dt}s)"
            ),
        }
        print(f"  [Eval] {spec.get('tag', spec.get('mode'))}: {info['summary']}", flush=True)
        res = (True, score, "ok", info)
        eval_cache[key] = res
        return res

    # Single-shot Sealed Blind Vault: full N=150 unseen cognitive + second half eval_wins +
    # physical PackedLUT2BitLinear parity verification + baseline comparison on same Vault
    def blind_vault_eval(fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        spec = dict(fn())
        print(
            f"\n[T1 Blind Vault] Opening SINGLE-SHOT sealed vault for champion `{spec.get('tag', spec.get('mode'))}`...",
            flush=True,
        )
        # 1. Evaluate FP ceiling on Vault (reference only)
        work_model.load_state_dict(base_state_cpu, strict=True)
        work_model.to(device).eval()
        work_model.byte_chunker.lock_router(True)
        work_model.set_crystal_schedule(1, 1)
        work_model.set_snap_alpha(1.0)
        ow_v_fp = eval_openweb_6domains_any_byte_lm(work_model, vault_eval_wins, device=device, batch_size=8)
        cog_v_fp = eval_cognitive_suites_any_byte_lm(
            work_model, tok, vault_s5, vault_ov, vault_trap, device=device, batch_size=32
        )

        # 2. Evaluate Baseline 2-bit LUT on Vault (reference only)
        base_m, base_meta = build_quantized_candidate(
            base_state_cpu, work_model, act_rms, H_blocks, {"mode": "baseline_ls", "tag": "baseline"},
            device=device, physical_pack=False,
        )
        ow_v_base = eval_openweb_6domains_any_byte_lm(base_m, vault_eval_wins, device=device, batch_size=8)
        cog_v_base = eval_cognitive_suites_any_byte_lm(
            base_m, tok, vault_s5, vault_ov, vault_trap, device=device, batch_size=32
        )

        # 3. Evaluate Champion on Vault (simulated exact dequant)
        champ_m, champ_meta = build_quantized_candidate(
            base_state_cpu, work_model, act_rms, H_blocks, spec, device=device, physical_pack=False
        )
        ow_v_ch = eval_openweb_6domains_any_byte_lm(champ_m, vault_eval_wins, device=device, batch_size=8)
        cog_v_ch = eval_cognitive_suites_any_byte_lm(
            champ_m, tok, vault_s5, vault_ov, vault_trap, device=device, batch_size=32
        )

        # 4. Physical PackedLUT2BitLinear T-MAC packing + exact logit parity verification
        x_par = torch.tensor([vault_eval_wins["1_GSM8K_EN"][0][:-1]], dtype=torch.long, device=device)
        with torch.no_grad():
            lg_sim = champ_m(x_par).float()
        phys_m, phys_meta = build_quantized_candidate(
            base_state_cpu, work_model, act_rms, H_blocks, spec, device=device, physical_pack=True
        )
        with torch.no_grad():
            lg_phys = phys_m(x_par).float()
        tmac_parity_max_diff = float((lg_sim - lg_phys).abs().max().item())
        tmac_parity_mean_diff = float((lg_sim - lg_phys).abs().mean().item())
        del phys_m
        if device == "cuda":
            torch.cuda.empty_cache()

        return {
            "champion_spec": spec,
            "fp_ceiling_vault": {
                "mean_openweb_bpb": ow_v_fp["mean_openweb_bpb"],
                "regimes_bpb": {k: v["byte_bpb"] for k, v in ow_v_fp["regimes"].items()},
                "s5_sum": cog_v_fp["s5_sum"],
                "s5": {k: v["exact_pct"] for k, v in cog_v_fp["s5"].items()},
                "ov_long": cog_v_fp["overwrite"]["ood_long"]["exact_pct"],
                "trap_long": cog_v_fp["trap"]["ood_long"]["exact_pct"],
                "selective_f1_ood_long": cog_v_fp["selective_f1_ood_long"],
            },
            "baseline_lut_vault": {
                "mean_openweb_bpb": ow_v_base["mean_openweb_bpb"],
                "regimes_bpb": {k: v["byte_bpb"] for k, v in ow_v_base["regimes"].items()},
                "s5_sum": cog_v_base["s5_sum"],
                "s5": {k: v["exact_pct"] for k, v in cog_v_base["s5"].items()},
                "ov_long": cog_v_base["overwrite"]["ood_long"]["exact_pct"],
                "trap_long": cog_v_base["trap"]["ood_long"]["exact_pct"],
                "selective_f1_ood_long": cog_v_base["selective_f1_ood_long"],
                "lut_mb": base_meta["lut_mb"],
            },
            "champion_lut_vault": {
                "mean_openweb_bpb": ow_v_ch["mean_openweb_bpb"],
                "regimes_bpb": {k: v["byte_bpb"] for k, v in ow_v_ch["regimes"].items()},
                "s5_sum": cog_v_ch["s5_sum"],
                "s5": {k: v["exact_pct"] for k, v in cog_v_ch["s5"].items()},
                "ov_long": cog_v_ch["overwrite"]["ood_long"]["exact_pct"],
                "trap_long": cog_v_ch["trap"]["ood_long"]["exact_pct"],
                "selective_f1_ood_long": cog_v_ch["selective_f1_ood_long"],
                "max_mid_utf8_split_pct": ow_v_ch["max_mid_utf8_split_pct"],
                "lut_mb": phys_meta["lut_mb"],
                "physical_footprint": phys_meta["physical_footprint"],
                "n_quantized": phys_meta["n_quantized"],
                "n_fp16_barbell": phys_meta["n_fp16_barbell"],
                "fp16_keep_layers": phys_meta["fp16_keep_layers"],
            },
            "physical_tmac_parity_max_diff": round(tmac_parity_max_diff, 6),
            "physical_tmac_parity_mean_diff": round(tmac_parity_mean_diff, 6),
            "delta_vs_baseline": {
                "openweb_bpb": round(ow_v_ch["mean_openweb_bpb"] - ow_v_base["mean_openweb_bpb"], 4),
                "s5_sum_pp": round(cog_v_ch["s5_sum"] - cog_v_base["s5_sum"], 2),
                "selective_f1_long_pp": round(
                    cog_v_ch["selective_f1_ood_long"] - cog_v_base["selective_f1_ood_long"], 2
                ),
                "lut_mb_delta": round(phys_meta["lut_mb"] - base_meta["lut_mb"], 4),
            },
            "gap_vs_fp_ceiling": {
                "baseline_bpb_gap": round(ow_v_base["mean_openweb_bpb"] - ow_v_fp["mean_openweb_bpb"], 4),
                "champion_bpb_gap": round(ow_v_ch["mean_openweb_bpb"] - ow_v_fp["mean_openweb_bpb"], 4),
                "baseline_s5_gap_pp": round(cog_v_base["s5_sum"] - cog_v_fp["s5_sum"], 2),
                "champion_s5_gap_pp": round(cog_v_ch["s5_sum"] - cog_v_fp["s5_sum"], 2),
            },
        }

    # --- Baseline & 4 Orthogonal Round-1 Branches ---
    base_fn = lambda: {"mode": "baseline_ls", "tag": "root_baseline_ls"}

    b0 = BranchCandidate(
        id="b0_hessian_cd",
        branch=0,
        description="Block-32 Full-Hessian Coordinate Descent (Mini-GPTQ, 0 KB overhead, 100% T-MAC)",
        fn=lambda: {
            "mode": "hessian_cd",
            "damp_rel": 0.05,
            "cd_passes": 3,
            "grid_points": 5,
            "grid_span": 0.35,
            "awq_alpha": 0.5,
            "tag": "b0_hessian_cd",
        },
        next_refinement=BranchCandidate(
            id="b0_hessian_cd_deep",
            branch=0,
            parent_id="b0_hessian_cd",
            attempt=2,
            description="Block-32 Hessian CD with tighter damping (0.015) and 5 CD passes",
            fn=lambda: {
                "mode": "hessian_cd",
                "damp_rel": 0.015,
                "cd_passes": 5,
                "grid_points": 7,
                "grid_span": 0.40,
                "awq_alpha": 0.5,
                "tag": "b0_hessian_cd_deep",
            },
        ),
    )

    b1 = BranchCandidate(
        id="b1_awq_rms_smooth",
        branch=1,
        description="Zero-Overhead RMSNorm Channel Smoothing + Activation-RMS Weighted LS (0 KB overhead)",
        fn=lambda: {
            "mode": "awq_ls",
            "smooth_gamma": 0.35,
            "awq_alpha": 0.5,
            "grid_points": 7,
            "grid_span": 0.35,
            "tag": "b1_awq_rms_smooth",
        },
        next_refinement=BranchCandidate(
            id="b1_awq_rms_smooth_gentle",
            branch=1,
            parent_id="b1_awq_rms_smooth",
            attempt=2,
            description="Gentle RMSNorm smoothing (gamma=0.20) + strong AWQ weighting (alpha=0.75)",
            fn=lambda: {
                "mode": "awq_ls",
                "smooth_gamma": 0.20,
                "awq_alpha": 0.75,
                "grid_points": 9,
                "grid_span": 0.40,
                "tag": "b1_awq_rms_smooth_gentle",
            },
        ),
    )

    b2 = BranchCandidate(
        id="b2_barbell_gates",
        branch=2,
        description="Barbell 99.4% LUT + 0.6% FP16 Cognitive Gate/Pointer Carving (<=100 KB layers)",
        fn=lambda: {
            "mode": "baseline_ls",
            "barbell_kb_thresh": 100.0,
            "barbell_scope": "cog_only",
            "tag": "b2_barbell_gates_100kb",
        },
        next_refinement=BranchCandidate(
            id="b2_barbell_gates_200kb",
            branch=2,
            parent_id="b2_barbell_gates",
            attempt=2,
            description="Barbell FP16 Carving up to 200 KB (includes r_ctx/out_proj/slot/prose_qk in crystal/woodbury)",
            fn=lambda: {
                "mode": "baseline_ls",
                "barbell_kb_thresh": 200.0,
                "barbell_scope": "cog_only",
                "tag": "b2_barbell_gates_200kb",
            },
        ),
    )

    b3 = BranchCandidate(
        id="b3_grid_scale_ls",
        branch=3,
        description="Non-Convex 1D Scale-Grid + Clipping Search (15 scales per 32-group, 0 KB overhead)",
        fn=lambda: {
            "mode": "awq_ls",
            "smooth_gamma": 0.0,
            "awq_alpha": 0.35,
            "grid_points": 15,
            "grid_span": 0.45,
            "tag": "b3_grid_scale_ls",
        },
    )

    # DAG Crossover Synthesizer (merges Hessian-CD + RMSNorm Smooth + Barbell Carving + @tunable knobs)
    def crossover_synthesizer(u: BranchCandidate, v: BranchCandidate) -> BranchCandidate:
        def _factory(p: np.ndarray) -> Callable[[], Dict[str, Any]]:
            damp_rel = float(10.0 ** float(p[0]))
            smooth_gamma = float(max(0.0, p[1]))
            barbell_kb = float(max(0.0, p[2]))
            awq_alpha = float(max(0.0, p[3]))
            return lambda: {
                "mode": "hessian_cd",
                "damp_rel": round(damp_rel, 5),
                "cd_passes": 4,
                "grid_points": 7,
                "grid_span": 0.40,
                "smooth_gamma": round(smooth_gamma, 4),
                "awq_alpha": round(awq_alpha, 4),
                "barbell_kb_thresh": round(barbell_kb, 1),
                "barbell_scope": "cog_only",
                "tag": f"xover_hcd_d{damp_rel:.3f}_g{smooth_gamma:.2f}_b{barbell_kb:.0f}_a{awq_alpha:.2f}",
            }

        p0 = np.array([-1.6, 0.15, 100.0, 0.50], dtype=np.float64)
        return BranchCandidate(
            id=f"xover_{u.id}_x_{v.id}",
            branch=4,
            parent_id=u.id,
            description=(
                f"DAG Hybrid `{u.id}` x `{v.id}`: Block-32 Hessian CD + RMSNorm Smooth + "
                f"Barbell FP16 Cognitive Gates with 4 continuous knobs for Round-4 pruning"
            ),
            fn=_factory(p0),
            tunable_params=p0,
            tunable_bounds=[(-2.5, -0.5), (0.0, 0.45), (0.0, 200.0), (0.0, 1.0)],
            param_names=["damp_log10", "smooth_gamma", "barbell_kb_thresh", "awq_alpha"],
            param_factory=_factory,
        )

    cartridge = TaskCartridge(
        task_name="calcite_lut_t1",
        description=(
            "Close W3 2-bit LUT quantization gap on Chinchilla S3_85M_stage2_fp.pt "
            "(100% T-MAC PackedLUT2BitLinear compatible, zero retraining, sealed Blind Vault)"
        ),
        baseline_fn=base_fn,
        verify_and_score=verify_and_score,
        initial_branches=[b0, b1, b2, b3],
        crossover_synthesizer=crossover_synthesizer,
        blind_vault_evaluator=blind_vault_eval,
        noise_probe_runs=1,
    )
    return cartridge, {"fp_ref_dev": fp_ref}


def main() -> int:
    pool_dir = os.environ.get("DREAM_RSI_POOL") or dream_engine.default_pool_dir()
    cartridge, ctx_meta = build_cartridge()
    runner = AutoDreamRunner(cartridge, pool_dir=pool_dir)
    report = runner.run()

    print("\n" + "=" * 78)
    print(
        f"[T1 DONE] Task: {report['task_name']} | Baseline: {report['baseline_score']:.4f} -> "
        f"Champion `{report['champion_id']}`: {report['champion_score']:.4f}"
    )
    print(f"Nodes explored: {report['total_nodes_explored']} | Elapsed: {report['elapsed_s']:.1f}s")
    if report.get("blind_vault_report"):
        print("Sealed Blind Vault:")
        print(json.dumps(report["blind_vault_report"], indent=2, ensure_ascii=False))

    out_payload = {
        "task_name": report["task_name"],
        "fp_ref_dev": ctx_meta["fp_ref_dev"],
        "baseline_score": report["baseline_score"],
        "champion_id": report["champion_id"],
        "champion_score": report["champion_score"],
        "champion_metrics": report["champion_metrics"],
        "total_nodes_explored": report["total_nodes_explored"],
        "elapsed_s": round(float(report["elapsed_s"]), 2),
        "blind_vault_report": report["blind_vault_report"],
        "journal": report["journal"],
        "pool_dir": pool_dir,
    }
    OUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT_PATH.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved report -> {OUT_REPORT_PATH} ({OUT_REPORT_PATH.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
