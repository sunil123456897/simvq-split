"""
Phase P-1: Controlled Single-Axis `S1_5M` Breakthrough Tournament + Auto-Combo + `2-bit LUT` QAT Stress Check
(`scripts/run_phase_p_5m_tournament.py`).

Strict `epistemic-ml-trading.md` / `calcite-phase-runner` discipline:
- Single `seed=2026` and 100% identical `(OpenWeb 75% + Cognitive S5/OV/Trap 15% + MQAR 10%)`
  stream matching Phase O (`700` FP steps, `B=64/mbs=32`, `L=512`, `compile_calcite_v2`,
  `WSDLRSchedule(warm=40, decay=250)`, `MuonAdamWHybrid(ns_schedule="quintic5")`).
- Gate-2 init-std, 100% finite live gradient, and `0.00%` mid-UTF8 cut tracer before EVERY arm.
- Base Control `P0_base_bn05`: `Calcite v3.1 (5M)` + Phase O strict Pareto winner `dw_bottleneck_ratio=0.5`.
- 5 Orthogonal Single-Axis Arms (each changes ONLY ONE axis vs `P0_base_bn05`):
    * `P1_w16_chunk`:       `window_size=16`
    * `P2_byte_swa32_h4`:   `byte_swa_window=32, byte_swa_dim=128` (`4 heads x D=32`)
    * `P3_engram_4gram_8k`: `engram_slots=8192, engram_use_4gram=True, engram_dim=32`
    * `P4_dca_depth`:       `use_dca=True` (`DeepCrossAttentionMix`, `+0 B` O(1) RAM)
    * `P5_dw_ratio_075`:    `dw_bottleneck_ratio=0.75`
- Automatic `P6_combo_winners`: combines ALL axes from `P1..P5` that strictly beat `P0_base_bn05`
  by `> 2*sigma_ref` (`Val delta <= -0.0057 BPB`) without regressing `OW6` or `Needle` (`P5` replaces
  `0.5` only if it strictly beats `0.5`).
- `250`-step Copy-Distilled `4-Tap QAT` (Law 8c: `12.5%` prefix-masked MQAR copy weight + dedicated
  `bic_q` LR `2.5e-4 * w_scale` + `AdamW(fused=True)` cache-invalidation hook) + `2-bit LUT` packing +
  full `Tier-1` eval (`Val`, `OW6`, `L=2048`, `S5`, `OV_L`, `Needle 512..8K + OrigRef`, `OverwriteNeedle`)
  + `CPU AVX-512` & `GPU CUDA Graph` streaming `B/s` (`max_err < 2e-5`, strict `O(1)` RAM) on BOTH
  `P0_base_bn05` and `P6_combo_winners`.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from calcite_v2 import (  # noqa: E402
    CalciteV2Config,
    CalciteV2LM,
    build_calcite_v2_tier_config,
    compute_v2_state_cache_kb,
)
from experiments.scaling_phase_k.eval_suite import (  # noqa: E402
    BatchPrefetcher,
    has_usable_cuda,
    load_utf8_corpus_array,
    sample_memmap_batch,
    sync_if_cuda,
)
from kernels.fast_calcite_scans import (  # noqa: E402
    clear_fast_qat_cache,
    compile_calcite_v2,
    unwrap_compiled_state_dict,
)
from kernels.fast_packed_lut import (  # noqa: E402
    FastPackedLUT2BitLinear,
    get_lut_backend_info,
    pack_fast_model_2bit_lut,
)
from scripts.colab_phase_k_scaling_runner import (  # noqa: E402
    _assert_router_unchanged,
    _attach_4tap_hooks,
    _cosine_lr,
    _make_adamw,
    _run_4tap_qat_step,
    _snapshot_router_params,
)
from scripts.colab_phase_l_v2_runner import (  # noqa: E402
    build_eval_bundles,
    ensure_phase_k_corpus,
    load_frozen_router_into,
    set_all_seeds,
)
from scripts.experiment_velocity import (  # noqa: E402
    WSDLRSchedule,
    run_tiered_eval,
)
from scripts.muon_opt import MuonAdamWHybrid  # noqa: E402
from scripts.run_phase_m_stage_c_v3_5m import _split_v3_qat_4groups  # noqa: E402
from scripts.run_phase_n_v31_lm_check_5m import eval_overwrite_needle_ood  # noqa: E402
from scripts.run_phase_o_tier2_ablations_5m import (  # noqa: E402
    _measure_muon_adamw_split,
    benchmark_arm_lut_streaming,
    run_gate2_init_probe,
    sample_tournament_batch,
)
from simvq_split.memory import (  # noqa: E402
    compute_2bit_lut_footprint,
    quantize_ternary_weights_ls,
)
from simvq_split.tokenizer import ByteBPETokenizer  # noqa: E402


TWO_SIGMA_REF_BPB: float = 0.0057

# Ordered single-axis arms vs P0_base_bn05 (each arm changes ONLY ONE orthogonal axis)
SINGLE_AXIS_SPECS: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("P0_base_bn05", {}),
    ("P1_w16_chunk", {"window_size": 16}),
    ("P2_byte_swa32_h4", {"byte_swa_window": 32, "byte_swa_dim": 128}),
    ("P3_engram_4gram_8k", {"engram_slots": 8192, "engram_use_4gram": True, "engram_dim": 32}),
    ("P4_dca_depth", {"use_dca": True}),
    ("P5_dw_ratio_075", {"dw_bottleneck_ratio": 0.75}),
)


# ---------------------------------------------------------------------------
# 1. Config Builder & Summary Extraction
# ---------------------------------------------------------------------------


def build_phase_p_5m_config(
    *,
    window_size: int = 8,
    byte_swa_window: int = 24,
    byte_swa_dim: int = 64,
    engram_slots: int = 4096,
    engram_dim: int = 32,
    engram_use_4gram: bool = False,
    use_dca: bool = False,
    dw_bottleneck_ratio: float = 0.5,
    dec_dw3_enabled: bool = True,
    bf16_residual_stream: bool = False,
    smoke: bool = False,
) -> CalciteV2Config:
    """Builds `Calcite v3.1 (5M)` with Phase O `dw_bottleneck_ratio=0.5` base + opt-in Phase P axes."""
    tier_sig = inspect.signature(build_calcite_v2_tier_config).parameters
    kwargs: Dict[str, Any] = dict(
        topology="hourglass",
        use_scaled_decoder=True,
        use_kalman_precond=False,
        use_latent_feedback=False,
        use_clean_gates=False,
        window_size=int(window_size),
        use_sym_byte_swa=True,
        byte_swa_window=int(byte_swa_window),
        byte_swa_dim=int(byte_swa_dim),
        use_byte_engram=True,
        engram_slots=int(engram_slots),
        engram_dim=int(engram_dim),
        use_byte_induct_cache=True,
        byte_induct_slots=128 if smoke else 512,
        byte_induct_dim=16,
        dec_dw3_enabled=bool(dec_dw3_enabled),
        dw_bottleneck_ratio=float(dw_bottleneck_ratio),
        bf16_residual_stream=bool(bf16_residual_stream),
    )
    if "engram_use_4gram" in tier_sig:
        kwargs["engram_use_4gram"] = bool(engram_use_4gram)
    if "use_dca" in tier_sig:
        kwargs["use_dca"] = bool(use_dca)

    cfg = build_calcite_v2_tier_config("S1_5M", **kwargs)
    if "engram_use_4gram" not in tier_sig:
        cfg.engram_use_4gram = bool(engram_use_4gram)
    if "use_dca" not in tier_sig:
        cfg.use_dca = bool(use_dca)

    if smoke:
        # Keep head_dim = 64 (`128 / 2`) so native CUDA `swa_rope_attn<64, W>` still triggers!
        cfg.d_model, cfg.n_heads, cfg.d_ff = 128, 2, 256
        cfg.byte_char_dim, cfg.byte_d_ff = 16, 128
    return cfg


def _cfg_axis_dict(cfg: CalciteV2Config) -> Dict[str, Any]:
    return {
        "window_size": int(cfg.window_size),
        "byte_swa_window": int(cfg.byte_swa_window),
        "byte_swa_dim": int(cfg.byte_swa_dim),
        "engram_slots": int(cfg.engram_slots),
        "engram_dim": int(cfg.engram_dim),
        "engram_use_4gram": bool(getattr(cfg, "engram_use_4gram", False)),
        "use_dca": bool(getattr(cfg, "use_dca", False)),
        "dw_bottleneck_ratio": float(cfg.dw_bottleneck_ratio),
        "dec_dw3_enabled": bool(cfg.dec_dw3_enabled),
        "bf16_residual_stream": bool(cfg.bf16_residual_stream),
        "byte_induct_slots": int(cfg.byte_induct_slots),
    }


def _extract_tier1_summary(t1: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Extracts complete Phase P summary metrics (`Val`, `OW6`, `L2048`, `S5`, `OV_L`, `Needle 512..8K + OrigRef`)."""
    cog = t1.get("cognitive") or {}
    ndl = t1.get("ood_verbatim_ladder") or {}
    ext = t1.get("extrap_4x") or {}
    ow6 = t1.get("openweb_6") or {}

    n512 = ndl.get("L_512") or ndl.get("L_256") or {}
    n1500 = ndl.get("L_1500") or ndl.get("L_256") or {}
    n4096 = ndl.get("L_4096") or {}
    n8192 = ndl.get("L_8192") or {}
    orig = ndl.get("OrigRefCode_1500B") or {}

    ndl_entries = [x for x in (n512, n1500, n4096, n8192, orig) if x.get("exact_match_pct") is not None]
    mean_ex = (
        round(float(np.mean([float(x["exact_match_pct"]) for x in ndl_entries])), 2)
        if ndl_entries
        else None
    )
    mean_ba = (
        round(float(np.mean([float(x["byte_acc_pct"]) for x in ndl_entries if x.get("byte_acc_pct") is not None])), 2)
        if ndl_entries
        else None
    )

    return {
        "val_bpb": t1.get("val_bpb"),
        "ow6_bpb": t1.get("ow6_mean_bpb"),
        "max_mid_utf8_split_pct": ow6.get("max_mid_utf8_split_pct"),
        "l2048_bpb": ext.get("bpb_L2048", ext.get("bpb_L1024")),
        "s5_k2_pct": ((cog.get("s5") or {}).get("k2") or {}).get("exact_pct"),
        "s5_sum_pct": cog.get("s5_sum"),
        "ov_long_pct": (cog.get("ov_ood_long") or {}).get("exact_pct"),
        "sel_f1_long_pct": cog.get("selective_f1_ood_long"),
        "needle_512_exact_pct": n512.get("exact_match_pct"),
        "needle_512_byte_pct": n512.get("byte_acc_pct"),
        "needle_1500_exact_pct": n1500.get("exact_match_pct"),
        "needle_1500_byte_pct": n1500.get("byte_acc_pct"),
        "needle_4096_exact_pct": n4096.get("exact_match_pct"),
        "needle_4096_byte_pct": n4096.get("byte_acc_pct"),
        "needle_8192_exact_pct": n8192.get("exact_match_pct"),
        "needle_8192_byte_pct": n8192.get("byte_acc_pct"),
        "orig_ref_1500_exact_pct": orig.get("exact_match_pct"),
        "orig_ref_1500_byte_pct": orig.get("byte_acc_pct"),
        "needle_mean_exact_pct": mean_ex,
        "needle_mean_byte_pct": mean_ba,
    }


def _check_utf8_zero_mid_cuts(model: CalciteV2LM, device: str) -> Dict[str, Any]:
    """1-batch multilingual tracer verifying 0.00% mid-UTF8 splits across ASCII/Cyrillic/CJK/Emoji."""
    samples = [
        "Calcite v3.2 UTF-8 safety: Привет, мир! 東京 2026 🚀🌌🧪 def f(x): return x * 2\n" * 4,
        "Проверка многобайтовых символов: 漢字カナ交じり文 🧭🔋💡 {\"key\": [1, 2, 3]} ∑∏∫√∞\n" * 4,
    ]
    enc = [[b + 3 for b in s.encode("utf-8")[:128]] for s in samples]
    xb = torch.tensor(enc, dtype=torch.long, device=device)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            _, _, ctx = model.byte_chunker.encode_to_chunks(model.tok_emb(xb), xb)
            raw_b = (xb - 3).clamp(min=0)
            is_cont = (xb >= 3) & ((raw_b & 0xC0) == 0x80)
            mid_cuts = int((ctx.b_mask & is_cont).sum().item())
            total_b = int(ctx.b_mask.sum().item())
    finally:
        model.train(was_training)
    return {
        "mid_utf8_cuts": mid_cuts,
        "total_boundaries": total_b,
        "zero_mid_utf8_cuts": mid_cuts == 0,
    }


def _compute_std_ratio_vs_ref(
    cur_stds: Dict[str, float], ref_stds: Optional[Dict[str, float]]
) -> Optional[float]:
    if not ref_stds:
        return None
    ratios = []
    for k, v in cur_stds.items():
        rv = ref_stds.get(k)
        if rv is not None and rv > 1e-6 and v > 0.0:
            ratios.append(max(v / rv, rv / v))
    return round(max(ratios), 4) if ratios else None


# ---------------------------------------------------------------------------
# 2. Core Single-Axis FP Arm Trainer (Exact Phase O Stream & WSD Schedule)
# ---------------------------------------------------------------------------


def train_fp_arm(
    arm_name: str,
    cfg: CalciteV2Config,
    router_state: Dict[str, torch.Tensor],
    train_mm: np.ndarray,
    val_mm: np.ndarray,
    tok: ByteBPETokenizer,
    tr_sents: List[str],
    eval_wins: Dict[str, List[List[int]]],
    s5_suites: Dict[str, list],
    ov_suites: Dict[str, list],
    trap_suites: Dict[str, list],
    *,
    device: str,
    total_steps: int,
    warmup_steps: int,
    decay_steps: int,
    batch_size: int,
    micro_bs: int,
    seq_len: int,
    seed: int,
    base_lr: float = 1.8e-3,
    min_lr: float = 1.2e-4,
    use_compile: bool = True,
    ref_init_stds: Optional[Dict[str, float]] = None,
    run_sigma_cal: bool = False,
    smoke: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    print(f"\n==================== [{arm_name}] ====================", flush=True)
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
    n_cog = max(1, int(round(batch_size * 0.15))) if batch_size >= 8 else 1
    n_copy = max(1, int(round(batch_size * 0.10))) if batch_size >= 8 else 1

    set_all_seeds(seed)
    model = CalciteV2LM(cfg).to(device)
    load_frozen_router_into(model, router_state)
    router_snap = _snapshot_router_params(model)
    n_params = int(sum(p.numel() for p in model.parameters()))
    cache_kb = round(compute_v2_state_cache_kb(cfg, 32768), 2)

    # Gate-2 Init-Std, 100% live finite gradients, and 0 mid-UTF8 cuts tracer
    utf8_chk = _check_utf8_zero_mid_cuts(model, device)
    assert utf8_chk["zero_mid_utf8_cuts"], f"[{arm_name}] Mid-UTF8 cuts detected: {utf8_chk}"

    rng_probe_py = random.Random(seed + 999)
    rng_probe_np = np.random.default_rng(seed + 999)
    xb_p, yb_p, _ = sample_tournament_batch(
        train_mm, tok, tr_sents, min(8, batch_size), seq_len, rng_probe_py, rng_probe_np, 1, 1
    )
    gate2 = run_gate2_init_probe(model, xb_p.to(device), yb_p.to(device), device, use_amp)
    gate2["utf8_tracer"] = utf8_chk
    std_ratio = _compute_std_ratio_vs_ref(gate2["block_std"], ref_init_stds)
    gate2["max_std_ratio_vs_p0"] = std_ratio
    assert gate2["all_grads_finite"], f"Gate-2 non-finite gradients in {arm_name}: {gate2}"
    if std_ratio is not None:
        assert std_ratio <= 1.5, f"Gate-2 init-std ratio > 1.5x in {arm_name}: {std_ratio} ({gate2['block_std']})"
    print(
        f"  [Gate-2 {arm_name}] params={n_params:,} | O1_cache={cache_kb:.2f}KB | loss={gate2['init_loss']} | "
        f"res_std={gate2['block_std'].get('cortex_post_last')} (ratio_vs_p0={std_ratio}) | "
        f"live_grads={gate2['live_finite_grad_tensors']}/{gate2['trainable_params_tensors']} | "
        f"mid_utf8_cuts={utf8_chk['mid_utf8_cuts']}",
        flush=True,
    )

    compiled_ok = False
    if use_compile and device == "cuda":
        compiled_ok = bool(compile_calcite_v2(model, mode="default", dynamic=False))

    opt = MuonAdamWHybrid(
        model,
        base_lr=base_lr,
        muon_lr_mult=6.0,
        weight_decay=0.04,
        device=device,
        woodbury_lr_mult=0.45,
        protect_crystal=False,
        ns_schedule="quintic5",
    )
    sched = WSDLRSchedule(
        base_lr=base_lr,
        min_lr=min_lr,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        decay_mode="cosine",
    )

    # Identical RNG seed across ALL arms (100% identical data stream to Phase O)
    rng_py = random.Random(seed + 3141)
    rng_np = np.random.default_rng(seed + 3141)
    n_acc = max(1, (batch_size + micro_bs - 1) // micro_bs)
    warmup_measure_steps = 4 if compiled_ok else 2
    chunk_span = 2 if smoke else 25

    def _sample_next():
        return sample_tournament_batch(
            train_mm, tok, tr_sents, batch_size, seq_len, rng_py, rng_np, n_cog, n_copy
        )

    t_train0 = time.perf_counter()
    t_steady0: Optional[float] = None
    steady_steps = 0
    chunk_ms_samples: List[float] = []
    chunk_t0: Optional[float] = None
    chunk_steps = 0
    grad_norms: Dict[int, float] = {}
    last_batch: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

    with BatchPrefetcher(
        _sample_next, device=device, total_steps=total_steps, prefetch_depth=4, enabled=(total_steps >= 4)
    ) as pf:
        for step in range(1, total_steps + 1):
            if step == warmup_measure_steps + 1:
                sync_if_cuda(device)
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                t_steady0 = time.perf_counter()
                chunk_t0 = t_steady0
                chunk_steps = 0

            model.train()
            model.set_crystal_schedule(step, total_steps)
            sched.apply(opt, step)
            xb_f, yb_f, mb_f = pf.next_batch()
            last_batch = (xb_f, yb_f, mb_f)

            opt.zero_grad(set_to_none=True)
            for st in range(0, batch_size, micro_bs):
                xb = xb_f[st : st + micro_bs]
                yb = yb_f[st : st + micro_bs]
                mb = mb_f[st : st + micro_bs]
                with torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                    lg = model(xb)
                    lg_f = lg.float()
                    tok_ce = F.cross_entropy(
                        lg_f.reshape(-1, lg_f.size(-1)),
                        yb.reshape(-1),
                        ignore_index=-100,
                        reduction="none",
                    ).view_as(yb)
                    valid = yb != -100
                    m_copy_f = (mb & valid).float()
                    m_pref_f = ((~mb) & valid).float()
                    n_c_tok = m_copy_f.sum()
                    n_p_tok = m_pref_f.sum()
                    ce_p = (tok_ce * m_pref_f).sum() / n_p_tok.clamp(min=1.0)
                    has_c = n_c_tok > 0.0
                    ce_c = torch.where(
                        has_c,
                        (tok_ce * m_copy_f).sum() / n_c_tok.clamp(min=1.0),
                        torch.zeros_like(ce_p),
                    )
                    ce = torch.where(has_c, 0.85 * ce_p + 0.15 * ce_c, ce_p)
                    loss = (ce + 0.04 * model.get_commitment_loss()) / float(n_acc)
                loss.backward()

            gn_t = torch.nn.utils.clip_grad_norm_(opt.trainable_params, 1.0)
            opt.step()

            if t_steady0 is not None:
                steady_steps += 1
                chunk_steps += 1
                if chunk_steps >= chunk_span and chunk_t0 is not None:
                    sync_if_cuda(device)
                    now_c = time.perf_counter()
                    chunk_ms_samples.append(1000.0 * (now_c - chunk_t0) / float(chunk_steps))
                    chunk_t0 = now_c
                    chunk_steps = 0

            if step in (1, total_steps) or (step % max(1, total_steps // 4) == 0):
                gn_v = round(float(gn_t.item()), 3)
                grad_norms[step] = gn_v
                dt_so_far = time.perf_counter() - t_train0
                print(
                    f"  [{arm_name:20s}] step {step:4d}/{total_steps} | lr={sched.lr_at(step):.2e} | "
                    f"gn={gn_v:6.3f} | elapsed={dt_so_far:5.1f}s",
                    flush=True,
                )

    sync_if_cuda(device)
    total_train_s = time.perf_counter() - t_train0
    if t_steady0 is not None and steady_steps > 0:
        raw_steady_ms = 1000.0 * (time.perf_counter() - t_steady0) / float(steady_steps)
    else:
        raw_steady_ms = 1000.0 * total_train_s / float(max(1, total_steps))
    steady_ms = float(np.median(chunk_ms_samples)) if chunk_ms_samples else raw_steady_ms
    compile_warmup_s = max(0.0, total_train_s - (steady_ms / 1000.0) * float(total_steps))
    steady_bps = (batch_size * seq_len) / max(1e-9, steady_ms / 1000.0)
    peak_alloc_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 1) if device == "cuda" else 0.0
    peak_res_mb = round(torch.cuda.max_memory_reserved() / (1024**2), 1) if device == "cuda" else 0.0

    opt_split = (
        _measure_muon_adamw_split(
            model, opt, last_batch[0], last_batch[1], last_batch[2], micro_bs, device, use_amp, n_iters=3 if smoke else 8
        )
        if last_batch is not None
        else {"muon_ns_ms": 0.0, "adamw_ms": 0.0, "opt_total_ms": 0.0}
    )

    router_mut_diff = _assert_router_unchanged(model, router_snap, f"{arm_name}_FP")
    trained_sd_cpu = unwrap_compiled_state_dict(model)
    del model, opt
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # Clean unwrapped Tier-1 evaluation (zero Inductor eval-shape recompiles)
    eval_model = CalciteV2LM(cfg).to(device)
    eval_model.load_state_dict(trained_sd_cpu, strict=True)
    eval_model.byte_chunker.lock_router(True)
    eval_model.set_crystal_schedule(1, 1)
    eval_model.set_snap_alpha(1.0)
    eval_model.eval()

    t1 = run_tiered_eval(
        eval_model,
        tier=1,
        val_mm=val_mm,
        eval_wins=eval_wins,
        tok=tok,
        s5_suites=s5_suites,
        ov_suites=ov_suites,
        trap_suites=trap_suites,
        device=device,
        seq_len=seq_len,
        eval_bs=16,
        smoke=smoke,
    )

    sigma_cal: Optional[Dict[str, Any]] = None
    if run_sigma_cal:
        rng_cal = np.random.default_rng(77777)
        bpb_samples: List[float] = []
        n_cal_b = 4 if smoke else 32
        with torch.inference_mode():
            for _ in range(n_cal_b):
                xb_v, yb_v = sample_memmap_batch(val_mm, 16, seq_len, rng_cal, device)
                with torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                    lg_v = eval_model(xb_v)
                ce_v = float(
                    F.cross_entropy(lg_v.float().reshape(-1, lg_v.size(-1)), yb_v.reshape(-1), ignore_index=0).item()
                )
                bpb_samples.append(ce_v / math.log(2.0))
        arr_b = np.asarray(bpb_samples, dtype=np.float64)
        sem = float(arr_b.std(ddof=1) / math.sqrt(len(arr_b))) if len(arr_b) > 1 else 0.0
        sigma_cal = {
            "n_batches": n_cal_b,
            "mean_bpb": round(float(arr_b.mean()), 4),
            "batch_std_bpb": round(float(arr_b.std(ddof=1)), 4) if len(arr_b) > 1 else 0.0,
            "sem_bpb": round(sem, 5),
            "two_sigma_bpb": round(2.0 * sem, 5),
            "phase_m_ref_two_sigma_bpb": TWO_SIGMA_REF_BPB,
        }

    summary = _extract_tier1_summary(t1)
    print(
        f"  [{arm_name:20s} DONE] {steady_ms:.2f}ms/st ({steady_bps:,.0f} B/s) | VRAM={peak_alloc_mb:.0f}MB | "
        f"Val={summary['val_bpb']} OW6={summary['ow6_bpb']} L2K={summary['l2048_bpb']} | "
        f"S5k2={summary['s5_k2_pct']}% OVL={summary['ov_long_pct']}% | "
        f"Ndl(512/1.5K/4K/8K/Orig)={summary['needle_512_exact_pct']}/{summary['needle_1500_exact_pct']}/"
        f"{summary['needle_4096_exact_pct']}/{summary['needle_8192_exact_pct']}/{summary['orig_ref_1500_exact_pct']}% "
        f"(mean={summary['needle_mean_exact_pct']}% ex / {summary['needle_mean_byte_pct']}% b)",
        flush=True,
    )

    del eval_model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    arm_out: Dict[str, Any] = {
        "arm": arm_name,
        "params": n_params,
        "cache_kb_o1_l32k": cache_kb,
        "axis_config": _cfg_axis_dict(cfg),
        "router_max_abs_diff": router_mut_diff,
        "gate2_init": gate2,
        "train": {
            "steps_completed": int(total_steps),
            "total_train_s": round(total_train_s, 2),
            "compile_warmup_s": round(compile_warmup_s, 2),
            "steady_ms_per_step": round(steady_ms, 2),
            "raw_steady_ms_per_step": round(raw_steady_ms, 2),
            "steady_bytes_per_sec": round(steady_bps, 1),
            "peak_alloc_mb": peak_alloc_mb,
            "peak_reserved_mb": peak_res_mb,
            "opt_split_ms": opt_split,
            "grad_norms": grad_norms,
            "compiled": compiled_ok,
        },
        "summary": summary,
        "eval_tier1": t1,
        "two_sigma_calibration": sigma_cal,
        "lut_streaming": None,
    }
    return arm_out, trained_sd_cpu


# ---------------------------------------------------------------------------
# 3. Automatic Winner Selection for `P6_combo_winners`
# ---------------------------------------------------------------------------


def select_combo_winners(
    arms: Dict[str, Dict[str, Any]],
    *,
    two_sigma_ref: float = TWO_SIGMA_REF_BPB,
    smoke: bool = False,
) -> Dict[str, Any]:
    """
    Applies the pre-registered Phase P promotion rule against `P0_base_bn05`:
    - `delta_val_bpb <= -two_sigma_ref` (`<= -0.0057 BPB`)
    - `delta_ow6_bpb <= +two_sigma_ref` (no OW6 regression beyond noise floor)
    - `Needle` no regression: `mean_exact >= p0 - 5.0%`, `mean_byte >= p0 - 2.0%`,
      and `OrigRef byte >= p0 - 5.0%` (catches single-mode needle collapse).
    - `P5_dw_ratio_075` replaces `dw_bottleneck_ratio=0.5` ONLY if it strictly passes.
    """
    p0_s = arms["P0_base_bn05"]["summary"]
    p0_val = float(p0_s["val_bpb"])
    p0_ow6 = float(p0_s["ow6_bpb"])
    p0_n_ex = float(p0_s["needle_mean_exact_pct"] or 0.0)
    p0_n_ba = float(p0_s["needle_mean_byte_pct"] or 0.0)
    p0_orig_ba = float(p0_s["orig_ref_1500_byte_pct"] or 0.0)

    spec_map = {nm: kw for nm, kw in SINGLE_AXIS_SPECS}
    per_arm_gates: Dict[str, Dict[str, Any]] = {}
    winners: List[str] = []

    for arm_nm, _ in SINGLE_AXIS_SPECS:
        if arm_nm == "P0_base_bn05" or arm_nm not in arms:
            continue
        s = arms[arm_nm]["summary"]
        d_val = round(float(s["val_bpb"]) - p0_val, 4)
        d_ow6 = round(float(s["ow6_bpb"]) - p0_ow6, 4)
        a_n_ex = float(s["needle_mean_exact_pct"] or 0.0)
        a_n_ba = float(s["needle_mean_byte_pct"] or 0.0)
        a_orig_ba = float(s["orig_ref_1500_byte_pct"] or 0.0)
        d_n_ex = round(a_n_ex - p0_n_ex, 2)
        d_n_ba = round(a_n_ba - p0_n_ba, 2)
        d_orig_ba = round(a_orig_ba - p0_orig_ba, 2)

        val_pass = d_val <= -two_sigma_ref
        ow6_pass = d_ow6 <= two_sigma_ref
        needle_pass = (d_n_ex >= -5.0) and (d_n_ba >= -2.0) and (d_orig_ba >= -5.0)
        passed = bool(val_pass and ow6_pass and needle_pass)
        per_arm_gates[arm_nm] = {
            "delta_val_bpb": d_val,
            "delta_ow6_bpb": d_ow6,
            "delta_needle_mean_exact_pct": d_n_ex,
            "delta_needle_mean_byte_pct": d_n_ba,
            "delta_orig_ref_byte_pct": d_orig_ba,
            "val_beats_2sigma": val_pass,
            "ow6_no_regress": ow6_pass,
            "needle_no_regress": needle_pass,
            "promoted_to_combo": passed,
        }
        if passed:
            winners.append(arm_nm)

    if smoke:
        # In smoke mode, ALWAYS exercise the full combined axis stack (P1+P2+P3+P4)
        combo_kwargs: Dict[str, Any] = {}
        smoke_axes = ["P1_w16_chunk", "P2_byte_swa32_h4", "P3_engram_4gram_8k", "P4_dca_depth"]
        for nm in smoke_axes:
            combo_kwargs.update(spec_map[nm])
        return {
            "mode": "smoke_full_combo",
            "two_sigma_ref_bpb": two_sigma_ref,
            "selected_arms": smoke_axes,
            "strict_gate_winners": winners,
            "combo_kwargs": combo_kwargs,
            "per_arm_gates": per_arm_gates,
        }

    combo_kwargs = {}
    for nm in winners:
        combo_kwargs.update(spec_map[nm])

    if len(winners) >= 2:
        mode = "combo_2plus"
    elif len(winners) == 1:
        mode = "single_winner_reused"
    else:
        # Fallback informational check: pick best Val arm (if any beat P0 at all) so QAT still stress-tests top candidate
        cand_names = [nm for nm, _ in SINGLE_AXIS_SPECS if nm != "P0_base_bn05" and nm in arms]
        best_nm = min(cand_names, key=lambda n: float(arms[n]["summary"]["val_bpb"]))
        combo_kwargs = dict(spec_map[best_nm])
        mode = f"no_strict_winner_fallback_{best_nm}"

    return {
        "mode": mode,
        "two_sigma_ref_bpb": two_sigma_ref,
        "selected_arms": winners,
        "combo_kwargs": combo_kwargs,
        "per_arm_gates": per_arm_gates,
    }


# ---------------------------------------------------------------------------
# 4. Copy-Distilled 4-Tap QAT (Law 8c) -> 2-bit LUT Packing & Full Eval
# ---------------------------------------------------------------------------


def _split_v32_qat_5groups(
    model: CalciteV2LM,
) -> Tuple[List[nn.Parameter], List[nn.Parameter], List[nn.Parameter], List[nn.Parameter], List[nn.Parameter]]:
    """
    5-Group QAT partition for `Calcite v3.1 / v3.2`:
    - `gain_q`: non-quantized FP params (`tok_emb`, `RMSNorm/CalciteRMSNorm`, `nn.Conv1d`,
      `byte_offset_emb`, `stream_gate/cryst2_gate`, `engram_emb2/3/4`, `engram_gate`, `dca`).
    - `cryst_q`: `crystal`, `crystal_2`, and `woodbury` scalar thresholds/lags.
    - `chunk_q`: quantized `byte_chunker` linear weights (excluding `byte_induct_cache`).
    - `core_q`: quantized `cortex_pre`, `cortex_post`, `mid_mlp`, `woodbury` 2D projections.
    - `bic_q`: dedicated `byte_induct_cache` group (`2.5e-4 * w_scale` per Law 8c).
    """
    gain_p, cryst_p, chunk_p, core_p = _split_v3_qat_4groups(model)

    # Ensure `engram_emb4` and all `dca` (FP routing weights) live cleanly in `gain_p`
    extra_gain_ids: set[int] = set()
    emb4 = getattr(model.byte_chunker, "engram_emb4", None)
    if emb4 is not None:
        for p in emb4.parameters():
            if p.requires_grad:
                extra_gain_ids.add(id(p))
    for nm, p in model.named_parameters():
        if p.requires_grad and (".dca" in nm or nm.startswith("dca") or "deep_cross" in nm):
            extra_gain_ids.add(id(p))

    if extra_gain_ids:
        gain_ids = {id(p) for p in gain_p}
        cryst_p = [p for p in cryst_p if id(p) not in extra_gain_ids]
        chunk_p = [p for p in chunk_p if id(p) not in extra_gain_ids]
        core_p = [p for p in core_p if id(p) not in extra_gain_ids]
        for _, p in model.named_parameters():
            if p.requires_grad and id(p) in extra_gain_ids and id(p) not in gain_ids:
                gain_p.append(p)
                gain_ids.add(id(p))

    # Split `byte_induct_cache` into dedicated gentle group `bic_p` (Law 8c)
    bic_mod = getattr(model.byte_chunker, "byte_induct_cache", None)
    bic_ids = {id(p) for p in bic_mod.parameters() if p.requires_grad} if bic_mod is not None else set()
    bic_p: List[nn.Parameter] = []
    if bic_ids:
        seen_bic: set[int] = set()
        for grp in (gain_p, cryst_p, chunk_p, core_p):
            for p in grp:
                if id(p) in bic_ids and id(p) not in seen_bic:
                    bic_p.append(p)
                    seen_bic.add(id(p))
        gain_p = [p for p in gain_p if id(p) not in bic_ids]
        cryst_p = [p for p in cryst_p if id(p) not in bic_ids]
        chunk_p = [p for p in chunk_p if id(p) not in bic_ids]
        core_p = [p for p in core_p if id(p) not in bic_ids]

    all_assigned = gain_p + cryst_p + chunk_p + core_p + bic_p
    expected_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assigned_ids = {id(p) for p in all_assigned}
    assert len(all_assigned) == len(assigned_ids), "Duplicate parameter detected across QAT groups!"
    assert assigned_ids == expected_ids, (
        f"QAT param group coverage mismatch: assigned={len(assigned_ids)} expected={len(expected_ids)}"
    )
    return gain_p, cryst_p, chunk_p, core_p, bic_p


def run_copy_distilled_qat_and_lut_eval(
    arm_name: str,
    cfg: CalciteV2Config,
    fp_sd_cpu: Dict[str, torch.Tensor],
    fp_summary: Dict[str, Optional[float]],
    train_mm: np.ndarray,
    val_mm: np.ndarray,
    tok: ByteBPETokenizer,
    tr_sents: List[str],
    eval_wins: Dict[str, List[List[int]]],
    s5_suites: Dict[str, list],
    ov_suites: Dict[str, list],
    trap_suites: Dict[str, list],
    *,
    device: str,
    qat_steps: int,
    batch_size: int,
    micro_bs_q: int,
    seq_len: int,
    seed: int,
    smoke: bool = False,
) -> Dict[str, Any]:
    """
    Runs Law-8c Copy-Distilled `4-Tap + Logit-KL STE QAT` (`250` steps), packs to `2-bit uint8 LUT`,
    verifies DCA FP immunity & router lock, runs full `Tier-1` + `OverwriteNeedle` eval, and measures
    `CPU AVX-512` + `GPU CUDA Graph` streaming `B/s` (`max_err < 2e-5`, strict `O(1)` RAM).
    """
    print(f"\n================ [2-BIT LUT QAT STRESS CHECK: {arm_name}] ================", flush=True)
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
    set_all_seeds(seed + 5050)

    teacher_m = CalciteV2LM(cfg).to(device)
    teacher_m.load_state_dict(fp_sd_cpu, strict=True)
    teacher_m.byte_chunker.use_static_m_cap = True
    teacher_m.byte_chunker.lock_router(True)
    teacher_m.set_crystal_schedule(1, 1)
    teacher_m.set_snap_alpha(1.0)
    for p in teacher_m.parameters():
        p.requires_grad_(False)
    teacher_m.eval()

    qat_m = CalciteV2LM(cfg).to(device)
    qat_m.load_state_dict(fp_sd_cpu, strict=True)
    qat_m.byte_chunker.use_static_m_cap = True
    qat_m.byte_chunker.lock_router(True)
    qat_m.set_crystal_schedule(1, 1)
    qat_m.set_snap_alpha(1.0)
    qat_m.set_qat_mix(0.30)
    qat_m.set_ternary_qat(True)
    router_snap = _snapshot_router_params(qat_m)

    gain_q, cryst_q, chunk_q, core_q, bic_q = _split_v32_qat_5groups(qat_m)
    all_qp = gain_q + cryst_q + chunk_q + core_q + bic_q
    w_scale = (384.0 / float(cfg.d_model)) ** 0.5
    q_lrs = (1.2e-3 * w_scale, 8.0e-5, 1.1e-3 * w_scale, 4.5e-4 * w_scale, 2.5e-4 * w_scale)
    q_mins = (1.0e-4 * w_scale, 1.5e-5, 8.0e-5 * w_scale, 4.5e-5 * w_scale, 3.0e-5 * w_scale)
    opt_groups = []
    lr_pairs = []
    for grp, blr, mlr in zip((gain_q, cryst_q, chunk_q, core_q, bic_q), q_lrs, q_mins):
        if grp:
            opt_groups.append({"params": grp, "lr": blr})
            lr_pairs.append((blr, mlr))

    opt_q = _make_adamw(opt_groups, device, weight_decay=0.0)
    # Gate 3.5 invariant: `AdamW(fused=True)` skips `weight._version`; invalidate cache via post-hook!
    opt_q.register_step_post_hook(lambda *_a, **_kw: clear_fast_qat_cache())

    warm_q = min(25, max(1, qat_steps // 10))
    ramp_q = min(100, max(1, qat_steps // 3))
    kl_temp = 1.5
    taps_fp: List[torch.Tensor] = []
    taps_q: List[torch.Tensor] = []
    hk_fp = _attach_4tap_hooks(teacher_m, taps_fp, detach=True)
    hk_q = _attach_4tap_hooks(qat_m, taps_q, detach=False)

    # Identical Copy-Distilled QAT stream (Law 8c: 12.5% prefix-masked MQAR rows!)
    rng_q_py = random.Random(seed + 7777)
    rng_q_np = np.random.default_rng(seed + 7777)
    n_cog_q = max(1, int(round(batch_size * 0.15))) if batch_size >= 8 else 1
    n_copy_q = max(1, int(round(batch_size * 0.125)))
    cur_mbs_q = min(batch_size, int(micro_bs_q))
    t_q0 = time.perf_counter()

    def _sample_qat_batch():
        xb_q, yb_q, mb_q = sample_tournament_batch(
            train_mm, tok, tr_sents, batch_size, seq_len, rng_q_py, rng_q_np, n_cog_q, n_copy_q
        )
        # Law 8c: mask prefix tokens on injected MQAR rows so per-row CE+KL gives full ~12.5% weight to verbatim copy!
        is_copy_row = mb_q.any(dim=1, keepdim=True)
        yb_q = torch.where((yb_q == 0) | (is_copy_row & (~mb_q)), torch.full_like(yb_q, -100), yb_q)
        return xb_q, yb_q

    try:
        with BatchPrefetcher(
            _sample_qat_batch, device=device, total_steps=qat_steps, prefetch_depth=4, enabled=(qat_steps >= 4)
        ) as pf_q:
            for qs in range(1, qat_steps + 1):
                qat_m.train()
                mix = min(1.0, 0.30 + 0.70 * (qs / float(ramp_q)))
                qat_m.set_qat_mix(mix)
                clear_fast_qat_cache()
                for gi, (blr, mlr) in enumerate(lr_pairs):
                    opt_q.param_groups[gi]["lr"] = _cosine_lr(qs, qat_steps, warm_q, blr, mlr)
                xb_f, yb_f = pf_q.next_batch()
                cur_mbs_q, q_ce_sum_t = _run_4tap_qat_step(
                    teacher_m,
                    qat_m,
                    taps_fp,
                    taps_q,
                    xb_f,
                    yb_f,
                    cur_mbs_q,
                    opt_q,
                    -100,
                    kl_temp,
                    use_amp,
                    device,
                    f"{arm_name}-QAT qs={qs}",
                )
                torch.nn.utils.clip_grad_norm_(all_qp, 1.0)
                opt_q.step()
                clear_fast_qat_cache()
                if qs in (1, max(1, qat_steps // 2), qat_steps) or (qs % 50 == 0):
                    n_acc_q = max(1, (batch_size + cur_mbs_q - 1) // cur_mbs_q)
                    qce = float(q_ce_sum_t.item()) / float(n_acc_q) if q_ce_sum_t is not None else 0.0
                    print(
                        f"  [QAT {arm_name:18s}] step {qs:4d}/{qat_steps} | mix={mix:.2f} | "
                        f"ce={qce:.4f} | {(time.perf_counter()-t_q0):.1f}s",
                        flush=True,
                    )
    finally:
        for h in hk_fp + hk_q:
            h.remove()
        clear_fast_qat_cache()

    sync_if_cuda(device)
    qat_s = round(time.perf_counter() - t_q0, 2)
    router_mut_diff = _assert_router_unchanged(qat_m, router_snap, f"{arm_name}_QAT")

    qat_m.set_ternary_qat(False)
    qat_m.set_qat_mix(1.0)
    qat_sd_cpu = unwrap_compiled_state_dict(qat_m)
    del teacher_m, qat_m, opt_q
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # Snap ternary weights on CPU and pack to native 2-bit LUT
    clean_q = CalciteV2LM(cfg).cpu()
    clean_q.load_state_dict(qat_sd_cpu, strict=True)
    clean_q.byte_chunker.lock_router(True)
    clean_q.set_crystal_schedule(1, 1)
    clean_q.set_snap_alpha(1.0)
    with torch.no_grad():
        for nm, mod in clean_q.named_modules():
            if isinstance(mod, nn.Linear) and nm != "lm_head":
                if clean_q.byte_chunker.is_locked_router_module(mod):
                    continue
                wt, _, _ = quantize_ternary_weights_ls(mod.weight.data)
                mod.weight.data.copy_(wt)
    clean_q.eval()

    lut_cpu = pack_fast_model_2bit_lut(clean_q, backend="auto").eval()
    lut_foot = compute_2bit_lut_footprint(lut_cpu)
    n_lut_layers = sum(1 for m in lut_cpu.modules() if isinstance(m, FastPackedLUT2BitLinear))

    # Verify DCA routing weights stay FP (never packed into 2-bit LUT!)
    dca_fp_preserved = True
    if bool(getattr(cfg, "use_dca", False)):
        dca_mods = [
            m
            for nm, m in lut_cpu.named_modules()
            if hasattr(m, "w_dyn") and hasattr(m, "b_stat")
        ]
        dca_fp_preserved = bool(
            dca_mods
            and all(
                (not isinstance(m, FastPackedLUT2BitLinear))
                and m.w_dyn.is_floating_point()
                and m.b_stat.is_floating_point()
                for m in dca_mods
            )
        )
        assert dca_fp_preserved, f"[{arm_name}] DCA routing weights were not preserved in FP!"

    binfo = get_lut_backend_info()
    lut_bk = "cuda_ext" if (device == "cuda" and binfo["cuda_ext_available"]) else "auto"
    lut_gpu = pack_fast_model_2bit_lut(clean_q, backend=lut_bk).to(device).eval()

    t1_lut = run_tiered_eval(
        lut_gpu,
        tier=1,
        val_mm=val_mm,
        eval_wins=eval_wins,
        tok=tok,
        s5_suites=s5_suites,
        ov_suites=ov_suites,
        trap_suites=trap_suites,
        device=device,
        seq_len=seq_len,
        eval_bs=16,
        smoke=smoke,
    )
    ov_ndl_lut = eval_overwrite_needle_ood(
        lut_gpu, val_mm, device=device, haystack_bytes=256 if smoke else 1500, n_samples=4 if smoke else 16
    )
    del lut_gpu, lut_cpu
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # Exact CPU AVX-512 + GPU CUDA Graph streaming benchmark & parity verification (< 2e-5, strict O(1) RAM)
    stream_lut = benchmark_arm_lut_streaming(clean_q, cfg, val_mm, device=device, smoke=smoke)
    del clean_q
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    cpu_s = stream_lut["cpu_avx512_fused_lut"]
    gpu_s = stream_lut.get("gpu_cuda_graph_fused_lut") or {}
    assert cpu_s["strict_o1_ram"], f"[{arm_name} LUT] CPU streaming violated strict O(1) RAM!"
    assert cpu_s["vs_batch_max_err"] < 2e-5, f"[{arm_name} LUT] CPU stream vs batch err={cpu_s['vs_batch_max_err']}"
    if gpu_s:
        assert gpu_s["strict_o1_ram"], f"[{arm_name} LUT] GPU CG streaming violated strict O(1) RAM!"
        assert gpu_s["vs_batch_max_err"] < 2e-5, f"[{arm_name} LUT] GPU CG vs batch err={gpu_s['vs_batch_max_err']}"
        assert gpu_s["vs_eager_max_err"] < 2e-5, f"[{arm_name} LUT] GPU CG vs eager err={gpu_s['vs_eager_max_err']}"

    lut_sum = _extract_tier1_summary(t1_lut)

    def _gap(k: str) -> Optional[float]:
        lv, fv = lut_sum.get(k), fp_summary.get(k)
        return round(float(lv) - float(fv), 4) if (lv is not None and fv is not None) else None

    lut_gaps = {
        "delta_val_bpb": _gap("val_bpb"),
        "delta_ow6_bpb": _gap("ow6_bpb"),
        "delta_l2048_bpb": _gap("l2048_bpb"),
        "delta_needle_mean_exact_pct": _gap("needle_mean_exact_pct"),
        "delta_needle_mean_byte_pct": _gap("needle_mean_byte_pct"),
        "delta_orig_ref_exact_pct": _gap("orig_ref_1500_exact_pct"),
    }

    print(
        f"  [2-BIT LUT {arm_name:14s}] Val={lut_sum['val_bpb']} (gap {lut_gaps['delta_val_bpb']:+.4f}) | "
        f"OW6={lut_sum['ow6_bpb']} (gap {lut_gaps['delta_ow6_bpb']:+.4f}) | L2K={lut_sum['l2048_bpb']} | "
        f"Ndl(512/1.5K/4K/8K/Orig)={lut_sum['needle_512_exact_pct']}/{lut_sum['needle_1500_exact_pct']}/"
        f"{lut_sum['needle_4096_exact_pct']}/{lut_sum['needle_8192_exact_pct']}/{lut_sum['orig_ref_1500_exact_pct']}% | "
        f"OvNdl={ov_ndl_lut['new_exact_pct']}%/{ov_ndl_lut['stale_old_exact_pct']}% | "
        f"CPU={cpu_s['decode_bps']} B/s | GPU_CG={gpu_s.get('decode_bps')} B/s | "
        f"err={max(cpu_s['vs_batch_max_err'], gpu_s.get('vs_batch_max_err', 0.0), gpu_s.get('vs_eager_max_err', 0.0)):.1e} | "
        f"RAM={cpu_s['state_kb']}KB (O1={cpu_s['strict_o1_ram']})",
        flush=True,
    )

    return {
        "arm": arm_name,
        "axis_config": _cfg_axis_dict(cfg),
        "qat_steps": int(qat_steps),
        "qat_seconds": qat_s,
        "router_max_abs_diff": router_mut_diff,
        "n_packed_lut_layers": n_lut_layers,
        "dca_fp_preserved": dca_fp_preserved,
        "lut_footprint": lut_foot,
        "summary_lut": lut_sum,
        "gaps_lut_minus_fp": lut_gaps,
        "eval_tier1_lut": t1_lut,
        "overwrite_needle_1500b_lut": ov_ndl_lut,
        "lut_streaming": stream_lut,
    }


# ---------------------------------------------------------------------------
# 5. Main Phase P-1 Tournament Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase P-1: Controlled Single-Axis 5M Breakthrough Tournament + Auto-Combo + 2-bit LUT QAT"
    )
    ap.add_argument("--data-dir", type=Path, default=ROOT_DIR / "data_cache")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--warmup-steps", type=int, default=40)
    ap.add_argument("--decay-steps", type=int, default=250)
    ap.add_argument("--qat-steps", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--micro-bs", type=int, default=32)
    ap.add_argument("--qat-micro-bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--smoke", "--smoke-local", dest="smoke", action="store_true")
    args = ap.parse_args()

    out_json = (
        args.out_json
        if args.out_json is not None
        else (
            ROOT_DIR / "benchmarks" / ("smoke_phase_p_5m_tournament.json" if args.smoke else "phase_p_5m_tournament.json")
        )
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if has_usable_cuda() else "cpu"
    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram_gb < 12.0:
            torch.cuda.set_per_process_memory_fraction(0.90, 0)

    tok = ByteBPETokenizer(merges=[])
    seq_len = 128 if args.smoke else 512
    tr_sents, eval_wins, s5_suites, ov_suites, trap_suites = build_eval_bundles(
        ROOT_DIR, tok, n_eval=16 if args.smoke else 80, seq_len=seq_len
    )

    if args.smoke:
        rng_f = np.random.default_rng(args.seed)
        fake = rng_f.integers(32 + 3, 126 + 3, size=60_000, dtype=np.uint16)
        train_mm = val_mm = fake
        total_steps, warm_steps, decay_steps, qat_steps = 8, 2, 3, 4
        batch_size, micro_bs, qat_micro_bs = 8, 4, 4
    else:
        train_bin, val_bin = ensure_phase_k_corpus(args.data_dir)
        train_mm = load_utf8_corpus_array(train_bin)
        val_mm = load_utf8_corpus_array(val_bin)
        total_steps = int(args.steps)
        warm_steps = int(args.warmup_steps)
        decay_steps = int(args.decay_steps)
        qat_steps = int(args.qat_steps)
        batch_size = int(args.batch_size)
        micro_bs = int(args.micro_bs)
        qat_micro_bs = int(args.qat_micro_bs)

    router_state = torch.load(
        ROOT_DIR / "benchmarks" / "phase_l_v2" / "frozen_utf8_router.pt", map_location="cpu", weights_only=True
    )
    use_compile = (not args.no_compile) and (not args.smoke) and (device == "cuda")
    t_wall0 = time.perf_counter()

    results: Dict[str, Any] = {
        "meta": {
            "phase": "Phase_P1_SingleAxis_5M_Breakthrough_Tournament",
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "seed": int(args.seed),
            "total_steps": total_steps,
            "warmup_steps": warm_steps,
            "stable_end_step": total_steps - decay_steps,
            "decay_steps": decay_steps,
            "qat_steps": qat_steps,
            "batch_size": batch_size,
            "micro_bs": micro_bs,
            "qat_micro_bs": qat_micro_bs,
            "seq_len": seq_len,
            "compiled": use_compile,
            "two_sigma_ref_bpb": TWO_SIGMA_REF_BPB,
            "smoke": bool(args.smoke),
        },
        "arms": {},
        "deltas_vs_p0": {},
        "winner_selection": {},
        "lut_qat_stress_check": {},
    }

    def _save() -> None:
        results["meta"]["elapsed_wall_s"] = round(time.perf_counter() - t_wall0, 2)
        out_json.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    def _recompute_deltas() -> None:
        p0_a = results["arms"]["P0_base_bn05"]
        p0_s = p0_a["summary"]
        p0_t = p0_a["train"]
        p0_st = p0_a.get("lut_streaming") or {}
        results["deltas_vs_p0"] = {}

        def _diff(a, b, nd=4):
            return round(float(a) - float(b), nd) if (a is not None and b is not None) else None

        def _sp(a, b):
            return round(float(a) / max(1e-9, float(b)), 3) if (a is not None and b is not None) else None

        for arm_nm, arm_d in results["arms"].items():
            if arm_nm == "P0_base_bn05":
                continue
            s = arm_d["summary"]
            t = arm_d["train"]
            st = arm_d.get("lut_streaming") or {}
            results["deltas_vs_p0"][arm_nm] = {
                "params_delta": int(arm_d["params"]) - int(p0_a["params"]),
                "params_delta_pct": round(100.0 * (arm_d["params"] - p0_a["params"]) / max(1, p0_a["params"]), 2),
                "delta_val_bpb": _diff(s.get("val_bpb"), p0_s.get("val_bpb")),
                "delta_ow6_bpb": _diff(s.get("ow6_bpb"), p0_s.get("ow6_bpb")),
                "delta_l2048_bpb": _diff(s.get("l2048_bpb"), p0_s.get("l2048_bpb")),
                "delta_s5_k2_pct": _diff(s.get("s5_k2_pct"), p0_s.get("s5_k2_pct"), 2),
                "delta_ov_long_pct": _diff(s.get("ov_long_pct"), p0_s.get("ov_long_pct"), 2),
                "delta_needle_512_pct": _diff(s.get("needle_512_exact_pct"), p0_s.get("needle_512_exact_pct"), 2),
                "delta_needle_1500_pct": _diff(s.get("needle_1500_exact_pct"), p0_s.get("needle_1500_exact_pct"), 2),
                "delta_needle_4096_pct": _diff(s.get("needle_4096_exact_pct"), p0_s.get("needle_4096_exact_pct"), 2),
                "delta_needle_8192_pct": _diff(s.get("needle_8192_exact_pct"), p0_s.get("needle_8192_exact_pct"), 2),
                "delta_orig_ref_pct": _diff(s.get("orig_ref_1500_exact_pct"), p0_s.get("orig_ref_1500_exact_pct"), 2),
                "delta_needle_mean_exact_pct": _diff(
                    s.get("needle_mean_exact_pct"), p0_s.get("needle_mean_exact_pct"), 2
                ),
                "delta_needle_mean_byte_pct": _diff(
                    s.get("needle_mean_byte_pct"), p0_s.get("needle_mean_byte_pct"), 2
                ),
                "train_step_speedup": _sp(p0_t.get("steady_ms_per_step"), t.get("steady_ms_per_step")),
                "vram_alloc_delta_mb": _diff(t.get("peak_alloc_mb"), p0_t.get("peak_alloc_mb"), 1),
                "ram_delta_kb": _diff(arm_d.get("cache_kb_o1_l32k"), p0_a.get("cache_kb_o1_l32k"), 2),
                "cpu_stream_speedup": _sp(
                    (st.get("cpu_avx512_fused_lut") or {}).get("decode_bps"),
                    (p0_st.get("cpu_avx512_fused_lut") or {}).get("decode_bps"),
                ),
                "gpu_cg_stream_speedup": _sp(
                    (st.get("gpu_cuda_graph_fused_lut") or {}).get("decode_bps"),
                    (p0_st.get("gpu_cuda_graph_fused_lut") or {}).get("decode_bps"),
                ),
            }

    # 1. Run Base Control P0 + 5 Single-Axis Orthogonal Arms (P1..P5)
    trained_sds: Dict[str, Tuple[CalciteV2Config, Dict[str, torch.Tensor]]] = {}
    p0_init_stds: Optional[Dict[str, float]] = None

    for arm_nm, axis_kw in SINGLE_AXIS_SPECS:
        cfg_a = build_phase_p_5m_config(smoke=args.smoke, **axis_kw)
        arm_res, sd_cpu = train_fp_arm(
            arm_nm,
            cfg_a,
            router_state,
            train_mm,
            val_mm,
            tok,
            tr_sents,
            eval_wins,
            s5_suites,
            ov_suites,
            trap_suites,
            device=device,
            total_steps=total_steps,
            warmup_steps=warm_steps,
            decay_steps=decay_steps,
            batch_size=batch_size,
            micro_bs=micro_bs,
            seq_len=seq_len,
            seed=args.seed,
            use_compile=use_compile,
            ref_init_stds=p0_init_stds,
            run_sigma_cal=(arm_nm == "P0_base_bn05"),
            smoke=args.smoke,
        )
        if arm_nm == "P0_base_bn05":
            p0_init_stds = dict(arm_res["gate2_init"]["block_std"])
        results["arms"][arm_nm] = arm_res
        trained_sds[arm_nm] = (cfg_a, sd_cpu)
        _recompute_deltas()
        _save()

    # 2. Automatic P6_combo_winners Selection & Training
    sel = select_combo_winners(results["arms"], two_sigma_ref=TWO_SIGMA_REF_BPB, smoke=args.smoke)
    results["winner_selection"] = sel
    print(
        f"\n================ [AUTO-COMBO SELECTION] mode={sel['mode']} | "
        f"selected={sel['selected_arms']} | kwargs={sel['combo_kwargs']} ================",
        flush=True,
    )

    p6_cfg = build_phase_p_5m_config(smoke=args.smoke, **sel["combo_kwargs"])
    if sel["mode"] == "single_winner_reused":
        win_nm = sel["selected_arms"][0]
        p6_cfg, p6_sd_cpu = trained_sds[win_nm]
        p6_res = json.loads(json.dumps(results["arms"][win_nm]))
        p6_res["arm"] = "P6_combo_winners"
        p6_res["reused_from"] = win_nm
        results["arms"]["P6_combo_winners"] = p6_res
        trained_sds["P6_combo_winners"] = (p6_cfg, p6_sd_cpu)
        print(f"  [P6_combo_winners] Single strict winner `{win_nm}` -> reused identical FP weights (0s re-train).", flush=True)
    elif sel["mode"].startswith("no_strict_winner_fallback_"):
        fb_nm = sel["mode"][len("no_strict_winner_fallback_") :]
        p6_cfg, p6_sd_cpu = trained_sds[fb_nm]
        p6_res = json.loads(json.dumps(results["arms"][fb_nm]))
        p6_res["arm"] = "P6_combo_winners"
        p6_res["reused_from"] = fb_nm
        p6_res["fallback_informational_only"] = True
        results["arms"]["P6_combo_winners"] = p6_res
        trained_sds["P6_combo_winners"] = (p6_cfg, p6_sd_cpu)
        print(
            f"  [P6_combo_winners] 0 strict winners -> reusing top Val arm `{fb_nm}` as informational QAT stress check.",
            flush=True,
        )
    else:
        p6_res, p6_sd_cpu = train_fp_arm(
            "P6_combo_winners",
            p6_cfg,
            router_state,
            train_mm,
            val_mm,
            tok,
            tr_sents,
            eval_wins,
            s5_suites,
            ov_suites,
            trap_suites,
            device=device,
            total_steps=total_steps,
            warmup_steps=warm_steps,
            decay_steps=decay_steps,
            batch_size=batch_size,
            micro_bs=micro_bs,
            seq_len=seq_len,
            seed=args.seed,
            use_compile=use_compile,
            ref_init_stds=p0_init_stds,
            run_sigma_cal=False,
            smoke=args.smoke,
        )
        results["arms"]["P6_combo_winners"] = p6_res
        trained_sds["P6_combo_winners"] = (p6_cfg, p6_sd_cpu)
    _recompute_deltas()
    _save()

    # 3. Post-Training CPU AVX-512 & GPU CUDA Graph LUT Streaming Benchmark for ALL Arms (Zero VRAM Contamination)
    print("\n================ [POST-TRAIN LUT STREAMING BENCHMARK (ALL ARMS)] ================", flush=True)
    reused_stream_src = results["arms"]["P6_combo_winners"].get("reused_from")
    for arm_nm in list(results["arms"].keys()):
        if arm_nm == "P6_combo_winners" and reused_stream_src and reused_stream_src in results["arms"]:
            results["arms"][arm_nm]["lut_streaming"] = results["arms"][reused_stream_src]["lut_streaming"]
            continue
        cfg_s, sd_cpu = trained_sds[arm_nm]
        m_tmp = CalciteV2LM(cfg_s).cpu()
        m_tmp.load_state_dict(sd_cpu, strict=True)
        m_tmp.byte_chunker.lock_router(True)
        m_tmp.set_crystal_schedule(1, 1)
        m_tmp.set_snap_alpha(1.0)
        st_res = benchmark_arm_lut_streaming(m_tmp, cfg_s, val_mm, device=device, smoke=args.smoke)
        results["arms"][arm_nm]["lut_streaming"] = st_res
        cpu_r = st_res["cpu_avx512_fused_lut"]
        gpu_r = st_res.get("gpu_cuda_graph_fused_lut") or {}
        assert cpu_r["strict_o1_ram"], f"[{arm_nm}] CPU streaming violated O(1) RAM!"
        assert cpu_r["vs_batch_max_err"] < 2e-5, f"[{arm_nm}] CPU stream err={cpu_r['vs_batch_max_err']}"
        if gpu_r:
            assert gpu_r["strict_o1_ram"], f"[{arm_nm}] GPU CG violated O(1) RAM!"
            assert gpu_r["vs_batch_max_err"] < 2e-5, f"[{arm_nm}] GPU CG vs batch err={gpu_r['vs_batch_max_err']}"
            assert gpu_r["vs_eager_max_err"] < 2e-5, f"[{arm_nm}] GPU CG vs eager err={gpu_r['vs_eager_max_err']}"
        print(
            f"  [Stream {arm_nm:20s}] CPU={cpu_r['decode_bps']:6.1f} B/s ({cpu_r['decode_ms_per_byte']:.3f}ms/B, err={cpu_r['vs_batch_max_err']:.1e}) | "
            f"GPU_CG={gpu_r.get('decode_bps')} B/s ({gpu_r.get('decode_ms_per_byte')}ms/B, err={gpu_r.get('vs_eager_max_err')}) | "
            f"RAM={cpu_r['state_kb']} KB (O1={cpu_r['strict_o1_ram']})",
            flush=True,
        )
        del m_tmp
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    _recompute_deltas()
    _save()

    # 4. 250-Step Copy-Distilled 4-Tap QAT + 2-bit LUT Stress Check on BOTH P0_base_bn05 AND P6_combo_winners
    for q_arm in ("P0_base_bn05", "P6_combo_winners"):
        cfg_q, sd_q_cpu = trained_sds[q_arm]
        q_res = run_copy_distilled_qat_and_lut_eval(
            q_arm,
            cfg_q,
            sd_q_cpu,
            results["arms"][q_arm]["summary"],
            train_mm,
            val_mm,
            tok,
            tr_sents,
            eval_wins,
            s5_suites,
            ov_suites,
            trap_suites,
            device=device,
            qat_steps=qat_steps,
            batch_size=batch_size,
            micro_bs_q=qat_micro_bs,
            seq_len=seq_len,
            seed=args.seed,
            smoke=args.smoke,
        )
        results["lut_qat_stress_check"][q_arm] = q_res
        _save()

    # Compute direct P6_lut vs P0_lut deltas for instant KEEP/REJECT visibility
    p0_l = results["lut_qat_stress_check"]["P0_base_bn05"]["summary_lut"]
    p6_l = results["lut_qat_stress_check"]["P6_combo_winners"]["summary_lut"]
    results["lut_qat_stress_check"]["delta_p6_lut_vs_p0_lut"] = {
        k: (
            round(float(p6_l[k]) - float(p0_l[k]), 4)
            if (p6_l.get(k) is not None and p0_l.get(k) is not None)
            else None
        )
        for k in (
            "val_bpb",
            "ow6_bpb",
            "l2048_bpb",
            "needle_512_exact_pct",
            "needle_1500_exact_pct",
            "needle_4096_exact_pct",
            "needle_8192_exact_pct",
            "orig_ref_1500_exact_pct",
            "needle_mean_exact_pct",
            "needle_mean_byte_pct",
        )
    }
    _save()
    print(
        f"\n[PHASE P-1 TOURNAMENT COMPLETE] elapsed={results['meta']['elapsed_wall_s']:.1f}s | Saved -> {out_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()
