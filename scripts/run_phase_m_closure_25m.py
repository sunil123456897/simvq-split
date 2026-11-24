"""
Phase M Stage D: Flagship `Calcite v3 (S2_25M)` Unified Closure
(`FP -> 4-Tap 2-bit LUT -> Disk Roundtrip -> L=32,768 -> Cognitive n=150 -> Live CPU/GPU Streaming`)
(`scripts/run_phase_m_closure_25m.py`).

Synthesizes the verified `Via Negativa` Phase M winners on top of `Calcite v2 (S2_25M)`:
  1. `M3_sym_byte_swa`: Symmetric encoder dilated conv (`k=4, d=2`) + 3rd-scale decoder conv
     (`k=4, d=4`) + `6 KB` micro byte-SWA (`W=24, d=64, 2h`).
  2. `M8_byte_engram`: Pure-tensor 2/3-byte polynomial rolling hash table (`4096 x 32`,
     `+24 B` dynamic streaming RAM).
  3. `Pre-frozen UTF-8 Router`: `0.00%` mid-UTF8 cuts from step 1 (`router_max_abs_diff = 0.0`).
  4. `MuonAdamWHybrid`: 5-step BF16 Newton-Schulz `Muon` on hidden 2D matrices + fused `AdamW`
     with Crystal/Woodbury protection across `Stage 1A -> Stage 1B -> Stage 2`.
  5. `4-Group + 4-Tap + Logit-KL STE QAT` -> `checkpoints/calcite_v3_s2_25m_utf8_lut.pt`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from calcite_v2 import (  # noqa: E402
    CalciteV2Config,
    CalciteV2LM,
    build_calcite_v2_tier_config,
    compute_v2_state_cache_kb,
)
from experiments.scaling_phase_k.eval_suite import has_usable_cuda  # noqa: E402
from kernels.fast_calcite_scans import clear_fast_qat_cache, unwrap_compiled_state_dict  # noqa: E402
from kernels.fast_packed_lut import (  # noqa: E402
    get_lut_backend_info,
    pack_fast_model_2bit_lut,
    save_packed_lut_checkpoint,
)
from run_phase_e_fast_utf8_benchmark import build_disjoint_openweb_corpora  # noqa: E402
from scripts.colab_phase_k_scaling_runner import (  # noqa: E402
    StratifiedSystem1Batcher,
    _assert_router_unchanged,
    _attach_4tap_hooks,
    _cosine_lr,
    _fp_micro_ce,
    _make_adamw,
    _qat_micro_step,
    _run_accum_microbatches,
    _snapshot_router_params,
)
from scripts.colab_phase_l_v2_runner import (  # noqa: E402
    benchmark_stream_engine_live,
    build_eval_bundles,
    ensure_phase_k_corpus,
    eval_long_context_stress,
    evaluate_full_model,
    load_frozen_router_into,
    set_all_seeds,
)
from scripts.muon_opt import MuonAdamWHybrid  # noqa: E402
from scripts.run_phase_l_closure_25m import (  # noqa: E402
    _default_micro_bs,
    load_calcite_v2_lut_checkpoint,
)
from scripts.run_phase_m_ablations_5m import eval_zeroshot_needle_recall_bpb  # noqa: E402
from scripts.run_phase_m_stage_c_v3_5m import _split_v3_qat_4groups  # noqa: E402
from simvq_split.eval_utils import evaluate_suite_batched  # noqa: E402
from simvq_split.memory import (  # noqa: E402
    compute_2bit_lut_footprint,
    quantize_ternary_weights_ls,
)
from simvq_split.tasks import load_gsm8k_corpora  # noqa: E402
from simvq_split.tokenizer import ByteBPETokenizer  # noqa: E402


def build_calcite_v3_25m_config(smoke: bool = False) -> CalciteV2Config:
    if smoke:
        cfg = build_calcite_v2_tier_config(
            "S1_5M",
            topology="hourglass",
            use_scaled_decoder=True,
            use_kalman_precond=False,
            use_latent_feedback=False,
            use_clean_gates=False,
        )
        cfg.d_model = 128
        cfg.n_heads = 2
        cfg.d_ff = 256
        cfg.byte_char_dim = 16
        cfg.byte_d_ff = 128
    else:
        cfg = build_calcite_v2_tier_config(
            "S2_25M",
            topology="hourglass",
            use_scaled_decoder=True,
            use_kalman_precond=False,
            use_latent_feedback=False,
            use_clean_gates=False,
        )
    # Verified Phase M `Via Negativa` winners: M3 + M8 (Byte Engram)
    cfg.use_sym_byte_swa = True
    cfg.byte_swa_window = 24
    cfg.byte_swa_dim = 64
    cfg.use_byte_engram = True
    cfg.engram_slots = 4096
    cfg.engram_dim = 32
    cfg.byte_chunk_mode = "utf8"
    cfg.use_static_m_cap = True
    return cfg


def run_v3_closure_25m(
    data_dir: Path,
    out_dir: Path,
    ckpt_dir: Path,
    s1a_steps: int = 1600,
    s1b_steps: int = 500,
    s2_steps: int = 1600,
    qat_steps: int = 500,
    batch_size: int = 64,
    seq_len: int = 512,
    seed: int = 2026,
    smoke: bool = False,
    force: bool = False,
) -> Dict[str, object]:
    device = "cuda" if has_usable_cuda() else "cpu"
    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram_gb < 12.0:
            torch.cuda.set_per_process_memory_fraction(0.72, 0)
    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tok = ByteBPETokenizer(merges=[])
    tr_docs, te_docs, tr_sents, _ = load_gsm8k_corpora(ROOT_DIR)
    train_wins, eval_wins, _, _ = build_disjoint_openweb_corpora(
        ROOT_DIR, tok, tr_docs, te_docs, seq_len=seq_len
    )
    n_ev = 6 if smoke else 150
    _, _, s5_suites, ov_suites, trap_suites = build_eval_bundles(
        ROOT_DIR, tok, n_eval=n_ev, seq_len=seq_len
    )

    router_path = ROOT_DIR / "benchmarks" / "phase_l_v2" / "frozen_utf8_router.pt"
    assert router_path.is_file(), f"Missing canonical frozen UTF-8 router: {router_path}"
    router_state = torch.load(router_path, map_location="cpu", weights_only=True)

    cfg = build_calcite_v3_25m_config(smoke=smoke)
    if smoke:
        rng_f = np.random.default_rng(seed)
        fake_arr = rng_f.integers(3, 128, size=100_000, dtype=np.uint16)
        train_mm = fake_arr
        val_mm = fake_arr
        s1a_steps, s1b_steps, s2_steps, qat_steps = 4, 2, 4, 4
        batch_size, seq_len = 8, 128
    else:
        train_bin, val_bin = ensure_phase_k_corpus(data_dir)
        train_mm = np.memmap(train_bin, dtype=np.uint16, mode="r")
        val_mm = np.memmap(val_bin, dtype=np.uint16, mode="r")

    grok_steps = s1a_steps + s1b_steps
    total_fp_steps = grok_steps + s2_steps
    batcher = StratifiedSystem1Batcher(
        train_mm=train_mm,
        tr_docs=tr_docs,
        tr_sents=tr_sents,
        tok=tok,
        seq_len=seq_len,
        train_wins=train_wins,
        seed=seed,
    )

    set_all_seeds(seed)
    model = CalciteV2LM(cfg).to(device)
    load_frozen_router_into(model, router_state)
    fp_router_snap = _snapshot_router_params(model)
    n_params = int(sum(p.numel() for p in model.parameters()))
    cache_kb = round(compute_v2_state_cache_kb(cfg, 32768), 2)
    micro_bs = min(batch_size, _default_micro_bs(device, is_qat=False))
    print(
        f"\n=== [PHASE M CLOSURE: Calcite v3 S2_25M] params={n_params:,} | O(1)_cache={cache_kb:.2f} KB | "
        f"device={device} | micro_bs={micro_bs} | steps=({s1a_steps}+{s1b_steps}+{s2_steps}, qat={qat_steps}) ===",
        flush=True,
    )

    s1b_ckpt = out_dir / "closure_v3_25m_s1b_fp.pt"
    s2_ckpt = out_dir / "closure_v3_25m_stage2_fp.pt"
    lut_ckpt = ckpt_dir / ("smoke_v3_25m_lut.pt" if smoke else "calcite_v3_s2_25m_utf8_lut.pt")
    res_path = out_dir / "phase_m_closure_25m_results.json"

    t_all = time.perf_counter()
    s1a_s5_k2 = 0.0
    s1_s5_k2 = 0.0
    s1_ov_long = 0.0
    s1_trap_long = 0.0

    if (not smoke) and (not force) and s2_ckpt.is_file():
        print(f"  [Resume] Loading saved Stage-2 FP checkpoint -> {s2_ckpt}", flush=True)
        blob = torch.load(s2_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["state_dict"], strict=True)
        model.byte_chunker.lock_router(True)
        s1a_s5_k2 = float(blob.get("s1a_s5_k2", 0.0))
        s1_s5_k2 = float(blob.get("s1_s5_k2", 0.0))
        s1_ov_long = float(blob.get("s1_ov_long", 0.0))
        s1_trap_long = float(blob.get("s1_trap_long", 0.0))
        _assert_router_unchanged(model, fp_router_snap, "V3_25M_RESUME_S2")
    else:
        if (not smoke) and (not force) and s1b_ckpt.is_file():
            print(f"  [Resume] Loading saved Stage-1B FP checkpoint -> {s1b_ckpt}", flush=True)
            blob = torch.load(s1b_ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(blob["state_dict"], strict=True)
            model.byte_chunker.lock_router(True)
            s1a_s5_k2 = float(blob.get("s1a_s5_k2", 0.0))
            s1_s5_k2 = float(blob.get("s1_s5_k2", 0.0))
            s1_ov_long = float(blob.get("s1_ov_long", 0.0))
            s1_trap_long = float(blob.get("s1_trap_long", 0.0))
            _assert_router_unchanged(model, fp_router_snap, "V3_25M_RESUME_S1B")
        else:
            # ----------------------------------------------------------
            # STAGE 1A: Algebraic S_5 Grokking (`MuonAdamW`, Woodbury quiet 0.05x)
            # ----------------------------------------------------------
            base_lr_1a = 1.5e-3
            opt_1a = MuonAdamWHybrid(
                model,
                base_lr=base_lr_1a,
                muon_lr_mult=5.0,
                weight_decay=0.02,
                device=device,
                woodbury_lr_mult=0.05,
                protect_crystal=False,
            )
            warm_1a = min(60, max(1, s1a_steps // 12))
            live_diag = (800, 1200, s1a_steps) if ((not smoke) and s1a_steps >= 800) else ()
            t_1a = time.perf_counter()

            for step in range(1, s1a_steps + 1):
                model.train()
                model.set_crystal_schedule(step, s1a_steps)
                lr_s = _cosine_lr(step, s1a_steps, warm_1a, base_lr_1a, 2.0e-4)
                sc = lr_s / base_lr_1a
                for pg in opt_1a.param_groups:
                    pg["lr"] = base_lr_1a * float(pg.get("lr_mult", 1.0)) * sc
                xb_full, yb_full = batcher.sample_batch(
                    batch_size,
                    device,
                    step=step,
                    grok_steps=grok_steps,
                    total_steps=total_fp_steps,
                    s1b_steps=s1b_steps,
                    s1a_offset=0,
                )
                micro_bs, ce_sum_t = _run_accum_microbatches(
                    xb_full,
                    yb_full,
                    micro_bs,
                    opt_1a,
                    device,
                    f"S1A s={step}",
                    lambda xb, yb, cur_acc, m_ref=model: _fp_micro_ce(
                        m_ref, xb, yb, tok.PAD_ID, use_amp, cur_acc, 0.04
                    ),
                )
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt_1a.step()

                if step in (1, s1a_steps) or (step % 200 == 0):
                    ce_v = float(ce_sum_t.item()) / float(max(1, batch_size // micro_bs)) if ce_sum_t is not None else 0.0
                    b_ch = model.byte_chunker.last_chunk_ratio
                    vram_mb = torch.cuda.memory_allocated() / 1e6 if device == "cuda" else 0.0
                    print(
                        f"  [S1A] step {step:4d}/{s1a_steps} | b/ch={b_ch:.2f} | ce={ce_v:.4f} | "
                        f"VRAM={vram_mb:.0f}MB | {(time.perf_counter()-t_1a)/60:.1f}m",
                        flush=True,
                    )
                if step in live_diag:
                    model.eval()
                    model.set_snap_alpha(1.0)
                    pk2, _, _, _ = evaluate_suite_batched(
                        model, s5_suites["k2"][:32], pad_id=tok.PAD_ID, batch_size=16, device=device
                    )
                    print(f"  [S1A LIVE step={step}] s5_k2_32={pk2:.1f}%", flush=True)
                    model.train()
                    model.set_crystal_schedule(step, s1a_steps)

            del opt_1a
            if device == "cuda":
                torch.cuda.empty_cache()

            model.eval()
            model.set_crystal_schedule(s1a_steps, s1a_steps)
            model.set_snap_alpha(1.0)
            s1a_s5_k2, _, _, _ = evaluate_suite_batched(
                model, s5_suites["k2"][: min(64, len(s5_suites["k2"]))], pad_id=tok.PAD_ID, batch_size=16, device=device
            )
            print(f"  [S1A DONE] s5_k2={s1a_s5_k2:.2f}%", flush=True)

            # ----------------------------------------------------------
            # STAGE 1B: Associative Woodbury Consolidation (Crystal protected 0.04x)
            # ----------------------------------------------------------
            base_lr_1b = 9.0e-4
            opt_1b = MuonAdamWHybrid(
                model,
                base_lr=base_lr_1b,
                muon_lr_mult=3.5,
                weight_decay=0.01,
                device=device,
                woodbury_lr_mult=1.8,
                protect_crystal=True,
                crystal_lr_mult=0.04,
            )
            warm_1b = min(35, max(1, s1b_steps // 8))
            t_1b = time.perf_counter()

            for o_step in range(1, s1b_steps + 1):
                step = s1a_steps + o_step
                model.train()
                model.set_crystal_schedule(s1a_steps, s1a_steps)
                model.set_snap_alpha(1.0)
                lr_s = _cosine_lr(o_step, s1b_steps, warm_1b, base_lr_1b, 1.5e-4)
                sc = lr_s / base_lr_1b
                for pg in opt_1b.param_groups:
                    pg["lr"] = base_lr_1b * float(pg.get("lr_mult", 1.0)) * sc
                xb_full, yb_full = batcher.sample_batch(
                    batch_size, device, step=step, grok_steps=grok_steps, total_steps=total_fp_steps, s1b_steps=s1b_steps
                )
                micro_bs, ce_sum_t = _run_accum_microbatches(
                    xb_full,
                    yb_full,
                    micro_bs,
                    opt_1b,
                    device,
                    f"S1B s={o_step}",
                    lambda xb, yb, cur_acc, m_ref=model: _fp_micro_ce(
                        m_ref, xb, yb, tok.PAD_ID, use_amp, cur_acc, 0.04
                    ),
                )
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt_1b.step()
                if o_step in (1, max(1, s1b_steps // 2), s1b_steps) or (o_step % 125 == 0):
                    ce_v = float(ce_sum_t.item()) / float(max(1, batch_size // micro_bs)) if ce_sum_t is not None else 0.0
                    print(
                        f"  [S1B] step {o_step:4d}/{s1b_steps} | ce={ce_v:.4f} | {(time.perf_counter()-t_1b)/60:.1f}m",
                        flush=True,
                    )

            del opt_1b
            if device == "cuda":
                torch.cuda.empty_cache()

            model.eval()
            model.set_crystal_schedule(s1a_steps, s1a_steps)
            model.set_snap_alpha(1.0)
            s1_s5_k2, _, _, _ = evaluate_suite_batched(
                model, s5_suites["k2"][: min(64, len(s5_suites["k2"]))], pad_id=tok.PAD_ID, batch_size=16, device=device
            )
            s1_ov_long, _, _, _ = evaluate_suite_batched(
                model, ov_suites["ood_long"][: min(64, len(ov_suites["ood_long"]))], pad_id=tok.PAD_ID, batch_size=16, device=device
            )
            s1_trap_long, _, _, _ = evaluate_suite_batched(
                model, trap_suites["ood_long"][: min(64, len(trap_suites["ood_long"]))], pad_id=tok.PAD_ID, batch_size=16, device=device
            )
            print(
                f"  [S1 DUAL GATE] s5_k2={s1_s5_k2:.2f}% | ov_long={s1_ov_long:.2f}% | trap_long={s1_trap_long:.2f}%",
                flush=True,
            )
            if not smoke:
                torch.save(
                    {
                        "config": asdict(cfg),
                        "state_dict": {k: v.cpu() for k, v in unwrap_compiled_state_dict(model).items()},
                        "s1a_s5_k2": float(s1a_s5_k2),
                        "s1_s5_k2": float(s1_s5_k2),
                        "s1_ov_long": float(s1_ov_long),
                        "s1_trap_long": float(s1_trap_long),
                    },
                    s1b_ckpt,
                )
                print(f"  [Saved] Stage-1B FP checkpoint -> {s1b_ckpt}", flush=True)

        # ----------------------------------------------------------
        # STAGE 2: UTF-8 OpenWeb Language Scaling + Protected Crystal/Woodbury (`MuonAdamW`)
        # ----------------------------------------------------------
        base_lr_2 = 1.0e-3
        opt_2 = MuonAdamWHybrid(
            model,
            base_lr=base_lr_2,
            muon_lr_mult=4.8,
            weight_decay=0.03,
            device=device,
            woodbury_lr_mult=0.07,
            protect_crystal=True,
            crystal_lr_mult=0.05,
        )
        warm_2 = min(40, max(1, s2_steps // 8))
        t_2 = time.perf_counter()
        for s2 in range(1, s2_steps + 1):
            step = grok_steps + s2
            model.train()
            model.set_crystal_schedule(s1a_steps, s1a_steps)
            model.set_snap_alpha(1.0)
            lr_s = _cosine_lr(s2, s2_steps, warm_2, base_lr_2, 8.0e-5)
            sc = lr_s / base_lr_2
            for pg in opt_2.param_groups:
                pg["lr"] = base_lr_2 * float(pg.get("lr_mult", 1.0)) * sc
            xb_full, yb_full = batcher.sample_batch(
                batch_size, device, step=step, grok_steps=grok_steps, total_steps=total_fp_steps, s1b_steps=s1b_steps
            )
            micro_bs, ce_sum_t = _run_accum_microbatches(
                xb_full,
                yb_full,
                micro_bs,
                opt_2,
                device,
                f"S2 s={s2}",
                lambda xb, yb, cur_acc, m_ref=model: _fp_micro_ce(
                    m_ref, xb, yb, tok.PAD_ID, use_amp, cur_acc, 0.04
                ),
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_2.step()
            if s2 in (1, s2_steps) or (s2 % 200 == 0):
                ce_v = float(ce_sum_t.item()) / float(max(1, batch_size // micro_bs)) if ce_sum_t is not None else 0.0
                print(
                    f"  [S2-utf8] step {s2:4d}/{s2_steps} | ce={ce_v:.4f} | {(time.perf_counter()-t_2)/60:.1f}m",
                    flush=True,
                )

        del opt_2
        if device == "cuda":
            torch.cuda.empty_cache()

        _assert_router_unchanged(model, fp_router_snap, "V3_25M_FP_POSTLOCK")
        if not smoke:
            torch.save(
                {
                    "config": asdict(cfg),
                    "state_dict": {k: v.cpu() for k, v in unwrap_compiled_state_dict(model).items()},
                    "s1a_s5_k2": float(s1a_s5_k2),
                    "s1_s5_k2": float(s1_s5_k2),
                    "s1_ov_long": float(s1_ov_long),
                    "s1_trap_long": float(s1_trap_long),
                },
                s2_ckpt,
            )
            print(f"  [Saved] Stage-2 FP checkpoint -> {s2_ckpt}", flush=True)

    # Evaluate FP Champion (`eval_bs=8` matched identically to LUT for exact paired `val_bpb`!)
    model.eval()
    model.set_crystal_schedule(1, 1)
    model.set_snap_alpha(1.0)
    print("  [Eval] Evaluating Stage-2 FP Calcite v3 (25M) Champion...", flush=True)
    eval_fp = evaluate_full_model(
        model, val_mm, eval_wins, tok, s5_suites, ov_suites, trap_suites, device=device, quick=smoke, eval_bs=8
    )
    long_fp = eval_long_context_stress(
        model, val_mm, device=device, lengths=(512, 2048) if smoke else (2048, 8192, 32768)
    )
    needle_fp = eval_zeroshot_needle_recall_bpb(
        model, val_mm, device=device, haystack_bytes=1500, n_samples=4 if smoke else 24, seed=7777
    )
    print(
        f"  [FP DONE] Val={eval_fp['val_bpb']:.4f} | OW6={eval_fp['openweb_6']['mean_openweb_bpb']:.4f} | "
        f"L2048={long_fp.get('L_2048', {}).get('bpb')} | L32K={long_fp.get('L_32768', {}).get('bpb')} | "
        f"S5={eval_fp['cognitive']['s5_sum']:.1f}% | OV_L={eval_fp['cognitive']['overwrite']['ood_long']['exact_pct']:.1f}% | "
        f"Trap_L={eval_fp['cognitive']['trap']['ood_long']['exact_pct']:.1f}% | "
        f"SelF1_L={eval_fp['cognitive']['selective_f1_ood_long']:.1f}% | Needle={needle_fp['recall_gain_bpb']:+.4f}",
        flush=True,
    )

    # ----------------------------------------------------------
    # STAGE 3: 4-Group + 4-Tap + Logit-KL STE QAT -> 2-bit uint8 LUT
    # ----------------------------------------------------------
    for p in model.parameters():
        p.requires_grad_(False)
    fp_sd_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    qat_model = CalciteV2LM(cfg).to(device)
    qat_model.load_state_dict(fp_sd_cpu, strict=True)
    model.byte_chunker.use_static_m_cap = False
    qat_model.byte_chunker.use_static_m_cap = False
    qat_model.byte_chunker.lock_router(True)
    qat_model.set_crystal_schedule(1, 1)
    qat_model.set_snap_alpha(1.0)
    qat_model.set_qat_mix(0.30)
    qat_model.set_ternary_qat(True)

    gain_q, cryst_q, chunk_q, core_q = _split_v3_qat_4groups(qat_model)
    all_qp = gain_q + cryst_q + chunk_q + core_q
    w_scale = (384.0 / float(cfg.d_model)) ** 0.5
    # Causal fix: per-layer PTQ probe showed cortex is +0.0117 zero-shot while `byte_chunker` carries 96% of PTQ shock;
    # give `chunk_q` 1.1e-3*w_scale and 800 steps to close the Muon->2-bit LUT gap cleanly.
    q_lrs = (1.2e-3 * w_scale, 8.0e-5, 1.1e-3 * w_scale, 4.5e-4 * w_scale)
    q_mins = (1.0e-4 * w_scale, 1.5e-5, 8.0e-5 * w_scale, 4.5e-5 * w_scale)
    opt_q = _make_adamw(
        [
            {"params": gain_q, "lr": q_lrs[0]},
            {"params": cryst_q, "lr": q_lrs[1]},
            {"params": chunk_q, "lr": q_lrs[2]},
            {"params": core_q, "lr": q_lrs[3]},
        ],
        device,
        weight_decay=0.0,
    )
    opt_q.register_step_post_hook(lambda *_a, **_kw: clear_fast_qat_cache())
    warm_q = min(48, max(1, qat_steps // 12))
    ramp_q = min(240, max(1, qat_steps // 4))
    kl_temp = 1.5
    taps_fp: List[torch.Tensor] = []
    taps_q: List[torch.Tensor] = []
    hk_fp = _attach_4tap_hooks(model, taps_fp, detach=True)
    hk_q = _attach_4tap_hooks(qat_model, taps_q, detach=False)
    micro_bs_q = min(batch_size, _default_micro_bs(device, is_qat=True))
    t_q0 = time.perf_counter()

    try:
        for qs in range(1, qat_steps + 1):
            qat_model.train()
            mix = min(1.0, 0.30 + 0.70 * (qs / float(ramp_q)))
            qat_model.set_qat_mix(mix)
            clear_fast_qat_cache()
            for gi, (blr, mlr) in enumerate(zip(q_lrs, q_mins)):
                opt_q.param_groups[gi]["lr"] = _cosine_lr(qs, qat_steps, warm_q, blr, mlr)
            xb_full, yb_full = batcher.sample_batch(
                batch_size, device, step=total_fp_steps, grok_steps=grok_steps, total_steps=total_fp_steps, s1b_steps=s1b_steps
            )
            micro_bs_q, q_ce_sum_t = _run_accum_microbatches(
                xb_full,
                yb_full,
                micro_bs_q,
                opt_q,
                device,
                f"4Tap-QAT qs={qs}",
                lambda xb, yb, cur_acc, t_m=model, s_m=qat_model: _qat_micro_step(
                    t_m, s_m, taps_fp, taps_q, kl_temp, xb, yb, tok.PAD_ID, use_amp, cur_acc
                ),
            )
            torch.nn.utils.clip_grad_norm_(all_qp, 1.0)
            opt_q.step()
            clear_fast_qat_cache()
            if qs in (1, max(1, qat_steps // 2), qat_steps) or (qs % 100 == 0):
                qce = float(q_ce_sum_t.item()) / float(max(1, batch_size // micro_bs_q)) if q_ce_sum_t is not None else 0.0
                print(
                    f"  [4Tap-QAT] step {qs:4d}/{qat_steps} | mix={mix:.2f} | ce={qce:.4f} | "
                    f"{(time.perf_counter()-t_q0)/60:.1f}m",
                    flush=True,
                )
    finally:
        for h in hk_fp + hk_q:
            h.remove()
        clear_fast_qat_cache()
        qat_model.byte_chunker.use_static_m_cap = True

    router_mut_diff = _assert_router_unchanged(qat_model, fp_router_snap, "V3_25M_QAT")
    qat_s = round(time.perf_counter() - t_q0, 2)
    del model, opt_q
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    qat_model.set_ternary_qat(False)
    qat_model.set_qat_mix(1.0)
    qat_sd_clean = unwrap_compiled_state_dict(qat_model)
    del qat_model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    clean_qat = CalciteV2LM(cfg)
    clean_qat.load_state_dict(qat_sd_clean, strict=True)
    clean_qat.byte_chunker.byte_chunk_mode = "utf8"
    clean_qat.byte_chunker.target_chunk_bytes = float(cfg.target_chunk_bytes)
    clean_qat.byte_chunker.use_static_m_cap = True
    clean_qat.byte_chunker.lock_router(True)
    clean_qat.set_crystal_schedule(1, 1)
    clean_qat.set_snap_alpha(1.0)
    with torch.no_grad():
        for nm, mod in clean_qat.named_modules():
            if isinstance(mod, nn.Linear) and nm != "lm_head":
                if clean_qat.byte_chunker.is_locked_router_module(mod):
                    continue
                wt, _, _ = quantize_ternary_weights_ls(mod.weight.data)
                mod.weight.data.copy_(wt)
    clean_qat.eval()

    binfo = get_lut_backend_info()
    lut_bk = "cuda_ext" if (device == "cuda" and binfo["cuda_ext_available"]) else "auto"
    lut_cpu = pack_fast_model_2bit_lut(clean_qat, backend="auto").eval()
    lut_foot = compute_2bit_lut_footprint(lut_cpu)
    save_packed_lut_checkpoint(
        lut_ckpt,
        lut_cpu,
        cfg,
        tok,
        metadata={
            "phase": "Phase_M_Closure_V3_25M",
            "params": n_params,
            "router_max_abs_diff": router_mut_diff,
            "s1a_s5_k2": float(s1a_s5_k2),
            "s1_s5_k2": float(s1_s5_k2),
            "s1_ov_long": float(s1_ov_long),
            "s1_trap_long": float(s1_trap_long),
        },
    )
    ckpt_bytes = int(lut_ckpt.stat().st_size)
    print(f"  [Saved] 2-bit LUT checkpoint -> {lut_ckpt} ({ckpt_bytes:,} B)", flush=True)
    del clean_qat, lut_cpu
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # Strict disk roundtrip load (`strict=True`) on CPU and GPU!
    lut_cpu_rt, _, _, _ = load_calcite_v2_lut_checkpoint(lut_ckpt, device="cpu", backend="auto")
    stream_cpu = benchmark_stream_engine_live(lut_cpu_rt, device="cpu", n_bytes=32 if smoke else 256)
    print(
        f"  [Stream CPU RT] {stream_cpu['bytes_per_sec']} B/s | err={stream_cpu['stream_vs_batch_max_err']} | "
        f"RAM={stream_cpu['state_kb']} KB | O1={stream_cpu['strict_o1_ram_invariant']}",
        flush=True,
    )
    del lut_cpu_rt
    gc.collect()

    lut_gpu_rt, _, _, _ = load_calcite_v2_lut_checkpoint(lut_ckpt, device=device, backend=lut_bk)
    print("  [Eval] Evaluating 2-bit LUT Calcite v3 (25M) from disk roundtrip...", flush=True)
    eval_lut = evaluate_full_model(
        lut_gpu_rt, val_mm, eval_wins, tok, s5_suites, ov_suites, trap_suites, device=device, quick=smoke, eval_bs=8, cog_bs=2
    )
    if device == "cuda":
        torch.cuda.empty_cache()
    long_lut = eval_long_context_stress(
        lut_gpu_rt, val_mm, device=device, lengths=(512, 2048) if smoke else (2048, 8192, 32768)
    )
    if device == "cuda":
        torch.cuda.empty_cache()
    stream_gpu = benchmark_stream_engine_live(lut_gpu_rt, device=device, n_bytes=32 if smoke else 256)
    del lut_gpu_rt
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    gap_val = round(float(eval_lut["val_bpb"]) - float(eval_fp["val_bpb"]), 4)
    gap_ow = round(
        float(eval_lut["openweb_6"]["mean_openweb_bpb"]) - float(eval_fp["openweb_6"]["mean_openweb_bpb"]), 4
    )
    result: Dict[str, object] = {
        "phase": "Phase_M_Closure_V3_25M",
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "params": n_params,
        "config": asdict(cfg),
        "schedule": {
            "s1a_steps": s1a_steps,
            "s1b_steps": s1b_steps,
            "s2_steps": s2_steps,
            "qat_steps": qat_steps,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "qat_seconds": qat_s,
            "total_seconds": round(time.perf_counter() - t_all, 1),
        },
        "stage1_gates": {
            "s1a_s5_k2_pct": round(s1a_s5_k2, 2),
            "s1_s5_k2_pct": round(s1_s5_k2, 2),
            "s1_ov_long_pct": round(s1_ov_long, 2),
            "s1_trap_long_pct": round(s1_trap_long, 2),
        },
        "router_max_abs_diff": router_mut_diff,
        "cache_kb_o1": cache_kb,
        "ckpt_bytes": ckpt_bytes,
        "lut_footprint": lut_foot,
        "gap_val_bpb": gap_val,
        "gap_ow_bpb": gap_ow,
        "eval_fp": eval_fp,
        "eval_lut": eval_lut,
        "long_context_fp": long_fp,
        "long_context_lut": long_lut,
        "needle_fp": needle_fp,
        "stream_cpu": stream_cpu,
        "stream_gpu": stream_gpu,
        "checkpoint_path": str(lut_ckpt),
    }

    if smoke:
        lut_ckpt.unlink(missing_ok=True)
        print(
            f"[SMOKE PASS] FP_val={eval_fp['val_bpb']} | LUT_val={eval_lut['val_bpb']} | "
            f"gap={gap_val:+.4f} | stream_cpu={stream_cpu['bytes_per_sec']} B/s | err={stream_cpu['stream_vs_batch_max_err']}",
            flush=True,
        )
    else:
        res_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"\n=== [PHASE M 25M CLOSURE DONE] Saved -> {res_path} ===\n"
            f"  FP  val={eval_fp['val_bpb']:.4f} | OW6={eval_fp['openweb_6']['mean_openweb_bpb']:.4f} | "
            f"L2048={long_fp.get('L_2048', {}).get('bpb')} | L32K={long_fp.get('L_32768', {}).get('bpb')} | "
            f"S5={eval_fp['cognitive']['s5_sum']:.1f}% | OV_L={eval_fp['cognitive']['overwrite']['ood_long']['exact_pct']:.1f}% | "
            f"SelF1_L={eval_fp['cognitive']['selective_f1_ood_long']:.1f}%\n"
            f"  LUT val={eval_lut['val_bpb']:.4f} (gap={gap_val:+.4f}) | OW6={eval_lut['openweb_6']['mean_openweb_bpb']:.4f} (gap={gap_ow:+.4f}) | "
            f"L2048={long_lut.get('L_2048', {}).get('bpb')} | L32K={long_lut.get('L_32768', {}).get('bpb')} | "
            f"S5={eval_lut['cognitive']['s5_sum']:.1f}% | OV_L={eval_lut['cognitive']['overwrite']['ood_long']['exact_pct']:.1f}% | "
            f"SelF1_L={eval_lut['cognitive']['selective_f1_ood_long']:.1f}% | "
            f"CPU_stream={stream_cpu['bytes_per_sec']} B/s (RAM={stream_cpu['state_kb']} KB, err={stream_cpu['stream_vs_batch_max_err']})",
            flush=True,
        )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase M Stage D: Flagship Calcite v3 (S2_25M) Unified Closure")
    ap.add_argument("--data-dir", type=str, default="data_cache")
    ap.add_argument("--out-dir", type=str, default="benchmarks/phase_m_v3")
    ap.add_argument("--ckpt-dir", type=str, default="checkpoints")
    ap.add_argument("--s1a-steps", type=int, default=1600)
    ap.add_argument("--s1b-steps", type=int, default=500)
    ap.add_argument("--s2-steps", type=int, default=1600)
    ap.add_argument("--qat-steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke-local", action="store_true")
    args = ap.parse_args()

    run_v3_closure_25m(
        data_dir=ROOT_DIR / args.data_dir,
        out_dir=ROOT_DIR / args.out_dir,
        ckpt_dir=ROOT_DIR / args.ckpt_dir,
        s1a_steps=args.s1a_steps,
        s1b_steps=args.s1b_steps,
        s2_steps=args.s2_steps,
        qat_steps=args.qat_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        seed=args.seed,
        smoke=args.smoke_local,
        force=args.force,
    )


if __name__ == "__main__":
    main()
