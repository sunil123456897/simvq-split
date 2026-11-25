"""
Phase N Stage 3: Flagship `Calcite v3.1 (S2_25M)` Unified Closure
(`FP -> 4-Tap 2-bit LUT -> Disk Roundtrip -> L=32,768 Prose & Verbatim Needle -> Cognitive n=150 -> Live CPU/GPU Streaming`)
(`scripts/run_phase_n_closure_25m.py`).

Unifies ALL verified `Calcite` breakthroughs into one `~10.0 MB` `2-bit uint8 T-MAC LUT` checkpoint
(`checkpoints/calcite_v31_s2_25m_utf8_lut.pt`):
  1. `Calcite v3` Core: `M3_sym_byte_swa` + `M8_byte_engram` + Pre-frozen UTF-8 Router (`0.00%` mid-UTF8 cuts)
     + `MuonAdamWHybrid` (`protect_crystal` + `Woodbury` WD=0 discipline).
  2. `N1 Dual-Timescale ByteInductionCache` (`2 x 1024` slots, `d_sig=8, d_v=16`, `+201 KB` O(1)):
     - Scale 0 (`k=4`): fast in-context prose/code/JSON repeater (slashes `Val` & `OW6` BPB).
     - Scale 1 (`k=7`): sparse long-horizon verbatim needle vault (`<= 8%` write rate -> survives `32,768 B`!).
  3. `S_5` Recovery (`S1A = 2000` steps + `S1B = 500` + `S2 = 1600` with `9.4%` dense multi-format MQAR on web batches).
  4. `800`-step `4-Group + 4-Tap + Logit-KL STE QAT` -> strict `strict=True` disk roundtrip + paired `eval_bs=8`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

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
)
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
    _run_4tap_qat_step,
    _run_accum_microbatches,
    _snapshot_router_params,
)
from scripts.colab_phase_l_v2_runner import (  # noqa: E402
    benchmark_stream_engine_live,
    build_eval_bundles,
    compile_v2_swa_blocks,
    ensure_phase_k_corpus,
    eval_long_context_stress,
    evaluate_full_model,
    inject_l2048_from_extrap,
    load_frozen_router_into,
    set_all_seeds,
)
from scripts.muon_opt import MuonAdamWHybrid  # noqa: E402
from scripts.run_phase_l_closure_25m import (  # noqa: E402
    _default_micro_bs,
    _recommended_lut_cog_bs,
    load_calcite_v2_lut_checkpoint,
)
from scripts.run_phase_m_stage_c_v3_5m import _split_v3_qat_4groups  # noqa: E402
from scripts.run_phase_n_mqar_probe_5m import eval_ood_verbatim_ladder  # noqa: E402
from scripts.run_phase_n_v31_lm_check_5m import (  # noqa: E402
    eval_live_stream_needle,
    eval_overwrite_needle_ood,
    sample_multiformat_mqar_row,
)
from simvq_split.eval_utils import evaluate_suite_batched  # noqa: E402
from simvq_split.memory import (  # noqa: E402
    compute_2bit_lut_footprint,
    quantize_ternary_weights_ls,
)
from simvq_split.tasks import load_gsm8k_corpora  # noqa: E402
from simvq_split.tokenizer import ByteBPETokenizer  # noqa: E402


def build_calcite_v31_25m_config(smoke: bool = False, slots: int = 1024) -> CalciteV2Config:
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
    # Verified Phase M + Phase N winners: M3 + M8 (Byte Engram) + N1 (Dual-Timescale ByteInductionCache)
    cfg.use_sym_byte_swa = True
    cfg.byte_swa_window = 24
    cfg.byte_swa_dim = 64
    cfg.use_byte_engram = True
    cfg.engram_slots = 4096
    cfg.engram_dim = 32
    cfg.use_byte_induct_cache = True
    cfg.byte_induct_slots = 128 if smoke else int(slots)
    cfg.byte_induct_dim = 16
    cfg.byte_chunk_mode = "utf8"
    cfg.use_static_m_cap = True
    return cfg


def _inject_mqar_rows(
    xb_full: torch.Tensor,
    yb_full: torch.Tensor,
    train_mm: np.ndarray,
    n_copy: int,
    rng_py: random.Random,
    rng_np: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replaces `n_copy` evenly spaced rows in an OpenWeb batch with dense multi-format MQAR rows
    and returns `(xb, yb, copy_mask)` so every microbatch gets clean balanced copy supervision.
    """
    B, T = xb_full.shape
    n_copy = max(1, min(B - 1, int(n_copy)))
    dev = xb_full.device
    xb = xb_full.clone()
    yb = yb_full.clone()
    mb = torch.zeros((B, T), dtype=torch.bool, device=dev)
    stride = max(1, B // n_copy)
    idxs = [min(B - 1, i * stride) for i in range(n_copy)]
    xs, ys, ms = [], [], []
    for _ in idxs:
        xr, yr, mr = sample_multiformat_mqar_row(train_mm, T, rng_py, rng_np)
        xs.append(xr)
        ys.append(yr)
        ms.append(mr)
    idx_t = torch.tensor(idxs, dtype=torch.long)
    xs_t = torch.from_numpy(np.asarray(xs, dtype=np.int64))
    ys_t = torch.from_numpy(np.asarray(ys, dtype=np.int64))
    ms_t = torch.from_numpy(np.asarray(ms, dtype=np.bool_))
    if dev.type == "cuda":
        idx_t = idx_t.pin_memory().to(device=dev, non_blocking=True)
        xs_t = xs_t.pin_memory().to(device=dev, non_blocking=True)
        ys_t = ys_t.pin_memory().to(device=dev, non_blocking=True)
        ms_t = ms_t.pin_memory().to(device=dev, non_blocking=True)
    elif dev.type != "cpu":
        idx_t = idx_t.to(device=dev)
        xs_t = xs_t.to(device=dev)
        ys_t = ys_t.to(device=dev)
        ms_t = ms_t.to(device=dev)
    xb[idx_t] = xs_t
    yb[idx_t] = ys_t
    mb[idx_t] = ms_t
    return xb, yb, mb


def run_v31_closure_25m(
    data_dir: Path,
    out_dir: Path,
    ckpt_dir: Path,
    s1a_steps: int = 2000,
    s1b_steps: int = 500,
    s2_steps: int = 1600,
    qat_steps: int = 800,
    slots: int = 1024,
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
            torch.cuda.set_per_process_memory_fraction(0.90, 0)
    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tok = ByteBPETokenizer(merges=[])
    tr_docs, te_docs, tr_sents, _ = load_gsm8k_corpora(ROOT_DIR)
    train_wins, eval_wins, _, _ = build_disjoint_openweb_corpora(
        ROOT_DIR, tok, tr_docs, te_docs, seq_len=128 if smoke else seq_len
    )
    n_ev = 6 if smoke else 150
    _, _, s5_suites, ov_suites, trap_suites = build_eval_bundles(
        ROOT_DIR, tok, n_eval=n_ev, seq_len=128 if smoke else seq_len
    )

    router_path = ROOT_DIR / "benchmarks" / "phase_l_v2" / "frozen_utf8_router.pt"
    assert router_path.is_file(), f"Missing canonical frozen UTF-8 router: {router_path}"
    router_state = torch.load(router_path, map_location="cpu", weights_only=True)

    cfg = build_calcite_v31_25m_config(smoke=smoke, slots=slots)
    if smoke:
        rng_f = np.random.default_rng(seed)
        fake_arr = rng_f.integers(32 + 3, 126 + 3, size=100_000, dtype=np.uint16)
        train_mm = val_mm = fake_arr
        s1a_steps, s1b_steps, s2_steps, qat_steps = 4, 2, 4, 4
        batch_size, seq_len = 8, 128
    else:
        train_bin, val_bin = ensure_phase_k_corpus(data_dir)
        train_mm = load_utf8_corpus_array(train_bin)
        val_mm = load_utf8_corpus_array(val_bin)

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
    mqar_rng_py = random.Random(seed + 7001)
    mqar_rng_np = np.random.default_rng(seed + 7001)

    set_all_seeds(seed)
    model = CalciteV2LM(cfg).to(device)
    load_frozen_router_into(model, router_state)
    fp_router_snap = _snapshot_router_params(model)
    n_params = int(sum(p.numel() for p in model.parameters()))
    cache_kb = round(compute_v2_state_cache_kb(cfg, 32768), 2)
    micro_bs = min(batch_size, _default_micro_bs(device, is_qat=False))
    print(
        f"\n=== [PHASE N CLOSURE: Calcite v3.1 S2_25M] params={n_params:,} | O(1)_cache={cache_kb:.2f} KB | "
        f"slots=2x{cfg.byte_induct_slots} | device={device} | micro_bs={micro_bs} | "
        f"steps=({s1a_steps}+{s1b_steps}+{s2_steps}, qat={qat_steps}) ===",
        flush=True,
    )

    s1b_ckpt = out_dir / "closure_v31_25m_s1b_fp.pt"
    s2_ckpt = out_dir / "closure_v31_25m_stage2_fp.pt"
    lut_ckpt = ckpt_dir / ("smoke_v31_25m_lut.pt" if smoke else "calcite_v31_s2_25m_utf8_lut.pt")
    res_path = out_dir / ("smoke_v31_25m_results.json" if smoke else "phase_n_closure_25m_results.json")

    t_all = time.perf_counter()
    s1a_s5_k2 = s1_s5_k2 = s1_ov_long = s1_trap_long = 0.0

    if (not smoke) and (not force) and s2_ckpt.is_file():
        print(f"  [Resume] Loading saved Stage-2 FP checkpoint -> {s2_ckpt}", flush=True)
        blob = torch.load(s2_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["state_dict"], strict=True)
        model.byte_chunker.lock_router(True)
        s1a_s5_k2 = float(blob.get("s1a_s5_k2", 0.0))
        s1_s5_k2 = float(blob.get("s1_s5_k2", 0.0))
        s1_ov_long = float(blob.get("s1_ov_long", 0.0))
        s1_trap_long = float(blob.get("s1_trap_long", 0.0))
        _assert_router_unchanged(model, fp_router_snap, "V31_25M_RESUME_S2")
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
            _assert_router_unchanged(model, fp_router_snap, "V31_25M_RESUME_S1B")
        else:
            # ----------------------------------------------------------
            # STAGE 1A: Algebraic S_5 Grokking (`2000` steps -> recovers S_5 >= 155%)
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
            live_diag = (1000, 1500, s1a_steps) if ((not smoke) and s1a_steps >= 1000) else ()
            t_1a = time.perf_counter()
            s1a_it = iter(range(1, s1a_steps + 1))

            def _sample_n_s1a():
                st = next(s1a_it)
                return batcher.sample_batch(
                    batch_size,
                    "cpu",
                    step=st,
                    grok_steps=grok_steps,
                    total_steps=total_fp_steps,
                    s1b_steps=s1b_steps,
                    s1a_offset=0,
                )

            with BatchPrefetcher(
                _sample_n_s1a, device=device, total_steps=s1a_steps, prefetch_depth=4, enabled=(s1a_steps >= 4)
            ) as pf_1a:
                for step in range(1, s1a_steps + 1):
                    model.train()
                    model.set_crystal_schedule(step, s1a_steps)
                    lr_s = _cosine_lr(step, s1a_steps, warm_1a, base_lr_1a, 2.0e-4)
                    sc = lr_s / base_lr_1a
                    for pg in opt_1a.param_groups:
                        pg["lr"] = base_lr_1a * float(pg.get("lr_mult", 1.0)) * sc
                    xb_full, yb_full = pf_1a.next_batch()
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
                    torch.nn.utils.clip_grad_norm_(opt_1a.trainable_params, 1.0)
                    opt_1a.step()

                    if step in (1, s1a_steps) or (step % 250 == 0):
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
            s1b_it = iter(range(1, s1b_steps + 1))

            def _sample_n_s1b():
                st = s1a_steps + next(s1b_it)
                return batcher.sample_batch(
                    batch_size, "cpu", step=st, grok_steps=grok_steps, total_steps=total_fp_steps, s1b_steps=s1b_steps
                )

            with BatchPrefetcher(
                _sample_n_s1b, device=device, total_steps=s1b_steps, prefetch_depth=4, enabled=(s1b_steps >= 4)
            ) as pf_1b:
                for o_step in range(1, s1b_steps + 1):
                    model.train()
                    model.set_crystal_schedule(s1a_steps, s1a_steps)
                    model.set_snap_alpha(1.0)
                    lr_s = _cosine_lr(o_step, s1b_steps, warm_1b, base_lr_1b, 1.5e-4)
                    sc = lr_s / base_lr_1b
                    for pg in opt_1b.param_groups:
                        pg["lr"] = base_lr_1b * float(pg.get("lr_mult", 1.0)) * sc
                    xb_full, yb_full = pf_1b.next_batch()
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
                    torch.nn.utils.clip_grad_norm_(opt_1b.trainable_params, 1.0)
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
        # STAGE 2: UTF-8 OpenWeb + 9.4% Verbatim MQAR + Cognitive Maintenance (`MuonAdamW`)
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
        n_copy_s2 = max(1, int(round(batch_size * 0.094)))
        t_2 = time.perf_counter()
        s2_it = iter(range(1, s2_steps + 1))

        def _sample_n_s2():
            st = grok_steps + next(s2_it)
            xb_c, yb_c, is_web = batcher.sample_batch(
                batch_size,
                "cpu",
                step=st,
                grok_steps=grok_steps,
                total_steps=total_fp_steps,
                s1b_steps=s1b_steps,
                return_is_web=True,
            )
            if is_web:
                return _inject_mqar_rows(xb_c, yb_c, train_mm, n_copy_s2, mqar_rng_py, mqar_rng_np)
            return xb_c, yb_c, None

        with BatchPrefetcher(
            _sample_n_s2, device=device, total_steps=s2_steps, prefetch_depth=4, enabled=(s2_steps >= 4)
        ) as pf_2:
            for s2 in range(1, s2_steps + 1):
                model.train()
                model.set_crystal_schedule(s1a_steps, s1a_steps)
                model.set_snap_alpha(1.0)
                lr_s = _cosine_lr(s2, s2_steps, warm_2, base_lr_2, 8.0e-5)
                sc = lr_s / base_lr_2
                for pg in opt_2.param_groups:
                    pg["lr"] = base_lr_2 * float(pg.get("lr_mult", 1.0)) * sc

                xb_full, yb_full, mb_full = pf_2.next_batch()
                is_log = (s2 in (1, s2_steps)) or (s2 % 200 == 0)

                # OOM-safe gradient accumulation with branchless zero-sync balanced copy CE
                while True:
                    opt_2.zero_grad(set_to_none=True)
                    n_acc = max(1, (batch_size + micro_bs - 1) // micro_bs)
                    ce_acc_t: Optional[torch.Tensor] = None
                    c_acc_t: Optional[torch.Tensor] = None
                    n_c_m_t: Optional[torch.Tensor] = None
                    try:
                        for st in range(0, batch_size, micro_bs):
                            xb = xb_full[st : st + micro_bs]
                            yb = yb_full[st : st + micro_bs]
                            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                                lg = model(xb)
                                lg_f = lg.float()
                                if mb_full is not None:
                                    mb = mb_full[st : st + micro_bs]
                                    tok_ce = F.cross_entropy(
                                        lg_f.reshape(-1, lg_f.size(-1)),
                                        yb.reshape(-1),
                                        ignore_index=tok.PAD_ID,
                                        reduction="none",
                                    ).view_as(yb)
                                    valid = yb != tok.PAD_ID
                                    v_f = valid.float()
                                    m_copy_f = (mb & valid).float()
                                    m_pref_f = ((~mb) & valid).float()
                                    n_c_tok = m_copy_f.sum()
                                    n_p_tok = m_pref_f.sum()
                                    ce_v_all = (tok_ce * v_f).sum() / v_f.sum().clamp(min=1.0)
                                    ce_p = torch.where(n_p_tok > 0.0, (tok_ce * m_pref_f).sum() / n_p_tok.clamp(min=1.0), ce_v_all)
                                    has_c = n_c_tok > 0.0
                                    ce_c = torch.where(has_c, (tok_ce * m_copy_f).sum() / n_c_tok.clamp(min=1.0), torch.zeros_like(ce_p))
                                    ce = torch.where(has_c, 0.85 * ce_p + 0.15 * ce_c, ce_p)
                                    if is_log:
                                        with torch.no_grad():
                                            eq_c = ((lg_f.argmax(dim=-1) == yb).float() * m_copy_f).sum() / n_c_tok.clamp(min=1.0)
                                            eq_c = torch.where(has_c, eq_c, torch.zeros_like(eq_c))
                                            has_c_f = has_c.float()
                                            c_acc_t = eq_c if c_acc_t is None else (c_acc_t + eq_c)
                                            n_c_m_t = has_c_f if n_c_m_t is None else (n_c_m_t + has_c_f)
                                else:
                                    ce = F.cross_entropy(
                                        lg_f.reshape(-1, lg_f.size(-1)),
                                        yb.reshape(-1),
                                        ignore_index=tok.PAD_ID,
                                    )
                                aux = model.get_commitment_loss()
                                loss = (ce + 0.04 * aux) / float(n_acc)
                            loss.backward()
                            if is_log:
                                ce_d = ce.detach()
                                ce_acc_t = ce_d if ce_acc_t is None else (ce_acc_t + ce_d)
                        break
                    except torch.cuda.OutOfMemoryError:
                        if micro_bs <= 2:
                            raise
                        micro_bs = max(2, micro_bs // 2)
                        opt_2.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        print(f"  [OOM adapt] S2 s={s2}: micro_bs -> {micro_bs}", flush=True)

                torch.nn.utils.clip_grad_norm_(opt_2.trainable_params, 1.0)
                opt_2.step()
                if is_log:
                    ce_sum_v = float(ce_acc_t.item()) if ce_acc_t is not None else 0.0
                    n_c_m = int(round(float(n_c_m_t.item()))) if n_c_m_t is not None else 0
                    c_acc_sum = float(c_acc_t.item()) if c_acc_t is not None else 0.0
                    bic = model.byte_chunker.byte_induct_cache
                    b_info = f" | bwr={bic.last_write_rate:.2f}(s1={bic.last_write_rate_s1:.2f})" if bic is not None else ""
                    c_info = f" | cAcc={100.0*c_acc_sum/n_c_m:5.1f}%" if n_c_m > 0 else ""
                    print(
                        f"  [S2-utf8] step {s2:4d}/{s2_steps} | ce={ce_sum_v/n_acc:.4f}{c_info}{b_info} | "
                        f"{(time.perf_counter()-t_2)/60:.1f}m",
                        flush=True,
                    )

        del opt_2
        if device == "cuda":
            torch.cuda.empty_cache()

        _assert_router_unchanged(model, fp_router_snap, "V31_25M_FP_POSTLOCK")
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

    # ----------------------------------------------------------
    # Evaluate FP Champion (`eval_bs=8` matched identically to LUT)
    # ----------------------------------------------------------
    model.eval()
    model.set_crystal_schedule(1, 1)
    model.set_snap_alpha(1.0)
    print("  [Eval] Evaluating Stage-2 FP Calcite v3.1 (25M) Champion...", flush=True)
    eval_fp = evaluate_full_model(
        model, val_mm, eval_wins, tok, s5_suites, ov_suites, trap_suites, device=device, quick=smoke, eval_bs=8
    )
    long_fp = inject_l2048_from_extrap(
        eval_long_context_stress(
            model, val_mm, device=device, lengths=(256, 512) if smoke else (8192, 32768)
        ),
        None if smoke else eval_fp.get("extrap_4x"),
    )
    ndl_hs = (256,) if smoke else (512, 1500, 4096, 8192, 32768)
    needle_fp = eval_ood_verbatim_ladder(
        model, val_mm, device=device, haystacks=ndl_hs, n_samples=4 if smoke else 16, val_len=4 if smoke else 8, seed=9901
    )
    ov_needle_fp = eval_overwrite_needle_ood(
        model, val_mm, device=device, haystack_bytes=256 if smoke else 1500, n_samples=4 if smoke else 16
    )
    print(
        f"  [FP DONE] Val={eval_fp['val_bpb']:.4f} | OW6={eval_fp['openweb_6']['mean_openweb_bpb']:.4f} | "
        f"L2048={long_fp.get('L_2048', {}).get('bpb')} | L32K={long_fp.get('L_32768', {}).get('bpb')} | "
        f"S5={eval_fp['cognitive']['s5_sum']:.1f}% | OV_L={eval_fp['cognitive']['overwrite']['ood_long']['exact_pct']:.1f}% | "
        f"Trap_L={eval_fp['cognitive']['trap']['ood_long']['exact_pct']:.1f}% | "
        f"SelF1_L={eval_fp['cognitive']['selective_f1_ood_long']:.1f}% | "
        f"Orig1500={needle_fp['OrigRefCode_1500B']['exact_match_pct']:.1f}% | "
        f"N32K={needle_fp.get('L_32768', {}).get('exact_match_pct')} | OvN={ov_needle_fp['new_exact_pct']:.1f}%",
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
    model.byte_chunker.use_static_m_cap = True
    qat_model.byte_chunker.use_static_m_cap = True
    qat_model.byte_chunker.lock_router(True)
    qat_model.set_crystal_schedule(1, 1)
    qat_model.set_snap_alpha(1.0)
    qat_model.set_qat_mix(0.30)
    qat_model.set_ternary_qat(True)
    if (not smoke) and device == "cuda":
        compile_v2_swa_blocks(qat_model)

    gain_q, cryst_q, chunk_q, core_q = _split_v3_qat_4groups(qat_model)
    all_qp = gain_q + cryst_q + chunk_q + core_q
    w_scale = (384.0 / float(cfg.d_model)) ** 0.5
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
    n_copy_q = max(1, int(round(batch_size * 0.0625)))
    t_q0 = time.perf_counter()

    def _sample_n_qat():
        xb_c, yb_c, is_web = batcher.sample_batch(
            batch_size,
            "cpu",
            step=total_fp_steps,
            grok_steps=grok_steps,
            total_steps=total_fp_steps,
            s1b_steps=s1b_steps,
            return_is_web=True,
        )
        if is_web:
            xb_c, yb_c, _ = _inject_mqar_rows(xb_c, yb_c, train_mm, n_copy_q, mqar_rng_py, mqar_rng_np)
        return xb_c, yb_c

    try:
        with BatchPrefetcher(
            _sample_n_qat, device=device, total_steps=qat_steps, prefetch_depth=4, enabled=(qat_steps >= 4)
        ) as pf_q:
            for qs in range(1, qat_steps + 1):
                qat_model.train()
                mix = min(1.0, 0.30 + 0.70 * (qs / float(ramp_q)))
                qat_model.set_qat_mix(mix)
                clear_fast_qat_cache()
                for gi, (blr, mlr) in enumerate(zip(q_lrs, q_mins)):
                    opt_q.param_groups[gi]["lr"] = _cosine_lr(qs, qat_steps, warm_q, blr, mlr)
                xb_full, yb_full = pf_q.next_batch()
                micro_bs_q, q_ce_sum_t = _run_4tap_qat_step(
                    model,
                    qat_model,
                    taps_fp,
                    taps_q,
                    xb_full,
                    yb_full,
                    micro_bs_q,
                    opt_q,
                    tok.PAD_ID,
                    kl_temp,
                    use_amp,
                    device,
                    f"4Tap-QAT qs={qs}",
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

    router_mut_diff = _assert_router_unchanged(qat_model, fp_router_snap, "V31_25M_QAT")
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
            "phase": "Phase_N_Closure_V31_25M",
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

    lut_cog_bs = _recommended_lut_cog_bs(device, binfo)
    lut_gpu_rt, _, _, _ = load_calcite_v2_lut_checkpoint(lut_ckpt, device=device, backend=lut_bk)
    print(f"  [Eval] Evaluating 2-bit LUT Calcite v3.1 (25M) from disk roundtrip (cog_bs={lut_cog_bs})...", flush=True)
    eval_lut = evaluate_full_model(
        lut_gpu_rt, val_mm, eval_wins, tok, s5_suites, ov_suites, trap_suites, device=device, quick=smoke, eval_bs=8, cog_bs=lut_cog_bs
    )
    if device == "cuda":
        torch.cuda.empty_cache()
    long_lut = inject_l2048_from_extrap(
        eval_long_context_stress(
            lut_gpu_rt, val_mm, device=device, lengths=(256, 512) if smoke else (8192, 32768)
        ),
        None if smoke else eval_lut.get("extrap_4x"),
    )
    if device == "cuda":
        torch.cuda.empty_cache()
    needle_lut = eval_ood_verbatim_ladder(
        lut_gpu_rt, val_mm, device=device, haystacks=ndl_hs, n_samples=4 if smoke else 16, val_len=4 if smoke else 8, seed=9901
    )
    ov_needle_lut = eval_overwrite_needle_ood(
        lut_gpu_rt, val_mm, device=device, haystack_bytes=256 if smoke else 1500, n_samples=4 if smoke else 16
    )
    stream_gpu = benchmark_stream_engine_live(lut_gpu_rt, device=device, n_bytes=32 if smoke else 256)
    stream_needle_lut = eval_live_stream_needle(
        lut_gpu_rt, val_mm, device=device, haystack_bytes=256 if smoke else 1500, n_samples=2 if smoke else 4
    )
    del lut_gpu_rt
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    val_gap = round(float(eval_lut["val_bpb"]) - float(eval_fp["val_bpb"]), 4)
    ow6_gap = round(
        float(eval_lut["openweb_6"]["mean_openweb_bpb"]) - float(eval_fp["openweb_6"]["mean_openweb_bpb"]), 4
    )
    print(
        f"  [2bit-LUT DONE] Val={eval_lut['val_bpb']:.4f} (gap {val_gap:+.4f}) | "
        f"OW6={eval_lut['openweb_6']['mean_openweb_bpb']:.4f} (gap {ow6_gap:+.4f}) | "
        f"L32K={long_lut.get('L_32768', {}).get('bpb')} | S5={eval_lut['cognitive']['s5_sum']:.1f}% | "
        f"OV_L={eval_lut['cognitive']['overwrite']['ood_long']['exact_pct']:.1f}% | "
        f"Trap_L={eval_lut['cognitive']['trap']['ood_long']['exact_pct']:.1f}% | "
        f"SelF1_L={eval_lut['cognitive']['selective_f1_ood_long']:.1f}% | "
        f"Orig1500={needle_lut['OrigRefCode_1500B']['exact_match_pct']:.1f}% | "
        f"N32K={needle_lut.get('L_32768', {}).get('exact_match_pct')} | OvN={ov_needle_lut['new_exact_pct']:.1f}% | "
        f"StreamNeedle={stream_needle_lut['stream_greedy_exact_pct']:.1f}%",
        flush=True,
    )

    payload = {
        "meta": {
            "phase": "Phase_N_Closure_Calcite_v31_S2_25M",
            "params": n_params,
            "cache_kb_o1": cache_kb,
            "byte_induct_slots": cfg.byte_induct_slots,
            "steps": {"s1a": s1a_steps, "s1b": s1b_steps, "s2": s2_steps, "qat": qat_steps},
            "seed": seed,
            "smoke": smoke,
            "elapsed_s": round(time.perf_counter() - t_all, 1),
            "router_max_abs_diff": router_mut_diff,
            "lut_checkpoint": str(lut_ckpt.relative_to(ROOT_DIR)) if lut_ckpt.is_relative_to(ROOT_DIR) else str(lut_ckpt),
            "lut_checkpoint_bytes": ckpt_bytes,
            "qat_seconds": qat_s,
            "stage1_gates": {
                "s1a_s5_k2": s1a_s5_k2,
                "s1_s5_k2": s1_s5_k2,
                "s1_ov_long": s1_ov_long,
                "s1_trap_long": s1_trap_long,
            },
        },
        "eval_fp": eval_fp,
        "long_fp": long_fp,
        "needle_fp": needle_fp,
        "overwrite_needle_fp": ov_needle_fp,
        "eval_lut": eval_lut,
        "long_lut": long_lut,
        "needle_lut": needle_lut,
        "overwrite_needle_lut": ov_needle_lut,
        "lut_gaps": {"val_bpb_gap": val_gap, "ow6_mean_bpb_gap": ow6_gap},
        "lut_footprint": lut_foot,
        "stream_cpu_rt": stream_cpu,
        "stream_gpu_rt": stream_gpu,
        "stream_needle_lut": stream_needle_lut,
    }
    res_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[Phase N Closure 25M Complete] Saved -> {res_path}", flush=True)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT_DIR / "data_cache")
    ap.add_argument("--out-dir", type=Path, default=ROOT_DIR / "benchmarks" / "phase_n_needle")
    ap.add_argument("--ckpt-dir", type=Path, default=ROOT_DIR / "checkpoints")
    ap.add_argument("--s1a-steps", type=int, default=2000)
    ap.add_argument("--s1b-steps", type=int, default=500)
    ap.add_argument("--s2-steps", type=int, default=1600)
    ap.add_argument("--qat-steps", type=int, default=800)
    ap.add_argument("--slots", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke-local", action="store_true")
    args = ap.parse_args()

    run_v31_closure_25m(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        ckpt_dir=args.ckpt_dir,
        s1a_steps=args.s1a_steps,
        s1b_steps=args.s1b_steps,
        s2_steps=args.s2_steps,
        qat_steps=args.qat_steps,
        slots=args.slots,
        batch_size=args.batch_size,
        seed=args.seed,
        smoke=args.smoke_local,
        force=args.force,
    )


if __name__ == "__main__":
    main()
