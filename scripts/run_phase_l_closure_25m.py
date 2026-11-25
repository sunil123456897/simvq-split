"""
Phase L Unified Closure Runner (`scripts/run_phase_l_closure_25m.py`):
Combines `Calcite v2` (`A1b` 2-scale causal gated byte-mixer + `O(T*W)` SWA) with:
  1. Exact Phase K 3-Stage Curriculum (`Stage 1A` S_5 grokking -> `Stage 1B` Woodbury consolidation ->
     `Stage 2` protected 6-domain OpenWeb + cognitive replay) using `StratifiedSystem1Batcher` and
     per-row normalized `_fp_micro_ce`.
  2. Exact Phase H/K `4-Group + 4-Tap + Logit-KL STE QAT` (`500` steps) collapsing the `2-bit LUT` gap.
  3. Immediate stage checkpointing (`stage1b_fp`, `stage2_fp`, `calcite_v2_s2_25m_utf8_lut.pt`) BEFORE eval.
  4. Complete evaluation: `Val BPB`, `6` OpenWeb domains (`0.00%` mid-UTF8 cuts), `S_5`, `Overwrite`,
     `Trap`, `Selective F1`, `L = 2K / 8K / 32K` (`64x`) stress test, and live `CalciteStreamEngine` (GPU & CPU).
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import gc
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Dict, List, Optional, Tuple

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
from experiments.scaling_phase_k.eval_suite import (  # noqa: E402
    BatchPrefetcher,
    has_usable_cuda,
    load_utf8_corpus_array,
    sync_if_cuda,
)
from kernels.fast_calcite_scans import (  # noqa: E402
    clear_fast_qat_cache,
    get_fast_scans_backend_info,
    unwrap_compiled_state_dict,
)
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
    _calibrate_non_ascii_router_emb,
    _cosine_lr,
    _fp_micro_ce,
    _make_adamw,
    _qat_micro_step,
    _run_4tap_qat_step,
    _run_accum_microbatches,
    _snapshot_router_params,
    _split_calcite_1b_groups,
    _split_calcite_fp_groups,
    _split_calcite_stage2_groups,
)
from scripts.colab_phase_l_v2_runner import (  # noqa: E402
    benchmark_stream_engine_live,
    build_eval_bundles,
    compile_v2_swa_blocks,
    ensure_phase_k_corpus,
    eval_long_context_stress,
    evaluate_full_model,
    inject_l2048_from_extrap,
)
from simvq_split.baselines import RMSNorm  # noqa: E402
from simvq_split.eval_utils import evaluate_suite_batched  # noqa: E402
from simvq_split.memory import (  # noqa: E402
    compute_2bit_lut_footprint,
    quantize_ternary_weights_ls,
)
from simvq_split.tasks import load_gsm8k_corpora  # noqa: E402
from simvq_split.tokenizer import ByteBPETokenizer  # noqa: E402


def _split_v2_qat_4groups(
    model: CalciteV2LM,
) -> Tuple[List[nn.Parameter], List[nn.Parameter], List[nn.Parameter], List[nn.Parameter]]:
    """
    4-Group QAT parameter partition adapted cleanly for `CalciteV2LM`:
    - `gain_p`: `tok_emb`, all `RMSNorm`, `stream_gate/cryst2_gate`, `byte_offset_emb`, and FP16 `dec_dw*_conv`.
    - `cryst_p`: `crystal`, `crystal_2`, and `woodbury` scalar thresholds/lags.
    - `chunk_p`: quantized `byte_chunker` linear weights (`enc_*`, `dec_fuse`, `dec_dw*_gate/out`, `dec_mlp`).
    - `core_p`: `cortex_pre`, `cortex_post`, `mid_mlp`, and `woodbury` 2D projections.
    """
    gain_ids: set[int] = {id(model.tok_emb.weight)}
    for mod in model.modules():
        if isinstance(mod, (RMSNorm, nn.Conv1d)):
            for p in mod.parameters():
                if p.requires_grad:
                    gain_ids.add(id(p))
    if model.byte_chunker.byte_offset_emb is not None:
        gain_ids.add(id(model.byte_chunker.byte_offset_emb.weight))
    for gm in (model.stream_gate, model.cryst2_gate):
        if gm is not None:
            for p in gm.parameters():
                if p.requires_grad:
                    gain_ids.add(id(p))

    gain_p, cryst_p, chunk_p, core_p = [], [], [], []
    seen: set[int] = set()
    for nm, p in model.named_parameters():
        if (not p.requires_grad) or (id(p) in seen):
            continue
        seen.add(id(p))
        is_wood_scalar = nm.startswith("woodbury.") and (
            (p.ndim < 2 and "out_norm" not in nm) or ("subj_lag_bias" in nm)
        )
        if is_wood_scalar:
            cryst_p.append(p)
        elif id(p) in gain_ids:
            gain_p.append(p)
        elif nm.startswith(("crystal.", "crystal_2.")):
            cryst_p.append(p)
        elif nm.startswith("byte_chunker."):
            chunk_p.append(p)
        else:
            core_p.append(p)
    return gain_p, cryst_p, chunk_p, core_p


def _default_micro_bs(device: str, is_qat: bool) -> int:
    if device != "cuda" or not has_usable_cuda():
        return 4
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if vram_gb >= 35.0:
        return 64
    if vram_gb >= 20.0:
        return 32 if is_qat else 64
    return 16 if is_qat else 32


def _recommended_lut_cog_bs(device: str, binfo: Optional[Dict[str, object]] = None) -> int:
    """Returns safe cognitive eval batch size for 2-bit LUT (`16` with `cuda_ext` or `n_blk=512`)."""
    if device != "cuda" or not has_usable_cuda():
        return 16
    info = binfo if binfo is not None else get_lut_backend_info()
    if info.get("cuda_ext_available"):
        return 16
    return 16


def run_closure_25m(
    data_dir: Path,
    out_dir: Path,
    ckpt_dir: Path,
    s1a_steps: int = 2400,
    s1b_steps: int = 800,
    s2_steps: int = 1600,
    qat_steps: int = 500,
    batch_size: int = 64,
    seq_len: int = 512,
    core_lr: float = 1.4e-3,
    stage2_core_lr: float = 7.5e-4,
    stage2_emb_lr: float = 1.10e-3,
    min_lr: float = 2.0e-5,
    smoke: bool = False,
    compile_swa: bool = True,
) -> Dict[str, object]:
    device = "cuda" if has_usable_cuda() else "cpu"
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

    if smoke:
        rng_f = np.random.default_rng(2026)
        fake_arr = rng_f.integers(3, 128, size=100_000, dtype=np.uint16)
        train_mm = fake_arr
        val_mm = fake_arr
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
        s1a_steps, s1b_steps, s2_steps, qat_steps = 6, 4, 4, 4
        batch_size, seq_len = 8, 128
        compile_swa = False
    else:
        train_bin, val_bin = ensure_phase_k_corpus(data_dir)
        train_mm = load_utf8_corpus_array(train_bin)
        val_mm = load_utf8_corpus_array(val_bin)
        cfg = build_calcite_v2_tier_config(
            "S2_25M",
            topology="hourglass",
            use_scaled_decoder=True,
            use_kalman_precond=False,
            use_latent_feedback=False,
            use_clean_gates=False,
        )

    grok_steps = s1a_steps + s1b_steps
    total_fp_steps = grok_steps + s2_steps

    batcher = StratifiedSystem1Batcher(
        train_mm=train_mm,
        tr_docs=tr_docs,
        tr_sents=tr_sents,
        tok=tok,
        seq_len=seq_len,
        train_wins=train_wins,
        seed=2026,
    )

    torch.manual_seed(102)
    random.seed(102)
    np.random.seed(102)

    model = CalciteV2LM(cfg).to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    micro_bs = min(batch_size, _default_micro_bs(device, is_qat=False))
    if compile_swa and device == "cuda":
        compile_v2_swa_blocks(model)

    fk_info = get_fast_scans_backend_info()
    print(
        f"\n=== [Phase L Closure | Calcite v2 {'SMOKE' if smoke else 'S2_25M'}] "
        f"params={n_params:,} | FastScans={fk_info.get('cuda_ext_available')} | "
        f"s1a={s1a_steps} + s1b={s1b_steps} + s2={s2_steps} + qat={qat_steps} | "
        f"B={batch_size}(mbs={micro_bs}) L={seq_len} ===",
        flush=True,
    )

    s1b_ckpt = out_dir / ("smoke_s1b.pt" if smoke else "closure_25m_stage1b_fp.pt")
    s2_ckpt = out_dir / ("smoke_s2.pt" if smoke else "closure_25m_stage2_fp.pt")
    fp_router_snap: Dict[str, torch.Tensor] = {}
    s1a_s5_k2 = 0.0
    s1_s5_k2 = 0.0
    s1_ov_long = 0.0
    s1_trap_long = 0.0
    t_all0 = time.perf_counter()

    if (not smoke) and s2_ckpt.is_file():
        print(f"  [Resume] Loading finished Stage-2 FP from {s2_ckpt}", flush=True)
        blob = torch.load(s2_ckpt, map_location="cpu", weights_only=False)
        model = CalciteV2LM(cfg).to(device)
        model.load_state_dict(blob["state_dict"], strict=True)
        model.byte_chunker.byte_chunk_mode = "utf8"
        model.byte_chunker.target_chunk_bytes = float(cfg.target_chunk_bytes)
        model.byte_chunker.use_static_m_cap = True
        model.byte_chunker.lock_router(True)
        fp_router_snap = _snapshot_router_params(model)
        s1a_s5_k2 = float(blob.get("s1a_s5_k2", 0.0))
        s1_s5_k2 = float(blob.get("s1_s5_k2", 0.0))
        s1_ov_long = float(blob.get("s1_ov_long", 0.0))
        s1_trap_long = float(blob.get("s1_trap_long", 0.0))
    else:
        skip_s1 = (not smoke) and s1b_ckpt.is_file()
        if skip_s1:
            print(f"  [Resume] Loading finished Stage-1B FP from {s1b_ckpt}", flush=True)
            blob = torch.load(s1b_ckpt, map_location="cpu", weights_only=False)
            model = CalciteV2LM(cfg).to(device)
            model.load_state_dict(blob["state_dict"], strict=True)
            if compile_swa and device == "cuda":
                compile_v2_swa_blocks(model)
            model.byte_chunker.byte_chunk_mode = "utf8"
            model.byte_chunker.target_chunk_bytes = float(cfg.target_chunk_bytes)
            model.byte_chunker.use_static_m_cap = False
            model.byte_chunker.lock_router(True)
            fp_router_snap = _snapshot_router_params(model)
            s1a_s5_k2 = float(blob.get("s1a_s5_k2", 0.0))
            s1_s5_k2 = float(blob.get("s1_s5_k2", 0.0))
            s1_ov_long = float(blob.get("s1_ov_long", 0.0))
            s1_trap_long = float(blob.get("s1_trap_long", 0.0))
        else:
            # ----------------------------------------------------------
            # STAGE 1A: Unmasked prose router warmup -> Crystal S_5 grokking
            # ----------------------------------------------------------
            model.byte_chunker.byte_chunk_mode = "byte"
            model.byte_chunker.target_chunk_bytes = 4.2
            model.byte_chunker.use_static_m_cap = False

            lock_step = int(float(cfg.router_lock_frac) * s1a_steps) if ((not smoke) and s1a_steps >= 100) else 2
            cryst_p, chunk_p, core_p, wood_p = _split_calcite_fp_groups(model)
            all_1a = cryst_p + chunk_p + core_p + wood_p
            fixed_lr_1 = max(core_lr, 2.0e-3)
            base_lrs_1 = (fixed_lr_1, 2.5e-3, core_lr, 0.05 * core_lr)
            min_lrs_1 = (1.5e-4, 1.0e-4, 0.08 * core_lr, 0.01 * core_lr)
            opt_1a = _make_adamw(
                [
                    {"params": cryst_p, "lr": base_lrs_1[0], "weight_decay": 0.01},
                    {"params": chunk_p, "lr": base_lrs_1[1], "weight_decay": 0.01},
                    {"params": core_p, "lr": base_lrs_1[2], "weight_decay": 0.01},
                    {"params": wood_p, "lr": base_lrs_1[3], "weight_decay": 0.0},
                ],
                device,
                betas=(0.9, 0.999),
            )
            warm_1 = min(60, max(1, s1a_steps // 12))
            live_diag = (1200, 1800, s1a_steps) if ((not smoke) and s1a_steps >= 1000) else ()
            t_1a = time.perf_counter()
            s1a_step_it = iter(range(1, s1a_steps + 1))

            def _sample_s1a() -> Tuple[torch.Tensor, torch.Tensor]:
                st = next(s1a_step_it)
                if lock_step > 0 and st <= lock_step:
                    return batcher.sample_prose_batch(batch_size, "cpu")
                return batcher.sample_batch(
                    batch_size,
                    "cpu",
                    step=st,
                    grok_steps=grok_steps,
                    total_steps=total_fp_steps,
                    s1b_steps=s1b_steps,
                    s1a_offset=lock_step,
                )

            with BatchPrefetcher(
                _sample_s1a, device=device, total_steps=s1a_steps, prefetch_depth=4, enabled=(s1a_steps >= 4)
            ) as pf_1a:
                for step in range(1, s1a_steps + 1):
                    model.train()
                    model.set_crystal_schedule(step, s1a_steps)
                    for gi, (blr, mlr) in enumerate(zip(base_lrs_1, min_lrs_1)):
                        opt_1a.param_groups[gi]["lr"] = _cosine_lr(step, s1a_steps, warm_1, blr, mlr)
                    if lock_step > 0 and step <= lock_step:
                        opt_1a.param_groups[1]["lr"] = _cosine_lr(
                            step, lock_step, min(40, max(1, lock_step // 6)), 2.5e-3, 4.0e-4
                        )
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
                    torch.nn.utils.clip_grad_norm_(all_1a, 1.0)
                    opt_1a.step()

                    if step in (1, s1a_steps) or (step % 400 == 0):
                        ce_v = float(ce_sum_t.item()) / float(max(1, batch_size // micro_bs)) if ce_sum_t is not None else 0.0
                        b_ch = model.byte_chunker.last_chunk_ratio
                        print(
                            f"  [S1A] step {step:4d}/{s1a_steps} | b/ch={b_ch:.2f} | ce={ce_v:.4f} | "
                            f"{(time.perf_counter()-t_1a)/60:.1f}m",
                            flush=True,
                        )
                    if step in live_diag:
                        was_tr = model.training
                        model.eval()
                        model.set_snap_alpha(1.0)
                        pk2, _, _, _ = evaluate_suite_batched(
                            model, s5_suites["k2"][:32], pad_id=tok.PAD_ID, batch_size=16, device=device
                        )
                        print(f"  [S1A LIVE step={step}] s5_k2_32={pk2:.1f}%", flush=True)
                        model.train(was_tr)
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
            # STAGE 1B: Associative Woodbury Consolidation (router locked)
            # ----------------------------------------------------------
            model.byte_chunker.lock_router(True)
            c1b, pre_n, pre_2, post_n, post_2, wood_n, wood_2 = _split_calcite_1b_groups(model)
            all_1b = c1b + pre_n + pre_2 + post_n + post_2 + wood_n + wood_2
            pre_lr = min(3.0e-4, 0.30 * core_lr)
            post_lr = min(1.2e-3, 1.00 * core_lr)
            wood_lr = min(2.4e-3, 2.20 * core_lr)
            base_lrs_1b = (4.0e-5, pre_lr, pre_lr, post_lr, post_lr, wood_lr, wood_lr)
            opt_1b = _make_adamw(
                [
                    {"params": c1b, "lr": base_lrs_1b[0], "weight_decay": 0.0},
                    {"params": pre_n, "lr": base_lrs_1b[1], "weight_decay": 0.0},
                    {"params": pre_2, "lr": base_lrs_1b[2], "weight_decay": 0.01},
                    {"params": post_n, "lr": base_lrs_1b[3], "weight_decay": 0.0},
                    {"params": post_2, "lr": base_lrs_1b[4], "weight_decay": 0.01},
                    {"params": wood_n, "lr": base_lrs_1b[5], "weight_decay": 0.0},
                    {"params": wood_2, "lr": base_lrs_1b[6], "weight_decay": 0.01},
                ],
                device,
                betas=(0.9, 0.999),
            )
            warm_1b = min(40, max(1, s1b_steps // 8))
            t_1b = time.perf_counter()
            s1b_step_it = iter(range(1, s1b_steps + 1))

            def _sample_s1b() -> Tuple[torch.Tensor, torch.Tensor]:
                st = s1a_steps + next(s1b_step_it)
                return batcher.sample_batch(
                    batch_size, "cpu", step=st, grok_steps=grok_steps, total_steps=total_fp_steps, s1b_steps=s1b_steps
                )

            with BatchPrefetcher(
                _sample_s1b, device=device, total_steps=s1b_steps, prefetch_depth=4, enabled=(s1b_steps >= 4)
            ) as pf_1b:
                for o_step in range(1, s1b_steps + 1):
                    model.train()
                    model.set_crystal_schedule(s1a_steps, s1a_steps)
                    model.set_snap_alpha(1.0)
                    for gi, blr in enumerate(base_lrs_1b):
                        opt_1b.param_groups[gi]["lr"] = _cosine_lr(o_step, s1b_steps, warm_1b, blr, 0.15 * blr)
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
                    torch.nn.utils.clip_grad_norm_(all_1b, 1.0)
                    opt_1b.step()
                    if o_step in (1, max(1, s1b_steps // 2), s1b_steps):
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

            # Switch to UTF-8 mode & calibrate non-ASCII router embeddings (exact Phase K boundary)
            model.byte_chunker.byte_chunk_mode = "utf8"
            model.byte_chunker.target_chunk_bytes = float(cfg.target_chunk_bytes)
            model.byte_chunker.use_static_m_cap = False
            model.byte_chunker.lock_router(True)
            cal_steps = 2 if smoke else 120
            _calibrate_non_ascii_router_emb(model, train_wins, device, cal_steps=cal_steps, seed=2026)
            fp_router_snap = _snapshot_router_params(model)
            model.set_crystal_schedule(s1a_steps, s1a_steps)
            model.set_snap_alpha(1.0)

            if not smoke:
                torch.save(
                    {
                        "state_dict": unwrap_compiled_state_dict(model),
                        "s1a_s5_k2": float(s1a_s5_k2),
                        "s1_s5_k2": float(s1_s5_k2),
                        "s1_ov_long": float(s1_ov_long),
                        "s1_trap_long": float(s1_trap_long),
                    },
                    s1b_ckpt,
                )
                print(f"  [Saved] Stage-1B FP checkpoint -> {s1b_ckpt}", flush=True)

        # ----------------------------------------------------------
        # STAGE 2: UTF-8 OpenWeb Language Scaling + Protected Crystal/Woodbury
        # ----------------------------------------------------------
        cryst_2, wood_2, nodc_other_2, emb_chunk_2, core_2 = _split_calcite_stage2_groups(model)
        all_fp2 = cryst_2 + wood_2 + nodc_other_2 + emb_chunk_2 + core_2
        base_lrs_2 = (6.0e-5, 8.0e-5, stage2_core_lr, stage2_emb_lr, stage2_core_lr)
        min_lrs_2 = (1.0e-5, 1.0e-5, min_lr, min_lr, min_lr)
        opt_2 = _make_adamw(
            [
                {"params": cryst_2, "lr": base_lrs_2[0], "weight_decay": 0.0},
                {"params": wood_2, "lr": base_lrs_2[1], "weight_decay": 0.0},
                {"params": nodc_other_2, "lr": base_lrs_2[2], "weight_decay": 0.0},
                {"params": emb_chunk_2, "lr": base_lrs_2[3], "weight_decay": 0.01},
                {"params": core_2, "lr": base_lrs_2[4], "weight_decay": 0.01},
            ],
            device,
            betas=(0.9, 0.95),
        )
        warm_2 = min(40, max(1, s2_steps // 8))
        t_2 = time.perf_counter()
        s2_step_it = iter(range(1, s2_steps + 1))

        def _sample_s2() -> Tuple[torch.Tensor, torch.Tensor]:
            st = grok_steps + next(s2_step_it)
            return batcher.sample_batch(
                batch_size, "cpu", step=st, grok_steps=grok_steps, total_steps=total_fp_steps, s1b_steps=s1b_steps
            )

        with BatchPrefetcher(
            _sample_s2, device=device, total_steps=s2_steps, prefetch_depth=4, enabled=(s2_steps >= 4)
        ) as pf_2:
            for s2 in range(1, s2_steps + 1):
                model.train()
                model.set_crystal_schedule(s1a_steps, s1a_steps)
                model.set_snap_alpha(1.0)
                for gi, (blr, mlr) in enumerate(zip(base_lrs_2, min_lrs_2)):
                    opt_2.param_groups[gi]["lr"] = _cosine_lr(s2, s2_steps, warm_2, blr, mlr)
                xb_full, yb_full = pf_2.next_batch()
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
                torch.nn.utils.clip_grad_norm_(all_fp2, 1.0)
                opt_2.step()
                if s2 in (1, s2_steps) or (s2 % 400 == 0):
                    ce_v = float(ce_sum_t.item()) / float(max(1, batch_size // micro_bs)) if ce_sum_t is not None else 0.0
                    print(
                        f"  [S2-utf8] step {s2:4d}/{s2_steps} | ce={ce_v:.4f} | {(time.perf_counter()-t_2)/60:.1f}m",
                        flush=True,
                    )
        del opt_2
        if device == "cuda":
            torch.cuda.empty_cache()

        fp_sd_clean = unwrap_compiled_state_dict(model)
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        model = CalciteV2LM(cfg).to(device)
        model.load_state_dict(fp_sd_clean, strict=True)
        model.byte_chunker.byte_chunk_mode = "utf8"
        model.byte_chunker.target_chunk_bytes = float(cfg.target_chunk_bytes)
        model.byte_chunker.use_static_m_cap = True
        model.byte_chunker.lock_router(True)
        model.set_crystal_schedule(grok_steps, grok_steps)
        model.set_snap_alpha(1.0)
        if not fp_router_snap:
            fp_router_snap = _snapshot_router_params(model)
        _assert_router_unchanged(model, fp_router_snap, "V2_FP_POSTLOCK")

        if not smoke:
            torch.save(
                {
                    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "s1a_s5_k2": float(s1a_s5_k2),
                    "s1_s5_k2": float(s1_s5_k2),
                    "s1_ov_long": float(s1_ov_long),
                    "s1_trap_long": float(s1_trap_long),
                },
                s2_ckpt,
            )
            print(f"  [Saved] Stage-2 FP checkpoint -> {s2_ckpt}", flush=True)

    # Evaluate FP model
    model.eval()
    model.set_crystal_schedule(1, 1)
    model.set_snap_alpha(1.0)
    print("  [Eval] Evaluating Stage-2 FP Champion...", flush=True)
    eval_fp = evaluate_full_model(
        model, val_mm, eval_wins, tok, s5_suites, ov_suites, trap_suites, device=device, quick=smoke, eval_bs=16
    )
    long_fp = inject_l2048_from_extrap(
        eval_long_context_stress(
            model, val_mm, device=device, lengths=(512, 2048) if smoke else (8192, 32768)
        ),
        None if smoke else eval_fp.get("extrap_4x"),
    )
    print(f"  [Eval] FP long-context: {long_fp}", flush=True)

    # ----------------------------------------------------------
    # STAGE 3: Exact 4-Group + 4-Tap + Logit-KL STE QAT -> 2-bit uint8 LUT
    # ----------------------------------------------------------
    for p in model.parameters():
        p.requires_grad_(False)
    fp_sd_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    qat_model = CalciteV2LM(cfg).to(device)
    qat_model.load_state_dict(fp_sd_cpu, strict=True)
    model.byte_chunker.byte_chunk_mode = "utf8"
    model.byte_chunker.target_chunk_bytes = float(cfg.target_chunk_bytes)
    model.byte_chunker.use_static_m_cap = True
    qat_model.byte_chunker.byte_chunk_mode = "utf8"
    qat_model.byte_chunker.target_chunk_bytes = float(cfg.target_chunk_bytes)
    qat_model.byte_chunker.use_static_m_cap = True
    qat_model.byte_chunker.lock_router(True)
    qat_model.set_crystal_schedule(1, 1)
    qat_model.set_snap_alpha(1.0)
    qat_model.set_qat_mix(0.30)
    qat_model.set_ternary_qat(True)
    if compile_swa and device == "cuda":
        compile_v2_swa_blocks(qat_model)

    gain_q, cryst_q, chunk_q, core_q = _split_v2_qat_4groups(qat_model)
    all_qp = gain_q + cryst_q + chunk_q + core_q
    w_scale = (384.0 / float(cfg.d_model)) ** 0.5
    q_lrs = (1.2e-3 * w_scale, 8.0e-5, 6.0e-4 * w_scale, 4.2e-4 * w_scale)
    q_mins = (1.2e-4 * w_scale, 1.5e-5, 7.0e-5 * w_scale, 5.0e-5 * w_scale)
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

    def _sample_qat_l() -> Tuple[torch.Tensor, torch.Tensor]:
        return batcher.sample_batch(
            batch_size, "cpu", step=total_fp_steps, grok_steps=grok_steps, total_steps=total_fp_steps, s1b_steps=s1b_steps
        )

    try:
        with BatchPrefetcher(
            _sample_qat_l, device=device, total_steps=qat_steps, prefetch_depth=4, enabled=(qat_steps >= 4)
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

    router_mut_diff = _assert_router_unchanged(qat_model, fp_router_snap, "V2_QAT")
    qat_s = round(time.perf_counter() - t_q0, 2)
    del model, opt_q
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # Unwrap, LS-quantize weights, pack into 2-bit uint8 T-MAC LUT, and save IMMEDIATELY before eval
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
    ckpt_out = ckpt_dir / ("smoke_v2_lut.pt" if smoke else "calcite_v2_s2_25m_utf8_lut.pt")
    save_packed_lut_checkpoint(
        ckpt_out,
        lut_cpu,
        cfg,
        tok,
        metadata={
            "phase": "Phase_L_Closure_4Tap_QAT",
            "params": n_params,
            "router_max_abs_diff": router_mut_diff,
            "s1a_s5_k2": float(s1a_s5_k2),
            "s1_s5_k2": float(s1_s5_k2),
            "s1_ov_long": float(s1_ov_long),
            "s1_trap_long": float(s1_trap_long),
        },
    )
    print(f"  [Saved] 2-bit LUT checkpoint -> {ckpt_out} ({ckpt_out.stat().st_size:,} B)", flush=True)

    stream_cpu = benchmark_stream_engine_live(lut_cpu, device="cpu", n_bytes=32 if smoke else 128)
    print(
        f"  [Stream CPU] {stream_cpu['bytes_per_sec']} B/s | err={stream_cpu['stream_vs_batch_max_err']} | "
        f"RAM={stream_cpu['state_kb']} KB | skip={stream_cpu['effective_chunk_ratio']}x",
        flush=True,
    )

    lut_gpu = pack_fast_model_2bit_lut(clean_qat, backend=lut_bk).to(device).eval()
    lut_gpu.set_crystal_schedule(1, 1)
    lut_gpu.set_snap_alpha(1.0)
    if device == "cuda":
        torch.cuda.empty_cache()
    print("  [Eval] Evaluating 2-bit LUT Champion (4-Tap QAT)...", flush=True)
    lut_cog_bs = _recommended_lut_cog_bs(device, binfo)
    eval_lut = evaluate_full_model(
        lut_gpu, val_mm, eval_wins, tok, s5_suites, ov_suites, trap_suites, device=device, quick=smoke, eval_bs=8, cog_bs=lut_cog_bs
    )
    if device == "cuda":
        torch.cuda.empty_cache()
    long_lut = inject_l2048_from_extrap(
        eval_long_context_stress(
            lut_gpu, val_mm, device=device, lengths=(512, 2048) if smoke else (8192, 32768)
        ),
        None if smoke else eval_lut.get("extrap_4x"),
    )
    if device == "cuda":
        torch.cuda.empty_cache()
    stream_gpu = benchmark_stream_engine_live(lut_gpu, device=device, n_bytes=32 if smoke else 256)
    print(
        f"  [Stream GPU] {stream_gpu['bytes_per_sec']} B/s | err={stream_gpu['stream_vs_batch_max_err']} | "
        f"RAM={stream_gpu['state_kb']} KB",
        flush=True,
    )

    gap_val = round(float(eval_lut["val_bpb"]) - float(eval_fp["val_bpb"]), 4)
    gap_ow = round(float(eval_lut["openweb_6"]["mean_openweb_bpb"]) - float(eval_fp["openweb_6"]["mean_openweb_bpb"]), 4)
    result = {
        "phase": "Phase_L_Closure_25M",
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "params": n_params,
        "config": asdict(cfg),
        "schedule": {
            "s1a_steps": s1a_steps,
            "s1b_steps": s1b_steps,
            "s2_steps": s2_steps,
            "qat_steps": qat_steps,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "total_seconds": round(time.perf_counter() - t_all0, 2),
            "qat_seconds": qat_s,
        },
        "stage1_gates": {
            "s1a_s5_k2_pct": round(float(s1a_s5_k2), 2),
            "s1_s5_k2_pct": round(float(s1_s5_k2), 2),
            "s1_ov_long_pct": round(float(s1_ov_long), 2),
            "s1_trap_long_pct": round(float(s1_trap_long), 2),
        },
        "router_max_abs_diff": router_mut_diff,
        "cache_kb_L2048": round(compute_v2_state_cache_kb(cfg, 2048), 2),
        "cache_kb_L32768": round(compute_v2_state_cache_kb(cfg, 32768), 2),
        "lut_footprint": lut_foot,
        "gap_val_bpb": gap_val,
        "gap_ow_bpb": gap_ow,
        "eval_fp": eval_fp,
        "eval_lut": eval_lut,
        "long_context_fp": long_fp,
        "long_context_lut": long_lut,
        "stream_cpu": stream_cpu,
        "stream_gpu": stream_gpu,
        "checkpoint_path": str(ckpt_out),
    }

    if smoke:
        ckpt_out.unlink(missing_ok=True)
        print(
            f"[SMOKE PASS] FP_val={eval_fp['val_bpb']} | LUT_val={eval_lut['val_bpb']} | "
            f"gap={gap_val:+.4f} | stream_cpu={stream_cpu['bytes_per_sec']} B/s | err={stream_cpu['stream_vs_batch_max_err']}",
            flush=True,
        )
    else:
        res_path = out_dir / "phase_l_closure_25m_results.json"
        res_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(
            f"\n=== [PHASE L CLOSURE DONE] Saved -> {res_path} ===\n"
            f"  FP  val={eval_fp['val_bpb']:.4f} | L2048={eval_fp['extrap_4x']['bpb_L2048']:.4f} | "
            f"S5={eval_fp['cognitive']['s5_sum']:.1f}% | OV_L={eval_fp['cognitive']['overwrite']['ood_long']['exact_pct']:.1f}% | "
            f"SelF1={eval_fp['cognitive']['selective_f1_ood_long']:.1f}%\n"
            f"  LUT val={eval_lut['val_bpb']:.4f} (gap={gap_val:+.4f}) | L2048={eval_lut['extrap_4x']['bpb_L2048']:.4f} | "
            f"S5={eval_lut['cognitive']['s5_sum']:.1f}% | OV_L={eval_lut['cognitive']['overwrite']['ood_long']['exact_pct']:.1f}% | "
            f"SelF1={eval_lut['cognitive']['selective_f1_ood_long']:.1f}%",
            flush=True,
        )
    return result


def load_calcite_v2_lut_checkpoint(
    path: Path,
    device: str = "cpu",
    backend: str = "auto",
) -> Tuple[CalciteV2LM, CalciteV2Config, ByteBPETokenizer, Dict[str, object]]:
    """Loads a saved `CalciteV2LM` 2-bit uint8 LUT checkpoint with `strict=True`."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = CalciteV2Config(**payload["simvq_split_config"])
    tok = ByteBPETokenizer.from_dict(payload["tokenizer"])
    skel = CalciteV2LM(cfg).eval()
    skel.byte_chunker.byte_chunk_mode = "utf8"
    skel.byte_chunker.target_chunk_bytes = float(cfg.target_chunk_bytes)
    skel.byte_chunker.use_static_m_cap = True
    skel.byte_chunker.lock_router(True)
    skel.set_crystal_schedule(1, 1)
    skel.set_snap_alpha(1.0)
    lut_m = pack_fast_model_2bit_lut(skel, backend=backend)
    lut_m.load_state_dict(payload["lut_state_dict"], strict=True)
    lut_m = lut_m.to(device).eval()
    lut_m.set_crystal_schedule(1, 1)
    lut_m.set_snap_alpha(1.0)
    return lut_m, cfg, tok, payload


def eval_saved_closure_25m(
    data_dir: Path,
    out_dir: Path,
    ckpt_dir: Path,
    seq_len: int = 512,
    s1a_steps: int = 2400,
    s1b_steps: int = 800,
    s2_steps: int = 1600,
    qat_steps: int = 500,
    batch_size: int = 64,
) -> Dict[str, object]:
    """Evaluates saved Stage-2 FP and 2-bit LUT checkpoints cleanly with low-VRAM batching."""
    device = "cuda" if has_usable_cuda() else "cpu"
    binfo_init = get_lut_backend_info()
    if device == "cuda":
        # Keep higher VRAM headroom when CUDA LUT kernel is active; cap only on pytorch_vec fallback
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram_gb < 12.0:
            torch.cuda.set_per_process_memory_fraction(0.90 if binfo_init.get("cuda_ext_available") else 0.72, 0)
    t0 = time.perf_counter()
    s2_ckpt = out_dir / "closure_25m_stage2_fp.pt"
    lut_ckpt = ckpt_dir / "calcite_v2_s2_25m_utf8_lut.pt"
    assert s2_ckpt.is_file(), f"Missing Stage-2 FP checkpoint: {s2_ckpt}"
    assert lut_ckpt.is_file(), f"Missing 2-bit LUT checkpoint: {lut_ckpt}"

    _, val_bin = ensure_phase_k_corpus(data_dir)
    val_mm = load_utf8_corpus_array(val_bin)

    lut_cpu, cfg, tok, lut_payload = load_calcite_v2_lut_checkpoint(lut_ckpt, device="cpu", backend="auto")
    meta = lut_payload.get("metadata", {})
    tr_docs, te_docs, _, _ = load_gsm8k_corpora(ROOT_DIR)
    _, eval_wins, _, _ = build_disjoint_openweb_corpora(ROOT_DIR, tok, tr_docs, te_docs, seq_len=seq_len)
    _, _, s5_suites, ov_suites, trap_suites = build_eval_bundles(ROOT_DIR, tok, n_eval=150, seq_len=seq_len)

    # 1. Stage-2 FP evaluation
    fp_blob = torch.load(s2_ckpt, map_location="cpu", weights_only=False)
    model_fp = CalciteV2LM(cfg).to(device)
    model_fp.load_state_dict(fp_blob["state_dict"], strict=True)
    model_fp.byte_chunker.byte_chunk_mode = "utf8"
    model_fp.byte_chunker.target_chunk_bytes = float(cfg.target_chunk_bytes)
    model_fp.byte_chunker.use_static_m_cap = True
    model_fp.byte_chunker.lock_router(True)
    model_fp.set_crystal_schedule(1, 1)
    model_fp.set_snap_alpha(1.0)
    model_fp.eval()
    n_params = int(sum(p.numel() for p in model_fp.parameters()))
    fp_router_snap = _snapshot_router_params(model_fp)

    print(f"\n=== [Phase L Closure Eval-Only | params={n_params:,}] ===", flush=True)
    print("  [Eval] Evaluating Stage-2 FP Champion (paired eval_bs=8)...", flush=True)
    eval_fp = evaluate_full_model(
        model_fp, val_mm, eval_wins, tok, s5_suites, ov_suites, trap_suites, device=device, quick=False, eval_bs=8
    )
    long_fp = inject_l2048_from_extrap(
        eval_long_context_stress(model_fp, val_mm, device=device, lengths=(8192, 32768)),
        eval_fp.get("extrap_4x"),
    )
    print(f"  [Eval] FP long-context: {long_fp}", flush=True)
    del model_fp
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # 2. Disk-roundtrip 2-bit LUT verification + CPU streaming
    router_mut_diff = _assert_router_unchanged(lut_cpu, fp_router_snap, "V2_LUT_DISK_ROUNDTRIP")
    lut_foot = compute_2bit_lut_footprint(lut_cpu)
    stream_cpu = benchmark_stream_engine_live(lut_cpu, device="cpu", n_bytes=128)
    print(
        f"  [Stream CPU] {stream_cpu['bytes_per_sec']} B/s | err={stream_cpu['stream_vs_batch_max_err']} | "
        f"RAM={stream_cpu['state_kb']} KB | skip={stream_cpu['effective_chunk_ratio']}x | "
        f"router_diff={router_mut_diff:.1e}",
        flush=True,
    )

    # 3. Low-VRAM GPU evaluation of exact saved 2-bit LUT checkpoint
    binfo = get_lut_backend_info()
    lut_bk = "cuda_ext" if (device == "cuda" and binfo["cuda_ext_available"]) else "auto"
    lut_cog_bs = _recommended_lut_cog_bs(device, binfo)
    lut_gpu, _, _, _ = load_calcite_v2_lut_checkpoint(lut_ckpt, device=device, backend=lut_bk)
    print(f"  [Eval] Evaluating saved 2-bit LUT Champion (paired eval_bs=8, cog_bs={lut_cog_bs})...", flush=True)
    eval_lut = evaluate_full_model(
        lut_gpu,
        val_mm,
        eval_wins,
        tok,
        s5_suites,
        ov_suites,
        trap_suites,
        device=device,
        quick=False,
        eval_bs=8,
        cog_bs=lut_cog_bs,
    )
    if device == "cuda":
        torch.cuda.empty_cache()
    long_lut = inject_l2048_from_extrap(
        eval_long_context_stress(lut_gpu, val_mm, device=device, lengths=(8192, 32768)),
        eval_lut.get("extrap_4x"),
    )
    print(f"  [Eval] LUT long-context: {long_lut}", flush=True)
    if device == "cuda":
        torch.cuda.empty_cache()
    stream_gpu = benchmark_stream_engine_live(lut_gpu, device=device, n_bytes=256)
    print(
        f"  [Stream GPU] {stream_gpu['bytes_per_sec']} B/s | err={stream_gpu['stream_vs_batch_max_err']} | "
        f"RAM={stream_gpu['state_kb']} KB",
        flush=True,
    )

    gap_val = round(float(eval_lut["val_bpb"]) - float(eval_fp["val_bpb"]), 4)
    gap_ow = round(float(eval_lut["openweb_6"]["mean_openweb_bpb"]) - float(eval_fp["openweb_6"]["mean_openweb_bpb"]), 4)
    result = {
        "phase": "Phase_L_Closure_25M",
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "params": n_params,
        "config": asdict(cfg),
        "schedule": {
            "s1a_steps": s1a_steps,
            "s1b_steps": s1b_steps,
            "s2_steps": s2_steps,
            "qat_steps": qat_steps,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "eval_only_seconds": round(time.perf_counter() - t0, 2),
        },
        "stage1_gates": {
            "s1a_s5_k2_pct": round(float(fp_blob.get("s1a_s5_k2", meta.get("s1a_s5_k2", 0.0))), 2),
            "s1_s5_k2_pct": round(float(fp_blob.get("s1_s5_k2", meta.get("s1_s5_k2", 0.0))), 2),
            "s1_ov_long_pct": round(float(fp_blob.get("s1_ov_long", meta.get("s1_ov_long", 0.0))), 2),
            "s1_trap_long_pct": round(float(fp_blob.get("s1_trap_long", meta.get("s1_trap_long", 0.0))), 2),
        },
        "router_max_abs_diff": router_mut_diff,
        "cache_kb_L2048": round(compute_v2_state_cache_kb(cfg, 2048), 2),
        "cache_kb_L32768": round(compute_v2_state_cache_kb(cfg, 32768), 2),
        "lut_footprint": lut_foot,
        "gap_val_bpb": gap_val,
        "gap_ow_bpb": gap_ow,
        "eval_fp": eval_fp,
        "eval_lut": eval_lut,
        "long_context_fp": long_fp,
        "long_context_lut": long_lut,
        "stream_cpu": stream_cpu,
        "stream_gpu": stream_gpu,
        "checkpoint_path": str(lut_ckpt),
        "checkpoint_bytes": int(lut_ckpt.stat().st_size),
    }
    res_path = out_dir / "phase_l_closure_25m_results.json"
    res_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"\n=== [PHASE L CLOSURE EVAL DONE] Saved -> {res_path} ===\n"
        f"  FP  val={eval_fp['val_bpb']:.4f} | OW6={eval_fp['openweb_6']['mean_openweb_bpb']:.4f} | "
        f"L2048={eval_fp['extrap_4x']['bpb_L2048']:.4f} | L32K={long_fp['L_32768']['bpb']:.4f} | "
        f"S5={eval_fp['cognitive']['s5_sum']:.1f}% | OV_L={eval_fp['cognitive']['overwrite']['ood_long']['exact_pct']:.1f}% | "
        f"SelF1={eval_fp['cognitive']['selective_f1_ood_long']:.1f}%\n"
        f"  LUT val={eval_lut['val_bpb']:.4f} (gap={gap_val:+.4f}) | OW6={eval_lut['openweb_6']['mean_openweb_bpb']:.4f} (gap={gap_ow:+.4f}) | "
        f"L2048={eval_lut['extrap_4x']['bpb_L2048']:.4f} | L32K={long_lut['L_32768']['bpb']:.4f} | "
        f"S5={eval_lut['cognitive']['s5_sum']:.1f}% | OV_L={eval_lut['cognitive']['overwrite']['ood_long']['exact_pct']:.1f}% | "
        f"SelF1={eval_lut['cognitive']['selective_f1_ood_long']:.1f}%",
        flush=True,
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data_cache")
    ap.add_argument("--out-dir", type=str, default="benchmarks/phase_l_v2")
    ap.add_argument("--ckpt-dir", type=str, default="checkpoints")
    ap.add_argument("--s1a-steps", type=int, default=2400)
    ap.add_argument("--s1b-steps", type=int, default=800)
    ap.add_argument("--s2-steps", type=int, default=1600)
    ap.add_argument("--qat-steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--smoke-local", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    if args.eval_only:
        eval_saved_closure_25m(
            data_dir=Path(args.data_dir),
            out_dir=Path(args.out_dir),
            ckpt_dir=Path(args.ckpt_dir),
            seq_len=args.seq_len,
            s1a_steps=args.s1a_steps,
            s1b_steps=args.s1b_steps,
            s2_steps=args.s2_steps,
            qat_steps=args.qat_steps,
            batch_size=args.batch_size,
        )
        return

    run_closure_25m(
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir),
        ckpt_dir=Path(args.ckpt_dir),
        s1a_steps=args.s1a_steps,
        s1b_steps=args.s1b_steps,
        s2_steps=args.s2_steps,
        qat_steps=args.qat_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        smoke=args.smoke_local,
        compile_swa=not args.no_compile,
    )


if __name__ == "__main__":
    main()
