"""
Phase N Stage 4: Surgical Zero-Regression `Calcite v3 -> v3.1 (S2_25M)` Retrofit
(`scripts/run_phase_n_v31_retrofit_25m.py`).

Resolves BOTH defects diagnosed in the from-scratch `v3.1` run in ~28 minutes:
1. **Zero `S_5` Shortcut Theft:** Warm-starts from the verified `Calcite v3 (25M)` FP champion
   (`benchmarks/phase_m_v3/closure_v3_25m_stage2_fp.pt`, which ALREADY grokked `S_5 = 137.3%`,
   `Val = 2.0395`, `L32K = 1.6769`, `OV_L = 89.3%`, `Trap_L = 93.3%`, `SelF1_L = 91.3%`)
   and attaches the upgraded dual-timescale `ByteInductionCache` (`2 x 1024` slots) post-`S1A`
   while protecting `Crystal` (`0.03x`) & `Woodbury` (`0.05x`).
2. **Ultra-Sparse Scale-1 32K Needle Vault:** With `tau_1 = 0.45` and `0.80 * relu(mean - 0.025)`
   sparsity hinge ONLY on Scale 1 (`k=7`), Scale 0 (`k=4`) stays open for the massive `Val`/`OW6`
   prose/code boost while Scale 1 stays ultra-sparse (`<= 2%`) so needles survive `32,768 B` (`64x` OOD)!
3. **Full `4-Tap 2-bit LUT` Closure + Strict Disk Roundtrip (`checkpoints/calcite_v31_s2_25m_utf8_lut.pt`).**
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
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from calcite_v2 import (  # noqa: E402
    CalciteV2LM,
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
from scripts.run_phase_m_stage_c_v3_5m import _split_v3_qat_4groups  # noqa: E402
from scripts.run_phase_n_closure_25m import (  # noqa: E402
    _inject_mqar_rows,
    build_calcite_v31_25m_config,
)
from scripts.run_phase_n_mqar_probe_5m import eval_ood_verbatim_ladder  # noqa: E402
from scripts.run_phase_n_v31_lm_check_5m import (  # noqa: E402
    eval_live_stream_needle,
    eval_overwrite_needle_ood,
)
from simvq_split.memory import (  # noqa: E402
    compute_2bit_lut_footprint,
    quantize_ternary_weights_ls,
)
from simvq_split.tasks import load_gsm8k_corpora  # noqa: E402
from simvq_split.tokenizer import ByteBPETokenizer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-fp-ckpt",
        type=Path,
        default=ROOT_DIR / "benchmarks" / "phase_m_v3" / "closure_v3_25m_stage2_fp.pt",
    )
    ap.add_argument("--data-dir", type=Path, default=ROOT_DIR / "data_cache")
    ap.add_argument("--out-dir", type=Path, default=ROOT_DIR / "benchmarks" / "phase_n_needle")
    ap.add_argument("--ckpt-dir", type=Path, default=ROOT_DIR / "checkpoints")
    ap.add_argument("--retrofit-steps", type=int, default=650)
    ap.add_argument("--qat-steps", type=int, default=600)
    ap.add_argument("--slots", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--qat-only", action="store_true")
    ap.add_argument("--smoke-local", action="store_true")
    args = ap.parse_args()

    device = "cuda" if has_usable_cuda() else "cpu"
    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram_gb < 12.0:
            torch.cuda.set_per_process_memory_fraction(0.72, 0)
    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    tok = ByteBPETokenizer(merges=[])
    tr_docs, te_docs, tr_sents, _ = load_gsm8k_corpora(ROOT_DIR)
    seq_len = 128 if args.smoke_local else 512
    train_wins, eval_wins, _, _ = build_disjoint_openweb_corpora(
        ROOT_DIR, tok, tr_docs, te_docs, seq_len=seq_len
    )
    n_ev = 6 if args.smoke_local else 150
    _, _, s5_suites, ov_suites, trap_suites = build_eval_bundles(
        ROOT_DIR, tok, n_eval=n_ev, seq_len=seq_len
    )

    cfg = build_calcite_v31_25m_config(smoke=args.smoke_local, slots=args.slots)
    if args.smoke_local:
        rng_f = np.random.default_rng(args.seed)
        fake = rng_f.integers(32 + 3, 126 + 3, size=100_000, dtype=np.uint16)
        train_mm = val_mm = fake
        ret_steps, qat_steps, batch_size = 4, 4, 8
    else:
        train_bin, val_bin = ensure_phase_k_corpus(args.data_dir)
        train_mm = np.memmap(train_bin, dtype=np.uint16, mode="r")
        val_mm = np.memmap(val_bin, dtype=np.uint16, mode="r")
        ret_steps, qat_steps, batch_size = int(args.retrofit_steps), int(args.qat_steps), int(args.batch_size)

    set_all_seeds(args.seed)
    model = CalciteV2LM(cfg).to(device)
    fp_ckpt = args.out_dir / "closure_v31_retrofit_25m_fp.pt"
    res_path = args.out_dir / ("smoke_v31_retrofit_25m.json" if args.smoke_local else "phase_n_v31_retrofit_25m_results.json")

    if (not args.smoke_local) and args.qat_only:
        assert fp_ckpt.is_file(), f"Missing saved Retrofit FP checkpoint for --qat-only: {fp_ckpt}"
        blob = torch.load(fp_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["state_dict"], strict=True)
        print(f"  [Resume --qat-only] Loaded saved Retrofit FP champion <- {fp_ckpt}", flush=True)
    elif not args.smoke_local:
        assert args.base_fp_ckpt.is_file(), f"Missing Calcite v3 FP base checkpoint: {args.base_fp_ckpt}"
        blob = torch.load(args.base_fp_ckpt, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(blob["state_dict"], strict=False)
        assert not unexpected, f"Unexpected keys loading v3 base: {unexpected}"
        assert all("byte_induct_cache" in k for k in missing), f"Unexpected missing keys: {missing}"
        print(f"  [Retrofit] Loaded Calcite v3 (25M) base <- {args.base_fp_ckpt} (attached {len(missing)} ByteInductionCache tensors)", flush=True)
    else:
        router_state = torch.load(
            ROOT_DIR / "benchmarks" / "phase_l_v2" / "frozen_utf8_router.pt", map_location="cpu", weights_only=True
        )
        load_frozen_router_into(model, router_state)

    model.byte_chunker.lock_router(True)
    model.set_crystal_schedule(1, 1)
    model.set_snap_alpha(1.0)
    fp_router_snap = _snapshot_router_params(model)
    n_params = int(sum(p.numel() for p in model.parameters()))
    cache_kb = round(compute_v2_state_cache_kb(cfg, 32768), 2)
    micro_bs = min(batch_size, _default_micro_bs(device, is_qat=False))

    print(
        f"\n=== [PHASE N RETROFIT: Calcite v3 -> v3.1 S2_25M] params={n_params:,} | "
        f"O(1)_cache={cache_kb:.2f} KB | slots=2x{cfg.byte_induct_slots} | "
        f"retrofit={ret_steps} + qat={qat_steps} ===",
        flush=True,
    )

    # ----------------------------------------------------------
    # STAGE 2B RETROFIT: Protected Crystal (0.03x) + Woodbury (0.05x) + Active ByteInductionCache
    # ----------------------------------------------------------
    batcher = StratifiedSystem1Batcher(
        train_mm=train_mm,
        tr_docs=tr_docs,
        tr_sents=tr_sents,
        tok=tok,
        seq_len=seq_len,
        train_wins=train_wins,
        seed=args.seed + 99,
    )
    mqar_rng_py = random.Random(args.seed + 8888)
    mqar_rng_np = np.random.default_rng(args.seed + 8888)

    t_all = time.perf_counter()
    ret_s = 0.0
    ndl_hs = (256,) if args.smoke_local else (512, 1500, 4096, 8192, 32768)

    if (not args.smoke_local) and args.qat_only and res_path.is_file():
        prev_res = json.loads(res_path.read_text(encoding="utf-8"))
        eval_fp = prev_res["eval_fp"]
        long_fp = prev_res["long_fp"]
        needle_fp = prev_res["needle_fp"]
        ov_needle_fp = prev_res["overwrite_needle_fp"]
        ret_s = float(prev_res.get("meta", {}).get("retrofit_seconds", 0.0))
        print(
            f"  [Resume --qat-only] Reused verified FP metrics (Val={eval_fp['val_bpb']:.4f}, "
            f"S5={eval_fp['cognitive']['s5_sum']:.1f}%, N1500={needle_fp['L_1500']['exact_match_pct']}%)",
            flush=True,
        )
    else:
        base_lr = 6.5e-4
        opt = MuonAdamWHybrid(
            model,
            base_lr=base_lr,
            muon_lr_mult=3.5,
            weight_decay=0.02,
            device=device,
            woodbury_lr_mult=0.05,
            protect_crystal=True,
            crystal_lr_mult=0.03,
        )
        # Give new `byte_induct_cache` parameters a dedicated fast 3.0x AdamW LR while base stays gentle!
        bic_ids = {id(p) for p in model.byte_chunker.byte_induct_cache.parameters()} if model.byte_chunker.byte_induct_cache else set()
        bic_params = []
        if opt.adamw is not None:
            for pg in opt.adamw.param_groups:
                kept = []
                for p in pg["params"]:
                    if id(p) in bic_ids:
                        bic_params.append(p)
                    else:
                        kept.append(p)
                pg["params"] = kept
            if bic_params:
                opt.adamw.add_param_group(
                    {"params": bic_params, "lr": base_lr * 3.0, "weight_decay": 0.0, "lr_mult": 3.0, "is_muon": False}
                )
                opt.param_groups = list(opt.adamw.param_groups) + ([opt.muon_group] if opt.muon_group else [])

        warm = min(30, max(1, ret_steps // 12))
        n_copy_rows = max(1, int(round(batch_size * 0.125)))
        t_ret = time.perf_counter()

        for r_step in range(1, ret_steps + 1):
            model.train()
            model.set_crystal_schedule(1, 1)
            model.set_snap_alpha(1.0)
            lr_s = _cosine_lr(r_step, ret_steps, warm, base_lr, 6.0e-5)
            sc = lr_s / base_lr
            for pg in opt.param_groups:
                pg["lr"] = base_lr * float(pg.get("lr_mult", 1.0)) * sc

            # Stage-2 regime (68% OpenWeb+MQAR, 32% S_5/OV/Trap maintenance!)
            xb_full, yb_full = batcher.sample_batch(
                batch_size, device, step=4000 + r_step, grok_steps=2000, total_steps=5000, s1b_steps=500
            )
            is_web_step = float((yb_full != tok.PAD_ID).float().mean().item()) > 0.60
            if is_web_step:
                xb_full, yb_full, mb_full = _inject_mqar_rows(
                    xb_full, yb_full, train_mm, n_copy_rows, mqar_rng_py, mqar_rng_np
                )
            else:
                mb_full = None

            while True:
                opt.zero_grad(set_to_none=True)
                n_acc = max(1, (batch_size + micro_bs - 1) // micro_bs)
                ce_sum_v, c_acc_sum, n_c_m = 0.0, 0.0, 0
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
                                m_copy = mb & valid
                                m_pref = (~mb) & valid
                                ce_p = tok_ce[m_pref].mean() if m_pref.any() else tok_ce[valid].mean()
                                if m_copy.any():
                                    ce_c = tok_ce[m_copy].mean()
                                    ce = 0.82 * ce_p + 0.18 * ce_c
                                    with torch.no_grad():
                                        c_acc_sum += float((lg_f.argmax(dim=-1)[m_copy] == yb[m_copy]).float().mean().item())
                                    n_c_m += 1
                                else:
                                    ce = ce_p
                            else:
                                ce = F.cross_entropy(
                                    lg_f.reshape(-1, lg_f.size(-1)),
                                    yb.reshape(-1),
                                    ignore_index=tok.PAD_ID,
                                )
                            aux = model.get_commitment_loss()
                            loss = (ce + 0.04 * aux) / float(n_acc)
                        loss.backward()
                        ce_sum_v += float(ce.detach().item())
                    break
                except torch.cuda.OutOfMemoryError:
                    if micro_bs <= 2:
                        raise
                    micro_bs = max(2, micro_bs // 2)
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if r_step in (1, ret_steps) or (r_step % 100 == 0):
                bic = model.byte_chunker.byte_induct_cache
                b_info = f" | bwr={bic.last_write_rate:.2f}(s1={bic.last_write_rate_s1:.3f})" if bic is not None else ""
                c_info = f" | cAcc={100.0*c_acc_sum/n_c_m:5.1f}%" if n_c_m > 0 else ""
                print(
                    f"  [Retrofit] step {r_step:4d}/{ret_steps} | ce={ce_sum_v/n_acc:.4f}{c_info}{b_info} | "
                    f"{(time.perf_counter()-t_ret)/60:.1f}m",
                    flush=True,
                )

        ret_s = round(time.perf_counter() - t_ret, 2)
        del opt
        if device == "cuda":
            torch.cuda.empty_cache()

        _assert_router_unchanged(model, fp_router_snap, "V31_RETROFIT_FP")
        if not args.smoke_local:
            torch.save(
                {"config": asdict(cfg), "state_dict": {k: v.cpu() for k, v in unwrap_compiled_state_dict(model).items()}},
                fp_ckpt,
            )
            print(f"  [Saved] Retrofit FP checkpoint -> {fp_ckpt}", flush=True)

        # Evaluate FP Retrofit Champion
        model.eval()
        model.set_crystal_schedule(1, 1)
        model.set_snap_alpha(1.0)
        print("  [Eval] Evaluating Calcite v3.1 (25M Retrofit) FP Champion...", flush=True)
        eval_fp = evaluate_full_model(
            model, val_mm, eval_wins, tok, s5_suites, ov_suites, trap_suites, device=device, quick=args.smoke_local, eval_bs=8
        )
        long_fp = eval_long_context_stress(
            model, val_mm, device=device, lengths=(256, 512) if args.smoke_local else (2048, 8192, 32768)
        )
        needle_fp = eval_ood_verbatim_ladder(
            model, val_mm, device=device, haystacks=ndl_hs, n_samples=4 if args.smoke_local else 16, val_len=4 if args.smoke_local else 8, seed=9901
        )
        ov_needle_fp = eval_overwrite_needle_ood(
            model, val_mm, device=device, haystack_bytes=256 if args.smoke_local else 1500, n_samples=4 if args.smoke_local else 16
        )
        print(
            f"  [FP DONE] Val={eval_fp['val_bpb']:.4f} | OW6={eval_fp['openweb_6']['mean_openweb_bpb']:.4f} | "
            f"L2048={eval_fp['extrap_4x']['bpb_L2048']:.4f} | L32K={long_fp.get('L_32768', {}).get('bpb')} | "
            f"S5={eval_fp['cognitive']['s5_sum']:.1f}% | OV_L={eval_fp['cognitive']['overwrite']['ood_long']['exact_pct']:.1f}% | "
            f"Trap_L={eval_fp['cognitive']['trap']['ood_long']['exact_pct']:.1f}% | "
            f"SelF1_L={eval_fp['cognitive']['selective_f1_ood_long']:.1f}% | "
            f"Orig1500={needle_fp['OrigRefCode_1500B']['exact_match_pct']:.1f}% | "
            f"N1500={needle_fp.get('L_1500', {}).get('exact_match_pct')} | "
            f"N8K={needle_fp.get('L_8192', {}).get('exact_match_pct')} | "
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
    model.byte_chunker.use_static_m_cap = False
    qat_model.byte_chunker.use_static_m_cap = False
    qat_model.byte_chunker.lock_router(True)
    qat_model.set_crystal_schedule(1, 1)
    qat_model.set_snap_alpha(1.0)
    qat_model.set_qat_mix(0.30)
    qat_model.set_ternary_qat(True)

    gain_q, cryst_q, chunk_q, core_q = _split_v3_qat_4groups(qat_model)
    # Keep `byte_induct_cache` on a dedicated gentle LR (2.5e-4*w_scale) so `write_proj` never loses s1<=0.01 sparsity
    bic_q_ids = {
        id(p) for p in qat_model.byte_chunker.byte_induct_cache.parameters()
    } if qat_model.byte_chunker.byte_induct_cache else set()
    bic_q = [p for p in (gain_q + chunk_q) if id(p) in bic_q_ids]
    gain_q = [p for p in gain_q if id(p) not in bic_q_ids]
    chunk_q = [p for p in chunk_q if id(p) not in bic_q_ids]
    all_qp = gain_q + cryst_q + chunk_q + core_q + bic_q
    w_scale = (384.0 / float(cfg.d_model)) ** 0.5
    q_lrs = (1.2e-3 * w_scale, 8.0e-5, 1.1e-3 * w_scale, 4.5e-4 * w_scale, 2.5e-4 * w_scale)
    q_mins = (1.0e-4 * w_scale, 1.5e-5, 8.0e-5 * w_scale, 4.5e-5 * w_scale, 3.0e-5 * w_scale)
    opt_q = _make_adamw(
        [
            {"params": gain_q, "lr": q_lrs[0]},
            {"params": cryst_q, "lr": q_lrs[1]},
            {"params": chunk_q, "lr": q_lrs[2]},
            {"params": core_q, "lr": q_lrs[3]},
            {"params": bic_q, "lr": q_lrs[4]},
        ],
        device,
        weight_decay=0.0,
    )
    opt_q.register_step_post_hook(lambda *_a, **_kw: clear_fast_qat_cache())
    warm_q = min(40, max(1, qat_steps // 12))
    ramp_q = min(200, max(1, qat_steps // 3))
    kl_temp = 1.5
    taps_fp: List[torch.Tensor] = []
    taps_q: List[torch.Tensor] = []
    hk_fp = _attach_4tap_hooks(model, taps_fp, detach=True)
    hk_q = _attach_4tap_hooks(qat_model, taps_q, detach=False)
    micro_bs_q = min(batch_size, _default_micro_bs(device, is_qat=True))
    n_copy_q = max(1, int(round(batch_size * 0.125)))
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
                batch_size, device, step=4500 + qs, grok_steps=2000, total_steps=5000, s1b_steps=500
            )
            if float((yb_full != tok.PAD_ID).float().mean().item()) > 0.60:
                xb_full, yb_full, mb_q = _inject_mqar_rows(
                    xb_full, yb_full, train_mm, n_copy_q, mqar_rng_py, mqar_rng_np
                )
                # Causal QAT fix: mask prefix tokens on injected MQAR rows so per-row CE+KL gives
                # full 12.5% distillation weight directly to exact verbatim copy tokens!
                is_copy_row = mb_q.any(dim=1, keepdim=True)
                yb_full = torch.where(is_copy_row & (~mb_q), torch.full_like(yb_full, tok.PAD_ID), yb_full)
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

    router_mut_diff = _assert_router_unchanged(qat_model, fp_router_snap, "V31_RETROFIT_QAT")
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

    lut_ckpt = args.ckpt_dir / ("smoke_v31_ret_lut.pt" if args.smoke_local else "calcite_v31_s2_25m_utf8_lut.pt")
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
            "phase": "Phase_N_Retrofit_V31_25M",
            "params": n_params,
            "router_max_abs_diff": router_mut_diff,
            "base_fp_ckpt": str(args.base_fp_ckpt.name),
        },
    )
    ckpt_bytes = int(lut_ckpt.stat().st_size)
    print(f"  [Saved] 2-bit LUT checkpoint -> {lut_ckpt} ({ckpt_bytes:,} B)", flush=True)
    del clean_qat, lut_cpu
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    lut_cpu_rt, _, _, _ = load_calcite_v2_lut_checkpoint(lut_ckpt, device="cpu", backend="auto")
    stream_cpu = benchmark_stream_engine_live(lut_cpu_rt, device="cpu", n_bytes=32 if args.smoke_local else 256)
    print(
        f"  [Stream CPU RT] {stream_cpu['bytes_per_sec']} B/s | err={stream_cpu['stream_vs_batch_max_err']} | "
        f"RAM={stream_cpu['state_kb']} KB | O1={stream_cpu['strict_o1_ram_invariant']}",
        flush=True,
    )
    del lut_cpu_rt
    gc.collect()

    lut_gpu_rt, _, _, _ = load_calcite_v2_lut_checkpoint(lut_ckpt, device=device, backend=lut_bk)
    print("  [Eval] Evaluating 2-bit LUT Calcite v3.1 (25M Retrofit) from disk roundtrip...", flush=True)
    eval_lut = evaluate_full_model(
        lut_gpu_rt, val_mm, eval_wins, tok, s5_suites, ov_suites, trap_suites, device=device, quick=args.smoke_local, eval_bs=8, cog_bs=2
    )
    if device == "cuda":
        torch.cuda.empty_cache()
    long_lut = eval_long_context_stress(
        lut_gpu_rt, val_mm, device=device, lengths=(256, 512) if args.smoke_local else (2048, 8192, 32768)
    )
    if device == "cuda":
        torch.cuda.empty_cache()
    needle_lut = eval_ood_verbatim_ladder(
        lut_gpu_rt, val_mm, device=device, haystacks=ndl_hs, n_samples=4 if args.smoke_local else 16, val_len=4 if args.smoke_local else 8, seed=9901
    )
    ov_needle_lut = eval_overwrite_needle_ood(
        lut_gpu_rt, val_mm, device=device, haystack_bytes=256 if args.smoke_local else 1500, n_samples=4 if args.smoke_local else 16
    )
    stream_gpu = benchmark_stream_engine_live(lut_gpu_rt, device=device, n_bytes=32 if args.smoke_local else 256)
    stream_needle_lut = eval_live_stream_needle(
        lut_gpu_rt, val_mm, device=device, haystack_bytes=256 if args.smoke_local else 1500, n_samples=2 if args.smoke_local else 4
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
        f"L2048={eval_lut['extrap_4x']['bpb_L2048']:.4f} | L32K={long_lut.get('L_32768', {}).get('bpb')} | "
        f"S5={eval_lut['cognitive']['s5_sum']:.1f}% | OV_L={eval_lut['cognitive']['overwrite']['ood_long']['exact_pct']:.1f}% | "
        f"Trap_L={eval_lut['cognitive']['trap']['ood_long']['exact_pct']:.1f}% | "
        f"SelF1_L={eval_lut['cognitive']['selective_f1_ood_long']:.1f}% | "
        f"Orig1500={needle_lut['OrigRefCode_1500B']['exact_match_pct']:.1f}% | "
        f"N1500={needle_lut.get('L_1500', {}).get('exact_match_pct')} | "
        f"N8K={needle_lut.get('L_8192', {}).get('exact_match_pct')} | "
        f"N32K={needle_lut.get('L_32768', {}).get('exact_match_pct')} | OvN={ov_needle_lut['new_exact_pct']:.1f}% | "
        f"StreamNeedle={stream_needle_lut['stream_greedy_exact_pct']:.1f}%",
        flush=True,
    )

    payload = {
        "meta": {
            "phase": "Phase_N_Retrofit_Calcite_v31_S2_25M",
            "params": n_params,
            "cache_kb_o1": cache_kb,
            "byte_induct_slots": cfg.byte_induct_slots,
            "steps": {"retrofit": ret_steps, "qat": qat_steps},
            "seed": args.seed,
            "smoke": args.smoke_local,
            "elapsed_s": round(time.perf_counter() - t_all, 1),
            "retrofit_seconds": ret_s,
            "qat_seconds": qat_s,
            "router_max_abs_diff": router_mut_diff,
            "lut_checkpoint": str(lut_ckpt.relative_to(ROOT_DIR)),
            "lut_checkpoint_bytes": ckpt_bytes,
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
    print(f"\n[Phase N Retrofit 25M Complete] Saved -> {res_path}", flush=True)


if __name__ == "__main__":
    main()
