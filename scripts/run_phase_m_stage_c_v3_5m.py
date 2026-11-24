"""
Phase M Stage C (`Calcite v3 5M` Synthesis, Fair `RoPE + Muon` Head-to-Head,
Unified Cognitive Grokking, `Temporal Residual (TR)` Retrofit, and `4-Tap 2-bit LUT` Closure)
(`scripts/run_phase_m_stage_c_v3_5m.py`).

Parts:
  - C1 (Fair Muon Head-to-Head, 700 steps, cog_prob=0.20):
      * `C0_RoPE_5M_Muon`:               Equal-budget `RoPE_5M` trained with `MuonAdamWHybrid`
      * `C1_Calcite_v3_Muon`:            `M3_sym_byte_swa + Byte Engram + MuonAdamWHybrid`
      * `C2_Calcite_v3_DeltaNet2_Muon`:  `M3_sym_byte_swa + Byte Engram + Gated DeltaNet-2 + MuonAdamWHybrid`
  - C2 (Unified Cognitive + Language + `TR` Retrofit + `4-Tap 2-bit LUT` on winning `Calcite v3 5M`):
      * Trains winning `Calcite v3 (5M)` through `Stage 1A -> Stage 1B -> Stage 2` (`StratifiedSystem1Batcher`)
        to verify `S_5` grokking + `Overwrite Long` + `Trap` + `L=32,768` + `Needle-in-a-Haystack`.
      * Clones trained `stage2_fp` into TWO arms (`C3_noTR_control` vs `C3_TR_retrofit` with `rB=0` init,
        `4` Jacobi passes, `last-2` backprop) for `250`-step reasoning SFT, evaluated strictly at `K=8` passes!
      * Runs `4-Tap + Logit-KL STE QAT` (`350` steps) -> `checkpoints/calcite_v3_s1_5m_utf8_lut.pt` ->
        strict disk roundtrip paired eval (`FP vs 2-bit LUT`) + live `CalciteStreamEngine` CPU/GPU check.
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
from experiments.scaling_phase_k.byte_baselines import compute_rope_cache_kb  # noqa: E402
from experiments.scaling_phase_k.eval_suite import (  # noqa: E402
    eval_openweb_6domains_any_byte_lm,
    eval_val_memmap_bpb,
    has_usable_cuda,
    sync_if_cuda,
)
from kernels.fast_calcite_scans import clear_fast_qat_cache, unwrap_compiled_state_dict  # noqa: E402
from kernels.fast_packed_lut import pack_fast_model_2bit_lut, save_packed_lut_checkpoint  # noqa: E402
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
    build_matched_rope_for_tier,
    ensure_phase_k_corpus,
    eval_cognitive_v2,
    eval_long_context_stress,
    load_frozen_router_into,
    set_all_seeds,
)
from scripts.muon_opt import MuonAdamWHybrid  # noqa: E402
from scripts.run_phase_l_closure_25m import (  # noqa: E402
    _split_v2_qat_4groups,
    load_calcite_v2_lut_checkpoint,
)
from scripts.run_phase_m_ablations_5m import (  # noqa: E402
    eval_cognitive_fast,
    eval_zeroshot_needle_recall_bpb,
)
from scripts.run_phase_m_stage_b_5m import make_m3_base_cfg, train_stage_b_model  # noqa: E402
from simvq_split.memory import compute_2bit_lut_footprint, quantize_ternary_weights_ls  # noqa: E402
from simvq_split.tasks import load_gsm8k_corpora  # noqa: E402
from simvq_split.tokenizer import ByteBPETokenizer  # noqa: E402


def _split_v3_qat_4groups(
    model: CalciteV2LM,
) -> Tuple[List[nn.Parameter], List[nn.Parameter], List[nn.Parameter], List[nn.Parameter]]:
    gain_p, cryst_p, chunk_p, core_p = _split_v2_qat_4groups(model)
    # Ensure Engram embeddings (not 2-bit quantized) live cleanly in gain_p
    engram_ids = set()
    for mod in (model.byte_chunker.engram_emb2, model.byte_chunker.engram_emb3, model.byte_chunker.engram_gate):
        if mod is not None:
            for p in mod.parameters():
                if p.requires_grad:
                    engram_ids.add(id(p))
    if engram_ids:
        gain_ids = {id(p) for p in gain_p}
        chunk_p = [p for p in chunk_p if id(p) not in engram_ids]
        core_p = [p for p in core_p if id(p) not in engram_ids]
        for mod in (model.byte_chunker.engram_emb2, model.byte_chunker.engram_emb3, model.byte_chunker.engram_gate):
            if mod is not None:
                for p in mod.parameters():
                    if p.requires_grad and id(p) not in gain_ids:
                        gain_p.append(p)
                        gain_ids.add(id(p))
    return gain_p, cryst_p, chunk_p, core_p


def run_part_c1_fair_muon(
    train_mm: np.memmap,
    val_mm: np.memmap,
    router_state: Dict[str, torch.Tensor],
    tok: ByteBPETokenizer,
    tr_sents: List[str],
    eval_wins: Dict[str, List[List[int]]],
    s5_suites: Dict[str, list],
    ov_suites: Dict[str, list],
    trap_suites: Dict[str, list],
    device: str,
    steps: int,
    batch_size: int,
    micro_bs: int,
    n_val_batches: int,
    n_needle: int,
    seed: int,
    results: Dict[str, object],
    save_fn,
    force: bool,
    smoke: bool,
) -> Dict[str, object]:
    c1_cells = [
        ("C0_RoPE_5M_Muon", True, {}),
        ("C1_Calcite_v3_Muon", False, {"use_byte_engram": True, "engram_slots": 4096, "engram_dim": 32}),
        (
            "C2_Calcite_v3_DeltaNet2_Muon",
            False,
            {"use_byte_engram": True, "engram_slots": 4096, "engram_dim": 32, "use_deltanet2_gates": True},
        ),
    ]
    c1_out: Dict[str, object] = results.setdefault("part_c1_fair_muon", {})
    for name, is_rope, kw in c1_cells:
        if name in c1_out and not force and not smoke:
            print(f"[skip C1] {name} already present", flush=True)
            continue
        print(f"\n==================== [{name}] ====================", flush=True)
        set_all_seeds(seed)
        if is_rope:
            model: nn.Module = build_matched_rope_for_tier("S1_5M", 5_382_253).to(device)
            n_params = sum(p.numel() for p in model.parameters())
            cache_kb = round(float(compute_rope_cache_kb(model.cfg, 2048)), 2)
        else:
            cfg = make_m3_base_cfg(kw)
            model = CalciteV2LM(cfg).to(device)
            load_frozen_router_into(model, router_state)
            n_params = sum(p.numel() for p in model.parameters())
            cache_kb = round(compute_v2_state_cache_kb(cfg, 2048), 2)

        print(f"[{name}] params={n_params:,} | cache_L2048={cache_kb:.2f} KB | optimizer=MuonAdamW", flush=True)
        train_info = train_stage_b_model(
            model=model,
            train_mm=train_mm,
            tok=tok,
            tr_sents=tr_sents,
            device=device,
            total_steps=steps,
            batch_size=batch_size,
            seq_len=512,
            micro_bs=micro_bs,
            base_lr=1.8e-3,
            min_lr=1.2e-4,
            weight_decay=0.04,
            cog_prob=0.20,
            seed=seed,
            use_muon=True,
            muon_lr_mult=6.0,
            tag=name,
        )
        model.eval()
        val_bpb = float(
            eval_val_memmap_bpb(model, val_mm, seq_len=512, n_batches=n_val_batches, batch_size=8, device=device)
        )
        ow6 = eval_openweb_6domains_any_byte_lm(model, eval_wins, device=device, batch_size=8)
        long_ctx = eval_long_context_stress(model, val_mm, device=device, lengths=(2048, 8192))
        needle = eval_zeroshot_needle_recall_bpb(
            model, val_mm, device=device, haystack_bytes=1500, n_samples=n_needle, seed=7777
        )
        cog = eval_cognitive_fast(model, tok, s5_suites, ov_suites, trap_suites, device=device, batch_size=16)

        c1_out[name] = {
            "name": name,
            "params": int(n_params),
            "cache_kb_L2048": cache_kb,
            "train": train_info,
            "val_bpb": round(val_bpb, 4),
            "ow6_mean_bpb": round(float(ow6["mean_openweb_bpb"]), 4),
            "ow6_domains": {k: round(float(v["byte_bpb"]), 4) for k, v in ow6["regimes"].items()},
            "mid_utf8_split_pct": round(float(ow6.get("max_mid_utf8_split_pct", 0.0)), 4),
            "long_context": long_ctx,
            "needle_1500b": needle,
            "cognitive": cog,
        }
        print(
            f"[{name}] DONE | Val={val_bpb:.4f} | OW6={ow6['mean_openweb_bpb']:.4f} | "
            f"Py={c1_out[name]['ow6_domains'].get('3_Python_Code')} | "
            f"CJK={c1_out[name]['ow6_domains'].get('5_CJK_UTF8')} | "
            f"L2048={long_ctx.get('L_2048', {}).get('bpb')} | "
            f"L8192={long_ctx.get('L_8192', {}).get('bpb')} | "
            f"Needle={needle['recall_gain_bpb']:+.4f} | {train_info['ms_per_step']:.1f} ms/st",
            flush=True,
        )
        save_fn()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return c1_out


def train_unified_v3_5m(
    cfg: CalciteV2Config,
    router_state: Dict[str, torch.Tensor],
    batcher: StratifiedSystem1Batcher,
    tok: ByteBPETokenizer,
    device: str,
    s1a_steps: int,
    s1b_steps: int,
    s2_steps: int,
    batch_size: int,
    micro_bs: int,
    seed: int,
    ckpt_fp_path: Path,
) -> Tuple[CalciteV2LM, Dict[str, object], Dict[str, torch.Tensor]]:
    set_all_seeds(seed)
    model = CalciteV2LM(cfg).to(device)
    load_frozen_router_into(model, router_state)
    router_snap = _snapshot_router_params(model)
    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
    grok_steps = s1a_steps + s1b_steps
    total_fp_steps = grok_steps + s2_steps
    t0 = time.perf_counter()

    # Stage 1A: S_5 + Overwrite grokking with MuonAdamW (Woodbury quiet at 0.05x AdamW wd=0.0!)
    opt_1a = MuonAdamWHybrid(
        model,
        base_lr=1.8e-3,
        muon_lr_mult=5.5,
        weight_decay=0.02,
        device=device,
        woodbury_lr_mult=0.05,
        protect_crystal=False,
    )
    warm_1a = min(40, max(1, s1a_steps // 12))
    print(f"\n[C2-Unified] Stage 1A ({s1a_steps} steps)...", flush=True)
    for st in range(1, s1a_steps + 1):
        model.train()
        model.set_crystal_schedule(st, s1a_steps)
        lr_s = _cosine_lr(st, s1a_steps, warm_1a, 1.8e-3, 2.5e-4)
        scale = lr_s / 1.8e-3
        for pg in opt_1a.param_groups:
            pg["lr"] = 1.8e-3 * float(pg.get("lr_mult", 1.0)) * scale
        xb_full, yb_full = batcher.sample_batch(
            batch_size,
            device,
            step=st,
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
            f"1A s={st}",
            lambda xb, yb, cur_acc, m_ref=model: _fp_micro_ce(
                m_ref, xb, yb, tok.PAD_ID, use_amp, cur_acc, 0.04
            ),
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_1a.step()
        if st % max(1, s1a_steps // 4) == 0 or st == s1a_steps:
            ce_v = float(ce_sum_t.item()) / float(max(1, batch_size // micro_bs)) if ce_sum_t is not None else 0.0
            print(f"  [1A] step={st}/{s1a_steps} | ce={ce_v:.4f} | {time.perf_counter()-t0:.1f}s", flush=True)

    # Stage 1B: Woodbury Overwrite/Trap consolidation (Crystal protected at 0.04x AdamW, Woodbury at 1.8x)
    model.set_crystal_schedule(1, 1)
    model.set_snap_alpha(1.0)
    opt_1b = MuonAdamWHybrid(
        model,
        base_lr=1.0e-3,
        muon_lr_mult=3.5,
        weight_decay=0.01,
        device=device,
        woodbury_lr_mult=1.8,
        protect_crystal=True,
        crystal_lr_mult=0.04,
    )
    warm_1b = min(30, max(1, s1b_steps // 8))
    print(f"[C2-Unified] Stage 1B ({s1b_steps} steps)...", flush=True)
    for st in range(1, s1b_steps + 1):
        step = s1a_steps + st
        model.train()
        lr_s = _cosine_lr(st, s1b_steps, warm_1b, 1.0e-3, 1.5e-4)
        scale = lr_s / 1.0e-3
        for pg in opt_1b.param_groups:
            pg["lr"] = 1.0e-3 * float(pg.get("lr_mult", 1.0)) * scale
        xb_full, yb_full = batcher.sample_batch(
            batch_size,
            device,
            step=step,
            grok_steps=grok_steps,
            total_steps=total_fp_steps,
            s1b_steps=s1b_steps,
        )
        micro_bs, ce_sum_t = _run_accum_microbatches(
            xb_full,
            yb_full,
            micro_bs,
            opt_1b,
            device,
            f"1B s={st}",
            lambda xb, yb, cur_acc, m_ref=model: _fp_micro_ce(
                m_ref, xb, yb, tok.PAD_ID, use_amp, cur_acc, 0.04
            ),
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_1b.step()
        if st % max(1, s1b_steps // 2) == 0 or st == s1b_steps:
            ce_v = float(ce_sum_t.item()) / float(max(1, batch_size // micro_bs)) if ce_sum_t is not None else 0.0
            print(f"  [1B] step={st}/{s1b_steps} | ce={ce_v:.4f} | {time.perf_counter()-t0:.1f}s", flush=True)

    # Stage 2: Protected 6-domain OpenWeb + cognitive replay (Crystal 0.05x & Woodbury 0.07x, Muon on hidden 2D)
    opt_s2 = MuonAdamWHybrid(
        model,
        base_lr=1.2e-3,
        muon_lr_mult=5.0,
        weight_decay=0.03,
        device=device,
        woodbury_lr_mult=0.07,
        protect_crystal=True,
        crystal_lr_mult=0.05,
    )
    warm_s2 = min(40, max(1, s2_steps // 8))
    print(f"[C2-Unified] Stage 2 ({s2_steps} steps)...", flush=True)
    for st in range(1, s2_steps + 1):
        step = grok_steps + st
        model.train()
        lr_s = _cosine_lr(st, s2_steps, warm_s2, 1.2e-3, 1.0e-4)
        scale = lr_s / 1.2e-3
        for pg in opt_s2.param_groups:
            pg["lr"] = 1.2e-3 * float(pg.get("lr_mult", 1.0)) * scale
        xb_full, yb_full = batcher.sample_batch(
            batch_size,
            device,
            step=step,
            grok_steps=grok_steps,
            total_steps=total_fp_steps,
            s1b_steps=s1b_steps,
        )
        micro_bs, ce_sum_t = _run_accum_microbatches(
            xb_full,
            yb_full,
            micro_bs,
            opt_s2,
            device,
            f"S2 s={st}",
            lambda xb, yb, cur_acc, m_ref=model: _fp_micro_ce(
                m_ref, xb, yb, tok.PAD_ID, use_amp, cur_acc, 0.04
            ),
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_s2.step()
        if st % max(1, s2_steps // 4) == 0 or st == s2_steps:
            ce_v = float(ce_sum_t.item()) / float(max(1, batch_size // micro_bs)) if ce_sum_t is not None else 0.0
            print(f"  [S2] step={st}/{s2_steps} | ce={ce_v:.4f} | {time.perf_counter()-t0:.1f}s", flush=True)

    r_diff = _assert_router_unchanged(model, router_snap, "V3_5M_FP")
    sd_clean = unwrap_compiled_state_dict(model)
    ckpt_fp_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(cfg), "state_dict": {k: v.cpu() for k, v in sd_clean.items()}}, ckpt_fp_path)
    print(f"[C2-Unified] Saved FP checkpoint -> {ckpt_fp_path} (router_diff={r_diff})", flush=True)
    return model, {"total_seconds": round(time.perf_counter() - t0, 1), "router_max_abs_diff": r_diff}, router_snap


def run_tr_retrofit_ablation(
    base_cfg: CalciteV2Config,
    base_sd: Dict[str, torch.Tensor],
    batcher: StratifiedSystem1Batcher,
    val_mm: np.memmap,
    tok: ByteBPETokenizer,
    s5_suites: Dict[str, list],
    ov_suites: Dict[str, list],
    trap_suites: Dict[str, list],
    device: str,
    sft_steps: int,
    batch_size: int,
    micro_bs: int,
    seed: int,
    n_val_batches: int,
) -> Dict[str, object]:
    """
    Princeton Temporal Residual (TR) Retrofit Test:
    Clones the exact trained FP checkpoint into:
      1. `noTR_control`: SFT (1 pass)
      2. `TR_retrofit`:  SFT with `rB=0` init, `4` Jacobi passes, `last-2` backprop
    Evaluated strictly at `K=8` passes (exact decoding proxy per Section 5.2)!
    """
    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
    out: Dict[str, object] = {}

    for arm_name, enable_tr in (("noTR_control", False), ("TR_retrofit", True)):
        print(f"\n---------- [C3 {arm_name}] ({sft_steps} SFT steps) ----------", flush=True)
        set_all_seeds(seed + 303)
        cfg_arm = copy.deepcopy(base_cfg)
        cfg_arm.use_temporal_residual = enable_tr
        cfg_arm.tr_train_passes = 4
        cfg_arm.tr_eval_passes = 8
        cfg_arm.tr_backprop_passes = 2
        m = CalciteV2LM(cfg_arm).to(device)
        m.byte_chunker.lock_router(True)
        m.load_state_dict(base_sd, strict=False)
        m.set_crystal_schedule(1, 1)
        m.set_snap_alpha(1.0)

        # Princeton Table 1 LR ratio: small LR on pretrained weights, higher LR on TR scalars & phi
        tr_scalar_p, tr_phi_p, base_p = [], [], []
        for nm, p in m.named_parameters():
            if not p.requires_grad:
                continue
            if nm in ("tr_rA", "tr_rB", "tr_sB"):
                tr_scalar_p.append(p)
            elif nm.startswith("tr_phi"):
                tr_phi_p.append(p)
            else:
                base_p.append(p)
        groups = [{"params": base_p, "lr": 2.0e-4, "weight_decay": 0.01, "lr_mult": 1.0}]
        if tr_phi_p:
            groups.append({"params": tr_phi_p, "lr": 1.0e-3, "weight_decay": 0.0, "lr_mult": 5.0})
        if tr_scalar_p:
            groups.append({"params": tr_scalar_p, "lr": 1.0e-2, "weight_decay": 0.0, "lr_mult": 50.0})
        opt = _make_adamw(groups, device=device)

        warm_tr = min(20, max(1, sft_steps // 10))
        cur_mbs = micro_bs
        t0 = time.perf_counter()
        for st in range(1, sft_steps + 1):
            m.train()
            lr_s = _cosine_lr(st, sft_steps, warm_tr, 2.0e-4, 2.0e-5)
            sc = lr_s / 2.0e-4
            for pg in opt.param_groups:
                pg["lr"] = 2.0e-4 * float(pg["lr_mult"]) * sc
            # Mix 25% S1A (S_5), 25% S1B (Overwrite/Trap), 50% Stage 2 replay so language never drifts
            fake_step = 1 if (st % 4 == 1) else (11 if (st % 4 == 2) else 25)
            xb_full, yb_full = batcher.sample_batch(
                batch_size,
                device,
                step=fake_step,
                grok_steps=20,
                total_steps=40,
                s1b_steps=10,
                s1a_offset=0,
            )
            cur_mbs, ce_sum_t = _run_accum_microbatches(
                xb_full,
                yb_full,
                cur_mbs,
                opt,
                device,
                f"TR s={st}",
                lambda xb, yb, cur_acc, m_ref=m: _fp_micro_ce(
                    m_ref, xb, yb, tok.PAD_ID, use_amp, cur_acc, 0.04
                ),
            )
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            if st % max(1, sft_steps // 3) == 0 or st == sft_steps:
                ce_v = float(ce_sum_t.item()) / float(max(1, batch_size // cur_mbs)) if ce_sum_t is not None else 0.0
                tr_info = ""
                if enable_tr and m.tr_rB is not None:
                    tr_info = f" | max|rB|={float(m.tr_rB.detach().abs().max().item()):.3f}"
                print(
                    f"  [{arm_name}] step={st}/{sft_steps} | ce={ce_v:.4f} | "
                    f"{time.perf_counter()-t0:.1f}s{tr_info}",
                    flush=True,
                )

        m.eval()
        val_bpb = float(
            eval_val_memmap_bpb(m, val_mm, seq_len=512, n_batches=n_val_batches, batch_size=8, device=device)
        )
        cog = eval_cognitive_v2(m, tok, s5_suites, ov_suites, trap_suites, device=device, batch_size=16)
        arm_res: Dict[str, object] = {
            "sft_seconds": round(time.perf_counter() - t0, 1),
            "val_bpb_K8": round(val_bpb, 4),
            "s5_sum": cog["s5_sum"],
            "s5": {k: cog["s5"][k]["exact_pct"] for k in ("k2", "k4", "k8", "k16")},
            "ov": {k: cog["overwrite"][k]["exact_pct"] for k in ("in_dist", "ood_med", "ood_long")},
            "trap": {k: cog["trap"][k]["exact_pct"] for k in ("in_dist", "ood_long")},
            "sel_f1_long": cog["selective_f1_ood_long"],
        }
        if enable_tr and m.tr_rB is not None and m.tr_sB is not None:
            arm_res["tr_rB_learned"] = [round(float(v), 4) for v in m.tr_rB.detach().cpu().tolist()]
            arm_res["tr_sB_learned"] = [round(float(v), 4) for v in m.tr_sB.detach().cpu().tolist()]
        out[arm_name] = arm_res
        print(
            f"[{arm_name}] DONE (K=8) | Val={val_bpb:.4f} | S5={cog['s5_sum']:.1f}% "
            f"(k2={arm_res['s5']['k2']}, k4={arm_res['s5']['k4']}, k8={arm_res['s5']['k8']}) | "
            f"OV_L={arm_res['ov']['ood_long']}% | SelF1_L={cog['selective_f1_ood_long']}%",
            flush=True,
        )
        del m
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def run_4tap_qat_5m(
    cfg: CalciteV2Config,
    fp_sd: Dict[str, torch.Tensor],
    batcher: StratifiedSystem1Batcher,
    tok: ByteBPETokenizer,
    device: str,
    qat_steps: int,
    batch_size: int,
    micro_bs: int,
    seed: int,
    lut_ckpt_path: Path,
) -> Tuple[CalciteV2LM, Dict[str, object]]:
    set_all_seeds(seed + 505)
    teacher = CalciteV2LM(cfg).to(device)
    teacher.byte_chunker.byte_chunk_mode = "utf8"
    teacher.byte_chunker.target_chunk_bytes = float(cfg.target_chunk_bytes)
    teacher.byte_chunker.use_static_m_cap = False
    teacher.byte_chunker.lock_router(True)
    teacher.load_state_dict(fp_sd, strict=True)
    teacher.set_crystal_schedule(1, 1)
    teacher.set_snap_alpha(1.0)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = CalciteV2LM(cfg).to(device)
    student.byte_chunker.byte_chunk_mode = "utf8"
    student.byte_chunker.target_chunk_bytes = float(cfg.target_chunk_bytes)
    student.byte_chunker.use_static_m_cap = False
    student.byte_chunker.lock_router(True)
    student.load_state_dict(fp_sd, strict=True)
    student.set_crystal_schedule(1, 1)
    student.set_snap_alpha(1.0)
    student.set_qat_mix(0.30)
    student.set_ternary_qat(True)

    gain_p, cryst_p, chunk_p, core_p = _split_v3_qat_4groups(student)
    all_qp = gain_p + cryst_p + chunk_p + core_p
    q_lrs = (1.2e-3, 8.0e-5, 6.0e-4, 4.2e-4)
    q_mins = (1.2e-4, 1.5e-5, 7.0e-5, 5.0e-5)
    opt = _make_adamw(
        [
            {"params": gain_p, "lr": q_lrs[0]},
            {"params": cryst_p, "lr": q_lrs[1]},
            {"params": chunk_p, "lr": q_lrs[2]},
            {"params": core_p, "lr": q_lrs[3]},
        ],
        device=device,
        weight_decay=0.0,
    )
    opt.register_step_post_hook(lambda *_a, **_kw: clear_fast_qat_cache())
    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
    warm_q = min(40, max(1, qat_steps // 10))
    ramp_q = min(180, max(1, qat_steps // 3))
    kl_temp = 1.5
    taps_fp: List[torch.Tensor] = []
    taps_q: List[torch.Tensor] = []
    hk_fp = _attach_4tap_hooks(teacher, taps_fp, detach=True)
    hk_q = _attach_4tap_hooks(student, taps_q, detach=False)
    cur_mbs = min(batch_size, max(4, micro_bs // 2))
    t0 = time.perf_counter()
    print(f"\n[C2-QAT] 4-Group + 4-Tap STE QAT ({qat_steps} steps)...", flush=True)
    try:
        for st in range(1, qat_steps + 1):
            student.train()
            mix = min(1.0, 0.30 + 0.70 * (st / float(ramp_q)))
            student.set_qat_mix(mix)
            clear_fast_qat_cache()
            for gi, (blr, mlr) in enumerate(zip(q_lrs, q_mins)):
                opt.param_groups[gi]["lr"] = _cosine_lr(st, qat_steps, warm_q, blr, mlr)
            xb_full, yb_full = batcher.sample_batch(
                batch_size, device, step=40, grok_steps=20, total_steps=40, s1b_steps=10
            )
            cur_mbs, q_ce_sum_t = _run_accum_microbatches(
                xb_full,
                yb_full,
                cur_mbs,
                opt,
                device,
                f"4Tap-QAT s={st}",
                lambda xb, yb, cur_acc, t_m=teacher, s_m=student: _qat_micro_step(
                    t_m, s_m, taps_fp, taps_q, kl_temp, xb, yb, tok.PAD_ID, use_amp, cur_acc
                ),
            )
            torch.nn.utils.clip_grad_norm_(all_qp, 1.0)
            opt.step()
            clear_fast_qat_cache()
            if st % max(1, qat_steps // 3) == 0 or st == qat_steps:
                qce = float(q_ce_sum_t.item()) / float(max(1, batch_size // cur_mbs)) if q_ce_sum_t is not None else 0.0
                print(
                    f"  [QAT] step={st}/{qat_steps} | mix={mix:.2f} | ce={qce:.4f} | "
                    f"{time.perf_counter()-t0:.1f}s",
                    flush=True,
                )
    finally:
        for h in hk_fp + hk_q:
            h.remove()
        clear_fast_qat_cache()

    student.set_ternary_qat(False)
    student.set_qat_mix(1.0)
    qat_sd = unwrap_compiled_state_dict(student)
    del teacher, student, opt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    clean_qat = CalciteV2LM(cfg)
    clean_qat.load_state_dict(qat_sd, strict=True)
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

    lut_model = pack_fast_model_2bit_lut(clean_qat, backend="auto").to("cpu").eval()
    lut_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    save_packed_lut_checkpoint(
        lut_ckpt_path,
        lut_model,
        cfg,
        tok,
        metadata={"tier": "S1_5M", "arch": "CalciteV3", "phase": "Phase_M_Stage_C_5M"},
    )
    fp_info = compute_2bit_lut_footprint(lut_model)
    print(
        f"[C2-QAT] Saved 2-bit LUT checkpoint -> {lut_ckpt_path} "
        f"({lut_ckpt_path.stat().st_size:,} B, footprint={fp_info})",
        flush=True,
    )
    return lut_model, {
        "qat_seconds": round(time.perf_counter() - t0, 1),
        "ckpt_bytes": int(lut_ckpt_path.stat().st_size),
        "lut_footprint": fp_info,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase M Stage C: Calcite v3 5M synthesis, cognitive & TR retrofit")
    ap.add_argument("--c1-steps", type=int, default=700)
    ap.add_argument("--s1a-steps", type=int, default=1200)
    ap.add_argument("--s1b-steps", type=int, default=400)
    ap.add_argument("--s2-steps", type=int, default=800)
    ap.add_argument("--tr-steps", type=int, default=250)
    ap.add_argument("--qat-steps", type=int, default=350)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--micro-bs", type=int, default=16)
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--n-needle", type=int, default=24)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke-local", action="store_true")
    ap.add_argument("--out", type=str, default="benchmarks/phase_m_v3/stage_c_v3_5m_results.json")
    args = ap.parse_args()

    device = "cuda" if has_usable_cuda() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.74, 0)
        print(f"[env] GPU: {torch.cuda.get_device_name(0)} | BF16={torch.cuda.is_bf16_supported()}", flush=True)

    train_bin, val_bin = ensure_phase_k_corpus(ROOT_DIR / "data_cache")
    train_mm = np.memmap(train_bin, dtype=np.uint16, mode="r")
    val_mm = np.memmap(val_bin, dtype=np.uint16, mode="r")

    router_path = ROOT_DIR / "benchmarks" / "phase_l_v2" / "frozen_utf8_router.pt"
    router_state = torch.load(router_path, map_location="cpu", weights_only=True)

    tok = ByteBPETokenizer()
    n_eval = 4 if args.smoke_local else args.n_eval
    tr_docs, te_docs, tr_sents, _ = load_gsm8k_corpora(ROOT_DIR)
    train_wins, eval_wins, _, _ = build_disjoint_openweb_corpora(ROOT_DIR, tok, tr_docs, te_docs, seq_len=512)
    _, _, s5_suites, ov_suites, trap_suites = build_eval_bundles(ROOT_DIR, tok, n_eval=n_eval, seq_len=512)

    out_path = ROOT_DIR / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: Dict[str, object] = (
        json.loads(out_path.read_text(encoding="utf-8"))
        if (out_path.is_file() and not args.smoke_local)
        else {}
    )
    t_all = time.perf_counter()

    def _save() -> None:
        if not args.smoke_local:
            results["elapsed_s"] = round(time.perf_counter() - t_all, 1)
            out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    c1_steps = 2 if args.smoke_local else args.c1_steps
    s1a_steps = 4 if args.smoke_local else args.s1a_steps
    s1b_steps = 2 if args.smoke_local else args.s1b_steps
    s2_steps = 4 if args.smoke_local else args.s2_steps
    tr_steps = 2 if args.smoke_local else args.tr_steps
    qat_steps = 2 if args.smoke_local else args.qat_steps
    bs = 4 if args.smoke_local else args.batch_size
    mbs = 4 if args.smoke_local else args.micro_bs
    n_val_b = 2 if args.smoke_local else 24
    n_needle = 2 if args.smoke_local else args.n_needle

    # 1. Part C1: Fair Muon Head-to-Head (`RoPE + Muon` vs `Calcite v3 + Muon` vs `+ DeltaNet2`)
    c1_out = run_part_c1_fair_muon(
        train_mm=train_mm,
        val_mm=val_mm,
        router_state=router_state,
        tok=tok,
        tr_sents=tr_sents,
        eval_wins=eval_wins,
        s5_suites=s5_suites,
        ov_suites=ov_suites,
        trap_suites=trap_suites,
        device=device,
        steps=c1_steps,
        batch_size=bs,
        micro_bs=mbs,
        n_val_batches=n_val_b,
        n_needle=n_needle,
        seed=args.seed,
        results=results,
        save_fn=_save,
        force=args.force,
        smoke=args.smoke_local,
    )

    # Pick winner between C1 (`M3+Engram`) and C2 (`M3+Engram+DeltaNet2`)
    v_c1 = float(c1_out["C1_Calcite_v3_Muon"]["val_bpb"])
    v_c2 = float(c1_out["C2_Calcite_v3_DeltaNet2_Muon"]["val_bpb"])
    use_dn2 = bool(v_c2 < v_c1 - 0.0040)
    win_tag = "C2_Calcite_v3_DeltaNet2" if use_dn2 else "C1_Calcite_v3"
    print(f"\n[Stage C] Selected unified v3 winner: {win_tag} (C1={v_c1:.4f} vs C2={v_c2:.4f})", flush=True)
    results["selected_v3_winner"] = {"tag": win_tag, "use_deltanet2_gates": use_dn2}
    _save()

    # 2. Part C2: Unified Cognitive + Language + 32K + TR Retrofit + 4-Tap 2-bit LUT on 5M
    v3_cfg = make_m3_base_cfg(
        {
            "use_byte_engram": True,
            "engram_slots": 4096,
            "engram_dim": 32,
            "use_deltanet2_gates": use_dn2,
        }
    )
    batcher = StratifiedSystem1Batcher(
        train_mm=train_mm,
        tr_docs=tr_docs,
        tr_sents=tr_sents,
        tok=tok,
        seq_len=512,
        train_wins=train_wins,
        seed=args.seed + 101,
    )
    fp_ckpt_path = ROOT_DIR / "benchmarks" / "phase_m_v3" / (
        "smoke_v3_5m_fp.pt" if args.smoke_local else "c_v3_5m_stage2_fp.pt"
    )
    lut_ckpt_path = ROOT_DIR / (
        "benchmarks/phase_m_v3/smoke_v3_5m_lut.pt"
        if args.smoke_local
        else "checkpoints/calcite_v3_s1_5m_utf8_lut.pt"
    )

    if fp_ckpt_path.is_file() and not args.force and not args.smoke_local and "part_c2_unified_fp" in results:
        print(f"[skip C2-FP] Loading existing {fp_ckpt_path}", flush=True)
        blob = torch.load(fp_ckpt_path, map_location="cpu", weights_only=False)
        fp_sd = blob["state_dict"]
    else:
        model_fp, tr_meta, _ = train_unified_v3_5m(
            cfg=v3_cfg,
            router_state=router_state,
            batcher=batcher,
            tok=tok,
            device=device,
            s1a_steps=s1a_steps,
            s1b_steps=s1b_steps,
            s2_steps=s2_steps,
            batch_size=bs,
            micro_bs=mbs,
            seed=args.seed,
            ckpt_fp_path=fp_ckpt_path,
        )
        model_fp.eval()
        val_fp = float(
            eval_val_memmap_bpb(model_fp, val_mm, seq_len=512, n_batches=n_val_b, batch_size=8, device=device)
        )
        ow6_fp = eval_openweb_6domains_any_byte_lm(model_fp, eval_wins, device=device, batch_size=8)
        long_fp = eval_long_context_stress(
            model_fp, val_mm, device=device, lengths=(2048, 8192) if args.smoke_local else (2048, 8192, 32768)
        )
        needle_fp = eval_zeroshot_needle_recall_bpb(
            model_fp, val_mm, device=device, haystack_bytes=1500, n_samples=n_needle, seed=7777
        )
        cog_fp = eval_cognitive_v2(model_fp, tok, s5_suites, ov_suites, trap_suites, device=device, batch_size=16)
        results["part_c2_unified_fp"] = {
            "params": int(sum(p.numel() for p in model_fp.parameters())),
            "cache_kb_o1": round(compute_v2_state_cache_kb(v3_cfg, 32768), 2),
            "train": tr_meta,
            "val_bpb": round(val_fp, 4),
            "ow6_mean_bpb": round(float(ow6_fp["mean_openweb_bpb"]), 4),
            "ow6_domains": {k: round(float(v["byte_bpb"]), 4) for k, v in ow6_fp["regimes"].items()},
            "mid_utf8_split_pct": round(float(ow6_fp.get("max_mid_utf8_split_pct", 0.0)), 4),
            "long_context": long_fp,
            "needle_1500b": needle_fp,
            "cognitive": cog_fp,
        }
        print(
            f"[C2-FP] Val={val_fp:.4f} | OW6={ow6_fp['mean_openweb_bpb']:.4f} | "
            f"L2048={long_fp.get('L_2048', {}).get('bpb')} | L32K={long_fp.get('L_32768', {}).get('bpb')} | "
            f"S5={cog_fp['s5_sum']:.1f}% | OV_L={cog_fp['overwrite']['ood_long']['exact_pct']}% | "
            f"SelF1_L={cog_fp['selective_f1_ood_long']}%",
            flush=True,
        )
        fp_sd = {k: v.detach().cpu().clone() for k, v in unwrap_compiled_state_dict(model_fp).items()}
        del model_fp
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _save()

    # 3. Part C3: Temporal Residual (TR) Retrofit on trained FP checkpoint
    if "part_c3_tr_retrofit" not in results or args.force or args.smoke_local:
        results["part_c3_tr_retrofit"] = run_tr_retrofit_ablation(
            base_cfg=v3_cfg,
            base_sd=fp_sd,
            batcher=batcher,
            val_mm=val_mm,
            tok=tok,
            s5_suites=s5_suites,
            ov_suites=ov_suites,
            trap_suites=trap_suites,
            device=device,
            sft_steps=tr_steps,
            batch_size=bs,
            micro_bs=mbs,
            seed=args.seed,
            n_val_batches=n_val_b,
        )
        _save()

    # 4. Part C4: 4-Tap 2-bit LUT QAT + Strict Disk Roundtrip + Live CPU/GPU Stream Engine
    if "part_c4_lut_and_stream" not in results or args.force or args.smoke_local:
        lut_cpu, qat_meta = run_4tap_qat_5m(
            cfg=v3_cfg,
            fp_sd=fp_sd,
            batcher=batcher,
            tok=tok,
            device=device,
            qat_steps=qat_steps,
            batch_size=bs,
            micro_bs=mbs,
            seed=args.seed,
            lut_ckpt_path=lut_ckpt_path,
        )
        del lut_cpu
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Strict disk roundtrip load + paired eval (cog_bs=2 to avoid Windows WDDM swap!)
        lut_loaded_gpu, _, _, _ = load_calcite_v2_lut_checkpoint(lut_ckpt_path, device=device)
        val_lut = float(
            eval_val_memmap_bpb(lut_loaded_gpu, val_mm, seq_len=512, n_batches=n_val_b, batch_size=8, device=device)
        )
        ow6_lut = eval_openweb_6domains_any_byte_lm(lut_loaded_gpu, eval_wins, device=device, batch_size=8)
        long_lut = eval_long_context_stress(
            lut_loaded_gpu, val_mm, device=device, lengths=(2048, 8192) if args.smoke_local else (2048, 8192, 32768)
        )
        cog_lut = eval_cognitive_v2(lut_loaded_gpu, tok, s5_suites, ov_suites, trap_suites, device=device, batch_size=2)
        stream_gpu = benchmark_stream_engine_live(lut_loaded_gpu, device=device, n_bytes=32 if args.smoke_local else 256)
        del lut_loaded_gpu
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        lut_loaded_cpu, _, _, _ = load_calcite_v2_lut_checkpoint(lut_ckpt_path, device="cpu")
        stream_cpu = benchmark_stream_engine_live(lut_loaded_cpu, device="cpu", n_bytes=32 if args.smoke_local else 256)
        del lut_loaded_cpu
        if args.smoke_local:
            fp_ckpt_path.unlink(missing_ok=True)
            lut_ckpt_path.unlink(missing_ok=True)

        fp_val_ref = float(results["part_c2_unified_fp"]["val_bpb"])
        fp_ow6_ref = float(results["part_c2_unified_fp"]["ow6_mean_bpb"])
        results["part_c4_lut_and_stream"] = {
            "qat": qat_meta,
            "val_bpb_lut": round(val_lut, 4),
            "val_gap_lut_minus_fp": round(val_lut - fp_val_ref, 4),
            "ow6_mean_bpb_lut": round(float(ow6_lut["mean_openweb_bpb"]), 4),
            "ow6_gap_lut_minus_fp": round(float(ow6_lut["mean_openweb_bpb"]) - fp_ow6_ref, 4),
            "ow6_domains_lut": {k: round(float(v["byte_bpb"]), 4) for k, v in ow6_lut["regimes"].items()},
            "mid_utf8_split_pct": round(float(ow6_lut.get("max_mid_utf8_split_pct", 0.0)), 4),
            "long_context_lut": long_lut,
            "cognitive_lut": cog_lut,
            "stream_gpu": stream_gpu,
            "stream_cpu": stream_cpu,
        }
        print(
            f"[C4-LUT] Val={val_lut:.4f} (gap={val_lut-fp_val_ref:+.4f}) | "
            f"OW6={ow6_lut['mean_openweb_bpb']:.4f} (gap={float(ow6_lut['mean_openweb_bpb'])-fp_ow6_ref:+.4f}) | "
            f"S5={cog_lut['s5_sum']:.1f}% | OV_L={cog_lut['overwrite']['ood_long']['exact_pct']}% | "
            f"CPU_stream={stream_cpu['bytes_per_sec']} B/s (err={stream_cpu['stream_vs_batch_max_err']}, "
            f"RAM={stream_cpu['state_kb']} KB, O1={stream_cpu['strict_o1_ram_invariant']})",
            flush=True,
        )
        _save()

    print(f"\n[STAGE C ALL DONE] total_elapsed={time.perf_counter() - t_all:.1f}s | out={out_path}", flush=True)


if __name__ == "__main__":
    main()
