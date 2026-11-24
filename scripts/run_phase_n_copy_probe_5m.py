"""
Phase N Probe A: Pure Verbatim Multi-Needle Copy Curriculum (`L=128 -> 256 -> 512`)
(`scripts/run_phase_n_copy_probe_5m.py`).

Isolates the **Induction-Head Phase Transition** without waiting 10,000 LM steps:
  - `P0_RoPE_5M`:               Full quadratic attention (`12,288 KB` KV @ 2K)
  - `P1_Calcite_v3_5M`:         Exact `Calcite v3` (`Woodbury`, `120.5 KB` O(1))
  - `P2_Calcite_v3_InductSlots`:`Calcite v3` + upgraded `InductionSlotMemory` (`145.5 KB` O(1))

Curriculum (identical data stream across all 3 cells):
  - Steps `1..200`:   `L = 128` (short haystack, triggers induction head transition fast)
  - Steps `201..400`: `L = 256` (medium haystack, 1–3 needles)
  - Steps `401..600`: `L = 512` (full haystack, 1–3 needles + 20% overwrite)

Evaluates cleanly on:
  1. Seen-template unseen-code/unseen-prose (`L=512`, `L=1500`) -> proves capacity & distance scaling.
  2. Strictly UNSEEN-template (`"Important reference code:"` @ `1500 B`, and 2-Needle `EVAL_MULTI_TEMPLATES`
     @ `L = 256 -> 512 -> 1500 -> 4096 -> 8192 B`) -> proves zero-shot generalization without template overfitting.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Tuple

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
from experiments.scaling_phase_k.eval_suite import has_usable_cuda  # noqa: E402
from scripts.colab_phase_k_scaling_runner import _cosine_lr  # noqa: E402
from scripts.colab_phase_l_v2_runner import (  # noqa: E402
    ensure_phase_k_corpus,
    load_frozen_router_into,
    set_all_seeds,
)
from scripts.muon_opt import MuonAdamWHybrid  # noqa: E402
from scripts.run_phase_n_needle_diag_5m import (  # noqa: E402
    TRAIN_COPY_TEMPLATES,
    _enc_utf8,
    _score_needle_pair,
    eval_heldout_multi_needle_ladder,
    eval_heldout_passkey_single,
    sample_train_copy_row,
)
from simvq_split.baselines import BaselineConfig, RoPETransformerLM  # noqa: E402


def sample_pure_copy_batch(
    train_mm: np.memmap,
    batch_size: int,
    seq_len: int,
    rng_py: random.Random,
    rng_np: np.random.Generator,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs: List[List[int]] = []
    ys: List[List[int]] = []
    for _ in range(batch_size):
        xr, yr = sample_train_copy_row(train_mm, seq_len, rng_py, rng_np, pad_id=0)
        xs.append(xr)
        ys.append(yr)
    return (
        torch.tensor(xs, dtype=torch.long, device=device),
        torch.tensor(ys, dtype=torch.long, device=device),
    )


@torch.no_grad()
def eval_seen_tpl_unseen_code(
    model: nn.Module,
    val_mm: np.memmap,
    device: str,
    haystacks: Tuple[int, ...] = (512, 1500, 4096),
    n_samples: int = 24,
    seed: int = 8801,
) -> Dict[str, Dict[str, float]]:
    """Evaluates unseen random 8-byte codes inside unseen `val_mm` prose using seen key templates."""
    model.eval()
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    use_amp = device == "cuda" and amp_dtype == torch.bfloat16
    out: Dict[str, Dict[str, float]] = {}

    for h_len in haystacks:
        rng_py = random.Random(seed + h_len)
        rng_np = np.random.default_rng(seed + h_len)
        bw_l: List[float] = []
        bwo_l: List[float] = []
        bacc_l: List[float] = []
        tf_hits = 0
        gr_hits = 0
        max_start = max(1, len(val_mm) - h_len - 64)
        for _ in range(n_samples):
            t1, t2 = rng_py.sample(TRAIN_COPY_TEMPLATES, k=2)
            c1 = "".join(rng_py.choice(alphabet) for _ in range(8))
            c2 = "".join(rng_py.choice(alphabet) for _ in range(8))
            q_which = rng_py.choice((0, 1))
            q_tpl = t1 if q_which == 0 else t2
            q_code = c1 if q_which == 0 else c2

            pos = int(rng_np.integers(0, max_start))
            raw = val_mm[pos : pos + h_len].astype(np.int64).tolist()
            p1 = max(24, int(h_len * 0.10))
            p2 = max(p1 + 32, int(h_len * 0.25))

            s1_w = _enc_utf8(t1[0].format(val=c1))
            s2_w = _enc_utf8(t2[0].format(val=c2))
            s1_wo = _enc_utf8(t1[0].format(val="STANDARD")) if q_which == 0 else s1_w
            s2_wo = _enc_utf8(t2[0].format(val="STANDARD")) if q_which == 1 else s2_w
            q_hdr = _enc_utf8(q_tpl[1])
            t_ids = _enc_utf8(q_code)

            seq_w = raw[:p1] + s1_w + raw[p1:p2] + s2_w + raw[p2:] + q_hdr + t_ids
            seq_wo = raw[:p1] + s1_wo + raw[p1:p2] + s2_wo + raw[p2:] + q_hdr + t_ids

            bw, bwo, ba, tf_ex, gr_ex = _score_needle_pair(
                model, seq_w, seq_wo, t_ids, device, use_amp, amp_dtype, run_greedy=True
            )
            bw_l.append(bw)
            bwo_l.append(bwo)
            bacc_l.append(ba)
            tf_hits += int(tf_ex)
            gr_hits += int(gr_ex)
        mw = float(np.mean(bw_l))
        mwo = float(np.mean(bwo_l))
        out[f"L_{h_len}"] = {
            "haystack_bytes": int(h_len),
            "n_samples": int(n_samples),
            "needle_bpb_with": round(mw, 4),
            "needle_bpb_without": round(mwo, 4),
            "recall_gain_bpb": round(mwo - mw, 4),
            "byte_acc_pct": round(100.0 * float(np.mean(bacc_l)), 2),
            "tf_exact_pct": round(100.0 * tf_hits / max(1, n_samples), 2),
            "greedy_exact_pct": round(100.0 * gr_hits / max(1, n_samples), 2),
        }
        if device == "cuda":
            torch.cuda.empty_cache()
    return out


def run_probe_cell(
    name: str,
    model: nn.Module,
    cache_kb: float,
    train_mm: np.memmap,
    val_mm: np.memmap,
    device: str,
    steps: int,
    batch_size: int,
    micro_bs: int,
    seed: int,
    smoke: bool,
) -> Dict[str, object]:
    print(f"\n==================== [{name}] ====================", flush=True)
    n_params = int(sum(p.numel() for p in model.parameters()))
    print(f"[{name}] params={n_params:,} | cache_O1={cache_kb:.2f} KB | steps={steps}", flush=True)

    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
    base_lr = 2.0e-3
    min_lr = 1.5e-4
    opt = MuonAdamWHybrid(
        model,
        base_lr=base_lr,
        muon_lr_mult=5.5,
        weight_decay=0.02,
        device=device,
        woodbury_lr_mult=0.80,
        protect_crystal=False,
    )
    warm = min(30, max(1, steps // 15))
    rng_py = random.Random(seed + 4242)
    rng_np = np.random.default_rng(seed + 4242)
    is_calcite = isinstance(model, CalciteV2LM)
    cur_mbs = min(batch_size, micro_bs)
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        model.train()
        if is_calcite:
            model.set_crystal_schedule(step, steps)
        # 3-stage length curriculum: 128 -> 256 -> 512
        frac = step / float(max(1, steps))
        if smoke:
            cur_L = 128
        elif frac <= 0.34:
            cur_L = 128
        elif frac <= 0.67:
            cur_L = 256
        else:
            cur_L = 512

        lr_s = _cosine_lr(step, steps, warm, base_lr, min_lr)
        sc = lr_s / base_lr
        for pg in opt.param_groups:
            pg["lr"] = base_lr * float(pg.get("lr_mult", 1.0)) * sc

        xb_full, yb_full = sample_pure_copy_batch(train_mm, batch_size, cur_L, rng_py, rng_np, device)

        while True:
            opt.zero_grad(set_to_none=True)
            n_acc = max(1, (batch_size + cur_mbs - 1) // cur_mbs)
            ce_sum = 0.0
            acc_sum = 0.0
            try:
                for st in range(0, batch_size, cur_mbs):
                    xb = xb_full[st : st + cur_mbs]
                    yb = yb_full[st : st + cur_mbs]
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                        lg = model(xb)
                        lg_f = lg.float()
                        ce = F.cross_entropy(
                            lg_f.reshape(-1, lg_f.size(-1)),
                            yb.reshape(-1),
                            ignore_index=0,
                        )
                        aux = model.get_commitment_loss() if is_calcite else 0.0
                        loss = (ce + 0.03 * aux) / float(n_acc)
                    loss.backward()
                    ce_sum += float(ce.detach().item())
                    with torch.no_grad():
                        mask = yb != 0
                        if mask.any():
                            acc_sum += float((lg_f.argmax(dim=-1)[mask] == yb[mask]).float().mean().item())
                break
            except torch.cuda.OutOfMemoryError:
                if cur_mbs <= 2:
                    raise
                cur_mbs = max(2, cur_mbs // 2)
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step in (1, steps) or (step % 50 == 0):
            es_info = ""
            if is_calcite and model.epistemic_slots is not None:
                es_info = f" | wr={model.epistemic_slots.last_write_rate:.2f}"
            print(
                f"  [{name}] step {step:4d}/{steps} (L={cur_L:3d}) | ce={ce_sum/n_acc:.4f} | "
                f"byteAcc={100.0*acc_sum/n_acc:5.1f}%{es_info} | {(time.perf_counter()-t0)/60:.1f}m",
                flush=True,
            )

    train_s = time.perf_counter() - t0
    del opt
    if device == "cuda":
        torch.cuda.empty_cache()

    model.eval()
    if is_calcite:
        model.set_crystal_schedule(1, 1)
        model.set_snap_alpha(1.0)

    n_ev = 4 if smoke else 24
    seen_eval = eval_seen_tpl_unseen_code(
        model, val_mm, device=device, haystacks=(256,) if smoke else (512, 1500, 4096), n_samples=n_ev, seed=8801
    )
    orig_ndl = eval_heldout_passkey_single(
        model, val_mm, device=device, haystack_bytes=256 if smoke else 1500, n_samples=n_ev, seed=7777, run_greedy=True
    )
    ood_ladder = eval_heldout_multi_needle_ladder(
        model,
        val_mm,
        device=device,
        haystacks=(256,) if smoke else (256, 512, 1500, 4096, 8192),
        n_samples=n_ev,
        seed=9901,
    )

    telemetry: Dict[str, object] = {}
    if is_calcite and model.epistemic_slots is not None:
        es = model.epistemic_slots
        lw = F.softmax(es.lag_logits.detach().float(), dim=-1).cpu().tolist()
        telemetry = {
            "last_write_rate": round(float(es.last_write_rate), 4),
            "lex_weight": round(float(torch.sigmoid(es.lex_logit.detach()).item()), 4),
            "lag_weights": [[round(v, 4) for v in row] for row in lw],
            "read_scale": [round(float(v), 3) for v in es.read_scale.detach().cpu().tolist()],
            "null_logit": [round(float(v), 3) for v in es.null_logit.detach().cpu().tolist()],
            "write_bias": [round(float(v), 3) for v in es.write_proj.bias.detach().cpu().tolist()],
            "slots_gate_bias": round(float(model.epistemic_slots_gate.bias.detach().item()), 3),
        }

    print(f"  [{name} EVAL SUMMARY] ({train_s:.1f}s)", flush=True)
    for lk, lv in seen_eval.items():
        print(
            f"    SeenTpl {lk}: gain={lv['recall_gain_bpb']:+.4f} | byte={lv['byte_acc_pct']:5.1f}% | "
            f"tfEx={lv['tf_exact_pct']:5.1f}% | grEx={lv['greedy_exact_pct']:5.1f}%",
            flush=True,
        )
    print(
        f"    OrigOOD 1500B: gain={orig_ndl['recall_gain_bpb']:+.4f} | byte={orig_ndl['byte_acc_pct']:5.1f}% | "
        f"tfEx={orig_ndl['tf_exact_pct']:5.1f}% | grEx={orig_ndl['greedy_exact_pct']:5.1f}%",
        flush=True,
    )
    for lk, lv in ood_ladder.items():
        print(
            f"    OODMulti {lk}: gain={lv['recall_gain_bpb']:+.4f} | byte={lv['byte_acc_pct']:5.1f}% | "
            f"tfEx={lv['tf_exact_pct']:5.1f}% | grEx={lv['greedy_exact_pct']:5.1f}%",
            flush=True,
        )
    if telemetry:
        print(f"    Slots telemetry: {telemetry}", flush=True)

    return {
        "name": name,
        "params": n_params,
        "cache_kb_o1": round(cache_kb, 2),
        "train_seconds": round(train_s, 2),
        "seen_tpl_eval": seen_eval,
        "orig_ood_1500b": orig_ndl,
        "ood_multi_ladder": ood_ladder,
        "slots_telemetry": telemetry,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT_DIR / "data_cache")
    ap.add_argument("--out-dir", type=Path, default=ROOT_DIR / "benchmarks" / "phase_n_needle")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--micro-bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--smoke-local", action="store_true")
    args = ap.parse_args()

    device = "cuda" if has_usable_cuda() else "cpu"
    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram_gb < 12.0:
            torch.cuda.set_per_process_memory_fraction(0.75, 0)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_local:
        rng_f = np.random.default_rng(args.seed)
        fake = rng_f.integers(32 + 3, 126 + 3, size=60_000, dtype=np.uint16)
        train_mm = fake
        val_mm = fake
        steps, batch_size, micro_bs = 6, 8, 4
    else:
        train_bin, val_bin = ensure_phase_k_corpus(args.data_dir)
        train_mm = np.memmap(train_bin, dtype=np.uint16, mode="r")
        val_mm = np.memmap(val_bin, dtype=np.uint16, mode="r")
        steps, batch_size, micro_bs = int(args.steps), int(args.batch_size), int(args.micro_bs)

    router_path = ROOT_DIR / "benchmarks" / "phase_l_v2" / "frozen_utf8_router.pt"
    router_state = torch.load(router_path, map_location="cpu", weights_only=True)
    res_path = args.out_dir / ("smoke_copy_probe_5m.json" if args.smoke_local else "phase_n_copy_probe_5m.json")
    t_all = time.perf_counter()
    cells: Dict[str, object] = {}

    def _save() -> None:
        res_path.write_text(
            json.dumps(
                {
                    "meta": {
                        "phase": "Phase_N_Probe_A_Pure_Copy_Curriculum_5M",
                        "steps": steps,
                        "batch_size": batch_size,
                        "seed": args.seed,
                        "smoke": args.smoke_local,
                        "elapsed_s": round(time.perf_counter() - t_all, 1),
                    },
                    "cells": cells,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # 1. RoPE 5M
    set_all_seeds(args.seed)
    rope_cfg = BaselineConfig(
        vocab_size=259,
        d_model=256 if not args.smoke_local else 128,
        n_heads=4 if not args.smoke_local else 2,
        n_layers=6 if not args.smoke_local else 2,
        d_ff=800 if not args.smoke_local else 256,
        window_size=36,
    )
    rope_m = RoPETransformerLM(rope_cfg).to(device)
    rope_kb = float(rope_cfg.n_layers * 2 * 2048 * rope_cfg.d_model * 2) / 1024.0
    cells["P0_RoPE_5M"] = run_probe_cell(
        "P0_RoPE_5M", rope_m, rope_kb, train_mm, val_mm, device, steps, batch_size, micro_bs, args.seed, args.smoke_local
    )
    _save()
    del rope_m
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # 2 & 3. Calcite v3 (Woodbury only) vs Calcite v3 + InductionSlotMemory
    def _build_cfg(use_slots: bool) -> CalciteV2Config:
        cfg = build_calcite_v2_tier_config(
            "S1_5M",
            topology="hourglass",
            use_scaled_decoder=True,
            use_kalman_precond=False,
            use_latent_feedback=False,
            use_clean_gates=False,
            use_sym_byte_swa=True,
            byte_swa_window=24,
            byte_swa_dim=64,
            use_byte_engram=True,
            engram_slots=4096,
            engram_dim=32,
            use_epistemic_slots=use_slots,
            epistemic_slots_n=32,
            epistemic_slots_dim=64,
        )
        if args.smoke_local:
            cfg.d_model = 128
            cfg.n_heads = 2
            cfg.d_ff = 256
            cfg.byte_char_dim = 16
            cfg.byte_d_ff = 128
            cfg.epistemic_slots_n = 16
            cfg.epistemic_slots_dim = 32
        return cfg

    for cname, use_slots in [("P1_Calcite_v3_5M", False), ("P2_Calcite_v3_InductSlots", True)]:
        set_all_seeds(args.seed)
        cfg = _build_cfg(use_slots=use_slots)
        m = CalciteV2LM(cfg).to(device)
        load_frozen_router_into(m, router_state)
        kb = compute_v2_state_cache_kb(cfg, 32768)
        cells[cname] = run_probe_cell(
            cname, m, kb, train_mm, val_mm, device, steps, batch_size, micro_bs, args.seed, args.smoke_local
        )
        _save()
        del m
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n[Probe A Complete] Saved -> {res_path}", flush=True)


if __name__ == "__main__":
    main()
