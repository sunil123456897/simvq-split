"""
Phase M (`Calcite v3`) Fast Local 5M Single-Axis Ablations (`scripts/run_phase_m_ablations_5m.py`).

Tests four orthogonal architectural upgrades on `S1_5M` (local RTX 5060, ~4 min/cell)
against exact `Calcite v2` control (`M0_v2_base`) using the exact Phase L pre-frozen UTF-8 router
and zero-leakage Phase K corpus:
  - `M0_v2_base`:         Exact Calcite v2 control (`A1b` 2-scale byte-mixer + hourglass)
  - `M1_sparse_woodbury`: Sparse Routed Woodbury (`sub_banks=4, top_k=1`, `75%` zero-decay freeze)
  - `M2_epistemic_slots`: Epistemic L2 Exact Slot Cache (`32 x 64` = `16 KB` O(1), driven by `q^T P q`)
  - `M3_sym_byte_swa`:    Symmetric encoder mixer + 3rd-scale decoder + `6 KB` micro byte-SWA (`W=24`)
  - `M4_swa_drop`:        Per-sample SWA Window-Drop (`p=0.15`, zero extra params/RAM)

Evaluates every cell on:
  1. `Val BPB` (`L=512`), `OW6 BPB` (`0.00%` Mid-UTF8 cuts), `L=2048` & `L=8192` (`tail-head`)
  2. Zero-shot natural-text `Needle-in-a-Haystack` Passkey Copy BPB Gain across `~1,500` bytes
  3. Unified single-stage cognitive memory (`S_5`, `Overwrite In/Med/Long`, `Trap Long`, `SelF1_L`)
  4. `ms/step`, parameter count, and exact `O(1)` state cache KB.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import math
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
from experiments.scaling_phase_k.eval_suite import (  # noqa: E402
    eval_openweb_6domains_any_byte_lm,
    eval_val_memmap_bpb,
    has_usable_cuda,
    sample_memmap_batch,
    sync_if_cuda,
)
from run_simvq_split_benchmark import harmonic_mean  # noqa: E402
from scripts.colab_phase_l_v2_runner import (  # noqa: E402
    build_eval_bundles,
    ensure_phase_k_corpus,
    eval_long_context_stress,
    load_frozen_router_into,
    set_all_seeds,
    train_byte_lm,
)
from simvq_split.eval_utils import evaluate_suite_batched  # noqa: E402
from simvq_split.tokenizer import ByteBPETokenizer  # noqa: E402


@torch.no_grad()
def eval_zeroshot_needle_recall_bpb(
    model: nn.Module,
    val_mm: np.memmap,
    device: str,
    haystack_bytes: int = 1500,
    n_samples: int = 24,
    seed: int = 7777,
) -> Dict[str, float]:
    """
    Zero-shot natural-text Needle-in-a-Haystack Passkey Recall probe:
    Injects an unseen 8-char alphanumeric reference code near the start of `haystack_bytes`
    of unseen `val_mm` text, then queries the same code at the end.
    Measures `needle_bpb_with` vs `needle_bpb_without` (paired exact same haystack & suffix):
    `recall_gain_bpb = needle_bpb_without - needle_bpb_with > 0` proves true long-range copying.
    """
    model.eval()
    rng_py = random.Random(seed)
    rng_np = np.random.default_rng(seed)
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    use_amp = device == "cuda" and amp_dtype == torch.bfloat16

    def _enc(s: str) -> List[int]:
        return [b + 3 for b in s.encode("utf-8")]

    bpb_with_list: List[float] = []
    bpb_without_list: List[float] = []
    exact_hits = 0

    max_start = max(1, len(val_mm) - haystack_bytes - 64)
    for _ in range(n_samples):
        code = "".join(rng_py.choice(alphabet) for _ in range(8))
        pos = int(rng_np.integers(0, max_start))
        raw = val_mm[pos : pos + haystack_bytes].astype(np.int64).tolist()
        prefix = raw[:128]
        middle = raw[128:]

        needle_stmt = _enc(f"\nImportant reference code: {code}.\n")
        decoy_stmt = _enc("\nImportant reference note: standard.\n")
        query_hdr = _enc("\nRepeat the important reference code: ")
        target_ids = _enc(code)

        seq_with = prefix + needle_stmt + middle + query_hdr + target_ids
        seq_without = prefix + decoy_stmt + middle + query_hdr + target_ids

        def _score(seq: List[int]) -> Tuple[float, bool]:
            x = torch.tensor([seq[:-1]], device=device, dtype=torch.long)
            y = torch.tensor([seq[1:]], device=device, dtype=torch.long)
            with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                lg = model(x)
            n_t = len(target_ids)
            lg_t = lg[:, -n_t:, :].float()
            y_t = y[:, -n_t:]
            bpb = float(
                F.cross_entropy(lg_t.reshape(-1, lg_t.size(-1)), y_t.reshape(-1)).item()
                / math.log(2.0)
            )
            exact = bool((lg_t.argmax(dim=-1) == y_t).all().item())
            return bpb, exact

        bw, hit = _score(seq_with)
        bwo, _ = _score(seq_without)
        bpb_with_list.append(bw)
        bpb_without_list.append(bwo)
        if hit:
            exact_hits += 1

    mean_w = float(np.mean(bpb_with_list))
    mean_wo = float(np.mean(bpb_without_list))
    return {
        "haystack_bytes": int(haystack_bytes),
        "n_samples": int(n_samples),
        "needle_bpb_with": round(mean_w, 4),
        "needle_bpb_without": round(mean_wo, 4),
        "recall_gain_bpb": round(mean_wo - mean_w, 4),
        "exact_match_pct": round(100.0 * exact_hits / max(1, n_samples), 2),
    }


@torch.no_grad()
def eval_cognitive_fast(
    model: nn.Module,
    tok: ByteBPETokenizer,
    s5_suites: Dict[str, list],
    ov_suites: Dict[str, list],
    trap_suites: Dict[str, list],
    device: str,
    batch_size: int = 16,
) -> Dict[str, float]:
    model.eval()
    def _acc(suite: list) -> float:
        ex_pct, _, _, _ = evaluate_suite_batched(
            model, suite, pad_id=tok.PAD_ID, batch_size=batch_size, device=device
        )
        return float(ex_pct)

    s5_k2 = _acc(s5_suites["k2"])
    s5_k4 = _acc(s5_suites["k4"])
    ov_in = _acc(ov_suites["in_dist"])
    ov_med = _acc(ov_suites["ood_med"])
    ov_long = _acc(ov_suites["ood_long"])
    tr_long = _acc(trap_suites["ood_long"])
    return {
        "s5_k2_pct": round(s5_k2, 2),
        "s5_k4_pct": round(s5_k4, 2),
        "s5_k2_k4_sum": round(s5_k2 + s5_k4, 2),
        "ov_in_pct": round(ov_in, 2),
        "ov_med_pct": round(ov_med, 2),
        "ov_long_pct": round(ov_long, 2),
        "trap_long_pct": round(tr_long, 2),
        "sel_f1_long_pct": round(float(harmonic_mean(ov_long, tr_long)), 2),
    }


def make_phase_m_cfg(name: str, extra_kw: Dict[str, object]) -> CalciteV2Config:
    cfg = build_calcite_v2_tier_config(
        "S1_5M",
        topology="hourglass",
        use_scaled_decoder=True,
        use_kalman_precond=False,
        use_latent_feedback=False,
        use_clean_gates=False,
        window_size=8,
        crystal_w_chunks=2,
        woodbury_latch_span=6,
        **extra_kw,
    )
    return cfg


def evaluate_cell(
    name: str,
    cfg: CalciteV2Config,
    router_state: Dict[str, torch.Tensor],
    train_mm: np.memmap,
    val_mm: np.memmap,
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
    cog_prob: float,
    seed: int,
    n_val_batches: int,
    n_needle: int,
) -> Dict[str, object]:
    print(f"\n==================== [{name}] ====================", flush=True)
    set_all_seeds(seed)
    model = CalciteV2LM(cfg).to(device)
    load_frozen_router_into(model, router_state)
    n_params = sum(p.numel() for p in model.parameters())
    cache_kb = round(compute_v2_state_cache_kb(cfg, 2048), 2)
    print(f"[{name}] params={n_params:,} | cache_O1={cache_kb:.2f} KB | steps={steps}", flush=True)

    train_info = train_byte_lm(
        model=model,
        train_mm=train_mm,
        val_mm=val_mm,
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
        cog_prob=cog_prob,
        seed=seed,
        use_compile=False,
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
    cog = eval_cognitive_fast(
        model, tok, s5_suites, ov_suites, trap_suites, device=device, batch_size=16
    )

    res = {
        "name": name,
        "params": int(n_params),
        "cache_kb_o1": cache_kb,
        "config": asdict(cfg),
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
        f"L2048={long_ctx.get('L_2048', {}).get('bpb')} | "
        f"NeedleGain={needle['recall_gain_bpb']:+.4f} (EM={needle['exact_match_pct']:.1f}%) | "
        f"S5={cog['s5_k2_k4_sum']:.1f}% | OV_L={cog['ov_long_pct']:.1f}% | "
        f"SelF1_L={cog['sel_f1_long_pct']:.1f}% | {train_info['ms_per_step']:.1f} ms/st",
        flush=True,
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase M (Calcite v3) 5M local single-axis ablations")
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--micro-bs", type=int, default=16)
    ap.add_argument("--cog-prob", type=float, default=0.20)
    ap.add_argument("--n-eval", type=int, default=80)
    ap.add_argument("--n-needle", type=int, default=24)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--only", type=str, default="", help="Comma-separated subset of cell names")
    ap.add_argument("--run-combo", action="store_true", help="Auto-run combo of winning axes vs M0")
    ap.add_argument("--smoke-local", action="store_true", help="2-step smoke tracer over all 5 cells")
    ap.add_argument("--out", type=str, default="benchmarks/phase_m_v3/stage1_5m_ablations.json")
    args = ap.parse_args()

    device = "cuda" if has_usable_cuda() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.85, 0)
        print(f"[env] GPU: {torch.cuda.get_device_name(0)} | BF16={torch.cuda.is_bf16_supported()}", flush=True)
    else:
        print("[env] Running on CPU", flush=True)

    train_bin, val_bin = ensure_phase_k_corpus(ROOT_DIR / "data_cache")
    train_mm = np.memmap(train_bin, dtype=np.uint16, mode="r")
    val_mm = np.memmap(val_bin, dtype=np.uint16, mode="r")

    router_path = ROOT_DIR / "benchmarks" / "phase_l_v2" / "frozen_utf8_router.pt"
    assert router_path.is_file(), f"Missing frozen UTF-8 router: {router_path}"
    router_state = torch.load(router_path, map_location="cpu", weights_only=True)

    tok = ByteBPETokenizer()
    n_eval = 4 if args.smoke_local else args.n_eval
    tr_sents, eval_wins, s5_suites, ov_suites, trap_suites = build_eval_bundles(
        ROOT_DIR, tok, n_eval=n_eval, seq_len=512
    )

    steps = 2 if args.smoke_local else args.steps
    batch_size = 4 if args.smoke_local else args.batch_size
    micro_bs = 4 if args.smoke_local else args.micro_bs
    n_val_batches = 2 if args.smoke_local else 24
    n_needle = 2 if args.smoke_local else args.n_needle

    cells: List[Tuple[str, Dict[str, object]]] = [
        ("M0_v2_base", {}),
        ("M1_sparse_woodbury", {"woodbury_sub_banks": 4, "woodbury_top_k_banks": 1}),
        ("M2_epistemic_slots", {"use_epistemic_slots": True, "epistemic_slots_n": 32, "epistemic_slots_dim": 64}),
        ("M3_sym_byte_swa", {"use_sym_byte_swa": True, "byte_swa_window": 24, "byte_swa_dim": 64}),
        ("M4_swa_drop", {"swa_drop_prob": 0.15}),
    ]
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        cells = [(n, kw) for n, kw in cells if n in want]

    out_path = ROOT_DIR / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.is_file() and not args.smoke_local:
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        results: Dict[str, object] = payload.get("cells", {})
    else:
        results = {}

    t_all = time.perf_counter()
    for name, kw in cells:
        cfg = make_phase_m_cfg(name, kw)
        res = evaluate_cell(
            name=name,
            cfg=cfg,
            router_state=router_state,
            train_mm=train_mm,
            val_mm=val_mm,
            tok=tok,
            tr_sents=tr_sents,
            eval_wins=eval_wins,
            s5_suites=s5_suites,
            ov_suites=ov_suites,
            trap_suites=trap_suites,
            device=device,
            steps=steps,
            batch_size=batch_size,
            micro_bs=micro_bs,
            cog_prob=args.cog_prob,
            seed=args.seed,
            n_val_batches=n_val_batches,
            n_needle=n_needle,
        )
        results[name] = res
        if not args.smoke_local:
            out_path.write_text(
                json.dumps(
                    {
                        "meta": {
                            "tier": "S1_5M",
                            "steps": steps,
                            "batch_size": batch_size,
                            "cog_prob": args.cog_prob,
                            "seed": args.seed,
                            "noise_floor_2sigma_bpb": 0.0057,
                            "elapsed_s": round(time.perf_counter() - t_all, 1),
                        },
                        "cells": results,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

    # Optional auto-combo of axes that clearly won vs M0_v2_base
    if args.run_combo and not args.smoke_local and "M0_v2_base" in results:
        base = results["M0_v2_base"]
        base_val = float(base["val_bpb"])
        base_ov = float(base["cognitive"]["ov_long_pct"])
        base_s5 = float(base["cognitive"]["s5_k2_k4_sum"])
        base_nd = float(base["needle_1500b"]["recall_gain_bpb"])

        combo_kw: Dict[str, object] = {}
        winners: List[str] = []
        axis_map = {
            "M1_sparse_woodbury": {"woodbury_sub_banks": 4, "woodbury_top_k_banks": 1},
            "M2_epistemic_slots": {"use_epistemic_slots": True, "epistemic_slots_n": 32, "epistemic_slots_dim": 64},
            "M3_sym_byte_swa": {"use_sym_byte_swa": True, "byte_swa_window": 24, "byte_swa_dim": 64},
            "M4_swa_drop": {"swa_drop_prob": 0.15},
        }
        for ax_name, ax_kw in axis_map.items():
            if ax_name not in results:
                continue
            r = results[ax_name]
            val_gain = base_val - float(r["val_bpb"])
            ov_gain = float(r["cognitive"]["ov_long_pct"]) - base_ov
            s5_gain = float(r["cognitive"]["s5_k2_k4_sum"]) - base_s5
            nd_gain = float(r["needle_1500b"]["recall_gain_bpb"]) - base_nd
            # Accept if beats 2*sigma on Val BPB OR clearly boosts memory/needle without hurting Val BPB > 0.015
            if (val_gain > 0.0057) or ((ov_gain >= 5.0 or s5_gain >= 8.0 or nd_gain >= 0.05) and val_gain >= -0.015):
                combo_kw.update(ax_kw)
                winners.append(ax_name)

        if len(winners) >= 2:
            print(f"\n[combo] Combining winning axes: {winners}", flush=True)
            cfg_c = make_phase_m_cfg("M_combo_winners", combo_kw)
            res_c = evaluate_cell(
                name="M_combo_winners",
                cfg=cfg_c,
                router_state=router_state,
                train_mm=train_mm,
                val_mm=val_mm,
                tok=tok,
                tr_sents=tr_sents,
                eval_wins=eval_wins,
                s5_suites=s5_suites,
                ov_suites=ov_suites,
                trap_suites=trap_suites,
                device=device,
                steps=steps,
                batch_size=batch_size,
                micro_bs=micro_bs,
                cog_prob=args.cog_prob,
                seed=args.seed,
                n_val_batches=n_val_batches,
                n_needle=n_needle,
            )
            res_c["combined_axes"] = winners
            results["M_combo_winners"] = res_c
            out_path.write_text(
                json.dumps(
                    {
                        "meta": {
                            "tier": "S1_5M",
                            "steps": steps,
                            "batch_size": batch_size,
                            "cog_prob": args.cog_prob,
                            "seed": args.seed,
                            "noise_floor_2sigma_bpb": 0.0057,
                            "winners": winners,
                            "elapsed_s": round(time.perf_counter() - t_all, 1),
                        },
                        "cells": results,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        else:
            print(f"\n[combo] Fewer than 2 winning axes ({winners}); skipping combo run.", flush=True)

    print(f"\n[ALL DONE] total_elapsed={time.perf_counter() - t_all:.1f}s | out={out_path}", flush=True)


if __name__ == "__main__":
    main()
