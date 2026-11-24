"""
Phase M Stage B (`Calcite v3`) Single-Axis Ablation Ladder #2 on Top of `M3_sym_byte_swa`
(`scripts/run_phase_m_stage_b_5m.py`).

Tests every research-radar axis strictly one at a time on top of `M3_sym_byte_swa` (`Val=2.3313`)
plus equal-budget `RoPE_5M` control on the exact same `700`-step stream (`seed=2026`, `B=64, L=512`,
`cog_prob=0.20`, pre-frozen UTF-8 router):
  - `B0_RoPE_5M_matched`:   Equal-budget `RoPETransformerLM (5M)` control
  - `B1_M3_plus_M1`:        `M3` + `Sparse Routed Woodbury (4 banks, top-1)` (clean, no M2 conflict)
  - `B2_M3_plus_AttnRes`:   `M3` + `Kimi AttnRes` depth-wise softmax routing (`+0` RAM)
  - `B3_M3_plus_DeltaNet2`: `M3` + `NVIDIA Gated DeltaNet-2` decoupled erase/write/decay (`+0` RAM)
  - `B4_M3_plus_Canon`:     `M3` + `Allen-Zhu Canon` 4-tap convs in chunk `SWA` blocks (`+12 KB` RAM)
  - `B5_M3_plus_Engram`:    `M3` + pure-tensor hashed 2/3-byte `Engram` (`+24 B` dynamic RAM)
  - `B6_M3_plus_Muon`:      `M3` trained with `MuonAdamWHybrid` optimizer on 2D hidden matrices
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
    sample_memmap_batch,
    sync_if_cuda,
)
from scripts.colab_phase_l_v2_runner import (  # noqa: E402
    build_eval_bundles,
    build_matched_rope_for_tier,
    build_optimizer_for_model,
    ensure_phase_k_corpus,
    eval_long_context_stress,
    load_frozen_router_into,
    sample_cognitive_batch,
    set_all_seeds,
)
from scripts.muon_opt import MuonAdamWHybrid  # noqa: E402
from scripts.run_phase_m_ablations_5m import (  # noqa: E402
    eval_cognitive_fast,
    eval_zeroshot_needle_recall_bpb,
)
from simvq_split.tokenizer import ByteBPETokenizer  # noqa: E402


def train_stage_b_model(
    model: nn.Module,
    train_mm: np.memmap,
    tok: ByteBPETokenizer,
    tr_sents: List[str],
    device: str,
    total_steps: int,
    batch_size: int = 64,
    seq_len: int = 512,
    micro_bs: int = 16,
    base_lr: float = 1.8e-3,
    min_lr: float = 1.2e-4,
    weight_decay: float = 0.04,
    cog_prob: float = 0.20,
    seed: int = 2026,
    use_muon: bool = False,
    muon_lr_mult: float = 6.0,
    tag: str = "run",
) -> Dict[str, object]:
    set_all_seeds(seed)
    model = model.to(device)
    if use_muon:
        opt = MuonAdamWHybrid(
            model,
            base_lr=base_lr,
            muon_lr_mult=muon_lr_mult,
            weight_decay=weight_decay,
            device=device,
        )
    else:
        opt = build_optimizer_for_model(model, base_lr=base_lr, weight_decay=weight_decay, device=device)

    rng_np = np.random.default_rng(seed)
    rng_py = random.Random(seed + 17)
    mbs = int(micro_bs)
    accum = max(1, batch_size // mbs)
    is_calcite = isinstance(model, CalciteV2LM)
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    use_amp = device == "cuda" and amp_dtype == torch.bfloat16

    t0 = time.perf_counter()
    warmup = max(20, int(0.05 * total_steps))
    bytes_per_step = batch_size * seq_len

    for step in range(1, total_steps + 1):
        model.train()
        if is_calcite:
            model.set_crystal_schedule(step, total_steps)

        if step <= warmup:
            scale = float(step) / float(warmup)
        else:
            prog = float(step - warmup) / float(max(1, total_steps - warmup))
            scale = (min_lr / base_lr) + 0.5 * (1.0 - min_lr / base_lr) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = base_lr * float(pg.get("lr_mult", 1.0)) * scale

        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(accum):
            if cog_prob > 0.0 and rng_py.random() < cog_prob:
                xb, yb = sample_cognitive_batch(tok, tr_sents, mbs, seq_len, rng_py, device)
            else:
                xb, yb = sample_memmap_batch(train_mm, mbs, seq_len, rng_np, device)

            with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                logits = model(xb)
                ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1), ignore_index=-100)
                loss = ce + (model.get_commitment_loss() if is_calcite else 0.0)
            (loss / float(accum)).backward()
            step_loss += float(ce.item()) / float(accum)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % max(1, total_steps // 5) == 0 or step == total_steps:
            elapsed = time.perf_counter() - t0
            ms = 1000.0 * elapsed / step
            vram_str = ""
            if device == "cuda" and torch.cuda.is_available():
                alloc_mb = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
                res_mb = torch.cuda.memory_reserved() / (1024.0 * 1024.0)
                vram_str = f" | VRAM={alloc_mb:.0f}/{res_mb:.0f}MB"
            print(
                f"  [{tag}] step={step}/{total_steps} | ce={step_loss:.4f} | "
                f"{ms:.1f} ms/step | {elapsed:.1f}s{vram_str}",
                flush=True,
            )

    sync_if_cuda(device)
    total_s = time.perf_counter() - t0
    return {
        "total_steps": int(total_steps),
        "bytes_seen": int(total_steps * bytes_per_step),
        "train_seconds": round(total_s, 2),
        "ms_per_step": round(1000.0 * total_s / max(1, total_steps), 2),
        "optimizer": "muon_adamw" if use_muon else "adamw",
    }


def make_m3_base_cfg(extra_kw: Dict[str, object]) -> CalciteV2Config:
    base_kw: Dict[str, object] = {
        "topology": "hourglass",
        "use_scaled_decoder": True,
        "use_kalman_precond": False,
        "use_latent_feedback": False,
        "use_clean_gates": False,
        "window_size": 8,
        "crystal_w_chunks": 2,
        "woodbury_latch_span": 6,
        "use_sym_byte_swa": True,
        "byte_swa_window": 24,
        "byte_swa_dim": 64,
    }
    base_kw.update(extra_kw)
    return build_calcite_v2_tier_config("S1_5M", **base_kw)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase M Stage B: 5M single-axis ablations on M3_sym_byte_swa")
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--micro-bs", type=int, default=16)
    ap.add_argument("--cog-prob", type=float, default=0.20)
    ap.add_argument("--n-eval", type=int, default=80)
    ap.add_argument("--n-needle", type=int, default=24)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke-local", action="store_true")
    ap.add_argument("--out", type=str, default="benchmarks/phase_m_v3/stage_b_5m_ablations.json")
    args = ap.parse_args()

    device = "cuda" if has_usable_cuda() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.78, 0)
        print(f"[env] GPU: {torch.cuda.get_device_name(0)} | BF16={torch.cuda.is_bf16_supported()}", flush=True)

    train_bin, val_bin = ensure_phase_k_corpus(ROOT_DIR / "data_cache")
    train_mm = np.memmap(train_bin, dtype=np.uint16, mode="r")
    val_mm = np.memmap(val_bin, dtype=np.uint16, mode="r")

    router_path = ROOT_DIR / "benchmarks" / "phase_l_v2" / "frozen_utf8_router.pt"
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

    # (name, is_rope, extra_cfg_kw, use_muon)
    cells: List[Tuple[str, bool, Dict[str, object], bool]] = [
        ("B0_RoPE_5M_matched", True, {}, False),
        ("B1_M3_plus_M1", False, {"woodbury_sub_banks": 4, "woodbury_top_k_banks": 1}, False),
        ("B2_M3_plus_AttnRes", False, {"use_attnres": True}, False),
        ("B3_M3_plus_DeltaNet2", False, {"use_deltanet2_gates": True}, False),
        ("B4_M3_plus_Canon", False, {"use_canon_chunks": True}, False),
        ("B5_M3_plus_Engram", False, {"use_byte_engram": True, "engram_slots": 4096, "engram_dim": 32}, False),
        ("B6_M3_plus_Muon", False, {}, True),
    ]
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        cells = [c for c in cells if c[0] in want]

    out_path = ROOT_DIR / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.is_file() and not args.smoke_local:
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        results: Dict[str, object] = payload.get("cells", {})
    else:
        results = {}

    # Seed reference M3_sym_byte_swa from Stage 1 if present
    s1_path = ROOT_DIR / "benchmarks" / "phase_m_v3" / "stage1_5m_ablations.json"
    if s1_path.is_file() and "M3_sym_byte_swa_ref" not in results and not args.smoke_local:
        s1_data = json.loads(s1_path.read_text(encoding="utf-8"))
        if "M3_sym_byte_swa" in s1_data.get("cells", {}):
            results["M3_sym_byte_swa_ref"] = s1_data["cells"]["M3_sym_byte_swa"]

    t_all = time.perf_counter()
    for name, is_rope, kw, use_muon in cells:
        if name in results and not args.force and not args.smoke_local:
            print(f"[skip] {name} already verified in {out_path.name}", flush=True)
            continue

        print(f"\n==================== [{name}] ====================", flush=True)
        set_all_seeds(args.seed)
        if is_rope:
            model: nn.Module = build_matched_rope_for_tier("S1_5M", 5_382_253).to(device)
            n_params = sum(p.numel() for p in model.parameters())
            cache_kb = round(float(compute_rope_cache_kb(model.cfg, 2048)), 2)
            cfg_dict = asdict(model.cfg)
        else:
            cfg = make_m3_base_cfg(kw)
            model = CalciteV2LM(cfg).to(device)
            load_frozen_router_into(model, router_state)
            n_params = sum(p.numel() for p in model.parameters())
            cache_kb = round(compute_v2_state_cache_kb(cfg, 2048), 2)
            cfg_dict = asdict(cfg)

        print(f"[{name}] params={n_params:,} | cache_L2048={cache_kb:.2f} KB | muon={use_muon}", flush=True)
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
            cog_prob=args.cog_prob,
            seed=args.seed,
            use_muon=use_muon,
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
        cog = eval_cognitive_fast(
            model, tok, s5_suites, ov_suites, trap_suites, device=device, batch_size=16
        )

        res = {
            "name": name,
            "params": int(n_params),
            "cache_kb_L2048": cache_kb,
            "config": cfg_dict,
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
            f"Py={res['ow6_domains'].get('3_Python_Code')} | "
            f"L2048={long_ctx.get('L_2048', {}).get('bpb')} | "
            f"L8192={long_ctx.get('L_8192', {}).get('bpb')} | "
            f"NeedleGain={needle['recall_gain_bpb']:+.4f} | {train_info['ms_per_step']:.1f} ms/st",
            flush=True,
        )
        results[name] = res
        if not args.smoke_local:
            out_path.write_text(
                json.dumps(
                    {
                        "meta": {
                            "tier": "S1_5M",
                            "base_ref": "M3_sym_byte_swa (Val=2.3313, OW6=3.1883, L8192=2.2055)",
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
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n[STAGE B ALL DONE] elapsed={time.perf_counter() - t_all:.1f}s | out={out_path}", flush=True)


if __name__ == "__main__":
    main()
