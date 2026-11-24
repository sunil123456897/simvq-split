"""
Phase N Stage 2: `Calcite v3.1` (`Calcite v3` + `ByteInductionCache`, `S=512`, `+100 KB` O(1))
Full Language-Modeling, `L=32,768` Verbatim Needle, `2-bit LUT`, and Live Streaming Verification on `S1_5M`
(`scripts/run_phase_n_v31_lm_check_5m.py`).

Verifies every single promotion gate BEFORE launching the `25M` flagship closure:
1. **Zero LM Regression (`700` steps `MuonAdamW` vs Phase M `C1_Calcite_v3_Muon` `Val 2.1524`, `OW6 2.9619`):**
   - `90%` pure OpenWeb UTF-8 prose + `10%` dense multi-format MQAR verbatim copy (`0.85 * ce_p + 0.15 * ce_c`).
   - Strictly disjoint train vs eval key names & templates.
2. **OOD Verbatim Needle Ladder (`L = 512 -> 1500 -> 4096 -> 8192 -> 32,768 B`, up to `64x` OOD!):**
   - Untouched Phase M `"Important reference code:"` (`1500 B` & `8192 B` & `32,768 B`).
   - Unseen 2-Needle `EVAL_KEYS` (`Cipher seed`, `Parcel tag`, `Beacon id`, `Ledger pin`).
   - Unseen **Overwrite Needle** (key written TWICE with two different random 8-byte values; must return the 2nd!).
3. **`4-Tap 2-bit LUT` Quantization + Disk Roundtrip + Live `CalciteStreamEngine` Greedy Needle Test:**
   - Packs `Calcite v3.1` to `FastPackedLUT2BitLinear`, checks `Val BPB`, `L=32K` prose, `L=1500/8192/32768`
     Needle exact match IN 2-BIT LUT, and runs live `CalciteStreamEngine` byte-by-byte generation on a `1,500 B`
     unseen passkey to prove strict `O(1)` streaming verbatim recall!
"""

from __future__ import annotations

import argparse
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
    CalciteStreamEngine,
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
)
from kernels.fast_calcite_scans import clear_fast_qat_cache  # noqa: E402
from kernels.fast_packed_lut import pack_fast_model_2bit_lut  # noqa: E402
from run_phase_e_fast_utf8_benchmark import build_disjoint_openweb_corpora  # noqa: E402
from scripts.colab_phase_k_scaling_runner import (  # noqa: E402
    _attach_4tap_hooks,
    _cosine_lr,
    _make_adamw,
    _qat_micro_step,
)
from scripts.colab_phase_l_v2_runner import (  # noqa: E402
    benchmark_stream_engine_live,
    ensure_phase_k_corpus,
    eval_long_context_stress,
    load_frozen_router_into,
    set_all_seeds,
)
from scripts.muon_opt import MuonAdamWHybrid  # noqa: E402
from scripts.run_phase_m_stage_c_v3_5m import _split_v3_qat_4groups  # noqa: E402
from scripts.run_phase_n_mqar_probe_5m import (  # noqa: E402
    ALPHABET,
    EVAL_KEYS,
    TRAIN_KEYS,
    _enc,
    eval_ood_verbatim_ladder,
)
from simvq_split.memory import compute_2bit_lut_footprint, quantize_ternary_weights_ls  # noqa: E402
from simvq_split.tasks import load_gsm8k_corpora  # noqa: E402
from simvq_split.tokenizer import ByteBPETokenizer  # noqa: E402

# Multi-format punctuation templates for training (zero overlap with EVAL_KEYS or "Important reference code")
TRAIN_FMT_PAIRS: Tuple[Tuple[str, str], ...] = (
    (" [{k}: {v}] ", " {k}: "),
    ("\n{k}: {v}.\n", "\n{k}: "),
    ("\n{k} = {v};\n", "\n{k} = "),
    ("\n{k} -> {v}\n", "\n{k} -> "),
)


def sample_multiformat_mqar_row(
    train_mm: np.memmap,
    seq_len: int,
    rng_py: random.Random,
    rng_np: np.random.Generator,
) -> Tuple[List[int], List[int], List[bool]]:
    """Multi-format 3–4 pair verbatim copy row (with 25% in-stream overwrite) inside `train_mm` prose."""
    n_pairs = rng_py.choice((3, 4))
    val_len = rng_py.choice((6, 8))
    keys = rng_py.sample(TRAIN_KEYS, k=n_pairs)
    fmt_st, fmt_q = rng_py.choice(TRAIN_FMT_PAIRS)

    pairs: List[Tuple[str, str]] = []
    for k in keys:
        k_str = f"{k} {''.join(rng_py.choice(ALPHABET[:16]) for _ in range(3))}" if rng_py.random() < 0.5 else k
        val = "".join(rng_py.choice(ALPHABET) for _ in range(val_len))
        pairs.append((k_str, val))

    stmts = [_enc(fmt_st.format(k=k, v=v)) for k, v in pairs]
    # 25% overwrite of one key in-stream so overwrite + verbatim recall reinforce each other
    if rng_py.random() < 0.25:
        ov_i = rng_py.randrange(n_pairs)
        new_v = "".join(rng_py.choice(ALPHABET) for _ in range(val_len))
        pairs[ov_i] = (pairs[ov_i][0], new_v)
        stmts.append(_enc(fmt_st.format(k=pairs[ov_i][0], v=new_v)))

    q_order = list(pairs)
    rng_py.shuffle(q_order)
    q_blocks = [(_enc(fmt_q.format(k=k)), _enc(v)) for k, v in q_order]

    fixed_len = sum(len(s) for s in stmts) + sum(len(qh) + len(qv) for qh, qv in q_blocks)
    prose_budget = max(16, seq_len + 1 - fixed_len)
    max_st = max(1, len(train_mm) - prose_budget - 16)
    pos = int(rng_np.integers(0, max_st))
    for _ in range(3):
        tv = int(train_mm[pos])
        if tv >= 3 and ((tv - 3) & 0xC0) == 0x80:
            pos += 1
        else:
            break
    prose = train_mm[pos : pos + prose_budget].astype(np.int64).tolist()

    n_st = len(stmts)
    hi = max(n_st + 2, int(prose_budget * 0.55))
    cuts = [0] + sorted(
        rng_py.sample(range(1, hi), k=n_st)
        if hi > n_st + 1
        else [int(hi * (i + 1) / (n_st + 1)) for i in range(n_st)]
    ) + [prose_budget]

    seq: List[int] = []
    is_target: List[bool] = []
    for i, st in enumerate(stmts):
        seg = prose[cuts[i] : cuts[i + 1]]
        seq.extend(seg)
        is_target.extend([False] * len(seg))
        seq.extend(st)
        is_target.extend([False] * len(st))
    tail_seg = prose[cuts[-2] : cuts[-1]]
    seq.extend(tail_seg)
    is_target.extend([False] * len(tail_seg))

    for qh, qv in q_blocks:
        seq.extend(qh)
        is_target.extend([False] * len(qh))
        seq.extend(qv)
        is_target.extend([True] * len(qv))

    if len(seq) > seq_len + 1:
        q_start = len(seq) - sum(len(qh) + len(qv) for qh, qv in q_blocks)
        excess = len(seq) - (seq_len + 1)
        t0 = max(0, q_start - excess)
        seq = seq[:t0] + seq[q_start:]
        is_target = is_target[:t0] + is_target[q_start:]

    x_row = [0] * seq_len
    y_row = [0] * seq_len
    m_row = [False] * seq_len
    L_eff = min(seq_len, len(seq) - 1)
    x_row[:L_eff] = seq[:L_eff]
    y_row[:L_eff] = seq[1 : L_eff + 1]
    m_row[:L_eff] = is_target[1 : L_eff + 1]
    return x_row, y_row, m_row


def sample_lm_plus_copy_batch(
    train_mm: np.memmap,
    batch_size: int,
    seq_len: int,
    n_copy: int,
    rng_py: random.Random,
    rng_np: np.random.Generator,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_copy = max(0, min(batch_size - 1, int(n_copy)))
    n_prose = batch_size - n_copy
    xp, yp = sample_memmap_batch(train_mm, n_prose, seq_len, rng_np, device)
    mp = torch.zeros_like(yp, dtype=torch.bool)
    if n_copy == 0:
        return xp, yp, mp
    xs, ys, ms = [], [], []
    for _ in range(n_copy):
        xr, yr, mr = sample_multiformat_mqar_row(train_mm, seq_len, rng_py, rng_np)
        xs.append(xr)
        ys.append(yr)
        ms.append(mr)
    xc = torch.tensor(xs, dtype=torch.long, device=device)
    yc = torch.tensor(ys, dtype=torch.long, device=device)
    mc = torch.tensor(ms, dtype=torch.bool, device=device)

    x_all = torch.cat([xp, xc], dim=0)
    y_all = torch.cat([yp, yc], dim=0)
    m_all = torch.cat([mp, mc], dim=0)
    # Interleave so every microbatch gets copy rows
    stride = max(1, batch_size // n_copy)
    order: List[int] = []
    pi, ci = 0, n_prose
    for idx in range(batch_size):
        if (idx % stride == stride - 1) and ci < batch_size:
            order.append(ci)
            ci += 1
        elif pi < n_prose:
            order.append(pi)
            pi += 1
        else:
            order.append(ci)
            ci += 1
    perm = torch.tensor(order, dtype=torch.long, device=device)
    return x_all[perm], y_all[perm], m_all[perm]


@torch.no_grad()
def eval_overwrite_needle_ood(
    model: nn.Module,
    val_mm: np.memmap,
    device: str,
    haystack_bytes: int = 1500,
    n_samples: int = 16,
    seed: int = 7331,
) -> Dict[str, float]:
    """Unseen key written TWICE with different random 8-byte codes; verifies exact overwrite to 2nd value."""
    model.eval()
    rng_py = random.Random(seed)
    rng_np = np.random.default_rng(seed)
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    use_amp = device == "cuda" and amp_dtype == torch.bfloat16
    hits_new = 0
    hits_old = 0
    bacc_l: List[float] = []
    max_st = max(1, len(val_mm) - haystack_bytes - 64)
    for _ in range(n_samples):
        k = rng_py.choice(EVAL_KEYS[:4])
        v_old = "".join(rng_py.choice(ALPHABET) for _ in range(8))
        v_new = "".join(rng_py.choice(ALPHABET) for _ in range(8))
        pos = int(rng_np.integers(0, max_st))
        raw = val_mm[pos : pos + haystack_bytes].astype(np.int64).tolist()
        p1, p2 = int(haystack_bytes * 0.15), int(haystack_bytes * 0.50)
        seq = (
            raw[:p1]
            + _enc(f"\n{k}: {v_old}.\n")
            + raw[p1:p2]
            + _enc(f"\n{k}: {v_new}.\n")
            + raw[p2:]
            + _enc(f"\n{k}: ")
            + _enc(v_new)
        )
        t_new = _enc(v_new)
        t_old = _enc(v_old)
        n_t = len(t_new)
        x = torch.tensor([seq[:-1]], device=device, dtype=torch.long)
        y = torch.tensor([seq[1:]], device=device, dtype=torch.long)
        with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            lg = model(x)
        pr = lg[:, -n_t:, :].float().argmax(dim=-1)
        y_t = y[:, -n_t:]
        bacc_l.append(float((pr == y_t).float().mean().item()))
        hits_new += int(bool((pr == y_t).all().item()))
        hits_old += int(pr[0].tolist() == t_old)
    return {
        "haystack_bytes": int(haystack_bytes),
        "n_samples": int(n_samples),
        "new_byte_acc_pct": round(100.0 * float(np.mean(bacc_l)), 2),
        "new_exact_pct": round(100.0 * hits_new / max(1, n_samples), 2),
        "stale_old_exact_pct": round(100.0 * hits_old / max(1, n_samples), 2),
    }


@torch.no_grad()
def eval_live_stream_needle(
    model: CalciteV2LM,
    val_mm: np.memmap,
    device: str,
    haystack_bytes: int = 1500,
    n_samples: int = 4,
    seed: int = 7777,
) -> Dict[str, object]:
    """Runs live O(1) `CalciteStreamEngine` byte-by-byte prefill + greedy generation on unseen 1500B needles."""
    model.eval()
    eng = CalciteStreamEngine(model)
    rng_py = random.Random(seed)
    rng_np = np.random.default_rng(seed)
    hits = 0
    ram_before = 0
    ram_after = 0
    sample_ex: List[Dict[str, str]] = []
    max_st = max(1, len(val_mm) - haystack_bytes - 64)
    for i in range(n_samples):
        code = "".join(rng_py.choice(ALPHABET) for _ in range(8))
        pos = int(rng_np.integers(0, max_st))
        raw = val_mm[pos : pos + haystack_bytes].astype(np.int64).tolist()
        prompt = (
            raw[:128]
            + _enc(f"\nImportant reference code: {code}.\n")
            + raw[128:]
            + _enc("\nRepeat the important reference code: ")
        )
        p_t = torch.tensor([prompt], device=device, dtype=torch.long)
        st0 = eng.init_state(batch_size=1, device=torch.device(device), dtype=model.tok_emb.weight.dtype)
        ram_before = st0.memory_bytes()
        out_ids, st = eng.generate(p_t, max_new_bytes=8, temperature=0.0)
        ram_after = st.memory_bytes()
        gen_bytes = [max(0, min(255, int(v) - 3)) for v in out_ids[0, -8:].tolist()]
        gen_str = bytes(gen_bytes).decode("utf-8", errors="replace")
        ok = gen_str == code
        hits += int(ok)
        if i < 3:
            sample_ex.append({"target": code, "generated": gen_str, "exact": str(ok)})
    return {
        "haystack_bytes": int(haystack_bytes),
        "n_samples": int(n_samples),
        "stream_greedy_exact_pct": round(100.0 * hits / max(1, n_samples), 2),
        "state_kb": round(ram_after / 1024.0, 2),
        "strict_o1_ram": ram_before == ram_after,
        "examples": sample_ex,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT_DIR / "data_cache")
    ap.add_argument("--out-dir", type=Path, default=ROOT_DIR / "benchmarks" / "phase_n_needle")
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--qat-steps", type=int, default=250)
    ap.add_argument("--slots", type=int, default=512)
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
    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tok = ByteBPETokenizer(merges=[])
    tr_docs, te_docs, _, _ = load_gsm8k_corpora(ROOT_DIR)
    seq_len = 128 if args.smoke_local else 512
    _, eval_wins, _, _ = build_disjoint_openweb_corpora(ROOT_DIR, tok, tr_docs, te_docs, seq_len=seq_len)

    if args.smoke_local:
        rng_f = np.random.default_rng(args.seed)
        fake = rng_f.integers(32 + 3, 126 + 3, size=60_000, dtype=np.uint16)
        train_mm = val_mm = fake
        steps, qat_steps, batch_size, micro_bs, n_slots = 6, 4, 8, 4, 128
    else:
        train_bin, val_bin = ensure_phase_k_corpus(args.data_dir)
        train_mm = np.memmap(train_bin, dtype=np.uint16, mode="r")
        val_mm = np.memmap(val_bin, dtype=np.uint16, mode="r")
        steps, qat_steps, batch_size, micro_bs, n_slots = (
            int(args.steps),
            int(args.qat_steps),
            int(args.batch_size),
            int(args.micro_bs),
            int(args.slots),
        )

    router_state = torch.load(
        ROOT_DIR / "benchmarks" / "phase_l_v2" / "frozen_utf8_router.pt", map_location="cpu", weights_only=True
    )

    set_all_seeds(args.seed)
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
        use_byte_induct_cache=True,
        byte_induct_slots=n_slots,
        byte_induct_dim=16,
    )
    if args.smoke_local:
        cfg.d_model, cfg.n_heads, cfg.d_ff = 128, 2, 256
        cfg.byte_char_dim, cfg.byte_d_ff = 16, 128

    model = CalciteV2LM(cfg).to(device)
    load_frozen_router_into(model, router_state)
    n_params = int(sum(p.numel() for p in model.parameters()))
    cache_kb = round(compute_v2_state_cache_kb(cfg, 32768), 2)
    print(
        f"\n=== [Calcite v3.1 S1_5M Verification] params={n_params:,} | O(1)_cache={cache_kb:.2f} KB | "
        f"slots=2x{n_slots} | steps={steps} + qat={qat_steps} ===",
        flush=True,
    )

    # 1. FP Training (90% OpenWeb UTF-8 LM + 10% dense multi-format MQAR)
    base_lr, min_lr = 1.8e-3, 1.2e-4
    opt = MuonAdamWHybrid(
        model,
        base_lr=base_lr,
        muon_lr_mult=6.0,
        weight_decay=0.04,
        device=device,
        woodbury_lr_mult=0.45,
        protect_crystal=False,
    )
    warm = min(40, max(1, steps // 12))
    rng_py = random.Random(args.seed + 3141)
    rng_np = np.random.default_rng(args.seed + 3141)
    n_copy_rows = max(1, int(round(batch_size * 0.10)))
    cur_mbs = min(batch_size, micro_bs)
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        model.train()
        model.set_crystal_schedule(step, steps)
        lr_s = _cosine_lr(step, steps, warm, base_lr, min_lr)
        sc = lr_s / base_lr
        for pg in opt.param_groups:
            pg["lr"] = base_lr * float(pg.get("lr_mult", 1.0)) * sc

        xb_f, yb_f, mb_f = sample_lm_plus_copy_batch(
            train_mm, batch_size, seq_len, n_copy_rows, rng_py, rng_np, device
        )
        while True:
            opt.zero_grad(set_to_none=True)
            n_acc = max(1, (batch_size + cur_mbs - 1) // cur_mbs)
            ce_p_s, ce_c_s, acc_c_s, n_c_m = 0.0, 0.0, 0.0, 0
            try:
                for st in range(0, batch_size, cur_mbs):
                    xb = xb_f[st : st + cur_mbs]
                    yb = yb_f[st : st + cur_mbs]
                    mb = mb_f[st : st + cur_mbs]
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                        lg = model(xb)
                        lg_f = lg.float()
                        tok_ce = F.cross_entropy(
                            lg_f.reshape(-1, lg_f.size(-1)),
                            yb.reshape(-1),
                            ignore_index=0,
                            reduction="none",
                        ).view_as(yb)
                        valid = yb != 0
                        m_copy = mb & valid
                        m_pref = (~mb) & valid
                        ce_p = tok_ce[m_pref].mean() if m_pref.any() else tok_ce.mean()
                        has_c = bool(m_copy.any().item())
                        ce_c = tok_ce[m_copy].mean() if has_c else torch.zeros((), device=device)
                        ce = (0.85 * ce_p + 0.15 * ce_c) if has_c else ce_p
                        aux = model.get_commitment_loss()
                        loss = (ce + 0.04 * aux) / float(n_acc)
                    loss.backward()
                    ce_p_s += float(ce_p.detach().item())
                    if has_c:
                        ce_c_s += float(ce_c.detach().item())
                        with torch.no_grad():
                            acc_c_s += float((lg_f.argmax(dim=-1)[m_copy] == yb[m_copy]).float().mean().item())
                        n_c_m += 1
                break
            except torch.cuda.OutOfMemoryError:
                if cur_mbs <= 2:
                    raise
                cur_mbs = max(2, cur_mbs // 2)
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step in (1, steps) or (step % 100 == 0):
            bwr = model.byte_chunker.byte_induct_cache.last_write_rate if model.byte_chunker.byte_induct_cache else 0.0
            print(
                f"  [FP] step {step:4d}/{steps} | ce_p={ce_p_s/n_acc:.4f} | "
                f"ce_c={ce_c_s/max(1, n_c_m):.4f} | copyAcc={100.0*acc_c_s/max(1, n_c_m):5.1f}% | "
                f"bwr={bwr:.2f} | {(time.perf_counter()-t0)/60:.1f}m",
                flush=True,
            )

    fp_train_s = round(time.perf_counter() - t0, 2)
    del opt
    if device == "cuda":
        torch.cuda.empty_cache()

    model.eval()
    model.set_crystal_schedule(1, 1)
    model.set_snap_alpha(1.0)

    def _eval_bundle(m_eval: CalciteV2LM, tag: str) -> Dict[str, object]:
        val_bpb = float(
            eval_val_memmap_bpb(m_eval, val_mm, seq_len=seq_len, n_batches=4 if args.smoke_local else 24, batch_size=8, device=device)
        )
        ow6 = eval_openweb_6domains_any_byte_lm(m_eval, eval_wins, device=device, batch_size=8)
        long_ctx = eval_long_context_stress(
            m_eval, val_mm, device=device, lengths=(256, 512) if args.smoke_local else (2048, 8192, 32768)
        )
        ndl_hs = (256,) if args.smoke_local else (512, 1500, 4096, 8192, 32768)
        ndl = eval_ood_verbatim_ladder(
            m_eval,
            val_mm,
            device=device,
            haystacks=ndl_hs,
            n_samples=4 if args.smoke_local else 16,
            val_len=4 if args.smoke_local else 8,
            seed=9901,
        )
        ov_ndl = eval_overwrite_needle_ood(
            m_eval, val_mm, device=device, haystack_bytes=256 if args.smoke_local else 1500, n_samples=4 if args.smoke_local else 16
        )
        print(
            f"  [{tag}] Val={val_bpb:.4f} | OW6={ow6['mean_openweb_bpb']:.4f} | "
            f"L2048={long_ctx.get('L_2048', {}).get('bpb')} | L32K={long_ctx.get('L_32768', {}).get('bpb')} | "
            f"Orig1500={ndl['OrigRefCode_1500B']['exact_match_pct']:.1f}% ({ndl['OrigRefCode_1500B']['byte_acc_pct']:.1f}%b) | "
            f"OvNew={ov_ndl['new_exact_pct']:.1f}% (stale={ov_ndl['stale_old_exact_pct']:.1f}%)",
            flush=True,
        )
        for lk in [f"L_{h}" for h in ndl_hs]:
            lv = ndl[lk]
            print(
                f"    [{tag}] MultiNeedle {lk:7s}: gain={lv['recall_gain_bpb']:+.4f} | "
                f"byte={lv['byte_acc_pct']:5.1f}% | exact={lv['exact_match_pct']:5.1f}%",
                flush=True,
            )
        return {
            "val_bpb": round(val_bpb, 4),
            "ow6_mean_bpb": round(float(ow6["mean_openweb_bpb"]), 4),
            "ow6_domains": {k: round(float(v["byte_bpb"]), 4) for k, v in ow6["regimes"].items()},
            "long_context": long_ctx,
            "ood_verbatim_ladder": ndl,
            "overwrite_needle_1500b": ov_ndl,
        }

    eval_fp = _eval_bundle(model, "FP")

    # 2. Fast 4-Tap STE QAT -> Packed 2-bit uint8 LUT
    for p in model.parameters():
        p.requires_grad_(False)
    fp_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    qat_m = CalciteV2LM(cfg).to(device)
    qat_m.load_state_dict(fp_sd, strict=True)
    qat_m.byte_chunker.lock_router(True)
    qat_m.set_crystal_schedule(1, 1)
    qat_m.set_snap_alpha(1.0)
    qat_m.set_qat_mix(0.30)
    qat_m.set_ternary_qat(True)

    gain_q, cryst_q, chunk_q, core_q = _split_v3_qat_4groups(qat_m)
    all_qp = gain_q + cryst_q + chunk_q + core_q
    w_sc = (384.0 / float(cfg.d_model)) ** 0.5
    q_lrs = (1.2e-3 * w_sc, 8.0e-5, 1.1e-3 * w_sc, 4.5e-4 * w_sc)
    q_mins = (1.0e-4 * w_sc, 1.5e-5, 8.0e-5 * w_sc, 4.5e-5 * w_sc)
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
    warm_q = min(25, max(1, qat_steps // 10))
    ramp_q = min(100, max(1, qat_steps // 3))
    taps_fp: List[torch.Tensor] = []
    taps_q: List[torch.Tensor] = []
    hk_fp = _attach_4tap_hooks(model, taps_fp, detach=True)
    hk_q = _attach_4tap_hooks(qat_m, taps_q, detach=False)
    mbs_q = min(batch_size, 8 if not args.smoke_local else 4)
    tq0 = time.perf_counter()
    try:
        for qs in range(1, qat_steps + 1):
            qat_m.train()
            mix = min(1.0, 0.30 + 0.70 * (qs / float(ramp_q)))
            qat_m.set_qat_mix(mix)
            clear_fast_qat_cache()
            for gi, (blr, mlr) in enumerate(zip(q_lrs, q_mins)):
                opt_q.param_groups[gi]["lr"] = _cosine_lr(qs, qat_steps, warm_q, blr, mlr)
            xb_f, yb_f, _ = sample_lm_plus_copy_batch(
                train_mm, batch_size, seq_len, n_copy_rows, rng_py, rng_np, device
            )
            opt_q.zero_grad(set_to_none=True)
            n_acc = max(1, (batch_size + mbs_q - 1) // mbs_q)
            q_ce = 0.0
            for st in range(0, batch_size, mbs_q):
                q_ce += float(
                    _qat_micro_step(
                        model,
                        qat_m,
                        taps_fp,
                        taps_q,
                        1.5,
                        xb_f[st : st + mbs_q],
                        yb_f[st : st + mbs_q],
                        0,
                        use_amp,
                        n_acc,
                    )
                )
            torch.nn.utils.clip_grad_norm_(all_qp, 1.0)
            opt_q.step()
            clear_fast_qat_cache()
            if qs in (1, qat_steps) or (qs % 50 == 0):
                print(
                    f"  [QAT] step {qs:4d}/{qat_steps} | mix={mix:.2f} | ce={q_ce/n_acc:.4f} | {(time.perf_counter()-tq0)/60:.1f}m",
                    flush=True,
                )
    finally:
        for h in hk_fp + hk_q:
            h.remove()
        clear_fast_qat_cache()

    qat_s = round(time.perf_counter() - tq0, 2)
    del model, opt_q
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    qat_m.set_ternary_qat(False)
    qat_m.set_qat_mix(1.0)
    qat_sd = {k: v.detach().cpu().clone() for k, v in qat_m.state_dict().items()}
    del qat_m
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    clean_q = CalciteV2LM(cfg)
    clean_q.load_state_dict(qat_sd, strict=True)
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
    stream_cpu = benchmark_stream_engine_live(lut_cpu, device="cpu", n_bytes=32 if args.smoke_local else 128)
    print(
        f"  [LUT Stream CPU] {stream_cpu['bytes_per_sec']} B/s | err={stream_cpu['stream_vs_batch_max_err']} | "
        f"RAM={stream_cpu['state_kb']} KB | O1={stream_cpu['strict_o1_ram_invariant']}",
        flush=True,
    )

    lut_gpu = lut_cpu.to(device).eval()
    eval_lut = _eval_bundle(lut_gpu, "2bit-LUT")
    stream_needle_lut = eval_live_stream_needle(
        lut_gpu, val_mm, device=device, haystack_bytes=256 if args.smoke_local else 1500, n_samples=2 if args.smoke_local else 4
    )
    print(f"  [LUT Live O(1) Stream Needle 1500B] {stream_needle_lut}", flush=True)

    res = {
        "meta": {
            "phase": "Phase_N_Stage2_Calcite_v31_5M_LM_L32K_LUT",
            "params": n_params,
            "cache_kb_o1": cache_kb,
            "byte_induct_slots": n_slots,
            "steps_fp": steps,
            "steps_qat": qat_steps,
            "fp_train_seconds": fp_train_s,
            "qat_seconds": qat_s,
            "smoke": args.smoke_local,
        },
        "eval_fp": eval_fp,
        "eval_lut": eval_lut,
        "lut_footprint": lut_foot,
        "stream_cpu": stream_cpu,
        "stream_needle_lut": stream_needle_lut,
    }
    out_p = args.out_dir / ("smoke_v31_5m_lm_lut.json" if args.smoke_local else "phase_n_v31_5m_lm_lut.json")
    out_p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n[Calcite v3.1 5M Verification Complete] Saved -> {out_p}", flush=True)


if __name__ == "__main__":
    main()
