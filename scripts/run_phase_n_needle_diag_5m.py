"""
Phase N: Causal Verbatim Needle-in-a-Haystack Diagnosis on `S1_5M`
(`scripts/run_phase_n_needle_diag_5m.py`).

Separates **Hypothesis A (Training Data Blindness)** from **Hypothesis B (Architectural
O(1) Exact Induction Slot Capacity)** under strict zero-cheat, template-disjoint OOD rules:
  - `N2_RoPE_copydata`:      Full-attention `RoPE 5M + Muon` (in-dist upper bound & OOD collapse baseline)
  - `N0_v3_copydata`:        Exact `Calcite v3` (`M3 + Byte Engram + Muon`, `120.5 KB` O(1))
  - `N1_v3_induct_slots`:    `Calcite v3` + parallel `InductionSlotMemory` (`M2-v2`, `145.5 KB` O(1))

All three cells see the EXACT same `82%` OpenWeb UTF-8 prose + `18%` diverse multi-needle
verbatim copy stream (`seq_len = 512`, `14` training key-value templates, `1–3` needles/seq,
random `5–9` byte keys, `0%` overlap with eval templates).

Evaluated strictly out-of-distribution on:
  1. Untouched Phase M `eval_zeroshot_needle_recall_bpb` (`1500 B`, `n=24`, seed `7777`,
     `"Important reference code:"` — NEVER in train!) + true greedy exact match.
  2. Held-out Multi-Needle (2 distinct unseen keys per haystack) across `L = 512 -> 1500 -> 4096 -> 8192 B`
     (`3x–16x` beyond `L_train = 512`) with teacher-forced AND greedy exact match.
  3. Zero-regression language guard: `Val BPB (L=512)`, `OW6 BPB`, `L=2048` & `L=8192` prose BPB.
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
from simvq_split.baselines import (  # noqa: E402
    BaselineConfig,
    RoPETransformerLM,
)
from experiments.scaling_phase_k.eval_suite import (  # noqa: E402
    eval_openweb_6domains_any_byte_lm,
    eval_val_memmap_bpb,
    has_usable_cuda,
    sample_memmap_batch,
)
from run_phase_e_fast_utf8_benchmark import build_disjoint_openweb_corpora  # noqa: E402
from scripts.colab_phase_k_scaling_runner import _cosine_lr  # noqa: E402
from scripts.colab_phase_l_v2_runner import (  # noqa: E402
    ensure_phase_k_corpus,
    eval_long_context_stress,
    load_frozen_router_into,
    set_all_seeds,
)
from scripts.muon_opt import MuonAdamWHybrid  # noqa: E402
from simvq_split.tasks import load_gsm8k_corpora  # noqa: E402
from simvq_split.tokenizer import ByteBPETokenizer  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Strictly Disjoint Train vs Held-Out Eval Verbatim Copy Templates
# ---------------------------------------------------------------------------

TRAIN_COPY_TEMPLATES: Tuple[Tuple[str, str], ...] = (
    ("\nTicket ID for this session is {val}.\n", "\nConfirm the Ticket ID: "),
    ("\nThe assigned Vault PIN = {val}; keep it safe.\n", "\nEnter the Vault PIN: "),
    ("\nOrder ref [{val}] was logged.\n", "\nLookup Order ref: "),
    ("\nDevice Serial #{val} passed check.\n", "\nEcho Device Serial #"),
    ('\nconfig["api_token"] = "{val}"\n', '\nprint(config["api_token"]) # '),
    ("\nSession hex -> {val}.\n", "\nCurrent Session hex -> "),
    ("\nVisitor Badge: {val} (active).\n", "\nVerify Visitor Badge: "),
    ("\nBlock checksum is {val}.\n", "\nExpected block checksum: "),
    ("\nClient key={val}, status=ok.\n", "\nQuery Client key="),
    ("\nUse routing code {val} for dispatch.\n", "\nDispatch routing code: "),
    ("\nArchive tag ({val}) saved.\n", "\nRetrieve Archive tag: "),
    ("\nrelease_hash = '{val}'\n", "\nactive release_hash: "),
    ("\nAccount ID: {val} verified.\n", "\nShow Account ID: "),
    ("\nLocker pass is {val}.\n", "\nLocker pass: "),
)

# Strictly UNSEEN in training!
EVAL_MULTI_TEMPLATES: Tuple[Tuple[str, str, str], ...] = (
    ("\nCipher seed: {val}.\n", "\nCipher seed: standard.\n", "\nRecall the Cipher seed: "),
    ("\nParcel tag = {val}.\n", "\nParcel tag = standard.\n", "\nRecall the Parcel tag: "),
    ("\nBeacon id [{val}].\n", "\nBeacon id [standard].\n", "\nRecall the Beacon id: "),
    ("\nLedger pin -> {val}.\n", "\nLedger pin -> standard.\n", "\nRecall the Ledger pin: "),
)

TRAIN_ALPHABETS: Tuple[str, ...] = (
    "ABCDEFGHJKLMNPQRSTUVWXYZ23456789",
    "0123456789abcdef",
    "abcdefghijklmnopqrstuvwxyz0123456789",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
)


def _enc_utf8(s: str) -> List[int]:
    return [b + 3 for b in s.encode("utf-8")]


def sample_train_copy_row(
    train_mm: np.memmap,
    seq_len: int,
    rng_py: random.Random,
    rng_np: np.random.Generator,
    pad_id: int = 0,
) -> Tuple[List[int], List[int]]:
    """
    Constructs one `seq_len`-byte multi-needle verbatim copy sequence from `train_mm`
    using 1..3 distinct `TRAIN_COPY_TEMPLATES` (strictly disjoint from eval).
    Returns `(x_ids, y_ids)` where `y_ids` supervises ONLY the copied target bytes.
    """
    n_keys = rng_py.choice((1, 2, 2, 3))
    chosen_tpls = rng_py.sample(TRAIN_COPY_TEMPLATES, k=n_keys)
    vals: List[str] = []
    stmts: List[List[int]] = []
    for st_tpl, _ in chosen_tpls:
        alph = rng_py.choice(TRAIN_ALPHABETS)
        vlen = rng_py.randint(5, 9)
        val = "".join(rng_py.choice(alph) for _ in range(vlen))
        vals.append(val)
        stmts.append(_enc_utf8(st_tpl.format(val=val)))

    # 20% chance to overwrite one key in-stream so overwrite + verbatim recall co-adapt
    do_overwrite = (n_keys >= 2) and (rng_py.random() < 0.20)
    q_idx = rng_py.randrange(n_keys)
    if do_overwrite:
        alph = rng_py.choice(TRAIN_ALPHABETS)
        new_val = "".join(rng_py.choice(alph) for _ in range(len(vals[q_idx])))
        vals[q_idx] = new_val
        stmts.append(_enc_utf8(chosen_tpls[q_idx][0].format(val=new_val)))

    q_hdr = _enc_utf8(chosen_tpls[q_idx][1])
    t_ids = _enc_utf8(vals[q_idx])

    fixed_len = sum(len(s) for s in stmts) + len(q_hdr) + len(t_ids)
    prose_budget = max(32, seq_len + 1 - fixed_len)

    max_st = max(1, len(train_mm) - prose_budget - 16)
    pos = int(rng_np.integers(0, max_st))
    # Align start to UTF-8 lead byte
    for _ in range(3):
        tv = int(train_mm[pos])
        if tv >= 3 and ((tv - 3) & 0xC0) == 0x80:
            pos += 1
        else:
            break
    prose = train_mm[pos : pos + prose_budget].astype(np.int64).tolist()

    # Distribute statements across the first ~60% of prose so haystack gap is ~150..360 B
    n_seg = len(stmts) + 1
    cuts: List[int] = [0]
    hi_zone = max(n_seg + 4, int(prose_budget * 0.55))
    inner = sorted(rng_py.sample(range(8, hi_zone), k=len(stmts))) if hi_zone > 8 + len(stmts) else [
        int(prose_budget * (i + 1) / (n_seg + 1)) for i in range(len(stmts))
    ]
    cuts.extend(inner)
    cuts.append(prose_budget)

    seq: List[int] = []
    for i, st in enumerate(stmts):
        seq.extend(prose[cuts[i] : cuts[i + 1]])
        seq.extend(st)
    seq.extend(prose[cuts[-2] : cuts[-1]])
    seq.extend(q_hdr)
    target_start = len(seq)
    seq.extend(t_ids)

    # Fit into exact `seq_len + 1`
    if len(seq) > seq_len + 1:
        excess = len(seq) - (seq_len + 1)
        # Trim from the last filler gap right before `q_hdr`, never cutting statements or target
        trim_end = target_start - len(q_hdr)
        trim_start = max(0, trim_end - excess)
        seq = seq[:trim_start] + seq[trim_end:]
        target_start -= trim_end - trim_start

    x_row = [pad_id] * seq_len
    y_row = [pad_id] * seq_len
    L_eff = min(seq_len, len(seq) - 1)
    x_row[:L_eff] = seq[:L_eff]
    # Supervise ONLY target value bytes: x[p-1] predicts seq[p] for p in [target_start .. len(seq)-1]
    for p in range(target_start, min(len(seq), seq_len + 1)):
        if p - 1 < seq_len:
            y_row[p - 1] = seq[p]
    return x_row, y_row


def sample_mixed_batch(
    train_mm: np.memmap,
    batch_size: int,
    seq_len: int,
    copy_prob: float,
    rng_py: random.Random,
    rng_np: np.random.Generator,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns `(x, y, is_copy_row_mask)` with interleaved prose/copy rows so every microbatch has both."""
    n_copy = int(round(batch_size * copy_prob)) if copy_prob > 0.0 else 0
    n_prose = max(1, batch_size - n_copy)
    n_copy = batch_size - n_prose

    xp, yp = sample_memmap_batch(train_mm, n_prose, seq_len, rng_np, device)
    if n_copy <= 0:
        return xp, yp, torch.zeros(batch_size, dtype=torch.bool, device=device)

    xc_rows: List[List[int]] = []
    yc_rows: List[List[int]] = []
    for _ in range(n_copy):
        xr, yr = sample_train_copy_row(train_mm, seq_len, rng_py, rng_np, pad_id=0)
        xc_rows.append(xr)
        yc_rows.append(yr)
    xc = torch.tensor(xc_rows, dtype=torch.long, device=device)
    yc = torch.tensor(yc_rows, dtype=torch.long, device=device)
    x_all = torch.cat([xp, xc], dim=0)
    y_all = torch.cat([yp, yc], dim=0)
    m_all = torch.zeros(batch_size, dtype=torch.bool, device=device)
    m_all[n_prose:] = True
    # Interleave deterministically so every microbatch receives both prose and copy rows
    perm = torch.arange(batch_size, device=device)
    if n_copy > 0:
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


# ---------------------------------------------------------------------------
# 2. Strictly Held-Out OOD Verbatim Needle Evaluators (Teacher-Forced + Greedy)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _score_needle_pair(
    model: nn.Module,
    seq_with: List[int],
    seq_without: List[int],
    target_ids: List[int],
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    run_greedy: bool = True,
) -> Tuple[float, float, float, bool, bool]:
    """Returns `(bpb_with, bpb_without, tf_byte_acc, tf_exact, greedy_exact)`."""
    n_t = len(target_ids)

    def _tf(seq: List[int]) -> Tuple[float, float, bool]:
        x = torch.tensor([seq[:-1]], device=device, dtype=torch.long)
        y = torch.tensor([seq[1:]], device=device, dtype=torch.long)
        with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            lg = model(x)
        lg_t = lg[:, -n_t:, :].float()
        y_t = y[:, -n_t:]
        bpb = float(
            F.cross_entropy(lg_t.reshape(-1, lg_t.size(-1)), y_t.reshape(-1)).item()
            / math.log(2.0)
        )
        preds = lg_t.argmax(dim=-1)
        b_acc = float((preds == y_t).float().mean().item())
        exact = bool((preds == y_t).all().item())
        return bpb, b_acc, exact

    bw, b_acc, tf_ex = _tf(seq_with)
    bwo, _, _ = _tf(seq_without)

    gr_ex = False
    if run_greedy:
        # True autoregressive greedy decoding (zero teacher forcing on the needle!)
        prefix_ids = list(seq_with[:-n_t])
        gen: List[int] = []
        for _ in range(n_t):
            x_cur = torch.tensor([prefix_ids + gen], device=device, dtype=torch.long)
            with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                lg = model(x_cur)
            nxt = int(lg[0, -1, :].float().argmax().item())
            gen.append(nxt)
        gr_ex = gen == target_ids
    return bw, bwo, b_acc, tf_ex, gr_ex


@torch.no_grad()
def eval_heldout_passkey_single(
    model: nn.Module,
    val_mm: np.memmap,
    device: str,
    haystack_bytes: int = 1500,
    n_samples: int = 24,
    seed: int = 7777,
    run_greedy: bool = True,
) -> Dict[str, float]:
    """Exact original Phase M `"Important reference code:"` probe + byte_acc & greedy exact."""
    model.eval()
    rng_py = random.Random(seed)
    rng_np = np.random.default_rng(seed)
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    use_amp = device == "cuda" and amp_dtype == torch.bfloat16

    bw_l: List[float] = []
    bwo_l: List[float] = []
    bacc_l: List[float] = []
    tf_hits = 0
    gr_hits = 0

    max_start = max(1, len(val_mm) - haystack_bytes - 64)
    for _ in range(n_samples):
        code = "".join(rng_py.choice(alphabet) for _ in range(8))
        pos = int(rng_np.integers(0, max_start))
        raw = val_mm[pos : pos + haystack_bytes].astype(np.int64).tolist()
        prefix = raw[:128]
        middle = raw[128:]

        needle_stmt = _enc_utf8(f"\nImportant reference code: {code}.\n")
        decoy_stmt = _enc_utf8("\nImportant reference note: standard.\n")
        query_hdr = _enc_utf8("\nRepeat the important reference code: ")
        target_ids = _enc_utf8(code)

        seq_w = prefix + needle_stmt + middle + query_hdr + target_ids
        seq_wo = prefix + decoy_stmt + middle + query_hdr + target_ids
        bw, bwo, ba, tf_ex, gr_ex = _score_needle_pair(
            model, seq_w, seq_wo, target_ids, device, use_amp, amp_dtype, run_greedy=run_greedy
        )
        bw_l.append(bw)
        bwo_l.append(bwo)
        bacc_l.append(ba)
        tf_hits += int(tf_ex)
        gr_hits += int(gr_ex)

    mw = float(np.mean(bw_l))
    mwo = float(np.mean(bwo_l))
    return {
        "haystack_bytes": int(haystack_bytes),
        "n_samples": int(n_samples),
        "needle_bpb_with": round(mw, 4),
        "needle_bpb_without": round(mwo, 4),
        "recall_gain_bpb": round(mwo - mw, 4),
        "byte_acc_pct": round(100.0 * float(np.mean(bacc_l)), 2),
        "tf_exact_pct": round(100.0 * tf_hits / max(1, n_samples), 2),
        "greedy_exact_pct": round(100.0 * gr_hits / max(1, n_samples), 2),
    }


@torch.no_grad()
def eval_heldout_multi_needle_ladder(
    model: nn.Module,
    val_mm: np.memmap,
    device: str,
    haystacks: Tuple[int, ...] = (512, 1500, 4096, 8192),
    n_samples: int = 24,
    seed: int = 9901,
) -> Dict[str, Dict[str, float]]:
    """
    Held-out 2-needle content-addressed verbatim recall across `512 -> 1500 -> 4096 -> 8192 B`
    using strictly unseen `EVAL_MULTI_TEMPLATES`.
    """
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
        # Run greedy on up to 4096B (and on 8192B only for first 8 samples to keep eval fast)
        max_start = max(1, len(val_mm) - h_len - 64)
        n_greedy = n_samples if h_len <= 4096 else min(n_samples, 8)

        for si in range(n_samples):
            t1, t2 = rng_py.sample(EVAL_MULTI_TEMPLATES, k=2)
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
            s1_wo = _enc_utf8(t1[1]) if q_which == 0 else s1_w
            s2_wo = _enc_utf8(t2[1]) if q_which == 1 else s2_w
            q_hdr = _enc_utf8(q_tpl[2])
            t_ids = _enc_utf8(q_code)

            seq_w = raw[:p1] + s1_w + raw[p1:p2] + s2_w + raw[p2:] + q_hdr + t_ids
            seq_wo = raw[:p1] + s1_wo + raw[p1:p2] + s2_wo + raw[p2:] + q_hdr + t_ids

            do_gr = si < n_greedy
            bw, bwo, ba, tf_ex, gr_ex = _score_needle_pair(
                model, seq_w, seq_wo, t_ids, device, use_amp, amp_dtype, run_greedy=do_gr
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
            "n_greedy": int(n_greedy),
            "needle_bpb_with": round(mw, 4),
            "needle_bpb_without": round(mwo, 4),
            "recall_gain_bpb": round(mwo - mw, 4),
            "byte_acc_pct": round(100.0 * float(np.mean(bacc_l)), 2),
            "tf_exact_pct": round(100.0 * tf_hits / max(1, n_samples), 2),
            "greedy_exact_pct": round(100.0 * gr_hits / max(1, n_greedy), 2),
        }
        if device == "cuda":
            torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# 3. Unified Fair Muon+AdamW Training Loop (Identical Stream for All 3 Cells)
# ---------------------------------------------------------------------------


def train_and_eval_cell(
    name: str,
    model: nn.Module,
    cache_kb: float,
    train_mm: np.memmap,
    val_mm: np.memmap,
    eval_wins: Dict[str, List[List[int]]],
    device: str,
    steps: int,
    batch_size: int,
    micro_bs: int,
    copy_prob: float,
    seed: int,
    smoke: bool,
) -> Dict[str, object]:
    print(f"\n==================== [{name}] ====================", flush=True)
    n_params = int(sum(p.numel() for p in model.parameters()))
    print(
        f"[{name}] params={n_params:,} | cache_O1={cache_kb:.2f} KB | steps={steps} | copy_prob={copy_prob:.2f}",
        flush=True,
    )

    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
    base_lr = 1.8e-3
    min_lr = 1.2e-4
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
    rng_py = random.Random(seed + 1337)
    rng_np = np.random.default_rng(seed + 1337)
    is_calcite = isinstance(model, CalciteV2LM)

    t0 = time.perf_counter()
    cur_mbs = min(batch_size, micro_bs)
    for step in range(1, steps + 1):
        model.train()
        if is_calcite:
            model.set_crystal_schedule(step, steps)
        lr_s = _cosine_lr(step, steps, warm, base_lr, min_lr)
        sc = lr_s / base_lr
        for pg in opt.param_groups:
            pg["lr"] = base_lr * float(pg.get("lr_mult", 1.0)) * sc

        xb_full, yb_full, cm_full = sample_mixed_batch(
            train_mm, batch_size, 128 if smoke else 512, copy_prob, rng_py, rng_np, device
        )

        # Auto-halving OOM-safe gradient accumulation with balanced prose/copy CE
        while True:
            opt.zero_grad(set_to_none=True)
            n_acc = max(1, (batch_size + cur_mbs - 1) // cur_mbs)
            ce_p_sum = 0.0
            ce_c_sum = 0.0
            n_c_acc = 0
            try:
                for st in range(0, batch_size, cur_mbs):
                    xb = xb_full[st : st + cur_mbs]
                    yb = yb_full[st : st + cur_mbs]
                    cm = cm_full[st : st + cur_mbs]
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                        lg = model(xb)
                        lg_f = lg.float()
                        pm = ~cm
                        has_p = bool(pm.any().item())
                        has_c = bool(cm.any().item())
                        ce_p = (
                            F.cross_entropy(
                                lg_f[pm].reshape(-1, lg_f.size(-1)),
                                yb[pm].reshape(-1),
                                ignore_index=0,
                            )
                            if has_p
                            else torch.zeros((), device=device, dtype=torch.float32)
                        )
                        ce_c = (
                            F.cross_entropy(
                                lg_f[cm].reshape(-1, lg_f.size(-1)),
                                yb[cm].reshape(-1),
                                ignore_index=0,
                            )
                            if has_c
                            else torch.zeros((), device=device, dtype=torch.float32)
                        )
                        if has_p and has_c:
                            ce = 0.75 * ce_p + 0.25 * ce_c
                        elif has_p:
                            ce = ce_p
                        else:
                            ce = ce_c
                        aux = model.get_commitment_loss() if is_calcite else 0.0
                        loss = (ce + 0.04 * aux) / float(n_acc)
                    loss.backward()
                    ce_p_sum += float(ce_p.detach().item())
                    if has_c:
                        ce_c_sum += float(ce_c.detach().item())
                        n_c_acc += 1
                break
            except torch.cuda.OutOfMemoryError:
                if cur_mbs <= 2:
                    raise
                cur_mbs = max(2, cur_mbs // 2)
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(f"  [OOM adapt] {name} step {step}: reducing micro_bs -> {cur_mbs}", flush=True)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step in (1, steps) or (step % 100 == 0):
            es_info = ""
            if is_calcite and model.epistemic_slots is not None:
                es = model.epistemic_slots
                es_info = f" | wr={es.last_write_rate:.2f}"
            vram_mb = torch.cuda.memory_allocated() / 1e6 if device == "cuda" else 0.0
            print(
                f"  [{name}] step {step:4d}/{steps} | ce_p={ce_p_sum/n_acc:.4f} | "
                f"ce_c={ce_c_sum/max(1, n_c_acc):.4f}{es_info} | "
                f"VRAM={vram_mb:.0f}MB | {(time.perf_counter()-t0)/60:.1f}m",
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

    n_val_b = 4 if smoke else 24
    val_bpb = float(
        eval_val_memmap_bpb(model, val_mm, seq_len=128 if smoke else 512, n_batches=n_val_b, batch_size=8, device=device)
    )
    ow6 = eval_openweb_6domains_any_byte_lm(model, eval_wins, device=device, batch_size=8)
    long_ctx = eval_long_context_stress(
        model, val_mm, device=device, lengths=(512, 1024) if smoke else (2048, 8192)
    )
    n_ndl = 4 if smoke else 24
    needle_orig = eval_heldout_passkey_single(
        model, val_mm, device=device, haystack_bytes=256 if smoke else 1500, n_samples=n_ndl, seed=7777, run_greedy=True
    )
    needle_ladder = eval_heldout_multi_needle_ladder(
        model,
        val_mm,
        device=device,
        haystacks=(256, 512) if smoke else (512, 1500, 4096, 8192),
        n_samples=n_ndl,
        seed=9901,
    )

    telemetry: Dict[str, object] = {}
    if is_calcite and model.epistemic_slots is not None:
        es = model.epistemic_slots
        lw = F.softmax(es.lag_logits.detach().float(), dim=-1).cpu().tolist()
        telemetry = {
            "last_write_rate": round(float(es.last_write_rate), 4),
            "lag_weights": [[round(v, 4) for v in row] for row in lw],
            "read_scale": [round(float(v), 3) for v in es.read_scale.detach().cpu().tolist()],
            "null_logit": [round(float(v), 3) for v in es.null_logit.detach().cpu().tolist()],
            "write_bias": [round(float(v), 3) for v in es.write_proj.bias.detach().cpu().tolist()],
            "novelty_scale": [round(float(v), 3) for v in es.novelty_scale.detach().cpu().tolist()],
            "slots_gate_bias": round(float(model.epistemic_slots_gate.bias.detach().item()), 3),
        }

    print(
        f"  [{name} DONE] Val={val_bpb:.4f} | OW6={ow6['mean_openweb_bpb']:.4f} | "
        f"OrigNeedle1500: gain={needle_orig['recall_gain_bpb']:+.4f} byte={needle_orig['byte_acc_pct']:.1f}% "
        f"tfEx={needle_orig['tf_exact_pct']:.1f}% grEx={needle_orig['greedy_exact_pct']:.1f}%",
        flush=True,
    )
    for lk, lv in needle_ladder.items():
        print(
            f"    MultiNeedle {lk}: gain={lv['recall_gain_bpb']:+.4f} | byte={lv['byte_acc_pct']:.1f}% | "
            f"tfEx={lv['tf_exact_pct']:.1f}% | grEx={lv['greedy_exact_pct']:.1f}%",
            flush=True,
        )
    if telemetry:
        print(f"    Slots telemetry: {telemetry}", flush=True)

    return {
        "name": name,
        "params": n_params,
        "cache_kb_o1": round(cache_kb, 2),
        "train": {
            "steps": int(steps),
            "batch_size": int(batch_size),
            "copy_prob": float(copy_prob),
            "train_seconds": round(train_s, 2),
            "ms_per_step": round(1000.0 * train_s / max(1, steps), 2),
        },
        "val_bpb": round(val_bpb, 4),
        "ow6_mean_bpb": round(float(ow6["mean_openweb_bpb"]), 4),
        "ow6_domains": {k: round(float(v["byte_bpb"]), 4) for k, v in ow6["regimes"].items()},
        "long_context": long_ctx,
        "orig_needle_1500b": needle_orig,
        "multi_needle_ladder": needle_ladder,
        "slots_telemetry": telemetry,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT_DIR / "data_cache")
    ap.add_argument("--out-dir", type=Path, default=ROOT_DIR / "benchmarks" / "phase_n_needle")
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--micro-bs", type=int, default=16)
    ap.add_argument("--copy-prob", type=float, default=0.18)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--smoke-local", action="store_true")
    args = ap.parse_args()

    device = "cuda" if has_usable_cuda() else "cpu"
    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram_gb < 12.0:
            torch.cuda.set_per_process_memory_fraction(0.75, 0)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tok = ByteBPETokenizer(merges=[])
    tr_docs, te_docs, _, _ = load_gsm8k_corpora(ROOT_DIR)
    seq_len = 128 if args.smoke_local else 512
    _, eval_wins, _, _ = build_disjoint_openweb_corpora(ROOT_DIR, tok, tr_docs, te_docs, seq_len=seq_len)

    if args.smoke_local:
        rng_f = np.random.default_rng(args.seed)
        fake = rng_f.integers(32 + 3, 126 + 3, size=60_000, dtype=np.uint16)
        train_mm = fake
        val_mm = fake
        steps = 6
        batch_size = 8
        micro_bs = 4
    else:
        train_bin, val_bin = ensure_phase_k_corpus(args.data_dir)
        train_mm = np.memmap(train_bin, dtype=np.uint16, mode="r")
        val_mm = np.memmap(val_bin, dtype=np.uint16, mode="r")
        steps = int(args.steps)
        batch_size = int(args.batch_size)
        micro_bs = int(args.micro_bs)

    router_path = ROOT_DIR / "benchmarks" / "phase_l_v2" / "frozen_utf8_router.pt"
    assert router_path.is_file(), f"Missing frozen UTF-8 router: {router_path}"
    router_state = torch.load(router_path, map_location="cpu", weights_only=True)

    res_path = args.out_dir / ("smoke_needle_diag_5m.json" if args.smoke_local else "phase_n_needle_diag_5m.json")
    t_all = time.perf_counter()
    cells: Dict[str, object] = {}

    def _save() -> None:
        payload = {
            "meta": {
                "phase": "Phase_N_Causal_Needle_Diagnosis_5M",
                "steps": steps,
                "batch_size": batch_size,
                "copy_prob": args.copy_prob,
                "seed": args.seed,
                "smoke": args.smoke_local,
                "elapsed_s": round(time.perf_counter() - t_all, 1),
            },
            "cells": cells,
        }
        res_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Cell 1: N2_RoPE_copydata (Full-Attention Upper Bound @512 & OOD Ref)
    # ------------------------------------------------------------------
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
    rope_kv_kb = float(rope_cfg.n_layers * 2 * 2048 * rope_cfg.d_model * 2) / 1024.0
    cells["N2_RoPE_copydata"] = train_and_eval_cell(
        "N2_RoPE_copydata",
        rope_m,
        rope_kv_kb,
        train_mm,
        val_mm,
        eval_wins,
        device,
        steps,
        batch_size,
        micro_bs,
        args.copy_prob,
        args.seed,
        args.smoke_local,
    )
    _save()
    del rope_m
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Cell 2: N0_v3_copydata (Exact Calcite v3 — Tests Hypothesis A: Data)
    # ------------------------------------------------------------------
    def _build_v3_cfg(use_slots: bool) -> CalciteV2Config:
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

    for cname, use_slots in [("N0_v3_copydata", False), ("N1_v3_induct_slots", True)]:
        set_all_seeds(args.seed)
        cfg = _build_v3_cfg(use_slots=use_slots)
        m = CalciteV2LM(cfg).to(device)
        load_frozen_router_into(m, router_state)
        kb = compute_v2_state_cache_kb(cfg, 32768)
        cells[cname] = train_and_eval_cell(
            cname,
            m,
            kb,
            train_mm,
            val_mm,
            eval_wins,
            device,
            steps,
            batch_size,
            micro_bs,
            args.copy_prob,
            args.seed,
            args.smoke_local,
        )
        _save()
        del m
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n[Phase N Diagnosis Complete] Saved -> {res_path}", flush=True)


if __name__ == "__main__":
    main()
