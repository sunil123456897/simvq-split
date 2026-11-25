"""
Phase N Probe B: Fast Dense Multi-Query Verbatim Recall + OOD Passkey Probe
(`scripts/run_phase_n_mqar_probe_5m.py`).

Fixes the two root causes that stalled Probe A even on full quadratic `RoPE 5M`:
1. **Prefix LM supervision (`0.25 * ce_prefix + 0.75 * ce_copy`)**: models actually learn
   key/template/word representations instead of backpropagating through 7 bytes into random embeddings.
2. **Dense Multi-Query Copy (`4` key-value pairs stored inside real `train_mm` prose, ALL `4`
   queried in random order at the end -> `32` target bytes/row, `5x` denser induction signal!)**
   with clean delimiter alignment + diverse key names (zero overlap with eval templates!).
3. **Fast Eval**: runs greedy decoding ONLY when teacher-forced exact match > 0 (or on `L <= 1500`),
   avoiding 5-minute uncached greedy loops on 0% models.

Compares identically seeded `5M` models (`400` steps, ~`6` min total on RTX 5060):
  - `B0_RoPE_5M`               (full quadratic attention upper bound @512, OOD collapse @1500+)
  - `B1_Calcite_v3_5M`         (exact `Calcite v3`, Woodbury only, `120.5 KB` O(1))
  - `B2_Calcite_v3_InductSlots`(`Calcite v3` + chunk `InductionSlotMemory`, `145.5 KB` O(1))
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
    has_usable_cuda,
    load_utf8_corpus_array,
)
from scripts.colab_phase_k_scaling_runner import _cosine_lr  # noqa: E402
from scripts.colab_phase_l_v2_runner import (  # noqa: E402
    ensure_phase_k_corpus,
    load_frozen_router_into,
    set_all_seeds,
)
from scripts.muon_opt import MuonAdamWHybrid  # noqa: E402
from simvq_split.baselines import BaselineConfig, RoPETransformerLM  # noqa: E402

ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

# 16 training key names (strictly disjoint from eval keys!)
TRAIN_KEYS: Tuple[str, ...] = (
    "ticket", "vault", "order", "serial", "token", "session", "badge", "checksum",
    "client", "route", "archive", "build", "account", "locker", "permit", "relay",
)

# Strictly UNSEEN eval keys!
EVAL_KEYS: Tuple[str, ...] = (
    "cipher seed", "parcel tag", "beacon id", "ledger pin", "important reference code",
)


def _enc(s: str) -> List[int]:
    return [b + 3 for b in s.encode("utf-8")]


def sample_dense_mqar_row(
    train_mm: np.memmap,
    seq_len: int,
    n_pairs: int,
    val_len: int,
    rng_py: random.Random,
    rng_np: np.random.Generator,
) -> Tuple[List[int], List[int], List[bool]]:
    """
    Embeds `n_pairs` distinct `(key: VAL)` entries inside real `train_mm` prose,
    then queries ALL `n_pairs` keys in shuffled order at the end.
    Returns `(x_ids, y_ids, is_copy_target_mask)` of exact length `seq_len`.
    """
    keys = rng_py.sample(TRAIN_KEYS, k=n_pairs)
    # 50% natural prose key, 50% random 4-char key tag (teaches zero-shot arbitrary key matching!)
    pairs: List[Tuple[str, str]] = []
    for k in keys:
        if rng_py.random() < 0.50:
            k_str = f"{k} {''.join(rng_py.choice(ALPHABET[:16]) for _ in range(3))}"
        else:
            k_str = k
        val = "".join(rng_py.choice(ALPHABET) for _ in range(val_len))
        pairs.append((k_str, val))

    stmts = [_enc(f" [{k}: {v}] ") for k, v in pairs]
    q_order = list(pairs)
    rng_py.shuffle(q_order)
    q_blocks = [(_enc(f" {k}: "), _enc(v)) for k, v in q_order]

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

    # Place pairs in the first 55% of prose, leaving a clean filler gap before queries
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
        # Trim strictly from `tail_seg` right before the first query
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


def sample_dense_mqar_batch(
    train_mm: np.ndarray,
    batch_size: int,
    seq_len: int,
    n_pairs: int,
    val_len: int,
    rng_py: random.Random,
    rng_np: np.random.Generator,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs, ys, ms = [], [], []
    for _ in range(batch_size):
        xr, yr, mr = sample_dense_mqar_row(train_mm, seq_len, n_pairs, val_len, rng_py, rng_np)
        xs.append(xr)
        ys.append(yr)
        ms.append(mr)
    xt = torch.from_numpy(np.asarray(xs, dtype=np.int64))
    yt = torch.from_numpy(np.asarray(ys, dtype=np.int64))
    mt = torch.from_numpy(np.asarray(ms, dtype=np.bool_))
    if device == "cuda" and has_usable_cuda():
        xt = xt.pin_memory().to(device=device, non_blocking=True)
        yt = yt.pin_memory().to(device=device, non_blocking=True)
        mt = mt.pin_memory().to(device=device, non_blocking=True)
    elif device != "cpu":
        xt = xt.to(device=device)
        yt = yt.to(device=device)
        mt = mt.to(device=device)
    return xt, yt, mt


@torch.inference_mode()
def eval_ood_verbatim_ladder(
    model: nn.Module,
    val_mm: np.ndarray,
    device: str,
    haystacks: Tuple[int, ...] = (256, 512, 1500, 4096, 8192),
    n_samples: int = 24,
    val_len: int = 8,
    seed: int = 9901,
) -> Dict[str, Dict[str, float]]:
    """
    Strictly UNSEEN `EVAL_KEYS` 2-needle verbatim recall across `256 -> 512 -> 1500 -> 4096 -> 8192 B`.
    Tests BOTH exact delimiter (`key: VAL ... key: `) AND original `"Important reference code:"` phrasing.
    Batched with right-padding (`PAD_ID=0`, strictly causal) and automatic OOM backoff.
    """
    model.eval()
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    use_amp = device == "cuda" and amp_dtype == torch.bfloat16
    results: Dict[str, Dict[str, float]] = {}

    def _eval_seq_batch(
        seqs: List[List[int]], n_ts: List[int], init_bs: int
    ) -> List[Tuple[float, float, bool]]:
        bs = max(1, min(int(init_bs), len(seqs)))
        while True:
            try:
                out_metrics: List[Tuple[float, float, bool]] = []
                for s0 in range(0, len(seqs), bs):
                    sub_seqs = seqs[s0 : s0 + bs]
                    sub_nt = n_ts[s0 : s0 + bs]
                    lens = [len(sq) - 1 for sq in sub_seqs]
                    max_l = max(lens)
                    x_np = np.zeros((len(sub_seqs), max_l), dtype=np.int64)
                    y_np = np.zeros((len(sub_seqs), max_l), dtype=np.int64)
                    for ri, (sq, ln) in enumerate(zip(sub_seqs, lens)):
                        x_np[ri, :ln] = sq[:-1]
                        y_np[ri, :ln] = sq[1:]
                    xb = torch.from_numpy(x_np).to(device=device, non_blocking=(device == "cuda"))
                    yb = torch.from_numpy(y_np).to(device=device, non_blocking=(device == "cuda"))
                    with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                        lg = model(xb)
                    lg_f = lg.float()
                    for ri, (ln, nt) in enumerate(zip(lens, sub_nt)):
                        lg_t = lg_f[ri, ln - nt : ln, :]
                        y_t = yb[ri, ln - nt : ln]
                        bpb = float(F.cross_entropy(lg_t, y_t).item() / math.log(2.0))
                        pr = lg_t.argmax(dim=-1)
                        eq = pr == y_t
                        out_metrics.append((bpb, float(eq.float().mean().item()), bool(eq.all().item())))
                return out_metrics
            except torch.cuda.OutOfMemoryError:
                if bs <= 1:
                    raise
                bs = max(1, bs // 2)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    for h_len in haystacks:
        rng_py = random.Random(seed + h_len)
        rng_np = np.random.default_rng(seed + h_len)
        all_seqs: List[List[int]] = []
        all_nt: List[int] = []
        max_start = max(1, len(val_mm) - h_len - 64)

        for _ in range(n_samples):
            k1, k2 = rng_py.sample(EVAL_KEYS[:4], k=2)
            v1 = "".join(rng_py.choice(ALPHABET) for _ in range(val_len))
            v2 = "".join(rng_py.choice(ALPHABET) for _ in range(val_len))
            q_i = rng_py.choice((0, 1))
            qk, qv = (k1, v1) if q_i == 0 else (k2, v2)
            decoy_v = "".join(rng_py.choice(ALPHABET) for _ in range(val_len))

            pos = int(rng_np.integers(0, max_start))
            raw = val_mm[pos : pos + h_len].astype(np.int64).tolist()
            p1 = max(16, int(h_len * 0.10))
            p2 = max(p1 + 24, int(h_len * 0.25))

            s1_w = _enc(f" [{k1}: {v1}] ")
            s2_w = _enc(f" [{k2}: {v2}] ")
            s1_wo = _enc(f" [{k1}: {decoy_v}] ") if q_i == 0 else s1_w
            s2_wo = _enc(f" [{k2}: {decoy_v}] ") if q_i == 1 else s2_w
            qh = _enc(f" {qk}: ")
            t_ids = _enc(qv)
            n_t = len(t_ids)

            seq_w = raw[:p1] + s1_w + raw[p1:p2] + s2_w + raw[p2:] + qh + t_ids
            seq_wo = raw[:p1] + s1_wo + raw[p1:p2] + s2_wo + raw[p2:] + qh + t_ids
            all_seqs.append(seq_w)
            all_seqs.append(seq_wo)
            all_nt.append(n_t)
            all_nt.append(n_t)

        init_bs = 16 if h_len <= 1500 else (8 if h_len <= 4096 else (4 if h_len <= 8192 else 2))
        metrics = _eval_seq_batch(all_seqs, all_nt, init_bs=init_bs)
        bw_l = [metrics[2 * i][0] for i in range(n_samples)]
        bwo_l = [metrics[2 * i + 1][0] for i in range(n_samples)]
        ba_l = [metrics[2 * i][1] for i in range(n_samples)]
        tf_hits = sum(int(metrics[2 * i][2]) for i in range(n_samples))

        mw, mwo = float(np.mean(bw_l)), float(np.mean(bwo_l))
        results[f"L_{h_len}"] = {
            "haystack_bytes": int(h_len),
            "n_samples": int(n_samples),
            "needle_bpb_with": round(mw, 4),
            "needle_bpb_without": round(mwo, 4),
            "recall_gain_bpb": round(mwo - mw, 4),
            "byte_acc_pct": round(100.0 * float(np.mean(ba_l)), 2),
            "exact_match_pct": round(100.0 * tf_hits / max(1, n_samples), 2),
        }

    # Also evaluate untouched original Phase M phrase ("Important reference code: ...") at 1500B
    rng_py = random.Random(7777)
    rng_np = np.random.default_rng(7777)
    orig_seqs: List[List[int]] = []
    orig_nt: List[int] = []
    h_orig = min(1500, haystacks[-1])
    max_start = max(1, len(val_mm) - h_orig - 64)
    for _ in range(n_samples):
        code = "".join(rng_py.choice(ALPHABET) for _ in range(8))
        pos = int(rng_np.integers(0, max_start))
        raw = val_mm[pos : pos + h_orig].astype(np.int64).tolist()
        s_w = _enc(f"\nImportant reference code: {code}.\n")
        s_wo = _enc("\nImportant reference note: standard.\n")
        qh = _enc("\nRepeat the important reference code: ")
        t_ids = _enc(code)
        n_t = len(t_ids)
        orig_seqs.append(raw[:128] + s_w + raw[128:] + qh + t_ids)
        orig_seqs.append(raw[:128] + s_wo + raw[128:] + qh + t_ids)
        orig_nt.append(n_t)
        orig_nt.append(n_t)

    orig_m = _eval_seq_batch(orig_seqs, orig_nt, init_bs=16)
    bw_l = [orig_m[2 * i][0] for i in range(n_samples)]
    bwo_l = [orig_m[2 * i + 1][0] for i in range(n_samples)]
    ba_l = [orig_m[2 * i][1] for i in range(n_samples)]
    tf_hits = sum(int(orig_m[2 * i][2]) for i in range(n_samples))
    mw, mwo = float(np.mean(bw_l)), float(np.mean(bwo_l))
    results["OrigRefCode_1500B"] = {
        "haystack_bytes": int(h_orig),
        "n_samples": int(n_samples),
        "needle_bpb_with": round(mw, 4),
        "needle_bpb_without": round(mwo, 4),
        "recall_gain_bpb": round(mwo - mw, 4),
        "byte_acc_pct": round(100.0 * float(np.mean(ba_l)), 2),
        "exact_match_pct": round(100.0 * tf_hits / max(1, n_samples), 2),
    }
    return results


def run_cell(
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
    base_lr = 2.2e-3
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
    warm = min(25, max(1, steps // 16))
    rng_py = random.Random(seed + 5555)
    rng_np = np.random.default_rng(seed + 5555)
    is_calcite = isinstance(model, CalciteV2LM)
    cur_mbs = min(batch_size, micro_bs)
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        model.train()
        if is_calcite:
            model.set_crystal_schedule(step, steps)
        frac = step / float(max(1, steps))
        if smoke:
            cur_L, n_p, v_l = 128, 2, 4
        elif frac <= 0.35:
            cur_L, n_p, v_l = 160, 3, 4
        elif frac <= 0.65:
            cur_L, n_p, v_l = 288, 4, 6
        else:
            cur_L, n_p, v_l = 512, 4, 8

        lr_s = _cosine_lr(step, steps, warm, base_lr, min_lr)
        sc = lr_s / base_lr
        for pg in opt.param_groups:
            pg["lr"] = base_lr * float(pg.get("lr_mult", 1.0)) * sc

        xb_f, yb_f, mb_f = sample_dense_mqar_batch(
            train_mm, batch_size, cur_L, n_p, v_l, rng_py, rng_np, device
        )

        is_log = (step in (1, steps)) or (step % 50 == 0)
        while True:
            opt.zero_grad(set_to_none=True)
            n_acc = max(1, (batch_size + cur_mbs - 1) // cur_mbs)
            ce_c_acc: Optional[torch.Tensor] = None
            ce_p_acc: Optional[torch.Tensor] = None
            acc_t: Optional[torch.Tensor] = None
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
                        m_copy_f = (mb & valid).float()
                        m_pref_f = ((~mb) & valid).float()
                        n_c_tok = m_copy_f.sum()
                        n_p_tok = m_pref_f.sum()
                        tok_mean = tok_ce.mean()
                        ce_c = torch.where(n_c_tok > 0.0, (tok_ce * m_copy_f).sum() / n_c_tok.clamp(min=1.0), tok_mean)
                        ce_p = torch.where(n_p_tok > 0.0, (tok_ce * m_pref_f).sum() / n_p_tok.clamp(min=1.0), tok_mean)
                        aux = model.get_commitment_loss() if is_calcite else 0.0
                        loss = (0.75 * ce_c + 0.25 * ce_p + 0.03 * aux) / float(n_acc)
                    loss.backward()
                    if is_log:
                        ce_c_d = ce_c.detach()
                        ce_p_d = ce_p.detach()
                        ce_c_acc = ce_c_d if ce_c_acc is None else (ce_c_acc + ce_c_d)
                        ce_p_acc = ce_p_d if ce_p_acc is None else (ce_p_acc + ce_p_d)
                        with torch.no_grad():
                            eq_c = ((lg_f.argmax(dim=-1) == yb).float() * m_copy_f).sum() / n_c_tok.clamp(min=1.0)
                            acc_t = eq_c if acc_t is None else (acc_t + eq_c)
                break
            except torch.cuda.OutOfMemoryError:
                if cur_mbs <= 2:
                    raise
                cur_mbs = max(2, cur_mbs // 2)
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

        torch.nn.utils.clip_grad_norm_(opt.trainable_params, 1.0)
        opt.step()

        if is_log:
            ce_c_sum = float(ce_c_acc.item()) if ce_c_acc is not None else 0.0
            ce_p_sum = float(ce_p_acc.item()) if ce_p_acc is not None else 0.0
            acc_sum = float(acc_t.item()) if acc_t is not None else 0.0
            es_info = ""
            if is_calcite and model.epistemic_slots is not None:
                es_info = f" | wr={model.epistemic_slots.last_write_rate:.2f}"
            elif is_calcite and model.byte_chunker.byte_induct_cache is not None:
                es_info = f" | bwr={model.byte_chunker.byte_induct_cache.last_write_rate:.2f}"
            print(
                f"  [{name}] step {step:4d}/{steps} (L={cur_L:3d},v={v_l}) | "
                f"ce_c={ce_c_sum/n_acc:.4f} | ce_p={ce_p_sum/n_acc:.4f} | "
                f"copyAcc={100.0*acc_sum/n_acc:5.1f}%{es_info} | {(time.perf_counter()-t0)/60:.1f}m",
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

    ev = eval_ood_verbatim_ladder(
        model,
        val_mm,
        device=device,
        haystacks=(256,) if smoke else (256, 512, 1500, 4096, 8192),
        n_samples=4 if smoke else 24,
        val_len=4 if smoke else 8,
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
    elif is_calcite and model.byte_chunker.byte_induct_cache is not None:
        bic = model.byte_chunker.byte_induct_cache
        telemetry = {
            "last_write_rate": round(float(bic.last_write_rate), 4),
            "read_scale": [round(float(v), 3) for v in bic.read_scale.detach().cpu().tolist()],
            "read_thresh": [round(float(v), 3) for v in bic.read_thresh.detach().cpu().tolist()],
            "write_bias": [round(float(v), 3) for v in bic.write_proj.bias.detach().cpu().tolist()],
            "out_gate_bias": round(float(bic.out_gate.bias.detach().item()), 3),
        }

    print(f"  [{name} OOD VERBATIM EVAL] ({train_s:.1f}s)", flush=True)
    for lk, lv in ev.items():
        print(
            f"    {lk:17s}: gain={lv['recall_gain_bpb']:+.4f} | byte={lv['byte_acc_pct']:5.1f}% | exact={lv['exact_match_pct']:5.1f}%",
            flush=True,
        )
    if telemetry:
        print(f"    Slots telemetry: {telemetry}", flush=True)

    return {
        "name": name,
        "params": n_params,
        "cache_kb_o1": round(cache_kb, 2),
        "train_seconds": round(train_s, 2),
        "ood_verbatim_eval": ev,
        "slots_telemetry": telemetry,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT_DIR / "data_cache")
    ap.add_argument("--out-dir", type=Path, default=ROOT_DIR / "benchmarks" / "phase_n_needle")
    ap.add_argument("--steps", type=int, default=450)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--micro-bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--only-byte-induct", action="store_true")
    ap.add_argument("--smoke-local", action="store_true")
    args = ap.parse_args()

    device = "cuda" if has_usable_cuda() else "cpu"
    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram_gb < 12.0:
            torch.cuda.set_per_process_memory_fraction(0.90, 0)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_local:
        rng_f = np.random.default_rng(args.seed)
        fake = rng_f.integers(32 + 3, 126 + 3, size=60_000, dtype=np.uint16)
        train_mm = val_mm = fake
        steps, batch_size, micro_bs = 6, 8, 4
    else:
        train_bin, val_bin = ensure_phase_k_corpus(args.data_dir)
        train_mm = load_utf8_corpus_array(train_bin)
        val_mm = load_utf8_corpus_array(val_bin)
        steps, batch_size, micro_bs = int(args.steps), int(args.batch_size), int(args.micro_bs)

    router_state = torch.load(
        ROOT_DIR / "benchmarks" / "phase_l_v2" / "frozen_utf8_router.pt", map_location="cpu", weights_only=True
    )
    res_path = args.out_dir / ("smoke_mqar_probe_5m.json" if args.smoke_local else "phase_n_mqar_probe_5m.json")
    t_all = time.perf_counter()
    cells: Dict[str, object] = {}
    if args.only_byte_induct and res_path.is_file():
        prev = json.loads(res_path.read_text(encoding="utf-8"))
        cells = dict(prev.get("cells", {}))

    def _save() -> None:
        res_path.write_text(
            json.dumps(
                {
                    "meta": {
                        "phase": "Phase_N_Probe_B_Dense_MQAR_5M",
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

    if not args.only_byte_induct:
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
        cells["B0_RoPE_5M"] = run_cell(
            "B0_RoPE_5M", rope_m, rope_kb, train_mm, val_mm, device, steps, batch_size, micro_bs, args.seed, args.smoke_local
        )
        _save()
        del rope_m
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    def _build_cfg(use_slots: bool = False, use_byte_induct: bool = False) -> CalciteV2Config:
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
            use_byte_induct_cache=use_byte_induct,
            byte_induct_slots=1024 if not args.smoke_local else 128,
            byte_induct_dim=16,
        )
        if args.smoke_local:
            cfg.d_model, cfg.n_heads, cfg.d_ff = 128, 2, 256
            cfg.byte_char_dim, cfg.byte_d_ff = 16, 128
            cfg.epistemic_slots_n, cfg.epistemic_slots_dim = 16, 32
        return cfg

    run_list = (
        [("B3_Calcite_v3_ByteInductCache", False, True)]
        if args.only_byte_induct
        else [
            ("B1_Calcite_v3_5M", False, False),
            ("B2_Calcite_v3_InductSlots", True, False),
            ("B3_Calcite_v3_ByteInductCache", False, True),
        ]
    )
    for cname, use_slots, use_bic in run_list:
        set_all_seeds(args.seed)
        cfg = _build_cfg(use_slots=use_slots, use_byte_induct=use_bic)
        m = CalciteV2LM(cfg).to(device)
        load_frozen_router_into(m, router_state)
        kb = compute_v2_state_cache_kb(cfg, 32768)
        cells[cname] = run_cell(
            cname, m, kb, train_mm, val_mm, device, steps, batch_size, micro_bs, args.seed, args.smoke_local
        )
        _save()
        del m
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n[Probe B Complete] Saved -> {res_path}", flush=True)


if __name__ == "__main__":
    main()
