"""
Tracer #2 (`experiments/v5_proto/probe_phase_a_v5.py`):
Fast Phase-A + Physical `2-bit uint8 T-MAC LUT` Verification Probe (`Calcite-5 System-1 Core`).

Trains ONLY `SimVQSplitLM` and `SimVQSplit_CrystalOnly` (`STEPS=2000, BS=64, SEQ_LEN=192, seed=2026`),
evaluates `FP`, `0-QAT PTQ (LS)`, `300-step progressive STE QAT`, and physical `PackedLUT2BitLinear`
(`uint8` T-MAC addition/subtraction-only inference) against the `OEB` acceptance gates before running
the full 7-model official tournament.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import random
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_simvq_split_benchmark import (  # noqa: E402
    build_shared_training_stream,
    compute_state_cache_kb,
    count_params,
    harmonic_mean,
    measure_crystal_drift_at_1024,
    tune_simvq_ablation_dff,
)
from simvq_split.eval_utils import (  # noqa: E402
    evaluate_prose_perplexity,
    evaluate_suite_batched,
    wilson_ci_95,
)
from simvq_split.memory import (  # noqa: E402
    compute_2bit_lut_footprint,
    pack_model_2bit_lut,
    quantize_ternary_weights_ls,
)
from simvq_split.models import SimVQSplitConfig, SimVQSplitLM  # noqa: E402
from simvq_split.tasks import (  # noqa: E402
    build_eval_overwrite_suite,
    build_eval_s5_suite,
    build_eval_trap_suite,
    load_gsm8k_corpora,
    make_overwrite_story,
    make_s5_story,
    make_trap_overwrite_story,
)
from simvq_split.tokenizer import ByteBPETokenizer  # noqa: E402


def train_and_eval_fp(
    name: str,
    model: SimVQSplitLM,
    tok: ByteBPETokenizer,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    te_docs: list[str],
    s5_suites: dict,
    ov_suites: dict,
    trap_suites: dict,
    steps: int,
    batch_size: int,
    seq_len: int,
    dev: str,
) -> dict:
    n_p = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=0.01)
    use_amp = dev == "cuda" and torch.cuda.is_available()
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        model.train()
        model.set_crystal_schedule(step, steps)
        alpha = model.crystal.snap_alpha if model.crystal is not None else 0.0

        lr = 2.5e-3 * 0.5 * (1.0 + math.cos(math.pi * step / steps))
        if step < 60:
            lr = 2.5e-3 * (step / 60.0)
        for pg in opt.param_groups:
            pg["lr"] = lr

        sl = slice((step - 1) * batch_size, step * batch_size)
        xb = X_train[sl].to(dev)
        yb = Y_train[sl].to(dev)
        opt.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(xb)
                tok_ce = F.cross_entropy(
                    logits.transpose(1, 2),
                    yb,
                    ignore_index=tok.PAD_ID,
                    reduction="none",
                )
                denom = (yb != tok.PAD_ID).sum(dim=1).clamp(min=1)
                loss = (tok_ce.sum(dim=1) / denom).mean() + 0.04 * model.get_commitment_loss()
        else:
            logits = model(xb)
            tok_ce = F.cross_entropy(
                logits.transpose(1, 2),
                yb,
                ignore_index=tok.PAD_ID,
                reduction="none",
            )
            denom = (yb != tok.PAD_ID).sum(dim=1).clamp(min=1)
            loss = (tok_ce.sum(dim=1) / denom).mean() + 0.04 * model.get_commitment_loss()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 500 == 0:
            print(
                f"  [{name:24s} | {n_p:,}p] step {step:4d}/{steps} ({time.perf_counter() - t0:5.1f}s) alpha={alpha:.2f} loss={loss.item():.3f}",
                flush=True,
            )

    train_sec = time.perf_counter() - t0
    model.set_crystal_schedule(steps, steps)
    model.set_snap_alpha(1.0)

    ppl, nll, eval_toks = evaluate_prose_perplexity(
        model, tok, te_docs[:500], seq_len=seq_len, device=dev
    )
    s5_res = {}
    for k_key, suite in s5_suites.items():
        ex_pct, tk_pct, k_ok, n_tot = evaluate_suite_batched(
            model, suite, pad_id=tok.PAD_ID, device=dev
        )
        ci = wilson_ci_95(k_ok, n_tot)
        s5_res[k_key] = {
            "exact_pct": round(ex_pct, 2),
            "token_pct": round(tk_pct, 2),
            "ok": k_ok,
            "n": n_tot,
            "wilson_95": [round(ci[0], 2), round(ci[1], 2)],
        }
    ov_res = {}
    for o_key, suite in ov_suites.items():
        ex_pct, tk_pct, k_ok, n_tot = evaluate_suite_batched(
            model, suite, pad_id=tok.PAD_ID, device=dev
        )
        ci = wilson_ci_95(k_ok, n_tot)
        ov_res[o_key] = {
            "exact_pct": round(ex_pct, 2),
            "token_pct": round(tk_pct, 2),
            "ok": k_ok,
            "n": n_tot,
            "wilson_95": [round(ci[0], 2), round(ci[1], 2)],
        }
    trap_res = {}
    for t_key, suite in trap_suites.items():
        ex_pct, tk_pct, k_ok, n_tot = evaluate_suite_batched(
            model, suite, pad_id=tok.PAD_ID, device=dev
        )
        ci = wilson_ci_95(k_ok, n_tot)
        trap_res[t_key] = {
            "exact_pct": round(ex_pct, 2),
            "token_pct": round(tk_pct, 2),
            "ok": k_ok,
            "n": n_tot,
            "wilson_95": [round(ci[0], 2), round(ci[1], 2)],
        }

    sel_f1_in = round(
        harmonic_mean(ov_res["in_dist"]["exact_pct"], trap_res["in_dist"]["exact_pct"]),
        2,
    )
    sel_f1_long = round(
        harmonic_mean(ov_res["ood_long"]["exact_pct"], trap_res["ood_long"]["exact_pct"]),
        2,
    )
    drift = measure_crystal_drift_at_1024(model, tok.vocab_size, dev)
    s5_sum = round(sum(s5_res[k]["exact_pct"] for k in ("k2", "k4", "k8", "k16")), 2)

    print(
        f"-> {name} FP ({train_sec:.1f}s): PPL={ppl:.2f} | "
        f"S5={s5_res['k2']['exact_pct']:.1f}/{s5_res['k4']['exact_pct']:.1f}/{s5_res['k8']['exact_pct']:.1f}/{s5_res['k16']['exact_pct']:.1f} (sum={s5_sum:.1f}) | "
        f"OV={ov_res['in_dist']['exact_pct']:.1f}/{ov_res['ood_med']['exact_pct']:.1f}/{ov_res['ood_long']['exact_pct']:.1f} | "
        f"Trap={trap_res['in_dist']['exact_pct']:.1f}/{trap_res['ood_long']['exact_pct']:.1f} | "
        f"SelF1={sel_f1_in:.1f}/{sel_f1_long:.1f} | drift=({drift['norm_drift_L1024']:.2e}, {drift['voronoi_lattice_err_L1024']:.2e})",
        flush=True,
    )
    return {
        "params": n_p,
        "train_seconds": round(train_sec, 2),
        "gsm8k_ppl": round(ppl, 3),
        "s5": s5_res,
        "s5_sum": s5_sum,
        "overwrite": ov_res,
        "trap_unknown": trap_res,
        "selective_f1_in_dist": sel_f1_in,
        "selective_f1_ood_long": sel_f1_long,
        "drift_1024": drift,
    }


def eval_ptq_ls(
    model: SimVQSplitLM,
    tok: ByteBPETokenizer,
    te_docs: list[str],
    s5_k8,
    ov_med,
    trap_long,
    dev: str,
) -> dict:
    q_model = copy.deepcopy(model).to(dev).eval()
    q_model.set_snap_alpha(1.0)
    quantized_params = 0
    with torch.no_grad():
        for name, mod in q_model.named_modules():
            if isinstance(mod, nn.Linear) and name != "lm_head":
                w_tern, _, _ = quantize_ternary_weights_ls(mod.weight.data)
                g_size = 32 if mod.weight.numel() % 32 == 0 else mod.weight.size(-1)
                for row in w_tern.float().reshape(-1, g_size):
                    assert torch.unique(torch.round(row * 1e5)).numel() <= 3
                mod.weight.data.copy_(w_tern)
                quantized_params += mod.weight.numel()

    ppl, _, _ = evaluate_prose_perplexity(q_model, tok, te_docs[:400], seq_len=192, device=dev)
    s5, _, _, _ = evaluate_suite_batched(q_model, s5_k8, pad_id=tok.PAD_ID, device=dev)
    ov, _, _, _ = evaluate_suite_batched(q_model, ov_med, pad_id=tok.PAD_ID, device=dev)
    trap, _, _, _ = evaluate_suite_batched(q_model, trap_long, pad_id=tok.PAD_ID, device=dev)
    res = {
        "ternary_quantized_linear_params": quantized_params,
        "gsm8k_ppl_158bit_ptq": round(ppl, 3),
        "s5_k8_158bit_ptq_pct": round(s5, 2),
        "ov_ood_med_158bit_ptq_pct": round(ov, 2),
        "trap_ood_long_158bit_ptq_pct": round(trap, 2),
    }
    print(f"-> 1.58-bit PTQ (LS): {res}", flush=True)
    return res


def eval_qat_and_tmac_lut(
    model: SimVQSplitLM,
    tok: ByteBPETokenizer,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    te_docs: list[str],
    s5_k2,
    s5_k8,
    ov_med,
    ov_long,
    trap_long,
    dev: str,
    total_steps: int,
    batch_size: int,
) -> tuple[dict, dict]:
    QAT_STEPS = 300
    Q_LR = 1.5e-4
    torch.manual_seed(777)
    qat_model = copy.deepcopy(model).to(dev)
    qat_model.set_crystal_schedule(total_steps, total_steps)
    qat_model.set_snap_alpha(1.0)
    qat_model.set_qat_mix(0.30)
    qat_model.set_ternary_qat(True)

    opt = torch.optim.AdamW(qat_model.parameters(), lr=Q_LR, weight_decay=0.0)
    use_amp = dev == "cuda" and torch.cuda.is_available()
    start_row = (total_steps - QAT_STEPS) * batch_size
    t0 = time.perf_counter()

    for q_step in range(1, QAT_STEPS + 1):
        qat_model.train()
        mix = min(1.0, 0.30 + 0.70 * (q_step / 90.0)) if q_step <= 90 else 1.0
        qat_model.set_qat_mix(mix)

        if q_step <= 15:
            lr = Q_LR * (q_step / 15.0)
        else:
            lr = Q_LR * 0.1 + 0.5 * (Q_LR * 0.9) * (
                1.0 + math.cos(math.pi * (q_step - 15) / (QAT_STEPS - 15))
            )
        for pg in opt.param_groups:
            pg["lr"] = lr

        sl = slice(start_row + (q_step - 1) * batch_size, start_row + q_step * batch_size)
        xb = X_train[sl].to(dev)
        yb = Y_train[sl].to(dev)
        opt.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = qat_model(xb)
                tok_ce = F.cross_entropy(
                    logits.transpose(1, 2),
                    yb,
                    ignore_index=tok.PAD_ID,
                    reduction="none",
                )
                denom = (yb != tok.PAD_ID).sum(dim=1).clamp(min=1)
                loss = (tok_ce.sum(dim=1) / denom).mean() + 0.04 * qat_model.get_commitment_loss()
        else:
            logits = qat_model(xb)
            tok_ce = F.cross_entropy(
                logits.transpose(1, 2),
                yb,
                ignore_index=tok.PAD_ID,
                reduction="none",
            )
            denom = (yb != tok.PAD_ID).sum(dim=1).clamp(min=1)
            loss = (tok_ce.sum(dim=1) / denom).mean() + 0.04 * qat_model.get_commitment_loss()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(qat_model.parameters(), 1.0)
        opt.step()

        if q_step in (90, 150, QAT_STEPS):
            print(
                f"  [1.58-bit STE QAT] step {q_step:3d}/{QAT_STEPS} mix={qat_model.qat_mix:.2f} ({time.perf_counter() - t0:5.1f}s) loss={loss.item():.3f}",
                flush=True,
            )

    qat_model.set_ternary_qat(False)
    qat_model.set_qat_mix(1.0)
    with torch.no_grad():
        quantized_params = 0
        for name, mod in qat_model.named_modules():
            if isinstance(mod, nn.Linear) and name != "lm_head":
                w_tern, _, _ = quantize_ternary_weights_ls(mod.weight.data)
                g_size = 32 if mod.weight.numel() % 32 == 0 else mod.weight.size(-1)
                for row in w_tern.float().reshape(-1, g_size):
                    assert torch.unique(torch.round(row * 1e5)).numel() <= 3, (
                        f"Baked ternary weight in {name} exceeded 3 levels!"
                    )
                mod.weight.data.copy_(w_tern)
                quantized_params += mod.weight.numel()

    qat_model.eval()
    qat_model.set_snap_alpha(1.0)

    def _eval_bundle(m: nn.Module) -> dict:
        with torch.no_grad():
            ppl, _, _ = evaluate_prose_perplexity(m, tok, te_docs[:500], seq_len=192, device=dev)
            k2, _, _, _ = evaluate_suite_batched(m, s5_k2, pad_id=tok.PAD_ID, device=dev)
            k8, _, _, _ = evaluate_suite_batched(m, s5_k8, pad_id=tok.PAD_ID, device=dev)
            om, _, _, _ = evaluate_suite_batched(m, ov_med, pad_id=tok.PAD_ID, device=dev)
            ol, _, _, _ = evaluate_suite_batched(m, ov_long, pad_id=tok.PAD_ID, device=dev)
            tl, _, _, _ = evaluate_suite_batched(m, trap_long, pad_id=tok.PAD_ID, device=dev)
        return {
            "ppl": round(ppl, 3),
            "s5_k2": round(k2, 2),
            "s5_k8": round(k8, 2),
            "ov_med": round(om, 2),
            "ov_long": round(ol, 2),
            "trap_long": round(tl, 2),
        }

    qat_bundle = _eval_bundle(qat_model)
    qat_stats = {
        "qat_steps": QAT_STEPS,
        "qat_seconds": round(time.perf_counter() - t0, 2),
        "ternary_quantized_linear_params": quantized_params,
        "gsm8k_ppl_158bit_qat": qat_bundle["ppl"],
        "s5_k2_158bit_qat_pct": qat_bundle["s5_k2"],
        "s5_k8_158bit_qat_pct": qat_bundle["s5_k8"],
        "ov_ood_med_158bit_qat_pct": qat_bundle["ov_med"],
        "ov_ood_long_158bit_qat_pct": qat_bundle["ov_long"],
        "trap_ood_long_158bit_qat_pct": qat_bundle["trap_long"],
    }
    print(f"-> 1.58-bit STE QAT (baked LS): {qat_stats}", flush=True)

    lut_model = pack_model_2bit_lut(qat_model).to(dev)
    fp = compute_2bit_lut_footprint(lut_model)
    lut_bundle = _eval_bundle(lut_model)
    max_pp_diff = max(
        abs(lut_bundle[k] - qat_bundle[k])
        for k in ("s5_k2", "s5_k8", "ov_med", "ov_long", "trap_long")
    )
    lut_stats = {
        **fp,
        "gsm8k_ppl_2bit_lut": lut_bundle["ppl"],
        "s5_k2_2bit_lut_pct": lut_bundle["s5_k2"],
        "s5_k8_2bit_lut_pct": lut_bundle["s5_k8"],
        "ov_ood_med_2bit_lut_pct": lut_bundle["ov_med"],
        "ov_ood_long_2bit_lut_pct": lut_bundle["ov_long"],
        "trap_ood_long_2bit_lut_pct": lut_bundle["trap_long"],
        "max_pp_diff_vs_qat": round(max_pp_diff, 3),
        "ppl_diff_vs_qat": round(abs(lut_bundle["ppl"] - qat_bundle["ppl"]), 4),
    }
    print(f"-> 2-bit uint8 T-MAC LUT: {lut_stats}", flush=True)
    return qat_stats, lut_stats


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Tracer #2 running on device={dev}", flush=True)

    tr_docs, te_docs, tr_sents, te_sents = load_gsm8k_corpora(REPO_ROOT)
    rng = random.Random(42)
    tok_corpus = list(tr_docs[:2000])
    for _ in range(500):
        p, t, _ = make_s5_story(rng, rng.randint(1, 4), tr_sents, (0, 1))
        tok_corpus.append(p + t)
        p2, t2, _ = make_overwrite_story(rng, 4, 2, tr_sents, 2, 0.3)
        tok_corpus.append(p2 + t2)
    for _ in range(150):
        p3, t3, _ = make_trap_overwrite_story(rng, 4, 2, tr_sents, 2, 0.3)
        tok_corpus.append(p3 + t3)
    tok = ByteBPETokenizer.train_from_iterator(tok_corpus, vocab_size=1536)

    STEPS, BS, SEQ_LEN = 2000, 64, 192
    X_train, Y_train = build_shared_training_stream(
        tok, tr_docs, tr_sents, n_steps=STEPS, batch_size=BS, seq_len=SEQ_LEN, seed=2026
    )

    N_EVAL = 200
    s5_suites = {
        "k2": build_eval_s5_suite(tok, te_sents, 2, N_EVAL, (0, 1), 3100),
        "k4": build_eval_s5_suite(tok, te_sents, 4, N_EVAL, (0, 1), 3200),
        "k8": build_eval_s5_suite(tok, te_sents, 8, N_EVAL, (1, 1), 3300),
        "k16": build_eval_s5_suite(tok, te_sents, 16, N_EVAL, (1, 2), 3400),
    }
    ov_suites = {
        "in_dist": build_eval_overwrite_suite(tok, te_sents, 4, 2, 2, N_EVAL, 4100),
        "ood_med": build_eval_overwrite_suite(tok, te_sents, 7, 4, 10, N_EVAL, 4200),
        "ood_long": build_eval_overwrite_suite(tok, te_sents, 9, 5, 18, N_EVAL, 4300),
    }
    trap_suites = {
        "in_dist": build_eval_trap_suite(tok, te_sents, 4, 2, 2, N_EVAL, 5100),
        "ood_long": build_eval_trap_suite(tok, te_sents, 9, 5, 18, N_EVAL, 5300),
    }

    base_scfg = SimVQSplitConfig(
        vocab_size=tok.vocab_size,
        d_model=160,
        n_heads=4,
        n_layers_pre=2,
        n_layers_post=1,
        window_size=12,
        num_spheres=24,
        d_sphere=32,
        codebook_size=32,
        mem_heads=5,
        mem_dim=32,
        ridge_lambda=0.04,
        use_crystal=True,
        use_woodbury=True,
    )
    target_params = count_params(SimVQSplitLM(base_scfg))
    cryst_only_cfg = tune_simvq_ablation_dff(
        base_scfg, use_crystal=True, use_woodbury=False, target_params=target_params
    )

    torch.manual_seed(100)
    full_model = SimVQSplitLM(base_scfg).to(dev)
    full_fp = train_and_eval_fp(
        "SimVQSplitLM",
        full_model,
        tok,
        X_train,
        Y_train,
        te_docs,
        s5_suites,
        ov_suites,
        trap_suites,
        STEPS,
        BS,
        SEQ_LEN,
        dev,
    )

    torch.manual_seed(100)
    cryst_model = SimVQSplitLM(cryst_only_cfg).to(dev)
    cryst_fp = train_and_eval_fp(
        "SimVQSplit_CrystalOnly",
        cryst_model,
        tok,
        X_train,
        Y_train,
        te_docs,
        s5_suites,
        ov_suites,
        trap_suites,
        STEPS,
        BS,
        SEQ_LEN,
        dev,
    )

    ptq_res = eval_ptq_ls(
        full_model,
        tok,
        te_docs,
        s5_suites["k8"],
        ov_suites["ood_med"],
        trap_suites["ood_long"],
        dev,
    )
    qat_res, lut_res = eval_qat_and_tmac_lut(
        full_model,
        tok,
        X_train,
        Y_train,
        te_docs,
        s5_suites["k2"],
        s5_suites["k8"],
        ov_suites["ood_med"],
        ov_suites["ood_long"],
        trap_suites["ood_long"],
        dev,
        STEPS,
        BS,
    )

    gates = {
        "fp_ppl_le_19": full_fp["gsm8k_ppl"] <= 19.0,
        "fp_s5_sum_ge_231_5": full_fp["s5_sum"] >= 231.5,
        "fp_ov_long_ge_95": full_fp["overwrite"]["ood_long"]["exact_pct"] >= 95.0,
        "fp_trap_long_ge_96": full_fp["trap_unknown"]["ood_long"]["exact_pct"] >= 96.0,
        "fp_drift_lt_1e6": (
            full_fp["drift_1024"]["norm_drift_L1024"] < 1e-6
            and full_fp["drift_1024"]["voronoi_lattice_err_L1024"] < 1e-6
        ),
        "fp_runtime_le_850s": full_fp["train_seconds"] <= 850.0,
        "cryst_only_improved": (
            cryst_fp["gsm8k_ppl"] < 20.65
            and (
                cryst_fp["s5"]["k2"]["exact_pct"] >= 10.0
                or cryst_fp["selective_f1_in_dist"] >= 37.0
            )
        ),
        "ptq_ppl_lt_35": ptq_res["gsm8k_ppl_158bit_ptq"] < 35.0,
        "ptq_s5_k8_gt_25": ptq_res["s5_k8_158bit_ptq_pct"] > 25.0,
        "qat_s5_k8_ge_45": qat_res["s5_k8_158bit_qat_pct"] >= 45.0,
        "qat_trap_long_ge_88": qat_res["trap_ood_long_158bit_qat_pct"] >= 88.0,
        "lut_s5_k8_ge_45": lut_res["s5_k8_2bit_lut_pct"] >= 45.0,
        "lut_trap_long_ge_88": lut_res["trap_ood_long_2bit_lut_pct"] >= 88.0,
        "lut_vs_qat_diff_le_1_5pp": lut_res["max_pp_diff_vs_qat"] <= 1.5,
    }
    print(f"[ACCEPTANCE CHECK] {gates}", flush=True)

    out_path = Path(__file__).resolve().parent / "probe_phase_a_v5_results.json"
    out_path.write_text(
        json.dumps(
            {
                "SimVQSplitLM": full_fp,
                "SimVQSplit_CrystalOnly": cryst_fp,
                "ptq_158bit_ls": ptq_res,
                "qat_158bit_ls": qat_res,
                "lut_2bit_uint8": lut_res,
                "state_cache_kb": round(
                    compute_state_cache_kb("SimVQSplitLM", base_scfg, None, seq_len=2048),
                    2,
                ),
                "gates": gates,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved Tracer #2 results -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
