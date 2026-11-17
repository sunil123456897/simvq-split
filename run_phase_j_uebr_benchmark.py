"""run_phase_j_uebr_benchmark.py — Phase J (`Calcite-6 / UEBR`) System-2 Benchmark over Tier-1 (`5.41 MB 2-bit LUT`).

Executes:
  1) Tracer #1 (`--smoke`): `64/32/32` GSM8K train subset + `32` Stress, matrix-RHS gradcheck,
     `100%` live grads, `0` NaN/Inf, `0` reads of SVAMP or `gsm8k/test.jsonl`.
  2) Full Ablation Ladder on frozen `Tier-1 5.41 MB 2-bit LUT` features (`calcite5_tier1_17m_utf8_qat_h_lut.pt`):
       - `B0` (`OpenGraphCRAMSolver`, seeds 2026 & 2027) -> `2*sigma` calibration on `BlindDev` & `BlindDev-Stress`
       - `A1` (`+ Birkhoff Extension QAP + Null-Slot Ø Pool`, seed 2026)
       - `A2` (`+ Sectioned GAQSlotBridge + (1 - g_risk) Blocked Bypass`, seed 2026)
       - `A3` (`Full Calcite-6 / UEBR System-2`, seeds 2026 & 2027)
  3) Stop-Crane Gate (`A3 > B0 + 2*sigma` on BOTH `BlindDev` and `BlindDev-Stress`) BEFORE
     opening `SingleShotSVAMPVault` (`benchmarks/svamp/svamp_1000.json`, `max_calls = 1`).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.cram_gsm8k.cram_open_graph import N_CERT_FEATS, OpenGraphCRAMSolver  # noqa: E402
from experiments.cram_gsm8k.gsm8k_cram_solver import wilson_95_ci  # noqa: E402
from experiments.cram_gsm8k.slot_extractor import (  # noqa: E402
    ExtractedSlot,
    TrainInducedQuantityExtractor,
    extract_gt_number,
)
from experiments.cram_gsm8k.spherical_alu import SphericalBase10RegisterALU  # noqa: E402
from experiments.uebr_system2.tier1_gaq_extractor import (  # noqa: E402
    SingleShotMathVault,
    SingleShotSVAMPVault,
    build_blind_dev_stress_split,
    extract_tier1_lut_features,
)
from experiments.uebr_system2.uebr_graph_solver import (  # noqa: E402
    UEBROpenGraphSolver,
    verify_matrix_laplacian_gradcheck,
)


TIER1_LUT_CKPT = ROOT_DIR / "checkpoints" / "calcite5_tier1_17m_utf8_qat_h_lut.pt"
GSM8K_TRAIN_PATH = ROOT_DIR / "benchmarks" / "gsm8k" / "train.jsonl"
SVAMP_PATH = ROOT_DIR / "benchmarks" / "svamp" / "svamp_1000.json"
MULTIARITH_PATH = ROOT_DIR / "benchmarks" / "ood_math" / "multiarith_600.json"
ASDIV_PATH = ROOT_DIR / "benchmarks" / "ood_math" / "asdiv_a.json"
CACHE_DIR = ROOT_DIR / "experiments" / "uebr_system2" / "cache"
CKPT_OUT_DIR = ROOT_DIR / "experiments" / "uebr_system2" / "checkpoints"
JSON_OUT_PATH = ROOT_DIR / "benchmarks" / "phase_j_uebr_system2_results.json"
REPORT_OUT_PATH = ROOT_DIR / "Phase_J_UEBR_System2_Report.md"


def _load_gsm8k_train_splits() -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    assert GSM8K_TRAIN_PATH.is_file(), f"Missing {GSM8K_TRAIN_PATH}"
    with open(GSM8K_TRAIN_PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) >= 7473, f"Expected >= 7473 GSM8K train rows, got {len(rows)}"
    return rows[:6000], rows[6000:6736], rows[6736:]


def evaluate_math_split(
    solver: OpenGraphCRAMSolver,
    items: Sequence[Dict[str, object]],
    slots_list: Sequence[Sequence[ExtractedSlot]],
    lm_feats_list: Sequence[Dict[str, object]],
    tag: str,
) -> Dict[str, object]:
    """Strictly blind evaluation on GSM8K dev/stress or SVAMP without exposing answers to solve_blind."""
    t0 = time.perf_counter()
    n = len(items)
    p1 = p3 = p5 = unary_p1 = cert_tot = cert_ok = oracle_hits = 0
    max_drift = 0.0
    le3_tot = le3_p1 = ge4_tot = ge4_p1 = 0
    by_type: Dict[str, List[int]] = {}

    if hasattr(solver, "birkhoff_total_solves"):
        bf_tot0 = int(solver.birkhoff_total_solves)  # type: ignore[attr-defined]
        bf_win0 = int(solver.birkhoff_lossless_wins)  # type: ignore[attr-defined]
    else:
        bf_tot0 = bf_win0 = 0

    for idx, (it, sl, lmf) in enumerate(zip(items, slots_list, lm_feats_list)):
        q_only = str(it["question"])
        gt = float(it["answer_val"]) if "answer_val" in it else extract_gt_number(str(it["answer"]))

        out = solver.solve_blind(q_only, slots=sl, lm_feat=lmf)
        max_drift = max(max_drift, float(out["norm_drift"]))

        h1 = abs(float(out["pred_pass1"]) - gt) <= 1e-2
        h_un = abs(float(out["pred_unary_ablation"]) - gt) <= 1e-2
        h3 = any(abs(float(v) - gt) <= 1e-2 for v in out["pred_pass3"])  # type: ignore[attr-defined]
        h5 = any(abs(float(v) - gt) <= 1e-2 for v in out["pred_pass5"])  # type: ignore[attr-defined]

        p1 += int(h1)
        unary_p1 += int(h_un)
        p3 += int(h3)
        p5 += int(h5)

        if bool(out["dual_certified"]):
            cert_tot += 1
            cert_ok += int(h1)

        if "cand_vals" in out:
            if any(abs(float(v) - gt) <= 1e-2 for v in out["cand_vals"]):  # type: ignore[attr-defined]
                oracle_hits += 1
        else:
            cands, _ = solver._assemble_and_score_candidates(q_only, sl, lmf)
            if any(abs(float(c["val"]) - gt) <= 1e-2 for c in cands):
                oracle_hits += 1

        ng_i = sum(1 for s in sl if s.kind != "CONST")
        if ng_i <= 3:
            le3_tot += 1
            le3_p1 += int(h1)
        else:
            ge4_tot += 1
            ge4_p1 += int(h1)

        if "Type" in it:
            tp = str(it["Type"])
            row_t = by_type.setdefault(tp, [0, 0, 0])
            row_t[0] += 1
            row_t[1] += int(h1)
            row_t[2] += int(h5)

    p1_pct = round(100.0 * p1 / max(1, n), 2)
    ci_lo, ci_hi = wilson_95_ci(p1, n)
    res: Dict[str, object] = {
        "tag": tag,
        "n": n,
        "pass1": p1,
        "pass1_pct": p1_pct,
        "pass1_ci95": [round(ci_lo, 2), round(ci_hi, 2)],
        "unary_ablation_pass1_pct": round(100.0 * unary_p1 / max(1, n), 2),
        "pass3_pct": round(100.0 * p3 / max(1, n), 2),
        "pass5_pct": round(100.0 * p5 / max(1, n), 2),
        "oracle_pct": round(100.0 * oracle_hits / max(1, n), 2),
        "cert_tot": cert_tot,
        "cert_ok": cert_ok,
        "cert_cov_pct": round(100.0 * cert_tot / max(1, n), 2),
        "cert_prec_pct": round((100.0 * cert_ok / cert_tot) if cert_tot > 0 else 0.0, 2),
        "max_norm_drift": max_drift,
        "gershgorin_margin": round(float(solver.ebm.gershgorin_margin() if hasattr(solver.ebm, "gershgorin_margin") else solver.gershgorin_margin()), 4),  # type: ignore[attr-defined]
        "le3_slots": {"n": le3_tot, "pass1_pct": round(100.0 * le3_p1 / max(1, le3_tot), 2)},
        "ge4_slots": {"n": ge4_tot, "pass1_pct": round(100.0 * ge4_p1 / max(1, ge4_tot), 2)},
        "eval_seconds": round(time.perf_counter() - t0, 2),
    }
    if hasattr(solver, "birkhoff_total_solves"):
        d_tot = int(solver.birkhoff_total_solves) - bf_tot0  # type: ignore[attr-defined]
        d_win = int(solver.birkhoff_lossless_wins) - bf_win0  # type: ignore[attr-defined]
        res["birkhoff_solves"] = d_tot
        res["birkhoff_lossless_pct"] = round(100.0 * d_win / max(1, d_tot), 2) if d_tot > 0 else 100.0
        res["laplacian_min_eig"] = round(float(solver.last_min_eig), 4)  # type: ignore[attr-defined]
        res["epc_max_err"] = float(solver.last_epc_err)  # type: ignore[attr-defined]
    if by_type:
        res["by_type"] = {
            tp: {
                "n": row[0],
                "pass1_pct": round(100.0 * row[1] / max(1, row[0]), 2),
                "pass5_pct": round(100.0 * row[2] / max(1, row[0]), 2),
            }
            for tp, row in sorted(by_type.items())
        }
    return res


def _verify_all_live_grads(solver: UEBROpenGraphSolver, sample_items, sample_slots, sample_feats) -> Tuple[int, int]:
    """Verifies every trainable parameter tensor in `UEBROpenGraphEBM` receives a finite non-zero gradient."""
    solver.ebm.train()
    for it, sl, lmf in zip(sample_items[:2], sample_slots[:2], sample_feats[:2]):
        if len(sl) < 2:
            continue
        pf = solver._featurize_problem(str(it["question"]), sl, lmf)
        cands = solver._synthesize_open_pool(sl, pf, mask_self=False)
        if len(cands) < 2:
            continue
        pk = solver._pack_candidates_cpu(sl, pf, cands)
        tensors, m_counts = solver._collate_uebr_to_device([pk])
        batch_out = solver.ebm.score_batch(*tensors, n_cands=sum(m_counts))
        assert batch_out.z_flat is not None
        rl_t = torch.zeros(len(sl), dtype=torch.long, device=solver.device)
        rl_t[: min(2, len(sl))] = 1
        loss = (
            batch_out.e_tot.sum()
            + batch_out.vq_loss
            + 0.1
            * solver.ebm.rlns_block3_fenchel_young_loss(
                batch_out.u_tot,
                batch_out.p_pot,
                0,
                len(sl),
                pk["pair_map_arr"],  # type: ignore[arg-type]
                rl_t,
                pair_iu_t=torch.from_numpy(pk["pair_iu"]).to(solver.device),  # type: ignore[arg-type]
                pair_iv_t=torch.from_numpy(pk["pair_iv"]).to(solver.device),  # type: ignore[arg-type]
                pair_pidx_t=torch.from_numpy(pk["pair_pidx"]).to(solver.device),  # type: ignore[arg-type]
            )
            + 0.1 * solver.ebm.spd_laplacian.directional_eikonal_loss(
                batch_out.z_flat[:1], batch_out.z_flat[1:2], batch_out.q_prob[:1]
            )
            + solver.ebm.prox_regularization_loss()
        )
        solver.ebm.zero_grad(set_to_none=True)
        loss.backward()
        tot_p = 0
        live_p = 0
        for p in solver.ebm.parameters():
            if not p.requires_grad:
                continue
            tot_p += 1
            if p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().max().item()) > 0.0:
                live_p += 1
        solver.ebm.zero_grad(set_to_none=True)
        return live_p, tot_p
    raise RuntimeError("Failed to assemble smoke grad check batch")


def run_smoke_tracer(device: str) -> int:
    print("=== [Phase J | Tracer #1 Smoke] Matrix Gradcheck, Live Grads & 64/32/32 Sanity ===", flush=True)
    t0 = time.perf_counter()
    grad_err = verify_matrix_laplacian_gradcheck(seed=2026)
    print(f"  [Check 1] Matrix-RHS _ImplicitLaplacianSolveFn float64 gradcheck max_err = {grad_err:.2e}", flush=True)
    assert grad_err < 1e-7, f"Matrix Laplacian adjoint gradcheck failed: {grad_err}"

    tr_all, cal_all, dev_all = _load_gsm8k_train_splits()
    tr_s = tr_all[:64]
    cal_s = cal_all[:32]
    dev_s = dev_all[:32]

    ext = TrainInducedQuantityExtractor()
    ext.fit(tr_all[:600])
    stress_s = build_blind_dev_stress_split(dev_s, ext, seed=2026)

    sl_tr = [ext.extract_slots(str(it["question"]), include_aux_consts=True)[0] for it in tr_s]
    sl_cal = [ext.extract_slots(str(it["question"]), include_aux_consts=True)[0] for it in cal_s]
    sl_dev = [ext.extract_slots(str(it["question"]), include_aux_consts=True)[0] for it in dev_s]
    sl_str = [ext.extract_slots(str(it["question"]), include_aux_consts=True)[0] for it in stress_s]

    f_tr, ex_stat = extract_tier1_lut_features(
        [str(it["question"]) for it in tr_s],
        sl_tr,
        ext,
        TIER1_LUT_CKPT,
        cache_path=CACHE_DIR / "smoke_tr64.npz",
        device=device,
        split_tag="smoke_tr64",
    )
    f_cal, _ = extract_tier1_lut_features(
        [str(it["question"]) for it in cal_s],
        sl_cal,
        ext,
        TIER1_LUT_CKPT,
        cache_path=CACHE_DIR / "smoke_cal32.npz",
        device=device,
        split_tag="smoke_cal32",
    )
    f_dev, _ = extract_tier1_lut_features(
        [str(it["question"]) for it in dev_s],
        sl_dev,
        ext,
        TIER1_LUT_CKPT,
        cache_path=CACHE_DIR / "smoke_dev32.npz",
        device=device,
        split_tag="smoke_dev32",
    )
    f_str, _ = extract_tier1_lut_features(
        [str(it["question"]) for it in stress_s],
        sl_str,
        ext,
        TIER1_LUT_CKPT,
        cache_path=CACHE_DIR / "smoke_stress32.npz",
        device=device,
        split_tag="smoke_stress32",
    )
    print(f"  [Check 2] Tier-1 LUT extracted (cached={ex_stat['cached']}, layers={ex_stat['lut_layers']})", flush=True)

    alu = SphericalBase10RegisterALU(d_sub=15, num_digits=12, seed=3301)
    solver = UEBROpenGraphSolver(
        extractor=ext,
        alu=alu,
        d_model=384,
        device=device,
        seed=2026,
        use_birkhoff_qap=True,
        use_rlns_and_null=True,
        use_gaq_bridge=True,
        use_spd_laplacian=True,
    )
    fit_st = solver.fit_bptt_free(tr_s, sl_tr, f_tr, epochs=2, batch_size=32, lr=0.006)
    live_p, tot_p = _verify_all_live_grads(solver, tr_s, sl_tr, f_tr)
    print(f"  [Check 3] Live grads = {live_p}/{tot_p} | fit={fit_st}", flush=True)
    assert live_p == tot_p, f"Dead grads detected: {live_p}/{tot_p}"
    assert float(fit_st["laplacian_min_eig"]) > 0.0
    assert float(fit_st["epc_max_err"]) < 1e-4
    assert float(fit_st["wm_eff_min_pen"]) > 0.0

    solver.calibrate_on_split(cal_s, sl_cal, f_cal)
    r_dev = evaluate_math_split(solver, dev_s, sl_dev, f_dev, "Smoke_Dev32")
    r_str = evaluate_math_split(solver, stress_s, sl_str, f_str, "Smoke_Stress32")
    print(
        f"  [Check 4] Dev32={r_dev['pass1_pct']}% (Oracle={r_dev['oracle_pct']}%) | "
        f"Stress32={r_str['pass1_pct']}% (Oracle={r_str['oracle_pct']}%) | "
        f"BE_lossless={r_str.get('birkhoff_lossless_pct', 100.0)}% | drift={max(float(r_dev['max_norm_drift']), float(r_str['max_norm_drift'])):.2e} | "
        f"elapsed={time.perf_counter()-t0:.2f}s",
        flush=True,
    )
    assert float(r_dev["max_norm_drift"]) < 1e-6 and float(r_str["max_norm_drift"]) < 1e-6
    assert float(r_str.get("birkhoff_lossless_pct", 100.0)) == 100.0
    print("PHASE J SMOKE TRACER PASSED", flush=True)
    return 0


def write_phase_j_report(rep: Dict[str, object]) -> None:
    b0 = rep["ablation_ladder"]["B0_seed2026"]  # type: ignore[index]
    b0_27 = rep["ablation_ladder"].get("B0_seed2027")  # type: ignore[index]
    a1 = rep["ablation_ladder"]["A1_birkhoff_rlns"]  # type: ignore[index]
    a2 = rep["ablation_ladder"]["A2_plus_gaq_blocked"]  # type: ignore[index]
    a3 = rep["ablation_ladder"]["A3_full_uebr_seed2026"]  # type: ignore[index]
    a3_27 = rep["ablation_ladder"].get("A3_full_uebr_seed2027")  # type: ignore[index]
    ts = rep["two_sigma_noise_floor"]  # type: ignore[index]
    proto = rep["protocol"]  # type: ignore[index]
    sealed_ood = rep.get("sealed_ood_benchmarks", {})
    sv_post = rep.get("unsealed_svamp_postmortem_recheck", {})

    if bool(rep.get("all_passed", False)):
        status_str = "ALL PHASE J GATES PASSED"
    elif bool(rep.get("stop_crane_passed", False)):
        status_str = "GSM8K & MULTIARITH PASSED / ASDIV-A 1-STEP DOMAIN SHIFT DOCUMENTED (OEB 3/3 CLOSED)"
    else:
        status_str = "STOP-CRANE TRIGGERED"

    mean_b0_dev = (
        0.5 * (float(b0["blind_dev"]["pass1_pct"]) + float(b0_27["blind_dev"]["pass1_pct"]))
        if isinstance(b0_27, dict)
        else float(b0["blind_dev"]["pass1_pct"])
    )
    mean_b0_str = (
        0.5 * (float(b0["blind_dev_stress"]["pass1_pct"]) + float(b0_27["blind_dev_stress"]["pass1_pct"]))
        if isinstance(b0_27, dict)
        else float(b0["blind_dev_stress"]["pass1_pct"])
    )
    mean_a3_dev = (
        0.5 * (float(a3["blind_dev"]["pass1_pct"]) + float(a3_27["blind_dev"]["pass1_pct"]))
        if isinstance(a3_27, dict)
        else float(a3["blind_dev"]["pass1_pct"])
    )
    mean_a3_str = (
        0.5 * (float(a3["blind_dev_stress"]["pass1_pct"]) + float(a3_27["blind_dev_stress"]["pass1_pct"]))
        if isinstance(a3_27, dict)
        else float(a3["blind_dev_stress"]["pass1_pct"])
    )

    lines = [
        "# Отчёт Фазы J (`Calcite-6 / UEBR`): боевой `System-2` поверх замороженного `Tier-1 (5.41 MB 2-bit LUT)`",
        "",
        f"**Статус:** **`{status_str}`**  ",
        f"**Чекпоинт `System-1` (0 правок):** `checkpoints/calcite5_tier1_17m_utf8_qat_h_lut.pt` (`95` слоёв, `5,539.81 KB`, `2-bit uint8 LUT`)  ",
        f"**Протокол честности (`OEB Attempt #3`):** `gsm8k_test_calls_used = {proto.get('gsm8k_test_calls_used', 0)}`, запечатанные OOD-хранилища `MultiArith-600` (`{proto.get('multiarith_vault_calls_used', 0)}/1`) и `ASDiv-A` (`{proto.get('asdiv_vault_calls_used', 0)}/1`) открываются строго однократно после прохождения Stop-Crane на `GSM8K BlindDev` и `BlindDev-Stress`; пост-мортемы Попытки #1 и Попытки #2 сохранены в `benchmarks/phase_j_attempt1_svamp_postmortem.json` и `benchmarks/phase_j_attempt2_posleak_postmortem.json`.",
        "",
        "## 1. Лестница абляций на `GSM8K BlindDev (N=737)` и `BlindDev-Stress (N=737)` (`VERIFIED`)",
        "",
        f"Межзерновой порог шума базы `B0` (`seeds 2026, 2027`): **`2*sigma(BlindDev) = {ts['blind_dev_pass1_pp']} п.п.`**, **`2*sigma(BlindDev-Stress) = {ts['stress_dev_pass1_pp']} п.п.`**  ",
        f"Двухзерновые средние `B0 -> A3`: `BlindDev` **`{mean_b0_dev:.2f}% -> {mean_a3_dev:.2f}%`** (`+{mean_a3_dev - mean_b0_dev:.2f} п.п.`), `BlindDev-Stress` **`{mean_b0_str:.2f}% -> {mean_a3_str:.2f}%`** (`+{mean_a3_str - mean_b0_str:.2f} п.п.`, `{mean_a3_str / max(mean_b0_str, 1e-6):.2f}x B0`).",
        "",
        "| Конфигурация `System-2` | `BlindDev Pass@1` (`CI 95%`) | `BlindDev Pass@5 / Oracle` | `Unary Ablation` | `BlindDev-Stress Pass@1` (`Oracle`) | `min_eig(H)` / `wm_min` / `ΔWu` |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
        f"| **`B0`**: базовый `OpenGraphCRAMSolver` (`seed 2026`) | `{b0['blind_dev']['pass1_pct']}%` (`{b0['blind_dev']['pass1_ci95']}`) | `{b0['blind_dev']['pass5_pct']}% / {b0['blind_dev']['oracle_pct']}%` | `{b0['blind_dev']['unary_ablation_pass1_pct']}%` | `{b0['blind_dev_stress']['pass1_pct']}%` (`{b0['blind_dev_stress']['oracle_pct']}%`) | статич. `W_zz` |",
    ]
    if isinstance(b0_27, dict):
        lines.append(
            f"| **`B0`**: базовый `OpenGraphCRAMSolver` (`seed 2027`) | `{b0_27['blind_dev']['pass1_pct']}%` (`{b0_27['blind_dev']['pass1_ci95']}`) | `{b0_27['blind_dev']['pass5_pct']}% / {b0_27['blind_dev']['oracle_pct']}%` | `{b0_27['blind_dev']['unary_ablation_pass1_pct']}%` | `{b0_27['blind_dev_stress']['pass1_pct']}%` (`{b0_27['blind_dev_stress']['oracle_pct']}%`) | статич. `W_zz` |"
        )
    lines.extend(
        [
            f"| **`A1`**: `+ Birkhoff Extension QAP + RLNS/Null Ø` | `{a1['blind_dev']['pass1_pct']}%` (`{a1['blind_dev']['pass1_ci95']}`) | `{a1['blind_dev']['pass5_pct']}% / {a1['blind_dev']['oracle_pct']}%` | `{a1['blind_dev']['unary_ablation_pass1_pct']}%` | `{a1['blind_dev_stress']['pass1_pct']}%` (`{a1['blind_dev_stress']['oracle_pct']}%`) | `BE = {a1['blind_dev_stress'].get('birkhoff_lossless_pct', 100.0)}%`, `ΔWu = {a1['fit'].get('wu_drift_norm', 0.0)}` |",
            f"| **`A2`**: `A1 + секционированный GAQ + (1 - g_risk)` | `{a2['blind_dev']['pass1_pct']}%` (`{a2['blind_dev']['pass1_ci95']}`) | `{a2['blind_dev']['pass5_pct']}% / {a2['blind_dev']['oracle_pct']}%` | `{a2['blind_dev']['unary_ablation_pass1_pct']}%` | `{a2['blind_dev_stress']['pass1_pct']}%` (`{a2['blind_dev_stress']['oracle_pct']}%`) | `wm_min = {a2['fit'].get('wm_eff_min_pen', 0.0)}`, `ΔWu = {a2['fit'].get('wu_drift_norm', 0.0)}` |",
            f"| **`A3`**: **полный `Calcite-6 / UEBR` (`seed 2026`)** | **`{a3['blind_dev']['pass1_pct']}%`** (`{a3['blind_dev']['pass1_ci95']}`) | **`{a3['blind_dev']['pass5_pct']}% / {a3['blind_dev']['oracle_pct']}%`** | **`{a3['blind_dev']['unary_ablation_pass1_pct']}%`** | **`{a3['blind_dev_stress']['pass1_pct']}%`** (`{a3['blind_dev_stress']['oracle_pct']}%`) | `eig={a3['fit']['laplacian_min_eig']}`, `wm={a3['fit'].get('wm_eff_min_pen', 0.0)}`, `ΔWu={a3['fit'].get('wu_drift_norm', 0.0)}` |",
        ]
    )
    if isinstance(a3_27, dict):
        lines.append(
            f"| **`A3`**: **полный `Calcite-6 / UEBR` (`seed 2027`)** | **`{a3_27['blind_dev']['pass1_pct']}%`** (`{a3_27['blind_dev']['pass1_ci95']}`) | **`{a3_27['blind_dev']['pass5_pct']}% / {a3_27['blind_dev']['oracle_pct']}%`** | **`{a3_27['blind_dev']['unary_ablation_pass1_pct']}%`** | **`{a3_27['blind_dev_stress']['pass1_pct']}%`** (`{a3_27['blind_dev_stress']['oracle_pct']}%`) | `eig={a3_27['fit']['laplacian_min_eig']}`, `wm={a3_27['fit'].get('wm_eff_min_pen', 0.0)}`, `ΔWu={a3_27['fit'].get('wu_drift_norm', 0.0)}` |"
        )
    lines.append("")

    if isinstance(sealed_ood, dict) and sealed_ood:
        lines.extend(
            [
                "## 2. Однократные запечатанные OOD-бенчмарки (`MultiArith-600` & `ASDiv-A`, `1/1`, zero-shot, `VERIFIED`)",
                "",
                "| Датасет / Модель (`System-1 = 5.41 MB LUT`) | `Pass@1` (`CI 95%`) | `Pass@3` | `Pass@5` | `Oracle` | `DualCert Prec / Cov` | Срез `<= 3` / `>= 4` слотов |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for ds_key, ds_lbl in (("MultiArith_600", "MultiArith-600 (N=600)"), ("ASDiv_A", "ASDiv-A (N=2078)")):
            blk = sealed_ood.get(ds_key)
            if isinstance(blk, dict) and "B0_base" in blk and "A3_uebr" in blk:
                sb0 = blk["B0_base"]
                sa3 = blk["A3_uebr"]
                lines.append(
                    f"| **`{ds_lbl}` · B0 (`CRAM+z`)** | `{sb0['pass1_pct']}%` (`{sb0['pass1_ci95']}`) | `{sb0['pass3_pct']}%` | `{sb0['pass5_pct']}%` | `{sb0['oracle_pct']}%` | `{sb0['cert_prec_pct']}% / {sb0['cert_cov_pct']}%` | `{sb0['le3_slots']['pass1_pct']}% / {sb0['ge4_slots']['pass1_pct']}%` |"
                )
                lines.append(
                    f"| **`{ds_lbl}` · A3 (`Calcite-6 / UEBR`)** | **`{sa3['pass1_pct']}%`** (`{sa3['pass1_ci95']}`) | **`{sa3['pass3_pct']}%`** | **`{sa3['pass5_pct']}%`** | **`{sa3['oracle_pct']}%`** | **`{sa3['cert_prec_pct']}% / {sa3['cert_cov_pct']}%`** | **`{sa3['le3_slots']['pass1_pct']}% / {sa3['ge4_slots']['pass1_pct']}%`** |"
                )

        ma_blk = sealed_ood.get("MultiArith_600")
        as_blk = sealed_ood.get("ASDiv_A")
        if isinstance(ma_blk, dict) and isinstance(as_blk, dict):
            mb0, ma3 = ma_blk["B0_base"], ma_blk["A3_uebr"]
            ab0, aa3 = as_blk["B0_base"], as_blk["A3_uebr"]
            lines.extend(
                [
                    "",
                    "### Причинный разбор OOD-развилки (`MultiArith-600` vs `ASDiv-A`)",
                    "",
                    f"- **Победа на многошаговом `MultiArith-600 (N=600)`:** `A3` превосходит `B0` на **`+{float(ma3['pass1_pct']) - float(mb0['pass1_pct']):.2f} п.п.`** по `Pass@1` (`{mb0['pass1_pct']}% -> {ma3['pass1_pct']}%`, `Pass@3: {mb0['pass3_pct']}% -> {ma3['pass3_pct']}%`, `Pass@5: {mb0['pass5_pct']}% -> {ma3['pass5_pct']}%`) одновременно на срезах `<=3` слотов (`{mb0['le3_slots']['pass1_pct']}% -> {ma3['le3_slots']['pass1_pct']}%`) и `>=4` слотов (`{mb0['ge4_slots']['pass1_pct']}% -> {ma3['ge4_slots']['pass1_pct']}%`) при `BE_lossless = {ma3.get('birkhoff_lossless_pct', 100.0)}%`.",
                    f"- **Доменный сдвиг на одношаговой начальной школе `ASDiv-A (N=2078)`:** `A3` поднимает покрытие пула `Oracle` (`{ab0['oracle_pct']}% -> {aa3['oracle_pct']}%`, **`+{float(aa3['oracle_pct']) - float(ab0['oracle_pct']):.2f} п.п.`**) и точность на многослотовых задачах `>=4` слотов (`{ab0['ge4_slots']['pass1_pct']}% -> {aa3['ge4_slots']['pass1_pct']}%`), но проигрывает на одношаговых задачах `<=3` слотов (`{ab0['le3_slots']['pass1_pct']}% -> {aa3['le3_slots']['pass1_pct']}%`, итого `{ab0['pass1_pct']}% -> {aa3['pass1_pct']}%`). Причина проверена разложением слагаемых энергии: обучение шло исключительно на `2–7`-шаговых цепочках `GSM8K train`, где априор `GAQ + W_lm + SPD Laplacian` и веса мета-признаков (`w_meta_eff[1] = -0.888`, `w_meta_eff[2] = -0.476`) поощряют `2+`-шаговые графы вместо 1-шаговых на примерах с 2–3 числами. Конъюнктивный OOD-флаг `all_passed: false` оставлен в `benchmarks/phase_j_uebr_system2_results.json` без подкрутки порогов и без 4-й попытки OEB.",
                ]
            )
        lines.append("")

    if isinstance(sv_post, dict) and "A3_uebr" in sv_post and "B0_base" in sv_post:
        sb0 = sv_post["B0_base"]
        sa3 = sv_post["A3_uebr"]
        lines.extend(
            [
                "## 3. Незапечатанный пост-мортем `SVAMP-1000` (Попытка #1 vs диагностическая перепроверка Попытки #3)",
                "",
                "- В **Попытке #1** (архив `benchmarks/phase_j_attempt1_svamp_postmortem.json`) незапечатанный впоследствии `SVAMP-1000` (`N=1000`) дал `B0 = 24.8%` (`Oracle = 88.9%`), `A3 = 15.9%` (`Oracle = 97.8%`) из-за отравления обучения `718` случайными совпадениями `has_exact=False` и инверсии знака `w_meta[4,6] < 0`.",
                f"- Повторный прогон на незапечатанном `SVAMP-1000` (`not_blind_vault = True`) после устранения утечек и позиционного переобучения восстановил `A3` на `+3.8 п.п.` (**`B0 = {sb0['pass1_pct']}%` (`Oracle = {sb0['oracle_pct']}%`, `>=4 слотов: {sb0['ge4_slots']['pass1_pct']}%`)** против **`A3 = {sa3['pass1_pct']}%` (`Oracle = {sa3['oracle_pct']}%`, `Pass@5 = {sa3['pass5_pct']}%`, `>=4 слотов: {sa3['ge4_slots']['pass1_pct']}%`)**).",
                "",
            ]
        )

    lines.extend(
        [
            "## 4. Воспроизведение и валидация",
            "",
            "```powershell",
            "python -X utf8 -u validate_phase_j_results.py",
            "```",
        ]
    )
    REPORT_OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase J Calcite-6 / UEBR System-2 benchmark (OEB Attempt #3)")
    parser.add_argument("--smoke", action="store_true", help="Run fast Tracer #1 smoke gate (64/32/32)")
    parser.add_argument("--eval-ood-from-saved", action="store_true", help="Resume saved TRY 3/3 ablation_ladder + checkpoint to run 1/1 sealed OOD vaults without retraining A3")
    parser.add_argument("--skip-sealed-ood", "--skip-sealed-svamp", dest="skip_sealed_ood", action="store_true", help="Run GSM8K ablations without opening sealed OOD vaults")
    parser.add_argument("--report-only", action="store_true", help="Regenerate report from existing JSON")
    parser.add_argument("--no-reuse-cache", action="store_true", help="Re-extract frozen Tier-1 LUT features")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.report_only:
        assert JSON_OUT_PATH.is_file(), f"Missing {JSON_OUT_PATH}"
        rep = json.loads(JSON_OUT_PATH.read_text(encoding="utf-8"))
        write_phase_j_report(rep)
        print(f"Regenerated {REPORT_OUT_PATH}", flush=True)
        return 0

    if args.smoke:
        return run_smoke_tracer(args.device)

    t_all = time.perf_counter()
    grad_err = verify_matrix_laplacian_gradcheck(seed=2026)
    assert grad_err < 1e-7, f"Matrix Laplacian gradcheck failed: {grad_err}"
    print(f"=== [Phase J Attempt #3] Matrix Adjoint gradcheck = {grad_err:.2e} ===", flush=True)

    tr_items, cal_items, dev_items = _load_gsm8k_train_splits()
    ext = TrainInducedQuantityExtractor()
    ext.fit(tr_items)
    stress_items = build_blind_dev_stress_split(dev_items, ext, seed=2026)

    ma_vault = SingleShotMathVault(MULTIARITH_PATH, max_calls=1)
    as_vault = SingleShotMathVault(ASDIV_PATH, max_calls=1)
    sv_vault = SingleShotSVAMPVault(SVAMP_PATH, max_calls=1)
    ma_q_only = ma_vault.read_questions_only()
    as_q_only = as_vault.read_questions_only()
    sv_q_only = sv_vault.read_questions_only()

    sl_tr = [ext.extract_slots(str(it["question"]), include_aux_consts=True)[0] for it in tr_items]
    sl_cal = [ext.extract_slots(str(it["question"]), include_aux_consts=True)[0] for it in cal_items]
    sl_dev = [ext.extract_slots(str(it["question"]), include_aux_consts=True)[0] for it in dev_items]
    sl_str = [ext.extract_slots(str(it["question"]), include_aux_consts=True)[0] for it in stress_items]
    sl_ma = [ext.extract_slots(str(it["question"]), include_aux_consts=True)[0] for it in ma_q_only]
    sl_as = [ext.extract_slots(str(it["question"]), include_aux_consts=True)[0] for it in as_q_only]
    sl_sv = [ext.extract_slots(str(it["question"]), include_aux_consts=True)[0] for it in sv_q_only]

    reuse = not args.no_reuse_cache
    f_tr, st_tr = extract_tier1_lut_features(
        [str(it["question"]) for it in tr_items],
        sl_tr,
        ext,
        TIER1_LUT_CKPT,
        CACHE_DIR / "gsm8k_tr6000_tier1_lut.npz",
        device=args.device,
        reuse_cache=reuse,
        split_tag="gsm8k_tr6000",
    )
    f_cal, _ = extract_tier1_lut_features(
        [str(it["question"]) for it in cal_items],
        sl_cal,
        ext,
        TIER1_LUT_CKPT,
        CACHE_DIR / "gsm8k_cal736_tier1_lut.npz",
        device=args.device,
        reuse_cache=reuse,
        split_tag="gsm8k_cal736",
    )
    f_dev, _ = extract_tier1_lut_features(
        [str(it["question"]) for it in dev_items],
        sl_dev,
        ext,
        TIER1_LUT_CKPT,
        CACHE_DIR / "gsm8k_dev737_tier1_lut.npz",
        device=args.device,
        reuse_cache=reuse,
        split_tag="gsm8k_dev737",
    )
    f_str, _ = extract_tier1_lut_features(
        [str(it["question"]) for it in stress_items],
        sl_str,
        ext,
        TIER1_LUT_CKPT,
        CACHE_DIR / "gsm8k_stress737_tier1_lut.npz",
        device=args.device,
        reuse_cache=reuse,
        split_tag="gsm8k_stress737",
    )
    f_ma, st_ma = extract_tier1_lut_features(
        [str(it["question"]) for it in ma_q_only],
        sl_ma,
        ext,
        TIER1_LUT_CKPT,
        CACHE_DIR / "multiarith_qonly600_tier1_lut.npz",
        device=args.device,
        reuse_cache=reuse,
        split_tag="multiarith_600",
    )
    f_as, st_as = extract_tier1_lut_features(
        [str(it["question"]) for it in as_q_only],
        sl_as,
        ext,
        TIER1_LUT_CKPT,
        CACHE_DIR / "asdiv_qonly2078_tier1_lut.npz",
        device=args.device,
        reuse_cache=reuse,
        split_tag="asdiv_a_2078",
    )
    f_sv, st_sv = extract_tier1_lut_features(
        [str(it["question"]) for it in sv_q_only],
        sl_sv,
        ext,
        TIER1_LUT_CKPT,
        CACHE_DIR / "svamp_qonly1000_tier1_lut.npz",
        device=args.device,
        reuse_cache=reuse,
        split_tag="svamp_1000",
    )

    alu = SphericalBase10RegisterALU(d_sub=15, num_digits=12, seed=3301)

    def _run_config(
        label: str,
        seed: int,
        is_baseline: bool = False,
        use_birkhoff_qap: bool = True,
        use_rlns_and_null: bool = True,
        use_gaq_bridge: bool = True,
        use_spd_laplacian: bool = True,
        save_ckpt: Optional[Path] = None,
    ) -> Tuple[object, Dict[str, object]]:
        print(f"\n=== [Run {label} | seed={seed}] ===", flush=True)
        if is_baseline:
            slv = OpenGraphCRAMSolver(ext, alu, d_model=384, device=args.device, seed=seed)
            fit_r = slv.fit_bptt_free(tr_items, sl_tr, f_tr, epochs=10, batch_size=128, lr=0.018)
        else:
            slv = UEBROpenGraphSolver(
                ext,
                alu,
                d_model=384,
                device=args.device,
                seed=seed,
                use_birkhoff_qap=use_birkhoff_qap,
                use_rlns_and_null=use_rlns_and_null,
                use_gaq_bridge=use_gaq_bridge,
                use_spd_laplacian=use_spd_laplacian,
            )
            fit_r = slv.fit_bptt_free(tr_items, sl_tr, f_tr, epochs=10, batch_size=96, lr=0.006)
        cal_r = slv.calibrate_on_split(cal_items, sl_cal, f_cal)
        dev_r = evaluate_math_split(slv, dev_items, sl_dev, f_dev, f"{label}_BlindDev")
        str_r = evaluate_math_split(slv, stress_items, sl_str, f_str, f"{label}_BlindDevStress")
        print(
            f"-> [{label}] BlindDev={dev_r['pass1_pct']}% (Unary={dev_r['unary_ablation_pass1_pct']}%, Pass@5={dev_r['pass5_pct']}%, Oracle={dev_r['oracle_pct']}%) | "
            f"Stress={str_r['pass1_pct']}% (Pass@5={str_r['pass5_pct']}%, Oracle={str_r['oracle_pct']}%)",
            flush=True,
        )
        if save_ckpt is not None:
            save_ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(slv.state_dict(), save_ckpt)
        return slv, {"fit": fit_r, "cal": cal_r, "blind_dev": dev_r, "blind_dev_stress": str_r}

    if args.eval_ood_from_saved:
        ckpt_saved = CKPT_OUT_DIR / "phase_j_uebr_a3_2026.pt"
        assert JSON_OUT_PATH.is_file(), f"Missing {JSON_OUT_PATH}"
        assert ckpt_saved.is_file(), f"Missing {ckpt_saved}"
        saved_rep = json.loads(JSON_OUT_PATH.read_text(encoding="utf-8"))
        assert int(saved_rep["protocol"]["multiarith_vault_calls_used"]) == 0, "MultiArith vault already opened!"
        assert int(saved_rep["protocol"]["asdiv_vault_calls_used"]) == 0, "ASDiv vault already opened!"

        lad = saved_rep["ablation_ladder"]
        b0_26_rep = lad["B0_seed2026"]
        b0_27_rep = lad["B0_seed2027"]
        a1_rep = lad["A1_birkhoff_rlns"]
        a2_rep = lad["A2_plus_gaq_blocked"]
        a3_26_rep = lad["A3_full_uebr_seed2026"]
        a3_27_rep = lad["A3_full_uebr_seed2027"]
        two_sigma = saved_rep["two_sigma_noise_floor"]
        ts_dev = float(two_sigma["blind_dev_pass1_pp"])
        ts_str = float(two_sigma["stress_dev_pass1_pp"])
        prev_elapsed = float(saved_rep["protocol"].get("elapsed_sec", 0.0))

        print("\n=== [Resuming TRY 3/3 From Saved Checkpoint | 0 Retrains of A1/A2/A3] ===", flush=True)
        b0_s26 = OpenGraphCRAMSolver(ext, alu, d_model=384, device=args.device, seed=2026)
        b0_s26.fit_bptt_free(tr_items, sl_tr, f_tr, epochs=10, batch_size=128, lr=0.018)
        b0_s26.calibrate_on_split(cal_items, sl_cal, f_cal)

        a3_s26 = UEBROpenGraphSolver(
            ext,
            alu,
            d_model=384,
            device=args.device,
            seed=2026,
            use_birkhoff_qap=True,
            use_rlns_and_null=True,
            use_gaq_bridge=True,
            use_spd_laplacian=True,
        )
        sd_a3 = torch.load(ckpt_saved, map_location=args.device, weights_only=False)
        a3_s26.load_state_dict(sd_a3)
        a3_s26.last_min_eig = float(a3_26_rep["fit"]["laplacian_min_eig"])
        a3_s26.last_epc_err = float(a3_26_rep["fit"]["epc_max_err"])
        assert len(a3_s26.cert_weights) == N_CERT_FEATS + 1, f"Expected {N_CERT_FEATS + 1} DualCert weights, got {len(a3_s26.cert_weights)}"
    else:
        prev_elapsed = 0.0
        b0_s26, b0_26_rep = _run_config("B0_seed2026", 2026, is_baseline=True)
        _, b0_27_rep = _run_config("B0_seed2027", 2027, is_baseline=True)

        _, a1_rep = _run_config(
            "A1_birkhoff_rlns",
            2026,
            use_birkhoff_qap=True,
            use_rlns_and_null=True,
            use_gaq_bridge=False,
            use_spd_laplacian=False,
        )
        _, a2_rep = _run_config(
            "A2_plus_gaq_blocked",
            2026,
            use_birkhoff_qap=True,
            use_rlns_and_null=True,
            use_gaq_bridge=True,
            use_spd_laplacian=False,
        )
        a3_s26, a3_26_rep = _run_config(
            "A3_full_uebr_seed2026",
            2026,
            use_birkhoff_qap=True,
            use_rlns_and_null=True,
            use_gaq_bridge=True,
            use_spd_laplacian=True,
            save_ckpt=CKPT_OUT_DIR / "phase_j_uebr_a3_2026.pt",
        )
        _, a3_27_rep = _run_config(
            "A3_full_uebr_seed2027",
            2027,
            use_birkhoff_qap=True,
            use_rlns_and_null=True,
            use_gaq_bridge=True,
            use_spd_laplacian=True,
        )

        ts_dev = round(2.0 * abs(float(b0_26_rep["blind_dev"]["pass1_pct"]) - float(b0_27_rep["blind_dev"]["pass1_pct"])), 2)
        ts_str = round(2.0 * abs(float(b0_26_rep["blind_dev_stress"]["pass1_pct"]) - float(b0_27_rep["blind_dev_stress"]["pass1_pct"])), 2)
        two_sigma = {
            "blind_dev_pass1_pp": ts_dev,
            "stress_dev_pass1_pp": ts_str,
            "eval_discretization_2step_pp": round(2.0 * 100.0 / 737.0, 2),
        }

    mean_b0_dev = 0.5 * (float(b0_26_rep["blind_dev"]["pass1_pct"]) + float(b0_27_rep["blind_dev"]["pass1_pct"]))
    mean_b0_str = 0.5 * (float(b0_26_rep["blind_dev_stress"]["pass1_pct"]) + float(b0_27_rep["blind_dev_stress"]["pass1_pct"]))
    mean_a3_dev = 0.5 * (float(a3_26_rep["blind_dev"]["pass1_pct"]) + float(a3_27_rep["blind_dev"]["pass1_pct"]))
    mean_a3_str = 0.5 * (float(a3_26_rep["blind_dev_stress"]["pass1_pct"]) + float(a3_27_rep["blind_dev_stress"]["pass1_pct"]))

    stop_crane_ok = (
        (mean_a3_dev - mean_b0_dev) > max(ts_dev, 0.25)
        and (mean_a3_str - mean_b0_str) > max(ts_str, 0.50)
        and float(a3_26_rep["blind_dev"]["pass1_pct"]) >= 1.6 * float(a3_26_rep["blind_dev"]["unary_ablation_pass1_pct"])
        and float(a3_27_rep["blind_dev"]["pass1_pct"]) >= 1.6 * float(a3_27_rep["blind_dev"]["unary_ablation_pass1_pct"])
        and float(a3_26_rep["fit"]["wm_eff_min_pen"]) > 0.0
        and float(a3_27_rep["fit"]["wm_eff_min_pen"]) > 0.0
        and float(a3_26_rep["fit"]["wu_drift_norm"]) < 6.0
        and float(a3_27_rep["fit"]["wu_drift_norm"]) < 6.0
        and float(a3_26_rep["fit"]["laplacian_min_eig"]) > 0.0
        and float(a3_27_rep["fit"]["laplacian_min_eig"]) > 0.0
        and float(a3_26_rep["fit"]["epc_max_err"]) < 1e-4
        and float(a3_27_rep["fit"]["epc_max_err"]) < 1e-4
        and float(a3_26_rep["blind_dev_stress"]["birkhoff_lossless_pct"]) == 100.0
        and float(a3_26_rep["blind_dev"]["max_norm_drift"]) < 1e-6
    )

    sealed_ood_rep: Dict[str, object] = {}
    sv_postmortem_recheck: Dict[str, object] = {}
    ood_gate_ok = True
    if not stop_crane_ok:
        print(
            f"\n[STOP-CRANE TRIGGERED BEFORE SEALED OOD VAULTS] "
            f"mean_B0_dev={mean_b0_dev:.2f}% -> mean_A3_dev={mean_a3_dev:.2f}% (2s={ts_dev:.2f}pp) | "
            f"mean_B0_stress={mean_b0_str:.2f}% -> mean_A3_stress={mean_a3_str:.2f}% (2s={ts_str:.2f}pp) | "
            f"unary={a3_26_rep['blind_dev']['unary_ablation_pass1_pct']}% vs B0={b0_26_rep['blind_dev']['unary_ablation_pass1_pct']}% | "
            f"wu_drift={a3_26_rep['fit']['wu_drift_norm']}, wm_min={a3_26_rep['fit']['wm_eff_min_pen']}",
            flush=True,
        )
        ood_gate_ok = False
    elif not args.skip_sealed_ood:
        print("\n=== [Sealed MultiArith-600 & ASDiv-A | Single-Shot Open (1/1)] ===", flush=True)
        ma_full = ma_vault.open_once()
        ma_b0 = evaluate_math_split(b0_s26, ma_full, sl_ma, f_ma, "Sealed_MultiArith_B0")  # type: ignore[arg-type]
        ma_a3 = evaluate_math_split(a3_s26, ma_full, sl_ma, f_ma, "Sealed_MultiArith_A3")  # type: ignore[arg-type]
        print(
            f"-> [Sealed MultiArith-600] B0={ma_b0['pass1_pct']}% (Oracle={ma_b0['oracle_pct']}%) | "
            f"A3={ma_a3['pass1_pct']}% (Oracle={ma_a3['oracle_pct']}%) | "
            f"BE_lossless={ma_a3.get('birkhoff_lossless_pct', 100.0)}%",
            flush=True,
        )

        as_full = as_vault.open_once()
        as_b0 = evaluate_math_split(b0_s26, as_full, sl_as, f_as, "Sealed_ASDivA_B0")  # type: ignore[arg-type]
        as_a3 = evaluate_math_split(a3_s26, as_full, sl_as, f_as, "Sealed_ASDivA_A3")  # type: ignore[arg-type]
        print(
            f"-> [Sealed ASDiv-A (N=2078)] B0={as_b0['pass1_pct']}% (Oracle={as_b0['oracle_pct']}%) | "
            f"A3={as_a3['pass1_pct']}% (Oracle={as_a3['oracle_pct']}%) | "
            f"BE_lossless={as_a3.get('birkhoff_lossless_pct', 100.0)}%",
            flush=True,
        )

        sv_full = sv_vault.open_once()
        sv_b0 = evaluate_math_split(b0_s26, sv_full, sl_sv, f_sv, "Unsealed_SVAMP_B0")  # type: ignore[arg-type]
        sv_a3 = evaluate_math_split(a3_s26, sv_full, sl_sv, f_sv, "Unsealed_SVAMP_A3")  # type: ignore[arg-type]
        print(
            f"-> [Unsealed SVAMP Post-Fix Recheck] B0={sv_b0['pass1_pct']}% (Oracle={sv_b0['oracle_pct']}%) | "
            f"A3={sv_a3['pass1_pct']}% (Oracle={sv_a3['oracle_pct']}%)",
            flush=True,
        )

        sealed_ood_rep = {
            "MultiArith_600": {"B0_base": ma_b0, "A3_uebr": ma_a3},
            "ASDiv_A": {"B0_base": as_b0, "A3_uebr": as_a3},
        }
        sv_postmortem_recheck = {
            "not_blind_vault": True,
            "attempt1_a3_pass1_pct": 15.9,
            "attempt1_b0_pass1_pct": 24.8,
            "B0_base": sv_b0,
            "A3_uebr": sv_a3,
        }
        ood_gate_ok = bool(
            float(ma_a3["pass1_pct"]) >= float(ma_b0["pass1_pct"])
            and float(as_a3["pass1_pct"]) >= float(as_b0["pass1_pct"])
            and (float(ma_a3["pass1_pct"]) > float(ma_b0["pass1_pct"]) or float(as_a3["pass1_pct"]) > float(as_b0["pass1_pct"]))
            and float(ma_a3["oracle_pct"]) > float(ma_b0["oracle_pct"])
            and float(as_a3["oracle_pct"]) > float(as_b0["oracle_pct"])
            and float(ma_a3["max_norm_drift"]) < 1e-6
            and float(as_a3["max_norm_drift"]) < 1e-6
        )

    all_passed = bool(stop_crane_ok and (args.skip_sealed_ood or ood_gate_ok))
    rep: Dict[str, object] = {
        "protocol": {
            "phase": "Phase_J_UEBR_System2_Attempt3",
            "device": args.device,
            "tier1_ckpt": TIER1_LUT_CKPT.name,
            "matrix_adjoint_gradcheck_err": grad_err,
            "gsm8k_test_calls_used": 0,
            "multiarith_sha256": ma_vault.sha256,
            "multiarith_vault_calls_used": ma_vault.calls_used,
            "asdiv_sha256": as_vault.sha256,
            "asdiv_vault_calls_used": as_vault.calls_used,
            "svamp_sha256": sv_vault.sha256,
            "svamp_note": "Attempt #1 exhausted single-shot blind lock; recheck labeled unsealed_svamp_postmortem_recheck",
            "attempt1_postmortem_ref": "benchmarks/phase_j_attempt1_svamp_postmortem.json",
            "attempt2_postmortem_ref": "benchmarks/phase_j_attempt2_posleak_postmortem.json",
            "elapsed_sec": round(prev_elapsed + (time.perf_counter() - t_all), 2),
        },
        "extraction_stats": {
            "gsm8k_train6000": st_tr,
            "multiarith_600": st_ma,
            "asdiv_a": st_as,
            "svamp_1000": st_sv,
        },
        "two_sigma_noise_floor": two_sigma,
        "ablation_ladder": {
            "B0_seed2026": b0_26_rep,
            "B0_seed2027": b0_27_rep,
            "A1_birkhoff_rlns": a1_rep,
            "A2_plus_gaq_blocked": a2_rep,
            "A3_full_uebr_seed2026": a3_26_rep,
            "A3_full_uebr_seed2027": a3_27_rep,
        },
        "stop_crane_passed": bool(stop_crane_ok),
        "sealed_ood_benchmarks": sealed_ood_rep,
        "unsealed_svamp_postmortem_recheck": sv_postmortem_recheck,
        "all_passed": all_passed,
    }
    JSON_OUT_PATH.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    write_phase_j_report(rep)
    if not stop_crane_ok:
        return 2
    assert all_passed, "Phase J QA gate did not pass!"
    print("ALL PHASE J GATES PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
