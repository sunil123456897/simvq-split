"""
Standalone Zero-Manual QA Validator for Phase J (`validate_phase_j_results.py`).

Verifies integrity of:
1. Zero diff on protected `simvq_split/`, `experiments/cram_gsm8k/`, `experiments/uebr_proto/`,
   `kernels/fast_packed_lut.py`, Tier-1 checkpoints, `benchmarks/gsm8k/test.jsonl`, previous
   tests, and Phase A–I benchmark JSONs vs `HEAD`; ruff clean; `17 + 6 + 4 + 5 + 5 + 5 + 5 = 47`
   passing Zero-Cheat tests; and `simvq_cram_adapter.py --verify-alignment` (`100.00%`).
2. `benchmarks/phase_j_uebr_system2_results.json`, `benchmarks/phase_j_attempt1_svamp_postmortem.json`,
   and `benchmarks/phase_j_attempt2_posleak_postmortem.json` finiteness, `GSM8K Stop-Crane PASS`,
   `MultiArith-600` OOD win, honest `ASDiv-A` 1-step shift (`all_passed=False`), and optional local
   checkpoint `experiments/uebr_system2/checkpoints/phase_j_uebr_a3_2026.pt` roundtrip (`13` cert weights).
3. `Phase_J_UEBR_System2_Report.md` and `research/post_transformer_synthesis/14_phase_j_uebr_system2_ledger.md`
   existence, UTF-8 integrity, and strict Russian lexical quarantine across all repository Markdown docs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.cram_gsm8k.cram_open_graph import N_CERT_FEATS  # noqa: E402
from experiments.cram_gsm8k.slot_extractor import TrainInducedQuantityExtractor  # noqa: E402
from experiments.cram_gsm8k.spherical_alu import SphericalBase10RegisterALU  # noqa: E402
from experiments.uebr_system2.uebr_graph_solver import UEBROpenGraphSolver  # noqa: E402
from scripts.repo_guardrail import (  # noqa: E402
    check_anti_cheat_pytest,
    check_baselines_untouched,
    check_markdown_quarantine,
    check_ruff_lint,
    collect_guarded_markdown_paths,
)

PROTECTED_TARGETS = (
    "simvq_split/",
    "experiments/cram_gsm8k/",
    "experiments/uebr_proto/",
    "kernels/fast_packed_lut.py",
    "checkpoints/calcite5_tier1_17m_utf8_lut.pt",
    "checkpoints/calcite5_tier1_17m_utf8_qat_h_lut.pt",
    "benchmarks/gsm8k/test.jsonl",
    "tests/test_anti_cheat.py",
    "tests/test_cram_phase_c.py",
    "tests/test_cram_phase_f.py",
    "tests/test_phase_g_lut_and_scale.py",
    "tests/test_phase_h_qat_closure.py",
    "tests/test_phase_i_uebr_tracer.py",
    "benchmarks/simvq_split_results.json",
    "benchmarks/phase_b_byte_chunker_results.json",
    "benchmarks/phase_c_cram_gsm8k_results.json",
    "benchmarks/phase_e_fast_utf8_results.json",
    "benchmarks/phase_f_utf8_iterative_cram_results.json",
    "benchmarks/phase_g_colab_scaleup_results.json",
    "benchmarks/phase_h_tier1_qat_closure_results.json",
    "benchmarks/phase_i_uebr_tracer_results.json",
)


def _assert_no_nan_inf(obj: Any, trail: str = "root") -> None:
    if isinstance(obj, float):
        assert math.isfinite(obj), f"Non-finite float at {trail}: {obj}"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _assert_no_nan_inf(v, f"{trail}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_no_nan_inf(v, f"{trail}[{i}]")


def main() -> int:
    root = ROOT_DIR
    json_path = root / "benchmarks" / "phase_j_uebr_system2_results.json"
    pm1_path = root / "benchmarks" / "phase_j_attempt1_svamp_postmortem.json"
    pm2_path = root / "benchmarks" / "phase_j_attempt2_posleak_postmortem.json"
    ckpt_path = root / "experiments" / "uebr_system2" / "checkpoints" / "phase_j_uebr_a3_2026.pt"
    report_path = root / "Phase_J_UEBR_System2_Report.md"
    ledger_path = root / "research" / "post_transformer_synthesis" / "14_phase_j_uebr_system2_ledger.md"

    print("=== [Phase J QA 1/4] Protected diffs, ruff, 47 pytest laws & alignment ===", flush=True)
    check_baselines_untouched(root)
    for prot in PROTECTED_TARGETS:
        chk_diff = subprocess.run(
            ["git", "diff", "--exit-code", "HEAD", "--", prot],
            cwd=str(root),
            capture_output=True,
            text=True,
        )
        assert chk_diff.returncode == 0, f"Protected target modified `{prot}`:\n{chk_diff.stdout}\n{chk_diff.stderr}"

    check_ruff_lint(root)
    print(check_anti_cheat_pytest(root), flush=True)
    total_laws = 17
    for t_file, exp_n, cnt in (
        ("test_cram_phase_c.py", "6 passed", 6),
        ("test_cram_phase_f.py", "4 passed", 4),
        ("test_phase_g_lut_and_scale.py", "5 passed", 5),
        ("test_phase_h_qat_closure.py", "5 passed", 5),
        ("test_phase_i_uebr_tracer.py", "5 passed", 5),
        ("test_phase_j_uebr_system2.py", "5 passed", 5),
    ):
        res_t = subprocess.run(
            [sys.executable, "-m", "pytest", str(root / "tests" / t_file), "-q"],
            cwd=str(root),
            capture_output=True,
            text=True,
        )
        assert res_t.returncode == 0 and exp_n in res_t.stdout, (
            f"{t_file} failed (`{exp_n}` required):\n{res_t.stdout}\n{res_t.stderr}"
        )
        total_laws += cnt
        print(f"OK {t_file}: {res_t.stdout.strip().splitlines()[-1]}", flush=True)
    assert total_laws == 47, f"Expected 47 total Zero-Cheat laws, counted {total_laws}"

    res_al = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-u",
            str(root / "experiments" / "cram_gsm8k" / "simvq_cram_adapter.py"),
            "--verify-alignment",
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert res_al.returncode == 0 and "100.00%" in res_al.stdout, (
        f"Alignment verifier failed:\n{res_al.stdout}\n{res_al.stderr}"
    )
    print("OK alignment: 100.00% verified", flush=True)

    print("=== [Phase J QA 2/4] Validating Phase J JSON, post-mortems & checkpoint ===", flush=True)
    for jpath in (json_path, pm1_path, pm2_path):
        assert jpath.is_file(), f"Missing {jpath}"
        _assert_no_nan_inf(json.loads(jpath.read_text(encoding="utf-8")), trail=jpath.name)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    proto = data["protocol"]
    assert proto["tier1_ckpt"] == "calcite5_tier1_17m_utf8_qat_h_lut.pt"
    assert int(proto["gsm8k_test_calls_used"]) == 0
    assert int(proto["multiarith_vault_calls_used"]) == 1
    assert int(proto["asdiv_vault_calls_used"]) == 1
    assert float(proto["matrix_adjoint_gradcheck_err"]) < 1e-7
    assert data["stop_crane_passed"] is True
    assert data["all_passed"] is False

    lad = data["ablation_ladder"]
    b0_dev = 0.5 * (lad["B0_seed2026"]["blind_dev"]["pass1_pct"] + lad["B0_seed2027"]["blind_dev"]["pass1_pct"])
    a3_dev = 0.5 * (
        lad["A3_full_uebr_seed2026"]["blind_dev"]["pass1_pct"]
        + lad["A3_full_uebr_seed2027"]["blind_dev"]["pass1_pct"]
    )
    b0_str = 0.5 * (
        lad["B0_seed2026"]["blind_dev_stress"]["pass1_pct"]
        + lad["B0_seed2027"]["blind_dev_stress"]["pass1_pct"]
    )
    a3_str = 0.5 * (
        lad["A3_full_uebr_seed2026"]["blind_dev_stress"]["pass1_pct"]
        + lad["A3_full_uebr_seed2027"]["blind_dev_stress"]["pass1_pct"]
    )
    ts_dev = float(data["two_sigma_noise_floor"]["blind_dev_pass1_pp"])
    assert a3_dev > b0_dev + ts_dev
    assert a3_str >= 4.0 * b0_str
    for sk in ("A3_full_uebr_seed2026", "A3_full_uebr_seed2027"):
        a3 = lad[sk]
        assert float(a3["blind_dev"]["pass1_pct"]) >= 1.6 * float(a3["blind_dev"]["unary_ablation_pass1_pct"])
        assert float(a3["blind_dev_stress"]["birkhoff_lossless_pct"]) == 100.0
        assert float(a3["fit"]["laplacian_min_eig"]) > 0.0
        assert float(a3["fit"]["epc_max_err"]) < 1e-4

    ood = data["sealed_ood_benchmarks"]
    assert float(ood["MultiArith_600"]["A3_uebr"]["pass1_pct"]) > float(ood["MultiArith_600"]["B0_base"]["pass1_pct"])
    assert float(ood["ASDiv_A"]["A3_uebr"]["oracle_pct"]) > float(ood["ASDiv_A"]["B0_base"]["oracle_pct"])
    assert (
        float(ood["ASDiv_A"]["A3_uebr"]["ge4_slots"]["pass1_pct"])
        > float(ood["ASDiv_A"]["B0_base"]["ge4_slots"]["pass1_pct"])
    )

    if ckpt_path.is_file():
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        slv = UEBROpenGraphSolver(
            extractor=TrainInducedQuantityExtractor(),
            alu=SphericalBase10RegisterALU(d_sub=15, num_digits=12, seed=3301),
            d_model=384,
            device="cpu",
            seed=2026,
        )
        slv.load_state_dict(sd)
        assert len(slv.cert_weights) == N_CERT_FEATS + 1 == 13
        assert int(slv.birkhoff_total_solves) > 0 and float(slv.last_min_eig) > 0.0
        print(
            f"OK checkpoint roundtrip: {ckpt_path.name} "
            f"(cert_dims={len(slv.cert_weights)}, be_solves={slv.birkhoff_total_solves}, min_eig={slv.last_min_eig:.4f})",
            flush=True,
        )
    else:
        print(f"SKIP gitignored local checkpoint not on disk: {ckpt_path.name}", flush=True)

    print("=== [Phase J QA 3/4] Validating Phase J report, ledger & lexical quarantine ===", flush=True)
    assert report_path.is_file(), f"Missing {report_path}"
    assert ledger_path.is_file(), f"Missing {ledger_path}"
    guarded = collect_guarded_markdown_paths(root)
    for req_p in (report_path, ledger_path):
        if req_p not in guarded:
            guarded.append(req_p)
    n_docs = check_markdown_quarantine(root, paths=guarded)
    print(f"OK quarantine: {n_docs} Markdown files verified clean", flush=True)

    print("=== [Phase J QA 4/4] ALL PHASE J QA CHECKS PASSED ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
