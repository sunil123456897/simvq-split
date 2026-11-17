"""
Standalone Zero-Manual QA Validator for Phase I (`validate_phase_i_results.py`).

Verifies integrity of:
1. Zero diff on protected `simvq_split/`, `experiments/cram_gsm8k/`, `kernels/fast_packed_lut.py`,
   Tier-1 checkpoints, previous tests, and Phase A–H benchmark JSONs vs `HEAD`; ruff clean;
   `17 + 6 + 4 + 5 + 5 + 5 = 42` passing Zero-Cheat tests; and `simvq_cram_adapter.py --verify-alignment` (`100.00%`).
2. `benchmarks/phase_i_uebr_tracer_results.json` finiteness, two-seed `2*sigma` calibration (`2026`, `2027`),
   and all 4 UEBR mathematical gates + live Tier-1 `5.41 MB 2-bit LUT` probe (`AUROC >= 0.90`).
3. `research/post_transformer_synthesis/13_phase_i_uebr_blueprint.md` existence, UTF-8 integrity, and
   strict Russian lexical quarantine across all repository Markdown documents.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
    "kernels/fast_packed_lut.py",
    "checkpoints/calcite5_tier1_17m_utf8_lut.pt",
    "checkpoints/calcite5_tier1_17m_utf8_qat_h_lut.pt",
    "tests/test_anti_cheat.py",
    "tests/test_cram_phase_c.py",
    "tests/test_cram_phase_f.py",
    "tests/test_phase_g_lut_and_scale.py",
    "tests/test_phase_h_qat_closure.py",
    "benchmarks/simvq_split_results.json",
    "benchmarks/phase_b_byte_chunker_results.json",
    "benchmarks/phase_c_cram_gsm8k_results.json",
    "benchmarks/phase_e_fast_utf8_results.json",
    "benchmarks/phase_f_utf8_iterative_cram_results.json",
    "benchmarks/phase_g_colab_scaleup_results.json",
    "benchmarks/phase_h_tier1_qat_closure_results.json",
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
    json_path = root / "benchmarks" / "phase_i_uebr_tracer_results.json"
    blueprint_path = root / "research" / "post_transformer_synthesis" / "13_phase_i_uebr_blueprint.md"

    print("=== [Phase I QA 1/4] Protected diffs, ruff, 42 pytest laws & alignment ===", flush=True)
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
    for t_file, exp_n in (
        ("test_cram_phase_c.py", "6 passed"),
        ("test_cram_phase_f.py", "4 passed"),
        ("test_phase_g_lut_and_scale.py", "5 passed"),
        ("test_phase_h_qat_closure.py", "5 passed"),
        ("test_phase_i_uebr_tracer.py", "5 passed"),
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
        print(f"OK {t_file}: {res_t.stdout.strip().splitlines()[-1]}", flush=True)

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

    print("=== [Phase I QA 2/4] Validating Phase I UEBR tracer JSON & 2*sigma calibration ===", flush=True)
    assert json_path.exists(), f"Missing {json_path}"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    _assert_no_nan_inf(data)
    assert data["all_passed"] is True
    assert list(data["protocol"]["seeds"]) == [2026, 2027]
    assert float(data["protocol"]["elapsed_sec"]) < 60.0
    assert "two_sigma_noise_floor" in data
    for sd_key in ("seed_2026", "seed_2027"):
        for inv_k in ("invariant_1", "invariant_2", "invariant_3", "invariant_4a"):
            assert data[sd_key][inv_k]["passed"] is True, f"{sd_key}.{inv_k} failed"
    assert data["invariant_4b_live_tier1_lut"]["passed"] is True
    assert int(data["invariant_4b_live_tier1_lut"]["lut_layers"]) == 95
    assert float(data["invariant_4b_live_tier1_lut"]["auroc_combined_risk"]) >= 0.90

    print("=== [Phase I QA 3/4] Validating Phase I blueprint & lexical quarantine ===", flush=True)
    assert blueprint_path.exists(), f"Missing {blueprint_path}"
    guarded = collect_guarded_markdown_paths(root)
    if blueprint_path not in guarded:
        guarded.append(blueprint_path)
    n_docs = check_markdown_quarantine(root, paths=guarded)
    print(f"OK quarantine: {n_docs} Markdown files verified clean", flush=True)

    print("=== [Phase I QA 4/4] ALL PHASE I QA CHECKS PASSED ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
