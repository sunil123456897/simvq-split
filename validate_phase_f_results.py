"""
Standalone Zero-Manual QA Validator for Phase F (`validate_phase_f_results.py`).

Verifies integrity of:
1. Untouched `simvq_split/` & `benchmarks/phase_c_cram_gsm8k_results.json` vs `HEAD`, ruff fatal lint,
   `17 + 6 + 4 = 27` Zero-Cheat laws across Phase A–F pytest suites, and `--verify-alignment`
   (`100.00%` BPE-1536 & UTF-8 alignment without torch/test.jsonl).
2. `benchmarks/phase_f_utf8_iterative_cram_results.json` (Track 1 `N=16..256` & `ng=4..16`, Track 2
   `548.34 KB` UTF-8 ByteHNet `Cal`/`BlindDev` gates, Track 3 single-shot `Sealed Test` `max_calls=1`).
3. `Phase_F_UTF8_Iterative_CRAM_Report.md` UTF-8 integrity and strict Russian lexical quarantine
   across all repository Markdown documents.
"""

from __future__ import annotations

import argparse
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


def _check_split_block(res: dict, min_p1: float, min_p5: float | None, name: str) -> None:
    p1 = float(res["pass1_pct"])
    un = float(res["unary_ablation_pass1_pct"])
    p5 = float(res["pass5_pct"])
    lo, hi = [float(x) for x in res["pass1_ci95"]]
    assert 0.0 <= lo <= p1 <= hi <= 100.0, f"Invalid Wilson CI in {name}: {res['pass1_ci95']}"
    assert p1 >= min_p1, f"{name} Pass@1 too low: {p1:.2f}% < {min_p1}%"
    if min_p5 is not None:
        assert p5 >= min_p5, f"{name} Pass@5 too low: {p5:.2f}% < {min_p5}%"
    assert p1 >= 1.6 * un, f"{name} pairwise coupling lift too weak: {p1:.2f}% < 1.6 * {un:.2f}%"
    prec = float(res["cert_prec_pct"])
    cov = float(res["cert_cov_pct"])
    assert cov >= 6.0, f"{name} DualCert coverage too low: {cov:.2f}% < 6.0%"
    assert prec >= p1 + 15.0, f"{name} DualCert precision gain < +15 pp: {prec:.2f}% vs {p1:.2f}%"
    assert float(res["max_norm_drift"]) < 1e-6, f"{name} ALU drift >= 1e-6: {res['max_norm_drift']}"
    assert float(res["gershgorin_margin"]) >= 4.0, f"{name} Gershgorin < 4.0: {res['gershgorin_margin']}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase F JSON, tests, and Markdown quarantine")
    parser.add_argument("--allow-dev-only", action="store_true", help="Allow `--skip-sealed-test` dev validation")
    args = parser.parse_args()

    root = ROOT_DIR
    json_path = root / "benchmarks" / "phase_f_utf8_iterative_cram_results.json"
    report_path = root / "Phase_F_UTF8_Iterative_CRAM_Report.md"

    print("=== [Phase F QA 1/4] Protected diffs, ruff, 17+6+4 pytest laws & alignment ===", flush=True)
    check_baselines_untouched(root)
    for prot in (
        "simvq_split/",
        "tests/test_anti_cheat.py",
        "tests/test_cram_phase_c.py",
        "benchmarks/phase_c_cram_gsm8k_results.json",
        "benchmarks/phase_c_v5_calibration_slice.json",
    ):
        chk_diff = subprocess.run(
            ["git", "diff", "--exit-code", "HEAD", "--", prot],
            cwd=str(root),
            capture_output=True,
            text=True,
        )
        assert chk_diff.returncode == 0, f"Protected target modified `{prot}`:\n{chk_diff.stdout}\n{chk_diff.stderr}"

    check_ruff_lint(root)
    chk_ruff_f = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--select",
            "E9,F63,F7,F82",
            str(root / "run_phase_f_utf8_cram_benchmark.py"),
            str(root / "validate_phase_f_results.py"),
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    assert chk_ruff_f.returncode == 0, f"Phase F ruff lint failed:\n{chk_ruff_f.stdout}\n{chk_ruff_f.stderr}"

    print(check_anti_cheat_pytest(root), flush=True)
    for t_file, exp_n in (("test_cram_phase_c.py", "6 passed"), ("test_cram_phase_f.py", "4 passed")):
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
        [sys.executable, "-X", "utf8", "-u", str(root / "experiments" / "cram_gsm8k" / "simvq_cram_adapter.py"), "--verify-alignment"],
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert res_al.returncode == 0 and "100.00%" in res_al.stdout, (
        f"Alignment verifier failed:\n{res_al.stdout}\n{res_al.stderr}"
    )

    print("=== [Phase F QA 2/4] Validating Track 1 Scaling & Track 2/3 GSM8K JSON ===", flush=True)
    assert json_path.exists(), f"Missing {json_path}"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    _assert_no_nan_inf(data)

    proto = data["protocol"]
    assert int(proto["vocab_size"]) == 259
    assert int(proto["chunker_state_bytes"]) == 1268
    assert int(proto["sealed_test_max_calls"]) == 1

    lut = data["lut_footprint"]
    assert int(lut["num_lut_layers"]) == 70, f"Expected 70 UTF-8 LUT layers, got {lut['num_lut_layers']}"
    assert abs(float(lut["matrix_compression_vs_fp32"]) - 16.0) < 1e-3
    assert 535.0 <= float(lut["total_lut_kb"]) <= 555.0, f"Unexpected LUT KB: {lut['total_lut_kb']}"

    alu = data["alu_invariants"]
    for opk in ("add_acc_pct", "sub_acc_pct", "mul_acc_pct", "div_acc_pct"):
        assert float(alu[opk]) == 100.0
    assert float(alu["max_norm_drift"]) < 1e-6

    t1 = data["track1_barrier3_scaling"]
    for nk in ("N16", "N64", "N128", "N256"):
        zr = t1["continuous_z_uniqueness"][nk]
        assert float(zr["max_pair_linf"]) < 1e-6
        assert float(zr["max_err_vs_closed_form"]) < 1e-6
        assert float(zr["gershgorin_margin"]) >= 4.0

    for nk in ("N16", "N32", "N64", "N128"):
        cr = t1["coupled_2opt_vs_1flip"][nk]
        assert float(cr["block_gibbs_2opt"]["exact_recovery_pct"]) >= 95.0, f"2-opt < 95% on {nk}: {cr}"
        if nk in ("N64", "N128"):
            assert float(cr["besag_1flip"]["exact_recovery_pct"]) <= 5.0, f"1-flip unexpectedly high on {nk}: {cr}"

    for ngk in ("ng4", "ng6", "ng8", "ng10", "ng12", "ng16"):
        sr = t1["high_slot_synthesis"][ngk]
        assert float(sr["non_empty_pool_pct"]) == 100.0
        assert float(sr["mean_latency_ms"]) < 100.0
        if ngk in ("ng10", "ng12", "ng16"):
            assert float(sr["oracle_hit_pct"]) > 0.0, f"Expected non-zero high-slot oracle on {ngk}"

    dev = data["dev_validation"]
    bd_fp = dev["blind_dev_fp"]
    bd_lut = dev["blind_dev_lut"]
    _check_split_block(bd_fp, min_p1=15.0, min_p5=28.0, name="BlindDev_FP")
    _check_split_block(bd_lut, min_p1=15.0, min_p5=28.0, name="BlindDev_2Bit_LUT")
    assert abs(float(bd_fp["pass1_pct"]) - float(bd_lut["pass1_pct"])) <= 2.0

    if not args.allow_dev_only:
        assert int(proto["sealed_test_calls_used"]) == 1, (
            f"Expected exactly 1 sealed test call, got {proto['sealed_test_calls_used']}"
        )
        sealed = data["sealed_test"]
        assert sealed is not None, "Missing sealed_test section!"
        st_fp = sealed["sealed_test_fp"]
        st_lut = sealed["sealed_test_2bit_lut"]
        _check_split_block(st_fp, min_p1=13.5, min_p5=28.0, name="SealedTest_FP")
        _check_split_block(st_lut, min_p1=13.5, min_p5=28.0, name="SealedTest_2Bit_LUT")
        assert abs(float(st_fp["pass1_pct"]) - float(st_lut["pass1_pct"])) <= 2.0
    else:
        assert int(proto["sealed_test_calls_used"]) == 0, "Expected 0 sealed calls in dev-only mode!"

    print("=== [Phase F QA 3/4] Validating Markdown report & lexical quarantine ===", flush=True)
    assert report_path.exists(), f"Missing {report_path}"
    from scripts.repo_guardrail import collect_guarded_markdown_paths  # noqa: E402

    guarded = collect_guarded_markdown_paths(root)
    if report_path not in guarded:
        guarded.append(report_path)
    n_docs = check_markdown_quarantine(root, paths=guarded)
    print(f"OK quarantine: {n_docs} Markdown files verified clean", flush=True)

    print("=== [Phase F QA 4/4] ALL PHASE F QA CHECKS PASSED ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
