"""
Standalone Zero-Manual QA Validator for Phase H (`validate_phase_h_results.py`).

Verifies integrity of:
1. Zero diff on `simvq_split/`, `experiments/cram_gsm8k/`, Phase G LUT checkpoint, and Phase A–G
   benchmark JSONs vs `HEAD`; ruff clean; `17 + 6 + 4 + 5 + 5 = 37` passing Zero-Cheat tests;
   and `simvq_cram_adapter.py --verify-alignment` (`100.00%`).
2. `checkpoints/calcite5_tier1_17m_utf8_qat_h_lut.pt` (`95` layers, `5,539.81 KB`, `16.00x`,
   `2,164 B` buffer, `230.11 KB` cache) and `benchmarks/phase_h_tier1_qat_closure_results.json`
   (Track 2 closure vs Phase G at `0.00%` mid-UTF8 cuts + Track 3 `CRAM+z` `vault_calls_used == 1`).
3. `Phase_H_Tier1_QAT_Closure_Report.md` UTF-8 integrity and strict Russian lexical quarantine across
   all repository Markdown documents.
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

from kernels.fast_packed_lut import (  # noqa: E402
    FastPackedLUT2BitLinear,
    load_packed_lut_checkpoint,
)
from scripts.colab_phase_g_runner import TIER1_CONFIG_DICT  # noqa: E402
from scripts.repo_guardrail import (  # noqa: E402
    check_anti_cheat_pytest,
    check_baselines_untouched,
    check_markdown_quarantine,
    check_ruff_lint,
    collect_guarded_markdown_paths,
)
from simvq_split.memory import compute_chunker_state_bytes  # noqa: E402
from simvq_split.quant_lut import compute_2bit_lut_footprint  # noqa: E402


def _assert_no_nan_inf(obj: Any, trail: str = "root") -> None:
    if isinstance(obj, float):
        assert math.isfinite(obj), f"Non-finite float at {trail}: {obj}"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _assert_no_nan_inf(v, f"{trail}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_no_nan_inf(v, f"{trail}[{i}]")


def _check_cram_split(res: dict, min_p1: float, min_p5: float, name: str) -> None:
    p1 = float(res["pass1_pct"])
    un = float(res["unary_ablation_pass1_pct"])
    p5 = float(res["pass5_pct"])
    lo, hi = [float(x) for x in res["pass1_ci95"]]
    assert 0.0 <= lo <= p1 <= hi <= 100.0, f"Invalid Wilson CI in {name}: {res['pass1_ci95']}"
    assert p1 >= min_p1, f"{name} Pass@1 below gate: {p1:.2f}% < {min_p1}%"
    assert p5 >= min_p5, f"{name} Pass@5 below gate: {p5:.2f}% < {min_p5}%"
    assert p1 >= 1.6 * un, f"{name} pairwise CRAM+z lift too weak: {p1:.2f}% vs 1.6 * {un:.2f}%"
    prec = float(res["cert_prec_pct"])
    cov = float(res["cert_cov_pct"])
    assert cov >= 10.0 and prec >= p1 + 15.0, f"{name} DualCert weak: prec={prec:.2f}% @ cov={cov:.2f}%"
    assert float(res["max_norm_drift"]) < 1e-6, f"{name} ALU drift >= 1e-6: {res['max_norm_drift']}"
    assert float(res["gershgorin_margin"]) >= 4.0, f"{name} Gershgorin < 4.0: {res['gershgorin_margin']}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase H JSON, LUT checkpoint, tests, and Markdown")
    parser.add_argument("--allow-dev-only", action="store_true", help="Allow dev validation without sealed test")
    args = parser.parse_args()

    root = ROOT_DIR
    json_path = root / "benchmarks" / "phase_h_tier1_qat_closure_results.json"
    lut_ckpt_path = root / "checkpoints" / "calcite5_tier1_17m_utf8_qat_h_lut.pt"
    report_path = root / "Phase_H_Tier1_QAT_Closure_Report.md"

    print("=== [Phase H QA 1/4] Protected diffs, ruff, 17+6+4+5+5=37 pytest laws & alignment ===", flush=True)
    check_baselines_untouched(root)
    for prot in (
        "simvq_split/",
        "experiments/cram_gsm8k/",
        "checkpoints/calcite5_tier1_17m_utf8_lut.pt",
        "tests/test_anti_cheat.py",
        "tests/test_cram_phase_c.py",
        "tests/test_cram_phase_f.py",
        "tests/test_phase_g_lut_and_scale.py",
        "benchmarks/simvq_split_results.json",
        "benchmarks/phase_b_byte_chunker_results.json",
        "benchmarks/phase_c_cram_gsm8k_results.json",
        "benchmarks/phase_e_fast_utf8_results.json",
        "benchmarks/phase_f_utf8_iterative_cram_results.json",
        "benchmarks/phase_g_colab_scaleup_results.json",
    ):
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

    print("=== [Phase H QA 2/4] Validating Phase H LUT ckpt, System-1 closure & CRAM+z ===", flush=True)
    assert lut_ckpt_path.exists(), f"Missing {lut_ckpt_path}"
    lut_m, loaded_cfg, _, _ = load_packed_lut_checkpoint(lut_ckpt_path, device="cpu", backend="pytorch_vec")
    fp_ckpt = compute_2bit_lut_footprint(lut_m)
    assert int(fp_ckpt["num_lut_layers"]) == 95
    assert abs(float(fp_ckpt["matrix_compression_vs_fp32"]) - 16.0) < 1e-3
    assert abs(float(fp_ckpt["total_lut_kb"]) - 5539.81) <= 0.1
    assert len([m for m in lut_m.modules() if isinstance(m, FastPackedLUT2BitLinear)]) == 95
    from simvq_split.models import SimVQSplitLM  # noqa: E402

    assert sum(p.numel() for p in SimVQSplitLM(loaded_cfg).parameters()) == 17_488_961
    assert (
        compute_chunker_state_bytes(
            loaded_cfg.d_model,
            loaded_cfg.byte_char_dim,
            loaded_cfg.byte_router_char_dim,
            byte_chunk_mode=loaded_cfg.byte_chunk_mode,
        )
        == 2164
    )

    assert json_path.exists(), f"Missing {json_path}"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    _assert_no_nan_inf(data)

    s12 = data["system1_qat_closure"]
    spec = s12["model_spec"]
    # Raw Colab JSON recorded `0` because `count_params(m_fp)` ran after freezing `m_fp` as teacher; ckpt enforces 17,488,961.
    assert int(spec["params"]) in (0, 17_488_961)
    assert dict(spec["config"]) == TIER1_CONFIG_DICT
    assert int(spec["chunker_state_bytes"]) == 2164
    assert 230.0 <= float(spec["state_cache_kb_L2048"]) <= 230.2
    assert float(s12["training"]["router_max_abs_diff"]) == 0.0
    assert s12["gap_closure"]["beats_phase_g_all_axes"] is True
    assert "block_alignment_v2" in s12 and float(s12["block_alignment_v2"]["mid_mlp"]["std_ratio"]) >= 0.75

    g_lut = data["phase_g_reference"]["phase_g_tier1_lut"]
    h_fp = s12["evaluations"]["tier1_h_fp"]
    h_qat = s12["evaluations"]["tier1_h_qat"]
    h_lut = s12["evaluations"]["tier1_h_lut_5_41mb"]
    for ev_b in (h_fp, h_qat, h_lut):
        assert float(ev_b["openweb"]["max_mid_utf8_split_pct"]) == 0.0
        assert float(ev_b["crystal_drift_L2048"]["norm_drift_L2048"]) < 1e-6
        assert float(ev_b["crystal_drift_L2048"]["voronoi_lattice_err_L2048"]) < 1e-6
    assert abs(float(h_qat["val_memmap_bpb"]) - float(h_lut["val_memmap_bpb"])) <= 0.01
    assert float(h_lut["val_memmap_bpb"]) < float(g_lut["val_memmap_bpb"])
    assert float(h_lut["openweb"]["mean_openweb_bpb"]) < float(g_lut["openweb"]["mean_openweb_bpb"]) - 0.30
    assert float(h_lut["gsm8k_te500"]["byte_bpb"]) < float(g_lut["gsm8k_te500"]["byte_bpb"]) - 0.12
    assert float(h_lut["s5_sum"]) >= 115.0 and float(h_fp["s5_sum"]) >= 130.0
    assert float(h_lut["overwrite"]["ood_long"]["exact_pct"]) >= 83.5

    t3 = data["track3_cram_gsm8k"]
    for opk in ("add_acc_pct", "sub_acc_pct", "mul_acc_pct", "div_acc_pct"):
        assert float(t3["alu_invariants"][opk]) == 100.0
    assert float(t3["alu_invariants"]["max_norm_drift"]) < 1e-6
    assert float(t3["cram_fit_fp"]["gershgorin_margin"]) >= 4.0
    assert float(t3["cram_fit_lut"]["gershgorin_margin"]) >= 4.0

    bd = t3["blind_dev"]
    _check_cram_split(bd["fp"], min_p1=18.0, min_p5=34.0, name="BlindDev_Tier1H_FP")
    _check_cram_split(bd["lut_on_fp_solver"], min_p1=18.0, min_p5=34.0, name="BlindDev_Tier1H_LUT_on_FPSolver")
    _check_cram_split(bd["lut_2bit"], min_p1=17.0, min_p5=34.0, name="BlindDev_Tier1H_2Bit_LUT")

    if not args.allow_dev_only:
        st = t3["sealed_test"]
        assert int(st["vault_calls_used"]) == 1
        assert int(st["fp"]["n"]) == 1319 and int(st["lut_2bit"]["n"]) == 1319
        _check_cram_split(st["fp"], min_p1=13.0, min_p5=30.0, name="SealedTest_Tier1H_FP")
        _check_cram_split(st["lut_2bit"], min_p1=14.0, min_p5=31.0, name="SealedTest_Tier1H_2Bit_LUT")

    print("=== [Phase H QA 3/4] Validating Markdown report & lexical quarantine ===", flush=True)
    assert report_path.exists(), f"Missing {report_path}"
    guarded = collect_guarded_markdown_paths(root)
    if report_path not in guarded:
        guarded.append(report_path)
    n_docs = check_markdown_quarantine(root, paths=guarded)
    print(f"OK quarantine: {n_docs} Markdown files verified clean", flush=True)

    print("=== [Phase H QA 4/4] ALL PHASE H QA CHECKS PASSED ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
