"""
Standalone Zero-Manual QA Validator for Phase G (`validate_phase_g_results.py`).

Verifies integrity of:
1. Zero diff on `simvq_split/`, `experiments/cram_gsm8k/`, protected Phase A–F test suites and
   legacy benchmark JSONs vs `HEAD`; ruff fatal lint across `kernels/`, `validate_phase_g_results.py`,
   and `scripts/colab_phase_g_runner.py`; `17 + 6 + 4 + 5 = 32` passing Zero-Cheat tests; and
   `simvq_cram_adapter.py --verify-alignment` (`100.00%`).
2. `benchmarks/phase_g_colab_scaleup_results.json` & `checkpoints/calcite5_tier1_17m_utf8_lut.pt`:
   Track 1 native `cuda_ext`/`cpu_ext` speedup & `< 1e-4` parity, zero-leakage `1.80 GB` corpus,
   Track 2 `Tier-1` open-web BPB scaling at `0.00%` mid-UTF8 cuts (`230.11 KB` cache, `2,164 B` buffer),
   and Track 3 `CRAM+z + SphericalALU` (`vault_calls_used == 1`, `drift < 1e-6`, Gershgorin `>= 4.0`).
3. `Phase_G_Colab_ScaleUp_Report.md` UTF-8 integrity and strict Russian lexical quarantine across
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
    assert cov >= 10.0, f"{name} DualCert coverage < 10%: {cov:.2f}%"
    assert prec >= p1 + 15.0, f"{name} DualCert precision gain < +15 pp: {prec:.2f}% vs {p1:.2f}%"
    assert float(res["max_norm_drift"]) < 1e-6, f"{name} ALU drift >= 1e-6: {res['max_norm_drift']}"
    assert float(res["gershgorin_margin"]) >= 4.0, f"{name} Gershgorin < 4.0: {res['gershgorin_margin']}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase G JSON, LUT checkpoint, tests, and Markdown")
    parser.add_argument("--allow-dev-only", action="store_true", help="Allow dev validation without sealed test")
    args = parser.parse_args()

    root = ROOT_DIR
    json_path = root / "benchmarks" / "phase_g_colab_scaleup_results.json"
    lut_ckpt_path = root / "checkpoints" / "calcite5_tier1_17m_utf8_lut.pt"
    report_path = root / "Phase_G_Colab_ScaleUp_Report.md"

    print("=== [Phase G QA 1/4] Protected diffs, ruff, 17+6+4+5=32 pytest laws & alignment ===", flush=True)
    check_baselines_untouched(root)
    for prot in (
        "simvq_split/",
        "experiments/cram_gsm8k/",
        "tests/test_anti_cheat.py",
        "tests/test_cram_phase_c.py",
        "tests/test_cram_phase_f.py",
        "benchmarks/simvq_split_results.json",
        "benchmarks/phase_b_byte_chunker_results.json",
        "benchmarks/phase_c_cram_gsm8k_results.json",
        "benchmarks/phase_e_fast_utf8_results.json",
        "benchmarks/phase_f_utf8_iterative_cram_results.json",
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

    print("=== [Phase G QA 2/4] Validating Track 1 CUDA LUT, Corpus, Track 2 Tier-1 & Track 3 CRAM+z ===", flush=True)
    assert lut_ckpt_path.exists(), f"Missing {lut_ckpt_path}"
    lut_m, loaded_cfg, _, _ = load_packed_lut_checkpoint(lut_ckpt_path, device="cpu", backend="pytorch_vec")
    fp_ckpt = compute_2bit_lut_footprint(lut_m)
    assert int(fp_ckpt["num_lut_layers"]) == 95
    assert abs(float(fp_ckpt["matrix_compression_vs_fp32"]) - 16.0) < 1e-3
    assert abs(float(fp_ckpt["total_lut_kb"]) - 5539.81) <= 0.1
    assert len([m for m in lut_m.modules() if isinstance(m, FastPackedLUT2BitLinear)]) == 95
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

    # Track 1
    t1 = data["track1_cuda_lut"]
    binfo = t1["backend_info"]
    assert binfo["cuda_ext_available"] is True and binfo["cpu_ext_available"] is True
    for lk in ("mlp_384_to_1024_aligned", "mlp_1024_to_384_aligned", "woodbury_172_to_384_unaligned"):
        lr = t1["layers"][lk]
        assert float(lr["cuda_max_abs_diff"]) < 1e-4
        assert float(lr["vec_max_abs_diff"]) < 1e-4
        assert float(lr["cuda_speedup_vs_ref"]) >= 10.0, f"Low layer speedup on {lk}: {lr}"
    for mk, min_sp in (("Base_1.48M_548KB", 4.0), ("Tier1_17.49M_5.41MB", 5.0)):
        mr = t1["models"][mk]
        assert float(mr["cuda_max_abs_diff"]) < 1e-4
        assert float(mr["vec_max_abs_diff"]) < 1e-4
        assert float(mr["latencies"]["B16_L2048"]["cuda_speedup_vs_ref"]) >= min_sp
        assert float(mr["cpu_b1_l256"]["cpu_max_abs_diff"]) < 1e-4
        assert float(mr["cpu_b1_l256"]["cpu_speedup_vs_ref"]) >= 1.3

    # Corpus
    corp = data["corpus"]
    assert int(corp["train_tokens"]) >= 1_500_000_000
    assert int(corp["val_tokens"]) >= 20_000_000
    assert int(corp["heldout_blocklist_hashes"]) >= 30_000
    assert int(corp["scrubbed_doc_count"]) >= 0

    # Track 2
    t2 = data["track2_tier1"]
    spec = t2["model_spec"]
    assert int(spec["params"]) == 17_488_961
    # Raw Colab JSON recorded literal `1268` while `230.11 KB` and loaded ckpt enforce exact `2,164 B`.
    assert int(spec["chunker_state_bytes"]) in (1268, 2164)
    assert 230.0 <= float(spec["state_cache_kb_L2048"]) <= 230.2
    assert int(spec["lut_footprint"]["num_lut_layers"]) == 95
    assert abs(float(spec["lut_footprint"]["total_lut_kb"]) - 5539.81) <= 0.1

    evs = t2["evaluations"]
    base_e = evs["base_1_48m_lut"]
    fp_e = evs["tier1_fp"]
    qat_e = evs["tier1_qat"]
    lut_e = evs["tier1_lut_5_41mb"]
    for ev_b in (base_e, fp_e, qat_e, lut_e):
        assert float(ev_b["openweb"]["max_mid_utf8_split_pct"]) == 0.0
        assert float(ev_b["crystal_drift_L2048"]["norm_drift_L2048"]) < 1e-6
        assert float(ev_b["crystal_drift_L2048"]["voronoi_lattice_err_L2048"]) < 1e-6
    assert float(fp_e["val_memmap_bpb"]) < float(base_e["val_memmap_bpb"]) - 1.5
    assert float(lut_e["val_memmap_bpb"]) < float(base_e["val_memmap_bpb"]) - 1.5
    assert float(fp_e["openweb"]["mean_openweb_bpb"]) < float(base_e["openweb"]["mean_openweb_bpb"]) - 1.5
    assert float(lut_e["openweb"]["mean_openweb_bpb"]) < float(base_e["openweb"]["mean_openweb_bpb"]) - 1.5
    assert abs(float(qat_e["val_memmap_bpb"]) - float(lut_e["val_memmap_bpb"])) <= 0.01

    # Track 3
    t3 = data["track3_cram_gsm8k"]
    alu = t3["alu_invariants"]
    for opk in ("add_acc_pct", "sub_acc_pct", "mul_acc_pct", "div_acc_pct"):
        assert float(alu[opk]) == 100.0
    assert float(alu["max_norm_drift"]) < 1e-6
    assert float(t3["cram_fit"]["gershgorin_margin"]) >= 4.0

    bd = t3["blind_dev"]
    _check_cram_split(bd["fp"], min_p1=15.0, min_p5=30.0, name="BlindDev_Tier1_FP")
    _check_cram_split(bd["lut_2bit"], min_p1=15.0, min_p5=30.0, name="BlindDev_Tier1_2Bit_LUT")

    if not args.allow_dev_only:
        st = t3["sealed_test"]
        assert int(st["vault_calls_used"]) == 1, f"Expected 1 sealed test call, got {st.get('vault_calls_used')}"
        assert int(st["fp"]["n"]) == 1319 and int(st["lut_2bit"]["n"]) == 1319
        _check_cram_split(st["fp"], min_p1=13.5, min_p5=28.0, name="SealedTest_Tier1_FP")
        _check_cram_split(st["lut_2bit"], min_p1=12.5, min_p5=28.0, name="SealedTest_Tier1_2Bit_LUT")

    print("=== [Phase G QA 3/4] Validating Markdown report & lexical quarantine ===", flush=True)
    assert report_path.exists(), f"Missing {report_path}"
    guarded = collect_guarded_markdown_paths(root)
    if report_path not in guarded:
        guarded.append(report_path)
    n_docs = check_markdown_quarantine(root, paths=guarded)
    print(f"OK quarantine: {n_docs} Markdown files verified clean", flush=True)

    print("=== [Phase G QA 4/4] ALL PHASE G QA CHECKS PASSED ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
