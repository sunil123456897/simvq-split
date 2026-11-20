"""
Standalone Zero-Manual QA Validator for Phase K (`validate_phase_k_results.py`).

Verifies integrity of:
1. Zero diff on protected Phases A–J artifacts vs `HEAD`, ruff syntax/fatal lint checks across
   the codebase, and all 52 passing Zero-Cheat tests (`17 + 6 + 4 + 5 + 5 + 5 + 5 + 5`).
2. `benchmarks/phase_k_via_negativa_probe.json` and `benchmarks/phase_k_scaling_law_results.json`
   finiteness, zero-ALU / zero-leakage corpus metadata (`>= 40,000` held-out hashes), `< 1.0%`
   parameter parity, `0.00%` mid-UTF8 cuts, `0.0` router drift, `O(1)` ring-buffer cache parity,
   monotone `LUT val_bpb` + `22.55x` quant-gap collapse, honest Stage-2 Chinchilla regression flags,
   and CPU roundtrip loading of all 3 packed `2-bit uint8 LUT` checkpoints (`75 / 95 / 105` layers).
3. `Phase_K_System1_Scaling_Report.md` and `research/post_transformer_synthesis/15_phase_k_scaling_law_ledger.md`
   existence, UTF-8 integrity, and strict Russian lexical quarantine across all repository docs.
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

from experiments.scaling_phase_k.eval_suite import load_lean_packed_lut_checkpoint  # noqa: E402
from experiments.scaling_phase_k.lean_system1 import (  # noqa: E402
    LeanSystem1Config,
    compute_lean_state_cache_kb,
)
from scripts.repo_guardrail import (  # noqa: E402
    check_anti_cheat_pytest,
    check_baselines_untouched,
    check_markdown_quarantine,
    check_ruff_lint,
    collect_guarded_markdown_paths,
)
from simvq_split.quant_lut import compute_2bit_lut_footprint  # noqa: E402

PROTECTED_TARGETS = (
    "simvq_split/",
    "experiments/cram_gsm8k/",
    "experiments/uebr_proto/",
    "experiments/uebr_system2/",
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
    "tests/test_phase_j_uebr_system2.py",
    "benchmarks/simvq_split_results.json",
    "benchmarks/phase_b_byte_chunker_results.json",
    "benchmarks/phase_c_cram_gsm8k_results.json",
    "benchmarks/phase_e_fast_utf8_results.json",
    "benchmarks/phase_f_utf8_iterative_cram_results.json",
    "benchmarks/phase_g_colab_scaleup_results.json",
    "benchmarks/phase_h_tier1_qat_closure_results.json",
    "benchmarks/phase_i_uebr_tracer_results.json",
    "benchmarks/phase_j_uebr_system2_results.json",
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
    probe_path = root / "benchmarks" / "phase_k_via_negativa_probe.json"
    res_path = root / "benchmarks" / "phase_k_scaling_law_results.json"
    report_path = root / "Phase_K_System1_Scaling_Report.md"
    ledger_path = root / "research" / "post_transformer_synthesis" / "15_phase_k_scaling_law_ledger.md"

    print("=== [Phase K QA 1/4] Protected diffs, ruff & all 52 Zero-Cheat pytest laws ===", flush=True)
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
        ("test_phase_k_scaling_law.py", "5 passed", 5),
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
    assert total_laws == 52, f"Expected 52 total Zero-Cheat laws, counted {total_laws}"

    print("=== [Phase K QA 2/4] Validating Phase K JSONs, honest scaling laws & 3 LUT checkpoints ===", flush=True)
    for jpath in (probe_path, res_path):
        assert jpath.is_file(), f"Missing {jpath}"
        _assert_no_nan_inf(json.loads(jpath.read_text(encoding="utf-8")), trail=jpath.name)

    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    for req_k in ("v0_seed2026", "v0_seed2027", "v1_attempt1", "selected_lean_params", "prior_diagnostics"):
        assert req_k in probe, f"Missing key `{req_k}` in {probe_path.name}"
    sel_p = probe["selected_lean_params"]
    assert sel_p["variant"] == "V1_lean_pruned"
    assert (int(sel_p["window_size"]), int(sel_p["crystal_w_chunks"]), int(sel_p["woodbury_latch_span"])) == (8, 2, 6)

    res = json.loads(res_path.read_text(encoding="utf-8"))
    proto = res["protocol"]
    assert proto["pure_system1_zero_alu_traces"] is True
    assert int(proto["vocab_size"]) == 259
    assert int(proto["corpus"]["alu_traces_injected"]) == 0
    assert int(proto["corpus"]["heldout_blocklist_hashes"]) >= 40_000
    assert (
        int(proto["selected_lean_params"]["window_size"]),
        int(proto["selected_lean_params"]["crystal_w_chunks"]),
        int(proto["selected_lean_params"]["woodbury_latch_span"]),
    ) == (8, 2, 6)

    tiers = res["tiers"]
    tier_specs = (
        ("S1_5M", "calcite5_s1_5m_utf8_lut.pt", 75, 99.6),
        ("S2_25M", "calcite5_s2_25m_utf8_lut.pt", 95, 209.25),
        ("S3_85M", "calcite5_s3_85m_utf8_lut.pt", 105, 449.54),
    )
    fp_vals, lut_vals, gaps = [], [], []
    for tname, ckname, exp_layers, exp_cache_kb in tier_specs:
        assert tname in tiers and len(tiers[tname]) == 3
        cal = tiers[tname]["CalciteLeanSystem1LM"]
        rope = tiers[tname]["RoPETransformerLM"]
        hyb = tiers[tname]["ScaledHybridDeltaSWALM"]

        c_p = float(cal["params"])
        for other_n, other in (("RoPE", rope), ("Hybrid", hyb)):
            rel_diff_pct = abs(float(other["params"]) - c_p) / c_p * 100.0
            assert rel_diff_pct < 1.0, f"{tname} {other_n} parity drift {rel_diff_pct:.3f}% >= 1.0%"

        assert float(cal["router_max_abs_diff"]) == 0.0
        assert float(cal["fp"]["max_mid_utf8_split_pct"]) == 0.0
        assert float(cal["lut_2bit"]["max_mid_utf8_split_pct"]) == 0.0
        assert abs(float(cal["lut_footprint"]["matrix_compression_vs_fp32"]) - 16.0) < 0.01
        assert abs(float(cal["cache_kb_L2048"]) - exp_cache_kb) < 0.02

        cfg_obj = LeanSystem1Config(**cal["config"])
        calc_kb = round(compute_lean_state_cache_kb(cfg_obj, 2048), 2)
        assert abs(calc_kb - float(cal["cache_kb_L2048"])) < 0.02

        ck_path = root / "checkpoints" / ckname
        assert ck_path.is_file(), f"Missing packed checkpoint {ck_path}"
        lm_k, _, _, meta = load_lean_packed_lut_checkpoint(ck_path, backend="pytorch_vec", device="cpu")
        fp_ck = compute_2bit_lut_footprint(lm_k)
        assert int(fp_ck["num_lut_layers"]) == exp_layers
        assert abs(float(fp_ck["matrix_compression_vs_fp32"]) - 16.0) < 0.01
        print(
            f"OK checkpoint {ckname}: {ck_path.stat().st_size:,} B, "
            f"{exp_layers} LUT layers, tier={meta['metadata'].get('tier')}",
            flush=True,
        )

        fp_v = float(cal["fp"]["val_memmap_bpb"])
        lut_v = float(cal["lut_2bit"]["val_memmap_bpb"])
        fp_vals.append(fp_v)
        lut_vals.append(lut_v)
        gaps.append(lut_v - fp_v)

    sf = res["scaling_fits"]
    assert lut_vals[0] > lut_vals[1] > lut_vals[2]
    assert gaps[0] > gaps[1] > gaps[2] and float(sf["quant_gap_val_bpb_collapse_ratio"]) >= 20.0
    assert sf["lut_val_bpb_monotonic_passed"] is True
    assert sf["quant_gap_val_bpb_monotonic_passed"] is True
    assert sf["strict_fp_val_monotonic_passed"] is False
    assert sf["strict_openweb_monotonic_passed"] is False
    assert sf["s1_to_s2_fp_and_ow_improved"] is True
    assert sf["cjk_utf8_fp_monotonic_passed"] is True

    print("=== [Phase K QA 3/4] Validating Phase K report, ledger & lexical quarantine ===", flush=True)
    assert report_path.is_file(), f"Missing {report_path}"
    assert ledger_path.is_file(), f"Missing {ledger_path}"
    guarded = collect_guarded_markdown_paths(root)
    for req_p in (report_path, ledger_path):
        if req_p not in guarded:
            guarded.append(req_p)
    n_docs = check_markdown_quarantine(root, paths=guarded)
    print(f"OK quarantine: {n_docs} Markdown files verified clean", flush=True)

    print("=== [Phase K QA 4/4] ALL PHASE K QA CHECKS PASSED ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
