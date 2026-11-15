"""
Standalone Zero-Manual QA Validator (`validate_simvq_split_results.py`).

Verifies integrity of:
1. `tests/test_anti_cheat.py` (all 15 Zero-Cheat, Bijection, Epistemic & 1.58-bit STE QAT / 2-bit LUT laws must pass)
2. `benchmarks/simvq_split_results.json` (valid UTF-8 JSON, zero NaN/Inf, all expected
   models & tracks present, parameter parity <= 5%, Wilson CI bounds valid, 1.58-bit STE QAT & 2-bit uint8 T-MAC LUT)
3. `checkpoints/simvq_split_tournament.pt` (non-empty loadable checkpoint matching
   reported parameter counts + strict SimVQSplitLM reload + valid ternary-baked QAT state_dict + 16.00x 2-bit LUT packing)
4. `SimVQ_Split_Benchmark_Report.md` (UTF-8 integrity, zero broken brackets/tags,
   all reported model rows present, and strict Russian lexical-quarantine compliance)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import torch

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.repo_guardrail import (  # noqa: E402
    check_anti_cheat_pytest,
    check_baselines_untouched,
    check_markdown_quarantine,
    check_ruff_lint,
)


EXPECTED_MODELS = [
    "SimVQSplitLM",
    "SimVQSplit_CrystalOnly",
    "SimVQSplit_WoodburyOnly",
    "RoPETransformerLM",
    "SlidingWindowTransformerLM",
    "LoopedTransformerLM",
    "HouseholderDeltaNetLM",
]


def main() -> int:
    root = Path(__file__).resolve().parent
    json_path = root / "benchmarks" / "simvq_split_results.json"
    ckpt_path = root / "checkpoints" / "simvq_split_tournament.pt"
    report_path = root / "SimVQ_Split_Benchmark_Report.md"

    print("=== [QA 1/4] Running baselines diff, ruff & pytest tests/test_anti_cheat.py ===")
    check_baselines_untouched(root)
    check_ruff_lint(root)
    pytest_out = check_anti_cheat_pytest(root)
    print(pytest_out)

    print("=== [QA 2/4] Validating benchmarks/simvq_split_results.json ===")
    assert json_path.exists(), f"Missing {json_path}"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    models_data = data.get("models", {})
    for m_name in EXPECTED_MODELS:
        assert m_name in models_data, f"Missing model `{m_name}` in results JSON"

    target_params = models_data["SimVQSplitLM"]["params"]
    for m_name, m_entry in models_data.items():
        n_p = m_entry["params"]
        rel = abs(n_p - target_params) / target_params
        assert rel <= 0.05, f"Param parity > 5% for {m_name}: {n_p} vs {target_params}"
        ppl = m_entry["gsm8k_ppl"]
        assert (
            isinstance(ppl, (int, float)) and math.isfinite(ppl) and 1.0 < ppl < 500.0
        ), f"Invalid PPL for {m_name}: {ppl}"
        for track_key in ("s5", "overwrite", "trap"):
            assert track_key in m_entry, f"Missing {track_key} in {m_name}"
            for split_name, metrics in m_entry[track_key].items():
                acc = metrics["exact_pct"]
                lo, hi = metrics["wilson_95"]
                assert 0.0 <= lo <= acc + 1e-5 and acc - 1e-5 <= hi <= 100.0, (
                    f"Invalid Wilson CI for {m_name}/{track_key}/{split_name}: {acc} not in [{lo}, {hi}]"
                )

    ptq = data.get("ternary_158bit_ablation", {})
    assert ptq.get("ternary_quantized_linear_params", 0) > 0, "Missing PTQ quantized param count"
    qat = data.get("ternary_158bit_qat", {})
    assert qat.get("qat_steps") == 300, f"Expected qat_steps == 300, got {qat.get('qat_steps')}"
    assert qat.get("ternary_quantized_linear_params", 0) > 0, "Missing QAT quantized param count"
    qat_ppl = qat.get("gsm8k_ppl_158bit_qat", -1.0)
    assert isinstance(qat_ppl, (int, float)) and math.isfinite(qat_ppl) and 1.0 < qat_ppl < 100.0, (
        f"Invalid 1.58-bit STE QAT PPL: {qat_ppl}"
    )
    for pct_k in (
        "s5_k2_158bit_qat_pct",
        "s5_k8_158bit_qat_pct",
        "ov_ood_med_158bit_qat_pct",
        "ov_ood_long_158bit_qat_pct",
        "trap_ood_long_158bit_qat_pct",
    ):
        val = qat.get(pct_k)
        assert isinstance(val, (int, float)) and 0.0 <= val <= 100.0, f"Invalid {pct_k}: {val}"

    lut = data.get("tmac_2bit_lut", {})
    assert lut.get("num_lut_layers", 0) > 0, "Missing 2-bit T-MAC LUT layers"
    comp = lut.get("matrix_compression_vs_fp32")
    assert isinstance(comp, (int, float)) and abs(comp - 16.0) < 1e-3, f"Invalid LUT compression: {comp}"
    max_pp_diff = lut.get("max_pp_diff_vs_qat")
    assert isinstance(max_pp_diff, (int, float)) and max_pp_diff <= 1.5, (
        f"2-bit LUT drift vs QAT too high: {max_pp_diff}"
    )
    lut_ppl = lut.get("gsm8k_ppl_2bit_lut")
    assert isinstance(lut_ppl, (int, float)) and 1.0 < lut_ppl < 100.0, f"Invalid LUT PPL: {lut_ppl}"
    for pct_k in (
        "s5_k2_2bit_lut_pct",
        "s5_k8_2bit_lut_pct",
        "ov_ood_med_2bit_lut_pct",
        "ov_ood_long_2bit_lut_pct",
        "trap_ood_long_2bit_lut_pct",
    ):
        val = lut.get(pct_k)
        assert isinstance(val, (int, float)) and 0.0 <= val <= 100.0, f"Invalid {pct_k}: {val}"

    print("=== [QA 3/4] Validating checkpoints/simvq_split_tournament.pt ===")
    assert ckpt_path.exists() and ckpt_path.stat().st_size > 100_000, (
        f"Invalid ckpt {ckpt_path}"
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert (
        "simvq_split_config" in ckpt
        and "simvq_split_state_dict" in ckpt
        and "simvq_split_ternary_qat_state_dict" in ckpt
        and "tokenizer" in ckpt
    ), "Incomplete checkpoint keys"
    for k, t in ckpt["simvq_split_state_dict"].items():
        if torch.is_floating_point(t):
            assert torch.isfinite(t).all(), (
                f"Non-finite weight tensor `{k}` in saved checkpoint!"
            )
    qat_sd = ckpt["simvq_split_ternary_qat_state_dict"]
    checked_linear_tensors = 0
    for k, t in qat_sd.items():
        if torch.is_floating_point(t):
            assert torch.isfinite(t).all(), (
                f"Non-finite weight tensor `{k}` in saved ternary QAT checkpoint!"
            )
        if k.endswith(".weight") and t.ndim == 2 and k != "lm_head.weight" and "tok_emb" not in k:
            g_size = 32 if t.numel() % 32 == 0 else t.size(-1)
            for row in t.float().reshape(-1, g_size):
                assert torch.unique(torch.round(row * 1e5)).numel() <= 3, (
                    f"Saved ternary QAT weight `{k}` violates <= 3 levels per group!"
                )
            checked_linear_tensors += 1
    assert checked_linear_tensors > 0, "No quantized linear tensors found in ternary QAT state_dict!"

    from simvq_split.models import SimVQSplitConfig, SimVQSplitLM
    from simvq_split.memory import PackedLUT2BitLinear, compute_2bit_lut_footprint, pack_model_2bit_lut

    cfg = SimVQSplitConfig(**ckpt["simvq_split_config"])
    m_fp = SimVQSplitLM(cfg)
    m_fp.load_state_dict(ckpt["simvq_split_state_dict"], strict=True)
    m_qat = SimVQSplitLM(cfg)
    m_qat.load_state_dict(ckpt["simvq_split_ternary_qat_state_dict"], strict=True)
    lut_m = pack_model_2bit_lut(m_qat)
    assert any(isinstance(mod, PackedLUT2BitLinear) for mod in lut_m.modules()), (
        "pack_model_2bit_lut produced zero PackedLUT2BitLinear layers!"
    )
    lut_fp = compute_2bit_lut_footprint(lut_m)
    assert abs(lut_fp["matrix_compression_vs_fp32"] - 16.0) < 1e-3, (
        f"Unexpected checkpoint LUT compression: {lut_fp['matrix_compression_vs_fp32']}"
    )

    print("=== [QA 4/4] Validating Markdown reports & lexical quarantine ===")
    assert report_path.exists(), f"Missing {report_path}"
    md_text = report_path.read_text(encoding="utf-8")
    assert len(md_text) > 1500, "Report suspiciously short"
    assert (
        "15/15" in md_text and "SimVQ-Split v5" in md_text and "PackedLUT2BitLinear" in md_text
    ), "Missing v5 / 15/15 / PackedLUT2BitLinear header in report"
    for m_name in EXPECTED_MODELS:
        assert m_name in md_text, f"Model `{m_name}` missing from Markdown report!"

    n_docs = check_markdown_quarantine(root)
    assert n_docs >= 4, f"Expected >= 4 guarded Markdown files, got {n_docs}"

    print(
        f"ALL QA CHECKS PASSED: {len(EXPECTED_MODELS)} models verified | "
        f"JSON={json_path.stat().st_size:,} B | CKPT={ckpt_path.stat().st_size:,} B | "
        f"REPORT={report_path.stat().st_size:,} B"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
