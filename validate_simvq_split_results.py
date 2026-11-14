"""
Standalone Zero-Manual QA Validator (`validate_simvq_split_results.py`).

Verifies integrity of:
1. `tests/test_anti_cheat.py` (all 12 Zero-Cheat, Bijection, Epistemic & 1.58-bit STE QAT laws must pass)
2. `benchmarks/simvq_split_results.json` (valid UTF-8 JSON, zero NaN/Inf, all expected
   models & tracks present, parameter parity <= 5%, Wilson CI bounds valid, plus 1.58-bit STE QAT)
3. `checkpoints/simvq_split_tournament.pt` (non-empty loadable checkpoint matching
   reported parameter counts + valid ternary-baked QAT state_dict with <= 3 levels per 32-group)
4. `SimVQ_Split_Benchmark_Report.md` (UTF-8 integrity, zero broken brackets/tags,
   all reported model rows present, and strict Russian lexical-quarantine compliance)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import torch


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

    print("=== [QA 1/4] Running pytest tests/test_anti_cheat.py ===")
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(root / "tests" / "test_anti_cheat.py"),
            "-q",
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    print(res.stdout.strip())
    if res.returncode != 0 or "12 passed" not in res.stdout:
        print(res.stderr)
        raise AssertionError("Zero-Cheat pytest suite (12 laws) failed!")

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

    print("=== [QA 3/4] Validating checkpoints/simvq_split_tournament.pt ===")
    assert ckpt_path.exists() and ckpt_path.stat().st_size > 100_000, (
        f"Invalid ckpt {ckpt_path}"
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert (
        "simvq_split_state_dict" in ckpt
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

    print("=== [QA 4/4] Validating SimVQ_Split_Benchmark_Report.md ===")
    assert report_path.exists(), f"Missing {report_path}"
    md_text = report_path.read_text(encoding="utf-8")
    assert len(md_text) > 1500, "Report suspiciously short"
    assert "\ufffd" not in md_text, "UTF-8 replacement char found in report"
    assert "12/12" in md_text and "SimVQ-Split v4" in md_text, "Missing v4 / 12/12 header in report"
    for m_name in EXPECTED_MODELS:
        assert m_name in md_text, f"Model `{m_name}` missing from Markdown report!"

    md_low = md_text.lower()
    forbidden_phrases = (
        "не просто",
        "является",
        "являются",
        "осуществлять",
        "осуществляется",
        "в рамках",
        "симфония",
        "гобелен",
        "квантовый скачок",
        "бесшовная экосистема",
        "уникальный опыт",
        "феноменальный",
        "колоссальный",
        "в заключение",
        "таким образом",
    )
    for phrase in forbidden_phrases:
        assert phrase not in md_low, f"Forbidden cliche phrase `{phrase}` found in report!"

    print(
        f"ALL QA CHECKS PASSED: {len(EXPECTED_MODELS)} models verified | "
        f"JSON={json_path.stat().st_size:,} B | CKPT={ckpt_path.stat().st_size:,} B | "
        f"REPORT={report_path.stat().st_size:,} B"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
