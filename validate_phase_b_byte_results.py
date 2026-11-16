"""
Standalone Zero-Manual QA Validator for Phase B (`validate_phase_b_byte_results.py`).

Verifies integrity of:
1. Untouched `simvq_split/baselines.py` vs HEAD, ruff fatal lint, `16/16` Zero-Cheat laws
   (`tests/test_anti_cheat.py`) and `6/6` Phase C laws (`tests/test_cram_phase_c.py`).
2. `benchmarks/phase_b_byte_chunker_results.json` (all 7 raw-byte models, <= 5% param parity,
   valid Wilson CIs, `O(1)` state cache 80..83.5 KB, chunk ratio 3.8..5.2, 2-bit LUT <= 555 KB at 16.0x,
   `bpe_equiv_ppl <= 48.0`, `max_pp_diff_vs_qat <= 2.5`).
3. `checkpoints/simvq_split_byte_hnet.pt` (`strict=True` FP & QAT reload into `SimVQSplitLM`,
   finite tensors, <= 3 ternary levels per 32-group across all `nn.Linear` layers except `lm_head`,
   physical `PackedLUT2BitLinear` packing, and live CUDA forward sanity).
4. `Phase_B_Byte_Chunker_Report.md` and strict Russian lexical quarantine across all repo docs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.repo_guardrail import (  # noqa: E402
    check_anti_cheat_pytest,
    check_baselines_untouched,
    check_markdown_quarantine,
    check_ruff_lint,
)

EXPECTED_MODELS = (
    "SimVQSplitLM",
    "SimVQSplit_CrystalOnly",
    "SimVQSplit_WoodburyOnly",
    "RoPETransformerLM",
    "SlidingWindowTransformerLM",
    "LoopedTransformerLM",
    "HouseholderDeltaNetLM",
)


def main() -> int:
    root = ROOT_DIR
    json_path = root / "benchmarks" / "phase_b_byte_chunker_results.json"
    ckpt_path = root / "checkpoints" / "simvq_split_byte_hnet.pt"
    report_path = root / "Phase_B_Byte_Chunker_Report.md"

    print("=== [Phase B QA 1/4] Checking baselines diff, ruff, 16+6 pytest laws ===", flush=True)
    check_baselines_untouched(root)
    check_ruff_lint(root)
    print(check_anti_cheat_pytest(root), flush=True)
    chk_c = subprocess.run(
        [sys.executable, "-m", "pytest", str(root / "tests" / "test_cram_phase_c.py"), "-q"],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    assert chk_c.returncode == 0 and "6 passed" in chk_c.stdout, (
        f"Phase C pytest failed:\n{chk_c.stdout}\n{chk_c.stderr}"
    )
    print(chk_c.stdout.strip(), flush=True)

    print("=== [Phase B QA 2/4] Validating benchmarks/phase_b_byte_chunker_results.json ===", flush=True)
    assert json_path.exists(), f"Missing {json_path}"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    models_data = data.get("models", {})
    for m_name in EXPECTED_MODELS:
        assert m_name in models_data, f"Missing model `{m_name}` in {json_path.name}"

    target_params = models_data["SimVQSplitLM"]["params"]
    for m_name, m_entry in models_data.items():
        n_p = m_entry["params"]
        assert abs(n_p - target_params) / target_params <= 0.05, (
            f"Param parity > 5% for {m_name}: {n_p} vs {target_params}"
        )
        bpb = m_entry["byte_bpb"]
        eq_ppl = m_entry["bpe_equiv_ppl"]
        assert isinstance(bpb, (int, float)) and math.isfinite(bpb) and 1.0 < bpb < 10.0
        assert isinstance(eq_ppl, (int, float)) and math.isfinite(eq_ppl) and 1.0 < eq_ppl < 500.0
        for track_key in ("s5", "overwrite", "trap"):
            for split_name, metrics in m_entry[track_key].items():
                acc = metrics["exact_pct"]
                lo, hi = metrics["wilson_95"]
                assert 0.0 <= lo <= acc + 1e-5 and acc - 1e-5 <= hi <= 100.0, (
                    f"Invalid Wilson CI for {m_name}/{track_key}/{split_name}"
                )

    sim_entry = models_data["SimVQSplitLM"]
    assert 80.0 <= sim_entry["state_cache_kb_L2048"] <= 83.5, (
        f"SimVQSplitLM cache out of bounds: {sim_entry['state_cache_kb_L2048']}"
    )
    assert 3.8 <= sim_entry["mean_bytes_per_chunk"] <= 5.2, (
        f"SimVQSplitLM chunk ratio out of bounds: {sim_entry['mean_bytes_per_chunk']}"
    )

    qat = data.get("ternary_158bit_qat", {})
    assert qat.get("qat_steps") == 500, f"Expected qat_steps == 500, got {qat.get('qat_steps')}"
    lut = data.get("tmac_2bit_lut", {})
    assert lut.get("total_lut_kb", 999.0) <= 555.0, f"2-bit LUT exceeded 555 KB: {lut.get('total_lut_kb')}"
    assert abs(lut.get("matrix_compression_vs_fp32", 0.0) - 16.0) < 1e-3
    assert 3.8 <= lut.get("mean_bytes_per_chunk", 0.0) <= 5.2
    assert lut.get("bpe_equiv_ppl", 999.0) <= 48.0, f"LUT BPE_EqPPL > 48.0: {lut.get('bpe_equiv_ppl')}"
    assert lut.get("max_pp_diff_vs_qat", 999.0) <= 2.5

    print("=== [Phase B QA 3/4] Validating checkpoints/simvq_split_byte_hnet.pt ===", flush=True)
    assert ckpt_path.exists() and ckpt_path.stat().st_size > 100_000, f"Invalid {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for req_k in (
        "simvq_split_config",
        "simvq_split_state_dict",
        "simvq_split_ternary_qat_state_dict",
        "tokenizer",
    ):
        assert req_k in ckpt, f"Missing checkpoint key `{req_k}`"

    for k, t in ckpt["simvq_split_state_dict"].items():
        if torch.is_floating_point(t):
            assert torch.isfinite(t).all(), f"Non-finite FP tensor `{k}`"
    for k, t in ckpt["simvq_split_ternary_qat_state_dict"].items():
        if torch.is_floating_point(t):
            assert torch.isfinite(t).all(), f"Non-finite QAT tensor `{k}`"

    from simvq_split.memory import (
        PackedLUT2BitLinear,
        compute_2bit_lut_footprint,
        pack_model_2bit_lut,
    )
    from simvq_split.models import SimVQSplitConfig, SimVQSplitLM

    cfg = SimVQSplitConfig(**ckpt["simvq_split_config"])
    assert cfg.use_byte_chunker is True
    m_fp = SimVQSplitLM(cfg)
    m_fp.load_state_dict(ckpt["simvq_split_state_dict"], strict=True)
    m_qat = SimVQSplitLM(cfg)
    m_qat.load_state_dict(ckpt["simvq_split_ternary_qat_state_dict"], strict=True)

    checked_linears = 0
    for name, mod in m_qat.named_modules():
        if isinstance(mod, nn.Linear) and name != "lm_head":
            w = mod.weight.detach().float()
            g_size = 32 if w.numel() % 32 == 0 else w.size(-1)
            for row in w.reshape(-1, g_size):
                assert torch.unique(torch.round(row * 1e5)).numel() <= 3, (
                    f"QAT linear `{name}` violates <= 3 ternary levels!"
                )
            checked_linears += 1
    assert checked_linears == 70, f"Expected 70 Linear modules, checked {checked_linears}"

    lut_m = pack_model_2bit_lut(m_qat)
    assert any(isinstance(mod, PackedLUT2BitLinear) for mod in lut_m.modules())
    lut_fp = compute_2bit_lut_footprint(lut_m)
    assert lut_fp["total_lut_kb"] <= 555.0
    assert abs(lut_fp["matrix_compression_vs_fp32"] - 16.0) < 1e-3

    # Live forward sanity on local CUDA/CPU
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m_fp.to(dev).eval()
    m_fp.set_snap_alpha(1.0)
    lut_m.to(dev).eval()
    lut_m.set_snap_alpha(1.0)
    with torch.no_grad():
        xb = torch.randint(3, cfg.vocab_size, (2, 128), device=dev)
        lg_fp = m_fp(xb)
        lg_lut = lut_m(xb)
    assert torch.isfinite(lg_fp).all() and torch.isfinite(lg_lut).all()

    print("=== [Phase B QA 4/4] Validating Phase_B_Byte_Chunker_Report.md & quarantine ===", flush=True)
    assert report_path.exists(), f"Missing {report_path}"
    md_text = report_path.read_text(encoding="utf-8")
    assert len(md_text) > 1500
    assert "16/16" in md_text and "SurpriseHNetChunker" in md_text and "PackedLUT2BitLinear" in md_text
    for m_name in EXPECTED_MODELS:
        assert m_name in md_text, f"Missing `{m_name}` in {report_path.name}"
    n_docs = check_markdown_quarantine(root)
    assert n_docs >= 5

    print(
        f"ALL PHASE B QA CHECKS PASSED: {len(EXPECTED_MODELS)} models verified ({dev}) | "
        f"JSON={json_path.stat().st_size:,} B | CKPT={ckpt_path.stat().st_size:,} B | "
        f"REPORT={report_path.stat().st_size:,} B",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
