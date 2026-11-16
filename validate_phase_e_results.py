"""
Standalone Zero-Manual QA Validator for Phase E (`validate_phase_e_results.py`).

Verifies integrity of:
1. Untouched `simvq_split/baselines.py` vs HEAD, ruff fatal lint, `>= 17` Zero-Cheat laws
   (`tests/test_anti_cheat.py`) and `6/6` Phase C laws (`tests/test_cram_phase_c.py`).
2. `benchmarks/phase_e_fast_utf8_results.json` (Track 1 kernel parity/VRAM/speedup, Track 2
   legacy Phase A/B checkpoint parity, Track 3 `0.00%` mid-UTF8 splits across 6 disjoint
   open-web regimes, `1,268 B` streaming state, `<= 555.0 KB` `2-bit LUT` at `16.00x`).
3. `checkpoints/simvq_split_utf8_openweb.pt` (`strict=True` FP & QAT reload, `70` ternary
   linear layers `<= 3` levels/group, physical `PackedLUT2BitLinear`, live UTF-8 forward).
4. `Phase_E_Fast_UTF8_Report.md` and strict Russian lexical quarantine across all repo docs.
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
from simvq_split.memory import (  # noqa: E402
    PackedLUT2BitLinear,
    compute_2bit_lut_footprint,
    pack_model_2bit_lut,
)
from simvq_split.models import SimVQSplitConfig, SimVQSplitLM  # noqa: E402
from simvq_split.tokenizer import ByteBPETokenizer  # noqa: E402


def main() -> int:
    root = ROOT_DIR
    json_path = root / "benchmarks" / "phase_e_fast_utf8_results.json"
    ckpt_path = root / "checkpoints" / "simvq_split_utf8_openweb.pt"
    report_path = root / "Phase_E_Fast_UTF8_Report.md"

    print("=== [Phase E QA 1/4] Checking baselines diff, ruff, 17+6 pytest laws ===", flush=True)
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

    print("=== [Phase E QA 2/4] Validating benchmarks/phase_e_fast_utf8_results.json ===", flush=True)
    assert json_path.exists(), f"Missing {json_path}"
    data = json.loads(json_path.read_text(encoding="utf-8"))

    t1 = data["track1_speed_vram"]
    k2048 = t1["kernels"]["L2048"]
    assert k2048["chunker"]["span_speedup"] >= 3.0, f"Low chunker speedup: {k2048['chunker']['span_speedup']}"
    assert k2048["woodbury_fp"]["fwd_diff"] < 2e-5
    assert k2048["woodbury_fp"]["bwd_diff"] < 5e-5
    assert k2048["woodbury_fp"]["vram_drop_ckpt32"] >= 2.5
    assert k2048["woodbury_tern"]["fwd_diff"] < 2e-5
    assert k2048["woodbury_tern"]["bwd_diff"] < 5e-5
    assert k2048["woodbury_tern"]["vram_drop_ckpt32"] >= 2.5
    assert k2048["crystal"]["fwd_diff"] < 1e-6
    assert k2048["crystal"]["bwd_diff"] < 1e-5
    assert k2048["crystal"]["vram_drop_ckpt32"] >= 2.5
    e2e2048 = t1["end_to_end"]["L2048"]
    for v_name in ("BPE_FP", "BPE_QAT", "ByteHNet_FP_byte", "ByteHNet_UTF8_staticM"):
        assert not e2e2048[v_name]["phase_e"]["oom"], f"Phase E OOM at L=2048 on {v_name}!"

    t2 = data["track2_legacy_parity"]
    for p_key in ("phase_a_bpe", "phase_b_byte"):
        for var in ("FP", "QAT"):
            assert t2[p_key][var]["max_logit_abs_diff"] < 1e-4, (
                f"Legacy checkpoint logit drift in {p_key}/{var}: {t2[p_key][var]}"
            )
            assert t2[p_key][var]["argmax_agreement_pct"] >= 99.5
    drift_direct = t2["phase_b_byte"]["cognitive_pp_drift_legacy_vs_phase_e"]
    for dk, dv in drift_direct.items():
        assert dv <= 2.0, f"Direct Legacy-vs-PhaseE cognitive drift in {dk}: {dv} pp"
    drift_b = t2["phase_b_byte"]["cognitive_pp_drift_vs_phase_b_json"]
    for dk, dv in drift_b.items():
        assert dv <= 2.5, f"Phase B cognitive drift in {dk}: {dv} pp"

    t3 = data["track3_multilingual_utf8"]
    assert t3["chunker_state_bytes"] == 1268
    assert t3["streaming_bytes_L64_256_2048"] == [1268, 1268, 1268]
    assert t3["streaming_bitwise_match"] is True
    assert 80.0 <= t3["cache_kb_L2048"] <= 83.5
    assert t3["unshielded_phase_b_byte"]["max_mid_utf8_split_pct"] > 5.0
    for st_key in ("zeroshot_utf8", "adapted_fp_utf8", "adapted_qat_utf8", "adapted_lut_utf8"):
        assert t3[st_key]["max_mid_utf8_split_pct"] == 0.0, (
            f"Non-zero mid-UTF8 cuts in {st_key}: {t3[st_key]['max_mid_utf8_split_pct']}"
        )

    lut = t3["adapted_lut_utf8"]
    assert lut["total_lut_kb"] <= 555.0
    assert abs(lut["matrix_compression_vs_fp32"] - 16.0) < 1e-3
    assert lut["max_pp_diff_vs_qat"] <= 2.5
    assert lut["gsm8k_te500"]["bpe_equiv_ppl"] <= 48.0
    assert lut["s5"]["k2"]["exact_pct"] >= 60.0, f"Low UTF8 LUT S5 k2: {lut['s5']['k2']}"
    assert lut["selective_f1_ood_long"] >= 75.0, f"Low UTF8 LUT SelF1 Long: {lut['selective_f1_ood_long']}"
    for r_name, r_val in lut["regimes"].items():
        assert math.isfinite(r_val["byte_bpb"]) and 0.1 < r_val["byte_bpb"] < 12.0, (
            f"Bad byte_bpb for {r_name}: {r_val['byte_bpb']}"
        )
        assert r_val["mid_utf8_split_pct"] == 0.0

    print("=== [Phase E QA 3/4] Validating checkpoints/simvq_split_utf8_openweb.pt ===", flush=True)
    assert ckpt_path.exists() and ckpt_path.stat().st_size > 100_000, f"Invalid {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for req_k in (
        "simvq_split_config",
        "simvq_split_state_dict",
        "simvq_split_ternary_qat_state_dict",
        "tokenizer",
    ):
        assert req_k in ckpt, f"Missing checkpoint key `{req_k}`"

    cfg = SimVQSplitConfig(**ckpt["simvq_split_config"])
    assert cfg.use_byte_chunker is True
    assert cfg.byte_chunk_mode == "utf8"
    assert cfg.use_static_m_cap is True

    m_fp = SimVQSplitLM(cfg)
    m_fp.load_state_dict(ckpt["simvq_split_state_dict"], strict=True)
    m_qat = SimVQSplitLM(cfg)
    m_qat.load_state_dict(ckpt["simvq_split_ternary_qat_state_dict"], strict=True)

    checked_linears = 0
    for name, mod in m_qat.named_modules():
        if isinstance(mod, nn.Linear) and name != "lm_head":
            w = mod.weight.detach().float()
            assert torch.isfinite(w).all()
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

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = ByteBPETokenizer(merges=[])
    sample = "Проверка UTF-8: Фаза E без Triton, 中文測試 🚀🔥🧠, `def f(x): return x`!"
    s_ids = tok.encode(sample, add_bos=True, add_eos=True)
    xb = torch.tensor([s_ids[:-1]], dtype=torch.long, device=dev)
    lut_m.to(dev).eval()
    lut_m.set_snap_alpha(1.0)
    with torch.no_grad():
        logits_lut = lut_m(xb)
        _, _, ctx = lut_m.byte_chunker.encode_to_chunks(lut_m.tok_emb(xb), xb)
    assert torch.isfinite(logits_lut).all()
    raw_b = sample.encode("utf-8")
    b_flags = ctx.b_mask[0, 1 : 1 + len(raw_b)].tolist()
    # First token (BOS=1) starts chunk 0; byte boundaries after BOS must keep UTF-8 intact
    b_all = [True] + b_flags[1:]
    cuts = [i for i, fl in enumerate(b_all) if fl] + [len(raw_b)]
    for cs, ce in zip(cuts[:-1], cuts[1:]):
        raw_b[cs:ce].decode("utf-8", errors="strict")

    print("=== [Phase E QA 4/4] Validating Phase_E_Fast_UTF8_Report.md & quarantine ===", flush=True)
    assert report_path.exists(), f"Missing {report_path}"
    md_text = report_path.read_text(encoding="utf-8")
    assert len(md_text) > 1500
    assert "PackedLUT2BitLinear" in md_text and "1,268 B" in md_text
    n_docs = check_markdown_quarantine(root)
    assert n_docs >= 6

    print(
        f"ALL PHASE E QA CHECKS PASSED ({dev}) | "
        f"JSON={json_path.stat().st_size:,} B | CKPT={ckpt_path.stat().st_size:,} B | "
        f"REPORT={report_path.stat().st_size:,} B",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
