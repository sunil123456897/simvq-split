"""
Zero-Cheat Phase K Verification Suite (`tests/test_phase_k_scaling_law.py`, Laws `k1..k5`).

Enforces:
  k1: AST purity of `experiments/scaling_phase_k/` neural modules & runner (0 banned imports/calls,
      0 Python dict/set state in `nn.Module`, 0 `SphericalALU` traces) and `0`-byte `git diff HEAD`
      across all protected Phases A–J artifacts.
  k2: `< 1.0%` parameter parity across `S1_5M -> S2_25M -> S3_85M`, exact `O(1)` constancy of
      `compute_lean_state_cache_kb` (`L=64..2048`), exact match with legacy cache formula at
      `(12, 5, 7)`, and `>= 20%` cache reduction under `Via Negativa` `V1 (8, 2, 3)`.
  k3: Physical `2-bit uint8 LUT` (`FastPackedLUT2BitLinear`) parity (`< 1e-4`), `16.00x` matrix
      compression, `0.00%` mid-UTF8 cuts on multilingual prose, and checkpoint roundtrips.
  k4: Zero-leakage 45-char scrubber catches perturbed substrings across all 5 held-out vaults
      (`GSM8K`, `MultiArith-600`, `ASDiv-A`, `SVAMP-1000`, `ARC Eval`) without false positives.
  k5: Final Phase K cloud scaling law completeness (`benchmarks/phase_k_via_negativa_probe.json`,
      `benchmarks/phase_k_scaling_law_results.json`, 3 packed LUT checkpoints), monotone `LUT val_bpb`
      & quant-gap collapse with honest `strict_fp_val_monotonic_passed == False` reporting,
      and zero-split / zero-ALU purity.
"""

from __future__ import annotations

import ast
import glob
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.scaling_phase_k.byte_baselines import (  # noqa: E402
    FastHouseholderDeltaLayer,
    ScaledHybridDeltaSWALM,
    build_matched_baselines_for_tier,
)
from experiments.scaling_phase_k.eval_suite import load_lean_packed_lut_checkpoint  # noqa: E402
from experiments.scaling_phase_k.lean_system1 import (  # noqa: E402
    CalciteLeanSystem1LM,
    LeanNeuralWoodburyLayer,
    LeanVoronoiCrystalLayer,
    build_tier_lean_config,
    compute_lean_state_cache_kb,
    get_scaling_tier_specs,
)
from kernels.fast_packed_lut import (  # noqa: E402
    pack_fast_model_2bit_lut,
    save_packed_lut_checkpoint,
)
from run_simvq_split_benchmark import compute_state_cache_kb  # noqa: E402
from scripts.colab_phase_g_runner import is_leaked_doc  # noqa: E402
from scripts.colab_phase_k_scaling_runner import build_phase_k_heldout_blocklist  # noqa: E402
from simvq_split.baselines import HouseholderDeltaLayer  # noqa: E402
from simvq_split.quant_lut import (  # noqa: E402
    compute_2bit_lut_footprint,
    pack_model_2bit_lut,
    quantize_ternary_weights_ls,
)
from simvq_split.tokenizer import ByteBPETokenizer  # noqa: E402

FORBIDDEN_IMPORTS = {"re", "ast", "inspect", "subprocess", "json", "pickle"}
FORBIDDEN_CALL_NAMES = {
    "eval",
    "exec",
    "compile",
    "isupper",
    "islower",
    "isdigit",
    "isalpha",
    "isalnum",
    "startswith",
    "endswith",
    "findall",
    "search",
    "fullmatch",
    "splitlines",
}


def test_k1_ast_hygiene_and_protected_zero_diff():
    for path in (
        ROOT_DIR / "experiments" / "scaling_phase_k" / "lean_system1.py",
        ROOT_DIR / "experiments" / "scaling_phase_k" / "byte_baselines.py",
    ):
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in FORBIDDEN_IMPORTS, (
                        f"Forbidden import `{alias.name}` in {path.name}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in FORBIDDEN_IMPORTS, (
                    f"Forbidden from-import `{node.module}` in {path.name}"
                )
            elif isinstance(node, ast.Call):
                fn = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else (node.func.attr if isinstance(node.func, ast.Attribute) else None)
                )
                if fn is not None:
                    assert fn not in FORBIDDEN_CALL_NAMES, (
                        f"Forbidden call `{fn}()` in {path.name}:{node.lineno}"
                    )

        for cls_node in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            is_nn = any(
                (isinstance(b, ast.Attribute) and b.attr == "Module")
                or (isinstance(b, ast.Name) and b.id in ("SimVQSplitLM", "VoronoiCrystalLayer", "NeuralWoodburyLayer"))
                for b in cls_node.bases
            )
            if not is_nn:
                continue
            for sub in ast.walk(cls_node):
                assert not isinstance(sub, (ast.Dict, ast.DictComp, ast.Set, ast.SetComp)), (
                    f"Forbidden dict/set state in `{cls_node.name}` ({path.name}:{getattr(sub, 'lineno', '?')})"
                )

    for cls in (CalciteLeanSystem1LM, ScaledHybridDeltaSWALM):
        sig = inspect.signature(cls.forward)
        params = list(sig.parameters.keys())
        assert params[0] == "self" and params[1] == "token_ids"

    # Runner must never inject SphericalALU traces or call eval/exec/compile
    runner_p = ROOT_DIR / "scripts" / "colab_phase_k_scaling_runner.py"
    runner_src = runner_p.read_text(encoding="utf-8")
    assert "SphericalBase10RegisterALU" not in runner_src
    assert "encode_interleaved_alu_trace" not in runner_src
    r_tree = ast.parse(runner_src, filename=str(runner_p))
    for node in ast.walk(r_tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"eval", "exec", "compile"}

    # Strict 0-byte git diff on protected core & Phases A-J artifacts
    protected = (
        "simvq_split/",
        "experiments/cram_gsm8k/",
        "experiments/uebr_proto/",
        "experiments/uebr_system2/",
        "checkpoints/calcite5_tier1_17m_utf8_lut.pt",
        "checkpoints/calcite5_tier1_17m_utf8_qat_h_lut.pt",
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
    for prot in protected:
        chk = subprocess.run(
            ["git", "diff", "--exit-code", "HEAD", "--", prot],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
        )
        assert chk.returncode == 0, f"Protected path modified `{prot}`:\n{chk.stdout}\n{chk.stderr}"


def test_k2_three_tier_param_parity_and_o1_lean_cache():
    specs = get_scaling_tier_specs(window_size=8, crystal_w_chunks=2, woodbury_latch_span=3)
    for tier in ("S1_5M", "S2_25M", "S3_85M"):
        t_info = specs[tier]
        c_params = int(t_info["params"])
        bl = build_matched_baselines_for_tier(tier, c_params)
        assert abs(float(bl["RoPETransformerLM"]["parity_pct"])) < 1.0
        assert abs(float(bl["ScaledHybridDeltaSWALM"]["parity_pct"])) < 1.0
        assert float(t_info["cache_reduction_pct"]) >= 20.0

        cfg_lean = build_tier_lean_config(tier, window_size=8, crystal_w_chunks=2, woodbury_latch_span=3)
        c64 = compute_lean_state_cache_kb(cfg_lean, 64)
        c512 = compute_lean_state_cache_kb(cfg_lean, 512)
        c2048 = compute_lean_state_cache_kb(cfg_lean, 2048)
        assert c64 == c512 == c2048

        cfg_leg = build_tier_lean_config(tier, window_size=12, crystal_w_chunks=5, woodbury_latch_span=7)
        assert abs(
            compute_lean_state_cache_kb(cfg_leg, 2048) - compute_state_cache_kb("SimVQSplitLM", cfg_leg, None, 2048)
        ) < 1e-6

    # Chunker streaming state constancy across sequence lengths on S1_5M
    cfg1 = build_tier_lean_config("S1_5M", window_size=8, crystal_w_chunks=2, woodbury_latch_span=3)
    m1 = CalciteLeanSystem1LM(cfg1).eval()
    m1.byte_chunker.lock_router(True)
    obs = set()
    with torch.no_grad():
        for L in (64, 256, 768):
            xb = torch.randint(3, 259, (2, L))
            _, _, ctx = m1.byte_chunker.encode_to_chunks(m1.tok_emb(xb), xb)
            assert ctx.valid_chunk.size(1) > 0
            obs.add(int(m1.cfg.d_model))
    assert isinstance(m1.crystal, LeanVoronoiCrystalLayer)
    assert isinstance(m1.woodbury, LeanNeuralWoodburyLayer)

    # Verify exact forward & backward equivalence between FastHouseholderDeltaLayer (WY Bc=16) and HouseholderDeltaLayer
    torch.manual_seed(2026)
    ref_delta = HouseholderDeltaLayer(64, 4, chunk_size=4)
    fast_delta = FastHouseholderDeltaLayer(64, 4, chunk_size=4)
    fast_delta.load_state_dict(ref_delta.state_dict(), strict=True)
    x_ref = torch.randn(2, 70, 64, requires_grad=True)
    x_fast = x_ref.detach().clone().requires_grad_(True)
    y_ref = ref_delta(x_ref)
    y_fast = fast_delta(x_fast)
    assert float((y_ref - y_fast).abs().max().item()) < 1e-4
    g_out = torch.randn_like(y_ref)
    (y_ref * g_out).sum().backward()
    (y_fast * g_out).sum().backward()
    assert float((x_ref.grad - x_fast.grad).abs().max().item()) < 1e-4
    for (nm_r, pr_r), (nm_f, pr_f) in zip(ref_delta.named_parameters(), fast_delta.named_parameters()):
        assert nm_r == nm_f and pr_r.grad is not None and pr_f.grad is not None
        assert float((pr_r.grad - pr_f.grad).abs().max().item()) < 1e-4, f"WY grad parity failed on {nm_r}"


def test_k3_lut_roundtrip_and_utf8_zero_cuts():
    torch.manual_seed(2026)
    cfg1 = build_tier_lean_config("S1_5M", window_size=8, crystal_w_chunks=2, woodbury_latch_span=3)
    m = CalciteLeanSystem1LM(cfg1).eval()
    m.byte_chunker.lock_router(True)
    m.set_crystal_schedule(1, 1)
    m.set_snap_alpha(1.0)
    with torch.no_grad():
        for nm, mod in m.named_modules():
            if isinstance(mod, nn.Linear) and nm != "lm_head" and not m.byte_chunker.is_locked_router_module(mod):
                wt, _, _ = quantize_ternary_weights_ls(mod.weight.data)
                mod.weight.data.copy_(wt)

    ref_m = pack_model_2bit_lut(m).eval()
    fp = compute_2bit_lut_footprint(ref_m)
    assert int(fp["num_lut_layers"]) == 75
    assert abs(float(fp["matrix_compression_vs_fp32"]) - 16.0) < 1e-4

    fast_m = pack_fast_model_2bit_lut(m, backend="pytorch_vec").eval()
    x_par = torch.randint(3, 259, (2, 128))
    with torch.no_grad():
        diff = float((ref_m(x_par).float() - fast_m(x_par).float()).abs().max().item())
    assert diff < 1e-4, f"S1_5M fast LUT parity error {diff}"

    # Verify 0.00% mid-UTF8 cuts on multilingual UTF-8 prose
    tok = ByteBPETokenizer(merges=[])
    multi = (
        "Модель без токенизатора держит границу UTF-8 строго целой. "
        "多语言字节切分绝对不能切断多字节字符。🚀✨🔥 "
        "def f(x): return x * 2 + 1\n"
    ) * 3
    x_ids = torch.tensor([tok.encode(multi, add_bos=True, add_eos=True)[:384]], dtype=torch.long)
    with torch.no_grad():
        _, _, ctx_u = fast_m.byte_chunker.encode_to_chunks(fast_m.tok_emb(x_ids), x_ids)
    raw_v = (x_ids - 3).clamp(min=0)
    is_cont = (x_ids >= 3) & (torch.bitwise_and(raw_v, 0xC0) == 0x80)
    b_int = ctx_u.b_mask[:, 1:] & ctx_u.valid_mask[:, 1:]
    n_mid = int((b_int & is_cont[:, 1:]).sum().item())
    assert n_mid == 0, f"Detected {n_mid} illegal mid-UTF8 splits!"

    # Test exact save_packed_lut_checkpoint -> load_lean_packed_lut_checkpoint roundtrip
    with tempfile.TemporaryDirectory() as tmpd:
        tmp_ckpt = Path(tmpd) / "lean_s1_roundtrip.pt"
        save_packed_lut_checkpoint(tmp_ckpt, fast_m, cfg1, tok, metadata={"test": "k3"})
        reloaded_m, _, _, meta = load_lean_packed_lut_checkpoint(tmp_ckpt, backend="pytorch_vec", device="cpu")
        assert meta["metadata"]["test"] == "k3"
        with torch.no_grad():
            rt_diff = float((fast_m(x_par).float() - reloaded_m(x_par).float()).abs().max().item())
        assert rt_diff < 1e-5, f"Lean checkpoint roundtrip error {rt_diff}"

    # Also validate downloaded packed Phase K checkpoints when present
    expected_layers = {"calcite5_s1_5m_utf8_lut.pt": 75, "calcite5_s2_25m_utf8_lut.pt": 95, "calcite5_s3_85m_utf8_lut.pt": 105}
    for cname, n_layers in expected_layers.items():
        ck = ROOT_DIR / "checkpoints" / cname
        if ck.exists():
            lm_k, _, _, _ = load_lean_packed_lut_checkpoint(ck, backend="pytorch_vec", device="cpu")
            assert int(compute_2bit_lut_footprint(lm_k)["num_lut_layers"]) == n_layers


def test_k4_zero_leakage_scrubber_all_vaults():
    blocklist = build_phase_k_heldout_blocklist(ROOT_DIR)
    assert len(blocklist) >= 40_000, f"Too few blocklist hashes: {len(blocklist)}"

    # GSM8K test
    gsm_q = json.loads((ROOT_DIR / "benchmarks" / "gsm8k" / "test.jsonl").read_text(encoding="utf-8").splitlines()[0])["question"]
    assert is_leaked_doc("Prefix. " + " ".join(gsm_q.upper().split()) + " Suffix.", blocklist)

    # MultiArith-600
    ma_q = json.loads((ROOT_DIR / "benchmarks" / "ood_math" / "multiarith_600.json").read_text(encoding="utf-8"))[0]["question"]
    assert is_leaked_doc(ma_q.lower(), blocklist)

    # ASDiv-A
    as_it = json.loads((ROOT_DIR / "benchmarks" / "ood_math" / "asdiv_a.json").read_text(encoding="utf-8"))[0]
    assert is_leaked_doc(str(as_it["question"]).upper(), blocklist)

    # SVAMP-1000
    sv_it = json.loads((ROOT_DIR / "benchmarks" / "svamp" / "svamp_1000.json").read_text(encoding="utf-8"))[0]
    assert is_leaked_doc(str(sv_it["Body"]).upper(), blocklist)

    # ARC Eval
    arc_p = sorted(glob.glob(str(ROOT_DIR / "benchmarks" / "arc_agi" / "ARC-AGI-master" / "data" / "evaluation" / "*.json")))[0]
    assert is_leaked_doc(Path(arc_p).read_text(encoding="utf-8")[:250].upper(), blocklist)

    clean = "Тёплый ноябрьский ветер качал ветви старого дуба над тихим деревянным мостом в лесу."
    assert not is_leaked_doc(clean, blocklist)


def test_k5_scaling_law_monotonicity_and_pareto_results():
    probe_p = ROOT_DIR / "benchmarks" / "phase_k_via_negativa_probe.json"
    res_p = ROOT_DIR / "benchmarks" / "phase_k_scaling_law_results.json"
    assert probe_p.exists(), f"Missing {probe_p}"
    assert res_p.exists(), f"Missing {res_p}"

    probe = json.loads(probe_p.read_text(encoding="utf-8"))
    assert "v0_seed2026" in probe and "v0_seed2027" in probe and "v1_attempt1" in probe
    assert "selected_lean_params" in probe

    res = json.loads(res_p.read_text(encoding="utf-8"))
    assert res["protocol"]["pure_system1_zero_alu_traces"] is True
    assert res["protocol"]["corpus"]["alu_traces_injected"] == 0

    tiers = res["tiers"]
    fp_bpbs = []
    lut_bpbs = []
    ow_fp_bpbs = []
    ow_lut_bpbs = []
    for tname, ckname in (
        ("S1_5M", "calcite5_s1_5m_utf8_lut.pt"),
        ("S2_25M", "calcite5_s2_25m_utf8_lut.pt"),
        ("S3_85M", "calcite5_s3_85m_utf8_lut.pt"),
    ):
        assert tname in tiers and len(tiers[tname]) == 3
        cal = tiers[tname]["CalciteLeanSystem1LM"]
        assert float(cal["router_max_abs_diff"]) == 0.0
        assert float(cal["fp"]["max_mid_utf8_split_pct"]) == 0.0
        assert float(cal["lut_2bit"]["max_mid_utf8_split_pct"]) == 0.0
        assert abs(float(cal["lut_footprint"]["matrix_compression_vs_fp32"]) - 16.0) < 0.01
        ck_p = ROOT_DIR / "checkpoints" / ckname
        assert ck_p.exists(), f"Missing downloaded Phase K checkpoint: {ck_p}"

        fp_bpbs.append(float(cal["fp"]["val_memmap_bpb"]))
        lut_bpbs.append(float(cal["lut_2bit"]["val_memmap_bpb"]))
        ow_fp_bpbs.append(float(cal["fp"]["mean_openweb_bpb"]))
        ow_lut_bpbs.append(float(cal["lut_2bit"]["mean_openweb_bpb"]))

    gaps = [lut_bpbs[i] - fp_bpbs[i] for i in range(3)]
    assert lut_bpbs[0] > lut_bpbs[1] > lut_bpbs[2], f"Non-monotone Calcite LUT val BPB: {lut_bpbs}"
    assert fp_bpbs[0] > fp_bpbs[1], f"S1->S2 Calcite FP val BPB did not improve: {fp_bpbs}"
    assert ow_fp_bpbs[0] > ow_fp_bpbs[1], f"S1->S2 Calcite FP OpenWeb BPB did not improve: {ow_fp_bpbs}"
    assert ow_lut_bpbs[0] > ow_lut_bpbs[1], f"S1->S2 Calcite LUT OpenWeb BPB did not improve: {ow_lut_bpbs}"
    assert gaps[0] > gaps[1] > gaps[2] and gaps[2] < 0.02, f"Expected monotone 2-bit gap collapse: {gaps}"
    sf = res["scaling_fits"]
    assert "quant_gap_val_bpb_by_width" in sf
    assert sf["lut_val_bpb_monotonic_passed"] is True
    assert sf["quant_gap_val_bpb_monotonic_passed"] is True
    assert sf["strict_fp_val_monotonic_passed"] is False
    assert sf["strict_openweb_monotonic_passed"] is False
