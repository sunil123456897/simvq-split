"""
Zero-Cheat Phase I (`Calcite-6 / UEBR`) Verification Suite (`tests/test_phase_i_uebr_tracer.py`).

Enforces 5 deterministic invariants (`i1..i5`, `< 15 s` runtime):
  i1: AST purity of `experiments/uebr_proto/smoke_uebr_invariants.py` (`0` `re`/`ast`/`inspect`/`scipy`,
      `0` `eval`/`exec`/`compile`, `0` `dict`/`set` inside `nn.Module` classes) and strict `0`-byte
      `git diff HEAD` on all protected core, CRAM, kernel, checkpoint, test, and Phase A–H JSON paths.
  i2: `SPDGraphLaplacianField` strict SPD (`min_eig > 0`), matrix-free `ePC` convergence `< 1e-4`,
      and `float64` `torch.autograd.gradcheck` on `_ImplicitLaplacianSolveFn` (`atol=1e-6`).
  i3: Closed-form `PermFlow` affine-hull projector `C(U)` exactness (`< 1e-6` FP32 / `< 1e-14` FP64)
      and `100%` lossless rounding compliance `f(P*) <= F_S(A) + 1e-7` for `birkhoff_extension_round`
      across `20` random QAP instances.
  i4: `RLNSVariableSlotEBM` (`0-BPTT` block Fenchel-Young) 100% live non-zero gradients across all
      parameters + `adaptive_mincut_blocks` partitioning with absorbing null slot `0`.
  i5: `GAQSlotBridge` zero radial-tangential gradient leakage (`< 1e-6` vs `> 0.05` STE), `100.0%`
      Step-1 active codebook utilization, and full schema/gate compliance of
      `benchmarks/phase_i_uebr_tracer_results.json` (`seed_2026`, `seed_2027`, `invariant_4b_live_tier1_lut`).
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import subprocess
import sys

import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.uebr_proto.smoke_uebr_invariants import (  # noqa: E402
    GAQSlotBridge,
    RLNSVariableSlotEBM,
    SPDGraphLaplacianField,
    _ImplicitLaplacianSolveFn,
    _measure_gaq_radial_tangential_leakage,
    birkhoff_extension_round,
    permflow_tangent_project,
)

TRACER_PATH = ROOT_DIR / "experiments" / "uebr_proto" / "smoke_uebr_invariants.py"
PHASE_I_JSON = ROOT_DIR / "benchmarks" / "phase_i_uebr_tracer_results.json"

BANNED_IMPORTS = {"re", "ast", "inspect", "scipy"}
BANNED_CALLS = {"eval", "exec", "compile"}

PROTECTED_PATHS = (
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


def _assert_finite_nested(obj: object, trail: str = "root") -> None:
    if isinstance(obj, float):
        assert math.isfinite(obj), f"Non-finite float at {trail}: {obj}"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _assert_finite_nested(v, f"{trail}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_finite_nested(v, f"{trail}[{i}]")


def test_i1_ast_zero_cheat_and_protected_zero_diff():
    assert TRACER_PATH.exists(), f"Missing UEBR tracer: {TRACER_PATH}"
    src = TRACER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(TRACER_PATH))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in BANNED_IMPORTS, (
                    f"Banned import `{alias.name}` in {TRACER_PATH.name}"
                )
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in BANNED_IMPORTS, (
                f"Banned from-import `{node.module}` in {TRACER_PATH.name}"
            )
        elif isinstance(node, ast.Call):
            fn_name = None
            if isinstance(node.func, ast.Name):
                fn_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                fn_name = node.func.attr
            if fn_name is not None:
                assert fn_name not in BANNED_CALLS, (
                    f"Banned call `{fn_name}()` in {TRACER_PATH.name}:{node.lineno}"
                )

    for cls_node in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
        is_nn_mod = any(
            (isinstance(b, ast.Attribute) and b.attr == "Module")
            or (isinstance(b, ast.Name) and b.id == "Module")
            for b in cls_node.bases
        )
        if not is_nn_mod:
            continue
        for sub in ast.walk(cls_node):
            assert not isinstance(sub, (ast.Dict, ast.DictComp, ast.Set, ast.SetComp)), (
                f"Forbidden dict/set inside nn.Module `{cls_node.name}` at line {getattr(sub, 'lineno', '?')}"
            )
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id in {"dict", "set"}:
                raise AssertionError(
                    f"Forbidden dict()/set() inside nn.Module `{cls_node.name}` at line {getattr(sub, 'lineno', '?')}"
                )

    for prot in PROTECTED_PATHS:
        chk = subprocess.run(
            ["git", "diff", "--exit-code", "HEAD", "--", prot],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
        )
        assert chk.returncode == 0, f"Protected artifact `{prot}` modified:\n{chk.stdout}\n{chk.stderr}"


def test_i2_spd_laplacian_cholesky_adjoint_gradcheck():
    torch.manual_seed(2026)
    mod = SPDGraphLaplacianField(d_slot=64, d_z=8, d_k=64).eval()
    q_hat = torch.randn(3, 12, 64, dtype=torch.float32)
    H_psi, s_psi, _ = mod.build_system(q_hat)
    min_eig = float(torch.linalg.eigvalsh(H_psi).min().item())
    assert min_eig >= math.exp(-1.5) - 1e-6, f"Laplacian H_psi not strictly SPD: {min_eig}"

    z_chol = _ImplicitLaplacianSolveFn.apply(H_psi, s_psi)
    z_epc = mod.relax_epc_no_grad(H_psi, s_psi, steps=40)
    epc_err = float((z_chol - z_epc).abs().max().item())
    assert epc_err < 1e-4, f"ePC failed to converge to Cholesky: max_err={epc_err}"

    # Exact float64 autograd.gradcheck for 0-BPTT Cholesky Implicit Adjoint
    A = torch.randn(2, 5, 5, dtype=torch.float64)
    H64 = (torch.bmm(A, A.transpose(-1, -2)) + 1.5 * torch.eye(5, dtype=torch.float64)).requires_grad_(True)
    s64 = torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(_ImplicitLaplacianSolveFn.apply, (H64, s64), eps=1e-6, atol=1e-6)


def test_i3_permflow_projector_and_birkhoff_extension_lossless_rounding():
    torch.manual_seed(2026)
    U32 = torch.randn(16, 10, 10, dtype=torch.float32)
    C32 = permflow_tangent_project(U32)
    assert float(C32.sum(dim=-1).abs().max().item()) < 1e-6
    assert float(C32.sum(dim=-2).abs().max().item()) < 1e-6

    U64 = torch.randn(8, 12, 12, dtype=torch.float64)
    C64 = permflow_tangent_project(U64)
    assert float(C64.sum(dim=-1).abs().max().item()) < 1e-14
    assert float(C64.sum(dim=-2).abs().max().item()) < 1e-14

    n = 8
    for i in range(20):
        W = torch.rand(n, n, dtype=torch.float64) + 0.05
        # Sinkhorn doubly-stochastic interior point
        A = W.clone()
        for _ in range(15):
            A = A / A.sum(dim=-1, keepdim=True)
            A = A / A.sum(dim=-2, keepdim=True)
        F_m = torch.randn(n, n, dtype=torch.float64)
        D_m = torch.randn(n, n, dtype=torch.float64)
        F_m = 0.5 * (F_m + F_m.t())
        D_m = 0.5 * (D_m + D_m.t())
        S_m = -2.0 * (F_m @ A @ D_m)

        def _ev(P: torch.Tensor) -> float:
            return float((P * (F_m @ P @ D_m)).sum().item())

        P_best, f_best, F_ext, supp_sz = birkhoff_extension_round(A, _ev, S=S_m)
        assert supp_sz >= 1
        assert torch.allclose(P_best.sum(dim=-1), torch.ones(n, dtype=torch.float64))
        assert torch.allclose(P_best.sum(dim=-2), torch.ones(n, dtype=torch.float64))
        assert float(((P_best == 0.0) | (P_best == 1.0)).float().min().item()) == 1.0
        assert f_best <= F_ext + 1e-7, f"Lossless rounding violated on instance {i}: {f_best} > {F_ext}"


def test_i4_rlns_block_fenchel_young_and_absorbing_null_slot():
    torch.manual_seed(2026)
    ebm = RLNSVariableSlotEBM(d_feat=24, n_slots=10, n_states=5)
    x_slots = torch.randn(8, 10, 24)
    y_true = torch.randint(0, 5, (8, 10))
    with torch.no_grad():
        _, a_sym, _ = ebm.compute_potentials(x_slots)
        blocks = ebm.adaptive_mincut_blocks(a_sym, k_blk=4)
    assert len(blocks) >= 2 and all(b.shape == (8, 4) for b in blocks)

    loss = ebm.block_fenchel_young_loss(x_slots, y_true, blocks[0], gamma=0.35)
    assert torch.isfinite(loss)
    loss.backward()
    n_tot = 0
    n_live = 0
    for p in ebm.parameters():
        n_tot += 1
        assert p.grad is not None and torch.isfinite(p.grad).all()
        if float(p.grad.abs().sum().item()) > 0.0:
            n_live += 1
    assert n_live == n_tot, f"Expected 100% live gradients on RLNSVariableSlotEBM, got {n_live}/{n_tot}"

    y_inf = ebm.infer_block_gibbs(x_slots, k_blk=4, sweeps=2)
    assert y_inf.shape == (8, 10) and int(y_inf.min().item()) >= 0 and int(y_inf.max().item()) <= 4


def test_i5_gaq_zero_leakage_and_phase_i_json():
    torch.manual_seed(2026)
    bridge = GAQSlotBridge(d_in=16, d_code=16, num_codes=64)
    x_probe = torch.randn(32, 16)
    gaq_leak, ste_leak = _measure_gaq_radial_tangential_leakage(bridge, x_probe)
    assert gaq_leak < 1e-6, f"Householder leakage too high: {gaq_leak}"
    assert ste_leak > 0.05, f"Expected STE leakage > 0.05, got {ste_leak}"

    with torch.no_grad():
        c0 = bridge.codebook().clone()
    opt = torch.optim.SGD([bridge.E_vq], lr=0.1)
    _, vql, _, _ = bridge(x_probe)
    opt.zero_grad()
    vql.backward()
    opt.step()
    with torch.no_grad():
        c1 = bridge.codebook()
        active_pct = 100.0 * float(((c1 - c0).abs().max(dim=-1).values > 1e-7).float().mean().item())
    assert active_pct == 100.0, f"Expected 100% Step-1 codebook update, got {active_pct}%"

    assert PHASE_I_JSON.exists(), f"Missing Phase I tracer JSON: {PHASE_I_JSON}"
    rep = json.loads(PHASE_I_JSON.read_text(encoding="utf-8"))
    _assert_finite_nested(rep)
    assert rep["all_passed"] is True
    assert float(rep["protocol"]["elapsed_sec"]) < 60.0

    for sd_key in ("seed_2026", "seed_2027"):
        s_res = rep[sd_key]
        i1 = s_res["invariant_1"]
        assert i1["passed"] is True
        assert float(i1["adjoint_grad_err"]) < 1e-5
        assert float(i1["epc_init_max_err"]) < 1e-4
        assert float(i1["uebr_fp32"]["acc_pct"]) >= 90.0
        assert float(i1["uebr_qat_158"]["acc_pct"]) >= 90.0
        assert float(i1["uebr_qat_158"]["acc_pct"]) >= float(i1["static_wzz_acc_pct"]) + 25.0

        i2 = s_res["invariant_2"]
        assert i2["passed"] is True
        assert float(i2["part2a_permflow_mode_cov_pct"]) >= 87.5
        assert float(i2["part2a_sinkhorn_mode_cov_pct"]) <= 25.0
        assert float(i2["part2b_ba_err_f64"]) < 1e-13
        assert float(i2["part2b_lossless_rounding_pct"]) == 100.0
        assert int(i2["part2b_be_wins"]) >= 40

        i3 = s_res["invariant_3"]
        assert i3["passed"] is True
        assert float(i3["rlns_block4_exact_pct"]) >= 85.0
        assert float(i3["rlns_null_slot_acc_pct"]) == 100.0
        assert float(i3["rlns_block4_exact_pct"]) >= float(i3["besag_1flip_exact_pct"]) + 35.0

        i4a = s_res["invariant_4a"]
        assert i4a["passed"] is True
        assert float(i4a["gaq_radial_tangential_leak"]) < 1e-6
        assert float(i4a["gaq_step1_active_codes_pct"]) == 100.0
        assert float(i4a["open_bypass"]["s2_contrib_pp"]) < 5.0
        assert float(i4a["blocked_gaq"]["hard_acc_pct"]) >= 90.0
        assert float(i4a["blocked_gaq"]["s2_contrib_pp"]) >= 35.0
        assert float(i4a["blocked_gaq"]["easy_false_trigger_pct"]) == 0.0

    i4b = rep["invariant_4b_live_tier1_lut"]
    assert i4b["passed"] is True
    assert int(i4b["lut_layers"]) == 95
    assert abs(float(i4b["cache_kb_L2048"]) - 230.11) <= 0.1
    assert int(i4b["chunker_state_bytes"]) == 2164
    assert float(i4b["auroc_woodbury_trap"]) >= 0.90
    assert float(i4b["auroc_combined_risk"]) >= 0.90
