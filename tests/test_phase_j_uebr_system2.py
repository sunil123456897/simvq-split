"""
Zero-Cheat Phase J (`Calcite-6 / UEBR`) Verification Suite (`tests/test_phase_j_uebr_system2.py`).

Enforces 5 deterministic laws (`j1..j5`, `< 15 s` runtime):
  j1: AST purity across `experiments/uebr_system2/*.py` (`0` `re`/`ast`/`inspect`/`scipy`,
      `0` bare `eval`/`exec`/`compile`, `0` `dict`/`set` inside `nn.Module`), `0` code reads of
      `test.jsonl` across `experiments/uebr_system2/*.py` & `run_phase_j_uebr_benchmark.py`, and
      `0`-byte `git diff HEAD` on all protected core, CRAM, UEBR-proto, kernel, checkpoint,
      test, and Phase A–I benchmark JSON paths.
  j2: 3D matrix-RHS `float64` `autograd.gradcheck` on `_ImplicitLaplacianSolveFn`, strict SPD
      under `torch.amp.autocast(..., dtype=torch.bfloat16)`, and `ePC` convergence `< 1e-4`.
  j3: `100%` lossless `Birkhoff Extension` rounding + `PermFlow` QAP across `20` random QAPs,
      and `SingleShotMathVault` leak-free `.read_questions_only()` + strict `max_calls=1` lock.
  j4: `100%` Step-1 `SectionedGAQBridge` codebook utilization, exact zeroing of positional cols
      `[14:17]` in `_compute_null_slot_features`, `100%` live non-zero gradients across `68` EBM
      parameters, and `(N_CERT_FEATS + 1 == 13)` `cert_weights` `state_dict` checkpoint roundtrip.
  j5: Full finiteness & protocol verification of `benchmarks/phase_j_uebr_system2_results.json`
      (`Stop-Crane PASS` on `GSM8K`, `MultiArith-600` OOD win, honest `ASDiv-A` `Oracle`/`ge4` vs
      `le3` `all_passed=False`), plus `phase_j_attempt1_svamp_postmortem.json` and
      `phase_j_attempt2_posleak_postmortem.json`.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.cram_gsm8k.cram_open_graph import D_STEP_PSI, N_CERT_FEATS  # noqa: E402
from experiments.cram_gsm8k.slot_extractor import TrainInducedQuantityExtractor  # noqa: E402
from experiments.cram_gsm8k.spherical_alu import SphericalBase10RegisterALU  # noqa: E402
from experiments.uebr_system2.tier1_gaq_extractor import SingleShotMathVault  # noqa: E402
from experiments.uebr_system2.uebr_graph_solver import (  # noqa: E402
    D_NULL_INV,
    SectionedGAQBridge,
    SPDGraphLaplacianField,
    UEBROpenGraphSolver,
    _ImplicitLaplacianSolveFn,
    _compute_null_slot_features,
    birkhoff_extension_round_np,
    epc_equilibrium_solve,
    solve_qap_permflow_birkhoff,
    verify_matrix_laplacian_gradcheck,
)

UEBR_SYS2_DIR = ROOT_DIR / "experiments" / "uebr_system2"
RUNNER_PATH = ROOT_DIR / "run_phase_j_uebr_benchmark.py"
PHASE_J_JSON = ROOT_DIR / "benchmarks" / "phase_j_uebr_system2_results.json"
PM1_JSON = ROOT_DIR / "benchmarks" / "phase_j_attempt1_svamp_postmortem.json"
PM2_JSON = ROOT_DIR / "benchmarks" / "phase_j_attempt2_posleak_postmortem.json"
PHASE_J_CKPT = UEBR_SYS2_DIR / "checkpoints" / "phase_j_uebr_a3_2026.pt"

BANNED_IMPORTS = {"re", "ast", "inspect", "scipy"}
BANNED_CALLS = {"eval", "exec", "compile"}

PROTECTED_PATHS = (
    "simvq_split/",
    "experiments/cram_gsm8k/",
    "experiments/uebr_proto/",
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
    "benchmarks/simvq_split_results.json",
    "benchmarks/phase_b_byte_chunker_results.json",
    "benchmarks/phase_c_cram_gsm8k_results.json",
    "benchmarks/phase_e_fast_utf8_results.json",
    "benchmarks/phase_f_utf8_iterative_cram_results.json",
    "benchmarks/phase_g_colab_scaleup_results.json",
    "benchmarks/phase_h_tier1_qat_closure_results.json",
    "benchmarks/phase_i_uebr_tracer_results.json",
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


def _docstring_node_ids(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                ids.add(id(first.value))
    return ids


def test_j1_ast_purity_zero_test_reads_and_protected_diffs():
    py_files = sorted(UEBR_SYS2_DIR.glob("*.py"))
    assert len(py_files) == 3, f"Expected 3 modules in {UEBR_SYS2_DIR}, got {[p.name for p in py_files]}"

    for py_path in py_files:
        src = py_path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(py_path))
        doc_ids = _docstring_node_ids(tree)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in BANNED_IMPORTS, (
                        f"Banned import `{alias.name}` in {py_path.name}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in BANNED_IMPORTS, (
                    f"Banned from-import `{node.module}` in {py_path.name}"
                )
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in BANNED_CALLS, (
                    f"Banned call `{node.func.id}()` in {py_path.name}:{node.lineno}"
                )
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in doc_ids:
                assert "test.jsonl" not in node.value.lower(), (
                    f"Forbidden `test.jsonl` string literal in {py_path.name}:{node.lineno}"
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
                    f"Forbidden dict/set inside nn.Module `{cls_node.name}` ({py_path.name}:{getattr(sub, 'lineno', '?')})"
                )
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id in {"dict", "set"}:
                    raise AssertionError(
                        f"Forbidden dict()/set() inside nn.Module `{cls_node.name}` ({py_path.name}:{getattr(sub, 'lineno', '?')})"
                    )

    assert RUNNER_PATH.is_file(), f"Missing runner: {RUNNER_PATH}"
    r_tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"), filename=str(RUNNER_PATH))
    r_doc_ids = _docstring_node_ids(r_tree)
    for node in ast.walk(r_tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in r_doc_ids:
            assert "test.jsonl" not in node.value.lower(), (
                f"Forbidden `test.jsonl` literal in {RUNNER_PATH.name}:{node.lineno}"
            )

    for prot in PROTECTED_PATHS:
        chk = subprocess.run(
            ["git", "diff", "--exit-code", "HEAD", "--", prot],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
        )
        assert chk.returncode == 0, f"Protected artifact `{prot}` modified:\n{chk.stdout}\n{chk.stderr}"


def test_j2_matrix_cholesky_adjoint_amp_and_epc():
    grad_err = verify_matrix_laplacian_gradcheck(seed=2026)
    assert grad_err < 1e-7, f"Matrix Laplacian adjoint FD error too high: {grad_err}"

    torch.manual_seed(2026)
    A = torch.randn(2, 5, 5, dtype=torch.float64)
    H64 = (torch.bmm(A, A.transpose(-1, -2)) + 1.4 * torch.eye(5, dtype=torch.float64)).requires_grad_(True)
    S64 = torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(_ImplicitLaplacianSolveFn.apply, (H64, S64), eps=1e-6, atol=1e-6)

    lap_mod = SPDGraphLaplacianField(d_slot=64, d_lap=16, d_psi_step=D_STEP_PSI).eval()
    q_slots_pad = torch.randn(3, 8, 64, dtype=torch.float32)
    slot_mask = torch.ones(3, 8, dtype=torch.bool)
    q_prob = torch.randn(3, 64, dtype=torch.float32)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        H_amp = lap_mod.build_laplacian(q_slots_pad, slot_mask, q_prob)
        S_rhs = torch.randn(3, 16, 4, dtype=torch.float32)
        Z_chol = _ImplicitLaplacianSolveFn.apply(H_amp.float(), S_rhs)

    min_eig = float(torch.linalg.eigvalsh(H_amp.float()).min().item())
    assert min_eig >= math.exp(-1.2) - 1e-5, f"SPD bound violated under AMP: min_eig={min_eig}"
    Z_epc = epc_equilibrium_solve(H_amp.float(), S_rhs, steps=45)
    epc_err = float((Z_chol - Z_epc).abs().max().item())
    assert epc_err < 1e-4, f"ePC failed to converge to Cholesky solve: max_err={epc_err}"


def test_j3_birkhoff_qap_lossless_and_single_shot_vault(tmp_path: Path):
    n = 7
    for i in range(20):
        rng = np.random.default_rng(2026 + i)
        A = rng.uniform(0.05, 1.0, size=(n, n))
        for _ in range(15):
            A /= A.sum(axis=1, keepdims=True)
            A /= A.sum(axis=0, keepdims=True)
        F_m = rng.normal(size=(n, n))
        D_m = rng.normal(size=(n, n))
        F_m = 0.5 * (F_m + F_m.T)
        D_m = 0.5 * (D_m + D_m.T)
        S_m = -2.0 * (F_m @ A @ D_m)

        P_best, best_v, ext_v, top_ps = birkhoff_extension_round_np(
            A, eval_fn=lambda P_m: float(np.sum(P_m * (F_m @ P_m @ D_m))), S_mat=S_m
        )
        assert len(top_ps) >= 1
        assert np.allclose(P_best.sum(axis=1), np.ones(n), atol=1e-9)
        assert np.allclose(P_best.sum(axis=0), np.ones(n), atol=1e-9)
        assert np.all((P_best == 0.0) | (P_best == 1.0))
        assert best_v <= ext_v + 1e-7, f"Birkhoff Extension lossless violated on i={i}: {best_v} > {ext_v}"

        C_lin = rng.normal(size=(n, n))
        edges = [(0, 1, 0), (1, 2, 1), (2, 3, 2)]
        p_mats = [rng.normal(size=(n, n)) for _ in edges]
        assigns, lossless = solve_qap_permflow_birkhoff(C_lin, edges, p_mats)
        assert lossless is True and len(assigns) >= 1
        assert sorted(assigns[0]) == list(range(n)), f"Non-bijective QAP assignment: {assigns[0]}"

    vault_file = tmp_path / "mini_ood.json"
    vault_file.write_text(
        json.dumps(
            [
                {
                    "ID": "sv-1",
                    "Body": "Alice had 5 apples",
                    "Question": "She bought 3 more apples. How many apples does she have?",
                    "Answer": 8,
                    "Equation": "5 + 3",
                    "Type": "Addition",
                },
                {
                    "ID": "ma-1",
                    "question": "Bob has 4 pens and gets 2 more. How many pens total?",
                    "answer_val": 6.0,
                },
            ]
        ),
        encoding="utf-8",
    )
    vault = SingleShotMathVault(vault_file, max_calls=1)
    q_only = vault.read_questions_only()
    assert len(q_only) == 2 and vault.calls_used == 0
    assert q_only[0]["question"] == "Alice had 5 apples. She bought 3 more apples. How many apples does she have?"
    for rec in q_only:
        for secret_k in ("Answer", "answer_val", "Equation", "Body", "Question"):
            assert secret_k not in rec, f"Leaked answer field `{secret_k}` in read_questions_only()"

    unsealed = vault.open_once()
    assert vault.calls_used == 1
    assert unsealed[0]["answer_val"] == 8.0 and unsealed[1]["answer_val"] == 6.0
    try:
        vault.open_once()
    except RuntimeError:
        pass
    else:
        raise AssertionError("SingleShotMathVault allowed second open_once() call!")


def test_j4_gaq_live_grads_null_invariance_and_ckpt_roundtrip():
    torch.manual_seed(2026)
    bridge = SectionedGAQBridge(d_in=32, m_sections=4, d_code=16, num_codes=32, seed=2026)
    with torch.no_grad():
        c0 = bridge.codebooks().clone()
    opt = torch.optim.SGD([bridge.E_vq], lr=0.1)
    _, vql = bridge(torch.randn(32, 32))
    opt.zero_grad()
    vql.backward()
    opt.step()
    with torch.no_grad():
        c1 = bridge.codebooks()
        active_pct = 100.0 * float(((c1 - c0).abs().max(dim=-1).values > 1e-7).float().mean().item())
    assert active_pct == 100.0, f"Expected 100% Step-1 GAQ codebook utilization, got {active_pct}%"

    mini_items = [
        {
            "question": "Tom has 4 apples. He buys 3 more apples. How many apples does Tom have?",
            "answer": "Tom gets 4 + 3 = <<4+3=7>>7 apples.\n#### 7",
        },
        {
            "question": "Sara has 10 books. She gives away 2 books. How many books remain?",
            "answer": "Sara keeps 10 - 2 = <<10-2=8>>8 books.\n#### 8",
        },
        {
            "question": "A box holds 6 cups. John fills 5 boxes. How many cups in total?",
            "answer": "Total cups = 6 * 5 = <<6*5=30>>30.\n#### 30",
        },
        {
            "question": "Mia splits 12 cookies among 3 friends. How many cookies does each friend get?",
            "answer": "Each gets 12 / 3 = <<12/3=4>>4.\n#### 4",
        },
    ]
    d_toy = 16
    ext = TrainInducedQuantityExtractor()
    ext.fit(mini_items)
    alu = SphericalBase10RegisterALU(d_sub=15, num_digits=12, seed=3301)
    solver = UEBROpenGraphSolver(ext, alu, d_model=d_toy, device="cpu", seed=2026)
    solver._fit_corpus_stats(mini_items)
    # Emulate non-zero Woodbury linear anchors (set by ridge solve in fit_bptt_free) so (1 - g_risk) * u_lm is active
    with torch.no_grad():
        solver.ebm.Wlm0.normal_(std=0.05)
        solver.ebm.W_lm.copy_(solver.ebm.Wlm0)

    rng_np = np.random.default_rng(2026)
    sl0, _, _ = ext.extract_slots(mini_items[0]["question"], include_aux_consts=True)
    sl1, _, _ = ext.extract_slots(mini_items[1]["question"], include_aux_consts=True)
    lmf0 = {
        "h_num": rng_np.normal(size=(len(sl0), d_toy)).astype(np.float32),
        "h_post": rng_np.normal(size=(len(sl0), d_toy)).astype(np.float32),
        "h_q": rng_np.normal(size=(d_toy,)).astype(np.float32),
        "u_prob": 0.65,
        "u_risk": 0.72,
    }
    lmf1 = {
        "h_num": rng_np.normal(size=(len(sl1), d_toy)).astype(np.float32),
        "h_post": rng_np.normal(size=(len(sl1), d_toy)).astype(np.float32),
        "h_q": rng_np.normal(size=(d_toy,)).astype(np.float32),
        "u_prob": 0.61,
        "u_risk": 0.69,
    }

    pf0 = solver._featurize_problem(mini_items[0]["question"], sl0, lmf0)
    nf0 = _compute_null_slot_features(
        mini_items[0]["question"], sl0, pf0, ext, solver.vocab_counts, solver.stop_words, d_toy
    )
    assert nf0.shape == (len(sl0), D_NULL_INV) and np.isfinite(nf0).all()
    assert float(np.max(np.abs(nf0[:, 14:17]))) == 0.0, "Non-invariant positional cols [14:17] must be zeroed!"

    pf1 = solver._featurize_problem(mini_items[1]["question"], sl1, lmf1)
    cands0 = solver._synthesize_open_pool(sl0, pf0, mask_self=False)
    cands1 = solver._synthesize_open_pool(sl1, pf1, mask_self=False)
    pk0 = solver._pack_candidates_cpu(sl0, pf0, cands0)
    pk1 = solver._pack_candidates_cpu(sl1, pf1, cands1)
    tensors, m_counts = solver._collate_uebr_to_device([pk0, pk1])

    solver.ebm.train()
    out = solver.ebm.score_batch(*tensors, n_cands=sum(m_counts))
    assert out.z_flat is not None
    rl_t = torch.zeros(len(sl0), dtype=torch.long)
    rl_t[: min(2, len(sl0))] = 1
    loss = (
        out.e_tot.sum()
        + out.vq_loss
        + 0.1
        * solver.ebm.rlns_block3_fenchel_young_loss(
            out.u_tot,
            out.p_pot,
            0,
            len(sl0),
            pk0["pair_map_arr"],  # type: ignore[arg-type]
            rl_t,
            pair_iu_t=torch.from_numpy(pk0["pair_iu"]),  # type: ignore[arg-type]
            pair_iv_t=torch.from_numpy(pk0["pair_iv"]),  # type: ignore[arg-type]
            pair_pidx_t=torch.from_numpy(pk0["pair_pidx"]),  # type: ignore[arg-type]
        )
        + 0.1 * solver.ebm.spd_laplacian.directional_eikonal_loss(out.z_flat[:1], out.z_flat[1:2], out.q_prob[:1])
        + solver.ebm.prox_regularization_loss()
    )
    solver.ebm.zero_grad(set_to_none=True)
    loss.backward()
    n_tot = n_live = 0
    for p in solver.ebm.parameters():
        if not p.requires_grad:
            continue
        n_tot += 1
        assert p.grad is not None and torch.isfinite(p.grad).all()
        if float(p.grad.abs().max().item()) > 0.0:
            n_live += 1
    assert n_tot >= 60 and n_live == n_tot, f"Dead grads detected in UEBROpenGraphEBM: {n_live}/{n_tot}"

    # Verify (N_CERT_FEATS + 1 == 13) cert_weights & telemetry state_dict roundtrip
    solver.cert_weights = np.linspace(-0.5, 0.7, N_CERT_FEATS + 1, dtype=np.float64)
    solver.birkhoff_total_solves = 42
    solver.birkhoff_lossless_wins = 42
    solver.last_min_eig = 0.625
    solver.last_epc_err = 1.5e-5
    sd_rt = solver.state_dict()
    clone = UEBROpenGraphSolver(ext, alu, d_model=d_toy, device="cpu", seed=99)
    clone.load_state_dict(sd_rt)
    assert len(clone.cert_weights) == N_CERT_FEATS + 1 == 13
    assert np.allclose(clone.cert_weights, solver.cert_weights, atol=1e-12)
    assert clone.birkhoff_total_solves == 42 and abs(clone.last_min_eig - 0.625) < 1e-12

    if PHASE_J_CKPT.is_file():
        sd_disk = torch.load(PHASE_J_CKPT, map_location="cpu", weights_only=False)
        disk_slv = UEBROpenGraphSolver(ext, alu, d_model=384, device="cpu", seed=2026)
        disk_slv.load_state_dict(sd_disk)
        assert len(disk_slv.cert_weights) == N_CERT_FEATS + 1 == 13


def test_j5_phase_j_json_and_postmortem_archives():
    for path in (PHASE_J_JSON, PM1_JSON, PM2_JSON):
        assert path.is_file(), f"Missing Phase J artifact: {path}"
        _assert_finite_nested(json.loads(path.read_text(encoding="utf-8")), trail=path.name)

    rep = json.loads(PHASE_J_JSON.read_text(encoding="utf-8"))
    proto = rep["protocol"]
    assert int(proto["gsm8k_test_calls_used"]) == 0
    assert int(proto["multiarith_vault_calls_used"]) == 1
    assert int(proto["asdiv_vault_calls_used"]) == 1
    assert float(proto["matrix_adjoint_gradcheck_err"]) < 1e-7
    assert rep["unsealed_svamp_postmortem_recheck"]["not_blind_vault"] is True

    ts = rep["two_sigma_noise_floor"]
    lad = rep["ablation_ladder"]
    b0_26, b0_27 = lad["B0_seed2026"], lad["B0_seed2027"]
    a3_26, a3_27 = lad["A3_full_uebr_seed2026"], lad["A3_full_uebr_seed2027"]

    assert rep["stop_crane_passed"] is True
    mean_b0_dev = 0.5 * (float(b0_26["blind_dev"]["pass1_pct"]) + float(b0_27["blind_dev"]["pass1_pct"]))
    mean_a3_dev = 0.5 * (float(a3_26["blind_dev"]["pass1_pct"]) + float(a3_27["blind_dev"]["pass1_pct"]))
    mean_b0_str = 0.5 * (float(b0_26["blind_dev_stress"]["pass1_pct"]) + float(b0_27["blind_dev_stress"]["pass1_pct"]))
    mean_a3_str = 0.5 * (float(a3_26["blind_dev_stress"]["pass1_pct"]) + float(a3_27["blind_dev_stress"]["pass1_pct"]))
    assert mean_a3_dev > mean_b0_dev + float(ts["blind_dev_pass1_pp"])
    assert mean_a3_str >= 4.0 * mean_b0_str

    for a3_r in (a3_26, a3_27):
        bd = a3_r["blind_dev"]
        bs = a3_r["blind_dev_stress"]
        fit = a3_r["fit"]
        assert float(bd["pass1_pct"]) >= 1.6 * float(bd["unary_ablation_pass1_pct"])
        assert float(bs["birkhoff_lossless_pct"]) == 100.0
        assert float(bd["max_norm_drift"]) < 1e-6 and float(bs["max_norm_drift"]) < 1e-6
        assert float(fit["laplacian_min_eig"]) > 0.0
        assert float(fit["epc_max_err"]) < 1e-4
        assert float(fit["wm_eff_min_pen"]) > 0.0
        assert float(fit["wu_drift_norm"]) < 6.0

    ood = rep["sealed_ood_benchmarks"]
    ma = ood["MultiArith_600"]
    assert float(ma["A3_uebr"]["pass1_pct"]) > float(ma["B0_base"]["pass1_pct"])
    assert float(ma["A3_uebr"]["pass5_pct"]) > float(ma["B0_base"]["pass5_pct"])
    assert float(ma["A3_uebr"]["birkhoff_lossless_pct"]) == 100.0

    asdiv = ood["ASDiv_A"]
    assert float(asdiv["A3_uebr"]["oracle_pct"]) > float(asdiv["B0_base"]["oracle_pct"])
    assert float(asdiv["A3_uebr"]["ge4_slots"]["pass1_pct"]) > float(asdiv["B0_base"]["ge4_slots"]["pass1_pct"])
    assert float(asdiv["A3_uebr"]["birkhoff_lossless_pct"]) == 100.0
    # Honest preservation of OEB Attempt #3 mixed OOD gate (ASDiv-A 1-step le3_slots domain shift)
    assert rep["all_passed"] is False
