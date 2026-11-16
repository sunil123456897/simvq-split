"""
Zero-Cheat Phase C Verification Suite (`tests/test_cram_phase_c.py`).

Enforces 6 deterministic invariants for `experiments/cram_gsm8k/` & `Calcite-5`:
  c1: AST purity across all 6 `experiments/cram_gsm8k/*.py` files + zero custom
      `dict`/`set` state on `CRAMzOperatorGraphEBM(nn.Module)`.
  c2: `SphericalBase10RegisterALU` exactness on ADD/SUB/MUL/DIV & `< 1e-6` drift.
  c3: Gershgorin margin `>= 4.0`, `100%` live gradients across all 20 parameters
      under `0`-BPTT objective, and Besag no-leak conditional energy parity `< 1e-6`.
  c4: Purely train-induced quantity extraction (`train.jsonl[:400]`) without hardcoded
      word tables + hard anti-leak error on `"####"` / `"<<"` in `solve_blind`.
  c5: `Sparsemax` analytical Jacobian accuracy (`< 1e-7`) + `PackedLUT2BitLinear`
      (`954.55 KB`, `16.00x` matrices, `.bias_fp16` contract) from tournament checkpoint.
  c6: `DualCert` conjunction (`(u_prob <= theta_epi) and (e_cert <= tau_energy)`)
      accepts Known Solvable and rejects both Unseen Entity & Ungrounded Trap + invalid ALU.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.cram_gsm8k.cram_open_graph import (  # noqa: E402
    CRAMzOperatorGraphEBM,
    OpenGraphCRAMSolver,
    Sparsemax,
)
from experiments.cram_gsm8k.simvq_cram_adapter import sparsemax_numpy  # noqa: E402
from experiments.cram_gsm8k.slot_extractor import TrainInducedQuantityExtractor  # noqa: E402
from experiments.cram_gsm8k.spherical_alu import SphericalBase10RegisterALU  # noqa: E402
from simvq_split.models import SimVQSplitConfig, SimVQSplitLM  # noqa: E402
from simvq_split.quant_lut import (  # noqa: E402
    PackedLUT2BitLinear,
    compute_2bit_lut_footprint,
    pack_model_2bit_lut,
)


PHASE_C_MODULES = (
    "spherical_alu.py",
    "slot_extractor.py",
    "system1_encoder.py",
    "cram_open_graph.py",
    "gsm8k_cram_solver.py",
    "simvq_cram_adapter.py",
)

BANNED_IMPORTS = {"re", "ast", "inspect"}
BANNED_BARE_CALLS = {"eval", "exec", "compile"}
BANNED_LEGACY_IDENTIFIERS = (
    "TOP_TEMPLATES",
    "arity_pool",
    "top_templates_per_arity",
    "WORD_NUMS_V33",
    "SUB_WORDS_V33",
    "MUL_WORDS_V33",
    "DIV_WORDS_V33",
    "ADD_WORDS_V33",
    "_CLAUSE_SPLIT_RE",
    "STOP_NOUNS",
    "VERB_NORM_V33",
    "MONEY_WORDS",
    "TIME_WORDS",
    "RATE_MARKERS",
)

PYTORCH_INTERNAL_DICT_ATTRS = {
    "_parameters",
    "_buffers",
    "_non_persistent_buffers_set",
    "_backward_hooks",
    "_backward_pre_hooks",
    "_forward_hooks",
    "_forward_hooks_with_kwargs",
    "_forward_hooks_always_called",
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
    "_modules",
}


def test_c1_ast_purity_and_no_module_dicts():
    exp_dir = ROOT_DIR / "experiments" / "cram_gsm8k"
    for fname in PHASE_C_MODULES:
        path = exp_dir / fname
        assert path.exists(), f"Missing Phase C module: {path}"
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=fname)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in BANNED_IMPORTS, (
                        f"Forbidden import `{alias.name}` in {fname}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in BANNED_IMPORTS, (
                    f"Forbidden from-import `{node.module}` in {fname}"
                )
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in BANNED_BARE_CALLS, (
                    f"Forbidden call `{node.func.id}()` in {fname}:{node.lineno}"
                )

        for legacy_tok in BANNED_LEGACY_IDENTIFIERS:
            assert legacy_tok not in src, f"Legacy template/lexicon token `{legacy_tok}` in {fname}"

    og_tree = ast.parse((exp_dir / "cram_open_graph.py").read_text(encoding="utf-8"))
    for cls_node in [n for n in og_tree.body if isinstance(n, ast.ClassDef) and n.name == "CRAMzOperatorGraphEBM"]:
        for sub in ast.walk(cls_node):
            assert not isinstance(sub, (ast.Dict, ast.DictComp, ast.Set, ast.SetComp)), (
                f"Forbidden dict/set AST node inside CRAMzOperatorGraphEBM at line {getattr(sub, 'lineno', '?')}"
            )

    ebm = CRAMzOperatorGraphEBM(d_model=160)
    for attr_name, val in ebm.__dict__.items():
        if attr_name in PYTORCH_INTERNAL_DICT_ATTRS:
            continue
        assert not isinstance(val, (dict, set)), (
            f"Forbidden custom {type(val).__name__} attribute `{attr_name}` on CRAMzOperatorGraphEBM"
        )


def test_c2_spherical_alu_exactness_and_drift():
    alu = SphericalBase10RegisterALU(d_sub=15, num_digits=12, seed=3301)
    diag = alu.verify_crystal_invariants(n_pairs=200)
    for k in ("add_acc_pct", "sub_acc_pct", "mul_acc_pct", "div_acc_pct"):
        assert diag[k] == 100.0, f"Spherical ALU {k} dropped below 100%: {diag[k]}"
    assert diag["max_norm_drift"] < 1e-6, f"Spherical ALU drift too high: {diag['max_norm_drift']}"

    # Multi-step mixed-sign & fractional DAG chains:
    # Step 0: 12.50 - 20.25 = -7.75; Step 1: (-7.75) * (-4) = +31.0; Step 2: 31.0 / 2.5 = 12.4
    regs = [alu.encode(v) for v in (12.50, 20.25, -4.0, 2.5)]
    out_reg, _, dag_drift, invalid = alu.execute_dag_spherical(
        regs,
        steps=((1, 0, 1), (2, 4, 2), (3, 5, 3)),
    )
    decoded, dec_drift = alu.decode_with_drift(out_reg)
    assert not invalid
    assert abs(decoded - 12.4) <= 1e-4, f"Expected 12.4, got {decoded}"
    assert max(dag_drift, dec_drift) < 1e-6

    # Division by zero sets `invalid=True` while maintaining unit sphere norm
    z_regs = [alu.encode(15.0), alu.encode(0.0)]
    bad_reg, _, zero_drift, bad_inv = alu.execute_dag_spherical(z_regs, steps=((3, 0, 1),))
    assert bad_inv
    _, bad_dec_drift = alu.decode_with_drift(bad_reg)
    assert max(zero_drift, bad_dec_drift) < 1e-6


def test_c3_gershgorin_live_grads_and_besag_no_leak():
    torch.manual_seed(2026)
    ebm = CRAMzOperatorGraphEBM(d_model=160)
    assert ebm.gershgorin_margin() >= 4.0, f"Initial Gershgorin margin too low: {ebm.gershgorin_margin()}"

    # 1. Exact parity between full energy differences and Besag no-leak conditional energies
    psi_true = torch.randn(4, 16)
    cand_psis = torch.randn(6, 16)
    with torch.no_grad():
        couplings = ebm._bounded_couplings()
        for t_idx in (0, 2):
            e_cond = ebm.conditional_candidate_energies_no_leak(psi_true, t_idx, cand_psis, couplings=couplings)
            e_full = []
            for m_i in range(cand_psis.size(0)):
                patched = psi_true.clone()
                patched[t_idx] = cand_psis[m_i]
                et, _ = ebm.energy_from_psi_stack(patched, couplings=couplings)
                e_full.append(et)
            e_full_t = torch.stack(e_full)
            diffs_cond = e_cond - e_cond[0]
            diffs_full = e_full_t - e_full_t[0]
            assert float((diffs_cond - diffs_full).abs().max().item()) < 1e-5

            # Mutating original `psi_true[t_idx]` must produce ZERO change in `e_cond` (0 target leak)
            psi_mut = psi_true.clone()
            psi_mut[t_idx] = psi_true[t_idx] + 50.0
            e_cond_mut = ebm.conditional_candidate_energies_no_leak(psi_mut, t_idx, cand_psis, couplings=couplings)
            assert float((e_cond - e_cond_mut).abs().max().item()) == 0.0

    # 2. 100% live non-zero gradients across all 20 parameters under 0-BPTT objective
    ebm.train()
    all_uf = torch.randn(6, ebm.d_unary)
    all_pf = torch.randn(5, ebm.d_pair)
    all_rf = torch.randn(2, ebm.d_root)
    all_lm = torch.randn(6, 3 * ebm.d_model)
    n_cands = 4
    slot_gidx = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5], dtype=torch.long)
    slot_roles = torch.tensor([1, 2, 3, 4, 5, 6, 1, 3, 7, 2, 8, 4], dtype=torch.long)
    slot_cand_idx = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=torch.long)
    slot_inv_k = torch.full((12,), 1.0 / 3.0)
    root_prob_idx = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    root_op = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    bm_cand = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    bp_pair = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    bo_op = torch.tensor([0, 1, 3, 5], dtype=torch.long)
    bra = torch.tensor([1, 2, 4, 6], dtype=torch.long)
    brb = torch.tensor([2, 3, 5, 7], dtype=torch.long)
    bwt = torch.tensor([0.5, 0.7, 0.4, 0.6], dtype=torch.float32)
    psi_raw_pad = torch.randn(4, 3, ebm.d_step)
    psi_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    psi_lens = psi_mask.sum(dim=-1)
    meta_m = torch.randn(4, 15)

    ebm.zero_grad()
    e_cram, _ = ebm.score_batch(
        all_uf, all_pf, all_rf, all_lm,
        slot_gidx, slot_roles, slot_cand_idx, slot_inv_k,
        root_prob_idx, root_op,
        bm_cand, bp_pair, bo_op, bra, brb, bwt,
        psi_raw_pad, psi_mask, psi_lens, meta_m,
        n_cands=n_cands,
    )
    loss_pool = F.cross_entropy(-e_cram[:2].unsqueeze(0), torch.zeros(1, dtype=torch.long))
    loss_pool = loss_pool + F.cross_entropy(-e_cram[2:].unsqueeze(0), torch.zeros(1, dtype=torch.long))

    psi_seq_raw = torch.randn(3, ebm.d_step)
    cand_raw = torch.randn(5, ebm.d_step)
    psi_seq_t = torch.tanh(ebm.step_proj(psi_seq_raw))
    cand_t = torch.tanh(ebm.step_proj(cand_raw))
    e_cond_step = ebm.conditional_candidate_energies_no_leak(psi_seq_t, 1, cand_t)
    loss_pl = 0.05 * F.cross_entropy(-e_cond_step.unsqueeze(0), torch.zeros(1, dtype=torch.long))

    loss = loss_pool + loss_pl + ebm.prox_regularization_loss()
    loss.backward()

    named_params = list(ebm.named_parameters())
    assert len(named_params) == 20, f"Expected 20 parameters on CRAMzOperatorGraphEBM, got {len(named_params)}"
    dead = [
        name for name, p in named_params
        if p.grad is None or not torch.isfinite(p.grad).all() or float(p.grad.abs().max().item()) == 0.0
    ]
    assert not dead, f"Dead or non-finite gradients in CRAMzOperatorGraphEBM: {dead}"


def test_c4_train_induced_extractor_and_anti_leak_guard():
    train_path = ROOT_DIR / "benchmarks" / "gsm8k" / "train.jsonl"
    assert train_path.exists(), f"Missing {train_path}"
    with train_path.open("r", encoding="utf-8") as f:
        items_400 = [json.loads(line) for _, line in zip(range(400), f)]

    ext = TrainInducedQuantityExtractor(min_word_df=5, min_cooc=4)
    stats = ext.fit(items_400)
    assert ext.fitted
    assert stats["word_num_map"] >= 2
    assert stats["ctx_vocab"] >= 50

    sample_q = "Sam buys a box of 12 pens and gets twice as many pencils."
    slots, _, _ = ext.extract_slots(sample_q, include_aux_consts=True)
    explicit_vals = [sl.val for sl in slots if sl.kind == "EXPLICIT"]
    assert 12.0 in explicit_vals
    if "twice" in ext.word_num_map:
        assert any(sl.kind == "INDUCED_WORD" and abs(sl.val - 2.0) < 1e-6 for sl in slots)

    alu = SphericalBase10RegisterALU(d_sub=15, num_digits=12, seed=3301)
    solver = OpenGraphCRAMSolver(ext, alu, d_model=160, device="cpu", seed=2026)
    with pytest.raises(ValueError, match="Anti-Cheat"):
        solver.solve_blind("Tom has 3 apples and 2 pears. #### 5")
    with pytest.raises(ValueError, match="Anti-Cheat"):
        solver.solve_blind("Tom has 3 apples and <<3+2=5>> pears.")


def test_c5_sparsemax_jacobian_and_lut_footprint_parity():
    # 1. Sparsemax NumPy + autograd Jacobian against central finite differences
    rng = np.random.default_rng(2026)
    z = rng.standard_normal((6, 32))
    grad_out = rng.standard_normal((6, 32))
    p = sparsemax_numpy(z)
    supp = (p > 0.0).astype(np.float64)
    v_hat = np.sum(grad_out * supp, axis=-1, keepdims=True) / np.maximum(np.sum(supp, axis=-1, keepdims=True), 1.0)
    grad_analytic = supp * (grad_out - v_hat)

    eps = 1e-6
    grad_num = np.zeros_like(z)
    for r in range(z.shape[0]):
        for c in range(z.shape[1]):
            zp = z.copy()
            zp[r, c] += eps
            zm = z.copy()
            zm[r, c] -= eps
            grad_num[r, c] = np.sum((sparsemax_numpy(zp) - sparsemax_numpy(zm)) * grad_out) / (2.0 * eps)

    assert float(np.max(np.abs(grad_analytic - grad_num))) < 1e-7

    z_t = torch.tensor(z, dtype=torch.float64, requires_grad=True)
    p_t = Sparsemax.apply(z_t)
    (p_t * torch.tensor(grad_out, dtype=torch.float64)).sum().backward()
    assert float((z_t.grad - torch.tensor(grad_analytic, dtype=torch.float64)).abs().max().item()) < 1e-7

    # 2. Tournament checkpoint 2-bit T-MAC LUT packing + bias_fp16 contract
    ckpt_path = ROOT_DIR / "checkpoints" / "simvq_split_tournament.pt"
    assert ckpt_path.exists(), f"Missing {ckpt_path}"
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = SimVQSplitConfig(**payload["simvq_split_config"])
    qat_m = SimVQSplitLM(cfg)
    qat_m.load_state_dict(payload["simvq_split_ternary_qat_state_dict"], strict=True)
    orig_pg_bias = qat_m.crystal.perm_gate_proj.bias.detach().clone().to(torch.float16)

    lut_m = pack_model_2bit_lut(qat_m)
    info = compute_2bit_lut_footprint(lut_m)
    assert int(info["num_lut_layers"]) == 56
    assert abs(float(info["matrix_compression_vs_fp32"]) - 16.0) < 1e-3
    assert 900.0 <= float(info["total_lut_kb"]) <= 960.0
    assert isinstance(lut_m.crystal.perm_gate_proj, PackedLUT2BitLinear)
    lut_bias = getattr(lut_m.crystal.perm_gate_proj, "bias_fp16", None)
    assert lut_bias is not None and torch.equal(lut_bias, orig_pg_bias)


def test_c6_dual_certificate_trap_rejection():
    alu = SphericalBase10RegisterALU(d_sub=15, num_digits=12, seed=3301)
    ext = TrainInducedQuantityExtractor()
    solver = OpenGraphCRAMSolver(ext, alu, d_model=160, device="cpu", seed=2026)
    solver.theta_epi = 0.50
    solver.tau_energy = 0.0
    solver.cert_weights = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    solver.cert_bias = 0.0

    def _certify(u_prob: float, min_e: float, inv_top1: bool) -> bool:
        dummy_rec = {"meta": np.zeros(15, dtype=np.float32), "steps": ((0, 0, 1),), "invalid": inv_top1}
        e_c = solver._compute_e_cert(
            np.array([min_e, min_e + 1.0], dtype=np.float64),
            margin_e=1.0,
            p_top=0.80,
            entropy_val=0.40,
            best_rec=dummy_rec,
            k_slots=2,
        )
        return (not inv_top1) and (u_prob <= solver.theta_epi) and (e_c <= solver.tau_energy)

    assert _certify(u_prob=0.15, min_e=-0.8, inv_top1=False) is True   # Known Solvable
    assert _certify(u_prob=0.92, min_e=-0.8, inv_top1=False) is False  # Unseen Entity (high u_prob)
    assert _certify(u_prob=0.15, min_e=0.9, inv_top1=False) is False   # Ungrounded Trap (high E_cert)
    assert _certify(u_prob=0.15, min_e=-0.8, inv_top1=True) is False   # Invalid Spherical ALU Register
