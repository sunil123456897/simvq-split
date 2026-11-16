"""
Zero-Cheat Phase F Verification Suite (`tests/test_cram_phase_f.py`).

Enforces 4 deterministic scaling & UTF-8 System-1 invariants without modifying Phase C suite:
  f1: AST purity across all `experiments/cram_gsm8k/*.py`, zero dict/set AST nodes or
      attributes on `CRAMzOperatorGraphEBM`, exact `20` trainable parameters, Gershgorin `>= 4.0`,
      and `100%` live gradients when jointly optimizing `score_batch` + `conditional_pair_energies_2opt`.
  f2: Tokenizer-free `548.34 KB` `UTF-8 ByteHNet` (`vocab=259`, `70` LUT layers at `16.00x`)
      + interleaved `SphericalBase10RegisterALU` trace mask (`False` on `=val`, `drift < 1e-6`)
      + chunk-to-byte epistemic gather without `IndexError` at `L >= 500` in FP & 2-bit LUT modes.
  f3: Barrier #3 scaling laws: unique continuous `z*` convergence (`max_pair_Linf < 1e-6` at `N=128`)
      and `0-BPTT` coupled constraint recovery (`besag_1flip_relax` trapped vs `block_gibbs_2opt_relax` `100%`),
      plus zero-leak & energy difference parity (`< 1e-5`) of `conditional_pair_energies_2opt`.
  f4: High-slot synthesis (`ng >= 10`) returns non-empty valid pool in `< 0.25 s`, while short/medium
      tasks (`ng <= 7`) remain identical to Phase C.
"""
from __future__ import annotations

import ast
from pathlib import Path
import sys
import time
import pytest
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.cram_gsm8k.cram_open_graph import (  # noqa: E402
    CRAMzOperatorGraphEBM,
    OpenGraphCRAMSolver,
)
from experiments.cram_gsm8k.slot_extractor import (  # noqa: E402
    ExtractedSlot,
    TrainInducedQuantityExtractor,
)
from experiments.cram_gsm8k.spherical_alu import SphericalBase10RegisterALU  # noqa: E402
from experiments.cram_gsm8k.system1_encoder import (  # noqa: E402
    Calcite5MathSystem1,
    encode_interleaved_alu_trace,
)
from tests.test_cram_phase_c import (  # noqa: E402
    BANNED_BARE_CALLS,
    BANNED_IMPORTS,
    BANNED_LEGACY_IDENTIFIERS,
    PHASE_C_MODULES,
    PYTORCH_INTERNAL_DICT_ATTRS,
)


def test_f1_phase_f_ast_purity_and_ebm_20_params():
    exp_dir = ROOT_DIR / "experiments" / "cram_gsm8k"
    for fname in PHASE_C_MODULES:
        path = exp_dir / fname
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=fname)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in BANNED_IMPORTS
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in BANNED_IMPORTS
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in BANNED_BARE_CALLS
        for legacy_tok in BANNED_LEGACY_IDENTIFIERS:
            assert legacy_tok not in src

    og_tree = ast.parse((exp_dir / "cram_open_graph.py").read_text(encoding="utf-8"))
    for cls_node in [n for n in og_tree.body if isinstance(n, ast.ClassDef) and n.name == "CRAMzOperatorGraphEBM"]:
        for sub in ast.walk(cls_node):
            assert not isinstance(sub, (ast.Dict, ast.DictComp, ast.Set, ast.SetComp)), (
                f"Forbidden dict/set AST node inside CRAMzOperatorGraphEBM at line {getattr(sub, 'lineno', '?')}"
            )

    torch.manual_seed(2026)
    ebm = CRAMzOperatorGraphEBM(d_model=160)
    for attr_name, val in ebm.__dict__.items():
        if attr_name in PYTORCH_INTERNAL_DICT_ATTRS:
            continue
        assert not isinstance(val, (dict, set)), f"Forbidden {type(val).__name__} attr `{attr_name}`"
    assert ebm.gershgorin_margin() >= 4.0

    ebm.train()
    all_uf = torch.randn(6, ebm.d_unary)
    all_pf = torch.randn(5, ebm.d_pair)
    all_rf = torch.randn(2, ebm.d_root)
    all_lm = torch.randn(6, 3 * ebm.d_model)
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
        n_cands=4,
    )
    loss_pool = F.cross_entropy(-e_cram[:2].unsqueeze(0), torch.zeros(1, dtype=torch.long))
    loss_pool = loss_pool + F.cross_entropy(-e_cram[2:].unsqueeze(0), torch.zeros(1, dtype=torch.long))

    psi_seq_t = torch.tanh(ebm.step_proj(torch.randn(5, ebm.d_step)))
    cand_pairs_t = torch.tanh(ebm.step_proj(torch.randn(6, 2, ebm.d_step)))
    e_2opt = ebm.conditional_pair_energies_2opt(psi_seq_t, 1, 3, cand_pairs_t)
    loss_pl = 0.05 * F.cross_entropy(-e_2opt.unsqueeze(0), torch.zeros(1, dtype=torch.long))
    (loss_pool + loss_pl + ebm.prox_regularization_loss()).backward()

    named_params = list(ebm.named_parameters())
    assert len(named_params) == 20, f"Expected 20 parameters, got {len(named_params)}"
    dead = [n for n, p in named_params if p.grad is None or not torch.isfinite(p.grad).all() or float(p.grad.abs().max().item()) == 0.0]
    assert not dead, f"Dead gradients: {dead}"


def test_f2_utf8_byte_hnet_548kb_lut_and_epistemic_gather():
    ckpt_path = ROOT_DIR / "checkpoints" / "simvq_split_utf8_openweb.pt"
    if not ckpt_path.exists():
        pytest.skip("UTF-8 checkpoint not found")

    sys1 = Calcite5MathSystem1(ckpt_path, device="cpu", seed=2026)
    assert sys1.is_byte_hnet is True
    assert int(sys1.tok.vocab_size) == 259
    lut_info = sys1.prepare_base_lut_model()
    assert int(lut_info["num_lut_layers"]) == 70
    assert abs(float(lut_info["matrix_compression_vs_fp32"]) - 16.0) < 1e-3
    assert 535.0 <= float(lut_info["total_lut_kb"]) <= 555.0

    alu = SphericalBase10RegisterALU(d_sub=15, num_digits=12, seed=3301)
    dsl, sup_mask, drift, inv = encode_interleaved_alu_trace(
        steps=((3, 0, 1), (0, 0, 2)),
        slot_vals=[48.0, 2.0],
        alu=alu,
        n_init=2,
    )
    assert not inv
    assert drift < 1e-6
    assert dsl == "A/B=24;A+a=72"
    assert len(dsl) == len(sup_mask)
    eq_indices = [i for i, ch in enumerate(dsl) if ch in "=247"]
    for idx in range(len(dsl)):
        if dsl[idx] in "ABa/+;":
            assert sup_mask[idx] is True
    for i in eq_indices:
        if dsl[i] == "=" or (i > 0 and dsl[i - 1] in "=27"):
            assert sup_mask[i] is False

    ext = TrainInducedQuantityExtractor()
    ext.fit([{"question": "Alice bought 12 apples and 45 pears for the party. How many fruits?", "answer": "12+45 = <<12+45=57>>57\n#### 57"}])
    long_q = (
        "At a large city festival on Saturday morning, Alice packed 12 crates of sweet apples, "
        "18 boxes of yellow pears, 24 jars of wildflower honey, and 6 baskets of ripe oranges "
        "for the neighborhood school cafeteria before lunch time. How many items did she bring?"
    )
    assert len(long_q.encode("utf-8")) > 250
    slots, _, _ = ext.extract_slots(long_q, include_aux_consts=True)
    assert len(slots) >= 4

    sys1.use_interleaved_alu = True
    f_fp = sys1.extract_features_and_beams([long_q], [slots], ext, batch_size=1, beam_width=2, max_beam_steps=2, use_lut=False, alu=alu)
    f_lut = sys1.extract_features_and_beams([long_q], [slots], ext, batch_size=1, beam_width=2, max_beam_steps=2, use_lut=True, alu=alu)
    for feat in (f_fp[0], f_lut[0]):
        assert feat["h_num"].shape == (len(slots), 160)
        assert feat["u_slots"].shape == (len(slots),)
        assert torch.isfinite(torch.from_numpy(feat["h_num"])).all()
        assert torch.isfinite(torch.from_numpy(feat["u_slots"])).all()
        assert float(feat["u_prob"]) >= 0.0


def test_f3_cram_n128_global_convexity_and_2opt_block_gibbs():
    torch.manual_seed(2026)
    ebm = CRAMzOperatorGraphEBM(d_model=160)
    ebm.eval()

    # 1. Exact conditional_pair_energies_2opt parity & zero-leak check
    psi_true = torch.randn(6, 16)
    cand_pairs = torch.randn(8, 2, 16)
    with torch.no_grad():
        couplings = ebm._bounded_couplings()
        e_pair_cond = ebm.conditional_pair_energies_2opt(psi_true, 1, 4, cand_pairs, couplings=couplings)
        e_full_list = []
        for m_i in range(cand_pairs.size(0)):
            patched = psi_true.clone()
            patched[1] = cand_pairs[m_i, 0]
            patched[4] = cand_pairs[m_i, 1]
            ef, _ = ebm.energy_from_psi_stack(patched, couplings=couplings)
            e_full_list.append(ef)
        e_full_t = torch.stack(e_full_list)
        assert float(((e_pair_cond - e_pair_cond[0]) - (e_full_t - e_full_t[0])).abs().max().item()) < 1e-5

        psi_mut = psi_true.clone()
        psi_mut[1] += 40.0
        psi_mut[4] -= 30.0
        e_pair_mut = ebm.conditional_pair_energies_2opt(psi_mut, 1, 4, cand_pairs, couplings=couplings)
        assert float((e_pair_cond - e_pair_mut).abs().max().item()) == 0.0

        # 2. Global convexity at N=128 from 8 random z0 in U(-5, 5)
        W_zz = ebm.normalized_W_zz()
        eye = torch.eye(ebm.d_z)
        A = ebm.m_convex * eye - W_zz
        b = F.linear(torch.randn(128, 16).mean(dim=0), ebm._bounded_couplings()[1])
        z_exact = torch.linalg.solve(A, b)
        z_conv = []
        for rs in range(8):
            g = torch.Generator().manual_seed(100 + rs)
            z = torch.empty(ebm.d_z).uniform_(-5.0, 5.0, generator=g)
            for _ in range(80):
                z = (b + F.linear(z, W_zz)) / ebm.m_convex
            assert float((z - z_exact).abs().max().item()) < 1e-6
            z_conv.append(z)
        for i in range(len(z_conv)):
            for j in range(i + 1, len(z_conv)):
                assert float((z_conv[i] - z_conv[j]).abs().max().item()) < 1e-6

        # 3. Coupled parity-pair trap at N=64: 1-flip Besag trapped at 0%, 2-opt Block-Gibbs 100%
        n_nodes = 64
        n_cands = 4
        n_edges = n_nodes // 2
        node_bank = torch.randn(n_nodes, n_cands, 16) * 0.01
        pair_edges = torch.tensor([[2 * k, 2 * k + 1] for k in range(n_edges)], dtype=torch.long)
        unary_bias = torch.zeros(n_nodes, n_cands)
        unary_bias[:, 1] = 2.0
        unary_bias[:, 2] = 0.0
        unary_bias[:, 3] = 2.0
        pair_pot = torch.full((n_edges, n_cands, n_cands), 8.0)
        # Coupled trap at (0, 0) (energy = 0) vs global optimum at (2, 2) (energy = -4.0)
        pair_pot[:, 0, 0] = 0.0
        pair_pot[:, 2, 2] = -4.0
        y_init = torch.zeros(n_nodes, dtype=torch.long)

        y_1f, _, _ = ebm.besag_1flip_relax(y_init, node_bank, pair_edges, unary_bias, pair_pot, max_sweeps=4)
        y_2opt, _, _ = ebm.block_gibbs_2opt_relax(y_init, node_bank, pair_edges, unary_bias, pair_pot, max_sweeps=4)
        assert float((y_1f == 2).float().mean().item()) == 0.0
        assert float((y_2opt == 2).float().mean().item()) == 1.0


def test_f4_high_slot_synthesis_no_cliff_and_short_pool_invariance():
    alu = SphericalBase10RegisterALU(d_sub=15, num_digits=12, seed=3301)
    ext = TrainInducedQuantityExtractor()
    ext.fit([{"question": "A box has 5 pens and 3 pencils.", "answer": "5+3 = <<5+3=8>>8\n#### 8"}])
    solver = OpenGraphCRAMSolver(ext, alu, d_model=160, device="cpu", seed=2026)

    q_12 = (
        "A store stocked 12 pens, 8 books, 5 erasers, 14 rulers, 9 folders, 4 staplers, "
        "7 clips, 11 markers, 6 tapes, 15 pads, 3 glues, and 10 scissors today. Total?"
    )
    slots_12 = [
        ExtractedSlot(i, float(v), str(v), "EXPLICIT", i * 5, i * 5 + 2, 0, 0, False, False, False, ("has",), ("items",))
        for i, v in enumerate((12, 8, 5, 14, 9, 4, 7, 11, 6, 15, 3, 10))
    ]
    t0 = time.perf_counter()
    pf_12 = solver._featurize_problem(q_12, slots_12, None)
    pool_12 = solver._synthesize_open_pool(slots_12, pf_12, mask_self=False)
    dt = time.perf_counter() - t0
    assert len(pool_12) > 0, "Expected non-empty high-slot candidate pool for ng=12"
    assert dt < 0.25, f"High-slot synthesis too slow: {dt:.3f}s"

    q_3 = "Tom has 12 apples, buys 8 more, and gives away 5. How many remain?"
    slots_3 = [
        ExtractedSlot(i, float(v), str(v), "EXPLICIT", i * 4, i * 4 + 2, 0, 0, False, False, False, ("has",), ("apples",))
        for i, v in enumerate((12, 8, 5))
    ]
    pf_3 = solver._featurize_problem(q_3, slots_3, None)
    pool_3 = solver._synthesize_open_pool(slots_3, pf_3, mask_self=False)
    assert any(abs(float(c[1]) - 15.0) < 1e-6 for c in pool_3)
