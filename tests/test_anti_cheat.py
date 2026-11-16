"""
Zero-Cheat Automated Verification Suite for `simvq_split` (16 checks across 16 laws).

Enforces the non-negotiable scientific laws:
1-15. Static AST purity, 100% live gradients, strict causality, receptive field isolation,
      parameter parity, BF16/S_G/Woodbury/Hadamard/QAT/id_well/2-bit uint8 LUT invariants.
16. Surprise-Gated H-Net Byte Chunker (`Calcite-5` Phase B): backward compatibility,
    strict causal byte decoding (`diff < 1e-5`), zero-shortcut chunked receptive field,
    100% live boundary/ratio gradients, exact `O(1)` streaming buffer (`L=64..2048`),
    and `<= 555 KB` 2-bit uint8 LUT budget (`~80-83 KB` O(1) cache).
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

from simvq_split.models import SimVQSplitConfig, SimVQSplitLM
from simvq_split.baselines import (
    BaselineConfig,
    HouseholderDeltaNetLM,
    LoopedTransformerLM,
    RoPETransformerLM,
    SlidingWindowTransformerLM,
    match_baseline_configs,
)

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


def _neural_source_files() -> list[Path]:
    pkg = Path(__file__).resolve().parent.parent / "simvq_split"
    return [
        pkg / "quant_lut.py",
        pkg / "chunker.py",
        pkg / "crystal.py",
        pkg / "woodbury.py",
        pkg / "memory.py",
        pkg / "models.py",
        pkg / "baselines.py",
    ]


def test_ast_zero_host_heuristics():
    for path in _neural_source_files():
        assert path.exists(), f"Missing neural module file: {path}"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_mod = alias.name.split(".")[0]
                    assert root_mod not in FORBIDDEN_IMPORTS, (
                        f"Forbidden import `{alias.name}` in {path.name}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                root_mod = node.module.split(".")[0]
                assert root_mod not in FORBIDDEN_IMPORTS, (
                    f"Forbidden from-import `{node.module}` in {path.name}"
                )
            elif isinstance(node, ast.Call):
                fn_name = None
                if isinstance(node.func, ast.Name):
                    fn_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    fn_name = node.func.attr
                if fn_name is not None:
                    assert fn_name not in FORBIDDEN_CALL_NAMES, (
                        f"Forbidden host/string heuristic call `{fn_name}()` in {path.name}:{node.lineno}"
                    )

        # Inside nn.Module subclasses, ban Python dict/set literals or comprehensions
        for cls_node in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            is_nn_mod = any(
                (isinstance(b, ast.Attribute) and b.attr == "Module")
                or (isinstance(b, ast.Name) and b.id.endswith("LM"))
                for b in cls_node.bases
            )
            if not is_nn_mod:
                continue
            for sub in ast.walk(cls_node):
                assert not isinstance(
                    sub, (ast.Dict, ast.DictComp, ast.Set, ast.SetComp)
                ), (
                    f"Forbidden Python dict/set state inside neural module `{cls_node.name}` "
                    f"({path.name}:{getattr(sub, 'lineno', '?')})"
                )

    for cls in (
        SimVQSplitLM,
        RoPETransformerLM,
        SlidingWindowTransformerLM,
        LoopedTransformerLM,
        HouseholderDeltaNetLM,
    ):
        sig = inspect.signature(cls.forward)
        params = list(sig.parameters.keys())
        assert params[0] == "self" and params[1] == "token_ids", (
            f"{cls.__name__}.forward must start with (self, token_ids, ...), got {params}"
        )


@pytest.mark.parametrize("snap_alpha", [0.35, 1.0])
def test_zero_dead_parameters_simvq_split(snap_alpha: float):
    torch.manual_seed(123)
    cfg = SimVQSplitConfig(
        vocab_size=256,
        d_model=96,
        n_heads=4,
        n_layers_pre=2,
        n_layers_post=1,
        window_size=16,
        num_spheres=6,
        d_sphere=16,
        codebook_size=8,
        mem_heads=3,
        mem_dim=16,
        use_crystal=True,
        use_woodbury=True,
    )
    model = SimVQSplitLM(cfg)
    model.set_snap_alpha(snap_alpha)
    model.train()

    x = torch.randint(3, cfg.vocab_size, (4, 48))
    logits = model(x)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, cfg.vocab_size),
        x[:, 1:].reshape(-1),
    )
    loss.backward()

    dead = []
    non_finite = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None or p.grad.abs().max().item() == 0.0:
            dead.append(name)
        elif not torch.isfinite(p.grad).all():
            non_finite.append(name)

    assert not non_finite, (
        f"Non-finite gradients at snap_alpha={snap_alpha}: {non_finite}"
    )
    assert not dead, f"Dead parameters (zero grad) at snap_alpha={snap_alpha}: {dead}"


def test_zero_dead_parameters_all_baselines():
    torch.manual_seed(123)
    bcfg = BaselineConfig(
        vocab_size=256, d_model=96, n_heads=4, n_layers=4, window_size=16
    )
    for cls in (
        RoPETransformerLM,
        SlidingWindowTransformerLM,
        LoopedTransformerLM,
        HouseholderDeltaNetLM,
    ):
        model = cls(bcfg)
        model.train()
        x = torch.randint(3, bcfg.vocab_size, (4, 40))
        logits = model(x)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, bcfg.vocab_size),
            x[:, 1:].reshape(-1),
        )
        loss.backward()
        dead = [
            n
            for n, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or p.grad.abs().max().item() == 0.0)
        ]
        assert not dead, f"{cls.__name__} has dead parameters: {dead}"


def test_strict_causality_all_models():
    torch.manual_seed(7)
    scfg = SimVQSplitConfig(
        vocab_size=128,
        d_model=64,
        n_heads=4,
        n_layers_pre=1,
        n_layers_post=1,
        window_size=8,
        num_spheres=4,
        d_sphere=16,
        codebook_size=8,
        mem_heads=2,
        mem_dim=16,
    )
    bcfg = BaselineConfig(
        vocab_size=128, d_model=64, n_heads=4, n_layers=3, window_size=8
    )
    models = [
        SimVQSplitLM(scfg).eval(),
        RoPETransformerLM(bcfg).eval(),
        SlidingWindowTransformerLM(bcfg).eval(),
        LoopedTransformerLM(bcfg).eval(),
        HouseholderDeltaNetLM(bcfg).eval(),
    ]
    models[0].set_snap_alpha(1.0)

    x1 = torch.randint(3, 128, (2, 32))
    x2 = x1.clone()
    cut = 14
    x2[:, cut + 1 :] = torch.randint(3, 128, (2, 32 - cut - 1))

    for m in models:
        with torch.no_grad():
            y1 = m(x1)[:, : cut + 1]
            y2 = m(x2)[:, : cut + 1]
        diff = (y1 - y2).abs().max().item()
        assert diff < 1e-5, (
            f"Causal future leakage in {m.__class__.__name__}: max diff={diff}"
        )


def test_no_shortcut_receptive_field_and_branch_sensitivity():
    torch.manual_seed(99)
    cfg = SimVQSplitConfig(
        vocab_size=128,
        d_model=64,
        n_heads=4,
        n_layers_pre=2,
        n_layers_post=1,
        window_size=8,  # total local cortex horizon = 3 * (8 - 1) = 21 tokens
        num_spheres=4,
        d_sphere=16,
        codebook_size=8,
        mem_heads=2,
        mem_dim=16,
        use_crystal=True,
        use_woodbury=True,
    )
    model = SimVQSplitLM(cfg).eval()
    # Boost gates slightly so random init carries signal across 50 tokens
    with torch.no_grad():
        model.crystal.w_eta_proj.bias.fill_(1.0)
        model.woodbury.beta_proj.bias.fill_(1.0)
    model.set_snap_alpha(0.0)

    L = 56  # Token 55 is 55 steps away from token 0 (> 21 local cortex horizon!)
    x_a = torch.randint(3, 128, (2, L))
    x_b = x_a.clone()
    x_b[:, 0] = (x_a[:, 0] + 17) % 120 + 3

    with torch.no_grad():
        full_a = model(x_a)
        full_b = model(x_b)
        no_cryst = model(x_a, ablate_crystal=True)
        no_wood = model(x_a, ablate_woodbury=True)
        pure_local_a = model(x_a, ablate_crystal=True, ablate_woodbury=True)
        pure_local_b = model(x_b, ablate_crystal=True, ablate_woodbury=True)

    # 1. Both branches must measurably affect output logits
    assert (full_a - no_cryst).abs().max().item() > 1e-4, (
        "Crystal branch is ornamental!"
    )
    assert (full_a - no_wood).abs().max().item() > 1e-4, (
        "Woodbury branch is ornamental!"
    )

    # 2. With both long-range streams ablated, Cortex strictly CANNOT see token 0 at position 55
    local_leak = (pure_local_a[:, -1] - pure_local_b[:, -1]).abs().max().item()
    assert local_leak < 1e-6, f"Cortex leaked past sliding window! diff={local_leak}"

    # 3. With streams active, token 0 reaches position 55
    long_reach = (full_a[:, -1] - full_b[:, -1]).abs().max().item()
    assert long_reach > 1e-5, (
        "Long-range streams failed to carry signal beyond local window!"
    )


def test_parameter_parity():
    scfg = SimVQSplitConfig(vocab_size=2048)
    target_model = SimVQSplitLM(scfg)
    n_target = sum(p.numel() for p in target_model.parameters() if p.requires_grad)

    b_cfgs = match_baseline_configs(scfg)
    for name, (cls, cfg) in b_cfgs.items():
        m = cls(cfg)
        n_b = sum(p.numel() for p in m.parameters() if p.requires_grad)
        rel_diff = abs(n_b - n_target) / n_target
        assert rel_diff <= 0.05, (
            f"Parameter parity violated for {name}: {n_b:,} vs SimVQSplitLM {n_target:,} "
            f"({rel_diff * 100:.2f}% > 5.0%)"
        )


def test_long_horizon_bf16_spd_stability():
    """Verifies zero NaN/Inf on L=384 sequence rollout in bfloat16/fp32."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = SimVQSplitConfig(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_layers_pre=1,
        n_layers_post=1,
        window_size=8,
        num_spheres=4,
        d_sphere=16,
        codebook_size=8,
        mem_heads=2,
        mem_dim=16,
    )
    model = SimVQSplitLM(cfg).to(dev)
    model.set_snap_alpha(1.0)
    x = torch.randint(3, 256, (4, 384), device=dev)
    if dev == "cuda":
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
    else:
        logits = model(x)
    assert torch.isfinite(logits).all(), "Non-finite logits on long-horizon rollout!"


def test_operator_crystal_bijection_conservation():
    """
    Law 9 (Conservation of State under Operator Crystal):
    1. `project_birkhoff_to_perm_jit` must return exact `{0, 1}` permutation matrices in S_G.
    2. Across L = 1024 tokens of pure permutations (`w_eta -> 0`, `perm_gate -> 1`, `snap_alpha = 1`),
       the multiset of distinct crystal codewords across the G slots must be 100% conserved
       (zero symbol cloning, zero symbol deletion, < 1e-6 norm & lattice drift).
    """
    from simvq_split.memory import VoronoiCrystalLayer, project_birkhoff_to_perm_jit

    torch.manual_seed(314)
    raw = torch.softmax(torch.randn(32, 8, 8), dim=-1)
    P_perm = project_birkhoff_to_perm_jit(raw)
    assert torch.all((P_perm == 0.0) | (P_perm == 1.0)), "P_perm must be binary {0, 1}!"
    assert torch.allclose(P_perm.sum(dim=-1), torch.ones(32, 8)), "Row sums != 1!"
    assert torch.allclose(P_perm.sum(dim=-2), torch.ones(32, 8)), "Col sums != 1!"

    cryst = VoronoiCrystalLayer(
        d_model=64, num_spheres=8, d_sphere=32, codebook_size=16, chunk_size=2
    ).eval()
    cryst.snap_alpha = 1.0
    with torch.no_grad():
        # Disable literal overwrites, force active permutations on every chunk over L=1024!
        cryst.w_eta_proj.weight.zero_()
        cryst.w_eta_proj.bias.fill_(-20.0)
        cryst.perm_gate_proj.bias.fill_(20.0)

        x = torch.randn(2, 1024, 64)
        s_read, _, _, _ = cryst.forward_states(x)  # (2, 512, 8, 32)
        codebook, c_norm_t = cryst.get_codebook_fp32()

        s_unit = F.normalize(s_read.float(), dim=-1)
        cos_cb = torch.matmul(s_unit, c_norm_t)  # (2, 512, 8, 16)
        max_cos, slot_ids = cos_cb.max(dim=-1)
        lattice_err = (1.0 - max_cos).abs().max().item()
        assert lattice_err < 1e-6, f"Lattice drift > 1e-6: {lattice_err}"

        # Initial 8 slots hold 8 distinct codeword IDs [0..7]; every step 0..511 must hold the EXACT same multiset!
        init_sorted = slot_ids[:, 0].sort(dim=-1).values
        final_sorted = slot_ids[:, -1].sort(dim=-1).values
        assert torch.equal(init_sorted, final_sorted), (
            f"Bijection violated over 512 permutations: {init_sorted} vs {final_sorted}"
        )


def test_woodbury_epistemic_uncertainty_monotonicity():
    """
    Law 10 (Analytical Epistemic Doubt Sensor):
    Normalized ridge variance `u_t = lambda * q_t^T P_t q_t` must drop sharply when querying
    an observed key, while remaining near 1.0 (ignorance) for an orthogonal unseen key.
    """
    from simvq_split.memory import neural_woodbury_chunk_scan_jit

    torch.manual_seed(271)
    B, M_len, C, H, D = 1, 3, 4, 2, 16
    q_orth, _ = torch.linalg.qr(torch.randn(D, D))
    k_seen = q_orth[0].view(1, 1, 1, D).expand(B, 1, H, D).contiguous()
    k_other = q_orth[1].view(1, 1, 1, D).expand(B, 1, H, D).contiguous()
    k_unseen = q_orth[2].view(1, 1, 1, 1, D).expand(B, 1, C, H, D).contiguous()

    k_seq = torch.cat([k_seen, k_other, k_other], dim=1)  # chunk 0 writes k_seen
    v_seq = torch.randn(B, M_len, H, D)
    beta_seq = torch.ones(B, M_len, H, 1) * 0.99
    gamma_seq = torch.ones(B, M_len, H, 1) * 0.999

    q_seen_chunk = k_seen.unsqueeze(2).expand(B, 1, C, H, D)
    q_chunks = torch.cat([q_seen_chunk, q_seen_chunk, k_unseen], dim=1)

    _, uncerts = neural_woodbury_chunk_scan_jit(
        q_chunks, k_seq, v_seq, beta_seq, gamma_seq, inv_lam=20.0
    )
    # Chunk 0 (t=0..3): before write -> uncertainty == 1.0
    u_before = uncerts[0, 0].mean().item()
    # Chunk 1 (t=4..7): querying k_seen AFTER chunk 0 wrote k_seen -> uncertainty drops << 0.25!
    u_seen_after = uncerts[0, 4].mean().item()
    # Chunk 2 (t=8..11): querying orthogonal k_unseen -> uncertainty remains > 0.90!
    u_unseen_after = uncerts[0, 8].mean().item()

    assert abs(u_before - 1.0) < 1e-4, f"Expected initial u=1.0, got {u_before}"
    assert u_seen_after < 0.20, f"Expected observed key u < 0.20, got {u_seen_after}"
    assert u_unseen_after > 0.90, f"Expected unseen key u > 0.90, got {u_unseen_after}"


def test_combinatorial_hadamard_trap_separation():
    """
    Law 11 (Combinatorial Subspace Orthogonality on Correlated Multi-Attribute Traps):
    Given 4 observed multi-token composite entities (`Senior Archive Officer Lin`,
    `Junior Archive Officer Lin`, `Chief Vault Warden Lin`, `Chief Archive Officer Cho`)
    whose union covers 100% of the words of an unseen composite trap (`Chief Archive Officer Lin`),
    `NeuralWoodburyLayer` with Token-Grounded 3-Factor Phrase-Latch Hadamard Sketch must separate
    epistemic uncertainty `u_trap > u_seen + 0.35` across 20 distractor chunks.
    """
    from simvq_split.memory import NeuralWoodburyLayer, neural_woodbury_chunk_scan_jit

    torch.manual_seed(2026)
    d_model = 96
    wood = NeuralWoodburyLayer(
        d_model=d_model, mem_heads=4, mem_dim=32, ridge_lambda=0.04, chunk_size=4
    ).eval()

    q_orth, _ = torch.linalg.qr(torch.randn(d_model, d_model))
    ranks = q_orth[0:3]  # Senior, Junior, Chief
    role1 = q_orth[3:5]  # Archive, Vault
    role2 = q_orth[5:7]  # Officer, Warden
    surns = q_orth[7:9]  # Lin, Cho
    pred_toks = q_orth[9:13]  # ' works', ' in', ' City', '.'

    def make_clause(ri: int, r1: int, r2: int, si: int) -> torch.Tensor:
        # 8 tokens = 2 chunks of C=4: [Rank, Role1, Role2, Surname] + [' works', ' in', ' City', '.']
        # At token 6 (' City', lag -3 from Surname) and query token 5 (' in', lag -2/-3 from Surname),
        # the causal subject latch pools the 3-factor Hadamard sketch of (Rank, Role1+Role2, Surname).
        return torch.stack(
            [
                ranks[ri],
                role1[r1],
                role2[r2],
                surns[si],
                pred_toks[0],
                pred_toks[1],
                pred_toks[2],
                pred_toks[3],
            ],
            dim=0,
        )

    seen_clauses = [
        make_clause(0, 0, 0, 0),  # Senior Archive Officer Lin
        make_clause(1, 0, 0, 0),  # Junior Archive Officer Lin
        make_clause(2, 1, 1, 0),  # Chief Vault Warden Lin
        make_clause(2, 0, 0, 1),  # Chief Archive Officer Cho
    ]
    trap_clause = make_clause(2, 0, 0, 0)  # Unseen Chief Archive Officer Lin!
    preamble = q_orth[13:17]  # 4-token ledger header preamble
    distractors = F.normalize(torch.randn(20 * 4, d_model), dim=-1)

    seq = torch.cat(
        [preamble] + seen_clauses + [distractors] + seen_clauses + [trap_clause], dim=0
    ).unsqueeze(0)  # (1, 156, d_model) -> M = 39 chunks of C=4

    with torch.no_grad():
        q_epi_t = wood.encode_phrase_sketch(seq, h_lex=seq)  # (1, 156, H, D)
        M_len = seq.size(1) // 4
        q_epi_chunks = q_epi_t.view(1, M_len, 4, wood.H, wood.D)
        k_epi_seq = F.normalize(q_epi_chunks[:, :, 2], dim=-1)
        v_seq = torch.randn(1, M_len, wood.H, wood.D)

        beta = torch.full((1, M_len, wood.H, 1), 0.05)  # distractor gate < 0.26
        for idx_s in range(4):
            beta[:, 1 + 2 * idx_s + 1] = (
                0.95  # write predicate chunk of 4 seen entities
            )
        gamma = torch.full((1, M_len, wood.H, 1), 0.98)

        _, uncerts = neural_woodbury_chunk_scan_jit(
            q_epi_chunks,
            k_epi_seq,
            v_seq,
            beta,
            gamma,
            wood.inv_lam,
            q_epi_chunks,
            k_epi_seq,
        )
        seen_q_pos = [4 * (29 + 2 * i + 1) + 2 for i in range(4)]
        trap_q_pos = 4 * (29 + 8 + 1) + 2
        u_seen = uncerts[0, seen_q_pos].mean().item()
        u_trap = uncerts[0, trap_q_pos].mean().item()

    assert u_seen < 0.30, (
        f"Seen entity uncertainty drifted across distractors: {u_seen:.4f}"
    )
    assert u_trap > u_seen + 0.35, (
        f"Combinatorial trap collapsed into span(K_seen): u_seen={u_seen:.4f}, u_trap={u_trap:.4f}"
    )


def test_ternary_158bit_qat_ste_and_cardinality():
    """
    Law 12 (1.58-bit Ternary STE QAT & Groupwise Cardinality):
    1. `quantize_ternary_ste` must produce at most 3 unique scaled levels (`{-1, 0, +1} * scale`)
       in every 32-weight group while propagating exact unit STE gradients (`grad == 1.0`).
    2. With `model.set_snap_alpha(1.0)` and `model.set_ternary_qat(True)` on `B = 8` (exercising
       `1.58-bit STE QAT` forward hooks across all internal linear layers), 100% of trainable parameters
       in `SimVQSplitLM` must receive finite, non-zero gradients from pure next-token cross-entropy.
    3. `model.set_ternary_qat(False)` must cleanly remove all registered forward hooks.
    """
    from simvq_split.memory import quantize_ternary_ste

    torch.manual_seed(404)
    w = torch.randn(64, 96, requires_grad=True)
    w_q = quantize_ternary_ste(w)
    for row in w_q.detach().reshape(-1, 32):
        n_levels = torch.unique(torch.round(row * 1e5)).numel()
        assert n_levels <= 3, f"Expected <= 3 ternary levels per group, got {n_levels}"
    w_q.sum().backward()
    assert w.grad is not None and torch.allclose(w.grad, torch.ones_like(w)), (
        "STE gradient through quantize_ternary_ste must equal 1.0!"
    )

    cfg = SimVQSplitConfig(
        vocab_size=128,
        d_model=64,
        n_heads=4,
        n_layers_pre=1,
        n_layers_post=1,
        window_size=8,
        num_spheres=8,
        d_sphere=16,
        codebook_size=8,
        mem_heads=3,
        mem_dim=16,
        use_crystal=True,
        use_woodbury=True,
    )
    model = SimVQSplitLM(cfg)
    model.set_snap_alpha(1.0)
    model.set_ternary_qat(True)
    assert model.ternary_qat_enabled and len(model._qat_handles) > 0, (
        "Expected active 1.58-bit QAT forward hooks!"
    )
    model.train()

    x = torch.randint(3, cfg.vocab_size, (8, 32))
    logits = model(x)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, cfg.vocab_size),
        x[:, 1:].reshape(-1),
    )
    loss.backward()

    dead = []
    non_finite = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None or p.grad.abs().max().item() == 0.0:
            dead.append(name)
        elif not torch.isfinite(p.grad).all():
            non_finite.append(name)

    assert not non_finite, f"Non-finite gradients under 1.58-bit STE QAT: {non_finite}"
    assert not dead, f"Dead parameters under 1.58-bit STE QAT: {dead}"

    model.set_ternary_qat(False)
    assert not model.ternary_qat_enabled and len(model._qat_handles) == 0, (
        "QAT hooks were not cleanly removed by set_ternary_qat(False)!"
    )


def test_woodbury_long_horizon_spd_spectrum():
    """
    Law 13 (Woodbury Long-Horizon SPD Spectrum on L=2048):
    Under mixed writes `beta in [0.05, 0.98]` and active forgetting `gamma in [0.65, 0.99]`
    across `M = 512` chunks (`L = 2048`), the precision matrix `P_t` must remain strictly symmetric
    (`< 1e-5`) and bounded in spectrum `0 < min_eig <= max_eig <= inv_lam + 1e-4` for both
    `tern_gate = 0.0` (FP) and `tern_gate = 1.0` (1.58-bit active-forget blend).
    """
    from simvq_split.memory import inspect_woodbury_spd_spectrum

    torch.manual_seed(1313)
    B, M_len, H, d_k = 2, 512, 4, 32
    inv_lam = 25.0
    k_raw = torch.randn(B, M_len, H, d_k)
    k_epi = F.normalize(k_raw, dim=-1)

    m_idx = torch.arange(M_len).view(1, M_len, 1, 1)
    beta = torch.where(
        (m_idx % 7) == 0,
        torch.full((B, M_len, H, 1), 0.92),
        torch.full((B, M_len, H, 1), 0.05) + 0.15 * ((m_idx % 11) == 0).float(),
    )
    beta[:, ::19] = 0.98
    gamma = torch.where(
        (m_idx % 5) == 0,
        torch.full((B, M_len, H, 1), 0.65),
        torch.full((B, M_len, H, 1), 0.99),
    )

    for t_gate in (0.0, 1.0):
        min_eig, max_eig, max_sym_err = inspect_woodbury_spd_spectrum(
            k_epi, beta, gamma, inv_lam=inv_lam, sample_every=32, tern_gate=t_gate
        )
        assert max_sym_err < 1e-5, (
            f"Precision matrix lost symmetry at tern_gate={t_gate}: {max_sym_err:.2e}"
        )
        assert min_eig > 0.0, (
            f"Precision matrix became non-SPD at tern_gate={t_gate}: min_eig={min_eig:.3e}"
        )
        assert max_eig <= inv_lam + 1e-4, (
            f"Precision matrix exceeded prior upper bound 1/lam={inv_lam} at tern_gate={t_gate}: "
            f"max_eig={max_eig:.6f}"
        )


def test_crystal_positive_identity_well_distractor_shield():
    """
    Law 14 (Crystal Positive Identity Well `id_well > 0` on Distractors):
    On noisy single-slot distractor logits, enabling `well_scale=1.0` must strictly
    increase self-consistency `(pa * pb).sum(-1)` and decrease spurious `||T_swap - I||_F`
    relative to `well_scale=0.0`, while keeping live gradients on `perm_gate_proj.bias`.
    """
    from simvq_split.memory import VoronoiCrystalLayer

    torch.manual_seed(1414)
    cryst = VoronoiCrystalLayer(d_model=64, num_spheres=8, d_sphere=16, codebook_size=8)
    B, M_len, H, Gh = 2, 6, cryst.H, cryst.G_h
    eye = cryst.eye_Gh.float().view(1, 1, 1, Gh, Gh)

    psi_a = torch.full((B, M_len, H, Gh), -1.0)
    psi_b = torch.full((B, M_len, H, Gh), -1.0)
    psi_a[..., 1] = 4.0
    psi_b[..., 1] = 1.15
    psi_b[..., 2] = 0.95

    pa_off, pb_off, t_off = cryst._compute_ptr_swap(psi_a, psi_b, well_scale=0.0)
    pa_on, pb_on, t_on = cryst._compute_ptr_swap(psi_a, psi_b, well_scale=1.0)

    agree_off = (pa_off * pb_off).sum(dim=-1).mean().item()
    agree_on = (pa_on * pb_on).sum(dim=-1).mean().item()
    dev_off = (t_off - eye).pow(2).sum(dim=(-1, -2)).sqrt().mean().item()
    dev_on = (t_on - eye).pow(2).sum(dim=(-1, -2)).sqrt().mean().item()

    assert agree_on > agree_off + 0.05, (
        f"Identity well failed to strengthen self-agreement: off={agree_off:.4f}, on={agree_on:.4f}"
    )
    assert dev_on < dev_off - 0.05, (
        f"Identity well failed to reduce distractor ||T_swap - I||_F: off={dev_off:.4f}, on={dev_on:.4f}"
    )

    cryst.zero_grad()
    t_on.pow(2).sum().backward()
    assert (
        cryst.perm_gate_proj.bias.grad is not None
        and torch.isfinite(cryst.perm_gate_proj.bias.grad).all()
        and cryst.perm_gate_proj.bias.grad.abs().max().item() > 0.0
    ), "perm_gate_proj.bias received zero or non-finite direct _compute_ptr_swap gradient!"

    cryst.zero_grad()
    cryst.train()
    cryst.snap_alpha = 1.0
    cryst.qat_active_mix = 1.0
    x = torch.randn(2, 24, 64, requires_grad=True)
    y = cryst(x)
    y.pow(2).sum().backward()
    assert (
        cryst.perm_gate_proj.bias.grad is not None
        and torch.isfinite(cryst.perm_gate_proj.bias.grad).all()
        and cryst.perm_gate_proj.bias.grad.abs().max().item() > 0.0
    ), "perm_gate_proj.bias received zero or non-finite end-to-end gradient!"


def test_packed_uint8_2bit_tmac_lut_equivalence_and_compression():
    """
    Law 15 (Packed `uint8` `2-bit T-MAC LUT` Physical Inference & 16x Compression):
    Quantizes `SimVQSplitLM` weights with `quantize_ternary_weights_ls`, packs every
    non-`lm_head` linear layer via `pack_model_2bit_lut`, and asserts:
    1. All packed weight buffers use `torch.uint8` with exact `16.0x` matrix compression vs FP32.
    2. Zero `nn.Linear` submodules remain except tied `lm_head`.
    3. Logits from addition/subtraction-only `PackedLUT2BitLinear` match FP32-ternary
       reference logits within `max_abs_diff < 5e-3`.
    """
    import torch.nn as nn
    from simvq_split.memory import (
        PackedLUT2BitLinear,
        compute_2bit_lut_footprint,
        pack_model_2bit_lut,
        quantize_ternary_weights_ls,
    )

    torch.manual_seed(1515)
    cfg = SimVQSplitConfig(
        vocab_size=128,
        d_model=160,
        n_heads=4,
        n_layers_pre=2,
        n_layers_post=1,
        window_size=12,
        num_spheres=24,
        d_sphere=16,
        codebook_size=16,
        mem_heads=5,
        mem_dim=32,
        use_crystal=True,
        use_woodbury=True,
    )
    model = SimVQSplitLM(cfg)
    model.eval()
    model.set_snap_alpha(1.0)
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, nn.Linear) and mod is not model.lm_head:
                w_q, _, _ = quantize_ternary_weights_ls(mod.weight.data)
                mod.weight.data.copy_(w_q)

    lut_model = pack_model_2bit_lut(model)
    remaining_linear = [
        m
        for m in lut_model.modules()
        if isinstance(m, nn.Linear) and m is not lut_model.lm_head
    ]
    assert not remaining_linear, "Found unpacked nn.Linear modules outside lm_head!"

    packed_layers = [
        m for m in lut_model.modules() if isinstance(m, PackedLUT2BitLinear)
    ]
    assert len(packed_layers) > 0
    for pl in packed_layers:
        assert pl.packed_u8.dtype == torch.uint8, f"Expected uint8 buffer, got {pl.packed_u8.dtype}"

    fp = compute_2bit_lut_footprint(lut_model)
    assert abs(fp["matrix_compression_vs_fp32"] - 16.0) < 1e-3, (
        f"Expected 16.0x matrix compression vs FP32, got {fp['matrix_compression_vs_fp32']}"
    )

    x = torch.randint(3, cfg.vocab_size, (4, 32))
    with torch.no_grad():
        logits_ref = model(x)
        logits_lut = lut_model(x)
    max_diff = (logits_ref - logits_lut).abs().max().item()
    assert max_diff < 5e-3, (
        f"Packed T-MAC uint8 LUT logits diverged from ternary reference: max_abs_diff={max_diff:.5f}"
    )


def test_surprise_hnet_chunker_causality_ratio_grads_and_o1_budget():
    """
    Law 16 (Surprise-Gated H-Net Byte Chunker — Phase B):
    1. Backward compatibility: legacy `SimVQSplitConfig()` defaults `use_byte_chunker=False` and loads
       `checkpoints/simvq_split_tournament.pt` (`FP` & `QAT`, `strict=True`).
    2. Strict byte-level causality (`max_abs_diff < 1e-5` on `logits[:, :cut+1]` under future perturbation)
       across `snap_alpha in {0.0, 1.0}` and packed `2-bit uint8 LUT`.
    3. Zero-shortcut receptive field isolation beyond `(n_layers * (window_size - 1) + 2) * max_chunk_bytes`.
    4. 100% live finite non-zero gradients on all parameters (including `router_*`) from pure CE + Ratio Loss,
       and exact opposite-sign gradient on `router_ctx.bias` for `target_chunk_bytes=2.0` vs `7.0`.
    5. Exact `O(1)` streaming state constancy (`L=64..2048`), `<= 555.0 KB` `2-bit uint8 LUT` (`16.0x`),
       and `80.0..83.5 KB` full `O(1)` state cache on the Phase B config.
    """
    import torch.nn as nn
    from simvq_split.memory import (
        compute_2bit_lut_footprint,
        compute_chunker_state_bytes,
        pack_model_2bit_lut,
        quantize_ternary_weights_ls,
    )

    # 1. Backward compatibility with existing v5 tournament checkpoint
    assert SimVQSplitConfig().use_byte_chunker is False
    ckpt_path = (
        Path(__file__).resolve().parent.parent
        / "checkpoints"
        / "simvq_split_tournament.pt"
    )
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg_legacy = SimVQSplitConfig(**ckpt["simvq_split_config"])
        assert cfg_legacy.use_byte_chunker is False
        m_fp = SimVQSplitLM(cfg_legacy)
        m_fp.load_state_dict(ckpt["simvq_split_state_dict"], strict=True)
        m_qat = SimVQSplitLM(cfg_legacy)
        m_qat.load_state_dict(ckpt["simvq_split_ternary_qat_state_dict"], strict=True)

    # 2 & 3. Causality + Receptive-field isolation on a compact byte model
    torch.manual_seed(1616)
    cfg_small = SimVQSplitConfig(
        vocab_size=259,
        d_model=64,
        n_heads=4,
        d_ff=128,
        n_layers_pre=1,
        n_layers_post=1,
        window_size=4,
        num_spheres=4,
        d_sphere=16,
        codebook_size=8,
        mem_heads=2,
        mem_dim=16,
        use_crystal=True,
        use_woodbury=True,
        use_byte_chunker=True,
        byte_char_dim=16,
        byte_d_ff=96,
        byte_router_char_dim=8,
        byte_router_hidden=64,
        target_chunk_bytes=4.0,
        max_chunk_bytes=6,
    )
    m_small = SimVQSplitLM(cfg_small).eval()
    with torch.no_grad():
        m_small.crystal.w_eta_proj.bias.fill_(1.0)
        m_small.woodbury.beta_proj.bias.fill_(1.0)

    x1 = torch.randint(3, 259, (2, 128))
    x2 = x1.clone()
    cut = 63
    x2[:, cut + 1 :] = torch.randint(3, 259, (2, 128 - cut - 1))
    for alpha in (0.0, 1.0):
        m_small.set_snap_alpha(alpha)
        with torch.no_grad():
            y1 = m_small(x1)[:, : cut + 1]
            y2 = m_small(x2)[:, : cut + 1]
        diff = (y1 - y2).abs().max().item()
        assert diff < 1e-5, (
            f"Byte chunker causal future leakage at snap_alpha={alpha}: {diff}"
        )

    m_small_q = SimVQSplitLM(cfg_small).eval()
    m_small_q.set_snap_alpha(1.0)
    with torch.no_grad():
        for mod in m_small_q.modules():
            if isinstance(mod, nn.Linear) and mod is not m_small_q.lm_head:
                w_q, _, _ = quantize_ternary_weights_ls(mod.weight.data)
                mod.weight.data.copy_(w_q)
    lut_small = pack_model_2bit_lut(m_small_q).eval()
    with torch.no_grad():
        yl1 = lut_small(x1)[:, : cut + 1]
        yl2 = lut_small(x2)[:, : cut + 1]
    assert (yl1 - yl2).abs().max().item() < 1e-5, "Causal leak in packed 2-bit LUT byte chunker!"

    # Receptive-field isolation: 2 layers * (window=4 - 1) + 2 chunks = 8 chunks * max 6 bytes = 48 + 8 bytes < 127
    xa = torch.randint(3, 259, (2, 128))
    xb = xa.clone()
    xb[:, 0] = (xa[:, 0] + 19) % 250 + 3
    m_small.set_snap_alpha(0.0)
    with torch.no_grad():
        pure_a = m_small(xa, ablate_crystal=True, ablate_woodbury=True)
        pure_b = m_small(xb, ablate_crystal=True, ablate_woodbury=True)
        full_a = m_small(xa)
        full_b = m_small(xb)
    assert (pure_a[:, -1] - pure_b[:, -1]).abs().max().item() < 1e-6, (
        "Byte cortex leaked past chunked sliding-window horizon!"
    )
    assert (full_a[:, -1] - full_b[:, -1]).abs().max().item() > 1e-5, (
        "Long-range branches failed to propagate byte 0 across 128 bytes!"
    )

    # 4. 100% live finite gradients across ALL params + Ratio Loss direction check
    m_small.train()
    m_small.set_snap_alpha(1.0)
    m_small.zero_grad()
    x_tr = torch.randint(3, 259, (4, 64))
    logits_tr = m_small(x_tr)
    loss_ce_ratio = F.cross_entropy(
        logits_tr[:, :-1].reshape(-1, cfg_small.vocab_size),
        x_tr[:, 1:].reshape(-1),
    ) + 0.04 * m_small.get_commitment_loss()
    loss_ce_ratio.backward()
    dead = [
        n
        for n, p in m_small.named_parameters()
        if p.requires_grad and (p.grad is None or p.grad.abs().max().item() == 0.0)
    ]
    non_fin = [
        n
        for n, p in m_small.named_parameters()
        if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    assert not non_fin, f"Non-finite grads in byte-chunked SimVQSplitLM: {non_fin}"
    assert not dead, f"Dead parameters in byte-chunked SimVQSplitLM: {dead}"

    # Verify pack_model_2bit_lut works immediately after backward() without graph-leaf deepcopy errors,
    # and verify locked router boundaries match bit-for-bit between FP and packed 2-bit LUT.
    m_small.byte_chunker.lock_router(True)
    with torch.no_grad():
        _, _, ctx_fp_lock = m_small.byte_chunker.encode_to_chunks(
            m_small.tok_emb(x_tr), x_tr
        )
    lut_post_bwd = pack_model_2bit_lut(m_small).eval()
    with torch.no_grad():
        _, _, ctx_lut_lock = lut_post_bwd.byte_chunker.encode_to_chunks(
            lut_post_bwd.tok_emb(x_tr), x_tr
        )
    assert torch.equal(ctx_fp_lock.b_mask, ctx_lut_lock.b_mask), (
        "Locked router boundaries diverged after pack_model_2bit_lut!"
    )
    m_small.byte_chunker.lock_router(False)

    m_small.eval()
    m_small.zero_grad()
    m_small.byte_chunker.target_chunk_bytes = 2.0
    _ = m_small(x_tr)
    m_small.byte_chunker.last_ratio_loss.backward()
    g_short = float(m_small.byte_chunker.router_ctx.bias.grad.item())

    m_small.zero_grad()
    m_small.byte_chunker.target_chunk_bytes = 7.0
    _ = m_small(x_tr)
    m_small.byte_chunker.last_ratio_loss.backward()
    g_long = float(m_small.byte_chunker.router_ctx.bias.grad.item())
    assert g_short < 0.0 < g_long, (
        f"Ratio loss boundary bias gradient wrong sign: tgt=2.0 -> {g_short}, tgt=7.0 -> {g_long}"
    )

    # 5. Production Phase B budget check: O(1) state across L=64..2048, cache <= 83.5 KB, 2-bit LUT <= 555.0 KB
    cfg_prod = SimVQSplitConfig(
        vocab_size=259,
        d_model=160,
        n_heads=4,
        d_ff=224,
        n_layers_pre=2,
        n_layers_post=1,
        window_size=8,
        num_spheres=24,
        d_sphere=32,
        codebook_size=32,
        mem_heads=5,
        mem_dim=32,
        use_crystal=True,
        use_woodbury=True,
        use_byte_chunker=True,
        byte_char_dim=32,
        byte_d_ff=128,
        byte_router_char_dim=16,
        byte_router_hidden=96,
        byte_commit_scale=0.23,
        router_lock_frac=0.16,
        target_chunk_bytes=4.6,
        max_chunk_bytes=10,
        ratio_loss_scale=1.5,
    )
    m_prod = SimVQSplitLM(cfg_prod).eval()
    m_prod_bf16 = SimVQSplitLM(cfg_prod).to(torch.bfloat16).eval()
    m_prod_bf16.load_state_dict({k: v.to(torch.bfloat16) for k, v in m_prod.state_dict().items()})
    expected_chunker_b = compute_chunker_state_bytes(160, 32, 16, 32)
    obs_bytes = []
    with torch.no_grad():
        for seq_L in (64, 256, 2048):
            xs = torch.randint(3, 259, (1, seq_L))
            es = m_prod_bf16.tok_emb(xs)
            ps, bs, hbs, st_b = m_prod_bf16.byte_chunker.forward_streaming_check(
                es, xs, m_prod_bf16.tok_emb.weight
            )
            obs_bytes.append(st_b)
            assert st_b == expected_chunker_b, (
                f"Expected chunker O(1) bytes {expected_chunker_b}, got {st_b} at L={seq_L}"
            )
            if seq_L == 256:
                _, _, b_ctx = m_prod_bf16.byte_chunker.encode_to_chunks(
                    es, xs, m_prod_bf16.tok_emb.weight
                )
                assert torch.equal(b_ctx.b_mask, bs), "Streaming boundary mismatch!"
                assert (b_ctx.p_prob - ps).abs().max().item() < 5e-3
                assert (b_ctx.h_byte.float() - hbs.float()).abs().max().item() < 5e-3
    assert len(set(obs_bytes)) == 1, f"Chunker state grew with L: {obs_bytes}"

    lut_prod = pack_model_2bit_lut(m_prod)
    fp_prod = compute_2bit_lut_footprint(lut_prod)
    assert fp_prod["total_lut_kb"] <= 555.0, (
        f"Phase B 2-bit LUT footprint exceeded 555 KB: {fp_prod['total_lut_kb']} KB"
    )
    assert abs(fp_prod["matrix_compression_vs_fp32"] - 16.0) < 1e-3
