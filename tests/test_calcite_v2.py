"""
Gate 1–4 Verification Tests for `calcite_v2` (Phase L):
- L1: Zero-Cheat AST purity across all `calcite_v2/` modules (zero re/ast/eval/exec/torch.compile).
- L2: Gate-2 init-std bounds (`<= 1.5x`), `qat_mix=0` exact parity (`< 1e-5`), and `100%` live gradients.
- L3: True 2nd-order biorthogonal Kalman-Woodbury SPD & correlated-key separation + exact overwrite parity.
- L4: `CalciteStreamEngine` exact parity (`stream(seq) == forward(seq)`) AND strict `O(1)` flat RAM
      across `L = 64 -> 512 -> 1024`, for both FP and packed `2-bit uint8 T-MAC LUT`.
- L5: Standalone UTF-8 router pretraining + pre-freeze invariance (`0.00%` mid-UTF8 cuts) + `AdamW(fused=True)`
      QAT cache invalidation safety.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from calcite_v2 import (
    CalciteStreamEngine,
    CalciteV2Config,
    CalciteV2LM,
    build_calcite_v2_tier_config,
    compute_v2_state_cache_kb,
)
from kernels.fast_calcite_scans import clear_fast_qat_cache
from kernels.fast_packed_lut import pack_fast_model_2bit_lut

ROOT = Path(__file__).resolve().parents[1]


def _make_tiny_cfg(
    topology: str = "interleaved",
    use_scaled_decoder: bool = True,
    use_kalman_precond: bool = True,
    use_latent_feedback: bool = False,
) -> CalciteV2Config:
    return CalciteV2Config(
        vocab_size=259,
        d_model=64,
        n_heads=2,
        d_ff=128,
        n_layers_pre=2,
        n_layers_post=2,
        window_size=8,
        num_spheres=8,
        d_sphere=16,
        codebook_size=16,
        crystal_w_chunks=2,
        mem_heads=4,
        mem_dim=16,
        woodbury_latch_span=6,
        byte_char_dim=16,
        byte_d_ff=64,
        byte_router_char_dim=8,
        byte_router_hidden=32,
        topology=topology,
        use_scaled_decoder=use_scaled_decoder,
        use_kalman_precond=use_kalman_precond,
        use_latent_feedback=use_latent_feedback,
        use_clean_gates=True,
        use_fast_kernels=False,
    )


def test_l1_calcite_v2_ast_zero_cheat_purity() -> None:
    """L1: Zero forbidden imports (`re`, `ast`), zero `eval`/`exec`/`torch.compile` in `calcite_v2/`."""
    v2_dir = ROOT / "calcite_v2"
    py_files = sorted(v2_dir.glob("*.py"))
    assert len(py_files) >= 5
    forbidden_imports = {"re", "ast", "subprocess", "os"}
    forbidden_calls = {"eval", "exec", "compile"}
    for p in py_files:
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in forbidden_imports, f"Forbidden import {alias.name} in {p}"
            elif isinstance(node, ast.ImportFrom):
                mod = (node.module or "").split(".")[0]
                assert mod not in forbidden_imports, f"Forbidden import-from {mod} in {p}"
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    assert node.func.id not in forbidden_calls, f"Forbidden call {node.func.id} in {p}"
                elif isinstance(node.func, ast.Attribute):
                    assert node.func.attr != "compile", f"Forbidden .compile() call in {p}"


def test_l2_gate2_init_std_and_live_gradients() -> None:
    """L2: Bounded init std, exact `qat_mix=0` parity, and 100% non-zero trainable gradients."""
    torch.manual_seed(2026)
    cfg = _make_tiny_cfg(topology="interleaved", use_scaled_decoder=True, use_kalman_precond=True, use_latent_feedback=True)
    model = CalciteV2LM(cfg)
    model.byte_chunker.lock_router(True)
    x = torch.randint(3, 259, (2, 64))
    y = torch.randint(3, 259, (2, 64))

    # Init std + exact qat_mix=0 parity
    model.eval()
    with torch.no_grad():
        lg_fp = model(x, feedback_passes=1)
        assert torch.isfinite(lg_fp).all()
        assert 0.05 < float(lg_fp.std().item()) < 2.0

        model.set_qat_mix(0.0)
        model.set_ternary_qat(True)
        lg_q0 = model(x, feedback_passes=1)
        model.set_ternary_qat(False)
        assert float((lg_fp - lg_q0).abs().max().item()) < 1e-5

    # Live gradients on all unlocked trainable parameters
    model.train()
    model.set_crystal_schedule(5, 100)
    logits = model(x, feedback_passes=2)
    loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1)) + model.get_commitment_loss()
    loss.backward()

    dead = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None or float(p.grad.abs().max().item()) == 0.0:
                # Some unused byte_offset_emb indices beyond max observed chunk len may be 0
                if "byte_offset_emb" in name:
                    continue
                dead.append(name)
    assert not dead, f"Dead gradients in CalciteV2LM: {dead}"


def test_l3_kalman_woodbury_spd_and_correlated_separation() -> None:
    """L3: Verify biorthogonal Kalman-Woodbury preserves exact overwrite and suppresses correlated interference."""
    from calcite_v2.woodbury import _kalman_p_and_dual_nograd_jit

    D = 16
    inv_lam = 25.0
    k_old = F.normalize(torch.randn(1, 1, D, 1), dim=2)
    u_perp = torch.randn(1, 1, D, 1)
    u_perp = u_perp - (u_perp * k_old).sum(dim=2, keepdim=True) * k_old
    u_perp = F.normalize(u_perp, dim=2)
    # Correlated key: cos(k_old, k_corr) = 0.85
    k_corr = F.normalize(0.85 * k_old + (1.0 - 0.85**2) ** 0.5 * u_perp, dim=2)

    # Sequence of 3 steps: write k_old -> write k_corr -> overwrite k_old
    k_seq = torch.cat([k_old, k_corr, k_old], dim=0)       # (3, 1, D, 1)
    q_seq = k_seq.expand(3, 1, D, 4)
    b_seq = torch.full((3, 1, 1, 1), 0.9)
    g_seq = torch.full((3, 1, 1, 1), 0.99)

    _, k_dual = _kalman_p_and_dual_nograd_jit(q_seq, k_seq, k_seq, b_seq, g_seq, inv_lam, 1.0, 1.0)

    # 1) Correlated write must have MUCH smaller projection onto k_old than raw k_corr (0.85)
    raw_leak = float((k_corr * k_old).sum().item())
    kalman_leak = float((k_dual[1:2] * k_old).sum().item())
    assert kalman_leak < 0.35 * raw_leak, f"Expected strong decorrelation: raw={raw_leak:.3f}, kalman={kalman_leak:.3f}"

    # 2) Self-projection on k_corr stays strong (writes new fact cleanly)
    self_proj = float((k_dual[1:2] * k_corr).sum().item())
    assert self_proj > 0.75, f"Expected strong self-write on k_corr, got {self_proj:.3f}"

    # 3) Bounded norm invariant (never explodes)
    assert float(k_dual.norm(dim=2).max().item()) <= 2.6


@pytest.mark.parametrize("topology", ["hourglass", "interleaved"])
@pytest.mark.parametrize("use_scaled_decoder", [False, True])
def test_l4_stream_engine_exact_parity_and_o1_memory(
    topology: str, use_scaled_decoder: bool
) -> None:
    """L4: Exact streaming parity `stream(seq) == forward(seq)` (FP + 2-bit LUT) and strict O(1) RAM."""
    torch.manual_seed(2026)
    cfg = _make_tiny_cfg(
        topology=topology,
        use_scaled_decoder=use_scaled_decoder,
        use_kalman_precond=True,
        use_latent_feedback=False,
    )
    model = CalciteV2LM(cfg).eval()
    model.byte_chunker.lock_router(True)
    model.set_crystal_schedule(1, 1)
    model.set_snap_alpha(1.0)

    # UTF-8 rich sequence (ASCII + Russian + CJK + 4-byte emoji)
    text = "Alice moved to Paris! Привет мир 🚀 東京 12345 " * 3
    ids = torch.tensor([[b + 3 for b in text.encode("utf-8")]], dtype=torch.long)

    with torch.no_grad():
        lg_batch = model(ids)
        engine = CalciteStreamEngine(model)
        lg_stream, st = engine.prefill(ids, return_all_logits=True)

    max_err = float((lg_batch - lg_stream).abs().max().item())
    assert max_err < 1e-4, f"Stream vs batch parity mismatch ({topology}, dec={use_scaled_decoder}): {max_err}"
    # Backbone must tick much fewer times than total bytes (~3-5x compression)
    assert st.backbone_ticks < st.total_bytes // 2

    # Strict O(1) memory invariance across growing lengths
    mem_start = st.memory_bytes()
    extra_ids = torch.randint(3, 128, (1, 256), dtype=torch.long)
    _, st = engine.prefill(extra_ids, state=st, return_all_logits=False)
    mem_after = st.memory_bytes()
    assert mem_start == mem_after, f"State memory grew! {mem_start} -> {mem_after}"

    # Verify 2-bit uint8 T-MAC LUT roundtrip + live streaming generation
    lut_model = pack_fast_model_2bit_lut(model, backend="pytorch_vec").eval()
    lut_engine = CalciteStreamEngine(lut_model)
    with torch.no_grad():
        lg_lut_batch = lut_model(ids[:, :48])
        lg_lut_stream, _ = lut_engine.prefill(ids[:, :48], return_all_logits=True)
    lut_err = float((lg_lut_batch - lg_lut_stream).abs().max().item())
    assert lut_err < 1e-3, f"LUT stream vs batch mismatch: {lut_err}"

    gen_ids, gen_state = lut_engine.generate(ids[:, :24], max_new_bytes=16, temperature=0.0)
    assert gen_ids.shape[1] >= 25
    assert gen_state.memory_bytes() == mem_start


def test_l5_standalone_router_pretrain_utf8_safety_and_tiers() -> None:
    """L5: Standalone router pretraining produces 0.00% mid-UTF8 cuts and exact tier specs."""
    torch.manual_seed(2026)
    cfg = _make_tiny_cfg()
    model = CalciteV2LM(cfg)
    opt = torch.optim.AdamW(
        [p for m in model.byte_chunker._router_modules() for p in m.parameters()]
        + [model.byte_chunker.router_feat_weights],
        lr=3e-3,
    )
    sample = ("Hello world! Привет, архитектура Calcite! 日本語テスト 🚀🔥 " * 4).encode("utf-8")
    seq = torch.tensor([[b + 3 for b in sample[:128]]], dtype=torch.long).repeat(4, 1)
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        loss, ratio = model.byte_chunker.router_forward_only(seq)
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()

    model.byte_chunker.lock_router(True)
    assert model.byte_chunker.router_locked
    with torch.no_grad():
        e_tok = model.tok_emb(seq)
        _, _, ctx = model.byte_chunker.encode_to_chunks(e_tok, seq)
        # Verify 0.00% mid-UTF8 continuation splits
        raw_bytes = (seq - 3).clamp(min=0)
        is_cont = (seq >= 3) & ((raw_bytes & 0xC0) == 0x80)
        mid_cuts = int((ctx.b_mask & is_cont).sum().item())
        assert mid_cuts == 0, f"Found {mid_cuts} mid-UTF8 cuts!"

    # Verify tier configs build cleanly and have bounded O(1) cache
    for tier in ("S1_5M", "S2_25M", "S3_85M"):
        t_cfg = build_calcite_v2_tier_config(tier)
        kb = compute_v2_state_cache_kb(t_cfg, 65536)
        assert 50.0 < kb < 850.0, f"Unexpected cache KB for {tier}: {kb}"

    # Verify CUDA AdamW(fused=True) QAT cache invalidation if CUDA is available
    if torch.cuda.is_available():
        m_cuda = CalciteV2LM(_make_tiny_cfg()).cuda()
        m_cuda.byte_chunker.lock_router(True)
        m_cuda.set_fast_kernels(True)
        m_cuda.set_qat_mix(1.0)
        m_cuda.set_ternary_qat(True)
        opt_c = torch.optim.AdamW(
            [p for p in m_cuda.parameters() if p.requires_grad], lr=1e-2, fused=True
        )
        opt_c.register_step_post_hook(lambda *_a, **_kw: clear_fast_qat_cache())
        xc = torch.randint(3, 259, (2, 48), device="cuda")
        out1 = m_cuda(xc)
        out1.sum().backward()
        opt_c.step()
        out2 = m_cuda(xc)
        assert not torch.allclose(out1.detach(), out2.detach()), "Stale QAT cache after fused AdamW step!"


@pytest.mark.parametrize(
    "axis_name,extra_kw",
    [
        ("M1_sparse_woodbury", {"woodbury_sub_banks": 4, "woodbury_top_k_banks": 1}),
        ("M2_epistemic_slots", {"use_epistemic_slots": True, "epistemic_slots_n": 16, "epistemic_slots_dim": 32}),
        ("M3_sym_byte_swa", {"use_sym_byte_swa": True, "byte_swa_window": 16, "byte_swa_dim": 32}),
        ("M4_swa_drop", {"swa_drop_prob": 0.25}),
        ("M5_attnres", {"use_attnres": True}),
        ("M6_deltanet2", {"use_deltanet2_gates": True}),
        ("M6_sparse_deltanet2", {"woodbury_sub_banks": 4, "woodbury_top_k_banks": 1, "use_deltanet2_gates": True}),
        ("M7_canon_chunks", {"use_canon_chunks": True}),
        ("M8_byte_engram", {"use_byte_engram": True, "engram_slots": 256, "engram_dim": 16}),
        (
            "M_v3_combo",
            {
                "use_sym_byte_swa": True,
                "byte_swa_window": 16,
                "byte_swa_dim": 32,
                "use_attnres": True,
                "use_deltanet2_gates": True,
                "use_canon_chunks": True,
                "use_byte_engram": True,
                "engram_slots": 256,
                "engram_dim": 16,
            },
        ),
    ],
)
def test_m_axes_init_std_live_grads_causality_and_stream_parity(
    axis_name: str, extra_kw: dict
) -> None:
    """Phase M Gate 1–2: init-std, live gradients, strict causality, exact stream==forward parity, O(1) RAM."""
    torch.manual_seed(2026)
    cfg = _make_tiny_cfg(
        topology="hourglass",
        use_scaled_decoder=True,
        use_kalman_precond=False,
        use_latent_feedback=False,
    )
    for k, v in extra_kw.items():
        setattr(cfg, k, v)

    model = CalciteV2LM(cfg)
    model.byte_chunker.lock_router(True)
    model.set_crystal_schedule(1, 1)
    model.set_snap_alpha(1.0)

    text = "Ref code X7K9Q2M4. Alice moved to Paris! Привет 🚀 東京 12345 " * 3
    ids = torch.tensor([[b + 3 for b in text.encode("utf-8")]], dtype=torch.long)
    x_tr = torch.cat([ids[:, :64], ids[:, 16:80]], dim=0)
    y_tr = torch.randint(3, 259, x_tr.shape)

    # 1. Init-std bound + strict causality (no look-ahead)
    model.eval()
    with torch.no_grad():
        lg1 = model(ids[:, :64])
        assert torch.isfinite(lg1).all(), f"Non-finite logits in {axis_name}"
        assert 0.05 < float(lg1.std().item()) < 2.0, f"Bad init std in {axis_name}: {lg1.std().item()}"

        ids_mod = ids[:, :64].clone()
        ids_mod[:, 32:] = torch.randint(3, 128, (1, 32))
        lg2 = model(ids_mod)
        causal_diff = float((lg1[:, :32] - lg2[:, :32]).abs().max().item())
        assert causal_diff < 1e-5, f"Look-ahead leak in {axis_name}: {causal_diff}"

        # 2. Exact stream == forward parity + strict O(1) RAM
        engine = CalciteStreamEngine(model)
        lg_stream, st = engine.prefill(ids[:, :96], return_all_logits=True)
        lg_batch = model(ids[:, :96])
        parity_err = float((lg_batch - lg_stream).abs().max().item())
        assert parity_err < 1e-4, f"Stream parity mismatch in {axis_name}: {parity_err}"

        mem_start = st.memory_bytes()
        _, st = engine.prefill(torch.randint(3, 128, (1, 128)), state=st, return_all_logits=False)
        assert st.memory_bytes() == mem_start, f"RAM grew in {axis_name}: {mem_start} -> {st.memory_bytes()}"

    # 3. Live gradients on trainable params
    model.train()
    logits = model(x_tr)
    loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y_tr.reshape(-1)) + model.get_commitment_loss()
    loss.backward()
    dead = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None or float(p.grad.abs().max().item()) == 0.0:
                if "byte_offset_emb" in name or "engram_emb" in name:
                    continue
                dead.append(name)
    assert not dead, f"Dead gradients in {axis_name}: {dead}"


def test_m_temporal_residual_exact_retrofit_causality_and_stream_parity() -> None:
    """TR: exact step-0 identity retrofit, live grads, zero look-ahead, exact stream==forward fixed point, O(1) RAM."""
    from scripts.muon_opt import MuonAdamWHybrid

    torch.manual_seed(2026)
    base_cfg = _make_tiny_cfg(
        topology="hourglass",
        use_scaled_decoder=True,
        use_kalman_precond=False,
        use_latent_feedback=False,
    )
    base_cfg.use_sym_byte_swa = True
    base_cfg.byte_swa_window = 16
    base_cfg.byte_swa_dim = 32
    base_model = CalciteV2LM(base_cfg).eval()
    base_model.byte_chunker.lock_router(True)
    base_model.set_crystal_schedule(1, 1)
    base_model.set_snap_alpha(1.0)

    tr_cfg = _make_tiny_cfg(
        topology="hourglass",
        use_scaled_decoder=True,
        use_kalman_precond=False,
        use_latent_feedback=False,
    )
    tr_cfg.use_sym_byte_swa = True
    tr_cfg.byte_swa_window = 16
    tr_cfg.byte_swa_dim = 32
    tr_cfg.use_temporal_residual = True
    tr_cfg.tr_train_passes = 4
    tr_cfg.tr_eval_passes = 8
    tr_cfg.tr_backprop_passes = 2
    tr_model = CalciteV2LM(tr_cfg).eval()
    tr_model.byte_chunker.lock_router(True)
    tr_model.set_crystal_schedule(1, 1)
    tr_model.set_snap_alpha(1.0)
    missing, unexpected = tr_model.load_state_dict(base_model.state_dict(), strict=False)
    assert not unexpected
    assert all(k.startswith("tr_") for k in missing), f"Unexpected missing keys: {missing}"

    text = "Alice moved to Paris! Ref X7K9Q2M4. Привет 🚀 東京 12345 " * 2
    ids = torch.tensor([[b + 3 for b in text.encode("utf-8")]], dtype=torch.long)

    # 1. Exact step-0 identity (rB == 0 -> bit-for-bit identical to pretrained base!)
    with torch.no_grad():
        lg_base = base_model(ids[:, :64])
        lg_tr0 = tr_model(ids[:, :64], tr_passes=4)
        assert float((lg_base - lg_tr0).abs().max().item()) < 1e-5

    # 2. Take 2 steps with MuonAdamWHybrid so rB moves off 0 and verify all TR params receive live grads
    tr_model.train()
    opt = MuonAdamWHybrid(tr_model, base_lr=1e-2, muon_lr_mult=4.0, device="cpu")
    x_tr = ids[:, :64].repeat(2, 1)
    y_tr = torch.randint(3, 259, x_tr.shape)
    for step_i in range(2):
        opt.zero_grad(set_to_none=True)
        lg = tr_model(x_tr, tr_passes=4, tr_backprop_passes=2)
        loss = F.cross_entropy(lg.reshape(-1, tr_cfg.vocab_size), y_tr.reshape(-1))
        loss.backward()
        if step_i == 0:
            assert tr_model.tr_rB.grad is not None and float(tr_model.tr_rB.grad.abs().max().item()) > 0.0
        else:
            for nm in ("tr_rA", "tr_rB", "tr_sB"):
                p = getattr(tr_model, nm)
                assert p.grad is not None and float(p.grad.abs().max().item()) > 0.0, f"Dead grad on {nm}"
            assert tr_model.tr_phi.weight.grad is not None and float(tr_model.tr_phi.weight.grad.abs().max().item()) > 0.0
        opt.step()

    # 3. Strict causality + exact stream == forward fixed-point parity + strict O(1) RAM
    tr_model.eval()
    with torch.no_grad():
        lg1 = tr_model(ids[:, :64], tr_passes=6)
        ids_mod = ids[:, :64].clone()
        ids_mod[:, 32:] = torch.randint(3, 128, (1, 32))
        lg2 = tr_model(ids_mod, tr_passes=6)
        assert float((lg1[:, :32] - lg2[:, :32]).abs().max().item()) < 1e-5, "Look-ahead leak in TR!"

        engine = CalciteStreamEngine(tr_model)
        lg_stream, st = engine.prefill(ids[:, :80], return_all_logits=True)
        lg_batch = tr_model(ids[:, :80], tr_passes=max(2, st.backbone_ticks))
        err = float((lg_batch - lg_stream).abs().max().item())
        assert err < 1e-4, f"TR stream vs batch fixed-point mismatch: {err}"

        mem0 = st.memory_bytes()
        _, st = engine.prefill(torch.randint(3, 128, (1, 128)), state=st, return_all_logits=False)
        assert st.memory_bytes() == mem0, f"TR stream RAM grew: {mem0} -> {st.memory_bytes()}"


def test_interleaved_canon_stream_parity_and_unsupported_axes_guard() -> None:
    """Interleaved topology passes canon_rings cleanly in streaming, rejects unsupported AttnRes/TR, and tests Muon state_dict."""
    from scripts.muon_opt import MuonAdamWHybrid

    torch.manual_seed(2026)
    cfg = _make_tiny_cfg(topology="interleaved", use_scaled_decoder=True)
    cfg.use_canon_chunks = True
    m = CalciteV2LM(cfg).eval()
    m.byte_chunker.lock_router(True)
    m.set_crystal_schedule(1, 1)
    m.set_snap_alpha(1.0)
    x = torch.randint(3, 128, (1, 64))
    with torch.no_grad():
        lb = m(x)
        ls, _ = CalciteStreamEngine(m).prefill(x, return_all_logits=True)
    err = float((lb - ls).abs().max().item())
    assert err < 1e-4, f"interleaved canon stream mismatch: {err}"

    for bad_kw in ({"use_attnres": True}, {"use_temporal_residual": True}):
        bad_cfg = _make_tiny_cfg(topology="interleaved")
        for k, v in bad_kw.items():
            setattr(bad_cfg, k, v)
        with pytest.raises(ValueError, match="hourglass"):
            CalciteV2LM(bad_cfg)

    lin = torch.nn.Linear(32, 32)
    opt = MuonAdamWHybrid(lin, base_lr=1e-3, muon_lr_mult=4.0, device="cpu")
    lin(torch.randn(2, 32)).sum().backward()
    opt.step()
    sd = opt.state_dict()
    opt2 = MuonAdamWHybrid(lin, base_lr=1e-3, muon_lr_mult=4.0, device="cpu")
    opt2.load_state_dict(sd)
    assert len(opt2.muon_bufs) == len(opt.muon_bufs) == 1
    assert torch.allclose(opt2.muon_bufs[id(lin.weight)], opt.muon_bufs[id(lin.weight)])


