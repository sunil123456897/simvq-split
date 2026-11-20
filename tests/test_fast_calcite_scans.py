"""
Automated Parity, Regression & Speedup Verification Suite for Calcite-5 v3 CUDA Kernels
(`kernels/fast_calcite_scans.py` + `kernels/fast_calcite_swa_chunker.py`).

Verifies:
  1. Exact parity of all 18 v2+v3 kernel & integration checks (`verify_kernel_parity("cpu")`),
     including fused SWA RoPE attention, zero-`argsort` UTF-8 chunker + locked router,
     zero-double-encode `fast_crystal_forward`, lean `fast_woodbury_forward`, fused ternary STE,
     2-microbatch QAT gradient accumulation + cache invalidation, flat CE & chunked CE+KL losses,
     and `unwrap_compiled_state_dict`.
  2. Adversarial v3 regression checks: dynamic `use_static_m_cap=False` (`M_max > static_cap`),
     `patch_fast_swa_attention` / `patch_fast_chunker` clean toggling + `PackedLUT2BitLinear`
     compatibility, and `FastTernaryQAT` `clear_fast_qat_cache` / `qat_mix` change /
     `inference_mode -> train` / slot cleanup invariants.
  3. Verified Colab `NVIDIA L4` v3 native CUDA + `torch.compile` benchmark artifact
     (`benchmarks/fast_calcite_kernels_results.json`): both `fast_calcite_scans` v3 and
     `fast_calcite_swa_chunker` `cuda_ext_available == True`, all 18 parity checks clean,
     strictly > v2 compiled FP/QAT/QAT-accum2 speedups, honest cache hit/miss counts,
     > 30% peak VRAM savings, and per-block breakdown across `S1_5M`, `S2_25M`, and `S3_85M`.
  4. Zero-byte `git diff HEAD` on protected Phases A–J core paths.
"""

from __future__ import annotations

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

from experiments.scaling_phase_k.lean_system1 import (  # noqa: E402
    CalciteLeanSystem1LM,
    build_tier_lean_config,
)
from kernels.fast_calcite_scans import (  # noqa: E402
    clear_fast_qat_cache,
    get_fast_qat_cache_stats,
    unwrap_compiled_state_dict,
)
from kernels.fast_calcite_swa_chunker import (  # noqa: E402
    fast_encode_to_chunks,
    patch_fast_swa_attention,
)
from kernels.fast_packed_lut import pack_fast_model_2bit_lut  # noqa: E402
from scripts.bench_fast_calcite_kernels import verify_kernel_parity  # noqa: E402
from simvq_split.baselines import CausalRoPEAttention  # noqa: E402
from simvq_split.chunker import SurpriseHNetChunker  # noqa: E402
from simvq_split.quant_lut import quantize_ternary_weights_ls  # noqa: E402


def _assert_all_finite(obj: object, trail: str = "root") -> None:
    if isinstance(obj, float):
        assert math.isfinite(obj), f"Non-finite float at {trail}: {obj}"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _assert_all_finite(v, f"{trail}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_all_finite(v, f"{trail}[{i}]")


def test_fast_scans_and_single_pass_qat_parity():
    parity = verify_kernel_parity("cpu")
    _assert_all_finite(parity, "cpu_parity")

    assert parity["utf8_chunk_span_exact"] is True
    assert float(parity["birkhoff_perm_exact_err"]) == 0.0
    assert max(float(v) for v in parity["swa_rope_attn_fp32"].values()) < 2e-4
    assert max(float(v) for v in parity["swa_rope_attn_bf16"].values()) < 2e-2

    chk = parity["utf8_span_compact_zero_argsort"]
    assert chk["b_mask_slot_gather_exact"] is True
    assert int(chk["mid_utf8_cuts"]) == 0
    assert float(chk["router_max_abs_diff"]) == 0.0
    assert max(float(chk[k]) for k in ("z_in_err", "h_lex_err", "grad_e_err", "grad_param_err")) < 1e-4

    cr = parity["fast_crystal_forward"]
    assert int(cr["encode_tokens_calls"]) == 1
    assert max(float(cr[k]) for k in ("fwd", "commit", "gx", "g_param")) < 1e-4

    for tag in ("d32_fp", "d48_qat"):
        assert max(float(v) for v in parity["fast_woodbury_forward"][tag].values()) < 1e-4

    for tag in ("mix_1.0", "mix_0.6"):
        assert max(float(v) for v in parity["fused_ternary_ste"][tag].values()) < 1e-5

    e2e = parity["e2e_s1_qat_parity"]
    assert float(e2e["fwd_max_diff"]) < 1e-5
    assert float(e2e["grad_max_diff"]) < 1e-5
    assert float(e2e["router_max_abs_diff"]) == 0.0

    acc = parity["qat_2microbatch_accum_parity"]
    assert float(acc["fwd_max_diff"]) < 1e-5
    assert float(acc["accum_grad_max_diff"]) < 1e-4
    assert float(acc["post_step_fwd_diff"]) < 1e-5
    assert int(acc["mb0_misses"]) == 70
    assert int(acc["mb1_extra_misses"]) == 0
    assert int(acc["mb1_hits"]) == 70
    assert int(acc["post_step_misses"]) == 70
    assert float(acc["router_max_abs_diff"]) == 0.0

    assert max(float(v) for v in parity["loss_fp_ce_and_qat_cekl_parity"].values()) < 1e-5

    # Verify `_orig_mod.` prefix stripping for torch.compile checkpoints
    class _DummyOrig(nn.Module):
        def __init__(self):
            super().__init__()
            self._orig_mod = nn.Linear(4, 4, bias=False)

    class _Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.mid_mlp = _DummyOrig()

    sd = unwrap_compiled_state_dict(_Wrap())
    assert "mid_mlp.weight" in sd and all("_orig_mod" not in k for k in sd)


def test_v3_swa_chunker_edge_cases_and_qat_cache_reset():
    torch.manual_seed(2026)

    # 1. Dynamic `use_static_m_cap=False` where rounded 32-multiple `M_max > static_cap`
    for seq_t in (1, 17, 64, 192):
        chk = SurpriseHNetChunker(
            vocab_size=259,
            d_model=128,
            byte_chunk_mode="utf8",
            use_static_m_cap=False,
        )
        chk.lock_router(True)
        with torch.no_grad():
            chk.router_ctx.bias.fill_(10.0)
        r_snap = {nm: p.detach().clone() for nm, p in chk.named_parameters() if nm.startswith("router_")}
        ids = torch.full((2, seq_t), 68, dtype=torch.long)
        e_tok = torch.randn(2, seq_t, 128)
        z_r, hl_r, ctx_r = chk.encode_to_chunks(e_tok, ids)
        z_f, hl_f, ctx_f = fast_encode_to_chunks(chk, e_tok, ids)
        assert z_r.shape == z_f.shape and hl_r.shape == hl_f.shape
        assert torch.equal(ctx_r.b_mask, ctx_f.b_mask)
        assert torch.equal(ctx_r.slot_idx, ctx_f.slot_idx)
        assert torch.equal(ctx_r.valid_chunk, ctx_f.valid_chunk)
        assert float((z_r - z_f).abs().max().item()) < 1e-5
        assert float((hl_r - hl_f).abs().max().item()) < 1e-5
        for nm, p in chk.named_parameters():
            if nm in r_snap:
                assert float((p.detach() - r_snap[nm]).abs().max().item()) == 0.0

    # 2. Global attention (`window_size=None`) must stay untouched by `patch_fast_swa_attention`
    global_attn = CausalRoPEAttention(d_model=128, n_heads=2, window_size=None)
    assert patch_fast_swa_attention(global_attn, enabled=True) == 0

    # 3. Cache reset, `qat_mix` change invalidation, `inference_mode -> train`, toggle cleanup & LUT parity
    cfg = build_tier_lean_config("S1_5M", window_size=8, crystal_w_chunks=2, woodbury_latch_span=6)
    m_ref = CalciteLeanSystem1LM(cfg)
    m_fast = CalciteLeanSystem1LM(cfg)
    m_fast.load_state_dict(m_ref.state_dict(), strict=True)
    assert list(m_ref.state_dict().keys()) == list(m_fast.state_dict().keys())

    for mm in (m_ref, m_fast):
        mm.byte_chunker.lock_router(True)
        mm.set_crystal_schedule(1, 1)
        mm.set_snap_alpha(1.0)
        mm.set_qat_mix(0.4)
    m_ref.set_fast_kernels(False)
    m_ref.set_ternary_qat(True)
    m_fast.set_fast_kernels(True)
    m_fast.set_ternary_qat(True)

    clear_fast_qat_cache(reset_stats=True)
    xb = torch.randint(3, 259, (2, 64))
    # Populate mid_mlp QAT cache slots under `torch.inference_mode()` (must never cache inference tensors)
    with torch.inference_mode():
        _ = m_fast.mid_mlp(torch.randn(2, 8, cfg.d_model))
    st_inf = get_fast_qat_cache_stats()
    assert st_inf["misses"] == 3 and st_inf["hits"] == 0

    # Populate remaining slots under `no_grad`, then verify full 70/70 cache hits + live STE grads in train
    with torch.no_grad():
        _ = m_fast(xb)
    st_ng = get_fast_qat_cache_stats()
    assert st_ng["misses"] == 70 and st_ng["hits"] == 3

    m_ref.zero_grad(set_to_none=True)
    m_fast.zero_grad(set_to_none=True)
    y_r = m_ref(xb)
    y_f = m_fast(xb)
    st_hit = get_fast_qat_cache_stats()
    assert st_hit["hits"] == 73 and st_hit["misses"] == 70
    y_r.square().mean().backward()
    y_f.square().mean().backward()
    assert float((y_r - y_f).abs().max().item()) < 1e-5
    for (n1, p1), (n2, p2) in zip(m_ref.named_parameters(), m_fast.named_parameters()):
        assert n1 == n2
        if p1.grad is not None and p2.grad is not None:
            assert float((p1.grad - p2.grad).abs().max().item()) < 1e-4, f"Cache-hit STE grad drift on {n1}"

    # Explicit `clear_fast_qat_cache` & `qat_mix` change must both invalidate
    clear_fast_qat_cache(reset_stats=True)
    m_ref.set_qat_mix(0.9)
    m_fast.set_qat_mix(0.9)
    y_r2 = m_ref(xb)
    y_f2 = m_fast(xb)
    st_mix = get_fast_qat_cache_stats()
    assert st_mix["misses"] == 70 and st_mix["hits"] == 0
    assert float((y_r2 - y_f2).abs().max().item()) < 1e-5

    # Disabling fast kernels under QAT must drop all fast QAT slots cleanly
    m_fast.set_fast_kernels(False)
    assert get_fast_qat_cache_stats()["active_slots"] == 0
    m_fast.set_ternary_qat(False)
    m_ref.set_ternary_qat(False)

    # Packed 2-bit LUT compatibility with fast SWA + fast chunker + fast crystal/woodbury
    m_lut_src = CalciteLeanSystem1LM(cfg).eval()
    m_lut_src.byte_chunker.lock_router(True)
    m_lut_src.set_crystal_schedule(1, 1)
    m_lut_src.set_snap_alpha(1.0)
    with torch.no_grad():
        for nm, mod in m_lut_src.named_modules():
            if isinstance(mod, nn.Linear) and nm != "lm_head" and not m_lut_src.byte_chunker.is_locked_router_module(mod):
                wt, _, _ = quantize_ternary_weights_ls(mod.weight.data)
                mod.weight.data.copy_(wt)
    m_lut_ref = pack_fast_model_2bit_lut(m_lut_src, backend="pytorch_vec").eval()
    m_lut_ref.set_fast_kernels(False)
    m_lut_fast = pack_fast_model_2bit_lut(m_lut_src, backend="pytorch_vec").eval()
    m_lut_fast.set_fast_kernels(True)
    with torch.no_grad():
        lut_diff = float((m_lut_ref(xb) - m_lut_fast(xb)).abs().max().item())
    assert lut_diff < 1e-4, f"PackedLUT2BitLinear fast_kernels=True drift: {lut_diff}"


def test_fast_calcite_colab_benchmark_artifact():
    art = ROOT_DIR / "benchmarks" / "fast_calcite_kernels_results.json"
    assert art.is_file(), f"Missing Colab kernel benchmark artifact: {art}"
    data = json.loads(art.read_text(encoding="utf-8"))
    _assert_all_finite(data, art.name)

    assert data["device"] == "NVIDIA L4"
    assert data["backend_info"]["cuda_ext_available"] is True
    assert data["backend_info"]["cuda_build_error"] is None
    assert data["backend_info"]["version"] == "v3"

    swa_bi = data["swa_chunker_backend_info"]
    assert swa_bi["cuda_ext_available"] is True
    assert swa_bi["cuda_build_error"] is None
    assert swa_bi["custom_op_registered"] is True

    par = data["parity"]
    assert par["utf8_chunk_span_exact"] is True
    assert float(par["birkhoff_perm_exact_err"]) == 0.0
    for key in (
        "refractory_gate_err",
        "crystal_scan_hard_0",
        "crystal_scan_hard_1",
        "woodbury_fp_d32",
        "woodbury_fp_d48",
        "woodbury_tern_d32",
        "woodbury_tern_d48",
        "swa_rope_attn_fp32",
        "loss_fp_ce_and_qat_cekl_parity",
    ):
        assert max(float(v) for v in par[key].values()) < 1e-4, f"Parity drift in {key}: {par[key]}"
    assert max(float(v) for v in par["swa_rope_attn_bf16"].values()) < 2e-2

    chk = par["utf8_span_compact_zero_argsort"]
    assert chk["b_mask_slot_gather_exact"] is True
    assert int(chk["mid_utf8_cuts"]) == 0
    assert float(chk["router_max_abs_diff"]) == 0.0
    assert max(float(chk[k]) for k in ("z_in_err", "h_lex_err", "grad_e_err", "grad_param_err")) < 1e-4

    cr = par["fast_crystal_forward"]
    assert int(cr["encode_tokens_calls"]) == 1
    assert max(float(cr[k]) for k in ("fwd", "commit", "gx", "g_param")) < 1e-4

    for tag in ("d32_fp", "d48_qat"):
        assert max(float(v) for v in par["fast_woodbury_forward"][tag].values()) < 1e-4
    for tag in ("mix_1.0", "mix_0.6"):
        assert max(float(v) for v in par["fused_ternary_ste"][tag].values()) < 1e-5

    e2e = par["e2e_s1_qat_parity"]
    assert float(e2e["fwd_max_diff"]) < 1e-5
    assert float(e2e["grad_max_diff"]) < 1e-5
    assert float(e2e["router_max_abs_diff"]) == 0.0

    acc = par["qat_2microbatch_accum_parity"]
    assert float(acc["fwd_max_diff"]) < 1e-5
    assert float(acc["accum_grad_max_diff"]) < 1e-4
    assert float(acc["post_step_fwd_diff"]) < 1e-5
    assert int(acc["mb0_misses"]) == 70
    assert int(acc["mb1_extra_misses"]) == 0
    assert int(acc["mb1_hits"]) == 70
    assert int(acc["post_step_misses"]) == 70
    assert float(acc["router_max_abs_diff"]) == 0.0

    # Strict > v2 L4 speedup thresholds, honest QAT cache counts, > 30% peak VRAM drop & block breakdown
    spd = data["speed"]
    blks = data["block_breakdown"]
    tier_gates = (
        ("S1_5M", 70.0, 5.0, 7.0, 12.0),
        ("S2_25M", 90.0, 3.5, 4.5, 7.0),
        ("S3_85M", 100.0, 3.3, 4.5, 7.0),
    )
    for tier, exp_misses, min_fp_x, min_qat_x, min_acc2_x in tier_gates:
        fp_e = spd[tier]["FP"]
        qat_e = spd[tier]["QAT"]
        assert float(fp_e["compiled_speedup_x"]) >= min_fp_x, f"{tier} FP speedup regressed: {fp_e}"
        assert float(qat_e["compiled_speedup_x"]) >= min_qat_x, f"{tier} QAT speedup regressed: {qat_e}"
        assert float(qat_e["accum2_compiled_speedup_x"]) >= min_acc2_x, f"{tier} QAT accum2 regressed: {qat_e}"
        assert float(qat_e["qat_misses_per_step"]) == exp_misses
        assert float(qat_e["qat_hits_per_step"]) == 0.0
        assert float(qat_e["accum2_qat_misses_per_step"]) == exp_misses
        assert float(qat_e["accum2_qat_hits_per_step"]) == exp_misses
        assert float(fp_e["compiled_peak_mem_mb"]) < float(fp_e["baseline_peak_mem_mb"]) * 0.70
        assert float(qat_e["compiled_peak_mem_mb"]) < float(qat_e["baseline_peak_mem_mb"]) * 0.70

        b_base = blks[tier]["FP_baseline"]
        b_fast = blks[tier]["FP_fast_v3"]
        assert float(b_fast["step_total_ms"]) < float(b_base["step_total_ms"]) / 3.0
        assert float(b_fast["chunker_enc"]) < float(b_base["chunker_enc"]) / 3.0
        assert float(b_fast["crystal_1"]) < float(b_base["crystal_1"]) / 5.0
        assert float(b_fast["woodbury"]) < float(b_base["woodbury"]) / 3.0


def test_fast_qat_fused_adamw_cache_invalidation():
    """Regression test: `AdamW(fused=True)` skips `weight._version` bump; post-step hook must still invalidate cache."""
    import copy

    from kernels.fast_calcite_scans import clear_fast_qat_cache, get_fast_qat_cache_stats

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(2026)
    cfg = build_tier_lean_config("S1_5M", window_size=8, crystal_w_chunks=2, woodbury_latch_span=6)
    m_ref = CalciteLeanSystem1LM(cfg).to(dev)
    m_fast = copy.deepcopy(m_ref).to(dev)
    for m in (m_ref, m_fast):
        if m.byte_chunker is not None:
            m.byte_chunker.lock_router(True)
        m.set_crystal_schedule(1, 1)
        m.set_snap_alpha(1.0)
        m.set_qat_mix(1.0)
    m_ref.set_fast_kernels(False)
    m_ref.set_ternary_qat(True)
    m_fast.set_fast_kernels(True)
    m_fast.set_ternary_qat(True)

    use_fused = dev == "cuda"
    opt_ref = torch.optim.AdamW([p for p in m_ref.parameters() if p.requires_grad], lr=5e-3, fused=use_fused)
    opt_fast = torch.optim.AdamW([p for p in m_fast.parameters() if p.requires_grad], lr=5e-3, fused=use_fused)

    x1 = torch.randint(3, 128, (2, 32), device=dev)
    x2 = torch.randint(3, 128, (2, 32), device=dev)
    y1 = torch.randint(3, 128, (2, 32), device=dev)
    y2 = torch.randint(3, 128, (2, 32), device=dev)

    clear_fast_qat_cache(reset_stats=True)
    # 2 microbatches before step: mb0 misses, mb1 hits
    for m, opt in ((m_ref, opt_ref), (m_fast, opt_fast)):
        opt.zero_grad(set_to_none=True)
        for xb, yb in ((x1, y1), (x2, y2)):
            lg = m(xb)
            loss = torch.nn.functional.cross_entropy(lg.float().reshape(-1, lg.size(-1)), yb.reshape(-1))
            (0.5 * loss).backward()
    st_pre = get_fast_qat_cache_stats()
    assert st_pre["misses"] > 0 and st_pre["hits"] == st_pre["misses"], f"Expected 100% mb1 hits: {st_pre}"

    # Optimizer step (fused=True on CUDA does NOT increment weight._version!)
    opt_ref.step()
    opt_fast.step()

    # Post-step forward at constant mix=1.0 MUST miss cache and match uncached reference!
    m_ref.eval()
    m_fast.eval()
    with torch.no_grad():
        lg_ref = m_ref(x1)
        lg_fast = m_fast(x1)
    st_post = get_fast_qat_cache_stats()
    extra_misses = st_post["misses"] - st_pre["misses"]
    max_diff = float((lg_ref - lg_fast).abs().max().item())
    # Explicitly clear cache and verify zero staleness (`0.0` diff vs freshly cleared cache)
    clear_fast_qat_cache()
    with torch.no_grad():
        lg_fast_fresh = m_fast(x1)
    stale_diff = float((lg_fast - lg_fast_fresh).abs().max().item())
    m_ref.set_ternary_qat(False)
    m_fast.set_ternary_qat(False)
    assert extra_misses == st_pre["misses"], f"Fused AdamW failed to invalidate QAT cache: pre={st_pre} post={st_post}"
    assert stale_diff == 0.0, f"Stale QAT cache detected after opt.step(): {stale_diff:.3e}"
    assert max_diff < 1.5e-4, f"Post-step QAT logits drifted from reference: {max_diff:.3e}"


def test_fast_kernels_zero_diff_protected_core():
    protected = (
        "simvq_split/",
        "kernels/fast_packed_lut.py",
        "experiments/cram_gsm8k/",
        "experiments/uebr_proto/",
        "experiments/uebr_system2/",
    )
    for prot in protected:
        chk = subprocess.run(
            ["git", "diff", "--exit-code", "HEAD", "--", prot],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
        )
        assert chk.returncode == 0, f"Protected path modified `{prot}`:\n{chk.stdout}\n{chk.stderr}"
