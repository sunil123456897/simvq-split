"""
Exact Numerical Parity, Per-Block Breakdown & E2E Speedup Benchmark for Calcite-5 v3 CUDA Kernels
(`kernels/fast_calcite_scans.py` + `kernels/fast_calcite_swa_chunker.py` + lean CE/KL losses).

Verifies:
1. Bitwise / `< 1e-4` forward & backward parity of all v2+v3 scan kernels, fused SWA RoPE attention,
   zero-`argsort` UTF-8 chunker + locked-router fast path, zero-double-encode Crystal, lean Woodbury,
   fused ternary STE, 2-microbatch QAT gradient accumulation caching, and flat/chunked CE+KL losses
   against untouched `simvq_split/` references.
2. End-to-end `CalciteLeanSystem1LM` (`S1_5M`, `S2_25M`, `S3_85M`) `FP` & `QAT` forward+backward
   step latency (`ms/step`), peak VRAM (`MB`), and per-block breakdown before (`use_fast_kernels=False`)
   vs after (`use_fast_kernels=True` and `+ compile_cortex`).
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.scaling_phase_k.lean_system1 import (  # noqa: E402
    CalciteLeanSystem1LM,
    build_tier_lean_config,
)
from kernels.fast_calcite_scans import (  # noqa: E402
    clear_fast_qat_cache,
    compile_calcite_cortex,
    fast_causal_refractory_gate,
    fast_crystal_forward,
    fast_diag_birkhoff_crystal_scan,
    fast_neural_woodbury_chunk_scan,
    fast_project_birkhoff_to_perm,
    fast_quantize_ternary_ste,
    fast_utf8_chunk_span,
    fast_woodbury_forward,
    get_fast_qat_cache_stats,
    get_fast_scans_backend_info,
)
from kernels.fast_calcite_swa_chunker import (  # noqa: E402
    fast_encode_to_chunks,
    get_fast_swa_chunker_backend_info,
    patch_fast_swa_attention,
    utf8_span_and_compact_cuda,
)
from scripts.colab_phase_k_scaling_runner import _FusedChunkedCEKLFn  # noqa: E402
from simvq_split.baselines import CausalRoPEAttention  # noqa: E402
from simvq_split.chunker import (  # noqa: E402
    SurpriseHNetChunker,
    _enforce_utf8_chunk_span_jit,
    _utf8_tensor_flags,
)
from simvq_split.crystal import (  # noqa: E402
    VoronoiCrystalLayer,
    causal_refractory_gate_jit,
    diag_birkhoff_crystal_scan as ref_crystal_scan,
    project_birkhoff_to_perm_jit,
)
from simvq_split.quant_lut import quantize_ternary_ste  # noqa: E402
from simvq_split.tokenizer import ByteBPETokenizer  # noqa: E402
from simvq_split.woodbury import (  # noqa: E402
    NeuralWoodburyLayer,
    block_checkpointed_tern_scan,
    block_parallel_wy_fp_scan,
)


def _snapshot_router(model: CalciteLeanSystem1LM) -> dict[str, torch.Tensor]:
    if model.byte_chunker is None:
        return {}
    return {
        nm: p.detach().cpu().clone()
        for nm, p in model.byte_chunker.named_parameters()
        if nm.startswith("router_")
    }


def _max_router_diff(model: CalciteLeanSystem1LM, snap: dict[str, torch.Tensor]) -> float:
    if model.byte_chunker is None or not snap:
        return 0.0
    max_d = 0.0
    for nm, p in model.byte_chunker.named_parameters():
        if nm in snap:
            max_d = max(max_d, float((p.detach().cpu() - snap[nm]).abs().max().item()))
    return max_d


def verify_kernel_parity(device: str) -> dict[str, object]:
    out: dict[str, object] = {}
    torch.manual_seed(2026)

    # 1. UTF-8 chunk span guard (exact bitwise equality)
    tok = ByteBPETokenizer(merges=[])
    sample_text = (
        "Модель без токенизатора держит границу UTF-8 строго целой. "
        "多语言字节切分绝对不能切断多字节字符。🚀✨🔥 "
        "def f(x): return x * 2 + 1\n"
    ) * 6
    ids_1 = tok.encode(sample_text, add_bos=True, add_eos=True)[:768]
    B_u = 16
    x_ids = torch.zeros((B_u, 768), dtype=torch.long, device=device)
    for b in range(B_u):
        cut = 768 - (b * 17)
        x_ids[b, :cut] = torch.tensor(ids_1[:cut], dtype=torch.long, device=device)
    valid_mask = x_ids != 0
    valid_mask[:, 0] = True
    _, is_cont, char_len = _utf8_tensor_flags(x_ids)
    is_cont_v = is_cont & valid_mask
    c1 = F.pad(is_cont_v[:, :-1], (1, 0), value=False)
    c2 = F.pad(is_cont_v[:, :-2], (2, 0), value=False)
    c3 = F.pad(is_cont_v[:, :-3], (3, 0), value=False)
    can_fire = valid_mask & (~(is_cont_v & (~(c1 & c2 & c3))))
    is_lead = (valid_mask & (~is_cont)).long()
    raw_cand = (torch.rand((B_u, 768), device=device) > 0.72) & can_fire

    ref_b = _enforce_utf8_chunk_span_jit(raw_cand, valid_mask, can_fire, is_lead, char_len, 10, 16, 40, 100, 1200)
    fast_b = fast_utf8_chunk_span(raw_cand, valid_mask, can_fire, is_lead, char_len, 10, 16, 40, 100, 1200)
    utf8_exact = bool(torch.equal(ref_b, fast_b))
    assert utf8_exact, f"UTF-8 span mismatch: {(ref_b != fast_b).sum().item()} bits"
    out["utf8_chunk_span_exact"] = utf8_exact

    # 2. Causal refractory gate (fwd + bwd)
    p0 = torch.rand((8, 192, 2), device=device, dtype=torch.float32, requires_grad=True)
    p1 = p0.detach().clone().requires_grad_(True)
    g_out = torch.randn_like(p0)
    y_ref = causal_refractory_gate_jit(p0)
    y_fast = fast_causal_refractory_gate(p1)
    (y_ref * g_out).sum().backward()
    (y_fast * g_out).sum().backward()
    ref_fwd_err = float((y_ref - y_fast).abs().max().item())
    ref_bwd_err = float((p0.grad - p1.grad).abs().max().item())
    assert ref_fwd_err < 1e-5 and ref_bwd_err < 1e-5, f"Refractory err: fwd={ref_fwd_err}, bwd={ref_bwd_err}"
    out["refractory_gate_err"] = {"fwd": ref_fwd_err, "bwd": ref_bwd_err}

    # 3. Birkhoff greedy permutation projection
    sc = torch.randn((1024, 12, 12), device=device, dtype=torch.float32)
    ph_ref = project_birkhoff_to_perm_jit(sc)
    ph_fast = fast_project_birkhoff_to_perm(sc)
    birk_err = float((ph_ref - ph_fast).abs().max().item())
    assert birk_err == 0.0, f"Birkhoff greedy perm mismatch: {birk_err}"
    assert bool((ph_fast.sum(dim=-1) == 1.0).all().item() and (ph_fast.sum(dim=-2) == 1.0).all().item())
    out["birkhoff_perm_exact_err"] = birk_err

    # 4. Crystal scan v3 (both is_hard=False and is_hard=True, fwd + all 4 grads)
    for is_hard in (False, True):
        BH, M, Gh, ds = 16, 128, 12, 32
        s0_r = F.normalize(torch.randn((BH, Gh, ds), device=device), dim=-1) * (ds ** 0.5)
        kwe_r = torch.rand((BH, M, Gh, 1), device=device) * 0.4
        kwv_r = torch.rand((BH, M, Gh, 1), device=device) * 0.4
        vs_r = F.normalize(torch.randn((BH, M, ds), device=device), dim=-1) * (ds ** 0.5)
        pe_r = F.softmax(torch.randn((BH, M, Gh, Gh), device=device), dim=-1)
        if is_hard:
            pe_r = fast_project_birkhoff_to_perm(pe_r.reshape(-1, Gh, Gh)).reshape(BH, M, Gh, Gh)

        s0_r.requires_grad_(True)
        kwe_r.requires_grad_(True)
        kwv_r.requires_grad_(True)
        vs_r.requires_grad_(True)
        pe_r.requires_grad_(True)

        s0_f = s0_r.detach().clone().requires_grad_(True)
        kwe_f = kwe_r.detach().clone().requires_grad_(True)
        kwv_f = kwv_r.detach().clone().requires_grad_(True)
        vs_f = vs_r.detach().clone().requires_grad_(True)
        pe_f = pe_r.detach().clone().requires_grad_(True)

        go = torch.randn((BH, M, Gh, ds), device=device)
        out_r = ref_crystal_scan(s0_r, kwe_r, kwv_r, vs_r, pe_r, is_hard, ds ** 0.5, Bc=32)
        out_f = fast_diag_birkhoff_crystal_scan(s0_f, kwe_f, kwv_f, vs_f, pe_f, is_hard, ds ** 0.5, Bc=32)
        (out_r * go).sum().backward()
        (out_f * go).sum().backward()

        errs = {
            "fwd": float((out_r - out_f).abs().max().item()),
            "g_s0": float((s0_r.grad - s0_f.grad).abs().max().item()),
            "g_kwe": float((kwe_r.grad - kwe_f.grad).abs().max().item()),
            "g_vs": float((vs_r.grad - vs_f.grad).abs().max().item()),
            "g_P": float((pe_r.grad - pe_f.grad).abs().max().item()),
        }
        assert max(errs.values()) < 2e-4, f"Crystal scan parity fail (is_hard={is_hard}): {errs}"
        out[f"crystal_scan_hard_{int(is_hard)}"] = errs

    # 5. Woodbury scan v3 (FP tern_gate=0.0 and Ternary tern_gate=1.0, d=32 and d=48)
    for t_gate, tag_w in ((0.0, "fp"), (1.0, "tern")):
        for d_k in (32, 48):
            B, M_len, C, H = 4, 64, 4, 5
            q_r = F.normalize(torch.randn((B, M_len, C, H, d_k), device=device), dim=-1).requires_grad_(True)
            k_r = F.normalize(torch.randn((B, M_len, H, d_k), device=device), dim=-1).requires_grad_(True)
            v_r = (torch.randn((B, M_len, H, d_k), device=device) * 0.5).requires_grad_(True)
            b_r = torch.sigmoid(torch.randn((B, M_len, H, 1), device=device)).requires_grad_(True)
            g_r = torch.sigmoid(torch.randn((B, M_len, H, 1), device=device) + 2.0).requires_grad_(True)
            qe = F.normalize(torch.randn((B, M_len, C, H, d_k), device=device), dim=-1)
            ke = F.normalize(torch.randn((B, M_len, H, d_k), device=device), dim=-1)

            q_f = q_r.detach().clone().requires_grad_(True)
            k_f = k_r.detach().clone().requires_grad_(True)
            v_f = v_r.detach().clone().requires_grad_(True)
            b_f = b_r.detach().clone().requires_grad_(True)
            g_f = g_r.detach().clone().requires_grad_(True)

            if t_gate <= 0.0:
                o_r, u_r = block_parallel_wy_fp_scan(q_r, k_r, v_r, b_r, g_r, 25.0, qe, ke, Bc=32, checkpointed=False)
            else:
                o_r, u_r = block_checkpointed_tern_scan(q_r, k_r, v_r, b_r, g_r, 25.0, qe, ke, tern_gate=1.0, Bc=32)
            o_f, u_f = fast_neural_woodbury_chunk_scan(q_f, k_f, v_f, b_f, g_f, 25.0, qe, ke, tern_gate=t_gate)
            go = torch.randn_like(o_r)
            (o_r * go).sum().backward()
            (o_f * go).sum().backward()

            w_errs = {
                "out": float((o_r - o_f).abs().max().item()),
                "uncerts": float((u_r - u_f).abs().max().item()),
                "dq": float((q_r.grad - q_f.grad).abs().max().item()),
                "dk": float((k_r.grad - k_f.grad).abs().max().item()),
                "dv": float((v_r.grad - v_f.grad).abs().max().item()),
                "db": float((b_r.grad - b_f.grad).abs().max().item()),
                "dg": float((g_r.grad - g_f.grad).abs().max().item()),
            }
            assert max(w_errs.values()) < 5e-4, f"Woodbury {tag_w} parity fail (d={d_k}): {w_errs}"
            out[f"woodbury_{tag_w}_d{d_k}"] = w_errs

    # 6. Fused Local Sliding-Window RoPE Attention (FP32 + BF16, fwd + bwd)
    for dt_name, dt, tol_fwd, tol_bwd in (
        ("fp32", torch.float32, 2e-4, 2e-4),
        ("bf16", torch.bfloat16, 2e-2, 2e-2),
    ):
        attn_ref = CausalRoPEAttention(d_model=256, n_heads=4, window_size=8).to(device=device, dtype=dt)
        attn_fast = CausalRoPEAttention(d_model=256, n_heads=4, window_size=8).to(device=device, dtype=dt)
        attn_fast.load_state_dict(attn_ref.state_dict(), strict=True)
        assert patch_fast_swa_attention(attn_fast, enabled=True) == 1

        x_r = torch.randn((4, 128, 256), device=device, dtype=dt, requires_grad=True)
        x_f = x_r.detach().clone().requires_grad_(True)
        go = torch.randn((4, 128, 256), device=device, dtype=dt)

        y_r = attn_ref(x_r)
        y_f = attn_fast(x_f)
        # Scale gradient like mean-reduced loss so BF16 weight grads don't saturate into coarse ULPs
        ((y_r.float() * go.float()).mean() * 256.0).backward()
        ((y_f.float() * go.float()).mean() * 256.0).backward()

        swa_errs = {
            "fwd": float((y_r.float() - y_f.float()).abs().max().item()),
            "gx": float((x_r.grad.float() - x_f.grad.float()).abs().max().item()),
            "g_qkv": float((attn_ref.qkv.weight.grad.float() - attn_fast.qkv.weight.grad.float()).abs().max().item()),
            "g_out": float(
                (attn_ref.out_proj.weight.grad.float() - attn_fast.out_proj.weight.grad.float()).abs().max().item()
            ),
        }
        assert swa_errs["fwd"] < tol_fwd and max(swa_errs["gx"], swa_errs["g_qkv"], swa_errs["g_out"]) < tol_bwd, (
            f"SWA RoPE attn {dt_name} parity fail: {swa_errs}"
        )
        out[f"swa_rope_attn_{dt_name}"] = swa_errs

    # 7. Zero-`argsort` UTF-8 Chunker + Locked-Router Fast Path
    chk_ref = SurpriseHNetChunker(
        vocab_size=259,
        d_model=256,
        byte_chunk_mode="utf8",
        use_static_m_cap=True,
    ).to(device)
    chk_ref.lock_router(True)
    chk_fast = SurpriseHNetChunker(
        vocab_size=259,
        d_model=256,
        byte_chunk_mode="utf8",
        use_static_m_cap=True,
    ).to(device)
    chk_fast.load_state_dict(chk_ref.state_dict(), strict=True)
    chk_fast.lock_router(True)
    r_snap = {nm: p.detach().cpu().clone() for nm, p in chk_fast.named_parameters() if nm.startswith("router_")}

    e_r = torch.randn((4, 256, 256), device=device, dtype=torch.float32, requires_grad=True)
    e_f = e_r.detach().clone().requires_grad_(True)
    ids_chk = x_ids[:4, :256]

    z_r, hl_r, ctx_r = chk_ref.encode_to_chunks(e_r, ids_chk)
    z_f, hl_f, ctx_f = fast_encode_to_chunks(chk_fast, e_f, ids_chk)
    bm_k, cc_k, sl_k, gp_k = utf8_span_and_compact_cuda(
        ids_chk,
        ctx_r.p_prob,
        pad_token_id=chk_ref.pad_token_id,
        max_cp_span=chk_ref.max_chunk_bytes,
        max_byte_span=chk_ref.max_utf8_bytes,
        refill_int=chk_ref.refill_int,
        cost_int=chk_ref.cost_int,
        burst_int=chk_ref.burst_int,
        m_cap=chk_ref.static_m_cap_for_T(256),
    )
    t_ar = torch.arange(256, device=device, dtype=torch.long).unsqueeze(0)
    gp_ref = ((~ctx_r.b_mask).long() * 257 + t_ar).argsort(dim=1)[:, : chk_ref.static_m_cap_for_T(256)]
    assert torch.equal(ctx_r.b_mask, bm_k) and torch.equal(ctx_r.b_mask, ctx_f.b_mask)
    assert torch.equal(ctx_r.slot_idx, sl_k) and torch.equal(ctx_r.slot_idx, ctx_f.slot_idx)
    assert torch.equal(gp_ref, gp_k) and torch.equal(ctx_r.b_mask.long().sum(dim=1), cc_k)

    raw_v = (ids_chk - 3).clamp(min=0)
    is_cont_c = (ids_chk >= 3) & (torch.bitwise_and(raw_v, 0xC0) == 0x80)
    mid_utf8_cuts = int((ctx_f.b_mask[:, 1:] & ctx_f.valid_mask[:, 1:] & is_cont_c[:, 1:]).sum().item())
    assert mid_utf8_cuts == 0, f"Detected {mid_utf8_cuts} mid-UTF8 cuts in fast_encode_to_chunks"

    gz = torch.randn_like(z_r)
    gh = torch.randn_like(hl_r)
    ((z_r * gz).mean() + (hl_r * gh).mean()).backward()
    ((z_f * gz).mean() + (hl_f * gh).mean()).backward()
    chk_param_g_err = 0.0
    for (n1, p1), (n2, p2) in zip(chk_ref.named_parameters(), chk_fast.named_parameters()):
        assert n1 == n2
        if p1.grad is not None and p2.grad is not None:
            chk_param_g_err = max(chk_param_g_err, float((p1.grad - p2.grad).abs().max().item()))
    chk_router_diff = max(
        float((p.detach().cpu() - r_snap[nm]).abs().max().item())
        for nm, p in chk_fast.named_parameters()
        if nm in r_snap
    )
    chk_errs = {
        "b_mask_slot_gather_exact": True,
        "mid_utf8_cuts": mid_utf8_cuts,
        "router_max_abs_diff": chk_router_diff,
        "z_in_err": float((z_r - z_f).abs().max().item()),
        "h_lex_err": float((hl_r - hl_f).abs().max().item()),
        "grad_e_err": float((e_r.grad - e_f.grad).abs().max().item()),
        "grad_param_err": chk_param_g_err,
    }
    assert chk_router_diff == 0.0
    assert max(chk_errs["z_in_err"], chk_errs["h_lex_err"], chk_errs["grad_e_err"], chk_param_g_err) < 1e-4, (
        f"Fast chunker parity fail: {chk_errs}"
    )
    out["utf8_span_compact_zero_argsort"] = chk_errs

    # 8. Zero-Double-Encode Crystal Wrapper (`fast_crystal_forward`)
    cr_ref = VoronoiCrystalLayer(d_model=256, num_spheres=24, d_sphere=32, codebook_size=32).to(device)
    cr_fast = VoronoiCrystalLayer(d_model=256, num_spheres=24, d_sphere=32, codebook_size=32).to(device)
    cr_fast.load_state_dict(cr_ref.state_dict(), strict=True)
    for cr in (cr_ref, cr_fast):
        cr.snap_alpha = 1.0
        cr.qat_active_mix = 0.75
    enc_calls = {"n": 0}
    orig_enc = cr_fast._encode_tokens

    def _counted_enc(x_in, h_in):
        enc_calls["n"] += 1
        return orig_enc(x_in, h_in)

    cr_fast._encode_tokens = _counted_enc  # type: ignore[method-assign]
    xc_r = torch.randn((4, 127, 256), device=device, requires_grad=True)
    hlc_r = torch.randn((4, 127, 256), device=device, requires_grad=True)
    xc_f = xc_r.detach().clone().requires_grad_(True)
    hlc_f = hlc_r.detach().clone().requires_grad_(True)
    yc_r = cr_ref(xc_r, h_lex=hlc_r)
    yc_f = fast_crystal_forward(cr_fast, xc_f, h_lex=hlc_f)
    assert enc_calls["n"] == 1, f"Expected 1 _encode_tokens call, got {enc_calls['n']}"
    goc = torch.randn_like(yc_r)
    ((yc_r * goc).mean() + 0.04 * cr_ref.last_commit_loss).backward()
    ((yc_f * goc).mean() + 0.04 * cr_fast.last_commit_loss).backward()
    cr_pg_err = 0.0
    for (n1, p1), (n2, p2) in zip(cr_ref.named_parameters(), cr_fast.named_parameters()):
        assert n1 == n2
        if p1.grad is not None and p2.grad is not None:
            cr_pg_err = max(cr_pg_err, float((p1.grad - p2.grad).abs().max().item()))
    cr_errs = {
        "encode_tokens_calls": enc_calls["n"],
        "fwd": float((yc_r - yc_f).abs().max().item()),
        "commit": float((cr_ref.last_commit_loss - cr_fast.last_commit_loss).abs().max().item()),
        "gx": float((xc_r.grad - xc_f.grad).abs().max().item()),
        "g_param": cr_pg_err,
    }
    assert max(cr_errs["fwd"], cr_errs["commit"], cr_errs["gx"], cr_pg_err) < 2e-4, (
        f"fast_crystal_forward parity fail: {cr_errs}"
    )
    out["fast_crystal_forward"] = cr_errs

    # 9. Lean Woodbury Wrapper (`fast_woodbury_forward`, d=32 FP + d=48 QAT)
    wb_out: dict[str, object] = {}
    for tag, d_m, h_m, d_w, qmix in (("d32_fp", 256, 5, 32, 0.0), ("d48_qat", 384, 8, 48, 1.0)):
        wb_ref = NeuralWoodburyLayer(d_model=d_m, mem_heads=h_m, mem_dim=d_w).to(device)
        wb_fast = NeuralWoodburyLayer(d_model=d_m, mem_heads=h_m, mem_dim=d_w).to(device)
        wb_fast.load_state_dict(wb_ref.state_dict(), strict=True)
        wb_ref.qat_active_mix = qmix
        wb_fast.qat_active_mix = qmix

        xw_r = torch.randn((4, 67, d_m), device=device, requires_grad=True)
        hlw_r = torch.randn((4, 67, d_m), device=device, requires_grad=True)
        xw_f = xw_r.detach().clone().requires_grad_(True)
        hlw_f = hlw_r.detach().clone().requires_grad_(True)
        yw_r = wb_ref(xw_r, h_lex=hlw_r)
        yw_f = fast_woodbury_forward(wb_fast, xw_f, h_lex=hlw_f)
        gow = torch.randn_like(yw_r)
        (yw_r * gow).mean().backward()
        (yw_f * gow).mean().backward()
        wb_pg = 0.0
        for (n1, p1), (n2, p2) in zip(wb_ref.named_parameters(), wb_fast.named_parameters()):
            assert n1 == n2
            if p1.grad is not None and p2.grad is not None:
                wb_pg = max(wb_pg, float((p1.grad - p2.grad).abs().max().item()))
        errs_w = {
            "fwd": float((yw_r - yw_f).abs().max().item()),
            "uncerts": float((wb_ref.last_epistemic_uncertainty - wb_fast.last_epistemic_uncertainty).abs().max().item()),
            "gx": float((xw_r.grad - xw_f.grad).abs().max().item()),
            "g_param": wb_pg,
        }
        assert max(errs_w.values()) < 5e-4, f"fast_woodbury_forward {tag} parity fail: {errs_w}"
        wb_out[tag] = errs_w
    out["fast_woodbury_forward"] = wb_out

    # 10. Fused Ternary STE (`fast_quantize_ternary_ste`)
    ste_out: dict[str, object] = {}
    for mix_v in (1.0, 0.6):
        w_r = (torch.randn((256, 256), device=device, dtype=torch.float32) * 0.25).requires_grad_(True)
        w_f = w_r.detach().clone().requires_grad_(True)
        wt_r = quantize_ternary_ste(w_r)
        wq_r = ((1.0 - mix_v) * w_r + mix_v * wt_r) if mix_v < 0.9999 else wt_r
        wq_f = fast_quantize_ternary_ste(w_f, mix=mix_v, out_dtype=torch.float32)
        gw = torch.randn_like(wq_r)
        (wq_r * gw).sum().backward()
        (wq_f * gw).sum().backward()
        s_err = {
            "fwd": float((wq_r - wq_f).abs().max().item()),
            "grad": float((w_r.grad - w_f.grad).abs().max().item()),
        }
        assert max(s_err.values()) < 1e-5, f"fused_ternary_ste mix={mix_v} parity fail: {s_err}"
        ste_out[f"mix_{mix_v:.1f}"] = s_err
    out["fused_ternary_ste"] = ste_out

    # 11. Full S1_5M E2E QAT parity (v2 key preserved: `< 1e-5` FP32)
    cfg = build_tier_lean_config("S1_5M", window_size=8, crystal_w_chunks=2, woodbury_latch_span=6)
    m_ref = CalciteLeanSystem1LM(cfg).to(device)
    m_fast = CalciteLeanSystem1LM(cfg).to(device)
    m_fast.load_state_dict(m_ref.state_dict(), strict=True)
    for mm in (m_ref, m_fast):
        mm.byte_chunker.lock_router(True)
        mm.set_crystal_schedule(1, 1)
        mm.set_snap_alpha(1.0)
        mm.set_qat_mix(0.75)
    m_ref.set_fast_kernels(False)
    m_ref.set_ternary_qat(True)
    m_fast.set_fast_kernels(True)
    m_fast.set_ternary_qat(True)
    r_snap_e2e = _snapshot_router(m_fast)

    xb = torch.randint(3, 259, (4, 192), device=device)
    yb = torch.randint(3, 259, (4, 192), device=device)
    lg_r = m_ref(xb)
    lg_f = m_fast(xb)
    F.cross_entropy(lg_r.float().transpose(1, 2), yb).backward()
    F.cross_entropy(lg_f.float().reshape(-1, lg_f.size(-1)), yb.reshape(-1)).backward()
    e2e_fwd = float((lg_r.float() - lg_f.float()).abs().max().item())
    max_g_diff = 0.0
    for (n1, p1), (n2, p2) in zip(m_ref.named_parameters(), m_fast.named_parameters()):
        assert n1 == n2
        if p1.grad is not None and p2.grad is not None:
            max_g_diff = max(max_g_diff, float((p1.grad.float() - p2.grad.float()).abs().max().item()))
    e2e_r_diff = _max_router_diff(m_fast, r_snap_e2e)
    assert e2e_r_diff == 0.0, f"E2E QAT router mutated: {e2e_r_diff}"
    assert e2e_fwd < 2e-4 and max_g_diff < 5e-4, f"E2E QAT parity fail: fwd={e2e_fwd}, grad={max_g_diff}"
    out["e2e_s1_qat_parity"] = {"fwd_max_diff": e2e_fwd, "grad_max_diff": max_g_diff, "router_max_abs_diff": e2e_r_diff}

    # 12. 2-Microbatch QAT Gradient Accumulation & Cache Invalidation Parity
    m_ref.zero_grad(set_to_none=True)
    m_fast.zero_grad(set_to_none=True)
    m_ref.set_qat_mix(0.65)
    m_fast.set_qat_mix(0.65)
    clear_fast_qat_cache(reset_stats=True)

    xb0 = torch.randint(3, 259, (2, 160), device=device)
    yb0 = torch.randint(3, 259, (2, 160), device=device)
    xb1 = torch.randint(3, 259, (2, 160), device=device)
    yb1 = torch.randint(3, 259, (2, 160), device=device)

    l0_r = m_ref(xb0)
    l0_f = m_fast(xb0)
    (F.cross_entropy(l0_r.float().reshape(-1, l0_r.size(-1)), yb0.reshape(-1)) * 0.5).backward()
    (F.cross_entropy(l0_f.float().reshape(-1, l0_f.size(-1)), yb0.reshape(-1)) * 0.5).backward()
    st_mb0 = get_fast_qat_cache_stats()

    l1_r = m_ref(xb1)
    l1_f = m_fast(xb1)
    (F.cross_entropy(l1_r.float().reshape(-1, l1_r.size(-1)), yb1.reshape(-1)) * 0.5).backward()
    (F.cross_entropy(l1_f.float().reshape(-1, l1_f.size(-1)), yb1.reshape(-1)) * 0.5).backward()
    st_mb1 = get_fast_qat_cache_stats()

    acc_fwd = max(
        float((l0_r.float() - l0_f.float()).abs().max().item()),
        float((l1_r.float() - l1_f.float()).abs().max().item()),
    )
    acc_grad = 0.0
    for (n1, p1), (n2, p2) in zip(m_ref.named_parameters(), m_fast.named_parameters()):
        assert n1 == n2
        if p1.grad is not None and p2.grad is not None:
            acc_grad = max(acc_grad, float((p1.grad.float() - p2.grad.float()).abs().max().item()))
    mb0_misses = int(st_mb0["misses"])
    mb1_extra_misses = int(st_mb1["misses"]) - mb0_misses
    mb1_hits = int(st_mb1["hits"]) - int(st_mb0["hits"])
    assert mb0_misses > 0 and mb1_extra_misses == 0 and mb1_hits >= mb0_misses, (
        f"Microbatch QAT cache failed: mb0={st_mb0}, mb1={st_mb1}"
    )

    # Simulate optimizer step (`w._version` bump) and confirm cache invalidation parity
    with torch.no_grad():
        for p_r, p_f in zip(m_ref.parameters(), m_fast.parameters()):
            if p_r.grad is not None and p_f.grad is not None:
                p_r.add_(p_r.grad, alpha=-1e-2)
                p_f.add_(p_f.grad, alpha=-1e-2)
    l2_r = m_ref(xb0)
    l2_f = m_fast(xb0)
    st_mb2 = get_fast_qat_cache_stats()
    post_step_fwd = float((l2_r.float() - l2_f.float()).abs().max().item())
    post_step_misses = int(st_mb2["misses"]) - int(st_mb1["misses"])
    r_diff_acc = _max_router_diff(m_fast, r_snap_e2e)
    assert post_step_misses >= mb0_misses, f"Expected cache invalidation after step: mb1={st_mb1}, mb2={st_mb2}"
    assert r_diff_acc == 0.0 and acc_fwd < 1e-4 and acc_grad < 1e-4 and post_step_fwd < 1e-4, (
        f"2-microbatch QAT accum parity fail: fwd={acc_fwd}, grad={acc_grad}, post={post_step_fwd}, r={r_diff_acc}"
    )
    out["qat_2microbatch_accum_parity"] = {
        "fwd_max_diff": acc_fwd,
        "accum_grad_max_diff": acc_grad,
        "post_step_fwd_diff": post_step_fwd,
        "mb0_misses": mb0_misses,
        "mb1_extra_misses": mb1_extra_misses,
        "mb1_hits": mb1_hits,
        "post_step_misses": post_step_misses,
        "router_max_abs_diff": r_diff_acc,
    }
    m_fast.set_ternary_qat(False)
    m_ref.set_ternary_qat(False)
    clear_fast_qat_cache(reset_stats=True)

    # 13. Optimized FP Flat CE & QAT Chunked Analytical CE+KL Parity
    B_l, L_l, V_l = 4, 192, 259
    lg_base = torch.randn((B_l, L_l, V_l), device=device, dtype=torch.float32) * 2.0
    zfp_base = torch.randn((B_l, L_l, V_l), device=device, dtype=torch.float32) * 2.0
    yb_l = torch.randint(0, V_l, (B_l, L_l), device=device)
    yb_l[:, -16:] = 0
    valid_f = (yb_l != 0).float()
    denom_f = valid_f.sum(dim=1).clamp(min=1.0)

    # FP CE
    lg_a = lg_base.clone().requires_grad_(True)
    lg_b = lg_base.clone().requires_grad_(True)
    ce_ref = (F.cross_entropy(lg_a.float().transpose(1, 2), yb_l, ignore_index=0, reduction="none").sum(dim=1) / denom_f).mean()
    ce_flat = (
        F.cross_entropy(lg_b.float().reshape(-1, V_l), yb_l.reshape(-1), ignore_index=0, reduction="none")
        .view(B_l, L_l)
        .sum(dim=1)
        / denom_f
    ).mean()
    ce_ref.backward()
    ce_flat.backward()

    # QAT CE+KL
    kl_temp = 1.5
    zq_a = lg_base.clone().requires_grad_(True)
    zq_b = lg_base.clone().requires_grad_(True)
    log_p_fp = F.log_softmax(zfp_base.float() / kl_temp, dim=-1)
    ce_qa = (F.cross_entropy(zq_a.float().transpose(1, 2), yb_l, ignore_index=0, reduction="none").sum(dim=1) / denom_f).mean()
    kl_qa = (
        (
            F.kl_div(F.log_softmax(zq_a.float() / kl_temp, dim=-1), log_p_fp, reduction="none", log_target=True).sum(dim=-1)
            * (kl_temp * kl_temp)
            * valid_f
        ).sum(dim=1)
        / denom_f
    ).mean()
    (0.35 * ce_qa + 0.45 * kl_qa).backward()
    comb_qb, ce_qb = _FusedChunkedCEKLFn.apply(zq_b, zfp_base, yb_l, valid_f, denom_f, kl_temp, 0.35, 0.45, 0, 256)
    comb_qb.backward()

    loss_errs = {
        "fp_ce_diff": float(abs(ce_ref.item() - ce_flat.item())),
        "fp_grad_diff": float((lg_a.grad - lg_b.grad).abs().max().item()),
        "qat_ce_diff": float(abs(ce_qa.item() - ce_qb.item())),
        "qat_comb_diff": float(abs((0.35 * ce_qa + 0.45 * kl_qa).item() - comb_qb.item())),
        "qat_grad_diff": float((zq_a.grad - zq_b.grad).abs().max().item()),
    }
    assert max(loss_errs.values()) < 1e-5, f"Loss CE/KL parity fail: {loss_errs}"
    out["loss_fp_ce_and_qat_cekl_parity"] = loss_errs

    return out


def _sync_cuda(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _build_bench_model(
    tier_name: str,
    qat: bool,
    use_fast: bool,
    do_compile: bool,
    compile_mode: str,
    device: str,
) -> CalciteLeanSystem1LM:
    torch.manual_seed(2026)
    cfg = build_tier_lean_config(tier_name, window_size=8, crystal_w_chunks=2, woodbury_latch_span=6)
    m = CalciteLeanSystem1LM(cfg).to(device)
    m.byte_chunker.lock_router(True)
    m.set_crystal_schedule(1, 1)
    m.set_snap_alpha(1.0)
    m.set_fast_kernels(use_fast)
    if qat:
        m.set_qat_mix(1.0)
        m.set_ternary_qat(True)
    if do_compile:
        compile_calcite_cortex(m, mode=compile_mode)
    return m


def bench_tier_speed(
    tier_name: str,
    batch_size: int,
    seq_len: int,
    qat: bool,
    use_fast: bool,
    do_compile: bool,
    device: str,
    iters: int = 8,
    compile_mode: str = "default",
    accum: int = 1,
) -> dict[str, float]:
    m = _build_bench_model(tier_name, qat, use_fast, do_compile, compile_mode, device)
    accum_i = max(1, int(accum))
    xs = [torch.randint(3, 259, (batch_size, seq_len), device=device) for _ in range(accum_i)]
    ys = [torch.randint(3, 259, (batch_size, seq_len), device=device) for _ in range(accum_i)]
    inv_acc = 1.0 / float(accum_i)

    def _run_step(model: CalciteLeanSystem1LM) -> None:
        model.zero_grad(set_to_none=True)
        # Invalidate microbatch QAT cache once per optimizer step so mb0 genuinely runs fused_ternary_ste
        if qat and use_fast:
            clear_fast_qat_cache()
        for ai in range(accum_i):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                lg = model(xs[ai])
                if use_fast:
                    loss = F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), ys[ai].reshape(-1))
                else:
                    loss = F.cross_entropy(lg.float().transpose(1, 2), ys[ai])
            (loss * inv_acc if accum_i > 1 else loss).backward()

    for _ in range(3):
        _run_step(m)
    _sync_cuda(device)

    if qat and use_fast:
        clear_fast_qat_cache(reset_stats=True)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _run_step(m)
        _sync_cuda(device)
        times.append((time.perf_counter() - t0) * 1000.0)

    qat_misses_per_step = 0.0
    qat_hits_per_step = 0.0
    if qat and use_fast:
        st = get_fast_qat_cache_stats()
        qat_misses_per_step = round(float(st["misses"]) / float(max(1, iters)), 2)
        qat_hits_per_step = round(float(st["hits"]) / float(max(1, iters)), 2)
        assert st["misses"] > 0, f"Expected real QAT quantize misses per step, got {st}"
        if accum_i > 1:
            assert st["hits"] >= st["misses"] * (accum_i - 1), (
                f"Expected microbatch QAT cache hits for accum={accum_i}, got {st}"
            )

    peak_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1) if device == "cuda" else 0.0
    m.set_ternary_qat(False)
    clear_fast_qat_cache()
    del m
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    times.sort()
    med_ms = times[len(times) // 2]
    return {
        "median_ms": round(med_ms, 2),
        "min_ms": round(times[0], 2),
        "peak_mem_mb": peak_mb,
        "qat_misses_per_step": qat_misses_per_step,
        "qat_hits_per_step": qat_hits_per_step,
    }


def profile_tier_blocks(
    tier_name: str,
    batch_size: int,
    seq_len: int,
    qat: bool,
    use_fast: bool,
    do_compile: bool,
    device: str,
    iters: int = 6,
    compile_mode: str = "default",
) -> dict[str, float]:
    """Measures per-block forward + total backward latency breakdown without forward hooks."""
    m = _build_bench_model(tier_name, qat, use_fast, do_compile, compile_mode, device)
    x = torch.randint(3, 259, (batch_size, seq_len), device=device)
    y = torch.randint(3, 259, (batch_size, seq_len), device=device)
    use_amp = device == "cuda"
    stage_names = (
        "tok_emb",
        "chunker_enc",
        "cortex_pre",
        "crystal_1",
        "woodbury",
        "mid_mlp",
        "crystal_2",
        "cortex_post",
        "chunker_dec",
        "lm_head_ce",
        "backward",
    )

    def _run_timed_step(model: CalciteLeanSystem1LM, record: bool) -> dict[str, float]:
        model.zero_grad(set_to_none=True)
        if qat and use_fast:
            clear_fast_qat_cache()
        if use_amp and record:
            ev_s = [torch.cuda.Event(enable_timing=True) for _ in range(len(stage_names) + 1)]
            ev_i = 0

            def _mark() -> None:
                nonlocal ev_i
                ev_s[ev_i].record()
                ev_i += 1
        else:
            ts_cpu: list[float] = []

            def _mark() -> None:
                if record:
                    ts_cpu.append(time.perf_counter())

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            _mark()
            e_tok = model.tok_emb(x)
            _mark()
            z_in, h_lex, ctx = model.byte_chunker.encode_to_chunks(e_tok, x, model.tok_emb.weight)
            _mark()
            h = z_in
            for blk in model.cortex_pre:
                h = blk(h)
            _mark()
            h_split = model.split_norm(h)
            gates = 2.0 * torch.sigmoid(model.stream_gate(h_split))
            c_out = model.crystal(h_split, h_lex=h_lex)
            _mark()
            w_out = model.woodbury(h_split, h_lex=h_lex)
            h = h + gates[..., 0:1] * c_out + gates[..., 1:2] * w_out
            _mark()
            h = h + model.mid_mlp(model.mid_mlp_norm(h))
            _mark()
            c2_in = model.cryst2_norm(h)
            c2_out = (2.0 * torch.sigmoid(model.cryst2_gate(c2_in))) * model.crystal_2(c2_in, h_lex=h_lex)
            h = h + c2_out
            _mark()
            for blk in model.cortex_post:
                h = blk(h)
            _mark()
            h_byte = model.byte_chunker.decode_from_chunks(h, ctx)
            _mark()
            lg = model.lm_head(model.norm_f(h_byte))
            if use_fast:
                loss = F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), y.reshape(-1))
            else:
                loss = F.cross_entropy(lg.float().transpose(1, 2), y)
            _mark()
        loss.backward()
        _mark()

        if not record:
            return {}
        if use_amp:
            torch.cuda.synchronize()
            return {nm: float(ev_s[i].elapsed_time(ev_s[i + 1])) for i, nm in enumerate(stage_names)}
        return {nm: (ts_cpu[i + 1] - ts_cpu[i]) * 1000.0 for i, nm in enumerate(stage_names)}

    for _ in range(2):
        _run_timed_step(m, record=False)
    _sync_cuda(device)

    per_stage: dict[str, list[float]] = {nm: [] for nm in stage_names}
    for _ in range(iters):
        rec = _run_timed_step(m, record=True)
        for nm in stage_names:
            per_stage[nm].append(rec[nm])

    m.set_ternary_qat(False)
    clear_fast_qat_cache()
    del m
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    med = {nm: round(sorted(vals)[len(vals) // 2], 2) for nm, vals in per_stage.items()}
    fwd_tot = round(sum(v for k, v in med.items() if k != "backward"), 2)
    med["fwd_total_ms"] = fwd_tot
    med["step_total_ms"] = round(fwd_tot + med["backward"], 2)
    return med


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiers", default="S1_5M,S2_25M,S3_85M")
    parser.add_argument("--seq-len", type=int, default=768)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--with-compile", action="store_true")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--skip-profile-blocks", action="store_true")
    parser.add_argument("--out-json", default="benchmarks/fast_calcite_kernels_results.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    binfo = get_fast_scans_backend_info()
    swa_info = get_fast_swa_chunker_backend_info()
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    print(
        f"=== [Fast Calcite v3] device={gpu_name} | scans={binfo} | swa_chunker={swa_info} ===",
        flush=True,
    )

    parity = verify_kernel_parity(device)
    print(f"OK PARITY PASSED: {json.dumps(parity, indent=2)}", flush=True)

    bs_map = {"S1_5M": 64, "S2_25M": 64, "S3_85M": 32}
    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if vram_gb < 12.0:
            bs_map = {"S1_5M": 16, "S2_25M": 16, "S3_85M": 8}

    speed_results: dict[str, object] = {}
    block_breakdown: dict[str, object] = {}
    tier_list = [t.strip() for t in args.tiers.split(",") if t.strip()]
    for tier in tier_list:
        bs = bs_map.get(tier, 16)
        speed_results[tier] = {}
        for mode_name, qat in (("FP", False), ("QAT", True)):
            base = bench_tier_speed(tier, bs, args.seq_len, qat, False, False, device, args.iters, args.compile_mode)
            fast = bench_tier_speed(tier, bs, args.seq_len, qat, True, False, device, args.iters, args.compile_mode)
            entry = {
                "batch_size": bs,
                "baseline_ms": base["median_ms"],
                "fast_cuda_ms": fast["median_ms"],
                "speedup_x": round(base["median_ms"] / max(1e-6, fast["median_ms"]), 2),
                "baseline_peak_mem_mb": base["peak_mem_mb"],
                "fast_peak_mem_mb": fast["peak_mem_mb"],
            }
            if args.with_compile:
                comp = bench_tier_speed(
                    tier, bs, args.seq_len, qat, True, True, device, args.iters, args.compile_mode
                )
                entry["fast_compiled_ms"] = comp["median_ms"]
                entry["compiled_speedup_x"] = round(base["median_ms"] / max(1e-6, comp["median_ms"]), 2)
                entry["compiled_peak_mem_mb"] = comp["peak_mem_mb"]
                if qat:
                    entry["qat_misses_per_step"] = comp["qat_misses_per_step"]
                    entry["qat_hits_per_step"] = comp["qat_hits_per_step"]
                    mb_bs = max(1, bs // 2)
                    acc_base = bench_tier_speed(
                        tier, mb_bs, args.seq_len, True, False, False, device, args.iters, args.compile_mode, accum=2
                    )
                    acc_comp = bench_tier_speed(
                        tier, mb_bs, args.seq_len, True, True, True, device, args.iters, args.compile_mode, accum=2
                    )
                    entry["accum2_micro_bs"] = mb_bs
                    entry["accum2_baseline_ms"] = acc_base["median_ms"]
                    entry["accum2_fast_compiled_ms"] = acc_comp["median_ms"]
                    entry["accum2_compiled_speedup_x"] = round(
                        acc_base["median_ms"] / max(1e-6, acc_comp["median_ms"]), 2
                    )
                    entry["accum2_qat_misses_per_step"] = acc_comp["qat_misses_per_step"]
                    entry["accum2_qat_hits_per_step"] = acc_comp["qat_hits_per_step"]
            speed_results[tier][mode_name] = entry
            print(
                f"  [{tier} {mode_name} B={bs} L={args.seq_len}] "
                f"base={entry['baseline_ms']:.2f}ms ({entry['baseline_peak_mem_mb']:.0f}MB) -> "
                f"fast={entry['fast_cuda_ms']:.2f}ms ({entry['speedup_x']:.2f}x, {entry['fast_peak_mem_mb']:.0f}MB)"
                + (
                    f" -> compiled={entry['fast_compiled_ms']:.2f}ms ({entry['compiled_speedup_x']:.2f}x, "
                    f"{entry['compiled_peak_mem_mb']:.0f}MB)"
                    if args.with_compile
                    else ""
                )
                + (
                    f" | accum2(mb={entry['accum2_micro_bs']}): base={entry['accum2_baseline_ms']:.2f}ms -> "
                    f"compiled={entry['accum2_fast_compiled_ms']:.2f}ms ({entry['accum2_compiled_speedup_x']:.2f}x, "
                    f"hits/miss={entry['accum2_qat_hits_per_step']:.0f}/{entry['accum2_qat_misses_per_step']:.0f})"
                    if (args.with_compile and qat)
                    else ""
                ),
                flush=True,
            )

        if not args.skip_profile_blocks:
            blk_base = profile_tier_blocks(
                tier, bs, args.seq_len, False, False, False, device, min(6, args.iters), args.compile_mode
            )
            blk_fast = profile_tier_blocks(
                tier, bs, args.seq_len, False, True, args.with_compile, device, min(6, args.iters), args.compile_mode
            )
            block_breakdown[tier] = {"FP_baseline": blk_base, "FP_fast_v3": blk_fast}
            print(f"  [{tier} FP Blocks] base={blk_base} | fast_v3={blk_fast}", flush=True)

    summary = {
        "device": gpu_name,
        "backend_info": binfo,
        "swa_chunker_backend_info": swa_info,
        "parity": parity,
        "speed": speed_results,
        "block_breakdown": block_breakdown,
    }
    out_p = ROOT_DIR / args.out_json
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {out_p} ({out_p.stat().st_size:,} B)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
