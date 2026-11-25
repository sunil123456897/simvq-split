"""
Wave 2 Regression & Parity Verification Suite for `Calcite v3.1` Optimizations
(`tests/test_calcite_v31_optimizations.py`).

Verifies all `[LOSSLESS]` Wave 2 engineering upgrades across `calcite_v2/`, `kernels/`,
`scripts/`, and `experiments/`:
  1. `CalciteRMSNorm` `0.00e+00` BF16 parity, autocast dtype preservation, `isinstance(..., RMSNorm)`,
     and identical `state_dict` keys vs legacy checkpoints.
  2. `FastTernaryQAT` autocast BF16 cache dtype (`0` FP32 bloat), `MuonAdamWHybrid` batched `bmm`
     NS5 parity (`< 1e-4`), `Woodbury wd=0.0`, and post-step QAT cache invalidation (`stale_diff == 0.0`).
  3. `ByteInductionCache` zero-`.item()` lazy 0-d stats, hoisted PyTorch scan parity, and
     Shard-Parallel SMEM `fast_bic_scan_cuda` fwd + bwd parity (`< 1e-5` incl. live STE `bt==0` grad).
  4. `sliding_window_sdpa`, `byte_swa` (`D=32, W=24`) native CUDA fwd/bwd parity (`< 2e-5`),
     `utf8_span_compact_and_offset_cuda` exact match (`0` mid-UTF8 cuts), and `inference_mode -> bwd`
     RoPE cache hygiene.
  5. `CalciteV2Chunker` 4-tap shift-add conv parity (`< 1e-5`), tight eval `M_max` parity (`< 1e-5`),
     shared `precomputed_chunks` QAT fwd/grad parity (`< 1e-5`), and `Crystal` eval guards + `GH=16`.
  6. `CalciteStreamEngine` vectorized `prefill()` parity vs byte-by-byte `step()` & `forward()`
     (`< 2e-5` on multi-block `T=600` FP and 2-bit LUT + strict `O(1)` RAM), lazy `h_lex_t` (`0.0`),
     unsynced `B=2` parity (`< 2e-5`), `FastPackedLUT2BitLinear` C++ AVX-512 / CUDA `B=1` GEMV +
     `fuse_packed_swiglu_for_inference` parity (`< 2e-5`), and GPU CUDA Graph streaming parity.
  7. `eval_suite` RAM loader, fast raw-UTF8 encoding parity, single-pass `OW6` `encode_to_chunks`,
     `compile_calcite_v2` smoke + `unwrap_compiled_state_dict`, and `0`-byte diff on `simvq_split/`.
"""

from __future__ import annotations

import ast
import copy
import math
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from calcite_v2 import (  # noqa: E402
    CalciteRMSNorm,
    CalciteStreamEngine,
    CalciteV2Config,
    CalciteV2CrystalLayer,
    CalciteV2LM,
)
from calcite_v2.byte_induct_cache import ByteInductionCache  # noqa: E402
from calcite_v2.chunker import CalciteV2Chunker, _depthwise_conv4_shift_add  # noqa: E402
from calcite_v2.norm import build_swa_block_masks, sliding_window_sdpa  # noqa: E402
from experiments.scaling_phase_k.eval_suite import (  # noqa: E402
    encode_utf8_fast,
    eval_openweb_6domains_any_byte_lm,
    load_utf8_corpus_array,
)
from kernels.fast_calcite_bic import (  # noqa: E402
    fast_bic_scan_available,
    fast_bic_scan_cuda,
)
from kernels.fast_calcite_scans import (  # noqa: E402
    _QAT_SLOTS,
    clear_fast_qat_cache,
    compile_calcite_v2,
    fast_crystal_forward,
    get_fast_qat_cache_stats,
    unwrap_compiled_state_dict,
)
from kernels.fast_calcite_swa_chunker import (  # noqa: E402
    _load_swa_chunker_cuda_ext,
    _swa_rope_attn_bwd_pytorch,
    _swa_rope_attn_fwd_pytorch,
    fast_swa_rope_attn_cuda,
    get_cached_rope_cos_sin,
    utf8_span_and_compact_cuda,
    utf8_span_compact_and_offset_cuda,
)
from kernels.fast_packed_lut import (  # noqa: E402
    FastPackedLUT2BitLinear,
    fuse_packed_swiglu_for_inference,
    get_lut_backend_info,
    pack_fast_model_2bit_lut,
)
from calcite_v2.model import compute_v2_state_cache_kb  # noqa: E402
from scripts.experiment_velocity import (  # noqa: E402
    OEBEarlyStopper,
    WSDLRSchedule,
    load_plateau_checkpoint,
    run_tiered_eval,
    save_plateau_checkpoint,
)
from scripts.muon_opt import (  # noqa: E402
    MuonAdamWHybrid,
    zeropower_via_newtonschulz5,
    zeropower_via_newtonschulz5_batched,
)
from scripts.repo_guardrail import (  # noqa: E402
    AGENTS_MD_MAX_BYTES,
    check_agents_md_size,
    check_markdown_quarantine,
)
from simvq_split.baselines import RMSNorm  # noqa: E402
from simvq_split.quant_lut import PackedLUT2BitLinear  # noqa: E402
from simvq_split.tokenizer import ByteBPETokenizer  # noqa: E402


def _make_v31_tiny_cfg(
    *,
    d_model: int = 128,
    n_heads: int = 2,
    d_ff: int = 256,
    use_fast_kernels: bool = True,
    byte_induct_slots: int = 128,
    dec_dw3_enabled: bool = True,
    dw_bottleneck_ratio: float = 1.0,
    bf16_residual_stream: bool = False,
) -> CalciteV2Config:
    return CalciteV2Config(
        vocab_size=259,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers_pre=2,
        n_layers_post=2,
        window_size=8,
        num_spheres=8,
        d_sphere=32,
        codebook_size=16,
        crystal_w_chunks=2,
        mem_heads=4,
        mem_dim=16,
        woodbury_latch_span=6,
        byte_char_dim=16,
        byte_d_ff=128,
        byte_router_char_dim=8,
        byte_router_hidden=32,
        byte_chunk_mode="utf8",
        use_static_m_cap=True,
        topology="hourglass",
        use_scaled_decoder=True,
        use_kalman_precond=False,
        use_latent_feedback=False,
        use_clean_gates=False,
        use_fast_kernels=use_fast_kernels,
        use_sym_byte_swa=True,
        byte_swa_window=24,
        byte_swa_dim=64,
        use_byte_engram=True,
        engram_slots=512,
        engram_dim=16,
        use_byte_induct_cache=True,
        byte_induct_slots=byte_induct_slots,
        byte_induct_dim=16,
        dec_dw3_enabled=dec_dw3_enabled,
        dw_bottleneck_ratio=dw_bottleneck_ratio,
        bf16_residual_stream=bf16_residual_stream,
    )


def _seq_bic_reference(
    sq: torch.Tensor,
    sw: torch.Tensor,
    bt: torch.Tensor,
    u_all: torch.Tensor,
    n_slots: int,
) -> torch.Tensor:
    """Exact timestep-by-timestep differentiable BIC recurrence (ground truth)."""
    B, T, K_sc, D_u = u_all.shape
    dev = u_all.device
    bank = torch.zeros((B, K_sc, n_slots, D_u), device=dev, dtype=torch.float32)
    b_idx = torch.arange(B, device=dev, dtype=torch.long)
    outs = []
    for t in range(T):
        row = []
        for k in range(K_sc):
            sw_t = sw[:, t, k].long()
            sq_t = sq[:, t, k].long()
            bt_t = bt[:, t, k].float().unsqueeze(-1)
            old = bank[b_idx, k, sw_t]
            upd = (1.0 - bt_t) * old + bt_t * u_all[:, t, k].float()
            new_bank_k = bank[:, k].clone()
            new_bank_k[b_idx, sw_t] = upd
            bank = torch.stack(
                [new_bank_k if ki == k else bank[:, ki] for ki in range(K_sc)], dim=1
            )
            row.append(bank[b_idx, k, sq_t])
        outs.append(torch.stack(row, dim=1))
    return torch.stack(outs, dim=1)


def test_calcite_rms_norm_parity_and_state_dict() -> None:
    torch.manual_seed(2026)
    ref = RMSNorm(128)
    fast = CalciteRMSNorm(128)
    assert isinstance(fast, RMSNorm)
    assert list(ref.state_dict().keys()) == list(fast.state_dict().keys()) == ["weight"]
    with torch.no_grad():
        ref.weight.uniform_(0.5, 1.5)
        fast.weight.copy_(ref.weight)

    x_fp32 = torch.randn(2, 32, 128, dtype=torch.float32)
    assert float((ref(x_fp32) - fast(x_fp32)).abs().max().item()) < 1e-6

    x_bf16 = x_fp32.bfloat16()
    ref_bf16 = ref(x_bf16).bfloat16()
    fast_bf16 = fast(x_bf16)
    assert fast_bf16.dtype == torch.bfloat16
    # Native F.rms_norm casts weight to bf16 before multiply (within 1-2 BF16 ULPs of FP32-promoted RMSNorm)
    assert float((ref_bf16.float() - fast_bf16.float()).abs().max().item()) <= 0.05

    # All norms in CalciteV2LM must be CalciteRMSNorm (subclassing RMSNorm)
    m = CalciteV2LM(_make_v31_tiny_cfg(use_fast_kernels=False))
    norms = [mod for mod in m.modules() if isinstance(mod, RMSNorm)]
    assert len(norms) >= 25
    assert all(isinstance(mod, CalciteRMSNorm) for mod in norms)

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        ref_c = ref.cuda()
        fast_c = fast.cuda()
        xc = x_fp32.cuda()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            yc_ref = ref_c(xc).bfloat16()
            yc = fast_c(xc)
        assert yc.dtype == torch.bfloat16
        # Exact 0.00e+00 BF16 parity on the FP32 residual autocast path
        assert float((yc_ref.float() - yc.float()).abs().max().item()) == 0.0


def test_fast_qat_bf16_cache_and_muon_invalidation() -> None:
    torch.manual_seed(2026)
    dev = "cuda" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "cpu"

    # 1. Batched Muon NS5 vs serial reference across mixed transposed 2D shapes
    cfg = _make_v31_tiny_cfg(use_fast_kernels=True)
    m_batched = CalciteV2LM(cfg).to(dev)
    m_batched.byte_chunker.lock_router(True)
    m_serial = copy.deepcopy(m_batched)

    opt_b = MuonAdamWHybrid(m_batched, base_lr=1.5e-3, muon_lr_mult=4.0, weight_decay=0.04, device=dev)
    # Verify Woodbury weight_decay == 0.0 invariant
    wood_ids = {id(p) for n, p in m_batched.named_parameters() if "woodbury" in n and p.requires_grad}
    for pg in opt_b.param_groups:
        if any(id(p) in wood_ids for p in pg["params"]):
            assert float(pg["weight_decay"]) == 0.0

    # Assign identical synthetic gradients and compare one step against serial NS5
    x = torch.randint(3, 259, (2, 48), device=dev)
    y = torch.randint(3, 259, (2, 48), device=dev)
    lg_b = m_batched(x)
    F.cross_entropy(lg_b.float().reshape(-1, 259), y.reshape(-1)).backward()
    for p_s, p_b in zip(m_serial.parameters(), m_batched.parameters()):
        if p_b.grad is not None:
            p_s.grad = p_b.grad.detach().clone()

    # Step batched MuonAdamWHybrid (capture post-hook firing order!)
    hook_events: list[str] = []
    h_post = opt_b.register_step_post_hook(lambda _o: hook_events.append("post_hook"))
    opt_b.step()
    h_post.remove()
    assert hook_events == ["post_hook"]

    # Reference serial Muon update on muon_group params
    lr_m = float(opt_b.muon_group["lr"])
    wd_m = float(opt_b.muon_group["weight_decay"])
    beta_m = opt_b.muon_momentum
    muon_names = {
        id(p): nm for nm, p in m_batched.named_parameters()
    }
    serial_map = dict(m_serial.named_parameters())
    max_muon_diff = 0.0
    with torch.no_grad():
        for p_b in opt_b.muon_group["params"]:
            nm = muon_names[id(p_b)]
            p_s = serial_map[nm]
            g = p_s.grad
            if g is None:
                continue
            buf = g.clone()
            g_nest = g.add(buf, alpha=beta_m)
            orth = zeropower_via_newtonschulz5(g_nest, steps=opt_b.ns_steps).to(p_s.dtype)
            scale = 0.2 * (float(max(p_s.size(0), p_s.size(1))) ** 0.5)
            p_s.mul_(1.0 - lr_m * wd_m)
            p_s.add_(orth, alpha=-lr_m * scale)
            max_muon_diff = max(max_muon_diff, float((p_b - p_s).abs().max().item()))
    assert max_muon_diff < 1e-4, f"Batched Muon NS5 drifted from serial ref: {max_muon_diff:.3e}"

    # 2. FastTernaryQAT BF16 autocast cache dtype + MuonAdamWHybrid post-step cache invalidation
    m_qat = CalciteV2LM(cfg).to(dev)
    m_qat.byte_chunker.lock_router(True)
    m_qat.set_crystal_schedule(1, 1)
    m_qat.set_snap_alpha(1.0)
    m_qat.set_qat_mix(1.0)
    m_qat.set_ternary_qat(True)
    opt_q = MuonAdamWHybrid(m_qat, base_lr=3e-3, muon_lr_mult=4.0, weight_decay=0.0, device=dev)

    clear_fast_qat_cache(reset_stats=True)
    use_amp = dev == "cuda"
    for _ in range(2):
        with torch.amp.autocast(dev, dtype=torch.bfloat16, enabled=use_amp):
            lg = m_qat(x)
            loss = F.cross_entropy(lg.float().reshape(-1, 259), y.reshape(-1)) * 0.5
        loss.backward()
    st_pre = get_fast_qat_cache_stats()
    assert st_pre["misses"] > 0 and st_pre["hits"] == st_pre["misses"], f"Expected mb1 100% hits: {st_pre}"

    if use_amp:
        cached_dtypes = {
            e.tensor.dtype for e in _QAT_SLOTS.values() if e.tensor is not None
        }
        assert cached_dtypes == {torch.bfloat16}, f"FP32 QAT cache bloat detected: {cached_dtypes}"

    opt_q.step()
    m_qat.eval()
    with torch.no_grad(), torch.amp.autocast(dev, dtype=torch.bfloat16, enabled=use_amp):
        lg_post = m_qat(x)
    clear_fast_qat_cache()
    with torch.no_grad(), torch.amp.autocast(dev, dtype=torch.bfloat16, enabled=use_amp):
        lg_fresh = m_qat(x)
    stale_diff = float((lg_post.float() - lg_fresh.float()).abs().max().item())
    m_qat.set_ternary_qat(False)
    clear_fast_qat_cache()
    assert stale_diff == 0.0, f"Stale QAT cache after MuonAdamWHybrid.step(): {stale_diff:.3e}"


def test_bic_lazy_stats_and_cuda_scan_parity() -> None:
    # 1. Verify zero `.item()` inside ByteInductionCache.forward via AST
    bic_src = (ROOT_DIR / "calcite_v2" / "byte_induct_cache.py").read_text(encoding="utf-8")
    tree = ast.parse(bic_src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ByteInductionCache":
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "forward":
                    for sub in ast.walk(fn):
                        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                            assert sub.func.attr != "item", f"Forbidden .item() sync in BIC.forward:{sub.lineno}"

    # 2. Lazy 0-d stats + optimized PyTorch scan parity vs exact sequential recurrence
    torch.manual_seed(7)
    bic = ByteInductionCache(vocab_size=259, d_model=64, n_slots=128, d_sig=8, d_v=16)
    ids = torch.randint(3, 259, (2, 80))
    h = torch.randn(2, 80, 64)
    _ = bic(ids, h)
    assert isinstance(bic._last_write_rate_t, torch.Tensor) and bic._last_write_rate_t.ndim == 0
    assert isinstance(bic._last_write_rate_s1_t, torch.Tensor) and bic._last_write_rate_s1_t.ndim == 0
    wr = bic.last_write_rate
    wr1 = bic.last_write_rate_s1
    assert 0.0 <= wr <= 1.0 and 0.0 <= wr1 <= 1.0 and bic._last_write_rate_t is None

    sq = torch.randint(0, 64, (2, 70, 2), dtype=torch.long)
    sw = torch.randint(0, 64, (2, 70, 2), dtype=torch.long)
    bt = torch.rand(2, 70, 2) * 0.9
    bt[:, 0, :] = 0.0
    u_all = torch.randn(2, 70, 2, 25)
    out_blk = bic._scan_blocks_pytorch(sq, sw, bt, u_all)
    out_seq = _seq_bic_reference(sq, sw, bt, u_all, n_slots=64)
    assert float((out_blk - out_seq).abs().max().item()) < 1e-5

    # 3. Native CUDA Shard-Parallel SMEM BIC fwd + bwd parity (incl. live bt==0 STE gradient!)
    if torch.cuda.is_available() and fast_bic_scan_available():
        for n_slots, T in ((64, 80), (512, 300)):
            sq_c = torch.randint(0, n_slots, (2, T, 2), device="cuda", dtype=torch.long)
            sw_c = torch.randint(0, n_slots, (2, T, 2), device="cuda", dtype=torch.long)
            raw_bt = torch.rand(2, T, 2, device="cuda")
            bt_mask = (raw_bt > 0.7).float()
            bt_vals = raw_bt * bt_mask * 0.9999
            bt_vals[:, 0, :] = 0.0

            bt_cuda = bt_vals.clone().requires_grad_(True)
            u_cuda = torch.randn(2, T, 2, 25, device="cuda", requires_grad=True)
            bt_ref = bt_vals.clone().requires_grad_(True)
            u_ref = u_cuda.detach().clone().requires_grad_(True)

            out_cuda = fast_bic_scan_cuda(sq_c, sw_c, bt_cuda, u_cuda, n_slots=n_slots)
            out_ref = _seq_bic_reference(sq_c, sw_c, bt_ref, u_ref, n_slots=n_slots)
            assert float((out_cuda - out_ref).abs().max().item()) < 1e-5

            go = torch.randn_like(out_cuda)
            (out_cuda * go).sum().backward()
            (out_ref * go).sum().backward()
            assert float((bt_cuda.grad - bt_ref.grad).abs().max().item()) < 1e-5
            assert float((u_cuda.grad - u_ref.grad).abs().max().item()) < 1e-5
            # Verify non-zero STE gradient flows through bt == 0 positions
            zero_mask = bt_vals == 0.0
            assert float(bt_cuda.grad[zero_mask].abs().max().item()) > 0.0


def test_swa_d32_w24_and_utf8_offset_cuda_parity() -> None:
    torch.manual_seed(11)

    # 1. sliding_window_sdpa fwd + bwd parity vs exact causal band attention
    for H, D, W, T in ((4, 64, 8, 137), (2, 32, 24, 150)):
        q = torch.randn(2, H, T, D, requires_grad=True)
        k = torch.randn(2, H, T, D, requires_grad=True)
        v = torch.randn(2, H, T, D, requires_grad=True)
        mask0, mask_inner, bq = build_swa_block_masks(W)
        y_fast = sliding_window_sdpa(q, k, v, W, mask0, mask_inner, bq)

        t_i = torch.arange(T).unsqueeze(1)
        t_j = torch.arange(T).unsqueeze(0)
        full_mask = (t_i >= t_j) & ((t_i - t_j) < W)
        q2, k2, v2 = q.detach().clone().requires_grad_(True), k.detach().clone().requires_grad_(True), v.detach().clone().requires_grad_(True)
        y_ref = F.scaled_dot_product_attention(q2, k2, v2, attn_mask=full_mask)
        assert float((y_fast - y_ref).abs().max().item()) < 1e-5
        go = torch.randn_like(y_fast)
        (y_fast * go).sum().backward()
        (y_ref * go).sum().backward()
        assert float((q.grad - q2.grad).abs().max().item()) < 1e-5
        assert float((k.grad - k2.grad).abs().max().item()) < 1e-5
        assert float((v.grad - v2.grad).abs().max().item()) < 1e-5

    # 2. RoPE cache hygiene: inference_mode() -> autograd backward() must not fail (#23 regression)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.inference_mode():
        c_inf, s_inf = get_cached_rope_cos_sin(160, dev, torch.float32, head_dim=32)
    assert not c_inf.is_inference() and not s_inf.is_inference()
    qkv_inf = torch.randn(2, 160, 3, 2, 32, device=dev, requires_grad=True)
    y_inf = fast_swa_rope_attn_cuda(qkv_inf, c_inf, s_inf, 24)
    y_inf.sum().backward()
    assert qkv_inf.grad is not None and torch.isfinite(qkv_inf.grad).all()

    # 3. Native CUDA D=32, W=24 & W=32 SWA fwd+bwd parity + utf8_span_compact_and_offset_cuda
    if torch.cuda.is_available() and _load_swa_chunker_cuda_ext() is not None:
        for D, W in ((32, 24), (32, 32), (64, 24)):
            T = 137
            qkv = torch.randn(2, T, 3, 2, D, device="cuda", dtype=torch.float32, requires_grad=True)
            qkv_ref = qkv.detach().clone().requires_grad_(True)
            cos, sin = get_cached_rope_cos_sin(T, qkv.device, qkv.dtype, head_dim=D)
            y_cu = fast_swa_rope_attn_cuda(qkv, cos, sin, W)
            y_py, lse_py = _swa_rope_attn_fwd_pytorch(qkv_ref, cos, sin, W)
            assert float((y_cu - y_py).abs().max().item()) < 2e-5
            go = torch.randn_like(y_cu)
            y_cu.backward(go)
            dqkv_py = _swa_rope_attn_bwd_pytorch(qkv_ref, cos, sin, y_py, lse_py, go, W)
            assert float((qkv.grad - dqkv_py).abs().max().item()) < 2e-5

        # UTF-8 span + offset exact match (ASCII + Russian + CJK + Emoji + PAD)
        text = "Calcite v3.1 Привет мир! 東京 🚀 " * 6
        raw = [b + 3 for b in text.encode("utf-8")][:192]
        raw_pad = raw[:160] + [0] * (192 - 160)
        toks = torch.tensor([raw, raw_pad], device="cuda", dtype=torch.long)
        probs = torch.rand(2, 192, device="cuda")
        bm4, cnt4, slot4, gpos4 = utf8_span_and_compact_cuda(toks, probs, 0, 10, 16, 100, 331, 1324, 96)
        bm5, cnt5, slot5, gpos5, off5 = utf8_span_compact_and_offset_cuda(toks, probs, 0, 10, 16, 100, 331, 1324, 96)
        off_ref = CalciteV2Chunker._compute_offset_in_chunk(bm4, 16)
        assert torch.equal(bm4, bm5) and torch.equal(cnt4, cnt5)
        assert torch.equal(slot4, slot5) and torch.equal(gpos4, gpos5)
        assert torch.equal(off5, off_ref)
        # 0 mid-UTF8 cuts
        raw_b = (toks - 3).clamp(min=0)
        is_cont = (toks >= 3) & ((raw_b & 0xC0) == 0x80)
        assert int((bm5[:, 1:] & is_cont[:, 1:]).sum().item()) == 0


def test_chunker_shift_add_tight_eval_and_shared_qat_chunks() -> None:
    torch.manual_seed(17)

    # 1. 4-tap shift-add parity vs nn.Conv1d for dilations d=1, 2, 4
    for dil in (1, 2, 4):
        conv = nn.Conv1d(64, 64, kernel_size=4, dilation=dil, groups=64, bias=True)
        x1 = torch.randn(2, 53, 64, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)
        y_sa = _depthwise_conv4_shift_add(x1, conv, dil)
        y_cv = conv(F.pad(x2.transpose(1, 2), (3 * dil, 0))).transpose(1, 2)
        assert float((y_sa - y_cv).abs().max().item()) < 1e-5
        go = torch.randn_like(y_sa)
        (y_sa * go).sum().backward()
        (y_cv * go).sum().backward()
        assert float((x1.grad - x2.grad).abs().max().item()) < 1e-5

    # 2. Tight M_max in eval matches static M_cap forward + saves padded chunks
    cfg = _make_v31_tiny_cfg(use_fast_kernels=True)
    m = CalciteV2LM(cfg).eval()
    m.byte_chunker.lock_router(True)
    m.set_crystal_schedule(1, 1)
    m.set_snap_alpha(1.0)
    x = torch.randint(3, 259, (2, 160))
    with torch.no_grad():
        lg_tight = m(x)
        m_tight = m.byte_chunker.last_m_cap_used
    assert m_tight < m.byte_chunker.static_m_cap_for_T(160)
    # Compare tight eval M_max against static M_cap while keeping backbone/Crystal in identical eval mode
    m.byte_chunker.training = True
    try:
        with torch.no_grad():
            lg_static = m(x)
            m_stat = m.byte_chunker.last_m_cap_used
    finally:
        m.byte_chunker.training = False
    assert m_stat == m.byte_chunker.static_m_cap_for_T(160)
    assert float((lg_tight - lg_static).abs().max().item()) < 1e-5

    # 3. Shared precomputed_chunks QAT teacher -> student parity (logits & all param grads)
    #    + static M_max preservation under eval `return_chunks=True` (#40/#41 regression)
    teacher = copy.deepcopy(m).eval()
    student_a = copy.deepcopy(m).train()
    student_b = copy.deepcopy(m).train()
    x4 = torch.randint(3, 259, (4, 96))
    y4 = torch.randint(3, 259, (4, 96))
    with torch.inference_mode():
        lg_t_tight = teacher(x4, return_chunks=False)
        m_t_tight = teacher.byte_chunker.last_m_cap_used
        lg_t_stat, chunks_full = teacher(x4, return_chunks=True)
    m_stat_96 = teacher.byte_chunker.static_m_cap_for_T(96)
    assert m_t_tight < m_stat_96
    assert int(chunks_full["M_max"]) == m_stat_96
    assert float((lg_t_tight - lg_t_stat).abs().max().item()) < 1e-5
    ch_mb = student_a.byte_chunker.slice_precomputed_chunks(chunks_full, 0, 2)
    lg_shared = student_a(x4[:2], precomputed_chunks=ch_mb)
    lg_direct = student_b(x4[:2])
    assert float((lg_shared - lg_direct).abs().max().item()) < 1e-5
    F.cross_entropy(lg_shared.reshape(-1, 259), y4[:2].reshape(-1)).backward()
    F.cross_entropy(lg_direct.reshape(-1, 259), y4[:2].reshape(-1)).backward()
    max_gdiff = max(
        float((pa.grad - pb.grad).abs().max().item())
        for pa, pb in zip(student_a.parameters(), student_b.parameters())
        if pa.grad is not None and pb.grad is not None
    )
    assert max_gdiff < 1e-5

    # 4. Crystal eval guard (skips commitment loss) + GH=16 (num_spheres=32, S3_85M) parity
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cr32 = CalciteV2CrystalLayer(d_model=128, num_spheres=32, d_sphere=32, codebook_size=32, use_fast_kernels=True).to(dev)
    cr32.snap_alpha = 1.0
    xc = torch.randn(2, 40, 128, device=dev)
    cr32.eval()
    with torch.no_grad():
        y_ev = fast_crystal_forward(cr32, xc)
    assert cr32.last_commit_loss is None
    cr32_ref = copy.deepcopy(cr32).eval()
    cr32_ref.use_fast_kernels = False
    with torch.no_grad():
        y_ref = cr32_ref(xc)
    assert float((y_ev - y_ref).abs().max().item()) < 1e-5


def test_stream_engine_prefill_lazy_lex_b2_and_packed_lut() -> None:
    torch.manual_seed(29)
    cfg = _make_v31_tiny_cfg(use_fast_kernels=False, byte_induct_slots=128)
    m_fp = CalciteV2LM(cfg).eval()
    m_fp.byte_chunker.lock_router(True)
    m_fp.set_crystal_schedule(1, 1)
    m_fp.set_snap_alpha(1.0)

    # 1. Lazy h_lex_t exact 0.0 parity
    st_a = m_fp.byte_chunker.init_stream_state(1, torch.device("cpu"), torch.float32)
    st_b = m_fp.byte_chunker.init_stream_state(1, torch.device("cpu"), torch.float32)
    tok_t = torch.tensor([68], dtype=torch.long)
    e_t = m_fp.tok_emb(tok_t)
    fa, pa, ha, ca, la, oa, va = m_fp.byte_chunker.encode_byte_step(tok_t, e_t, st_a, compute_lex=True)
    fb, pb, hb, cb, w8b, ob, vb = m_fp.byte_chunker.encode_byte_step(tok_t, e_t, st_b, compute_lex=False)
    lb = m_fp.byte_chunker.compute_lex_from_win8(w8b)
    assert torch.equal(fa, fb) and torch.equal(oa, ob)
    assert float((la - lb).abs().max().item()) == 0.0

    # 2. Vectorized prefill + continuation parity on T=600 (multi-block BIC!) for FP & 2-bit LUT
    utf8_s = "Calcite v3.1 streaming check: Привет, мир! 東京 🚀 key: X7K9Q2M4. " * 10
    seq_ids = [b + 3 for b in utf8_s.encode("utf-8")][:600]
    x_full = torch.tensor([seq_ids], dtype=torch.long)

    m_lut = pack_fast_model_2bit_lut(m_fp, backend="auto").eval()
    for tag, mod in (("FP", m_fp), ("LUT", m_lut)):
        with torch.no_grad():
            lg_fwd = mod(x_full)
            eng = CalciteStreamEngine(mod)
            lg_pre, st = eng.prefill(x_full[:, :560], return_all_logits=True)
            mem_pre = st.memory_bytes()
            step_lgs = []
            for t in range(560, 600):
                lg_t, st = eng.step(x_full[:, t], st)
                step_lgs.append(lg_t)
            mem_end = st.memory_bytes()
            lg_all = torch.cat([lg_pre, torch.stack(step_lgs, dim=1)], dim=1)
        err = float((lg_fwd - lg_all).abs().max().item())
        assert mem_pre == mem_end, f"{tag} violated strict O(1) RAM: {mem_pre} != {mem_end}"
        assert err < 2e-5, f"{tag} prefill+step parity failed on T=600: max_err={err:.3e}"

    # 3. Unsynced B=2 batch streaming parity (regression test for 0.30 bug)
    x_b2 = torch.randint(3, 259, (2, 64))
    with torch.no_grad():
        lg_b2_fwd = m_fp(x_b2)
        lg_b2_str, _ = CalciteStreamEngine(m_fp).prefill(x_b2, return_all_logits=True)
    assert float((lg_b2_fwd - lg_b2_str).abs().max().item()) < 2e-5

    # 4. FastPackedLUT2BitLinear native cpu_ext (AVX-512/AVX2) & cuda_ext (B=1 GEMV + tiled) + fused SwiGLU
    binfo = get_lut_backend_info()
    for in_f, out_f, has_bias in ((128, 256, False), (172, 160, True)):
        lin = nn.Linear(in_f, out_f, bias=has_bias)
        ref_mod = PackedLUT2BitLinear.from_linear(lin)
        assert isinstance(ref_mod, PackedLUT2BitLinear)
        fast_ref = FastPackedLUT2BitLinear.from_packed(ref_mod, backend="reference")
        for n_tok in (1, 4, 33):
            x_t = torch.randn(n_tok, in_f)
            with torch.no_grad():
                y_ref = fast_ref(x_t)
                if binfo["cpu_ext_available"]:
                    y_cpu = FastPackedLUT2BitLinear.from_packed(ref_mod, backend="cpu_ext")(x_t)
                    assert float((y_cpu - y_ref).abs().max().item()) < 2e-5
                if torch.cuda.is_available() and binfo["cuda_ext_available"]:
                    y_cu = FastPackedLUT2BitLinear.from_packed(ref_mod, backend="cuda_ext").cuda()(x_t.cuda()).cpu()
                    assert float((y_cu - y_ref).abs().max().item()) < 2e-5

    m_lut_fused = fuse_packed_swiglu_for_inference(m_lut, inplace=False)
    with torch.no_grad():
        err_swi = float((m_lut(x_full[:, :64]) - m_lut_fused(x_full[:, :64])).abs().max().item())
    assert err_swi < 2e-5, f"FusedPackedSwiGLU parity failed: {err_swi:.3e}"

    # 5. GPU CUDA Graph streaming parity (when CUDA + cuda_ext available)
    if torch.cuda.is_available() and binfo["cuda_ext_available"]:
        m_gpu = pack_fast_model_2bit_lut(m_fp, backend="cuda_ext").cuda().eval()
        x_g = x_full[:, :96].cuda()
        with torch.no_grad():
            lg_g_fwd = m_gpu(x_g)
            eng_g = CalciteStreamEngine(m_gpu, use_cuda_graph=True)
            lg_g_pre, st_g = eng_g.prefill(x_g[:, :64], return_all_logits=True)
            g_steps = []
            for t in range(64, 96):
                lg_t, st_g = eng_g.step(x_g[:, t], st_g)
                g_steps.append(lg_t)
            lg_g_all = torch.cat([lg_g_pre, torch.stack(g_steps, dim=1)], dim=1)
            eng_g.enable_cuda_graph(False, state=st_g)
        assert float((lg_g_fwd - lg_g_all).abs().max().item()) < 2e-5


def test_eval_suite_ram_loader_fast_utf8_and_compile_v2(tmp_path: Path) -> None:
    # 1. RAM loader bit-exact vs memmap
    arr_src = np.arange(1024, dtype=np.uint16) + 3
    bin_p = tmp_path / "tiny_utf8.u16.bin"
    arr_src.tofile(bin_p)
    loaded = load_utf8_corpus_array(bin_p, max_ram_mb=16)
    mm = np.memmap(bin_p, dtype=np.uint16, mode="r")
    assert np.array_equal(loaded, mm)
    del mm

    # 2. encode_utf8_fast bit-exact vs ByteBPETokenizer(merges=[])
    tok = ByteBPETokenizer(merges=[])
    for sample in ("Hello world!", "Привет, мир! 東京 🚀 2026", "def f(x):\n    return x + 1\n"):
        for bos in (False, True):
            for eos in (False, True):
                assert encode_utf8_fast(tok, sample, add_bos=bos, add_eos=eos) == tok.encode(
                    sample, add_bos=bos, add_eos=eos
                )

    # 3. eval_openweb_6domains_any_byte_lm calls encode_to_chunks ONCE per batch (no duplicate!)
    cfg = _make_v31_tiny_cfg(use_fast_kernels=False)
    m = CalciteV2LM(cfg).eval()
    m.byte_chunker.lock_router(True)
    m.set_crystal_schedule(1, 1)
    m.set_snap_alpha(1.0)
    win = encode_utf8_fast(tok, "Привет, мир! 東京 🚀 Calcite v3.1 test " * 4)[:65]
    eval_wins = {"dom_a": [win, win], "dom_b": [win, win]}
    call_cnt = [0]
    orig_enc = m.byte_chunker.encode_to_chunks

    def _counted(*a, **kw):
        call_cnt[0] += 1
        return orig_enc(*a, **kw)

    m.byte_chunker.encode_to_chunks = _counted
    res = eval_openweb_6domains_any_byte_lm(m, eval_wins, device="cpu", batch_size=2)
    m.byte_chunker.encode_to_chunks = orig_enc
    assert call_cnt[0] == 2, f"Expected 1 encode_to_chunks call per batch (2 total), got {call_cnt[0]}"
    assert all(float(v["mid_utf8_split_pct"]) == 0.0 for v in res["regimes"].values())

    # 4. compile_calcite_v2 smoke + compiled-vs-eager step-1 grad parity (#59 Woodbury h_lex regression)
    if torch.cuda.is_available():
        try:
            import triton  # noqa: F401
        except ImportError:
            triton = None
        if triton is not None:
            torch.manual_seed(5901)
            cfg_cmp = _make_v31_tiny_cfg(use_fast_kernels=True, bf16_residual_stream=False)
            me = CalciteV2LM(cfg_cmp).cuda().train()
            me.byte_chunker.lock_router(True)
            me.set_crystal_schedule(1, 1)
            me.set_snap_alpha(1.0)
            mc = copy.deepcopy(me).cuda().train()
            mc.byte_chunker.lock_router(True)
            ok = compile_calcite_v2(mc, mode="default")
            assert ok is True
            # Include a 60% PAD=0 tail row (exact #59/#63 Woodbury h_lex==0 Inductor trigger)
            sample_ids = [b + 3 for b in ("Calcite v3.1 compiled grad parity check. Привет! " * 4).encode("utf-8")][:128]
            pad_row = sample_ids[:48] + [0] * (128 - 48)
            xc = torch.tensor([sample_ids, pad_row], device="cuda", dtype=torch.long)
            yc = torch.where(xc == 0, torch.zeros_like(xc), torch.roll(xc, shifts=-1, dims=1))

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                lg_e = me(xc)
                loss_e = F.cross_entropy(lg_e.float().reshape(-1, 259), yc.reshape(-1)) + 0.04 * me.get_commitment_loss()
            loss_e.backward()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                lg_c = mc(xc)
                loss_c = F.cross_entropy(lg_c.float().reshape(-1, 259), yc.reshape(-1)) + 0.04 * mc.get_commitment_loss()
            loss_c.backward()

            assert float((lg_c.float() - lg_e.float()).abs().max().item()) < 5e-2
            gn_e = math.sqrt(sum(float(p.grad.float().pow(2).sum().item()) for p in me.parameters() if p.grad is not None))
            gn_c = math.sqrt(sum(float(p.grad.float().pow(2).sum().item()) for p in mc.parameters() if p.grad is not None))
            gn_ratio = gn_c / max(1e-8, gn_e)
            assert 0.85 <= gn_ratio <= 1.15, f"Compiled grad_norm corrupted vs eager: {gn_c:.4f} vs {gn_e:.4f} ({gn_ratio:.2f}x)"

            gl_e = me.byte_chunker.lex_up.weight.grad.float()
            gl_c = mc.byte_chunker.lex_up.weight.grad.float()
            lex_ratio = float(gl_c.norm().item()) / max(1e-8, float(gl_e.norm().item()))
            lex_cos = float(F.cosine_similarity(gl_c.reshape(1, -1), gl_e.reshape(1, -1)).item())
            assert 0.80 <= lex_ratio <= 1.25 and lex_cos > 0.98, (
                f"Compiled lex_up.weight.grad corrupted (#59 regression): ratio={lex_ratio:.2f}x cos={lex_cos:.4f}"
            )
            sd = unwrap_compiled_state_dict(mc)
            assert all("_orig_mod" not in k for k in sd)


def test_wave2_simvq_split_zero_diff() -> None:
    chk = subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", "simvq_split/"],
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
    )
    assert chk.returncode == 0, f"Protected simvq_split/ modified:\n{chk.stdout}\n{chk.stderr}"


def test_wave3_defaults_and_dw_via_negativa_streaming() -> None:
    torch.manual_seed(301)

    # 1. Default CalciteV2Config() preserves exact Wave 2 architecture & state_dict keys
    dcfg = CalciteV2Config()
    assert dcfg.dec_dw3_enabled is True
    assert dcfg.dw_bottleneck_ratio == 1.0
    assert dcfg.bf16_residual_stream is False

    cfg_c0 = _make_v31_tiny_cfg(use_fast_kernels=False)
    m_c0 = CalciteV2LM(cfg_c0).eval()
    sd_keys = set(m_c0.state_dict().keys())
    assert "byte_chunker.dec_dw3_conv.weight" in sd_keys
    assert "byte_chunker.dec_dw_gate.weight" in sd_keys
    assert not any("_in_gate" in k for k in sd_keys)
    c0_params = sum(p.numel() for p in m_c0.parameters())
    c0_cache_kb = compute_v2_state_cache_kb(cfg_c0)
    c0_mem_b = CalciteStreamEngine(m_c0).init_state(1, torch.device("cpu"), torch.float32).memory_bytes()

    utf8_s = "Calcite v3.1 Wave3 Axis A streaming check: Привет, мир! 東京 🚀 key=Q9W2E4R6. " * 4
    x = torch.tensor([[b + 3 for b in utf8_s.encode("utf-8")][:200]], dtype=torch.long)
    binfo = get_lut_backend_info()

    # 2. A1 (dec_dw3_enabled=False), A2 (dw_bottleneck_ratio=0.5), and A1+A2 combo
    for tag, dec_dw3, bn_ratio in (
        ("A1_no_dec_dw3", False, 1.0),
        ("A2_dw_bn05", True, 0.5),
        ("A1_A2_combo", False, 0.5),
    ):
        cfg = _make_v31_tiny_cfg(
            use_fast_kernels=False,
            dec_dw3_enabled=dec_dw3,
            dw_bottleneck_ratio=bn_ratio,
        )
        m = CalciteV2LM(cfg).eval()
        m.byte_chunker.lock_router(True)
        m.set_crystal_schedule(1, 1)
        m.set_snap_alpha(1.0)
        assert sum(p.numel() for p in m.parameters()) < c0_params
        assert compute_v2_state_cache_kb(cfg) < c0_cache_kb
        if not dec_dw3:
            assert m.byte_chunker.dec_dw3_conv is None
            assert m.byte_chunker.byte_swa_qkv is not None
        if bn_ratio < 1.0:
            assert m.byte_chunker.d_dw == 64
            assert m.byte_chunker.dec_dw_in_gate is not None and m.byte_chunker.dec_dw_gate is None

        m_lut = pack_fast_model_2bit_lut(m, backend="auto").eval()
        for mode_tag, mod in (("FP", m), ("LUT", m_lut)):
            with torch.no_grad():
                lg_fwd = mod(x)
                eng = CalciteStreamEngine(mod)
                lg_pre, st = eng.prefill(x[:, :160], return_all_logits=True)
                mem0 = st.memory_bytes()
                steps = []
                for t in range(160, 200):
                    lt, st = eng.step(x[:, t], st)
                    steps.append(lt)
                mem1 = st.memory_bytes()
                lg_str = torch.cat([lg_pre, torch.stack(steps, dim=1)], dim=1)
            err = float((lg_fwd - lg_str).abs().max().item())
            assert mem0 == mem1 and mem1 < c0_mem_b, f"{tag}/{mode_tag} O(1) RAM fail: {mem0} vs {mem1} vs {c0_mem_b}"
            assert err < 2e-5, f"{tag}/{mode_tag} prefill+step parity failed: {err:.3e}"

        if torch.cuda.is_available() and binfo["cuda_ext_available"]:
            m_gpu = pack_fast_model_2bit_lut(m, backend="cuda_ext").cuda().eval()
            xg = x[:, :96].cuda()
            with torch.no_grad():
                lg_gf = m_gpu(xg)
                eng_g = CalciteStreamEngine(m_gpu, use_cuda_graph=True)
                lg_gp, st_g = eng_g.prefill(xg[:, :64], return_all_logits=True)
                g_steps = []
                for t in range(64, 96):
                    lt, st_g = eng_g.step(xg[:, t], st_g)
                    g_steps.append(lt)
                lg_gs = torch.cat([lg_gp, torch.stack(g_steps, dim=1)], dim=1)
                eng_g.enable_cuda_graph(False, state=st_g)
            assert float((lg_gf - lg_gs).abs().max().item()) < 2e-5


def test_wave3_bf16_residual_stream_gate2_and_grads() -> None:
    torch.manual_seed(302)
    use_cuda = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dev_type = "cuda" if use_cuda else "cpu"

    cfg_fp = _make_v31_tiny_cfg(use_fast_kernels=use_cuda, bf16_residual_stream=False)
    cfg_bf = _make_v31_tiny_cfg(use_fast_kernels=use_cuda, bf16_residual_stream=True)
    m_fp = CalciteV2LM(cfg_fp).to(dev_type).train()
    m_bf = CalciteV2LM(cfg_bf).to(dev_type).train()
    m_bf.load_state_dict(m_fp.state_dict(), strict=True)
    m_fp.byte_chunker.lock_router(True)
    m_bf.byte_chunker.lock_router(True)
    m_fp.set_crystal_schedule(1, 1)
    m_bf.set_crystal_schedule(1, 1)
    m_fp.set_snap_alpha(1.0)
    m_bf.set_snap_alpha(1.0)

    x = torch.randint(3, 259, (2, 64), device=dev_type)
    y = torch.randint(3, 259, (2, 64), device=dev_type)

    # Without autocast, bf16_residual_stream=True is bit-identical to False
    with torch.no_grad():
        assert float((m_fp(x) - m_bf(x)).abs().max().item()) == 0.0

    # Under BF16 autocast: verify residual reaches norm_f in BF16, Gate-2 std <= 1.5x, finite grads
    seen_dtypes: list[torch.dtype] = []
    hk = m_bf.norm_f.register_forward_pre_hook(lambda _m, inp: seen_dtypes.append(inp[0].dtype))
    try:
        with torch.amp.autocast(dev_type, dtype=torch.bfloat16):
            lg_fp = m_fp(x)
            lg_bf = m_bf(x)
            loss_bf = F.cross_entropy(lg_bf.float().reshape(-1, 259), y.reshape(-1))
    finally:
        hk.remove()
    assert seen_dtypes == [torch.bfloat16], f"Expected BF16 residual at norm_f, got {seen_dtypes}"
    assert torch.isfinite(lg_bf).all()
    std_ratio = float(lg_bf.float().std().item()) / max(1e-6, float(lg_fp.float().std().item()))
    assert (1.0 / 1.5) <= std_ratio <= 1.5, f"Gate-2 init-std ratio out of bounds: {std_ratio:.4f}"
    loss_bf.backward()
    grad_params = [p for p in m_bf.parameters() if p.requires_grad and p.grad is not None]
    assert len(grad_params) > 20
    assert all(torch.isfinite(p.grad).all() for p in grad_params)


def test_wave3_polar_express_muon_and_qat_invalidation() -> None:
    torch.manual_seed(303)
    dev = "cuda" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "cpu"

    # 1. Polar Express spectral orthogonality & batched==serial parity
    G_rect = torch.randn(3, 64, 160, device=dev)
    G_sq = torch.randn(2, 64, 64, device=dev)
    maes: dict[str, float] = {}
    for sched in ("quintic5", "polar4", "polar3"):
        B_out = zeropower_via_newtonschulz5_batched(G_rect, ns_schedule=sched)
        S_out = torch.stack(
            [zeropower_via_newtonschulz5(G_rect[i], ns_schedule=sched) for i in range(G_rect.size(0))]
        )
        assert float((B_out.float() - S_out.float()).abs().max().item()) < 1e-4
        sv_rect = torch.linalg.svdvals(B_out[0].float())
        sv_sq = torch.linalg.svdvals(zeropower_via_newtonschulz5_batched(G_sq, ns_schedule=sched)[0].float())
        assert torch.isfinite(sv_rect).all() and torch.isfinite(sv_sq).all()
        assert float(sv_rect.max().item()) < 1.25 and float(sv_sq.max().item()) < 1.25
        maes[sched] = float((sv_rect - 1.0).abs().mean().item())
    assert maes["polar4"] < maes["quintic5"] and maes["polar4"] < 0.15
    assert maes["polar3"] < maes["quintic5"] and maes["polar3"] < 0.15

    with pytest.raises(ValueError):
        zeropower_via_newtonschulz5(G_rect[0], ns_schedule="invalid_sched")

    # 2. MuonAdamWHybrid polar4/polar3: Woodbury wd==0.0, state_dict persistence, QAT cache invalidation
    cfg = _make_v31_tiny_cfg(use_fast_kernels=True)
    x = torch.randint(3, 259, (2, 48), device=dev)
    y = torch.randint(3, 259, (2, 48), device=dev)
    use_amp = dev == "cuda"
    for sched in ("polar4", "polar3"):
        m = CalciteV2LM(cfg).to(dev)
        m.byte_chunker.lock_router(True)
        m.set_crystal_schedule(1, 1)
        m.set_snap_alpha(1.0)
        m.set_qat_mix(1.0)
        m.set_ternary_qat(True)
        opt = MuonAdamWHybrid(m, base_lr=3e-3, muon_lr_mult=4.0, weight_decay=0.04, device=dev, ns_schedule=sched)
        wood_ids = {id(p) for n, p in m.named_parameters() if "woodbury" in n and p.requires_grad}
        for pg in opt.param_groups:
            if any(id(p) in wood_ids for p in pg["params"]):
                assert float(pg["weight_decay"]) == 0.0
        assert opt.state_dict()["ns_schedule"] == sched

        clear_fast_qat_cache(reset_stats=True)
        with torch.amp.autocast(dev, dtype=torch.bfloat16, enabled=use_amp):
            lg = m(x)
            loss = F.cross_entropy(lg.float().reshape(-1, 259), y.reshape(-1))
        loss.backward()
        opt.step()
        m.eval()
        with torch.no_grad(), torch.amp.autocast(dev, dtype=torch.bfloat16, enabled=use_amp):
            lg_post = m(x)
        clear_fast_qat_cache()
        with torch.no_grad(), torch.amp.autocast(dev, dtype=torch.bfloat16, enabled=use_amp):
            lg_fresh = m(x)
        stale_diff = float((lg_post.float() - lg_fresh.float()).abs().max().item())
        m.set_ternary_qat(False)
        clear_fast_qat_cache()
        assert stale_diff == 0.0, f"Stale QAT cache after MuonAdamWHybrid({sched}).step(): {stale_diff:.3e}"


def test_wave3_experiment_velocity_and_agents_guardrail(tmp_path: Path) -> None:
    torch.manual_seed(304)

    # 1. WSDLRSchedule (cosine & 1-sqrt)
    for dmode in ("cosine", "1-sqrt"):
        sched = WSDLRSchedule(
            base_lr=2e-3, min_lr=2e-4, total_steps=100, warmup_steps=10, decay_steps=30, decay_mode=dmode
        )
        assert sched.stable_end == 70
        assert sched.is_plateau_save_step(70) and not sched.is_plateau_save_step(69)
        assert abs(sched.lr_at(1) - 2e-4) < 1e-12
        assert abs(sched.lr_at(10) - 2e-3) < 1e-12
        assert abs(sched.lr_at(70) - 2e-3) < 1e-12
        assert abs(sched.lr_at(100) - 2e-4) < 1e-12
        assert sched.lr_at(85) < 2e-3 and sched.lr_at(85) > 2e-4
        assert WSDLRSchedule.from_dict(sched.to_dict()).to_dict() == sched.to_dict()

    # 2. Plateau checkpoint save/load roundtrip + Law 8a S1A+BIC guard + quintic5->polar4 branching
    cfg = _make_v31_tiny_cfg(use_fast_kernels=False)
    m1 = CalciteV2LM(cfg)
    m1.byte_chunker.lock_router(True)
    opt1 = MuonAdamWHybrid(m1, base_lr=1e-3, device="cpu", ns_schedule="quintic5")
    x = torch.randint(3, 259, (2, 32))
    y = torch.randint(3, 259, (2, 32))
    F.cross_entropy(m1(x).reshape(-1, 259), y.reshape(-1)).backward()
    opt1.step()

    with pytest.raises(ValueError, match="Law 8a"):
        save_plateau_checkpoint(tmp_path / "bad_s1a.pt", m1, opt1, step=50, config=cfg, stage="S1A")

    rng_py1 = random.Random(123)
    rng_np1 = np.random.default_rng(456)
    _ = rng_py1.random(), float(rng_np1.random())
    ckpt_p = save_plateau_checkpoint(
        tmp_path / "plateau.pt",
        m1,
        opt1,
        step=70,
        config=cfg,
        rng_py=rng_py1,
        rng_np=rng_np1,
        stage="S2",
    )
    expected_py = rng_py1.random()
    expected_np = float(rng_np1.random())

    m2 = CalciteV2LM(cfg)
    m2.byte_chunker.lock_router(True)
    opt2 = MuonAdamWHybrid(m2, base_lr=1e-3, device="cpu", ns_schedule="polar4")
    rng_py2 = random.Random(999)
    rng_np2 = np.random.default_rng(999)
    meta = load_plateau_checkpoint(
        ckpt_p, m2, opt2, rng_py=rng_py2, rng_np=rng_np2, preserve_ns_schedule=True
    )
    assert meta["step"] == 70 and meta["stage"] == "S2"
    assert opt2.ns_schedule == "polar4"
    # Verify #66/#67 fix: live param_groups identity preserved across load_state_dict & WSD decay reaches AdamW
    assert opt2.adamw is not None and opt2.muon_group is not None
    for i, apg in enumerate(opt2.adamw.param_groups):
        assert opt2.param_groups[i] is apg
    assert opt2.param_groups[-1] is opt2.muon_group
    sched_end = WSDLRSchedule(base_lr=1e-3, min_lr=1e-4, total_steps=100, warmup_steps=10, decay_steps=30)
    sched_end.apply(opt2, 100)
    for apg in opt2.adamw.param_groups:
        assert abs(float(apg["lr"]) - 1e-4 * float(apg.get("lr_mult", 1.0))) < 1e-12
    assert abs(float(opt2.muon_group["lr"]) - 1e-4 * float(opt2.muon_group.get("lr_mult", 1.0))) < 1e-12
    assert rng_py2.random() == expected_py and float(rng_np2.random()) == expected_np
    with torch.no_grad():
        assert float((m1(x) - m2(x)).abs().max().item()) == 0.0

    # 3. OEBEarlyStopper (PASS, KILL, NaN-KILL, disabled_for_axis)
    oeb = OEBEarlyStopper({250: 2.30, 450: 2.15}, margin_bpb=0.025)
    assert oeb.should_check(250) and not oeb.should_check(200)
    killed, _ = oeb.check(250, 2.32)
    assert not killed
    killed, reason = oeb.check(450, 2.20)
    assert killed and "KILL" in reason and oeb.kill_step == 450

    oeb_nan = OEBEarlyStopper({250: 2.30}, margin_bpb=0.025)
    killed_n, _ = oeb_nan.check(250, float("nan"))
    assert killed_n

    oeb_dis = OEBEarlyStopper({250: 2.30}, margin_bpb=0.025, disabled_for_axis=True, axis_name="needle")
    assert not oeb_dis.should_check(250)
    killed_d, _ = oeb_dis.check(250, 9.99)
    assert not killed_d

    # 4. run_tiered_eval(tier=0, smoke=True) CPU smoke
    tok = ByteBPETokenizer(merges=[])
    val_mm = np.random.default_rng(7).integers(3, 259, size=(16384,), dtype=np.uint16)
    win = encode_utf8_fast(tok, "Calcite v3.1 Tier-0 eval smoke test Привет мир! " * 4)[:65]
    eval_wins = {"dom_a": [win] * 4}
    m1.eval()
    m1.set_crystal_schedule(1, 1)
    m1.set_snap_alpha(1.0)
    ev0 = run_tiered_eval(
        m1, tier=0, val_mm=val_mm, eval_wins=eval_wins, tok=tok, device="cpu", seq_len=64, eval_bs=2, smoke=True
    )
    assert ev0["tier"] == 0 and float(ev0["val_bpb"]) > 0.0 and float(ev0["ow6_mean_bpb"]) > 0.0
    assert "ood_verbatim_ladder" in ev0

    # 5. AGENTS.md size <= 24,000 B & zero Russian lexical quarantine hits
    sz = check_agents_md_size(ROOT_DIR)
    assert sz <= AGENTS_MD_MAX_BYTES
    n_md = check_markdown_quarantine(ROOT_DIR)
    assert n_md >= 1
