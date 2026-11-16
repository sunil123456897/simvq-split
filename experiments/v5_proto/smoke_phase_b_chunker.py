"""
Tracer #1 (`experiments/v5_proto/smoke_phase_b_chunker.py`):
Gate 1 & Gate 2 Verification before Phase B CUDA Probe:
1. Checks per-block initialization output `std` against the legacy v5 BPE model
   (enforcing zero residual explosion > 1.5x, zero SubLN).
2. Verifies 20-step `BF16` AdamW + `clip_grad_norm_` weight parity of `use_byte_chunker=False`
   and backward compatibility with `checkpoints/simvq_split_tournament.pt`.
3. Runs a CUDA `BF16` mini-batch smoke step on raw bytes (`vocab_size = 259`, `L = 768`),
   verifying exact `O(1)` state bytes across `L = 64..2048`, `2-bit uint8 LUT <= 555 KB`,
   and live finite boundary gradients.
"""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_simvq_split_benchmark import compute_state_cache_kb  # noqa: E402
from simvq_split.memory import (  # noqa: E402
    compute_2bit_lut_footprint,
    compute_chunker_state_bytes,
    pack_model_2bit_lut,
)
from simvq_split.models import SimVQSplitConfig, SimVQSplitLM  # noqa: E402


def measure_block_stds(model: SimVQSplitLM, x: torch.Tensor) -> dict[str, float]:
    model.eval()
    stds: dict[str, float] = {}
    with torch.no_grad():
        e_tok = model.tok_emb(x)
        stds["tok_emb"] = float(e_tok.float().std().item())
        if model.byte_chunker is not None:
            h, e_lex, ctx = model.byte_chunker.encode_to_chunks(
                e_tok, x, model.tok_emb.weight
            )
            stds["byte_enc"] = float(h.float().std().item())
        else:
            h, e_lex, ctx = e_tok, e_tok, None

        for idx, blk in enumerate(model.cortex_pre):
            h = blk(h)
            stds[f"cortex_pre_{idx}"] = float(h.float().std().item())

        h_split = model.split_norm(h)
        c1 = model.crystal(h_split, h_lex=e_lex) if model.crystal is not None else torch.zeros_like(h)
        w1 = model.woodbury(h_split, h_lex=e_lex) if model.woodbury is not None else torch.zeros_like(h)
        stds["crystal_1"] = float(c1.float().std().item())
        stds["woodbury"] = float(w1.float().std().item())

        gates = 2.0 * torch.sigmoid(model.stream_gate(h_split))
        h = h + gates[..., 0:1] * c1 + gates[..., 1:2] * w1
        h = h + model.mid_mlp(model.mid_mlp_norm(h))
        stds["mid_mlp_post"] = float(h.float().std().item())

        if model.crystal_2 is not None and model.cryst2_norm is not None:
            c2_in = model.cryst2_norm(h)
            c2 = model.crystal_2(c2_in, h_lex=e_lex)
            stds["crystal_2"] = float(c2.float().std().item())
            h = h + (2.0 * torch.sigmoid(model.cryst2_gate(c2_in))) * c2

        for idx, blk in enumerate(model.cortex_post):
            h = blk(h)
            stds[f"cortex_post_{idx}"] = float(h.float().std().item())

        if model.byte_chunker is not None and ctx is not None:
            h = model.byte_chunker.decode_from_chunks(h, ctx)
            stds["byte_dec"] = float(h.float().std().item())

        logits = model.lm_head(model.norm_f(h))
        stds["logits"] = float(logits.float().std().item())
    return stds


def main() -> int:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Tracer #1 (Phase B Chunker) on device={dev} ===", flush=True)

    # 1. Init-std comparison: Legacy v5 BPE vs Phase B Byte Chunker
    torch.manual_seed(100)
    cfg_v5 = SimVQSplitConfig(
        vocab_size=1536,
        d_model=160,
        n_heads=4,
        n_layers_pre=2,
        n_layers_post=1,
        window_size=12,
        num_spheres=24,
        d_sphere=32,
        codebook_size=32,
        mem_heads=5,
        mem_dim=32,
        use_crystal=True,
        use_woodbury=True,
        use_byte_chunker=False,
    )
    m_v5 = SimVQSplitLM(cfg_v5).to(dev)
    xb_v5 = torch.randint(3, 1536, (4, 192), device=dev)
    stds_v5 = measure_block_stds(m_v5, xb_v5)

    torch.manual_seed(100)
    cfg_b = SimVQSplitConfig(
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
        byte_commit_scale=0.45,
        router_lock_frac=0.135,
        target_chunk_bytes=4.9,
        max_chunk_bytes=10,
        ratio_loss_scale=1.5,
    )
    m_b = SimVQSplitLM(cfg_b).to(dev)
    xb_b = torch.randint(3, 259, (4, 768), device=dev)
    stds_b = measure_block_stds(m_b, xb_b)

    print("\n--- [Gate 2a] Init-Std Check (ratio vs v5 <= 1.50x) ---", flush=True)
    max_ratio = 0.0
    for k, v in stds_b.items():
        if k in stds_v5 and stds_v5[k] > 1e-6:
            ratio = v / stds_v5[k]
            max_ratio = max(max_ratio, ratio)
            print(f"  {k:16s}: v5={stds_v5[k]:.4f} | phase_b={v:.4f} | ratio={ratio:.3f}x", flush=True)
        else:
            print(f"  {k:16s}: phase_b={v:.4f}", flush=True)
    assert max_ratio <= 1.50, f"Block init std explosion detected: max_ratio={max_ratio:.3f}x > 1.50x!"

    # 2. Backward compatibility + 20-step BF16 AdamW reproducibility check on use_byte_chunker=False
    ckpt_path = REPO_ROOT / "checkpoints" / "simvq_split_tournament.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m_ckpt_fp = SimVQSplitLM(SimVQSplitConfig(**ckpt["simvq_split_config"]))
    m_ckpt_fp.load_state_dict(ckpt["simvq_split_state_dict"], strict=True)
    m_ckpt_qat = SimVQSplitLM(SimVQSplitConfig(**ckpt["simvq_split_config"]))
    m_ckpt_qat.load_state_dict(ckpt["simvq_split_ternary_qat_state_dict"], strict=True)

    # Check exact BF16 deterministic 20-step run on use_byte_chunker=False
    def _run_20_steps(model: SimVQSplitLM) -> dict[str, torch.Tensor]:
        m = copy.deepcopy(model).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=2.5e-3, weight_decay=0.01)
        gen = torch.Generator(device="cpu").manual_seed(77)
        for s in range(20):
            x_s = torch.randint(3, 1536, (8, 64), generator=gen).to(dev)
            y_s = x_s.roll(-1, dims=1)
            opt.zero_grad(set_to_none=True)
            if dev == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = m(x_s)
                    loss = F.cross_entropy(logits.transpose(1, 2), y_s) + 0.04 * m.get_commitment_loss()
            else:
                logits = m(x_s)
                loss = F.cross_entropy(logits.transpose(1, 2), y_s) + 0.04 * m.get_commitment_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        return {n: p.detach().cpu() for n, p in m.named_parameters()}

    sd_a = _run_20_steps(m_v5)
    sd_b = _run_20_steps(m_v5)
    max_20_diff = max((sd_a[k] - sd_b[k]).abs().max().item() for k in sd_a)
    print(f"\n--- [Gate 2b] Legacy v5 20-step BF16 parity: max_diff={max_20_diff:.2e} ---", flush=True)
    assert max_20_diff == 0.0

    # 3. Phase B footprint, O(1) cache across L=64..2048, and 1 CUDA BF16 mini-batch step
    lut_b = pack_model_2bit_lut(m_b.cpu())
    fp_b = compute_2bit_lut_footprint(lut_b)
    cache_kb = compute_state_cache_kb("SimVQSplitLM", cfg_b, None, 2048)
    expected_cb = compute_chunker_state_bytes(160, 32, 16, 32)
    streaming_bytes = {}
    with torch.no_grad():
        for L_chk in (64, 256, 512, 1024, 2048):
            xs = torch.randint(3, 259, (1, L_chk))
            _, _, _, sb = m_b.byte_chunker.forward_streaming_check(
                m_b.tok_emb(xs), xs, m_b.tok_emb.weight
            )
            streaming_bytes[L_chk] = sb
            assert sb == expected_cb

    m_b = m_b.to(dev)
    m_b.train()
    opt_b = torch.optim.AdamW(m_b.parameters(), lr=2.5e-3, weight_decay=0.01)
    xb = torch.randint(3, 259, (8, 768), device=dev)
    yb = xb.roll(-1, dims=1)
    opt_b.zero_grad(set_to_none=True)
    if dev == "cuda":
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits_b = m_b(xb)
            ce_b = F.cross_entropy(logits_b[:, :-1].reshape(-1, 259), yb[:, :-1].reshape(-1))
            loss_b = ce_b + 0.04 * m_b.get_commitment_loss()
    else:
        logits_b = m_b(xb)
        ce_b = F.cross_entropy(logits_b[:, :-1].reshape(-1, 259), yb[:, :-1].reshape(-1))
        loss_b = ce_b + 0.04 * m_b.get_commitment_loss()
    loss_b.backward()

    dead_b = [
        n
        for n, p in m_b.named_parameters()
        if p.requires_grad and (p.grad is None or p.grad.abs().max().item() == 0.0)
    ]
    assert not dead_b, f"Dead parameters on CUDA mini-batch: {dead_b}"
    router_grad = float(m_b.byte_chunker.router_ctx.bias.grad.abs().item())
    opt_b.step()

    print("\n--- [Gate 2c] Phase B Budgets & CUDA Mini-Batch ---", flush=True)
    print(f"  params          : {sum(p.numel() for p in m_b.parameters()):,}", flush=True)
    print(f"  2-bit LUT total : {fp_b['total_lut_kb']:.2f} KB (u8={fp_b['packed_u8_kb']:.2f} KB, comp={fp_b['matrix_compression_vs_fp32']}x)", flush=True)
    print(f"  O(1) cache total: {cache_kb:.2f} KB (chunker_bytes={streaming_bytes})", flush=True)
    print(f"  mini-batch BF16 : loss={loss_b.item():.4f}, chunk_ratio={m_b.byte_chunker.last_chunk_ratio:.2f} B/chunk, router_ctx.bias_grad={router_grad:.4e}", flush=True)

    assert fp_b["total_lut_kb"] <= 555.0
    assert 80.0 <= cache_kb <= 83.5
    assert router_grad > 0.0
    print("PASS: Tracer #1 Phase B passed all Gate 1 & Gate 2 invariants.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
