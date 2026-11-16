"""
Gate 3 Single-Seed CUDA Probe (`experiments/v5_proto/probe_phase_b_byte_hnet.py`):
Phase B (`SurpriseHNetChunker` over 256 raw UTF-8 bytes, `vocab_size = 259`).

Trains `SimVQSplitLM(vocab_size=259, use_byte_chunker=True)` on the shared multi-track stream
(`STEPS=2200, BS=64, SEQ_LEN_BYTES=768, seed=2026`), measures realized byte compression ratio
and exact `BPE-1536 Equiv PPL` on `te_docs[:500]`, evaluates `FP`, `0-QAT PTQ (LS)`, `300-step STE QAT`,
and physical `2-bit uint8 T-MAC LUT` (`PackedLUT2BitLinear`), checks all Phase B acceptance gates
(`<= 555 KB` LUT, `80.0..83.5 KB` O(1) cache, `3.8..5.2` B/chunk, cognitive non-degradation),
and writes `benchmarks/phase_b_byte_chunker_results.json`.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_simvq_split_benchmark import (  # noqa: E402
    build_shared_training_stream,
    compute_state_cache_kb,
    count_params,
    harmonic_mean,
)
from simvq_split.eval_utils import (  # noqa: E402
    evaluate_prose_perplexity,
    evaluate_suite_batched,
    wilson_ci_95,
)
from simvq_split.memory import (  # noqa: E402
    compute_2bit_lut_footprint,
    compute_chunker_state_bytes,
    pack_model_2bit_lut,
    quantize_ternary_weights_ls,
)
from simvq_split.models import SimVQSplitConfig, SimVQSplitLM  # noqa: E402
from simvq_split.tasks import (  # noqa: E402
    build_eval_overwrite_suite,
    build_eval_s5_suite,
    build_eval_trap_suite,
    load_gsm8k_corpora,
    make_overwrite_story,
    make_s5_story,
    make_trap_overwrite_story,
)
from simvq_split.tokenizer import ByteBPETokenizer  # noqa: E402


def compute_bpe_to_byte_scale(repo_root: Path, te_docs: list[str], tr_docs: list[str], tr_sents: list[str]) -> float:
    rng = random.Random(42)
    tok_corpus = list(tr_docs[:2000])
    for _ in range(500):
        p, t, _ = make_s5_story(rng, rng.randint(1, 4), tr_sents, (0, 1))
        tok_corpus.append(p + t)
        p2, t2, _ = make_overwrite_story(rng, 4, 2, tr_sents, 2, 0.3)
        tok_corpus.append(p2 + t2)
    for _ in range(150):
        p3, t3, _ = make_trap_overwrite_story(rng, 4, 2, tr_sents, 2, 0.3)
        tok_corpus.append(p3 + t3)
    tok_bpe = ByteBPETokenizer.train_from_iterator(tok_corpus, vocab_size=1536)
    tok_byte = ByteBPETokenizer(merges=[])
    n_bpe = sum(len(tok_bpe.encode(d, add_bos=True, add_eos=True)) for d in te_docs[:500])
    n_byte = sum(len(tok_byte.encode(d, add_bos=True, add_eos=True)) for d in te_docs[:500])
    return float(n_byte / max(1, n_bpe))


@torch.no_grad()
def evaluate_byte_prose_and_ratio(
    model: SimVQSplitLM,
    tok: ByteBPETokenizer,
    docs: list[str],
    byte_per_bpe: float,
    seq_len: int = 768,
    device: str = "cuda",
) -> dict[str, float]:
    byte_ppl, byte_nll, eval_bytes = evaluate_prose_perplexity(
        model, tok, docs, seq_len=seq_len, batch_size=32, device=device
    )
    byte_bpb = byte_nll / math.log(2.0)
    bpe_equiv_ppl = math.exp(min(byte_nll * byte_per_bpe, 20.0))

    # Measure realized bytes/chunk over te_docs[:500] chunks
    stream: list[int] = []
    for d in docs:
        stream.extend(tok.encode(d, add_bos=True, add_eos=True))
    w = seq_len + 1
    chunks = [stream[i : i + w] for i in range(0, len(stream) - w, seq_len)]
    data = torch.tensor(chunks, dtype=torch.long)
    tot_valid = 0
    tot_bounds = 0
    use_amp = device == "cuda" and torch.cuda.is_available()
    model.eval()
    for start in range(0, data.size(0), 32):
        xb = data[start : start + 32, :-1].to(device)
        e_tok = model.tok_emb(xb)
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, _, ctx = model.byte_chunker.encode_to_chunks(
                    e_tok, xb, model.tok_emb.weight
                )
        else:
            _, _, ctx = model.byte_chunker.encode_to_chunks(
                e_tok, xb, model.tok_emb.weight
            )
        tot_valid += int(ctx.valid_mask[:, 1:].sum().item())
        tot_bounds += int(ctx.b_mask[:, 1:][ctx.valid_mask[:, 1:]].sum().item())
    realized_ratio = float(tot_valid / max(1, tot_bounds))
    return {
        "byte_ppl": round(byte_ppl, 4),
        "byte_bpb": round(byte_bpb, 4),
        "byte_nll": round(byte_nll, 5),
        "bpe_equiv_ppl": round(bpe_equiv_ppl, 3),
        "eval_bytes": int(eval_bytes),
        "mean_bytes_per_chunk": round(realized_ratio, 3),
    }


@torch.no_grad()
def measure_crystal_drift_at_L(
    model: SimVQSplitLM, vocab_size: int, device: str, length: int = 2048
) -> dict[str, float]:
    model.eval()
    model.set_snap_alpha(1.0)
    x = torch.randint(3, vocab_size, (4, length), device=device)
    e_tok = model.tok_emb(x)
    if model.byte_chunker is not None:
        h, e_lex, _ = model.byte_chunker.encode_to_chunks(
            e_tok, x, model.tok_emb.weight
        )
    else:
        h, e_lex = e_tok, e_tok
    n_chunks = int(h.size(1))
    for blk in model.cortex_pre:
        h = blk(h)
    hs = model.split_norm(h)

    max_norm_drift = 0.0
    max_lattice_err = 0.0
    stages_checked = 0

    def _accumulate(cryst: nn.Module, inp: torch.Tensor) -> None:
        nonlocal max_norm_drift, max_lattice_err, stages_checked
        scale_r = math.sqrt(cryst.d_s)
        _, c_norm_t = cryst.get_codebook_fp32()
        s_read, _, _, _ = cryst.forward_states(inp, h_lex=e_lex)
        norms = s_read.float().norm(dim=-1) / scale_r
        max_norm_drift = max(max_norm_drift, float((norms - 1.0).abs().max().item()))
        s_unit = F.normalize(s_read.float(), dim=-1)
        cos_to_cb = torch.matmul(s_unit, c_norm_t)
        max_lattice_err = max(
            max_lattice_err,
            float((1.0 - cos_to_cb.max(dim=-1).values).abs().max().item()),
        )
        stages_checked += 1

    if model.crystal is not None:
        _accumulate(model.crystal, hs)
    if model.crystal_2 is not None and model.cryst2_norm is not None:
        gates = 2.0 * torch.sigmoid(model.stream_gate(hs))
        delta_long = (
            gates[..., 0:1] * model.crystal(hs, h_lex=e_lex)
            + gates[..., 1:2] * model.woodbury(hs, h_lex=e_lex)
        )
        h2 = h + delta_long
        h2 = h2 + model.mid_mlp(model.mid_mlp_norm(h2))
        _accumulate(model.crystal_2, model.cryst2_norm(h2))

    return {
        "length_bytes": length,
        "chunks_at_L": n_chunks,
        "norm_drift_L2048": max_norm_drift,
        "voronoi_lattice_err_L2048": max_lattice_err,
        "crystal_stages_checked": stages_checked,
    }


def _eval_cognitive_suites(
    model: nn.Module,
    tok: ByteBPETokenizer,
    s5_suites: dict,
    ov_suites: dict,
    trap_suites: dict,
    device: str,
) -> tuple[dict, dict, dict, float, float]:
    s5_res = {}
    for k_key, suite in s5_suites.items():
        ex_pct, tk_pct, k_ok, n_tot = evaluate_suite_batched(
            model, suite, pad_id=tok.PAD_ID, batch_size=32, device=device
        )
        ci = wilson_ci_95(k_ok, n_tot)
        s5_res[k_key] = {
            "exact_pct": round(ex_pct, 2),
            "token_pct": round(tk_pct, 2),
            "ok": k_ok,
            "n": n_tot,
            "wilson_95": [round(ci[0], 2), round(ci[1], 2)],
        }
    ov_res = {}
    for o_key, suite in ov_suites.items():
        ex_pct, tk_pct, k_ok, n_tot = evaluate_suite_batched(
            model, suite, pad_id=tok.PAD_ID, batch_size=32, device=device
        )
        ci = wilson_ci_95(k_ok, n_tot)
        ov_res[o_key] = {
            "exact_pct": round(ex_pct, 2),
            "token_pct": round(tk_pct, 2),
            "ok": k_ok,
            "n": n_tot,
            "wilson_95": [round(ci[0], 2), round(ci[1], 2)],
        }
    trap_res = {}
    for t_key, suite in trap_suites.items():
        ex_pct, tk_pct, k_ok, n_tot = evaluate_suite_batched(
            model, suite, pad_id=tok.PAD_ID, batch_size=32, device=device
        )
        ci = wilson_ci_95(k_ok, n_tot)
        trap_res[t_key] = {
            "exact_pct": round(ex_pct, 2),
            "token_pct": round(tk_pct, 2),
            "ok": k_ok,
            "n": n_tot,
            "wilson_95": [round(ci[0], 2), round(ci[1], 2)],
        }
    sel_f1_in = round(
        harmonic_mean(ov_res["in_dist"]["exact_pct"], trap_res["in_dist"]["exact_pct"]),
        2,
    )
    sel_f1_long = round(
        harmonic_mean(ov_res["ood_long"]["exact_pct"], trap_res["ood_long"]["exact_pct"]),
        2,
    )
    return s5_res, ov_res, trap_res, sel_f1_in, sel_f1_long


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate 3 Phase B CUDA Probe")
    parser.add_argument(
        "--resume-fp-ckpt",
        type=str,
        default="",
        help="Optional checkpoint path to resume converged FP ByteHNet weights and run QAT/LUT evaluation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Gate 3 Phase B CUDA Probe (`SurpriseHNetChunker` over 256 bytes) on device={dev} ===", flush=True)

    tr_docs, te_docs, tr_sents, te_sents = load_gsm8k_corpora(REPO_ROOT)
    tok = ByteBPETokenizer(merges=[])
    byte_per_bpe = compute_bpe_to_byte_scale(REPO_ROOT, te_docs, tr_docs, tr_sents)
    print(f"Tokenizer vocab_size={tok.vocab_size} | BPE-to-byte ratio on te_docs[:500]={byte_per_bpe:.4f}", flush=True)

    STEPS, BS, SEQ_LEN_BYTES = 2400, 64, 768
    t_data = time.perf_counter()
    X_train, Y_train = build_shared_training_stream(
        tok, tr_docs, tr_sents, n_steps=STEPS, batch_size=BS, seq_len=SEQ_LEN_BYTES, seed=2026
    )
    print(f"Built unified byte stream {tuple(X_train.shape)} in {time.perf_counter() - t_data:.1f}s", flush=True)

    N_EVAL = 200
    s5_suites = {
        "k2": build_eval_s5_suite(tok, te_sents, 2, N_EVAL, (0, 1), 3100),
        "k4": build_eval_s5_suite(tok, te_sents, 4, N_EVAL, (0, 1), 3200),
        "k8": build_eval_s5_suite(tok, te_sents, 8, N_EVAL, (1, 1), 3300),
        "k16": build_eval_s5_suite(tok, te_sents, 16, N_EVAL, (1, 2), 3400),
    }
    ov_suites = {
        "in_dist": build_eval_overwrite_suite(tok, te_sents, 4, 2, 2, N_EVAL, 4100),
        "ood_med": build_eval_overwrite_suite(tok, te_sents, 7, 4, 10, N_EVAL, 4200),
        "ood_long": build_eval_overwrite_suite(tok, te_sents, 9, 5, 18, N_EVAL, 4300),
    }
    trap_suites = {
        "in_dist": build_eval_trap_suite(tok, te_sents, 4, 2, 2, N_EVAL, 5100),
        "ood_long": build_eval_trap_suite(tok, te_sents, 9, 5, 18, N_EVAL, 5300),
    }

    cfg = SimVQSplitConfig(
        vocab_size=tok.vocab_size,
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
        ridge_lambda=0.04,
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
        pad_token_id=tok.PAD_ID,
    )

    torch.manual_seed(100)
    model = SimVQSplitLM(cfg).to(dev)
    n_p = count_params(model)
    cache_kb = round(compute_state_cache_kb("SimVQSplitLM", cfg, None, 2048), 2)
    chunker_bytes = compute_chunker_state_bytes(
        cfg.d_model, cfg.byte_char_dim, cfg.byte_router_char_dim
    )
    print(f"Model params={n_p:,} | O(1) cache@2048={cache_kb:.2f} KB (chunker={chunker_bytes} B)", flush=True)

    # 1. Train FP model (or resume converged FP checkpoint)
    use_amp = dev == "cuda" and torch.cuda.is_available()
    resume_path = Path(args.resume_fp_ckpt) if args.resume_fp_ckpt else None
    if resume_path is not None and resume_path.exists():
        ckpt_in = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt_in["simvq_split_state_dict"], strict=True)
        model = model.to(dev)
        if model.byte_chunker is not None:
            model.byte_chunker.lock_router(True)
        train_sec = float(
            ckpt_in.get("probe_summary", {}).get("fp", {}).get("train_seconds", 0.0)
        )
        print(
            f"  [Resumed FP ByteHNet from {resume_path.name} (orig train={train_sec:.1f}s)]",
            flush=True,
        )
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=0.01)
        t0 = time.perf_counter()

        for step in range(1, STEPS + 1):
            model.train()
            model.set_crystal_schedule(step, STEPS)
            alpha = model.crystal.snap_alpha if model.crystal is not None else 0.0

            lr = 2.5e-3 * 0.5 * (1.0 + math.cos(math.pi * step / STEPS))
            if step < 60:
                lr = 2.5e-3 * (step / 60.0)
            for pg in opt.param_groups:
                pg["lr"] = lr

            sl = slice((step - 1) * BS, step * BS)
            xb = X_train[sl].to(dev)
            yb = Y_train[sl].to(dev)
            opt.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(xb)
                    tok_ce = F.cross_entropy(
                        logits.transpose(1, 2),
                        yb,
                        ignore_index=tok.PAD_ID,
                        reduction="none",
                    )
                    denom = (yb != tok.PAD_ID).sum(dim=1).clamp(min=1)
                    loss = (tok_ce.sum(dim=1) / denom).mean() + 0.04 * model.get_commitment_loss()
            else:
                logits = model(xb)
                tok_ce = F.cross_entropy(
                    logits.transpose(1, 2),
                    yb,
                    ignore_index=tok.PAD_ID,
                    reduction="none",
                )
                denom = (yb != tok.PAD_ID).sum(dim=1).clamp(min=1)
                loss = (tok_ce.sum(dim=1) / denom).mean() + 0.04 * model.get_commitment_loss()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step in (100, 415, 1000, 1800, STEPS):
                ratio_cur = model.byte_chunker.last_chunk_ratio if model.byte_chunker is not None else 1.0
                locked = model.byte_chunker.router_locked if model.byte_chunker is not None else False
                print(
                    f"  [FP ByteHNet | {n_p:,}p] step {step:4d}/{STEPS} ({time.perf_counter() - t0:5.1f}s) "
                    f"alpha={alpha:.2f} lock={int(locked)} b/ch={ratio_cur:.2f} loss={loss.item():.3f}",
                    flush=True,
                )

        train_sec = round(time.perf_counter() - t0, 2)

    model.set_crystal_schedule(STEPS, STEPS)
    model.set_snap_alpha(1.0)

    fp_lang = evaluate_byte_prose_and_ratio(
        model, tok, te_docs[:500], byte_per_bpe, seq_len=SEQ_LEN_BYTES, device=dev
    )
    fp_s5, fp_ov, fp_trap, fp_f1_in, fp_f1_long = _eval_cognitive_suites(
        model, tok, s5_suites, ov_suites, trap_suites, dev
    )
    fp_s5_sum = round(sum(fp_s5[k]["exact_pct"] for k in ("k2", "k4", "k8", "k16")), 2)
    drift_2048 = measure_crystal_drift_at_L(model, tok.vocab_size, dev, length=2048)

    print(
        f"-> FP ByteHNet ({train_sec:.1f}s): BPB={fp_lang['byte_bpb']:.3f} BPE_EqPPL={fp_lang['bpe_equiv_ppl']:.2f} "
        f"b/chunk={fp_lang['mean_bytes_per_chunk']:.2f} | "
        f"S5={fp_s5['k2']['exact_pct']:.1f}/{fp_s5['k4']['exact_pct']:.1f}/{fp_s5['k8']['exact_pct']:.1f}/{fp_s5['k16']['exact_pct']:.1f} (sum={fp_s5_sum:.1f}) | "
        f"OV={fp_ov['in_dist']['exact_pct']:.1f}/{fp_ov['ood_med']['exact_pct']:.1f}/{fp_ov['ood_long']['exact_pct']:.1f} | "
        f"Trap={fp_trap['in_dist']['exact_pct']:.1f}/{fp_trap['ood_long']['exact_pct']:.1f} | "
        f"drift2048=({drift_2048['norm_drift_L2048']:.2e}, {drift_2048['voronoi_lattice_err_L2048']:.2e})",
        flush=True,
    )

    # 2. 0-QAT PTQ (LS)
    ptq_model = copy.deepcopy(model).to(dev).eval()
    ptq_model.set_snap_alpha(1.0)
    ptq_qparams = 0
    bc_ptq = ptq_model.byte_chunker
    with torch.no_grad():
        for name, mod in ptq_model.named_modules():
            if isinstance(mod, nn.Linear) and name != "lm_head":
                if bc_ptq is not None and bc_ptq.is_locked_router_module(mod):
                    continue
                w_tern, _, _ = quantize_ternary_weights_ls(mod.weight.data)
                mod.weight.data.copy_(w_tern)
                ptq_qparams += mod.weight.numel()
    ptq_lang = evaluate_byte_prose_and_ratio(
        ptq_model, tok, te_docs[:400], byte_per_bpe, seq_len=SEQ_LEN_BYTES, device=dev
    )
    ptq_s5k8, _, _, _ = evaluate_suite_batched(ptq_model, s5_suites["k8"], pad_id=tok.PAD_ID, batch_size=32, device=dev)
    ptq_ovmed, _, _, _ = evaluate_suite_batched(ptq_model, ov_suites["ood_med"], pad_id=tok.PAD_ID, batch_size=32, device=dev)
    ptq_trapl, _, _, _ = evaluate_suite_batched(ptq_model, trap_suites["ood_long"], pad_id=tok.PAD_ID, batch_size=32, device=dev)
    ptq_stats = {
        "ternary_quantized_linear_params": ptq_qparams,
        "byte_bpb_ptq": ptq_lang["byte_bpb"],
        "bpe_equiv_ppl_ptq": ptq_lang["bpe_equiv_ppl"],
        "s5_k8_158bit_ptq_pct": round(ptq_s5k8, 2),
        "ov_ood_med_158bit_ptq_pct": round(ptq_ovmed, 2),
        "trap_ood_long_158bit_ptq_pct": round(ptq_trapl, 2),
    }
    print(f"-> 1.58-bit PTQ (LS): {ptq_stats}", flush=True)

    # 3. 500-step Differential-LR Progressive STE QAT (`mix: 0.30 -> 1.0`, `CORE_LR=1.5e-4`, `CHUNK_LR=9.0e-4`)
    QAT_STEPS = 500
    CORE_LR = 1.5e-4
    CHUNK_LR = 9.0e-4
    torch.manual_seed(777)
    qat_model = copy.deepcopy(model).to(dev)
    qat_model.set_crystal_schedule(STEPS, STEPS)
    if qat_model.byte_chunker is not None:
        qat_model.byte_chunker.lock_router(True)
    qat_model.set_snap_alpha(1.0)
    qat_model.set_qat_mix(0.30)
    qat_model.set_ternary_qat(True)

    chunker_params = []
    core_params = []
    for pname, p in qat_model.named_parameters():
        if not p.requires_grad:
            continue
        if pname.startswith("byte_chunker."):
            chunker_params.append(p)
        else:
            core_params.append(p)
    all_qat_params = core_params + chunker_params
    opt_q = torch.optim.AdamW(
        [
            {"params": core_params, "lr": CORE_LR},
            {"params": chunker_params, "lr": CHUNK_LR},
        ],
        weight_decay=0.0,
    )
    start_row = (STEPS - QAT_STEPS) * BS
    t_qat = time.perf_counter()
    for q_step in range(1, QAT_STEPS + 1):
        qat_model.train()
        mix = min(1.0, 0.30 + 0.70 * (q_step / 120.0)) if q_step <= 120 else 1.0
        qat_model.set_qat_mix(mix)
        if q_step <= 20:
            lr_mult = q_step / 20.0
        else:
            lr_mult = 0.1 + 0.9 * 0.5 * (
                1.0 + math.cos(math.pi * (q_step - 20) / (QAT_STEPS - 20))
            )
        opt_q.param_groups[0]["lr"] = CORE_LR * lr_mult
        opt_q.param_groups[1]["lr"] = CHUNK_LR * lr_mult

        sl = slice(start_row + (q_step - 1) * BS, start_row + q_step * BS)
        xb = X_train[sl].to(dev)
        yb = Y_train[sl].to(dev)
        opt_q.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = qat_model(xb)
                tok_ce = F.cross_entropy(
                    logits.transpose(1, 2), yb, ignore_index=tok.PAD_ID, reduction="none"
                )
                denom = (yb != tok.PAD_ID).sum(dim=1).clamp(min=1)
                loss = (tok_ce.sum(dim=1) / denom).mean() + 0.04 * qat_model.get_commitment_loss()
        else:
            logits = qat_model(xb)
            tok_ce = F.cross_entropy(
                logits.transpose(1, 2), yb, ignore_index=tok.PAD_ID, reduction="none"
            )
            denom = (yb != tok.PAD_ID).sum(dim=1).clamp(min=1)
            loss = (tok_ce.sum(dim=1) / denom).mean() + 0.04 * qat_model.get_commitment_loss()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_qat_params, 1.0)
        opt_q.step()

        if q_step in (120, 250, QAT_STEPS):
            rc = qat_model.byte_chunker.last_chunk_ratio if qat_model.byte_chunker is not None else 1.0
            print(
                f"  [1.58-bit STE QAT] step {q_step:3d}/{QAT_STEPS} mix={qat_model.qat_mix:.2f} b/ch={rc:.2f} ({time.perf_counter() - t_qat:5.1f}s) loss={loss.item():.3f}",
                flush=True,
            )

    qat_model.set_ternary_qat(False)
    qat_model.set_qat_mix(1.0)
    qat_qparams = 0
    bc_qat = qat_model.byte_chunker
    with torch.no_grad():
        for name, mod in qat_model.named_modules():
            if isinstance(mod, nn.Linear) and name != "lm_head":
                if bc_qat is not None and bc_qat.is_locked_router_module(mod):
                    continue
                w_tern, _, _ = quantize_ternary_weights_ls(mod.weight.data)
                g_size = 32 if mod.weight.numel() % 32 == 0 else mod.weight.size(-1)
                for row in w_tern.float().reshape(-1, g_size):
                    assert torch.unique(torch.round(row * 1e5)).numel() <= 3
                mod.weight.data.copy_(w_tern)
                qat_qparams += mod.weight.numel()
    qat_model.eval()
    qat_model.set_snap_alpha(1.0)

    qat_lang = evaluate_byte_prose_and_ratio(
        qat_model, tok, te_docs[:500], byte_per_bpe, seq_len=SEQ_LEN_BYTES, device=dev
    )
    qat_s5, qat_ov, qat_trap, qat_f1_in, qat_f1_long = _eval_cognitive_suites(
        qat_model, tok, s5_suites, ov_suites, trap_suites, dev
    )
    qat_s5_sum = round(sum(qat_s5[k]["exact_pct"] for k in ("k2", "k4", "k8", "k16")), 2)
    qat_stats = {
        "qat_steps": QAT_STEPS,
        "qat_seconds": round(time.perf_counter() - t_qat, 2),
        "ternary_quantized_linear_params": qat_qparams,
        **qat_lang,
        "s5": qat_s5,
        "s5_sum": qat_s5_sum,
        "overwrite": qat_ov,
        "trap": qat_trap,
        "selective_f1_in": qat_f1_in,
        "selective_f1_ood_long": qat_f1_long,
    }
    print(
        f"-> STE QAT ByteHNet: BPE_EqPPL={qat_lang['bpe_equiv_ppl']:.2f} b/ch={qat_lang['mean_bytes_per_chunk']:.2f} | "
        f"S5={qat_s5['k2']['exact_pct']:.1f}/{qat_s5['k4']['exact_pct']:.1f}/{qat_s5['k8']['exact_pct']:.1f}/{qat_s5['k16']['exact_pct']:.1f} (sum={qat_s5_sum:.1f}) | "
        f"OV={qat_ov['in_dist']['exact_pct']:.1f}/{qat_ov['ood_med']['exact_pct']:.1f}/{qat_ov['ood_long']['exact_pct']:.1f} | "
        f"Trap={qat_trap['in_dist']['exact_pct']:.1f}/{qat_trap['ood_long']['exact_pct']:.1f}",
        flush=True,
    )

    # 4. Physical Packed uint8 2-bit T-MAC LUT evaluation
    lut_model = pack_model_2bit_lut(qat_model).to(dev).eval()
    fp_lut = compute_2bit_lut_footprint(lut_model)
    lut_lang = evaluate_byte_prose_and_ratio(
        lut_model, tok, te_docs[:500], byte_per_bpe, seq_len=SEQ_LEN_BYTES, device=dev
    )
    lut_s5, lut_ov, lut_trap, lut_f1_in, lut_f1_long = _eval_cognitive_suites(
        lut_model, tok, s5_suites, ov_suites, trap_suites, dev
    )
    lut_s5_sum = round(sum(lut_s5[k]["exact_pct"] for k in ("k2", "k4", "k8", "k16")), 2)
    max_pp_diff = max(
        abs(lut_s5[k]["exact_pct"] - qat_s5[k]["exact_pct"]) for k in ("k2", "k4", "k8", "k16")
    )
    max_pp_diff = max(
        max_pp_diff,
        abs(lut_ov["in_dist"]["exact_pct"] - qat_ov["in_dist"]["exact_pct"]),
        abs(lut_ov["ood_med"]["exact_pct"] - qat_ov["ood_med"]["exact_pct"]),
        abs(lut_ov["ood_long"]["exact_pct"] - qat_ov["ood_long"]["exact_pct"]),
        abs(lut_trap["in_dist"]["exact_pct"] - qat_trap["in_dist"]["exact_pct"]),
        abs(lut_trap["ood_long"]["exact_pct"] - qat_trap["ood_long"]["exact_pct"]),
    )
    lut_stats = {
        **fp_lut,
        **lut_lang,
        "s5": lut_s5,
        "s5_sum": lut_s5_sum,
        "overwrite": lut_ov,
        "trap": lut_trap,
        "selective_f1_in": lut_f1_in,
        "selective_f1_ood_long": lut_f1_long,
        "max_pp_diff_vs_qat": round(max_pp_diff, 3),
        "bpe_equiv_ppl_diff_vs_qat": round(abs(lut_lang["bpe_equiv_ppl"] - qat_lang["bpe_equiv_ppl"]), 4),
    }
    print(
        f"-> 2-bit uint8 LUT ({fp_lut['total_lut_kb']:.2f} KB, {fp_lut['matrix_compression_vs_fp32']}x): "
        f"BPE_EqPPL={lut_lang['bpe_equiv_ppl']:.2f} b/ch={lut_lang['mean_bytes_per_chunk']:.2f} | "
        f"S5={lut_s5['k2']['exact_pct']:.1f}/{lut_s5['k4']['exact_pct']:.1f}/{lut_s5['k8']['exact_pct']:.1f}/{lut_s5['k16']['exact_pct']:.1f} (sum={lut_s5_sum:.1f}) | "
        f"OV={lut_ov['in_dist']['exact_pct']:.1f}/{lut_ov['ood_med']['exact_pct']:.1f}/{lut_ov['ood_long']['exact_pct']:.1f} | "
        f"Trap={lut_trap['in_dist']['exact_pct']:.1f}/{lut_trap['ood_long']['exact_pct']:.1f} | diff_vs_qat={max_pp_diff:.2f}pp",
        flush=True,
    )

    # 5. Evaluate calibrated Attempt #2 Gate 3 Acceptance Criteria
    gates = {
        "lut_kb_le_555": fp_lut["total_lut_kb"] <= 555.0,
        "matrix_comp_16x": abs(fp_lut["matrix_compression_vs_fp32"] - 16.0) < 1e-3,
        "cache_kb_in_80_83_5": 80.0 <= cache_kb <= 83.5,
        "fp_chunk_ratio_3_8_to_5_2": 3.8 <= fp_lang["mean_bytes_per_chunk"] <= 5.2,
        "lut_chunk_ratio_3_8_to_5_2": 3.8 <= lut_lang["mean_bytes_per_chunk"] <= 5.2,
        "fp_bpb_le_1_81": fp_lang["byte_bpb"] <= 1.81 and fp_lang["bpe_equiv_ppl"] <= 39.82,
        "lut_bpb_le_1_92": lut_lang["byte_bpb"] <= 1.92 and lut_lang["bpe_equiv_ppl"] <= 48.0,
        "fp_s5_k2_ge_55": fp_s5["k2"]["exact_pct"] >= 55.0,
        "fp_s5_sum_ge_80": fp_s5_sum >= 80.0,
        "lut_s5_k2_ge_48": lut_s5["k2"]["exact_pct"] >= 48.0,
        "fp_ov_med_ge_90": fp_ov["ood_med"]["exact_pct"] >= 90.0,
        "fp_ov_long_ge_86": fp_ov["ood_long"]["exact_pct"] >= 86.0,
        "lut_ov_med_ge_90": lut_ov["ood_med"]["exact_pct"] >= 90.0,
        "lut_ov_long_ge_86": lut_ov["ood_long"]["exact_pct"] >= 86.0,
        "fp_trap_in_ge_92": fp_trap["in_dist"]["exact_pct"] >= 92.0,
        "fp_trap_long_ge_86": fp_trap["ood_long"]["exact_pct"] >= 86.0,
        "lut_trap_in_ge_92": lut_trap["in_dist"]["exact_pct"] >= 92.0,
        "lut_trap_long_ge_86": lut_trap["ood_long"]["exact_pct"] >= 86.0,
        "crystal_drift_L2048_lt_1e5": max(drift_2048["norm_drift_L2048"], drift_2048["voronoi_lattice_err_L2048"]) < 1e-5,
        "lut_qat_parity_le_2_5pp": max_pp_diff <= 2.5,
    }
    all_pass = all(gates.values())
    print("\n=== Gate 3 Acceptance Summary ===", flush=True)
    for g_name, g_ok in gates.items():
        print(f"  {g_name:30s}: {'PASS' if g_ok else 'FAIL'}", flush=True)
    print(f"OVERALL GATE 3: {'PASS' if all_pass else 'FAIL'}", flush=True)

    payload = {
        "probe_config": {
            "phase": "B_SurpriseHNetChunker_256_bytes",
            "steps": STEPS,
            "batch_size": BS,
            "seq_len_bytes": SEQ_LEN_BYTES,
            "bpe_to_byte_scale_te500": round(byte_per_bpe, 6),
            "simvq_split_config": asdict(cfg),
            "params": n_p,
            "state_cache_kb_L2048": cache_kb,
            "chunker_state_bytes": chunker_bytes,
            "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
        },
        "fp": {
            "train_seconds": train_sec,
            **fp_lang,
            "s5": fp_s5,
            "s5_sum": fp_s5_sum,
            "overwrite": fp_ov,
            "trap": fp_trap,
            "selective_f1_in": fp_f1_in,
            "selective_f1_ood_long": fp_f1_long,
        },
        "crystal_drift_L2048": drift_2048,
        "ternary_158bit_ptq": ptq_stats,
        "ternary_158bit_qat": qat_stats,
        "tmac_2bit_lut": lut_stats,
        "gates": gates,
        "all_gates_passed": all_pass,
    }

    out_json = REPO_ROOT / "benchmarks" / "phase_b_probe_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if qat_model.byte_chunker is not None:
        qat_model.byte_chunker.lock_router(False)
    ckpt_out = REPO_ROOT / "checkpoints" / "simvq_split_byte_hnet.pt"
    ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "simvq_split_config": asdict(cfg),
            "simvq_split_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "simvq_split_ternary_qat_state_dict": {k: v.cpu() for k, v in qat_model.state_dict().items()},
            "tokenizer": tok.to_dict(),
            "probe_summary": payload,
        },
        ckpt_out,
    )
    print(f"Saved probe results to {out_json} and checkpoint to {ckpt_out}", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
