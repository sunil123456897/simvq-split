"""
Gate 3 Single-Axis `Via Negativa` Ring-Buffer Pruning Probe (`K15`, `OEB <= 2`).

Measures the `2*sigma` seed noise floor of `V0_phase_be_rings` (`window_size=8, crystal_w_chunks=5,
woodbury_latch_span=7`, actual Phase B/E baseline) across `seed=2026` (`model_seed=100`) and
`seed=2027` (`model_seed=101`) at `d_model=160, 1.48M` params using pure `K8` `CalciteLeanSystem1LM`
(`fused=False` standard `torch.optim.AdamW`, `ov_steps=800`, `target_chunk_bytes=4.85`) and Orthogonal
Two-Substage Consolidation (`Stage 1A`: `2800`-step `"byte"` Crystal `S_5` grokking with
`woodbury` at `0.05x LR, wd=0.01`, `min_lr=1.5e-4` -> `Stage 1B`: `800`-step `"byte"` Woodbury
consolidation with `15% prose / 15% S_5 / 52% OV / 18% Trap`, `woodbury=2.4e-3`, `pre=2.5e-4`,
`post=1.2e-3`, `crystal 4e-5 wd=0`, cosine floor `0.15x` -> `Stage 1.5`: `120`-step `router_emb[131:]`
UTF-8 calibration -> `Stage 2`: `320`-step `"utf8"` adaptation protecting `crystal 6e-5` &
`woodbury 8e-5` at `BS=64`), asserting `v0_mean` `s5_sum >= 80.0` & `ov_ood_long_pct >= 80.0`, then
evaluates `V1_lean_pruned (8, 2, 6)` and fallback `V1b_moderate_pruned (8, 3, 7)` (respecting the
`17 B` causality floor proven in `K10`+`K11`). Writes `benchmarks/phase_k_via_negativa_probe.json`.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import sys
import time
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.scaling_phase_k.eval_suite import (  # noqa: E402
    check_gate2_block_std_parity,
    eval_cognitive_suites_any_byte_lm,
    eval_openweb_6domains_any_byte_lm,
    has_usable_cuda,
    measure_infer_latency_ms,
    sync_if_cuda,
)
from experiments.scaling_phase_k.lean_system1 import (  # noqa: E402
    CalciteLeanSystem1LM,
    LeanSystem1Config,
    build_tier_lean_config,
    compute_lean_state_cache_kb,
)
from run_phase_e_fast_utf8_benchmark import (  # noqa: E402
    build_disjoint_openweb_corpora,
    build_multilingual_plus_cognitive_stream,
)
from run_simvq_split_benchmark import build_shared_training_stream  # noqa: E402
from simvq_split.eval_utils import evaluate_suite_batched  # noqa: E402
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


def _build_stage1b_consolidation_stream(
    tok: ByteBPETokenizer,
    tr_docs: list[str],
    tr_sents: list[str],
    n_steps: int,
    batch_size: int,
    seq_len: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Builds zero-leakage Stage-1B Consolidation stream from `tr_docs` & `tr_sents` only:
    `15%` prose + `15%` `S_5 (k in 1..4)` + `52%` `Overwrite` + `18%` `Trap`.
    """
    r = random.Random(seed)
    tw = seq_len + 1
    total = n_steps * batch_size

    docs = list(tr_docs)
    r.shuffle(docs)
    p_stream: list[int] = []
    for d in docs:
        p_stream.extend(tok.encode(d, add_bos=True, add_eos=True))
    prose_chunks = [p_stream[i : i + tw] for i in range(0, max(0, len(p_stream) - tw), seq_len)]
    if not prose_chunks and len(p_stream) >= 2:
        ch = p_stream[:tw] + [tok.PAD_ID] * max(0, tw - len(p_stream))
        prose_chunks = [ch]

    X_all = torch.full((total, seq_len), tok.PAD_ID, dtype=torch.long)
    Y_all = torch.full((total, seq_len), tok.PAD_ID, dtype=torch.long)
    dr_map = {1: (1, 1), 2: (0, 1), 3: (0, 1), 4: (0, 1)}

    for idx in range(total):
        u = r.random()
        if u < 0.15 and prose_chunks:
            ch = r.choice(prose_chunks)
            X_all[idx] = torch.tensor(ch[:-1], dtype=torch.long)
            Y_all[idx] = torch.tensor(ch[1:], dtype=torch.long)
            continue

        while True:
            if u < 0.30:
                k = r.choice((1, 2, 3, 4))
                p, t, _ = make_s5_story(r, k, tr_sents, dr_map[k])
            elif u < 0.82:
                p, t, _ = make_overwrite_story(
                    r, r.randint(3, 5), r.randint(1, 2), tr_sents, 2, 0.25
                )
            else:
                p, t, _ = make_trap_overwrite_story(
                    r, r.randint(3, 5), r.randint(1, 2), tr_sents, 2, 0.25
                )
            p_ids = tok.encode(p, add_bos=True, add_eos=False)
            f_ids = tok.encode(p + t, add_bos=True, add_eos=True)
            if len(f_ids) <= tw and f_ids[: len(p_ids)] == p_ids:
                break
        L = len(f_ids)
        X_all[idx, : L - 1] = torch.tensor(f_ids[:-1], dtype=torch.long)
        p_len = len(p_ids)
        Y_all[idx, p_len - 1 : L - 1] = torch.tensor(f_ids[p_len:], dtype=torch.long)

    return X_all, Y_all


def _split_stage1b_groups(model: CalciteLeanSystem1LM) -> tuple[list, list, list, list, list, list, list]:
    """Splits trainable params into Crystal (`4e-5`), Pre (`1.5e-4`), Post (`8e-4`), and Woodbury (`1.8e-3`)."""
    cryst_1b, pre_nodc, pre_2d, post_nodc, post_2d, wood_nodc, wood_2d = [], [], [], [], [], [], []
    seen: set[int] = set()
    for pname, p in model.named_parameters():
        if (not p.requires_grad) or (id(p) in seen):
            continue
        seen.add(id(p))
        if pname.startswith(("crystal.", "crystal_2.")):
            cryst_1b.append(p)
        elif pname.startswith("woodbury."):
            if p.ndim < 2:
                wood_nodc.append(p)
            else:
                wood_2d.append(p)
        elif pname.startswith(("tok_emb.", "byte_chunker.", "cortex_pre.", "split_norm.")):
            if p.ndim < 2:
                pre_nodc.append(p)
            else:
                pre_2d.append(p)
        else:
            if p.ndim < 2 or pname.startswith(("stream_gate.", "cryst2_gate.")):
                post_nodc.append(p)
            else:
                post_2d.append(p)
    return cryst_1b, pre_nodc, pre_2d, post_nodc, post_2d, wood_nodc, wood_2d


def _make_adamw(param_groups, device: str, **kwargs) -> torch.optim.AdamW:
    del device
    return torch.optim.AdamW(param_groups, **kwargs)


def _probe_config(window_size: int, crystal_w_chunks: int, woodbury_latch_span: int) -> LeanSystem1Config:
    return LeanSystem1Config(
        vocab_size=259,
        d_model=160,
        n_heads=4,
        d_ff=224,
        n_layers_pre=2,
        n_layers_post=1,
        window_size=int(window_size),
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
        target_chunk_bytes=4.6,
        max_chunk_bytes=10,
        byte_chunk_mode="utf8",
        max_utf8_bytes=16,
        use_static_m_cap=True,
        ratio_loss_scale=1.5,
        pad_token_id=0,
        crystal_w_chunks=int(crystal_w_chunks),
        woodbury_latch_span=int(woodbury_latch_span),
    )


def _train_and_eval_variant(
    tag: str,
    window_size: int,
    crystal_w_chunks: int,
    woodbury_latch_span: int,
    seed: int,
    steps: int,
    utf8_steps: int,
    batch_size: int,
    seq_len: int,
    train_wins: dict,
    eval_wins: dict,
    tr_docs: list[str],
    tr_sents: list[str],
    s5_suites: dict,
    ov_suites: dict,
    trap_suites: dict,
    tok: ByteBPETokenizer,
    device: str,
    ov_steps: int = 800,
    on_stage1_abort=None,
) -> dict:
    ov_steps_eff = int(ov_steps) if steps >= 2000 else max(2, min(int(ov_steps), max(2, steps // 2)))
    print(
        f"\n--- [Via Negativa Probe K15: {tag} | seed={seed}] W={window_size}, "
        f"W_chunks={crystal_w_chunks}, L_latch={woodbury_latch_span} | "
        f"d=160 s1a={steps} + s1b={ov_steps_eff} + cal=120 + utf8={utf8_steps} B={batch_size} L={seq_len} ---",
        flush=True,
    )
    cfg = _probe_config(window_size, crystal_w_chunks, woodbury_latch_span)
    cfg_s1 = replace(cfg, byte_chunk_mode="byte", target_chunk_bytes=4.85, use_static_m_cap=False)
    use_amp = device == "cuda" and has_usable_cuda()

    X_tr, Y_tr = build_shared_training_stream(
        tok, tr_docs, tr_sents, n_steps=steps, batch_size=batch_size, seq_len=seq_len, seed=seed
    )
    _gate_tmp = CalciteLeanSystem1LM(cfg).to(device)
    gate2 = check_gate2_block_std_parity(_gate_tmp, device)
    assert gate2["gate2_passed"], f"Gate 2 failed for {tag}"
    del _gate_tmp

    model_seed = 100 if seed == 2026 else 101
    torch.manual_seed(model_seed)
    random.seed(model_seed)
    model = CalciteLeanSystem1LM(cfg_s1).to(device)
    bc = model.byte_chunker
    t0 = time.perf_counter()

    # Stage 1A (`1 .. steps`): Crystal S_5 grokking (`woodbury` at 0.05x LR, wd=0.01 + tail floor `min_lr=1.5e-4`)
    non_wood_p = []
    wood_p = []
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if pname.startswith("woodbury."):
            wood_p.append(p)
        else:
            non_wood_p.append(p)
    all_1a_params = non_wood_p + wood_p
    base_lr_1a = 2.5e-3
    min_lr_1a = 1.5e-4
    opt = _make_adamw(
        [
            {"params": non_wood_p, "lr": base_lr_1a, "weight_decay": 0.01},
            {"params": wood_p, "lr": 0.05 * base_lr_1a, "weight_decay": 0.01},
        ],
        device,
    )

    for step in range(1, steps + 1):
        model.train()
        model.set_crystal_schedule(step, steps)
        if step < 60:
            lr_now = base_lr_1a * (step / 60.0)
        else:
            lr_now = min_lr_1a + 0.5 * (base_lr_1a - min_lr_1a) * (
                1.0 + math.cos(math.pi * (step - 60) / float(max(1, steps - 60)))
            )
        opt.param_groups[0]["lr"] = lr_now
        opt.param_groups[1]["lr"] = 0.05 * lr_now

        st = (step - 1) * batch_size
        xb = X_tr[st : st + batch_size].to(device, non_blocking=True)
        yb = Y_tr[st : st + batch_size].to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            lg = model(xb)
            t_ce = F.cross_entropy(
                lg.transpose(1, 2), yb, ignore_index=tok.PAD_ID, reduction="none"
            )
            denom = (yb != tok.PAD_ID).sum(dim=1).clamp(min=1)
            ce_loss = (t_ce.sum(dim=1) / denom).mean()
            loss = ce_loss + 0.04 * model.get_commitment_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_1a_params, 1.0)
        opt.step()

        if step in (1, 800, 1600, 2200, steps):
            sync_if_cuda(device)
            ce_v = float(ce_loss.item())
            assert math.isfinite(ce_v), f"Non-finite Stage 1A CE at step {step}"
            alpha = model.crystal.snap_alpha if model.crystal is not None else 0.0
            b_ch = bc.last_chunk_ratio if bc is not None else 1.0
            c2_b0 = (
                float(model.crystal_2.perm_gate_proj.bias[0].item())
                if model.crystal_2 is not None
                else 0.0
            )
            s5_str = ""
            if step in (800, 1600, 2200, steps) and steps >= 800:
                py_rng_st = random.getstate()
                th_rng_st = torch.get_rng_state()
                cu_rng_st = torch.cuda.get_rng_state_all() if use_amp else None
                try:
                    n_q = min(32, len(s5_suites["k2"]))
                    model.eval()
                    model.set_snap_alpha(1.0)
                    pk2, _, _, _ = evaluate_suite_batched(
                        model, s5_suites["k2"][:n_q], pad_id=tok.PAD_ID, batch_size=32, device=device
                    )
                    s5_str = f" s5_k2_32={pk2:.1f}%"
                finally:
                    random.setstate(py_rng_st)
                    torch.set_rng_state(th_rng_st)
                    if cu_rng_st is not None:
                        torch.cuda.set_rng_state_all(cu_rng_st)
                    model.train()
                    model.set_crystal_schedule(step, steps)
            print(
                f"  [{tag} s={seed} S1A-byte] step {step:4d}/{steps} alpha={alpha:.2f} "
                f"b/ch={b_ch:.2f} c2_b0={c2_b0:+.2f} ce={ce_v:.4f}{s5_str} ({time.perf_counter()-t0:.1f}s)",
                flush=True,
            )

    del opt, X_tr, Y_tr
    if device == "cuda":
        torch.cuda.empty_cache()

    model.eval()
    model.set_crystal_schedule(steps, steps)
    model.set_snap_alpha(1.0)
    s1a_k2_pct, _, _, _ = evaluate_suite_batched(
        model, s5_suites["k2"], pad_id=tok.PAD_ID, batch_size=32, device=device
    )
    s1a_byte_s5_k2_pct = round(float(s1a_k2_pct), 2)
    sg_b_1a = (
        [round(float(x), 3) for x in model.stream_gate.bias.detach().tolist()]
        if getattr(model, "stream_gate", None) is not None
        else []
    )
    print(
        f"  [{tag} s={seed} S1A-byte EVAL] s1a_byte_s5_k2_pct={s1a_byte_s5_k2_pct:.2f}% sg_b={sg_b_1a}",
        flush=True,
    )
    s1a_min = 50.0
    abort_s1a = (tag in ("V0_phase_be_rings", "V1_lean_pruned")) and seed == 2026
    if steps >= 2000 and abort_s1a and s1a_byte_s5_k2_pct < s1a_min:
        if on_stage1_abort is not None:
            on_stage1_abort(
                {
                    "tag": tag,
                    "seed": seed,
                    "window_size": window_size,
                    "crystal_w_chunks": crystal_w_chunks,
                    "woodbury_latch_span": woodbury_latch_span,
                    "stage1a_abort": True,
                    "s1a_byte_s5_k2_pct": s1a_byte_s5_k2_pct,
                }
            )
        raise AssertionError(
            f"Stop-Crane K15 Stage-1A triggered on {tag} s={seed}: "
            f"s1a_byte_s5_k2_pct={s1a_byte_s5_k2_pct:.2f}% < {s1a_min:.1f}%"
        )

    # Stage 1B (`1 .. ov_steps_eff`): Associative Woodbury Consolidation (router locked, snap_alpha=1.0)
    if bc is not None:
        bc.lock_router(True)
    X_1b, Y_1b = _build_stage1b_consolidation_stream(
        tok, tr_docs, tr_sents, n_steps=ov_steps_eff, batch_size=batch_size, seq_len=seq_len, seed=seed + 50
    )
    cryst_1b, pre_nodc, pre_2d, post_nodc, post_2d, wood_nodc, wood_2d = _split_stage1b_groups(model)
    all_1b_params = cryst_1b + pre_nodc + pre_2d + post_nodc + post_2d + wood_nodc + wood_2d
    base_lrs_1b = (4.0e-5, 2.5e-4, 2.5e-4, 1.2e-3, 1.2e-3, 2.4e-3, 2.4e-3)
    opt_1b = _make_adamw(
        [
            {"params": cryst_1b, "lr": base_lrs_1b[0], "weight_decay": 0.0},
            {"params": pre_nodc, "lr": base_lrs_1b[1], "weight_decay": 0.0},
            {"params": pre_2d, "lr": base_lrs_1b[2], "weight_decay": 0.01},
            {"params": post_nodc, "lr": base_lrs_1b[3], "weight_decay": 0.0},
            {"params": post_2d, "lr": base_lrs_1b[4], "weight_decay": 0.01},
            {"params": wood_nodc, "lr": base_lrs_1b[5], "weight_decay": 0.0},
            {"params": wood_2d, "lr": base_lrs_1b[6], "weight_decay": 0.01},
        ],
        device,
    )
    warm_1b = min(40, max(1, ov_steps_eff // 8))
    log_1b_steps = {1, max(1, ov_steps_eff // 3), max(1, (2 * ov_steps_eff) // 3), ov_steps_eff}
    for ostep in range(1, ov_steps_eff + 1):
        model.train()
        model.set_crystal_schedule(steps, steps)
        model.set_snap_alpha(1.0)
        if ostep < warm_1b:
            lr_mult = ostep / float(warm_1b)
        else:
            lr_mult = 0.15 + 0.85 * 0.5 * (
                1.0 + math.cos(math.pi * (ostep - warm_1b) / max(1, ov_steps_eff - warm_1b))
            )
        for gi, blr in enumerate(base_lrs_1b):
            opt_1b.param_groups[gi]["lr"] = blr * lr_mult

        st = (ostep - 1) * batch_size
        xb = X_1b[st : st + batch_size].to(device, non_blocking=True)
        yb = Y_1b[st : st + batch_size].to(device, non_blocking=True)
        opt_1b.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            lg = model(xb)
            t_ce = F.cross_entropy(
                lg.transpose(1, 2), yb, ignore_index=tok.PAD_ID, reduction="none"
            )
            denom = (yb != tok.PAD_ID).sum(dim=1).clamp(min=1)
            ce_loss = (t_ce.sum(dim=1) / denom).mean()
            loss = ce_loss + 0.04 * model.get_commitment_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_1b_params, 1.0)
        opt_1b.step()

        if ostep in log_1b_steps:
            sync_if_cuda(device)
            ce_v = float(ce_loss.item())
            assert math.isfinite(ce_v), f"Non-finite Stage 1B CE at step {ostep}"
            sg_b_str = (
                f" sg_b={[round(float(x), 3) for x in model.stream_gate.bias.detach().tolist()]}"
                if getattr(model, "stream_gate", None) is not None
                else ""
            )
            ov32_str = ""
            if ov_steps_eff >= 300 and ostep > 1:
                py_rng_st = random.getstate()
                th_rng_st = torch.get_rng_state()
                cu_rng_st = torch.cuda.get_rng_state_all() if use_amp else None
                try:
                    n_ovq = min(32, len(ov_suites["ood_long"]))
                    model.eval()
                    model.set_crystal_schedule(steps, steps)
                    model.set_snap_alpha(1.0)
                    pov, _, _, _ = evaluate_suite_batched(
                        model, ov_suites["ood_long"][:n_ovq], pad_id=tok.PAD_ID, batch_size=32, device=device
                    )
                    ov32_str = f" ov32={pov:.1f}%"
                finally:
                    random.setstate(py_rng_st)
                    torch.set_rng_state(th_rng_st)
                    if cu_rng_st is not None:
                        torch.cuda.set_rng_state_all(cu_rng_st)
                    model.train()
                    model.set_crystal_schedule(steps, steps)
                    model.set_snap_alpha(1.0)
            print(
                f"  [{tag} s={seed} S1B-wood] step {ostep:4d}/{ov_steps_eff} ce={ce_v:.4f}{ov32_str}{sg_b_str} ({time.perf_counter()-t0:.1f}s)",
                flush=True,
            )

    del opt_1b, X_1b, Y_1b
    if device == "cuda":
        torch.cuda.empty_cache()

    model.eval()
    model.set_crystal_schedule(steps, steps)
    model.set_snap_alpha(1.0)
    s1_k2_pct, _, _, _ = evaluate_suite_batched(
        model, s5_suites["k2"], pad_id=tok.PAD_ID, batch_size=32, device=device
    )
    s1_ov_pct, _, _, _ = evaluate_suite_batched(
        model, ov_suites["ood_long"], pad_id=tok.PAD_ID, batch_size=32, device=device
    )
    s1_tr_pct, _, _, _ = evaluate_suite_batched(
        model, trap_suites["ood_long"], pad_id=tok.PAD_ID, batch_size=32, device=device
    )
    s1_byte_s5_k2_pct = round(float(s1_k2_pct), 2)
    s1_byte_ov_long_pct = round(float(s1_ov_pct), 2)
    s1_byte_trap_long_pct = round(float(s1_tr_pct), 2)
    print(
        f"  [{tag} s={seed} S1B-byte DUAL GATE] s1_s5_k2={s1_byte_s5_k2_pct:.2f}% | "
        f"s1_ov_long={s1_byte_ov_long_pct:.2f}% | s1_trap_long={s1_byte_trap_long_pct:.2f}%",
        flush=True,
    )
    s1b_s5_min = 55.0
    abort_s1b = (tag in ("V0_phase_be_rings", "V1_lean_pruned")) and seed == 2026
    if steps >= 2000 and abort_s1b and (s1_byte_s5_k2_pct < s1b_s5_min or s1_byte_ov_long_pct < 70.0):
        if on_stage1_abort is not None:
            on_stage1_abort(
                {
                    "tag": tag,
                    "seed": seed,
                    "window_size": window_size,
                    "crystal_w_chunks": crystal_w_chunks,
                    "woodbury_latch_span": woodbury_latch_span,
                    "stage1b_abort": True,
                    "s1a_byte_s5_k2_pct": s1a_byte_s5_k2_pct,
                    "s1_byte_s5_k2_pct": s1_byte_s5_k2_pct,
                    "s1_byte_ov_long_pct": s1_byte_ov_long_pct,
                    "s1_byte_trap_long_pct": s1_byte_trap_long_pct,
                }
            )
        raise AssertionError(
            f"Stop-Crane K15 Stage-1B Dual Gate triggered on {tag} s={seed}: "
            f"s1_s5_k2={s1_byte_s5_k2_pct:.2f}%, s1_ov_long={s1_byte_ov_long_pct:.2f}%"
        )

    # Stage 1.5: 120-step Non-ASCII Router Calibration (`router_emb[131:]` only, keeping `use_static_m_cap=False` in train)
    cal_steps = min(120, max(4, utf8_steps // 4)) if utf8_steps < 120 else 120
    if bc is not None:
        bc.byte_chunk_mode = "utf8"
        bc.use_static_m_cap = False
        bc.lock_router(True)
        for p in model.parameters():
            p.requires_grad_(False)
        bc.router_locked = False
        bc.router_emb.weight.requires_grad_(True)
        bc.target_chunk_bytes = 4.3
        opt_r = _make_adamw([bc.router_emb.weight], device, lr=2.0e-3, weight_decay=0.0)
        cal_pools = [
            train_wins[k]
            for k in ("2_Russian_UTF8", "5_CJK_UTF8", "6_Emoji_4Byte")
            if len(train_wins.get(k, [])) > 0
        ]
        rng_cal = random.Random(seed)
        model.train()
        for _ in range(cal_steps):
            batch = [rng_cal.choice(rng_cal.choice(cal_pools))[:-1] for _ in range(8)]
            xb_c = torch.tensor(batch, dtype=torch.long, device=device)
            e_tok_c = model.tok_emb(xb_c)
            _, _, _ = bc.encode_to_chunks(e_tok_c, xb_c)
            loss_r = bc.last_ratio_loss
            opt_r.zero_grad(set_to_none=True)
            loss_r.backward()
            if bc.router_emb.weight.grad is not None:
                bc.router_emb.weight.grad[:131].zero_()
            torch.nn.utils.clip_grad_norm_([bc.router_emb.weight], 1.0)
            opt_r.step()
        del opt_r
        bc.router_emb.weight.requires_grad_(False)
        bc.target_chunk_bytes = float(cfg.target_chunk_bytes)
        bc.router_locked = True
        for p in model.parameters():
            p.requires_grad_(True)
        bc.lock_router(True)
        print(
            f"  [{tag} s={seed} S1.5-cal] calibrated router_emb[131:] ({cal_steps} steps, utf8, tgt={cfg.target_chunk_bytes})",
            flush=True,
        )

    # Stage 2: Multilingual UTF-8 + Cognitive Maintenance at `BS=64`, protecting Crystal (`6e-5`) & Woodbury (`8e-5`)
    bs_u = batch_size
    X_mix, Y_mix = build_multilingual_plus_cognitive_stream(
        tok, train_wins, tr_sents, n_steps=utf8_steps, batch_size=bs_u, seq_len=seq_len, seed=seed + 100
    )
    cryst_fp_params = []
    wood_fp_params = []
    nodc_other_params = []
    decay_other_params = []
    seen_s2: set[int] = set()
    for pname, p in model.named_parameters():
        if (not p.requires_grad) or (id(p) in seen_s2):
            continue
        seen_s2.add(id(p))
        if pname.startswith(("crystal.", "crystal_2.")):
            cryst_fp_params.append(p)
        elif pname.startswith("woodbury."):
            wood_fp_params.append(p)
        elif p.ndim < 2 or pname.startswith(("stream_gate.", "cryst2_gate.")):
            nodc_other_params.append(p)
        else:
            decay_other_params.append(p)
    all_u_params = cryst_fp_params + wood_fp_params + nodc_other_params + decay_other_params
    s2_lrs = (6.0e-5, 8.0e-5, 2.5e-4, 2.5e-4)
    opt_u = _make_adamw(
        [
            {"params": cryst_fp_params, "lr": s2_lrs[0], "weight_decay": 0.0},
            {"params": wood_fp_params, "lr": s2_lrs[1], "weight_decay": 0.0},
            {"params": nodc_other_params, "lr": s2_lrs[2], "weight_decay": 0.0},
            {"params": decay_other_params, "lr": s2_lrs[3], "weight_decay": 0.01},
        ],
        device,
    )
    warm_u = min(30, max(1, utf8_steps // 5))
    for ustep in range(1, utf8_steps + 1):
        model.train()
        model.set_crystal_schedule(steps, steps)
        model.set_snap_alpha(1.0)
        if ustep < warm_u:
            lr_mult = ustep / warm_u
        else:
            lr_mult = 0.15 + 0.85 * 0.5 * (
                1.0 + math.cos(math.pi * (ustep - warm_u) / max(1, utf8_steps - warm_u))
            )
        for gi, blr in enumerate(s2_lrs):
            opt_u.param_groups[gi]["lr"] = blr * lr_mult

        st = (ustep - 1) * bs_u
        xb = X_mix[st : st + bs_u].to(device, non_blocking=True)
        yb = Y_mix[st : st + bs_u].to(device, non_blocking=True)
        opt_u.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            lg = model(xb)
            t_ce = F.cross_entropy(
                lg.transpose(1, 2), yb, ignore_index=tok.PAD_ID, reduction="none"
            )
            denom = (yb != tok.PAD_ID).sum(dim=1).clamp(min=1)
            ce_loss = (t_ce.sum(dim=1) / denom).mean()
            loss = ce_loss + 0.04 * model.get_commitment_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_u_params, 1.0)
        opt_u.step()

        if ustep in (1, max(1, utf8_steps // 2), utf8_steps):
            sync_if_cuda(device)
            ce_v = float(ce_loss.item())
            assert math.isfinite(ce_v), f"Non-finite Stage 2 CE at step {ustep}"
            print(
                f"  [{tag} s={seed} S2-utf8] step {ustep:4d}/{utf8_steps} "
                f"ce={ce_v:.4f} ({time.perf_counter()-t0:.1f}s)",
                flush=True,
            )

    train_s = round(time.perf_counter() - t0, 2)
    del opt_u, X_mix, Y_mix
    if device == "cuda":
        torch.cuda.empty_cache()

    model.eval()
    if bc is not None:
        bc.byte_chunk_mode = "utf8"
        bc.target_chunk_bytes = float(cfg.target_chunk_bytes)
        bc.use_static_m_cap = True
    model.set_crystal_schedule(steps, steps)
    model.set_snap_alpha(1.0)
    ow = eval_openweb_6domains_any_byte_lm(model, eval_wins, device, batch_size=16)
    cog = eval_cognitive_suites_any_byte_lm(model, tok, s5_suites, ov_suites, trap_suites, device, batch_size=32)
    lat_ms = measure_infer_latency_ms(model, cfg.vocab_size, device, batch_size=8, seq_len=seq_len)
    cache_kb = round(compute_lean_state_cache_kb(cfg, 2048), 2)
    n_params = int(sum(p.numel() for p in model.parameters()))

    out = {
        "tag": tag,
        "seed": seed,
        "window_size": window_size,
        "crystal_w_chunks": crystal_w_chunks,
        "woodbury_latch_span": woodbury_latch_span,
        "params": n_params,
        "cache_kb_L2048": cache_kb,
        "train_seconds": train_s,
        "infer_b8_l768_ms": lat_ms,
        "gate2_init_std": gate2,
        "s1a_byte_s5_k2_pct": s1a_byte_s5_k2_pct,
        "s1_byte_s5_k2_pct": s1_byte_s5_k2_pct,
        "s1_byte_ov_long_pct": s1_byte_ov_long_pct,
        "s1_byte_trap_long_pct": s1_byte_trap_long_pct,
        "mean_openweb_bpb": ow["mean_openweb_bpb"],
        "max_mid_utf8_split_pct": ow["max_mid_utf8_split_pct"],
        "openweb": ow,
        "s5_sum": cog["s5_sum"],
        "s5": cog["s5"],
        "ov_ood_long_pct": cog["overwrite"]["ood_long"]["exact_pct"],
        "overwrite": cog["overwrite"],
        "trap": cog["trap"],
        "selective_f1_in": cog["selective_f1_in"],
        "selective_f1_ood_long": cog["selective_f1_ood_long"],
        "crystal_drift_L2048": cog["crystal_drift_L2048"],
    }
    print(
        f"  => [{tag} s={seed}] cache={cache_kb:.2f}KB | ow_bpb={out['mean_openweb_bpb']:.4f} "
        f"(mid_utf8={out['max_mid_utf8_split_pct']:.2f}%) | S5_sum={out['s5_sum']:.1f}% | "
        f"OV_long={out['ov_ood_long_pct']:.1f}% | SelF1_long={out['selective_f1_ood_long']:.1f}% | "
        f"lat={lat_ms:.2f}ms",
        flush=True,
    )
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def _tier_reduction_summary(window_size: int, crystal_w_chunks: int, woodbury_latch_span: int) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for tier in ("S1_5M", "S2_25M", "S3_85M"):
        c_lean = compute_lean_state_cache_kb(
            build_tier_lean_config(tier, window_size, crystal_w_chunks, woodbury_latch_span), 2048
        )
        c_leg = compute_lean_state_cache_kb(build_tier_lean_config(tier, 12, 5, 7), 2048)
        c_w8 = compute_lean_state_cache_kb(build_tier_lean_config(tier, 8, 5, 7), 2048)
        out[tier] = {
            "legacy_kb": round(c_leg, 2),
            "phase_be_w8_kb": round(c_w8, 2),
            "lean_kb": round(c_lean, 2),
            "reduction_pct": round(100.0 * (1.0 - c_lean / c_leg), 2),
            "reduction_vs_w8_pct": round(100.0 * (1.0 - c_lean / c_w8), 2),
        }
    return out


VARIANT_SPECS: dict[str, tuple[str, int, int, int, int]] = {
    "v0_seed2026": ("V0_phase_be_rings", 8, 5, 7, 2026),
    "v0_seed2027": ("V0_phase_be_rings", 8, 5, 7, 2027),
    "v1_attempt1": ("V1_lean_pruned", 8, 2, 6, 2026),
    "v1b_attempt2": ("V1b_moderate_pruned", 8, 3, 7, 2026),
}
PROBE_RECIPE = "k15_s1a005_s1b800_2800_800_320"


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_p = path.with_suffix(path.suffix + ".tmp")
    tmp_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_p.replace(path)


def _check_v0_variant_gate(key: str, res: dict, steps: int) -> None:
    if steps < 2000:
        return
    if key == "v0_seed2026":
        assert res["s5_sum"] >= 80.0 and res["ov_ood_long_pct"] >= 80.0, (
            f"Stop-Crane K15 triggered on V0_seed2026: s5_sum={res['s5_sum']} ov_long={res['ov_ood_long_pct']}"
        )
    elif key == "v0_seed2027":
        assert res["ov_ood_long_pct"] >= 80.0, (
            f"Stop-Crane K15 triggered on V0_seed2027: s5_sum={res['s5_sum']} ov_long={res['ov_ood_long_pct']}"
        )
    elif key == "v1_attempt1":
        assert res["s5_sum"] >= 80.0 and res["ov_ood_long_pct"] >= 80.0, (
            f"Stop-Crane K15 triggered on {key}: s5_sum={res['s5_sum']} ov_long={res['ov_ood_long_pct']}"
        )


def _build_selected_summary(
    variant_name: str,
    window_size: int,
    crystal_w_chunks: int,
    woodbury_latch_span: int,
    cache_kb: float,
    v0_cache_kb: float,
    c_w12_probe: float,
) -> dict:
    return {
        "variant": variant_name,
        "window_size": window_size,
        "crystal_w_chunks": crystal_w_chunks,
        "woodbury_latch_span": woodbury_latch_span,
        "probe_cache_reduction_pct": round(100.0 * (1.0 - cache_kb / v0_cache_kb), 2),
        "probe_cache_reduction_vs_w12_pct": round(100.0 * (1.0 - cache_kb / c_w12_probe), 2),
        "tier_cache_reductions": _tier_reduction_summary(
            window_size, crystal_w_chunks, woodbury_latch_span
        ),
    }


def _passes_non_regression(cand: dict, v0_mean: dict, tolerances: dict) -> tuple[bool, dict]:
    tol = tolerances["effective_gate_tol"]
    checks = {
        "utf8_zero_cuts": float(cand["max_mid_utf8_split_pct"]) == 0.0,
        "openweb_bpb_ok": cand["mean_openweb_bpb"] <= v0_mean["mean_openweb_bpb"] + tol["mean_openweb_bpb"],
        "s5_sum_ok": cand["s5_sum"] >= v0_mean["s5_sum"] - tol["s5_sum"],
        "ov_long_ok": cand["ov_ood_long_pct"] >= v0_mean["ov_ood_long_pct"] - tol["ov_ood_long_pct"],
        "sel_f1_long_ok": cand["selective_f1_ood_long"] >= v0_mean["selective_f1_ood_long"] - tol["selective_f1_ood_long"],
    }
    return all(checks.values()), checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2800)
    parser.add_argument("--ov-steps", type=int, default=800)
    parser.add_argument("--utf8-steps", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=768)
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-variants", type=int, default=0)
    parser.add_argument(
        "--only-variant",
        default="all",
        choices=("all", "v0_seed2026", "v0_seed2027", "v1_attempt1", "v1b_attempt2", "finalize"),
    )
    parser.add_argument("--variant-out", default="")
    args = parser.parse_args()
    assert args.seq_len >= 512, f"Raw-byte S_5 episodes require seq_len >= 512 (got {args.seq_len})"

    device = "cuda" if has_usable_cuda() else "cpu"
    n_ev = int(args.n_eval)
    partial_p = ROOT_DIR / "benchmarks" / "phase_k_via_negativa_probe.partial.json"
    partial_p.parent.mkdir(parents=True, exist_ok=True)
    proto_info = {
        "recipe": PROBE_RECIPE,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "probe_d_model": 160,
        "steps": args.steps,
        "ov_steps": args.ov_steps,
        "utf8_steps": args.utf8_steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "n_eval": n_ev,
    }
    partial_state: dict = {"protocol": proto_info}
    if (args.resume or args.only_variant == "finalize") and partial_p.exists():
        try:
            loaded = json.loads(partial_p.read_text(encoding="utf-8"))
            l_proto = loaded.get("protocol", {}) if isinstance(loaded, dict) else {}
            if (
                isinstance(loaded, dict)
                and l_proto.get("recipe") == PROBE_RECIPE
                and l_proto.get("steps") == args.steps
                and l_proto.get("ov_steps") == args.ov_steps
                and l_proto.get("utf8_steps") == args.utf8_steps
            ):
                partial_state = loaded
                proto_info = l_proto
                print(f"[Resume] Loaded partial probe state keys={list(partial_state.keys())}", flush=True)
        except Exception as exc:
            print(f"[Resume] Ignoring unreadable partial {partial_p}: {exc}", flush=True)

    def _save_partial(key: str, val: dict) -> None:
        partial_state[key] = val
        _write_json_atomic(partial_p, partial_state)

    if args.only_variant != "finalize":
        tok = ByteBPETokenizer(merges=[])
        tr_docs, te_docs, tr_sents, te_sents = load_gsm8k_corpora(ROOT_DIR)
        train_wins, eval_wins, _, _ = build_disjoint_openweb_corpora(
            ROOT_DIR, tok, tr_docs, te_docs, seq_len=args.seq_len
        )
        s5_suites = {
            "k2": build_eval_s5_suite(tok, te_sents, 2, n_ev, (0, 1), 3100),
            "k4": build_eval_s5_suite(tok, te_sents, 4, n_ev, (0, 1), 3200),
            "k8": build_eval_s5_suite(tok, te_sents, 8, n_ev, (1, 1), 3300),
            "k16": build_eval_s5_suite(tok, te_sents, 16, n_ev, (1, 2), 3400),
        }
        ov_suites = {
            "in_dist": build_eval_overwrite_suite(tok, te_sents, 4, 2, 2, n_ev, 4100),
            "ood_med": build_eval_overwrite_suite(tok, te_sents, 7, 4, 10, n_ev, 4200),
            "ood_long": build_eval_overwrite_suite(tok, te_sents, 9, 5, 18, n_ev, 4300),
        }
        trap_suites = {
            "in_dist": build_eval_trap_suite(tok, te_sents, 4, 2, 2, n_ev, 5100),
            "ood_long": build_eval_trap_suite(tok, te_sents, 9, 5, 18, n_ev, 5300),
        }

        if args.only_variant in VARIANT_SPECS:
            key = args.only_variant
            tag, w_sz, w_ch, l_sp, vseed = VARIANT_SPECS[key]
            out_v = Path(args.variant_out) if args.variant_out else (ROOT_DIR / "benchmarks" / f"phase_k_probe_{key}.json")

            def _abort_v(d: dict) -> None:
                _write_json_atomic(out_v, {"protocol": proto_info, "key": f"{key}_stage1_abort", "result": d})

            res_v = _train_and_eval_variant(
                tag, w_sz, w_ch, l_sp, vseed, args.steps, args.utf8_steps, args.batch_size, args.seq_len,
                train_wins, eval_wins, tr_docs, tr_sents, s5_suites, ov_suites, trap_suites, tok, device,
                ov_steps=args.ov_steps,
                on_stage1_abort=_abort_v,
            )
            _write_json_atomic(out_v, {"protocol": proto_info, "key": key, "result": res_v})
            _check_v0_variant_gate(key, res_v, args.steps)
            print(f"[DONE] Variant `{key}` saved to {out_v}", flush=True)
            return 0

        ran_now = 0
        # 1. Pre-flight 2*sigma calibration on Phase B/E baseline V0_phase_be_rings (W=8, W_chunks=5, L_latch=7)
        if "v0_seed2026" in partial_state:
            v0_s1 = partial_state["v0_seed2026"]
            print("[Resume] Reusing completed v0_seed2026", flush=True)
        else:
            v0_s1 = _train_and_eval_variant(
                "V0_phase_be_rings", 8, 5, 7, 2026, args.steps, args.utf8_steps, args.batch_size, args.seq_len,
                train_wins, eval_wins, tr_docs, tr_sents, s5_suites, ov_suites, trap_suites, tok, device,
                ov_steps=args.ov_steps,
                on_stage1_abort=lambda d: _save_partial("v0_seed2026_stage1_abort", d),
            )
            _save_partial("v0_seed2026", v0_s1)
            ran_now += 1
        _check_v0_variant_gate("v0_seed2026", v0_s1, args.steps)
        if args.max_variants > 0 and ran_now >= args.max_variants:
            print(f"[Yield] Reached --max-variants={args.max_variants} after v0_seed2026.", flush=True)
            return 0

        if "v0_seed2027" in partial_state:
            v0_s2 = partial_state["v0_seed2027"]
            print("[Resume] Reusing completed v0_seed2027", flush=True)
        else:
            v0_s2 = _train_and_eval_variant(
                "V0_phase_be_rings", 8, 5, 7, 2027, args.steps, args.utf8_steps, args.batch_size, args.seq_len,
                train_wins, eval_wins, tr_docs, tr_sents, s5_suites, ov_suites, trap_suites, tok, device,
                ov_steps=args.ov_steps,
                on_stage1_abort=lambda d: _save_partial("v0_seed2027_stage1_abort", d),
            )
            _save_partial("v0_seed2027", v0_s2)
            ran_now += 1
        _check_v0_variant_gate("v0_seed2027", v0_s2, args.steps)
        if args.max_variants > 0 and ran_now >= args.max_variants:
            print(f"[Yield] Reached --max-variants={args.max_variants} after v0_seed2027.", flush=True)
            return 0

        if "v1_attempt1" not in partial_state:
            v1_tmp = _train_and_eval_variant(
                "V1_lean_pruned", 8, 2, 6, 2026, args.steps, args.utf8_steps, args.batch_size, args.seq_len,
                train_wins, eval_wins, tr_docs, tr_sents, s5_suites, ov_suites, trap_suites, tok, device,
                ov_steps=args.ov_steps,
            )
            _save_partial("v1_attempt1", v1_tmp)
            ran_now += 1
    else:
        ran_now = 0

    assert "v0_seed2026" in partial_state and "v0_seed2027" in partial_state and "v1_attempt1" in partial_state, (
        f"Missing required probe variants in {partial_p}: got {list(partial_state.keys())}"
    )
    v0_s1 = partial_state["v0_seed2026"]
    v0_s2 = partial_state["v0_seed2027"]
    eff_steps = int(proto_info.get("steps", args.steps))
    _check_v0_variant_gate("v0_seed2026", v0_s1, eff_steps)
    _check_v0_variant_gate("v0_seed2027", v0_s2, eff_steps)

    sigma_ow = abs(v0_s1["mean_openweb_bpb"] - v0_s2["mean_openweb_bpb"])
    sigma_s5 = abs(v0_s1["s5_sum"] - v0_s2["s5_sum"])
    sigma_ov = abs(v0_s1["ov_ood_long_pct"] - v0_s2["ov_ood_long_pct"])
    sigma_f1 = abs(v0_s1["selective_f1_ood_long"] - v0_s2["selective_f1_ood_long"])

    v0_mean = {
        "cache_kb_L2048": v0_s1["cache_kb_L2048"],
        "mean_openweb_bpb": round(0.5 * (v0_s1["mean_openweb_bpb"] + v0_s2["mean_openweb_bpb"]), 4),
        "s5_sum": round(0.5 * (v0_s1["s5_sum"] + v0_s2["s5_sum"]), 2),
        "ov_ood_long_pct": round(0.5 * (v0_s1["ov_ood_long_pct"] + v0_s2["ov_ood_long_pct"]), 2),
        "selective_f1_ood_long": round(0.5 * (v0_s1["selective_f1_ood_long"] + v0_s2["selective_f1_ood_long"]), 2),
    }
    v1 = dict(partial_state["v1_attempt1"])
    if eff_steps >= 2000:
        assert v0_mean["s5_sum"] >= 80.0 and v0_mean["ov_ood_long_pct"] >= 80.0, (
            f"V0 failed grokking bar before pruning: s5_sum={v0_mean['s5_sum']} ov_long={v0_mean['ov_ood_long_pct']}"
        )

    tolerances = {
        "raw_2sigma": {
            "mean_openweb_bpb": round(2.0 * sigma_ow, 4),
            "s5_sum": round(2.0 * sigma_s5, 2),
            "ov_ood_long_pct": round(2.0 * sigma_ov, 2),
            "selective_f1_ood_long": round(2.0 * sigma_f1, 2),
        },
        "effective_gate_tol": {
            "mean_openweb_bpb": round(max(2.0 * sigma_ow, 0.04), 4),
            "s5_sum": round(max(2.0 * sigma_s5, 4.0), 2),
            "ov_ood_long_pct": round(max(2.0 * sigma_ov, 3.0), 2),
            "selective_f1_ood_long": round(max(2.0 * sigma_f1, 3.0), 2),
        },
    }
    c_w12_probe = compute_lean_state_cache_kb(_probe_config(12, 5, 7), 2048)

    v1_pass, v1_checks = _passes_non_regression(v1, v0_mean, tolerances)
    if eff_steps >= 2000:
        anti_relax = {
            "v1_abs_s5_ok": v1["s5_sum"] >= 80.0 and v1["s5_sum"] >= v0_s1["s5_sum"] - 25.0,
            "v1_abs_ov_ok": v1["ov_ood_long_pct"] >= 80.0 and v1["ov_ood_long_pct"] >= v0_s1["ov_ood_long_pct"] - 10.0,
            "v1_abs_f1_ok": v1["selective_f1_ood_long"] >= v0_s1["selective_f1_ood_long"] - 10.0,
            "v1_abs_ow_ok": v1["mean_openweb_bpb"] <= v0_s1["mean_openweb_bpb"] + 0.08,
            "v1_abs_utf8_ok": float(v1["max_mid_utf8_split_pct"]) == 0.0,
        }
        v1_checks.update(anti_relax)
        v1_pass = all(v1_checks.values())
    v1["parity_checks"] = v1_checks
    v1["passed"] = v1_pass
    partial_state["v1_attempt1"] = v1

    v1b = None
    if v1_pass:
        selected = _build_selected_summary(
            "V1_lean_pruned", 8, 2, 6, v1["cache_kb_L2048"], v0_s1["cache_kb_L2048"], c_w12_probe
        )
    else:
        if "v1b_attempt2" not in partial_state:
            if args.only_variant == "finalize":
                print("[NEED_V1B] V1_lean_pruned failed non-regression; v1b_attempt2 required.", flush=True)
                return 2
            if args.max_variants > 0 and ran_now >= args.max_variants:
                print(f"[Yield] Reached --max-variants={args.max_variants} before v1b_attempt2.", flush=True)
                return 0
            v1b_raw = _train_and_eval_variant(
                "V1b_moderate_pruned", 8, 3, 7, 2026, args.steps, args.utf8_steps, args.batch_size, args.seq_len,
                train_wins, eval_wins, tr_docs, tr_sents, s5_suites, ov_suites, trap_suites, tok, device,
                ov_steps=args.ov_steps,
            )
            _save_partial("v1b_attempt2", v1b_raw)
        v1b = dict(partial_state["v1b_attempt2"])
        v1b_pass, v1b_checks = _passes_non_regression(v1b, v0_mean, tolerances)
        v1b["parity_checks"] = v1b_checks
        v1b["passed"] = v1b_pass
        if v1b_pass:
            selected = _build_selected_summary(
                "V1b_moderate_pruned", 8, 3, 7, v1b["cache_kb_L2048"], v0_s1["cache_kb_L2048"], c_w12_probe
            )
        else:
            c_fb = compute_lean_state_cache_kb(_probe_config(8, 5, 7), 2048)
            selected = _build_selected_summary(
                "V0_phase_be_rings", 8, 5, 7, c_fb, v0_s1["cache_kb_L2048"], c_w12_probe
            )

    prior_diagnostics = {
        "k1c_w12_mixed_batch": {
            "tag": "W12_mixed_batch_K1c",
            "window_size": 12,
            "crystal_w_chunks": 5,
            "woodbury_latch_span": 7,
            "seed": 2026,
            "steps": 2200,
            "device": "NVIDIA L4",
            "cache_kb_L2048": 90.23,
            "train_seconds": 1181.37,
            "mean_openweb_bpb": 4.1357,
            "max_mid_utf8_split_pct": 0.0,
            "en_bytes_per_chunk": 4.499,
            "s5_sum": 5.0,
            "ov_ood_long_pct": 97.0,
            "selective_f1_ood_long": 90.6,
            "stop_crane_triggered": True,
            "root_cause": (
                "Mixed high-entropy CJK/Emoji/RU rows in same batch as S_5 starved crystal capacity (crystal.py:646-650) and attenuated S_5 grad norm"
            ),
        },
        "k2a_w8_twophase_mixed_batch": {
            "tag": "W8_twophase_mixed_batch_K2a",
            "window_size": 8,
            "crystal_w_chunks": 5,
            "woodbury_latch_span": 7,
            "seed": 2026,
            "steps": 2400,
            "device": "NVIDIA A100-SXM4-40GB",
            "cache_kb_L2048": 82.73,
            "train_seconds": 1195.55,
            "mean_openweb_bpb": 3.6583,
            "max_mid_utf8_split_pct": 0.0,
            "en_bytes_per_chunk": 4.286,
            "s5_sum": 7.0,
            "ov_ood_long_pct": 32.0,
            "selective_f1_ood_long": 33.43,
            "stop_crane_triggered": True,
            "root_cause": (
                "Still mixed web/cog inside same batch AND artificial byte->utf8 router switch + d_r=(3,4) destabilized Overwrite"
            ),
        },
        "k2b_w12_pure_batch_tgt46": {
            "tag": "W12_pure_batch_K2b",
            "window_size": 12,
            "crystal_w_chunks": 5,
            "woodbury_latch_span": 7,
            "seed": 2026,
            "steps": 2600,
            "device": "NVIDIA A100-SXM4-40GB",
            "cache_kb_L2048": 90.23,
            "train_seconds": 1076.24,
            "mean_openweb_bpb": 3.6339,
            "max_mid_utf8_split_pct": 0.0,
            "en_bytes_per_chunk": 3.699,
            "arc_json_bytes_per_chunk": 9.716,
            "s5_sum": 3.0,
            "ov_ood_long_pct": 83.0,
            "trap_ood_long_pct": 92.0,
            "selective_f1_ood_long": 87.27,
            "stop_crane_triggered": True,
            "root_cause": (
                "Cold-start target_chunk_bytes=4.6 on 72% ASCII rows dropped 1_GSM8K_EN to 3.699 < 3.80 B/chunk (Phase B Attempt #2 dead zone), mixed CJK/Emoji/ARC_JSON before S_5 grokked, and delayed k=3,4 to frac=0.42 right before crystal schedule freeze at frac=0.438"
            ),
        },
        "k3_w12_shortcut_proof": {
            "tag": "W12_shortcut_proof_K3",
            "window_size": 12,
            "crystal_w_chunks": 5,
            "woodbury_latch_span": 7,
            "seed": 2026,
            "steps": 2400,
            "utf8_steps": 320,
            "device": "NVIDIA L4",
            "cache_kb_L2048": 90.23,
            "train_seconds": 1113.4,
            "infer_b8_l768_ms": 147.98,
            "mean_openweb_bpb": 3.5191,
            "max_mid_utf8_split_pct": 0.0,
            "s5_sum": 12.0,
            "ov_ood_long_pct": 94.0,
            "selective_f1_ood_long": 93.5,
            "stop_crane_triggered": True,
            "root_cause": (
                "Even on exact Phase B+E two-stage stream, window_size=12 (132 B over 3 local layers) shortcuts early k=0,1 swaps around stream_gate[0]=-1.8 while Phase B/E W=8 (0.0 weight diff at init) reaches S_5=139%-150%; proves Via Negativa W:12->8 eliminates local-cortex crystal starvation"
            ),
        },
        "k4_w8_unshielded_amp": {
            "tag": "V0_phase_be_rings_K4",
            "window_size": 8,
            "crystal_w_chunks": 5,
            "woodbury_latch_span": 7,
            "seed": 2026,
            "steps": 2400,
            "utf8_steps": 320,
            "device": "NVIDIA L4",
            "cache_kb_L2048": 82.73,
            "train_seconds": 1132.5,
            "infer_b8_l768_ms": 159.79,
            "s1_byte_s5_k2_pct": 1.0,
            "mean_openweb_bpb": 3.3643,
            "max_mid_utf8_split_pct": 0.0,
            "en_bytes_per_chunk": 3.963,
            "s5_sum": 7.0,
            "ov_ood_long_pct": 96.0,
            "trap_ood_long_pct": 93.0,
            "selective_f1_ood_long": 94.48,
            "stop_crane_triggered": True,
            "root_cause": (
                "_CheckpointedCrystalScanFn (Phase E crystal.py:442) ran forward in AMP BFloat16 and recomputed backward outside autocast in Float32, injecting 1.6%-14.55% gradient mismatch across crystal projections; combined with weight_decay drag on perm_gate_proj.bias and early snap_alpha onset at frac=0.50, gate_bimodal snapped p_eta to 0 (identity P=I)"
            ),
        },
        "k5_amp_shield_delay_alpha": {
            "tag": "V0_phase_be_rings_K5",
            "window_size": 8,
            "crystal_w_chunks": 5,
            "woodbury_latch_span": 7,
            "seed": 2026,
            "steps": 2800,
            "utf8_steps": 320,
            "device": "NVIDIA A100-SXM4-40GB",
            "cache_kb_L2048": 82.73,
            "s1_byte_s5_k2_pct": 22.0,
            "s5_k2_32_progression": [3.1, 0.0, 3.1, 28.1],
            "stop_crane_triggered": True,
            "root_cause": (
                "Full forward_states FP32 shield raised S_5 k=2 22x over K4 (1.0%->22.0%, 28.1% on N=32 at step 2800), but gating only is_grad_enabled() mismatched write (FP32) vs read (BF16) and train vs eval, delayed snap_alpha (0.60->0.80) left only 485 low-LR STE steps, wd=0.0 + min_lr=2.5e-4 altered trajectory vs Phase B, and Gate-2 reset global torch seed after model_seed"
            ),
        },
        "k6_scan_shield_cold_saddle": {
            "tag": "V0_phase_be_rings_K6",
            "window_size": 8,
            "crystal_w_chunks": 5,
            "woodbury_latch_span": 7,
            "seed": 2026,
            "steps": 2400,
            "utf8_steps": 320,
            "device": "NVIDIA L4",
            "cache_kb_L2048": 82.73,
            "train_seconds": 1100.9,
            "infer_b8_l768_ms": 139.49,
            "s1_byte_s5_k2_pct": 78.0,
            "s5_k2_32_progression": [50.0, 71.9, 81.2],
            "mean_openweb_bpb": 3.5055,
            "max_mid_utf8_split_pct": 0.0,
            "s5_sum": 127.0,
            "ov_ood_long_pct": 15.0,
            "trap_ood_long_pct": 50.0,
            "selective_f1_ood_long": 23.08,
            "stop_crane_triggered": True,
            "root_cause": (
                "_amp_shielded_crystal_scan completely cured S_5 (78% S1 k=2, 127% post-UTF8 sum, 0.00% mid-UTF8 cuts), but under simultaneous cold-start FP32 scans Crystal and Woodbury exhibit saddle bistability (K4 Basin W: OV=96%/S5=7% vs K6 Basin C: S5=127%/OV=15%), while Stage-2 use_static_m_cap=True + BS=32 dragged mean_openweb_bpb (3.3643->3.5055)"
            ),
        },
        "k7_s2026_pass_s2027_tail_choke": {
            "tag": "V0_phase_be_rings_K7",
            "window_size": 8,
            "crystal_w_chunks": 5,
            "woodbury_latch_span": 7,
            "seed_2026": {
                "device": "NVIDIA A100-SXM4-40GB",
                "train_seconds": 1136.86,
                "s1a_byte_s5_k2_pct": 97.0,
                "s1_byte_s5_k2_pct": 96.0,
                "s1_byte_ov_long_pct": 92.0,
                "s1_byte_trap_long_pct": 91.0,
                "s5_sum": 176.0,
                "ov_ood_long_pct": 91.0,
                "trap_ood_long_pct": 92.0,
                "selective_f1_ood_long": 91.5,
                "mean_openweb_bpb": 3.465,
                "max_mid_utf8_split_pct": 0.0,
            },
            "seed_2027_stop_crane": {
                "s1a_byte_s5_k2_pct": 21.0,
                "s5_k2_32_progression": [0.0, 0.0, 12.5],
                "init_woodbury_to_crystal_grad_ratio": 1.61,
                "b_per_chunk_800": 5.04,
            },
            "root_cause": (
                "K7 orthogonal consolidation hit S_5=176.0% & OV_long=91.0% on seed=2026, while on seed=2027 (model_seed=101, 1.61x stronger init Woodbury grad, b/ch=5.04) woodbury 0.25x LR delayed S_5 takeoff past step 1600 where cosine decay to min_lr=0.0 choked Stage 1A on the upward ramp (0%->12.5%->21.0%)"
            ),
        },
        "k8_s2026_record_s2027_router_cliff": {
            "tag": "V0_phase_be_rings_K8",
            "window_size": 8,
            "crystal_w_chunks": 5,
            "woodbury_latch_span": 7,
            "seed_2026": {
                "device": "NVIDIA A100-SXM4-40GB",
                "train_seconds": 1439.06,
                "s1a_byte_s5_k2_pct": 98.0,
                "s1_byte_s5_k2_pct": 96.0,
                "s1_byte_ov_long_pct": 94.0,
                "s1_byte_trap_long_pct": 100.0,
                "s5_sum": 194.0,
                "ov_ood_long_pct": 96.0,
                "trap_ood_long_pct": 99.0,
                "selective_f1_ood_long": 97.48,
                "mean_openweb_bpb": 3.3628,
                "en_bytes_per_chunk": 3.804,
                "max_mid_utf8_split_pct": 0.0,
            },
            "seed_2027_stop_crane": {
                "device": "NVIDIA A100-SXM4-40GB",
                "train_seconds": 1395.42,
                "s1a_byte_s5_k2_pct": 4.0,
                "s1_byte_s5_k2_pct": 1.0,
                "s1_byte_ov_long_pct": 94.0,
                "s1_byte_trap_long_pct": 97.0,
                "s5_sum": 5.0,
                "ov_ood_long_pct": 94.0,
                "trap_ood_long_pct": 99.0,
                "selective_f1_ood_long": 96.44,
                "mean_openweb_bpb": 3.4184,
                "en_bytes_per_chunk": 3.797,
                "max_mid_utf8_split_pct": 0.0,
            },
            "root_cause": (
                "K8 set clean-stream record on seed=2026 (S_5=194.0%, OV=96.0%, Trap=99.0%, ow_bpb=3.3628), while on seed=2027 target_chunk_bytes=4.85 pushed unseeded std=0.04 RMSNorm (25x backward grad amplification) router over the <3.80 B/ch cliff (EN=3.797 B/ch, S_5=5.0%) and shifted S2_25M/S3_85M widths via post-init RNG offset"
            ),
        },
        "k9_std1_router_freeze": {
            "tag": "V0_phase_be_rings_K9",
            "window_size": 8,
            "crystal_w_chunks": 5,
            "woodbury_latch_span": 7,
            "seed": 2026,
            "steps": 2800,
            "device": "NVIDIA A100-SXM4-40GB",
            "train_seconds": 922.4,
            "b_per_chunk_2800": 5.02,
            "c2_b0_2800": -1.29,
            "s1a_byte_s5_k2_pct": 2.0,
            "stop_crane_triggered": True,
            "root_cause": (
                "_canonicalize_router_init replaced native std=0.04 RMSNorm router_emb init with nn.Embedding.reset_parameters() std=1.0, shrinking relative AdamW step size 25x during the 378 router-learning steps and freezing random Gaussian noise as word chunker"
            ),
        },
        "k10_fused_colon_and_latch3_proof": {
            "device": "NVIDIA A100-SXM4-40GB (3x parallel)",
            "v0_seed2026_stage1b_abort": {
                "window_size": 8,
                "crystal_w_chunks": 5,
                "woodbury_latch_span": 7,
                "s1a_byte_s5_k2_pct": 95.0,
                "s1_byte_s5_k2_pct": 96.0,
                "s1_byte_ov_long_pct": 50.0,
                "s1_byte_trap_long_pct": 11.0,
            },
            "v0_seed2027_stage1a_abort": {
                "window_size": 8,
                "crystal_w_chunks": 5,
                "woodbury_latch_span": 7,
                "s1a_byte_s5_k2_pct": 8.0,
            },
            "v1_attempt1_latch3": {
                "window_size": 8,
                "crystal_w_chunks": 2,
                "woodbury_latch_span": 3,
                "cache_kb_L2048": 75.45,
                "train_seconds": 1313.11,
                "infer_b8_l768_ms": 134.08,
                "s1a_byte_s5_k2_pct": 90.0,
                "s1_byte_s5_k2_pct": 92.0,
                "s1_byte_ov_long_pct": 0.0,
                "s1_byte_trap_long_pct": 100.0,
                "s5_sum": 153.0,
                "ov_ood_long_pct": 0.0,
                "trap_ood_long_pct": 100.0,
                "selective_f1_ood_long": 0.0,
                "mean_openweb_bpb": 3.4373,
                "max_mid_utf8_split_pct": 0.0,
                "drift": [1.19e-7, 2.38e-7],
            },
            "root_cause": (
                "(1) V1 L_latch=3 hits the physical 17 B / 3.5-4.5 chunk causality floor of ' is stationed in ' / ' is currently in' while W_chunks=2 passes (153% S_5); (2) unsplit-colon (colon=0) 1-byte shift cuts lexical c_win8 cosine (+0.091) while read_lag_logits peaks 68.3% at t-1, starving S_5 on non-colon router basins (mseed=101); (3) fused=True AdamW + target=4.90 shifted BF16 Stage-1B Woodbury saddle (50% vs K8 94%)"
            ),
        },
        "k11_v1_latch6_pass_and_wchunks5_distractor_proof": {
            "device": "NVIDIA L4 (VM0) + NVIDIA A100-SXM4-40GB (VM1, VM2)",
            "v1_attempt1_latch6": {
                "window_size": 8,
                "crystal_w_chunks": 2,
                "woodbury_latch_span": 6,
                "params": 1483789,
                "cache_kb_L2048": 76.98,
                "train_seconds": 1371.28,
                "infer_b8_l768_ms": 139.41,
                "s1a_byte_s5_k2_pct": 97.0,
                "s1_byte_s5_k2_pct": 98.0,
                "s1_byte_ov_long_pct": 60.0,
                "s1_byte_trap_long_pct": 89.0,
                "s5_sum": 181.0,
                "ov_ood_long_pct": 86.0,
                "trap_ood_long_pct": 93.0,
                "selective_f1_in": 94.74,
                "selective_f1_ood_long": 89.36,
                "mean_openweb_bpb": 3.3788,
                "max_mid_utf8_split_pct": 0.0,
                "drift": [1.19e-7, 2.38e-7],
            },
            "v0_seed2026_stage1b_abort": {
                "s1a_byte_s5_k2_pct": 94.0,
                "s1_byte_s5_k2_pct": 95.0,
                "s1_byte_ov_long_pct": 42.0,
                "s1_byte_trap_long_pct": 47.0,
            },
            "v0_seed2027_stage1a_abort": {
                "s1a_byte_s5_k2_pct": 2.0,
            },
            "root_cause": (
                "(1) V1 (8,2,6) clears both S_5 (181%) and Overwrite/Trap (86%/93%, SelF1=89.36%, ow_bpb=3.3788, -24.7% tier cache); (2) global _ShiftTolerantLexNorm blurred h_lex shared by Woodbury, hurting Officer surname discrimination (+0.311->+0.469) in Stage 1B; (3) W_chunks=5 allocates 66.1% pointer mass to pre-swap distractors (only 7.6% on Alpha vs 41.3%/83.0% under W_chunks=2)"
            ),
        },
        "k12_splitlex_duallag_l4_postmortem": {
            "device": "NVIDIA L4",
            "v0_seed2026": {
                "window_size": 8,
                "crystal_w_chunks": 5,
                "woodbury_latch_span": 7,
                "train_seconds": 1372.78,
                "infer_b8_l768_ms": 134.81,
                "s1a_byte_s5_k2_pct": 65.0,
                "s1_byte_s5_k2_pct": 80.0,
                "s1_byte_ov_long_pct": 44.0,
                "s1_byte_trap_long_pct": 43.0,
                "s5_sum": 117.0,
                "ov_ood_long_pct": 42.0,
                "trap_ood_long_pct": 54.0,
                "selective_f1_ood_long": 47.25,
                "mean_openweb_bpb": 3.5534,
                "max_mid_utf8_split_pct": 0.0,
            },
            "root_cause": (
                "Split lexical streams shared bc.lex_norm and bc.lex_up across blurred Crystal c_sm (Stage 1A) and unblurred Woodbury c_win8 (Stage 1B), while dual-lag read_lag_logits halved Crystal's t-1 read heads (degrading V0_seed2026 vs K8 from S1A 98%->65%, OV 96%->42%, ow_bpb 3.3628->3.5534)"
            ),
        },
        "k13_l4_s1b500_underfueled": {
            "device": "3x NVIDIA L4",
            "v0_seed2026_stage1b_abort": {
                "window_size": 8,
                "crystal_w_chunks": 5,
                "woodbury_latch_span": 7,
                "s1a_byte_s5_k2_pct": 99.0,
                "s1_byte_s5_k2_pct": 99.0,
                "s1_byte_ov_long_pct": 5.0,
                "s1_byte_trap_long_pct": 72.0,
            },
            "v1_attempt1_stage1b_abort": {
                "window_size": 8,
                "crystal_w_chunks": 2,
                "woodbury_latch_span": 6,
                "s1a_byte_s5_k2_pct": 96.0,
                "s1_byte_s5_k2_pct": 97.0,
                "s1_byte_ov_long_pct": 17.0,
                "s1_byte_trap_long_pct": 12.0,
            },
            "v0_seed2027": {
                "window_size": 8,
                "crystal_w_chunks": 5,
                "woodbury_latch_span": 7,
                "train_seconds": 1333.15,
                "infer_b8_l768_ms": 135.84,
                "s1a_byte_s5_k2_pct": 5.0,
                "s1_byte_s5_k2_pct": 10.0,
                "s1_byte_ov_long_pct": 17.0,
                "s1_byte_trap_long_pct": 41.0,
                "s5_sum": 14.0,
                "ov_in_dist_pct": 44.0,
                "ov_ood_med_pct": 38.0,
                "ov_ood_long_pct": 14.0,
                "trap_ood_long_pct": 62.0,
                "selective_f1_ood_long": 22.84,
                "mean_openweb_bpb": 3.4261,
                "max_mid_utf8_split_pct": 0.0,
            },
            "root_cause": (
                "Pure K8 core hit S1A=99%/96% on L4, proving Crystal health; Stage-1B OV failed (5%/17%/17%) because Stage 1A starved woodbury at 0.05x LR with wd=0.01 for 2800 steps while Stage 1B supplied only 500 x 35% = 175 effective OV batches with pre throttled at 1.5e-4 and cosine decay to 0.10x"
            ),
        },
        "k14_s1b900_v0_pass_v1_020_hijack": {
            "device": "3x NVIDIA L4",
            "v0_seed2026": {
                "window_size": 8,
                "crystal_w_chunks": 5,
                "woodbury_latch_span": 7,
                "train_seconds": 1528.76,
                "infer_b8_l768_ms": 136.85,
                "s1a_byte_s5_k2_pct": 98.0,
                "s1_byte_s5_k2_pct": 95.0,
                "s1_byte_ov_long_pct": 96.0,
                "s1_byte_trap_long_pct": 91.0,
                "s5_sum": 167.0,
                "ov_ood_long_pct": 95.0,
                "trap_ood_long_pct": 92.0,
                "selective_f1_ood_long": 93.48,
                "mean_openweb_bpb": 3.5257,
                "max_mid_utf8_split_pct": 0.0,
            },
            "v0_seed2027": {
                "window_size": 8,
                "crystal_w_chunks": 5,
                "woodbury_latch_span": 7,
                "train_seconds": 1600.32,
                "infer_b8_l768_ms": 142.77,
                "s1a_byte_s5_k2_pct": 0.0,
                "s1_byte_s5_k2_pct": 1.0,
                "s1_byte_ov_long_pct": 99.0,
                "s1_byte_trap_long_pct": 99.0,
                "s5_sum": 3.0,
                "ov_ood_long_pct": 98.0,
                "trap_ood_long_pct": 99.0,
                "selective_f1_ood_long": 98.5,
                "mean_openweb_bpb": 3.604,
                "max_mid_utf8_split_pct": 0.0,
            },
            "v1_attempt1_stage1a_abort": {
                "window_size": 8,
                "crystal_w_chunks": 2,
                "woodbury_latch_span": 6,
                "s1a_byte_s5_k2_pct": 8.0,
            },
            "root_cause": (
                "Axis orthogonality proven on 3x L4: (1) full-fuel Stage 1B (900 steps, 52% OV / 18% Trap, wood=2.4e-3, pre=3e-4, post=1.2e-3, floor=0.15x) cured OV on both V0 seeds (95%/98%, SelF1 93.48%/98.50%); (2) raising Stage 1A woodbury from 0.05x wd=0.01 (K13 V1 S1A=96%) to 0.20x wd=0.0 caused Woodbury to hijack V1's pruned W_chunks=2 crystal (S1A=8.0%)"
            ),
        },
        "k15_l4_bf16_nondet_and_woodbury_wd_proof": {
            "device": "3x NVIDIA L4",
            "v1_attempt1_stage1a_abort": {
                "window_size": 8,
                "crystal_w_chunks": 2,
                "woodbury_latch_span": 6,
                "s1a_byte_s5_k2_pct": 32.0,
            },
            "v0_seed2026_stage1b_abort": {
                "window_size": 8,
                "crystal_w_chunks": 5,
                "woodbury_latch_span": 7,
                "s1a_byte_s5_k2_pct": 89.0,
                "s1_byte_s5_k2_pct": 89.0,
                "s1_byte_ov_long_pct": 39.0,
                "s1_byte_trap_long_pct": 29.0,
            },
            "v0_seed2027": {
                "window_size": 8,
                "crystal_w_chunks": 5,
                "woodbury_latch_span": 7,
                "train_seconds": 1439.75,
                "infer_b8_l768_ms": 138.83,
                "s1a_byte_s5_k2_pct": 6.0,
                "s1_byte_s5_k2_pct": 6.0,
                "s1_byte_ov_long_pct": 93.0,
                "s1_byte_trap_long_pct": 98.0,
                "s5_sum": 8.0,
                "ov_ood_long_pct": 93.0,
                "trap_ood_long_pct": 100.0,
                "selective_f1_in": 94.18,
                "selective_f1_ood_long": 96.37,
                "mean_openweb_bpb": 3.5591,
                "max_mid_utf8_split_pct": 0.0,
            },
            "root_cause": (
                "(1) Stage-1A woodbury weight_decay=0.01 (vs K14 wd=0.0 yielding 96%/99% OV) atrophied Woodbury projections on V0_s2026 during 2800 steps (S1B OV=39%); (2) CUDA BF16 variance on tiny d=160 causes stochastic S1A outcomes on L4 while A100 runs K8 (V0) and K11 (V1 8,2,6) already established both anchors cleanly, motivating locking V1 (8,2,6) and moving directly to S1_5M/S2_25M/S3_85M scaling with Stage-1A woodbury lr=0.05x, wd=0.0 + K14 Stage-1B"
            ),
        },
    }

    payload = {
        "protocol": proto_info,
        "prior_diagnostics": prior_diagnostics,
        "v0_seed2026": v0_s1,
        "v0_seed2027": v0_s2,
        "v0_mean": v0_mean,
        "tolerances": tolerances,
        "v1_attempt1": v1,
        "v1b_attempt2": v1b,
        "selected_lean_params": selected,
    }
    out_p = ROOT_DIR / "benchmarks" / "phase_k_via_negativa_probe.json"
    _write_json_atomic(out_p, payload)
    print(f"\n[DONE] Saved {out_p} | selected={selected}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
