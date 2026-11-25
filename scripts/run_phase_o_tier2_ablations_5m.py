"""
Phase O: Tier 2 Controlled Single-Axis `S1_5M` Ablation Tournament, WSD Plateau Branching Proof,
`OEBEarlyStopper` Gate Verification, and Blackwell `sm_120` FP8 (`torch._scaled_mm`) Micro-Probe
(`scripts/run_phase_o_tier2_ablations_5m.py`).

Strict `epistemic-ml-trading.md` / `calcite-phase-runner` discipline:
- Single `seed=2026` and 100% identical `(OpenWeb 75% + Cognitive S5/OV/Trap 15% + MQAR 10%)`
  data stream across EVERY arm (`B=64, L=512, mbs=32, acc=2`).
- Gate-2 init-std & live gradient verification before training every arm.
- `C0_control` runs `WSDLRSchedule` (`40` warmup + `410` stable -> plateau checkpoint at `step=450`
  + `250` cosine decay), measures `Tier-0` vs `Tier-1` eval wall-time, `2*sigma` calibration,
  and deterministic `CPU AVX-512` & `GPU CUDA-Graph` 2-bit LUT streaming `B/s` + `O(1)` RAM.
- Single-axis arms: `A1_no_dec_dw3`, `A2_dw_bottleneck_05`, `B1_polar4`, `B2_polar3`,
  `C1_tf32_bf16res`, plus `WSD_branch_250` (250-step decay branched from `C0` plateau with `polar4`)
  and `OEB_degraded_synth` (live `OEBEarlyStopper` kill proof at `step=250`).
- Blackwell `sm_120` FP8 (`torch._scaled_mm`) vs BF16 GEMM micro-probe on `S2_25M` & `S3_85M` shapes.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from calcite_v2 import (  # noqa: E402
    CalciteStreamEngine,
    CalciteV2Config,
    CalciteV2LM,
    build_calcite_v2_tier_config,
    compute_v2_state_cache_kb,
)
from experiments.scaling_phase_k.eval_suite import (  # noqa: E402
    BatchPrefetcher,
    eval_val_memmap_bpb,
    has_usable_cuda,
    load_utf8_corpus_array,
    sample_memmap_batch,
    sync_if_cuda,
)
from kernels.fast_calcite_scans import (  # noqa: E402
    compile_calcite_v2,
    unwrap_compiled_state_dict,
)
from kernels.fast_packed_lut import (  # noqa: E402
    fuse_packed_swiglu_for_inference,
    pack_fast_model_2bit_lut,
)
from scripts.colab_phase_l_v2_runner import (  # noqa: E402
    build_eval_bundles,
    ensure_phase_k_corpus,
    load_frozen_router_into,
    sample_cognitive_batch,
    set_all_seeds,
)
from scripts.experiment_velocity import (  # noqa: E402
    OEBEarlyStopper,
    WSDLRSchedule,
    load_plateau_checkpoint,
    run_tiered_eval,
    save_plateau_checkpoint,
)
from scripts.muon_opt import MuonAdamWHybrid  # noqa: E402
from scripts.run_phase_n_v31_lm_check_5m import sample_multiformat_mqar_row  # noqa: E402
from simvq_split.memory import quantize_ternary_weights_ls  # noqa: E402
from simvq_split.tokenizer import ByteBPETokenizer  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Config & Deterministic Mixed Data Stream (Identical Across All Arms)
# ---------------------------------------------------------------------------


def build_v31_5m_arm_config(
    *,
    dec_dw3_enabled: bool = True,
    dw_bottleneck_ratio: float = 1.0,
    bf16_residual_stream: bool = False,
    smoke: bool = False,
) -> CalciteV2Config:
    cfg = build_calcite_v2_tier_config(
        "S1_5M",
        topology="hourglass",
        use_scaled_decoder=True,
        use_kalman_precond=False,
        use_latent_feedback=False,
        use_clean_gates=False,
        use_sym_byte_swa=True,
        byte_swa_window=24,
        byte_swa_dim=64,
        use_byte_engram=True,
        engram_slots=4096,
        engram_dim=32,
        use_byte_induct_cache=True,
        byte_induct_slots=128 if smoke else 512,
        byte_induct_dim=16,
        dec_dw3_enabled=dec_dw3_enabled,
        dw_bottleneck_ratio=dw_bottleneck_ratio,
        bf16_residual_stream=bf16_residual_stream,
    )
    if smoke:
        cfg.d_model, cfg.n_heads, cfg.d_ff = 128, 2, 256
        cfg.byte_char_dim, cfg.byte_d_ff = 16, 128
    return cfg


def sample_tournament_batch(
    train_mm: np.ndarray,
    tok: ByteBPETokenizer,
    tr_sents: List[str],
    batch_size: int,
    seq_len: int,
    rng_py: random.Random,
    rng_np: np.random.Generator,
    n_cog: int,
    n_copy: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Deterministic `OpenWeb (75%) + Cognitive S5/OV/Trap (15%) + MQAR Copy (10%)` CPU batch.
    Targets use `-100` for masked prompt/pad tokens across all row types, interleaved across
    microbatches so every microbatch sees identical task balance.
    """
    n_cog = max(0, min(batch_size - 2, int(n_cog)))
    n_copy = max(0, min(batch_size - n_cog - 1, int(n_copy)))
    n_prose = batch_size - n_cog - n_copy

    xp, yp = sample_memmap_batch(train_mm, n_prose, seq_len, rng_np, "cpu")
    yp = torch.where(yp == 0, torch.full_like(yp, -100), yp)
    mp = torch.zeros_like(yp, dtype=torch.bool)

    parts_x = [xp]
    parts_y = [yp]
    parts_m = [mp]

    if n_cog > 0:
        xg, yg = sample_cognitive_batch(tok, tr_sents, n_cog, seq_len, rng_py, "cpu")
        mg = torch.zeros_like(yg, dtype=torch.bool)
        parts_x.append(xg)
        parts_y.append(yg)
        parts_m.append(mg)

    if n_copy > 0:
        xs, ys, ms = [], [], []
        for _ in range(n_copy):
            xr, yr, mr = sample_multiformat_mqar_row(train_mm, seq_len, rng_py, rng_np)
            xs.append(xr)
            ys.append(yr)
            ms.append(mr)
        xc = torch.from_numpy(np.asarray(xs, dtype=np.int64))
        yc = torch.from_numpy(np.asarray(ys, dtype=np.int64))
        yc = torch.where(yc == 0, torch.full_like(yc, -100), yc)
        mc = torch.from_numpy(np.asarray(ms, dtype=np.bool_))
        parts_x.append(xc)
        parts_y.append(yc)
        parts_m.append(mc)

    x_all = torch.cat(parts_x, dim=0)
    y_all = torch.cat(parts_y, dim=0)
    m_all = torch.cat(parts_m, dim=0)

    # Deterministic round-robin interleaving across halves (microbatches)
    n_half = max(1, batch_size // 2)
    order: List[int] = []
    # Split each group evenly across first half and second half
    g_ranges = [
        (0, n_prose),
        (n_prose, n_prose + n_cog),
        (n_prose + n_cog, batch_size),
    ]
    half0: List[int] = []
    half1: List[int] = []
    for s, e in g_ranges:
        mid = s + (e - s) // 2
        half0.extend(range(s, mid))
        half1.extend(range(mid, e))
    order = (half0 + half1)[:batch_size]
    if len(order) != batch_size:
        order = list(range(batch_size))
    _ = n_half
    perm = torch.tensor(order, dtype=torch.long)
    return x_all[perm], y_all[perm], m_all[perm]


# ---------------------------------------------------------------------------
# 2. Gate-2 Init-Std & Live Gradient Check
# ---------------------------------------------------------------------------


def run_gate2_init_probe(
    model: CalciteV2LM,
    xb: torch.Tensor,
    yb: torch.Tensor,
    device: str,
    use_amp: bool,
) -> Dict[str, Any]:
    """Verifies bounded per-block output std and 100% finite live trainable gradients at init."""
    model.train()
    model.zero_grad(set_to_none=True)
    stds: Dict[str, float] = {}
    dtypes: Dict[str, str] = {}
    hooks: List[Any] = []

    def _mk_hook(name: str):
        def _fn(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor):
                stds[name] = round(float(t.detach().float().std().item()), 5)
                dtypes[name] = str(t.dtype)

        return _fn

    for nm, mod in [
        ("cortex_pre_last", model.cortex_pre[-1] if model.cortex_pre else None),
        ("crystal", model.crystal),
        ("woodbury", model.woodbury),
        ("mid_mlp", model.mid_mlp),
        ("crystal_2", model.crystal_2),
        ("cortex_post_last", model.cortex_post[-1] if model.cortex_post else None),
        ("norm_f", model.norm_f),
        ("lm_head", model.lm_head),
    ]:
        if mod is not None:
            hooks.append(mod.register_forward_hook(_mk_hook(nm)))

    try:
        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
            lg = model(xb)
            ce = F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), yb.reshape(-1), ignore_index=-100)
            loss = ce + 0.04 * model.get_commitment_loss()
        loss.backward()
    finally:
        for h in hooks:
            h.remove()

    n_req = 0
    n_finite_nz = 0
    n_nonfinite = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        n_req += 1
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            n_nonfinite += 1
        elif float(p.grad.abs().max().item()) > 0.0:
            n_finite_nz += 1

    model.zero_grad(set_to_none=True)
    return {
        "init_loss": round(float(loss.detach().item()), 4),
        "block_std": stds,
        "block_dtype": dtypes,
        "trainable_params_tensors": n_req,
        "live_finite_grad_tensors": n_finite_nz,
        "nonfinite_grad_tensors": n_nonfinite,
        "all_grads_finite": n_nonfinite == 0,
    }


# ---------------------------------------------------------------------------
# 3. Streaming Benchmark (Deterministic 256B Greedy CPU AVX-512 & GPU CUDA Graph)
# ---------------------------------------------------------------------------


def benchmark_arm_lut_streaming(
    model_fp: CalciteV2LM,
    cfg: CalciteV2Config,
    val_mm: np.ndarray,
    device: str,
    smoke: bool = False,
) -> Dict[str, Any]:
    """
    Packs arm weights to 2-bit LUT (`FastPackedLUT2BitLinear` + `FusedPackedSwiGLU`) and measures
    fixed-stream decode `B/s` (identical validation bytes -> identical frozen-router backbone ticks
    across all arms!), `ms/B`, `stream_vs_batch_max_err`, and `O(1)` RAM on CPU (`AVX512F`) and
    GPU (`cuda_ext + use_cuda_graph=True`).
    """
    sd_clean = unwrap_compiled_state_dict(model_fp)
    clean = CalciteV2LM(cfg).cpu()
    clean.load_state_dict(sd_clean, strict=True)
    clean.byte_chunker.lock_router(True)
    clean.set_crystal_schedule(1, 1)
    clean.set_snap_alpha(1.0)
    with torch.no_grad():
        for nm, mod in clean.named_modules():
            if isinstance(mod, nn.Linear) and nm != "lm_head":
                if clean.byte_chunker.is_locked_router_module(mod):
                    continue
                wt, _, _ = quantize_ternary_weights_ls(mod.weight.data)
                mod.weight.data.copy_(wt)
    clean.eval()

    lut_cpu = pack_fast_model_2bit_lut(clean, backend="auto").eval()
    lut_cpu_f = fuse_packed_swiglu_for_inference(lut_cpu, inplace=False).eval()

    prompt = "Calcite v3.1 Tier-2 ablation streaming check: Привет, мир! 東京 🚀 2026 "
    p_cpu = torch.tensor([[b + 3 for b in prompt.encode("utf-8")]], device="cpu", dtype=torch.long)
    n_gen = 32 if smoke else 256
    fixed_stream_cpu = torch.from_numpy(val_mm[:n_gen].astype(np.int64)).unsqueeze(0)

    def _run_det(mod: CalciteV2LM, p_ids: torch.Tensor, stream_ids: torch.Tensor, use_cg: bool) -> Dict[str, Any]:
        dev = p_ids.device
        eng = CalciteStreamEngine(mod, use_cuda_graph=use_cg)
        _, _ = eng.generate(p_ids, max_new_bytes=32, temperature=0.0, eos_token_id=None)
        if dev.type == "cuda":
            torch.cuda.synchronize()

        t_p0 = time.perf_counter()
        lg, st = eng.prefill(p_ids, return_all_logits=False)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        pref_ms = (time.perf_counter() - t_p0) * 1000.0
        ram_before = st.memory_bytes()
        ticks0 = int(st.chunk_pos)

        t0 = time.perf_counter()
        for k in range(n_gen):
            tok_cur = stream_ids[:, k]
            lg, st = eng.step(tok_cur, st)
            _ = lg.argmax(dim=-1)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        ram_after = st.memory_bytes()
        ticks = int(st.chunk_pos) - ticks0

        # Exact parity check vs batch forward on prompt + 32 generated bytes
        gen_seq, _ = eng.generate(p_ids, max_new_bytes=32, temperature=0.0, eos_token_id=None)
        eng_chk = CalciteStreamEngine(mod, use_cuda_graph=False)
        lg_s, _ = eng_chk.prefill(gen_seq, return_all_logits=True)
        with torch.inference_mode():
            lg_b = mod(gen_seq)
        err_batch = float((lg_b.float() - lg_s.float()).abs().max().item())
        err_cg = 0.0
        if use_cg:
            lg_e0, st_e0 = eng_chk.prefill(p_ids, return_all_logits=False)
            lg_c0, st_c0 = eng.prefill(p_ids, return_all_logits=False)
            err_cg = float((lg_e0.float() - lg_c0.float()).abs().max().item())
            for k in range(32):
                tk = gen_seq[:, p_ids.size(1) + k]
                lg_e0, st_e0 = eng_chk.step(tk, st_e0)
                lg_c0, st_c0 = eng.step(tk, st_c0)
                err_cg = max(err_cg, float((lg_e0.float() - lg_c0.float()).abs().max().item()))

        return {
            "decode_bps": round(float(n_gen) / max(1e-9, dt), 1),
            "decode_ms_per_byte": round(1000.0 * dt / max(1, n_gen), 3),
            "prefill_ms": round(pref_ms, 2),
            "prefill_bps": round(float(p_ids.size(1)) / max(1e-9, pref_ms / 1000.0), 1),
            "backbone_ticks": ticks,
            "state_kb": round(ram_after / 1024.0, 2),
            "strict_o1_ram": ram_before == ram_after,
            "vs_batch_max_err": round(err_batch, 7),
            "vs_eager_max_err": round(err_cg, 7) if use_cg else 0.0,
        }

    out: Dict[str, Any] = {}
    out["cpu_avx512_fused_lut"] = _run_det(lut_cpu_f, p_cpu, fixed_stream_cpu, use_cg=False)
    if device == "cuda" and torch.cuda.is_available():
        lut_gpu_f = lut_cpu_f.to("cuda").eval()
        p_gpu = p_cpu.to("cuda")
        s_gpu = fixed_stream_cpu.to("cuda")
        out["gpu_cuda_graph_fused_lut"] = _run_det(lut_gpu_f, p_gpu, s_gpu, use_cg=True)
        del lut_gpu_f, p_gpu, s_gpu
    del clean, lut_cpu, lut_cpu_f
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# 4. Core Arm Trainer (Supports WSD Plateau Save, Branch Resume, & OEBEarlyStopper)
# ---------------------------------------------------------------------------


def _measure_muon_adamw_split(
    model: nn.Module,
    opt: MuonAdamWHybrid,
    xb_f: torch.Tensor,
    yb_f: torch.Tensor,
    mb_f: torch.Tensor,
    micro_bs: int,
    device: str,
    use_amp: bool,
    n_iters: int = 8,
) -> Dict[str, float]:
    """Measures steady-state `muon_ms` and `adamw_ms` on dummy copies of current grads (no weight drift)."""
    orig_params = [p.detach().clone() for p in opt.trainable_params]
    orig_opt_sd = {
        "adamw": {k: v for k, v in opt.adamw.state_dict().items()} if opt.adamw is not None else None,
    }
    # Snapshot full state_dict cleanly on CPU so we leave model/opt untouched after probe
    opt_sd_snap = opt.state_dict()
    adamw_ms_acc = 0.0
    muon_ms_acc = 0.0
    orig_aw_step = opt.adamw.step if opt.adamw is not None else None
    aw_cur = [0.0]

    def _aw_wrap(*a, **kw):
        sync_if_cuda(device)
        t0 = time.perf_counter()
        r = orig_aw_step(*a, **kw)
        sync_if_cuda(device)
        aw_cur[0] += (time.perf_counter() - t0) * 1000.0
        return r

    if opt.adamw is not None:
        opt.adamw.step = _aw_wrap  # type: ignore[method-assign]
    try:
        # Compute one real gradient snapshot
        opt.zero_grad(set_to_none=True)
        xb = xb_f[:micro_bs]
        yb = yb_f[:micro_bs]
        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
            lg = model(xb)
            loss = F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), yb.reshape(-1), ignore_index=-100)
        loss.backward()
        saved_grads = [p.grad.detach().clone() if p.grad is not None else None for p in opt.trainable_params]

        for _ in range(n_iters):
            for p, g in zip(opt.trainable_params, saved_grads):
                if g is not None:
                    p.grad = g.clone()
            aw_cur[0] = 0.0
            sync_if_cuda(device)
            t0 = time.perf_counter()
            opt.step()
            sync_if_cuda(device)
            dt = (time.perf_counter() - t0) * 1000.0
            adamw_ms_acc += aw_cur[0]
            muon_ms_acc += max(0.0, dt - aw_cur[0])
    finally:
        if opt.adamw is not None and orig_aw_step is not None:
            opt.adamw.step = orig_aw_step  # type: ignore[method-assign]
        with torch.no_grad():
            for p, p0 in zip(opt.trainable_params, orig_params):
                p.copy_(p0)
        opt.load_state_dict(opt_sd_snap)
        opt.zero_grad(set_to_none=True)
        _ = orig_opt_sd
    return {
        "muon_ns_ms": round(muon_ms_acc / max(1, n_iters), 3),
        "adamw_ms": round(adamw_ms_acc / max(1, n_iters), 3),
        "opt_total_ms": round((muon_ms_acc + adamw_ms_acc) / max(1, n_iters), 3),
    }


def run_single_arm(
    arm_name: str,
    cfg: CalciteV2Config,
    router_state: Dict[str, torch.Tensor],
    train_mm: np.ndarray,
    val_mm: np.ndarray,
    tok: ByteBPETokenizer,
    tr_sents: List[str],
    eval_wins: Dict[str, List[List[int]]],
    s5_suites: Dict[str, list],
    ov_suites: Dict[str, list],
    trap_suites: Dict[str, list],
    *,
    device: str,
    total_steps: int,
    warmup_steps: int,
    decay_steps: int,
    batch_size: int,
    micro_bs: int,
    seq_len: int,
    seed: int,
    ns_schedule: str = "quintic5",
    allow_tf32: bool = False,
    base_lr: float = 1.8e-3,
    min_lr: float = 1.2e-4,
    use_compile: bool = True,
    plateau_save_path: Optional[Path] = None,
    resume_plateau_path: Optional[Path] = None,
    reuse_compiled_model: Optional[CalciteV2LM] = None,
    record_val_steps: Tuple[int, ...] = (),
    early_stopper: Optional[OEBEarlyStopper] = None,
    run_streaming: bool = False,
    run_tier0_and_tier1: bool = False,
    smoke: bool = False,
) -> Tuple[Dict[str, Any], Optional[CalciteV2LM]]:
    print(f"\n==================== [{arm_name}] ====================", flush=True)
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    prev_tf32_mm = bool(torch.backends.cuda.matmul.allow_tf32) if torch.cuda.is_available() else False
    prev_tf32_cudnn = bool(torch.backends.cudnn.allow_tf32) if torch.cuda.is_available() else False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)

    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
    n_cog = max(1, int(round(batch_size * 0.15))) if batch_size >= 8 else 1
    n_copy = max(1, int(round(batch_size * 0.10))) if batch_size >= 8 else 1

    try:
        set_all_seeds(seed)
        if reuse_compiled_model is not None:
            model = reuse_compiled_model
            already_compiled = True
        else:
            model = CalciteV2LM(cfg).to(device)
            load_frozen_router_into(model, router_state)
            already_compiled = False

        n_params = int(sum(p.numel() for p in model.parameters()))
        cache_kb = round(compute_v2_state_cache_kb(cfg, 32768), 2)

        # Gate-2 Init-Std & live gradients probe (before compile & before training)
        rng_probe_py = random.Random(seed + 999)
        rng_probe_np = np.random.default_rng(seed + 999)
        xb_p, yb_p, _ = sample_tournament_batch(
            train_mm, tok, tr_sents, min(8, batch_size), seq_len, rng_probe_py, rng_probe_np, 1, 1
        )
        gate2 = (
            {"skipped": "reused_plateau_model"}
            if resume_plateau_path is not None
            else run_gate2_init_probe(model, xb_p.to(device), yb_p.to(device), device, use_amp)
        )
        if not gate2.get("skipped"):
            assert gate2["all_grads_finite"], f"Gate-2 non-finite gradients in {arm_name}: {gate2}"
            print(
                f"  [Gate-2 {arm_name}] loss={gate2['init_loss']} | res_std={gate2['block_std'].get('cortex_post_last')} "
                f"({gate2['block_dtype'].get('cortex_post_last')}) | live_grads={gate2['live_finite_grad_tensors']}/{gate2['trainable_params_tensors']}",
                flush=True,
            )

        compiled_ok = already_compiled
        if use_compile and (not already_compiled) and device == "cuda":
            compiled_ok = bool(compile_calcite_v2(model, mode="default", dynamic=False))

        opt = MuonAdamWHybrid(
            model,
            base_lr=base_lr,
            muon_lr_mult=6.0,
            weight_decay=0.04,
            device=device,
            woodbury_lr_mult=0.45,
            protect_crystal=False,
            ns_schedule=ns_schedule,
        )
        sched = WSDLRSchedule(
            base_lr=base_lr,
            min_lr=min_lr,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            decay_mode="cosine",
        )

        # Identical data stream RNGs across ALL arms
        rng_py = random.Random(seed + 3141)
        rng_np = np.random.default_rng(seed + 3141)
        start_step = 1
        if resume_plateau_path is not None:
            meta_r = load_plateau_checkpoint(
                resume_plateau_path,
                model,
                opt,
                rng_py=rng_py,
                rng_np=rng_np,
                strict=True,
                restore_rng=True,
                preserve_ns_schedule=True,
            )
            start_step = int(meta_r["step"]) + 1
            print(
                f"  [WSD Resume] Loaded plateau step={meta_r['step']} -> running steps {start_step}..{total_steps} "
                f"with ns_schedule={opt.ns_schedule!r}",
                flush=True,
            )

        n_run_steps = max(0, total_steps - start_step + 1)
        n_acc = max(1, (batch_size + micro_bs - 1) // micro_bs)
        val_curve: Dict[int, float] = {}
        warmup_measure_steps = 4 if (compiled_ok and not already_compiled) else 2

        def _sample_next():
            return sample_tournament_batch(
                train_mm, tok, tr_sents, batch_size, seq_len, rng_py, rng_np, n_cog, n_copy
            )

        t_train0 = time.perf_counter()
        t_steady0: Optional[float] = None
        steady_steps = 0
        steps_completed = 0
        last_batch: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        killed_by_oeb = False
        grad_norms: Dict[int, float] = {}
        chunk_ms_samples: List[float] = []
        chunk_t0: Optional[float] = None
        chunk_steps = 0
        chunk_span = 25 if (not smoke) else 2

        with BatchPrefetcher(
            _sample_next, device=device, total_steps=n_run_steps, prefetch_depth=4, enabled=(n_run_steps >= 4)
        ) as pf:
            for step in range(start_step, total_steps + 1):
                local_i = step - start_step + 1
                if local_i == warmup_measure_steps + 1:
                    sync_if_cuda(device)
                    if device == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                    t_steady0 = time.perf_counter()
                    chunk_t0 = t_steady0
                    chunk_steps = 0

                model.train()
                model.set_crystal_schedule(step, total_steps)
                sched.apply(opt, step)
                xb_f, yb_f, mb_f = pf.next_batch()
                last_batch = (xb_f, yb_f, mb_f)

                opt.zero_grad(set_to_none=True)
                for st in range(0, batch_size, micro_bs):
                    xb = xb_f[st : st + micro_bs]
                    yb = yb_f[st : st + micro_bs]
                    mb = mb_f[st : st + micro_bs]
                    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                        lg = model(xb)
                        lg_f = lg.float()
                        tok_ce = F.cross_entropy(
                            lg_f.reshape(-1, lg_f.size(-1)),
                            yb.reshape(-1),
                            ignore_index=-100,
                            reduction="none",
                        ).view_as(yb)
                        valid = yb != -100
                        m_copy_f = (mb & valid).float()
                        m_pref_f = ((~mb) & valid).float()
                        n_c_tok = m_copy_f.sum()
                        n_p_tok = m_pref_f.sum()
                        ce_p = (tok_ce * m_pref_f).sum() / n_p_tok.clamp(min=1.0)
                        has_c = n_c_tok > 0.0
                        ce_c = torch.where(
                            has_c,
                            (tok_ce * m_copy_f).sum() / n_c_tok.clamp(min=1.0),
                            torch.zeros_like(ce_p),
                        )
                        ce = torch.where(has_c, 0.85 * ce_p + 0.15 * ce_c, ce_p)
                        loss = (ce + 0.04 * model.get_commitment_loss()) / float(n_acc)
                    loss.backward()

                gn_t = torch.nn.utils.clip_grad_norm_(opt.trainable_params, 1.0)
                opt.step()
                steps_completed += 1
                if t_steady0 is not None:
                    steady_steps += 1
                    chunk_steps += 1
                    if chunk_steps >= chunk_span and chunk_t0 is not None:
                        sync_if_cuda(device)
                        now_c = time.perf_counter()
                        chunk_ms_samples.append(1000.0 * (now_c - chunk_t0) / float(chunk_steps))
                        chunk_t0 = now_c
                        chunk_steps = 0

                # Save WSD plateau checkpoint at stable_end (step=450)
                if plateau_save_path is not None and sched.is_plateau_save_step(step):
                    sync_if_cuda(device)
                    t_sp = time.perf_counter()
                    save_plateau_checkpoint(
                        plateau_save_path,
                        model,
                        opt,
                        step,
                        config=cfg,
                        rng_py=rng_py,
                        rng_np=rng_np,
                        schedule=sched,
                        stage="S2_TOURNAMENT",
                        extra={"arm": arm_name, "ns_schedule": ns_schedule},
                    )
                    sp_dt = time.perf_counter() - t_sp
                    if t_steady0 is not None:
                        t_steady0 += sp_dt
                    if chunk_t0 is not None:
                        chunk_t0 += sp_dt
                    t_train0 += sp_dt
                    print(f"  [WSD Save] Saved plateau checkpoint at step={step} -> {plateau_save_path.name}", flush=True)

                # Record mid-run Val BPB curve / OEBEarlyStopper check on uncompiled eager copy
                # (avoids triggering Dynamo `tight_eval` recompilations on compiled `model`!)
                need_val = (step in record_val_steps) or (early_stopper is not None and early_stopper.should_check(step))
                if need_val:
                    sync_if_cuda(device)
                    t_pause = time.perf_counter()
                    mid_m = CalciteV2LM(cfg).to(device)
                    mid_m.load_state_dict(unwrap_compiled_state_dict(model), strict=True)
                    mid_m.byte_chunker.lock_router(True)
                    mid_m.set_crystal_schedule(1, 1)
                    mid_m.set_snap_alpha(1.0)
                    mid_m.eval()
                    v_mid = float(
                        eval_val_memmap_bpb(
                            mid_m, val_mm, seq_len=seq_len, n_batches=4 if smoke else 8, batch_size=min(16, batch_size), device=device
                        )
                    )
                    del mid_m
                    val_curve[step] = round(v_mid, 4)
                    pause_dt = time.perf_counter() - t_pause
                    if t_steady0 is not None:
                        t_steady0 += pause_dt
                    if chunk_t0 is not None:
                        chunk_t0 += pause_dt
                    t_train0 += pause_dt
                    if early_stopper is not None and early_stopper.should_check(step):
                        killed, reason = early_stopper.check(step, v_mid)
                        print(f"  [OEB {arm_name}] {reason}", flush=True)
                        if killed:
                            killed_by_oeb = True
                            break

                if step in (start_step, total_steps) or (step % max(1, total_steps // 4) == 0):
                    gn_v = round(float(gn_t.item()), 3)
                    grad_norms[step] = gn_v
                    dt_so_far = time.perf_counter() - t_train0
                    print(
                        f"  [{arm_name:20s}] step {step:4d}/{total_steps} | lr={sched.lr_at(step):.2e} | "
                        f"gn={gn_v:6.3f} | elapsed={dt_so_far:5.1f}s",
                        flush=True,
                    )

        sync_if_cuda(device)
        total_train_s = time.perf_counter() - t_train0
        if t_steady0 is not None and steady_steps > 0:
            steady_s = time.perf_counter() - t_steady0
            raw_steady_ms = 1000.0 * steady_s / float(steady_steps)
        else:
            steady_s = total_train_s
            raw_steady_ms = 1000.0 * total_train_s / float(max(1, steps_completed))
        # Median over 25-step chunks cleanly excludes 1-time mid-run `is_hard` Triton compile spikes
        steady_ms = float(np.median(chunk_ms_samples)) if chunk_ms_samples else raw_steady_ms
        compile_warmup_s = max(0.0, total_train_s - (steady_ms / 1000.0) * float(steps_completed))
        steady_bps = (batch_size * seq_len) / max(1e-9, steady_ms / 1000.0)
        peak_alloc_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 1) if device == "cuda" else 0.0
        peak_res_mb = round(torch.cuda.max_memory_reserved() / (1024**2), 1) if device == "cuda" else 0.0

        # Measure Muon vs AdamW split cleanly without altering trained weights
        opt_split = (
            _measure_muon_adamw_split(
                model, opt, last_batch[0], last_batch[1], last_batch[2], micro_bs, device, use_amp, n_iters=3 if smoke else 10
            )
            if last_batch is not None
            else {"muon_ns_ms": 0.0, "adamw_ms": 0.0, "opt_total_ms": 0.0}
        )

        # Evaluate trained arm (clean unwrapped model for eval so zero Inductor eval-shape recompiles!)
        trained_sd_cpu = unwrap_compiled_state_dict(model)
        eval_model = CalciteV2LM(cfg).to(device)
        eval_model.load_state_dict(trained_sd_cpu, strict=True)
        eval_model.byte_chunker.lock_router(True)
        eval_model.set_crystal_schedule(1, 1)
        eval_model.set_snap_alpha(1.0)
        eval_model.eval()

        eval_res: Dict[str, Any] = {}
        if not killed_by_oeb:
            if run_tier0_and_tier1:
                eval_res["tier0"] = run_tiered_eval(
                    eval_model,
                    tier=0,
                    val_mm=val_mm,
                    eval_wins=eval_wins,
                    tok=tok,
                    s5_suites=s5_suites,
                    ov_suites=ov_suites,
                    trap_suites=trap_suites,
                    device=device,
                    seq_len=seq_len,
                    eval_bs=16,
                    smoke=smoke,
                )
            eval_res["tier1"] = run_tiered_eval(
                eval_model,
                tier=1,
                val_mm=val_mm,
                eval_wins=eval_wins,
                tok=tok,
                s5_suites=s5_suites,
                ov_suites=ov_suites,
                trap_suites=trap_suites,
                device=device,
                seq_len=seq_len,
                eval_bs=16,
                smoke=smoke,
            )

        # 2*sigma calibration check on C0_control (per-batch Val BPB SEM across 32 batches)
        sigma_cal: Optional[Dict[str, Any]] = None
        if run_tier0_and_tier1 and (not killed_by_oeb):
            rng_cal = np.random.default_rng(77777)
            bpb_samples: List[float] = []
            n_cal_b = 4 if smoke else 32
            with torch.inference_mode():
                for _ in range(n_cal_b):
                    xb_v, yb_v = sample_memmap_batch(val_mm, 16, seq_len, rng_cal, device)
                    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                        lg_v = eval_model(xb_v)
                    ce_v = float(
                        F.cross_entropy(lg_v.float().reshape(-1, lg_v.size(-1)), yb_v.reshape(-1), ignore_index=0).item()
                    )
                    bpb_samples.append(ce_v / math.log(2.0))
            arr_b = np.asarray(bpb_samples, dtype=np.float64)
            sem = float(arr_b.std(ddof=1) / math.sqrt(len(arr_b))) if len(arr_b) > 1 else 0.0
            sigma_cal = {
                "n_batches": n_cal_b,
                "mean_bpb": round(float(arr_b.mean()), 4),
                "batch_std_bpb": round(float(arr_b.std(ddof=1)), 4) if len(arr_b) > 1 else 0.0,
                "sem_bpb": round(sem, 5),
                "two_sigma_bpb": round(2.0 * sem, 5),
                "phase_m_ref_two_sigma_bpb": 0.0057,
            }

        # Extract concise flat summary metrics for easy diff tables
        t1 = eval_res.get("tier1") or {}
        cog1 = t1.get("cognitive") or {}
        ndl1 = t1.get("ood_verbatim_ladder") or {}
        ndl_1500 = ndl1.get("L_1500") or ndl1.get("L_256") or {}
        orig_1500 = ndl1.get("OrigRefCode_1500B") or {}
        ext1 = t1.get("extrap_4x") or {}
        summary = {
            "val_bpb": t1.get("val_bpb"),
            "ow6_bpb": t1.get("ow6_mean_bpb"),
            "l2048_bpb": ext1.get("bpb_L2048", ext1.get("bpb_L1024")),
            "s5_k2_pct": ((cog1.get("s5") or {}).get("k2") or {}).get("exact_pct"),
            "s5_sum_pct": cog1.get("s5_sum"),
            "ov_long_pct": (cog1.get("ov_ood_long") or {}).get("exact_pct"),
            "sel_f1_long_pct": cog1.get("selective_f1_ood_long"),
            "needle_1500_exact_pct": ndl_1500.get("exact_match_pct"),
            "needle_1500_byte_pct": ndl_1500.get("byte_acc_pct"),
            "orig_ref_1500_exact_pct": orig_1500.get("exact_match_pct"),
        }

        print(
            f"  [{arm_name:20s} DONE] steady={steady_ms:.2f}ms/st ({steady_bps:,.0f} B/s) | "
            f"Muon={opt_split['muon_ns_ms']:.2f}ms | VRAM={peak_alloc_mb:.0f}/{peak_res_mb:.0f}MB | "
            f"Val={summary['val_bpb']} OW6={summary['ow6_bpb']} L2K={summary['l2048_bpb']} "
            f"S5k2={summary['s5_k2_pct']}% OVL={summary['ov_long_pct']}% Ndl1500={summary['needle_1500_exact_pct']}%",
            flush=True,
        )

        arm_out: Dict[str, Any] = {
            "arm": arm_name,
            "params": n_params,
            "cache_kb_o1_l32k": cache_kb,
            "axis_config": {
                "dec_dw3_enabled": cfg.dec_dw3_enabled,
                "dw_bottleneck_ratio": cfg.dw_bottleneck_ratio,
                "bf16_residual_stream": cfg.bf16_residual_stream,
                "ns_schedule": ns_schedule,
                "allow_tf32": allow_tf32,
                "base_lr": base_lr,
                "compiled": compiled_ok,
                "resumed_from_step": start_step - 1,
            },
            "gate2_init": gate2,
            "train": {
                "steps_completed": steps_completed,
                "total_train_s": round(total_train_s, 2),
                "compile_warmup_s": round(compile_warmup_s, 2),
                "steady_ms_per_step": round(steady_ms, 2),
                "raw_steady_ms_per_step": round(raw_steady_ms, 2),
                "steady_bytes_per_sec": round(steady_bps, 1),
                "peak_alloc_mb": peak_alloc_mb,
                "peak_reserved_mb": peak_res_mb,
                "opt_split_ms": opt_split,
                "grad_norms": grad_norms,
                "mid_val_curve": val_curve,
                "oeb_summary": early_stopper.summary() if early_stopper is not None else None,
            },
            "summary": summary,
            "eval": eval_res,
            "two_sigma_calibration": sigma_cal,
            "lut_streaming": None,
        }

        _ = run_streaming
        del eval_model, opt
        keep_model = model if plateau_save_path is not None else None
        if keep_model is None:
            del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        return arm_out, keep_model, trained_sd_cpu
    finally:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32_mm
            torch.backends.cudnn.allow_tf32 = prev_tf32_cudnn


# ---------------------------------------------------------------------------
# 5. Blackwell `sm_120` FP8 (`torch._scaled_mm`) vs BF16 Linear GEMM Micro-Probe
# ---------------------------------------------------------------------------


@torch.inference_mode()
def run_blackwell_fp8_micro_probe(device: str, smoke: bool = False) -> Dict[str, Any]:
    """Benchmarks BF16 vs FP8 (`e4m3fn` via `torch._scaled_mm`) on S2_25M & S3_85M shapes."""
    if device != "cuda" or not torch.cuda.is_available():
        return {"skipped": "CUDA unavailable"}

    props = torch.cuda.get_device_properties(0)
    cc = f"sm_{props.major}{props.minor}"
    has_fp8 = hasattr(torch, "float8_e4m3fn") and hasattr(torch, "_scaled_mm")
    out: Dict[str, Any] = {
        "gpu_name": props.name,
        "compute_capability": cc,
        "torch_version": torch.__version__,
        "fp8_supported": has_fp8,
        "shapes": [],
    }
    if not has_fp8:
        out["error"] = "torch.float8_e4m3fn or torch._scaled_mm unavailable"
        return out

    shapes = [
        ("S2_25M_ffn", 448, 1184),
        ("S2_25M_qkv", 448, 1344),
        ("S3_85M_ffn", 768, 2048),
        ("S3_85M_qkv", 768, 2304),
    ]
    ms = (4096,) if smoke else (4096, 8192, 16384)
    n_warm = 5 if smoke else 20
    n_it = 10 if smoke else 80
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)

    print(f"\n================ [BLACKWELL {cc} FP8 vs BF16 MICRO-PROBE] ================", flush=True)
    for label, k_dim, n_dim in shapes:
        for m_dim in ms:
            x_bf16 = torch.randn(m_dim, k_dim, device="cuda", dtype=torch.bfloat16)
            w_bf16 = torch.randn(n_dim, k_dim, device="cuda", dtype=torch.bfloat16) / math.sqrt(float(k_dim))

            # Pre-quantize for raw GEMM benchmark (column-major B required by cuBLASLt FP8)
            sa = (x_bf16.abs().amax().float().clamp(min=1e-12) / fp8_max).reshape(1)
            sb = (w_bf16.abs().amax().float().clamp(min=1e-12) / fp8_max).reshape(1)
            x_fp8 = (x_bf16.float() / sa).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
            w_fp8_col = (w_bf16.float() / sb).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn).t()

            def _bench(fn) -> float:
                for _ in range(n_warm):
                    _ = fn()
                torch.cuda.synchronize()
                s_ev = torch.cuda.Event(enable_timing=True)
                e_ev = torch.cuda.Event(enable_timing=True)
                s_ev.record()
                for _ in range(n_it):
                    _ = fn()
                e_ev.record()
                torch.cuda.synchronize()
                return float(s_ev.elapsed_time(e_ev)) / float(n_it)

            bf16_ms = _bench(lambda x=x_bf16, w=w_bf16: F.linear(x, w))
            y_ref = F.linear(x_bf16, w_bf16).float()

            fp8_raw_ms: Optional[float] = None
            fp8_e2e_ms: Optional[float] = None
            rel_err: Optional[float] = None
            err_msg: Optional[str] = None
            try:
                y_fp8 = torch._scaled_mm(
                    x_fp8, w_fp8_col, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16, use_fast_accum=True
                )
                if isinstance(y_fp8, tuple):
                    y_fp8 = y_fp8[0]
                rel_err = float((y_ref - y_fp8.float()).norm() / y_ref.norm().clamp(min=1e-8))
                fp8_raw_ms = _bench(
                    lambda xq=x_fp8, wq=w_fp8_col, s1=sa, s2=sb: torch._scaled_mm(
                        xq, wq, scale_a=s1, scale_b=s2, out_dtype=torch.bfloat16, use_fast_accum=True
                    )
                )

                def _e2e_fp8(xb_in=x_bf16, wb_in=w_bf16):
                    sa_d = (xb_in.abs().amax().float().clamp(min=1e-12) / fp8_max).reshape(1)
                    sb_d = (wb_in.abs().amax().float().clamp(min=1e-12) / fp8_max).reshape(1)
                    xq = (xb_in.float() / sa_d).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
                    wq_col = (wb_in.float() / sb_d).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn).t()
                    return torch._scaled_mm(
                        xq, wq_col, scale_a=sa_d, scale_b=sb_d, out_dtype=torch.bfloat16, use_fast_accum=True
                    )

                fp8_e2e_ms = _bench(_e2e_fp8)
            except Exception as exc:
                err_msg = f"{type(exc).__name__}: {exc}"

            flops = 2.0 * float(m_dim) * float(k_dim) * float(n_dim)
            bf16_tflops = round(flops / (bf16_ms * 1e9), 2)
            raw_sp = round(bf16_ms / fp8_raw_ms, 2) if fp8_raw_ms else None
            e2e_sp = round(bf16_ms / fp8_e2e_ms, 2) if fp8_e2e_ms else None
            row = {
                "label": label,
                "M": m_dim,
                "K": k_dim,
                "N": n_dim,
                "bf16_ms": round(bf16_ms, 4),
                "bf16_tflops": bf16_tflops,
                "fp8_raw_gemm_ms": round(fp8_raw_ms, 4) if fp8_raw_ms is not None else None,
                "fp8_raw_tflops": round(flops / (fp8_raw_ms * 1e9), 2) if fp8_raw_ms else None,
                "fp8_raw_speedup": raw_sp,
                "fp8_dyn_e2e_ms": round(fp8_e2e_ms, 4) if fp8_e2e_ms is not None else None,
                "fp8_dyn_e2e_speedup": e2e_sp,
                "rel_l2_err": round(rel_err, 5) if rel_err is not None else None,
                "error": err_msg,
            }
            out["shapes"].append(row)
            print(
                f"  [{label:11s} M={m_dim:5d} {k_dim:3d}x{n_dim:4d}] BF16={bf16_ms:.4f}ms ({bf16_tflops:5.1f} TF) | "
                f"FP8_raw={row['fp8_raw_gemm_ms']}ms ({raw_sp}x) | FP8_e2e={row['fp8_dyn_e2e_ms']}ms ({e2e_sp}x) | "
                f"relErr={row['rel_l2_err']} {err_msg or ''}",
                flush=True,
            )
    torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# 6. Main Tournament Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase O: Tier 2 Controlled Single-Axis 5M Tournament & FP8 Probe")
    ap.add_argument("--data-dir", type=Path, default=ROOT_DIR / "data_cache")
    ap.add_argument("--out-json", type=Path, default=ROOT_DIR / "benchmarks" / "phase_o_tier2_ablations_5m.json")
    ap.add_argument("--plateau-dir", type=Path, default=ROOT_DIR / "data_cache" / "phase_o_plateaus")
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--warmup-steps", type=int, default=40)
    ap.add_argument("--decay-steps", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--micro-bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--smoke-local", action="store_true")
    ap.add_argument(
        "--refresh-wsd-branch",
        action="store_true",
        help="Reload existing --out-json and re-run ONLY WSD_branch_250 from saved plateau checkpoint.",
    )
    args = ap.parse_args()

    device = "cuda" if has_usable_cuda() else "cpu"
    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram_gb < 12.0:
            torch.cuda.set_per_process_memory_fraction(0.90, 0)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.plateau_dir.mkdir(parents=True, exist_ok=True)

    tok = ByteBPETokenizer(merges=[])
    seq_len = 128 if args.smoke_local else 512
    tr_sents, eval_wins, s5_suites, ov_suites, trap_suites = build_eval_bundles(
        ROOT_DIR, tok, n_eval=16 if args.smoke_local else 80, seq_len=seq_len
    )

    if args.smoke_local:
        rng_f = np.random.default_rng(args.seed)
        fake = rng_f.integers(32 + 3, 126 + 3, size=60_000, dtype=np.uint16)
        train_mm = val_mm = fake
        total_steps, warm_steps, decay_steps = 8, 2, 3
        batch_size, micro_bs = 8, 4
        oeb_step = 3
    else:
        train_bin, val_bin = ensure_phase_k_corpus(args.data_dir)
        train_mm = load_utf8_corpus_array(train_bin)
        val_mm = load_utf8_corpus_array(val_bin)
        total_steps = int(args.steps)
        warm_steps = int(args.warmup_steps)
        decay_steps = int(args.decay_steps)
        batch_size = int(args.batch_size)
        micro_bs = int(args.micro_bs)
        oeb_step = 250

    router_state = torch.load(
        ROOT_DIR / "benchmarks" / "phase_l_v2" / "frozen_utf8_router.pt", map_location="cpu", weights_only=True
    )
    use_compile = (not args.no_compile) and (not args.smoke_local) and (device == "cuda")

    results: Dict[str, Any] = {
        "meta": {
            "phase": "Phase_O_Tier2_SingleAxis_5M_Ablation_Tournament",
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "seed": int(args.seed),
            "total_steps": total_steps,
            "warmup_steps": warm_steps,
            "stable_end_step": total_steps - decay_steps,
            "decay_steps": decay_steps,
            "batch_size": batch_size,
            "micro_bs": micro_bs,
            "seq_len": seq_len,
            "compiled": use_compile,
            "smoke": bool(args.smoke_local),
        },
        "arms": {},
        "deltas_vs_c0": {},
        "fp8_blackwell_probe": {},
    }

    def _save() -> None:
        args.out_json.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    def _recompute_deltas() -> None:
        c0_s = results["arms"]["C0_control"]["summary"]
        c0_t = results["arms"]["C0_control"]["train"]
        c0_st = results["arms"]["C0_control"].get("lut_streaming") or {}
        results["deltas_vs_c0"] = {}
        for arm_nm, arm_d in results["arms"].items():
            if arm_nm in ("C0_control", "OEB_degraded_synth"):
                continue
            s = arm_d["summary"]
            t = arm_d["train"]
            st = arm_d.get("lut_streaming") or {}

            def _diff(a, b):
                return round(float(a) - float(b), 4) if (a is not None and b is not None) else None

            def _sp(a, b):
                return round(float(a) / max(1e-9, float(b)), 3) if (a is not None and b is not None) else None

            results["deltas_vs_c0"][arm_nm] = {
                "delta_val_bpb": _diff(s.get("val_bpb"), c0_s.get("val_bpb")),
                "delta_ow6_bpb": _diff(s.get("ow6_bpb"), c0_s.get("ow6_bpb")),
                "delta_l2048_bpb": _diff(s.get("l2048_bpb"), c0_s.get("l2048_bpb")),
                "delta_s5_k2_pct": _diff(s.get("s5_k2_pct"), c0_s.get("s5_k2_pct")),
                "delta_ov_long_pct": _diff(s.get("ov_long_pct"), c0_s.get("ov_long_pct")),
                "delta_needle_1500_pct": _diff(s.get("needle_1500_exact_pct"), c0_s.get("needle_1500_exact_pct")),
                "train_step_speedup": _sp(c0_t.get("steady_ms_per_step"), t.get("steady_ms_per_step")),
                "muon_speedup": _sp(
                    c0_t["opt_split_ms"].get("muon_ns_ms"), t["opt_split_ms"].get("muon_ns_ms")
                ),
                "vram_alloc_delta_mb": _diff(t.get("peak_alloc_mb"), c0_t.get("peak_alloc_mb")),
                "cpu_stream_speedup": _sp(
                    (st.get("cpu_avx512_fused_lut") or {}).get("decode_bps"),
                    (c0_st.get("cpu_avx512_fused_lut") or {}).get("decode_bps"),
                ),
                "gpu_cg_stream_speedup": _sp(
                    (st.get("gpu_cuda_graph_fused_lut") or {}).get("decode_bps"),
                    (c0_st.get("gpu_cuda_graph_fused_lut") or {}).get("decode_bps"),
                ),
                "ram_delta_kb": _diff(arm_d.get("cache_kb_o1_l32k"), results["arms"]["C0_control"].get("cache_kb_o1_l32k")),
            }

    plateau_ckpt = args.plateau_dir / ("smoke_c0_plateau.pt" if args.smoke_local else "c0_control_step450_plateau.pt")
    cfg_c0 = build_v31_5m_arm_config(smoke=args.smoke_local)

    if args.refresh_wsd_branch:
        if not args.out_json.exists():
            raise FileNotFoundError(f"--refresh-wsd-branch requires existing {args.out_json}")
        if not plateau_ckpt.exists():
            raise FileNotFoundError(f"--refresh-wsd-branch requires saved plateau {plateau_ckpt}")
        results = json.loads(args.out_json.read_text(encoding="utf-8"))
        wsd_res, _, _ = run_single_arm(
            "WSD_branch_250",
            cfg_c0,
            router_state,
            train_mm,
            val_mm,
            tok,
            tr_sents,
            eval_wins,
            s5_suites,
            ov_suites,
            trap_suites,
            device=device,
            total_steps=total_steps,
            warmup_steps=warm_steps,
            decay_steps=decay_steps,
            batch_size=batch_size,
            micro_bs=micro_bs,
            seq_len=seq_len,
            seed=args.seed,
            ns_schedule="polar4",
            allow_tf32=False,
            use_compile=use_compile,
            resume_plateau_path=plateau_ckpt,
            run_streaming=False,
            run_tier0_and_tier1=False,
            smoke=args.smoke_local,
        )
        results["arms"]["WSD_branch_250"] = wsd_res
        _recompute_deltas()
        _save()
        print(f"\n[PHASE O WSD BRANCH REFRESH COMPLETE] Updated -> {args.out_json}", flush=True)
        return

    # 1. C0_control (WSD plateau saved at step=450, Tier-0 vs Tier-1 timing, 2*sigma)
    stream_candidates: Dict[str, Tuple[CalciteV2Config, Dict[str, torch.Tensor]]] = {}
    c0_res, c0_compiled_model, c0_sd_cpu = run_single_arm(
        "C0_control",
        cfg_c0,
        router_state,
        train_mm,
        val_mm,
        tok,
        tr_sents,
        eval_wins,
        s5_suites,
        ov_suites,
        trap_suites,
        device=device,
        total_steps=total_steps,
        warmup_steps=warm_steps,
        decay_steps=decay_steps,
        batch_size=batch_size,
        micro_bs=micro_bs,
        seq_len=seq_len,
        seed=args.seed,
        ns_schedule="quintic5",
        allow_tf32=False,
        use_compile=use_compile,
        plateau_save_path=plateau_ckpt,
        record_val_steps=(oeb_step, total_steps - decay_steps),
        run_streaming=False,
        run_tier0_and_tier1=True,
        smoke=args.smoke_local,
    )
    results["arms"]["C0_control"] = c0_res
    stream_candidates["C0_control"] = (cfg_c0, c0_sd_cpu)
    _save()

    # 2. WSD_branch_250 immediately reusing C0's compiled model (0s recompile, ~25s wall-time!)
    wsd_res, _, _ = run_single_arm(
        "WSD_branch_250",
        cfg_c0,
        router_state,
        train_mm,
        val_mm,
        tok,
        tr_sents,
        eval_wins,
        s5_suites,
        ov_suites,
        trap_suites,
        device=device,
        total_steps=total_steps,
        warmup_steps=warm_steps,
        decay_steps=decay_steps,
        batch_size=batch_size,
        micro_bs=micro_bs,
        seq_len=seq_len,
        seed=args.seed,
        ns_schedule="polar4",
        allow_tf32=False,
        use_compile=use_compile,
        resume_plateau_path=plateau_ckpt,
        reuse_compiled_model=c0_compiled_model,
        run_streaming=False,
        run_tier0_and_tier1=False,
        smoke=args.smoke_local,
    )
    results["arms"]["WSD_branch_250"] = wsd_res
    if c0_compiled_model is not None:
        del c0_compiled_model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    _save()

    # 3. B1_polar4 & B2_polar3 (identical architecture to C0 — clean empty GPU VRAM baseline!)
    for arm_nm, ns_sch in [("B1_polar4", "polar4"), ("B2_polar3", "polar3")]:
        arm_r, _, _ = run_single_arm(
            arm_nm,
            cfg_c0,
            router_state,
            train_mm,
            val_mm,
            tok,
            tr_sents,
            eval_wins,
            s5_suites,
            ov_suites,
            trap_suites,
            device=device,
            total_steps=total_steps,
            warmup_steps=warm_steps,
            decay_steps=decay_steps,
            batch_size=batch_size,
            micro_bs=micro_bs,
            seq_len=seq_len,
            seed=args.seed,
            ns_schedule=ns_sch,
            allow_tf32=False,
            use_compile=use_compile,
            run_streaming=False,
            run_tier0_and_tier1=False,
            smoke=args.smoke_local,
        )
        results["arms"][arm_nm] = arm_r
        _save()

    # 4. OEB_degraded_synth (live OEBEarlyStopper kill verification at step=250 on identical model)
    ctrl_curve = {int(k): float(v) for k, v in c0_res["train"]["mid_val_curve"].items()}
    oeb_gate = OEBEarlyStopper(control_curve=ctrl_curve, margin_bpb=0.025, axis_name="degraded_lr_synth")
    oeb_guard = OEBEarlyStopper(control_curve=ctrl_curve, margin_bpb=0.025, disabled_for_axis=True, axis_name="cog_axis")
    guard_killed, guard_msg = oeb_guard.check(oeb_step, ctrl_curve.get(oeb_step, 3.0) + 1.0)
    assert not guard_killed, f"disabled_for_axis guard failed: {guard_msg}"

    oeb_res, _, _ = run_single_arm(
        "OEB_degraded_synth",
        cfg_c0,
        router_state,
        train_mm,
        val_mm,
        tok,
        tr_sents,
        eval_wins,
        s5_suites,
        ov_suites,
        trap_suites,
        device=device,
        total_steps=total_steps,
        warmup_steps=warm_steps,
        decay_steps=decay_steps,
        batch_size=batch_size,
        micro_bs=micro_bs,
        seq_len=seq_len,
        seed=args.seed,
        ns_schedule="quintic5",
        allow_tf32=False,
        base_lr=3.0e-5,
        min_lr=1.0e-5,
        use_compile=False,
        early_stopper=oeb_gate,
        run_streaming=False,
        run_tier0_and_tier1=False,
        smoke=args.smoke_local,
    )
    oeb_res["disabled_for_axis_guard_check"] = {"killed": guard_killed, "reason": guard_msg}
    results["arms"]["OEB_degraded_synth"] = oeb_res
    _save()

    # 5. Architectural & Precision Axes: A1_no_dec_dw3, A2_dw_bottleneck_05, C1_tf32_bf16res
    arch_specs = [
        ("A1_no_dec_dw3", dict(dec_dw3_enabled=False, dw_bottleneck_ratio=1.0, bf16_residual_stream=False), False, True),
        ("A2_dw_bottleneck_05", dict(dec_dw3_enabled=True, dw_bottleneck_ratio=0.5, bf16_residual_stream=False), False, True),
        ("C1_tf32_bf16res", dict(dec_dw3_enabled=True, dw_bottleneck_ratio=1.0, bf16_residual_stream=True), True, False),
    ]
    for arm_nm, cfg_kw, tf32_flag, do_stream in arch_specs:
        cfg_a = build_v31_5m_arm_config(smoke=args.smoke_local, **cfg_kw)
        arm_r, _, arm_sd_cpu = run_single_arm(
            arm_nm,
            cfg_a,
            router_state,
            train_mm,
            val_mm,
            tok,
            tr_sents,
            eval_wins,
            s5_suites,
            ov_suites,
            trap_suites,
            device=device,
            total_steps=total_steps,
            warmup_steps=warm_steps,
            decay_steps=decay_steps,
            batch_size=batch_size,
            micro_bs=micro_bs,
            seq_len=seq_len,
            seed=args.seed,
            ns_schedule="quintic5",
            allow_tf32=tf32_flag,
            use_compile=use_compile,
            run_streaming=False,
            run_tier0_and_tier1=False,
            smoke=args.smoke_local,
        )
        results["arms"][arm_nm] = arm_r
        if do_stream:
            stream_candidates[arm_nm] = (cfg_a, arm_sd_cpu)
        _save()

    # 6. Post-Training 2-bit LUT CPU AVX-512 & GPU CUDA Graph Streaming (Zero VRAM Pool Contamination)
    print("\n================ [POST-TRAIN 2-BIT LUT STREAMING BENCHMARK (C0, A1, A2)] ================", flush=True)
    for arm_nm, (cfg_s, sd_cpu) in stream_candidates.items():
        m_tmp = CalciteV2LM(cfg_s).cpu()
        m_tmp.load_state_dict(sd_cpu, strict=True)
        m_tmp.byte_chunker.lock_router(True)
        m_tmp.set_crystal_schedule(1, 1)
        m_tmp.set_snap_alpha(1.0)
        st_res = benchmark_arm_lut_streaming(m_tmp, cfg_s, val_mm, device=device, smoke=args.smoke_local)
        results["arms"][arm_nm]["lut_streaming"] = st_res
        cpu_r = st_res["cpu_avx512_fused_lut"]
        gpu_r = st_res.get("gpu_cuda_graph_fused_lut") or {}
        print(
            f"  [LUT Stream {arm_nm:20s}] CPU={cpu_r['decode_bps']:6.1f} B/s ({cpu_r['decode_ms_per_byte']:.3f}ms/B, err={cpu_r['vs_batch_max_err']:.1e}) | "
            f"GPU_CG={gpu_r.get('decode_bps')} B/s ({gpu_r.get('decode_ms_per_byte')}ms/B, err={gpu_r.get('vs_eager_max_err')}) | "
            f"RAM={cpu_r['state_kb']} KB (O1={cpu_r['strict_o1_ram']})",
            flush=True,
        )
        del m_tmp
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    _save()

    # 6. Compute Exact Deltas vs C0_control
    _recompute_deltas()

    # 7. Blackwell sm_120 FP8 Micro-Probe
    results["fp8_blackwell_probe"] = run_blackwell_fp8_micro_probe(device=device, smoke=args.smoke_local)
    _save()
    print(f"\n[PHASE O TOURNAMENT COMPLETE] Saved -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
