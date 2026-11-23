"""
Phase L Execution Runner (`scripts/colab_phase_l_v2_runner.py`):
Calcite v2 Architecture Ablations (`5M`), `Skaling` L-Shape Grid (`5M -> 25M -> 85M`),
`25M` Champion (`FP + 2-bit uint8 LUT`), `L = 32K` Long-Context Stress Test, and
Live `CalciteStreamEngine` `O(1)` Inference Benchmark.

Stages (`--stage`):
  - `ablations`: Standalone UTF-8 router pretrain + 1-axis `5M` ablation ladder (`A0..A4` + `A_best` + `RoPE_5M`)
  - `skaling`:   `L-shape` Skaling grid (`D-band` on `5M` + `N-band` on `5M/25M/85M`) fitting
                 `L(N, D) = (A / N^a + B / D^b)^k + E` vs additive Chinchilla (`k = 1`)
  - `champion`:  Full `25M` Champion (`FP + QAT -> 2-bit uint8 LUT`) + `L = 8K / 32K` stress test
                 + live `CalciteStreamEngine` CPU/GPU streaming speed & flat `O(1)` RAM verification
  - `all`:       Runs `ablations -> skaling -> champion` sequentially with full JSON checkpointing.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import gc
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tarfile
import time
from typing import Dict, List, Optional, Tuple

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
from experiments.scaling_phase_k.byte_baselines import (  # noqa: E402
    build_matched_baselines_for_tier,
    compute_rope_cache_kb,
)
from experiments.scaling_phase_k.eval_suite import (  # noqa: E402
    eval_length_extrapolation_bpb,
    eval_openweb_6domains_any_byte_lm,
    eval_val_memmap_bpb,
    has_usable_cuda,
    sample_memmap_batch,
    sync_if_cuda,
)
from run_simvq_split_benchmark import harmonic_mean  # noqa: E402
from simvq_split.eval_utils import evaluate_suite_batched, wilson_ci_95  # noqa: E402
from kernels.fast_calcite_scans import (  # noqa: E402
    clear_fast_qat_cache,
    unwrap_compiled_state_dict,
)
from kernels.fast_packed_lut import (  # noqa: E402
    pack_fast_model_2bit_lut,
    save_packed_lut_checkpoint,
)
from run_phase_e_fast_utf8_benchmark import build_disjoint_openweb_corpora  # noqa: E402
from simvq_split.baselines import BaselineConfig, RoPETransformerLM  # noqa: E402
from simvq_split.memory import compute_2bit_lut_footprint  # noqa: E402
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


DRIVE_CORPUS_FILE_ID = "1uXTZ8Umh8RxisFILGfU51tx3b6YqlMhf"


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_phase_k_corpus(data_dir: Path) -> Tuple[Path, Path]:
    """Ensures the zero-leakage `0.65 GB` Phase K UTF-8 memmap corpus is present."""
    train_bin = data_dir / "train_utf8.u16.bin"
    val_bin = data_dir / "val_utf8.u16.bin"
    if train_bin.is_file() and val_bin.is_file():
        return train_bin, val_bin

    data_dir.mkdir(parents=True, exist_ok=True)
    drive_tar = Path("/content/drive/MyDrive/calcite_cache/phase_k_corpus_cache.tar.gz")
    if drive_tar.is_file():
        print(f"[corpus] Unpacking cached Drive corpus from {drive_tar}...", flush=True)
        with tarfile.open(drive_tar, "r:gz") as tf:
            tf.extractall(data_dir.parent)
        if train_bin.is_file() and val_bin.is_file():
            return train_bin, val_bin

    tmp_tar = data_dir.parent / "phase_k_corpus_cache.tar.gz"
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "gdown"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        res = subprocess.run(
            [sys.executable, "-m", "gdown", "--id", DRIVE_CORPUS_FILE_ID, "-O", str(tmp_tar)],
            check=False,
        )
        if res.returncode == 0 and tmp_tar.is_file() and tmp_tar.stat().st_size > 100_000_000:
            print(f"[corpus] Extracting gdown corpus ({tmp_tar.stat().st_size:,} B)...", flush=True)
            with tarfile.open(tmp_tar, "r:gz") as tf:
                tf.extractall(data_dir.parent)
            if train_bin.is_file() and val_bin.is_file():
                return train_bin, val_bin
    except Exception as exc:
        print(f"[corpus] gdown fallback notice: {exc}", flush=True)

    print("[corpus] Building fresh scrubbed Phase K UTF-8 corpus...", flush=True)
    from scripts.colab_phase_k_scaling_runner import build_phase_k_utf8_corpus

    build_phase_k_utf8_corpus(ROOT_DIR, data_dir)
    assert train_bin.is_file() and val_bin.is_file()
    return train_bin, val_bin


def build_eval_bundles(
    repo_root: Path, tok: ByteBPETokenizer, n_eval: int = 150, seq_len: int = 512
) -> Tuple[List[str], Dict[str, List[List[int]]], Dict[str, list], Dict[str, list], Dict[str, list]]:
    tr_docs, te_docs, tr_sents, te_sents = load_gsm8k_corpora(repo_root)
    _, eval_wins, _, _ = build_disjoint_openweb_corpora(repo_root, tok, tr_docs, te_docs, seq_len=seq_len)
    s5_suites = {
        "k2": build_eval_s5_suite(tok, te_sents, 2, n_eval, (0, 1), 3100),
        "k4": build_eval_s5_suite(tok, te_sents, 4, n_eval, (0, 1), 3200),
        "k8": build_eval_s5_suite(tok, te_sents, 8, n_eval, (1, 1), 3300),
        "k16": build_eval_s5_suite(tok, te_sents, 16, n_eval, (1, 2), 3400),
    }
    ov_suites = {
        "in_dist": build_eval_overwrite_suite(tok, te_sents, 4, 2, 2, n_eval, 4100),
        "ood_med": build_eval_overwrite_suite(tok, te_sents, 7, 4, 10, n_eval, 4200),
        "ood_long": build_eval_overwrite_suite(tok, te_sents, 9, 5, 18, n_eval, 4300),
    }
    trap_suites = {
        "in_dist": build_eval_trap_suite(tok, te_sents, 4, 2, 2, n_eval, 5100),
        "ood_long": build_eval_trap_suite(tok, te_sents, 9, 5, 18, n_eval, 5300),
    }
    return tr_sents, eval_wins, s5_suites, ov_suites, trap_suites


def build_matched_rope_for_tier(tier: str, target_params: int) -> RoPETransformerLM:
    specs = build_matched_baselines_for_tier(tier, target_params)
    rcfg = BaselineConfig(**specs["RoPETransformerLM"]["config"])
    return RoPETransformerLM(rcfg)


def sample_cognitive_batch(
    tok: ByteBPETokenizer,
    tr_sents: List[str],
    batch_size: int,
    seq_len: int,
    rng: random.Random,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Samples mixed in-distribution `S_5`, `Overwrite`, and `Trap` byte episodes (answer-weighted)."""
    x_rows = np.zeros((batch_size, seq_len), dtype=np.int64)
    y_rows = np.full((batch_size, seq_len), -100, dtype=np.int64)
    for bi in range(batch_size):
        kind = bi % 3
        if kind == 0:
            k_ops = rng.choice([2, 3, 4, 6])
            prompt, ans, _ = make_s5_story(rng, k_ops, tr_sents, (0, 1))
        elif kind == 1:
            prompt, ans, _ = make_overwrite_story(rng, rng.randint(4, 6), rng.randint(2, 3), tr_sents, rng.randint(2, 5))
        else:
            prompt, ans, _ = make_trap_overwrite_story(rng, rng.randint(4, 6), rng.randint(2, 3), tr_sents, rng.randint(2, 5))
        p_ids = tok.encode(prompt)
        a_ids = tok.encode(ans)
        full = (p_ids + a_ids)[: seq_len + 1]
        n = len(full) - 1
        if n <= 1:
            continue
        x_rows[bi, :n] = full[:-1]
        # Focus supervision on the answer span + last 25% of prompt so cognitive circuits sharpen fast
        ans_start = max(0, len(p_ids) - 1)
        if ans_start < n:
            y_rows[bi, ans_start:n] = full[ans_start + 1 : n + 1]
        else:
            y_rows[bi, :n] = full[1 : n + 1]
    x_t = torch.from_numpy(x_rows).to(device=device, dtype=torch.long)
    y_t = torch.from_numpy(y_rows).to(device=device, dtype=torch.long)
    return x_t, y_t


def pretrain_standalone_utf8_router(
    train_mm: np.memmap,
    device: str,
    steps: int = 800,
    batch_size: int = 64,
    seq_len: int = 512,
    seed: int = 2026,
) -> Dict[str, torch.Tensor]:
    """Pretrains ONE canonical UTF-8 boundary router on raw prose, quantizes to 2-bit trits, and locks it."""
    set_all_seeds(seed)
    cfg = build_calcite_v2_tier_config("S1_5M", topology="hourglass", use_scaled_decoder=False)
    ref = CalciteV2LM(cfg).to(device)
    chunker = ref.byte_chunker
    router_params = [p for m in chunker._router_modules() for p in m.parameters()] + [
        chunker.router_feat_weights
    ]
    opt = torch.optim.AdamW(router_params, lr=3.0e-3, weight_decay=0.01)
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    last_ratio = 1.0
    last_loss = 0.0
    for st in range(steps):
        lr = 4.0e-4 + 0.5 * (3.0e-3 - 4.0e-4) * (1.0 + math.cos(math.pi * st / max(1, steps)))
        for pg in opt.param_groups:
            pg["lr"] = lr
        xb, _ = sample_memmap_batch(train_mm, batch_size, seq_len, rng, device)
        opt.zero_grad(set_to_none=True)
        loss, ratio = chunker.router_forward_only(xb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router_params, 1.0)
        opt.step()
        last_ratio = ratio
        last_loss = float(loss.item())

    chunker.lock_router(True)
    xb_v, _ = sample_memmap_batch(train_mm, 16, seq_len, rng, device)
    with torch.no_grad():
        _, _, ctx = chunker.encode_to_chunks(ref.tok_emb(xb_v), xb_v)
        raw_b = (xb_v - 3).clamp(min=0)
        is_cont = (xb_v >= 3) & ((raw_b & 0xC0) == 0x80)
        mid_cuts = int((ctx.b_mask & is_cont).sum().item())
    assert mid_cuts == 0, f"Pretrained router violated UTF-8 safety: {mid_cuts} cuts!"

    elapsed = time.perf_counter() - t0
    print(
        f"[router-pretrain] {steps} steps in {elapsed:.1f}s | loss={last_loss:.4f} | "
        f"chunk_ratio={last_ratio:.2f} B/ch | mid_utf8_cuts={mid_cuts}",
        flush=True,
    )
    r_state = {}
    for k, v in chunker.state_dict().items():
        if k.startswith("router_"):
            r_state[k] = v.detach().cpu().clone()
    return r_state


def load_frozen_router_into(model: CalciteV2LM, router_state: Dict[str, torch.Tensor]) -> None:
    cur = model.byte_chunker.state_dict()
    for k, v in router_state.items():
        cur[k] = v.to(device=cur[k].device, dtype=cur[k].dtype)
    model.byte_chunker.load_state_dict(cur, strict=True)
    model.byte_chunker.lock_router(True)


def compile_v2_swa_blocks(model: nn.Module) -> None:
    """Compiles pure SWA blocks cleanly outside `calcite_v2/` on Linux CUDA (skips Windows without Triton)."""
    if not torch.cuda.is_available() or sys.platform == "win32":
        return
    if isinstance(model, CalciteV2LM):
        if model.swa_stages is not None:
            for stg in model.swa_stages:
                for i in range(len(stg)):
                    stg[i] = torch.compile(stg[i], dynamic=True)
        else:
            for i in range(len(model.cortex_pre)):
                model.cortex_pre[i] = torch.compile(model.cortex_pre[i], dynamic=True)
            for i in range(len(model.cortex_post)):
                model.cortex_post[i] = torch.compile(model.cortex_post[i], dynamic=True)


def build_optimizer_for_model(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    device: str,
) -> torch.optim.Optimizer:
    """Builds HP-transfer-aware AdamW keeping woodbury weight_decay=0.0 and fused CUDA hook."""
    decay_p, no_decay_p, wood_p = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "woodbury" in name:
            wood_p.append(p)
        elif p.ndim < 2 or "norm" in name or "bias" in name or "emb" in name:
            no_decay_p.append(p)
        else:
            decay_p.append(p)

    groups = []
    if decay_p:
        groups.append({"params": decay_p, "lr": base_lr, "weight_decay": weight_decay, "lr_mult": 1.0})
    if no_decay_p:
        groups.append({"params": no_decay_p, "lr": base_lr, "weight_decay": 0.0, "lr_mult": 1.0})
    if wood_p:
        groups.append({"params": wood_p, "lr": base_lr * 0.35, "weight_decay": 0.0, "lr_mult": 0.35})

    use_fused = device == "cuda" and torch.cuda.is_available()
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    opt.register_step_post_hook(lambda *_a, **_kw: clear_fast_qat_cache())
    return opt


def train_byte_lm(
    model: nn.Module,
    train_mm: np.memmap,
    val_mm: np.memmap,
    tok: ByteBPETokenizer,
    tr_sents: List[str],
    device: str,
    total_steps: int,
    batch_size: int = 64,
    seq_len: int = 512,
    micro_bs: Optional[int] = None,
    base_lr: float = 1.6e-3,
    min_lr: float = 1.0e-4,
    weight_decay: float = 0.04,
    cog_prob: float = 0.22,
    seed: int = 2026,
    eval_checkpoints: Optional[List[int]] = None,
    use_compile: bool = True,
    keep_crystal_locked: bool = False,
    tag: str = "run",
) -> Dict[str, object]:
    """Unified clean training loop with optional D-band checkpoint logging."""
    set_all_seeds(seed)
    model = model.to(device)
    if use_compile and device == "cuda":
        compile_v2_swa_blocks(model)

    opt = build_optimizer_for_model(model, base_lr=base_lr, weight_decay=weight_decay, device=device)
    rng_np = np.random.default_rng(seed)
    rng_py = random.Random(seed + 17)

    mbs = int(micro_bs or batch_size)
    accum = max(1, batch_size // mbs)
    is_calcite = isinstance(model, CalciteV2LM)
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    use_amp = device == "cuda" and amp_dtype == torch.bfloat16

    ckpt_set = set(eval_checkpoints or [])
    d_curve: List[Dict[str, float]] = []
    t0 = time.perf_counter()
    warmup = max(20, int(0.05 * total_steps))
    bytes_per_step = batch_size * seq_len

    for step in range(1, total_steps + 1):
        model.train()
        if is_calcite:
            if keep_crystal_locked:
                model.set_crystal_schedule(1, 1)
                model.set_snap_alpha(1.0)
            else:
                model.set_crystal_schedule(step, total_steps)

        if step <= warmup:
            scale = float(step) / float(warmup)
        else:
            prog = float(step - warmup) / float(max(1, total_steps - warmup))
            scale = (min_lr / base_lr) + 0.5 * (1.0 - min_lr / base_lr) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = base_lr * float(pg.get("lr_mult", 1.0)) * scale

        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(accum):
            if cog_prob > 0.0 and rng_py.random() < cog_prob:
                xb, yb = sample_cognitive_batch(tok, tr_sents, mbs, seq_len, rng_py, device)
            else:
                xb, yb = sample_memmap_batch(train_mm, mbs, seq_len, rng_np, device)

            with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                logits = model(xb)
                ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1), ignore_index=-100)
                loss = ce + (model.get_commitment_loss() if is_calcite else 0.0)
            (loss / float(accum)).backward()
            step_loss += float(ce.item()) / float(accum)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step in ckpt_set:
            model.eval()
            val_bpb = eval_val_memmap_bpb(model, val_mm, seq_len=min(512, seq_len), n_batches=16, batch_size=min(16, mbs), device=device)
            d_curve.append(
                {
                    "step": int(step),
                    "bytes_seen": int(step * bytes_per_step),
                    "val_bpb": float(val_bpb),
                    "elapsed_s": round(time.perf_counter() - t0, 1),
                }
            )
            print(
                f"  [{tag}] ckpt step={step}/{total_steps} | D={step*bytes_per_step/1e6:.1f}M B | "
                f"val_bpb={val_bpb:.4f} | {time.perf_counter()-t0:.1f}s",
                flush=True,
            )
        elif step % max(1, total_steps // 5) == 0 or step == total_steps:
            elapsed = time.perf_counter() - t0
            ms = 1000.0 * elapsed / step
            print(
                f"  [{tag}] step={step}/{total_steps} | ce={step_loss:.4f} | {ms:.1f} ms/step | {elapsed:.1f}s",
                flush=True,
            )

    sync_if_cuda(device)
    total_s = time.perf_counter() - t0
    return {
        "total_steps": int(total_steps),
        "bytes_seen": int(total_steps * bytes_per_step),
        "train_seconds": round(total_s, 2),
        "ms_per_step": round(1000.0 * total_s / max(1, total_steps), 2),
        "d_curve": d_curve,
    }


def run_qat_and_pack_lut(
    model_fp: CalciteV2LM,
    train_mm: np.memmap,
    tok: ByteBPETokenizer,
    tr_sents: List[str],
    device: str,
    qat_steps: int = 500,
    batch_size: int = 64,
    seq_len: int = 512,
    micro_bs: Optional[int] = None,
    seed: int = 2026,
) -> CalciteV2LM:
    """STE QAT + Logit-KL distillation -> packed `2-bit uint8 T-MAC LUT` model."""
    sd_clean = unwrap_compiled_state_dict(model_fp)
    teacher = CalciteV2LM(model_fp.cfg).to(device)
    teacher.byte_chunker.lock_router(True)
    teacher.load_state_dict(sd_clean, strict=True)
    teacher.set_crystal_schedule(1, 1)
    teacher.set_snap_alpha(1.0)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = CalciteV2LM(model_fp.cfg).to(device)
    student.byte_chunker.lock_router(True)
    student.load_state_dict(sd_clean, strict=True)
    student.set_crystal_schedule(1, 1)
    student.set_snap_alpha(1.0)
    student.set_qat_mix(0.0)
    student.set_ternary_qat(True)

    opt = build_optimizer_for_model(student, base_lr=3.5e-4, weight_decay=0.0, device=device)
    rng_np = np.random.default_rng(seed + 77)
    rng_py = random.Random(seed + 88)
    mbs = int(micro_bs or batch_size)
    accum = max(1, batch_size // mbs)
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    use_amp = device == "cuda" and amp_dtype == torch.bfloat16

    t0 = time.perf_counter()
    for st in range(1, qat_steps + 1):
        mix = min(1.0, float(st) / float(max(1, int(0.40 * qat_steps))))
        student.set_qat_mix(mix)
        clear_fast_qat_cache()
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            if rng_py.random() < 0.25:
                xb, yb = sample_cognitive_batch(tok, tr_sents, mbs, seq_len, rng_py, device)
            else:
                xb, yb = sample_memmap_batch(train_mm, mbs, seq_len, rng_np, device)
            with torch.no_grad(), torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                t_logits = teacher(xb)
            with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                s_logits = student(xb)
                ce = F.cross_entropy(s_logits.reshape(-1, s_logits.size(-1)), yb.reshape(-1), ignore_index=-100)
                kl = F.kl_div(
                    F.log_softmax(s_logits.float(), dim=-1),
                    F.softmax(t_logits.float(), dim=-1),
                    reduction="batchmean",
                ) / float(seq_len)
                loss = 0.65 * ce + 0.35 * kl + student.get_commitment_loss()
            (loss / float(accum)).backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        if st % max(1, qat_steps // 4) == 0 or st == qat_steps:
            print(f"  [QAT] step={st}/{qat_steps} mix={mix:.2f} | {time.perf_counter()-t0:.1f}s", flush=True)

    student.set_ternary_qat(False)
    student.eval()
    lut_model = pack_fast_model_2bit_lut(student, backend="auto").to(device).eval()
    return lut_model


@torch.no_grad()
def eval_long_context_stress(
    model: nn.Module,
    val_mm: np.memmap,
    device: str,
    lengths: Tuple[int, ...] = (2048, 8192, 32768),
) -> Dict[str, object]:
    """Evaluates long-context BPB & head-vs-tail drift up to `L = 32,768` bytes."""
    model.eval()
    rng = np.random.default_rng(4242)
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    use_amp = device == "cuda" and amp_dtype == torch.bfloat16
    out: Dict[str, object] = {}
    for L in lengths:
        if L + 8 >= len(val_mm):
            continue
        try:
            bpb_all, bpb_head, bpb_tail = [], [], []
            n_seq = 4 if L <= 8192 else 2
            for _ in range(n_seq):
                xb, yb = sample_memmap_batch(val_mm, 1, L, rng, device)
                with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                    logits = model(xb)
                per_tok = F.cross_entropy(
                    logits.float().reshape(-1, logits.size(-1)),
                    yb.reshape(-1),
                    reduction="none",
                ) / math.log(2.0)
                span = min(512, L // 2)
                bpb_all.append(float(per_tok.mean().item()))
                bpb_head.append(float(per_tok[:span].mean().item()))
                bpb_tail.append(float(per_tok[-span:].mean().item()))
            out[f"L_{L}"] = {
                "bpb": round(float(np.mean(bpb_all)), 4),
                "head_bpb": round(float(np.mean(bpb_head)), 4),
                "tail_bpb": round(float(np.mean(bpb_tail)), 4),
                "tail_minus_head": round(float(np.mean(bpb_tail) - np.mean(bpb_head)), 4),
            }
        except RuntimeError as exc:
            out[f"L_{L}"] = {"error": str(exc)[:180]}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return out


@torch.no_grad()
def benchmark_stream_engine_live(
    model_lut: CalciteV2LM,
    device: str,
    n_bytes: int = 256,
) -> Dict[str, object]:
    """Benchmarks live `CalciteStreamEngine` throughput, parity, and exact O(1) RAM."""
    model_lut.eval()
    eng = CalciteStreamEngine(model_lut)
    prompt = "Calcite v2 post-transformer streaming check: Привет, мир! 東京 🚀 2026 "
    p_ids = torch.tensor([[b + 3 for b in prompt.encode("utf-8")]], device=device, dtype=torch.long)

    lg_stream, st = eng.prefill(p_ids, return_all_logits=True)
    # When Latent Feedback is active, streaming computes the exact causal fixed point (`M` chunks);
    # verify batch fixed-point convergence using `feedback_passes = st.backbone_ticks`.
    fb_p = max(2, st.backbone_ticks) if model_lut.cfg.use_latent_feedback else None
    lg_batch = model_lut(p_ids, feedback_passes=fb_p)
    max_err = float((lg_batch.float() - lg_stream.float()).abs().max().item())
    mem_prompt = st.memory_bytes()

    sync_if_cuda(device)
    t0 = time.perf_counter()
    gen_ids, gen_st = eng.generate(p_ids, max_new_bytes=n_bytes, temperature=0.8, top_p=0.9, eos_token_id=None)
    sync_if_cuda(device)
    dt = time.perf_counter() - t0
    mem_end = gen_st.memory_bytes()

    raw_bytes = bytes([int(v) - 3 for v in gen_ids[0].tolist() if 3 <= int(v) <= 258])
    sample_text = raw_bytes.decode("utf-8", errors="replace")[:260]
    return {
        "device": device,
        "stream_vs_batch_max_err": round(max_err, 6),
        "state_bytes_prompt": int(mem_prompt),
        "state_bytes_after_gen": int(mem_end),
        "strict_o1_ram_invariant": bool(mem_prompt == mem_end),
        "state_kb": round(mem_end / 1024.0, 2),
        "generated_bytes": int(n_bytes),
        "backbone_ticks": int(gen_st.backbone_ticks),
        "total_bytes_processed": int(gen_st.total_bytes),
        "effective_chunk_ratio": round(gen_st.total_bytes / max(1, gen_st.backbone_ticks), 2),
        "bytes_per_sec": round(n_bytes / max(1e-6, dt), 1),
        "ms_per_byte": round(1000.0 * dt / max(1, n_bytes), 3),
        "sample_preview": sample_text,
    }


def fit_skaling_vs_chinchilla(points: List[Dict[str, float]]) -> Dict[str, object]:
    """
    Fits coupled Skaling law `L(N, D) = (A / N^a + B / D^b)^k + E` (arXiv:2608.07222v1)
    vs additive Chinchilla (`k = 1`) on `(N, D, val_bpb)` points using SciPy least_squares.
    """
    from scipy.optimize import least_squares

    N_arr = np.array([p["N"] for p in points], dtype=np.float64)
    D_arr = np.array([p["D"] for p in points], dtype=np.float64)
    L_arr = np.array([p["val_bpb"] for p in points], dtype=np.float64)
    N0, D0 = float(np.median(N_arr)), float(np.median(D_arr))
    n_s = N_arr / N0
    d_s = D_arr / D0

    def _fit(coupled: bool) -> Dict[str, object]:
        def _res(theta):
            if coupled:
                la, lb, a, b, k, e = theta
            else:
                la, lb, a, b, e = theta
                k = 1.0
            pred = np.power(np.exp(la) * np.power(n_s, -a) + np.exp(lb) * np.power(d_s, -b), k) + e
            return pred - L_arr

        e_ub = max(0.08, float(L_arr.min()) - 0.02)
        e_0 = max(0.06, min(e_ub - 0.01, float(L_arr.min()) - 0.4))
        if coupled:
            x0 = [0.0, 0.0, 0.30, 0.30, 0.75, e_0]
            lb = [-8.0, -8.0, 0.02, 0.02, 0.15, 0.05]
            ub = [8.0, 8.0, 2.0, 2.0, 3.0, e_ub]
        else:
            x0 = [0.0, 0.0, 0.30, 0.30, e_0]
            lb = [-8.0, -8.0, 0.02, 0.02, 0.05]
            ub = [8.0, 8.0, 2.0, 2.0, e_ub]

        sol = least_squares(_res, x0, bounds=(lb, ub), max_nfev=4000)
        preds = L_arr + sol.fun
        mape = float(np.mean(np.abs(sol.fun) / np.maximum(1e-6, L_arr)) * 100.0)
        rmse = float(np.sqrt(np.mean(sol.fun ** 2)))
        if coupled:
            la, lb_v, a, b, k, e = sol.x
        else:
            la, lb_v, a, b, e = sol.x
            k = 1.0
        pred_85m_500m = float(
            (math.exp(la) * ((82e6 / N0) ** (-a)) + math.exp(lb_v) * ((500e6 / D0) ** (-b))) ** k + e
        )
        return {
            "A_scaled": round(float(math.exp(la)), 5),
            "B_scaled": round(float(math.exp(lb_v)), 5),
            "alpha": round(float(a), 4),
            "beta": round(float(b), 4),
            "k_coupling": round(float(k), 4),
            "E_irred": round(float(e), 4),
            "mape_pct": round(mape, 4),
            "rmse_bpb": round(rmse, 5),
            "pred_85m_500MB_bpb": round(pred_85m_500m, 4),
            "preds": [round(float(x), 4) for x in preds],
        }

    chinch = _fit(coupled=False)
    skal = _fit(coupled=True)
    return {
        "n_points": len(points),
        "N0_median": N0,
        "D0_median": D0,
        "chinchilla_k1": chinch,
        "skaling_coupled": skal,
        "mape_gain_x": round(chinch["mape_pct"] / max(1e-6, skal["mape_pct"]), 2),
    }


@torch.no_grad()
def eval_cognitive_v2(
    model: nn.Module,
    tok: ByteBPETokenizer,
    s5_suites: Dict[str, list],
    ov_suites: Dict[str, list],
    trap_suites: Dict[str, list],
    device: str,
    batch_size: int = 32,
) -> Dict[str, object]:
    """Evaluates unseen `S_5`, `Overwrite`, `Trap`, `Selective F1`, and `L=2048` crystal norm drift."""
    model.eval()
    if hasattr(model, "set_snap_alpha"):
        model.set_snap_alpha(1.0)

    def _eval_group(suites: Dict[str, list], grp_name: str = "") -> Dict[str, Dict[str, object]]:
        out: Dict[str, Dict[str, object]] = {}
        for k, suite in suites.items():
            t_s = time.perf_counter()
            ex_pct, tk_pct, k_ok, n_tot = evaluate_suite_batched(
                model, suite, pad_id=tok.PAD_ID, batch_size=batch_size, device=device
            )
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                f"      [cog] {grp_name}.{k}: {ex_pct:.1f}% ({k_ok}/{n_tot}) in {time.perf_counter()-t_s:.1f}s",
                flush=True,
            )
            lo, hi = wilson_ci_95(k_ok, n_tot)
            out[k] = {
                "exact_pct": round(ex_pct, 2),
                "token_pct": round(tk_pct, 2),
                "ok": int(k_ok),
                "n": int(n_tot),
                "wilson_95": [round(lo, 2), round(hi, 2)],
            }
        return out

    s5_res = _eval_group(s5_suites, "s5")
    ov_res = _eval_group(ov_suites, "ov")
    trap_res = _eval_group(trap_suites, "trap")
    s5_sum = round(sum(float(s5_res[k]["exact_pct"]) for k in ("k2", "k4", "k8", "k16")), 2)
    sel_in = round(
        harmonic_mean(float(ov_res["in_dist"]["exact_pct"]), float(trap_res["in_dist"]["exact_pct"])),
        2,
    )
    sel_long = round(
        harmonic_mean(float(ov_res["ood_long"]["exact_pct"]), float(trap_res["ood_long"]["exact_pct"])),
        2,
    )
    out: Dict[str, object] = {
        "s5": s5_res,
        "s5_sum": s5_sum,
        "overwrite": ov_res,
        "trap": trap_res,
        "selective_f1_in": sel_in,
        "selective_f1_ood_long": sel_long,
    }
    if isinstance(model, CalciteV2LM):
        xb = torch.randint(3, tok.vocab_size, (2, 2048), device=device)
        e_tok = model.tok_emb(xb)
        z_in, h_lex, _ = model.byte_chunker.encode_to_chunks(e_tok, xb)
        c_in = model.cryst1_norm(z_in) if model.cryst1_norm is not None else model.split_norm(z_in)
        s_read, _, _, _ = model.crystal.forward_states(c_in.float(), h_lex=h_lex.float())
        target_r = math.sqrt(float(model.crystal.d_s))
        norm_drift = float((s_read.float().norm(dim=-1) - target_r).abs().max().item())
        out["crystal_norm_drift_L2048"] = round(norm_drift, 8)
    return out


def evaluate_full_model(
    model: nn.Module,
    val_mm: np.memmap,
    eval_wins: Dict[str, List[List[int]]],
    tok: ByteBPETokenizer,
    s5_suites: Dict[str, list],
    ov_suites: Dict[str, list],
    trap_suites: Dict[str, list],
    device: str,
    quick: bool = False,
    eval_bs: int = 16,
    cog_bs: Optional[int] = None,
) -> Dict[str, object]:
    """Runs the unified Phase K evaluation battery on any raw-byte model."""
    model.eval()
    bs = int(max(2, eval_bs))
    c_bs = int(max(1, cog_bs if cog_bs is not None else bs))
    val_bpb = eval_val_memmap_bpb(
        model, val_mm, seq_len=512, n_batches=12 if quick else 32, batch_size=bs, device=device
    )
    print(f"    [eval] val_bpb={val_bpb:.4f}", flush=True)
    ext = eval_length_extrapolation_bpb(
        model, val_mm, device, l_in=512, l_ood=2048, n_batches=6 if quick else 16, batch_size_ood=min(4, bs)
    )
    print(f"    [eval] L2048={ext['bpb_L2048']:.4f} (t-h={ext['tail_minus_head_bpb']:+.4f})", flush=True)
    ow_wins = {k: v[:8] for k, v in eval_wins.items()} if quick else eval_wins
    ow = eval_openweb_6domains_any_byte_lm(model, ow_wins, device=device, batch_size=bs)
    print(f"    [eval] OW6={ow['mean_openweb_bpb']:.4f}", flush=True)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    cog = eval_cognitive_v2(
        model, tok, s5_suites, ov_suites, trap_suites, device=device, batch_size=c_bs
    )
    print(
        f"    [eval] S5={cog['s5_sum']:.1f}% | OV_L={cog['overwrite']['ood_long']['exact_pct']:.1f}% | "
        f"Trap_L={cog['trap']['ood_long']['exact_pct']:.1f}% | SelF1={cog['selective_f1_ood_long']:.1f}%",
        flush=True,
    )
    return {
        "val_bpb": float(val_bpb),
        "extrap_4x": ext,
        "openweb_6": ow,
        "cognitive": cog,
    }


def run_smoke_local(device: str) -> None:
    """Fast local smoke tracer verifying EVERY function end-to-end before Colab GPU allocation."""
    print(f"[smoke-local] Running on {device}...", flush=True)
    tok = ByteBPETokenizer(merges=[])
    tr_sents, eval_wins, s5_s, ov_s, tr_s = build_eval_bundles(ROOT_DIR, tok, n_eval=4, seq_len=512)
    rng = np.random.default_rng(2026)
    fake_mm = rng.integers(3, 128, size=100_000, dtype=np.uint16)

    r_state = pretrain_standalone_utf8_router(fake_mm, device=device, steps=4, batch_size=4, seq_len=96)
    cfg = build_calcite_v2_tier_config(
        "S1_5M", topology="interleaved", use_scaled_decoder=True, use_kalman_precond=True, use_latent_feedback=True
    )
    cfg.d_model = 128
    cfg.n_heads = 2
    cfg.d_ff = 256
    cfg.byte_d_ff = 128
    model = CalciteV2LM(cfg).to(device)
    load_frozen_router_into(model, r_state)

    stats = train_byte_lm(
        model, fake_mm, fake_mm, tok, tr_sents, device=device, total_steps=4, batch_size=4, seq_len=96,
        cog_prob=0.5, eval_checkpoints=[2, 4], use_compile=False, tag="smoke_v2"
    )
    ev = evaluate_full_model(model, fake_mm, eval_wins, tok, s5_s, ov_s, tr_s, device=device, quick=True)
    lc = eval_long_context_stress(model, fake_mm, device=device, lengths=(512, 2048))
    lut_m = run_qat_and_pack_lut(model, fake_mm, tok, tr_sents, device=device, qat_steps=4, batch_size=4, seq_len=96)
    st_bench = benchmark_stream_engine_live(lut_m, device=device, n_bytes=24)
    assert st_bench["strict_o1_ram_invariant"], "O(1) RAM invariant broken!"
    assert st_bench["stream_vs_batch_max_err"] < 1e-2, f"Stream parity err too high: {st_bench['stream_vs_batch_max_err']}"

    # Verify Skaling fit math
    fake_pts = [
        {"N": 5e6, "D": 10e6, "val_bpb": 2.90},
        {"N": 5e6, "D": 40e6, "val_bpb": 2.45},
        {"N": 5e6, "D": 100e6, "val_bpb": 2.28},
        {"N": 25e6, "D": 25e6, "val_bpb": 2.35},
        {"N": 85e6, "D": 25e6, "val_bpb": 2.22},
        {"N": 25e6, "D": 100e6, "val_bpb": 2.02},
    ]
    sf = fit_skaling_vs_chinchilla(fake_pts)
    assert sf["skaling_coupled"]["mape_pct"] <= sf["chinchilla_k1"]["mape_pct"] + 0.05

    # Verify RoPE matched baseline builder
    r_m = build_matched_rope_for_tier("S1_5M", 5_000_000)
    assert sum(p.numel() for p in r_m.parameters()) > 4_000_000

    print(
        f"[smoke-local] ALL CHECKS PASSED | val_bpb={ev['val_bpb']} | L2048={lc['L_2048']['bpb']} | "
        f"stream={st_bench['bytes_per_sec']} B/s | err={st_bench['stream_vs_batch_max_err']} | "
        f"RAM={st_bench['state_kb']} KB | skaling_k={sf['skaling_coupled']['k_coupling']}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["ablations", "skaling", "champion", "all"], default="all")
    ap.add_argument("--data-dir", type=str, default="/content/phase_k_data")
    ap.add_argument("--out-dir", type=str, default="benchmarks/phase_l_v2")
    ap.add_argument("--ckpt-dir", type=str, default="checkpoints")
    ap.add_argument("--ablation-steps", type=int, default=1600)
    ap.add_argument("--d-band-steps", type=int, default=3200)
    ap.add_argument("--n-band-steps", type=int, default=800)
    ap.add_argument("--champion-steps", type=int, default=4000)
    ap.add_argument("--qat-steps", type=int, default=500)
    ap.add_argument("--smoke-local", action="store_true")
    args = ap.parse_args()

    device = "cuda" if has_usable_cuda() else "cpu"
    if args.smoke_local:
        run_smoke_local(device)
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_bin, val_bin = ensure_phase_k_corpus(Path(args.data_dir))
    train_mm = np.memmap(train_bin, dtype=np.uint16, mode="r")
    val_mm = np.memmap(val_bin, dtype=np.uint16, mode="r")
    tok = ByteBPETokenizer(merges=[])
    tr_sents, eval_wins, s5_quick, ov_quick, trap_quick = build_eval_bundles(ROOT_DIR, tok, n_eval=60, seq_len=512)
    _, _, s5_full, ov_full, trap_full = build_eval_bundles(ROOT_DIR, tok, n_eval=150, seq_len=512)

    router_path = out_dir / "frozen_utf8_router.pt"
    if router_path.is_file():
        router_state = torch.load(router_path, map_location="cpu", weights_only=False)
        print(f"[router] Reusing frozen UTF-8 router from {router_path}", flush=True)
    else:
        router_state = pretrain_standalone_utf8_router(train_mm, device=device, steps=800, batch_size=64, seq_len=512)
        torch.save(router_state, router_path)

    abl_path = out_dir / "stage1_5m_ablations.json"
    winner_cfg_path = out_dir / "winning_v2_axes.json"

    # ------------------------------------------------------------------
    # STAGE 1: Single-Axis 5M Ablation Ladder (+ 2*sigma noise floor)
    # ------------------------------------------------------------------
    if args.stage in ("ablations", "all"):
        abl_results: Dict[str, object] = {}
        if abl_path.is_file():
            abl_results = json.loads(abl_path.read_text(encoding="utf-8"))

        ladder = [
            ("A0_v1_seed2026", dict(topology="hourglass", use_scaled_decoder=False, use_kalman_precond=False, use_latent_feedback=False, use_clean_gates=False), 2026, args.ablation_steps),
            ("A0_v1_seed2027", dict(topology="hourglass", use_scaled_decoder=False, use_kalman_precond=False, use_latent_feedback=False, use_clean_gates=False), 2027, args.ablation_steps),
            ("A1_scaled_decoder", dict(topology="hourglass", use_scaled_decoder=True, use_kalman_precond=False, use_latent_feedback=False, use_clean_gates=False), 2026, args.ablation_steps),
            ("A2_kalman_woodbury", dict(topology="hourglass", use_scaled_decoder=False, use_kalman_precond=True, use_latent_feedback=False, use_clean_gates=False), 2026, args.ablation_steps),
            ("A3_interleaved_clean", dict(topology="interleaved", use_scaled_decoder=False, use_kalman_precond=False, use_latent_feedback=False, use_clean_gates=True), 2026, args.ablation_steps),
            ("A4_latent_feedback", dict(topology="hourglass", use_scaled_decoder=False, use_kalman_precond=False, use_latent_feedback=True, use_clean_gates=False), 2026, int(round(args.ablation_steps * 0.75))),
        ]

        ref_5m_params = 5_000_000
        for name, kw, seed, n_steps in ladder:
            if name in abl_results:
                print(f"[ablation] Skipping cached {name}", flush=True)
                continue
            print(f"\n=== [Stage 1 Ablation] {name} ({n_steps} steps, seed={seed}) ===", flush=True)
            cfg = build_calcite_v2_tier_config("S1_5M", **kw)
            m = CalciteV2LM(cfg).to(device)
            load_frozen_router_into(m, router_state)
            n_params = int(sum(p.numel() for p in m.parameters()))
            if name == "A0_v1_seed2026":
                ref_5m_params = n_params
            tr = train_byte_lm(
                m, train_mm, val_mm, tok, tr_sents, device=device, total_steps=n_steps,
                batch_size=64, seq_len=512, micro_bs=32, base_lr=1.8e-3, weight_decay=0.04,
                cog_prob=0.22, seed=seed, use_compile=True, tag=name,
            )
            ev = evaluate_full_model(m, val_mm, eval_wins, tok, s5_quick, ov_quick, trap_quick, device=device, quick=True)
            abl_results[name] = {
                "axes": kw,
                "params": n_params,
                "cache_kb_L2048": round(compute_v2_state_cache_kb(cfg, 2048), 2),
                "train": tr,
                "eval": ev,
            }
            abl_path.write_text(json.dumps(abl_results, indent=2), encoding="utf-8")
            del m
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if "RoPE_5M_ref" not in abl_results:
            print(f"\n=== [Stage 1 Baseline] RoPE_5M_ref ({args.ablation_steps} steps) ===", flush=True)
            rope_5m = build_matched_rope_for_tier("S1_5M", ref_5m_params).to(device)
            n_p = int(sum(p.numel() for p in rope_5m.parameters()))
            tr = train_byte_lm(
                rope_5m, train_mm, val_mm, tok, tr_sents, device=device, total_steps=args.ablation_steps,
                batch_size=64, seq_len=512, micro_bs=32, base_lr=1.8e-3, weight_decay=0.04,
                cog_prob=0.22, seed=2026, use_compile=False, tag="RoPE_5M",
            )
            ev = evaluate_full_model(rope_5m, val_mm, eval_wins, tok, s5_quick, ov_quick, trap_quick, device=device, quick=True)
            abl_results["RoPE_5M_ref"] = {
                "params": n_p,
                "cache_kb_L2048": round(compute_rope_cache_kb(rope_5m.cfg, 2048), 2),
                "train": tr,
                "eval": ev,
            }
            abl_path.write_text(json.dumps(abl_results, indent=2), encoding="utf-8")
            del rope_5m
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        a0_1 = abl_results["A0_v1_seed2026"]["eval"]
        a0_2 = abl_results["A0_v1_seed2027"]["eval"]
        bpb_base = 0.5 * (a0_1["val_bpb"] + a0_2["val_bpb"])
        sigma_bpb = abs(a0_1["val_bpb"] - a0_2["val_bpb"]) / math.sqrt(2.0)
        tol_bpb = max(0.015, 2.0 * sigma_bpb)
        ov_base = 0.5 * (
            a0_1["cognitive"]["overwrite"]["ood_long"]["exact_pct"]
            + a0_2["cognitive"]["overwrite"]["ood_long"]["exact_pct"]
        )
        s5_base = 0.5 * (a0_1["cognitive"]["s5_sum"] + a0_2["cognitive"]["s5_sum"])

        def _wins(key: str) -> bool:
            ev = abl_results[key]["eval"]
            better_bpb = ev["val_bpb"] < bpb_base - 0.005
            better_cog = (
                (ev["cognitive"]["overwrite"]["ood_long"]["exact_pct"] > ov_base + 3.0)
                or (ev["cognitive"]["s5_sum"] > s5_base + 8.0)
            ) and (ev["val_bpb"] <= bpb_base + tol_bpb)
            return bool(better_bpb or better_cog)

        winning_axes = {
            "use_scaled_decoder": _wins("A1_scaled_decoder"),
            "use_kalman_precond": _wins("A2_kalman_woodbury"),
            "topology": "interleaved" if _wins("A3_interleaved_clean") else "hourglass",
            "use_clean_gates": _wins("A3_interleaved_clean"),
            "use_latent_feedback": _wins("A4_latent_feedback"),
        }
        if not any([
            winning_axes["use_scaled_decoder"],
            winning_axes["use_kalman_precond"],
            winning_axes["topology"] == "interleaved",
            winning_axes["use_latent_feedback"],
        ]):
            best_k = min(
                ["A1_scaled_decoder", "A2_kalman_woodbury", "A3_interleaved_clean"],
                key=lambda k: abl_results[k]["eval"]["val_bpb"],
            )
            winning_axes.update(abl_results[best_k]["axes"])

        winner_cfg_path.write_text(
            json.dumps(
                {
                    "bpb_base": round(bpb_base, 4),
                    "two_sigma_bpb": round(2.0 * sigma_bpb, 4),
                    "ov_base": round(ov_base, 2),
                    "s5_base": round(s5_base, 2),
                    "winning_axes": winning_axes,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n[Stage 1 Verdict] Winning Via-Negativa axes: {winning_axes}", flush=True)

        if "A_best_combo" not in abl_results:
            cfg_best = build_calcite_v2_tier_config("S1_5M", **winning_axes)
            n_st = int(round(args.ablation_steps * 0.75)) if winning_axes["use_latent_feedback"] else args.ablation_steps
            m = CalciteV2LM(cfg_best).to(device)
            load_frozen_router_into(m, router_state)
            tr = train_byte_lm(
                m, train_mm, val_mm, tok, tr_sents, device=device, total_steps=n_st,
                batch_size=64, seq_len=512, micro_bs=32, base_lr=1.8e-3, weight_decay=0.04,
                cog_prob=0.22, seed=2026, use_compile=True, tag="A_best_combo",
            )
            ev = evaluate_full_model(m, val_mm, eval_wins, tok, s5_quick, ov_quick, trap_quick, device=device, quick=True)
            abl_results["A_best_combo"] = {
                "axes": winning_axes,
                "params": int(sum(p.numel() for p in m.parameters())),
                "cache_kb_L2048": round(compute_v2_state_cache_kb(cfg_best, 2048), 2),
                "train": tr,
                "eval": ev,
            }
            abl_path.write_text(json.dumps(abl_results, indent=2), encoding="utf-8")
            del m
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if winner_cfg_path.is_file():
        winning_axes = json.loads(winner_cfg_path.read_text(encoding="utf-8"))["winning_axes"]
    else:
        winning_axes = dict(
            topology="interleaved",
            use_scaled_decoder=True,
            use_kalman_precond=True,
            use_latent_feedback=False,
            use_clean_gates=True,
        )

    # ------------------------------------------------------------------
    # STAGE 2: `Skaling` L-Shape Profiling Grid (`5M D-band` + `5M/25M/85M N-band`)
    # ------------------------------------------------------------------
    skal_path = out_dir / "stage2_skaling_lshape.json"
    if args.stage in ("skaling", "all"):
        skal_data: Dict[str, object] = {}
        if skal_path.is_file():
            skal_data = json.loads(skal_path.read_text(encoding="utf-8"))

        d_max = int(args.d_band_steps)
        d_ckpts = sorted(set([max(100, d_max // 10), max(200, d_max // 5), max(400, (2 * d_max) // 5), max(600, (3 * d_max) // 4), d_max]))
        cal_5m_params = int(sum(p.numel() for p in CalciteV2LM(build_calcite_v2_tier_config("S1_5M", **winning_axes)).parameters()))
        for arch in ("CalciteV2", "RoPE"):
            key = f"D_band_{arch}"
            if key not in skal_data:
                print(f"\n=== [Stage 2 Skaling D-band] {arch} S1_5M ({d_max} steps) ===", flush=True)
                if arch == "CalciteV2":
                    cfg = build_calcite_v2_tier_config("S1_5M", **winning_axes)
                    m = CalciteV2LM(cfg).to(device)
                    load_frozen_router_into(m, router_state)
                else:
                    m = build_matched_rope_for_tier("S1_5M", cal_5m_params).to(device)
                n_p = int(sum(p.numel() for p in m.parameters()))
                tr = train_byte_lm(
                    m, train_mm, val_mm, tok, tr_sents, device=device, total_steps=d_max,
                    batch_size=64, seq_len=512, micro_bs=32, base_lr=1.8e-3, weight_decay=0.04,
                    cog_prob=0.0, seed=2026, eval_checkpoints=d_ckpts,
                    use_compile=(arch == "CalciteV2"), tag=key,
                )
                skal_data[key] = {"params": n_p, "train": tr}
                skal_path.write_text(json.dumps(skal_data, indent=2), encoding="utf-8")
                del m
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        n_st_cfg = int(args.n_band_steps)
        n_specs = [
            ("S2_25M", n_st_cfg, 16, 1.35e-3, 0.05),
            ("S3_85M", n_st_cfg, 8, 1.00e-3, 0.06),
        ]
        for tier, n_st, mbs, lr_t, wd_t in n_specs:
            cal_tier_params = int(sum(p.numel() for p in CalciteV2LM(build_calcite_v2_tier_config(tier, **winning_axes)).parameters()))
            for arch in ("CalciteV2", "RoPE"):
                key = f"N_band_{tier}_{arch}"
                if key in skal_data:
                    continue
                print(f"\n=== [Stage 2 Skaling N-band] {arch} {tier} ({n_st} steps, lr={lr_t}) ===", flush=True)
                if arch == "CalciteV2":
                    cfg = build_calcite_v2_tier_config(tier, **winning_axes)
                    m = CalciteV2LM(cfg).to(device)
                    load_frozen_router_into(m, router_state)
                else:
                    m = build_matched_rope_for_tier(tier, cal_tier_params).to(device)
                n_p = int(sum(p.numel() for p in m.parameters()))
                tr = train_byte_lm(
                    m, train_mm, val_mm, tok, tr_sents, device=device, total_steps=n_st,
                    batch_size=64, seq_len=512, micro_bs=mbs, base_lr=lr_t, weight_decay=wd_t,
                    cog_prob=0.0, seed=2026, eval_checkpoints=[max(100, n_st // 2), n_st],
                    use_compile=(arch == "CalciteV2"), tag=key,
                )
                skal_data[key] = {"params": n_p, "train": tr}
                skal_path.write_text(json.dumps(skal_data, indent=2), encoding="utf-8")
                del m
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        fits = {}
        for arch in ("CalciteV2", "RoPE"):
            pts = []
            d_rec = skal_data[f"D_band_{arch}"]
            for row in d_rec["train"]["d_curve"]:
                pts.append({"N": float(d_rec["params"]), "D": float(row["bytes_seen"]), "val_bpb": float(row["val_bpb"])})
            for tier in ("S2_25M", "S3_85M"):
                n_rec = skal_data[f"N_band_{tier}_{arch}"]
                for row in n_rec["train"]["d_curve"]:
                    pts.append({"N": float(n_rec["params"]), "D": float(row["bytes_seen"]), "val_bpb": float(row["val_bpb"])})
            fits[arch] = fit_skaling_vs_chinchilla(pts)
        skal_data["fits"] = fits
        skal_path.write_text(json.dumps(skal_data, indent=2), encoding="utf-8")
        print(f"\n[Stage 2 Skaling Fits] {json.dumps(fits, indent=2)}", flush=True)

    # ------------------------------------------------------------------
    # STAGE 3: Champion `Calcite v2 (25M)` + `2-bit LUT` + `L = 32K` + Live Stream
    # Equal-compute 2-stage curriculum (Stage 3A cognitive grokking -> Stage 3B language + 18% replay)
    # ------------------------------------------------------------------
    champ_path = out_dir / "stage3_25m_champion.json"
    if args.stage in ("champion", "all"):
        s3a_steps = max(400, int(round(args.champion_steps * 0.375)))
        s3b_steps = max(400, int(args.champion_steps - s3a_steps))
        print(
            f"\n=== [Stage 3 Champion] Calcite v2 S2_25M ({s3a_steps} S3A cog + {s3b_steps} S3B lang + {args.qat_steps} QAT) ===",
            flush=True,
        )
        cfg_25m = build_calcite_v2_tier_config("S2_25M", **winning_axes)
        m_25m = CalciteV2LM(cfg_25m).to(device)
        load_frozen_router_into(m_25m, router_state)
        n_params = int(sum(p.numel() for p in m_25m.parameters()))

        fp_ckpt_path = out_dir / "calcite_v2_s2_25m_fp_tmp.pt"
        if fp_ckpt_path.is_file():
            print(f"[Stage 3] Resuming FP Champion from cached {fp_ckpt_path}", flush=True)
            sd_clean = torch.load(fp_ckpt_path, map_location="cpu", weights_only=False)
            tr_25m = {"resumed_from": str(fp_ckpt_path)}
        else:
            tr_25m_a = train_byte_lm(
                m_25m, train_mm, val_mm, tok, tr_sents, device=device, total_steps=s3a_steps,
                batch_size=64, seq_len=512, micro_bs=16, base_lr=1.5e-3, min_lr=3.0e-4,
                weight_decay=0.03, cog_prob=0.75, seed=2026,
                eval_checkpoints=[s3a_steps], use_compile=True, tag="CalciteV2_25M_S3A",
            )
            tr_25m_b = train_byte_lm(
                m_25m, train_mm, val_mm, tok, tr_sents, device=device, total_steps=s3b_steps,
                batch_size=64, seq_len=512, micro_bs=16, base_lr=9.0e-4, min_lr=5.0e-5,
                weight_decay=0.05, cog_prob=0.18, seed=2027,
                eval_checkpoints=[max(200, s3b_steps // 2), s3b_steps],
                use_compile=True, keep_crystal_locked=True, tag="CalciteV2_25M_S3B",
            )
            tr_25m = {
                "total_steps": s3a_steps + s3b_steps,
                "s3a": tr_25m_a,
                "s3b": tr_25m_b,
                "train_seconds": round(tr_25m_a["train_seconds"] + tr_25m_b["train_seconds"], 2),
            }
            sd_clean = unwrap_compiled_state_dict(m_25m)
            torch.save({k: v.cpu() for k, v in sd_clean.items()}, fp_ckpt_path)
            print(f"[Stage 3] Saved FP checkpoint to {fp_ckpt_path}", flush=True)

        del m_25m
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        fp_eval_m = CalciteV2LM(cfg_25m).to(device)
        fp_eval_m.byte_chunker.lock_router(True)
        fp_eval_m.load_state_dict(sd_clean, strict=True)
        fp_eval_m.set_crystal_schedule(1, 1)
        fp_eval_m.set_snap_alpha(1.0)
        fp_eval_m.eval()

        print("  [Stage 3] Evaluating FP Champion...", flush=True)
        ev_fp = evaluate_full_model(fp_eval_m, val_mm, eval_wins, tok, s5_full, ov_full, trap_full, device=device, quick=False, eval_bs=16)
        long_fp = eval_long_context_stress(fp_eval_m, val_mm, device=device, lengths=(2048, 8192, 32768))
        print(f"  [Stage 3] FP long-context: {long_fp}", flush=True)

        lut_25m = run_qat_and_pack_lut(
            fp_eval_m, train_mm, tok, tr_sents, device=device, qat_steps=args.qat_steps,
            batch_size=64, seq_len=512, micro_bs=16, seed=2026,
        )
        ckpt_out = ckpt_dir / "calcite_v2_s2_25m_utf8_lut.pt"
        lut_foot = save_packed_lut_checkpoint(
            ckpt_out,
            lut_25m,
            cfg_25m,
            tok,
            metadata={"winning_axes": winning_axes, "phase": "Phase_L_Calcite_v2"},
        )
        print(f"  [Stage 3] Saved 2-bit LUT checkpoint to {ckpt_out} ({ckpt_out.stat().st_size:,} B)", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("  [Stage 3] Evaluating 2-bit LUT Champion (bs=8 VRAM-safe)...", flush=True)
        ev_lut = evaluate_full_model(lut_25m, val_mm, eval_wins, tok, s5_full, ov_full, trap_full, device=device, quick=False, eval_bs=8)
        long_lut = eval_long_context_stress(lut_25m, val_mm, device=device, lengths=(2048, 8192, 32768))
        print(f"  [Stage 3] LUT long-context: {long_lut}", flush=True)

        stream_gpu = benchmark_stream_engine_live(lut_25m, device=device, n_bytes=256)
        print(f"  [Stage 3] Stream GPU: {stream_gpu['bytes_per_sec']} B/s | err={stream_gpu['stream_vs_batch_max_err']} | RAM={stream_gpu['state_kb']} KB", flush=True)
        fp_eval_m = fp_eval_m.cpu()
        lut_sd_cpu = {k: v.cpu() for k, v in lut_25m.state_dict().items()}
        del lut_25m
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        lut_cpu = pack_fast_model_2bit_lut(copy.deepcopy(fp_eval_m), backend="auto").eval()
        lut_cpu.load_state_dict(lut_sd_cpu, strict=True)
        stream_cpu = benchmark_stream_engine_live(lut_cpu, device="cpu", n_bytes=128)
        print(f"  [Stage 3] Stream CPU: {stream_cpu['bytes_per_sec']} B/s | err={stream_cpu['stream_vs_batch_max_err']} | RAM={stream_cpu['state_kb']} KB", flush=True)
        del lut_cpu, fp_eval_m, lut_sd_cpu
        gc.collect()

        rope_25m = build_matched_rope_for_tier("S2_25M", n_params).to(device)
        tr_rope_a = train_byte_lm(
            rope_25m, train_mm, val_mm, tok, tr_sents, device=device, total_steps=s3a_steps,
            batch_size=64, seq_len=512, micro_bs=16, base_lr=1.5e-3, min_lr=3.0e-4,
            weight_decay=0.03, cog_prob=0.75, seed=2026, use_compile=False, tag="RoPE_25M_S3A",
        )
        tr_rope_b = train_byte_lm(
            rope_25m, train_mm, val_mm, tok, tr_sents, device=device, total_steps=s3b_steps,
            batch_size=64, seq_len=512, micro_bs=16, base_lr=9.0e-4, min_lr=5.0e-5,
            weight_decay=0.05, cog_prob=0.18, seed=2027, use_compile=False, tag="RoPE_25M_S3B",
        )
        tr_rope = {
            "total_steps": s3a_steps + s3b_steps,
            "s3a": tr_rope_a,
            "s3b": tr_rope_b,
            "train_seconds": round(tr_rope_a["train_seconds"] + tr_rope_b["train_seconds"], 2),
        }
        ev_rope = evaluate_full_model(rope_25m, val_mm, eval_wins, tok, s5_full, ov_full, trap_full, device=device, quick=False)
        long_rope = eval_long_context_stress(rope_25m, val_mm, device=device, lengths=(2048, 8192, 32768))

        summary = {
            "winning_axes": winning_axes,
            "calcite_v2_25m": {
                "params": n_params,
                "config": asdict(cfg_25m),
                "cache_kb_L2048": round(compute_v2_state_cache_kb(cfg_25m, 2048), 2),
                "cache_kb_L32768": round(compute_v2_state_cache_kb(cfg_25m, 32768), 2),
                "lut_footprint": lut_foot,
                "train": tr_25m,
                "eval_fp": ev_fp,
                "eval_lut": ev_lut,
                "long_context_fp": long_fp,
                "long_context_lut": long_lut,
                "stream_gpu": stream_gpu,
                "stream_cpu": stream_cpu,
                "checkpoint_path": str(ckpt_out),
            },
            "rope_25m": {
                "params": int(sum(p.numel() for p in rope_25m.parameters())),
                "cache_kb_L2048": round(compute_rope_cache_kb(rope_25m.cfg, 2048), 2),
                "cache_kb_L32768": round(compute_rope_cache_kb(rope_25m.cfg, 32768), 2),
                "train": tr_rope,
                "eval_fp": ev_rope,
                "long_context_fp": long_rope,
            },
        }
        champ_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n[Stage 3 Champion DONE] Saved to {champ_path}", flush=True)


if __name__ == "__main__":
    main()
