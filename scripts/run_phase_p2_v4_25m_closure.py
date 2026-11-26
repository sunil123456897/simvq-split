"""
Phase P-2: Flagship `Calcite v4 (S2_25M)` Unified Closure + Canonical WSD Plateau Checkpoint
(`scripts/run_phase_p2_v4_25m_closure.py`).

Unifies ALL verified Phase O + Phase P-1 breakthroughs into one `~10 MB` `2-bit uint8 T-MAC LUT`
checkpoint (`checkpoints/calcite_v4_s2_25m_utf8_lut.pt`) plus a reusable `25M` WSD plateau
checkpoint (`checkpoints/phase_p_fp/calcite_v4_s2_25m_wsd_plateau.pt`):
  1. `Calcite v4 (S2_25M)` Architecture:
     - `Calcite v3.1` hourglass core + `dw_bottleneck_ratio=0.5` (Phase O Pareto winner)
     - `use_dca=True` (`DeepCrossAttentionMix`, `+0 B` O(1) RAM, FP-immune to 2-bit LUT)
     - `ByteInductionCache` (`2 x 2048` slots, Scale-1 ultra-sparse hinge `0.80 * relu(mean(beta_1) - 0.018)`)
     - Pre-frozen UTF-8 router (`0.00%` mid-UTF8 cuts, `0.0` mutation verified across all stages)
  2. Strict 4-Stage Training Discipline (enforcing Law 8a, Law 8c, and `auditor-p` #81 guards):
     - **Stage 1A (`2,000` steps — Pure `S_5` Grokking):** `set_bic_active(False)` + `set_dca_active(False)`
       with frozen `BIC` & `DCA` grads (`Woodbury wd=0.0, 0.05x LR`) -> **HARD GATE `S_5 k2 >= 80%`**.
     - **Stage 1B (`500` steps — `Overwrite / Trap` + `S_5` Maint):** unfreeze `DCA` (`0.5x LR`),
       keep `BIC` bypassed, protect `Crystal` (`0.15x LR`) -> check `OV_L >= 85%` & `S_5` preserved.
     - **Stage 2 (`1,800` steps — `WSD` Language + Verbatim Needle + Cog Maint):**
       `WSDLRSchedule(warm=60, stable=1240, decay=500)` saving canonical WSD plateau at `step=1300`,
       unfreeze `BIC` (`2048` slots, `0.018` Scale-1 hinge, `BIC 2.5x`, `DCA 0.4x`, `Crystal 0.05x`,
       `Woodbury 0.12x`), `70%` 6-domain OpenWeb + `12%` multi-format MQAR + `18%` Cog maintenance.
     - **Stage 3 (`1,000` steps — Copy-Distilled `4-Tap QAT` with FROZEN `DCA`):**
       **CRITICAL GUARD #1:** freeze all `dca_mixers` params (`requires_grad=False`, `0.0` diff!)
       so zero OOD Emoji drift occurs while staying exact FP.
       **CRITICAL GUARD #2 (Law 8c):** `bic_q LR = 2.5e-4 * w_scale`, `chunk_q LR = 1.1e-3 * w_scale`,
       prefix-masked MQAR copy rows (~12.5% copy gradient share), `AdamW(fused=True)` cache hook.
  3. Full Closure Evaluation & Disk Roundtrip (`strict=True`):
     - `Val`, `OW6` (6 domains + `0.00%` mid-UTF8 cuts), `L=2K/8K/32K`, `S_5`, `OV_L`, `Trap_L`,
       `SelF1_L`, OOD Needle `512..32K`, `OrigRefCode 1500B`, `OverwriteNeedle 1500B`, live streaming
       Needle, plus `CPU AVX-512F` & `GPU CUDA Graph` (`fuse_packed_swiglu_for_inference`) `B/s`
       (`err < 2e-5`, strict `O(1)` RAM).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
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
    has_usable_cuda,
    load_utf8_corpus_array,
    sample_memmap_batch,
    sync_if_cuda,
)
from kernels.fast_calcite_scans import (  # noqa: E402
    clear_fast_qat_cache,
    compile_calcite_v2,
    unwrap_compiled_state_dict,
)
from kernels.fast_packed_lut import (  # noqa: E402
    FastPackedLUT2BitLinear,
    fuse_packed_swiglu_for_inference,
    get_lut_backend_info,
    pack_fast_model_2bit_lut,
    save_packed_lut_checkpoint,
)
from run_phase_e_fast_utf8_benchmark import build_disjoint_openweb_corpora  # noqa: E402
from scripts.colab_phase_k_scaling_runner import (  # noqa: E402
    StratifiedSystem1Batcher,
    _assert_router_unchanged,
    _attach_4tap_hooks,
    _cosine_lr,
    _fp_micro_ce,
    _make_adamw,
    _run_4tap_qat_step,
    _run_accum_microbatches,
    _snapshot_router_params,
)
from scripts.colab_phase_l_v2_runner import (  # noqa: E402
    build_eval_bundles,
    compile_v2_swa_blocks,
    ensure_phase_k_corpus,
    eval_long_context_stress,
    evaluate_full_model,
    inject_l2048_from_extrap,
    load_frozen_router_into,
    set_all_seeds,
)
from scripts.experiment_velocity import (  # noqa: E402
    WSDLRSchedule,
    save_plateau_checkpoint,
)
from scripts.muon_opt import MuonAdamWHybrid  # noqa: E402
from scripts.run_phase_l_closure_25m import (  # noqa: E402
    _recommended_lut_cog_bs,
    load_calcite_v2_lut_checkpoint,
)
from scripts.run_phase_m_stage_c_v3_5m import _split_v3_qat_4groups  # noqa: E402
from scripts.run_phase_n_closure_25m import _inject_mqar_rows  # noqa: E402
from scripts.run_phase_n_mqar_probe_5m import eval_ood_verbatim_ladder  # noqa: E402
from scripts.run_phase_n_v31_lm_check_5m import (  # noqa: E402
    eval_live_stream_needle,
    eval_overwrite_needle_ood,
)
from scripts.run_phase_o_tier2_ablations_5m import run_gate2_init_probe  # noqa: E402
from scripts.run_phase_p_5m_tournament import _check_utf8_zero_mid_cuts  # noqa: E402
from simvq_split.eval_utils import evaluate_suite_batched  # noqa: E402
from simvq_split.memory import (  # noqa: E402
    compute_2bit_lut_footprint,
    quantize_ternary_weights_ls,
)
from simvq_split.tasks import load_gsm8k_corpora  # noqa: E402
from simvq_split.tokenizer import ByteBPETokenizer  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Config, Module Toggles, Optimizer Groups & Snapshots
# ---------------------------------------------------------------------------


def build_calcite_v4_25m_config(smoke: bool = False, slots: int = 2048) -> CalciteV2Config:
    """Exact `Calcite v4 (S2_25M)` configuration from Phase P-1 audit & Stage 2 spec."""
    tier = "S1_5M" if smoke else "S2_25M"
    cfg = build_calcite_v2_tier_config(
        tier,
        topology="hourglass",
        use_scaled_decoder=True,
        use_kalman_precond=False,
        use_latent_feedback=False,
        use_clean_gates=False,
        window_size=8,
        use_sym_byte_swa=True,
        byte_swa_window=24,
        byte_swa_dim=64,
        dec_dw3_enabled=True,
        dw_bottleneck_ratio=0.5,
        use_byte_engram=True,
        engram_slots=4096,
        engram_dim=32,
        engram_use_4gram=False,
        use_byte_induct_cache=True,
        byte_induct_slots=128 if smoke else int(slots),
        byte_induct_dim=16,
        use_dca=True,
        bf16_residual_stream=False,
    )
    if smoke:
        cfg.d_model, cfg.n_heads, cfg.d_ff = 128, 2, 256
        cfg.byte_char_dim, cfg.byte_d_ff = 16, 128
    cfg.byte_chunk_mode = "utf8"
    cfg.use_static_m_cap = True
    return cfg


def _set_stage_module_modes(
    model: CalciteV2LM,
    *,
    dca_active: bool,
    dca_grad: bool,
    bic_active: bool,
    bic_grad: bool,
) -> None:
    """Explicitly sets runtime bypass flags AND `requires_grad` for `dca_mixers` & `byte_induct_cache`."""
    model.byte_chunker.lock_router(True)
    model.set_dca_active(dca_active)
    model.set_bic_active(bic_active)
    if model.dca_mixers is not None:
        for p in model.dca_mixers.parameters():
            p.requires_grad_(bool(dca_grad))
    bic = model.byte_chunker.byte_induct_cache
    if bic is not None:
        for p in bic.parameters():
            p.requires_grad_(bool(bic_grad))


def _snapshot_dca_params(model: CalciteV2LM) -> Dict[str, torch.Tensor]:
    if model.dca_mixers is None:
        return {}
    return {nm: p.detach().cpu().clone() for nm, p in model.dca_mixers.named_parameters()}


def _assert_dca_unchanged(model: CalciteV2LM, snap: Dict[str, torch.Tensor], tag: str) -> float:
    if model.dca_mixers is None or not snap:
        return 0.0
    max_d = 0.0
    for nm, p in model.dca_mixers.named_parameters():
        if nm in snap:
            diff = float((p.detach().cpu() - snap[nm]).abs().max().item())
            max_d = max(max_d, diff)
            assert diff == 0.0, f"[{tag}] Frozen DCA param `{nm}` mutated diff={diff:.3e}"
    return max_d


def _add_custom_adamw_groups(
    opt: MuonAdamWHybrid,
    model: CalciteV2LM,
    base_lr: float,
    *,
    dca_lr_mult: Optional[float] = None,
    bic_lr_mult: Optional[float] = None,
) -> None:
    """Re-assigns trainable `dca_mixers` and/or `byte_induct_cache` params to dedicated AdamW LR groups."""
    if opt.adamw is None:
        return

    def _extract_and_add(mod: Optional[nn.Module], lr_mult: Optional[float], label: str) -> None:
        if mod is None or lr_mult is None:
            return
        target_ids = {id(p) for p in mod.parameters() if p.requires_grad}
        if not target_ids:
            return
        extracted: List[nn.Parameter] = []
        seen: set[int] = set()
        for pg in opt.adamw.param_groups:
            kept: List[nn.Parameter] = []
            for p in pg["params"]:
                if id(p) in target_ids:
                    if id(p) not in seen:
                        extracted.append(p)
                        seen.add(id(p))
                else:
                    kept.append(p)
            pg["params"] = kept
        if opt.muon_group is not None:
            kept_m: List[nn.Parameter] = []
            for p in opt.muon_group["params"]:
                if id(p) in target_ids:
                    if id(p) not in seen:
                        extracted.append(p)
                        seen.add(id(p))
                else:
                    kept_m.append(p)
            opt.muon_group["params"] = kept_m
        if extracted:
            opt.adamw.add_param_group(
                {
                    "params": extracted,
                    "lr": float(base_lr) * float(lr_mult),
                    "weight_decay": 0.0,
                    "lr_mult": float(lr_mult),
                    "is_muon": False,
                    "group_name": label,
                }
            )

    _extract_and_add(model.dca_mixers, dca_lr_mult, "dca_mixers")
    _extract_and_add(model.byte_chunker.byte_induct_cache, bic_lr_mult, "byte_induct_cache")
    # Drop any emptied AdamW groups so fused AdamW never sees empty lists
    opt.adamw.param_groups[:] = [g for g in opt.adamw.param_groups if len(g["params"]) > 0]
    opt._sync_param_groups()
    opt._rebuild_muon_buckets()


def _split_v4_qat_5groups_frozen_dca(
    model: CalciteV2LM,
) -> Tuple[List[nn.Parameter], List[nn.Parameter], List[nn.Parameter], List[nn.Parameter], List[nn.Parameter]]:
    """
    5-Group QAT partition for `Calcite v4` with FROZEN `dca_mixers` (`auditor-p` #81 Guard #1):
    - Asserts EVERY `dca_mixers` parameter has `requires_grad == False`!
    - Splits `byte_induct_cache` into dedicated gentle group `bic_q` (`2.5e-4 * w_scale`, Law 8c).
    - Guarantees 100% disjoint, exact coverage of all remaining trainable parameters.
    """
    if model.dca_mixers is not None:
        for nm, p in model.dca_mixers.named_parameters():
            assert not p.requires_grad, f"Guard #1 violation: dca_mixers.{nm} must be frozen during QAT!"

    gain_p, cryst_p, chunk_p, core_p = _split_v3_qat_4groups(model)

    bic_mod = model.byte_chunker.byte_induct_cache
    bic_ids = {id(p) for p in bic_mod.parameters() if p.requires_grad} if bic_mod is not None else set()
    bic_p: List[nn.Parameter] = []
    if bic_ids:
        seen_b: set[int] = set()
        for grp in (gain_p, cryst_p, chunk_p, core_p):
            for p in grp:
                if id(p) in bic_ids and id(p) not in seen_b:
                    bic_p.append(p)
                    seen_b.add(id(p))
        gain_p = [p for p in gain_p if id(p) not in bic_ids]
        cryst_p = [p for p in cryst_p if id(p) not in bic_ids]
        chunk_p = [p for p in chunk_p if id(p) not in bic_ids]
        core_p = [p for p in core_p if id(p) not in bic_ids]

    all_p = gain_p + cryst_p + chunk_p + core_p + bic_p
    expected_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assigned_ids = {id(p) for p in all_p}
    assert len(all_p) == len(assigned_ids), "Duplicate parameter across v4 QAT groups!"
    assert assigned_ids == expected_ids, (
        f"v4 QAT group coverage mismatch: assigned={len(assigned_ids)} expected={len(expected_ids)}"
    )
    return gain_p, cryst_p, chunk_p, core_p, bic_p


# ---------------------------------------------------------------------------
# 2. Stage-2 / Stage-3 Balanced Data Sampler (`70% OpenWeb + 12% MQAR + 18% Cog`)
# ---------------------------------------------------------------------------


def sample_v4_s2_or_qat_batch(
    batcher: StratifiedSystem1Batcher,
    train_mm: np.ndarray,
    batch_size: int,
    seq_len: int,
    *,
    cog_prob: float,
    n_copy_web: int,
    rng_mix: random.Random,
    mqar_rng_py: random.Random,
    mqar_rng_np: np.random.Generator,
    mask_mqar_prefix_for_qat: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Exact `Calcite v4` Stage-2 / QAT stream:
    - With prob `cog_prob` (`0.18`): pure cognitive maintenance batch (`60% S_5 + 28% OV + 12% Trap`).
    - With prob `1 - cog_prob` (`0.82`): 6-domain OpenWeb batch (`train_mm` + all 6 `train_wins`
      domains incl. `5_CJK_UTF8` & `6_Emoji_4Byte`) with `n_copy_web` multi-format MQAR rows injected.
      * Stage 2 (`n_copy_web = 9` @ `B=64`): `70.5%` OpenWeb + `11.5%` MQAR + `18.0%` Cog.
      * Stage 3 QAT (`n_copy_web = 8` @ `B=64`, `mask_mqar_prefix_for_qat=True`): prefixes masked to
        `PAD_ID=0` on MQAR rows so per-row CE+KL gives ~12.5% copy distillation gradient (Law 8c).
    """
    pad_id = int(batcher.tok.PAD_ID)
    is_cog = rng_mix.random() < float(cog_prob)

    if is_cog:
        xb_np = np.full((batch_size, seq_len), pad_id, dtype=np.int64)
        yb_np = np.full((batch_size, seq_len), pad_id, dtype=np.int64)
        n_s5 = max(1, int(round(batch_size * 0.60)))
        n_ov = max(0, int(round(batch_size * 0.28)))
        while n_s5 + n_ov > batch_size:
            if n_s5 > 1:
                n_s5 -= 1
            else:
                n_ov -= 1
        kinds = ["s5"] * n_s5 + ["ov"] * n_ov + ["trap"] * (batch_size - n_s5 - n_ov)
        rng_mix.shuffle(kinds)
        for ri, kn in enumerate(kinds):
            f_arr, p_len = batcher._sample_cog_row_np(kn, local_frac=1.0, stage1=False)
            ls = len(f_arr)
            xb_np[ri, : ls - 1] = f_arr[:-1]
            yb_np[ri, p_len - 1 : ls - 1] = f_arr[p_len:]
        perm = mqar_rng_np.permutation(batch_size)
        return torch.from_numpy(xb_np[perm]), torch.from_numpy(yb_np[perm]), None

    # 6-Domain OpenWeb batch (guaranteed coverage of EN, RU, Python, ARC_JSON, CJK, and Emoji!)
    dom_keys = [k for k in batcher.domain_keys if batcher._train_wins_np.get(k)]
    n_dom = (
        len(dom_keys)
        if (dom_keys and batch_size >= 16)
        else (min(len(dom_keys), max(1, batch_size // 4)) if (dom_keys and batch_size >= 4) else 0)
    )
    n_web = max(0, batch_size - n_dom)
    xb_np = np.full((batch_size, seq_len), pad_id, dtype=np.int64)
    yb_np = np.full((batch_size, seq_len), pad_id, dtype=np.int64)
    if n_web > 0:
        xw, yw = sample_memmap_batch(train_mm, n_web, seq_len, batcher.rng_np, "cpu")
        xb_np[:n_web] = xw.numpy()
        yb_np[:n_web] = yw.numpy()
    if n_dom > 0:
        chosen = (
            dom_keys
            if n_dom == len(dom_keys)
            else [batcher.rng_py.choice(dom_keys) for _ in range(n_dom)]
        )
        for di, dk in enumerate(chosen):
            w = batcher.rng_py.choice(batcher._train_wins_np[dk])[: seq_len + 1]
            lw = len(w) - 1
            if lw > 0:
                xb_np[n_web + di, :lw] = w[:-1]
                yb_np[n_web + di, :lw] = w[1:]
    perm = mqar_rng_np.permutation(batch_size)
    xb_t = torch.from_numpy(xb_np[perm])
    yb_t = torch.from_numpy(yb_np[perm])
    xb_t, yb_t, mb_t = _inject_mqar_rows(xb_t, yb_t, train_mm, n_copy_web, mqar_rng_py, mqar_rng_np)
    if mask_mqar_prefix_for_qat:
        is_copy_row = mb_t.any(dim=1, keepdim=True)
        yb_t = torch.where(is_copy_row & (~mb_t), torch.full_like(yb_t, pad_id), yb_t)
    return xb_t, yb_t, mb_t


# ---------------------------------------------------------------------------
# 3. Fused 2-bit LUT Streaming Benchmark (`CPU AVX-512F` & `GPU CUDA Graph`)
# ---------------------------------------------------------------------------


def benchmark_reloaded_lut_streaming(
    lut_ckpt: Path,
    val_mm: np.ndarray,
    device: str,
    smoke: bool = False,
) -> Dict[str, Any]:
    """
    Loads `lut_ckpt` from disk (`strict=True`), applies `fuse_packed_swiglu_for_inference`, and
    measures deterministic `CPU AVX-512F` and `GPU CUDA Graph (use_cuda_graph=True)` streaming
    `B/s`, `ms/B`, `prefill_bps`, `vs_batch_max_err`, `vs_eager_max_err`, and strict `O(1)` RAM.
    """
    prompt = "Calcite v4 post-transformer streaming check: Привет, мир! 東京 🚀 2026 "
    p_cpu = torch.tensor([[b + 3 for b in prompt.encode("utf-8")]], device="cpu", dtype=torch.long)
    n_gen = 32 if smoke else 256
    fixed_stream_cpu = torch.from_numpy(val_mm[:n_gen].astype(np.int64)).unsqueeze(0)

    def _run_bench(mod: CalciteV2LM, p_ids: torch.Tensor, stream_ids: torch.Tensor, use_cg: bool) -> Dict[str, Any]:
        dev = p_ids.device
        eng = CalciteStreamEngine(mod, use_cuda_graph=use_cg)
        # Warmup (includes CUDA Graph capture when use_cg=True)
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

        # Exact batch-vs-stream parity on prompt + 32 generated bytes
        gen_seq, _ = eng.generate(p_ids, max_new_bytes=32, temperature=0.0, eos_token_id=None)
        eng_eager = CalciteStreamEngine(mod, use_cuda_graph=False)
        lg_s, _ = eng_eager.prefill(gen_seq, return_all_logits=True)
        with torch.inference_mode():
            lg_b = mod(gen_seq)
        err_batch = float((lg_b.float() - lg_s.float()).abs().max().item())

        err_cg = 0.0
        if use_cg:
            lg_e0, st_e0 = eng_eager.prefill(p_ids, return_all_logits=False)
            lg_c0, st_c0 = eng.prefill(p_ids, return_all_logits=False)
            err_cg = float((lg_e0.float() - lg_c0.float()).abs().max().item())
            for k in range(32):
                tk = gen_seq[:, p_ids.size(1) + k]
                lg_e0, st_e0 = eng_eager.step(tk, st_e0)
                lg_c0, st_c0 = eng.step(tk, st_c0)
                err_cg = max(err_cg, float((lg_e0.float() - lg_c0.float()).abs().max().item()))

        gen_bytes = [max(0, min(255, int(v) - 3)) for v in gen_seq[0].tolist()]
        preview = bytes(gen_bytes).decode("utf-8", errors="replace")

        return {
            "decode_bps": round(float(n_gen) / max(1e-9, dt), 1),
            "decode_ms_per_byte": round(1000.0 * dt / max(1, n_gen), 3),
            "prefill_ms": round(pref_ms, 2),
            "prefill_bps": round(float(p_ids.size(1)) / max(1e-9, pref_ms / 1000.0), 1),
            "backbone_ticks": ticks,
            "state_bytes": int(ram_after),
            "state_kb": round(ram_after / 1024.0, 2),
            "strict_o1_ram": ram_before == ram_after,
            "vs_batch_max_err": round(err_batch, 7),
            "vs_eager_max_err": round(err_cg, 7) if use_cg else 0.0,
            "sample_preview": preview[:180],
        }

    out: Dict[str, Any] = {}
    lut_cpu, _, _, _ = load_calcite_v2_lut_checkpoint(lut_ckpt, device="cpu", backend="auto")
    lut_cpu_f = fuse_packed_swiglu_for_inference(lut_cpu, inplace=False).eval()
    out["cpu_avx512_fused_lut"] = _run_bench(lut_cpu_f, p_cpu, fixed_stream_cpu, use_cg=False)
    del lut_cpu, lut_cpu_f
    gc.collect()

    if device == "cuda" and torch.cuda.is_available():
        binfo = get_lut_backend_info()
        lut_bk = "cuda_ext" if binfo.get("cuda_ext_available") else "auto"
        lut_gpu, _, _, _ = load_calcite_v2_lut_checkpoint(lut_ckpt, device="cuda", backend=lut_bk)
        lut_gpu_f = fuse_packed_swiglu_for_inference(lut_gpu, inplace=False).eval()
        out["gpu_cuda_graph_fused_lut"] = _run_bench(
            lut_gpu_f, p_cpu.to("cuda"), fixed_stream_cpu.to("cuda"), use_cg=True
        )
        del lut_gpu, lut_gpu_f
        gc.collect()
        torch.cuda.empty_cache()

    cpu_r = out["cpu_avx512_fused_lut"]
    assert cpu_r["strict_o1_ram"], f"CPU fused LUT violated strict O(1) RAM: {cpu_r}"
    assert cpu_r["vs_batch_max_err"] < 2e-5, f"CPU fused LUT err >= 2e-5: {cpu_r}"
    gpu_r = out.get("gpu_cuda_graph_fused_lut")
    if gpu_r:
        assert gpu_r["strict_o1_ram"], f"GPU CG fused LUT violated strict O(1) RAM: {gpu_r}"
        assert gpu_r["vs_batch_max_err"] < 2e-5, f"GPU CG vs batch err >= 2e-5: {gpu_r}"
        assert gpu_r["vs_eager_max_err"] < 2e-5, f"GPU CG vs eager err >= 2e-5: {gpu_r}"
    return out


# ---------------------------------------------------------------------------
# 4. Eager Mid-Run Eval Helper (Zero Dynamo Recompile Pollution)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _eval_unwrapped_suite_k2(
    model: CalciteV2LM,
    cfg: CalciteV2Config,
    suite: list,
    pad_id: int,
    device: str,
    *,
    dca_active: bool,
    bic_active: bool,
    n_limit: int = 64,
    batch_size: int = 16,
) -> float:
    sd_cpu = unwrap_compiled_state_dict(model)
    m_ev = CalciteV2LM(cfg).to(device)
    m_ev.load_state_dict(sd_cpu, strict=True)
    m_ev.byte_chunker.lock_router(True)
    m_ev.set_dca_active(dca_active)
    m_ev.set_bic_active(bic_active)
    m_ev.set_crystal_schedule(1, 1)
    m_ev.set_snap_alpha(1.0)
    m_ev.eval()
    ex_pct, _, _, _ = evaluate_suite_batched(
        m_ev, suite[: min(n_limit, len(suite))], pad_id=pad_id, batch_size=batch_size, device=device
    )
    del m_ev
    if device == "cuda":
        torch.cuda.empty_cache()
    return float(ex_pct)


# ---------------------------------------------------------------------------
# 5. Main Stage-2 `Calcite v4 (S2_25M)` Flagship Closure Pipeline
# ---------------------------------------------------------------------------


def run_v4_25m_closure(
    *,
    data_dir: Path,
    fp_dir: Path,
    ckpt_dir: Path,
    out_json: Path,
    s1a_steps: int = 2000,
    s1b_steps: int = 500,
    s2_steps: int = 1800,
    s2_warmup: int = 60,
    s2_decay: int = 500,
    qat_steps: int = 1000,
    slots: int = 2048,
    batch_size: int = 64,
    micro_bs_init: int = 32,
    qat_micro_bs: int = 16,
    seq_len: int = 512,
    seed: int = 2026,
    use_compile: bool = True,
    force: bool = False,
    smoke: bool = False,
) -> Dict[str, Any]:
    device = "cuda" if has_usable_cuda() else "cpu"
    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram_gb < 12.0:
            torch.cuda.set_per_process_memory_fraction(0.90, 0)
    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()

    fp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    tok = ByteBPETokenizer(merges=[])
    tr_docs, te_docs, tr_sents, _ = load_gsm8k_corpora(ROOT_DIR)
    eff_seq_len = 128 if smoke else int(seq_len)
    train_wins, eval_wins, _, _ = build_disjoint_openweb_corpora(
        ROOT_DIR, tok, tr_docs, te_docs, seq_len=eff_seq_len
    )
    n_ev = 6 if smoke else 150
    _, _, s5_suites, ov_suites, trap_suites = build_eval_bundles(
        ROOT_DIR, tok, n_eval=n_ev, seq_len=eff_seq_len
    )

    router_path = ROOT_DIR / "benchmarks" / "phase_l_v2" / "frozen_utf8_router.pt"
    assert router_path.is_file(), f"Missing canonical frozen UTF-8 router: {router_path}"
    router_state = torch.load(router_path, map_location="cpu", weights_only=True)

    cfg = build_calcite_v4_25m_config(smoke=smoke, slots=slots)
    if smoke:
        rng_f = np.random.default_rng(seed)
        fake_arr = rng_f.integers(32 + 3, 126 + 3, size=100_000, dtype=np.uint16)
        train_mm = val_mm = fake_arr
        s1a_steps, s1b_steps, s2_steps, s2_warmup, s2_decay, qat_steps = 4, 4, 6, 2, 2, 4
        batch_size, micro_bs_init, qat_micro_bs = 8, 4, 4
    else:
        train_bin, val_bin = ensure_phase_k_corpus(data_dir)
        train_mm = load_utf8_corpus_array(train_bin)
        val_mm = load_utf8_corpus_array(val_bin)

    grok_steps = s1a_steps + s1b_steps
    total_fp_steps = grok_steps + s2_steps
    batcher = StratifiedSystem1Batcher(
        train_mm=train_mm,
        tr_docs=tr_docs,
        tr_sents=tr_sents,
        tok=tok,
        seq_len=eff_seq_len,
        train_wins=train_wins,
        seed=seed,
    )

    s1a_ckpt = fp_dir / ("smoke_stage1a_fp.pt" if smoke else "stage1a_fp.pt")
    s1b_ckpt = fp_dir / ("smoke_stage1b_fp.pt" if smoke else "stage1b_fp.pt")
    wsd_plateau_ckpt = fp_dir / (
        "smoke_calcite_v4_wsd_plateau.pt" if smoke else "calcite_v4_s2_25m_wsd_plateau.pt"
    )
    s2_ckpt = fp_dir / ("smoke_stage2_fp.pt" if smoke else "stage2_fp.pt")
    lut_ckpt = ckpt_dir / (
        "smoke_calcite_v4_lut.pt" if smoke else "calcite_v4_s2_25m_utf8_lut.pt"
    )

    t_all = time.perf_counter()
    set_all_seeds(seed)

    # ------------------------------------------------------------------
    # Pre-Flight Gate-2 Init-Std, 0 Mid-UTF8 Cuts & S1A Isolation Tracer
    # ------------------------------------------------------------------
    probe_m = CalciteV2LM(cfg).to(device)
    load_frozen_router_into(probe_m, router_state)
    n_params = int(sum(p.numel() for p in probe_m.parameters()))
    cache_kb = round(compute_v2_state_cache_kb(cfg, 32768), 2)
    fp_router_snap = _snapshot_router_params(probe_m)

    utf8_chk = _check_utf8_zero_mid_cuts(probe_m, device)
    assert utf8_chk["zero_mid_utf8_cuts"], f"Mid-UTF8 cut violation: {utf8_chk}"

    # Verify Gate-2 with DCA+BIC active AND with S1A bypass active
    rng_p = np.random.default_rng(seed + 101)
    xb_p, yb_p = sample_memmap_batch(train_mm, min(4, batch_size), eff_seq_len, rng_p, device)
    yb_p = torch.where(yb_p == 0, torch.full_like(yb_p, -100), yb_p)
    gate2_full = run_gate2_init_probe(probe_m, xb_p, yb_p, device, use_amp)
    assert gate2_full["all_grads_finite"], f"Gate-2 non-finite grads: {gate2_full}"

    _set_stage_module_modes(probe_m, dca_active=False, dca_grad=False, bic_active=False, bic_grad=False)
    gate2_s1a = run_gate2_init_probe(probe_m, xb_p, yb_p, device, use_amp)
    assert gate2_s1a["all_grads_finite"], f"Gate-2 S1A non-finite grads: {gate2_s1a}"
    assert all(p.grad is None for p in probe_m.dca_mixers.parameters()), "S1A DCA isolation failed!"
    assert all(p.grad is None for p in probe_m.byte_chunker.byte_induct_cache.parameters()), "S1A BIC isolation failed!"
    del probe_m
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    print(
        f"\n=== [PHASE P-2 CLOSURE: Calcite v4 {'SMOKE' if smoke else 'S2_25M'}] "
        f"params={n_params:,} | O(1)_cache={cache_kb:.2f} KB | slots=2x{cfg.byte_induct_slots} | "
        f"device={device} | compile={use_compile} | "
        f"steps=(S1A={s1a_steps}, S1B={s1b_steps}, S2={s2_steps}[{s2_warmup}+{s2_steps-s2_warmup-s2_decay}+{s2_decay}], QAT={qat_steps}) ===\n"
        f"  [Gate-2 Preflight] loss={gate2_full['init_loss']} | res_std={gate2_full['block_std'].get('cortex_post_last')} | "
        f"mid_utf8_cuts={utf8_chk['mid_utf8_cuts']} | S1A_bypass_isolated=True",
        flush=True,
    )

    stage_timings: Dict[str, float] = {}
    stage1_gates: Dict[str, float] = {}
    s1_hinge_verified = False

    # ==================================================================
    # STAGE 1A: Pure S_5 Grokking (2,000 steps — Law 8a: DCA=OFF & BIC=OFF)
    # ==================================================================
    if (not smoke) and (not force) and (s2_ckpt.is_file() or s1b_ckpt.is_file() or s1a_ckpt.is_file()):
        src_p = s2_ckpt if s2_ckpt.is_file() else (s1b_ckpt if s1b_ckpt.is_file() else s1a_ckpt)
        blob_meta = torch.load(src_p, map_location="cpu", weights_only=False)
        stage1_gates["s1a_s5_k2"] = float(blob_meta.get("s1a_s5_k2", 0.0))
        print(f"  [Resume] Skipping Stage 1A (already saved; s1a_s5_k2={stage1_gates['s1a_s5_k2']:.2f}%)", flush=True)
    else:
        print(f"\n--- [STAGE 1A: Pure S_5 Grokking ({s1a_steps} steps, DCA=OFF, BIC=OFF)] ---", flush=True)
        set_all_seeds(seed)
        model = CalciteV2LM(cfg).to(device)
        load_frozen_router_into(model, router_state)
        _set_stage_module_modes(model, dca_active=False, dca_grad=False, bic_active=False, bic_grad=False)
        dca_snap_1a = _snapshot_dca_params(model)
        if use_compile and device == "cuda":
            compile_calcite_v2(model, mode="default", dynamic=False)

        base_lr_1a = 1.5e-3
        opt_1a = MuonAdamWHybrid(
            model,
            base_lr=base_lr_1a,
            muon_lr_mult=5.0,
            weight_decay=0.02,
            device=device,
            woodbury_lr_mult=0.05,
            protect_crystal=False,
            ns_schedule="quintic5",
        )
        warm_1a = min(60, max(1, s1a_steps // 12))
        live_diag = (1000, 1500, s1a_steps) if ((not smoke) and s1a_steps >= 1000) else ()
        cur_mbs = min(batch_size, int(micro_bs_init))
        t_1a = time.perf_counter()
        s1a_it = iter(range(1, s1a_steps + 1))

        def _sample_1a():
            st = next(s1a_it)
            return batcher.sample_batch(
                batch_size,
                "cpu",
                step=st,
                grok_steps=grok_steps,
                total_steps=total_fp_steps,
                s1b_steps=s1b_steps,
                s1a_offset=0,
            )

        with BatchPrefetcher(
            _sample_1a, device=device, total_steps=s1a_steps, prefetch_depth=4, enabled=(s1a_steps >= 4)
        ) as pf_1a:
            for step in range(1, s1a_steps + 1):
                model.train()
                model.set_crystal_schedule(step, s1a_steps)
                lr_s = _cosine_lr(step, s1a_steps, warm_1a, base_lr_1a, 2.0e-4)
                sc = lr_s / base_lr_1a
                for pg in opt_1a.param_groups:
                    pg["lr"] = base_lr_1a * float(pg.get("lr_mult", 1.0)) * sc
                xb_full, yb_full = pf_1a.next_batch()
                cur_mbs, ce_sum_t = _run_accum_microbatches(
                    xb_full,
                    yb_full,
                    cur_mbs,
                    opt_1a,
                    device,
                    f"S1A s={step}",
                    lambda xb, yb, cur_acc, m_ref=model: _fp_micro_ce(
                        m_ref, xb, yb, tok.PAD_ID, use_amp, cur_acc, 0.04
                    ),
                )
                torch.nn.utils.clip_grad_norm_(opt_1a.trainable_params, 1.0)
                opt_1a.step()

                if step in (1, s1a_steps) or (step % 250 == 0):
                    n_acc = max(1, (batch_size + cur_mbs - 1) // cur_mbs)
                    ce_v = float(ce_sum_t.item()) / float(n_acc) if ce_sum_t is not None else 0.0
                    vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0
                    print(
                        f"  [S1A] step {step:4d}/{s1a_steps} | mbs={cur_mbs} | ce={ce_v:.4f} | "
                        f"VRAM={vram_mb:.0f}MB | {(time.perf_counter()-t_1a)/60:.1f}m",
                        flush=True,
                    )
                if step in live_diag and step < s1a_steps:
                    pk2 = _eval_unwrapped_suite_k2(
                        model, cfg, s5_suites["k2"], tok.PAD_ID, device, dca_active=False, bic_active=False, n_limit=32
                    )
                    print(f"  [S1A LIVE step={step}] s5_k2_32={pk2:.1f}%", flush=True)

        sync_if_cuda(device)
        stage_timings["s1a_seconds"] = round(time.perf_counter() - t_1a, 2)
        _assert_router_unchanged(model, fp_router_snap, "V4_S1A")
        _assert_dca_unchanged(model, dca_snap_1a, "V4_S1A_DCA_FROZEN")
        s1a_s5_k2 = _eval_unwrapped_suite_k2(
            model, cfg, s5_suites["k2"], tok.PAD_ID, device, dca_active=False, bic_active=False, n_limit=64
        )
        stage1_gates["s1a_s5_k2"] = round(s1a_s5_k2, 2)
        print(
            f"  [S1A DONE] s5_k2={s1a_s5_k2:.2f}% | elapsed={stage_timings['s1a_seconds']:.1f}s",
            flush=True,
        )

        sd_1a = unwrap_compiled_state_dict(model)
        torch.save(
            {
                "config": asdict(cfg),
                "state_dict": sd_1a,
                "s1a_s5_k2": float(s1a_s5_k2),
            },
            s1a_ckpt,
        )
        print(f"  [Saved] Stage-1A FP checkpoint -> {s1a_ckpt}", flush=True)
        del model, opt_1a
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        if not smoke:
            assert s1a_s5_k2 >= 80.0, (
                f"HARD GATE FAILED: Stage 1A S_5 k2={s1a_s5_k2:.2f}% < 80.0%! Halting before S1B."
            )

    # ==================================================================
    # STAGE 1B: Overwrite / Trap + S_5 Maint (500 steps — DCA=ON 0.5x, BIC=OFF, Crystal=0.15x)
    # ==================================================================
    if (not smoke) and (not force) and (s2_ckpt.is_file() or s1b_ckpt.is_file()):
        src_p = s2_ckpt if s2_ckpt.is_file() else s1b_ckpt
        blob_meta = torch.load(src_p, map_location="cpu", weights_only=False)
        stage1_gates["s1_s5_k2"] = float(blob_meta.get("s1_s5_k2", 0.0))
        stage1_gates["s1_ov_long"] = float(blob_meta.get("s1_ov_long", 0.0))
        stage1_gates["s1_trap_long"] = float(blob_meta.get("s1_trap_long", 0.0))
        print(
            f"  [Resume] Skipping Stage 1B (s5_k2={stage1_gates['s1_s5_k2']:.1f}%, "
            f"ov_long={stage1_gates['s1_ov_long']:.1f}%, trap_long={stage1_gates['s1_trap_long']:.1f}%)",
            flush=True,
        )
    else:
        print(f"\n--- [STAGE 1B: Overwrite/Trap Consolidation ({s1b_steps} steps, DCA=ON 0.5x, BIC=OFF)] ---", flush=True)
        blob_1a = torch.load(s1a_ckpt, map_location="cpu", weights_only=False)
        model = CalciteV2LM(cfg).to(device)
        model.load_state_dict(blob_1a["state_dict"], strict=True)
        _set_stage_module_modes(model, dca_active=True, dca_grad=True, bic_active=False, bic_grad=False)
        if use_compile and device == "cuda":
            compile_calcite_v2(model, mode="default", dynamic=False)

        base_lr_1b = 9.0e-4
        opt_1b = MuonAdamWHybrid(
            model,
            base_lr=base_lr_1b,
            muon_lr_mult=3.5,
            weight_decay=0.01,
            device=device,
            woodbury_lr_mult=1.8,
            protect_crystal=True,
            crystal_lr_mult=0.15,
            ns_schedule="quintic5",
        )
        _add_custom_adamw_groups(opt_1b, model, base_lr_1b, dca_lr_mult=0.5, bic_lr_mult=None)
        warm_1b = min(35, max(1, s1b_steps // 8))
        cur_mbs = min(batch_size, int(micro_bs_init))
        t_1b = time.perf_counter()
        s1b_it = iter(range(1, s1b_steps + 1))

        def _sample_1b():
            st = s1a_steps + next(s1b_it)
            return batcher.sample_batch(
                batch_size,
                "cpu",
                step=st,
                grok_steps=grok_steps,
                total_steps=total_fp_steps,
                s1b_steps=s1b_steps,
            )

        with BatchPrefetcher(
            _sample_1b, device=device, total_steps=s1b_steps, prefetch_depth=4, enabled=(s1b_steps >= 4)
        ) as pf_1b:
            for o_step in range(1, s1b_steps + 1):
                model.train()
                model.set_crystal_schedule(s1a_steps, s1a_steps)
                model.set_snap_alpha(1.0)
                lr_s = _cosine_lr(o_step, s1b_steps, warm_1b, base_lr_1b, 1.5e-4)
                sc = lr_s / base_lr_1b
                for pg in opt_1b.param_groups:
                    pg["lr"] = base_lr_1b * float(pg.get("lr_mult", 1.0)) * sc
                xb_full, yb_full = pf_1b.next_batch()
                cur_mbs, ce_sum_t = _run_accum_microbatches(
                    xb_full,
                    yb_full,
                    cur_mbs,
                    opt_1b,
                    device,
                    f"S1B s={o_step}",
                    lambda xb, yb, cur_acc, m_ref=model: _fp_micro_ce(
                        m_ref, xb, yb, tok.PAD_ID, use_amp, cur_acc, 0.04
                    ),
                )
                torch.nn.utils.clip_grad_norm_(opt_1b.trainable_params, 1.0)
                opt_1b.step()
                if o_step in (1, max(1, s1b_steps // 2), s1b_steps) or (o_step % 125 == 0):
                    n_acc = max(1, (batch_size + cur_mbs - 1) // cur_mbs)
                    ce_v = float(ce_sum_t.item()) / float(n_acc) if ce_sum_t is not None else 0.0
                    print(
                        f"  [S1B] step {o_step:4d}/{s1b_steps} | mbs={cur_mbs} | ce={ce_v:.4f} | "
                        f"{(time.perf_counter()-t_1b)/60:.1f}m",
                        flush=True,
                    )

        sync_if_cuda(device)
        stage_timings["s1b_seconds"] = round(time.perf_counter() - t_1b, 2)
        _assert_router_unchanged(model, fp_router_snap, "V4_S1B")

        sd_1b = unwrap_compiled_state_dict(model)
        del model, opt_1b
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        ev_1b = CalciteV2LM(cfg).to(device)
        ev_1b.load_state_dict(sd_1b, strict=True)
        _set_stage_module_modes(ev_1b, dca_active=True, dca_grad=False, bic_active=False, bic_grad=False)
        ev_1b.set_crystal_schedule(1, 1)
        ev_1b.set_snap_alpha(1.0)
        ev_1b.eval()
        s1_s5_k2, _, _, _ = evaluate_suite_batched(
            ev_1b, s5_suites["k2"][: min(64, len(s5_suites["k2"]))], pad_id=tok.PAD_ID, batch_size=16, device=device
        )
        s1_ov_long, _, _, _ = evaluate_suite_batched(
            ev_1b, ov_suites["ood_long"][: min(64, len(ov_suites["ood_long"]))], pad_id=tok.PAD_ID, batch_size=16, device=device
        )
        s1_trap_long, _, _, _ = evaluate_suite_batched(
            ev_1b, trap_suites["ood_long"][: min(64, len(trap_suites["ood_long"]))], pad_id=tok.PAD_ID, batch_size=16, device=device
        )
        del ev_1b
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        stage1_gates["s1_s5_k2"] = round(float(s1_s5_k2), 2)
        stage1_gates["s1_ov_long"] = round(float(s1_ov_long), 2)
        stage1_gates["s1_trap_long"] = round(float(s1_trap_long), 2)
        print(
            f"  [S1 DUAL GATE] s5_k2={s1_s5_k2:.2f}% | ov_long={s1_ov_long:.2f}% | "
            f"trap_long={s1_trap_long:.2f}% | elapsed={stage_timings['s1b_seconds']:.1f}s",
            flush=True,
        )
        torch.save(
            {
                "config": asdict(cfg),
                "state_dict": sd_1b,
                "s1a_s5_k2": float(stage1_gates.get("s1a_s5_k2", 0.0)),
                "s1_s5_k2": float(s1_s5_k2),
                "s1_ov_long": float(s1_ov_long),
                "s1_trap_long": float(s1_trap_long),
            },
            s1b_ckpt,
        )
        print(f"  [Saved] Stage-1B FP checkpoint -> {s1b_ckpt}", flush=True)

    # ==================================================================
    # STAGE 2: WSD Language + Verbatim Needle + Cog Maint (1,800 steps)
    #   60 warmup + 1,240 stable -> SAVE WSD PLATEAU -> 500 cosine decay
    #   BIC=ON (2048 slots, 0.018 s1 hinge, 2.5x LR), DCA=ON (0.4x), Crystal=0.05x, Woodbury=0.12x
    # ==================================================================
    if (not smoke) and (not force) and s2_ckpt.is_file():
        print(f"\n  [Resume] Loading finished Stage-2 FP checkpoint <- {s2_ckpt}", flush=True)
        blob_s2 = torch.load(s2_ckpt, map_location="cpu", weights_only=False)
        sd_s2 = blob_s2["state_dict"]
        stage_timings["s2_seconds"] = float(blob_s2.get("s2_seconds", 0.0))
        s1_hinge_verified = bool(blob_s2.get("s1_hinge_018_verified", True))
    else:
        print(
            f"\n--- [STAGE 2: WSD Language + Needle + Cog ({s2_steps} steps: "
            f"{s2_warmup} warm + {s2_steps - s2_warmup - s2_decay} stable + {s2_decay} decay)] ---",
            flush=True,
        )
        blob_1b = torch.load(s1b_ckpt, map_location="cpu", weights_only=False)
        model = CalciteV2LM(cfg).to(device)
        model.load_state_dict(blob_1b["state_dict"], strict=True)
        _set_stage_module_modes(model, dca_active=True, dca_grad=True, bic_active=True, bic_grad=True)

        # Attach Scale-1 ultra-sparse hinge hook (`0.80 * relu(mean(beta_1) - 0.018)`) for 2048-slot 32K survival
        bic_mod = model.byte_chunker.byte_induct_cache
        assert bic_mod is not None
        s1_hinge_box: List[Optional[torch.Tensor]] = [None]

        def _bic_s1_hinge_hook(_m: nn.Module, _inp: Any, out: torch.Tensor) -> None:
            if bic_mod.training:
                beta_soft_1 = torch.sigmoid(out.float())[..., 1]
                s1_hinge_box[0] = 0.80 * F.relu(beta_soft_1.mean() - 0.018)

        hk_s1 = bic_mod.write_proj.register_forward_hook(_bic_s1_hinge_hook)
        if use_compile and device == "cuda":
            compile_calcite_v2(model, mode="default", dynamic=False)

        base_lr_2 = 1.0e-3
        min_lr_2 = 7.0e-5
        opt_2 = MuonAdamWHybrid(
            model,
            base_lr=base_lr_2,
            muon_lr_mult=4.5,
            weight_decay=0.03,
            device=device,
            woodbury_lr_mult=0.12,
            protect_crystal=True,
            crystal_lr_mult=0.05,
            ns_schedule="quintic5",
        )
        _add_custom_adamw_groups(opt_2, model, base_lr_2, dca_lr_mult=0.4, bic_lr_mult=2.5)
        sched_2 = WSDLRSchedule(
            base_lr=base_lr_2,
            min_lr=min_lr_2,
            total_steps=s2_steps,
            warmup_steps=s2_warmup,
            decay_steps=s2_decay,
            decay_mode="cosine",
        )

        rng_mix_2 = random.Random(seed + 4242)
        mqar_rng_py = random.Random(seed + 7001)
        mqar_rng_np = np.random.default_rng(seed + 7001)
        # 82% web batches * (9/64 = 14.1% MQAR) = 11.5% MQAR, 70.5% 6-dom OpenWeb, 18.0% Cog maint
        n_copy_s2 = max(1, int(round(batch_size * 0.141))) if batch_size >= 8 else 1
        cur_mbs = min(batch_size, int(micro_bs_init))
        t_2 = time.perf_counter()

        def _sample_s2():
            return sample_v4_s2_or_qat_batch(
                batcher,
                train_mm,
                batch_size,
                eff_seq_len,
                cog_prob=0.18,
                n_copy_web=n_copy_s2,
                rng_mix=rng_mix_2,
                mqar_rng_py=mqar_rng_py,
                mqar_rng_np=mqar_rng_np,
                mask_mqar_prefix_for_qat=False,
            )

        try:
            with BatchPrefetcher(
                _sample_s2, device=device, total_steps=s2_steps, prefetch_depth=4, enabled=(s2_steps >= 4)
            ) as pf_2:
                for s2 in range(1, s2_steps + 1):
                    model.train()
                    model.set_crystal_schedule(1, 1)
                    model.set_snap_alpha(1.0)
                    sched_2.apply(opt_2, s2)

                    xb_full, yb_full, mb_full = pf_2.next_batch()
                    is_log = (s2 in (1, sched_2.stable_end, s2_steps)) or (s2 % 200 == 0)

                    while True:
                        opt_2.zero_grad(set_to_none=True)
                        n_acc = max(1, (batch_size + cur_mbs - 1) // cur_mbs)
                        ce_acc_t: Optional[torch.Tensor] = None
                        c_acc_t: Optional[torch.Tensor] = None
                        n_c_m_t: Optional[torch.Tensor] = None
                        try:
                            for st in range(0, batch_size, cur_mbs):
                                xb = xb_full[st : st + cur_mbs]
                                yb = yb_full[st : st + cur_mbs]
                                s1_hinge_box[0] = None
                                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                                    lg = model(xb)
                                    lg_f = lg.float()
                                    if mb_full is not None:
                                        mb = mb_full[st : st + cur_mbs]
                                        tok_ce = F.cross_entropy(
                                            lg_f.reshape(-1, lg_f.size(-1)),
                                            yb.reshape(-1),
                                            ignore_index=tok.PAD_ID,
                                            reduction="none",
                                        ).view_as(yb)
                                        valid = yb != tok.PAD_ID
                                        v_f = valid.float()
                                        m_copy_f = (mb & valid).float()
                                        m_pref_f = ((~mb) & valid).float()
                                        n_c_tok = m_copy_f.sum()
                                        n_p_tok = m_pref_f.sum()
                                        ce_v_all = (tok_ce * v_f).sum() / v_f.sum().clamp(min=1.0)
                                        ce_p = torch.where(
                                            n_p_tok > 0.0,
                                            (tok_ce * m_pref_f).sum() / n_p_tok.clamp(min=1.0),
                                            ce_v_all,
                                        )
                                        has_c = n_c_tok > 0.0
                                        ce_c = torch.where(
                                            has_c,
                                            (tok_ce * m_copy_f).sum() / n_c_tok.clamp(min=1.0),
                                            torch.zeros_like(ce_p),
                                        )
                                        ce = torch.where(has_c, 0.85 * ce_p + 0.15 * ce_c, ce_p)
                                        if is_log:
                                            with torch.no_grad():
                                                eq_c = (
                                                    (lg_f.argmax(dim=-1) == yb).float() * m_copy_f
                                                ).sum() / n_c_tok.clamp(min=1.0)
                                                eq_c = torch.where(has_c, eq_c, torch.zeros_like(eq_c))
                                                has_c_f = has_c.float()
                                                c_acc_t = eq_c if c_acc_t is None else (c_acc_t + eq_c)
                                                n_c_m_t = has_c_f if n_c_m_t is None else (n_c_m_t + has_c_f)
                                    else:
                                        ce = F.cross_entropy(
                                            lg_f.reshape(-1, lg_f.size(-1)),
                                            yb.reshape(-1),
                                            ignore_index=tok.PAD_ID,
                                        )
                                    aux = model.get_commitment_loss()
                                    if s1_hinge_box[0] is not None:
                                        s1_hinge_verified = True
                                        if bic_mod.last_aux_loss is not None:
                                            aux = aux - bic_mod.last_aux_loss
                                        aux = aux + s1_hinge_box[0]
                                    loss = (ce + 0.04 * aux) / float(n_acc)
                                loss.backward()
                                if is_log:
                                    ce_d = ce.detach()
                                    ce_acc_t = ce_d if ce_acc_t is None else (ce_acc_t + ce_d)
                            break
                        except torch.cuda.OutOfMemoryError:
                            if cur_mbs <= 2:
                                raise
                            cur_mbs = max(2, cur_mbs // 2)
                            opt_2.zero_grad(set_to_none=True)
                            torch.cuda.empty_cache()
                            print(f"  [OOM adapt] S2 s={s2}: micro_bs -> {cur_mbs}", flush=True)

                    torch.nn.utils.clip_grad_norm_(opt_2.trainable_params, 1.0)
                    opt_2.step()

                    # Save canonical reusable WSD plateau checkpoint at stable_end (step=1300)
                    if sched_2.is_plateau_save_step(s2):
                        sync_if_cuda(device)
                        save_plateau_checkpoint(
                            wsd_plateau_ckpt,
                            model,
                            opt_2,
                            step=s2,
                            config=cfg,
                            rng_py=mqar_rng_py,
                            rng_np=mqar_rng_np,
                            schedule=sched_2,
                            stage="S2_V4_25M",
                            extra={
                                "s1a_s5_k2": stage1_gates.get("s1a_s5_k2", 0.0),
                                "s1_ov_long": stage1_gates.get("s1_ov_long", 0.0),
                                "bic_s1_hinge_target": 0.018,
                                "s1_hinge_018_verified": s1_hinge_verified,
                            },
                        )
                        print(
                            f"  [WSD PLATEAU SAVED] step={s2}/{s2_steps} -> {wsd_plateau_ckpt} "
                            f"({wsd_plateau_ckpt.stat().st_size / (1024**2):.1f} MB)",
                            flush=True,
                        )

                    if is_log:
                        ce_sum_v = float(ce_acc_t.item()) if ce_acc_t is not None else 0.0
                        n_c_m = int(round(float(n_c_m_t.item()))) if n_c_m_t is not None else 0
                        c_acc_sum = float(c_acc_t.item()) if c_acc_t is not None else 0.0
                        b_info = f" | bwr={bic_mod.last_write_rate:.2f}(s1={bic_mod.last_write_rate_s1:.3f})"
                        c_info = f" | cAcc={100.0*c_acc_sum/n_c_m:5.1f}%" if n_c_m > 0 else ""
                        print(
                            f"  [S2-WSD] step {s2:4d}/{s2_steps} | lr={sched_2.lr_at(s2):.2e} | mbs={cur_mbs} | "
                            f"ce={ce_sum_v/n_acc:.4f}{c_info}{b_info} | {(time.perf_counter()-t_2)/60:.1f}m",
                            flush=True,
                        )
        finally:
            hk_s1.remove()

        sync_if_cuda(device)
        stage_timings["s2_seconds"] = round(time.perf_counter() - t_2, 2)
        _assert_router_unchanged(model, fp_router_snap, "V4_S2_FP")
        sd_s2 = unwrap_compiled_state_dict(model)
        del model, opt_2
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        torch.save(
            {
                "config": asdict(cfg),
                "state_dict": sd_s2,
                "s1a_s5_k2": float(stage1_gates.get("s1a_s5_k2", 0.0)),
                "s1_s5_k2": float(stage1_gates.get("s1_s5_k2", 0.0)),
                "s1_ov_long": float(stage1_gates.get("s1_ov_long", 0.0)),
                "s1_trap_long": float(stage1_gates.get("s1_trap_long", 0.0)),
                "s2_seconds": stage_timings["s2_seconds"],
                "s1_hinge_018_verified": s1_hinge_verified,
            },
            s2_ckpt,
        )
        print(
            f"  [Saved] Stage-2 FP checkpoint -> {s2_ckpt} (s1_hinge_018_verified={s1_hinge_verified})",
            flush=True,
        )

    # ==================================================================
    # FULL FP CLOSURE EVALUATION (Clean Unwrapped Model, eval_bs=8)
    # ==================================================================
    model_fp = CalciteV2LM(cfg).to(device)
    model_fp.load_state_dict(sd_s2, strict=True)
    _set_stage_module_modes(model_fp, dca_active=True, dca_grad=False, bic_active=True, bic_grad=False)
    model_fp.set_crystal_schedule(1, 1)
    model_fp.set_snap_alpha(1.0)
    model_fp.eval()

    ndl_hs = (256,) if smoke else (512, 1500, 4096, 8192, 32768)
    print("\n--- [EVAL: Stage-2 FP Calcite v4 (25M) Full Closure Battery] ---", flush=True)
    eval_fp = evaluate_full_model(
        model_fp,
        val_mm,
        eval_wins,
        tok,
        s5_suites,
        ov_suites,
        trap_suites,
        device=device,
        quick=smoke,
        eval_bs=8,
    )
    long_fp = inject_l2048_from_extrap(
        eval_long_context_stress(
            model_fp, val_mm, device=device, lengths=(256, 512) if smoke else (8192, 32768)
        ),
        None if smoke else eval_fp.get("extrap_4x"),
    )
    needle_fp = eval_ood_verbatim_ladder(
        model_fp,
        val_mm,
        device=device,
        haystacks=ndl_hs,
        n_samples=4 if smoke else 16,
        val_len=4 if smoke else 8,
        seed=9901,
    )
    ov_needle_fp = eval_overwrite_needle_ood(
        model_fp, val_mm, device=device, haystack_bytes=256 if smoke else 1500, n_samples=4 if smoke else 16
    )
    print(
        f"  [FP DONE] Val={eval_fp['val_bpb']:.4f} | OW6={eval_fp['openweb_6']['mean_openweb_bpb']:.4f} | "
        f"L2048={eval_fp['extrap_4x']['bpb_L2048']:.4f} | L32K={long_fp.get('L_32768', {}).get('bpb')} | "
        f"S5={eval_fp['cognitive']['s5_sum']:.1f}% | OV_L={eval_fp['cognitive']['overwrite']['ood_long']['exact_pct']:.1f}% | "
        f"Trap_L={eval_fp['cognitive']['trap']['ood_long']['exact_pct']:.1f}% | "
        f"SelF1_L={eval_fp['cognitive']['selective_f1_ood_long']:.1f}% | "
        f"N512={needle_fp.get('L_512', {}).get('exact_match_pct')} | "
        f"N1500={needle_fp.get('L_1500', {}).get('exact_match_pct')} | "
        f"N8K={needle_fp.get('L_8192', {}).get('exact_match_pct')} | "
        f"N32K={needle_fp.get('L_32768', {}).get('exact_match_pct')} | "
        f"Orig1500={needle_fp['OrigRefCode_1500B']['exact_match_pct']:.1f}% | OvN={ov_needle_fp['new_exact_pct']:.1f}%",
        flush=True,
    )

    # Save intermediate FP JSON before QAT so progress is durably persisted
    interim_payload = {
        "meta": {
            "phase": "Phase_P2_Calcite_v4_S2_25M_Closure",
            "status": "FP_COMPLETE_QAT_PENDING",
            "params": n_params,
            "cache_kb_o1_l32k": cache_kb,
            "config": asdict(cfg),
            "stage_timings": stage_timings,
            "stage1_gates": stage1_gates,
            "s1_hinge_018_verified": s1_hinge_verified,
            "smoke": smoke,
        },
        "eval_fp": eval_fp,
        "long_fp": long_fp,
        "needle_fp": needle_fp,
        "overwrite_needle_fp": ov_needle_fp,
    }
    out_json.write_text(json.dumps(interim_payload, indent=2, default=str), encoding="utf-8")

    # ==================================================================
    # STAGE 3: Copy-Distilled 4-Tap QAT with FROZEN DCA (1,000 steps)
    #   CRITICAL GUARD #1 (`auditor-p` #81): FREEZE ALL `dca_mixers` params!
    #   CRITICAL GUARD #2 (Law 8c): `bic_q LR = 2.5e-4 * w_scale` + prefix-masked MQAR rows (~12.5%)
    # ==================================================================
    print(f"\n--- [STAGE 3: Copy-Distilled 4-Tap QAT ({qat_steps} steps, DCA=FROZEN FP)] ---", flush=True)
    for p in model_fp.parameters():
        p.requires_grad_(False)
    model_fp.byte_chunker.use_static_m_cap = True
    model_fp.eval()

    qat_model = CalciteV2LM(cfg).to(device)
    qat_model.load_state_dict(sd_s2, strict=True)
    qat_model.byte_chunker.use_static_m_cap = True
    # Guard #1: DCA active in forward (exact FP!), strictly frozen (`requires_grad=False`)!
    _set_stage_module_modes(qat_model, dca_active=True, dca_grad=False, bic_active=True, bic_grad=True)
    qat_model.set_crystal_schedule(1, 1)
    qat_model.set_snap_alpha(1.0)
    qat_model.set_qat_mix(0.30)
    qat_model.set_ternary_qat(True)
    dca_snap_qat = _snapshot_dca_params(qat_model)

    # Compile student SWA/blocks cleanly (try full compile_calcite_v2, fallback to compile_v2_swa_blocks)
    qat_compile_mode = "none"
    if use_compile and device == "cuda":
        try:
            if compile_calcite_v2(qat_model, mode="default", dynamic=False):
                qat_compile_mode = "compile_calcite_v2"
        except Exception as exc:
            print(f"  [QAT Compile Notice] compile_calcite_v2 fallback -> compile_v2_swa_blocks ({exc})", flush=True)
            compile_v2_swa_blocks(qat_model)
            qat_compile_mode = "compile_v2_swa_blocks"

    gain_q, cryst_q, chunk_q, core_q, bic_q = _split_v4_qat_5groups_frozen_dca(qat_model)
    all_qp = gain_q + cryst_q + chunk_q + core_q + bic_q
    w_scale = (384.0 / float(cfg.d_model)) ** 0.5
    q_lrs = (1.2e-3 * w_scale, 8.0e-5, 1.1e-3 * w_scale, 4.5e-4 * w_scale, 2.5e-4 * w_scale)
    q_mins = (1.0e-4 * w_scale, 1.5e-5, 8.0e-5 * w_scale, 4.5e-5 * w_scale, 3.0e-5 * w_scale)
    opt_q = _make_adamw(
        [
            {"params": gain_q, "lr": q_lrs[0]},
            {"params": cryst_q, "lr": q_lrs[1]},
            {"params": chunk_q, "lr": q_lrs[2]},
            {"params": core_q, "lr": q_lrs[3]},
            {"params": bic_q, "lr": q_lrs[4]},
        ],
        device,
        weight_decay=0.0,
    )
    # Gate 3.5 invariant: invalidate FastTernaryQAT cache after every `AdamW(fused=True)` step!
    opt_q.register_step_post_hook(lambda *_a, **_kw: clear_fast_qat_cache())

    warm_q = min(50, max(1, qat_steps // 12))
    ramp_q = min(300, max(1, qat_steps // 3))
    kl_temp = 1.5
    taps_fp: List[torch.Tensor] = []
    taps_q: List[torch.Tensor] = []
    hk_fp = _attach_4tap_hooks(model_fp, taps_fp, detach=True)
    hk_q = _attach_4tap_hooks(qat_model, taps_q, detach=False)

    cur_mbs_q = min(batch_size, int(qat_micro_bs))
    n_copy_q = max(1, int(round(batch_size * 0.125))) if batch_size >= 8 else 1
    rng_mix_q = random.Random(seed + 8888)
    mqar_q_py = random.Random(seed + 9001)
    mqar_q_np = np.random.default_rng(seed + 9001)
    t_q0 = time.perf_counter()

    def _sample_qat():
        xb_c, yb_c, _ = sample_v4_s2_or_qat_batch(
            batcher,
            train_mm,
            batch_size,
            eff_seq_len,
            cog_prob=0.18,
            n_copy_web=n_copy_q,
            rng_mix=rng_mix_q,
            mqar_rng_py=mqar_q_py,
            mqar_rng_np=mqar_q_np,
            mask_mqar_prefix_for_qat=True,
        )
        return xb_c, yb_c

    try:
        with BatchPrefetcher(
            _sample_qat, device=device, total_steps=qat_steps, prefetch_depth=4, enabled=(qat_steps >= 4)
        ) as pf_q:
            for qs in range(1, qat_steps + 1):
                qat_model.train()
                mix = min(1.0, 0.30 + 0.70 * (qs / float(ramp_q)))
                qat_model.set_qat_mix(mix)
                clear_fast_qat_cache()
                for gi, (blr, mlr) in enumerate(zip(q_lrs, q_mins)):
                    opt_q.param_groups[gi]["lr"] = _cosine_lr(qs, qat_steps, warm_q, blr, mlr)
                xb_full, yb_full = pf_q.next_batch()
                cur_mbs_q, q_ce_sum_t = _run_4tap_qat_step(
                    model_fp,
                    qat_model,
                    taps_fp,
                    taps_q,
                    xb_full,
                    yb_full,
                    cur_mbs_q,
                    opt_q,
                    tok.PAD_ID,
                    kl_temp,
                    use_amp,
                    device,
                    f"V4-QAT qs={qs}",
                )
                torch.nn.utils.clip_grad_norm_(all_qp, 1.0)
                opt_q.step()
                clear_fast_qat_cache()
                if qs in (1, max(1, qat_steps // 2), qat_steps) or (qs % 100 == 0):
                    n_acc_q = max(1, (batch_size + cur_mbs_q - 1) // cur_mbs_q)
                    qce = float(q_ce_sum_t.item()) / float(n_acc_q) if q_ce_sum_t is not None else 0.0
                    print(
                        f"  [4Tap-QAT] step {qs:4d}/{qat_steps} | mix={mix:.2f} | mbs={cur_mbs_q} | "
                        f"ce={qce:.4f} | {(time.perf_counter()-t_q0)/60:.1f}m",
                        flush=True,
                    )
    finally:
        for h in hk_fp + hk_q:
            h.remove()
        clear_fast_qat_cache()

    sync_if_cuda(device)
    stage_timings["qat_seconds"] = round(time.perf_counter() - t_q0, 2)
    router_mut_diff = _assert_router_unchanged(qat_model, fp_router_snap, "V4_25M_QAT")
    dca_mut_diff = _assert_dca_unchanged(qat_model, dca_snap_qat, "V4_25M_QAT_FROZEN_DCA")
    print(
        f"  [QAT DONE] elapsed={stage_timings['qat_seconds']:.1f}s | mode={qat_compile_mode} | "
        f"router_diff={router_mut_diff:.1e} | dca_diff={dca_mut_diff:.1e}",
        flush=True,
    )

    qat_model.set_ternary_qat(False)
    qat_model.set_qat_mix(1.0)
    qat_sd_clean = unwrap_compiled_state_dict(qat_model)
    del model_fp, qat_model, opt_q
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # ==================================================================
    # PACK 2-BIT UINT8 LUT + DISK ROUNDTRIP + FULL EVAL + STREAMING
    # ==================================================================
    clean_qat = CalciteV2LM(cfg).cpu()
    clean_qat.load_state_dict(qat_sd_clean, strict=True)
    clean_qat.byte_chunker.byte_chunk_mode = "utf8"
    clean_qat.byte_chunker.target_chunk_bytes = float(cfg.target_chunk_bytes)
    clean_qat.byte_chunker.use_static_m_cap = True
    clean_qat.byte_chunker.lock_router(True)
    clean_qat.set_dca_active(True)
    clean_qat.set_bic_active(True)
    clean_qat.set_crystal_schedule(1, 1)
    clean_qat.set_snap_alpha(1.0)
    with torch.no_grad():
        for nm, mod in clean_qat.named_modules():
            if isinstance(mod, nn.Linear) and nm != "lm_head":
                if clean_qat.byte_chunker.is_locked_router_module(mod):
                    continue
                wt, _, _ = quantize_ternary_weights_ls(mod.weight.data)
                mod.weight.data.copy_(wt)
    clean_qat.eval()

    lut_cpu = pack_fast_model_2bit_lut(clean_qat, backend="auto").eval()
    lut_foot = compute_2bit_lut_footprint(lut_cpu)
    n_lut_layers = sum(1 for m in lut_cpu.modules() if isinstance(m, FastPackedLUT2BitLinear))
    save_packed_lut_checkpoint(
        lut_ckpt,
        lut_cpu,
        cfg,
        tok,
        metadata={
            "phase": "Phase_P2_Calcite_v4_S2_25M",
            "params": n_params,
            "router_max_abs_diff": router_mut_diff,
            "dca_qat_max_abs_diff": dca_mut_diff,
            "stage1_gates": stage1_gates,
            "wsd_plateau_step": s2_steps - s2_decay,
            "s1_hinge_018_verified": s1_hinge_verified,
        },
    )
    ckpt_bytes = int(lut_ckpt.stat().st_size)
    print(
        f"  [Saved] 2-bit LUT flagship -> {lut_ckpt} ({ckpt_bytes:,} B = {ckpt_bytes / (1024**2):.2f} MB, "
        f"{n_lut_layers} packed LUT layers)",
        flush=True,
    )
    del clean_qat, lut_cpu
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # 1. Reload from disk (`strict=True`) -> verify DCA FP immunity + fused CPU AVX-512F & GPU CG streaming!
    binfo = get_lut_backend_info()
    lut_bk = "cuda_ext" if (device == "cuda" and binfo.get("cuda_ext_available")) else "auto"
    lut_cog_bs = _recommended_lut_cog_bs(device, binfo)
    lut_gpu_rt, cfg_rt, _, _ = load_calcite_v2_lut_checkpoint(lut_ckpt, device=device, backend=lut_bk)
    assert cfg_rt.use_dca is True and cfg_rt.dw_bottleneck_ratio == 0.5
    dca_fp_preserved = bool(
        lut_gpu_rt.dca_mixers is not None
        and all(
            (not isinstance(m, FastPackedLUT2BitLinear))
            and m.w_dyn.is_floating_point()
            and m.b_stat.is_floating_point()
            for m in lut_gpu_rt.dca_mixers
        )
    )
    assert dca_fp_preserved, "Reloaded LUT checkpoint failed DCA FP preservation check!"
    utf8_lut_chk = _check_utf8_zero_mid_cuts(lut_gpu_rt, device)
    assert utf8_lut_chk["zero_mid_utf8_cuts"], f"Reloaded LUT checkpoint mid-UTF8 cuts: {utf8_lut_chk}"

    print(
        f"\n--- [EVAL: 2-bit LUT Calcite v4 (25M) Disk Roundtrip (`strict=True`, cog_bs={lut_cog_bs})] ---",
        flush=True,
    )
    eval_lut = evaluate_full_model(
        lut_gpu_rt,
        val_mm,
        eval_wins,
        tok,
        s5_suites,
        ov_suites,
        trap_suites,
        device=device,
        quick=smoke,
        eval_bs=8,
        cog_bs=lut_cog_bs,
    )
    if device == "cuda":
        torch.cuda.empty_cache()
    long_lut = inject_l2048_from_extrap(
        eval_long_context_stress(
            lut_gpu_rt, val_mm, device=device, lengths=(256, 512) if smoke else (8192, 32768)
        ),
        None if smoke else eval_lut.get("extrap_4x"),
    )
    if device == "cuda":
        torch.cuda.empty_cache()
    needle_lut = eval_ood_verbatim_ladder(
        lut_gpu_rt,
        val_mm,
        device=device,
        haystacks=ndl_hs,
        n_samples=4 if smoke else 16,
        val_len=4 if smoke else 8,
        seed=9901,
    )
    ov_needle_lut = eval_overwrite_needle_ood(
        lut_gpu_rt, val_mm, device=device, haystack_bytes=256 if smoke else 1500, n_samples=4 if smoke else 16
    )
    stream_needle_lut = eval_live_stream_needle(
        lut_gpu_rt, val_mm, device=device, haystack_bytes=256 if smoke else 1500, n_samples=2 if smoke else 4
    )
    del lut_gpu_rt
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # 2. Fused SwiGLU CPU AVX-512F & GPU CUDA Graph (`use_cuda_graph=True`) streaming benchmarks from disk!
    stream_rt = benchmark_reloaded_lut_streaming(lut_ckpt, val_mm, device=device, smoke=smoke)
    cpu_st = stream_rt["cpu_avx512_fused_lut"]
    gpu_st = stream_rt.get("gpu_cuda_graph_fused_lut") or {}
    print(
        f"  [Stream RT Fused LUT] CPU AVX-512={cpu_st['decode_bps']} B/s ({cpu_st['decode_ms_per_byte']}ms/B, "
        f"err={cpu_st['vs_batch_max_err']:.1e}) | GPU CG={gpu_st.get('decode_bps')} B/s "
        f"({gpu_st.get('decode_ms_per_byte')}ms/B, err_b={gpu_st.get('vs_batch_max_err')}, "
        f"err_e={gpu_st.get('vs_eager_max_err')}) | RAM={cpu_st['state_kb']} KB (O1={cpu_st['strict_o1_ram']})",
        flush=True,
    )

    val_gap = round(float(eval_lut["val_bpb"]) - float(eval_fp["val_bpb"]), 4)
    ow6_gap = round(
        float(eval_lut["openweb_6"]["mean_openweb_bpb"]) - float(eval_fp["openweb_6"]["mean_openweb_bpb"]), 4
    )
    l2048_gap = round(
        float(eval_lut["extrap_4x"]["bpb_L2048"]) - float(eval_fp["extrap_4x"]["bpb_L2048"]), 4
    )
    l32k_fp = (long_fp.get("L_32768") or {}).get("bpb")
    l32k_lut = (long_lut.get("L_32768") or {}).get("bpb")
    l32k_gap = round(float(l32k_lut) - float(l32k_fp), 4) if (l32k_fp is not None and l32k_lut is not None) else None

    print(
        f"  [2bit-LUT DONE] Val={eval_lut['val_bpb']:.4f} (gap {val_gap:+.4f}) | "
        f"OW6={eval_lut['openweb_6']['mean_openweb_bpb']:.4f} (gap {ow6_gap:+.4f}) | "
        f"L2048={eval_lut['extrap_4x']['bpb_L2048']:.4f} | L32K={l32k_lut} | "
        f"S5={eval_lut['cognitive']['s5_sum']:.1f}% | OV_L={eval_lut['cognitive']['overwrite']['ood_long']['exact_pct']:.1f}% | "
        f"Trap_L={eval_lut['cognitive']['trap']['ood_long']['exact_pct']:.1f}% | "
        f"SelF1_L={eval_lut['cognitive']['selective_f1_ood_long']:.1f}% | "
        f"N512={needle_lut.get('L_512', {}).get('exact_match_pct')} | "
        f"N1500={needle_lut.get('L_1500', {}).get('exact_match_pct')} | "
        f"N8K={needle_lut.get('L_8192', {}).get('exact_match_pct')} | "
        f"N32K={needle_lut.get('L_32768', {}).get('exact_match_pct')} | "
        f"Orig1500={needle_lut['OrigRefCode_1500B']['exact_match_pct']:.1f}% | "
        f"OvN={ov_needle_lut['new_exact_pct']:.1f}% | StreamNdl={stream_needle_lut['stream_greedy_exact_pct']:.1f}%",
        flush=True,
    )

    payload: Dict[str, Any] = {
        "meta": {
            "phase": "Phase_P2_Calcite_v4_S2_25M_Closure",
            "status": "COMPLETE",
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "params": n_params,
            "cache_kb_o1_l32k": cache_kb,
            "byte_induct_slots": cfg.byte_induct_slots,
            "config": asdict(cfg),
            "steps": {
                "s1a": s1a_steps,
                "s1b": s1b_steps,
                "s2": s2_steps,
                "s2_warmup": s2_warmup,
                "s2_stable": s2_steps - s2_warmup - s2_decay,
                "s2_decay": s2_decay,
                "qat": qat_steps,
            },
            "seed": seed,
            "smoke": smoke,
            "elapsed_s": round(time.perf_counter() - t_all, 1),
            "stage_timings": stage_timings,
            "stage1_gates": stage1_gates,
            "gate2_preflight": {
                "full": gate2_full,
                "s1a_bypass": gate2_s1a,
                "utf8_fp": utf8_chk,
                "utf8_lut": utf8_lut_chk,
            },
            "s1_hinge_018_verified": s1_hinge_verified,
            "qat_compile_mode": qat_compile_mode,
            "router_max_abs_diff": router_mut_diff,
            "dca_qat_max_abs_diff": dca_mut_diff,
            "dca_fp_preserved": dca_fp_preserved,
            "n_packed_lut_layers": n_lut_layers,
            "wsd_plateau_checkpoint": str(wsd_plateau_ckpt.relative_to(ROOT_DIR))
            if wsd_plateau_ckpt.is_relative_to(ROOT_DIR)
            else str(wsd_plateau_ckpt),
            "wsd_plateau_bytes": int(wsd_plateau_ckpt.stat().st_size) if wsd_plateau_ckpt.is_file() else 0,
            "lut_checkpoint": str(lut_ckpt.relative_to(ROOT_DIR))
            if lut_ckpt.is_relative_to(ROOT_DIR)
            else str(lut_ckpt),
            "lut_checkpoint_bytes": ckpt_bytes,
        },
        "eval_fp": eval_fp,
        "long_fp": long_fp,
        "needle_fp": needle_fp,
        "overwrite_needle_fp": ov_needle_fp,
        "eval_lut": eval_lut,
        "long_lut": long_lut,
        "needle_lut": needle_lut,
        "overwrite_needle_lut": ov_needle_lut,
        "lut_gaps": {
            "val_bpb_gap": val_gap,
            "ow6_mean_bpb_gap": ow6_gap,
            "l2048_bpb_gap": l2048_gap,
            "l32k_bpb_gap": l32k_gap,
        },
        "lut_footprint": lut_foot,
        "stream_rt_fused": stream_rt,
        "stream_needle_lut": stream_needle_lut,
    }
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # Clean up temporary smoke checkpoints so repo stays spotless
    if smoke:
        for tmp_p in (s1a_ckpt, s1b_ckpt, wsd_plateau_ckpt, s2_ckpt, lut_ckpt):
            tmp_p.unlink(missing_ok=True)

    print(f"\n[PHASE P-2 CALCITE v4 25M CLOSURE COMPLETE] elapsed={payload['meta']['elapsed_s']}s | Saved -> {out_json}", flush=True)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase P-2: Calcite v4 (S2_25M) Flagship Closure + WSD Plateau")
    ap.add_argument("--data-dir", type=Path, default=ROOT_DIR / "data_cache")
    ap.add_argument("--fp-dir", type=Path, default=ROOT_DIR / "checkpoints" / "phase_p_fp")
    ap.add_argument("--ckpt-dir", type=Path, default=ROOT_DIR / "checkpoints")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--s1a-steps", type=int, default=2000)
    ap.add_argument("--s1b-steps", type=int, default=500)
    ap.add_argument("--s2-steps", type=int, default=1800)
    ap.add_argument("--s2-warmup", type=int, default=60)
    ap.add_argument("--s2-decay", type=int, default=500)
    ap.add_argument("--qat-steps", type=int, default=1000)
    ap.add_argument("--slots", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--micro-bs", type=int, default=32)
    ap.add_argument("--qat-micro-bs", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", "--smoke-local", dest="smoke", action="store_true")
    args = ap.parse_args()

    out_json = (
        args.out_json
        if args.out_json is not None
        else (
            ROOT_DIR
            / "benchmarks"
            / ("smoke_phase_p_v4_25m_closure.json" if args.smoke else "phase_p_v4_25m_closure_results.json")
        )
    )
    run_v4_25m_closure(
        data_dir=args.data_dir,
        fp_dir=args.fp_dir,
        ckpt_dir=args.ckpt_dir,
        out_json=out_json,
        s1a_steps=args.s1a_steps,
        s1b_steps=args.s1b_steps,
        s2_steps=args.s2_steps,
        s2_warmup=args.s2_warmup,
        s2_decay=args.s2_decay,
        qat_steps=args.qat_steps,
        slots=args.slots,
        batch_size=args.batch_size,
        micro_bs_init=args.micro_bs,
        qat_micro_bs=args.qat_micro_bs,
        seq_len=args.seq_len,
        seed=args.seed,
        use_compile=(not args.no_compile) and (not args.smoke),
        force=args.force,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
