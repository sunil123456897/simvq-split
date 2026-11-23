---
name: calcite-phase-runner
description: >-
  Use when implementing, ablating, accelerating with CUDA/torch.compile,
  quantizing (STE QAT / 2-bit LUT), or benchmarking any phase of Calcite-5 /
  Calcite-6 (`simvq_split/`, `kernels/`, `experiments/cram_gsm8k/`,
  `experiments/uebr_system2/`, `experiments/scaling_phase_k/`). Enforces
  Zero-Cheat pytest gates, init-std & BF16 parity, v3 CUDA/QAT cache safety
  (`AdamW(fused=True)` hook), Chinchilla Stage-2 scaling, and Russian lexical
  quarantine.
---

# Calcite-5 / Calcite-6 Phase Execution & QA Runbook

Run phases strictly one architectural axis at a time. Complete each gate before starting the next. For historical Phase A–J ledgers, CLI flags, and `v10.4` archive pointers, read [phase-archive.md](./references/phase-archive.md).

## Gate 1: Zero-Cheat & Protected Baseline Lock (`< 15 s`)
1. Verify untouched protected paths: `git diff --exit-code HEAD -- simvq_split/baselines.py` (and for Phase K+, `simvq_split/`, `kernels/fast_packed_lut.py`, `experiments/{cram_gsm8k,uebr_proto,uebr_system2}/`).
2. Run `python -m pytest tests/test_anti_cheat.py -q` and add a dedicated test law for any new invariant introduced by the phase.
*Completion criterion:* `0`-byte diff on protected paths and `100%` green `pytest`.

## Gate 2: Init-Std & `BF16` / `LUT` Parity Micro-Probe (`< 30 s`)
Before any multi-thousand-step training run:
1. Measure per-block output `std` at initialization (`cortex_pre`, `crystal`, `woodbury`, `mid_mlp`, `cortex_post`) — reject any block jumping `> 1.5x` (prevents `SubLN` residual washout).
2. Verify `qat_mix = 0` parity (`< 1e-5`), `PackedLUT2BitLinear` `.bias_fp16` parity (`Law 15`), and live gradients (`100%` non-zero on trainable parameters).
*Completion criterion:* Bounded init `std` and verified `BF16` / `LUT` micro-probe.

## Gate 3: Fast Single-Seed CUDA Probe & `OEB <= 2` Stop-Crane
1. Record noise floor (`2*sigma`) and explicit pass/fail criteria in `research/post_transformer_synthesis/` ledger before launching.
2. Run a minimal 1-batch local smoke tracer (`--smoke-local`) confirming shapes, finite loss/grads, and scrubber hash count before allocating Colab GPU.
3. **Hard Stop-Crane:** After `2` consecutive probe failures on one hypothesis, stop, log the root cause in the ledger, and pivot via `Via Negativa` — never launch a blind 3rd run.
*Completion criterion:* `100%` pre-registered acceptance checks verified in JSON output.

## Gate 3.5: Native CUDA v3, `torch.compile` & QAT Cache Safety
Whenever touching `kernels/fast_calcite_scans.py`, `kernels/fast_calcite_swa_chunker.py`, `experiments/scaling_phase_k/`, or `scripts/colab_phase_k_scaling_runner.py`:
1. **`AdamW(fused=True)` `_version` Trap:** On CUDA, `torch.optim.AdamW(..., fused=True)` updates `weight.data` in-place WITHOUT incrementing `weight._version`. Any quantized-weight cache (`FastTernaryQAT`) MUST stay wired to `register_optimizer_step_post_hook` + explicit `clear_fast_qat_cache()` after `opt.step()`, verified by `pytest tests/test_fast_calcite_scans.py -q` (`test_fast_qat_fused_adamw_cache_invalidation`: `stale_diff == 0.0`).
2. **`torch.compile` + `custom_op` + 4-Tap Distillation:**
   - Register custom CUDA kernels via `torch.library.custom_op` + `register_fake` + `register_autograd` so TorchInductor has zero graph breaks.
   - Keep `torch.compile(...)` calls strictly inside `kernels/` (never inside `lean_system1.py`, which fails `test_k1` AST purity).
   - Unwrap compiled models via `unwrap_compiled_state_dict(model)` before `eval_fp`, `teacher` distillation, and `2-bit LUT` packing; assert `len(taps_fp) == 4 and len(taps_q) == 4` inside `_qat_micro_step`.
3. **Stage-2 Chinchilla Scaling, Packaging & Resumability:** Pack Colab bundles via `python -X utf8 -u scripts/colab_pack.py --out <tar.gz> --verify-import scripts.colab_phase_k_scaling_runner`, persist intermediate `stage1b_fp.pt` / `stage2_fp.pt` (`--fp-ckpt-dir`, `--resume-stage2-from`), reuse cached Google Drive corpus (`/content/drive/MyDrive/calcite_cache/phase_k_corpus_cache.tar.gz`, Drive `FILE_ID = 1uXTZ8Umh8RxisFILGfU51tx3b6YqlMhf`, `322,796,697 B`), and scale `Stage 2` OpenWeb bytes proportionally to parameters (`>= 1.1–1.3 B/param`) with monotone `stage2_emb_lr`.

## Gate 4: Full Verification & Downstream Compatibility
1. Run `python -X utf8 -u validate_phase_k_results.py` (or the active phase validator) + `python -m pytest tests/test_fast_calcite_scans.py -q`.
2. Verify CPU roundtrip of every saved `2-bit LUT` `.pt` checkpoint (`0.00%` mid-UTF8 cuts, `router_max_abs_diff == 0.0`, finite BPB).
*Completion criterion:* `ALL PHASE <X> QA CHECKS PASSED` and clean LUT roundtrip.

## Gate 5: Documentation Sync & Lexical Quarantine
Update the phase report, ledger, `README.md`, and `AGENTS.md` (keep `AGENTS.md` strictly `< 24,000 B` to avoid context truncation — push older phase catalogs into `./references/phase-archive.md`), then run `python -X utf8 -u scripts/repo_guardrail.py` (`0` quarantine hits, `ruff` clean, `baselines.py` `0`-byte diff, `>= 17` anti-cheat tests passing).
