---
name: calcite-phase-runner
description: >-
  Use when implementing, ablating, quantizing, or benchmarking Phases B, C, or D
  of Calcite-5 (SimVQ-Split v5 and CRAM+z) in simvq_split/ or experiments/cram_gsm8k/.
  Enforces the 5-stage gate: Zero-Cheat pytest, init-std & 20-step BF16 parity check,
  fast single-seed CUDA probe with 2-try Stop-Crane, 7-model tournament + adapter
  compatibility check, and Russian report lexical quarantine.
---

# Calcite-5 Phase Execution & QA Runbook

Run phases strictly one architectural axis at a time. Complete each gate before starting the next.

## Gate 1: Zero-Cheat & Baseline Lock (`< 15 s`)
1. Verify untouched baselines: `git diff --exit-code HEAD -- simvq_split/baselines.py` (must exit `0`).
2. Run `python -m pytest tests/test_anti_cheat.py -q` and add a dedicated `Law N` test for any new invariant introduced by the phase.
*Completion criterion:* `0`-byte diff on `baselines.py` and `100%` green `pytest`.

## Gate 2: Init-Std & 20-Step `BF16` Parity Micro-Probe (`< 30 s`)
Before any 2000-step training run:
1. Measure per-block output `std` at initialization (`cortex_pre`, `crystal`, `woodbury`, `mid_mlp`, `cortex_post`) against the last accepted checkpoint (`SimVQSplitLM` v5) — reject any block jumping `> 1.5x` (prevents `SubLN` `5.56x` residual washout).
2. When extending `SimVQSplitLM` while preserving its `System-1` trajectory, verify 20-step `BF16` AdamW + `clip_grad_norm_` weight parity (`max_abs_diff == 0.0` when the new gate is zeroed) and verify `PackedLUT2BitLinear` `.bias_fp16` parity.
*Completion criterion:* Bounded init `std` and verified `BF16` / `LUT` micro-probe.

## Gate 3: Fast Single-Seed CUDA Probe & `OEB <= 2` Stop-Crane (`~7–9 min`)
1. Record noise floor (`2*sigma`) and explicit pass/fail gates in `research/post_transformer_synthesis/` ledger before launching.
2. Run single-seed probe (`experiments/v5_proto/probe_phase_*.py`) covering `FP`, `0-QAT PTQ (LS)`, `300-step STE QAT`, and `PackedLUT2BitLinear`.
3. **Hard Stop-Crane:** After `2` consecutive probe failures on one hypothesis, stop, log the root cause in the ledger, and re-plan via `Via Negativa` — do not launch a blind 3rd run or the full 7-model tournament.
*Completion criterion:* `100%` probe acceptance gates `True` in JSON output.

## Gate 4: Official 7-Model Tournament & Downstream Bridge Check (`~32 min`)
1. Run `python -u run_simvq_split_benchmark.py` followed by `python -u validate_simvq_split_results.py`.
2. Run a read-only CUDA smoke check loading `checkpoints/simvq_split_tournament.pt` into `experiments/cram_gsm8k/simvq_cram_adapter.py` (`strict=True` for both `FP` and `QAT` state dicts, verifying finite slot features).
*Completion criterion:* `ALL QA CHECKS PASSED` and finite downstream adapter tensors.

## Gate 5: Documentation Sync & Lexical Quarantine
Update `SimVQ_Split_Benchmark_Report.md`, phase ledger, `Post_Transformer_Architecture_v5_Blueprint.md`, and `AGENTS.md`, then scan all edited `.md` files with Python to assert `0` matches on forbidden clichés (`является`, `в рамках`, `осуществлять`, `это не просто`, `таким образом`, `в заключение`, `симфония`, `гобелен`, `квантовый скачок`, `бесшовная экосистема`, `уникальный опыт`, `феноменальный`, `колоссальный`) and re-run `python -u validate_simvq_split_results.py`.
*Completion criterion:* `0` quarantine hits and green post-edit `validate_simvq_split_results.py`.
