# Calcite-5 / Calcite-6 — Архив Фаз A–J, команды воспроизведения и Legacy `v10.4`

Справочник вынесен из `AGENTS.md` ради удержания корневого правила ниже лимита `24,000 B`.

## Журналы, отчёты и гейты Фаз A–J

| Фаза | Журнал / Чертёж | Отчёт и JSON | Тестовый гейт | Ключевые метрики |
| :--- | :--- | :--- | :--- | :--- |
| **A** (`BPE-1536`) | `research/post_transformer_synthesis/06_phase_a_calcite5_ledger.md` | `SimVQ_Split_Benchmark_Report.md`, `benchmarks/simvq_split_results.json`, `checkpoints/simvq_split_tournament.pt` | `tests/test_anti_cheat.py` (`Laws 1–15`) | `1,742,477` params, `89.0 KB` кэш, `954.55 KB` 2-bit LUT, `S_5` / `OV` / `Trap` |
| **B** (`ByteHNet`) | `research/post_transformer_synthesis/08_phase_b_byte_chunker_ledger.md` | `Phase_B_Byte_Chunker_Report.md`, `benchmarks/phase_b_byte_chunker_results.json`, `checkpoints/simvq_split_byte_hnet.pt` | `tests/test_anti_cheat.py` (`Law 16`) | `1,483,841` params, `548.34 KB` 2-bit LUT, `82.72 KB` кэш (`1,256 B` буфер) |
| **C** (`CRAM+z`) | `research/post_transformer_synthesis/07_phase_c_cram_gsm8k_ledger.md` | `benchmarks/phase_c_cram_gsm8k_results.json` | `tests/test_cram_phase_c.py` (`c1..c6`) | `BlindDev 16.01%/16.42%`, `Sealed Test 13.57%/13.57%` (`0.00` п.п., `0` BPTT) |
| **E** (`Fast Core + UTF-8`) | `research/post_transformer_synthesis/09_phase_e_fast_core_utf8_ledger.md` | `Phase_E_Fast_UTF8_Report.md`, `benchmarks/phase_e_fast_utf8_results.json`, `checkpoints/simvq_split_utf8_openweb.pt` | `tests/test_anti_cheat.py` (`Law 17`) | `654.44x` `cummax`, `5.44x` WY+Cholesky, `0.00%` UTF-8 разрывов, `1,268 B` буфер |
| **F** (`UTF-8 CRAM+z`) | `research/post_transformer_synthesis/10_phase_f_utf8_iterative_cram_ledger.md` | `Phase_F_UTF8_Iterative_CRAM_Report.md`, `benchmarks/phase_f_utf8_iterative_cram_results.json` | `tests/test_cram_phase_f.py` (`f1..f4`) | `100%` `2-Opt Block-Gibbs`, `Sealed Test 14.40% FP / 14.03% LUT` (`548.34 KB`) |
| **G** (`Tier-1 17M`) | `research/post_transformer_synthesis/11_phase_g_colab_scaleup_ledger.md` | `Phase_G_Colab_ScaleUp_Report.md`, `benchmarks/phase_g_colab_scaleup_results.json`, `checkpoints/calcite5_tier1_17m_utf8_lut.pt` | `tests/test_phase_g_lut_and_scale.py` (`g1..g5`) | `fast_packed_lut.py` до `18.34x`, `17.49M / 5.41 MB`, `val_bpb 1.78 / 1.99`, `14.63% / 13.04%` |
| **H** (`Tier-1 QAT`) | `research/post_transformer_synthesis/12_phase_h_qat_closure_ledger.md` | `Phase_H_Tier1_QAT_Closure_Report.md`, `benchmarks/phase_h_tier1_qat_closure_results.json`, `checkpoints/calcite5_tier1_17m_utf8_qat_h_lut.pt` | `tests/test_phase_h_qat_closure.py` (`h1..h5`) | 4-group LR + 4-tap + `solver_lut`, `ow_bpb 2.5091`, `S_5 135%/115%`, `Sealed Test 14.40% LUT` |
| **I** (`UEBR Tracer`) | `research/post_transformer_synthesis/13_phase_i_uebr_blueprint.md` | `experiments/uebr_proto/smoke_uebr_invariants.py`, `benchmarks/phase_i_uebr_tracer_results.json` | `tests/test_phase_i_uebr_tracer.py` (`i1..i5`) | `Cholesky Adjoint`, `PermFlow + Birkhoff Extension` `46/50`, `RLNS+Ø`, `GAQ AUROC=1.0000` |
| **J** (`UEBR System-2`) | `research/post_transformer_synthesis/14_phase_j_uebr_system2_ledger.md` | `Phase_J_UEBR_System2_Report.md`, `benchmarks/phase_j_uebr_system2_results.json` | `tests/test_phase_j_uebr_system2.py` (`j1..j5`) | `BlindDev 17.50%`, `Stress 12.69% (4.25x B0)`, `MultiArith-600 47.83%`, `ASDiv-A` развилка |

## Команды воспроизведения Фаз A–J

- **Фаза A:** `python -u run_simvq_split_benchmark.py && python -u validate_simvq_split_results.py` (флаги: `--report-only`, `--only`, `--steps`, `--skip-qat`, `--no-save-ckpt`).
- **Фаза B:** `python -u run_phase_b_byte_benchmark.py && python -u validate_phase_b_byte_results.py` (`--only`, `--from-probe-ckpt`, `--report-only`, `--skip-qat`, `--no-save-ckpt`).
- **Фаза C/F адаптер:** `python -X utf8 -u experiments/cram_gsm8k/simvq_cram_adapter.py --verify-alignment`.
- **Фаза E:** `python -X utf8 -u run_phase_e_fast_utf8_benchmark.py && python -X utf8 -u validate_phase_e_results.py` (`--report-only`, `--only-track3`, `--fast-track1`, `--steps`, `--qat-steps`, `--no-save-ckpt`, `--from-scratch`).
- **Фаза F:** `python -X utf8 -u run_phase_f_utf8_cram_benchmark.py && python -X utf8 -u validate_phase_f_results.py` (`--skip-sealed-test`, `--report-only`, `--reextract`).
- **Фаза G:** `python -X utf8 -u validate_phase_g_results.py` (Colab-раннер: `scripts/colab_phase_g_runner.py --stage track1|build-corpus|train-tier1|eval-cram|all`).
- **Фаза H:** `python -X utf8 -u validate_phase_h_results.py` (Colab-раннер: `scripts/colab_phase_h_qat_runner.py --smoke-local` / `--stage build-corpus|polish-and-qat|eval-cram|all`).
- **Фаза I:** `python -X utf8 -u experiments/uebr_proto/smoke_uebr_invariants.py && python -X utf8 -u validate_phase_i_results.py`.
- **Фаза J:** `python -X utf8 -u validate_phase_j_results.py` (пересборка отчёта: `python -X utf8 -u run_phase_j_uebr_benchmark.py --report-only`).

## Legacy Core (`v10.4`) и архив (`archive_v12_v36/`)

- `simvq_coprime_gpt.py` & `run_champion_local.py` (`checkpoints/simvq_champions_compact.pt.xz`, проверка: `python run_champion_local.py`). Инварианты `v10.4`: (1) `ZeroMaskStreamChunker_v104`: `self.force_full_mask = True` для `T >= 66`; (2) раздельные вентили `m_c4` `(1.32, 0.37, 0.61, 0.18)` и `m_nl` `(1.0, 0.35, 0.45, 0.25, force_full_mask=True)`; (3) `torch.cuda.synchronize()` под проверкой `if dev == "cuda" and torch.cuda.is_available():`.
- `archive_v12_v36/`, `Полный_отчет_исследования_SimVQ_v1_v21.md`, `Architecture_Freeze_Report_v28.md` (запуск из корня: `python archive_v12_v36/<script>.py`).
