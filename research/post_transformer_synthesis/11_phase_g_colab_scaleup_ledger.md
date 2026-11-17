# Эпистемический журнал: Фаза G (`Native C++/CUDA PackedLUT2BitLinear` + `Calcite-5 Tier-1 17.49M / 5.41 MB` в Google Colab и масштабирование `GSM8K + CRAM+z`)

**Проект:** `Calcite-5 / SimVQ-Split v5.0` -> [`kernels/`](file:///e:/эксперименты/Новая%20нейронка/kernels/fast_packed_lut.py) + [`scripts/colab_phase_g_runner.py`](file:///e:/эксперименты/Новая%20нейронка/scripts/colab_phase_g_runner.py)  
**Статус:** **`COMPLETED & VERIFIED`** (`ALL PHASE G QA CHECKS PASSED`, `32/32` автотестов Фаз A–G, нулевой дифф ядра `simvq_split/` и `experiments/cram_gsm8k/`).  
**Базовый отсчёт (Фаза E + Фаза F):** [`checkpoints/simvq_split_utf8_openweb.pt`](file:///e:/эксперименты/Новая%20нейронка/checkpoints/simvq_split_utf8_openweb.pt) (`1,483,841` параметров, `548.34 KB` `2-bit LUT`, `70` слоёв, кэш `O(1)` = `82.73 KB` на `L=2048`, буфер чанкера `1,268 B`, `Sealed Test = 14.40% FP / 14.03% LUT`).  
**Итоговый артефакт весов (`5.41 MB`):** [`checkpoints/calcite5_tier1_17m_utf8_lut.pt`](file:///e:/эксперименты/Новая%20нейронка/checkpoints/calcite5_tier1_17m_utf8_lut.pt) (`7,126,729 B`, `SHA256 = 46cad2ce108ae2c657ea988545e16caa5ca553bd416463db775a0669bd076fb0`).

---

## ЖУРНАЛ

- **ТЕКУЩЕЕ:** Фаза G полностью закрыта. Нативные ядра `cuda_ext`/`cpu_ext`, предобучение `Calcite-5 Tier-1` (`17,488,961` параметров, `5,539.81 KB` = `5.41 MB` `2-bit LUT`) на `1.70 GB` чистом UTF-8 корпусе в Google Colab (`NVIDIA A100-SXM4-80GB`), многоязычные оценки `Track 2` и стыковка с `CRAM+z` (`Track 3`) завершены, чекпоинты и [`benchmarks/phase_g_colab_scaleup_results.json`](file:///e:/эксперименты/Новая%20нейронка/benchmarks/phase_g_colab_scaleup_results.json) (`41,775 B`, `SHA256 = 96d1b620...`) скачаны локально, среда Colab немедленно освобождена (`runtime.unassign()`).
- **ФАКТЫ (`VERIFIED` после завершения всех стадий):**
  - **Track 1 (`NVIDIA A100-SXM4-80GB`, `Torch 2.14.1+cu130`):**
    - Одиночные слои: `mlp_384_to_1024_aligned` (`4.075 -> 0.383 ms`, **`10.64x`**), `mlp_1024_to_384_aligned` (`4.724 -> 0.381 ms`, **`12.40x`**), невыровненный `woodbury_172_to_384_unaligned` (`row_aligned=False`, `2.751 -> 0.150 ms`, **`18.34x`**), ошибка к Python `PackedLUT2BitLinear` `<= 7.2e-07` (`FP32`) и `0.00390625` (`1 ULP BF16`).
    - Сквозные модели на `B=16, L=2048`: `Base 1.48M` (`548.34 KB`, `70` слоёв) `1926.793 -> 418.473 ms` (**`4.60x`**, быстрее плотного `qat_fp32 = 421.948 ms`), `cuda_max_abs_diff = 8.58e-06`; `Tier-1 17.49M` (`5,539.81 KB` = `5.41 MB`, `95` слоёв `16.00x`) `2565.227 -> 484.966 ms` (**`5.29x`** к эталону, рядом с `qat_fp32 = 460.610 ms`), `cuda_max_abs_diff = 1.07e-06`. На CPU (`OpenMP B=1, L=256`): `817.350 -> 309.692 ms` (**`2.64x`**, `diff = 1.13e-06`).
  - **Stage 2 (`Zero-Leakage UTF-8 Stream`):** собраны `train = 1,698.23 MB` (`1,780,723,340` байтовых токенов) + `val = 24.00 MB` (`25,165,824` токенов) за `211.89 s`; проверено `37,866` хешей отложенных сплитов (`Cal + BlindDev + SealedTest + ARC eval`), вырезано `10` совпадений (`1` FineWeb-Edu + `9` FineMath), `0` утечек. После локального аудита `test_g4` сканнер `is_leaked_doc` переведён со шага `15` на шаг `1` для защиты при любом сдвиге префикса.
  - **Stage 3 & 3B (`Tier-1 17.49M` Pretraining + `STE QAT` -> `5.41 MB LUT`):**
    - `7,890` шагов `BF16` (`B=64, L=1024`, `0.6843 s/step`, пик VRAM `10.22 GB / 80 GB`, `87.84 min`, `468.25 MB` seen, `CE: 5.6544 -> 1.1459`) + `600` шагов двухгруппового `STE QAT` (`9.77 min`).
    - Постоянный кэш `O(1)` при `L=2048`: **`230.11 KB`**. Локальный `test_g3` выявил, что реальный буфер `SurpriseHNetChunker` при расширении `d_model: 160 -> 384` равен **`2,164 B`** (`1,268 + 896 B`), а не зашитому литералу `1268` в старом JSON; значение уже учитывалось в `230.11 KB`, а литерал в `colab_phase_g_runner.py` исправлен на вызов `compute_chunker_state_bytes(...)`.
    - Языковой сдвиг (`Track 2`): `val_memmap_bpb` упал с **`3.9423`** (`Base 1.48M LUT`) до **`1.7842` (`Tier-1 FP`, `-54.7%`)** и **`1.9879` (`Tier-1 5.41 MB LUT`, расхождение `0.0001` BPB к `QAT = 1.9880`)**; средний открытый веб (6 корпусов) снизился с **`4.9230`** до **`2.6119` (`FP`) / `2.8682` (`5.41 MB LUT`)** при строго **`0.00%` разрывов UTF-8** и `voronoi_lattice_err_L2048 = 2.38e-07`.
    - `GSM8K te500`: `1.8775` BPB (`EqPPL 45.67`) -> **`1.7118` (`EqPPL 32.60`, `FP`)** / `1.9059` (`EqPPL 48.39`, `LUT`).
    - Честный компромисс синтетики: при `82%` веб-прозы / `18%` когнитивных эпизодов `S_5 sum` составил `103.0% (FP) / 88.0% (LUT)` против `120.5%` у `Base 1.48M` (`Trap = 100%/100%`, `SelF1_long = 92.18% / 85.71%`).
  - **Stage 4 (`Track 3`: `Tier-1 + Interleaved SphericalALU + 0-BPTT CRAM+z`):**
    - Попытка №1 упала из-за несовпадения имени метода `solver.fit` -> исправлено по точному контракту (`fit_bptt_free`, `enrich_lm_feats_with_system1_tf`, `calibrate_on_split`), Попытка №2 завершилась с `EXIT: 0` с подхватом кэша `/content/phase_g_ckpts/tier1_cram_adapted.pt` (`final_loss = 0.1596`, `4,443` трассы).
    - `BlindDev (N=737)`: **`18.05%` `Pass@1`** (`Pass@5 = 35.96%`, `2.56x` к `Unary`, `DualCert = 40.00% @ 19.00%`) в `FP` и **`16.15%` `Pass@1`** (`Pass@5 = 35.41%`, `2.48x` к `Unary`, `DualCert = 41.48% @ 18.32%`) в `5.41 MB LUT`.
    - `Sealed Test (N=1,319, 1/1 вызов)`: **`14.63%` `Pass@1`** (`193/1319`, `[12.83%, 16.64%]`, `Pass@5 = 31.01%`, `2.76x` к `Unary 5.31%`, `DualCert = 35.19% @ 21.76%`) в `FP` и **`13.04%` `Pass@1`** (`172/1319`, `Pass@5 = 30.25%`, `2.65x` к `Unary 4.93%`, `DualCert = 31.97% @ 22.29%`) в `5.41 MB 2-bit LUT` при `max_norm_drift = 5.96e-08` и запасе Гершгорина `4.10`.
- **ТУПИКИ И ИСПРАВЛЕНИЯ В ХОДЕ ФАЗЫ:**
  1. Листовой случай `accelerate_lut_model(packed_linear)` в первой версии не преобразовывал одиночный слой `PackedLUT2BitLinear` -> исправлено прямым `from_packed`.
  2. `solver.fit` и ключ `c["dag"]` в первом черновике `run_stage_eval_cram` заменены на реальные API `OpenGraphCRAMSolver` (`fit_bptt_free`, `enrich_lm_feats_with_system1_tf`, `c["steps"]`).
  3. Выявлен зашитый литерал `chunker_state_bytes = 1268` при фактических `2,164 B` для `d_model = 384` -> исправлено на вызов `compute_chunker_state_bytes` без подмены сохранённого Colab JSON.

---

## Журнал запусков (`PROPOSITION / LAUNCH / CHECK / PIVOT`)

| Шаг | Тип | Действие / Команда | Наблюдаемая метрика | Вердикт |
| :---: | :---: | :--- | :--- | :--- |
| 0 | `CALIBRATION` | Локальный замер `Tier-1` (`RTX 5060`) | `17,488,961p`, `5,539.81 KB` (`95` слоёв), `230.11 KB` кэш | `VERIFIED` |
| 1 | `CHECK` | `colab_phase_g_runner.py --stage track1` (`A100`) | Слои `10.64x / 12.40x / 18.34x`, `Tier-1 5.29x`, `diff <= 1.07e-06` | `PASS (Tracer #1)` |
| 2 | `CHECK` | `--stage build-corpus` (`total_gb=1.80`) | `train=1698.23 MB` (`1.781B` tok), `val=24.00 MB`, `scrubbed=10` (`211.9s`) | `PASS (Zero-Leak)` |
| 3 | `CHECK` | `--stage train-tier1` (`B=64, L=1024`) | `7,890` FP + `600` QAT -> `val_bpb=1.7842 / 1.9879`, `ow_bpb=2.6119 / 2.8682`, `mid_utf8=0.00%` | `PASS` |
| 4a | `CHECK` | `--stage eval-cram` (Попытка №1) | `System-1 adapt` (`loss=0.1596`) ОК, падение `solver.fit` | `FAIL (TRY 1/2)` |
| 4b | `PIVOT + PASS` | Фикс `fit_bptt_free` + кэш `tier1_cram_adapted.pt` | `BlindDev = 18.05% / 16.15%`, `Sealed Test = 14.63% / 13.04%` (`1/1`) | `PASS (TRY 2/2)` |
| 5 | `CHECK` | Выгрузка `SHA256` -> `runtime.unassign()` + локальный `Zero-Manual QA` | `32/32` pytest + валидаторы A–G + `ALL PHASE G QA CHECKS PASSED` | `VERIFIED & CLOSED` |
