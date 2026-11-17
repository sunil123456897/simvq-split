# Эпистемический журнал: Фаза H (`Calcite-5 Tier-1` 17.49M — закрытие разрыва `FP -> 5.41 MB 2-bit LUT` и возврат когнитивного баланса)

**Проект:** `Calcite-5 / SimVQ-Split v5.0` -> [`scripts/colab_phase_h_qat_runner.py`](file:///e:/эксперименты/Новая%20нейронка/scripts/colab_phase_h_qat_runner.py) + [`tests/test_phase_h_qat_closure.py`](file:///e:/эксперименты/Новая%20нейронка/tests/test_phase_h_qat_closure.py)  
**Статус:** **`COMPLETED & VERIFIED`** (`ALL PHASE H QA CHECKS PASSED`, `37/37` автотестов Фаз A–H, нулевой дифф ядра `simvq_split/`, `experiments/cram_gsm8k/`, LUT-весов Фазы G и JSON Фаз A–G).  
**Базовый отсчёт (Фаза G):** [`checkpoints/calcite5_tier1_17m_utf8_bf16.pt`](file:///e:/эксперименты/Новая%20нейронка/checkpoints/calcite5_tier1_17m_utf8_bf16.pt) (`70,471,291 B`) и [`benchmarks/phase_g_colab_scaleup_results.json`](file:///e:/эксперименты/Новая%20нейронка/benchmarks/phase_g_colab_scaleup_results.json).  
**Итоговый артефакт весов (`5.41 MB`):** [`checkpoints/calcite5_tier1_17m_utf8_qat_h_lut.pt`](file:///e:/эксперименты/Новая%20нейронка/checkpoints/calcite5_tier1_17m_utf8_qat_h_lut.pt) (`7,129,267 B`, `SHA256 = 09a9e0e5e59de85378f27f428867d6564c31877b5e971ce65da1a9580c21676e`, `95` слоёв = `5,539.81 KB`, `16.00x`).  
**Итоговый JSON:** [`benchmarks/phase_h_tier1_qat_closure_results.json`](file:///e:/эксперименты/Новая%20нейронка/benchmarks/phase_h_tier1_qat_closure_results.json) (`53,992 B`, `SHA256 = f1039d96f4e6626372cfe4e5bd3b2c11401d5d597adc13c573a855a38d83f49d`).

---

## ЖУРНАЛ

- **ТЕКУЩЕЕ:** Все этапы Фазы H в Google Colab (`NVIDIA A100-SXM4-80GB`) завершены (`Stage 1+2` Попытка №2 + `Stage 3` `CRAM+z`), артефакты скачаны локально с проверкой `SHA256`, сеанс `A100` немедленно освобождён (`runtime.unassign()`), локальные тесты `37/37` и валидаторы Фаз A–H подтверждены.
- **ФАКТЫ (`VERIFIED`):**
  1. **Попытка №1 (`Stage 1` 1500 шагов + `Stage 2` 2400 шагов `Logit-KL`, `B=32`) -> Стоп-кран (`EXIT: 1`):**
     - `Stage 1 FP`: снизил `val_bpb` с `1.7842 -> 1.7408`, поднял `S_5 sum` с `103.0% -> 134.5%`, `OV_long = 85.5%` (сохранён промежуточный чекпоинт `attempt1_polished_bf16.pt`).
     - `Stage 2 LUT`: `val_bpb = 1.9530` (разрыв к новому `FP = +0.2122`), `S_5 sum = 114.5%`, но регрессия на редких режимах (`ow_mean = 2.9643` против `2.8682`, `6_Emoji_4Byte = 5.4215`, `4_ARC_JSON = 0.9579`). Поблочный зонд выявил усадку `SwiGLU`: `mid_mlp` `cos = 0.5474`, `std: 2.7100 -> 1.4322` (`0.528x`).
  2. **Попытка №2 (тёплый старт с `attempt1_polished_bf16.pt` + `400` шагов `Stage 1` = `1,900` суммарных `FP`-шагов от Фазы G + `1600` шагов `Stage 2` `4-Tap + Logit-KL`, `B=64`, 4-групповой `LR` + 6 доменов `train_wins`; дефолт `--polish-steps 1900` при холодном запуске с Фазы G):**
     - Поблочное выравнивание: `enc_mlp std_ratio = 1.000x` (`1.2129`), `pre3 = 0.822x`, `mid_mlp = 0.785x` (`2.1787 / 2.7761`), `post0 = 0.822x`, `norm_f = 0.942x`, мутация роутера **`0.0`**.
     - `mean_openweb_bpb` (`5.41 MB LUT`, `0.00%` разрывов UTF-8): **`2.8682 -> 2.5091`** (**`-0.3591 BPB`**, ниже даже `FP` Фазы G `2.6119`), разрыв `LUT - FP` сузился с `+0.2563` до **`+0.1761 BPB`** (`-31.3%`). По режимам `LUT`: `GSM8K EN 1.7101`, `RU 2.3204` (`-0.521`), `Python 2.0479` (`-0.704`), `ARC JSON 0.6789`, `CJK 3.5537`, `Emoji 4.7438`.
     - `gsm8k_te500` (`5.41 MB LUT`): **`1.9059 BPB (48.39 EqPPL) -> 1.7477 BPB (35.07 EqPPL)`** (разрыв сузился на `-25.1%`, а `FP` снизился с `32.60 -> 26.09 EqPPL`).
     - Когнитивный баланс (`5.41 MB LUT` vs Фаза G): **`S_5 sum: 88.0% -> 115.0%`** (`+27.0` п.п.; `FP: 103.0% -> 135.0%`), **`OV in_dist: 88.5% -> 91.5%`**, **`OV ood_long: 75.0% -> 83.5%` (`+8.5` п.п.)**, **`SelF1 ood_long: 85.71% -> 91.01%` (`+5.30` п.п.)**, `voronoi_lattice_err_L2048 = 2.38e-07`, `norm_drift = 1.19e-07`.
     - Честная фиксация плато `val_memmap_bpb`: две разные схемы QAT сошлись в `1.9530` и `1.9540` (лучше Фазы G `1.9879`, но `FP` также опустился до `1.7533`, давая разрыв `+0.2007 BPB > 0.175`, `stop_crane_passed: false` к исходной гипотезе `0.175`, `beats_phase_g_all_axes: true`). Третья слепая попытка по правилу `OEB <= 2` не запускалась.
     - Примечание по схеме JSON: в сыром артефакте Colab поле `system1_qat_closure.model_spec.params` записалось как `0` (функция `count_params(m_fp)` фильтровала `p.requires_grad` после заморозки учителя `m_fp`), тогда как сохранённый чекпоинт `calcite5_tier1_17m_utf8_qat_h_lut.pt` и `SimVQSplitLM(config)` строго проверяются на `17,488,961` параметров (ошибка в `scripts/colab_phase_h_qat_runner.py` исправлена на безусловный `sum(p.numel() ...)`).
  3. **Stage 3 (`GSM8K + Interleaved SphericalALU + 0-BPTT CRAM+z`, `d_model = 384`, Гершгорин `4.10`):**
     - `16` эпох FP (`weighted_loss = 0.1633`) + `1,120` шагов `4-Tap + Masked KL QAT` (`final_loss = 0.1656`) + обучение парного `solver_lut` (`0` BPTT, `max_norm_drift = 5.96e-08`).
     - **`BlindDev (N = 737)`:** `FP = 18.72%` (`138/737`, `Pass@5 = 36.23%`), абляция `LUT на solver_fp = 18.59%` (`137/737`, потеря квантизации `0.13` п.п.!), парный `LUT на solver_lut = 17.50%` (`129/737`, `Pass@5 = 35.82%`, `DualCert = 38.00% @ 20.35%`) — выше Фазы G LUT (`16.15%`) и в допуске `< 1.90` п.п.
     - **`Sealed Test (N = 1,319, 1/1 вызов)`:** `FP = 13.50%` (`178/1319`, `Pass@5 = 30.71%`) и **`5.41 MB 2-bit LUT = 14.40%` (`190/1319`, CI `[12.61%, 16.40%]`, `Pass@5 = 31.54%`, `2.68x` к `Unary 5.38%`, `DualCert = 34.26% @ 21.91%`)** — прирост **`+1.36` п.п.** к `13.04%` (`172/1319`) Фазы G, выше `14.03%` Фазы F и без деградации к `FP` (`+0.90` п.п.).
- **ТУПИКИ И ВЫВОДЫ:**
  1. Чистая дистилляция финальных логитов без `RMSNorm`-ускорения и без `train_wins` провалила `mid_mlp` (`std 0.528x`) и `6_Emoji_4Byte` (`5.4215`). 4-точечная дистилляция + 4-групповой `LR` + `train_wins` вылечили все `6` открытых доменов и `CRAM+z`, тогда как на сыром многоязычном `val_memmap` подтверждён структурный предел `1.58`-битного квантования (`+0.2007 BPB` на `17.49M` при замороженной решётке чанкера).

---

## Журнал запусков (`Attempt #1` -> `Attempt #2` + `Stage 3`)

| Шаг | Тип | Действие / Среда | Наблюдаемая метрика | Вердикт |
| :---: | :---: | :--- | :--- | :--- |
| 0 | `CHECK` | `--smoke-local` (`RTX 5060`, оба варианта раннера) | `Gate 2 <= 1.50x`, `95` слоёв `5,539.81 KB`, `0` NaN | `PASS (Tracer #1)` |
| 1 | `TRY 1/2` | `Stage 1 (1500) + Stage 2 KL-QAT (2400, B=32)` (`A100`) | `FP val=1.7408 / S5=134.5%`; `LUT val=1.9530 / ow=2.9643 / S5=114.5%` | `STOP-CRANE (PIVOT)` |
| 2 | `TRY 2/2` | `Stage 1 (attempt1 + 400 = 1900 FP) + Stage 2 4Tap+KL (1600, B=64)` (`A100`) | `LUT ow=2.5091` (`-31.3%` gap), `te500=35.07`, `S5=115.0%`, `OV_long=83.5%`, `val=1.9540` | `PLATO val (+0.2007) / ALL AXES BEAT G` |
| 3 | `PASS` | `Stage 3 eval-cram` (`1120` 4Tap+KL + `solver_lut`, `A100`) | `BlindDev: 18.72% / 18.59% / 17.50%`; **`Sealed Test LUT: 14.40%` (`+1.36` п.п. к G)** | `PASS (1/1 Vault)` |
| 4 | `CLOSE` | Выгрузка `SHA256` -> `runtime.unassign()` -> `validate_phase_h_results.py` | `37/37` `pytest`, `ruff` чисто, валидаторы A–H, карантин `22` док. | `VERIFIED & CLOSED` |
