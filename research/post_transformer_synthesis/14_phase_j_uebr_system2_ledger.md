# Эпистемический журнал: Фаза J (`Calcite-6 / UEBR` — боевой `System-2` поверх `Tier-1 5.41 MB 2-bit LUT`)

**Проект:** `Calcite-6 / UEBR` -> [`experiments/uebr_system2/`](../../experiments/uebr_system2/) + [`run_phase_j_uebr_benchmark.py`](../../run_phase_j_uebr_benchmark.py) + [`tests/test_phase_j_uebr_system2.py`](../../tests/test_phase_j_uebr_system2.py)  
**Статус:** **`GSM8K & MULTIARITH PASSED / ASDIV-A 1-STEP DOMAIN SHIFT DOCUMENTED (OEB 3/3 CLOSED)`** (`ALL PHASE J QA CHECKS PASSED`, `47/47` законов `Zero-Cheat`, `0`-байтовый дифф защищённых путей `simvq_split/`, `experiments/cram_gsm8k/`, `experiments/uebr_proto/`, `kernels/fast_packed_lut.py`, [`benchmarks/gsm8k/test.jsonl`](../../benchmarks/gsm8k/test.jsonl), весов и JSON Фаз A–I).  
**Замороженный `System-1` (`0` правок):** [`checkpoints/calcite5_tier1_17m_utf8_qat_h_lut.pt`](../../checkpoints/calcite5_tier1_17m_utf8_qat_h_lut.pt) (`95` слоёв `2-bit uint8 LUT = 5,539.81 KB`, кэш `230.11 KB`, буфер `2,164 B`).  
**Итоговый отчёт и JSON (`OEB Attempt #3`):** [`Phase_J_UEBR_System2_Report.md`](../../Phase_J_UEBR_System2_Report.md), [`benchmarks/phase_j_uebr_system2_results.json`](../../benchmarks/phase_j_uebr_system2_results.json) (`29,965 B`).  
**Архивы пост-мортемов (`Attempt #1–#2`):** [`phase_j_attempt1_svamp_postmortem.json`](../../benchmarks/phase_j_attempt1_svamp_postmortem.json) (`19,035 B`), [`phase_j_attempt2_posleak_postmortem.json`](../../benchmarks/phase_j_attempt2_posleak_postmortem.json) (`17,465 B`).

---

## ЖУРНАЛ

- **ТЕКУЩЕЕ:** Закрыт полный цикл Фазы J по дисциплине `OEB <= 3` поверх замороженного `5.41 MB 2-bit LUT` ядра `Tier-1`. На слепых сплитах `GSM8K` модуль `Calcite-6 / UEBR` (`A3`) статистически значимо превзошёл базовый `OpenGraphCRAMSolver` (`B0`): **`+1.62 п.п.` (`3.0x 2*sigma`)** на `BlindDev` и **`4.25x B0` (`+9.70 п.п.`, `18.0x 2*sigma`)** на состязательном `BlindDev-Stress` с перестановкой посылок и посторонним числовым клаузом. После прохождения Stop-Crane однократно вскрыты запечатанные OOD-хранилища: **`MultiArith-600` (`47.83% vs 46.00%`)** пройден, а **`ASDiv-A` (`N = 2,078`)** честно зафиксировал доменный сдвиг априора длины (`Oracle +1.92 п.п.` и `>=4` слотов `+0.80 п.п.`, но спад `le3_slots` на элементарных `1`-шаговых примерах, `"all_passed": false`). Четвёртая попытка OEB не запускалась.
- **ФАКТЫ (`VERIFIED`):**
  1. **Трассер №1 (`--smoke`, `64/32/32` + матричный `gradcheck`):**
     - `3D` матричный `float64` `autograd.gradcheck` для `_ImplicitLaplacianSolveFn` дал ошибку **`2.46e-09 < 1e-7`**; все **`68/68`** обучаемых тензоров `UEBROpenGraphEBM` получили конечные ненулевые градиенты при активных `Woodbury`-якорях (`0` мёртвых параметров, `0` BPTT, щит `torch.amp.autocast(enabled=False)`).
  2. **Попытка №1 -> Пост-мортем утечки совпадений и `w_meta` ([`phase_j_attempt1_svamp_postmortem.json`](../../benchmarks/phase_j_attempt1_svamp_postmortem.json)):**
     - `GSM8K BlindDev` дал `17.64% / 16.82%` (среднее `17.23%` vs `B0 15.95%`), `Stress = 5.43%`, однократный `SVAMP-1000 (1/1)` просел до **`15.9%`** при `Oracle = 97.8%` против `B0 = 24.8%` (`Oracle = 88.9%`).
     - Причина: разрешение `has_exact = False` (`+718` пакетов, `4,595` обучающих примеров против `3,877`) отравило графы случайными арифметическими совпадениями на подмножестве слотов и перевернуло знак калибровочных весов мета-признаков `w_meta[4, 6] < 0`.
  3. **Попытка №2 -> Пост-мортем позиционной утечки и честный `Stop-Crane` до OOD ([`phase_j_attempt2_posleak_postmortem.json`](../../benchmarks/phase_j_attempt2_posleak_postmortem.json)):**
     - Защита знаков `w_meta_effective()` и фильтрация `has_exact` подняли `GSM8K BlindDev` до рекорда серии **`19.27% / 19.40%` (среднее `19.34%`, `Pass@5 = 37.58% / 37.72%`)**, но на `BlindDev-Stress` модель рухнула до **`2.71% / 2.99%` (среднее `2.85%`)**.
     - Причина: позиционные доли `Uf[:, DU_HASH - 36 + 14 : 17]` (`i/n_s`, `sent_id/max_sent`, `clause_id/max_clause`) протекали в `null_prior_net` и `slot_gaq`, привязывая отбраковку дистракторов к позиции первого предложения. Механизм `Stop-Crane` остановил прогон **до** вскрытия `MultiArith-600` и `ASDiv-A` (`0/1` вызовов).
  4. **Попытка №3 -> Развязанный позиционно-инвариантный `Null Ø` и полная лестница на `GSM8K` ([`phase_j_uebr_system2_results.json`](../../benchmarks/phase_j_uebr_system2_results.json)):**
     - Зануление столбцов `[14:17]` в `_compute_null_slot_features` и добавление `5` дискурсивно-сферических сигналов связности (`[sent_peer_bridge, sent_q_bridge, iso_component, max_peer_hp_cos, hp_hq_cos]`) подняли разделение `CleanDev P(Ø) = 0.0579` против `Stress Distractor P(Ø) = 0.9115`.
     - Порог `2*sigma` базы `B0` (`seeds 2026 / 2027`): `BlindDev = 15.74% / 16.01%` (среднее **`15.88%`**, `2*sigma = 0.54 п.п.`, `Oracle = 64.31%`), `BlindDev-Stress = 2.85% / 3.12%` (среднее **`2.99%`**, `2*sigma = 0.54 п.п.`, `Oracle = 38.53%`, `DualCert = 0.0%`).
     - Лестница абляций: `A1 (+ Birkhoff Extension + RLNS/Null Ø)` = `14.11% / 10.72%` (`Oracle = 76.80% / 75.31%`); `A2 (+ GAQ + перекрытие (1 - g_risk))` = `16.69% / 11.26%`; полный `A3 (2026 / 2027)` = **`17.77% / 17.23%`** (среднее **`17.50%`**, `Pass@5 = 35.55% / 35.01%`, `Oracle = 76.93% / 76.12%`, `2.98x / 2.76x` к `Unary`) и **`BlindDev-Stress = 12.89% / 12.48%`** (среднее **`12.69%`, `4.25x B0`**, `Pass@5 = 29.72% / 28.77%`, `Oracle = 75.03% / 75.31%`, `DualCert = 33.33% @ 15.88%`).
     - Физические инварианты `A3` (`2026 / 2027`): `laplacian_min_eig = 0.6208 / 0.6301 > 0`, `epc_max_err = 1.81e-05 / 2.77e-05 < 1e-4`, `wu_drift_norm = 2.1549 / 2.4571 < 6.0`, `wm_eff_min_pen = 0.5815 / 0.5661 > 0`, `birkhoff_lossless_pct = 100.0%` (`72,332` и `81,946` решений), `max_norm_drift = 5.96e-08 < 1e-6`.
  5. **Однократные запечатанные OOD-проверки (`1/1`) и пост-мортем `SVAMP`:**
     - **`MultiArith-600 (N = 600, 1/1)` — ПОБЕДА:** `B0 = 46.00%` (`Pass@3 = 71.67%`, `Pass@5 = 80.83%`, `le3 = 47.56%`, `ge4 = 11.54%`) -> **`A3 = 47.83%`** (`+1.83 п.п.`, `Pass@3 = 75.17%`, `Pass@5 = 82.67%`, `le3 = 49.30%`, `ge4 = 15.38%`, `BE = 100.0%`).
     - **`ASDiv-A (N = 2,078, 1/1)` — СМЕШАННЫЙ ИТОГ (`all_passed = false`):** `A3` выше по покрытию `Oracle` (`81.42% vs 79.50%`, `+1.92 п.п.`) и сложным многослотовым задачам `ge4_slots` (`15.32% vs 14.52%`, `+0.80 п.п.`), но слабее на элементарных 1-шаговых задачах `le3_slots` (`21.69% vs 31.37%`, итого `20.93% vs 29.36%`). Разложение энергии показало: `eu_ablate + er` даёт `30.5%`, но совместный априор `GAQ + W_lm + SPD Laplacian` и веса `w_meta_eff[1] = -0.888`, `w_meta_eff[2] = -0.476`, обученные исключительно на `2–7`-шаговых цепочках `GSM8K train`, перевешивают в пользу 2-шаговых графов на простых примерах с 2 числами.
     - **Незапечатанный `SVAMP-1000` (`not_blind_vault = True`):** `A3` поднялся с `15.9%` до **`19.7%`** (`+3.8 п.п.`, `Oracle = 95.5% vs 88.9%`, `ge4_slots = 3.80% vs 3.16%`, `le3_slots = 22.68% vs 28.86%`), подтвердив тот же структурный профиль доменного сдвига при усилении многошаговых задач.
- **ТУПИКИ И ВЫВОДЫ:**
  1. Разрешение неточных совпадений `has_exact = False` отравляет обучение графов; жёсткая фильтрация восстановила калибровку.
  2. Позиционные признаки `[14:17]` в `Uf` не инвариантны к перестановке предложений и вставке дистракторов; перенос опоры на дискурсивно-сферические признаки дал `4.25x` устойчивость на стрессе без потери `MultiArith`.
  3. Обучение `UEBROpenGraphSolver` только на длинных `2–7`-шаговых цепочках `GSM8K` формирует устойчивый априор против 1-шаговых графов; сбалансированная калибровка длины цепочек на синтетическом 1-шаговом миксе внутри `train` остаётся задачей следующего цикла (вне квоты OEB `3/3`).

---

## Сводная таблица прогонов (`OEB Attempts #1–#3`)

| Попытка | Конфигурация / Фикс | `GSM8K BlindDev` (`B0 -> A3`) | `BlindDev-Stress` (`B0 -> A3`) | OOD / Запечатанный статус | Итог |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **1/3** | Базовый `UEBR` + `has_exact=False` | `15.95% -> 17.23%` | `~3.0% -> 5.43%` | `SVAMP (1/1): 24.8% -> 15.9%` | `PIVOT (утечка + w_meta)` |
| **2/3** | Чистые пакеты + `wm_pen_mask` | `15.88% -> 19.34%` | `2.85% -> 2.85%` | Vaults `MultiArith`/`ASDiv` закрыты (`0/1`) | `STOP-CRANE (утечка позиций)` |
| **3/3** | Инвариантный `Null Ø` (`cols 14:17 = 0`) | `15.88% -> 17.50%` | `2.99% -> 12.69%` (**`4.25x`**) | `MultiArith 47.83% (> 46.0%)`; `ASDiv 20.93% (ge4 +, le3 -)` | **`CLOSED (OEB 3/3)`** |
