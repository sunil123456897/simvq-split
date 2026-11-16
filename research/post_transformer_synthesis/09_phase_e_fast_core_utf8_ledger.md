# Журнал Фазы E (`Calcite-5`): Ускоренное ядро без Triton и `UTF-8` защита границ

> [!NOTE]
> Документ фиксирует прохождение всех 3 треков Фазы E (`run_phase_e_fast_utf8_benchmark.py` / `validate_phase_e_results.py`, полный финал `task-6682`) без написания внешних ядер Triton/CUDA, с сохранением полной побитовой совместимости с Фазами A–D (`23/23` закона `Zero-Cheat`, `0`-байтовый дифф на `simvq_split/baselines.py`).

---

## 1. Якоря и приёмочные пороги (`Track 1–3`)

- **Главный риск среды:** Поблочные сканы Вудбери и кристалла на `L=2048` под `torch.amp.autocast("cuda", dtype=torch.bfloat16)` и неразрывная сегментация произвольных байтовых потоков без потери уже обученных тернарных навыков `GSM8K`, $S_5$, `Overwrite` и `Trap Unknown`.
- **Пороги `validate_phase_e_results.py`:**
  - `Track 1` (`L=2048`): `chunker.span_speedup >= 3.0`, `woodbury_fp/tern` (`fwd < 2e-5`, `bwd < 5e-5`, `vram_drop >= 2.5x`), `crystal` (`fwd < 1e-6`, `bwd < 1e-5`, `vram_drop >= 2.5x`), `0` OOM на всех 4 конфигурациях;
  - `Track 2` (Legacy Regression): `max_logit_abs_diff < 1e-4`, `argmax >= 99.5%`, когнитивный дрейф `<= 2.0` п.п. (`<= 2.5` к JSON Фазы B);
  - `Track 3` (`UTF-8` + `2-bit LUT`): `0.00%` разрывов UTF-8 по 6 корпусам (`zeroshot`, `FP`, `QAT`, `LUT`), буфер чанкера `1,268 B` (`L=64..2048`, побитово `True`), кэш `80.0..83.5 KB`, `total_lut_kb <= 555.0` при `16.00x`, `max_pp_diff_vs_qat <= 2.5`, `GSM8K te500 EqPPL <= 48.0`, `S_5 k2 >= 60.0%`, `Selective F1 Long >= 75.0%`, ровно `70` `nn.Linear` с `<= 3` тернарными уровнями.

---

## 2. Инженерные открытия и устранение каузальных дефектов

### 2.1. Форма Жозефа для $P_t$ и экранирование `torch.bmm` от `AMP BFloat16`
1. **Вычитательное обновление `P - Z^T Z` в `float32`:** На почти коллинеарных хвостах правого `PAD` вычитательная форма матрицы прецезии уходила в отрицательный спектр (`min_eig = -0.0267`) и ломала `torch.linalg.cholesky(G_mat)`. Замена на стабилизированную форму Жозефа:
   $$P_{\text{next}} = (I - K k^\top) P (I - K k^\top)^\top + K K^\top$$
   удержала строгую полуопределённость при `float32`.
2. **Ловушка `torch.amp.autocast("cuda", dtype=torch.bfloat16)` (`task-6608`, `task-6635`, `task-6638`):** `evaluate_suite_batched` и циклы обучения оборачивают модель в `autocast(bfloat16)`. Даже при явном `.float()` на входах диспетчер PyTorch AMP понижает все вызовы `torch.bmm` до 7-битной мантиссы `BFloat16`. На 4+ блоках Вудбери (`M_len >= 128`, длинные примеры `ov_suites`) ошибка округления в `25.0 * I` накапливалась до `~0.2` и давала `_LinAlgError` на миноре 26–28 (`Batch element 59`, воспроизведено за 2с на `M_len=160` с `autocast_bf16=True`).
   - **Фикс (`simvq_split/woodbury.py`):** Явная блокировка `with torch.amp.autocast(device.type, enabled=False):` как снаружи `block_parallel_wy_fp_scan`, так и внутри `_wy_fp_block_fwd` (поскольку `checkpoint(..., use_reentrant=False)` восстанавливает внешний `autocast` при обратном проходе), с возвратом итоговых тензоров в исходном `v_seq.dtype`.
   - **Регрессионный тест:** В `Law 17` (`tests/test_anti_cheat.py`) закреплён тест прогона + `.backward()` на 5 коллинеарных блоках (`M_len=160`, `beta=0.99`) внутри `autocast(bfloat16)`.

### 2.2. Сдвиг `lo < 0` в `log-cumsum` и защита UTF-8 без разрывов символов
- В `_wy_fp_block_fwd` накопление произведений $\gamma$ переведено на `float64` `log-cumsum` со строгим маскированием `strict_lower_bool` **до** `exp`, что исключает `0.0 / 0.0 -> NaN` при малых удержаниях $\gamma$.
- В `simvq_split/chunker.py` внедрены: векторизованный `cummax`-ограничитель длины чанка `enforce_max_chunk_span_vec` (`654.44x` к `TorchScript`-циклу на `L=2048`), чистая битовая детекция UTF-8-байтов (`_utf8_tensor_flags`, GPU `@torch.jit.script _enforce_utf8_chunk_span_jit` без `.cpu()`), фиксированный zero-sync расчёт верхней ёмкости чанков (`static_m_cap_for_T`) и изоляция правого `PAD` от энтропийного/соотношенческого лоссов (`1,268 B` в `forward_streaming_check` на `L=64..2048`).

### 2.3. Хроника `Track 3`: Калибровка открытого веба и ортогональная адаптация подпространства `2-bit QAT`
1. **Диагностика переобучения `[:128]` -> `[:2500]` и стационирование ASCII-строк роутера (`3C`):** Узкая нарезка окон `[:128]` давала переобучение (`158.32 EqPPL` на `1_GSM8K_EN`). Расширение выборки на весь корпус `[:2500]` и 120-шаговая калибровка `router_emb.weight[131:]` с обнулением ASCII-градиентов `.grad[:131].zero_()` вывели адаптированную `FP`-модель (`adapted_fp_utf8`) на **`EqPPL = 41.265`** (`1.8277 BPB`), **`mean_openweb_bpb = 3.2772`**, **`S_5 k2 = 80.0%`**, **`OV Long = 99.5%`**, **`Trap Long = 99.0%`** и ровно `0.00%` разрывов UTF-8.
2. **Провалы Попыток #1–#2 `STE QAT` (`task-6492`: `EqPPL=65.80`; `task-6524`: `EqPPL=53.104`):**
   - Холодный перевод `FP` `m_utf8` в 400 шагов `QAT` на `18%` `GSM8K` дал `65.80 EqPPL`.
   - При перезапуске от тернарного `ckpt_b` нулевой шаг давал `44.322 EqPPL`, но полное дообучение на 6-доменном потоке за 400 шагов сместило веса до `53.111 EqPPL`.
   - Дополнительно обнаружены две инженерные ловушки: (а) скалярное ослабление градиента `.mul_(c)` компенсируется нормировкой второго момента в `AdamW`, разгоняя шаг на `Seen`-байтах (`58.48 EqPPL` в `task-6584`), и (б) перенос всех `_router_modules()` из `FP` `m_utf8` затирал тернарные уровни линейных слоёв роутера.
3. **Решение (`Track 3` в финале, `task-6596` -> `task-6682`):**
   - Определена маска виденных в `GSM8K` токенов `seen_mask` (`81/259` якорных, `178/259` пластичных не-ASCII/код/JSON байтов), переносится строго `router_emb` без 5 тернарных `Linear` роутера;
   - **Стадия `3E-1` (`200` шагов):** ортогональное выравнивание `tok_emb.weight` с жёстким `.grad[seen_mask] = 0.0` (`lr = 4.0e-3`);
   - **Стадия `3E-2` (`200` шагов):** `STE QAT` на кодере/декодере чанкера + `tok_emb[131:]` под квадратичным якорем $20.0 \cdot \|p - p_{\text{ref}}\|^2$.
   - **Итог `adapted_lut_utf8` (`548.34 KB` `2-bit uint8 LUT`, `70` слоёв, `16.00x`, `task-6682`):** `GSM8K te500 EqPPL = 45.668` (`1.8775 BPB`), `S_5 k2 = 80.5%`, `OV Long = 99.0%`, `Trap Long = 100.0%` (`Selective F1 Long = 99.5%`, `max_pp_diff_vs_qat = 1.50` п.п.), `max_mid_utf8_split_pct = 0.00%`.

---

## 3. Сводная ведомость замеров (`benchmarks/phase_e_fast_utf8_results.json`, `RTX 5060`)

### 3.1. `Track 1` — Скорость и память VRAM (`L = 2048`, `B = 16`, `BF16`)

| Узел / Модель | Legacy `ms` | Phase E `ms` | Ускорение | Legacy VRAM | Phase E VRAM | Экономия / Точность |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `Chunker span` (`vec` vs `jit`) | `128.924` | `0.197` | **`654.44x`** | — | — | побитово `True` |
| `Chunker encode` (`nosync`) | `143.24` | `18.40` | **`7.78x`** | `569.5 MB` | `569.5 MB` | zero `.cpu()` / zero sync |
| `Woodbury FP` (`WY+Chol ckpt=False`) | `261.72` | `48.07` | **`5.44x`** | `406.9 MB` (saved) | `193.7 MB` (saved) | `fwd=3.58e-7`, `bwd=1.82e-12` |
| `Woodbury FP` (`WY+Chol ckpt=True`) | `261.72` | `74.08` | **`3.53x`** | `406.9 MB` (saved) | `87.1 MB` (saved) | **`4.67x`** VRAM drop |
| `Woodbury Ternary` (`ckpt Bc=32`) | `324.23` | `345.57` | `0.94x` | `412.6 MB` (saved) | `87.1 MB` (saved) | **`4.74x`** VRAM drop, `0.00` diff |
| `Birkhoff Crystal` (`ckpt Bc=32`) | `437.40` | `438.61` | `1.00x` | `226.2 MB` (saved) | `75.9 MB` (saved) | **`2.98x`** VRAM drop, `9.54e-7` |
| E2E `SimVQSplitLM` (`BPE_FP`) | `1601.37` | `1308.52` | **`1.22x`** | `4417.6 MB` (peak) | `3805.2 MB` (peak) | **`1.16x`** пиковой VRAM |
| E2E `SimVQSplitLM` (`ByteHNet_FP_byte`) | `649.91` | `430.76` | **`1.51x`** | `1798.0 MB` (peak) | `1611.0 MB` (peak) | **`1.12x`** пиковой VRAM |
| E2E `SimVQSplitLM` (`ByteHNet_UTF8_staticM`) | `707.21` | `601.43` | **`1.18x`** | `2061.0 MB` (peak) | `1831.9 MB` (peak) | zero-sync + UTF-8 защита |

### 3.2. `Track 2` и `Track 3` — Регрессия и многоязычный `Open-Web + 2-bit LUT`

- **`Track 2` (Legacy Zero-Drift):** Фаза A `max_logit_abs_diff = 4.77e-06` (`FP`/`QAT`, `100.00%` argmax); Фаза B `3.81e-06` (`FP`) / `0.00e+00` (`QAT`); прямой сдвиг `Legacy vs Phase E` на когнитивных сьютах равен `0.50` / `0.00` / `0.00` п.п. (`S_5 k2` / `OV Long` / `Trap Long`).
- **`Track 3` (`6` открытых сплитов и когнитивные сьюты):**

| Модель / Этап | `1_GSM8K_EN` | `2_Russian_UTF8` | `3_Python_Code` | `4_ARC_JSON` | `5_CJK_UTF8` | `6_Emoji_4Byte` | Разрез UTF-8 | `te500 EqPPL` | `S_5 k2` / `SelF1 Long` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `unshielded_byte` (`FP`) | `1.8011` | `12.7455` | `7.0391` | `6.8167` | `15.7533` | `14.8562` | `83.76%` | — | — |
| `zeroshot_utf8` (`FP`) | `1.8010` | `12.7093` | `7.0466` | `6.8167` | `15.8412` | `15.0922` | **`0.00%`** | — | — |
| `adapted_fp_utf8` (`500 st`) | **`1.7957`** | **`3.6734`** | **`3.1527`** | **`1.1071`** | **`5.6985`** | **`4.2361`** | **`0.00%`** | **`41.27`** | `80.0%` / `99.25%` |
| `adapted_lut_utf8` (`548.34 KB`) | `1.9274` | `4.1638` | `5.6382` | `4.0732` | `8.1186` | `5.6167` | **`0.00%`** | **`45.67`** | **`80.5%`** / **`99.5%`** |

- **Статус артефактов (`task-6682`):** [Phase_E_Fast_UTF8_Report.md](file:///e:/эксперименты/Новая%20нейронка/Phase_E_Fast_UTF8_Report.md) (`8,238 B`), [`benchmarks/phase_e_fast_utf8_results.json`](file:///e:/эксперименты/Новая%20нейронка/benchmarks/phase_e_fast_utf8_results.json) (`31,944 B`), [`checkpoints/simvq_split_utf8_openweb.pt`](file:///e:/эксперименты/Новая%20нейронка/checkpoints/simvq_split_utf8_openweb.pt) (`12,299,191 B`) прошли `validate_phase_e_results.py` (`ALL PHASE E QA CHECKS PASSED`).
