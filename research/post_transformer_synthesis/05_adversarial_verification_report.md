# Отчёт независимого состязательного аудита `Calcite-5` (`SimVQ-CRAM v5`)

> [!IMPORTANT]
> **Итоговый вердикт:** **`PASS`** (с чётким разделением на **`VERIFIED`**, подтверждённое на `CUDA` / в `AST` / в первоисточниках, и **`TARGET`** для Фаз A–D интеграции в [`simvq_split/`](file:///e:/эксперименты/Новая%20нейронка/simvq_split/) и [`experiments/cram_gsm8k/`](file:///e:/эксперименты/Новая%20нейронка/experiments/cram_gsm8k/)). Все ранее выявленные дефекты в `01`–`04`, [`Post_Transformer_Architecture_v5_Blueprint.md`](file:///e:/эксперименты/Новая%20нейронка/Post_Transformer_Architecture_v5_Blueprint.md) и [`smoke_v5_invariants.py`](file:///e:/эксперименты/Новая%20нейронка/experiments/v5_proto/smoke_v5_invariants.py) исправлены авторами и перепроверены с чистого прогона.

---

## 1. Сводка ревизии документов и закрытых дефектов

Каждый файл синтеза прошёл цикл: первичный аудит против локальных PDF/HTML первоисточников и кода $\to$ возврат дефектов автору со ссылками на `#seq` $\to$ повторная проверка после финальной правки.

| Документ / артефакт | Объём | Первичные дефекты (`#seq`) | Исправление (`#seq`) | Статус |
|---|---|---|---|---|
| [`01_energy_and_bptt_free_reasoning.md`](file:///e:/эксперименты/Новая%20нейронка/research/post_transformer_synthesis/01_energy_and_bptt_free_reasoning.md) | `330` стр., `50,723` симв. | `#10` (`4` дефекта: заголовок и соавтор `CRAM`; направление `SQuAD ↓` в `EBT Table 3`; нумерация `Eq. 6/7/10` в `JEPA-Anything`; `Table 8` Loss vs `Table 3` PPL в `Depth Memory`) | `#13` (`@energy-lit`) | **`CLEAR`** |
| [`02_state_memory_and_chunking.md`](file:///e:/эксперименты/Новая%20нейронка/research/post_transformer_synthesis/02_state_memory_and_chunking.md) | `389` стр., `72,050` симв. | `#11` (`2` дефекта: заголовок `ZHA` в `p. 1`; `Eq. 17` vs `Eq. 19` для `wkv_t` в `RWKV-7`) | `#14` (`@ssm-memory-lit`) | **`CLEAR`** |
| [`03_our_discoveries_and_bottlenecks.md`](file:///e:/эксперименты/Новая%20нейронка/research/post_transformer_synthesis/03_our_discoveries_and_bottlenecks.md) | `207` стр., `40,000` симв. | `#12` (`2` правки: точное условие квантования `models.py#L45-47` и разбивка `245,760 + 10,797`; определение и точные метрики расхождения `du`) | `#15` (`@internal-audit`) | **`CLEAR`** |
| [`04_unified_architecture_v5_blueprint.md`](file:///e:/эксперименты/Новая%20нейронка/research/post_transformer_synthesis/04_unified_architecture_v5_blueprint.md) | `205` стр., `34,212` симв. | `#17` (границы `VERIFIED` vs `TARGET`, арифметика весов `425.4 / 3,403.3 KB / 7.3x` и кэша `82.66 KB / 10,016 B`) | `#18` (`@architect`) | **`CLEAR`** |
| [`Post_Transformer_Architecture_v5_Blueprint.md`](file:///e:/эксперименты/Новая%20нейронка/Post_Transformer_Architecture_v5_Blueprint.md) | `269` стр., `42,530` симв. | `#17` (согласование `Tracer #1` с реальным охватом, исправление байтов `79.59–83.05 KB` и весов `463.9 KB` / `6.47x`, обновление логов) | `#18` (`@architect`) | **`CLEAR`** |
| [`smoke_v5_invariants.py`](file:///e:/эксперименты/Новая%20нейронка/experiments/v5_proto/smoke_v5_invariants.py) | `1,213` стр., `58,424` Б | `#17` (`4` мёртвых параметра кристалла под чистым `CE` и отсутствие вывода в `logits`; падение под конкуренцией GPU `5.335 s > 3.0 s`; отсутствие `assert` на АЛУ; нецентрированный `unary` в `focused_gibbs_infer`; отсутствие проверки `P_t` SPD) | `#18` (`@architect`) | **`CLEAR`** |

---

## 2. Проверка цитат литературы, формул и внутренних замеров

Все ключевые ссылки, коэффициенты и формулы в досье `01`–`04` и итоговом блюпринте сверены с локальными первоисточниками, выгруженными текстами `arXiv` и файлами репозитория:

1. **Энергетические модели и `0-BPTT` обучение (`01` $\to$ `04` / `Blueprint`):**
   - `CRAM / CRAM+z` (`arXiv:2611.17309v1`, Hoover/Blayney/Whittington): подтверждены точное название (*Classical Energies Learn to Reason 1,000× Faster than Looped Models*), Поттс-ODE (`Eqs. 1–3`), Бесагов `PL` (`Eq. 4`, `App. D.1 Eq. 5–6`), скрытая диффузия `z` (`App. F Eq. 11–15`), результаты `Sudoku-Extreme 100.00%` (`7.7K` пар., `42 ms`), `Maze-Unique 100.00%` (`132 ms`), `Maze-Hard 99.90±0.00%` (`29` пар.), провалы `Implicit Diff = 0.45±0.50%`, `EqProp = 16.05±8.75%`, `Full BPTT (T=10) = 48.82±3.73%` (`5 ч`), а также `40,058×` перерасход FLOPs у `Truncated BPTT (t=5)` относительно `PL` (`Tables 6–7`, [`01#L60-L100,#L265-L280`](file:///e:/эксперименты/Новая%20нейронка/research/post_transformer_synthesis/01_energy_and_bptt_free_reasoning.md#L60-L100)).
   - `EBR` (`2611.08977`), `EBT` (`2507.02092`, `20N` FLOPs, OOD `3/4` побед с уточнением `SQuAD ↓ 53.1 vs 52.3`), `FRM` (`2606.29150v3`, `99.2±0.3%`, `AUROC=1.00`, `44×` дешевле EqR), `EqR` (`2605.21488`, оракул рушит `75.4 -> 13.6%`), `FPRM` (`2606.18206`, `S_5 98.8%`), `Solve the Loop` (`2605.12466`, `1-step IFT` роняет TRM `87.4 -> 56.5%`, полюс $(I-J_g)^{-1}$ в `App. B.3`), `TRMP` (`2610.18649`), `MMAM` (`2507.06211`), `OPF / JEPA-Anything` (`2609.20800v1`, `Eqs. 3–10`), `Depth Memory LDM` (`2610.18809v1`, `+1.88 / +0.33 PPL` в `Table 3` и `+0.1310 / +0.0241` nats в `Table 8`) — всё подтверждено [`01#L60-L100,#L265-L296`](file:///e:/эксперименты/Новая%20нейронка/research/post_transformer_synthesis/01_energy_and_bptt_free_reasoning.md#L60-L100).
2. **Память $O(1)$, $S_n$, динамическое чанкование и 2-битный инференс (`02` $\to$ `04` / `Blueprint`):**
   - `2411.12537v5` (`Thm 1–2`, барьер $\mathrm{TC}^0$ на `PARITY` и $\bmod\,3$), `DeltaProduct` (`2502.10297v7`, `Prop 1.1–1.3`, `Remark 1` с $\|\mathbf{A}\|_2=\sqrt{2}>1$), `Gated DeltaNet` (`2412.06464v3`), `RWKV-7 "Goose"` (`2503.14456v2`, `Eq. 17` для `wkv_t`, `Eq. 19` для `G_t`), `SE-RRM` (`2603.02193`, `S_K`-эквивариантность), `MesaNet` (`2506.05233v2`), `ATLAS` (`2505.23735v1`, `Omega` окно, явная маркировка `[UNKNOWN]` для промежуточных точек `BABILong`), `MIRAS/Yaad` (`2504.13173v1`), `Log-Linear Attention` (`2506.04761v3`), `ZHA` (`2610.42112v2`), `H-Net` (`2507.07955v2`, `Eqs. 4–10`, `58.2% vs 55.5%`, `Robustness 42.8 vs 22.2`), `BLT` (`2412.09871v1`, `Spelling 99.9 vs 1.1`), `T-MAC` (`2407.00088v2`, $2^{g-1}=8$ знаково-зеркальных сумм) — подтверждены [`02#L280-L305`](file:///e:/эксперименты/Новая%20нейронка/research/post_transformer_synthesis/02_state_memory_and_chunking.md#L280-L305), [`04#L128-L138`](file:///e:/эксперименты/Новая%20нейронка/research/post_transformer_synthesis/04_unified_architecture_v5_blueprint.md#L128-L138).
3. **База проекта (`03` + `benchmarks/simvq_split_results.json`):**
   - Подтверждены `SimVQSplitLM v4`: `1,742,477` параметров (`1,485,920` квантованных линейных весов + `245,760` связанных `tok_emb/lm_head` + `10,797` прочих с `22` bias-скалярами), `89.0 KB` кэш при `L=2048` (против `5,120.0 KB` у `RoPETransformerLM` и `7,680.0 KB` у `LoopedTransformerLM`), `GSM8K PPL = 18.579` (`20.811` в QAT), `S_5 = 80.0 / 62.5 / 57.0 / 32.0%` (`QAT k=8 = 24.5%`), `OV OOD-Long = 96.0%` (против `5.5%` у `RoPE`), `Trap OOD-Long = 97.5%` (против `30.5%` у `RoPE`, `Selective F1 = 96.7%` против `9.32%`), `CrystalOnly S_5 = 3.5 / 1.0 / 0.0 / 1.0%`, провал `2+1 JFB` при `alpha=0` (`231.5% -> 150.0%`), потолок `288` шаблонов `cram_gsm8k` (`49.90%` на `4+` шагах) и строковый парсинг `re.findall` + `float`-фоллбэк в `gsm8k_cram_solver.py` ([`03#L41-L44,#L121-L167`](file:///e:/эксперименты/Новая%20нейронка/research/post_transformer_synthesis/03_our_discoveries_and_bottlenecks.md#L41-L44), [`simvq_split_results.json#L438-L550`](file:///e:/эксперименты/Новая%20нейронка/benchmarks/simvq_split_results.json#L438-L550)).

---

## 3. Проверка `12/12 Zero-Cheat Laws`, `10` тупиков и точности расчётов

### 3.1. Дисциплина `Zero-Cheat` на базе и в `Tracer #1`

1. **Защищённый эталон [`simvq_split/`](file:///e:/эксперименты/Новая%20нейронка/simvq_split/) ([`tests/test_anti_cheat.py`](file:///e:/эксперименты/Новая%20нейронка/tests/test_anti_cheat.py)):** все **`12/12` автоматических тестов** пройдены без единого пропуска (`12 passed in 20.30s`), базовые файлы, тесты, валидатор, результаты турнира и чекпоинт не изменены.
2. **Аудит [`smoke_v5_invariants.py`](file:///e:/эксперименты/Новая%20нейронка/experiments/v5_proto/smoke_v5_invariants.py) (`Calcite5StreamingCore`, `EquivariantDeltaCrystal`, `OmegaWoodburyMemory`, `CRAMzOperatorGraphEBM`, `SphericalBase10RegisterALU`):**
   - **AST (`Law 1`):** `bad_imports=[]`, `nn_dict_set=[]`, `nn_regex_calls=[]` — ноль `dict`/`set`/`re`/`ast`/`inspect`/`.eval` внутри всех `nn.Module` классов.
   - **Живые градиенты под чистым `CE` (`Law 2`):** после добавления каузального считывания кристалла `cryst_q / cryst_out / cryst_gate` (`#L322-L324`, `#L385-L394`) обучение [`Calcite5StreamingCore`](file:///e:/эксперименты/Новая%20нейронка/experiments/v5_proto/smoke_v5_invariants.py#L298-L403) на **чистой** `F.cross_entropy` следующего токена (без вспомогательных лоссов на `gate`/`slots`) даёт **`0` мёртвых параметров из `27` тензоров** (`#L1150-L1161`).
   - **Каузальность (`Law 3`) и не-декоративность ветки (`Law 4`):** возмущение будущих токенов `t >= 5` даёт `causal_diff = 0.0e+00`, отключение ветки кристалла (`use_crystal=False`) смещает логиты на `crystal_ablation_diff = 0.2332`, а обнуление `id_well` роняет $S_5$ (`L=512`) с `100.00%` до `0.00%` (`#L1164-L1187`, `#L1028`).
   - **Ковариация $P_t$, биекция $S_G$, двойной сертификат и `BitLinear158 SubLN` (`Laws 7–12` в стенде):** `Calcite5State.total_bytes() == 15000 B (14.65 KB)` от `L=64` до `2048` при `eigvalsh(P_t) in [0.998, 9.317] <= 12.5` (`#L767-L780`) и полном `no_grad` на $P_t$ (`#L271`); норма кристалла и биекция $S_5$ держатся на `norm_drift = 0.00e+00`, `bij = 100.00%`, `voronoi_err = 0.00e+00` в `FP32` (`L=512, 1024`) и `1.58-bit STE QAT` (`L=512`); двойной сертификат `(theta_epi=0.2699, tau_energy=-30.4534)` даёт `100.00%` допуска валидных задач (`solve=93.75%`) и `100.00%` отсева обоих классов ловушек при `0.0%` одиночной фильтрации (`#L1122-L1138`); `CRAM+z Besag PL` решает графы за `1.397 s` (`83.98%`) при `100.00%` изолированного АЛУ на `+, -, *, /` и `91.02%` сквозной точности (`#L862-L937`).
   - **Честная граница между `Tracer #1` (`VERIFIED`) и переносом в продакшен (`TARGET`):** в [`smoke_v5_invariants.py`](file:///e:/эксперименты/Новая%20нейронка/experiments/v5_proto/smoke_v5_invariants.py) ещё **не реализованы** скользящее окно `Omega (c=4)`, 3-факторный скетч Хадамара, `Ratio Loss` на сырых 256 байтах, `T-MAC uint8` упаковка и `G=12` сферические таблицы умножения/деления без промежуточного декодирования в целые тензоры (`apply_op` в `#L677-L691` декодирует кодовые книги через `argmax`, вычисляет целочисленный `torch.div(..., rounding_mode="trunc")` без Python `float` и заново кодирует на сферы с `norm_err=0.00e+00`), равно как и паритет параметров $\pm 5\%$ на реальном турнире `simvq_split/`. Все эти элементы корректно помечены в [`04_unified_architecture_v5_blueprint.md`](file:///e:/эксперименты/Новая%20нейронка/research/post_transformer_synthesis/04_unified_architecture_v5_blueprint.md) и [`Post_Transformer_Architecture_v5_Blueprint.md`](file:///e:/эксперименты/Новая%20нейронка/Post_Transformer_Architecture_v5_Blueprint.md) как `[TARGET]` Фаз A–D.

### 3.2. Блокировка 10 исторических тупиков + `#B5` (`JFB` при $\alpha=0$)

Архитектура `Calcite-5` и код `Tracer #1` удерживают защиту от всех `10` тупиков реестра [`03#L173-L193`](file:///e:/эксперименты/Новая%20нейронка/research/post_transformer_synthesis/03_our_discoveries_and_bottlenecks.md#L173-L193): непрерывная локальная кора не квантуется (`#1`), жёсткий `Zero-Cheat` гейт сохраняется (`#2`), отсутствуют мёртвые пороги `ReLU` (`#3`), `P_t` строго под `no_grad` (`#4`), каузальное центрирование `h_centered` стоит **до** чтения/записи (`#5`, скетч Хадамара задан `[TARGET]` там же), `out_rms` в [`OmegaWoodburyMemory`](file:///e:/эксперименты/Новая%20нейронка/experiments/v5_proto/smoke_v5_invariants.py#L242-L245) всегда умножается на экспоненциальный эпистемический тормоз (`#6`), `G=24` в целевом конфиге и `id_well > 0` (`#7–#9`), `r=2` `DeltaBirkhoff` без затухания мод вместо 4 Хаусхолдеров (`#10`), и `Besag PL (0 BPTT)` на `CRAM+z` вместо `JFB` при $\alpha=0$ (`#B5`).

### 3.3. Контрольный пересчёт таблиц кэша и весов (`04#L105-L120`, `Blueprint#L148-L164`)

Проверено прямым целочисленным расчётом:
- **$O(1)$ кэш `Calcite-5` (`L=2048`):** `System-1 = 23,040 + 3,856 + 10,056 + 40,960 + 3,588 = 81,500 B = 79.5898 KB (79.59 KB)`; `+ System-2 = 81,500 + 2,048 + 1,500 = 85,048 B = 83.0547 KB (83.05 KB)` — совпадает со всеми ячейками таблиц `04` и `Blueprint`.
- **Веса байтового `Calcite-5` (`1,537,677` параметров):**
  - `FP16`: `3,003.28 KB` (`3,003.3 KB`) против `3,403.28 KB` (`3,403.3 KB`) у `v4`;
  - `2-bit uint8 LUT` (`1,485,920` линейных) + `FP16` (`40,960` байт-эмбеддингов + `10,797` прочих): `474,994 B = 463.86 KB (463.9 KB)` (`6.47x` против байтового `FP16`, `7.34x` против `v4` `FP16`);
  - с групповыми `FP16`-шкалами (`g=32`, `+92,870 B = +90.69 KB`): `567,864 B = 554.55 KB (554.6 KB)` (`5.42x`);
  - `2-bit` на все матрицы (`1,526,880`): `393.86 KB (393.9 KB)`; `2-bit` на все `1,537,677` параметров: `375.41 KB (375.4 KB)`.

---

## 4. Итоговые выводы консольных и статических проверок (выполнены лично после финальной правки)

### 4.1. `python -u experiments/v5_proto/smoke_v5_invariants.py` (изолированный GPU-прогон, `exit 0`)

*(Примечание о разбросе wall-clock таймера: в автономном прогоне аудитора скрипт отработал быстрее за `42.47 s`, `pl_0bptt = 1.397 s`, против `53.50 s` / `1.416 s` в логе автора, из-за чего внутри фиксированного окна `~1.40 s` у базовых `unrolled_bptt` и `jfb_1step` вышло `42 / 97` шагов вместо `41 / 100`; все инварианты, точность и `assert` стабильно пройдены).*

```text
Calcite-5 (SimVQ-CRAM v5) Tracer #1 | device=cuda

=== [Invariant 1] Strict O(1) State Bytes (L = 64 -> 512 -> 2048) ===
  L=  64 | state_bytes= 15000 B (14.65 KB) | chunk_rate=0.500 | norm_err=0.00e+00 | P_eig=[0.998, 9.317]
  L= 256 | state_bytes= 15000 B (14.65 KB) | chunk_rate=0.514 | norm_err=0.00e+00 | P_eig=[1.059, 7.600]
  L= 512 | state_bytes= 15000 B (14.65 KB) | chunk_rate=0.499 | norm_err=0.00e+00 | P_eig=[1.074, 8.093]
  L=2048 | state_bytes= 15000 B (14.65 KB) | chunk_rate=0.508 | norm_err=0.00e+00 | P_eig=[1.138, 7.693]
  PASS: Exact O(1) state invariant = 15000 B across L=64..2048; P_t SPD eig=[0.998, 9.317].

=== [Invariant 2] CRAM+z Besag PL (0 BPTT) vs JFB/BPTT (< 3.0 s) + Tensor ALU ===
  mode=pl_0bptt      | iters=220 | train_time=1.397 s ( 6.35 ms/it) | node_acc= 90.43% | exact_graph_acc= 83.98%
  mode=unrolled_bptt | iters= 42 | train_time=1.431 s (34.06 ms/it) | node_acc= 71.29% | exact_graph_acc= 36.72%
  mode=jfb_1step     | iters= 97 | train_time=1.405 s (14.49 ms/it) | node_acc= 59.57% | exact_graph_acc= 39.84%
  Tensor ALU (6-digit Base-10 on S^14): isolated_exact=100.00% | e2e_CRAM_acc= 91.02% | norm_err=0.00e+00
  PASS: Besag PL (0 BPTT) won in 1.397s (83.98% vs BPTT 36.72% / JFB 39.84%).

=== [Invariant 3] DeltaBirkhoff Crystal: Norm Drift < 1e-6 & S_5 id_well > 0 ===
  L= 512, id_well=1.980 > 0, FP32 | S_5 exact=100.00% | bij=100.00% | norm_drift=0.00e+00 | voronoi_err=0.00e+00
  L= 512, id_well=0.000 (off), FP32 | S_5 exact=  0.00% | bij=100.00% | norm_drift=0.00e+00
  L= 512, id_well=1.980 > 0, QAT  | S_5 exact=100.00% | bij=100.00% | norm_drift=0.00e+00 | voronoi_err=0.00e+00
  L=1024, id_well=1.980 > 0, FP32 | S_5 exact=100.00% | bij=100.00% | norm_drift=0.00e+00 | voronoi_err=0.00e+00
  PASS: id_well=1.980 > 0 holds S_5 at 100.00% (vs 0.00% without well), drift < 1e-6.

=== [Invariant 4] 100% Trap Rejection via Dual Certificate (sigma^2_q, E <= tau) ===
  Thresholds calibrated: theta_epi=0.2699, tau_energy=-30.4534
  (A) Known Solvable  | sig2=[0.025,0.188] E=[-46.33,-34.67] | accept=100.00% | solve= 93.75%
  (B) Unseen Entity   | sig2=[0.390,0.915] E=[-46.33,-35.21] | energy_only_rej=  0.0% | dual_rej=100.00%
  (C) Ungrounded Trap | sig2=[0.025,0.188] E=[-28.16,-23.96] | var_only_rej=  0.0%    | dual_rej=100.00%
  PASS: Dual Certificate (sigma^2_q <= theta, E <= tau) hit 100.00% trap rejection & 100.00% valid acceptance.

=== [Anti-Cheat Laws 2/3/4] 0 dead params under pure CE (27 tensors) | causal_diff=0.0e+00 | crystal_ablation_diff=0.2332 ===

ALL 4 CORE INVARIANTS + ANTI-CHEAT LAWS (2/3/4) PASSED in 42.47 s.
```

### 4.2. `python -m pytest tests/test_anti_cheat.py -q` (`exit 0`)

```text
............                                                             [100%]
============================== warnings summary ===============================
C:\Users\vanya\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\jit\_script.py:1487
C:\Users\vanya\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\jit\_script.py:1487
C:\Users\vanya\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\jit\_script.py:1487
C:\Users\vanya\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\jit\_script.py:1487
C:\Users\vanya\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\jit\_script.py:1487
C:\Users\vanya\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\jit\_script.py:1487
  C:\Users\vanya\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\jit\_script.py:1487: DeprecationWarning: `torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
12 passed, 6 warnings in 20.30s
```

### 4.3. `git status --short` и неприкосновенность эталонных файлов (`exit 0`)

```text
 M AGENTS.md
?? Post_Transformer_Architecture_v5_Blueprint.md
?? experiments/
?? research/
```

Команда `git status --short -- simvq_split tests/test_anti_cheat.py validate_simvq_split_results.py benchmarks/simvq_split_results.json checkpoints/simvq_split_tournament.pt` вернула пустой вывод (защищённые файлы не изменены); `git diff --name-only` показал только предсуществующую 1-строчную правку в `AGENTS.md` (ссылка на `experiments/cram_gsm8k/`).

### 4.4. Статический аудит текста, `AST`, ссылок и запрещённых штампов (`exit 0`)

- Все 5 документов (`01`, `02`, `03`, `04`, `Post_Transformer_Architecture_v5_Blueprint.md`): `fffd=0`, `exact_hits=[]` по всем `21` запрещённому штампу/имени, `missing=[]` по всем `137` локальным файловым ссылкам.
- `AST` сканирование [`smoke_v5_invariants.py`](file:///e:/эксперименты/Новая%20нейронка/experiments/v5_proto/smoke_v5_invariants.py): `bad_imports=[]`, `nn_dict_set=[]`, `nn_regex_calls=[]`.
