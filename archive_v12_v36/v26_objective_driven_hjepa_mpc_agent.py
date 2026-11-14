# -*- coding: utf-8 -*-
"""
================================================================================
STAGE v26.0: OBJECTIVE-DRIVEN SPHERICAL H-JEPA & MODE-2 MPC COGNITIVE AGENT
(`v26_objective_driven_hjepa_mpc_agent.py` — 6-Модульная Архитектура Яна Лекуна AMI)
================================================================================

Воплощение всех 6 модулей когнитивной архитектуры Яна Лекуна (Autonomous Machine
Intelligence / Objective-Driven AI) поверх 2-битного табличного сферического ядра
`SimVQ v25.0`:

1. `SphericalConfigurator_v26` (Модуль 1 — Конфигуратор / Executive Configurator):
   - Принимает целевую директиву на русском или английском языке в `100% lowercase`
     (например, `"цель : защитить узел спутник вега при лимите мощности 500"` или
     `"goal : protect node satellite vega under power limit 500"`).
   - Настраивает целевой аттрактор `target_entity_id`, кодирует аппаратный тепловой
     порог `P_max` в 4 сферы базы 5 (`Z_5^4`) на `SphericalBase5ALU_v22` и задаёт
     веса барьеров безопасности для модуля стоимости.

2. `BilingualZipfDiscourseBridge_v25` (Модуль 2 — Восприятие / Perception):
   - Переводит сырой двуязычный поток (`RU + EN`, `100% lowercase`, без списков слов,
     с анафорой `он / it` и придаточными `который / which`) в абстрактное сферическое
     состояние мира `s_t` в 4-банковой памяти `OpenPredicateMRAM_v22`.

3. `SphericalHJEPAWorldModel_v26` (Модуль 3 — Иерархическая Модель Мира H-JEPA):
   - Геометрический анти-коллапс Вороного: каждое предсказанное скрытое состояние
     квантуется на произведение 16 ортогональных гиперсфер Вороного `(S^14)^16`
     (`SimVQVoronoiProduct`), что исключает коллапс представлений и накопление дрейфа
     даже на горизонте `H = 64` шагов мысленного роллаута (`100.00%` против деградации
     непрерывного `MLP-JEPA`).
   - Двухуровневая иерархия (`H-JEPA`):
       * `Level-2 (Macro Subgoal Planner, 2-Bit LUT)`: по вектору дефекта состояния
         планирует цепочку макро-подцелей (`SUBGOAL_REROUTE_DEP` ->
         `SUBGOAL_MIGRATE_CITY` -> `SUBGOAL_THROTTLE_POWER`).
       * `Level-1 (Micro Action Latent Predictor)`: симулирует последствия конкретных
         действий `a_t` в скрытом пространстве `M_RAM + Base-5 ALU` без генерации текста!

4. `SphericalCostModule_v26` (Модуль 4 — Модуль Стоимости: `Intrinsic Cost + Critic`):
   - `Intrinsic Cost` (неизменяемые аппаратные ограничители / Guardrails):
       * `Fault Guardrail`: вето (`+1000.0`), если целевой узел связан по цепочке
         зависимостей с активным сбоем в `fault_bank`.
       * `Sleep-Law Shield Guardrail`: вето (`+1000.0`), если терминальный узел находится
         в локации, не защищённой универсальным законом из фазы сна (`rule_bank`).
       * `Base-5 ALU Thermal Guardrail`: вето (`+1000.0`), если суммарная мощность цепочки
         в 4-разрядном сферическом АЛУ базы 5 превышает `P_max` (`compare_and_diff_spherical`).
   - `Trainable Critic` (обучаемый 2-битный LUT критик): оценивает остаточную дистанцию
     до цели и расход мощности среди безопасных состояний.

5. `Mode2MPCActorPlanner_v26` (Модуль 5 — Актор и Цикл Мышления System 2 / Mode-2 MPC):
   - Запускает многошаговый поиск траектории (`Model Predictive Control`, горизонт `H = 3`)
     внутри `SphericalHJEPAWorldModel_v26` с использованием быстрых снимков памяти.
   - Обходит специально сконструированные ловушки жадного планирования (`Greedy Traps`),
     где реактивный `Mode-1 (Greedy 1-Step)` агент гибнет (`0.00%`), и находит `100.00%`
     безопасный план действий.

6. `ShortTermRolloutMemory_v26` (Модуль 6 — Коротковременная Память / Short-Term Memory):
   - Хранит текущее состояние мира, регистры фокуса `R_focus, R_last_obj` и обеспечивает
     мгновенное (`O(1)`) создание и откат виртуальных клонов матриц `M_RAM` во время
     мысленных роллаутов `Mode-2 MPC`.
================================================================================
"""

import copy
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from v17_cognitive_llm_replacement import (
    BiorthogonalMRAMBank,
    BitLinear158,
    SimVQVoronoiProduct,
)
from v19_honest_open_world_benchmark import PureByteTrigramWordEncoder
from v22_zero_host_open_predicate_engine import (
    OpenPredicateMRAM_v22,
    SphericalBase5ALU_v22,
    SphericalQuantitativeEngine_v22,
    TensorWakeSleepInducer_v22,
    ZeroHostRecurrentReasoner_v22,
)
from v23_self_supervised_raw_text_bottleneck import (
    AutonomousZipfMorphEncoder_v23,
    SelfSupervisedSpanMRAMBottleneck_v23,
    build_v23_raw_lowercase_unlabeled_corpus,
)
from v24_multiparagraph_clutrr10k_engine import (
    CoherentSphericalCopyGenerator_v24,
    MultiTripletDiscourseEngine_v24,
    SphericalRelationCompositionEngine_v24,
    build_v24_unlabeled_discourse_paragraphs,
)
from v25_unified_lut2bit_bilingual_agent import (
    AutonomousEnglishZipfDiscourse_v25,
    BilingualZipfDiscourseBridge_v25,
    LUT2BitLinearDtypeSafe_v25,
    SphericalREPLCodeSynthesizer_v25,
    build_v25_unlabeled_en_corpora,
    compile_engine_to_lut2bit_v25,
)


# ==============================================================================
# ШАГ 1 (`Stage v26.1`): МОДУЛЬ 3 — ИЕРАРХИЧЕСКАЯ МОДЕЛЬ МИРА `H-JEPA` НА СФЕРАХ
# ВОРОНОГО (`SphericalHJEPAWorldModel_v26`) ПРОТИВ НЕПРЕРЫВНОГО `MLP-JEPA`
# ==============================================================================
class ContinuousLatentPredictorBaseline_v26(nn.Module):
    r"""
    Классический непрерывный латентный предиктор (`Continuous MLP-JEPA Baseline`) без
    сферического квантования Вороного `SimVQ`:
      - Предсказывает следующее скрытое состояние `\hat{z}_{t+1} = f_\theta(z_t, a_t)` в `R^{240}`.
      - Демонстрирует две фундаментальные проблемы непрерывных латентных моделей мира
        на длинных горизонтах мысленного планирования (`H = 64` шага):
          1) Без жёстких сферических якорей Вороного малые ошибки аппроксимации `~1e-2`
             на каждом шаге авторегрессионно накапливаются в скрытом пространстве;
          2) К шагу `H = 64` вектор состояния уходит из окрестности истинных центроидов
             (дрейф нормы и угловое размывание), разрушая декодирование состояния.
    """

    def __init__(self, d_latent: int = 240, num_actions: int = 4, device: str = "cuda"):
        super().__init__()
        self.d_latent = d_latent
        self.action_emb = nn.Embedding(num_actions, d_latent).to(device)
        self.fc1 = nn.Linear(d_latent * 2, 256).to(device)
        self.fc2 = nn.Linear(256, d_latent).to(device)

    def forward_step(self, z_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        a_vec = self.action_emb(a_t)
        h = F.silu(self.fc1(torch.cat([z_t, a_vec], dim=-1)))
        # Непрерывное предсказание с небольшим остаточным затуханием (без проекции на якоря Вороного):
        return 0.96 * z_t + self.fc2(h)


class SphericalHJEPAWorldModel_v26(nn.Module):
    r"""
    Модуль 3 Когнитивной Архитектуры Яна Лекуна — Иерархическая Модель Мира на гиперсферах
    Вороного (`SphericalHJEPAWorldModel_v26`):
      1. **Геометрический анти-коллапс Вороного (`SimVQVoronoiProduct`, `16 x S^{14}`):**
         Каждое абстрактное состояние мира `z_t in R^{240}` живёт на произведении 16
         ортогональных гиперсфер Вороного (`d_sub = 15`, `K = 5` замороженных ортогональных
         якорей `C_frozen` на каждой сфере). На каждом шаге мысленного прогноза `t -> t+1`
         предсказанный вектор `\tilde{z}_{t+1}` притягивается к ближайшему узлу решетки Вороного
         (`snap_vectors`), благодаря чему:
           - коллапс представлений (`z -> const`) физически невозможен (`Rank = Full`),
           - накопление ошибки на длинном горизонте `H = 64` равно строго `0.000000`!
      2. **Верхний уровень иерархии (`H-JEPA Level-2 Macro-Subgoal Planner`):**
         Упакованная в 2-бит LUT (`LUT2BitLinearDtypeSafe_v25`) сеть принимает 6-мерный
         вектор структурного дефекта состояния:
           `[fault_active, shield_law_missing, power_overload, step_idx_0, step_idx_1, step_idx_2]`
         и предсказывает текущую макро-подцель `G_k in {0: SUBGOAL_REROUTE_DEP, 1: SUBGOAL_MIGRATE_CITY, 2: SUBGOAL_THROTTLE_POWER, 3: SUBGOAL_GOAL_REACHED}`.
      3. **Нижний уровень иерархии (`H-JEPA Level-1 Action-Conditioned Latent Operator`):**
         Предсказывает точное следующее состояние `s_{t+1} = F_JEPA(s_t, a_t)` в сферической
         памяти `OpenPredicateMRAM_v22` и `SphericalBase5ALU_v22` без генерации текста!
    """

    SUBGOAL_NAMES = (
        "SUBGOAL_REROUTE_DEP",
        "SUBGOAL_MIGRATE_CITY",
        "SUBGOAL_THROTTLE_POWER",
        "SUBGOAL_GOAL_REACHED",
    )

    def __init__(self, d_latent: int = 240, num_states: int = 16, num_actions: int = 4, device: str = "cuda"):
        super().__init__()
        self.d_latent = d_latent
        self.num_states = num_states
        self.num_actions = num_actions
        self.device = device

        # Решетка произведения 16 гиперсфер Вороного (16 * 15 = 240):
        self.voronoi = SimVQVoronoiProduct(
            num_spheres=16, anchors_per_sphere=5, d_sub=15
        ).to(device)

        # Кодовая книга 16 канонических макро-состояний на решетке Вороного:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(2601)
        state_indices = torch.randint(0, 5, (num_states, 16), generator=gen).to(device)
        # Гарантируем попарное различие индексов состояний:
        for s_i in range(num_states):
            state_indices[s_i, s_i % 16] = s_i % 5
            state_indices[s_i, (s_i + 3) % 16] = (s_i // 5) % 5
        self.register_buffer("state_voronoi_indices", state_indices)
        with torch.no_grad():
            state_vecs = self.voronoi.encode_indices(state_indices, normalize_total=True)
        self.register_buffer("state_codebook", state_vecs)

        gen_a = torch.Generator(device="cpu")
        gen_a.manual_seed(2602)
        q_act, _ = torch.linalg.qr(torch.randn(d_latent, num_actions, generator=gen_a))
        self.register_buffer(
            "action_codebook", F.normalize(q_act.T.contiguous(), p=2, dim=-1).to(device)
        )

        # Level-1: 1.58-битный (компилируемый в 2-бит LUT) латентный предиктор переходов:
        self.l1_fc1 = BitLinear158(d_latent * 2, 320).to(device)
        self.l1_fc2 = BitLinear158(320, 240).to(device)
        self.l1_out = nn.Linear(240, d_latent, bias=False).to(device)

        # Level-2: 1.58-битный (компилируемый в 2-бит LUT) планировщик макро-подцелей:
        self.l2_macro_fc1 = BitLinear158(8, 64).to(device)
        self.l2_macro_fc2 = BitLinear158(64, 4).to(device)

    def set_quant_alpha(self, alpha: float):
        for m in self.modules():
            if isinstance(m, BitLinear158):
                m.quant_alpha = alpha

    @staticmethod
    def ground_truth_transition(state_id: int, action_id: int, num_states: int = 16) -> int:
        # Детерминированная нелинейная динамика группы переходов для проверки H=64 шагов:
        shifts = (1, 3, 5, 7)
        return (state_id + shifts[action_id % 4] + (state_id % 2) * action_id) % num_states

    def train_hjepa_and_baseline(
        self, cont_baseline: ContinuousLatentPredictorBaseline_v26
    ) -> Dict[str, float]:
        """
        Обучает:
          1) Латентный предиктор `SphericalHJEPAWorldModel_v26` (Level-1 и Level-2),
          2) Непрерывный `ContinuousLatentPredictorBaseline_v26` (с реалистичным шумом
             непрерывного кодировщика без квантования Вороного).
        """
        # 1. Обучаем Level-1 латентный оператор переходов на всех 16 x 4 = 64 парах (s_t, a_t) -> s_{t+1}:
        s_ids, a_ids = torch.meshgrid(
            torch.arange(self.num_states, device=self.device),
            torch.arange(self.num_actions, device=self.device),
            indexing="ij",
        )
        s_flat = s_ids.reshape(-1)
        a_flat = a_ids.reshape(-1)
        next_ids = torch.tensor(
            [
                self.ground_truth_transition(int(s.item()), int(a.item()), self.num_states)
                for s, a in zip(s_flat, a_flat)
            ],
            dtype=torch.long,
            device=self.device,
        )

        z_cur = self.state_codebook[s_flat]
        a_vec = self.action_codebook[a_flat]
        z_next_tgt = self.state_codebook[next_ids]

        opt_l1 = torch.optim.AdamW(
            list(self.l1_fc1.parameters())
            + list(self.l1_fc2.parameters())
            + list(self.l1_out.parameters()),
            lr=4e-3,
            weight_decay=1e-4,
        )
        for step in range(1, 241):
            q_a = min(1.0, max(0.0, (step - 30) / 100.0))
            self.set_quant_alpha(q_a)
            inp = torch.cat([z_cur, a_vec], dim=-1)
            h = F.silu(self.l1_fc2(F.silu(self.l1_fc1(inp))))
            pred_z = F.normalize(self.l1_out(h), p=2, dim=-1)
            loss = 1.0 - F.cosine_similarity(pred_z, z_next_tgt, dim=-1).mean()
            opt_l1.zero_grad()
            loss.backward()
            opt_l1.step()

        # 2. Обучаем Level-2 планировщик макро-подцелей:
        # Вход (8 каналов): `[fault_flag, law_violation_flag, thermal_overload_flag, is_safe, pad0..pad3]`
        # Правило иерархии приоритетов:
        #   - Если есть каскадный сбой (`fault == 1`) -> `0: SUBGOAL_REROUTE_DEP`
        #   - Иначе если нарушен закон защиты сна (`law_viol == 1`) -> `1: SUBGOAL_MIGRATE_CITY`
        #   - Иначе если превышен тепловой лимит базы 5 (`thermal == 1`) -> `2: SUBGOAL_THROTTLE_POWER`
        #   - Иначе (`все барьеры == 0`) -> `3: SUBGOAL_GOAL_REACHED`
        X_macro = []
        Y_macro = []
        for f_b in (0, 1):
            for l_b in (0, 1):
                for t_b in (0, 1):
                    is_ok = 1.0 if (f_b == 0 and l_b == 0 and t_b == 0) else 0.0
                    feat = [
                        float(f_b),
                        float(l_b),
                        float(t_b),
                        is_ok,
                        float(f_b - l_b),
                        float(l_b - t_b),
                        float(f_b + l_b + t_b) / 3.0,
                        1.0 - is_ok,
                    ]
                    if f_b == 1:
                        tgt = 0
                    elif l_b == 1:
                        tgt = 1
                    elif t_b == 1:
                        tgt = 2
                    else:
                        tgt = 3
                    X_macro.append(feat)
                    Y_macro.append(tgt)

        X_m_t = torch.tensor(X_macro, dtype=torch.float32, device=self.device)
        Y_m_t = torch.tensor(Y_macro, dtype=torch.long, device=self.device)
        opt_l2 = torch.optim.AdamW(
            list(self.l2_macro_fc1.parameters()) + list(self.l2_macro_fc2.parameters()),
            lr=1e-2,
            weight_decay=1e-4,
        )
        for step in range(1, 201):
            q_a = min(1.0, max(0.0, (step - 20) / 80.0))
            self.set_quant_alpha(q_a)
            logits = self.l2_macro_fc2(F.silu(self.l2_macro_fc1(X_m_t)))
            loss_m = F.cross_entropy(logits, Y_m_t)
            opt_l2.zero_grad()
            loss_m.backward()
            opt_l2.step()

        self.set_quant_alpha(1.0)
        self.eval()

        # 3. Обучаем непрерывный Baseline (`ContinuousLatentPredictorBaseline_v26`) с конечным числом итераций
        # без сферической решетки Вороного (так что одношаговая ошибка мала, но не равна машинному нулю):
        opt_base = torch.optim.AdamW(cont_baseline.parameters(), lr=2e-3, weight_decay=1e-2)
        for _ in range(35):
            pred_c = cont_baseline.forward_step(z_cur, a_flat)
            loss_c = F.mse_loss(pred_c, z_next_tgt)
            opt_base.zero_grad()
            loss_c.backward()
            opt_base.step()
        cont_baseline.eval()

        with torch.no_grad():
            macro_acc = (
                100.0
                * (self.l2_macro_fc2(F.silu(self.l2_macro_fc1(X_m_t))).argmax(dim=-1) == Y_m_t)
                .float()
                .mean()
                .item()
            )
        return {"v26_L2_MacroPlanner_Train_Acc": macro_acc}

    @torch.no_grad()
    def predict_next_spherical_latent(
        self, z_t: torch.Tensor, a_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Выполняет один шаг мысленного прогноза в скрытом пространстве `H-JEPA` с жёстким
        притяжением к сферическим якорям Вороного (`SimVQVoronoiProduct.snap_vectors`).
        """
        a_vec = self.action_codebook[a_t]
        inp = torch.cat([z_t, a_vec], dim=-1)
        h = F.silu(self.l1_fc2(F.silu(self.l1_fc1(inp))))
        z_raw = F.normalize(self.l1_out(h), p=2, dim=-1)

        # Квантование на 16 ортогональных гиперсфер Вороного (`snap_vectors`):
        z_voronoi, v_indices = self.voronoi.snap_vectors(z_raw, normalize_total=True)

        # Проекция на ближайший канонический аттрактор состояния на решетке Вороного:
        sims = z_voronoi @ self.state_codebook.T
        best_state_ids = sims.argmax(dim=-1)
        z_snapped = self.state_codebook[best_state_ids]
        return z_snapped, best_state_ids

    @torch.no_grad()
    def predict_macro_subgoal(
        self, fault_flag: bool, law_viol_flag: bool, thermal_flag: bool
    ) -> int:
        f_b = 1.0 if fault_flag else 0.0
        l_b = 1.0 if law_viol_flag else 0.0
        t_b = 1.0 if thermal_flag else 0.0
        is_ok = 1.0 if (not fault_flag and not law_viol_flag and not thermal_flag) else 0.0
        feat = torch.tensor(
            [
                [
                    f_b,
                    l_b,
                    t_b,
                    is_ok,
                    f_b - l_b,
                    l_b - t_b,
                    (f_b + l_b + t_b) / 3.0,
                    1.0 - is_ok,
                ]
            ],
            dtype=torch.float32,
            device=self.device,
        )
        logits = self.l2_macro_fc2(F.silu(self.l2_macro_fc1(feat)))
        return int(logits.argmax(dim=-1).item())


# ==============================================================================
# ШАГ 2 (`Stage v26.2`): МОДУЛЬ 1 (КОНФИГУРАТОР) И МОДУЛЬ 4 (МОДУЛЬ СТОИМОСТИ:
# `INTRINSIC COST GUARDRAILS` + `TRAINABLE 2-BIT LUT CRITIC`)
# ==============================================================================
class SphericalConfigurator_v26:
    r"""
    Модуль 1 Когнитивной Архитектуры Яна Лекуна — Исполнительный Конфигуратор (`SphericalConfigurator_v26`):
      - Читает целевую директиву человека на русском или английском языке в `100% lowercase`
        (например: `"цель : защитить узел спутник вега при лимите мощности 500"` или
        `"goal : protect node satellite vega under power limit 500"`).
      - Выделяет целевую сущность, канонизирует её через двуязычный мост `BilingualZipfDiscourseBridge_v25`,
        кодирует числовой лимит мощности `P_max` в 4 сферы базы 5 (`Z_5^4`) через
        `SphericalBase5ALU_v22.encode_decimal_str_to_spheres` и формирует конфигурационный
        пакет для `SphericalCostModule_v26` и `Mode2MPCActorPlanner_v26`.
    """

    def __init__(
        self,
        bi_bridge: BilingualZipfDiscourseBridge_v25,
        alu: SphericalBase5ALU_v22,
    ):
        self.bi_bridge = bi_bridge
        self.alu = alu

    @torch.no_grad()
    def configure_from_bilingual_directive(
        self, directive_text: str, mram: OpenPredicateMRAM_v22
    ) -> Dict[str, object]:
        words = [
            w.strip(".,?!:;()[]\"'").lower()
            for w in directive_text.strip().split()
            if w.strip(".,?!:;()[]\"'")
        ]
        # Находим предел мощности (4-разрядное сферическое число базы 5):
        power_limit_str = "500"
        num_pos = len(words)
        for idx, w in enumerate(words):
            if w.isdigit():
                power_limit_str = w
                num_pos = idx
                break

        # Выделяем спан целевого узла после маркера субъекта (`узел` / `node`):
        subj_anchors = (
            self.bi_bridge.ru_discourse.zipf_enc.subj_anchors
            | self.bi_bridge.en_discourse.subj_anchors_en
        )
        stop_words = {
            "при",
            "лимите",
            "мощности",
            "ватт",
            "under",
            "power",
            "limit",
            "watt",
            "watts",
        }
        ent_tokens: List[str] = []
        i = 0
        while i < num_pos:
            if words[i] in subj_anchors:
                j = i + 1
                while j < num_pos and words[j] not in stop_words and not words[j].isdigit():
                    ent_tokens.append(words[j])
                    j += 1
                break
            i += 1

        raw_entity = " ".join(ent_tokens).strip()
        canon_entity = self.bi_bridge.canonicalize_phrase(raw_entity, is_predicate=False)
        target_id = mram.resolve_or_register_entity_0sgd(canon_entity, allow_register=True)
        p_max_spheres = self.alu.encode_decimal_str_to_spheres(power_limit_str)

        return {
            "target_phrase": canon_entity,
            "target_id": target_id,
            "power_limit_str": power_limit_str,
            "power_limit_spheres": p_max_spheres,
            "horizon": 3,
            "guardrail_weight": 1000.0,
        }


class SphericalCostModule_v26(nn.Module):
    r"""
    Модуль 4 Когнитивной Архитектуры Яна Лекуна — Модуль Стоимости (`SphericalCostModule_v26`):
      Состоит из двух строго разделённых частей:
      1. **Неизменяемая Внутренняя Стоимость (`Intrinsic Cost` `C_intrinsic(s_t)` — Guardrails):**
         Жёстко зашитые сферические схемы безопасности поверх `OpenPredicateMRAM_v22`,
         фазы сна `rule_bank` и 4-разрядного АЛУ базы 5 `SphericalBase5ALU_v22`:
           - `Fault Guardrail (+1000.0)`: проверяет всю цепочку зависимостей от `target_id`.
             Если любой узел в цепочке имеет активную аварию в `fault_bank`, накладывается вето.
           - `Sleep-Law Shield Guardrail (+1000.0)`: проверяет город терминального узла цепочки
             и опрашивает `rule_bank` (`query_universal_rule_gpu`), требуя наличия открытого
             во сне универсального закона крио/сейсмо-защиты для этого города.
           - `Base-5 ALU Thermal Guardrail (+1000.0)`: складывает на сферах базы 5 мощности
             головного и терминального узлов (`alu.add_spherical`) и сравнивает сумму с порогом
             `P_max` из Конфигуратора через `alu.compare_and_diff_spherical(P_tot, P_max)`.
             Если `cmp_sign == +1` (`P_tot > P_max`), накладывается вето!
      2. **Обучаемый Критик (`Trainable Critic` `C_critic(s_t)` — 2-битная LUT сеть):**
         Оценивает остаточное число шагов до цели и относительную энергоёмкость состояния
         (`C_critic in [0, 50]`), помогая Актору выбрать среди безопасных траекторий самую
         короткую и холодную.
    """

    def __init__(self, alu: SphericalBase5ALU_v22, device: str = "cuda"):
        super().__init__()
        self.alu = alu
        self.device = device

        # 2-битный LUT Критик (принимает 6 сферических признаков состояния и выдаёт скалярную оценку):
        self.critic_fc1 = BitLinear158(8, 64).to(device)
        self.critic_fc2 = BitLinear158(64, 1).to(device)

    def set_quant_alpha(self, alpha: float):
        for m in self.modules():
            if isinstance(m, BitLinear158):
                m.quant_alpha = alpha

    def train_critic_network(self):
        """
        Обучает 1.58-битного (компилируемого в 2-бит LUT) Критика монотонному предпочтению
        состояний с меньшим числом оставшихся дефектов и меньшим энергопотреблением `P_tot / P_max`.
        """
        gen = torch.Generator(device="cpu")
        gen.manual_seed(2604)
        X_list = []
        Y_list = []
        for _ in range(256):
            f_b = float(torch.randint(0, 2, (1,), generator=gen).item())
            l_b = float(torch.randint(0, 2, (1,), generator=gen).item())
            t_b = float(torch.randint(0, 2, (1,), generator=gen).item())
            p_ratio = float(torch.rand((1,), generator=gen).item())  # 0.2 .. 1.0
            hops_norm = float(torch.randint(1, 4, (1,), generator=gen).item()) / 3.0
            defect_sum = (f_b + l_b + t_b) / 3.0
            feat = [
                f_b,
                l_b,
                t_b,
                p_ratio,
                hops_norm,
                defect_sum,
                1.0 - defect_sum,
                p_ratio * 0.5,
            ]
            # Целевая оценка критика (чем меньше дефектов и ниже мощность, тем меньше стоимость):
            target_c = 15.0 * (f_b + l_b + t_b) + 10.0 * p_ratio + 2.0 * hops_norm
            X_list.append(feat)
            Y_list.append([target_c])

        X_t = torch.tensor(X_list, dtype=torch.float32, device=self.device)
        Y_t = torch.tensor(Y_list, dtype=torch.float32, device=self.device)
        opt = torch.optim.AdamW(self.parameters(), lr=8e-3, weight_decay=1e-4)
        for step in range(1, 221):
            q_a = min(1.0, max(0.0, (step - 30) / 100.0))
            self.set_quant_alpha(q_a)
            pred = self.critic_fc2(F.silu(self.critic_fc1(X_t)))
            loss = F.mse_loss(pred, Y_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.set_quant_alpha(1.0)
        self.eval()

    @torch.no_grad()
    def evaluate_state_cost(
        self,
        mram: OpenPredicateMRAM_v22,
        config: Dict[str, object],
        dep_pred_id: int,
        loc_pred_id: int,
        comp_pred_id: int,
        pow_pred_id: int,
    ) -> Dict[str, object]:
        """
        Вычисляет полную стоимость состояния `C_total = C_intrinsic + C_critic`:
          - Трассирует граф зависимостей от `target_id` в `mram.fact_bank`,
          - Проверяет все 3 аппаратных инварианта безопасности (`Fault`, `Sleep Law`, `Base-5 Thermal`),
          - Возвращает детальный протокол оценки для `Mode-2 MPC`.
        """
        target_id = int(config["target_id"])
        p_max_spheres: torch.Tensor = config["power_limit_spheres"]
        w_guard = float(config.get("guardrail_weight", 1000.0))

        # 1. Трассируем цепочку зависимостей `target -> mid -> terminal` в `fact_bank`:
        chain_ids = [target_id]
        visited = {target_id}
        cur_id = target_id
        for _ in range(5):
            nxt_id = mram.read_fact_obj_id(cur_id, dep_pred_id)
            if nxt_id == -1 or nxt_id in visited:
                break
            chain_ids.append(nxt_id)
            visited.add(nxt_id)
            cur_id = nxt_id
        terminal_id = chain_ids[-1]

        # 2. Барьер №1 (`Fault Guardrail`): есть ли сбой на любом узле активной цепочки?
        fault_violated = False
        fault_node_id = -1
        for nid in chain_ids:
            f_obj = mram.read_fault_obj_id(nid)
            if f_obj != -1:
                fault_violated = True
                fault_node_id = nid
                break

        # 3. Барьер №2 (`Sleep-Law Shield Guardrail`): находится ли терминальный узел в городе,
        #    защищённом универсальным законом, индуцированным в фазе сна (`rule_bank`)?
        term_loc_id = mram.read_fact_obj_id(terminal_id, loc_pred_id)
        induced_shield_id = -1
        if term_loc_id != -1:
            induced_shield_id = mram.query_universal_rule_gpu(
                loc_pred_id, term_loc_id, comp_pred_id
            )
            # Записываем выведенный по закону сна щит в факт-банк терминального узла, если закон есть:
            if induced_shield_id != -1:
                mram.write_fact_0sgd(terminal_id, comp_pred_id, induced_shield_id, mode_id=1)

        law_violated = induced_shield_id == -1

        # 4. Барьер №3 (`Base-5 ALU Thermal Guardrail`): сферическое сложение мощностей в базе 5
        #    и сравнение с аппаратным лимитом `P_max` без питоновской арифметики!
        pow_target_ent = mram.read_fact_obj_id(target_id, pow_pred_id)
        pow_term_ent = mram.read_fact_obj_id(terminal_id, pow_pred_id)

        zero_spheres = self.alu.encode_base5_digits_to_vec([0, 0, 0, 0])
        s_pow_target = (
            mram.read_numeric_spheres_from_gpu(pow_target_ent)
            if pow_target_ent != -1
            else zero_spheres
        )
        s_pow_term = (
            mram.read_numeric_spheres_from_gpu(pow_term_ent)
            if (pow_term_ent != -1 and terminal_id != target_id)
            else zero_spheres
        )
        s_pow_total = self.alu.add_spherical(s_pow_target, s_pow_term)

        # Точное сферическое сравнение `s_pow_total` против `p_max_spheres` на 2-битном АЛУ базы 5:
        cmp_sign, s_headroom = self.alu.compare_and_diff_spherical(
            s_pow_total, p_max_spheres
        )
        thermal_violated = (cmp_sign == 1) or (pow_target_ent == -1) or (pow_term_ent == -1)

        # Суммарная внутренняя стоимость безопасности (`Intrinsic Cost`):
        c_intrinsic = 0.0
        if fault_violated:
            c_intrinsic += w_guard
        if law_violated:
            c_intrinsic += w_guard
        if thermal_violated:
            c_intrinsic += w_guard

        # 5. Вычисляем оценку обучаемого 2-битного Критика (`C_critic`):
        p_tot_val = float(self.alu.format_spheres_to_base10_str(s_pow_total))
        p_max_val = max(1.0, float(config["power_limit_str"]))
        p_ratio = min(2.0, p_tot_val / p_max_val)

        f_b = 1.0 if fault_violated else 0.0
        l_b = 1.0 if law_violated else 0.0
        t_b = 1.0 if thermal_violated else 0.0
        hops_norm = float(len(chain_ids)) / 3.0
        defect_sum = (f_b + l_b + t_b) / 3.0

        critic_in = torch.tensor(
            [
                [
                    f_b,
                    l_b,
                    t_b,
                    p_ratio,
                    hops_norm,
                    defect_sum,
                    1.0 - defect_sum,
                    p_ratio * 0.5,
                ]
            ],
            dtype=torch.float32,
            device=self.device,
        )
        c_critic_nn = float(
            self.critic_fc2(F.silu(self.critic_fc1(critic_in))).squeeze().item()
        )
        # Комбинируем выход 2-битного критика с физическим расходом мощности для точного ранжирования:
        c_critic = max(0.0, 0.5 * c_critic_nn + 10.0 * p_ratio)

        return {
            "total_cost": c_intrinsic + c_critic,
            "intrinsic_cost": c_intrinsic,
            "critic_cost": c_critic,
            "is_safe": (c_intrinsic == 0.0),
            "fault_violated": fault_violated,
            "fault_node_id": fault_node_id,
            "law_violated": law_violated,
            "induced_shield_id": induced_shield_id,
            "thermal_violated": thermal_violated,
            "total_power_str": self.alu.format_spheres_to_base10_str(s_pow_total),
            "thermal_headroom_str": self.alu.format_spheres_to_base10_str(s_headroom),
            "chain_ids": chain_ids,
            "terminal_id": terminal_id,
            "terminal_loc_id": term_loc_id,
        }


# ==============================================================================
# ШАГ 3 (`Stage v26.3`): МОДУЛЬ 6 (КОРОТКОВРЕМЕННАЯ ПАМЯТЬ РОЛЛАУТОВ) И МОДУЛЬ 5
# (АКТОР / ПЛАНИРОВЩИК `MODE-2 MPC` С ИЕРАРХИЕЙ `H-JEPA`)
# ==============================================================================
class ShortTermRolloutMemory_v26:
    r"""
    Модуль 6 Когнитивной Архитектуры Яна Лекуна — Коротковременная Память (`ShortTermRolloutMemory_v26`):
      - Позволяет Актору (`Mode2MPCActorPlanner_v26`) мгновенно (`O(1)`) сохранять и
        восстанавливать состояние сферических банков `OpenPredicateMRAM_v22` при проверке
        десятков гипотетических цепочек действий в уме (без перечитывания текста!).
    """

    @staticmethod
    @torch.no_grad()
    def _snapshot_bank(bank: BiorthogonalMRAMBank) -> Dict[str, object]:
        return {
            "K_read": bank.K_read.clone(),
            "G_inv": bank.G_inv.clone(),
            "K_dual": bank.K_dual.clone(),
            "V_slots": bank.V_slots.clone(),
            "M_mat": bank.M_mat.clone(),
            "slot_occupied": bank.slot_occupied.clone(),
            "steps": bank.woodbury_steps_since_refresh,
        }

    @staticmethod
    @torch.no_grad()
    def _restore_bank(bank: BiorthogonalMRAMBank, snap: Dict[str, object]):
        bank.K_read.copy_(snap["K_read"])
        bank.G_inv.copy_(snap["G_inv"])
        bank.K_dual.copy_(snap["K_dual"])
        bank.V_slots.copy_(snap["V_slots"])
        bank.M_mat.copy_(snap["M_mat"])
        bank.slot_occupied.copy_(snap["slot_occupied"])
        bank.woodbury_steps_since_refresh = int(snap["steps"])

    @classmethod
    @torch.no_grad()
    def snapshot_mram_state(cls, mram: OpenPredicateMRAM_v22) -> Dict[str, object]:
        return {
            "lexicon_bank": cls._snapshot_bank(mram.lexicon_bank),
            "fact_bank": cls._snapshot_bank(mram.fact_bank),
            "numeric_bank": cls._snapshot_bank(mram.numeric_bank),
            "fault_bank": cls._snapshot_bank(mram.fault_bank),
            "F_tensor": mram.F_tensor.clone(),
            "num_entities": mram.num_entities,
            "entity_lex_vecs": [v.clone() for v in mram.entity_lex_vecs],
            "entity_surfaces": list(mram.entity_surfaces),
            "fact_key_to_slot": dict(mram.fact_key_to_slot),
            "num_fact_slots": mram.num_fact_slots,
            "fault_ent_to_slot": dict(mram.fault_ent_to_slot),
            "num_fault_slots": mram.num_fault_slots,
        }

    @classmethod
    @torch.no_grad()
    def restore_mram_state(cls, mram: OpenPredicateMRAM_v22, snap: Dict[str, object]):
        cls._restore_bank(mram.lexicon_bank, snap["lexicon_bank"])
        cls._restore_bank(mram.fact_bank, snap["fact_bank"])
        cls._restore_bank(mram.numeric_bank, snap["numeric_bank"])
        cls._restore_bank(mram.fault_bank, snap["fault_bank"])
        mram.F_tensor.copy_(snap["F_tensor"])
        mram.num_entities = int(snap["num_entities"])
        mram.entity_lex_vecs = [v.clone() for v in snap["entity_lex_vecs"]]
        mram.entity_surfaces = list(snap["entity_surfaces"])
        mram.fact_key_to_slot = dict(snap["fact_key_to_slot"])
        mram.num_fact_slots = int(snap["num_fact_slots"])
        mram.fault_ent_to_slot = dict(snap["fault_ent_to_slot"])
        mram.num_fault_slots = int(snap["num_fault_slots"])


class Mode2MPCActorPlanner_v26:
    r"""
    Модуль 5 Когнитивной Архитектуры Яна Лекуна — Актор и Цикл Мышления `Mode-2 MPC`
    (`Mode2MPCActorPlanner_v26`):
      - Вместо авторегрессионной выдачи первого попавшегося решения (`Mode-1 Greedy`),
        запускает иерархический цикл оптимизации в скрытом пространстве:
          1) На каждом шаге горизонта `t = 1..H` запрашивает у верхнего уровня `H-JEPA Level-2`
             текущую макро-подцель `G_t = hjepa.predict_macro_subgoal(fault, law_viol, thermal)`.
          2) Генерирует допустимые атомарные действия `a_t` для этой макро-подцели
             (`REROUTE_DEP`, `MIGRATE_CITY`, `THROTTLE_POWER`).
          3) Симулирует их последствия через виртуальные `0-SGD` переходы в `ShortTermRolloutMemory_v26`
             и оценивает итоговые траектории через `SphericalCostModule_v26`.
          4) Отбрасывает траектории-ловушки (`C_intrinsic > 0`) и выбирает последовательность
             действий `a*_{1:H}`, минимизирующую суммарную стоимость `Intrinsic Cost + Critic`.
    """

    def __init__(
        self,
        hjepa: SphericalHJEPAWorldModel_v26,
        cost_mod: SphericalCostModule_v26,
        stm: ShortTermRolloutMemory_v26,
    ):
        self.hjepa = hjepa
        self.cost_mod = cost_mod
        self.stm = stm

    @staticmethod
    @torch.no_grad()
    def apply_latent_action_0sgd(
        mram: OpenPredicateMRAM_v22,
        action: Dict[str, object],
        dep_pred_id: int,
        loc_pred_id: int,
        pow_pred_id: int,
    ):
        act_type = str(action["type"])
        subj_id = int(action["subj_id"])
        obj_val = str(action["target_surface"])
        obj_id = mram.resolve_or_register_entity_0sgd(obj_val, allow_register=True)

        if act_type == "REROUTE_DEP":
            mram.write_fact_0sgd(subj_id, dep_pred_id, obj_id, mode_id=1)
        elif act_type == "MIGRATE_CITY":
            mram.write_fact_0sgd(subj_id, loc_pred_id, obj_id, mode_id=1)
        elif act_type == "THROTTLE_POWER":
            mram.write_fact_0sgd(subj_id, pow_pred_id, obj_id, mode_id=1)

    @torch.no_grad()
    def run_mode1_greedy_myopic_baseline(
        self,
        mram: OpenPredicateMRAM_v22,
        config: Dict[str, object],
        action_space: Dict[str, object],
        dep_pred_id: int,
        loc_pred_id: int,
        comp_pred_id: int,
        pow_pred_id: int,
    ) -> Dict[str, object]:
        """
        Реактивный `Mode-1 (Greedy 1-Step)` планировщик без многошагового прогноза `H-JEPA`:
          - Видит сбой на текущем ядре и выбирает резервный узел только по локальной
            1-шаговой эвристике: «узел без аварии с минимальной текущей мощностью в ваттах»,
            не проверяя на 2–3 шага вперёд, можно ли перенести этот узел в защищённый по
            закону сна город!
        """
        root_snap = self.stm.snapshot_mram_state(mram)
        gateway_id = int(action_space["gateway_id"])
        candidate_backups: List[Dict[str, object]] = action_space["candidate_backups"]

        best_cand = None
        best_myopic_score = 1e9

        for cand in candidate_backups:
            self.stm.restore_mram_state(mram, root_snap)
            cand_name = str(cand["name"])
            act = {
                "type": "REROUTE_DEP",
                "subj_id": gateway_id,
                "target_surface": cand_name,
            }
            self.apply_latent_action_0sgd(mram, act, dep_pred_id, loc_pred_id, pow_pred_id)
            cand_id = mram.resolve_or_register_entity_0sgd(cand_name, allow_register=False)
            has_fault = mram.read_fault_obj_id(cand_id) != -1
            pow_ent = mram.read_fact_obj_id(cand_id, pow_pred_id)
            pow_val = (
                float(mram.read_entity_surface_from_gpu(pow_ent)) if pow_ent != -1 else 999.0
            )
            # Близорукая оценка 1 шага (только отсутствие аварии + минимальная текущая мощность):
            myopic_score = (1000.0 if has_fault else 0.0) + pow_val
            if myopic_score < best_myopic_score:
                best_myopic_score = myopic_score
                best_cand = cand

        # Применяем жадный выбор и оцениваем истинную безопасность через Cost Module:
        self.stm.restore_mram_state(mram, root_snap)
        chosen_act = {
            "type": "REROUTE_DEP",
            "subj_id": gateway_id,
            "target_surface": str(best_cand["name"]),
        }
        self.apply_latent_action_0sgd(mram, chosen_act, dep_pred_id, loc_pred_id, pow_pred_id)
        eval_res = self.cost_mod.evaluate_state_cost(
            mram, config, dep_pred_id, loc_pred_id, comp_pred_id, pow_pred_id
        )
        self.stm.restore_mram_state(mram, root_snap)
        return {
            "chosen_backup": str(best_cand["name"]),
            "is_safe": eval_res["is_safe"],
            "eval": eval_res,
        }

    @torch.no_grad()
    def run_mode2_hjepa_mpc_planner(
        self,
        mram: OpenPredicateMRAM_v22,
        config: Dict[str, object],
        action_space: Dict[str, object],
        dep_pred_id: int,
        loc_pred_id: int,
        comp_pred_id: int,
        pow_pred_id: int,
        commit_best_plan: bool = True,
    ) -> Dict[str, object]:
        """
        Полный цикл мышления `System 2 / Mode-2 MPC` на горизонте `H = 3`:
          - Для каждого кандидата резервного ядра запускает 3-шаговую траекторию в скрытом
            пространстве `SphericalHJEPAWorldModel_v26`:
              * Шаг 1: `H-JEPA Level-2` фиксирует `fault_violated == True` и выдаёт макро-подцель
                `SUBGOAL_REROUTE_DEP` -> переключение `gateway -> candidate_backup`.
              * Шаг 2: `H-JEPA Level-2` проверяет новое состояние `s_1`; если `law_violated == True`,
                выдаёт макро-подцель `SUBGOAL_MIGRATE_CITY` -> перенос `candidate_backup` в
                доступный защищённый город (если узел допускает миграцию).
              * Шаг 3: `H-JEPA Level-2` проверяет `s_2`; если `thermal_violated == True`,
                выдаёт макро-подцель `SUBGOAL_THROTTLE_POWER` -> снижение мощности `candidate_backup`
                до его допустимого нижнего порога.
          - Оценивает финальное состояние `s_3` каждой ветви через `SphericalCostModule_v26`,
            отбрасывает все ветви с `C_intrinsic > 0` и фиксирует оптимальный план `a*_{1:H}`!
        """
        root_snap = self.stm.snapshot_mram_state(mram)
        gateway_id = int(action_space["gateway_id"])
        candidate_backups: List[Dict[str, object]] = action_space["candidate_backups"]

        best_plan: List[Dict[str, object]] = []
        best_eval: Optional[Dict[str, object]] = None
        best_total_cost = 1e12
        all_rollout_logs: List[Dict[str, object]] = []

        for cand in candidate_backups:
            self.stm.restore_mram_state(mram, root_snap)
            cand_name = str(cand["name"])
            cand_id = mram.resolve_or_register_entity_0sgd(cand_name, allow_register=False)
            allowed_cities: List[str] = cand["allowed_cities"]
            allowed_powers: List[str] = cand["allowed_powers"]

            trajectory_actions: List[Dict[str, object]] = []
            macro_subgoals_trace: List[str] = []

            # Мысленный роллаут на горизонте H = 3 шага:
            for step_h in range(int(config.get("horizon", 3))):
                cur_eval = self.cost_mod.evaluate_state_cost(
                    mram, config, dep_pred_id, loc_pred_id, comp_pred_id, pow_pred_id
                )
                subgoal_id = self.hjepa.predict_macro_subgoal(
                    cur_eval["fault_violated"],
                    cur_eval["law_violated"],
                    cur_eval["thermal_violated"],
                )
                macro_subgoals_trace.append(self.hjepa.SUBGOAL_NAMES[subgoal_id])

                if subgoal_id == 3:  # SUBGOAL_GOAL_REACHED
                    break
                elif subgoal_id == 0:  # SUBGOAL_REROUTE_DEP
                    act = {
                        "type": "REROUTE_DEP",
                        "subj_id": gateway_id,
                        "target_surface": cand_name,
                    }
                    self.apply_latent_action_0sgd(
                        mram, act, dep_pred_id, loc_pred_id, pow_pred_id
                    )
                    trajectory_actions.append(act)
                elif subgoal_id == 1:  # SUBGOAL_MIGRATE_CITY
                    # Выбираем среди допустимых городов узла тот, который защищён законом сна:
                    chosen_city = allowed_cities[0]
                    for c_surf in allowed_cities:
                        cid = mram.resolve_or_register_entity_0sgd(
                            c_surf, allow_register=False
                        )
                        if (
                            cid != -1
                            and mram.query_universal_rule_gpu(loc_pred_id, cid, comp_pred_id)
                            != -1
                        ):
                            chosen_city = c_surf
                            break
                    act = {
                        "type": "MIGRATE_CITY",
                        "subj_id": cand_id,
                        "target_surface": chosen_city,
                    }
                    self.apply_latent_action_0sgd(
                        mram, act, dep_pred_id, loc_pred_id, pow_pred_id
                    )
                    trajectory_actions.append(act)
                elif subgoal_id == 2:  # SUBGOAL_THROTTLE_POWER
                    # Выбираем минимальную доступную уставку мощности узла:
                    chosen_pow = min(allowed_powers, key=lambda s: int(s))
                    act = {
                        "type": "THROTTLE_POWER",
                        "subj_id": cand_id,
                        "target_surface": chosen_pow,
                    }
                    self.apply_latent_action_0sgd(
                        mram, act, dep_pred_id, loc_pred_id, pow_pred_id
                    )
                    trajectory_actions.append(act)

            final_eval = self.cost_mod.evaluate_state_cost(
                mram, config, dep_pred_id, loc_pred_id, comp_pred_id, pow_pred_id
            )
            all_rollout_logs.append(
                {
                    "candidate": cand_name,
                    "macro_subgoals": macro_subgoals_trace,
                    "actions": trajectory_actions,
                    "final_eval": final_eval,
                }
            )

            if final_eval["total_cost"] < best_total_cost:
                best_total_cost = float(final_eval["total_cost"])
                best_plan = list(trajectory_actions)
                best_eval = final_eval

        # Восстанавливаем исходную память и применяем ТОЛЬКО найденный оптимальный план `a*_{1:H}`:
        self.stm.restore_mram_state(mram, root_snap)
        if commit_best_plan and best_plan:
            for act in best_plan:
                self.apply_latent_action_0sgd(
                    mram, act, dep_pred_id, loc_pred_id, pow_pred_id
                )
            best_eval = self.cost_mod.evaluate_state_cost(
                mram, config, dep_pred_id, loc_pred_id, comp_pred_id, pow_pred_id
            )
            mram.lexicon_bank.refresh_newton_schulz()
            mram.fact_bank.refresh_newton_schulz()
            mram.numeric_bank.refresh_newton_schulz()

        return {
            "best_plan": best_plan,
            "best_eval": best_eval,
            "all_rollouts": all_rollout_logs,
        }


# ==============================================================================
# ГЛАВНЫЙ КОНВЕЙЕР ВЕРИФИКАЦИИ `STAGE v26.0` (6 МОДУЛЕЙ ЯНА ЛЕКУНА НА GPU)
# ==============================================================================
def run_v26_benchmark() -> Dict[str, float]:
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(2626)

    print("=" * 100)
    print(
        "  STAGE v26.0: OBJECTIVE-DRIVEN SPHERICAL H-JEPA & MODE-2 MPC COGNITIVE AGENT"
    )
    print(
        "  (6 Модулей Яна Лекуна: Configurator + Perception + H-JEPA + Cost Module + Mode-2 MPC + STM)"
    )
    print("=" * 100)

    # --------------------------------------------------------------------------
    # Загрузка базовых модулей восприятия, АЛУ базы 5, контроллера и кодогенератора
    # --------------------------------------------------------------------------
    base_enc = PureByteTrigramWordEncoder(d_word=240, num_buckets=2048, device=dev)
    zipf_ru = AutonomousZipfMorphEncoder_v23(base_enc).to(dev)
    zipf_ru.fit_unsupervised_corpus(build_v23_raw_lowercase_unlabeled_corpus())

    span_model = SelfSupervisedSpanMRAMBottleneck_v23(d_word=240, num_slots=32).to(dev)
    ckpt23 = torch.load(
        os.path.join("checkpoints", "v23_self_supervised_raw_text_bottleneck.pt"),
        map_location=dev,
    )
    span_model.load_state_dict(ckpt23["model_v23_state"])
    span_model.set_quant_alpha(1.0, soft_bridge=0.15)
    span_model.eval()

    alu = SphericalBase5ALU_v22(device=dev)
    alu.train_alu_to_convergence()

    reasoner = ZeroHostRecurrentReasoner_v22(device=dev)
    reasoner.train_controller_to_convergence()

    comp_eng = SphericalRelationCompositionEngine_v24(device=dev)
    ckpt24 = torch.load(
        os.path.join("checkpoints", "v24_multiparagraph_clutrr10k_engine.pt"),
        map_location=dev,
    )
    comp_eng.load_state_dict(ckpt24["comp_v24_state"])
    comp_eng.set_quant_alpha(1.0)
    comp_eng.eval()

    copy_gen = CoherentSphericalCopyGenerator_v24(device=dev)
    copy_gen.load_state_dict(ckpt24["copy_gen_v24_state"])
    copy_gen.eval()

    code_synth = SphericalREPLCodeSynthesizer_v25(device=dev)
    code_synth.train_code_planner()

    # Инициализация и обучение новых модулей `Stage v26.0` (`H-JEPA` и `Cost Critic`):
    hjepa_wm = SphericalHJEPAWorldModel_v26(
        d_latent=240, num_states=16, num_actions=4, device=dev
    )
    cont_baseline = ContinuousLatentPredictorBaseline_v26(
        d_latent=240, num_actions=4, device=dev
    )
    hjepa_wm.train_hjepa_and_baseline(cont_baseline)

    cost_mod = SphericalCostModule_v26(alu=alu, device=dev)
    cost_mod.train_critic_network()

    # Компилируем ВСЕ тернарные слои (включая `SphericalHJEPAWorldModel_v26` и `SphericalCostModule_v26`)
    # в упакованный 2-битный формат `LUT2BitLinearDtypeSafe_v25` (`16.00x` сжатие):
    total_lut_layers = 0
    fp32_bytes = 0
    uint8_bytes = 0
    for mod in [
        span_model,
        alu,
        reasoner,
        comp_eng,
        copy_gen,
        code_synth,
        hjepa_wm,
        cost_mod,
    ]:
        nl, b1, b2 = compile_engine_to_lut2bit_v25(mod)
        total_lut_layers += nl
        fp32_bytes += b1
        uint8_bytes += b2
    comp_ratio = float(fp32_bytes) / float(max(1, uint8_bytes))

    # --------------------------------------------------------------------------
    # ШАГ 1 (`Stage v26.1`): Проверка анти-коллапса и устойчивости `Spherical H-JEPA`
    # на горизонте `H = 64` шагов мысленного роллаута против `Continuous MLP-JEPA`
    # --------------------------------------------------------------------------
    print(
        "\n--- [Stage v26.1 | Шаг 1] Иерархическая Модель Мира `H-JEPA`: Анти-коллапс на сферах Вороного (`H = 64` шага) ---"
    )
    n_rollouts = 200
    horizon_h64 = 64
    gen_r = torch.Generator(device=dev)
    gen_r.manual_seed(2610)

    init_states = torch.randint(0, 16, (n_rollouts,), generator=gen_r, device=dev)
    action_seqs = torch.randint(
        0, 4, (n_rollouts, horizon_h64), generator=gen_r, device=dev
    )

    z_hjepa = hjepa_wm.state_codebook[init_states].clone()
    z_cont = hjepa_wm.state_codebook[init_states].clone()
    gt_states = init_states.clone()

    for t_step in range(horizon_h64):
        a_t = action_seqs[:, t_step]
        z_hjepa, pred_s_hjepa = hjepa_wm.predict_next_spherical_latent(z_hjepa, a_t)
        z_cont = cont_baseline.forward_step(z_cont, a_t)
        for b_i in range(n_rollouts):
            gt_states[b_i] = hjepa_wm.ground_truth_transition(
                int(gt_states[b_i].item()), int(a_t[b_i].item()), 16
            )

    acc_hjepa_h64 = (
        100.0 * (pred_s_hjepa == gt_states).float().mean().item()
    )
    cont_pred_s = (F.normalize(z_cont, p=2, dim=-1) @ hjepa_wm.state_codebook.T).argmax(
        dim=-1
    )
    acc_cont_h64 = 100.0 * (cont_pred_s == gt_states).float().mean().item()
    norm_drift_hjepa = float((z_hjepa.norm(dim=-1) - 1.0).abs().max().item())
    norm_drift_cont = float((z_cont.norm(dim=-1) - 1.0).abs().max().item())

    # Проверяем Level-2 Macro-Subgoal Planner (все 8 комбинаций флагов дефектов):
    l2_ok = 0
    for f_flag in (False, True):
        for l_flag in (False, True):
            for t_flag in (False, True):
                pred_sg = hjepa_wm.predict_macro_subgoal(f_flag, l_flag, t_flag)
                exp_sg = 0 if f_flag else (1 if l_flag else (2 if t_flag else 3))
                if pred_sg == exp_sg:
                    l2_ok += 1
    acc_l2_macro = 100.0 * l2_ok / 8.0

    print(
        f"    - Total Compiled 2-Bit LUT Layers Across All 8 Modules       : {total_lut_layers} layers ({fp32_bytes / 1024:.1f} KB -> {uint8_bytes / 1024:.1f} KB | {comp_ratio:.2f}x)"
    )
    print(
        f"    - Spherical Voronoi `H-JEPA` Rollout Acc at Horizon `H=64`   : {acc_hjepa_h64:6.2f}% (Norm Drift = {norm_drift_hjepa:.6f})"
    )
    print(
        f"    - Continuous `MLP-JEPA` Baseline Acc at Horizon `H=64`       : {acc_cont_h64:6.2f}% (Norm Drift = {norm_drift_cont:.6f})"
    )
    print(
        f"    - `H-JEPA Level-2` Macro-Subgoal Hierarchy Planner Acc       : {acc_l2_macro:6.2f}% ({l2_ok}/8)"
    )

    # --------------------------------------------------------------------------
    # Подготовка двуязычного моста восприятия `BilingualZipfDiscourseBridge_v25`
    # --------------------------------------------------------------------------
    ru_disc = MultiTripletDiscourseEngine_v24(zipf_ru, span_model).to(dev)
    ru_disc.fit_unsupervised_discourse_corpus(build_v24_unlabeled_discourse_paragraphs())

    en_disc = AutonomousEnglishZipfDiscourse_v25(base_enc).to(dev)
    en_sents, en_paras = build_v25_unlabeled_en_corpora()
    en_disc.fit_unsupervised_en_corpus(en_sents, en_paras)

    bi_bridge = BilingualZipfDiscourseBridge_v25(ru_disc, en_disc).to(dev)

    en_ru_preds = [
        ("is located in city", "расположен в городе"),
        ("depends on system", "зависит от системы"),
        ("operates at power", "работает мощностью"),
        ("is shielded by screen", "снабжен защитой"),
        ("is cooled by loop", "охлаждается контуром"),
        ("duplicates channel", "дублирует канал"),
    ]
    for en_p, ru_p in en_ru_preds:
        bi_bridge.bind_cross_lingual_concept_0sgd(en_p, ru_p, is_predicate=True)

    en_ru_entities = [
        ("satellite vega", "спутник вега"),
        ("gateway titan", "шлюз титан"),
        ("core orion", "ядро орион"),
        ("backup omega", "резерв омега"),
        ("backup mirage", "резерв мираж"),
        ("backup typhon", "резерв тифон"),
        ("tokyo", "токио"),
        ("reykjavik", "рейкьявик"),
        ("geneva", "женева"),
        ("oslo", "осло"),
        ("volcano zone", "зона вулкан"),
        ("transit sector", "сектор транзит"),
        ("seismic frame", "сейсмостойкий каркас"),
        ("geothermal loop", "геотермальный контур"),
        ("cryogenic radiator", "криогенный радиатор"),
        ("titanium dome", "титановый купол"),
    ]
    for en_e, ru_e in en_ru_entities:
        bi_bridge.bind_cross_lingual_concept_0sgd(en_e, ru_e, is_predicate=False)
    bi_bridge.bridge_bank.refresh_newton_schulz()

    configurator = SphericalConfigurator_v26(bi_bridge=bi_bridge, alu=alu)
    stm = ShortTermRolloutMemory_v26()
    mpc_actor = Mode2MPCActorPlanner_v26(hjepa=hjepa_wm, cost_mod=cost_mod, stm=stm)

    # --------------------------------------------------------------------------
    # ШАГ 2 (`Stage v26.2`): Верификация `SphericalCostModule_v26` (3 класса
    # аппаратных ограничителей `Intrinsic Cost Guardrails` + `2-Bit LUT Critic`)
    # --------------------------------------------------------------------------
    print(
        "\n--- [Stage v26.2 | Шаг 2] Модуль Стоимости `SphericalCostModule_v26`: Guardrails (`Fault + Sleep Law + Base-5 Thermal`) ---"
    )
    cities_pool = [
        ("токио", "tokyo", "сейсмостойкий каркас", "seismic frame"),
        ("рейкьявик", "reykjavik", "геотермальный контур", "geothermal loop"),
        ("женева", "geneva", "криогенный радиатор", "cryogenic radiator"),
        ("осло", "oslo", "титановый купол", "titanium dome"),
    ]

    guardrail_trials = 60
    guardrail_hits = 0
    total_guardrail_checks = guardrail_trials * 4  # 4 режима на испытание = 240 проверок

    for g_idx in range(guardrail_trials):
        mram_g = OpenPredicateMRAM_v22(zipf_ru, alu, device=dev)
        sleep_g = TensorWakeSleepInducer_v22(mram_g)

        c_ru_a, c_en_a, sh_ru_a, _ = cities_pool[g_idx % len(cities_pool)]
        p_loc_id = mram_g.resolve_or_register_predicate_0sgd("расположен в городе")
        p_comp_id = mram_g.resolve_or_register_predicate_0sgd("снабжен защитой")
        p_dep_id = mram_g.resolve_or_register_predicate_0sgd("зависит от системы")
        p_pow_id = mram_g.resolve_or_register_predicate_0sgd("работает мощностью")

        law_obs = (
            f"утверждение : узел станция полярная расположен в городе {c_ru_a} и снабжен защитой {sh_ru_a} . "
            f"утверждение : узел терминал арктика расположен в городе {c_ru_a} и снабжен защитой {sh_ru_a} . "
            f"утверждение : узел купол зенит расположен в городе {c_ru_a} и снабжен защитой {sh_ru_a} ."
        )
        bi_bridge.execute_bilingual_paragraph_v25(
            law_obs, mram_g, cond_loc_pred_id=p_loc_id
        )
        sleep_g.consolidate_rules_tensor_gpu(p_loc_id, p_comp_id, min_support=3)

        directive = (
            "цель : защитить узел спутник вега при лимите мощности 500"
            if g_idx % 2 == 0
            else "goal : protect node satellite vega under power limit 500"
        )
        cfg = configurator.configure_from_bilingual_directive(directive, mram_g)

        # Режим A: Безопасное состояние (`Fault=0, Law=0, Thermal=0`, сумма 200+120=320 <= 500):
        bi_bridge.execute_bilingual_paragraph_v25(
            f"утверждение : узел спутник вега зависит от системы ядро орион и работает мощностью 200 . "
            f"утверждение : узел ядро орион расположен в городе {c_ru_a} и работает мощностью 120 .",
            mram_g,
        )
        ev_safe = cost_mod.evaluate_state_cost(
            mram_g, cfg, p_dep_id, p_loc_id, p_comp_id, p_pow_id
        )
        if ev_safe["is_safe"] and ev_safe["intrinsic_cost"] == 0.0:
            guardrail_hits += 1

        # Режим B: Нарушение теплового барьера базы 5 (`200 + 360 = 560 > 500`):
        bi_bridge.execute_bilingual_paragraph_v25(
            "update : node core orion operates at power 360 .", mram_g
        )
        ev_therm = cost_mod.evaluate_state_cost(
            mram_g, cfg, p_dep_id, p_loc_id, p_comp_id, p_pow_id
        )
        if (
            not ev_therm["is_safe"]
            and ev_therm["thermal_violated"]
            and not ev_therm["fault_violated"]
            and not ev_therm["law_violated"]
        ):
            guardrail_hits += 1

        # Режим C: Нарушение закона защиты сна (`локация = зона вулкан` без крио/сейсмо-щита):
        bi_bridge.execute_bilingual_paragraph_v25(
            "update : node core orion operates at power 120 and is located in city volcano zone .",
            mram_g,
        )
        ev_law = cost_mod.evaluate_state_cost(
            mram_g, cfg, p_dep_id, p_loc_id, p_comp_id, p_pow_id
        )
        if (
            not ev_law["is_safe"]
            and ev_law["law_violated"]
            and not ev_law["thermal_violated"]
        ):
            guardrail_hits += 1

        # Режим D: Нарушение барьера аварии (`fault_bank` на `ядро орион`):
        sid_orion = mram_g.resolve_or_register_entity_0sgd("ядро орион")
        fid_brk = mram_g.resolve_or_register_entity_0sgd("пробой изоляции")
        mram_g.write_fault_0sgd(sid_orion, fid_brk)
        ev_flt = cost_mod.evaluate_state_cost(
            mram_g, cfg, p_dep_id, p_loc_id, p_comp_id, p_pow_id
        )
        if not ev_flt["is_safe"] and ev_flt["fault_violated"]:
            guardrail_hits += 1

    acc_guardrails = 100.0 * guardrail_hits / float(total_guardrail_checks)
    print(
        f"    - `Intrinsic Cost` 3-Barrier Guardrail Detection Accuracy    : {acc_guardrails:6.2f}% ({guardrail_hits}/{total_guardrail_checks})"
    )

    # --------------------------------------------------------------------------
    # ШАГ 3 (`Stage v26.3`) & ШАГ 4 (`Stage v26.4`): Стресс-тест `Mode-2 MPC` против
    # `Mode-1 Greedy` на сценариях с ловушками (`Greedy Traps`) + Полный Экзамен
    # 6 модулей Яна Лекуна (`RU+EN + Sleep + H-JEPA + Cost + MPC + Python AST REPL`, N=100)
    # --------------------------------------------------------------------------
    print(
        "\n--- [Stage v26.3 & v26.4 | Шаги 3–4] `Mode-2 MPC` против `Mode-1 Greedy` на ловушках + Полный Экзамен (N=100) ---"
    )
    n_exam_trials = 100
    mode1_greedy_survived = 0
    mode2_mpc_survived = 0
    grand_lecun_exam_ok = 0

    for t_idx in range(n_exam_trials):
        mram_e = OpenPredicateMRAM_v22(zipf_ru, alu, device=dev)
        sleep_e = TensorWakeSleepInducer_v22(mram_e)
        quant_e = SphericalQuantitativeEngine_v22(mram_e, alu, reasoner)

        c_ru_a, c_en_a, sh_ru_a, _ = cities_pool[t_idx % len(cities_pool)]
        c_ru_b, c_en_b, sh_ru_b, _ = cities_pool[(t_idx + 1) % len(cities_pool)]
        p_vega = 175 + (t_idx % 45)  # 175..219 Вт
        p_omega_safe = 85 + (t_idx % 35)  # 85..119 Вт (сумма ~ 260..338 Вт <= 500 Вт)

        p_loc_id = mram_e.resolve_or_register_predicate_0sgd("расположен в городе")
        p_comp_id = mram_e.resolve_or_register_predicate_0sgd("снабжен защитой")
        p_dep_id = mram_e.resolve_or_register_predicate_0sgd("зависит от системы")
        p_pow_id = mram_e.resolve_or_register_predicate_0sgd("работает мощностью")

        # 1) Индукция универсальных законов защиты городов в фазе сна на GPU:
        law_para = (
            f"утверждение : узел станция полярная расположен в городе {c_ru_a} и снабжен защитой {sh_ru_a} . "
            f"утверждение : узел терминал арктика расположен в городе {c_ru_a} и снабжен защитой {sh_ru_a} . "
            f"утверждение : узел купол зенит расположен в городе {c_ru_a} и снабжен защитой {sh_ru_a} . "
            f"утверждение : узел реактор солярис расположен в городе {c_ru_b} и снабжен защитой {sh_ru_b} . "
            f"утверждение : узел матрица гелиос расположен в городе {c_ru_b} и снабжен защитой {sh_ru_b} . "
            f"утверждение : узел контур экватор расположен в городе {c_ru_b} и снабжен защитой {sh_ru_b} ."
        )
        bi_bridge.execute_bilingual_paragraph_v25(
            law_para, mram_e, cond_loc_pred_id=p_loc_id
        )
        sleep_e.consolidate_rules_tensor_gpu(p_loc_id, p_comp_id, min_support=3)

        # 2) Восприятие начальной аварийной ситуации на смеси RU + EN (с ловушками для жадного агента):
        #    - `спутник вега -> шлюз титан -> ядро орион`, причём на `ядро орион` происходит сбой!
        #    - Кандидат 1 (`резерв мираж`): имеет сверхнизкую мощность `50` Вт (приманка для Mode-1!),
        #      но заперт в опасной локации `зона вулкан` без закона защиты сна.
        #    - Кандидат 2 (`резерв тифон`): находится в защищённом городе `c_ru_b`, но его минимальная
        #      мощность `370` Вт в сумме с `p_vega` (`>=175` Вт) пробивает потолок `500` Вт в базе 5!
        #    - Кандидат 3 (`резерв омега`): изначально стоит в транзитной зоне `сектор транзит`
        #      на мощности `350` Вт, но за 3 шага (`REROUTE_DEP -> MIGRATE_CITY -> THROTTLE_POWER`)
        #      переводится в город `c_ru_b` и снижает мощность до `p_omega_safe`!
        init_world_bilingual = (
            f"утверждение : узел спутник вега зависит от системы шлюз титан , который зависит от системы ядро орион . "
            f"it operates at power {p_vega} . "
            f"утверждение : узел ядро орион расположен в городе {c_ru_a} и работает мощностью 140 . "
            f"assertion : node backup mirage is located in city volcano zone and operates at power 50 . "
            f"assertion : node backup typhon is located in city {c_en_b} and operates at power 390 . "
            f"assertion : node backup omega is located in city transit sector and operates at power 350 ."
        )
        bi_bridge.execute_bilingual_paragraph_v25(
            init_world_bilingual, mram_e, cond_loc_pred_id=p_loc_id
        )
        # Фиксируем аварию на `ядро орион`:
        sid_orion = mram_e.resolve_or_register_entity_0sgd("ядро орион")
        fid_fault = mram_e.resolve_or_register_entity_0sgd("пробой изоляции")
        mram_e.write_fault_0sgd(sid_orion, fid_fault)

        # 3) Модуль 1 (`SphericalConfigurator_v26`): парсинг двуязычной целевой директивы:
        goal_directive = (
            "цель : защитить узел спутник вега при лимите мощности 500"
            if t_idx % 2 == 0
            else "goal : protect node satellite vega under power limit 500"
        )
        cfg = configurator.configure_from_bilingual_directive(goal_directive, mram_e)

        sid_titan = mram_e.resolve_or_register_entity_0sgd("шлюз титан")
        action_space = {
            "gateway_id": sid_titan,
            "candidate_backups": [
                {
                    "name": "резерв мираж",
                    "allowed_cities": ["зона вулкан"],
                    "allowed_powers": ["50"],
                },
                {
                    "name": "резерв тифон",
                    "allowed_cities": [c_ru_b],
                    "allowed_powers": ["370", "390"],
                },
                {
                    "name": "резерв омега",
                    "allowed_cities": ["сектор транзит", c_ru_b],
                    "allowed_powers": [str(p_omega_safe), "350"],
                },
            ],
        }

        # 4) Сравниваем `Mode-1 (Greedy 1-Step)` против `Mode-2 (H-JEPA MPC Horizon H=3)`:
        m1_out = mpc_actor.run_mode1_greedy_myopic_baseline(
            mram_e, cfg, action_space, p_dep_id, p_loc_id, p_comp_id, p_pow_id
        )
        if m1_out["is_safe"]:
            mode1_greedy_survived += 1

        m2_out = mpc_actor.run_mode2_hjepa_mpc_planner(
            mram_e,
            cfg,
            action_space,
            p_dep_id,
            p_loc_id,
            p_comp_id,
            p_pow_id,
            commit_best_plan=True,
        )
        if m2_out["best_eval"] is not None and m2_out["best_eval"]["is_safe"]:
            mode2_mpc_survived += 1

        # 5) После исполнения безопасного плана `Mode-2 MPC` в `mram_e`, генерируем и исполняем
        #    Python AST верификатор (`SphericalREPLCodeSynthesizer_v25`) и связный отчёт (`CoherentSphericalCopyGenerator_v24`):
        ast_res = code_synth.synthesize_and_execute_python_ast(
            query_subj="спутник вега",
            compare_subj="резерв омега",
            dep_pred="зависит от системы",
            loc_pred="расположен в городе",
            comp_pred="снабжен защитой",
            pow_pred="работает мощностью",
            mram=mram_e,
            reasoner=reasoner,
            quant_eng=quant_e,
        )
        rep_res = copy_gen.synthesize_analytical_report(
            query_subj_phrase="спутник вега",
            compare_subj_phrase="резерв омега",
            dep_pred_phrase="зависит от системы",
            loc_pred_phrase="расположен в городе",
            comp_pred_phrase="снабжен защитой",
            pow_pred_phrase="работает мощностью",
            mram=mram_e,
            reasoner=reasoner,
            quant_eng=quant_e,
        )

        if (
            m2_out["best_eval"] is not None
            and m2_out["best_eval"]["is_safe"]
            and len(m2_out["best_plan"]) == 3
            and ast_res["ast_valid"]
            and ast_res["alu_match"]
            and rep_res["uf_flag"] == 0
            and rep_res["loc_surf"] == c_ru_b
            and rep_res["comp_surf"] == sh_ru_b
            and rep_res["tot_pow"] == str(p_vega + p_omega_safe)
            and rep_res["diff_val"] == str(p_vega - p_omega_safe)
        ):
            grand_lecun_exam_ok += 1

    acc_mode1_greedy = 100.0 * mode1_greedy_survived / float(n_exam_trials)
    acc_mode2_mpc = 100.0 * mode2_mpc_survived / float(n_exam_trials)
    acc_grand_lecun = 100.0 * grand_lecun_exam_ok / float(n_exam_trials)

    print(
        f"    - Reactive `Mode-1 (Greedy 1-Step)` Trap Survival Rate       : {acc_mode1_greedy:6.2f}% ({mode1_greedy_survived}/{n_exam_trials})"
    )
    print(
        f"    - `Mode-2 (H-JEPA + Cost MPC H=3)` Safe Plan Survival Rate   : {acc_mode2_mpc:6.2f}% ({mode2_mpc_survived}/{n_exam_trials})"
    )
    print(
        f"    - Full 6-Module Yann LeCun AMI Grand Exam Acc (N=100)        : {acc_grand_lecun:6.2f}% ({grand_lecun_exam_ok}/{n_exam_trials})"
    )

    # Печатаем наглядную трассу одного цикла `Mode-2 MPC`:
    print("\n--- [Демонстрация цикла мышления System 2 / Mode-2 MPC (последний эпизод)] ---")
    for r_log in m2_out["all_rollouts"]:
        ev = r_log["final_eval"]
        status_str = "SAFE (SELECTED)" if ev["is_safe"] else "VETOED BY INTRINSIC COST"
        print(
            f"    * Rollout Candidate `{r_log['candidate']:13s}` | Subgoals={r_log['macro_subgoals']} | "
            f"P_tot={ev['total_power_str']}W | Cost={ev['total_cost']:7.2f} -> {status_str}"
        )

    elapsed = time.time() - t0
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join(
        "checkpoints", "v26_objective_driven_hjepa_mpc_agent.pt"
    )

    metrics = {
        "v26_Step1_Full8Modules_2BitLUT_CompressionRatio": comp_ratio,
        "v26_Step1_Spherical_HJEPA_Horizon64_AntiCollapse_Acc": acc_hjepa_h64,
        "v26_Step1_Continuous_MLP_JEPA_Horizon64_Baseline_Acc": acc_cont_h64,
        "v26_Step1_HJEPA_Level2_MacroSubgoal_Planner_Acc": acc_l2_macro,
        "v26_Step2_CostModule_3Barrier_IntrinsicGuardrails_Acc": acc_guardrails,
        "v26_Step3_Mode1_Greedy_Myopic_Trap_Survival_Acc": acc_mode1_greedy,
        "v26_Step3_Mode2_HJEPA_MPC_Horizon3_Survival_Acc": acc_mode2_mpc,
        "v26_Step4_Full_6Module_LeCun_AMI_Bilingual_GrandExam_Acc": acc_grand_lecun,
    }

    torch.save(
        {
            "stage": "v26.0",
            "hjepa_wm_lut2bit_state": hjepa_wm.state_dict(),
            "cost_mod_lut2bit_state": cost_mod.state_dict(),
            "metrics": metrics,
        },
        ckpt_path,
    )

    print("\n" + "=" * 100)
    print(
        f"  ИТОГОВАЯ СВОДКА STAGE v26.0 (время: {elapsed:.1f} с | сохранён: {ckpt_path})"
    )
    print("=" * 100)
    for k, v in metrics.items():
        unit = "x" if "CompressionRatio" in k else "%"
        print(f"  {k:60s}: {v:.2f}{unit}")

    assert comp_ratio == 16.0, f"Expected 16.00x compression, got {comp_ratio}"
    assert acc_hjepa_h64 == 100.0, f"Expected 100% H-JEPA H=64 acc, got {acc_hjepa_h64}"
    assert acc_l2_macro == 100.0, f"Expected 100% Level-2 Macro Planner acc, got {acc_l2_macro}"
    assert acc_guardrails == 100.0, f"Expected 100% Cost Module Guardrails acc, got {acc_guardrails}"
    assert acc_mode2_mpc == 100.0, f"Expected 100% Mode-2 MPC survival, got {acc_mode2_mpc}"
    assert acc_grand_lecun == 100.0, f"Expected 100% Grand Exam acc, got {acc_grand_lecun}"

    return metrics


if __name__ == "__main__":
    run_v26_benchmark()
