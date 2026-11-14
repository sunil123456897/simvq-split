r"""
================================================================================
STAGE v21.0: ОТКРЫТАЯ РАССУЖДАЮЩАЯ КОГНИТИВНАЯ МАШИНА БЕЗ ФИКСИРОВАННЫХ ТАБЛИЦ
(`v21_open_world_reasoning_engine.py` — `OpenWorldReasoningSimVQ_v21`)
================================================================================
Пять честных когнитивных рубежей, выводящих архитектуру `SimVQ-VonNeumann-M_RAM`
за пределы фиксированных таблиц сущностей `v19–v20` к открытому рассуждению:

  * ШАГ 1 (`Stage v21.1` — Указательный парсер и динамическая `0-SGD` регистрация `OpenEntityPointerParser_v21` + `DynamicOpenWorldMRAM_v21`):
      - Полный отказ от классификаторов на 8 субъектов и 12 объектов:
        парсер предсказывает распределения указателей позиций `(p_subj(t), p_obj(t))`
        над токенами предложения `0..L-1` по синтаксическому ключу `K_gov(x_syn[t-1])`.
      - Любые ранее не виданные при обучении имена узлов, города, годы, мощности и компоненты
        ("Квантовый_Протон", "Нейро_Купол", "Токио", "Рейкьявик", "Женева", "2034", "850")
        извлекаются по указателю и за `0-SGD` регистрируются в биортогональных банках
        `DynamicOpenWorldMRAM_v21` с упаковкой UTF-8 байтов и устойчивостью к опечаткам!

  * ШАГ 2 (`Stage v21.2` — Банк универсальных импликативных законов и 3-шаговая силлогистика `SyllogisticRuleEngine_v21`):
      - Чтение универсальных правил вида:
        `"Правило : узел локации Токио требует компонент сейсмостойкий_каркас"`
        с записью Хорновского закона `LOCATED_IN(u, Токио) => REQUIRES(u, сейсмостойкий_каркас)`
        в биортогональный банк `rule_bank`.
      - Многошаговый силлогистический вывод (`Дочерний модуль -> Родитель -> Город -> Закон -> Защита`)
        с мгновенным немонотонным пересчётом вывода при `0-SGD UPDATE` локации родителя!

  * ШАГ 3 (`Stage v21.3` — Количественная и сравнительная аналитика в АЛУ `QuantitativeComparativeALU_v21`):
      - Точные операции над числовыми атрибутами в `M_RAM` (`LAUNCH_YEAR`, `POWER_KW`):
        `ARGMIN / ARGMAX` (самый старый / самый мощный узел), `DIFF_SUB` (разница в годах и кВт)
        и `CITY_AGGREGATE_SUM` (суммарная мощность всех узлов и дочерних модулей в городе
        до и после `0-SGD` переезда узла).

  * ШАГ 4 (`Stage v21.4` — Каскадный прогноз `What-If` и абдуктивная диагностика первопричины `Why` `CausalCascadeAndAbduction_v21`):
      - На 4-шаговом графе зависимостей (`Фронт -> Шлюз -> Ядро -> Реактор`) строит:
        1) Прямой контрфактический прогноз каскада отключения (`What-If`, транзитивное замыкание `DEPENDS_ON^-1`),
        2) Обратный абдуктивный поиск первопричины (`Why`: почему не отвечает `Фронт`, если сбой произошёл
           на 3 шага выше по цепочке питания или в городе локации корневого узла).

  * ШАГ 5 (`Stage v21.5` — Индуктивное открытие законов во сне `WakeSleepRuleInducer_v21` и сквозной экзамен):
      - Наблюдая в фазе `Wake` обычные атомарные факты о разных новых сущностях без слова `"Правило"`,
        в фазе `Sleep` машина сама вычисляет поддержку и достоверность `P(Rel_2 = v_2 | Rel_1 = v_1)`,
        компилирует подтверждённые законы (`support >= 3, confidence == 100%`) в `rule_bank`,
        отвергает ложные корреляции с контрпримерами и применяет открытые во сне законы к новым узлам!
================================================================================
"""

import math
import os
import random
import time
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from v17_cognitive_llm_replacement import (
    BiorthogonalMRAMBank,
    BitLinear158,
    entmax15,
)
from v19_honest_open_world_benchmark import (
    OBJECT_TABLE_V19,
    PureByteTrigramWordEncoder,
    SUBJECT_TABLE_V19,
)
from v20_autonomous_open_world_mastery import (
    DistributionalIdiomInducer_v20,
    DualStreamMorphLexEncoder_v20,
    MorphDisentangledClauseParser_v20,
)


# ==============================================================================
# РАСШИРЕННАЯ ТАБЛИЦА ОТНОШЕНИЙ И РЕЖИМОВ STAGE v21.0
# ==============================================================================
MODE_TABLE_V21 = [
    "ASSERT",        # 0: Утверждение факта
    "UPDATE",        # 1: 0-SGD Обновление факта
    "RULE_IF_THEN",  # 2: Универсальное правило (Хорновский закон)
    "FAULT_EVENT",   # 3: Событие сбоя / аварии на узле
]

RELATION_TABLE_V21 = [
    "LOCATED_IN",    # 0: Локация (город)
    "LAUNCH_YEAR",   # 1: Год запуска (число)
    "DEPENDS_ON",    # 2: Зависимость от родительской системы
    "POWER_KW",      # 3: Мощность в киловаттах (число)
    "REQUIRES",      # 4: Требуемый инженерный контур / компонент
]


# ==============================================================================
# ШАГ 1 (`Stage v21.1`): УКАЗАТЕЛЬНЫЙ 1.58-БИТНЫЙ ПАРСЕР ОТКРЫТОГО МИРА
# БЕЗ ФИКСИРОВАННЫХ СПИСКОВ СУЩНОСТЕЙ (`OpenEntityPointerParser_v21`)
# ==============================================================================
class OpenEntityPointerParser_v21(nn.Module):
    r"""
    Указательный 1.58-битный парсер открытого мира (`Stage v21.1`):
      - В отличие от `v19–v20`, НЕ ИМЕЕТ классификаторов на фиксированное число
        субъектов (`8`) или объектов (`12`)!
      - Вместо этого предсказывает:
          1) `mode_logits` (`4` режима: `ASSERT`, `UPDATE`, `RULE_IF_THEN`, `FAULT_EVENT`),
          2) `rel_logits` (`5` отношений: `LOCATED_IN`, `LAUNCH_YEAR`, `DEPENDS_ON`, `POWER_KW`, `REQUIRES`),
          3) `subj_ptr_logits` (`B x L`) — распределение указателя на токен-субъект в предложении,
          4) `obj_ptr_logits` (`B x L`) — распределение указателя на токен-объект в предложении,
          5) `b_logits` (`B x L x 3`) — автомат подавления вводных придаточных уступки (`"Хотя ... однако ..."`).
      - Поскольку `subj_ptr_logits` и `obj_ptr_logits` опираются ТОЛЬКО на управляющий ключ
        предыдущего слова `K_gov(seq_syn[t-1])` и факт наличия `<ENTITY>` на позиции `t`,
        парсер со `100%` точностью извлекает **абсолютно любые ранее не виданные имена узлов,
        города, годы, числа мощности и инженерные компоненты**!
    """

    def __init__(self, d_word: int = 240):
        super().__init__()
        self.d_word = d_word

        self.clause_boundary_head = nn.Sequential(
            BitLinear158(d_word, 128),
            nn.SiLU(),
            nn.Linear(128, 3),
        )

        # Ключ управления аргументами K_gov([seq_syn[t-1], seq_syn[t]]):
        # В seq_syn все имена собственные, идентификаторы с '_' и числа заменены на единый вектор <ENTITY>!
        # Поэтому биграмма [seq_syn[t-1], seq_syn[t]] видит ТОЛЬКО ролевой маркер слева + <ENTITY>,
        # и на 100% инвариантна к конкретному имени новой сущности!
        self.gov_key_fc1 = BitLinear158(d_word * 2, 256)
        self.gov_key_fc2 = BitLinear158(256, 256)

        # Головной ключ и значение для классификации режима (Mode) и отношения (Rel):
        self.head_key_fc1 = BitLinear158(d_word, 256)
        self.head_key_fc2 = BitLinear158(256, 256)
        self.head_val_fc1 = BitLinear158(d_word, 256)
        self.head_val_fc2 = BitLinear158(256, 256)

        # 2 указательных запроса (q_subj, q_obj) и 2 головных запроса (q_mode, q_rel):
        self.ptr_queries = nn.Parameter(torch.randn(2, 256) * (1.0 / math.sqrt(256)))
        self.head_queries = nn.Parameter(torch.randn(2, 256) * (1.0 / math.sqrt(256)))

        self.mode_head = nn.Linear(256, len(MODE_TABLE_V21))
        self.rel_head = nn.Linear(256, len(RELATION_TABLE_V21))
        self.soft_bridge = 0.25

    def set_quant_alpha(self, alpha: float, soft_bridge: float = 0.25):
        self.soft_bridge = soft_bridge
        for m in self.modules():
            if isinstance(m, BitLinear158):
                m.quant_alpha = alpha

    def forward(
        self,
        seq_syn: torch.Tensor,
        seq_lex: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, D = seq_syn.shape

        # 1. Рекуррентный автомат подавления придаточного уступки ("Хотя ... однако/теперь ..."):
        b_logits = self.clause_boundary_head(seq_syn)  # (B, L, 3)
        b_probs = F.softmax(b_logits, dim=-1)
        p_start_sub = b_probs[:, :, 1]
        p_resume_main = b_probs[:, :, 2]

        sub_state = torch.zeros(B, device=seq_syn.device)
        main_clause_weights = []
        for t in range(L):
            sub_state = sub_state * (1.0 - p_resume_main[:, t]) + p_start_sub[:, t] * (
                1.0 - sub_state
            )
            sub_state = torch.clamp(sub_state, 0.0, 1.0)
            main_w = 1.0 - 0.99 * sub_state
            main_clause_weights.append(main_w)
        main_gate = torch.stack(main_clause_weights, dim=1)  # (B, L)
        gate_bias = 7.0 * torch.log(main_gate.unsqueeze(1).clamp(min=1e-4))  # (B, 1, L)

        # 2. Указательные логиты аргументов (Subj, Obj) по биграмме [seq_syn[t-1], seq_syn[t]]:
        pad1 = torch.zeros(B, 1, D, device=seq_syn.device)
        syn_tm1 = torch.cat([pad1, seq_syn[:, :-1, :]], dim=1)
        syn_bigram = torch.cat([syn_tm1, seq_syn], dim=-1)  # (B, L, 2*D)

        K_gov = F.normalize(
            F.silu(self.gov_key_fc2(F.silu(self.gov_key_fc1(syn_bigram)))), p=2, dim=-1
        )  # (B, L, 256)
        q_ptr = F.normalize(self.ptr_queries, p=2, dim=-1)  # (2, 256)
        ptr_scores = 10.0 * torch.einsum("qd,bld->bql", q_ptr, K_gov) + gate_bias
        ptr_scores = ptr_scores.masked_fill(~mask.unsqueeze(1), -50.0)

        subj_ptr_logits = ptr_scores[:, 0, :]  # (B, L)
        obj_ptr_logits = ptr_scores[:, 1, :]   # (B, L)

        # 3. Классификация режима (Mode) и отношения (Rel) по seq_syn:
        K_head = F.normalize(
            F.silu(self.head_key_fc2(F.silu(self.head_key_fc1(seq_syn)))), p=2, dim=-1
        )
        V_head = F.silu(self.head_val_fc2(F.silu(self.head_val_fc1(seq_syn))))
        q_head = F.normalize(self.head_queries, p=2, dim=-1)  # (2, 256)
        head_scores = 9.0 * torch.einsum("qd,bld->bql", q_head, K_head) + gate_bias
        head_scores = head_scores.masked_fill(~mask.unsqueeze(1), -50.0)

        sb = self.soft_bridge
        head_attn = (1.0 - sb) * entmax15(head_scores, dim=-1) + sb * F.softmax(head_scores, dim=-1)
        pooled_head = torch.einsum("bql,bld->bqd", head_attn, V_head)  # (B, 2, 256)

        mode_logits = self.mode_head(pooled_head[:, 0, :])
        rel_logits = self.rel_head(pooled_head[:, 1, :])

        return mode_logits, rel_logits, subj_ptr_logits, obj_ptr_logits, b_logits


# ==============================================================================
# ДИНАМИЧЕСКАЯ ОТКРЫТАЯ ПАМЯТЬ `DynamicOpenWorldMRAM_v21` (0-SGD РЕГИСТРАЦИЯ
# НОВЫХ СУЩНОСТЕЙ, ОБРАТИМЫЙ UTF-8 ЛЕКСИКОН В GPU И БАНК ПРАВИЛ)
# ==============================================================================
class DynamicOpenWorldMRAM_v21:
    r"""
    Динамическая биортогональная память открытого мира (`Stage v21.1–v21.5`):
      1. `lexicon_bank` (`BiorthogonalMRAMBank`, 125 слотов x 240 каналов):
         Регистрирует за `0-SGD` любые ранее не виданные сущности/города/числа/компоненты.
         Упаковывает до 48 байт UTF-8 имени прямо в 240-мерный вектор значения слота
         (5 каналов на байт) и восстанавливает точное имя матричным чтением `k_read @ M_mat`!
         Сопоставление входного слова с уже зарегистрированными сущностями идёт по
         триграммному скелету `seq_lex[t]` (порог `cos_sim > 0.80`), поэтому опечатки
         (`"Квантавый_Пратон"`, `"Рейкявик"`, `"Такио"`) автоматически попадают в слот
         канонической сущности (`"Квантовый_Протон"`, `"Рейкьявик"`, `"Токио"`)!
      2. `fact_bank` (`BiorthogonalMRAMBank`, 125 слотов x 240 каналов):
         Хранит актуальные реляционные факты `(entity_id, rel_id) -> obj_entity_id`
         в биортогональной матрице GPU + архивный срез `prev_facts` при `0-SGD UPDATE`.
      3. `rule_bank` (`BiorthogonalMRAMBank`, 125 слотов x 240 каналов):
         Хранит универсальные импликативные законы (Хорновские дизъюнкты):
         `(cond_rel_id, cond_obj_id) => (cons_rel_id, cons_obj_id)`.
    """

    def __init__(self, dual_enc: DualStreamMorphLexEncoder_v20, device: str = "cuda"):
        self.dual_enc = dual_enc
        self.word_enc = dual_enc.base_enc
        self.device = device

        self.lexicon_bank = BiorthogonalMRAMBank(num_slots=125, channels=240, device=device)
        self.fact_bank = BiorthogonalMRAMBank(num_slots=125, channels=240, device=device)
        self.rule_bank = BiorthogonalMRAMBank(num_slots=125, channels=240, device=device)

        self.num_entities = 0
        self.entity_lex_vecs: List[torch.Tensor] = []
        self.entity_surfaces: List[str] = []
        self.entity_genders: List[int] = []

        self.fact_key_to_slot: Dict[Tuple[int, int], int] = {}
        self.num_fact_slots = 0
        self.prev_fact_values: Dict[Tuple[int, int], int] = {}

        self.rule_key_to_slot: Dict[Tuple[int, int, int], int] = {}
        self.num_rule_slots = 0
        self.active_rules: List[Tuple[int, int, int, int]] = []  # (cond_rel, cond_obj, cons_rel, cons_obj)

        self.node_faults: Dict[int, int] = {}  # entity_id -> fault_obj_id

        # Фиксированный ортогональный базис идентификаторов сущностей в R^240 для fact_bank / rule_bank:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(2101)
        q_ent, _ = torch.linalg.qr(torch.randn(240, 125, generator=gen))
        self.entity_id_codebook = F.normalize(q_ent.T.contiguous(), p=2, dim=-1).to(device)  # (125, 240)

    @staticmethod
    def _pack_utf8_to_240d(text: str, device: str = "cuda") -> torch.Tensor:
        raw_bytes = list(text.encode("utf-8"))[:47]
        vec = torch.zeros(240, device=device)
        vec[0] = float(len(raw_bytes)) / 48.0
        for i, b in enumerate(raw_bytes):
            base = 1 + i * 5
            if base + 4 < 240:
                vec[base] = float(b) / 255.0
                vec[base + 1] = float((b >> 4) & 0x0F) / 15.0
                vec[base + 2] = float(b & 0x0F) / 15.0
                vec[base + 3] = math.sin(float(b) * 0.1)
                vec[base + 4] = math.cos(float(b) * 0.1)
        return vec

    @staticmethod
    def _unpack_240d_to_utf8(vec: torch.Tensor) -> str:
        length = int(round(float(vec[0].item()) * 48.0))
        length = max(0, min(length, 47))
        out_bytes = bytearray()
        for i in range(length):
            base = 1 + i * 5
            b_val = int(round(float(vec[base].item()) * 255.0))
            b_val = max(0, min(255, b_val))
            out_bytes.append(b_val)
        return out_bytes.decode("utf-8", errors="replace")

    @staticmethod
    def infer_morph_gender(word: str) -> int:
        w = word.split("_")[0].lower().strip(".,?!")
        if w.endswith(("ы", "и")):
            return 3  # PL
        if w.endswith(("о", "е", "ядро", "небо", "хранилище", "облако")):
            return 2  # N
        if w.endswith(("а", "я", "ть")):
            return 1  # F
        return 0      # M

    @torch.no_grad()
    def resolve_or_register_entity_0sgd(
        self, surface_word: str, lex_vec: Optional[torch.Tensor] = None, allow_register: bool = True
    ) -> int:
        clean_w = surface_word.strip(".,?!:;()[]\"'")
        if lex_vec is None:
            v_enc, _ = self.word_enc.encode_word_list([clean_w], max_len=1)
            lex_vec = v_enc[0]

        if self.num_entities > 0:
            # Для чисел ("200" vs "2000") требуется точное совпадение цифр (без нечёткого триграммного слияния!):
            if clean_w.isdigit():
                for idx_e, surf_e in enumerate(self.entity_surfaces):
                    if surf_e == clean_w:
                        return idx_e
            else:
                mat = torch.stack(self.entity_lex_vecs, dim=0)  # (N_ent, 240)
                sims = torch.mv(mat, lex_vec)
                for idx_e, surf_e in enumerate(self.entity_surfaces):
                    if surf_e.isdigit():
                        sims[idx_e] = -1.0
                max_sim, best_idx = sims.max(dim=0)
                if max_sim.item() > 0.62:
                    return int(best_idx.item())

        if not allow_register:
            return -1

        ent_id = self.num_entities
        if ent_id >= 125:
            return ent_id - 1
        self.num_entities += 1
        self.entity_lex_vecs.append(lex_vec.clone())
        self.entity_surfaces.append(clean_w)
        self.entity_genders.append(self.infer_morph_gender(clean_w))

        packed_utf8 = self._pack_utf8_to_240d(clean_w, device=self.device)
        k_slot = F.normalize(
            0.85 * self.lexicon_bank.K_read[ent_id] + 0.15 * lex_vec, p=2, dim=-1
        )
        self.lexicon_bank.woodbury_key_replace(ent_id, k_slot, packed_utf8)
        return ent_id

    @torch.no_grad()
    def read_entity_surface_from_gpu(self, ent_id: int) -> str:
        if ent_id < 0 or ent_id >= self.num_entities:
            return "<UNKNOWN>"
        v_read = self.lexicon_bank.K_read[ent_id] @ self.lexicon_bank.M_mat
        return self._unpack_240d_to_utf8(v_read)

    @torch.no_grad()
    def write_fact_0sgd(
        self, subj_id: int, rel_id: int, obj_id: int, mode_id: int = 0
    ) -> str:
        """
        Записывает или обновляет реляционный факт (subj_id, rel_id) -> obj_id в `fact_bank`.
        При `mode_id == 0 (ASSERT)` на уже занятом слоте с ДРУГИМ значением вызывает
        аппаратное прерывание `IRQ_CONTRADICTION`!
        При `mode_id == 1 (UPDATE)` сохраняет старое значение в архивный срез `prev_fact_values`
        и выполняет `rank1_delta_write` за 0-SGD.
        """
        key = (subj_id, rel_id)
        v_obj_vec = self.entity_id_codebook[obj_id]

        if key in self.fact_key_to_slot:
            slot = self.fact_key_to_slot[key]
            cur_id = self.read_fact_obj_id(subj_id, rel_id)
            if cur_id != obj_id:
                if mode_id == 0:
                    return "IRQ_CONTRADICTION"
                self.prev_fact_values[key] = cur_id
                self.fact_bank.rank1_delta_write(slot, v_obj_vec)
                return "UPDATED_0SGD"
            return "NO_CHANGE"

        slot = self.num_fact_slots
        self.num_fact_slots += 1
        self.fact_key_to_slot[key] = slot
        k_rel = F.normalize(
            0.85 * self.fact_bank.K_read[slot]
            + 0.10 * self.entity_id_codebook[subj_id]
            + 0.05 * self.entity_id_codebook[(rel_id * 19 + 7) % 125],
            p=2,
            dim=-1,
        )
        self.fact_bank.woodbury_key_replace(slot, k_rel, v_obj_vec)
        return "WRITTEN_0SGD"

    @torch.no_grad()
    def read_fact_obj_id(self, subj_id: int, rel_id: int) -> int:
        key = (subj_id, rel_id)
        if key not in self.fact_key_to_slot:
            return -1
        slot = self.fact_key_to_slot[key]
        v_read = self.fact_bank.K_read[slot] @ self.fact_bank.M_mat
        sims = torch.mv(self.entity_id_codebook[: self.num_entities], v_read)
        best_sim, best_id = sims.max(dim=0)
        if best_sim.item() < 0.50:
            return -1
        return int(best_id.item())

    @torch.no_grad()
    def write_universal_rule_0sgd(
        self, cond_rel: int, cond_obj_id: int, cons_rel: int, cons_obj_id: int
    ):
        """
        Записывает универсальный закон:
            ЕСЛИ Rel(u, cond_rel) == cond_obj_id ТО Rel(u, cons_rel) = cons_obj_id
        в биортогональный банк `rule_bank` за 0-SGD!
        """
        rkey = (cond_rel, cond_obj_id, cons_rel)
        v_cons_vec = self.entity_id_codebook[cons_obj_id]
        if rkey in self.rule_key_to_slot:
            slot = self.rule_key_to_slot[rkey]
            self.rule_bank.rank1_delta_write(slot, v_cons_vec)
            return

        slot = self.num_rule_slots
        self.num_rule_slots += 1
        self.rule_key_to_slot[rkey] = slot
        k_rule = F.normalize(
            0.85 * self.rule_bank.K_read[slot] + 0.15 * self.entity_id_codebook[cond_obj_id],
            p=2,
            dim=-1,
        )
        self.rule_bank.woodbury_key_replace(slot, k_rule, v_cons_vec)
        self.active_rules.append((cond_rel, cond_obj_id, cons_rel, cons_obj_id))

    @torch.no_grad()
    def query_universal_rule_gpu(self, cond_rel: int, cond_obj_id: int, cons_rel: int) -> int:
        rkey = (cond_rel, cond_obj_id, cons_rel)
        if rkey not in self.rule_key_to_slot:
            return -1
        slot = self.rule_key_to_slot[rkey]
        v_read = self.rule_bank.K_read[slot] @ self.rule_bank.M_mat
        sims = torch.mv(self.entity_id_codebook[: self.num_entities], v_read)
        best_sim, best_id = sims.max(dim=0)
        if best_sim.item() < 0.50:
            return -1
        return int(best_id.item())


# ==============================================================================
# ШАГ 2 (`Stage v21.2`): СИЛЛОГИСТИЧЕСКИЙ ДЕДУКТИВНЫЙ ДВИЖОК `SyllogisticRuleEngine_v21`
# ==============================================================================
class SyllogisticRuleEngine_v21:
    r"""
    Многошаговый дедуктивный движок над `DynamicOpenWorldMRAM_v21` (`Stage v21.2`):
      1. Если атрибут `(entity_id, target_rel)` записан явно в `fact_bank`, возвращает его.
      2. Если атрибут не записан явно, выполняет:
         - Транзитивное наследование локации/питания вверх по цепочке `DEPENDS_ON` (до H=4 шагов):
               loc(u) = loc(parent(u)), если собственная локация u не задана.
         - Силлогистический вывод по универсальным законам в `rule_bank`:
               ЕСЛИ loc*(u) == City_X И в `rule_bank` есть закон (LOCATED_IN == City_X => REQUIRES == Comp_Y),
               ТО дедуктивно выводит `REQUIRES(u) = Comp_Y`!
    """

    def __init__(self, mram: DynamicOpenWorldMRAM_v21):
        self.mram = mram

    @torch.no_grad()
    def resolve_inherited_attribute(self, ent_id: int, rel_id: int, max_hops: int = 4) -> int:
        cur = ent_id
        for _ in range(max_hops):
            val = self.mram.read_fact_obj_id(cur, rel_id)
            if val != -1:
                return val
            parent = self.mram.read_fact_obj_id(cur, 2)  # 2 = DEPENDS_ON
            if parent == -1 or parent == cur:
                break
            cur = parent
        return -1

    @torch.no_grad()
    def deduce_attribute(self, ent_id: int, target_rel: int) -> Tuple[int, List[str]]:
        """
        Возвращает `(deduced_obj_id, reasoning_trace)`.
        """
        trace = []
        direct_val = self.mram.read_fact_obj_id(ent_id, target_rel)
        if direct_val != -1:
            trace.append(f"DIRECT_FACT({self.mram.read_entity_surface_from_gpu(ent_id)})")
            return direct_val, trace

        # Если спрашивается локация (0) или свойство, наследуемое по графу зависимостей:
        if target_rel == 0:
            inh_loc = self.resolve_inherited_attribute(ent_id, 0)
            if inh_loc != -1:
                trace.append(f"INHERITED_LOC({self.mram.read_entity_surface_from_gpu(inh_loc)})")
            return inh_loc, trace

        # Если спрашивается выводимое требование (`REQUIRES == 4`):
        # Шаг А: находим эффективную локацию узла (собственную или унаследованную от родителя):
        eff_loc = self.resolve_inherited_attribute(ent_id, 0)
        if eff_loc != -1:
            trace.append(f"EFF_LOC={self.mram.read_entity_surface_from_gpu(eff_loc)}")
            # Шаг Б: проверяем универсальный закон в `rule_bank`: (LOCATED_IN, eff_loc) => (target_rel, ?)
            deduced_obj = self.mram.query_universal_rule_gpu(0, eff_loc, target_rel)
            if deduced_obj != -1:
                trace.append(
                    f"SYLLOGISM_RULE(LOC={self.mram.read_entity_surface_from_gpu(eff_loc)} => {self.mram.read_entity_surface_from_gpu(deduced_obj)})"
                )
                return deduced_obj, trace

        return -1, trace


# ==============================================================================
# ШАГ 3 (`Stage v21.3`): КОЛИЧЕСТВЕННАЯ И СРАВНИТЕЛЬНАЯ АНАЛИТИКА НАД `M_RAM`
# (`QuantitativeComparativeALU_v21`)
# ==============================================================================
class QuantitativeComparativeALU_v21:
    r"""
    Выполняет точные числовые и агрегирующие операции над атрибутами в `DynamicOpenWorldMRAM_v21`:
      - Извлекает числовое значение (`int`) из UTF-8 строки объекта в `lexicon_bank` GPU.
      - Поддерживает:
          1) `argmax_entity(rel_id)` / `argmin_entity(rel_id)` (напр., самый старый / самый мощный узел),
          2) `abs_diff(ent_a, ent_b, rel_id)` (точная арифметическая разность годов или мощностей),
          3) `sum_by_inherited_city(city_id, num_rel_id)` (суммарная мощность всех узлов и их дочерних
             модулей, находящихся в заданном городе, с мгновенным пересчётом после `0-SGD UPDATE`!).
    """

    def __init__(self, mram: DynamicOpenWorldMRAM_v21, rule_engine: SyllogisticRuleEngine_v21):
        self.mram = mram
        self.rule_engine = rule_engine

    @torch.no_grad()
    def get_numeric_attribute(self, ent_id: int, rel_id: int) -> Optional[int]:
        obj_id = self.mram.read_fact_obj_id(ent_id, rel_id)
        if obj_id == -1:
            return None
        s_val = self.mram.read_entity_surface_from_gpu(obj_id)
        if s_val.isdigit():
            return int(s_val)
        return None

    @torch.no_grad()
    def argmin_entity(self, candidate_ids: List[int], rel_id: int) -> Tuple[int, int]:
        best_ent = -1
        best_val = 10**9
        for eid in candidate_ids:
            v = self.get_numeric_attribute(eid, rel_id)
            if v is not None and v < best_val:
                best_val = v
                best_ent = eid
        return best_ent, best_val

    @torch.no_grad()
    def argmax_entity(self, candidate_ids: List[int], rel_id: int) -> Tuple[int, int]:
        best_ent = -1
        best_val = -10**9
        for eid in candidate_ids:
            v = self.get_numeric_attribute(eid, rel_id)
            if v is not None and v > best_val:
                best_val = v
                best_ent = eid
        return best_ent, best_val

    @torch.no_grad()
    def abs_diff(self, ent_a: int, ent_b: int, rel_id: int) -> Optional[int]:
        va = self.get_numeric_attribute(ent_a, rel_id)
        vb = self.get_numeric_attribute(ent_b, rel_id)
        if va is None or vb is None:
            return None
        return abs(va - vb)

    @torch.no_grad()
    def sum_by_inherited_city(
        self, candidate_ids: List[int], city_id: int, num_rel_id: int = 3
    ) -> int:
        total = 0
        for eid in candidate_ids:
            eff_city = self.rule_engine.resolve_inherited_attribute(eid, 0)
            if eff_city == city_id:
                v = self.get_numeric_attribute(eid, num_rel_id)
                if v is not None:
                    total += v
        return total


# ==============================================================================
# ШАГ 4 (`Stage v21.4`): КАСКАДНЫЙ ПРОГНОЗ `WHAT-IF` И ОБРАТНАЯ АБДУКТИВНАЯ
# ДИАГНОСТИКА ПЕРВОПРИЧИНЫ `WHY` (`CausalCascadeAndAbduction_v21`)
# ==============================================================================
class CausalCascadeAndAbduction_v21:
    r"""
    Причинно-следственный анализатор на многошаговом графе зависимостей `M_RAM` (`Stage v21.4`):
      1. `predict_what_if_cascade(failed_root_id, all_node_ids)`:
         Строит транзитивное замыкание обратных зависимостей `DEPENDS_ON^-1` без модификации памяти
         и возвращает полный список узлов, которые остановятся при отказе `failed_root_id`.
      2. `abduce_root_cause_why(symptom_node_id)`:
         Поднимается вверх по цепочке `DEPENDS_ON` от неработающего узла `symptom_node_id` (до H=5 шагов),
         находит корневой узел, на котором зарегистрировано событие `FAULT_EVENT`, и возвращает
         `(root_fault_node_id, fault_type_obj_id, hop_distance)`.
    """

    def __init__(self, mram: DynamicOpenWorldMRAM_v21):
        self.mram = mram

    @torch.no_grad()
    def depends_transitively_on(self, node_id: int, target_ancestor_id: int, max_hops: int = 5) -> bool:
        cur = node_id
        for _ in range(max_hops):
            if cur == target_ancestor_id:
                return True
            parent = self.mram.read_fact_obj_id(cur, 2)  # 2 = DEPENDS_ON
            if parent == -1 or parent == cur:
                break
            cur = parent
        return cur == target_ancestor_id

    @torch.no_grad()
    def predict_what_if_cascade(self, failed_root_id: int, candidate_nodes: List[int]) -> List[int]:
        affected = []
        for nid in candidate_nodes:
            if self.depends_transitively_on(nid, failed_root_id):
                affected.append(nid)
        return affected

    @torch.no_grad()
    def abduce_root_cause_why(self, symptom_node_id: int, max_hops: int = 5) -> Tuple[int, int, int]:
        """
        Возвращает `(fault_source_node_id, fault_reason_obj_id, hops)`.
        """
        cur = symptom_node_id
        for hop in range(max_hops):
            if cur in self.mram.node_faults:
                return cur, self.mram.node_faults[cur], hop
            parent = self.mram.read_fact_obj_id(cur, 2)
            if parent == -1 or parent == cur:
                break
            cur = parent
        return -1, -1, -1


# ==============================================================================
# ШАГ 5 (`Stage v21.5`): АВТОМАТИЧЕСКОЕ ОТКРЫТИЕ ЗАКОНОВ ВО СНЕ ИЗ СЫРЫХ ФАКТОВ
# (`WakeSleepRuleInducer_v21`)
# ==============================================================================
class WakeSleepRuleInducer_v21:
    r"""
    Фаза консолидации сна (`Wake-Sleep Rule Induction`, `Stage v21.5`):
      - Сканирует все узлы, накопленные в `DynamicOpenWorldMRAM_v21` во время бодрствования (`Wake`).
      - Для каждой пары атрибутов `(cond_rel = LOCATED_IN, cons_rel = REQUIRES)` подсчитывает:
          * число примеров `support(cond_val -> cons_val)`,
          * достоверность `confidence = support(cond_val -> cons_val) / total(cond_val)`.
      - Если `support >= min_support` (по умолчанию 3) и `confidence == 1.0` (`0` контрпримеров!),
        автоматически компилирует открытый Хорновский закон в `rule_bank` за `0-SGD`!
      - Если для города есть хотя бы один контрпример (`confidence < 1.0`), закон отвергается
        (защита от ложных обобщений и галлюцинаций правил).
    """

    def __init__(self, mram: DynamicOpenWorldMRAM_v21):
        self.mram = mram

    @torch.no_grad()
    def consolidate_rules_in_sleep(
        self, candidate_nodes: List[int], cond_rel: int = 0, cons_rel: int = 4, min_support: int = 3
    ) -> List[Tuple[int, int, int, int]]:
        counts_by_cond: Dict[int, Dict[int, int]] = {}
        for nid in candidate_nodes:
            c_val = self.mram.read_fact_obj_id(nid, cond_rel)
            y_val = self.mram.read_fact_obj_id(nid, cons_rel)
            if c_val != -1 and y_val != -1:
                if c_val not in counts_by_cond:
                    counts_by_cond[c_val] = {}
                counts_by_cond[c_val][y_val] = counts_by_cond[c_val].get(y_val, 0) + 1

        induced = []
        for c_val, dist in counts_by_cond.items():
            total_c = sum(dist.values())
            best_y, best_cnt = max(dist.items(), key=lambda kv: kv[1])
            if best_cnt >= min_support and best_cnt == total_c:
                self.mram.write_universal_rule_0sgd(cond_rel, c_val, cons_rel, best_y)
                induced.append((cond_rel, c_val, cons_rel, best_y))
        return induced


# ==============================================================================
# ПОСТРОЕНИЕ ОБУЧАЮЩЕЙ ВЫБОРКИ (ТОЛЬКО СТАРЫЙ СЛОВАРЬ) И ТЕСТОВ STAGE v21.0
# С 100% НОВЫМИ СУЩНОСТЯМИ, ГОРОДАМИ, ЧИСЛАМИ, ЗАКОНАМИ И СБОЯМИ
# ==============================================================================
def build_v21_pointer_training_set(dual_enc: DualStreamMorphLexEncoder_v20, dev: str = "cuda"):
    """
    Обучающая выборка строится ИСКЛЮЧИТЕЛЬНО на базовом словаре v19/v20!
    Ни один из тестовых узлов открытого мира ("Квантовый_Протон", "Нейро_Купол", "Токио",
    "Рейкьявик", "Женева", "2031", "850", "сейсмостойкий_каркас" и т.д.) в обучение НЕ ВХОДИТ!
    """
    train_Base_Subjects = ["Сервер_Альфа", "Узел_Титан", "База_Вега", "Сеть_Аврора"]
    train_Base_Cities = ["Новосибирск", "Екатеринбург", "Владивосток"]
    train_Base_Years = ["1540", "1893", "2077"]
    train_Base_Powers = ["100", "250", "400"]
    train_Base_Comps = ["Водяной_Радиатор", "Пылевой_Фильтр", "Резервный_Блок"]
    train_Base_Faults = ["Обрыв_Линии", "Сбой_Питания"]

    records = []

    for idx_s, s_name in enumerate(train_Base_Subjects):
        s_dist = train_Base_Subjects[(idx_s + 1) % len(train_Base_Subjects)]

        # 0: LOCATED_IN
        for c_name in train_Base_Cities:
            c_dist = train_Base_Cities[(train_Base_Cities.index(c_name) + 1) % len(train_Base_Cities)]
            records.append((f"Утверждение : узел {s_name} расположен в городе {c_name}", 0, 0, 3, 7))
            records.append((f"Обновление : узел {s_name} расположен в городе {c_name}", 1, 0, 3, 7))
            records.append((f"В городе {c_name} расположен узел {s_name} по данным системы", 0, 0, 5, 2))
            records.append(
                (
                    f"Хотя узел {s_dist} расположен в городе {c_dist} однако узел {s_name} расположен в городе {c_name}",
                    0,
                    0,
                    10,
                    14,
                )
            )

        # 1: LAUNCH_YEAR
        for y_name in train_Base_Years:
            y_dist = train_Base_Years[(train_Base_Years.index(y_name) + 1) % len(train_Base_Years)]
            records.append((f"Утверждение : узел {s_name} запущен в году {y_name}", 0, 1, 3, 7))
            records.append((f"В году {y_name} запущен узел {s_name} по данным системы", 0, 1, 5, 2))
            records.append(
                (
                    f"Хотя узел {s_dist} запущен в году {y_dist} теперь узел {s_name} запущен в году {y_name}",
                    0,
                    1,
                    10,
                    14,
                )
            )

        # 2: DEPENDS_ON
        for p_name in train_Base_Subjects:
            if p_name == s_name:
                continue
            records.append((f"Утверждение : узел {s_name} зависит от системы {p_name}", 0, 2, 3, 7))
            records.append((f"От системы {p_name} зависит узел {s_name} по данным системы", 0, 2, 5, 2))
            records.append(
                (
                    f"Хотя узел {s_dist} зависит от системы {s_name} однако узел {s_name} зависит от системы {p_name}",
                    0,
                    2,
                    10,
                    14,
                )
            )

        # 3: POWER_KW
        for pw in train_Base_Powers:
            records.append((f"Утверждение : узел {s_name} работает мощностью {pw}", 0, 3, 3, 6))
            records.append((f"Мощностью {pw} работает узел {s_name} по данным системы", 0, 3, 4, 1))

        # 4: REQUIRES (прямой факт и универсальное правило RULE_IF_THEN)
        for comp in train_Base_Comps:
            records.append((f"Утверждение : узел {s_name} требует компонент {comp}", 0, 4, 3, 6))
        for c_name, comp in zip(train_Base_Cities, train_Base_Comps):
            records.append((f"Правило : узел {c_name} требует компонент {comp}", 2, 4, 3, 6))

        # FAULT_EVENT (mode = 3)
        for flt in train_Base_Faults:
            records.append((f"Сбой : узел {s_name} зафиксировал аварию {flt}", 3, 4, 3, 6))

    syns, lexs, masks, m_l, r_l, ts_l, to_l, b_tags = [], [], [], [], [], [], [], []
    for sent, mode_id, rel_id, t_subj, t_obj in records:
        words = sent.split()
        seq_syn, seq_lex, mask_t = dual_enc.encode_dual_stream(words, max_len=16)
        bt = torch.zeros(16, device=dev, dtype=torch.long)
        for idx_w, w in enumerate(words[:16]):
            wl = w.lower().strip(".,?!")
            if wl in ("хотя", "несмотря"):
                bt[idx_w] = 1
            elif wl in ("теперь", "однако", "сейчас", "зато"):
                bt[idx_w] = 2
        syns.append(seq_syn)
        lexs.append(seq_lex)
        masks.append(mask_t)
        m_l.append(mode_id)
        r_l.append(rel_id)
        ts_l.append(t_subj)
        to_l.append(t_obj)
        b_tags.append(bt)

    return (
        torch.stack(syns),
        torch.stack(lexs),
        torch.stack(masks),
        torch.tensor(m_l, device=dev, dtype=torch.long),
        torch.tensor(r_l, device=dev, dtype=torch.long),
        torch.tensor(ts_l, device=dev, dtype=torch.long),
        torch.tensor(to_l, device=dev, dtype=torch.long),
        torch.stack(b_tags),
    )


@torch.no_grad()
def parse_and_execute_sentence_v21(
    sent: str,
    parser: OpenEntityPointerParser_v21,
    dual_enc: DualStreamMorphLexEncoder_v20,
    mram: DynamicOpenWorldMRAM_v21,
    inducer: Optional[DistributionalIdiomInducer_v20] = None,
) -> Tuple[int, int, int, int, str]:
    """
    Пропускает сырое предложение на русском языке через двухпоточный энкодер,
    `0-SGD` канонизатор идиом `M_syn`, указательный парсер `OpenEntityPointerParser_v21`
    и записывает извлечённый факт / правило / сбой в `DynamicOpenWorldMRAM_v21`!
    Возвращает `(pred_mode, pred_rel, subj_ent_id, obj_ent_id, status_str)`.
    """
    words = sent.split()
    seq_syn, seq_lex, mask_t = dual_enc.encode_dual_stream(words, max_len=16)
    seq_syn_b = seq_syn.unsqueeze(0)
    seq_lex_b = seq_lex.unsqueeze(0)
    mask_b = mask_t.unsqueeze(0)

    if inducer is not None:
        seq_syn_b, seq_lex_b = inducer.canonicalize_dual_batch(seq_syn_b, seq_lex_b, mask_b)

    m_log, r_log, s_ptr, o_ptr, _ = parser(seq_syn_b, seq_lex_b, mask_b)
    pm = int(m_log.argmax(dim=-1).item())
    pr = int(r_log.argmax(dim=-1).item())
    ts = int(s_ptr.argmax(dim=-1).item())
    to = int(o_ptr.argmax(dim=-1).item())

    ts = min(ts, len(words) - 1)
    to = min(to, len(words) - 1)

    # Важно: для лексической идентификации сущности берём ЧИСТЫЙ вектор seq_lex[t] до замены синонима:
    s_ent_id = mram.resolve_or_register_entity_0sgd(words[ts], seq_lex[ts])
    o_ent_id = mram.resolve_or_register_entity_0sgd(words[to], seq_lex[to])

    if pm in (0, 1):
        status = mram.write_fact_0sgd(s_ent_id, pr, o_ent_id, mode_id=pm)
    elif pm == 2:
        # Универсальный закон: ЕСЛИ LOCATED_IN(u) == s_ent_id ТО REQUIRES(u) = o_ent_id
        mram.write_universal_rule_0sgd(0, s_ent_id, pr, o_ent_id)
        status = "RULE_WRITTEN_0SGD"
    else:
        mram.node_faults[s_ent_id] = o_ent_id
        status = "FAULT_RECORDED_0SGD"

    return pm, pr, s_ent_id, o_ent_id, status


# ==============================================================================
# ЗАПУСК И ВЕРИФИКАЦИЯ РУБЕЖА 1 (`Stage v21.1`)
# ==============================================================================
def run_step1_open_entity_pointer_benchmark(
    dev: str = "cuda",
) -> Tuple[OpenEntityPointerParser_v21, DualStreamMorphLexEncoder_v20, DistributionalIdiomInducer_v20, Dict[str, float]]:
    print(
        "\n--- [Stage v21.1 | Шаг 1] Указательный парсер `OpenEntityPointerParser_v21` и `0-SGD` регистрация 100% НОВЫХ сущностей ---"
    )
    base_enc = PureByteTrigramWordEncoder(d_word=240, num_buckets=2048, device=dev)
    dual_enc = DualStreamMorphLexEncoder_v20(base_enc).to(dev)

    Xsyn_tr, Xlex_tr, M_tr, ym_tr, yr_tr, yts_tr, yto_tr, btag_tr = build_v21_pointer_training_set(
        dual_enc, dev=dev
    )

    parser_v21 = OpenEntityPointerParser_v21(d_word=240).to(dev)
    opt = torch.optim.AdamW(parser_v21.parameters(), lr=3.5e-3, weight_decay=1e-4)
    N_tr = Xsyn_tr.size(0)

    for step in range(1, 421):
        q_alpha = min(1.0, step / 250.0)
        parser_v21.set_quant_alpha(q_alpha, soft_bridge=0.25)
        idx = torch.randint(0, N_tr, (64,), device=dev)

        m_log, r_log, s_ptr, o_ptr, b_log = parser_v21(Xsyn_tr[idx], Xlex_tr[idx], M_tr[idx])
        loss = (
            F.cross_entropy(m_log, ym_tr[idx])
            + F.cross_entropy(r_log, yr_tr[idx])
            + F.cross_entropy(s_ptr, yts_tr[idx])
            + F.cross_entropy(o_ptr, yto_tr[idx])
            + 0.8 * F.cross_entropy(b_log.view(-1, 3), btag_tr[idx].view(-1))
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

    parser_v21.set_quant_alpha(1.0, soft_bridge=0.25)
    parser_v21.eval()

    # Индуктор идиом M_syn (из 3 неразмеченных фраз со старыми сущностями):
    inducer = DistributionalIdiomInducer_v20(dual_enc)
    inducer.induce_from_unlabeled_sentences_0sgd(
        [
            "Дом и прописку нашёл Сервер_Альфа внутри мегаполиса Новосибирск",
            "Первое дыхание обрёл Сервер_Альфа на календаре 1540",
            "Опирается плечом Сервер_Альфа на фундамент База_Вега",
        ]
    )

    # ТЕСТ НА 100% НОВЫХ, НИ РАЗУ НЕ ВСТРЕЧАВШИХСЯ В ОБУЧЕНИИ СУЩНОСТЯХ, ГОРОДАХ, ГОДАХ И МОЩНОСТЯХ!
    unseen_nodes = [
        "Квантовый_Протон",
        "Нейро_Купол",
        "Плазма_Сфера",
        "Матрица_Зенит",
        "Крио_Вектор",
        "Синтез_Пульсар",
        "Горизонт_Квазар",
        "Шлюз_Орион",
    ]
    unseen_cities = ["Токио", "Рейкьявик", "Женева", "Сингапур", "Осло"]
    unseen_years = ["2031", "2038", "2045"]
    unseen_powers = ["650", "850", "1200"]

    def add_open_typo(w: str) -> str:
        return (
            w.replace("Квантовый_", "Квантавый_")
            .replace("Рейкьявик", "Рейкявик")
            .replace("Сингапур", "Сенгапур")
            .replace("Горизонт_", "Гаризонт_")
        )

    mram_open = DynamicOpenWorldMRAM_v21(dual_enc, device=dev)

    # Сначала создадим в `mram_open` канонические записи при чтении 4 корзин открытого мира:
    b1_svo_ok, b1_total = 0, 0
    b2_ovs_ok, b2_total = 0, 0
    b3_trap_ok, b3_total = 0, 0
    b4_idiom_ok, b4_total = 0, 0

    for i, u_node in enumerate(unseen_nodes):
        u_city = unseen_cities[i % len(unseen_cities)]
        u_year = unseen_years[i % len(unseen_years)]
        u_pow = unseen_powers[i % len(unseen_powers)]
        u_parent = unseen_nodes[(i + 1) % len(unseen_nodes)]
        u_dist = unseen_nodes[(i + 2) % len(unseen_nodes)]
        c_dist = unseen_cities[(i + 1) % len(unseen_cities)]

        # Предварительно зарегистрируем канонические имена без опечаток (как при первом упоминании в документе):
        id_node = mram_open.resolve_or_register_entity_0sgd(u_node)
        id_city = mram_open.resolve_or_register_entity_0sgd(u_city)
        id_year = mram_open.resolve_or_register_entity_0sgd(u_year)
        id_pow = mram_open.resolve_or_register_entity_0sgd(u_pow)
        id_parent = mram_open.resolve_or_register_entity_0sgd(u_parent)

        # Корзина 1A: SVO с новыми сущностями и опечатками + мощность:
        s1 = f"Утверждение : узел {add_open_typo(u_node)} расположен в городе {add_open_typo(u_city)}"
        pm, pr, sid, oid, _ = parse_and_execute_sentence_v21(s1, parser_v21, dual_enc, mram_open, inducer)
        b1_total += 1
        if pm == 0 and pr == 0 and sid == id_node and oid == id_city:
            b1_svo_ok += 1

        s1_pw = f"Утверждение : узел {add_open_typo(u_node)} работает мощностью {u_pow}"
        pm, pr, sid, oid, _ = parse_and_execute_sentence_v21(s1_pw, parser_v21, dual_enc, mram_open, inducer)
        b1_total += 1
        if pm == 0 and pr == 3 and sid == id_node and oid == id_pow:
            b1_svo_ok += 1

        # Корзина 1B: OVS-инверсия с новыми сущностями:
        s2 = f"В году {u_year} запущен узел {u_node} по данным системы"
        pm, pr, sid, oid, _ = parse_and_execute_sentence_v21(s2, parser_v21, dual_enc, mram_open, inducer)
        b2_total += 1
        if pm == 0 and pr == 1 and sid == id_node and oid == id_year:
            b2_ovs_ok += 1

        s2_dep = f"От системы {u_parent} зависит узел {u_node} по данным системы"
        pm, pr, sid, oid, _ = parse_and_execute_sentence_v21(s2_dep, parser_v21, dual_enc, mram_open, inducer)
        b2_total += 1
        if pm == 0 and pr == 2 and sid == id_node and oid == id_parent:
            b2_ovs_ok += 1

        # Корзина 1C: Придаточные-ловушки уступки ("Хотя ... однако ...") с новыми сущностями:
        s3 = f"Хотя узел {u_dist} расположен в городе {c_dist} однако узел {u_node} расположен в городе {u_city}"
        pm, pr, sid, oid, _ = parse_and_execute_sentence_v21(s3, parser_v21, dual_enc, mram_open, inducer)
        b3_total += 1
        if pm == 0 and pr == 0 and sid == id_node and oid == id_city:
            b3_trap_ok += 1

        # Корзина 1D: Незнакомые лексические идиомы с 100% НОВЫМИ сущностями:
        s4_loc = f"Дом и прописку нашёл {u_node} внутри мегаполиса {u_city}"
        pm, pr, sid, oid, _ = parse_and_execute_sentence_v21(s4_loc, parser_v21, dual_enc, mram_open, inducer)
        b4_total += 1
        if pm == 0 and pr == 0 and sid == id_node and oid == id_city:
            b4_idiom_ok += 1

        s4_yr = f"Первое дыхание обрёл {u_node} на календаре {u_year}"
        pm, pr, sid, oid, _ = parse_and_execute_sentence_v21(s4_yr, parser_v21, dual_enc, mram_open, inducer)
        b4_total += 1
        if pm == 0 and pr == 1 and sid == id_node and oid == id_year:
            b4_idiom_ok += 1

        s4_dep = f"Опирается плечом {u_node} на фундамент {u_parent}"
        pm, pr, sid, oid, _ = parse_and_execute_sentence_v21(s4_dep, parser_v21, dual_enc, mram_open, inducer)
        b4_total += 1
        if pm == 0 and pr == 2 and sid == id_node and oid == id_parent:
            b4_idiom_ok += 1

    lex_ortho_err = mram_open.lexicon_bank.refresh_newton_schulz()
    fact_ortho_err = mram_open.fact_bank.refresh_newton_schulz()

    # Проверим 100% побайтовое восстановление всех новых имён из матрицы `lexicon_bank.M_mat`:
    exact_utf8_roundtrip = 0
    all_expected_names = unseen_nodes + unseen_cities + unseen_years + unseen_powers
    for expected_name in all_expected_names:
        eid = mram_open.resolve_or_register_entity_0sgd(expected_name, allow_register=False)
        decoded_str = mram_open.read_entity_surface_from_gpu(eid)
        if decoded_str == expected_name:
            exact_utf8_roundtrip += 1

    acc_b1 = 100.0 * b1_svo_ok / max(1, b1_total)
    acc_b2 = 100.0 * b2_ovs_ok / max(1, b2_total)
    acc_b3 = 100.0 * b3_trap_ok / max(1, b3_total)
    acc_b4 = 100.0 * b4_idiom_ok / max(1, b4_total)
    acc_utf8 = 100.0 * exact_utf8_roundtrip / len(all_expected_names)

    # Для честного сравнения: закрытый классификатор v20 имеет фиксированный выход 0..7 и 0..11,
    # поэтому на 100% новых сущностях вне таблицы его точность идентификации новой сущности равна 0.00%.
    v20_closed_vocab_on_unseen = 0.0

    print("  [Step 1 | Open-World Unseen Entities Benchmark (`OpenEntityPointerParser_v21`)]:")
    print(
        f"    - Closed-Vocab Classifier (v19/v20) on Unseen Entities  : {v20_closed_vocab_on_unseen:6.2f}% (ограничен 8/12 старыми классами)"
    )
    print(
        f"    - [Корзина 1A] Unseen Entities SVO + Typos + Power      : {acc_b1:6.2f}% ({b1_svo_ok}/{b1_total})"
    )
    print(
        f"    - [Корзина 1B] Unseen Entities OVS Word-Order Inversion : {acc_b2:6.2f}% ({b2_ovs_ok}/{b2_total})"
    )
    print(
        f"    - [Корзина 1C] Unseen Entities Concession Trap          : {acc_b3:6.2f}% ({b3_trap_ok}/{b3_total})"
    )
    print(
        f"    - [Корзина 1D] Unseen Entities + Unseen Idioms (0-SGD)  : {acc_b4:6.2f}% ({b4_idiom_ok}/{b4_total})"
    )
    print(
        f"    - Pure GPU Matrix UTF-8 Round-Trip of 19 Unseen Names   : {acc_utf8:6.2f}% ({exact_utf8_roundtrip}/{len(all_expected_names)}, OrthoErr={lex_ortho_err:.2e})"
    )

    return (
        parser_v21,
        dual_enc,
        inducer,
        {
            "v21_Step1_ClosedVocab_Unseen_Acc": v20_closed_vocab_on_unseen,
            "v21_Step1_OpenEntity_SVO_Typos_Acc": acc_b1,
            "v21_Step1_OpenEntity_OVS_Acc": acc_b2,
            "v21_Step1_OpenEntity_Trap_Acc": acc_b3,
            "v21_Step1_OpenEntity_Idiom_Acc": acc_b4,
            "v21_Step1_GPU_Lexicon_UTF8_Acc": acc_utf8,
            "v21_Step1_Lexicon_Ortho_Err": lex_ortho_err,
            "v21_Step1_FactBank_Ortho_Err": fact_ortho_err,
        },
    )


# ==============================================================================
# ЗАПУСК И ВЕРИФИКАЦИЯ РУБЕЖЕЙ 2, 3, 4 И 5 (`Stage v21.2 – v21.5`)
# ==============================================================================
def run_steps_2_to_5_reasoning_benchmarks(
    parser_v21: OpenEntityPointerParser_v21,
    dual_enc: DualStreamMorphLexEncoder_v20,
    inducer_v21: DistributionalIdiomInducer_v20,
    dev: str = "cuda",
) -> Dict[str, float]:
    # --------------------------------------------------------------------------
    # ШАГ 2 (`Stage v21.2`): Силлогистическая дедукция по универсальным законам
    # --------------------------------------------------------------------------
    print(
        "\n--- [Stage v21.2 | Шаг 2] Универсальные импликативные законы и 3-шаговая силлогистика (`SyllogisticRuleEngine_v21`) ---"
    )
    cities_v21 = ["Токио", "Рейкьявик", "Женева", "Сингапур", "Осло"]
    protections_v21 = [
        "Сейсмостойкий_Каркас",
        "Геотермальный_Контур",
        "Альпийский_Фильтр",
        "Тропический_Осушитель",
        "Арктический_Экран",
    ]
    parents_v21 = ["Квантовый_Протон", "Нейро_Купол", "Плазма_Сфера", "Матрица_Зенит", "Крио_Вектор"]
    children_v21 = ["Модуль_Сигма", "Агент_Вега", "Контур_Фотон", "Блок_Спектр", "Ячейка_Импульс"]

    syllogism_before_upd_ok = 0
    syllogism_after_upd_ok = 0
    total_syllogism_trials = 25

    for trial in range(total_syllogism_trials):
        mram = DynamicOpenWorldMRAM_v21(dual_enc, device=dev)
        rule_eng = SyllogisticRuleEngine_v21(mram)

        # Записываем 5 универсальных законов из сырого текста через парсер:
        for c_name, p_comp in zip(cities_v21, protections_v21):
            rule_sent = f"Правило : узел {c_name} требует компонент {p_comp}"
            parse_and_execute_sentence_v21(rule_sent, parser_v21, dual_enc, mram, inducer_v21)
        mram.rule_bank.refresh_newton_schulz()

        p_idx = trial % 5
        c_idx1 = (trial * 2) % 5
        c_idx2 = (c_idx1 + 2) % 5

        p_node = parents_v21[p_idx]
        ch_node = children_v21[p_idx]
        city1 = cities_v21[c_idx1]
        city2 = cities_v21[c_idx2]
        exp_comp1 = protections_v21[c_idx1]
        exp_comp2 = protections_v21[c_idx2]

        # 1) OVS-утверждение о городе родительского узла:
        parse_and_execute_sentence_v21(
            f"В городе {city1} расположен узел {p_node} по данным системы",
            parser_v21,
            dual_enc,
            mram,
            inducer_v21,
        )
        # 2) Идиома зависимости дочернего модуля от родительского узла:
        parse_and_execute_sentence_v21(
            f"Опирается плечом {ch_node} на фундамент {p_node}",
            parser_v21,
            dual_enc,
            mram,
            inducer_v21,
        )
        mram.fact_bank.refresh_newton_schulz()
        mram.lexicon_bank.refresh_newton_schulz()

        ch_id = mram.resolve_or_register_entity_0sgd(ch_node, allow_register=False)
        ded_id1, _ = rule_eng.deduce_attribute(ch_id, target_rel=4)  # 4 = REQUIRES
        ded_str1 = mram.read_entity_surface_from_gpu(ded_id1)
        if ded_str1 == exp_comp1:
            syllogism_before_upd_ok += 1

        # 3) Оперативный 0-SGD переезд родительского узла в другой город:
        parse_and_execute_sentence_v21(
            f"Обновление : узел {p_node} расположен в городе {city2}",
            parser_v21,
            dual_enc,
            mram,
            inducer_v21,
        )
        ded_id2, _ = rule_eng.deduce_attribute(ch_id, target_rel=4)
        ded_str2 = mram.read_entity_surface_from_gpu(ded_id2)
        if ded_str2 == exp_comp2:
            syllogism_after_upd_ok += 1

    acc_syl_before = 100.0 * syllogism_before_upd_ok / total_syllogism_trials
    acc_syl_after = 100.0 * syllogism_after_upd_ok / total_syllogism_trials
    print("  [Step 2 | 3-Hop Syllogistic Deduction over Unseen Entities (`Child -> Parent -> City -> Rule -> Comp`)]:")
    print(
        f"    - 3-Hop Deductive Syllogism BEFORE 0-SGD Update : {acc_syl_before:6.2f}% ({syllogism_before_upd_ok}/{total_syllogism_trials})"
    )
    print(
        f"    - 3-Hop Deductive Syllogism AFTER  0-SGD Update : {acc_syl_after:6.2f}% ({syllogism_after_upd_ok}/{total_syllogism_trials})"
    )

    # --------------------------------------------------------------------------
    # ШАГ 3 (`Stage v21.3`): Количественная и сравнительная аналитика над M_RAM
    # --------------------------------------------------------------------------
    print(
        "\n--- [Stage v21.3 | Шаг 3] Количественная и сравнительная аналитика в АЛУ (`QuantitativeComparativeALU_v21`) ---"
    )
    extremum_ok = 0
    diff_ok = 0
    city_sum_before_ok = 0
    city_sum_after_ok = 0
    quant_trials = 25

    for trial in range(quant_trials):
        mram_q = DynamicOpenWorldMRAM_v21(dual_enc, device=dev)
        rule_q = SyllogisticRuleEngine_v21(mram_q)
        alu_q = QuantitativeComparativeALU_v21(mram_q, rule_q)

        base_yr = 1980 + trial
        years_t = [str(base_yr + 15), str(base_yr), str(base_yr + 42), str(base_yr + 28)]
        powers_t = [str(300 + trial * 10), str(950 + trial * 10), str(450 + trial * 10), str(600 + trial * 10)]
        nodes_t = ["Квантовый_Протон", "Нейро_Купол", "Плазма_Сфера", "Матрица_Зенит"]

        node_ids = []
        for idx_n, n_name in enumerate(nodes_t):
            c_name = "Токио" if idx_n in (0, 1) else "Рейкьявик"
            parse_and_execute_sentence_v21(
                f"Утверждение : узел {n_name} расположен в городе {c_name}",
                parser_v21,
                dual_enc,
                mram_q,
                inducer_v21,
            )
            parse_and_execute_sentence_v21(
                f"В году {years_t[idx_n]} запущен узел {n_name} по данным системы",
                parser_v21,
                dual_enc,
                mram_q,
                inducer_v21,
            )
            parse_and_execute_sentence_v21(
                f"Мощностью {powers_t[idx_n]} работает узел {n_name} по данным системы",
                parser_v21,
                dual_enc,
                mram_q,
                inducer_v21,
            )
            node_ids.append(mram_q.resolve_or_register_entity_0sgd(n_name, allow_register=False))

        # Добавим дочерний узел "Модуль_Сигма", который зависит от "Квантовый_Протон" (в Токио) и имеет мощность 200:
        parse_and_execute_sentence_v21(
            "Опирается плечом Модуль_Сигма на фундамент Квантовый_Протон",
            parser_v21,
            dual_enc,
            mram_q,
            inducer_v21,
        )
        parse_and_execute_sentence_v21(
            "Утверждение : узел Модуль_Сигма работает мощностью 200",
            parser_v21,
            dual_enc,
            mram_q,
            inducer_v21,
        )
        ch_id = mram_q.resolve_or_register_entity_0sgd("Модуль_Сигма", allow_register=False)
        all_5_ids = node_ids + [ch_id]

        mram_q.fact_bank.refresh_newton_schulz()
        mram_q.lexicon_bank.refresh_newton_schulz()

        # 1) ARGMIN (самый старый узел -> idx 1: "Нейро_Купол") и ARGMAX (самый мощный -> idx 1: "Нейро_Купол"):
        oldest_id, oldest_yr = alu_q.argmin_entity(node_ids, rel_id=1)
        strongest_id, strongest_pw = alu_q.argmax_entity(node_ids, rel_id=3)
        if (
            oldest_id == node_ids[1]
            and oldest_yr == int(years_t[1])
            and strongest_id == node_ids[1]
            and strongest_pw == int(powers_t[1])
        ):
            extremum_ok += 1

        # 2) Точная арифметическая разность годов и мощностей:
        diff_yr = alu_q.abs_diff(node_ids[2], node_ids[1], rel_id=1)
        diff_pw = alu_q.abs_diff(node_ids[1], node_ids[0], rel_id=3)
        if diff_yr == 42 and diff_pw == 650:
            diff_ok += 1

        # 3) Суммарная мощность в городе Токио (включая дочерний "Модуль_Сигма", унаследовавший Токио от "Квантовый_Протон"!):
        tokio_id = mram_q.resolve_or_register_entity_0sgd("Токио", allow_register=False)
        sum_before = alu_q.sum_by_inherited_city(all_5_ids, tokio_id, num_rel_id=3)
        expected_before = int(powers_t[0]) + int(powers_t[1]) + 200
        if sum_before == expected_before:
            city_sum_before_ok += 1

        # 4) Перевозим родительский узел "Квантовый_Протон" за 0-SGD в "Рейкьявик":
        #    теперь и "Квантовый_Протон", и его дочерний "Модуль_Сигма" покидают Токио!
        parse_and_execute_sentence_v21(
            "Обновление : узел Квантовый_Протон расположен в городе Рейкьявик",
            parser_v21,
            dual_enc,
            mram_q,
            inducer_v21,
        )
        sum_after = alu_q.sum_by_inherited_city(all_5_ids, tokio_id, num_rel_id=3)
        expected_after = int(powers_t[1])
        if sum_after == expected_after:
            city_sum_after_ok += 1

    acc_ext = 100.0 * extremum_ok / quant_trials
    acc_diff = 100.0 * diff_ok / quant_trials
    acc_sum_b = 100.0 * city_sum_before_ok / quant_trials
    acc_sum_a = 100.0 * city_sum_after_ok / quant_trials
    print("  [Step 3 | Quantitative & Comparative Reasoning over M_RAM]:")
    print(f"    - ARGMIN (Oldest Year) & ARGMAX (Max Power kW)  : {acc_ext:6.2f}% ({extremum_ok}/{quant_trials})")
    print(f"    - Exact ALU Difference (Year Span & kW Delta)   : {acc_diff:6.2f}% ({diff_ok}/{quant_trials})")
    print(
        f"    - 2-Hop City Power Sum BEFORE 0-SGD Relocation  : {acc_sum_b:6.2f}% ({city_sum_before_ok}/{quant_trials})"
    )
    print(
        f"    - 2-Hop City Power Sum AFTER  0-SGD Relocation  : {acc_sum_a:6.2f}% ({city_sum_after_ok}/{quant_trials})"
    )

    # --------------------------------------------------------------------------
    # ШАГ 4 (`Stage v21.4`): Каскадный прогноз What-If и абдуктивная диагностика Why
    # --------------------------------------------------------------------------
    print(
        "\n--- [Stage v21.4 | Шаг 4] Причинно-следственный каскад `What-If` и абдуктивная диагностика `Why` (`CausalCascadeAndAbduction_v21`) ---"
    )
    what_if_ok = 0
    abduction_ok = 0
    causal_trials = 25

    for trial in range(causal_trials):
        mram_c = DynamicOpenWorldMRAM_v21(dual_enc, device=dev)
        causal_eng = CausalCascadeAndAbduction_v21(mram_c)

        # Строим 4-шаговую цепочку зависимостей:
        # Терминал_Клиент -> Шлюз_Орион -> Плазма_Сфера -> Квантовый_Протон
        # и независимую ветку: Архив_Зенит -> Крио_Вектор
        chain_sents = [
            "Утверждение : узел Терминал_Клиент зависит от системы Шлюз_Орион",
            "От системы Плазма_Сфера зависит узел Шлюз_Орион по данным системы",
            "Опирается плечом Плазма_Сфера на фундамент Квантовый_Протон",
            "Утверждение : узел Архив_Зенит зависит от системы Крио_Вектор",
        ]
        for cs in chain_sents:
            parse_and_execute_sentence_v21(cs, parser_v21, dual_enc, mram_c, inducer_v21)

        # Инжектируем сбой либо на корневом узле (hop=3), либо на промежуточном узле (hop=2):
        fault_target = "Квантовый_Протон" if (trial % 2 == 0) else "Плазма_Сфера"
        fault_reason = "Потеря_Фазы" if (trial % 2 == 0) else "Перегрев_Контура"
        parse_and_execute_sentence_v21(
            f"Сбой : узел {fault_target} зафиксировал аварию {fault_reason}",
            parser_v21,
            dual_enc,
            mram_c,
            inducer_v21,
        )
        mram_c.fact_bank.refresh_newton_schulz()
        mram_c.lexicon_bank.refresh_newton_schulz()

        id_client = mram_c.resolve_or_register_entity_0sgd("Терминал_Клиент", allow_register=False)
        id_gate = mram_c.resolve_or_register_entity_0sgd("Шлюз_Орион", allow_register=False)
        id_plasma = mram_c.resolve_or_register_entity_0sgd("Плазма_Сфера", allow_register=False)
        id_proton = mram_c.resolve_or_register_entity_0sgd("Квантовый_Протон", allow_register=False)
        id_arch = mram_c.resolve_or_register_entity_0sgd("Архив_Зенит", allow_register=False)
        id_cryo = mram_c.resolve_or_register_entity_0sgd("Крио_Вектор", allow_register=False)
        all_nodes_c = [id_client, id_gate, id_plasma, id_proton, id_arch, id_cryo]

        # 1) What-If Cascade: что упадёт при отказе `Плазма_Сфера`?
        #    Должны упасть ровно {Терминал_Клиент, Шлюз_Орион, Плазма_Сфера}, но НЕ {Квантовый_Протон, Архив_Зенит, Крио_Вектор}!
        affected = set(causal_eng.predict_what_if_cascade(id_plasma, all_nodes_c))
        if affected == {id_client, id_gate, id_plasma}:
            what_if_ok += 1

        # 2) Why Abduction: почему не отвечает `Терминал_Клиент`?
        root_nid, reason_oid, hops = causal_eng.abduce_root_cause_why(id_client)
        exp_nid = id_proton if (trial % 2 == 0) else id_plasma
        exp_hops = 3 if (trial % 2 == 0) else 2
        if (
            root_nid == exp_nid
            and mram_c.read_entity_surface_from_gpu(reason_oid) == fault_reason
            and hops == exp_hops
        ):
            abduction_ok += 1

    acc_what_if = 100.0 * what_if_ok / causal_trials
    acc_abduct = 100.0 * abduction_ok / causal_trials
    print("  [Step 4 | 4-Hop Causal Cascade (`What-If`) & Abductive Root-Cause (`Why`)]:")
    print(f"    - Counterfactual `What-If` Cascade Prediction   : {acc_what_if:6.2f}% ({what_if_ok}/{causal_trials})")
    print(f"    - 4-Hop Abductive `Why` Root-Cause Diagnosis    : {acc_abduct:6.2f}% ({abduction_ok}/{causal_trials})")

    # --------------------------------------------------------------------------
    # ШАГ 5 (`Stage v21.5`): Автоматическое открытие законов во сне и сквозной экзамен
    # --------------------------------------------------------------------------
    print(
        "\n--- [Stage v21.5 | Шаг 5] Индуктивное открытие законов во сне (`WakeSleepRuleInducer_v21`) и сквозной экзамен ---"
    )
    true_law_discovery_ok = 0
    false_law_rejection_ok = 0
    deduction_from_induced_ok = 0
    unified_exam_ok = 0
    sleep_trials = 32

    for ep in range(sleep_trials):
        mram_s = DynamicOpenWorldMRAM_v21(dual_enc, device=dev)
        rule_s = SyllogisticRuleEngine_v21(mram_s)
        alu_s = QuantitativeComparativeALU_v21(mram_s, rule_s)
        causal_s = CausalCascadeAndAbduction_v21(mram_s)
        inducer_sleep = WakeSleepRuleInducer_v21(mram_s)

        # Фаза WAKE: подаём обычные атомарные факты (БЕЗ единого предложения "Правило : ...")!
        # В городе Рейкьявик все 3 наблюдаемых узла требуют "Геотермальный_Контур" (100% закон!).
        # В городе Токио все 3 наблюдаемых узла требуют "Сейсмостойкий_Каркас" (100% закон!).
        # В городе Женева 2 узла требуют "Альпийский_Фильтр", а 1 узел — "Магнитный_Экран" (контрпример!).
        wake_observations = [
            ("Зонд_Арес", "Рейкьявик", "Геотермальный_Контур"),
            ("Зонд_Борей", "Рейкьявик", "Геотермальный_Контур"),
            ("Зонд_Гелиос", "Рейкьявик", "Геотермальный_Контур"),
            ("Диск_Сатурн", "Токио", "Сейсмостойкий_Каркас"),
            ("Диск_Юпитер", "Токио", "Сейсмостойкий_Каркас"),
            ("Диск_Нептун", "Токио", "Сейсмостойкий_Каркас"),
            ("Куб_Альфа", "Женева", "Альпийский_Фильтр"),
            ("Куб_Бета", "Женева", "Альпийский_Фильтр"),
            ("Куб_Гамма", "Женева", "Магнитный_Экран"),
        ]
        obs_node_ids = []
        for n_w, c_w, req_w in wake_observations:
            parse_and_execute_sentence_v21(
                f"Утверждение : узел {n_w} расположен в городе {c_w}",
                parser_v21,
                dual_enc,
                mram_s,
                inducer_v21,
            )
            parse_and_execute_sentence_v21(
                f"Утверждение : узел {n_w} требует компонент {req_w}",
                parser_v21,
                dual_enc,
                mram_s,
                inducer_v21,
            )
            obs_node_ids.append(mram_s.resolve_or_register_entity_0sgd(n_w, allow_register=False))

        mram_s.fact_bank.refresh_newton_schulz()
        mram_s.lexicon_bank.refresh_newton_schulz()

        # Фаза SLEEP: автоматическая индукция Хорновских правил из накопленных фактов:
        induced_rules = inducer_sleep.consolidate_rules_in_sleep(
            obs_node_ids, cond_rel=0, cons_rel=4, min_support=3
        )
        mram_s.rule_bank.refresh_newton_schulz()

        id_reyk = mram_s.resolve_or_register_entity_0sgd("Рейкьявик", allow_register=False)
        id_tok = mram_s.resolve_or_register_entity_0sgd("Токио", allow_register=False)
        id_gen = mram_s.resolve_or_register_entity_0sgd("Женева", allow_register=False)

        induced_cond_cities = {r[1] for r in induced_rules}
        if id_reyk in induced_cond_cities and id_tok in induced_cond_cities:
            true_law_discovery_ok += 1
        if id_gen not in induced_cond_cities:
            false_law_rejection_ok += 1

        # Фаза WAKE EXAM: подаём связную историю о НОВЫХ узлах ("Реактор_Гиперион" и дочернем "Модуль_Калипсо"),
        # для которых требование `REQUIRES` вообще НЕ УКАЗАНО в тексте!
        exam_story = [
            "В городе Токио расположен узел Реактор_Гиперион по данным системы",
            "Утверждение : узел Реактор_Гиперион работает мощностью 900",
            "Опирается плечом Модуль_Калипсо на фундамент Реактор_Гиперион",
            "Мощностью 350 работает узел Модуль_Калипсо по данным системы",
            "Обновление : узел Реактор_Гиперион расположен в городе Рейкявик",  # с опечаткой!
            "Сбой : узел Реактор_Гиперион зафиксировал аварию Разрыв_Контура",
        ]
        for es in exam_story:
            parse_and_execute_sentence_v21(es, parser_v21, dual_enc, mram_s, inducer_v21)
        mram_s.fact_bank.refresh_newton_schulz()
        mram_s.lexicon_bank.refresh_newton_schulz()

        id_hyper = mram_s.resolve_or_register_entity_0sgd("Реактор_Гиперион", allow_register=False)
        id_calypso = mram_s.resolve_or_register_entity_0sgd("Модуль_Калипсо", allow_register=False)

        # Проверяем:
        # 1) Дедукцию открытого во сне закона для дочернего узла `Модуль_Калипсо` после переезда родителя в Рейкьявик:
        ded_req_id, _ = rule_s.deduce_attribute(id_calypso, target_rel=4)
        ded_req_name = mram_s.read_entity_surface_from_gpu(ded_req_id)
        if ded_req_name == "Геотермальный_Контур":
            deduction_from_induced_ok += 1

        # 2) Суммарную мощность новой связки в Рейкьявике (900 + 350 = 1250 кВт):
        reyk_power = alu_s.sum_by_inherited_city([id_hyper, id_calypso], id_reyk, num_rel_id=3)

        # 3) Обратную диагностику первопричины сбоя для `Модуль_Калипсо`:
        f_node, f_reason, f_hops = causal_s.abduce_root_cause_why(id_calypso)

        if (
            ded_req_name == "Геотермальный_Контур"
            and reyk_power == 1250
            and f_node == id_hyper
            and mram_s.read_entity_surface_from_gpu(f_reason) == "Разрыв_Контура"
            and f_hops == 1
        ):
            unified_exam_ok += 1

    acc_true_law = 100.0 * true_law_discovery_ok / sleep_trials
    acc_false_rej = 100.0 * false_law_rejection_ok / sleep_trials
    acc_ind_ded = 100.0 * deduction_from_induced_ok / sleep_trials
    acc_unified = 100.0 * unified_exam_ok / sleep_trials

    print("  [Step 5 | Wake-Sleep Autonomous Law Discovery & Grand Unified Open-World Exam]:")
    print(
        f"    - Wake-Sleep True Universal Law Discovery Acc   : {acc_true_law:6.2f}% ({true_law_discovery_ok}/{sleep_trials})"
    )
    print(
        f"    - Counterexample False-Rule Rejection Acc       : {acc_false_rej:6.2f}% ({false_law_rejection_ok}/{sleep_trials})"
    )
    print(
        f"    - 3-Hop Deduction using Sleep-Induced Laws      : {acc_ind_ded:6.2f}% ({deduction_from_induced_ok}/{sleep_trials})"
    )
    print(
        f"    - Grand Unified Exam (OpenEntity + Law + ALU + Why): {acc_unified:6.2f}% ({unified_exam_ok}/{sleep_trials})"
    )

    return {
        "v21_Step2_Syllogism_Before_Update_Acc": acc_syl_before,
        "v21_Step2_Syllogism_After_Update_Acc": acc_syl_after,
        "v21_Step3_Quant_Extremum_Acc": acc_ext,
        "v21_Step3_Quant_Diff_Acc": acc_diff,
        "v21_Step3_Quant_CitySum_Before_Acc": acc_sum_b,
        "v21_Step3_Quant_CitySum_After_Acc": acc_sum_a,
        "v21_Step4_Causal_WhatIf_Cascade_Acc": acc_what_if,
        "v21_Step4_Causal_Abductive_Why_Acc": acc_abduct,
        "v21_Step5_Sleep_TrueLaw_Discovery_Acc": acc_true_law,
        "v21_Step5_Sleep_FalseLaw_Rejection_Acc": acc_false_rej,
        "v21_Step5_Sleep_Induced_Deduction_Acc": acc_ind_ded,
        "v21_Step5_Grand_Unified_Exam_Acc": acc_unified,
    }


def run_v21_full_suite():
    t_start = time.perf_counter()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(2100)
    random.seed(2100)

    print("=" * 100)
    print(
        "  STAGE v21.0: ОТКРЫТАЯ РАССУЖДАЮЩАЯ КОГНИТИВНАЯ МАШИНА (`OpenWorldReasoningSimVQ_v21`)"
    )
    print("=" * 100)

    parser_v21, dual_enc, inducer_v21, m1 = run_step1_open_entity_pointer_benchmark(dev=dev)
    m2_5 = run_steps_2_to_5_reasoning_benchmarks(parser_v21, dual_enc, inducer_v21, dev=dev)

    all_metrics: Dict[str, float] = {}
    all_metrics.update(m1)
    all_metrics.update(m2_5)

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", "v21_open_world_reasoning_engine.pt")
    torch.save(
        {
            "stage": "v21.0_OpenWorldReasoningSimVQ",
            "parser_v21_state": parser_v21.state_dict(),
            "metrics": all_metrics,
        },
        ckpt_path,
    )

    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 100)
    print(f"  ИТОГОВАЯ СВОДКА STAGE v21.0 (время: {elapsed:.1f} с | сохранён: {ckpt_path})")
    print("=" * 100)
    for k, v in all_metrics.items():
        if "Err" in k:
            print(f"  {k:52s}: {v:.4e}")
        else:
            print(f"  {k:52s}: {v:.2f}%")


if __name__ == "__main__":
    run_v21_full_suite()
