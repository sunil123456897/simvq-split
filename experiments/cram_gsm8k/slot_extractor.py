"""
slot_extractor.py — Безрегексный посимвольный сканер величин и data-driven индуктор
словесных числительных / единичных констант строго по обучающей выборке `train.jsonl`.

Инварианты Фазы C:
  - Нулевой `import re`, нулевой `import ast`, нулевой `eval`/`exec`.
  - Отсутствуют зашитые вручную словари числительных и регулярные сплиттеры клауз.
  - Вся связь слово/биграмма -> числительное/константа и анти-контексты индуцируются
    в `.fit(train_items)` из статистики выравнивания `(question, <<lhs=rhs>>)`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


OP_TO_ID: Dict[str, int] = {"+": 0, "-": 1, "*": 2, "/": 3}
ID_TO_OP: Tuple[str, ...] = ("+", "-", "*", "/")
PREC: Dict[str, int] = {"+": 1, "-": 1, "*": 2, "/": 2}


@dataclass(frozen=True)
class ScannedToken:
    text: str
    kind: str  # "NUM", "WORD", "PUNCT"
    b0: int
    b1: int
    sent_id: int
    clause_id: int
    num_val: Optional[float] = None
    val_str: str = ""
    is_pct: bool = False
    has_dollar: bool = False
    lower: str = ""


@dataclass(frozen=True)
class ExtractedSlot:
    idx: int
    val: float
    val_str: str
    kind: str  # "EXPLICIT", "INDUCED_WORD", "CONST"
    b0: int
    b1: int
    sent_id: int
    clause_id: int
    is_in_question: bool
    is_pct: bool
    has_dollar: bool
    left_words: Tuple[str, ...]
    right_words: Tuple[str, ...]
    trigger_token: str = ""


def _parse_float_safe(s: str) -> Optional[float]:
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except ValueError:
        return None


def _fmt_decimal_str(val: float) -> str:
    try:
        d = Decimal(repr(val))
        return format(d, "f") if abs(val) < 1e12 else repr(val)
    except InvalidOperation:
        return repr(val)


def scan_question_tokens(text: str) -> List[ScannedToken]:
    """Посимвольно разбирает строку на токены с точными UTF-8 байтовыми границами `(b0, b1)`."""
    tokens: List[ScannedToken] = []
    n = len(text)
    i = 0
    sent_id = 0
    clause_id = 0

    # Таблица префиксных байтовых смещений UTF-8 (O(N), ноль повторных encode)
    byte_offsets = [0] * (n + 1)
    for idx_ch, ch in enumerate(text):
        byte_offsets[idx_ch + 1] = byte_offsets[idx_ch] + len(ch.encode("utf-8"))

    while i < n:
        ch = text[i]
        if ch.isspace():
            if ch == "\n":
                sent_id += 1
                clause_id += 1
            i += 1
            continue

        if ch.isdigit() or (ch == "." and i + 1 < n and text[i + 1].isdigit()):
            has_dollar = i > 0 and text[i - 1] == "$"
            start = i
            has_dot = ch == "."
            i += 1
            while i < n:
                c = text[i]
                if c.isdigit():
                    i += 1
                elif (
                    c == ","
                    and i + 3 < n
                    and text[i + 1 : i + 4].isdigit()
                    and (i + 4 >= n or not text[i + 4].isdigit())
                ):
                    i += 1
                elif c == "." and not has_dot and i + 1 < n and text[i + 1].isdigit():
                    has_dot = True
                    i += 1
                else:
                    break

            clean_num = text[start:i].replace(",", "")
            is_pct = False
            val: Optional[float] = None
            end_pos = i

            if i < n and text[i] == "%":
                is_pct = True
                end_pos = i + 1
                i = end_pos
                pv = _parse_float_safe(clean_num)
                if pv is not None:
                    val = pv / 100.0
            elif i < n and text[i] == "/" and i + 1 < n and text[i + 1].isdigit():
                j = i + 1
                while j < n and (text[j].isdigit() or text[j] == "."):
                    j += 1
                nv = _parse_float_safe(clean_num)
                dv = _parse_float_safe(text[i + 1 : j])
                if nv is not None and dv is not None and abs(dv) > 1e-12:
                    val = nv / dv
                    end_pos = j
                    i = j
            elif clean_num.isdigit() and i + 3 < n and text[i] == " " and text[i + 1].isdigit():
                # Поддержка смешанной дроби вида "1 1/2"
                j = i + 1
                while j < n and text[j].isdigit():
                    j += 1
                if j < n and text[j] == "/" and j + 1 < n and text[j + 1].isdigit():
                    k = j + 1
                    while k < n and text[k].isdigit():
                        k += 1
                    w_v = _parse_float_safe(clean_num)
                    num_v = _parse_float_safe(text[i + 1 : j])
                    den_v = _parse_float_safe(text[j + 1 : k])
                    if w_v is not None and num_v is not None and den_v is not None and abs(den_v) > 1e-12:
                        val = w_v + (num_v / den_v)
                        end_pos = k
                        i = k

            if val is None:
                val = _parse_float_safe(clean_num)
            if val is not None:
                raw_tok = text[start:end_pos]
                tokens.append(
                    ScannedToken(
                        text=raw_tok,
                        kind="NUM",
                        b0=byte_offsets[start],
                        b1=byte_offsets[end_pos],
                        sent_id=sent_id,
                        clause_id=clause_id,
                        num_val=val,
                        val_str=_fmt_decimal_str(val),
                        is_pct=is_pct,
                        has_dollar=has_dollar,
                        lower=raw_tok.lower(),
                    )
                )
            continue

        if ch.isalpha():
            start = i
            while i < n and (text[i].isalpha() or text[i] == "'"):
                i += 1
            raw_tok = text[start:i]
            tokens.append(
                ScannedToken(
                    text=raw_tok,
                    kind="WORD",
                    b0=byte_offsets[start],
                    b1=byte_offsets[i],
                    sent_id=sent_id,
                    clause_id=clause_id,
                    lower=raw_tok.lower(),
                )
            )
            continue

        tokens.append(
            ScannedToken(
                text=ch,
                kind="PUNCT",
                b0=byte_offsets[i],
                b1=byte_offsets[i + 1],
                sent_id=sent_id,
                clause_id=clause_id,
                lower=ch,
            )
        )
        if ch in ".?!":
            sent_id += 1
            clause_id += 1
        elif ch in ",;:":
            clause_id += 1
        i += 1

    return tokens


def extract_answer_tags(answer_text: str) -> List[Tuple[str, str]]:
    """Извлекает пары `(lhs, rhs)` внутри `<<...>>` без `re`."""
    tags: List[Tuple[str, str]] = []
    pos = 0
    n = len(answer_text)
    while pos < n:
        s_idx = answer_text.find("<<", pos)
        if s_idx < 0:
            break
        e_idx = answer_text.find(">>", s_idx + 2)
        if e_idx < 0:
            break
        inner = answer_text[s_idx + 2 : e_idx]
        if "=" in inner:
            eq_idx = inner.rfind("=")
            lhs = inner[:eq_idx].strip()
            rhs = inner[eq_idx + 1 :].strip().replace(",", "")
            tags.append((lhs, rhs))
        pos = e_idx + 2
    return tags


def extract_gt_number(answer_text: str) -> float:
    """Извлекает финальный эталон ответа после `####` строго для train/eval отчёта."""
    if "####" not in answer_text:
        raise ValueError("Missing #### delimiter in answer_text")
    tail = answer_text.split("####")[-1].strip().replace(",", "").replace("$", "")
    return float(tail)


def tokenize_arithmetic_expr(expr: str) -> Optional[List[float | str]]:
    """Лексер арифметических выражений в тегах `<<lhs>>` без `re` и `ast`."""
    s = expr.replace(",", "").replace("$", "").strip()
    out: List[float | str] = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            continue
        is_unary = (
            c in "+-"
            and (not out or out[-1] in ("(", "+", "-", "*", "/"))
            and i + 1 < n
            and (s[i + 1].isdigit() or s[i + 1] == ".")
        )
        if c.isdigit() or (c == "." and i + 1 < n and s[i + 1].isdigit()) or is_unary:
            st = i
            i += 1
            has_dot = c == "."
            while i < n and (s[i].isdigit() or (s[i] == "." and not has_dot)):
                if s[i] == ".":
                    has_dot = True
                i += 1
            v = _parse_float_safe(s[st:i])
            if v is None:
                return None
            if i < n and s[i] == "%":
                v /= 100.0
                i += 1
            out.append(v)
        elif c in "+-*/()":
            out.append(c)
            i += 1
        elif c.isalpha():
            while i < n and s[i].isalpha():
                i += 1
        else:
            return None
    return out


def _match_value_index(v: float, cur_vals: Sequence[float]) -> int:
    best_i = -1
    best_err = 1e12
    for idx, cv in enumerate(cur_vals):
        err = abs(v - cv)
        tol = max(1e-3, 1e-4 * abs(cv))
        if err <= tol and err < best_err:
            best_err = err
            best_i = idx
    return best_i


def _shunting_yard_rpn(toks: Sequence[float | str]) -> Optional[List[float | str]]:
    out: List[float | str] = []
    ops: List[str] = []
    for tk in toks:
        if isinstance(tk, float):
            out.append(tk)
        elif tk in OP_TO_ID:
            while ops and ops[-1] in OP_TO_ID and PREC[ops[-1]] >= PREC[tk]:
                out.append(ops.pop())
            ops.append(tk)
        elif tk == "(":
            ops.append("(")
        elif tk == ")":
            while ops and ops[-1] != "(":
                out.append(ops.pop())
            if not ops:
                return None
            ops.pop()
        else:
            return None
    while ops:
        if ops[-1] == "(":
            return None
        out.append(ops.pop())
    return out


def _collect_prose_intermediates(answer_text: str) -> List[float]:
    """Собирает числа, встречающиеся в пояснительном тексте train `answer` до `####`."""
    body = answer_text.split("####")[0]
    vals: List[float] = []
    for tk in scan_question_tokens(body):
        if tk.kind == "NUM" and tk.num_val is not None:
            vals.append(float(tk.num_val))
    return vals


class TrainInducedQuantityExtractor:
    """Автоматически индуцирует словесные числительные, единичные константы,
    контекстные анти-биграммы и словарь контекстных токенов исключительно на `train_split`."""

    def __init__(
        self,
        min_word_df: int = 12,
        min_cooc: int = 8,
        word_cond_thresh: float = 0.45,
        const_cond_thresh: float = 0.14,
        min_z_score: float = 4.8,
        min_lift: float = 2.0,
        max_vocab_tokens: int = 1024,
        max_aux_consts: int = 2,
    ):
        self.min_word_df = min_word_df
        self.min_cooc = min_cooc
        self.word_cond_thresh = word_cond_thresh
        self.const_cond_thresh = const_cond_thresh
        self.min_z_score = min_z_score
        self.min_lift = min_lift
        self.max_vocab_tokens = max_vocab_tokens
        self.max_aux_consts = max_aux_consts

        self.word_num_map: Dict[str, float] = {}
        self.unit_const_map: Dict[str, float] = {}
        self.bigram_suppress: Set[str] = set()
        self.structural_consts: List[float] = []
        self.ctx_token_to_id: Dict[str, int] = {}
        self.slot_op_affinity: Dict[str, np.ndarray] = {}
        self.q_root_affinity: Dict[str, np.ndarray] = {}
        self.pair_mid_affinity: Dict[str, np.ndarray] = {}
        self.fitted: bool = False

    def state_dict(self) -> Dict[str, object]:
        """Сериализует индуцированные словари и Empirical-Bayes аффинности без pickle-классов."""
        return {
            "min_word_df": int(self.min_word_df),
            "min_cooc": int(self.min_cooc),
            "word_cond_thresh": float(self.word_cond_thresh),
            "const_cond_thresh": float(self.const_cond_thresh),
            "min_z_score": float(self.min_z_score),
            "min_lift": float(self.min_lift),
            "max_vocab_tokens": int(self.max_vocab_tokens),
            "max_aux_consts": int(self.max_aux_consts),
            "word_num_map": dict(self.word_num_map),
            "unit_const_map": dict(self.unit_const_map),
            "bigram_suppress": sorted(self.bigram_suppress),
            "structural_consts": list(self.structural_consts),
            "ctx_token_to_id": dict(self.ctx_token_to_id),
            "slot_op_affinity": {k: v.tolist() for k, v in self.slot_op_affinity.items()},
            "q_root_affinity": {k: v.tolist() for k, v in self.q_root_affinity.items()},
            "pair_mid_affinity": {k: v.tolist() for k, v in self.pair_mid_affinity.items()},
            "fitted": bool(self.fitted),
        }

    def load_state_dict(self, sd: Dict[str, object]) -> None:
        """Восстанавливает индуцированное состояние экстрактора из словаря весов."""
        self.min_word_df = int(sd.get("min_word_df", self.min_word_df))  # type: ignore[arg-type]
        self.min_cooc = int(sd.get("min_cooc", self.min_cooc))  # type: ignore[arg-type]
        self.word_cond_thresh = float(sd.get("word_cond_thresh", self.word_cond_thresh))  # type: ignore[arg-type]
        self.const_cond_thresh = float(sd.get("const_cond_thresh", self.const_cond_thresh))  # type: ignore[arg-type]
        self.min_z_score = float(sd.get("min_z_score", self.min_z_score))  # type: ignore[arg-type]
        self.min_lift = float(sd.get("min_lift", self.min_lift))  # type: ignore[arg-type]
        self.max_vocab_tokens = int(sd.get("max_vocab_tokens", self.max_vocab_tokens))  # type: ignore[arg-type]
        self.max_aux_consts = int(sd.get("max_aux_consts", self.max_aux_consts))  # type: ignore[arg-type]
        self.word_num_map = {str(k): float(v) for k, v in dict(sd.get("word_num_map", {})).items()}  # type: ignore[arg-type]
        self.unit_const_map = {str(k): float(v) for k, v in dict(sd.get("unit_const_map", {})).items()}  # type: ignore[arg-type]
        self.bigram_suppress = {str(x) for x in list(sd.get("bigram_suppress", []))}  # type: ignore[arg-type]
        self.structural_consts = [float(x) for x in list(sd.get("structural_consts", []))]  # type: ignore[arg-type]
        self.ctx_token_to_id = {str(k): int(v) for k, v in dict(sd.get("ctx_token_to_id", {})).items()}  # type: ignore[arg-type]
        self.slot_op_affinity = {
            str(k): np.asarray(v, dtype=np.float32)
            for k, v in dict(sd.get("slot_op_affinity", {})).items()  # type: ignore[arg-type]
        }
        self.q_root_affinity = {
            str(k): np.asarray(v, dtype=np.float32)
            for k, v in dict(sd.get("q_root_affinity", {})).items()  # type: ignore[arg-type]
        }
        self.pair_mid_affinity = {
            str(k): np.asarray(v, dtype=np.float32)
            for k, v in dict(sd.get("pair_mid_affinity", {})).items()  # type: ignore[arg-type]
        }
        self.fitted = bool(sd.get("fitted", True))

    def _raw_explicit_slots(self, tokens: Sequence[ScannedToken]) -> List[Tuple[int, float, bool, int, int, bool]]:
        """Возвращает `(tok_idx, val, is_pct, b0, b1, has_dollar)` для явных чисел с учётом `percent*`."""
        out: List[Tuple[int, float, bool, int, int, bool]] = []
        n = len(tokens)
        for ti, tk in enumerate(tokens):
            if tk.kind == "NUM" and tk.num_val is not None:
                v = float(tk.num_val)
                is_pct = tk.is_pct
                if not is_pct and ti + 1 < n and tokens[ti + 1].kind == "WORD":
                    if tokens[ti + 1].lower.startswith("percent"):
                        v /= 100.0
                        is_pct = True
                out.append((ti, v, is_pct, tk.b0, tk.b1, tk.has_dollar))
        return out

    @staticmethod
    def _missing_operands(base_vals: Sequence[float], tags: Sequence[Tuple[str, str]]) -> Set[float]:
        """Возвращает числа из `<<lhs=rhs>>`, ещё не объяснённые текущими слотами и промежуточными `rhs`."""
        miss: Set[float] = set()
        cur = list(base_vals)
        for lhs, rhs in tags:
            for et in (tokenize_arithmetic_expr(lhs) or []):
                if isinstance(et, float) and _match_value_index(et, cur) < 0:
                    miss.add(et)
            rv = _parse_float_safe(rhs)
            if rv is not None:
                cur.append(rv)
        return miss

    def fit(self, train_items: Sequence[Dict[str, str]]) -> Dict[str, int]:
        """Учит все отображения и словарь контекстов в 3 каскада explaining-away строго из `train_items`."""
        df_uni: Counter[str] = Counter()
        df_bi: Counter[str] = Counter()
        cooc_uni1: Dict[str, Counter[float]] = defaultdict(Counter)
        cooc_bi1: Dict[str, Counter[float]] = defaultdict(Counter)
        mv_tot1: Counter[float] = Counter()
        ctx_freq: Counter[str] = Counter()
        parsed: List[Tuple[List[ScannedToken], List[float], List[Tuple[str, str]], Set[str], Set[str]]] = []

        n_tr = max(1, len(train_items))
        for it in train_items:
            q_str = str(it["question"])
            tokens = scan_question_tokens(q_str)
            expl = self._raw_explicit_slots(tokens)
            cur_vals: List[float] = []
            for _, v, is_pct, _, _, _ in expl:
                cur_vals.append(v)
                if is_pct:
                    # Проценты объясняют v, v*100, 100 и 1 до статистической индукции слов
                    cur_vals.extend([v * 100.0, 100.0, 1.0])

            q_unis: Set[str] = set()
            q_bis: Set[str] = set()
            for ti, tk in enumerate(tokens):
                if tk.kind == "WORD":
                    lw = tk.lower
                    q_unis.add(lw)
                    ctx_freq[lw] += 1
                    if ti > 0 and tokens[ti - 1].kind == "WORD":
                        bi = tokens[ti - 1].lower + "_" + lw
                        q_bis.add(bi)
                        ctx_freq["bi:" + bi] += 1

            for w in q_unis:
                df_uni[w] += 1
            for bi in q_bis:
                df_bi[bi] += 1

            tags = extract_answer_tags(str(it["answer"]))
            miss1 = self._missing_operands(cur_vals, tags)
            for mv in miss1:
                mv_tot1[mv] += 1
                for w in q_unis:
                    cooc_uni1[w][mv] += 1
                for bi in q_bis:
                    cooc_bi1[bi][mv] += 1
            parsed.append((tokens, cur_vals, tags, q_unis, q_bis))

        self.word_num_map.clear()
        self.unit_const_map.clear()
        self.bigram_suppress.clear()
        self.structural_consts.clear()

        def _pick_assoc(
            df: int,
            co_map: Counter[float],
            mv_tot: Counter[float],
            min_df: int,
            min_co: int,
            min_z: float,
            min_cond: float,
            min_diff: float,
            min_lift_val: float,
        ) -> Optional[float]:
            if df < min_df:
                return None
            cands: List[Tuple[float, float, float]] = []
            for mv, co in co_map.items():
                if co < min_co:
                    continue
                p_m = mv_tot[mv] / float(n_tr)
                cond = co / float(df)
                lift = cond / max(p_m, 1e-9)
                se = math.sqrt(max(p_m * (1.0 - p_m) / float(df), 1e-9))
                z = (cond - p_m) / se
                if z >= min_z and cond >= min_cond and (cond - p_m) >= min_diff and lift >= min_lift_val:
                    cands.append((cond, z, mv))
            if not cands:
                return None
            cands.sort(reverse=True)
            return cands[0][2]

        # Стадия 1: униграммы числительных (`word_num_map`) + биграммы-ингибиторы
        stage1_min_z = max(6.0, self.min_z_score)
        for w, df in df_uni.items():
            mv = _pick_assoc(
                df,
                cooc_uni1[w],
                mv_tot1,
                min_df=self.min_word_df,
                min_co=self.min_cooc,
                min_z=stage1_min_z,
                min_cond=self.word_cond_thresh,
                min_diff=0.25,
                min_lift_val=max(2.0, self.min_lift),
            )
            if mv is not None:
                self.word_num_map[w] = mv

        for w, target_mv in self.word_num_map.items():
            base_cond = cooc_uni1[w][target_mv] / max(1, df_uni[w])
            for bi, df in df_bi.items():
                if df < 6:
                    continue
                parts = bi.split("_", 1)
                if len(parts) == 2 and (parts[0] == w or parts[1] == w):
                    bi_cond = cooc_bi1[bi][target_mv] / float(df)
                    if bi_cond < 0.35 * base_cond:
                        self.bigram_suppress.add(bi)

        # Стадия 2: объясняем away слова Стадии 1, затем индуцируем единичные константы (`unit_const_map`)
        cooc_uni2: Dict[str, Counter[float]] = defaultdict(Counter)
        mv_tot2: Counter[float] = Counter()
        n_tokens_parsed = []
        for tokens, cur_vals, tags, q_unis, q_bis in parsed:
            cur2 = list(cur_vals)
            n_tk = len(tokens)
            for ti, tk in enumerate(tokens):
                if tk.kind != "WORD" or tk.lower not in self.word_num_map:
                    continue
                p_bi = (tokens[ti - 1].lower + "_" + tk.lower) if (ti > 0 and tokens[ti - 1].kind == "WORD") else ""
                n_bi = (tk.lower + "_" + tokens[ti + 1].lower) if (ti + 1 < n_tk and tokens[ti + 1].kind == "WORD") else ""
                if (p_bi in self.bigram_suppress) or (n_bi in self.bigram_suppress):
                    continue
                cur2.append(self.word_num_map[tk.lower])
            miss2 = self._missing_operands(cur2, tags)
            for mv in miss2:
                mv_tot2[mv] += 1
                for w in q_unis:
                    if w not in self.word_num_map:
                        cooc_uni2[w][mv] += 1
            n_tokens_parsed.append((tokens, cur2, tags, q_bis))

        stage2_co = max(7, self.min_cooc - 1)
        for w, df in df_uni.items():
            if w in self.word_num_map:
                continue
            mv = _pick_assoc(
                df,
                cooc_uni2[w],
                mv_tot2,
                min_df=self.min_word_df,
                min_co=stage2_co,
                min_z=self.min_z_score,
                min_cond=self.const_cond_thresh,
                min_diff=0.10,
                min_lift_val=max(2.2, self.min_lift),
            )
            if mv is not None:
                self.unit_const_map[w] = mv

        # Стадия 3: остаточные биграммы в `unit_const_map` (когда ни одно из слов ещё не объяснено)
        cooc_bi3: Dict[str, Counter[float]] = defaultdict(Counter)
        mv_tot3: Counter[float] = Counter()
        for tokens, cur2, tags, q_bis in n_tokens_parsed:
            cur3 = list(cur2) + [
                self.unit_const_map[tk.lower]
                for tk in tokens
                if tk.kind == "WORD" and tk.lower in self.unit_const_map
            ]
            miss3 = self._missing_operands(cur3, tags)
            for mv in miss3:
                mv_tot3[mv] += 1
                for bi in q_bis:
                    w1, w2 = bi.split("_", 1)
                    if (
                        w1 not in self.word_num_map
                        and w2 not in self.word_num_map
                        and w1 not in self.unit_const_map
                        and w2 not in self.unit_const_map
                    ):
                        cooc_bi3[bi][mv] += 1

        stage3_min_z = max(5.8, self.min_z_score)
        for bi, df in df_bi.items():
            w1, w2 = bi.split("_", 1)
            if (
                w1 in self.word_num_map
                or w2 in self.word_num_map
                or w1 in self.unit_const_map
                or w2 in self.unit_const_map
            ):
                continue
            mv = _pick_assoc(
                df,
                cooc_bi3[bi],
                mv_tot3,
                min_df=self.min_word_df,
                min_co=self.min_cooc,
                min_z=stage3_min_z,
                min_cond=0.28,
                min_diff=0.20,
                min_lift_val=2.8,
            )
            if mv is not None:
                self.unit_const_map[bi] = mv

        # Компактный словарь контекста вокруг слотов и вопроса
        self.ctx_token_to_id.clear()
        for idx_tok, (tok_key, _) in enumerate(ctx_freq.most_common(self.max_vocab_tokens)):
            self.ctx_token_to_id[tok_key] = idx_tok

        self.fitted = True
        self._fit_empirical_bayes_affinities(train_items)
        return {
            "word_num_map": len(self.word_num_map),
            "unit_const_map": len(self.unit_const_map),
            "bigram_suppress": len(self.bigram_suppress),
            "ctx_vocab": len(self.ctx_token_to_id),
            "slot_aff_keys": len(self.slot_op_affinity),
            "q_aff_keys": len(self.q_root_affinity),
            "pair_aff_keys": len(self.pair_mid_affinity),
        }

    @staticmethod
    def _slot_affinity_keys(sl: ExtractedSlot) -> List[str]:
        keys: List[str] = []
        lw = sl.left_words[-3:]
        rw = sl.right_words[:3]
        for w in lw:
            keys.append("L:" + w)
        for w in rw:
            keys.append("R:" + w)
        for j in range(len(lw) - 1):
            keys.append("LB:" + lw[j] + "_" + lw[j + 1])
        for j in range(len(rw) - 1):
            keys.append("RB:" + rw[j] + "_" + rw[j + 1])
        return keys

    def _question_affinity_keys(self, q_text: str) -> List[str]:
        q_focus = self.extract_question_sentence(q_text)
        toks = [tk.lower for tk in scan_question_tokens(q_focus) if tk.kind == "WORD"]
        keys: List[str] = list(toks)
        for j in range(len(toks) - 1):
            keys.append("bi:" + toks[j] + "_" + toks[j + 1])
        return list(dict.fromkeys(keys))

    @staticmethod
    def _between_pair_keys(tokens: Sequence[ScannedToken], sl_a: ExtractedSlot, sl_b: ExtractedSlot) -> List[str]:
        between = [
            tk.lower
            for tk in tokens
            if tk.kind == "WORD" and sl_a.b1 <= tk.b0 and tk.b1 <= sl_b.b0
        ]
        if not between or len(between) > 10:
            return []
        return list(dict.fromkeys("M:" + w for w in between))

    @staticmethod
    def _eb_log_odds(
        key_tot: Counter[str],
        key_cls: Dict[str, np.ndarray],
        tot_n: int,
        cls_tot: np.ndarray,
        min_df: int = 8,
        alpha_0: float = 5.0,
    ) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        if tot_n <= 0:
            return out
        p0 = (cls_tot + 1.0) / float(tot_n + 2.0)
        base_lo = np.log(np.clip(p0, 1e-5, 1.0 - 1e-5) / np.clip(1.0 - p0, 1e-5, 1.0 - 1e-5))
        for k, n_k in key_tot.items():
            if n_k < min_df:
                continue
            p_post = (key_cls[k] + alpha_0 * p0) / float(n_k + alpha_0)
            lo = np.log(np.clip(p_post, 1e-5, 1.0 - 1e-5) / np.clip(1.0 - p_post, 1e-5, 1.0 - 1e-5)) - base_lo
            out[k] = np.clip(lo, -3.0, 3.0).astype(np.float32)
        return out

    def _fit_empirical_bayes_affinities(self, train_items: Sequence[Dict[str, str]]) -> None:
        """Учит Empirical-Bayes log-odds связи контекстных токенов с ролями/операциями
        строго по скомпилированным графам на `train_items`."""
        slot_tot: Counter[str] = Counter()
        slot_cls: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(7, dtype=np.float64))
        slot_n_all = 0
        slot_cls_all = np.zeros(7, dtype=np.float64)

        q_tot: Counter[str] = Counter()
        q_cls: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(4, dtype=np.float64))
        q_n_all = 0
        q_cls_all = np.zeros(4, dtype=np.float64)

        pair_tot: Counter[str] = Counter()
        pair_cls: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(6, dtype=np.float64))
        pair_n_all = 0
        pair_cls_all = np.zeros(6, dtype=np.float64)

        for it in train_items:
            q_str = str(it["question"])
            slots, tokens, _ = self.extract_slots(q_str, include_aux_consts=True)
            steps, _, ok = self.compile_train_dag(slots, str(it["answer"]))
            if not ok or not steps:
                continue

            # 1. Финальная операция корневого вопроса
            root_op = steps[-1][0]
            q_n_all += 1
            q_cls_all[root_op] += 1.0
            for qk in self._question_affinity_keys(q_str):
                q_tot[qk] += 1
                q_cls[qk][root_op] += 1.0

            # 2. Участие заземлённых слотов в [UNUSED(0), ADD(1), SUB_L(2), SUB_R(3), MUL(4), DIV_N(5), DIV_D(6)]
            k_slots = len(slots)
            slot_traits = [set() for _ in range(k_slots)]
            bond_traits: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
            for op_id, a_i, b_i in steps:
                if op_id == 0:
                    if a_i < k_slots:
                        slot_traits[a_i].add(1)
                    if b_i < k_slots:
                        slot_traits[b_i].add(1)
                elif op_id == 1:
                    if a_i < k_slots:
                        slot_traits[a_i].add(2)
                    if b_i < k_slots:
                        slot_traits[b_i].add(3)
                elif op_id == 2:
                    if a_i < k_slots:
                        slot_traits[a_i].add(4)
                    if b_i < k_slots:
                        slot_traits[b_i].add(4)
                else:
                    if a_i < k_slots:
                        slot_traits[a_i].add(5)
                    if b_i < k_slots:
                        slot_traits[b_i].add(6)

                if a_i < k_slots and b_i < k_slots and a_i != b_i:
                    lo_i, hi_i = (a_i, b_i) if a_i < b_i else (b_i, a_i)
                    if op_id == 0:
                        bond_traits[(lo_i, hi_i)].add(0)
                    elif op_id == 1:
                        bond_traits[(lo_i, hi_i)].add(1 if a_i == lo_i else 2)
                    elif op_id == 2:
                        bond_traits[(lo_i, hi_i)].add(3)
                    else:
                        bond_traits[(lo_i, hi_i)].add(4 if a_i == lo_i else 5)

            for i, sl in enumerate(slots):
                if sl.kind not in ("EXPLICIT", "INDUCED_WORD"):
                    continue
                traits = slot_traits[i] if slot_traits[i] else {0}
                vec = np.zeros(7, dtype=np.float64)
                for tr in traits:
                    vec[tr] = 1.0
                slot_n_all += 1
                slot_cls_all += vec
                for sk in set(self._slot_affinity_keys(sl)):
                    slot_tot[sk] += 1
                    slot_cls[sk] += vec

            # 3. Прямые межслотовые связи (i < j) по словам между слотами
            for i in range(k_slots):
                for j in range(i + 1, k_slots):
                    mkeys = self._between_pair_keys(tokens, slots[i], slots[j])
                    if not mkeys:
                        continue
                    b_vec = np.zeros(6, dtype=np.float64)
                    for bt in bond_traits.get((i, j), ()):
                        b_vec[bt] = 1.0
                    pair_n_all += 1
                    pair_cls_all += b_vec
                    for mk in mkeys:
                        pair_tot[mk] += 1
                        pair_cls[mk] += b_vec

        self.slot_op_affinity = self._eb_log_odds(slot_tot, slot_cls, slot_n_all, slot_cls_all, min_df=8, alpha_0=5.0)
        self.q_root_affinity = self._eb_log_odds(q_tot, q_cls, q_n_all, q_cls_all, min_df=8, alpha_0=5.0)
        self.pair_mid_affinity = self._eb_log_odds(pair_tot, pair_cls, pair_n_all, pair_cls_all, min_df=8, alpha_0=5.0)

    def score_slot_affinities(self, slots: Sequence[ExtractedSlot]) -> np.ndarray:
        """Возвращает `(K, 7)` float32 усреднённый Empirical-Bayes log-odds по каждому слоту."""
        k = len(slots)
        out = np.zeros((k, 7), dtype=np.float32)
        if not self.slot_op_affinity or k == 0:
            return out
        for i, sl in enumerate(slots):
            matched = [self.slot_op_affinity[key] for key in self._slot_affinity_keys(sl) if key in self.slot_op_affinity]
            if matched:
                out[i] = np.mean(matched, axis=0)
        return out

    def score_question_root_affinity(self, q_text: str) -> np.ndarray:
        """Возвращает `(4,)` float32 Empirical-Bayes log-odds финальной операции вопроса."""
        if not self.q_root_affinity:
            return np.zeros((4,), dtype=np.float32)
        matched = [self.q_root_affinity[key] for key in self._question_affinity_keys(q_text) if key in self.q_root_affinity]
        if not matched:
            return np.zeros((4,), dtype=np.float32)
        return np.mean(matched, axis=0).astype(np.float32)

    def score_pair_mid_affinities(
        self, q_text: str, slots: Sequence[ExtractedSlot]
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """Возвращает словарь `(i, j) -> (6,) float32` Empirical-Bayes log-odds связи между парой слотов."""
        out: Dict[Tuple[int, int], np.ndarray] = {}
        if not self.pair_mid_affinity or len(slots) < 2:
            return out
        tokens = scan_question_tokens(q_text)
        k = len(slots)
        for i in range(k):
            for j in range(i + 1, k):
                mkeys = self._between_pair_keys(tokens, slots[i], slots[j])
                matched = [self.pair_mid_affinity[mk] for mk in mkeys if mk in self.pair_mid_affinity]
                if matched:
                    out[(i, j)] = np.mean(matched, axis=0).astype(np.float32)
        return out

    def extract_slots(self, q_text: str, include_aux_consts: bool = True) -> Tuple[List[ExtractedSlot], List[ScannedToken], int]:
        """Строго слепое извлечение слотов из вопроса `q_text`."""
        tokens = scan_question_tokens(q_text)
        n_toks = len(tokens)
        q_sent_id = tokens[-1].sent_id if tokens else 0
        for tk in reversed(tokens):
            if tk.kind == "PUNCT" and tk.text == "?":
                q_sent_id = tk.sent_id
                break

        def _ctx_windows(ti: int) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
            left: List[str] = []
            for j in range(ti - 1, -1, -1):
                if tokens[j].kind == "WORD":
                    left.append(tokens[j].lower)
                    if len(left) >= 4:
                        break
            left.reverse()
            right: List[str] = []
            for j in range(ti + 1, n_toks):
                if tokens[j].kind == "WORD":
                    right.append(tokens[j].lower)
                    if len(right) >= 5:
                        break
            return tuple(left), tuple(right)

        slots: List[ExtractedSlot] = []
        used_tok_indices: Set[int] = set()
        q_words_for_const: List[Tuple[str, int]] = []

        for ti, tk in enumerate(tokens):
            if ti in used_tok_indices:
                continue
            if tk.kind == "NUM" and tk.num_val is not None:
                v = float(tk.num_val)
                is_pct = tk.is_pct
                if not is_pct and ti + 1 < n_toks and tokens[ti + 1].kind == "WORD":
                    if tokens[ti + 1].lower.startswith("percent"):
                        v /= 100.0
                        is_pct = True
                lw, rw = _ctx_windows(ti)
                slots.append(
                    ExtractedSlot(
                        idx=len(slots),
                        val=v,
                        val_str=_fmt_decimal_str(v),
                        kind="EXPLICIT",
                        b0=tk.b0,
                        b1=tk.b1,
                        sent_id=tk.sent_id,
                        clause_id=tk.clause_id,
                        is_in_question=(tk.sent_id == q_sent_id),
                        is_pct=is_pct,
                        has_dollar=tk.has_dollar,
                        left_words=lw,
                        right_words=rw,
                        trigger_token=tk.text,
                    )
                )
                continue

            if tk.kind == "WORD" and self.fitted:
                lw_tok = tk.lower
                prev_bi = (tokens[ti - 1].lower + "_" + lw_tok) if (ti > 0 and tokens[ti - 1].kind == "WORD") else ""
                next_bi = (lw_tok + "_" + tokens[ti + 1].lower) if (ti + 1 < n_toks and tokens[ti + 1].kind == "WORD") else ""

                suppressed = (prev_bi in self.bigram_suppress) or (next_bi in self.bigram_suppress)
                if not suppressed:
                    induced_val: Optional[float] = None
                    b0_s, b1_s = tk.b0, tk.b1
                    trig = lw_tok
                    tok_start = ti
                    if prev_bi and prev_bi in self.word_num_map:
                        # Пропускаем повтор, если предыдущее слово не было поглощено или уже было явным числом
                        if (ti - 1) not in used_tok_indices and tokens[ti - 1].kind == "WORD":
                            induced_val = self.word_num_map[prev_bi]
                            b0_s = tokens[ti - 1].b0
                            tok_start = ti - 1
                            trig = prev_bi
                            used_tok_indices.add(ti - 1)
                    elif lw_tok in self.word_num_map:
                        induced_val = self.word_num_map[lw_tok]

                    if induced_val is not None:
                        lw_c, rw_c = _ctx_windows(ti)
                        has_dollar_ind = tok_start > 0 and tokens[tok_start - 1].text == "$"
                        slots.append(
                            ExtractedSlot(
                                idx=len(slots),
                                val=float(induced_val),
                                val_str=_fmt_decimal_str(float(induced_val)),
                                kind="INDUCED_WORD",
                                b0=b0_s,
                                b1=b1_s,
                                sent_id=tk.sent_id,
                                clause_id=tk.clause_id,
                                is_in_question=(tk.sent_id == q_sent_id),
                                is_pct=False,
                                has_dollar=has_dollar_ind,
                                left_words=lw_c,
                                right_words=rw_c,
                                trigger_token=trig,
                            )
                        )
                        used_tok_indices.add(ti)
                        continue

                if prev_bi and prev_bi in self.unit_const_map:
                    q_words_for_const.append((prev_bi, ti))
                if lw_tok in self.unit_const_map:
                    q_words_for_const.append((lw_tok, ti))

        if include_aux_consts and self.fitted:
            added_aux = 0
            # 1. Константы, вызванные триггерными словами (dozen/week/year/hour/cents)
            if self.max_aux_consts > 0:
                for key_w, tok_pos in q_words_for_const:
                    if added_aux >= self.max_aux_consts:
                        break
                    cv = float(self.unit_const_map[key_w])
                    if any(abs(cv - s.val) < 1e-4 for s in slots):
                        continue
                    tk_ref = tokens[tok_pos]
                    lw_c, rw_c = _ctx_windows(tok_pos)
                    slots.append(
                        ExtractedSlot(
                            idx=len(slots),
                            val=cv,
                            val_str=_fmt_decimal_str(cv),
                            kind="CONST",
                            b0=tk_ref.b0,
                            b1=tk_ref.b1,
                            sent_id=tk_ref.sent_id,
                            clause_id=tk_ref.clause_id,
                            is_in_question=(tk_ref.sent_id == q_sent_id),
                            is_pct=False,
                            has_dollar=False,
                            left_words=lw_c,
                            right_words=rw_c,
                            trigger_token=key_w,
                        )
                    )
                    added_aux += 1

            # 2. Процентные опоры (1.0 и 100.0) добавляются ТОЛЬКО при наличии процента в задаче
            if any(s.is_pct for s in slots) or ("%" in q_text):
                for pv in (1.0, 100.0):
                    if any(abs(pv - s.val) < 1e-4 for s in slots):
                        continue
                    slots.append(
                        ExtractedSlot(
                            idx=len(slots),
                            val=pv,
                            val_str=_fmt_decimal_str(pv),
                            kind="CONST",
                            b0=0,
                            b1=0,
                            sent_id=q_sent_id,
                            clause_id=0,
                            is_in_question=True,
                            is_pct=False,
                            has_dollar=False,
                            left_words=(),
                            right_words=(),
                            trigger_token="__pct__",
                        )
                    )

        clean_slots = [replace(s, idx=i) for i, s in enumerate(slots)]
        return clean_slots, tokens, q_sent_id

    def extract_question_sentence(self, q_text: str) -> str:
        """Выделяет финальное вопросное предложение без `re` для question-primed подачи."""
        tokens = scan_question_tokens(q_text)
        if not tokens:
            return q_text
        q_sent = tokens[-1].sent_id
        for tk in reversed(tokens):
            if tk.kind == "PUNCT" and tk.text == "?":
                q_sent = tk.sent_id
                break
        sent_toks = [tk for tk in tokens if tk.sent_id == q_sent]
        if not sent_toks:
            return q_text
        raw_bytes = q_text.encode("utf-8")
        return raw_bytes[sent_toks[0].b0 : sent_toks[-1].b1].decode("utf-8", errors="ignore").strip()

    def build_primed_prompt_and_align(
        self,
        q_text: str,
        slots: Sequence[ExtractedSlot],
        tok,
        post_window: int = 5,
    ) -> Tuple[List[int], List[Tuple[int, int]], List[Tuple[int, int]], int]:
        """Строит question-primed строку `Focus: <вопрос>\\nProblem: <q_text>\\nPlan:`
        и привязывает каждый слот `(b0, b1)` точно к его BPE-токенам внутри `<Problem>`.
        """
        q_focus = self.extract_question_sentence(q_text)
        prefix = f"Focus: {q_focus}\nProblem: "
        prefix_bytes = len(prefix.encode("utf-8"))
        full_text = prefix + q_text + "\nPlan:"

        token_ids = tok.encode(full_text, add_bos=True, add_eos=True)
        n_toks = len(token_ids)
        tok_bytes: List[Tuple[int, int]] = []
        b_cur = 0
        for tid in token_ids:
            raw_b = tok.id_to_bytes.get(int(tid), b"")
            tok_bytes.append((b_cur, b_cur + len(raw_b)))
            b_cur += len(raw_b)

        q_tok_idx = max(1, n_toks - 2)
        num_spans: List[Tuple[int, int]] = []
        post_spans: List[Tuple[int, int]] = []

        for sl in slots:
            sb0 = prefix_bytes + sl.b0
            sb1 = prefix_bytes + sl.b1
            if sb1 <= sb0:
                num_spans.append((q_tok_idx, q_tok_idx + 1))
                post_spans.append((q_tok_idx, q_tok_idx + 1))
                continue
            ov = [ti for ti, (tb0, tb1) in enumerate(tok_bytes) if tb1 > sb0 and tb0 < sb1 and 0 < ti < n_toks - 1]
            if not ov:
                t_s, t_e = q_tok_idx, q_tok_idx + 1
            else:
                t_s, t_e = ov[0], ov[-1] + 1
            p_s = min(t_e, q_tok_idx)
            p_e = min(t_e + post_window, n_toks - 1)
            if p_e <= p_s:
                p_s, p_e = t_s, t_e
            num_spans.append((t_s, t_e))
            post_spans.append((p_s, p_e))

        return token_ids, num_spans, post_spans, q_tok_idx

    # ------------------------------------------------------------------
    # Компиляция золотой пошаговой трассы ТОЛЬКО на train_split
    # ------------------------------------------------------------------
    def compile_train_dag(
        self,
        slots: Sequence[ExtractedSlot],
        answer_text: str,
    ) -> Tuple[List[Tuple[int, int, int]], List[int], bool]:
        """Компилирует `<<lhs=rhs>>` в последовательность элементарных бинарных шагов
        `[(op_id, a_idx, b_idx), ...]` над `slots` + промежуточными узлами `s_0..s_{T-1}`.

        Возвращает `(steps, slot_roles, compiled_ok)`, где:
          - `steps`: бинарные шаги (не более 12);
          - `slot_roles[i]`: 0 = UNUSED (дистрактор), 1 = USED_ONCE, 2 = REUSED (использован >= 2 раз);
          - `compiled_ok`: True, если последний результат совпал с GT.
        """
        cur_vals = [float(sl.val) for sl in slots]
        pct_map: Dict[int, float] = {}
        for idx_s, sl in enumerate(slots):
            if sl.is_pct:
                pct_map[idx_s] = float(sl.val) * 100.0

        tags = extract_answer_tags(answer_text)
        gt = extract_gt_number(answer_text)
        prose_nums = _collect_prose_intermediates(answer_text)

        if not tags:
            direct = _match_value_index(gt, cur_vals)
            roles = [0] * len(slots)
            if 0 <= direct < len(roles):
                roles[direct] = 1
            return [], roles, direct >= 0

        steps: List[Tuple[int, int, int]] = []

        def _try_bridge_missing(target_v: float) -> int:
            """Восстанавливает один неаннотированный в `<<...>>` промежуточный бинарный шаг,
            если его результат уже явно присутствует в тексте ответа."""
            if not any(abs(target_v - pv) <= max(1e-3, 1e-4 * abs(pv)) for pv in prose_nums):
                return -1
            m = len(cur_vals)
            for a_i in range(m):
                va = cur_vals[a_i]
                for b_i in range(m):
                    if a_i == b_i:
                        continue
                    vb = cur_vals[b_i]
                    for op_id, cand_v in (
                        (0, va + vb),
                        (1, va - vb),
                        (2, va * vb),
                        (3, va / vb if abs(vb) > 1e-12 else 1e18),
                    ):
                        if abs(cand_v - target_v) <= max(1e-3, 1e-4 * abs(target_v)):
                            new_idx = len(cur_vals)
                            cur_vals.append(cand_v)
                            sa, sb = (b_i, a_i) if (op_id in (0, 2) and a_i > b_i) else (a_i, b_i)
                            steps.append((op_id, sa, sb))
                            return new_idx
            return -1

        def _match_operand(v: float) -> int:
            idx = _match_value_index(v, cur_vals)
            if idx >= 0:
                return idx
            # Если в вопросе 20%, а автор написал `... * 20 / 100`
            for s_idx, raw_pct in pct_map.items():
                if abs(v - raw_pct) <= max(1e-3, 1e-4 * abs(raw_pct)):
                    return s_idx
            return _try_bridge_missing(v)

        last_out_idx = -1
        for lhs, rhs in tags:
            expr_toks = tokenize_arithmetic_expr(lhs)
            if not expr_toks:
                return [], [0] * len(slots), False
            rpn = _shunting_yard_rpn(expr_toks)
            if not rpn:
                return [], [0] * len(slots), False

            # Сворачиваем шаблоны `A * pct_raw / 100` -> `A * x_pct` до компиляции
            cleaned_rpn: List[float | str] = []
            for item in rpn:
                cleaned_rpn.append(item)

            stack: List[Tuple[int, float]] = []
            for idx_item, item in enumerate(cleaned_rpn):
                if isinstance(item, float):
                    # Проверяем, не является ли это `* 100` после процента
                    m_i = _match_operand(item)
                    if m_i < 0:
                        return [], [0] * len(slots), False
                    stack.append((m_i, cur_vals[m_i]))
                else:
                    if len(stack) < 2:
                        return [], [0] * len(slots), False
                    ib, vb = stack.pop()
                    ia, va = stack.pop()
                    op_id = OP_TO_ID[item]
                    # Пропускаем лишнее `/ 100`, если предыдущий шаг уже домножил на сжатый `x_pct = p/100`
                    if op_id == 3 and abs(vb - 100.0) < 1e-4 and ia >= len(slots):
                        last_op, last_a, last_b = steps[ia - len(slots)]
                        if last_op == 2 and (last_b in pct_map or last_a in pct_map):
                            stack.append((ia, va))
                            continue
                    if op_id == 0:
                        rv = va + vb
                    elif op_id == 1:
                        rv = va - vb
                    elif op_id == 2:
                        rv = va * vb
                    else:
                        if abs(vb) < 1e-12:
                            return [], [0] * len(slots), False
                        rv = va / vb
                    new_idx = len(cur_vals)
                    cur_vals.append(rv)
                    sa, sb = (ib, ia) if (op_id in (0, 2) and ia > ib) else (ia, ib)
                    steps.append((op_id, sa, sb))
                    stack.append((new_idx, rv))
                    last_out_idx = new_idx
            if len(stack) != 1:
                return [], [0] * len(slots), False
            last_out_idx = stack[0][0]
            rv_rhs = _parse_float_safe(rhs)
            if rv_rhs is not None and last_out_idx >= 0:
                if abs(cur_vals[last_out_idx] - rv_rhs) > max(0.05, 1e-2 * abs(rv_rhs)):
                    return [], [0] * len(slots), False

        if not steps or last_out_idx < 0:
            return [], [0] * len(slots), False

        if abs(cur_vals[last_out_idx] - gt) > max(1e-2, 1e-3 * abs(gt)):
            return [], [0] * len(slots), False

        # Сворачивание тождеств (+0, -0, *1, /1) и удаление мёртвых промежуточных узлов (DCE от корня)
        n_init = len(slots)
        vals_fold = [float(sl.val) for sl in slots]
        remap: Dict[int, int] = {i: i for i in range(n_init)}
        folded_steps: List[Tuple[int, int, int]] = []
        for s_idx, (op_id, a_i, b_i) in enumerate(steps):
            old_out = n_init + s_idx
            ra = remap[a_i]
            rb = remap[b_i]
            va = vals_fold[ra]
            vb = vals_fold[rb]
            if (op_id in (0, 1) and abs(vb) < 1e-9) or (op_id in (2, 3) and abs(vb - 1.0) < 1e-9):
                remap[old_out] = ra
                continue
            if (op_id == 0 and abs(va) < 1e-9) or (op_id == 2 and abs(va - 1.0) < 1e-9):
                remap[old_out] = rb
                continue
            if op_id in (1, 3) and ra == rb:
                return [], [0] * n_init, False
            if op_id == 0:
                rv = va + vb
            elif op_id == 1:
                rv = va - vb
            elif op_id == 2:
                rv = va * vb
            else:
                if abs(vb) < 1e-12:
                    return [], [0] * n_init, False
                rv = va / vb
            sa, sb = (rb, ra) if (op_id in (0, 2) and ra > rb) else (ra, rb)
            remap[old_out] = n_init + len(folded_steps)
            folded_steps.append((op_id, sa, sb))
            vals_fold.append(rv)

        mapped_root = remap[last_out_idx]
        if mapped_root < n_init:
            if abs(vals_fold[mapped_root] - gt) <= max(1e-2, 1e-3 * abs(gt)):
                roles = [0] * n_init
                roles[mapped_root] = 1
                return [], roles, True
            return [], [0] * n_init, False

        needed: Set[int] = {mapped_root}
        for s_idx in range(len(folded_steps) - 1, -1, -1):
            if (n_init + s_idx) in needed:
                _, a_f, b_f = folded_steps[s_idx]
                needed.add(a_f)
                needed.add(b_f)

        final_remap: Dict[int, int] = {i: i for i in range(n_init)}
        clean_steps: List[Tuple[int, int, int]] = []
        for s_idx, (op_id, a_f, b_f) in enumerate(folded_steps):
            if (n_init + s_idx) not in needed:
                continue
            ra = final_remap[a_f]
            rb = final_remap[b_f]
            sa, sb = (rb, ra) if (op_id in (0, 2) and ra > rb) else (ra, rb)
            final_remap[n_init + s_idx] = n_init + len(clean_steps)
            clean_steps.append((op_id, sa, sb))

        if not clean_steps or len(clean_steps) > 12:
            return [], [0] * n_init, False
        if abs(vals_fold[mapped_root] - gt) > max(1e-2, 1e-3 * abs(gt)):
            return [], [0] * n_init, False

        # Подсчёт ролей слотов строго по очищенному связному DAG: 0=UNUSED, 1=USED_ONCE, 2=REUSED
        slot_counts = [0] * n_init
        for _, a_i, b_i in clean_steps:
            if a_i < n_init:
                slot_counts[a_i] += 1
            if b_i < n_init:
                slot_counts[b_i] += 1
        roles = [2 if c >= 2 else (1 if c == 1 else 0) for c in slot_counts]
        return clean_steps, roles, True
