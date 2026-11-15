"""gsm8k_cram_solver.py — Сборка арифметических формул через CRAM+z без BPTT
перед 12-разрядным сферическим калькулятором (SphericalBase10FixedPointALU) на реальном GSM8K.

Честный протокол (Zero-Cheat Parity с v33_real_gsm8k_benchmark.py):
  1. Обучение ИСКЛЮЧИТЕЛЬНО на `benchmarks/gsm8k/train.jsonl` (N = 7,473).
  2. Слепой тест на `benchmarks/gsm8k/test.jsonl` (N = 1,319): в `solve_blind(question_text)`
     передаётся ТОЛЬКО строка вопроса (поле `answer` и теги `<<...>>` полностью отрезаны).
  3. Тот же самый парсер чисел/клауз (`extract_structured_quantities_and_clauses`) и пошаговое
     исполнение выбранной формулы на 12-разрядном сферическом кристалле базы 10 `(S^14)^13`,
     чтобы весь прирост над 12.89% (`v33`) шёл строго от замены одношагового угадывания
     на энергетическую сборку формулы `CRAM+z` без BPTT.
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
import time
import zlib
from collections import Counter
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# ==============================================================================
# 0. РУЧКИ НАСТРОЙКИ АРХИТЕКТУРЫ (TUNING_KNOBS — зафиксированы на dev-срезе train.jsonl)
# ==============================================================================
TUNING_KNOBS: Dict[str, float | int] = {
    "d_ctx": 4096,                 # Размерность сферической коры прецедентов SimVQ
    "d_unary": 1536,               # Размерность унарного контекста числового слота
    "d_pair": 1792,                # Размерность парной связи между слотами (i, j)
    "d_root": 512,                 # Размерность вектора целевого вопроса задачи
    "vocab_min_df": 15,            # Минимальная частота слова в train.jsonl
    "top_templates_per_arity": 288,# Число канонических AST-шаблонов на арность k
    "simvq_knn_k": 24,             # Число ближайших соседей в сферической памяти SimVQ
    "simvq_knn_temp": 16.0,        # Обратная температура косинусного резонанса SimVQ
    "cram_z_gamma": 0.85,          # Коэффициент диффузии латентного поля единиц z*
    "woodbury_ridge_lam": 18.0,    # Регуляризация замкнутого старта Вудбери (0 BPTT)
    "cram_epochs": 10,             # Эпохи контрастивной минимизации свободной энергии CRAM
    "cram_batch_size": 128,        # Размер батча контрастивного обучения (Eq. 5)
    "cram_lr": 0.022,              # Начальный шаг косинусного расписания AdamW
    "boltzmann_beta": 1.0,         # Обратная температура маргинализации ответов на тесте
    "uf_confidence_threshold": 0.25,  # Порог уверенности для честного отказа UF=1
    "seed": 2026,                  # Фиксированное зерно воспроизводимости
}


# ==============================================================================
# 1. СТАТИСТИКА (Доверительный интервал Уилсона 95%)
# ==============================================================================
def wilson_95_ci(k_ok: int, n_tot: int) -> Tuple[float, float]:
    """Возвращает 95% доверительный интервал Уилсона в процентах."""
    if n_tot <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = k_ok / float(n_tot)
    denom = 1.0 + (z * z) / n_tot
    center = (p + (z * z) / (2.0 * n_tot)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p)) / n_tot + (z * z) / (4.0 * n_tot * n_tot))
    return (max(0.0, (center - half) * 100.0), min(100.0, (center + half) * 100.0))


# ==============================================================================
# 2. СФЕРИЧЕСКИЙ 12-РАЗРЯДНЫЙ КАЛЬКУЛЯТОР БАЗЫ 10 НА (S^14)^12 x S^14_sign
#    Чистый NumPy-порт `SphericalBase10FixedPointALU_v33` с проверкой дрейфа нормы
# ==============================================================================
class SphericalBase10FixedPointALU:
    """Точное 12-разрядное АЛУ с фиксированной точкой (`SCALE = 10,000`, диапазон до `99,999,999.9999`)
    на декартовом произведении 13 единичных сфер `(S^14)^12 x S^14_sign` в `float32`.
    """

    SCALE: int = 10_000

    def __init__(self, d_sub: int = 15, num_digits: int = 12, seed: int = 3301):
        self.d_sub = d_sub
        self.num_digits = num_digits
        rng = np.random.default_rng(seed)

        # Ортонормированный кристалл цифр 0..9 на единичной сфере S^{d_sub - 1} в float32
        raw_q, _ = np.linalg.qr(rng.standard_normal((d_sub, d_sub)))
        d_cb = raw_q[:10].astype(np.float32)
        d_cb /= np.linalg.norm(d_cb, axis=-1, keepdims=True)
        self.digit_codebook = d_cb  # (10, 15)

        # Антиподальная пара знака (+/-) на S^{d_sub - 1}
        s_vec = raw_q[10:11].astype(np.float32)
        s_vec /= np.linalg.norm(s_vec, axis=-1, keepdims=True)
        self.sign_codebook = np.concatenate([s_vec, -s_vec], axis=0)  # (2, 15)

        # Верифицированные таблицы 200 состояний (10 x 10 x 2) поразрядного сложения и вычитания
        a_g, b_g, c_g = np.meshgrid(np.arange(10), np.arange(10), np.arange(2), indexing="ij")
        self.lut_add_sum = ((a_g + b_g + c_g) % 10).astype(np.int64)
        self.lut_add_cry = ((a_g + b_g + c_g) >= 10).astype(np.int64)
        self.lut_sub_dif = ((a_g - b_g - c_g) % 10).astype(np.int64)
        self.lut_sub_brw = ((a_g - b_g - c_g) < 0).astype(np.int64)

    def verify_crystal_invariants(self) -> Dict[str, float]:
        """Проверяет 200/200 состояний поразрядного АЛУ и отсутствие дрейфа сферической нормы."""
        ok_add = 0
        ok_sub = 0
        max_drift = 0.0
        for a in range(10):
            for b in range(10):
                for c in range(2):
                    ra = self.encode_float_to_spheres(float(a))
                    rb = self.encode_float_to_spheres(float(b))
                    max_drift = max(max_drift, self.compute_norm_drift(ra), self.compute_norm_drift(rb))
                    if self.lut_add_sum[a, b, c] == (a + b + c) % 10 and self.lut_add_cry[a, b, c] == int(a + b + c >= 10):
                        ok_add += 1
                    if self.lut_sub_dif[a, b, c] == (a - b - c) % 10 and self.lut_sub_brw[a, b, c] == int(a - b - c < 0):
                        ok_sub += 1
        return {
            "add_200_acc_pct": 100.0 * ok_add / 200.0,
            "sub_200_acc_pct": 100.0 * ok_sub / 200.0,
            "codebook_norm_drift": max_drift,
        }

    def encode_float_to_spheres(self, val: float) -> np.ndarray:
        scaled = min(int(round(abs(val) * self.SCALE)), 10 ** self.num_digits - 1)
        sign_idx = 1 if val < -1e-9 else 0
        digits = []
        tmp = scaled
        for _ in range(self.num_digits):
            digits.append(tmp % 10)
            tmp //= 10
        d_vecs = self.digit_codebook[digits]  # (12, 15)
        s_vec = self.sign_codebook[sign_idx : sign_idx + 1]  # (1, 15)
        return np.concatenate([d_vecs, s_vec], axis=0)  # (13, 15)

    def decode_spheres_to_digits_and_sign(self, reg_spheres: np.ndarray) -> Tuple[List[int], int]:
        d_vecs = reg_spheres[: self.num_digits]
        s_vec = reg_spheres[self.num_digits : self.num_digits + 1]
        # Проекция Вороного на кодовую книгу цифр и знака
        d_indices = np.argmax(d_vecs @ self.digit_codebook.T, axis=-1).tolist()
        sign_idx = int(np.argmax(s_vec @ self.sign_codebook.T, axis=-1)[0])
        return d_indices, sign_idx

    def decode_spheres_to_float(self, reg_spheres: np.ndarray) -> float:
        digits, sign_idx = self.decode_spheres_to_digits_and_sign(reg_spheres)
        scaled = 0
        for g in range(self.num_digits - 1, -1, -1):
            scaled = scaled * 10 + digits[g]
        val = float(scaled) / float(self.SCALE)
        return -val if (sign_idx == 1 and scaled > 0) else val

    def _pack_digits_and_sign_to_spheres(self, digits: List[int], sign_idx: int) -> np.ndarray:
        if all(d == 0 for d in digits):
            sign_idx = 0
        d_vecs = self.digit_codebook[digits[: self.num_digits]]
        s_vec = self.sign_codebook[sign_idx : sign_idx + 1]
        return np.concatenate([d_vecs, s_vec], axis=0)

    def _add_unsigned_digits(self, d_a: List[int], d_b: List[int]) -> List[int]:
        c = 0
        out = []
        for g in range(self.num_digits):
            ag, bg = d_a[g], d_b[g]
            sg = int(self.lut_add_sum[ag, bg, c])
            c = int(self.lut_add_cry[ag, bg, c])
            out.append(sg)
        return out

    def _sub_unsigned_digits(self, d_a: List[int], d_b: List[int]) -> List[int]:
        b = 0
        out = []
        for g in range(self.num_digits):
            ag, bg = d_a[g], d_b[g]
            dg = int(self.lut_sub_dif[ag, bg, b])
            b = int(self.lut_sub_brw[ag, bg, b])
            out.append(dg)
        return out

    @staticmethod
    def _cmp_unsigned_digits(d_a: List[int], d_b: List[int]) -> int:
        for g in range(len(d_a) - 1, -1, -1):
            if d_a[g] > d_b[g]:
                return 1
            if d_a[g] < d_b[g]:
                return -1
        return 0

    def add_spherical(self, reg_a: np.ndarray, reg_b: np.ndarray) -> np.ndarray:
        da, sa = self.decode_spheres_to_digits_and_sign(reg_a)
        db, sb = self.decode_spheres_to_digits_and_sign(reg_b)
        if sa == sb:
            return self._pack_digits_and_sign_to_spheres(self._add_unsigned_digits(da, db), sa)
        if self._cmp_unsigned_digits(da, db) >= 0:
            return self._pack_digits_and_sign_to_spheres(self._sub_unsigned_digits(da, db), sa)
        return self._pack_digits_and_sign_to_spheres(self._sub_unsigned_digits(db, da), sb)

    def sub_spherical(self, reg_a: np.ndarray, reg_b: np.ndarray) -> np.ndarray:
        db, sb = self.decode_spheres_to_digits_and_sign(reg_b)
        return self.add_spherical(reg_a, self._pack_digits_and_sign_to_spheres(db, 1 - sb))

    def mul_spherical(self, reg_a: np.ndarray, reg_b: np.ndarray) -> np.ndarray:
        return self.encode_float_to_spheres(self.decode_spheres_to_float(reg_a) * self.decode_spheres_to_float(reg_b))

    def div_spherical(self, reg_a: np.ndarray, reg_b: np.ndarray) -> np.ndarray:
        va = self.decode_spheres_to_float(reg_a)
        vb = self.decode_spheres_to_float(reg_b)
        if abs(vb) < 1e-12:
            return self.encode_float_to_spheres(0.0)
        return self.encode_float_to_spheres(va / vb)

    @staticmethod
    def compute_norm_drift(reg_spheres: np.ndarray) -> float:
        norms = np.linalg.norm(reg_spheres, axis=-1)
        return float(np.max(np.abs(norms - 1.0)))

    def eval_expr_spherical(self, expr_str: str) -> Tuple[np.ndarray, float]:
        """Исполняет арифметическое выражение через алгоритм Дейкстры (Shunting-Yard) на сферических регистрах."""
        s = expr_str.replace(",", "").strip()
        s = re.sub(r"(\d)\(", r"\1*(", s)
        s = re.sub(r"\)(\d)", r")*\1", s)
        s = re.sub(r"\)\(", r")*(", s)
        s = re.sub(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*\*\s*(\d+(?:\.\d+)?)", r"\1*\3/\2", s)

        tokens = re.findall(r"\d*\.\d+|\d+|[\+\-\*\/\(\)]", s)
        if not tokens:
            reg0 = self.encode_float_to_spheres(0.0)
            return reg0, self.compute_norm_drift(reg0)

        prec = {"+": 1, "-": 1, "*": 2, "/": 2}
        output_queue: List[object] = []
        op_stack: List[str] = []
        prev_tok = None
        for tok in tokens:
            if re.match(r"^\d*\.\d+$|^\d+$", tok):
                output_queue.append(self.encode_float_to_spheres(float(tok)))
                prev_tok = "NUM"
            elif tok == "(":
                op_stack.append(tok)
                prev_tok = "("
            elif tok == ")":
                while op_stack and op_stack[-1] != "(":
                    output_queue.append(op_stack.pop())
                if op_stack and op_stack[-1] == "(":
                    op_stack.pop()
                prev_tok = ")"
            elif tok in prec:
                if prev_tok is None or prev_tok in ("(", "+", "-", "*", "/"):
                    output_queue.append(self.encode_float_to_spheres(0.0))
                while op_stack and op_stack[-1] in prec and prec[op_stack[-1]] >= prec[tok]:
                    output_queue.append(op_stack.pop())
                op_stack.append(tok)
                prev_tok = tok
        while op_stack:
            output_queue.append(op_stack.pop())

        reg_stack: List[np.ndarray] = []
        max_drift = 0.0
        for item in output_queue:
            if isinstance(item, np.ndarray):
                max_drift = max(max_drift, self.compute_norm_drift(item))
                reg_stack.append(item)
            else:
                if len(reg_stack) < 2:
                    continue
                b_reg = reg_stack.pop()
                a_reg = reg_stack.pop()
                if item == "+":
                    res_reg = self.add_spherical(a_reg, b_reg)
                elif item == "-":
                    res_reg = self.sub_spherical(a_reg, b_reg)
                elif item == "*":
                    res_reg = self.mul_spherical(a_reg, b_reg)
                else:
                    res_reg = self.div_spherical(a_reg, b_reg)
                max_drift = max(max_drift, self.compute_norm_drift(res_reg))
                reg_stack.append(res_reg)

        final_reg = reg_stack[-1] if reg_stack else self.encode_float_to_spheres(0.0)
        return final_reg, max_drift

    def execute_program(self, prog: Tuple[str, ...], x_vals: Sequence[float]) -> Tuple[float, List[str], float]:
        """Пошагово исполняет цепочку шагов `s0, s1, ...` на сферическом кристалле."""
        env_val: Dict[str, float] = {f"x{i}": float(x_vals[i]) for i in range(len(x_vals))}
        trace_strs: List[str] = []
        max_drift = 0.0
        last_val = float(x_vals[0]) if x_vals else 0.0
        for j, step_expr in enumerate(prog):
            concrete_expr = step_expr
            for k_var in sorted(env_val.keys(), key=len, reverse=True):
                concrete_expr = concrete_expr.replace(k_var, f"{env_val[k_var]:.4f}")
            reg_s, drift = self.eval_expr_spherical(concrete_expr)
            max_drift = max(max_drift, drift)
            last_val = self.decode_spheres_to_float(reg_s)
            env_val[f"s{j}"] = last_val
            trace_strs.append(f"s{j}: <<{concrete_expr}={last_val:.2f}>>")
        return last_val, trace_strs, max_drift


# ==============================================================================
# 3. ЛЕКСИКОНЫ И ПАРСЕР ЧИСЛОВЫХ СЛОТОВ / КЛАУЗ (1-в-1 совместимость с v33)
# ==============================================================================
WORD_NUMS_V33 = {
    "two": 2.0, "three": 3.0, "four": 4.0, "five": 5.0,
    "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0, "ten": 10.0,
    "eleven": 11.0, "twelve": 12.0, "thirteen": 13.0, "fourteen": 14.0,
    "fifteen": 15.0, "sixteen": 16.0, "seventeen": 17.0, "eighteen": 18.0,
    "nineteen": 19.0, "twenty": 20.0, "thirty": 30.0, "forty": 40.0,
    "fifty": 50.0, "sixty": 60.0, "seventy": 70.0, "eighty": 80.0, "ninety": 90.0,
    "dozen": 12.0, "pair": 2.0,
    "half": 2.0, "halves": 2.0, "twice": 2.0, "double": 2.0, "doubled": 2.0,
    "triple": 3.0, "tripled": 3.0, "thrice": 3.0, "third": 3.0,
    "quarter": 4.0, "fourth": 4.0,
}

DIV_WORD_SET_V33 = {"half", "halves", "third", "quarter", "fourth"}
MUL_WORD_SET_V33 = {"twice", "double", "doubled", "triple", "tripled", "thrice", "dozen", "pair"}

VERB_NORM_V33 = {
    "ate": "eat", "eats": "eat", "eating": "eat", "eaten": "eat",
    "spent": "spend", "spends": "spend", "spending": "spend",
    "bought": "buy", "buys": "buy", "buying": "buy", "purchased": "buy", "purchases": "buy",
    "sold": "sell", "sells": "sell", "selling": "sell",
    "gave": "give", "gives": "give", "giving": "give", "given": "give",
    "lost": "lose", "loses": "lose", "losing": "lose",
    "used": "use", "uses": "use", "using": "use",
    "made": "make", "makes": "make", "making": "make",
    "earned": "earn", "earns": "earn", "earning": "earn",
    "paid": "pay", "pays": "pay", "paying": "pay",
    "took": "take", "takes": "take", "taking": "take", "taken": "take",
    "baked": "bake", "bakes": "bake", "baking": "bake",
    "drank": "drink", "drinks": "drink", "drinking": "drink",
    "drove": "drive", "drives": "drive", "driving": "drive",
    "walked": "walk", "walks": "walk", "walking": "walk",
    "ran": "run", "runs": "run", "running": "run",
    "worked": "work", "works": "work", "working": "work",
    "saved": "save", "saves": "save", "saving": "save",
    "collected": "collect", "collects": "collect",
    "picked": "pick", "picks": "pick",
    "planted": "plant", "plants": "plant",
    "shared": "share", "shares": "share", "sharing": "share",
    "divided": "divide", "divides": "divide", "dividing": "divide",
    "reduced": "reduce", "reduces": "reduce", "decreased": "decrease",
    "increased": "increase", "increases": "increase",
    "added": "add", "adds": "add", "adding": "add",
    "needed": "need", "needs": "need", "needing": "need",
    "wanted": "want", "wants": "want",
    "remaining": "remain", "remains": "remain", "remained": "remain",
    "costs": "cost", "costing": "cost",
    "hours": "hour", "days": "day", "weeks": "week", "months": "month", "years": "year",
    "minutes": "minute", "seconds": "second", "dollars": "$", "dollar": "$",
}

SUB_WORDS_V33 = {
    "left", "remain", "remaining", "remainder", "rest", "less", "fewer", "minus",
    "spent", "spend", "spends", "gave", "give", "gives", "given", "eats", "eat", "ate",
    "lost", "lose", "loses", "used", "use", "uses", "sold", "sells", "took", "take",
    "dropped", "broken", "died", "drank", "drink", "paid", "pay", "change", "difference",
    "profit", "discount", "off", "reduced", "reduce", "reduction", "decrease", "decreased",
    "already", "away", "out", "except", "without", "bakes", "bake", "needs", "need", "still",
    "donated", "donates", "donating", "shared", "wasted", "melted", "returned",
}

MUL_WORDS_V33 = {
    "each", "every", "per", "times", "twice", "double", "triple", "thrice",
    "rate", "speed", "mph", "hourly", "weekly", "monthly", "annual", "annually",
    "area", "product", "cost", "costs", "earns", "earn", "makes", "make", "worth", "apiece",
}

DIV_WORDS_V33 = {
    "half", "quarter", "third", "fourth", "split", "shared", "share", "divide", "divided",
    "equally", "average", "ratio", "into", "packs", "boxes", "groups", "dozens", "pair", "pairs",
}

ADD_WORDS_V33 = {
    "more", "plus", "added", "add", "adds", "additional", "another", "also", "together",
    "altogether", "total", "combined", "sum", "both", "all", "and", "increased", "increase",
    "extra", "bonus", "deposit", "bought", "buy", "received", "found", "earned", "grew", "joined",
}

MONEY_WORDS = {
    "$", "dollar", "dollars", "cent", "cents", "money", "cost", "costs",
    "price", "pay", "paid", "earn", "earned", "profit", "charge", "fee",
    "bill", "budget", "discount", "tax", "tip", "salary", "wage", "worth", "change",
}
TIME_WORDS = {"second", "minute", "hour", "day", "week", "month", "year", "daily", "weekly", "monthly"}
RATE_MARKERS = {"each", "every", "per", "apiece", "an", "a", "hourly", "daily", "weekly", "monthly"}
STOP_NOUNS = {
    "the", "a", "an", "of", "in", "to", "for", "on", "at", "by", "with", "from",
    "and", "or", "but", "if", "than", "as", "is", "are", "was", "were", "be",
    "he", "she", "they", "it", "his", "her", "their", "its", "that", "this",
    "how", "many", "much", "each", "every", "per", "more", "less", "fewer",
    "left", "remain", "total", "altogether", "together", "all", "old", "long",
    "far", "fast", "most", "other", "another", "rest", "same", "first", "second",
    "him", "them", "who", "what", "when", "where", "there", "had", "has", "have",
}

TEST_SIG_VEC = [math.sqrt(2.0 + i * 3.17) * 10.0 for i in range(25)]


def det_hash(s: str, mod: int) -> int:
    return zlib.crc32(s.encode("utf-8")) % mod


def norm_tok(w: str, vocab_keep: Optional[Set[str]] = None) -> str:
    wl = VERB_NORM_V33.get(w.lower(), w.lower())
    if vocab_keep is not None and wl not in vocab_keep:
        if wl.endswith("s") and len(wl) > 3:
            return "<NNS>"
        return "<ENT>"
    return wl


def singularize_noun(w: str) -> str:
    w = VERB_NORM_V33.get(w.lower(), w.lower())
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ves") and len(w) > 4:
        return w[:-3] + "f"
    if w.endswith(("ches", "shes", "xes", "boxes", "dishes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def extract_structured_quantities_and_clauses(
    q_text: str,
) -> Tuple[
    List[float],
    List[Tuple[List[str], List[str], int, int, bool, str, List[str]]],
    List[Tuple[str, int, bool]],
]:
    """Извлекает числа, локальные окна слов, клаузы и неявные множители времени/дюжин (1-в-1 v33)."""
    q_c = re.sub(r"(\d),(\d)", r"\1\2", q_text)
    q_c = (
        q_c.replace("half-hour", "30 minutes")
        .replace("half an hour", "0.5 hour")
        .replace("one-half", "half")
        .replace("one-third", "third")
        .replace("one-fourth", "fourth")
        .replace("a quarter", "quarter")
        .replace("a third", "third")
        .replace("a half", "half")
    )
    if not any(w in q_c.lower() for w in ("week", "month", "year", "days")):
        q_c = re.sub(r"\bevery\s+(morning|night|evening|afternoon)\b", r"in the \1", q_c, flags=re.IGNORECASE)
        if "per day" in q_c.lower() or "a day" in q_c.lower() or "daily" in q_c.lower():
            sents_tmp = re.split(r"(?<=[.?!])\s+", q_c)
            if len(sents_tmp) > 1:
                body_tmp = re.sub(r"\bevery\s+day\b", "daily", " ".join(sents_tmp[:-1]), flags=re.IGNORECASE)
                q_c = body_tmp + " " + sents_tmp[-1]

    sents = [s.strip() for s in re.split(r"(?<=[.?!])\s+", q_c) if s.strip()]
    clauses: List[Tuple[str, int, bool]] = []
    for s_idx, sent in enumerate(sents):
        is_q_sent = ("?" in sent) or (s_idx == len(sents) - 1)
        sub_clauses = re.split(
            r"(?:,\s*and\s+then\s+|,\s*and\s+|,\s*then\s+|\s+and\s+then\s+|;\s*|,\s*while\s+|,\s*but\s+"
            r"|\s+and\s+(?=(?:then\s+|also\s+|he\s+|she\s+|they\s+|it\s+)?(?:eat|eats|ate|bake|bakes|baked|buy|buys|bought|sell|sells|sold|give|gives|gave|spend|spends|spent|lose|loses|lost|use|uses|used|make|makes|made|take|takes|took|get|gets|got|find|finds|found|pay|pays|paid|save|saves|saved|drive|drives|drove|walk|walks|walked|run|runs|ran|work|works|worked|drink|drinks|drank|collect|collects|collected|pick|picks|picked|plant|plants|planted|drop|drops|dropped|donate|donates|donated)\b))",
            sent,
        )
        for sc in sub_clauses:
            if sc.strip():
                clauses.append((sc.strip(), s_idx, is_q_sent))

    nums: List[float] = []
    num_ctxs: List[Tuple[List[str], List[str], int, int, bool, str, List[str]]] = []
    for c_idx, (cl_text, s_idx, _) in enumerate(clauses):
        toks = re.findall(r"\d+(?:\.\d+)?%?|[A-Za-z]+|[.,;?!$]", cl_text)
        cl_alpha = [VERB_NORM_V33.get(w.lower(), w.lower()) for w in toks if w.isalpha()]
        for i, t in enumerate(toks):
            low = t.lower()
            val: Optional[float] = None
            is_pct = False
            w_kind = "DIGIT"
            if re.match(r"^\d+(?:\.\d+)?%?$", t):
                if t.endswith("%"):
                    val = float(t[:-1]) / 100.0
                    is_pct = True
                    w_kind = "PCT"
                else:
                    val = float(t)
                    if i + 1 < len(toks) and toks[i + 1].lower().startswith("percent"):
                        val = val / 100.0
                        is_pct = True
                        w_kind = "PCT"
            elif low in WORD_NUMS_V33:
                if low == "dozen" and ("per dozen" in q_c.lower() or q_c.lower().count("dozen") > 1):
                    continue
                if low == "pair" and ("pairs of" in cl_text.lower() or "a pair of" in cl_text.lower()):
                    if any(w in cl_text.lower() for w in ("shoes", "pants", "glasses", "socks", "gloves", "jeans", "scissors")):
                        continue
                if low in ("third", "fourth") and i > 0 and toks[i - 1].lower() in ("the", "her", "his", "their", "its", "every"):
                    if not (i + 1 < len(toks) and toks[i + 1].lower() == "of"):
                        continue
                val = WORD_NUMS_V33[low]
                w_kind = "DIV_WORD" if low in DIV_WORD_SET_V33 else ("MUL_WORD" if low in MUL_WORD_SET_V33 else "NUM_WORD")

            if val is not None:
                left_w = [VERB_NORM_V33.get(w.lower(), w.lower()) for w in toks[max(0, i - 9) : i] if w.isalpha() or w == "$"]
                right_w = [VERB_NORM_V33.get(w.lower(), w.lower()) for w in toks[i + 1 : min(len(toks), i + 7)] if w.isalpha() or w == "$"]
                nums.append(val)
                num_ctxs.append((left_w, right_w, c_idx, s_idx, is_pct, w_kind, cl_alpha))

    low_q = q_c.lower()
    last_c = max(0, len(clauses) - 1)
    last_s = max(0, len(sents) - 1)
    if ("week" in low_q and any(w in low_q for w in ("day", "daily", "morning", "night", "weekday"))) and 7.0 not in nums and 5.0 not in nums:
        if "weekday" in low_q:
            nums.append(5.0)
            num_ctxs.append((["in", "a"], ["week", "weekday"], last_c, last_s, False, "UNIT", ["week", "weekday"]))
        else:
            nums.append(7.0)
            num_ctxs.append((["in", "a"], ["week", "day"], last_c, last_s, False, "UNIT", ["week", "day"]))
    if ("hour" in low_q and "minute" in low_q) and 60.0 not in nums:
        nums.append(60.0)
        num_ctxs.append((["in", "an"], ["hour", "minute"], last_c, last_s, False, "UNIT", ["hour", "minute"]))
    if ("year" in low_q and "month" in low_q) and 12.0 not in nums:
        nums.append(12.0)
        num_ctxs.append((["in", "a"], ["year", "month"], last_c, last_s, False, "UNIT", ["year", "month"]))
    if ("year" in low_q and "week" in low_q) and 52.0 not in nums:
        nums.append(52.0)
        num_ctxs.append((["in", "a"], ["year", "week"], last_c, last_s, False, "UNIT", ["year", "week"]))
    if "dozen" in low_q and any(w in low_q for w in ("each", "egg", "cookie", "donut", "roll")) and 12.0 not in nums:
        nums.append(12.0)
        num_ctxs.append((["in", "a"], ["dozen"], last_c, last_s, False, "UNIT", ["dozen"]))
    if ("feet" in low_q or "foot" in low_q) and "inch" in low_q and 12.0 not in nums:
        nums.append(12.0)
        num_ctxs.append((["in", "a"], ["foot", "inch"], last_c, last_s, False, "UNIT", ["foot", "inch"]))

    return nums, num_ctxs, clauses


def prog_to_flat_expr(prog: Tuple[str, ...]) -> str:
    env_str: Dict[str, str] = {}
    last_s = "0.0"
    for j, step_expr in enumerate(prog):
        cur = step_expr
        for m in range(j - 1, -1, -1):
            cur = re.sub(rf"\bs{m}\b", f"({env_str[f's{m}']})", cur)
        env_str[f"s{j}"] = cur
        last_s = cur
    return last_s


def eval_prog_fast(prog: Tuple[str, ...], x_vals: Sequence[float]) -> float:
    env: Dict[str, float] = {f"x{i}": float(x_vals[i]) for i in range(len(x_vals))}
    last = 0.0
    for j, step_expr in enumerate(prog):
        last = float(eval(step_expr, {"__builtins__": {}}, env))
        env[f"s{j}"] = last
    return last


def compute_prog_sig(prog: Tuple[str, ...], k: int) -> Optional[float]:
    try:
        v = eval_prog_fast(prog, TEST_SIG_VEC[: max(k, 1)])
        if math.isfinite(v):
            return round(v, 4)
    except Exception:
        pass
    return None


def abstract_trace_from_train_answer(nums: List[float], ans: str) -> Optional[Tuple[str, ...]]:
    """Используется СТРОГО на train.jsonl для извлечения канонических DAG-шаблонов без чисел."""
    steps = re.findall(r"<<([^>]+)>>", ans)
    if not steps:
        return None
    gt_str = ans.split("####")[-1].strip().replace(",", "")
    try:
        gt_val = float(gt_str)
    except Exception:
        return None
    step_vals: List[float] = []
    prog: List[str] = []
    used_x: Set[int] = set()
    for st in steps:
        if "=" not in st:
            return None
        lhs, rhs = st.split("=", 1)
        lhs = re.sub(r"(?<![0-9])\.(\d+)", r"0.\1", lhs.replace(",", "").strip())
        try:
            r_val = float(rhs.replace(",", "").strip())
        except Exception:
            return None
        tokens = re.findall(r"\d+(?:\.\d+)?|[\+\-\*\/\(\)]", lhs)
        abs_tokens: List[str] = []
        for tok in tokens:
            if tok in "+-*/()":
                abs_tokens.append(tok)
            else:
                v = float(tok)
                matched = None
                for i, xv in enumerate(nums):
                    if i not in used_x and abs(xv - v) < 1e-4:
                        matched = f"x{i}"; used_x.add(i); break
                if matched is None:
                    for j in range(len(step_vals) - 1, -1, -1):
                        if abs(step_vals[j] - v) < 1e-4:
                            matched = f"s{j}"; break
                if matched is None:
                    for i, xv in enumerate(nums):
                        if abs(xv - v) < 1e-4:
                            matched = f"x{i}"; used_x.add(i); break
                        if abs(xv * 100.0 - v) < 1e-4:
                            matched = f"(x{i}*100)"; used_x.add(i); break
                        if xv in (2.0, 4.0) and abs(1.0 / xv - v) < 1e-4:
                            matched = f"(1/x{i})"; used_x.add(i); break
                if matched is None and v in (0.01, 0.5, 1.0, 2.0, 100.0):
                    matched = str(v) if v < 1.0 else str(int(v))
                if matched is None:
                    return None
                abs_tokens.append(matched)
        prog.append("".join(abs_tokens))
        step_vals.append(r_val)
    if not step_vals or abs(step_vals[-1] - gt_val) > 1e-2:
        return None
    return tuple(prog)


def synthesize_clause_sequential_programs(
    q_text: str,
    nums: List[float],
    num_ctxs: List[Tuple[List[str], List[str], int, int, bool, str, List[str]]],
    clauses: List[Tuple[str, int, bool]],
    role_probs: Optional[List[List[float]]] = None,
) -> List[Tuple[Tuple[str, ...], float]]:
    """Поклаузный генератор начальных гипотез-кандидатов."""
    k = len(nums)
    if k == 0:
        return []
    if k == 1:
        return [(("x0",), 1.0)]

    cands: List[Tuple[Tuple[str, ...], float]] = []
    seen_progs: Set[Tuple[str, ...]] = set()

    def add_cand(p: List[str], bonus: float):
        pt = tuple(p)
        if pt and pt not in seen_progs:
            seen_progs.add(pt)
            cands.append((pt, bonus))

    low_q = q_text.lower()
    q_clause_text = clauses[-1][0].lower() if clauses else low_q
    q_words = {VERB_NORM_V33.get(w, w) for w in re.findall(r"[a-z]+", q_clause_text)}
    asks_total = bool(q_words & {"total", "altogether", "together", "all", "combined", "both", "sum"})
    asks_diff_or_left = bool(q_words & {"left", "remain", "remaining", "more", "still", "need", "change", "difference", "farther", "longer", "older"})
    asks_profit = "profit" in q_words

    for accum_rel in (True, False):
        steps_seq: List[str] = []
        acc_var = "x0"
        last_term_var = "x0"
        i = 1
        while i < k:
            lw, rw, c_idx, _, is_pct, w_kind, cl_alpha = num_ctxs[i]
            ctx_set = set(lw + rw)
            cl_set = set(cl_alpha)
            is_rate_mul = bool((set(rw[:3]) & {"per", "each", "apiece"}) or ("$" in lw[-2:] and (set(rw[:3]) & {"per", "each", "apiece", "a", "an"})))

            if (
                i + 1 < k and num_ctxs[i + 1][2] == c_idx
                and w_kind == "DIGIT" and num_ctxs[i + 1][5] == "DIGIT"
                and (cl_set & {"each", "every", "per", "apiece", "an", "a", "for", "at", "hour", "day", "week", "boxes", "packs", "bags", "pairs"})
            ):
                steps_seq.append(f"x{i}*x{i+1}")
                term_var = f"s{len(steps_seq) - 1}"
                steps_seq.append(f"{acc_var}-{term_var}" if (cl_set & SUB_WORDS_V33) else f"{acc_var}+{term_var}")
                acc_var = f"s{len(steps_seq) - 1}"
                last_term_var = term_var
                i += 2
                continue

            if is_pct:
                if ctx_set & {"more", "increase", "up", "higher", "profit", "tip", "tax", "interest", "bonus", "extra"}:
                    steps_seq.append(f"{acc_var}*x{i}")
                    steps_seq.append(f"{acc_var}+s{len(steps_seq) - 1}")
                    acc_var = last_term_var = f"s{len(steps_seq) - 1}"
                elif ctx_set & {"less", "reduce", "discount", "off", "decrease", "lose", "fewer"}:
                    steps_seq.append(f"{acc_var}*x{i}")
                    steps_seq.append(f"{acc_var}-s{len(steps_seq) - 1}")
                    acc_var = last_term_var = f"s{len(steps_seq) - 1}"
                else:
                    steps_seq.append(f"{acc_var}*x{i}")
                    last_term_var = f"s{len(steps_seq) - 1}"
                    if accum_rel and (asks_total or asks_diff_or_left):
                        steps_seq.append(f"{acc_var}+{last_term_var}")
                        acc_var = f"s{len(steps_seq) - 1}"
                    else:
                        acc_var = last_term_var
            elif w_kind == "DIV_WORD":
                base_v = last_term_var if ("as" in ctx_set or "than" in ctx_set) else acc_var
                steps_seq.append(f"{base_v}/x{i}")
                term_var = f"s{len(steps_seq) - 1}"
                if accum_rel and (("as" in ctx_set) or asks_total) and not ("remain" in ctx_set or "left" in ctx_set or c_idx == len(clauses) - 1):
                    steps_seq.append(f"{acc_var}-{term_var}" if (ctx_set & {"less", "fewer"}) else f"{acc_var}+{term_var}")
                    acc_var = f"s{len(steps_seq) - 1}"
                else:
                    acc_var = term_var
                last_term_var = term_var
            elif w_kind == "MUL_WORD":
                base_v = last_term_var if ("as" in ctx_set or "parents" in ctx_set or "yesterday" in ctx_set or "first" in ctx_set) else acc_var
                steps_seq.append(f"{base_v}*x{i}")
                term_var = f"s{len(steps_seq) - 1}"
                if accum_rel and (("as" in ctx_set) or asks_total or asks_diff_or_left):
                    steps_seq.append(f"{acc_var}+{term_var}")
                    acc_var = f"s{len(steps_seq) - 1}"
                else:
                    acc_var = term_var
                last_term_var = term_var
            elif (ctx_set & {"times"}) and (ctx_set & {"as", "many", "much", "more", "than"}):
                steps_seq.append(f"{last_term_var}*x{i}")
                term_var = f"s{len(steps_seq) - 1}"
                if accum_rel and (asks_total or asks_diff_or_left):
                    steps_seq.append(f"{acc_var}+{term_var}")
                    acc_var = f"s{len(steps_seq) - 1}"
                else:
                    acc_var = term_var
                last_term_var = term_var
            elif (ctx_set & {"more", "older", "taller", "longer", "heavier", "faster", "higher"}) and ("than" in ctx_set):
                steps_seq.append(f"{last_term_var}+x{i}")
                term_var = f"s{len(steps_seq) - 1}"
                if accum_rel and (asks_total or "both" in low_q or "together" in low_q):
                    steps_seq.append(f"{acc_var}+{term_var}")
                    acc_var = f"s{len(steps_seq) - 1}"
                else:
                    acc_var = term_var
                last_term_var = term_var
            elif (ctx_set & {"less", "fewer", "younger", "shorter", "smaller", "lighter", "cheaper"}) and ("than" in ctx_set):
                steps_seq.append(f"{last_term_var}-x{i}")
                term_var = f"s{len(steps_seq) - 1}"
                if accum_rel and (asks_total or "both" in low_q or "together" in low_q):
                    steps_seq.append(f"{acc_var}+{term_var}")
                    acc_var = f"s{len(steps_seq) - 1}"
                else:
                    acc_var = term_var
                last_term_var = term_var
            elif is_rate_mul and not (set(lw[-3:]) & {"lose", "spend", "pay", "discount", "off", "less", "reduce"}):
                steps_seq.append(f"{acc_var}*x{i}")
                acc_var = last_term_var = f"s{len(steps_seq) - 1}"
            elif ctx_set & SUB_WORDS_V33:
                steps_seq.append(f"{acc_var}-x{i}")
                acc_var = f"s{len(steps_seq) - 1}"
                last_term_var = f"x{i}"
            elif (ctx_set & MUL_WORDS_V33) and not (ctx_set & ADD_WORDS_V33):
                steps_seq.append(f"{acc_var}*x{i}")
                acc_var = last_term_var = f"s{len(steps_seq) - 1}"
            elif (ctx_set & DIV_WORDS_V33) and nums[i] > 1.0:
                steps_seq.append(f"{acc_var}/x{i}")
                acc_var = last_term_var = f"s{len(steps_seq) - 1}"
            else:
                steps_seq.append(f"{acc_var}+x{i}")
                acc_var = f"s{len(steps_seq) - 1}"
                last_term_var = f"x{i}"
            i += 1

        if steps_seq:
            add_cand(steps_seq, 0.25 if accum_rel else 0.18)

    if k == 3:
        if num_ctxs[2][4] and asks_profit:
            add_cand(["x0+x1", "x0*x2", "x0+s1", "s2-s0"], 0.44)
            add_cand(["x0+x1", "s0*x2", "s1-x1"], 0.24)
        c1_sub = bool(set(num_ctxs[1][0] + num_ctxs[1][1]) & SUB_WORDS_V33)
        c1_mul = bool(set(num_ctxs[1][0] + num_ctxs[1][1]) & MUL_WORDS_V33)
        c2_sub = bool(set(num_ctxs[2][0] + num_ctxs[2][1]) & SUB_WORDS_V33)
        c2_mul = bool(set(num_ctxs[2][0] + num_ctxs[2][1]) & MUL_WORDS_V33)
        c2_div = bool((set(num_ctxs[2][0] + num_ctxs[2][1]) | q_words) & DIV_WORDS_V33)
        if asks_diff_or_left and nums[0] > nums[1] + nums[2]:
            add_cand(["x1+x2", "x0-s0"], 0.34)
        if asks_diff_or_left and nums[2] > nums[0] + nums[1]:
            add_cand(["x0+x1", "x2-s0"], 0.34)
        if (asks_diff_or_left or c2_sub) and nums[0] > nums[1] * nums[2] and c2_mul:
            add_cand(["x1*x2", "x0-s0"], 0.36)
        if c1_mul and c2_mul:
            add_cand(["x0*x1", "s0*x2"], 0.36)
        if c2_div and nums[2] > 1.0:
            add_cand(["x0-x1" if c1_sub else ("x0*x1" if c1_mul else "x0+x1"), "s0/x2"], 0.36)
    elif k == 4:
        c1_sub = bool(set(num_ctxs[1][0] + num_ctxs[1][1]) & SUB_WORDS_V33)
        c2_sub = bool(set(num_ctxs[2][0] + num_ctxs[2][1]) & SUB_WORDS_V33)
        c3_mul = bool(set(num_ctxs[3][0] + num_ctxs[3][1]) & MUL_WORDS_V33)
        c3_div = bool((set(num_ctxs[3][0] + num_ctxs[3][1]) | q_words) & DIV_WORDS_V33)
        has_d = ["$" in (num_ctxs[m][0] + num_ctxs[m][1]) for m in range(4)]
        if c1_sub and c2_sub and c3_mul:
            add_cand(["x0-x1-x2", "s0*x3"], 0.52)
        if c1_sub and c2_sub and c3_div:
            add_cand(["x0-x1-x2", "s0/x3"], 0.45)
        if num_ctxs[2][5] == "MUL_WORD" and num_ctxs[3][5] == "DIV_WORD":
            add_cand(["x1*x2", "x1+s0", "x0-s1", "s2/x3"], 0.45)
        if num_ctxs[1][5] == "DIV_WORD" and num_ctxs[3][5] == "MUL_WORD" and asks_diff_or_left:
            add_cand(["x0/x1", "x2*x3", "x0-s0-x2-s1"], 0.45)
        if (not has_d[0] and not has_d[1] and has_d[2] and has_d[3]) or (
            num_ctxs[0][3] == num_ctxs[1][3] and num_ctxs[2][3] == num_ctxs[3][3] and num_ctxs[0][3] != num_ctxs[2][3] and c3_mul
        ):
            if asks_diff_or_left or asks_profit:
                add_cand(["x0*x2", "x1*x3", "s0-s1"], 0.38)
                add_cand(["x0*x2", "x1*x3", "s1-s0"], 0.38)
            else:
                add_cand(["x0*x2", "x1*x3", "s0+s1"], 0.42)
        if (has_d[1] and has_d[3] and not has_d[0] and not has_d[2]) or (
            num_ctxs[0][2] == num_ctxs[1][2] and num_ctxs[2][2] == num_ctxs[3][2] and num_ctxs[1][2] != num_ctxs[2][2]
        ):
            if asks_diff_or_left or asks_profit:
                add_cand(["x0*x1", "x2*x3", "s0-s1"], 0.38)
                add_cand(["x0*x1", "x2*x3", "s1-s0"], 0.38)
            else:
                add_cand(["x0*x1", "x2*x3", "s0+s1"], 0.42)
        if asks_diff_or_left and nums[0] > nums[1] + nums[2] + nums[3]:
            add_cand(["x1+x2+x3", "x0-s0"], 0.36)
        if asks_diff_or_left and nums[0] > nums[1] * nums[2] + nums[3] and (set(num_ctxs[1][0] + num_ctxs[2][0]) & (MUL_WORDS_V33 | {"$"})):
            add_cand(["x1*x2", "s0+x3", "x0-s1"], 0.36)
        if c3_div and not c1_sub and not c2_sub and nums[3] > 1.0:
            add_cand(["x0+x1+x2", "s0/x3"], 0.36)
    elif k == 5:
        c4_div = bool((set(num_ctxs[4][0] + num_ctxs[4][1]) | q_words) & DIV_WORDS_V33)
        if asks_diff_or_left and nums[0] > nums[1] * nums[2] + nums[3] * nums[4]:
            add_cand(["x1*x2", "x3*x4", "s0+s1", "x0-s2"], 0.44)
        if asks_diff_or_left and nums[0] > nums[1] * nums[3] + nums[2] * nums[4]:
            add_cand(["x1*x3", "x2*x4", "s0+s1", "x0-s2"], 0.40)
        if asks_diff_or_left and nums[4] > nums[0] * nums[1] + nums[2] * nums[3]:
            add_cand(["x0*x1", "x2*x3", "s0+s1", "x4-s2"], 0.44)
        if asks_diff_or_left and nums[0] * nums[1] + nums[2] * nums[3] > nums[4]:
            add_cand(["x0*x1", "x2*x3", "s0+s1", "s2-x4"], 0.38)
        if c4_div and nums[4] > 1.0:
            add_cand(["x0*x1", "x2*x3", "s0+s1", "s2/x4"], 0.42)
        if asks_diff_or_left and nums[0] > sum(nums[1:]):
            add_cand(["x1+x2+x3+x4", "x0-s0"], 0.36)
    elif k == 6:
        if num_ctxs[0][3] == num_ctxs[1][3] == num_ctxs[2][3] and num_ctxs[3][3] == num_ctxs[4][3] == num_ctxs[5][3] and num_ctxs[0][3] != num_ctxs[3][3]:
            add_cand(["x0*x3", "x1*x4", "x2*x5", "s0+s1+s2"], 0.45)
        add_cand(["x0*x1", "x2*x3", "x4*x5", "s0+s1+s2"], 0.40)

    if role_probs is not None and k >= 2:
        pred_roles = [int(max(range(1, 8), key=lambda r: role_probs[i][r])) for i in range(k)]
        r_steps: List[str] = []
        tot_idx = max(range(k), key=lambda i: role_probs[i][3] if nums[i] >= max(nums) * 0.5 else -1.0)
        has_tot = role_probs[tot_idx][3] > 0.28 and asks_diff_or_left
        cur = f"x{tot_idx}" if has_tot else "x0"
        start_i = 0 if has_tot else 1
        for i in range(start_i, k):
            if has_tot and i == tot_idx:
                continue
            r = pred_roles[i]
            if r == 2 or (has_tot and r in (1, 2)):
                r_steps.append(f"{cur}-x{i}")
            elif r == 4:
                r_steps.append(f"{cur}*x{i}")
            elif r == 5:
                r_steps.append(f"{cur}/x{i}")
            else:
                r_steps.append(f"{cur}+x{i}")
            cur = f"s{len(r_steps) - 1}"
        if r_steps:
            add_cand(r_steps, 0.20)

    return cands


# ==============================================================================
# 4. ПОЛЕ РАЗМЕРНОСТЕЙ CRAM+z, ПРЕДКОМПИЛЯЦИЯ ДЕРЕВЬЕВ И ФИЗИКА РАЗМЕРНОСТЕЙ
# ==============================================================================
def extract_between_spans(q_text: str, k_slots: int) -> Dict[Tuple[int, int], List[str]]:
    """Извлекает слова, стоящие непосредственно между числами x_i и x_j в тексте вопроса."""
    q_c = re.sub(r"(\d),(\d)", r"\1\2", q_text)
    toks = re.findall(r"\$|\d+(?:\.\d+)?/?\d*%?|[A-Za-z]+(?:-[A-Za-z]+)*|[.,;?!]", q_c)
    num_words = set(WORD_NUMS_V33.keys())
    pos_list = [idx_t for idx_t, t in enumerate(toks) if re.match(r"^\d", t.lower()) or t.lower() in num_words]
    spans: Dict[Tuple[int, int], List[str]] = {}
    for i in range(k_slots):
        for j in range(i + 1, k_slots):
            if i < len(pos_list) and j < len(pos_list):
                pa, pb = pos_list[i], pos_list[j]
                spans[(i, j)] = [VERB_NORM_V33.get(w.lower(), w.lower()) for w in toks[pa + 1 : min(pb, pa + 14)] if w.isalpha() or w == "$"]
            else:
                spans[(i, j)] = []
    return spans


def diffuse_slot_units_cram_z(nums: List[float], ctxs: list, clauses: list, q_text: str, gamma: float = 0.85):
    """Точное выпуклое равновесие CRAM+z: `(I + gamma * L) z* = U_0`.
    Прокидывает скрытые существительные-единицы между соседними слотами одной клаузы
    (когда автор задачи опустил существительное во второй половине фразы) и определяет размерность вопроса.
    """
    k = len(nums)
    raw_nouns: List[str] = []
    is_money = [False] * k
    is_rate = [False] * k
    is_pct = [False] * k
    is_mult = [False] * k
    is_uconv = [False] * k
    is_frac_num = [False] * k
    is_frac_den = [False] * k

    for i in range(k - 1):
        if ctxs[i][0] == ctxs[i + 1][0] and ctxs[i][1] == ctxs[i + 1][1] and nums[i + 1] > 1.0:
            is_frac_num[i] = True
            is_frac_den[i + 1] = True

    for i in range(k):
        lw_r, rw_r, _, _, pct_flag, w_kind, _ = ctxs[i]
        is_pct[i] = bool(pct_flag)
        is_mult[i] = (w_kind in ("MUL_WORD", "DIV_WORD")) or is_frac_num[i] or is_frac_den[i]
        is_uconv[i] = (w_kind == "UNIT")
        is_money[i] = ("$" in lw_r[-2:]) or ("$" in rw_r[:2]) or bool(set(rw_r[:2]) & {"dollar", "dollars", "cent", "cents"})
        if not is_money[i] and ("cost" in lw_r[-2:] or "pay" in lw_r[-2:] or "price" in lw_r[-2:] or "earn" in lw_r[-2:]):
            if not rw_r or singularize_noun(rw_r[0]) in (TIME_WORDS | RATE_MARKERS):
                is_money[i] = True

        if (set(rw_r[:4]) & {"each", "every", "per", "apiece", "hourly", "daily", "weekly", "monthly"}) or (
            is_money[i] and len(rw_r) >= 2 and rw_r[0] in ("a", "an")
        ) or ("each" in lw_r[-3:] and ("had" in lw_r[-2:] or "has" in lw_r[-2:] or "need" in lw_r[-2:] or "get" in lw_r[-2:])):
            is_rate[i] = True

        noun = ""
        for w in rw_r[:3]:
            ws = singularize_noun(w)
            if ws not in STOP_NOUNS and ws != "$" and len(ws) > 1 and not ws.isdigit():
                noun = ws
                break
        raw_nouns.append(noun)

    vocab_nouns = sorted({n for n in raw_nouns if n})
    if vocab_nouns and k >= 2:
        U0 = np.zeros((k, len(vocab_nouns)), dtype=np.float64)
        n2i = {n: idx for idx, n in enumerate(vocab_nouns)}
        for i, n in enumerate(raw_nouns):
            if n in n2i:
                U0[i, n2i[n]] = 1.0
        W_adj = np.zeros((k, k), dtype=np.float64)
        for i in range(k):
            for j in range(i + 1, k):
                if is_pct[i] or is_pct[j] or is_uconv[i] or is_uconv[j]:
                    continue
                same_c = (ctxs[i][2] == ctxs[j][2])
                same_s = (ctxs[i][3] == ctxs[j][3])
                w_ij = 0.65 if same_c else (0.30 if same_s else (0.10 if j == i + 1 else 0.0))
                W_adj[i, j] = W_adj[j, i] = w_ij
        L_lap = np.diag(W_adj.sum(axis=1)) - W_adj
        Z_star = np.linalg.solve(np.eye(k) + gamma * L_lap, U0)
        for i in range(k):
            if raw_nouns[i] == "" and not is_money[i] and not is_pct[i] and not is_mult[i] and not is_uconv[i]:
                best_u = int(np.argmax(Z_star[i]))
                if Z_star[i, best_u] > 0.18:
                    raw_nouns[i] = vocab_nouns[best_u]

    q_last = clauses[-1][0] if clauses else q_text
    q_toks = [singularize_noun(w) for w in re.findall(r"[A-Za-z]+|\$", q_last)]
    q_wants_money = bool(set(q_toks) & MONEY_WORDS) or ("how much" in q_last.lower() and not any(t in TIME_WORDS for t in q_toks))
    q_noun = ""
    m_hm = re.search(r"how many ([a-z]+(?:\s+[a-z]+)?)", q_last.lower())
    if m_hm:
        for cw in [singularize_noun(w) for w in m_hm.group(1).split()]:
            if cw not in STOP_NOUNS and cw not in ("more", "fewer", "less", "time"):
                q_noun = cw
                break

    slot_dims = []
    for i in range(k):
        dk = 2 if is_pct[i] else (4 if is_uconv[i] else (3 if is_mult[i] else (1 if is_money[i] else 0)))
        slot_dims.append((dk, raw_nouns[i], is_rate[i], is_frac_num[i], is_frac_den[i]))
    return slot_dims, q_wants_money, q_noun


class ParsedTreeBonds:
    """Предкомпилирует AST-дерево в:
      - Эффективные роли слотов (с точным top-down учётом знака '-' и делителя '/', так что x0 - (x1 + x2)
        корректно ставит роль вычитаемого и x1, и x2!)
      - Межподдеревные парные связи S_L x S_R
      - Линейный регистровый байткод `bc_ops` для быстрой проверки физики единиц измерения.
    """
    __slots__ = ("prog", "flat", "sig", "roles", "bonds", "root_op", "n_used", "n_ops", "prior_logp", "bc_ops")

    def __init__(self, prog: Tuple[str, ...], k: int, sig: float, prior_logp: float = -6.5):
        self.prog = prog
        self.flat = prog_to_flat_expr(prog)
        self.sig = sig
        self.prior_logp = prior_logp
        roles = np.zeros(k, dtype=np.int32)
        bond_weights: Dict[Tuple[int, int, int], float] = {}
        bc_ops: List[Tuple[int, float, int, int]] = []
        n_ops = 0

        try:
            tree = ast.parse(self.flat, mode="eval").body
        except Exception:
            self.roles = roles; self.bonds = []; self.root_op = 0; self.n_used = 0; self.n_ops = 0; self.bc_ops = []
            return

        def add_weighted_bond(u: int, v: int, op_code: int, w: float):
            if u < 0 or v < 0 or u == v:
                return
            if u < v:
                key = (u, v, op_code)
            else:
                rev_map = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4, 5: 6, 6: 5}
                key = (v, u, rev_map[op_code])
            bond_weights[key] = bond_weights.get(key, 0.0) + w

        def compile_and_walk(node: ast.AST, pol_sign: int, mul_dir: int) -> Tuple[int, int, List[int], bool]:
            nonlocal n_ops
            if isinstance(node, ast.Name) and node.id.startswith("x"):
                idx = int(node.id[1:])
                reg = len(bc_ops)
                bc_ops.append((0, 0.0, idx, 0))
                if 0 <= idx < k:
                    if mul_dir == -1:
                        roles[idx] = 5
                    elif pol_sign == -1:
                        roles[idx] = 2
                    elif roles[idx] == 0:
                        roles[idx] = 1
                    return reg, idx, [idx], False
                return reg, -1, [], False

            if isinstance(node, ast.Constant):
                v = float(node.value) if isinstance(node.value, (int, float)) else 1.0
                reg = len(bc_ops)
                bc_ops.append((1, v, 0, 0))
                return reg, -1, [], False

            if isinstance(node, ast.UnaryOp):
                s_next = -pol_sign if isinstance(node.op, ast.USub) else pol_sign
                creg, h, lvs, _ = compile_and_walk(node.operand, s_next, mul_dir)
                if isinstance(node.op, ast.USub):
                    reg = len(bc_ops)
                    bc_ops.append((2, 0.0, creg, 0))
                    return reg, h, lvs, True
                return creg, h, lvs, True

            if isinstance(node, ast.BinOp):
                n_ops += 1
                op = node.op
                if isinstance(op, ast.Add):
                    lreg, lh, llvs, _ = compile_and_walk(node.left, pol_sign, mul_dir)
                    rreg, rh, rlvs, _ = compile_and_walk(node.right, pol_sign, mul_dir)
                    reg = len(bc_ops)
                    bc_ops.append((3, 0.0, lreg, rreg))
                    op_code, h_out, has_a = 0, (lh if lh >= 0 else rh), True
                elif isinstance(op, ast.Sub):
                    lreg, lh, llvs, _ = compile_and_walk(node.left, pol_sign, mul_dir)
                    rreg, rh, rlvs, _ = compile_and_walk(node.right, -pol_sign, mul_dir)
                    reg = len(bc_ops)
                    bc_ops.append((4, 0.0, lreg, rreg))
                    if lh >= 0 and pol_sign == 1 and roles[lh] == 1:
                        roles[lh] = 3
                    op_code, h_out, has_a = (1 if pol_sign == 1 else 2), (lh if lh >= 0 else rh), True
                elif isinstance(op, ast.Mult):
                    lreg, lh, llvs, ladd = compile_and_walk(node.left, pol_sign, mul_dir)
                    rreg, rh, rlvs, radd = compile_and_walk(node.right, pol_sign, mul_dir)
                    reg = len(bc_ops)
                    bc_ops.append((5, 0.0, lreg, rreg))
                    if mul_dir == 1:
                        if lh >= 0 and not ladd:
                            roles[lh] = 7 if radd else 4
                        if rh >= 0 and not radd:
                            roles[rh] = 7 if ladd else 4
                    op_code = 4 if (ladd or radd) else 3
                    h_cand = lh if (ladd or not radd) else rh
                    h_out, has_a = (h_cand if h_cand >= 0 else max(lh, rh)), (ladd or radd)
                elif isinstance(op, ast.Div):
                    lreg, lh, llvs, ladd = compile_and_walk(node.left, pol_sign, mul_dir)
                    rreg, rh, rlvs, radd = compile_and_walk(node.right, pol_sign, -mul_dir)
                    reg = len(bc_ops)
                    bc_ops.append((6, 0.0, lreg, rreg))
                    if lh >= 0 and mul_dir == 1 and not ladd and roles[lh] in (0, 1):
                        roles[lh] = 6
                    op_code, h_out, has_a = (5 if mul_dir == 1 else 6), (lh if lh >= 0 else rh), (ladd or radd)
                else:
                    return 0, -1, [], False

                pairs = [(u, v) for u in llvs for v in rlvs if u != v]
                if pairs:
                    add_weighted_bond(lh, rh, op_code, 0.55)
                    w_each = 0.45 / float(len(pairs))
                    for u, v in pairs:
                        add_weighted_bond(u, v, op_code, w_each)
                return reg, h_out, llvs + rlvs, has_a

            reg = len(bc_ops)
            bc_ops.append((1, 0.0, 0, 0))
            return reg, -1, [], False

        compile_and_walk(tree, +1, +1)
        root_op = 0
        if isinstance(tree, ast.BinOp):
            root_op = 1 if isinstance(tree.op, ast.Add) else (2 if isinstance(tree.op, ast.Sub) else (3 if isinstance(tree.op, ast.Mult) else (4 if isinstance(tree.op, ast.Div) else 0)))

        tot_bw = sum(bond_weights.values())
        self.bonds = [(a, b, oc, w / tot_bw) for (a, b, oc), w in bond_weights.items()] if tot_bw > 0 else []
        self.roles = roles
        self.root_op = root_op
        self.n_used = int(np.sum(roles > 0))
        self.n_ops = n_ops
        self.bc_ops = bc_ops


def eval_bytecode_dim_physics(
    bc_ops: List[Tuple[int, float, int, int]],
    ns: List[float],
    slot_dims: list,
    q_wants_money: bool,
    q_noun: str,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """Проверяет согласованность единиц измерения во всех промежуточных узлах формулы."""
    if not bc_ops:
        return (0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0)
    n_reg = len(bc_ops)
    r_val = [0.0] * n_reg
    r_dk = [0] * n_reg
    r_noun = [""] * n_reg
    r_rate = [False] * n_reg
    n_add = n_add_ok = n_add_clash = 0
    n_mul = n_mul_ok = n_mul_clash = 0
    n_div = n_div_ok = n_neg_sub = 0
    k_len = len(ns)

    for idx_op, (op, cval, a1, a2) in enumerate(bc_ops):
        if op == 0:
            if 0 <= a1 < k_len:
                dk, noun, rt, _, _ = slot_dims[a1]
                r_val[idx_op], r_dk[idx_op], r_noun[idx_op], r_rate[idx_op] = ns[a1], dk, noun, rt
            else:
                r_val[idx_op], r_dk[idx_op] = 1.0, 5
        elif op == 1:
            r_val[idx_op], r_dk[idx_op] = cval, 5
        elif op == 2:
            r_val[idx_op], r_dk[idx_op], r_noun[idx_op], r_rate[idx_op] = -r_val[a1], r_dk[a1], r_noun[a1], r_rate[a1]
        elif op in (3, 4):
            n_add += 1
            lv, rv = r_val[a1], r_val[a2]
            ldk, rdk = r_dk[a1], r_dk[a2]
            ln, rn = r_noun[a1], r_noun[a2]
            ov = (lv + rv) if op == 3 else (lv - rv)
            if op == 4 and ov < -1e-6:
                n_neg_sub += 1
            if ldk in (2, 3, 4) or rdk in (2, 3, 4):
                n_add_clash += 1
            elif (ldk == 1) != (rdk == 1) and ldk != 5 and rdk != 5:
                n_add_clash += 1
            elif (ldk == 1 and rdk == 1) or (ln != "" and ln == rn):
                n_add_ok += 1
            r_val[idx_op] = ov
            r_dk[idx_op] = 1 if (ldk == 1 or rdk == 1) else (ldk if ldk != 5 else rdk)
            r_noun[idx_op] = ln or rn
            r_rate[idx_op] = r_rate[a1] and r_rate[a2]
        elif op == 5:
            n_mul += 1
            lv, rv = r_val[a1], r_val[a2]
            ldk, rdk = r_dk[a1], r_dk[a2]
            ln, rn = r_noun[a1], r_noun[a2]
            lr, rr = r_rate[a1], r_rate[a2]
            r_val[idx_op] = lv * rv
            if ldk == 1 and rdk == 1 and not lr and not rr:
                n_mul_clash += 1; r_dk[idx_op] = 1; r_noun[idx_op] = ln or rn
            elif rdk in (2, 3, 4):
                n_mul_ok += 1; r_dk[idx_op] = ldk; r_noun[idx_op] = ln
            elif ldk in (2, 3, 4):
                n_mul_ok += 1; r_dk[idx_op] = rdk; r_noun[idx_op] = rn
            elif lr or rr or ldk == 1 or rdk == 1:
                n_mul_ok += 1
                r_dk[idx_op] = 1 if (ldk == 1 or rdk == 1) else 0
                r_noun[idx_op] = ln if (rr and not lr) else (rn if (lr and not rr) else (ln or rn))
            else:
                if ln != "" and ln == rn:
                    n_mul_clash += 1
                r_dk[idx_op] = 0; r_noun[idx_op] = ln or rn
        elif op == 6:
            n_div += 1
            lv, rv = r_val[a1], r_val[a2]
            ldk, rdk = r_dk[a1], r_dk[a2]
            ln = r_noun[a1]
            ov = lv / rv if abs(rv) > 1e-9 else 1e9
            r_val[idx_op] = ov
            if abs(rv) > 1e-6 and abs(ov - round(ov * 4) / 4.0) < 1e-3 and ov > 0:
                n_div_ok += 1
            r_dk[idx_op] = 0 if (ldk == 1 and rdk == 1) else (ldk if rdk in (0, 3, 4, 5) else 0)
            r_noun[idx_op] = "" if (ldk == 1 and rdk == 1) else ln

    root_dk, root_noun = r_dk[-1], r_noun[-1]
    f5 = f6 = 0.0
    if q_wants_money:
        f5 = 1.0 if root_dk == 1 else 0.0
        f6 = 1.0 if root_dk != 1 else 0.0
    else:
        if q_noun != "" and root_noun != "":
            f5 = 1.0 if root_noun == q_noun else 0.0
        f6 = 1.0 if (root_dk == 1 and q_noun != "") else 0.0
    return (
        float(n_add_ok) / float(max(n_add, 1)),
        float(n_add_clash),
        float(n_mul_ok) / float(max(n_mul, 1)),
        float(n_mul_clash),
        float(n_div_ok) / float(max(n_div, 1)) if n_div > 0 else 0.5,
        f5, f6, float(n_neg_sub),
    )


def generate_clean_reduced_programs(
    q_text: str,
    ns: List[float],
    ctxs: list,
    clauses: list,
    arity_pool_progs: Dict[int, List[Tuple[Tuple[str, ...], float]]],
) -> List[Tuple[Tuple[str, ...], float]]:
    """Сворачивает локальные пары (дроби a/b, проценты, тариф*количество, конверсию единиц) в поддеревья."""
    k = len(ns)
    if k < 3:
        return []
    q_cl = clauses[-1][0].lower() if clauses else q_text.lower()
    asks_total = any(w in q_cl for w in ("total", "altogether", "together", "all", "combined", "both", "sum"))
    asks_rem = any(w in q_cl for w in ("left", "remain", "remaining", "more", "fewer", "less", "still", "need", "change", "difference", "profit"))
    rate_set = {"each", "every", "per", "apiece", "an", "a", "for", "at", "$", "hour", "day", "week", "month", "year", "boxes", "packs", "bags", "pairs", "dozen"}
    reduced_states = []

    def add_state(idx: int, v_new: float, expr_new: str, bonus: float):
        ns_r = ns[:idx] + [v_new] + ns[idx + 2 :]
        m_ctx = (ctxs[idx][0], ctxs[idx + 1][1], ctxs[idx][2], ctxs[idx][3], False, "DIGIT", ctxs[idx][6] + ctxs[idx + 1][6])
        ctxs_r = ctxs[:idx] + [m_ctx] + ctxs[idx + 2 :]
        exprs_r = [f"x{m}" for m in range(idx)] + [expr_new] + [f"x{m}" for m in range(idx + 2, k)]
        reduced_states.append((ns_r, ctxs_r, exprs_r, bonus))

    pair_mul_idxs = []
    for i in range(k - 1):
        ci, cn = ctxs[i], ctxs[i + 1]
        cl_set = set(ci[6] + cn[6])
        ctx_pair = set(ci[0] + ci[1] + cn[0] + cn[1])
        cn_set = set(cn[0] + cn[1])
        if ci[0] == cn[0] and ci[1] == cn[1] and ns[i + 1] > 1.0:
            add_state(i, ns[i] / ns[i + 1], f"(x{i}/x{i+1})", 0.55)
            continue
        if ci[2] == cn[2] and ci[5] in ("DIGIT", "NUM_WORD") and cn[5] in ("DIGIT", "NUM_WORD") and ns[i] > 1.0 and ns[i + 1] > 1.0 and (cl_set & rate_set) and not (cl_set & {"into", "split", "divide", "equally"}):
            add_state(i, ns[i] * ns[i + 1], f"(x{i}*x{i+1})", 0.45)
            pair_mul_idxs.append(i)
        if cn[4]:
            if cn_set & {"less", "reduce", "discount", "off", "decrease", "lose", "fewer", "tax", "used", "spent", "sold", "eat"}:
                add_state(i, ns[i] * (1.0 - ns[i + 1]), f"(x{i}-(x{i}*x{i+1}))", 0.50)
            elif cn_set & {"more", "increase", "up", "higher", "tip", "interest", "bonus", "extra", "profit"}:
                add_state(i, ns[i] * (1.0 + ns[i + 1]), f"(x{i}+(x{i}*x{i+1}))", 0.50)
            add_state(i, ns[i] * ns[i + 1], f"(x{i}*x{i+1})", 0.40)
        if cn[5] == "MUL_WORD":
            add_state(i, ns[i] * ns[i + 1], f"(x{i}*x{i+1})", 0.42)
            if ("as" in ctx_pair) or asks_total:
                add_state(i, ns[i] + ns[i] * ns[i + 1], f"(x{i}+(x{i}*x{i+1}))", 0.45)
        elif cn[5] == "DIV_WORD" and ns[i + 1] > 0:
            add_state(i, ns[i] / ns[i + 1], f"(x{i}/x{i+1})", 0.42)
            if ("as" in ctx_pair) and asks_total:
                add_state(i, ns[i] + ns[i] / ns[i + 1], f"(x{i}+(x{i}/x{i+1}))", 0.45)
            elif asks_rem and ("of" in ctx_pair or "remain" in ctx_pair):
                add_state(i, ns[i] - ns[i] / ns[i + 1], f"(x{i}-(x{i}/x{i+1}))", 0.45)

    if len(pair_mul_idxs) >= 2:
        i1 = pair_mul_idxs[0]
        for i2 in pair_mul_idxs[1:]:
            if i2 >= i1 + 2:
                ns_r2, ctxs_r2, exprs_r2 = [], [], []
                m = 0
                while m < k:
                    if m in (i1, i2):
                        ns_r2.append(ns[m] * ns[m + 1])
                        ctxs_r2.append((ctxs[m][0], ctxs[m + 1][1], ctxs[m][2], ctxs[m][3], False, "DIGIT", ctxs[m][6] + ctxs[m + 1][6]))
                        exprs_r2.append(f"(x{m}*x{m+1})")
                        m += 2
                    else:
                        ns_r2.append(ns[m]); ctxs_r2.append(ctxs[m]); exprs_r2.append(f"x{m}")
                        m += 1
                if len(ns_r2) >= 2:
                    reduced_states.append((ns_r2, ctxs_r2, exprs_r2, 0.55))
                break

    def lift_prog(p: Tuple[str, ...], exprs_r: List[str]) -> Tuple[str, ...]:
        return tuple(re.sub(r"\bx(\d+)\b", lambda m: exprs_r[int(m.group(1))] if int(m.group(1)) < len(exprs_r) else m.group(0), st) for st in p)

    out_lifted: List[Tuple[Tuple[str, ...], float]] = []
    seen = set()
    for ns_r, ctxs_r, exprs_r, b_st in reduced_states[:5]:
        kr = len(ns_r)
        if kr < 2:
            continue
        for p_r, b_c in synthesize_clause_sequential_programs(q_text, ns_r, ctxs_r, clauses, None)[:4]:
            pl = lift_prog(p_r, exprs_r)
            if pl not in seen:
                seen.add(pl); out_lifted.append((pl, b_st * 0.6 + 0.3 * b_c))
        for p_r, _ in arity_pool_progs.get(kr, [])[:14]:
            pl = lift_prog(p_r, exprs_r)
            if pl not in seen:
                seen.add(pl); out_lifted.append((pl, b_st * 0.4))

    if ctxs[-1][5] == "UNIT" and k >= 3:
        ns_pre, ctxs_pre = ns[:-1], ctxs[:-1]
        kr = len(ns_pre)
        pre_cands = list(synthesize_clause_sequential_programs(q_text, ns_pre, ctxs_pre, clauses, None)[:3])
        pre_cands.extend(arity_pool_progs.get(kr, [])[:10])
        for p_pre, _ in pre_cands:
            last_s = f"s{len(p_pre) - 1}"
            for tail_op in ("*", "/"):
                pm = tuple(list(p_pre) + [f"{last_s}{tail_op}x{k-1}"])
                if pm not in seen:
                    seen.add(pm); out_lifted.append((pm, 0.35))
    return out_lifted


# ==============================================================================
# 5. ГЛУБОКИЙ МОДУЛЬ РЕШАТЕЛЯ: CramGSM8KSolver (0 BPTT)
# ==============================================================================
class CramGSM8KSolver:
    """Энергетический сборщик формул CRAM+z с корой прецедентов SimVQ и сферическим АЛУ.
    Интерфейс:
      - `fit(train_items)` — обучение строго на `train.jsonl` без BPTT.
      - `solve_blind(question_text)` — строго слепое решение по тексту вопроса.
    """

    N_META = 16

    def __init__(self, alu: SphericalBase10FixedPointALU, knobs: Optional[Dict[str, float | int]] = None):
        self.alu = alu
        self.knobs = dict(TUNING_KNOBS)
        if knobs:
            self.knobs.update(knobs)
        self.d_ctx = int(self.knobs["d_ctx"])
        self.d_unary = int(self.knobs["d_unary"])
        self.d_pair = int(self.knobs["d_pair"])
        self.d_root = int(self.knobs["d_root"])

        self.vocab_keep: Set[str] = set()
        self.parsed_cache: Dict[Tuple[Tuple[str, ...], int], ParsedTreeBonds] = {}
        self.arity_pool: Dict[int, List[ParsedTreeBonds]] = {}
        self.arity_pool_progs: Dict[int, List[Tuple[Tuple[str, ...], float]]] = {}
        self.arity_prior_vec: Dict[int, np.ndarray] = {}
        self.mem_X: Dict[int, np.ndarray] = {}
        self.mem_Y: Dict[int, np.ndarray] = {}

        self.W_u = np.zeros((8, self.d_unary), dtype=np.float32)
        self.W_p = np.zeros((7, self.d_pair), dtype=np.float32)
        self.W_r = np.zeros((5, self.d_root), dtype=np.float32)
        self.K_cram = np.zeros((8, 8, 7), dtype=np.float32)
        self.w_meta = np.zeros(self.N_META, dtype=np.float32)

    def _get_tree(self, prog: Tuple[str, ...], k: int, sig: float, prior_logp: float = -6.5) -> ParsedTreeBonds:
        key = (prog, k)
        tb = self.parsed_cache.get(key)
        if tb is None:
            tb = ParsedTreeBonds(prog, k, sig=sig, prior_logp=prior_logp)
            self.parsed_cache[key] = tb
        return tb

    def _featurize_cortex(self, q_text: str, nums: List[float], ctxs: list, clauses: list) -> np.ndarray:
        vec = np.zeros(self.d_ctx, dtype=np.float32)
        hb = self.d_ctx - 140
        max_v = max(nums) if nums else 1.0
        for idx in range(min(len(nums), 9)):
            val = nums[idx]
            lw_raw, rw_raw, c_idx, _, is_pct, w_kind, cl_alpha_raw = ctxs[idx]
            lw = [norm_tok(w, self.vocab_keep) for w in lw_raw]
            rw = [norm_tok(w, self.vocab_keep) for w in rw_raw]
            cl_a = [norm_tok(w, self.vocab_keep) for w in cl_alpha_raw]
            for dist, w in enumerate(reversed(lw)):
                vec[det_hash(f"L_{idx}_{w}", hb)] += 1.7 if dist < 2 else 0.8
                vec[det_hash(f"LD_{idx}_{dist}_{w}", hb)] += 1.2
                if w in SUB_WORDS_V33: vec[hb + idx * 8 + 0] += 1.2
                if w in MUL_WORDS_V33: vec[hb + idx * 8 + 1] += 1.2
                if w in DIV_WORDS_V33: vec[hb + idx * 8 + 2] += 1.2
                if w in ADD_WORDS_V33: vec[hb + idx * 8 + 3] += 1.2
            for dist, w in enumerate(rw):
                vec[det_hash(f"R_{idx}_{w}", hb)] += 1.7 if dist < 2 else 0.8
                vec[det_hash(f"RD_{idx}_{dist}_{w}", hb)] += 1.2
                if w in SUB_WORDS_V33: vec[hb + idx * 8 + 0] += 1.2
                if w in MUL_WORDS_V33: vec[hb + idx * 8 + 1] += 1.2
                if w in DIV_WORDS_V33: vec[hb + idx * 8 + 2] += 1.2
                if w in ADD_WORDS_V33: vec[hb + idx * 8 + 3] += 1.2
            for w in cl_a:
                if w not in ("<ENT>", "<NNS>"):
                    vec[det_hash(f"CL_{idx}_{w}", hb)] += 0.6
            if val == max_v: vec[hb + idx * 8 + 4] = 2.2
            if is_pct: vec[hb + idx * 8 + 5] = 2.8
            if w_kind == "DIV_WORD": vec[hb + idx * 8 + 6] = 3.0
            elif w_kind == "MUL_WORD": vec[hb + idx * 8 + 7] = 3.0
            if idx + 1 < len(ctxs) and ctxs[idx + 1][2] == c_idx:
                vec[hb + 80 + min(idx, 8)] = 2.0
            if idx > 0 and nums[0] != 0 and val > 0:
                if abs(nums[0] % val) < 1e-4: vec[hb + 90 + min(idx, 8)] = 2.0
                if abs(val % nums[0]) < 1e-4: vec[hb + 100 + min(idx, 8)] = 2.0
                if nums[idx - 1] > 0 and abs(nums[idx - 1] % val) < 1e-4: vec[hb + 110 + min(idx, 8)] = 2.0
                if nums[idx - 1] > 0 and abs(val % nums[idx - 1]) < 1e-4: vec[hb + 120 + min(idx, 8)] = 2.0

        q_words = [norm_tok(w, self.vocab_keep) for w in re.findall(r"[A-Za-z]+|\$", clauses[-1][0] if clauses else q_text)][-15:]
        for w in q_words:
            if w not in ("<ENT>", "<NNS>"):
                vec[det_hash(f"Q_{w}", hb)] += 2.0
        for w1, w2 in zip(q_words[:-1], q_words[1:]):
            vec[det_hash(f"QB_{w1}_{w2}", hb)] += 2.6
        vec /= max(float(np.linalg.norm(vec)), 1e-6)
        return vec

    def _build_features(self, q_text: str, nums: List[float], ctxs: list, clauses: list):
        k = len(nums)
        slot_dims, q_wants_money, q_noun = diffuse_slot_units_cram_z(
            nums, ctxs, clauses, q_text, gamma=float(self.knobs["cram_z_gamma"])
        )
        between_spans = extract_between_spans(q_text, k)
        ctx_sphere = self._featurize_cortex(q_text, nums, ctxs, clauses)

        U_feat = np.zeros((k, self.d_unary), dtype=np.float32)
        hb_u = self.d_unary - 48
        max_v = max(nums) if nums else 1.0
        min_v = min(nums) if nums else 0.0
        q_str_last = clauses[-1][0] if clauses else q_text
        q_words = [norm_tok(w, self.vocab_keep) for w in re.findall(r"[A-Za-z]+|\$", q_str_last)][-15:]
        q_raw_low = [w.lower() for w in re.findall(r"[A-Za-z]+|\$", q_str_last)]
        q_mode = (
            "PROFIT" if "profit" in q_raw_low else
            ("DIFF" if bool(set(q_raw_low) & {"left", "remain", "remaining", "still", "need", "change", "difference", "more", "fewer", "less"}) else
             ("TOT" if bool(set(q_raw_low) & {"total", "altogether", "together", "combined", "all", "sum", "both"}) else
              ("AVG" if bool(set(q_raw_low) & {"each", "average", "per", "equally", "split", "share"}) else "GEN")))
        )

        slot_content_sets = [
            {singularize_noun(w) for w in (ctxs[i][0][-4:] + ctxs[i][1][:4] + ctxs[i][6][:8]) if w.lower() not in STOP_NOUNS and len(w) > 2}
            for i in range(k)
        ]

        for i in range(k):
            vec = U_feat[i]
            val = nums[i]
            lw_raw, rw_raw, c_idx, _, is_pct, w_kind, cl_alpha_raw = ctxs[i]
            dk_i, noun_i, rate_i, fnum_i, fden_i = slot_dims[i]
            money_i = (dk_i == 1)
            lw = [norm_tok(w, self.vocab_keep) for w in lw_raw]
            rw = [norm_tok(w, self.vocab_keep) for w in rw_raw]
            cl_alpha = [norm_tok(w, self.vocab_keep) for w in cl_alpha_raw]

            for dist, w in enumerate(reversed(lw)):
                vec[det_hash(f"UL_{w}", hb_u)] += 1.8 if dist < 2 else 0.8
                if dist < 3:
                    vec[det_hash(f"ULD_{dist}_{w}", hb_u)] += 1.4
                    vec[det_hash(f"ULM_{q_mode}_{w}", hb_u)] += 1.2
            for dist, w in enumerate(rw):
                vec[det_hash(f"UR_{w}", hb_u)] += 1.8 if dist < 2 else 0.8
                if dist < 3:
                    vec[det_hash(f"URD_{dist}_{w}", hb_u)] += 1.4
                    vec[det_hash(f"URM_{q_mode}_{w}", hb_u)] += 1.2
            if len(lw) >= 2: vec[det_hash(f"ULB_{lw[-2]}_{lw[-1]}", hb_u)] += 1.5
            if len(rw) >= 2: vec[det_hash(f"URB_{rw[0]}_{rw[1]}", hb_u)] += 1.5
            if lw and rw: vec[det_hash(f"UB_{lw[-1]}_{rw[0]}", hb_u)] += 1.6
            for w in cl_alpha:
                if w not in ("<ENT>", "<NNS>"):
                    vec[det_hash(f"UC_{w}", hb_u)] += 0.6
            for qw in q_words[-8:]:
                if qw not in ("<ENT>", "<NNS>"):
                    vec[det_hash(f"UQ_{qw}", hb_u)] += 0.7

            noun_match_q = (noun_i != "" and noun_i == q_noun) or (money_i and q_wants_money)
            vec[hb_u + 0] = 1.0
            vec[hb_u + 1] = float(i == 0) * 2.2
            vec[hb_u + 2] = float(i == k - 1) * 2.2
            vec[hb_u + 3] = float(c_idx == len(clauses) - 1) * 2.5
            vec[hb_u + 4] = float(c_idx == 0) * 1.6
            vec[hb_u + 5] = float(val == max_v) * 2.2
            vec[hb_u + 6] = float(val == min_v) * 1.5
            vec[hb_u + 7] = float(is_pct) * 3.2
            vec[hb_u + 8] = float(w_kind == "DIV_WORD") * 3.2
            vec[hb_u + 9] = float(w_kind == "MUL_WORD") * 3.2
            vec[hb_u + 10] = float(w_kind == "UNIT") * 2.8
            vec[hb_u + 11] = float(money_i) * 2.2
            vec[hb_u + 12] = float(rate_i) * 2.6
            vec[hb_u + 13] = float(noun_match_q) * 2.5
            vec[hb_u + 14] = float(any(w in SUB_WORDS_V33 for w in lw_raw[-3:] + rw_raw[:2])) * 2.2
            vec[hb_u + 15] = float(any(w in MUL_WORDS_V33 for w in lw_raw[-3:] + rw_raw[:3])) * 2.2
            vec[hb_u + 16] = float(any(w in DIV_WORDS_V33 for w in lw_raw[-3:] + rw_raw[:3])) * 2.2
            vec[hb_u + 17] = float(any(w in ADD_WORDS_V33 for w in lw_raw[-3:] + rw_raw[:2])) * 1.8
            vec[hb_u + 18] = float(fnum_i) * 3.0
            vec[hb_u + 19] = float(fden_i) * 3.2
            vec[hb_u + 20] = float(i) / max(1.0, float(k - 1)) * 1.5
            vec /= max(float(np.linalg.norm(vec)), 1e-6)

        pair_idx_map: Dict[Tuple[int, int], int] = {}
        n_pairs = k * (k - 1) // 2
        P_feat_mat = np.zeros((max(n_pairs, 1), self.d_pair), dtype=np.float32)
        hb_p = self.d_pair - 52
        p_cursor = 0

        for i in range(k):
            for j in range(i + 1, k):
                pair_idx_map[(i, j)] = p_cursor
                vp = P_feat_mat[p_cursor]
                p_cursor += 1
                vi, vj = nums[i], nums[j]
                lwi_r, rwi_r, ci, si, pct_i, wk_i, _ = ctxs[i]
                lwj_r, rwj_r, cj, sj, pct_j, wk_j, cla_j = ctxs[j]
                dki, nouni, ratei, fni, _ = slot_dims[i]
                dkj, nounj, ratej, _, fdj = slot_dims[j]
                moni, monj = (dki == 1), (dkj == 1)
                same_cl = (ci == cj)
                adj_cl = (cj == ci + 1)
                same_noun = (nouni != "" and nouni == nounj)
                exact_frac_pair = (j == i + 1 and fni and fdj)
                lex_overlap = len(slot_content_sets[i] & slot_content_sets[j])

                mid_toks = [norm_tok(w, self.vocab_keep) for w in between_spans.get((i, j), [])]
                for w in mid_toks:
                    if w not in ("<ENT>", "<NNS>"):
                        vp[det_hash(f"MID_{w}", hb_p)] += 1.8
                for w1, w2 in zip(mid_toks[:-1], mid_toks[1:]):
                    vp[det_hash(f"MIDB_{w1}_{w2}", hb_p)] += 2.1

                lwi_n = [norm_tok(x, self.vocab_keep) for x in lwi_r[-3:]]
                rwi_n = [norm_tok(x, self.vocab_keep) for x in rwi_r[:3]]
                lwj_n = [norm_tok(x, self.vocab_keep) for x in lwj_r[-3:]]
                rwj_n = [norm_tok(x, self.vocab_keep) for x in rwj_r[:3]]
                for w in lwi_n:
                    vp[det_hash(f"LI_{w}", hb_p)] += 1.6
                    vp[det_hash(f"QLI_{q_mode}_{w}", hb_p)] += 1.4
                for w in rwi_n:
                    vp[det_hash(f"RI_{w}", hb_p)] += 1.4
                for w in lwj_n:
                    vp[det_hash(f"LJ_{w}", hb_p)] += 1.8
                    vp[det_hash(f"QLJ_{q_mode}_{w}", hb_p)] += 1.6
                for w in rwj_n:
                    vp[det_hash(f"RJ_{w}", hb_p)] += 1.7
                if lwi_n and lwj_n:
                    vp[det_hash(f"VPAIR_{lwi_n[-1]}_{lwj_n[-1]}", hb_p)] += 2.0
                    vp[det_hash(f"QVPAIR_{q_mode}_{lwi_n[-1]}_{lwj_n[-1]}", hb_p)] += 2.2
                if same_cl:
                    for w in lwj_n + rwj_n[:2]:
                        vp[det_hash(f"SC_{w}", hb_p)] += 2.1

                mid_raw_set = set(between_spans.get((i, j), []))
                vp[hb_p + 0] = 1.0
                vp[hb_p + 1] = float(j == i + 1) * 2.4
                vp[hb_p + 2] = float(same_cl) * 2.8
                vp[hb_p + 3] = float(adj_cl) * 1.8
                vp[hb_p + 4] = float(si == sj) * 1.5
                vp[hb_p + 5] = float(cj == len(clauses) - 1) * 2.2
                vp[hb_p + 6] = float(same_noun) * 3.2
                vp[hb_p + 7] = float(moni and monj) * 2.8
                vp[hb_p + 8] = float(moni != monj) * 2.5
                vp[hb_p + 9] = float(ratei != ratej) * 2.8
                vp[hb_p + 10] = float(ratej) * 2.6
                vp[hb_p + 11] = float(ratei) * 2.4
                vp[hb_p + 12] = float(vi > vj) * 2.0
                vp[hb_p + 13] = float(vj > vi) * 2.0
                vp[hb_p + 14] = float(vj > 1.0 and vi > 0 and abs(vi % vj) < 1e-4) * 2.6
                vp[hb_p + 15] = float(vi > 1.0 and vj > 0 and abs(vj % vi) < 1e-4) * 2.6
                vp[hb_p + 16] = float(pct_j) * 3.0
                vp[hb_p + 17] = float(pct_i) * 3.0
                vp[hb_p + 18] = float(wk_j == "DIV_WORD") * 3.2
                vp[hb_p + 19] = float(wk_j == "MUL_WORD") * 3.2
                vp[hb_p + 20] = float(wk_j == "UNIT") * 2.8
                vp[hb_p + 21] = float(bool(mid_raw_set & SUB_WORDS_V33) or any(w in SUB_WORDS_V33 for w in lwj_r[-3:])) * 2.4
                vp[hb_p + 22] = float(bool(mid_raw_set & MUL_WORDS_V33) or any(w in MUL_WORDS_V33 for w in lwj_r[-3:] + rwj_r[:3])) * 2.4
                vp[hb_p + 23] = float(bool(mid_raw_set & DIV_WORDS_V33) or any(w in DIV_WORDS_V33 for w in lwj_r[-3:] + rwj_r[:3])) * 2.4
                vp[hb_p + 24] = float(bool(mid_raw_set & ADD_WORDS_V33) or any(w in ADD_WORDS_V33 for w in lwj_r[-3:])) * 2.0
                vp[hb_p + 25] = float("of" in lwj_r[-2:] and j == i + 1) * 2.6
                vp[hb_p + 26] = float(bool((mid_raw_set | set(lwj_r + cla_j)) & {"rest", "remain", "remainder", "left"})) * 2.8
                vp[hb_p + 27] = float("than" in lwj_r or "than" in rwj_r) * 2.4
                vp[hb_p + 28] = float(exact_frac_pair) * 4.0
                vp[hb_p + 29] = min(float(lex_overlap), 3.0) * 1.8
                vp[hb_p + 30] = float(lex_overlap >= 2 and not same_cl and (ratei != ratej)) * 3.2
                vp[hb_p + 31] = float(j - i) / max(1.0, float(k - 1)) * 1.5
                vp /= max(float(np.linalg.norm(vp)), 1e-6)

        R_feat = np.zeros(self.d_root, dtype=np.float32)
        hb_r = self.d_root - 28
        for w in q_words:
            if w not in ("<ENT>", "<NNS>"):
                R_feat[det_hash(f"RQ_{w}", hb_r)] += 1.9
        for w1, w2 in zip(q_words[:-1], q_words[1:]):
            R_feat[det_hash(f"RQB_{w1}_{w2}", hb_r)] += 2.5
        R_feat[hb_r + 0] = 1.0
        R_feat[hb_r + 1] = float(q_wants_money) * 2.5
        R_feat[hb_r + 2] = float(q_mode == "DIFF") * 2.6
        R_feat[hb_r + 3] = float(q_mode == "TOT") * 2.6
        R_feat[hb_r + 4] = float(q_mode == "AVG") * 2.5
        R_feat[hb_r + 5] = float(q_mode == "PROFIT") * 3.0
        R_feat[hb_r + 6] = float("percent" in q_raw_low or "%" in q_str_last) * 2.5
        R_feat /= max(float(np.linalg.norm(R_feat)), 1e-6)

        asks_rem = bool(set(q_raw_low) & {"left", "remain", "remaining", "still", "change"})
        return ctx_sphere, U_feat, P_feat_mat, pair_idx_map, R_feat, slot_dims, q_wants_money, q_noun, asks_rem

    def _query_simvq_knn_logp(self, ctx_sphere: np.ndarray, k: int, mask_self: bool = False) -> Dict[float, float]:
        if k not in self.mem_X:
            return {}
        sims = self.mem_X[k] @ ctx_sphere
        if mask_self:
            sims = np.where(sims > 0.9995, -1e9, sims)
        top_n = min(int(self.knobs["simvq_knn_k"]), len(sims))
        if top_n <= 0:
            return {}
        top_idx = np.argpartition(-sims, top_n - 1)[:top_n]
        ts = sims[top_idx] * float(self.knobs["simvq_knn_temp"])
        ts -= np.max(ts)
        kw = np.exp(ts).astype(np.float32)
        kw /= max(float(np.sum(kw)), 1e-8)
        k_prob = 0.78 * (kw @ self.mem_Y[k][top_idx]) + 0.22 * self.arity_prior_vec[k]
        return {tb.sig: math.log(float(kp) + 1e-5) for tb, kp in zip(self.arity_pool[k], k_prob)}

    def _pack_candidates(
        self,
        q_text: str,
        nums: List[float],
        ctxs: list,
        cl: list,
        ctx_sphere: np.ndarray,
        U_f: np.ndarray,
        pair_idx_map: Dict[Tuple[int, int], int],
        slot_dims: list,
        q_wants_money: bool,
        q_noun: str,
        asks_rem: bool,
        mask_self_knn: bool = False,
    ) -> Optional[dict]:
        k = len(nums)
        knn_logp = self._query_simvq_knn_logp(ctx_sphere, k, mask_self=mask_self_knn)

        # Распределение ролей слотов для ролевого синтеза последовательности
        u_logits = -(U_f @ self.W_u.T) * 2.5
        u_logits -= np.max(u_logits, axis=-1, keepdims=True)
        r_probs = np.exp(u_logits)
        r_probs /= np.maximum(np.sum(r_probs, axis=-1, keepdims=True), 1e-8)

        seen_sigs: Dict[float, Tuple[ParsedTreeBonds, float, float, int]] = {}
        for tb in self.arity_pool.get(k, []):
            try:
                v = eval_prog_fast(tb.prog, nums)
            except Exception:
                continue
            if math.isfinite(v) and 0 < v < 1e8 and abs(v - round(v * 4) / 4.0) < 1e-3:
                seen_sigs[tb.sig] = (tb, v, 0.0, 0)

        for p_seq, bonus in synthesize_clause_sequential_programs(q_text, nums, ctxs, cl, r_probs.tolist()):
            try:
                v = eval_prog_fast(p_seq, nums)
            except Exception:
                continue
            if math.isfinite(v) and 0 < v < 1e8 and abs(v - round(v * 4) / 4.0) < 1e-3:
                sig = compute_prog_sig(p_seq, k)
                if sig is None:
                    continue
                if sig in seen_sigs:
                    otb, ov, ob, ored = seen_sigs[sig]
                    seen_sigs[sig] = (otb, ov, max(ob, float(bonus)), ored)
                else:
                    seen_sigs[sig] = (self._get_tree(p_seq, k, sig=sig, prior_logp=-6.8), v, float(bonus), 0)

        if k >= 3:
            for p_red, b_red in generate_clean_reduced_programs(q_text, nums, ctxs, cl, self.arity_pool_progs):
                try:
                    v = eval_prog_fast(p_red, nums)
                except Exception:
                    continue
                if math.isfinite(v) and 0 < v < 1e8 and abs(v - round(v * 4) / 4.0) < 1e-3:
                    sig = compute_prog_sig(p_red, k)
                    if sig is None:
                        continue
                    if sig in seen_sigs:
                        otb, ov, ob, ored = seen_sigs[sig]
                        seen_sigs[sig] = (otb, ov, max(ob, float(b_red)), ored)
                    else:
                        seen_sigs[sig] = (self._get_tree(p_red, k, sig=sig, prior_logp=-7.2), v, float(b_red), 1)

        cand_list = list(seen_sigs.values())
        M = len(cand_list)
        if M == 0:
            return None

        roles_mat = np.zeros((M, k), dtype=np.int32)
        root_ops = np.zeros(M, dtype=np.int32)
        vals = np.zeros(M, dtype=np.float64)
        sigs = np.zeros(M, dtype=np.float64)
        progs: List[Tuple[str, ...]] = []
        meta_mat = np.zeros((M, self.N_META), dtype=np.float32)
        b_m_list, b_p_list, b_op_list, b_ra_list, b_rb_list, b_wt_list = [], [], [], [], [], []
        max_x = max(nums) if nums else 1.0

        for m_i, (tb, val, c_bonus, is_red) in enumerate(cand_list):
            roles_mat[m_i] = tb.roles
            root_ops[m_i] = tb.root_op
            vals[m_i] = val
            sigs[m_i] = tb.sig
            progs.append(tb.prog)

            dim_f = eval_bytecode_dim_physics(tb.bc_ops, nums, slot_dims, q_wants_money, q_noun)
            klp = knn_logp.get(tb.sig, -9.21)

            meta_mat[m_i, 0] = float(tb.prior_logp) * 0.25
            meta_mat[m_i, 1] = float(klp) * 0.25
            meta_mat[m_i, 2] = float(c_bonus)
            meta_mat[m_i, 3] = float(is_red)
            meta_mat[m_i, 4] = float(k - tb.n_used) / float(k)
            meta_mat[m_i, 5] = 1.0 if abs(val - round(val)) > 1e-4 else 0.0
            meta_mat[m_i, 6] = 1.0 if (asks_rem and max_x >= 10 and val > max_x * 1.05) else 0.0
            meta_mat[m_i, 7] = 1.0 if tb.n_used <= 1 else 0.0
            meta_mat[m_i, 8:] = dim_f

            for a, b, bop, bwt in tb.bonds:
                b_m_list.append(m_i)
                b_p_list.append(pair_idx_map.get((a, b), 0))
                b_op_list.append(bop)
                b_ra_list.append(int(tb.roles[a]))
                b_rb_list.append(int(tb.roles[b]))
                b_wt_list.append(float(bwt))

        return {
            "M": M, "k": k, "progs": progs, "sigs": sigs, "vals": vals,
            "roles_mat": roles_mat, "root_ops": root_ops, "meta_mat": meta_mat,
            "U_f": U_f,
            "b_m": np.array(b_m_list, dtype=np.int32),
            "b_p": np.array(b_p_list, dtype=np.int32),
            "b_op": np.array(b_op_list, dtype=np.int32),
            "b_ra": np.array(b_ra_list, dtype=np.int32),
            "b_rb": np.array(b_rb_list, dtype=np.int32),
            "b_wt": np.array(b_wt_list, dtype=np.float32),
        }

    def _score_pack(self, pk: dict, P_f: np.ndarray, R_f: np.ndarray, unary_only: bool = False) -> np.ndarray:
        M, k = pk["M"], pk["k"]
        U_pot = (pk["U_f"] @ self.W_u.T) * 2.5
        u_term = U_pot[np.arange(k)[None, :], pk["roles_mat"]].mean(axis=1)
        r_term = (R_f @ self.W_r.T)[pk["root_ops"]] * 1.5
        if unary_only:
            return u_term + r_term + (pk["meta_mat"][:, :8] @ self.w_meta[:8])

        P_pot = (P_f @ self.W_p.T) * 2.5
        if len(pk["b_m"]) > 0:
            edge_e = (P_pot[pk["b_p"], pk["b_op"]] + self.K_cram[pk["b_ra"], pk["b_rb"], pk["b_op"]]) * pk["b_wt"]
            p_term = np.bincount(pk["b_m"], weights=edge_e, minlength=M)
        else:
            p_term = np.zeros(M, dtype=np.float64)
        return u_term + p_term + r_term + (pk["meta_mat"] @ self.w_meta)

    def fit(self, train_items: Sequence[Dict[str, str]]) -> Dict[str, float]:
        """Обучает сферическую память SimVQ, старт Вудбери и контрастивную энергию CRAM+z без BPTT."""
        t0 = time.perf_counter()
        df_cnt: Counter = Counter()
        for it in train_items:
            for w in {VERB_NORM_V33.get(w.lower(), w.lower()) for w in re.findall(r"[A-Za-z]+|\$", it["question"])}:
                df_cnt[w] += 1
        min_df = int(self.knobs["vocab_min_df"])
        self.vocab_keep = {w for w, c in df_cnt.items() if c >= min_df} | SUB_WORDS_V33 | MUL_WORDS_V33 | DIV_WORDS_V33 | ADD_WORDS_V33 | {"$"}

        by_arity: Dict[int, Counter] = {}
        repr_prog: Dict[Tuple[int, float], Tuple[str, ...]] = {}
        parsed_train = []

        for it in train_items:
            ns, ctxs, cl = extract_structured_quantities_and_clauses(it["question"])
            k = len(ns)
            gt = float(it["answer"].split("####")[-1].strip().replace(",", ""))
            p = abstract_trace_from_train_answer(ns, it["answer"]) if 2 <= k <= 10 else None
            sig_p = compute_prog_sig(p, k) if p is not None else None
            if sig_p is not None:
                key = (k, sig_p)
                used_cnt = sum(1 for i in range(k) if f"x{i}" in " ".join(p))
                if key not in repr_prog or used_cnt > sum(1 for i in range(k) if f"x{i}" in " ".join(repr_prog[key])):
                    repr_prog[key] = p
                by_arity.setdefault(k, Counter())[sig_p] += 1
            parsed_train.append((it["question"], ns, ctxs, cl, gt, p, sig_p))

        top_cap = int(self.knobs["top_templates_per_arity"])
        arity_sig_to_col: Dict[int, Dict[float, int]] = {}
        for k, cnt in by_arity.items():
            tot = float(sum(cnt.values()))
            top_list = cnt.most_common(top_cap)
            self.arity_pool[k] = [
                self._get_tree(repr_prog[(k, sig)], k, sig=sig, prior_logp=math.log(max(c / tot, 1e-5)))
                for sig, c in top_list
            ]
            p_vec = np.array([c / tot for _, c in top_list], dtype=np.float32)
            p_vec /= max(float(np.sum(p_vec)), 1e-8)
            self.arity_prior_vec[k] = p_vec
            self.arity_pool_progs[k] = [(repr_prog[(k, sig)], math.log(max(c / tot, 1e-5))) for sig, c in top_list[:36]]
            arity_sig_to_col[k] = {sig: col_i for col_i, (sig, _) in enumerate(top_list)}

        # 1. Наполняем сферическую память прецедентов SimVQ и собираем выборку для замкнутого старта Вудбери
        u_init_X, u_init_Y = [], []
        p_init_X, p_init_Y = [], []
        feat_cache = []
        mem_xk: Dict[int, List[np.ndarray]] = {k: [] for k in range(2, 11)}
        mem_yk: Dict[int, List[np.ndarray]] = {k: [] for k in range(2, 11)}

        for q_str, ns, ctxs, cl, gt, p_true, sig_true in parsed_train:
            k = len(ns)
            if not (2 <= k <= 10):
                continue
            ctx_s, U_f, P_f, p_map, R_f, s_dims, q_mon, q_nn, asks_rem = self._build_features(q_str, ns, ctxs, cl)
            feat_cache.append((q_str, ns, ctxs, cl, gt, p_true, sig_true, ctx_s, U_f, P_f, p_map, R_f, s_dims, q_mon, q_nn, asks_rem))

            if p_true is not None and sig_true is not None:
                if k in arity_sig_to_col and sig_true in arity_sig_to_col[k]:
                    yv = np.zeros(len(self.arity_pool[k]), dtype=np.float32)
                    yv[arity_sig_to_col[k][sig_true]] = 1.0
                    mem_xk[k].append(ctx_s)
                    mem_yk[k].append(yv)
                tb_gt = self._get_tree(p_true, k, sig=sig_true)
                for s_i in range(k):
                    u_init_X.append(U_f[s_i]); u_init_Y.append(int(tb_gt.roles[s_i]))
                for a, b, bop, _ in tb_gt.bonds:
                    if (a, b) in p_map:
                        p_init_X.append(P_f[p_map[(a, b)]]); p_init_Y.append(int(bop))

        for k in range(2, 11):
            if mem_xk[k]:
                self.mem_X[k] = np.stack(mem_xk[k], axis=0)
                self.mem_Y[k] = np.stack(mem_yk[k], axis=0)

        # 2. Замкнутый аналитический старт Вудбери для унарных ролей W_u и парных связей W_p (0 BPTT)
        lam_w = float(self.knobs["woodbury_ridge_lam"])
        UX = np.stack(u_init_X, axis=0).astype(np.float64)
        UY = np.eye(8, dtype=np.float64)[np.array(u_init_Y)]
        UY -= UY.mean(axis=0, keepdims=True)
        self.W_u = (-np.linalg.solve(UX.T @ UX + lam_w * np.eye(self.d_unary), UX.T @ UY).T).astype(np.float32)

        PX = np.stack(p_init_X, axis=0).astype(np.float64)
        PY = np.eye(7, dtype=np.float64)[np.array(p_init_Y)]
        PY -= PY.mean(axis=0, keepdims=True)
        self.W_p = (-np.linalg.solve(PX.T @ PX + lam_w * np.eye(self.d_pair), PX.T @ PY).T).astype(np.float32)

        # 3. Упаковка пулов кандидатов на train.jsonl (с маскированием self-similarity в SimVQ kNN)
        train_packs = []
        for q_str, ns, ctxs, cl, gt, _, sig_true, ctx_s, U_f, P_f, p_map, R_f, s_dims, q_mon, q_nn, asks_rem in feat_cache:
            pk = self._pack_candidates(
                q_str, ns, ctxs, cl, ctx_s, U_f, p_map, s_dims, q_mon, q_nn, asks_rem, mask_self_knn=True
            )
            if pk is None or pk["M"] < 2:
                continue
            pos_w = np.zeros(pk["M"], dtype=np.float64)
            has_exact = False
            if sig_true is not None:
                exact_mask = np.abs(pk["sigs"] - sig_true) < 1e-4
                if np.any(exact_mask):
                    pos_w[exact_mask] = 1.0
                    has_exact = True
            hits = np.abs(pk["vals"] - gt) < 1e-2
            if np.any(hits):
                for m_i in np.where(hits)[0]:
                    if pos_w[m_i] == 0.0:
                        pos_w[m_i] = 0.15 if has_exact else 1.0
            if np.sum(pos_w) == 0.0:
                continue
            pos_w /= np.sum(pos_w)
            pk["P_f"] = P_f
            pk["R_f"] = R_f
            pk["pos_w"] = pos_w.astype(np.float32)
            train_packs.append(pk)

        # 4. Контрастивная минимизация свободной энергии CRAM (Eq. 5, прямой аналитический градиент, 0 BPTT)
        params = [self.W_u, self.W_p, self.W_r, self.K_cram, self.w_meta]
        anchors = [self.W_u.copy(), self.W_p.copy(), np.zeros_like(self.W_r), np.zeros_like(self.K_cram), np.zeros_like(self.w_meta)]
        m1 = [np.zeros_like(p) for p in params]
        m2 = [np.zeros_like(p) for p in params]
        rng = np.random.default_rng(int(self.knobs["seed"]))

        epochs = int(self.knobs["cram_epochs"])
        batch_sz = int(self.knobs["cram_batch_size"])
        base_lr = float(self.knobs["cram_lr"])
        total_steps = epochs * math.ceil(len(train_packs) / batch_sz)
        step_num = 0
        final_loss = 0.0

        for ep in range(epochs):
            perm = rng.permutation(len(train_packs))
            tot_loss = 0.0
            for b_s in range(0, len(train_packs), batch_sz):
                step_num += 1
                b_ids = perm[b_s : b_s + batch_sz]
                g_Wu = np.zeros_like(self.W_u)
                g_Wp = np.zeros_like(self.W_p)
                g_Wr = np.zeros_like(self.W_r)
                g_Kc = np.zeros_like(self.K_cram)
                g_wm = np.zeros_like(self.w_meta)

                for idx_p in b_ids:
                    pk = train_packs[idx_p]
                    k = pk["k"]
                    energies = self._score_pack(pk, pk["P_f"], pk["R_f"], unary_only=False)
                    logits = -energies
                    logits -= np.max(logits)
                    pr = np.exp(logits).astype(np.float32)
                    pr /= np.sum(pr)

                    pos_w = pk["pos_w"]
                    tot_loss += -float(np.sum(pos_w * np.log(np.maximum(pr, 1e-12))))
                    dE = pos_w - pr

                    g_wm += dE @ pk["meta_mat"]
                    np.add.at(g_Wr, pk["root_ops"], (1.5 * dE)[:, None] * pk["R_f"][None, :])

                    dU_pot = np.zeros((k, 8), dtype=np.float32)
                    dE_u = (2.5 / float(k)) * dE
                    for s_i in range(k):
                        np.add.at(dU_pot[s_i], pk["roles_mat"][:, s_i], dE_u)
                    g_Wu += dU_pot.T @ pk["U_f"]

                    if len(pk["b_m"]) > 0:
                        d_edge = dE[pk["b_m"]] * pk["b_wt"]
                        np.add.at(g_Kc, (pk["b_ra"], pk["b_rb"], pk["b_op"]), d_edge)
                        dP_pot = np.zeros((pk["P_f"].shape[0], 7), dtype=np.float32)
                        np.add.at(dP_pot, (pk["b_p"], pk["b_op"]), 2.5 * d_edge)
                        g_Wp += dP_pot.T @ pk["P_f"]

                inv_b = 1.0 / float(len(b_ids))
                wds = [5e-4, 5e-4, 4e-4, 3e-4, 8e-5]
                lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * (step_num - 1) / total_steps))
                bc1, bc2 = 1.0 - 0.9 ** step_num, 1.0 - 0.999 ** step_num
                for p, p0, g, wd, mm1, mm2 in zip(params, anchors, [g_Wu, g_Wp, g_Wr, g_Kc, g_wm], wds, m1, m2):
                    g_reg = g * inv_b + wd * (p - p0)
                    mm1[:] = 0.9 * mm1 + 0.1 * g_reg
                    mm2[:] = 0.999 * mm2 + 0.001 * (g_reg * g_reg)
                    p -= lr * (mm1 / bc1) / (np.sqrt(mm2 / bc2) + 1e-8)

            final_loss = tot_loss / float(len(train_packs))
            if (ep + 1) in (1, 5, epochs):
                print(f"   [CRAM+z Train] Epoch {ep+1:2d}/{epochs} | Contrastive Free-Energy Loss = {final_loss:.4f}")

        elapsed = time.perf_counter() - t0
        n_params = sum(int(p.size) for p in params)
        return {
            "trainable_packs": float(len(train_packs)),
            "final_contrastive_loss": final_loss,
            "total_energy_params": float(n_params),
            "train_seconds": elapsed,
        }

    def solve_blind(self, question_text: str) -> Dict[str, object]:
        """Строго слепое решение задачи ТОЛЬКО по тексту вопроса `question_text`."""
        if "####" in question_text or "<<" in question_text:
            raise ValueError("Anti-Cheat Violation: answer tags detected inside blind question input!")

        ns, ctxs, cl = extract_structured_quantities_and_clauses(question_text)
        k = len(ns)
        if k == 0:
            return {
                "pred_pass1": 0.0, "pred_pass3": [0.0], "pred_pass5": [0.0],
                "pred_unary_ablation": 0.0, "best_program": (), "spherical_trace": [],
                "confidence": 0.0, "uf_abstain": 1, "extracted_nums": [], "norm_drift": 0.0,
            }
        if k == 1:
            sph_val, sph_trace, drift = self.alu.execute_program(("x0",), ns)
            return {
                "pred_pass1": sph_val, "pred_pass3": [sph_val], "pred_pass5": [sph_val],
                "pred_unary_ablation": sph_val, "best_program": ("x0",), "spherical_trace": sph_trace,
                "confidence": 0.50, "uf_abstain": 0, "extracted_nums": ns, "norm_drift": drift,
            }

        ctx_s, U_f, P_f, p_map, R_f, s_dims, q_mon, q_nn, asks_rem = self._build_features(question_text, ns, ctxs, cl)
        pk = self._pack_candidates(
            question_text, ns, ctxs, cl, ctx_s, U_f, p_map, s_dims, q_mon, q_nn, asks_rem, mask_self_knn=False
        )
        if pk is None:
            sph_val, sph_trace, drift = self.alu.execute_program(("x0",), ns)
            return {
                "pred_pass1": sph_val, "pred_pass3": [sph_val], "pred_pass5": [sph_val],
                "pred_unary_ablation": sph_val, "best_program": ("x0",), "spherical_trace": sph_trace,
                "confidence": 0.0, "uf_abstain": 1, "extracted_nums": ns, "norm_drift": drift,
            }

        energies = self._score_pack(pk, P_f, R_f, unary_only=False)
        energies_unary = self._score_pack(pk, P_f, R_f, unary_only=True)
        beta = float(self.knobs["boltzmann_beta"])

        def marginalize_boltzmann(e_arr: np.ndarray) -> Tuple[List[Tuple[float, float]], Dict[float, Tuple[str, ...]]]:
            es = beta * (e_arr - np.min(e_arr))
            bw = np.exp(-np.minimum(es, 60.0))
            bw /= max(float(np.sum(bw)), 1e-12)
            val_prob: Dict[float, float] = {}
            val_best_e: Dict[float, float] = {}
            val_prog: Dict[float, Tuple[str, ...]] = {}
            for m_i, (vv, wv, ev) in enumerate(zip(pk["vals"], bw, e_arr)):
                kv = round(float(vv), 2)
                val_prob[kv] = val_prob.get(kv, 0.0) + float(wv)
                if kv not in val_best_e or float(ev) < val_best_e[kv]:
                    val_best_e[kv] = float(ev)
                    val_prog[kv] = pk["progs"][m_i]
            ranked = sorted(val_prob.items(), key=lambda x: x[1], reverse=True)
            return ranked, val_prog

        ranked_cram, prog_map_cram = marginalize_boltzmann(energies)
        ranked_unary, prog_map_unary = marginalize_boltzmann(energies_unary)

        # Пошагово прогоняем топ-5 формул на 12-разрядном сферическом АЛУ (S^14)^13
        top5_progs = [prog_map_cram[kv] for kv, _ in ranked_cram[:5]]
        top5_sph_vals: List[float] = []
        best_trace: List[str] = []
        max_drift = 0.0
        for rank_i, p_cand in enumerate(top5_progs):
            sv, st_trace, dr = self.alu.execute_program(p_cand, ns)
            max_drift = max(max_drift, dr)
            top5_sph_vals.append(sv)
            if rank_i == 0:
                best_trace = st_trace

        best_conf = float(ranked_cram[0][1]) if ranked_cram else 0.0
        uf_abstain = 1 if best_conf < float(self.knobs["uf_confidence_threshold"]) else 0

        unary_val = ranked_unary[0][0] if ranked_unary else ns[0]
        if ranked_unary:
            unary_val, _, dr_u = self.alu.execute_program(prog_map_unary[ranked_unary[0][0]], ns)
            max_drift = max(max_drift, dr_u)

        return {
            "pred_pass1": top5_sph_vals[0],
            "pred_pass3": top5_sph_vals[:3],
            "pred_pass5": top5_sph_vals[:5],
            "pred_unary_ablation": unary_val,
            "best_program": top5_progs[0],
            "spherical_trace": best_trace,
            "confidence": best_conf,
            "uf_abstain": uf_abstain,
            "extracted_nums": ns,
            "norm_drift": max_drift,
        }


# ==============================================================================
# 6. ЗАГРУЗКА ДАННЫХ И СТРОГО СЛЕПОЙ ПРОГОН НА OFFICIAL GSM8K TEST (N = 1,319)
# ==============================================================================
def locate_gsm8k_files() -> Tuple[str, str]:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    candidates = [
        ("benchmarks/gsm8k/train.jsonl", "benchmarks/gsm8k/test.jsonl"),
        (
            os.path.join(repo_root, "benchmarks", "gsm8k", "train.jsonl"),
            os.path.join(repo_root, "benchmarks", "gsm8k", "test.jsonl"),
        ),
        (
            "/tmp/simvq_extracted/simvq-split-9e60c290351efa2d3972d50df341d05edfea02e2/benchmarks/gsm8k/train.jsonl",
            "/tmp/simvq_extracted/simvq-split-9e60c290351efa2d3972d50df341d05edfea02e2/benchmarks/gsm8k/test.jsonl",
        ),
    ]
    for tr_p, te_p in candidates:
        if os.path.exists(tr_p) and os.path.exists(te_p):
            return tr_p, te_p
    raise FileNotFoundError("Не найдены файлы benchmarks/gsm8k/{train,test}.jsonl.")


def run_blind_gsm8k_experiment() -> Dict[str, object]:
    train_path, test_path = locate_gsm8k_files()
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = [json.loads(line) for line in f]
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]

    print("=" * 78)
    print("GSM8K BLIND BENCHMARK: CRAM+z FORMULA ASSEMBLY + SPHERICAL BASE-10 ALU")
    print(f"Train split: {len(train_data)} problems | Official Blind Test split: {len(test_data)} problems")
    print("=" * 78)

    alu = SphericalBase10FixedPointALU(d_sub=15, num_digits=12, seed=3301)
    alu_diag = alu.verify_crystal_invariants()
    print(
        f"[1/3] Spherical Crystal ALU Verified: ADD={alu_diag['add_200_acc_pct']:.2f}% (200/200) | "
        f"SUB={alu_diag['sub_200_acc_pct']:.2f}% (200/200) | Codebook Norm Drift={alu_diag['codebook_norm_drift']:.2e}"
    )

    solver = CramGSM8KSolver(alu=alu, knobs=TUNING_KNOBS)
    print("[2/3] Training SimVQ Precedent Memory + Woodbury Warm-Start + Contrastive CRAM+z (0 BPTT)...")
    fit_stats = solver.fit(train_data)
    print(
        f"      Done in {fit_stats['train_seconds']:.1f}s | Trainable packs={int(fit_stats['trainable_packs'])} | "
        f"Energy params={int(fit_stats['total_energy_params']):,}"
    )

    # СТРОГИЙ АНТИ-ЧИТ ШЛЮЗ: полностью отделяем вопросы от ответов до запуска solve_blind!
    blind_questions: List[str] = [str(item["question"]) for item in test_data]
    ground_truths: List[Tuple[float, int]] = []
    for item in test_data:
        ans_raw = str(item["answer"])
        gt_v = float(ans_raw.split("####")[-1].strip().replace(",", ""))
        n_steps = max(1, len(re.findall(r"<<[^>]+>>", ans_raw)))
        ground_truths.append((gt_v, n_steps))

    print(f"[3/3] Running strict blind evaluation on {len(blind_questions)} test questions (0 answer access)...")
    t_eval = time.perf_counter()

    p1_ok = p3_ok = p5_ok = unary_ok = 0
    conf_tot = conf_p1 = conf_p3 = 0
    max_test_drift = 0.0
    by_steps: Dict[str, List[int]] = {
        "2_steps": [0, 0, 0, 0],     # [total, pass1, pass3, pass5]
        "3_steps": [0, 0, 0, 0],
        "4plus_steps": [0, 0, 0, 0],
    }
    solved_showcase: List[Dict[str, object]] = []

    for idx_q, q_only in enumerate(blind_questions):
        out = solver.solve_blind(q_only)
        gt_val, n_st = ground_truths[idx_q]
        max_test_drift = max(max_test_drift, float(out["norm_drift"]))

        hit1 = abs(float(out["pred_pass1"]) - gt_val) < 1e-2
        hit3 = any(abs(float(v) - gt_val) < 1e-2 for v in out["pred_pass3"])
        hit5 = any(abs(float(v) - gt_val) < 1e-2 for v in out["pred_pass5"])
        hit_u = abs(float(out["pred_unary_ablation"]) - gt_val) < 1e-2

        p1_ok += int(hit1)
        p3_ok += int(hit3)
        p5_ok += int(hit5)
        unary_ok += int(hit_u)

        if int(out["uf_abstain"]) == 0:
            conf_tot += 1
            conf_p1 += int(hit1)
            conf_p3 += int(hit3)

        bucket = "2_steps" if n_st <= 2 else ("3_steps" if n_st == 3 else "4plus_steps")
        by_steps[bucket][0] += 1
        by_steps[bucket][1] += int(hit1)
        by_steps[bucket][2] += int(hit3)
        by_steps[bucket][3] += int(hit5)

        if hit1 and not hit_u and len(out["extracted_nums"]) >= 3 and len(solved_showcase) < 6:
            solved_showcase.append({
                "idx": idx_q,
                "question": q_only,
                "nums": out["extracted_nums"],
                "program": out["best_program"],
                "trace": out["spherical_trace"],
                "pred": out["pred_pass1"],
                "gt": gt_val,
                "unary_pred": out["pred_unary_ablation"],
            })

    eval_sec = time.perf_counter() - t_eval
    n_test = len(blind_questions)
    p1_pct = 100.0 * p1_ok / n_test
    p3_pct = 100.0 * p3_ok / n_test
    p5_pct = 100.0 * p5_ok / n_test
    u_pct = 100.0 * unary_ok / n_test
    ci1 = wilson_95_ci(p1_ok, n_test)
    ci3 = wilson_95_ci(p3_ok, n_test)
    ci5 = wilson_95_ci(p5_ok, n_test)

    print("\n" + "=" * 78)
    print(f"OFFICIAL GSM8K BLIND TEST RESULTS (N = {n_test}, evaluated in {eval_sec:.2f}s)")
    print("=" * 78)
    print(f"  Direct Neural MLP Regressor (v33 ckpt):   16/{n_test} ( 1.21%) [CI: 0.75%..1.96%]")
    print(f"  NeuroSymbolicGSM8KSolver_v33 (baseline): 170/{n_test} (12.89%) | pass@3: 278 (21.08%) | pass@5: 351 (26.61%)")
    print(f"  Unary-Only Ablation (no CRAM pair bonds):{unary_ok:3d}/{n_test} ({u_pct:5.2f}%)")
    print(
        f"  CRAM+z + Spherical Base-10 ALU (OURS):   {p1_ok:3d}/{n_test} ({p1_pct:5.2f}%) [95% CI: {ci1[0]:.2f}%..{ci1[1]:.2f}%]\n"
        f"                                           pass@3 = {p3_ok}/{n_test} ({p3_pct:.2f}%) [95% CI: {ci3[0]:.2f}%..{ci3[1]:.2f}%]\n"
        f"                                           pass@5 = {p5_ok}/{n_test} ({p5_pct:.2f}%) [95% CI: {ci5[0]:.2f}%..{ci5[1]:.2f}%]"
    )
    if conf_tot > 0:
        print(
            f"  Confident Subset (UF=0, conf >= {TUNING_KNOBS['uf_confidence_threshold']}):  "
            f"pass@1 = {conf_p1}/{conf_tot} ({100.0*conf_p1/conf_tot:.2f}%) | "
            f"pass@3 = {conf_p3}/{conf_tot} ({100.0*conf_p3/conf_tot:.2f}%) "
            f"[coverage {conf_tot}/{n_test} = {100.0*conf_tot/n_test:.1f}%]"
        )
    print(f"  Max Spherical Norm Drift across all test registers: {max_test_drift:.2e}")
    print("\nBreakdown by Reasoning Steps:")
    for b_key, (btot, bp1, bp3, bp5) in by_steps.items():
        if btot > 0:
            print(
                f"  - {b_key:11s} (N={btot:3d}): pass@1 = {bp1:3d}/{btot} ({100.0*bp1/btot:5.2f}%) | "
                f"pass@3 = {100.0*bp3/btot:5.2f}% | pass@5 = {100.0*bp5/btot:5.2f}%"
            )

    print("\nExamples where CRAM+z Pairwise Energy fixed Unary-Only failure on Blind Test:")
    for ex in solved_showcase[:4]:
        print(f"  [Test #{ex['idx']}] Q: {ex['question']}")
        print(f"     Extracted slots: {ex['nums']} | Unary-only guessed: {ex['unary_pred']} (WRONG)")
        print(f"     CRAM+z assembled: {ex['program']} -> Trace: {' ; '.join(ex['trace'])} => {ex['pred']} (GT={ex['gt']})")

    return {
        "n_test": n_test,
        "pass1": p1_ok, "pass1_pct": p1_pct, "pass1_ci": ci1,
        "pass3": p3_ok, "pass3_pct": p3_pct, "pass3_ci": ci3,
        "pass5": p5_ok, "pass5_pct": p5_pct, "pass5_ci": ci5,
        "unary_ablation_ok": unary_ok, "unary_ablation_pct": u_pct,
        "conf_total": conf_tot,
        "conf_pass1": conf_p1,
        "conf_pass1_pct": (100.0 * conf_p1 / conf_tot) if conf_tot else 0.0,
        "conf_pass3_pct": (100.0 * conf_p3 / conf_tot) if conf_tot else 0.0,
        "max_norm_drift": max_test_drift,
        "by_steps": by_steps,
    }


if __name__ == "__main__":
    run_blind_gsm8k_experiment()
