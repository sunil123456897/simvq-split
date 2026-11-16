"""
spherical_alu.py — Чисто-сферический 12-разрядный регистровый АЛУ базы 10
на декартовом произведении `(S^14)^12 x S^14_sign` (`SCALE = 10,000`).

Архитектурный инвариант Фазы C:
  - Никакого `re`, `ast`, `eval`, `exec` и никакого отката в Python `float`
    внутри арифметических операций (`add`, `sub`, `mul`, `div`) и исполнения DAG.
  - Все 4 операции выполняются поразрядными таблицами истинности (carry/borrow,
    10x10 база-10 таблица умножения + сдвиг-сложение в 24-разрядный буфер,
    поразрядное деление столбиком с восстановлением остатка за 16 шагов)
    непосредственно над сферическими кодовыми книгами цифр `0..9` и знака `+/-`.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Sequence, Tuple

import numpy as np


class SphericalBase10RegisterALU:
    """Точное 12-разрядное АЛУ базы 10 (`SCALE = 10_000`, диапазон `[-99,999,999.9999, +99,999,999.9999]`)
    на `(S^14)^12 x S^14_sign` в `float32`.
    """

    SCALE: int = 10_000
    FRAC_DIGITS: int = 4
    NUM_DIGITS: int = 12
    OP_NAMES: Tuple[str, ...] = ("+", "-", "*", "/")

    def __init__(self, d_sub: int = 15, num_digits: int = 12, seed: int = 3301):
        self.d_sub = int(d_sub)
        self.num_digits = int(num_digits)
        rng = np.random.default_rng(seed)

        # Ортонормированный базис цифр 0..9 на единичной сфере S^{d_sub - 1}
        raw_q, _ = np.linalg.qr(rng.standard_normal((self.d_sub, self.d_sub)))
        d_cb = raw_q[:10].astype(np.float32)
        d_cb /= np.linalg.norm(d_cb, axis=-1, keepdims=True)
        self.digit_codebook: np.ndarray = d_cb  # (10, d_sub)

        # Антиподальная пара знака (+/-) на S^{d_sub - 1}
        s_vec = raw_q[10:11].astype(np.float32)
        s_vec /= np.linalg.norm(s_vec, axis=-1, keepdims=True)
        self.sign_codebook: np.ndarray = np.concatenate([s_vec, -s_vec], axis=0)  # (2, d_sub)

        # Таблицы поразрядного сложения/вычитания (carry/borrow in {0, 1}, a in 0..9, b in 0..9)
        add_d = np.zeros((2, 10, 10), dtype=np.int32)
        add_c = np.zeros((2, 10, 10), dtype=np.int32)
        sub_d = np.zeros((2, 10, 10), dtype=np.int32)
        sub_b = np.zeros((2, 10, 10), dtype=np.int32)
        for c in range(2):
            for a in range(10):
                for b in range(10):
                    s_val = a + b + c
                    add_d[c, a, b] = s_val % 10
                    add_c[c, a, b] = s_val // 10
                    d_val = a - b - c
                    if d_val < 0:
                        sub_d[c, a, b] = d_val + 10
                        sub_b[c, a, b] = 1
                    else:
                        sub_d[c, a, b] = d_val
                        sub_b[c, a, b] = 0
        self.add_digit_lut = add_d
        self.add_carry_lut = add_c
        self.sub_digit_lut = sub_d
        self.sub_borrow_lut = sub_b

        # Таблицы умножения цифр 0..9 x 0..9 (единицы и перенос в старший разряд)
        mul_p = np.zeros((10, 10), dtype=np.int32)
        mul_c = np.zeros((10, 10), dtype=np.int32)
        for a in range(10):
            for b in range(10):
                p_val = a * b
                mul_p[a, b] = p_val % 10
                mul_c[a, b] = p_val // 10
        self.mul_prod_lut = mul_p
        self.mul_carry_lut = mul_c

        # Знаковая XOR-таблица: 0=+ , 1=-
        self.sign_xor_lut = np.array([[0, 1], [1, 0]], dtype=np.int32)
        self.round_half_lut = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32)
        self.max_digits = np.full(self.num_digits, 9, dtype=np.int32)

    # ------------------------------------------------------------------
    # Кодирование/декодирование только на границах сферического конвейера
    # ------------------------------------------------------------------
    def _assemble_register(self, digits: np.ndarray, sign_idx: int) -> np.ndarray:
        reg = np.empty((self.num_digits + 1, self.d_sub), dtype=np.float32)
        reg[: self.num_digits] = self.digit_codebook[digits]
        reg[self.num_digits] = self.sign_codebook[sign_idx]
        norms = np.linalg.norm(reg, axis=-1, keepdims=True)
        reg /= np.maximum(norms, 1e-12)
        return reg

    def _read_spheres(self, reg: np.ndarray) -> Tuple[np.ndarray, int, float]:
        norms = np.linalg.norm(reg, axis=-1)
        drift = float(np.max(np.abs(norms - 1.0)))
        digits = np.argmax(reg[: self.num_digits] @ self.digit_codebook.T, axis=-1).astype(np.int32)
        sign_idx = int(np.argmax(reg[self.num_digits] @ self.sign_codebook.T))
        return digits, sign_idx, drift

    def encode(self, val: float | int | str | Decimal) -> np.ndarray:
        """Кодирует число на единичные сферы `(S^14)^12 x S^14_sign` с точностью 1e-4."""
        if isinstance(val, Decimal):
            dec = val
        elif isinstance(val, (int, str)):
            dec = Decimal(str(val))
        else:
            if not np.isfinite(val):
                return self._assemble_register(np.zeros(self.num_digits, dtype=np.int32), 0)
            dec = Decimal(repr(float(val)))

        sign_idx = 1 if dec < 0 else 0
        dec_abs = -dec if dec < 0 else dec
        scaled = (dec_abs * Decimal(self.SCALE)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        max_scaled = Decimal(10 ** self.num_digits - 1)
        if scaled > max_scaled:
            scaled = max_scaled
        s_str = str(int(scaled)).zfill(self.num_digits)[-self.num_digits :]
        digits = np.array([int(ch) for ch in reversed(s_str)], dtype=np.int32)
        if not np.any(digits):
            sign_idx = 0
        return self._assemble_register(digits, sign_idx)

    def decode_with_drift(self, reg: np.ndarray) -> Tuple[float, float]:
        """Декодирует финальный сферический регистр после вычислений и возвращает `(value, norm_drift)`."""
        digits, sign_idx, drift = self._read_spheres(reg)
        val_int = 0
        for p in range(self.num_digits - 1, -1, -1):
            val_int = val_int * 10 + int(digits[p])
        val = val_int / float(self.SCALE)
        if sign_idx == 1 and val_int > 0:
            val = -val
        return float(val), drift

    def format_register(self, reg: np.ndarray) -> str:
        digits, sign_idx, _ = self._read_spheres(reg)
        frac_s = "".join(str(int(digits[p])) for p in range(self.FRAC_DIGITS - 1, -1, -1))
        int_s = "".join(str(int(digits[p])) for p in range(self.num_digits - 1, self.FRAC_DIGITS - 1, -1)).lstrip("0") or "0"
        prefix = "-" if (sign_idx == 1 and int(np.any(digits))) else ""
        return f"{prefix}{int_s}.{frac_s}"

    # ------------------------------------------------------------------
    # Поразрядная табличная логика произвольной ширины n (без float!)
    # ------------------------------------------------------------------
    def _cmp_n(self, da: np.ndarray, db: np.ndarray) -> int:
        n = len(da)
        for p in range(n - 1, -1, -1):
            va, vb = int(da[p]), int(db[p])
            if va > vb:
                return 1
            if va < vb:
                return -1
        return 0

    def _add_n(self, da: np.ndarray, db: np.ndarray) -> Tuple[np.ndarray, int]:
        n = len(da)
        out = np.empty(n, dtype=np.int32)
        c = 0
        add_d = self.add_digit_lut
        add_c = self.add_carry_lut
        for p in range(n):
            a_p = int(da[p])
            b_p = int(db[p])
            out[p] = add_d[c, a_p, b_p]
            c = int(add_c[c, a_p, b_p])
        return out, c

    def _add_digit_at(self, buf: np.ndarray, pos: int, dig: int) -> None:
        if dig == 0:
            return
        n = len(buf)
        add_d = self.add_digit_lut
        add_c = self.add_carry_lut
        cur = int(buf[pos])
        buf[pos] = add_d[0, cur, dig]
        c = int(add_c[0, cur, dig])
        k = pos + 1
        while c > 0 and k < n:
            cur_k = int(buf[k])
            buf[k] = add_d[c, cur_k, 0]
            c = int(add_c[c, cur_k, 0])
            k += 1

    def _sub_n(self, big: np.ndarray, small: np.ndarray) -> np.ndarray:
        n = len(big)
        out = np.empty(n, dtype=np.int32)
        b = 0
        sub_d = self.sub_digit_lut
        sub_b = self.sub_borrow_lut
        for p in range(n):
            a_p = int(big[p])
            s_p = int(small[p])
            out[p] = sub_d[b, a_p, s_p]
            b = int(sub_b[b, a_p, s_p])
        return out

    def _inc_n(self, buf: np.ndarray, start_pos: int) -> int:
        n = len(buf)
        add_d = self.add_digit_lut
        add_c = self.add_carry_lut
        c = 1
        for p in range(start_pos, n):
            if c == 0:
                break
            cur_p = int(buf[p])
            buf[p] = add_d[c, cur_p, 0]
            c = int(add_c[c, cur_p, 0])
        return c

    # ------------------------------------------------------------------
    # Четыре чистые сферические операции (+, -, *, /) на (13, 15) сферах
    # ------------------------------------------------------------------
    def add_spherical(self, ra: np.ndarray, rb: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        da, sa, dr_a = self._read_spheres(ra)
        db, sb, dr_b = self._read_spheres(rb)
        overflow = False
        if sa == sb:
            out_d, carry = self._add_n(da, db)
            out_s = sa
            if carry > 0:
                overflow = True
                out_d = self.max_digits.copy()
        else:
            cmp_v = self._cmp_n(da, db)
            if cmp_v >= 0:
                out_d = self._sub_n(da, db)
                out_s = sa if cmp_v > 0 else 0
            else:
                out_d = self._sub_n(db, da)
                out_s = sb
        if not np.any(out_d):
            out_s = 0
        out_reg = self._assemble_register(out_d, out_s)
        _, _, dr_o = self._read_spheres(out_reg)
        return out_reg, max(dr_a, dr_b, dr_o), overflow

    def sub_spherical(self, ra: np.ndarray, rb: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        rb_neg = rb.copy()
        rb_neg[self.num_digits] = -rb_neg[self.num_digits]
        return self.add_spherical(ra, rb_neg)

    def mul_spherical(self, ra: np.ndarray, rb: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        da, sa, dr_a = self._read_spheres(ra)
        db, sb, dr_b = self._read_spheres(rb)
        n = self.num_digits
        acc = np.zeros(2 * n, dtype=np.int32)
        mul_p = self.mul_prod_lut
        mul_c = self.mul_carry_lut

        for i in range(n):
            ai = int(da[i])
            if ai == 0:
                continue
            for j in range(n):
                bj = int(db[j])
                if bj == 0:
                    continue
                self._add_digit_at(acc, i + j, int(mul_p[ai, bj]))
                self._add_digit_at(acc, i + j + 1, int(mul_c[ai, bj]))

        # Сдвиг на FRAC_DIGITS=4 десятичных разряда вправо и round-half-up по acc[3]
        if self.round_half_lut[int(acc[self.FRAC_DIGITS - 1])] == 1:
            self._inc_n(acc, self.FRAC_DIGITS)

        overflow = bool(np.any(acc[self.FRAC_DIGITS + n :] > 0))
        if overflow:
            out_d = self.max_digits.copy()
        else:
            out_d = acc[self.FRAC_DIGITS : self.FRAC_DIGITS + n].copy()

        out_s = int(self.sign_xor_lut[sa, sb]) if np.any(out_d) else 0
        out_reg = self._assemble_register(out_d, out_s)
        _, _, dr_o = self._read_spheres(out_reg)
        return out_reg, max(dr_a, dr_b, dr_o), overflow

    def div_spherical(self, ra: np.ndarray, rb: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        da, sa, dr_a = self._read_spheres(ra)
        db, sb, dr_b = self._read_spheres(rb)
        n = self.num_digits
        if not np.any(db):
            zero_reg = self._assemble_register(np.zeros(n, dtype=np.int32), 0)
            return zero_reg, max(dr_a, dr_b), True

        # Масштабированное делимое (A * SCALE) * SCALE -> 16 десятичных разрядов
        n_steps = n + self.FRAC_DIGITS
        dividend = np.zeros(n_steps, dtype=np.int32)
        dividend[self.FRAC_DIGITS :] = da

        # Таблица кратных делителя k * db для k in 0..9 (ширина n + 1 = 13 разрядов)
        w = n + 1
        db_w = np.zeros(w, dtype=np.int32)
        db_w[:n] = db
        mults = np.zeros((10, w), dtype=np.int32)
        for k in range(1, 10):
            mults[k], _ = self._add_n(mults[k - 1], db_w)

        quot = np.zeros(n_steps, dtype=np.int32)
        rem = np.zeros(w, dtype=np.int32)

        for pos in range(n_steps - 1, -1, -1):
            rem[1:] = rem[:-1]
            rem[0] = dividend[pos]
            q_dig = 0
            for cand in range(9, 0, -1):
                if self._cmp_n(rem, mults[cand]) >= 0:
                    q_dig = cand
                    rem = self._sub_n(rem, mults[cand])
                    break
            quot[pos] = q_dig

        # Округление младшего разряда: 2 * rem >= db
        twice_rem = np.zeros(w + 1, dtype=np.int32)
        sum_r, carry_r = self._add_n(rem, rem)
        twice_rem[:w] = sum_r
        twice_rem[w] = carry_r
        db_w1 = np.zeros(w + 1, dtype=np.int32)
        db_w1[:n] = db
        if self._cmp_n(twice_rem, db_w1) >= 0:
            self._inc_n(quot, 0)

        overflow = bool(np.any(quot[n:] > 0))
        if overflow:
            out_d = self.max_digits.copy()
        else:
            out_d = quot[:n].copy()

        out_s = int(self.sign_xor_lut[sa, sb]) if np.any(out_d) else 0
        out_reg = self._assemble_register(out_d, out_s)
        _, _, dr_o = self._read_spheres(out_reg)
        return out_reg, max(dr_a, dr_b, dr_o), overflow

    # ------------------------------------------------------------------
    # Исполнение графа бинарных шагов прямо на сферических регистрах
    # ------------------------------------------------------------------
    def execute_dag_spherical(
        self,
        slot_regs: Sequence[np.ndarray],
        steps: Sequence[Tuple[int, int, int]],
        direct_idx: int = 0,
    ) -> Tuple[np.ndarray, List[np.ndarray], float, bool]:
        """Исполняет DAG `[(op_id, a_idx, b_idx), ...]` строго в пространстве `(13, 15)` сфер.
        Возвращает `(final_reg, all_regs, max_norm_drift, invalid_flag)`."""
        regs: List[np.ndarray] = [r.copy() for r in slot_regs]
        max_drift = 0.0
        for r in regs:
            _, _, dr = self._read_spheres(r)
            if dr > max_drift:
                max_drift = dr

        if not steps:
            if not regs:
                zero_reg = self._assemble_register(np.zeros(self.num_digits, dtype=np.int32), 0)
                return zero_reg, [zero_reg], max_drift, True
            idx = max(0, min(int(direct_idx), len(regs) - 1))
            return regs[idx], regs, max_drift, False

        invalid = False
        for op_id, a_idx, b_idx in steps:
            if a_idx < 0 or b_idx < 0 or a_idx >= len(regs) or b_idx >= len(regs):
                zero_reg = self._assemble_register(np.zeros(self.num_digits, dtype=np.int32), 0)
                return zero_reg, regs, max_drift, True
            ra, rb = regs[a_idx], regs[b_idx]
            if op_id == 0:
                ro, dr, err = self.add_spherical(ra, rb)
            elif op_id == 1:
                ro, dr, err = self.sub_spherical(ra, rb)
            elif op_id == 2:
                ro, dr, err = self.mul_spherical(ra, rb)
            elif op_id == 3:
                ro, dr, err = self.div_spherical(ra, rb)
            else:
                zero_reg = self._assemble_register(np.zeros(self.num_digits, dtype=np.int32), 0)
                return zero_reg, regs, max_drift, True
            if dr > max_drift:
                max_drift = dr
            if err:
                invalid = True
            regs.append(ro)

        return regs[-1], regs, max_drift, invalid

    def register_geometric_features(self, reg: np.ndarray) -> np.ndarray:
        """Снимает чисто геометрические проекции сферического регистра на базис `0` и знак `+`:
        - sign_pos: скалярное произведение знаковой сферы с `+` (+1 или -1),
        - frac_zero_res: среднее совпадение 4 дробных разрядов со сферой нуля (1.0 -> целое число),
        - frac_last2_res: среднее совпадение сотых/десятых долей со сферой нуля,
        - int_active_span: доля ненулевых целых разрядов среди 8 позиций (прокси десятичного масштаба),
        - zero_frac_digits: 4 бинарных сферических совпадения для `10^-4..10^-1`.
        """
        zero_vec = self.digit_codebook[0]
        plus_vec = self.sign_codebook[0]
        frac_dots = reg[: self.FRAC_DIGITS] @ zero_vec
        int_dots = reg[self.FRAC_DIGITS : self.num_digits] @ zero_vec
        sign_dot = float(reg[self.num_digits] @ plus_vec)
        frac_zero_res = float(np.mean(frac_dots > 0.8))
        frac_low_res = float(np.mean(frac_dots[:2] > 0.8))
        int_nonzero_frac = float(np.mean(int_dots < 0.8))
        is_exact_zero = float(bool(np.all(frac_dots > 0.8) and np.all(int_dots > 0.8)))
        return np.array(
            [sign_dot, frac_zero_res, frac_low_res, int_nonzero_frac, is_exact_zero],
            dtype=np.float32,
        )

    def verify_crystal_invariants(self, n_pairs: int = 200) -> Dict[str, float]:
        """Автономная самопроверка сферической алгебры (ADD/SUB/MUL/DIV + дрейф нормы)."""
        rng = np.random.default_rng(777)
        c_norms = np.linalg.norm(self.digit_codebook, axis=-1)
        s_norms = np.linalg.norm(self.sign_codebook, axis=-1)
        codebook_drift = float(max(np.max(np.abs(c_norms - 1.0)), np.max(np.abs(s_norms - 1.0))))

        hits = {"add": 0, "sub": 0, "mul": 0, "div": 0}
        max_drift = codebook_drift
        for _ in range(n_pairs):
            a = round(float(rng.uniform(-400.0, 400.0)), 2)
            b = round(float(rng.uniform(-400.0, 400.0)), 2)
            if abs(b) < 0.5:
                b = 1.25
            ra = self.encode(a)
            rb = self.encode(b)

            r_add, d1, _ = self.add_spherical(ra, rb)
            v_add, d1b = self.decode_with_drift(r_add)
            hits["add"] += int(abs(v_add - round(a + b, 4)) <= 1e-4)

            r_sub, d2, _ = self.sub_spherical(ra, rb)
            v_sub, d2b = self.decode_with_drift(r_sub)
            hits["sub"] += int(abs(v_sub - round(a - b, 4)) <= 1e-4)

            r_mul, d3, _ = self.mul_spherical(ra, rb)
            v_mul, d3b = self.decode_with_drift(r_mul)
            hits["mul"] += int(abs(v_mul - round(a * b, 4)) <= 2e-4)

            r_div, d4, _ = self.div_spherical(ra, rb)
            v_div, d4b = self.decode_with_drift(r_div)
            hits["div"] += int(abs(v_div - round(a / b, 4)) <= 2e-4)

            max_drift = max(max_drift, d1, d1b, d2, d2b, d3, d3b, d4, d4b)

        return {
            "add_acc_pct": 100.0 * hits["add"] / n_pairs,
            "sub_acc_pct": 100.0 * hits["sub"] / n_pairs,
            "mul_acc_pct": 100.0 * hits["mul"] / n_pairs,
            "div_acc_pct": 100.0 * hits["div"] / n_pairs,
            "max_norm_drift": max_drift,
        }


SphericalBase10FixedPointALU = SphericalBase10RegisterALU
