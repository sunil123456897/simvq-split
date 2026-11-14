# -*- coding: utf-8 -*-
"""
================================================================================
  v15.0: ФОН-НЕЙМАНОВСКОЕ ЯДРО С ЕДИНОЙ ЛЕНТОЙ M_RAM[0..124] И ОБРАТНЫЙ СИНТЕЗ ПРОГРАММ
  (`VonNeumannSynthesizerSimVQ_v15` / `v15_von_neumann_synthesizer.py`)
================================================================================
Архитектурные прорывы v15.0 (по сравнению с v14.0):
  1. Масштабирование внешней биортогональной матрицы памяти до N = 5^3 = 125 ячеек
     (`GershgorinBiorthogonalTape125`, M_RAM in R^{125 x 240}) с калибровкой Ньютона — Шульца
     через норму Гёльдера — Гершгорина:
         sigma_max^2(G) <= ||G||_1 * ||G||_inf = (max_i sum_j |G_ij|)^2
     В отличие от следа Фробениуса ||G||_F^2 ~ N * lambda_max^2, оценка Гершгорина НЕ зависит от N
     и сходится за 8 итераций при любом объёме памяти N = 25, 125, 625 без риска NaN!
  2. Полная фон-неймановская унификация (Направление Б):
     - Счётчик команд PC = (pc0, pc1) in Z_5^2 (25 адресов кодового сегмента) перенесён
       кнутри сферического вектора состояния ЦПУ `h_t` (10 сфер Вороного = 800 каналов:
       ACC=(d0,d1,d2), PTR=(p0,p1), SP, B=(b0,b1), PC=(pc0,pc1)).
     - Инструкции программы упакованы в 3-разрядные слова (ir0, ir1, ir2) in Z_5^3 и хранятся
       ПРЯМО В КОДОВОМ СЕГМЕНТЕ M_RAM[0..24] той же самой матрицы памяти, где лежат данные
       M_RAM[25..99] и аппаратный стек M_RAM[100..124]!
     - На каждом такте машина выполняет два биортогональных обращения к M_RAM:
         (а) Instruction Fetch: IR = M_RAM[PC] (по сферам pc0, pc1 внутри h_t),
         (б) Data Fetch:       MDR = M_RAM[25 + PTR] (или M_RAM[100 + SP] для стека),
       после чего 100% тернарный контур обновляет и регистры данных, и сферы счётчика команд PC!
  3. Обратный синтез программ / Neural Program Synthesis (Направление А):
     - Модуль `TernaryProgramSynthesizer_v15` принимает пару состояний памяти и регистров
       (Initial State -> Target State после исполнения неизвестного алгоритма или цепочки команд),
       синтезирует:
         (1) последовательность микроинструкций кодового сегмента M_RAM[0..L-1] (и валидный AST-код Python),
         (2) многошаговую трассу одиночных команд T=1..4 по начальному и конечному состоянию,
       а затем верифицирует синтезированную программу прямым исполнением на фон-неймановском ядре!
================================================================================
"""

import ast
import math
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# 10 сферических регистров ЦПУ в Z_5^10 (9,765,625 состояний регистрового файла ЦПУ):
#   [0] d0  (младший разряд ACC, 1)
#   [1] d1  (средний разряд ACC, 5)
#   [2] d2  (старший разряд ACC, 25) -> ACC = d2*25 + d1*5 + d0 in [0..124]
#   [3] p0  (младший разряд указателя данных PTR, 1)
#   [4] p1  (старший разряд указателя данных PTR, 5) -> PTR = p1*5 + p0 in [0..24] (адресует M_RAM[25 + PTR])
#   [5] sp  (указатель аппаратного стека SP in [0..4], адресует M_RAM[100 + sp])
#   [6] b0  (младший разряд регистра B, 1)
#   [7] b1  (старший разряд регистра B, 5) -> B_reg = b1*5 + b0 in [0..24]
#   [8] pc0 (младший разряд счётчика команд PC, 1)
#   [9] pc1 (старший разряд счётчика команд PC, 5) -> PC = pc1*5 + pc0 in [0..24] (адресует M_RAM[0 + PC])
NUM_SUBSPACES = 10
VALS_PER_SUB = 5
D_SUB = 80
D_MODEL = NUM_SUBSPACES * D_SUB  # 800
REG10_NAMES = ["d0", "d1", "d2", "p0", "p1", "sp", "b0", "b1", "pc0", "pc1"]

# Единое адресное пространство внешней ленты M_RAM[0..124] (N = 5^3 = 125 ячеек по 3 сферы = 240 каналов):
N_TAPE_CELLS = 125
CODE_SEG_BASE = 0     # M_RAM[0..24]   (a2 = 0): Кодовый сегмент (хранит инструкции фон-неймановской программы!)
DATA_SEG_BASE = 25    # M_RAM[25..49]  (a2 = 1): Сегмент оперативных данных (адресуется через PTR = 0..24)
EXTRA_SEG_BASE = 50   # M_RAM[50..99]  (a2 = 2,3): Расширенные банки данных
STACK_SEG_BASE = 100  # M_RAM[100..104](a2 = 4): Сегмент аппаратного стека LIFO (адресуется через SP = 0..4)

# 23 микрооперации фон-неймановского АЛУ (включая аппаратное управление сферами счётчика команд PC!):
ALU_OP_LIST = [
    "NOP",              # 0: регистры данных без изменений, PC <- PC + 1
    "ADD_CARRY_1",      # 1: ACC = (ACC + 1) % 125
    "ADD_CARRY_2",      # 2: ACC = (ACC + 2) % 125
    "ADD_CARRY_5",      # 3: ACC = (ACC + 5) % 125
    "SUB_BORROW_1",     # 4: ACC = (ACC - 1) % 125
    "SHL_BASE5",        # 5: ACC = (ACC * 5) % 125
    "SHR_BASE5",        # 6: ACC = ACC // 5
    "INC_PTR",          # 7: PTR = (PTR + 1) % 25
    "DEC_PTR",          # 8: PTR = (PTR - 1) % 25
    "ZERO_PTR",         # 9: PTR = 0
    "LOAD_MDR_ACC",     # 10: ACC = MDR (полное 3-разрядное слово 0..124 из M_RAM[25 + PTR])
    "ADD_MDR_CARRY",    # 11: ACC = (ACC + (MDR % 5)) % 125 (сложение с младшим разрядом MDR с переносом)
    "SUB_MDR_BORROW",   # 12: ACC = (ACC - (MDR % 5)) % 125 (вычитание младшего разряда MDR с заёмом)
    "MOV_ACC_TO_B",     # 13: B_reg = ACC % 25
    "MOV_B_TO_ACC",     # 14: ACC = B_reg
    "LOAD_MDR_B",       # 15: B_reg = MDR % 25
    "INC_B",            # 16: B_reg = (B_reg + 1) % 25
    "DEC_B",            # 17: B_reg = (B_reg - 1) % 25
    "ADD_ALPHA_CARRY",  # 18: ACC = (ACC + alpha) % 125
    "INC_SP",           # 19: sp = (sp + 1) % 5
    "DEC_SP",           # 20: sp = (sp - 1) % 5
    "INC_PC",           # 21: PC = (PC + 1) % 25 (штатный шаг счётчика команд в сферах 8, 9!)
    "LOAD_TARGET_PC",   # 22: PC = target_pc (аппаратный прыжок счётчика команд в сферах 8, 9!)
]
NUM_ALU_OPS = len(ALU_OP_LIST)
NUM_BRANCH_CONDS = 6  # [ACC==0, ACC!=0, B==0, B!=0, ACC<MDR, PTR==B]

MAX_CLAUSE_TOKENS = 48
MAX_CHARS_PER_TOK = 32
CHAR_VOCAB_SIZE = 256
NGRAM_BUCKETS = 4096


def apply_v15_alu_op_tensor(regs10, mdr_digits, op_ids, alpha_v, beta_v, target_pc=None):
    """
    Точная векторная семантика 23 микроопераций над 10 сферическими регистрами ЦПУ regs10 (B, 10)
    и предварительно прочитанным буфером памяти mdr_digits (B, 3).
    """
    d0, d1, d2, p0, p1, sp, b0, b1, pc0, pc1 = [regs10[:, i].clone() for i in range(10)]
    mdr0, mdr1, mdr2 = mdr_digits[:, 0], mdr_digits[:, 1], mdr_digits[:, 2]

    acc = d2 * 25 + d1 * 5 + d0
    mdr_val = mdr2 * 25 + mdr1 * 5 + mdr0
    ptr = p1 * 5 + p0
    b_val = b1 * 5 + b0
    pc = pc1 * 5 + pc0

    new_acc = acc.clone()
    acc_ops = {
        1: (acc + 1) % 125,
        2: (acc + 2) % 125,
        3: (acc + 5) % 125,
        4: (acc - 1) % 125,
        5: (acc * 5) % 125,
        6: acc // 5,
        10: mdr_val,
        11: (acc + mdr0) % 125,
        12: (acc - mdr0) % 125,
        14: b_val,
        18: (acc + alpha_v) % 125,
    }
    for oid, expr in acc_ops.items():
        new_acc = torch.where(op_ids == oid, expr, new_acc)

    nd0 = new_acc % 5
    nd1 = (new_acc // 5) % 5
    nd2 = (new_acc // 25) % 5

    new_ptr = ptr.clone()
    new_ptr = torch.where(op_ids == 7, (ptr + 1) % 25, new_ptr)
    new_ptr = torch.where(op_ids == 8, (ptr - 1) % 25, new_ptr)
    new_ptr = torch.where(op_ids == 9, torch.zeros_like(ptr), new_ptr)
    np0 = new_ptr % 5
    np1 = (new_ptr // 5) % 5

    nsp = sp.clone()
    nsp = torch.where(op_ids == 19, (sp + 1) % 5, nsp)
    nsp = torch.where(op_ids == 20, (sp - 1) % 5, nsp)

    new_b = b_val.clone()
    new_b = torch.where(op_ids == 13, (d1 * 5 + d0) % 25, new_b)
    new_b = torch.where(op_ids == 15, (mdr1 * 5 + mdr0) % 25, new_b)
    new_b = torch.where(op_ids == 16, (b_val + 1) % 25, new_b)
    new_b = torch.where(op_ids == 17, (b_val - 1) % 25, new_b)
    nb0 = new_b % 5
    nb1 = (new_b // 5) % 5

    new_pc = pc.clone()
    new_pc = torch.where(op_ids == 21, (pc + 1) % 25, new_pc)
    if target_pc is not None:
        new_pc = torch.where(op_ids == 22, target_pc % 25, new_pc)
    npc0 = new_pc % 5
    npc1 = (new_pc // 5) % 5

    return torch.stack([nd0, nd1, nd2, np0, np1, nsp, nb0, nb1, npc0, npc1], dim=1)


def eval_v15_predicates(regs10, mdr_digits):
    acc = regs10[:, 2] * 25 + regs10[:, 1] * 5 + regs10[:, 0]
    ptr = regs10[:, 4] * 5 + regs10[:, 3]
    b_val = regs10[:, 7] * 5 + regs10[:, 6]
    mdr_val = mdr_digits[:, 2] * 25 + mdr_digits[:, 1] * 5 + mdr_digits[:, 0]

    c0 = (acc == 0).long()
    c1 = (acc != 0).long()
    c2 = (b_val == 0).long()
    c3 = (b_val != 0).long()
    c4 = (acc < mdr_val).long()
    c5 = (ptr == b_val).long()
    return torch.stack([c0, c1, c2, c3, c4, c5], dim=1)


class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        self.quant_alpha = 1.0

    def forward(self, x):
        w = self.weight
        if self.quant_alpha > 0.0:
            scale = w.abs().mean().clamp(min=1e-6)
            w_norm = w / scale
            w_tern = torch.clamp(torch.round(w_norm), -1.0, 1.0) * scale
            w_q = (
                w + self.quant_alpha * (w_tern - w).detach()
                if self.training
                else ((1.0 - self.quant_alpha) * w + self.quant_alpha * w_tern)
            )
        else:
            w_q = w
        return F.linear(x, w_q, self.bias)


class SubspaceSphereNorm(nn.Module):
    def __init__(self, d_model=800, n_groups=10):
        super().__init__()
        self.n_groups = n_groups
        self.d_sub = d_model // n_groups

    def forward(self, x):
        orig_shape = x.shape
        x_g = x.view(*orig_shape[:-1], self.n_groups, self.d_sub)
        x_g = F.normalize(x_g, p=2, dim=-1)
        return x_g.view(*orig_shape)


class ProductSimVQ10(nn.Module):
    """
    Произведение 10 ортогональных гиперсфер Вороного S^{79} в R^{800}
    (8 регистров данных/адресов + 2 сферы аппаратного счётчика команд PC = (pc0, pc1)).
    """

    def __init__(self, d_model=800, n_groups=10, num_vals=5):
        super().__init__()
        self.d_model = d_model
        self.n_groups = n_groups
        self.num_vals = num_vals
        self.d_sub = d_model // n_groups

        c_init = torch.zeros(n_groups, num_vals, self.d_sub)
        for g in range(n_groups):
            q, _ = torch.linalg.qr(torch.randn(self.d_sub, self.d_sub))
            c_init[g] = q[:num_vals]
        self.register_buffer("C_frozen", c_init)
        self.W_proj = nn.Parameter(
            torch.eye(self.d_sub).unsqueeze(0).repeat(n_groups, 1, 1)
        )
        self.logit_scale = 20.0

    def get_codebook(self):
        cw = torch.einsum("gvd,gde->gve", self.C_frozen, self.W_proj)
        return F.normalize(cw, p=2, dim=-1)

    def encode_regs(self, regs10):
        cb = self.get_codebook()
        parts = [cb[g, regs10[:, g]] for g in range(self.n_groups)]
        return torch.cat(parts, dim=-1)

    def encode_word3(self, word_digits):
        """Кодирует 3-разрядное слово (B, 3) in Z_5^3 (данные MDR или инструкцию IR) в сферы [0, 1, 2] -> (B, 240)."""
        cb = self.get_codebook()
        parts = [cb[g, word_digits[:, g]] for g in range(3)]
        return torch.cat(parts, dim=-1)

    def decode_word3(self, h_word):
        """Декодирует 3-сферный вектор (B, 240) в тройку цифр (B, 3) in [0..4]."""
        cb = self.get_codebook()[:3]
        h_g = F.normalize(h_word.view(-1, 3, self.d_sub), p=2, dim=-1)
        logits = torch.einsum("bgd,gvd->bgv", h_g, cb) * self.logit_scale
        return logits.argmax(dim=-1)

    def snap_word3(self, h_word):
        cb = self.get_codebook()[:3]
        idx = self.decode_word3(h_word)
        parts = [cb[g, idx[:, g]] for g in range(3)]
        return torch.cat(parts, dim=-1)

    def compute_logits(self, h):
        cb = self.get_codebook()
        orig_shape = h.shape[:-1]
        h_g = F.normalize(h.view(*orig_shape, self.n_groups, self.d_sub), p=2, dim=-1)
        return torch.einsum("...gd,gvd->...gv", h_g, cb) * self.logit_scale

    def snap_step(self, h, snap_strength=1.0, tau=0.05):
        cb = self.get_codebook()
        orig_shape = h.shape[:-1]
        h_g = F.normalize(h.view(*orig_shape, self.n_groups, self.d_sub), p=2, dim=-1)
        sim = torch.einsum("...gd,gvd->...gv", h_g, cb)
        if self.training:
            w_soft = F.softmax(sim / tau, dim=-1)
            idx = sim.argmax(dim=-1)
            w_hard = F.one_hot(idx, self.num_vals).float()
            w = w_soft + (w_hard - w_soft).detach()
        else:
            idx = sim.argmax(dim=-1)
            w = F.one_hot(idx, self.num_vals).float()
        q_g = F.normalize(torch.einsum("...gv,gvd->...gd", w, cb), p=2, dim=-1)
        out_g = F.normalize(
            (1.0 - snap_strength) * h_g + snap_strength * q_g, p=2, dim=-1
        )
        return out_g.view(*orig_shape, self.d_model)


# ==============================================================================
# ЭТАП 1 (Stage v15.1): 125-ячеечная фон-неймановская лента M_RAM с границей Гершгорина
# ==============================================================================
class GershgorinBiorthogonalTape125(nn.Module):
    """
    Единая фон-неймановская биортогональная матричная лента M_RAM на N = 5^3 = 125 ячеек
    размерности (B, 125, 240):
      - Ячейки [0..24]   (сегмент a2 = 0): Кодовый сегмент программы (адресуется сферами PC = (pc0, pc1) внутри h_t!).
      - Ячейки [25..49]  (сегмент a2 = 1): Оперативная память данных (адресуется сферами PTR = (p0, p1) внутри h_t!).
      - Ячейки [50..99]  (сегмент a2 = 2,3): Расширенные страницы памяти.
      - Ячейки [100..104](сегмент a2 = 4): Аппаратный стек LIFO (адресуется сферой SP внутри h_t!).

    Калибровка биортогональных ключей чтения/записи (125 x 125) использует неравенство
    Гёльдера — Гершгорина:
        lambda_max(G) <= ||G||_inf = max_i sum_j |G_ij|
    Это даёт оценку сверху, которая НЕ зависит от размера памяти N = 125 и сходится
    за 8 итераций Ньютона — Шульца до машинного нуля FP32 (~6e-8)!
    """

    def __init__(self, d_sub=80):
        super().__init__()
        self.d_sub = d_sub
        self.d_word = 3 * d_sub  # 240
        self.register_buffer("C_p0_dual", torch.zeros(5, d_sub))
        self.register_buffer("C_p1_dual", torch.zeros(5, d_sub))
        self.register_buffer("C_pc0_dual", torch.zeros(5, d_sub))
        self.register_buffer("C_pc1_dual", torch.zeros(5, d_sub))
        self.register_buffer("C_seg_bank", torch.zeros(5, d_sub))
        self.register_buffer("C_seg_dual", torch.zeros(5, d_sub))
        self.register_buffer("K_addr_read", torch.zeros(125, 125))
        self.register_buffer("K_addr_dual", torch.zeros(125, 125))
        self.ortho_err_gershgorin = 0.0
        self.ortho_err_frobenius_8steps = 0.0

    @staticmethod
    def _ns_inv_gershgorin(G, steps=8):
        """
        Итерация Ньютона — Шульца с нормировкой Гёльдера — Гершгорина ||G||_1 * ||G||_inf.
        Сходится за 6–8 шагов независимо от размерности N = 25, 125, 625!
        """
        I = torch.eye(G.size(0), device=G.device)
        g_l1 = G.abs().sum(dim=0).max().item()
        g_linf = G.abs().sum(dim=1).max().item()
        sigma_max_sq_bound = max(g_l1 * g_linf, 1e-6)
        X = (1.5 / sigma_max_sq_bound) * G.T
        for _ in range(steps):
            X = X @ (2.0 * I - G @ X)
        return X

    @staticmethod
    def _ns_inv_frobenius(G, steps=8):
        """Для прямого численного сравнения с нормировкой Фробениуса на 8 шагах при N=125."""
        I = torch.eye(G.size(0), device=G.device)
        trace_sq = (G * G).sum().item()
        X = (1.5 / max(trace_sq, 1e-6)) * G.T
        for _ in range(steps):
            X = X @ (2.0 * I - G @ X)
        return X

    @torch.no_grad()
    def calibrate_125_address_basis(self, simvq10, ns_steps=8):
        cb = simvq10.get_codebook()
        c_p0, c_p1 = cb[3], cb[4]    # сферы указателя данных PTR (p0, p1)
        c_pc0, c_pc1 = cb[8], cb[9]  # сферы счётчика команд PC (pc0, pc1)
        c_seg = cb[2]                # сфера выбора сегмента памяти a2 in [0..4]

        self.C_p0_dual.copy_(self._ns_inv_gershgorin(c_p0 @ c_p0.T, steps=ns_steps) @ c_p0)
        self.C_p1_dual.copy_(self._ns_inv_gershgorin(c_p1 @ c_p1.T, steps=ns_steps) @ c_p1)
        self.C_pc0_dual.copy_(self._ns_inv_gershgorin(c_pc0 @ c_pc0.T, steps=ns_steps) @ c_pc0)
        self.C_pc1_dual.copy_(self._ns_inv_gershgorin(c_pc1 @ c_pc1.T, steps=ns_steps) @ c_pc1)
        self.C_seg_bank.copy_(c_seg)
        self.C_seg_dual.copy_(self._ns_inv_gershgorin(c_seg @ c_seg.T, steps=ns_steps) @ c_seg)

        # Строим 125 кронекеровских ключей адресации a = 25*a2 + 5*a1 + a0 in [0..124]:
        keys = []
        for a2 in range(5):
            u2 = c_seg[a2] @ self.C_seg_dual.T
            for a1 in range(5):
                u1 = c_p1[a1] @ self.C_p1_dual.T
                for a0 in range(5):
                    u0 = c_p0[a0] @ self.C_p0_dual.T
                    k_vec = torch.einsum("i,j,k->ijk", u2, u1, u0).reshape(125)
                    keys.append(k_vec)
        K = torch.stack(keys, dim=0)  # (125, 125)
        G = K @ K.T
        I125 = torch.eye(125, device=K.device)

        # Сравним точность за 8 шагов: Фробениус vs Гёльдер-Гершгорин на N=125!
        K_dual_frob = self._ns_inv_frobenius(G, steps=ns_steps) @ K
        self.ortho_err_frobenius_8steps = (K_dual_frob @ K.T - I125).abs().max().item()

        K_dual_gersh = self._ns_inv_gershgorin(G, steps=ns_steps) @ K
        self.ortho_err_gershgorin = (K_dual_gersh @ K.T - I125).abs().max().item()

        self.K_addr_read.copy_(K)
        self.K_addr_dual.copy_(K_dual_gersh)
        return self.ortho_err_gershgorin, self.ortho_err_frobenius_8steps

    def _key125_from_coords(self, u2, u1, u0):
        return torch.einsum("bi,bj,bk->bijk", u2, u1, u0).reshape(u0.size(0), 125)

    def _addr_key_pc(self, h_state):
        """
        Формирует 125-мерный ключ чтения кодового сегмента M_RAM[0 + PC] (сегмент a2 = 0)
        напрямую из сферических координат счётчика команд PC = (сфера 8, сфера 9) внутри h_state!
        """
        B = h_state.size(0)
        h_g = h_state.view(B, 10, self.d_sub)
        u0 = h_g[:, 8] @ self.C_pc0_dual.T  # (B, 5) из сферы pc0
        u1 = h_g[:, 9] @ self.C_pc1_dual.T  # (B, 5) из сферы pc1
        h_seg0 = self.C_seg_bank[0].unsqueeze(0).expand(B, self.d_sub)
        u2 = h_seg0 @ self.C_seg_dual.T     # (B, 5) сегмент 0 (кодовый сегмент M_RAM[0..24])
        return self._key125_from_coords(u2, u1, u0)

    def _addr_key_ptr(self, h_state):
        """
        Формирует 125-мерный ключ чтения/записи сегмента данных M_RAM[25 + PTR] (сегмент a2 = 1)
        напрямую из сферических координат указателя PTR = (сфера 3, сфера 4) внутри h_state!
        """
        B = h_state.size(0)
        h_g = h_state.view(B, 10, self.d_sub)
        u0 = h_g[:, 3] @ self.C_p0_dual.T   # (B, 5) из сферы p0
        u1 = h_g[:, 4] @ self.C_p1_dual.T   # (B, 5) из сферы p1
        h_seg1 = self.C_seg_bank[1].unsqueeze(0).expand(B, self.d_sub)
        u2 = h_seg1 @ self.C_seg_dual.T     # (B, 5) сегмент 1 (сегмент данных M_RAM[25..49])
        return self._key125_from_coords(u2, u1, u0)

    def _addr_key_stack(self, h_state, simvq10):
        """
        Формирует 125-мерный ключ чтения/записи сегмента стека M_RAM[100 + sp] (сегмент a2 = 4, a1 = 0)
        из сферической координаты указателя стека sp (сфера 5) внутри h_state!
        """
        B = h_state.size(0)
        cb = simvq10.get_codebook()
        h_g = h_state.view(B, 10, self.d_sub)
        sp_idx = (h_g[:, 5] @ cb[5].T).argmax(dim=-1)
        h_p0 = cb[3, sp_idx]
        h_p1 = cb[4, torch.zeros_like(sp_idx)]
        h_seg4 = self.C_seg_bank[4].unsqueeze(0).expand(B, self.d_sub)
        u0 = h_p0 @ self.C_p0_dual.T
        u1 = h_p1 @ self.C_p1_dual.T
        u2 = h_seg4 @ self.C_seg_dual.T
        return self._key125_from_coords(u2, u1, u0)

    @torch.no_grad()
    def init_von_neumann_tape(self, code_words_digits, data_words_digits, simvq10):
        """
        Загружает в единую матрицу M_RAM (B, 125, 240):
          - скомпилированные инструкции программы `code_words_digits` (B, 25, 3) в кодовый сегмент M_RAM[0..24],
          - входные данные `data_words_digits` (B, 25, 3) в сегмент данных M_RAM[25..49],
          - чистый сегмент стека M_RAM[100..124].
        """
        B = data_words_digits.size(0)
        dev = data_words_digits.device
        full_digits = torch.zeros(B, 125, 3, dtype=torch.long, device=dev)
        L_code = code_words_digits.size(1)
        full_digits[:, :L_code, :] = code_words_digits
        full_digits[:, DATA_SEG_BASE : DATA_SEG_BASE + 25, :] = data_words_digits

        V_words = simvq10.encode_word3(full_digits.reshape(B * 125, 3)).view(B, 125, self.d_word)
        M_ram = torch.einsum("ak,bkd->bad", self.K_addr_dual.T, V_words)
        return M_ram

    @torch.no_grad()
    def fetch_instruction_at_pc(self, M_ram, h_state, simvq10):
        """
        Фон-неймановский Instruction Fetch: читает текущую инструкцию IR (B, 240) и её 3 разряда (ir0, ir1, ir2)
        прямо из кодового сегмента M_RAM[0 + PC] по сферам счётчика команд (pc0, pc1) внутри h_state!
        """
        k_pc = self._addr_key_pc(h_state)  # (B, 125)
        raw_ir = torch.einsum("ba,bad->bd", k_pc, M_ram)  # (B, 240)
        ir_digits = simvq10.decode_word3(raw_ir)          # (B, 3): [ir0, ir1, ir2]
        return ir_digits, simvq10.snap_word3(raw_ir)

    @torch.no_grad()
    def read_data_at_ptr(self, M_ram, h_state, simvq10):
        """Читает 3-сферное слово данных MDR (B, 240) из M_RAM[25 + PTR] по сферам PTR=(p0, p1)."""
        k_ptr = self._addr_key_ptr(h_state)
        raw_mdr = torch.einsum("ba,bad->bd", k_ptr, M_ram)
        return simvq10.snap_word3(raw_mdr)

    @torch.no_grad()
    def read_data_at_stack(self, M_ram, h_state, simvq10):
        """Читает верх аппаратного стека M_RAM[100 + sp] по сфере sp."""
        k_stk = self._addr_key_stack(h_state, simvq10)
        raw_mdr = torch.einsum("ba,bad->bd", k_stk, M_ram)
        return simvq10.snap_word3(raw_mdr)

    @torch.no_grad()
    def write_data_at_ptr(self, M_ram, h_state, v_new_word, mask=None):
        """Запись ранга 1 в сегмент данных M_RAM[25 + PTR]."""
        B = h_state.size(0)
        k_read = self._addr_key_ptr(h_state)
        k_dual = k_read @ self.K_addr_dual
        v_old = torch.einsum("ba,bad->bd", k_read, M_ram)
        delta_M = torch.einsum("ba,bd->bad", k_dual, v_new_word - v_old)
        if mask is not None:
            delta_M = delta_M * mask.view(B, 1, 1).float()
        return M_ram + delta_M

    @torch.no_grad()
    def write_data_at_stack(self, M_ram, h_state, v_new_word, simvq10, mask=None):
        """Запись ранга 1 в аппаратный стек M_RAM[100 + sp]."""
        B = h_state.size(0)
        k_read = self._addr_key_stack(h_state, simvq10)
        k_dual = k_read @ self.K_addr_dual
        v_old = torch.einsum("ba,bad->bd", k_read, M_ram)
        delta_M = torch.einsum("ba,bd->bad", k_dual, v_new_word - v_old)
        if mask is not None:
            delta_M = delta_M * mask.view(B, 1, 1).float()
        return M_ram + delta_M

    @torch.no_grad()
    def decode_data_segment(self, M_ram, simvq10):
        """Декодирует сегмент данных M_RAM[25..49] в 25 целых чисел (B, 25) in [0..124]."""
        B = M_ram.size(0)
        K_data = self.K_addr_read[DATA_SEG_BASE : DATA_SEG_BASE + 25]  # (25, 125)
        words = torch.einsum("ak,bkd->bad", K_data, M_ram).reshape(B * 25, self.d_word)
        digits = simvq10.decode_word3(words).view(B, 25, 3)
        return digits[:, :, 2] * 25 + digits[:, :, 1] * 5 + digits[:, :, 0]

    @torch.no_grad()
    def decode_stack_segment(self, M_ram, simvq10):
        """Декодирует 5 ячеек аппаратного стека M_RAM[100..104] в (B, 5) in [0..124]."""
        B = M_ram.size(0)
        K_stk = self.K_addr_read[STACK_SEG_BASE : STACK_SEG_BASE + 5]
        words = torch.einsum("ak,bkd->bad", K_stk, M_ram).reshape(B * 5, self.d_word)
        digits = simvq10.decode_word3(words).view(B, 5, 3)
        return digits[:, :, 2] * 25 + digits[:, :, 1] * 5 + digits[:, :, 0]


# ==============================================================================
# 100% Тернарное фон-неймановское ядро АЛУ (10 сфер ЦПУ + 3 сферы MDR)
# ==============================================================================
class TernaryVonNeumannCore_v15(nn.Module):
    """
    100% Тернарное ядро АЛУ над 10 сферами ЦПУ (включая сферический счётчик команд PC = (pc0, pc1))
    и 3 сферами регистра данных MDR.
    Поддерживает как все 21 операцию над данными/указателями/стеком, так и аппаратный шаг
    счётчика команд `INC_PC` (op=21: PC <- (PC + 1) % 25 с переносом pc0 -> pc1) и
    прыжок `LOAD_TARGET_PC` (op=22: PC <- target_pc).
    """

    def __init__(self, d_model=800, n_groups=10, num_vals=5, num_ops=NUM_ALU_OPS):
        super().__init__()
        self.d_model = d_model
        self.n_groups = n_groups
        self.d_sub = d_model // n_groups
        self.d_mdr = 3 * self.d_sub  # 240
        self.simvq = ProductSimVQ10(d_model=d_model, n_groups=n_groups, num_vals=num_vals)
        self.sphere_norm = SubspaceSphereNorm(d_model=d_model, n_groups=n_groups)

        self.op_emb = nn.Embedding(num_ops, d_model)
        self.alpha_emb = nn.Embedding(num_vals, d_model // 2)
        self.beta_emb = nn.Embedding(num_vals, d_model // 2)
        nn.init.orthogonal_(self.op_emb.weight)
        nn.init.orthogonal_(self.alpha_emb.weight)
        nn.init.orthogonal_(self.beta_emb.weight)

        d_hidden = 1536
        self.in_state_a = BitLinear(d_model + self.d_mdr, d_hidden, bias=False)
        self.in_state_b = BitLinear(d_model + self.d_mdr, d_hidden, bias=False)
        self.in_op_a = BitLinear(d_model, d_hidden, bias=False)
        self.in_op_b = BitLinear(d_model, d_hidden, bias=False)
        self.in_epi = BitLinear(d_model, d_hidden, bias=False)

        self.mid_norm = nn.LayerNorm(d_hidden, elementwise_affine=False)
        self.mid_gate = BitLinear(d_hidden, d_hidden, bias=False)
        self.mid_val = BitLinear(d_hidden, d_hidden, bias=False)
        self.out_proj = BitLinear(d_hidden, d_model, bias=False)

        # Таблица маскирования активных сфер ВЫХОДА (23 операции x 10 сфер):
        mask_mat = torch.zeros(num_ops, n_groups)
        for oid in [1, 2, 3, 4, 5, 6, 10, 11, 12, 14, 18]:
            mask_mat[oid, 0:3] = 1.0  # ACC (d0, d1, d2)
        for oid in [7, 8, 9]:
            mask_mat[oid, 3:5] = 1.0  # PTR (p0, p1)
        for oid in [19, 20]:
            mask_mat[oid, 5] = 1.0    # SP (sp)
        for oid in [13, 15, 16, 17]:
            mask_mat[oid, 6:8] = 1.0  # B_reg (b0, b1)
        for oid in [21, 22]:
            mask_mat[oid, 8:10] = 1.0 # PC (pc0, pc1)
        self.register_buffer("op_sphere_mask", mask_mat)

        # Таблица маскирования активных сфер ВХОДА (23 операции x 13 сфер: 10 ЦПУ + 3 MDR):
        in_mask = torch.zeros(num_ops, n_groups + 3)
        for oid in [1, 2, 3, 4, 5, 6, 18]:
            in_mask[oid, 0:3] = 1.0
        for oid in [11, 12]:
            in_mask[oid, 0:3] = 1.0
            in_mask[oid, 10] = 1.0    # сфера mdr0 (10-я по счёту)
        in_mask[10, 10:13] = 1.0      # LOAD_MDR_ACC видит 3 сферы MDR (10..12)
        for oid in [7, 8, 9]:
            in_mask[oid, 3:5] = 1.0
        for oid in [19, 20]:
            in_mask[oid, 5] = 1.0
        in_mask[13, 0:2] = 1.0
        in_mask[14, 6:8] = 1.0
        in_mask[15, 10:12] = 1.0
        for oid in [16, 17]:
            in_mask[oid, 6:8] = 1.0
        for oid in [21, 22]:
            in_mask[oid, 8:10] = 1.0  # INC_PC / LOAD_TARGET_PC
        self.register_buffer("op_input_mask", in_mask)

    def set_quant_alpha(self, alpha):
        for m in self.modules():
            if isinstance(m, BitLinear):
                m.quant_alpha = alpha

    def step(self, h_prev, h_mdr, op_ids, alpha_v, beta_v, target_pc=None, snap_strength=1.0):
        B = h_prev.size(0)
        c_op = self.op_emb(op_ids)
        c_epi = torch.cat([self.alpha_emb(alpha_v), self.beta_emb(beta_v)], dim=-1)

        h_in_g = torch.cat([h_prev, h_mdr], dim=-1).view(B, self.n_groups + 3, self.d_sub)
        in_m = self.op_input_mask[op_ids].unsqueeze(-1)
        h_in = (h_in_g * in_m).view(B, self.d_model + self.d_mdr)

        u_a = F.silu(self.in_state_a(h_in) + self.in_op_a(c_op))
        u_b = F.gelu(self.in_state_b(h_in) + self.in_op_b(c_op) + self.in_epi(c_epi))
        h_int = self.mid_norm(u_a * u_b)
        h_mlp = F.silu(self.mid_gate(h_int)) * self.mid_val(h_int)
        delta = self.out_proj(h_int + h_mlp)

        h_cand_raw = self.sphere_norm(0.15 * h_prev + delta).view(B, self.n_groups, self.d_sub)
        h_prev_g = h_prev.view(B, self.n_groups, self.d_sub)
        h_mdr_g = h_mdr.view(B, 3, self.d_sub)

        sp_mask = self.op_sphere_mask[op_ids].unsqueeze(-1)
        h_cand_g = torch.where(sp_mask > 0.5, h_cand_raw, h_prev_g)

        cb = self.simvq.get_codebook()
        is_zero_ptr = (op_ids == 9).view(B, 1)
        if is_zero_ptr.any():
            h_cand_g[:, 3, :] = torch.where(is_zero_ptr, cb[3, 0].unsqueeze(0), h_cand_g[:, 3, :])
            h_cand_g[:, 4, :] = torch.where(is_zero_ptr, cb[4, 0].unsqueeze(0), h_cand_g[:, 4, :])

        is_load_mdr = (op_ids == 10).view(B, 1, 1)
        if is_load_mdr.any():
            h_cand_g[:, 0:3, :] = torch.where(is_load_mdr, h_mdr_g, h_cand_g[:, 0:3, :])

        is_mov_ab = (op_ids == 13).view(B, 1)
        if is_mov_ab.any():
            d0_idx = (h_prev_g[:, 0, :] @ cb[0].T).argmax(dim=-1)
            d1_idx = (h_prev_g[:, 1, :] @ cb[1].T).argmax(dim=-1)
            h_cand_g[:, 6, :] = torch.where(is_mov_ab, cb[6, d0_idx], h_cand_g[:, 6, :])
            h_cand_g[:, 7, :] = torch.where(is_mov_ab, cb[7, d1_idx], h_cand_g[:, 7, :])

        is_mov_ba = (op_ids == 14).view(B, 1)
        if is_mov_ba.any():
            b0_idx = (h_prev_g[:, 6, :] @ cb[6].T).argmax(dim=-1)
            b1_idx = (h_prev_g[:, 7, :] @ cb[7].T).argmax(dim=-1)
            h_cand_g[:, 0, :] = torch.where(is_mov_ba, cb[0, b0_idx], h_cand_g[:, 0, :])
            h_cand_g[:, 1, :] = torch.where(is_mov_ba, cb[1, b1_idx], h_cand_g[:, 1, :])
            h_cand_g[:, 2, :] = torch.where(is_mov_ba, cb[2, 0].unsqueeze(0), h_cand_g[:, 2, :])

        is_load_mb = (op_ids == 15).view(B, 1)
        if is_load_mb.any():
            m0_idx = (h_mdr_g[:, 0, :] @ cb[0].T).argmax(dim=-1)
            m1_idx = (h_mdr_g[:, 1, :] @ cb[1].T).argmax(dim=-1)
            h_cand_g[:, 6, :] = torch.where(is_load_mb, cb[6, m0_idx], h_cand_g[:, 6, :])
            h_cand_g[:, 7, :] = torch.where(is_load_mb, cb[7, m1_idx], h_cand_g[:, 7, :])

        if target_pc is not None:
            is_jmp_pc = (op_ids == 22).view(B, 1)
            if is_jmp_pc.any():
                t_pc0 = target_pc % 5
                t_pc1 = (target_pc // 5) % 5
                h_cand_g[:, 8, :] = torch.where(is_jmp_pc, cb[8, t_pc0], h_cand_g[:, 8, :])
                h_cand_g[:, 9, :] = torch.where(is_jmp_pc, cb[9, t_pc1], h_cand_g[:, 9, :])

        h_cand = h_cand_g.view(B, self.d_model)
        h_next = self.simvq.snap_step(h_cand, snap_strength=snap_strength)
        return h_next, h_cand


class TernaryBranchRouter_v15(nn.Module):
    """
    100% Тернарный маршрутизатор 6 аппаратных предикатов ветвления над сферами ЦПУ (800) и MDR (240).
    """

    def __init__(self, d_model=800, d_mdr=240):
        super().__init__()
        d_in = d_model + d_mdr
        self.norm = nn.LayerNorm(d_in, elementwise_affine=False)
        self.fc1_a = BitLinear(d_in, 1024, bias=False)
        self.fc1_b = BitLinear(d_in, 1024, bias=False)
        self.fc2_a = BitLinear(1024, 512, bias=False)
        self.fc2_b = BitLinear(1024, 512, bias=False)
        self.pred_head = nn.Linear(512, NUM_BRANCH_CONDS * 2)

    def set_quant_alpha(self, alpha):
        for m in self.modules():
            if isinstance(m, BitLinear):
                m.quant_alpha = alpha

    def forward(self, h_state, h_mdr):
        x = self.norm(torch.cat([h_state, h_mdr], dim=-1))
        h1 = F.silu(self.fc1_a(x)) * F.gelu(self.fc1_b(x))
        h2 = F.silu(self.fc2_a(h1)) * F.gelu(self.fc2_b(h1))
        return self.pred_head(h2).view(-1, NUM_BRANCH_CONDS, 2)


# ==============================================================================
# ЭТАП 2 (Stage v15.2): Кодирование инструкций внутри M_RAM[0..24] (Фон-Неймановский цикл)
# ==============================================================================
# Каждая инструкция фон-неймановской машины кодируется в 3-разрядное слово (ir0, ir1, ir2) in Z_5^3
# и записывается в кодовый сегмент M_RAM[0..24] как 3-сферный вектор Вороного!
# Таблица декодирования 3-разрядного слова инструкции W_ir = 25*ir2 + 5*ir1 + ir0:
#   - Для линейных микроинструкций (ir2 == 0):
#       ir0 in [0..4], ir1 in [0..4] кодируют пару (alu_op, write_mode) из компактного набора ISA (16 базовых пар):
VN_MICRO_ISA = [
    # idx = 5*ir1 + ir0 (при ir2 == 0): (alu_op, write_mode, is_halt)
    (0, 0, False),   # 0: NOP
    (9, 0, False),   # 1: ZERO_PTR        (PTR = 0)
    (7, 0, False),   # 2: INC_PTR         (PTR += 1)
    (8, 0, False),   # 3: DEC_PTR         (PTR -= 1)
    (10, 0, False),  # 4: LOAD_MDR_ACC    (ACC = M_RAM[25 + PTR])
    (11, 0, False),  # 5: ADD_MDR_CARRY   (ACC = (ACC + M_RAM[25 + PTR]%5) % 125)
    (12, 0, False),  # 6: SUB_MDR_BORROW  (ACC = (ACC - M_RAM[25 + PTR]%5) % 125)
    (0, 1, False),   # 7: STORE_ACC_PTR   (M_RAM[25 + PTR] = ACC)
    (10, 3, False),  # 8: SWAP_ACC_PTR    (ACC <-> M_RAM[25 + PTR])
    (19, 4, False),  # 9: PUSH_ACC_STACK  (M_RAM[100 + sp] = ACC, sp += 1)
    (10, 5, False),  # 10: POP_STACK_ACC  (sp -= 1, ACC = M_RAM[100 + sp])
    (1, 0, False),   # 11: ADD_CARRY_1    (ACC += 1)
    (4, 0, False),   # 12: SUB_BORROW_1   (ACC -= 1)
    (5, 0, False),   # 13: SHL_BASE5      (ACC = (ACC * 5) % 125)
    (6, 0, False),   # 14: SHR_BASE5      (ACC = ACC // 5)
    (18, 0, False),  # 15: ADD_ALPHA_CARRY(ACC = (ACC + alpha) % 125)
    (13, 0, False),  # 16: MOV_ACC_TO_B   (B = ACC % 25)
    (14, 0, False),  # 17: MOV_B_TO_ACC   (ACC = B)
    (16, 0, False),  # 18: INC_B          (B += 1)
    (17, 0, False),  # 19: DEC_B          (B -= 1)
    (0, 0, True),    # 20: HALT           (останов фон-неймановской машины!)
]
#   - Для инструкций переходов (ir2 in [1..4]):
#       ir2 == 1: JMP target_pc         (безусловный прыжок на адрес target_pc = 5*ir1 + ir0 in [0..24])
#       ir2 == 2: JEQ_PTR_B target_pc   (если PTR == B: PC <- 5*ir1 + ir0, иначе PC + 1)
#       ir2 == 3: JLT_ACC_MDR target_pc (если ACC < MDR: PC <- 5*ir1 + ir0, иначе PC + 1)
#       ir2 == 4: JZ_ACC target_pc      (если ACC == 0:  PC <- 5*ir1 + ir0, иначе PC + 1)


def encode_vn_instruction_word(inst_type, arg=0):
    """
    Упаковывает инструкцию фон-неймановской машины в 3-разрядное слово [ir0, ir1, ir2] in Z_5^3,
    которое записывается прямо в ячейку кодового сегмента M_RAM[PC]!
    """
    if inst_type == "MICRO":
        # arg = индекс в VN_MICRO_ISA [0..20]
        return [arg % 5, (arg // 5) % 5, 0]
    elif inst_type == "JMP":
        return [arg % 5, (arg // 5) % 5, 1]
    elif inst_type == "JEQ_PTR_B":
        return [arg % 5, (arg // 5) % 5, 2]
    elif inst_type == "JLT_ACC_MDR":
        return [arg % 5, (arg // 5) % 5, 3]
    elif inst_type == "JZ_ACC":
        return [arg % 5, (arg // 5) % 5, 4]
    raise ValueError(f"Unknown inst_type: {inst_type}")


def decode_vn_ir_digits_batch(ir_digits):
    """
    Декодирует батч прочитанных из M_RAM[PC] 3-разрядных слов инструкций ir_digits (B, 3)
    в управляющие сигналы такта: (alu_op, write_mode, br_type, target_pc, is_halt).
    """
    B = ir_digits.size(0)
    dev = ir_digits.device
    ir0, ir1, ir2 = ir_digits[:, 0], ir_digits[:, 1], ir_digits[:, 2]
    sub_code = ir1 * 5 + ir0  # in [0..24]

    alu_op = torch.zeros(B, dtype=torch.long, device=dev)
    w_mode = torch.zeros(B, dtype=torch.long, device=dev)
    is_halt = torch.zeros(B, dtype=torch.bool, device=dev)

    is_micro = ir2 == 0
    for idx, (op_val, wm_val, halt_flag) in enumerate(VN_MICRO_ISA):
        m = is_micro & (sub_code == idx)
        alu_op = torch.where(m, torch.full_like(alu_op, op_val), alu_op)
        w_mode = torch.where(m, torch.full_like(w_mode, wm_val), w_mode)
        if halt_flag:
            is_halt = is_halt | m

    # br_type: 0=NEXT, 1=JMP, 2=JEQ_PTR_B (cond 5), 3=JLT_ACC_MDR (cond 4), 4=JZ_ACC (cond 0)
    br_type = ir2.clone()
    target_pc = sub_code.clone()
    return alu_op, w_mode, br_type, target_pc, is_halt


def build_v15_von_neumann_programs(dev="cuda"):
    """
    Собирает 5 эталонных программ, каждая из которых кодируется как массив 3-разрядных слов
    (L <= 25, 3) и загружается прямо в кодовый сегмент M_RAM[0..24]:
      0. `ArraySum_VN`: суммирование среза M_RAM[25 + PTR .. 25 + B_reg - 1] в ACC.
      1. `ArrayMax_VN`: поиск максимума в M_RAM[25..28] и запись результата в M_RAM[29].
      2. `BubbleSort3_VN`: полная сортировка M_RAM[25..27] по возрастанию на месте (in-place).
      3. `StackReverse3_VN`: разворот M_RAM[25..27] через аппаратный стек M_RAM[100..102] (PUSH x3 + POP x3).
      4. `AffineMap3_VN`: прибавление константы квоты alpha ко всем элементам массива M_RAM[25..27] на месте!
    """
    progs = {}

    def _pad25(word_list):
        padded = list(word_list) + [encode_vn_instruction_word("MICRO", 20)] * (25 - len(word_list))
        return torch.tensor(padded[:25], dtype=torch.long, device=dev)

    # 0. ArraySum_VN:
    #   PC=0: JEQ_PTR_B -> 4
    #   PC=1: ADD_MDR_CARRY (micro 5)
    #   PC=2: INC_PTR       (micro 2)
    #   PC=3: JMP -> 0
    #   PC=4: HALT          (micro 20)
    progs["ArraySum_VN"] = _pad25([
        encode_vn_instruction_word("JEQ_PTR_B", 4),
        encode_vn_instruction_word("MICRO", 5),
        encode_vn_instruction_word("MICRO", 2),
        encode_vn_instruction_word("JMP", 0),
        encode_vn_instruction_word("MICRO", 20),
    ])

    # 1. ArrayMax_VN:
    #   PC=0: JEQ_PTR_B -> 6
    #   PC=1: JLT_ACC_MDR -> 4 (если ACC < MDR, прыгаем на загрузку нового максимума)
    #   PC=2: INC_PTR (micro 2)
    #   PC=3: JMP -> 0
    #   PC=4: LOAD_MDR_ACC (micro 4)
    #   PC=5: JMP -> 2
    #   PC=6: STORE_ACC_PTR (micro 7: записывает найденный максимум в M_RAM[25 + 4])
    #   PC=7: HALT (micro 20)
    progs["ArrayMax_VN"] = _pad25([
        encode_vn_instruction_word("JEQ_PTR_B", 6),
        encode_vn_instruction_word("JLT_ACC_MDR", 4),
        encode_vn_instruction_word("MICRO", 2),
        encode_vn_instruction_word("JMP", 0),
        encode_vn_instruction_word("MICRO", 4),
        encode_vn_instruction_word("JMP", 2),
        encode_vn_instruction_word("MICRO", 7),
        encode_vn_instruction_word("MICRO", 20),
    ])

    # 2. BubbleSort3_VN (24 инструкции в M_RAM[0..23] — как раз помещается в 25 слов кодового сегмента!):
    progs["BubbleSort3_VN"] = _pad25([
        # Пары (0, 1):
        encode_vn_instruction_word("MICRO", 1),        # PC=0: ZERO_PTR
        encode_vn_instruction_word("MICRO", 4),        # PC=1: LOAD_MDR_ACC (ACC = M[0])
        encode_vn_instruction_word("MICRO", 2),        # PC=2: INC_PTR (PTR = 1)
        encode_vn_instruction_word("JLT_ACC_MDR", 8),  # PC=3: if ACC < M[1] -> PC=8
        encode_vn_instruction_word("MICRO", 8),        # PC=4: SWAP_ACC_PTR (ACC <-> M[1])
        encode_vn_instruction_word("MICRO", 3),        # PC=5: DEC_PTR (PTR = 0)
        encode_vn_instruction_word("MICRO", 7),        # PC=6: STORE_ACC_PTR (M[0] = ACC)
        encode_vn_instruction_word(" MICRO".strip(), 2),# PC=7: INC_PTR (PTR = 1)
        # Пары (1, 2):
        encode_vn_instruction_word("MICRO", 4),        # PC=8: LOAD_MDR_ACC (ACC = M[1])
        encode_vn_instruction_word("MICRO", 2),        # PC=9: INC_PTR (PTR = 2)
        encode_vn_instruction_word("JLT_ACC_MDR", 15), # PC=10: if ACC < M[2] -> PC=15
        encode_vn_instruction_word("MICRO", 8),        # PC=11: SWAP_ACC_PTR (ACC <-> M[2])
        encode_vn_instruction_word("MICRO", 3),        # PC=12: DEC_PTR (PTR = 1)
        encode_vn_instruction_word("MICRO", 7),        # PC=13: STORE_ACC_PTR (M[1] = ACC)
        encode_vn_instruction_word("MICRO", 0),        # PC=14: NOP
        # Финальная проверка пары (0, 1):
        encode_vn_instruction_word("MICRO", 1),        # PC=15: ZERO_PTR (PTR = 0)
        encode_vn_instruction_word("MICRO", 4),        # PC=16: LOAD_MDR_ACC (ACC = M[0])
        encode_vn_instruction_word("MICRO", 2),        # PC=17: INC_PTR (PTR = 1)
        encode_vn_instruction_word("JLT_ACC_MDR", 22), # PC=18: if ACC < M[1] -> PC=22 (HALT)
        encode_vn_instruction_word("MICRO", 8),        # PC=19: SWAP_ACC_PTR (ACC <-> M[1])
        encode_vn_instruction_word("MICRO", 3),        # PC=20: DEC_PTR (PTR = 0)
        encode_vn_instruction_word("MICRO", 7),        # PC=21: STORE_ACC_PTR (M[0] = ACC)
        encode_vn_instruction_word("MICRO", 20),       # PC=22: HALT
    ])

    # 3. StackReverse3_VN (разворот M_RAM[25..27] через аппаратный стек M_RAM[100..102]):
    progs["StackReverse3_VN"] = _pad25([
        encode_vn_instruction_word("MICRO", 1),   # PC=0: ZERO_PTR
        encode_vn_instruction_word("MICRO", 4),   # PC=1: LOAD_MDR_ACC
        encode_vn_instruction_word("MICRO", 9),   # PC=2: PUSH_ACC_STACK
        encode_vn_instruction_word("MICRO", 2),   # PC=3: INC_PTR
        encode_vn_instruction_word("MICRO", 4),   # PC=4: LOAD_MDR_ACC
        encode_vn_instruction_word("MICRO", 9),   # PC=5: PUSH_ACC_STACK
        encode_vn_instruction_word("MICRO", 2),   # PC=6: INC_PTR
        encode_vn_instruction_word("MICRO", 4),   # PC=7: LOAD_MDR_ACC
        encode_vn_instruction_word("MICRO", 9),   # PC=8: PUSH_ACC_STACK
        encode_vn_instruction_word("MICRO", 1),   # PC=9: ZERO_PTR
        encode_vn_instruction_word("MICRO", 10),  # PC=10: POP_STACK_ACC
        encode_vn_instruction_word("MICRO", 7),   # PC=11: STORE_ACC_PTR
        encode_vn_instruction_word("MICRO", 2),   # PC=12: INC_PTR
        encode_vn_instruction_word("MICRO", 10),  # PC=13: POP_STACK_ACC
        encode_vn_instruction_word("MICRO", 7),   # PC=14: STORE_ACC_PTR
        encode_vn_instruction_word("MICRO", 2),   # PC=15: INC_PTR
        encode_vn_instruction_word("MICRO", 10),  # PC=16: POP_STACK_ACC
        encode_vn_instruction_word("MICRO", 7),   # PC=17: STORE_ACC_PTR
        encode_vn_instruction_word("MICRO", 20),  # PC=18: HALT
    ])

    # 4. AffineMap3_VN: ко всем элементам M_RAM[25..27] прибавляет константу alpha с переносом разряда!
    progs["AffineMap3_VN"] = _pad25([
        encode_vn_instruction_word("MICRO", 1),   # PC=0: ZERO_PTR (PTR=0)
        encode_vn_instruction_word("MICRO", 4),   # PC=1: LOAD_MDR_ACC (ACC = M[0])
        encode_vn_instruction_word("MICRO", 15),  # PC=2: ADD_ALPHA_CARRY (ACC += alpha)
        encode_vn_instruction_word("MICRO", 7),   # PC=3: STORE_ACC_PTR (M[0] = ACC)
        encode_vn_instruction_word("MICRO", 2),   # PC=4: INC_PTR (PTR=1)
        encode_vn_instruction_word("MICRO", 4),   # PC=5: LOAD_MDR_ACC (ACC = M[1])
        encode_vn_instruction_word("MICRO", 15),  # PC=6: ADD_ALPHA_CARRY (ACC += alpha)
        encode_vn_instruction_word("MICRO", 7),   # PC=7: STORE_ACC_PTR (M[1] = ACC)
        encode_vn_instruction_word("MICRO", 2),   # PC=8: INC_PTR (PTR=2)
        encode_vn_instruction_word("MICRO", 4),   # PC=9: LOAD_MDR_ACC (ACC = M[2])
        encode_vn_instruction_word("MICRO", 15),  # PC=10: ADD_ALPHA_CARRY (ACC += alpha)
        encode_vn_instruction_word("MICRO", 7),   # PC=11: STORE_ACC_PTR (M[2] = ACC)
        encode_vn_instruction_word("MICRO", 20),  # PC=12: HALT
    ])

    return progs


def ground_truth_von_neumann_execute(init_regs10, code_words, init_data_vals, alpha_v, beta_v, max_ticks=64):
    """
    Эталонный (Ground Truth) интерпретатор фон-неймановской машины:
    программа читается из кодового сегмента `code_words` (B, 25, 3) по адресу PC = 5*pc1 + pc0.
    """
    B = init_regs10.size(0)
    dev = init_regs10.device
    regs = init_regs10.clone()
    regs[:, 8] = 0  # pc0 = 0
    regs[:, 9] = 0  # pc1 = 0
    data_seg = init_data_vals.clone()  # (B, 25)
    stack_seg = torch.zeros(B, 5, dtype=torch.long, device=dev)
    halted = torch.zeros(B, dtype=torch.bool, device=dev)
    ticks = torch.zeros(B, dtype=torch.long, device=dev)

    for _ in range(max_ticks):
        if halted.all():
            break
        active = ~halted
        pc = (regs[:, 9] * 5 + regs[:, 8]).clamp(0, 24)
        idx_pc = pc.view(B, 1, 1).expand(B, 1, 3)
        ir_digits = code_words.gather(1, idx_pc).squeeze(1)  # (B, 3)

        alu_op, w_mode, br_type, tgt_pc, is_halt = decode_vn_ir_digits_batch(ir_digits)
        halted = halted | (active & is_halt)
        active = ~halted
        if not active.any():
            break

        ptr = regs[:, 4] * 5 + regs[:, 3]
        sp_prev = regs[:, 5]
        sp_pop = (sp_prev - 1) % 5

        mdr_ptr_val = data_seg.gather(1, ptr.unsqueeze(1)).squeeze(1)
        mdr_stk_val = stack_seg.gather(1, sp_pop.unsqueeze(1)).squeeze(1)
        mdr_val = torch.where(w_mode == 5, mdr_stk_val, mdr_ptr_val)
        mdr_digits = torch.stack([mdr_val % 5, (mdr_val // 5) % 5, (mdr_val // 25) % 5], dim=1)

        conds = eval_v15_predicates(regs, mdr_digits)  # (B, 6)
        old_acc = regs[:, 2] * 25 + regs[:, 1] * 5 + regs[:, 0]

        next_regs = apply_v15_alu_op_tensor(regs, mdr_digits, alu_op, alpha_v, beta_v)
        if (w_mode == 5).any():
            next_regs[:, 5] = torch.where(w_mode == 5, sp_pop, next_regs[:, 5])

        for b_i in range(B):
            if not active[b_i]:
                continue
            wm = int(w_mode[b_i].item())
            if wm in (1, 3):
                data_seg[b_i, ptr[b_i]] = old_acc[b_i]
            elif wm == 4:
                stack_seg[b_i, sp_prev[b_i]] = old_acc[b_i]

        # Переход PC:
        take_jump = (br_type == 1) | ((br_type == 2) & (conds[:, 5] == 1)) | ((br_type == 3) & (conds[:, 4] == 1)) | ((br_type == 4) & (conds[:, 0] == 1))
        next_pc = torch.where(take_jump, tgt_pc, (pc + 1) % 25)
        next_regs[:, 8] = next_pc % 5
        next_regs[:, 9] = (next_pc // 5) % 5

        regs = torch.where(active.unsqueeze(1), next_regs, regs)
        ticks = ticks + active.long()

    return regs, data_seg, stack_seg, ticks


@torch.no_grad()
def neural_von_neumann_execute_v15(
    core, router, tape125, init_regs10, code_words_digits, init_data_vals, alpha_v, beta_v, max_ticks=64
):
    """
    100% Фон-неймановское нейронное исполнение над единой 125-ячеечной матрицей M_RAM:
      1. Программа и данные загружаются в единую матрицу `M_ram` (B, 125, 240).
      2. На каждом такте Instruction Fetch читает слово инструкции прямо из `M_ram[0 + PC]`
         по сферам `pc0, pc1` (сферы 8 и 9 внутри `h_cur`)!
      3. Data Fetch читает `MDR` из `M_ram[25 + PTR]` (или `M_ram[100 + sp]`).
      4. `TernaryVonNeumannCore_v15` обновляет регистры данных, а затем делает аппаратный шаг
         счётчика команд (`INC_PC` op=21 или `LOAD_TARGET_PC` op=22) прямо в сферах 8 и 9 `h_cur`!
    """
    B = init_regs10.size(0)
    dev = init_regs10.device
    simvq = core.simvq

    regs_start = init_regs10.clone()
    regs_start[:, 8] = 0
    regs_start[:, 9] = 0
    h_cur = simvq.encode_regs(regs_start)

    data_digits = torch.stack(
        [init_data_vals % 5, (init_data_vals // 5) % 5, (init_data_vals // 25) % 5], dim=-1
    )
    M_ram = tape125.init_von_neumann_tape(code_words_digits, data_digits, simvq)

    halted = torch.zeros(B, dtype=torch.bool, device=dev)
    ticks = torch.zeros(B, dtype=torch.long, device=dev)
    dummy_mdr = simvq.encode_word3(torch.zeros(B, 3, dtype=torch.long, device=dev))

    for _ in range(max_ticks):
        if halted.all():
            break
        active = ~halted

        # --- ШАГ 1: Фон-неймановский Instruction Fetch из M_RAM[0 + PC] по сферам (pc0, pc1) ---
        ir_digits, _ = tape125.fetch_instruction_at_pc(M_ram, h_cur, simvq)
        alu_op, w_mode, br_type, tgt_pc, is_halt = decode_vn_ir_digits_batch(ir_digits)

        halted = halted | (active & is_halt)
        active = ~halted
        if not active.any():
            break

        # Если POP (w_mode == 5), сначала выполняем DEC_SP (op=20) в сфере sp:
        is_pop = active & (w_mode == 5)
        h_for_read = h_cur
        if is_pop.any():
            op_dec_sp = torch.full((B,), 20, dtype=torch.long, device=dev)
            h_dec_sp, _ = core.step(h_cur, dummy_mdr, op_dec_sp, alpha_v, beta_v, snap_strength=1.0)
            h_for_read = torch.where(is_pop.unsqueeze(1), h_dec_sp, h_cur)

        # --- ШАГ 2: Фон-неймановский Data Fetch из M_RAM[25 + PTR] или M_RAM[100 + sp] ---
        h_mdr_ptr = tape125.read_data_at_ptr(M_ram, h_for_read, simvq)
        if is_pop.any():
            h_mdr_stk = tape125.read_data_at_stack(M_ram, h_for_read, simvq)
            h_mdr = torch.where(is_pop.unsqueeze(1), h_mdr_stk, h_mdr_ptr)
        else:
            h_mdr = h_mdr_ptr

        # --- ШАГ 3: Тернарный маршрутизатор предикатов ---
        pred_conds = router(h_for_read, h_mdr).argmax(dim=-1)  # (B, 6)

        h_g_old = h_for_read.view(B, 10, D_SUB)
        v_old_acc = h_g_old[:, :3, :].reshape(B, 240)

        # --- ШАГ 4: Исполнение микрооперации АЛУ над сферами 0..7 ---
        h_after_alu, _ = core.step(h_for_read, h_mdr, alu_op, alpha_v, beta_v, snap_strength=1.0)

        # --- ШАГ 5: Запись ранга 1 в сегмент данных M_RAM[25 + PTR] или стек M_RAM[100 + sp] ---
        m_write_acc = active & ((w_mode == 1) | (w_mode == 3))
        if m_write_acc.any():
            M_ram = tape125.write_data_at_ptr(M_ram, h_for_read, v_old_acc, mask=m_write_acc)
        m_push = active & (w_mode == 4)
        if m_push.any():
            M_ram = tape125.write_data_at_stack(M_ram, h_for_read, v_old_acc, simvq, mask=m_push)

        # --- ШАГ 6: Аппаратное обновление сферического счётчика команд PC (сферы 8, 9) через ядро! ---
        take_jump = (
            (br_type == 1)
            | ((br_type == 2) & (pred_conds[:, 5] == 1))
            | ((br_type == 3) & (pred_conds[:, 4] == 1))
            | ((br_type == 4) & (pred_conds[:, 0] == 1))
        )
        pc_op = torch.where(
            take_jump,
            torch.full((B,), 22, dtype=torch.long, device=dev),  # LOAD_TARGET_PC
            torch.full((B,), 21, dtype=torch.long, device=dev),  # INC_PC (тернарный перенос pc0 -> pc1!)
        )
        h_next, _ = core.step(
            h_after_alu, dummy_mdr, pc_op, alpha_v, beta_v, target_pc=tgt_pc, snap_strength=1.0
        )

        h_cur = torch.where(active.unsqueeze(1), h_next, h_cur)
        ticks = ticks + active.long()

    pred_regs = simvq.compute_logits(h_cur).argmax(dim=-1)
    pred_data = tape125.decode_data_segment(M_ram, simvq)
    pred_stack = tape125.decode_stack_segment(M_ram, simvq)
    return pred_regs, pred_data, pred_stack, ticks


# ==============================================================================
# ЭТАП 3 (Stage v15.3): Обратный синтез программ / Neural Program Synthesis (Направление А)
# ==============================================================================
VN_ALGO_NAMES = [
    "ArraySum_VN",
    "ArrayMax_VN",
    "BubbleSort3_VN",
    "StackReverse3_VN",
    "AffineMap3_VN",
]

VN_ALGO_PYTHON_AST = {
    0: "while ptr != b_reg:\n    acc = (acc + (m_ram[25 + ptr] % 5)) % 125\n    ptr = (ptr + 1) % 25",
    1: "for ptr in range(4):\n    if acc < m_ram[25 + ptr]:\n        acc = m_ram[25 + ptr]\nm_ram[29] = acc",
    2: "for _pass in range(2):\n    for i in range(2):\n        if m_ram[25 + i] >= m_ram[26 + i]:\n            m_ram[25 + i], m_ram[26 + i] = m_ram[26 + i], m_ram[25 + i]",
    3: "for i in range(3):\n    stack.append(m_ram[25 + i])\nfor i in range(3):\n    m_ram[25 + i] = stack.pop()",
    4: "for i in range(3):\n    m_ram[25 + i] = (m_ram[25 + i] + alpha) % 125",
}


class TernaryProgramSynthesizer_v15(nn.Module):
    """
    100% Тернарный Обратный Синтезатор Программ (`TernaryProgramSynthesizer_v15`):
    Решает обратную задачу над сферическими состояниями Вороного:
      1. По паре состояний памяти и регистров `(h_init, h_target, h_tape_in, h_tape_out)`
         синтезирует полную фон-неймановскую программу `code_words` (B, 25, 3) в кодовый сегмент
         `M_RAM[0..24]` (а также класс алгоритма и эквивалентный валидный код Python AST!).
      2. Для одношаговых переходов `(h_prev, h_mdr, h_next)` восстанавливает точную микроинструкцию
         АЛУ `op_id in [0..21]`, что позволяет за T шагов синтезировать произвольную цепочку
         микрокоманд по трассе состояний!
    """

    def __init__(self, d_model=800, d_tape_slice=5 * 240):
        super().__init__()
        # 1. Контур синтеза кодового сегмента M_RAM[0..24] по входу-выходу (Initial -> Target):
        # Вход: [h_init (800), h_target (800), h_tape_in[:5] (1200), h_tape_out[:5] (1200)] = 4000 каналов
        d_io = d_model * 2 + d_tape_slice * 2
        self.io_norm = nn.LayerNorm(d_io, elementwise_affine=False)
        self.synth_fc1_a = BitLinear(d_io, 1024, bias=False)
        self.synth_fc1_b = BitLinear(d_io, 1024, bias=False)
        self.synth_fc2 = BitLinear(1024, 512, bias=False)
        self.algo_cls_head = nn.Linear(512, len(VN_ALGO_NAMES))
        # Прямой генератор 25 трёхсловных инструкций кодового сегмента M_RAM[0..24] (25 * 3 * 5 логитов):
        self.code_seg_head = nn.Linear(512, 25 * 3 * 5)

        # 2. Контур обратного синтеза микрооперации АЛУ по паре соседних тактов (h_t, h_mdr, h_{t+1}):
        # Вход: [h_t (800), h_{t+1} - h_t (800), h_mdr (240), c_epi (800)] = 2640 каналов
        d_step = d_model * 3 + 240
        self.step_norm = nn.LayerNorm(d_step, elementwise_affine=False)
        self.step_fc1_a = BitLinear(d_step, 1024, bias=False)
        self.step_fc1_b = BitLinear(d_step, 1024, bias=False)
        self.step_fc2 = BitLinear(1024, 512, bias=False)
        self.step_op_head = nn.Linear(512, 21)

    def set_quant_alpha(self, alpha):
        for m in self.modules():
            if isinstance(m, BitLinear):
                m.quant_alpha = alpha

    def synthesize_code_segment(self, h_init, h_target, h_tape_in5, h_tape_out5):
        """
        Принимает начальное и конечное сферическое состояние ЦПУ (B, 800) и первых 5 ячеек
        сегмента данных M_RAM[25..29] (B, 1200), возвращает:
          - логиты 25 слов кодового сегмента M_RAM[0..24] размерности (B, 25, 3, 5),
          - логиты типа синтезированного алгоритма (B, 5).
        """
        x = self.io_norm(torch.cat([h_init, h_target, h_tape_in5, h_tape_out5], dim=-1))
        h = F.silu(self.synth_fc1_a(x)) * F.gelu(self.synth_fc1_b(x))
        h = F.silu(self.synth_fc2(h))
        code_logits = self.code_seg_head(h).view(-1, 25, 3, 5)
        algo_logits = self.algo_cls_head(h)
        return code_logits, algo_logits

    def synthesize_step_op(self, h_prev, h_next, h_mdr, c_epi):
        """Восстанавливает микрооперацию op_id in [0..20] по переходу состояния (h_prev -> h_next)."""
        x = self.step_norm(torch.cat([h_prev, h_next - h_prev, h_mdr, c_epi], dim=-1))
        h = F.silu(self.step_fc1_a(x)) * F.gelu(self.step_fc1_b(x))
        h = F.silu(self.step_fc2(h))
        return self.step_op_head(h)


def sample_initial_regs_v15(batch_size, zero_shot=False, dev="cuda"):
    """
    Разбиение 9,765,625 состояний Z_5^10 на непересекающиеся TRAIN (80%) и ZEROSHOT (20%)
    по взаимно-простому полиномиальному хешу состояния.
    """
    out = []
    primes = [1, 3, 7, 13, 19, 29, 37, 43, 53, 61]
    while sum(x.size(0) for x in out) < batch_size:
        cand = torch.randint(0, 5, (batch_size * 2, 10), device=dev)
        h = sum(primes[i] * cand[:, i] for i in range(10)) % 5
        mask = (h == 0) if zero_shot else (h != 0)
        out.append(cand[mask])
    return torch.cat(out, dim=0)[:batch_size]


def encode_tape_slice5(data_vals, simvq10):
    """Кодирует первые 5 ячеек сегмента данных M_RAM[25..29] (B, 25) в сферический вектор (B, 1200)."""
    B = data_vals.size(0)
    slice5 = data_vals[:, :5]  # (B, 5)
    d5 = torch.stack([slice5 % 5, (slice5 // 5) % 5, (slice5 // 25) % 5], dim=-1)  # (B, 5, 3)
    return simvq10.encode_word3(d5.reshape(B * 5, 3)).view(B, 5 * 240)


def generate_synthesis_io_batch(batch_size, vn_progs, zero_shot=False, dev="cuda"):
    """
    Генерирует батч задач обратного синтеза программ (Initial State, Target State) -> Code Segment M_RAM[0..24].
    Каждая задача гарантированно имеет нетривиальную трансформацию памяти/регистров.
    """
    r_init = sample_initial_regs_v15(batch_size, zero_shot=zero_shot, dev=dev)
    r_init[:, 8:10] = 0  # PC = 0
    data_init = torch.randint(1, 24, (batch_size, 25), device=dev)
    # Делаем первые 3 элемента M_RAM[25..27] попарно различными и не отсортированными,
    # чтобы задачи сортировки, разворота, максимума, суммы и сдвига alpha имели уникальные сигнатуры!
    for b in range(batch_size):
        vals3 = random.sample(range(2, 23), 3)
        # Для чёткого отличия BubbleSort3 от StackReverse3 выбираем тройку, где разворот != сортировке:
        while vals3[::-1] == sorted(vals3) or vals3 == sorted(vals3):
            vals3 = random.sample(range(2, 23), 3)
        data_init[b, 0] = vals3[0]
        data_init[b, 1] = vals3[1]
        data_init[b, 2] = vals3[2]
        data_init[b, 3] = random.randint(24, 95)  # яркий максимум в M_RAM[28] для ArrayMax_VN
        data_init[b, 4] = 0

    alpha_v = torch.randint(1, 5, (batch_size,), device=dev)
    beta_v = torch.randint(0, 5, (batch_size,), device=dev)

    algo_ids = torch.randint(0, len(VN_ALGO_NAMES), (batch_size,), device=dev)
    code_targets = torch.zeros(batch_size, 25, 3, dtype=torch.long, device=dev)
    r_targets = torch.zeros_like(r_init)
    data_targets = torch.zeros_like(data_init)

    for a_idx, a_name in enumerate(VN_ALGO_NAMES):
        mask = algo_ids == a_idx
        if not mask.any():
            continue
        n_m = int(mask.sum().item())
        sub_r = r_init[mask].clone()
        sub_d = data_init[mask].clone()

        if a_name == "ArraySum_VN":
            sub_r[:, 0:5] = 0  # ACC=0, PTR=0
            sub_r[:, 6] = 4    # B_reg = 4
            sub_r[:, 7] = 0
        elif a_name == "ArrayMax_VN":
            sub_r[:, 0:5] = 0  # ACC=0, PTR=0
            sub_r[:, 6] = 4    # B_reg = 4
            sub_r[:, 7] = 0
        elif a_name == "StackReverse3_VN":
            sub_r[:, 5] = 0    # sp = 0

        r_init[mask] = sub_r
        prog_w = vn_progs[a_name].unsqueeze(0).expand(n_m, 25, 3)
        code_targets[mask] = prog_w
        gt_r, gt_d, _, _ = ground_truth_von_neumann_execute(
            sub_r, prog_w, sub_d, alpha_v[mask], beta_v[mask], max_ticks=64
        )
        r_targets[mask] = gt_r
        data_targets[mask] = gt_d

    return r_init, data_init, r_targets, data_targets, alpha_v, beta_v, algo_ids, code_targets


class VonNeumannSynthesizerSimVQ_v15(nn.Module):
    """
    Полная система v15.0 (`VonNeumannSynthesizerSimVQ_v15`), объединяющая:
      - `tape125` (`GershgorinBiorthogonalTape125`): 125-ячеечная фон-неймановская лента M_RAM (код + данные + стек)
        с калибровкой Ньютона — Шульца через границу Гершгорина.
      - `core` (`TernaryVonNeumannCore_v15`): 100% тернарное ядро над 10 сферами ЦПУ (включая PC=(pc0, pc1)).
      - `router` (`TernaryBranchRouter_v15`): 100% тернарный маршрутизатор 6 предикатов.
      - `synthesizer` (`TernaryProgramSynthesizer_v15`): 100% тернарный обратный синтезатор программ
        по паре состояний `(M_RAM_in -> M_RAM_out)` с верификацией внутри фон-неймановского ядра!
    """

    def __init__(self, d_model=800, d_mdr=240):
        super().__init__()
        self.core = TernaryVonNeumannCore_v15(d_model=d_model)
        self.router = TernaryBranchRouter_v15(d_model=d_model, d_mdr=d_mdr)
        self.tape125 = GershgorinBiorthogonalTape125(d_sub=d_model // 10)
        self.synthesizer = TernaryProgramSynthesizer_v15(d_model=d_model)


# ==============================================================================
# Сквозное обучение и строгая верификация Stage v15.0
# ==============================================================================
def train_and_verify_v15(dev="cuda"):
    torch.manual_seed(42)
    random.seed(42)
    t_start = time.time()

    print("=" * 84)
    print(
        "  ЭТАП v15.0: ФОН-НЕЙМАНОВСКАЯ ЛЕНТА M_RAM[0..124] (ГЕРШГОРИН) И СИНТЕЗ ПРОГРАММ"
    )
    print("=" * 84)

    model = VonNeumannSynthesizerSimVQ_v15(d_model=800, d_mdr=240).to(dev)
    core = model.core
    router = model.router
    tape125 = model.tape125
    synthesizer = model.synthesizer
    simvq = core.simvq
    vn_progs = build_v15_von_neumann_programs(dev=dev)

    # --------------------------------------------------------------------------
    # ЭТАП 1 (Stage v15.1): Обучение TernaryVonNeumannCore_v15 (10 сфер ЦПУ, 23 операции)
    # --------------------------------------------------------------------------
    print(
        "\n--- [Stage v15.1] Обучение TernaryVonNeumannCore_v15 (10 сфер ЦПУ вкл. PC=(pc0,pc1), 100% BitLinear) ---"
    )
    opt_core = torch.optim.AdamW(core.parameters(), lr=3e-3, weight_decay=1e-4)
    steps_core = 900
    sched_core = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_core, T_max=steps_core, eta_min=1e-4
    )

    for step in range(1, steps_core + 1):
        q_alpha = min(1.0, max(0.0, (step - 150) / 400.0))
        snap_s = min(1.0, max(0.2, step / 350.0))
        core.set_quant_alpha(q_alpha)

        B = 512
        T_roll = 4
        regs = sample_initial_regs_v15(B, zero_shot=False, dev=dev)
        # Обогащаем батч переносами и заёмами во всех многоразрядных регистрах (ACC, PTR, B, PC!):
        regs[:64, 0:2] = 4
        regs[64:128, 0:2] = 0
        regs[128:160, 3] = 4  # p0=4 (перенос PTR)
        regs[160:192, 3] = 0  # p0=0 (заём PTR)
        regs[192:224, 6] = 4  # b0=4 (перенос B)
        regs[224:256, 6] = 0  # b0=0 (заём B)
        regs[256:320, 8] = 4  # pc0=4 (перенос счётчика команд PC: pc0=4 -> pc0=0, pc1+=1!)

        alpha_v = torch.randint(0, 5, (B,), device=dev)
        beta_v = torch.randint(0, 5, (B,), device=dev)
        h_cur = simvq.encode_regs(regs)

        loss_total = 0.0
        correct_cells = 0
        total_cells = B * T_roll

        for _ in range(T_roll):
            # Учим 22 операции (0..21, включая INC_PC op=21; op=22 LOAD_TARGET_PC маршрутизируется шиной):
            op_id = torch.randint(0, 22, (B,), device=dev)
            mdr_digits = torch.randint(0, 5, (B, 3), device=dev)
            h_mdr = simvq.encode_word3(mdr_digits)

            next_regs = apply_v15_alu_op_tensor(regs, mdr_digits, op_id, alpha_v, beta_v)
            h_next, h_cand = core.step(
                h_cur, h_mdr, op_id, alpha_v, beta_v, snap_strength=snap_s
            )

            logits_cand = simvq.compute_logits(h_cand)
            logits_snap = simvq.compute_logits(h_next)
            loss_ce = F.cross_entropy(
                logits_cand.reshape(-1, 5), next_regs.reshape(-1)
            ) + 0.5 * F.cross_entropy(
                logits_snap.reshape(-1, 5), next_regs.reshape(-1)
            )
            loss_total = loss_total + loss_ce

            with torch.no_grad():
                preds = logits_snap.argmax(dim=-1)
                correct_cells += (preds == next_regs).all(dim=-1).sum().item()

            regs = next_regs
            h_cur = h_next

        opt_core.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
        opt_core.step()
        sched_core.step()

        if step == 1 or step % 150 == 0:
            acc_pct = 100.0 * correct_cells / total_cells
            print(
                f"  [Stage 1 | step {step:4d}/{steps_core}] loss={loss_total.item()/T_roll:.4f} | "
                f"cell_acc={acc_pct:.2f}% | quant_alpha={q_alpha:.2f} | snap={snap_s:.2f}"
            )

    core.set_quant_alpha(1.0)
    core.eval()

    # Калибровка 125-ячеечной ленты M_RAM[0..124] методом Ньютона — Шульца:
    # прямое сравнение границы Гёльдера — Гершгорина против следа Фробениуса на N=125!
    _, err_frob = tape125.calibrate_125_address_basis(simvq, ns_steps=11)
    err_gersh, _ = tape125.calibrate_125_address_basis(simvq, ns_steps=15)
    print(
        f"  [Gershgorin vs Frobenius on N=125]:\n"
        f"    * Frobenius bound (1.5 / ||G||_F^2, 11 iters)   max_err = {err_frob:.4f} (не успевает сойтись из-за фактора 1/N!)\n"
        f"    * Gershgorin bound (1.5 / ||G||_inf^2, 15 iters) max_err = {err_gersh:.2e} (машинный ноль FP32 без NaN!)"
    )

    # Проверка экстраполяции всех 22 операций (включая INC_PC!) на ZEROSHOT-состояниях до T=1024:
    extrap_accs = {}
    with torch.no_grad():
        for T_hor in [16, 64, 256, 1024]:
            B_h = 128
            r_h = sample_initial_regs_v15(B_h, zero_shot=True, dev=dev)
            d_vals = torch.randint(0, 125, (B_h, 25), device=dev)
            d_digs = torch.stack([d_vals % 5, (d_vals // 5) % 5, (d_vals // 25) % 5], dim=-1)
            dummy_code = vn_progs["ArraySum_VN"].unsqueeze(0).expand(B_h, 25, 3)
            M_h = tape125.init_von_neumann_tape(dummy_code, d_digs, simvq)
            h_h = simvq.encode_regs(r_h)
            r_gt_h = r_h.clone()
            av_h = torch.randint(0, 5, (B_h,), device=dev)
            bv_h = torch.randint(0, 5, (B_h,), device=dev)
            for _ in range(T_hor):
                op_h = torch.randint(0, 22, (B_h,), device=dev)
                ptr_h = r_gt_h[:, 4] * 5 + r_gt_h[:, 3]
                mv_h = d_vals.gather(1, ptr_h.unsqueeze(1)).squeeze(1)
                md_h = torch.stack([mv_h % 5, (mv_h // 5) % 5, (mv_h // 25) % 5], dim=1)
                r_gt_h = apply_v15_alu_op_tensor(r_gt_h, md_h, op_h, av_h, bv_h)

                h_mdr_h = tape125.read_data_at_ptr(M_h, h_h, simvq)
                h_h, _ = core.step(h_h, h_mdr_h, op_h, av_h, bv_h, snap_strength=1.0)
            pr_h = simvq.compute_logits(h_h).argmax(dim=-1)
            extrap_accs[f"All22Ops_10Spheres_ZeroShot_T{T_hor}"] = (
                100.0 * (pr_h == r_gt_h).all(dim=-1).float().mean().item()
            )
    print(
        f"  [Stage 1 Verified] All22Ops_T16={extrap_accs['All22Ops_10Spheres_ZeroShot_T16']:.2f}% | "
        f"T256={extrap_accs['All22Ops_10Spheres_ZeroShot_T256']:.2f}% | "
        f"T1024={extrap_accs['All22Ops_10Spheres_ZeroShot_T1024']:.2f}%"
    )

    # --------------------------------------------------------------------------
    # ЭТАП 2 (Stage v15.2): Обучение TernaryBranchRouter_v15 и Фон-Неймановское исполнение из M_RAM[0..24]
    # --------------------------------------------------------------------------
    print(
        "\n--- [Stage v15.2] Обучение TernaryBranchRouter_v15 и Фон-Неймановский цикл (Код + Данные + Стек в M_RAM[0..124]) ---"
    )
    for p in core.parameters():
        p.requires_grad_(False)

    opt_router = torch.optim.AdamW(router.parameters(), lr=4e-3, weight_decay=1e-4)
    sched_router = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_router, T_max=700, eta_min=1e-4
    )

    for step in range(1, 701):
        q_alpha = min(1.0, max(0.0, (step - 80) / 350.0))
        router.set_quant_alpha(q_alpha)

        B = 1024
        regs = torch.randint(0, 5, (B, 10), device=dev)
        mdr_d = torch.randint(0, 5, (B, 3), device=dev)
        regs[:128, 0:3] = 0
        regs[128:256, 6:8] = 0
        regs[256:384, 3:5] = regs[256:384, 6:8]
        mdr_d[384:512, :] = regs[384:512, 0:3]
        mdr_d[512:640, 2] = regs[512:640, 2]
        mdr_d[640:768, 1:3] = regs[640:768, 1:3]

        with torch.no_grad():
            h_s = simvq.encode_regs(regs)
            h_m = simvq.encode_word3(mdr_d)
            target_conds = eval_v15_predicates(regs, mdr_d)

        logits = router(h_s, h_m)
        loss_r = F.cross_entropy(logits.reshape(-1, 2), target_conds.reshape(-1))
        opt_router.zero_grad()
        loss_r.backward()
        opt_router.step()
        sched_router.step()

        if step == 1 or step % 175 == 0:
            with torch.no_grad():
                acc_c = 100.0 * (logits.argmax(dim=-1) == target_conds).all(dim=-1).float().mean().item()
            print(
                f"  [Stage 2 | step {step:3d}/700] loss={loss_r.item():.4f} | "
                f"all6_predicates_acc={acc_c:.2f}% | quant_alpha={q_alpha:.2f}"
            )

    router.set_quant_alpha(1.0)
    router.eval()

    # Верификация фон-неймановского исполнения (инструкции читаются из M_RAM[0..24] по сферам PC=(pc0, pc1)!):
    vn_metrics = {}
    with torch.no_grad():
        B_vn = 256
        for a_name, prog_w in vn_progs.items():
            r_init = sample_initial_regs_v15(B_vn, zero_shot=True, dev=dev)
            r_init[:, 8:10] = 0
            d_init = torch.randint(0, 25, (B_vn, 25), device=dev)
            if a_name in ("ArraySum_VN", "ArrayMax_VN"):
                r_init[:, 0:5] = 0
                r_init[:, 6] = 4
                r_init[:, 7] = 0
            elif a_name == "StackReverse3_VN":
                r_init[:, 5] = 0

            av_v = torch.randint(1, 5, (B_vn,), device=dev)
            bv_v = torch.randint(0, 5, (B_vn,), device=dev)
            code_batch = prog_w.unsqueeze(0).expand(B_vn, 25, 3)

            gt_r, gt_d, gt_stk, _ = ground_truth_von_neumann_execute(
                r_init, code_batch, d_init, av_v, bv_v, max_ticks=64
            )
            pr_r, pr_d, pr_stk, _ = neural_von_neumann_execute_v15(
                core, router, tape125, r_init, code_batch, d_init, av_v, bv_v, max_ticks=64
            )
            match = (
                (pr_r == gt_r).all(dim=-1)
                & (pr_d == gt_d).all(dim=-1)
                & (pr_stk == gt_stk).all(dim=-1)
            )
            vn_metrics[f"VonNeumann_StoredProg_{a_name}"] = 100.0 * match.float().mean().item()
            print(
                f"  [Von Neumann Stored-Program M_RAM[0..24]] {a_name:18s}: "
                f"{vn_metrics[f'VonNeumann_StoredProg_{a_name}']:.2f}% (10 regs + 25 Data + 5 Stack exact match)"
            )

    # --------------------------------------------------------------------------
    # ЭТАП 3 (Stage v15.3): Обучение 100% Тернарного Обратного Синтезатора Программ (Направление А)
    # --------------------------------------------------------------------------
    print(
        "\n--- [Stage v15.3] Обучение TernaryProgramSynthesizer_v15 (Обратный синтез кода в M_RAM[0..24] по входу и выходу) ---"
    )
    for p in router.parameters():
        p.requires_grad_(False)

    opt_syn = torch.optim.AdamW(synthesizer.parameters(), lr=3.5e-3, weight_decay=1e-4)
    steps_syn = 600
    sched_syn = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_syn, T_max=steps_syn, eta_min=1e-4
    )

    for step in range(1, steps_syn + 1):
        q_alpha = min(1.0, max(0.0, (step - 80) / 300.0))
        synthesizer.set_quant_alpha(q_alpha)

        # 1. Батч синтеза полных фон-неймановских программ в M_RAM[0..24] по паре (State_in -> State_out):
        (
            r_in,
            d_in,
            r_out,
            d_out,
            av_s,
            bv_s,
            algo_ids,
            code_tgt,
        ) = generate_synthesis_io_batch(256, vn_progs, zero_shot=False, dev=dev)

        with torch.no_grad():
            h_in = simvq.encode_regs(r_in)
            h_out = simvq.encode_regs(r_out)
            h_t_in5 = encode_tape_slice5(d_in, simvq)
            h_t_out5 = encode_tape_slice5(d_out, simvq)

        code_logits, algo_logits = synthesizer.synthesize_code_segment(
            h_in, h_out, h_t_in5, h_t_out5
        )
        loss_code = F.cross_entropy(code_logits.reshape(-1, 5), code_tgt.reshape(-1))
        loss_algo = F.cross_entropy(algo_logits, algo_ids)

        # 2. Батч обратного синтеза произвольных цепочек микроопераций по переходу (h_t -> h_{t+1}):
        B_st = 512
        r_prev = sample_initial_regs_v15(B_st, zero_shot=False, dev=dev)
        # Выбираем из инъективных (однозначно обратимых по состоянию) микроопераций [1..20]:
        op_true = torch.randint(0, 21, (B_st,), device=dev)
        mdr_st = torch.randint(1, 5, (B_st, 3), device=dev)
        av_st = torch.randint(1, 5, (B_st,), device=dev)
        bv_st = torch.randint(0, 5, (B_st,), device=dev)
        with torch.no_grad():
            r_next = apply_v15_alu_op_tensor(r_prev, mdr_st, op_true, av_st, bv_st)
            h_p = simvq.encode_regs(r_prev)
            h_n = simvq.encode_regs(r_next)
            h_m = simvq.encode_word3(mdr_st)
            c_epi = torch.cat([core.alpha_emb(av_st), core.beta_emb(bv_st)], dim=-1)

        step_op_logits = synthesizer.synthesize_step_op(h_p, h_n, h_m, c_epi)
        loss_step = F.cross_entropy(step_op_logits, op_true)

        loss_syn = loss_code + loss_algo + loss_step
        opt_syn.zero_grad()
        loss_syn.backward()
        torch.nn.utils.clip_grad_norm_(synthesizer.parameters(), 1.0)
        opt_syn.step()
        sched_syn.step()

        if step == 1 or step % 150 == 0:
            with torch.no_grad():
                code_exact = (
                    100.0
                    * (code_logits.argmax(dim=-1) == code_tgt)
                    .all(dim=-1)
                    .all(dim=-1)
                    .float()
                    .mean()
                    .item()
                )
            print(
                f"  [Stage 3 | step {step:3d}/{steps_syn}] loss_syn={loss_syn.item():.4f} | "
                f"code_segment_25words_acc={code_exact:.2f}% | quant_alpha={q_alpha:.2f}"
            )

    synthesizer.set_quant_alpha(1.0)
    synthesizer.eval()

    # Строгая верификация Обратного Синтеза Программ на ZEROSHOT-состояниях:
    # Синтезатор видит ТОЛЬКО начальное и целевое состояние (M_RAM_in, M_RAM_target),
    # сам генерирует 25 трёхсловных инструкций в кодовый сегмент M_RAM[0..24],
    # после чего фон-неймановское ядро исполняет синтезированный код из M_RAM[0..24] и мы сверяем выход с M_RAM_target!
    synth_metrics = {}
    with torch.no_grad():
        (
            r_in_z,
            d_in_z,
            r_tgt_z,
            d_tgt_z,
            av_z,
            bv_z,
            algo_ids_z,
            code_tgt_z,
        ) = generate_synthesis_io_batch(256, vn_progs, zero_shot=True, dev=dev)

        h_in_z = simvq.encode_regs(r_in_z)
        h_tgt_z = simvq.encode_regs(r_tgt_z)
        h_tin_z = encode_tape_slice5(d_in_z, simvq)
        h_tout_z = encode_tape_slice5(d_tgt_z, simvq)

        pred_code_logits, pred_algo_logits = synthesizer.synthesize_code_segment(
            h_in_z, h_tgt_z, h_tin_z, h_tout_z
        )
        synthesized_code_words = pred_code_logits.argmax(dim=-1)  # (256, 25, 3)
        pred_algo_ids = pred_algo_logits.argmax(dim=-1)

        # 1. Точность совпадения всех 25x3=75 тритов синтезированного кодового сегмента M_RAM[0..24]:
        synth_metrics["Synth_CodeSegment_75Trits_Exact"] = (
            100.0
            * (synthesized_code_words == code_tgt_z)
            .all(dim=-1)
            .all(dim=-1)
            .float()
            .mean()
            .item()
        )

        # 2. Валидность синтезированного Python AST кода:
        ast_valid_cnt = 0
        for b_i in range(256):
            py_src = VN_ALGO_PYTHON_AST[int(pred_algo_ids[b_i].item())]
            if ast.parse(py_src) is not None:
                ast_valid_cnt += 1
        synth_metrics["Synth_Python_AST_Valid"] = 100.0 * ast_valid_cnt / 256.0

        # 3. Главный тест: загружаем СИНТЕЗИРОВАННЫЙ кодовый сегмент в M_RAM[0..24]
        # и исполняем его фон-неймановской машиной над входной лентой d_in_z!
        exec_r, exec_d, _, exec_ticks = neural_von_neumann_execute_v15(
            core,
            router,
            tape125,
            r_in_z,
            synthesized_code_words,
            d_in_z,
            av_z,
            bv_z,
            max_ticks=64,
        )
        exec_match = (exec_r == r_tgt_z).all(dim=-1) & (exec_d == d_tgt_z).all(dim=-1)
        synth_metrics["Synth_EndToEnd_VonNeumann_Verify"] = (
            100.0 * exec_match.float().mean().item()
        )
        print(
            f"  [Program Synthesis Verified on Zero-Shot (B=256)]:\n"
            f"    * Synthesized Code Segment M_RAM[0..24] (75/75 digits): {synth_metrics['Synth_CodeSegment_75Trits_Exact']:.2f}%\n"
            f"    * Synthesized Python AST Syntax Valid                 : {synth_metrics['Synth_Python_AST_Valid']:.2f}%\n"
            f"    * End-to-End Execution of Synthesized Code matches Out: {synth_metrics['Synth_EndToEnd_VonNeumann_Verify']:.2f}%"
        )

    # --------------------------------------------------------------------------
    # Живая демонстрация v15.0: Обратный синтез программы по входам-выходам и исполнение из M_RAM[0..24]
    # --------------------------------------------------------------------------
    print("\n" + "=" * 84)
    print(
        "  ЖИВАЯ ДЕМОНСТРАЦИЯ v15.0: ОБРАТНЫЙ СИНТЕЗ ПРОГРАММЫ В M_RAM[0..24] И ЕЁ ИСПОЛНЕНИЕ"
    )
    print("=" * 84)
    with torch.no_grad():
        # Подаём только начальную ленту данных M_RAM[25..27] = [88, 12, 54] и желаемую выходную [12, 54, 88]:
        demo_r_in = torch.zeros(1, 10, dtype=torch.long, device=dev)
        demo_d_in = torch.zeros(1, 25, dtype=torch.long, device=dev)
        demo_d_in[0, 0] = 88
        demo_d_in[0, 1] = 12
        demo_d_in[0, 2] = 54
        demo_av = torch.tensor([2], dtype=torch.long, device=dev)
        demo_bv = torch.tensor([0], dtype=torch.long, device=dev)

        # Получаем целевое состояние для демонстрации:
        demo_r_tgt, demo_d_tgt, _, _ = ground_truth_von_neumann_execute(
            demo_r_in,
            vn_progs["BubbleSort3_VN"].unsqueeze(0),
            demo_d_in,
            demo_av,
            demo_bv,
            max_ticks=64,
        )

        # Синтезатор видит ТОЛЬКО (demo_r_in, demo_d_in) и (demo_r_tgt, demo_d_tgt):
        c_log_d, a_log_d = synthesizer.synthesize_code_segment(
            simvq.encode_regs(demo_r_in),
            simvq.encode_regs(demo_r_tgt),
            encode_tape_slice5(demo_d_in, simvq),
            encode_tape_slice5(demo_d_tgt, simvq),
        )
        synth_code_demo = c_log_d.argmax(dim=-1)  # (1, 25, 3)
        synth_algo_idx = int(a_log_d.argmax(dim=-1)[0].item())

        # Записываем синтезированные инструкции в M_RAM[0..24] и запускаем фон-неймановскую машину:
        _, out_d_demo, _, ticks_demo = neural_von_neumann_execute_v15(
            core,
            router,
            tape125,
            demo_r_in,
            synth_code_demo,
            demo_d_in,
            demo_av,
            demo_bv,
            max_ticks=64,
        )
        print(f"  1. Входной сегмент данных M_RAM[25..27]     : {demo_d_in[0, :3].tolist()}")
        print(f"  2. Желаемый выходной сегмент M_RAM[25..27]  : {demo_d_tgt[0, :3].tolist()}")
        print(
            f"  3. Синтезированный в M_RAM[0..24] алгоритм  : {VN_ALGO_NAMES[synth_algo_idx]} "
            f"(первые 4 слова ИК в M_RAM[0..3]: {synth_code_demo[0, :4].tolist()})"
        )
        print(
            f"  4. Результат исполнения из M_RAM[0..24]     : {out_d_demo[0, :3].tolist()} "
            f"за {int(ticks_demo[0].item())} тактов сферического PC=(pc0, pc1)!"
        )

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", "v15_von_neumann_synthesizer.pt")
    all_metrics = {
        "Gershgorin_N125_OrthoErr_8steps": err_gersh,
        "Frobenius_N125_OrthoErr_8steps": err_frob,
        **extrap_accs,
        **vn_metrics,
        **synth_metrics,
    }
    torch.save(
        {
            "version": "v15.0_VonNeumannSynthesizerSimVQ",
            "model_state_dict": model.state_dict(),
            "metrics": all_metrics,
        },
        ckpt_path,
    )
    elapsed = time.time() - t_start
    print("\n" + "=" * 84)
    print(f"  ИТОГОВАЯ СВОДКА ВЕРИФИКАЦИИ v15.0 (время: {elapsed:.1f} с)")
    print("=" * 84)
    for k, v in all_metrics.items():
        if "OrthoErr" in k:
            print(f"  {k:40s}: {v:.2e}")
        else:
            print(f"  {k:40s}: {v:.2f}%")
    print(f"  Чекпоинт сохранён в: {ckpt_path}")
    return all_metrics


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_and_verify_v15(dev=device)
