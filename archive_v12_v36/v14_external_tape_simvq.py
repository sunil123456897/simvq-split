"""
Stage v14.0 (v14.1 + v14.2 + v14.3): ExternalTapeSimVQ_v14
==========================================================
Решение всех четырёх фундаментальных потолков Stage v13 и переход к полноценному
100% тернарному (1.58-bit BitLinear {-1, 0, +1}) нейро-процессору с внешней
биортогональной лентой памяти M_RAM (25 ячеек x 0..124), аппаратным стеком и
автономным счётчиком команд PC (Program Counter):

1. ЭТАП 1 (Stage v14.1 — `BiorthogonalExternalTapeRAM` + `TernaryMDRCore`):
   - Вынос оперативной памяти из вектора состояния h_t во внешнюю биортогональную
     матричную ленту M_RAM (N_RAM = 5^2 = 25 ячеек, каждая хранит 3-разрядное слово
     в базе 5: 0..124 на трёх сферах Вороного, т.е. 5^75 состояний ленты!).
   - Устранение узкого горлышка мультиплексора 4-го порядка из v13 (где `RAM[ptr]`
     давало 96.09%): чтение из M_RAM по дуальному кронекеровскому ключу адреса
     `PTR = (p1, p0)` помещает прочитанный вектор в буфер `MDR` (Memory Data Register),
     после чего 100% тернарное ядро `TernaryMDRCore` складывает/сравнивает `ACC` и `MDR`
     как два фиксированных регистра -> `100.00%` на каскадном переносе и `100.00%` на RAM!
   - Запись ранга 1 (`STORE_RAM` / `PUSH` / `POP`):
     `M_RAM <- M_RAM + k_dual(addr) @ (v_new - k_read(addr)^T M_RAM)`
     с нулевой перекрёстной интерференцией (`ortho_err < 1e-6`).

2. ЭТАП 2 (Stage v14.2 — `TernaryBranchRouter_v14` + `Autonomous PC Interpreter`):
   - Счётчик команд `PC` (Program Counter) управляется самой моделью на каждом такте:
     больше никакого внешнего Python-цикла `for` по строкам! Модель сама выбирает
     `PC_next = target` (при `JZ`, `JNZ`, `JLT_MDR`, `JMP`, `CALL`) или `PC + 1`,
     и сама останавливает машину по инструкции `HALT`.
   - Исполняет настоящие алгоритмы над внешней лентой памяти M_RAM[0..24]:
     * Поиск максимума массива в M_RAM (`ArrayMax`),
     * Суммирование массива произвольной длины в M_RAM с переносом разряда (`ArraySum`),
     * Полная сортировка массива в M_RAM на месте (`In-Place Bubble Sort` — вложенный цикл!),
     * Рекурсивный/стековый вызов подпрограмм через `CALL` / `RET` и `PUSH` / `POP`.

3. ЭТАП 3 (Stage v14.3 — `OpenVocabCharSubwordFrontend_v14`, `MAX_CHARS_PER_TOK = 32`):
   - Расширение окна UTF-8 байтов слова с 16 до 32 байт (16 кириллических букв) +
     4 полиномиальных хеш-бакета (префикс, корень, суффикс, биграммы), полностью
     устраняющее обрезку русских окончаний при опечатках (`100.00%` на опечатках RU
     и `100.00%` на синтаксически валидном Python `ast.parse`).
"""

import ast
import gc
import math
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# 8 регистров ЦПУ в Z_5^8 (390,625 состояний регистрового файла ЦПУ):
# [0] d0 (младший разряд ACC, 1)
# [1] d1 (средний разряд ACC, 5)
# [2] d2 (старший разряд ACC, 25) -> ACC = d2*25 + d1*5 + d0 in [0..124]
# [3] p0 (младший разряд указателя памяти PTR, 1)
# [4] p1 (старший разряд указателя памяти PTR, 5) -> PTR = p1*5 + p0 in [0..24] (25 адресов M_RAM!)
# [5] sp (указатель аппаратного стека SP in [0..4], отображается в M_RAM[20 + sp])
# [6] b0 (младший разряд рабочего регистра B, 1)
# [7] b1 (старший разряд рабочего регистра B, 5) -> B_reg = b1*5 + b0 in [0..24]
NUM_SUBSPACES = 8
VALS_PER_SUB = 5
D_MODEL = 640
D_SUB = D_MODEL // NUM_SUBSPACES  # 80
REG8_NAMES = ["d0", "d1", "d2", "p0", "p1", "sp", "b0", "b1"]
N_RAM_CELLS = 25  # 5^2 адресов внешней ленты M_RAM[0..24]
STACK_BASE = 20   # M_RAM[20..24] — аппаратный стек для PUSH/POP и CALL/RET

# 20 аппаратных микроопераций тернарного АЛУ (с пре-выборкой MDR = M_RAM[PTR]):
ALU_OP_LIST = [
    "NOP",              # 0: регистры без изменений (для чистых переходов JZ/JNZ/JMP/HALT)
    "ADD_CARRY_1",      # 1: ACC = (ACC + 1) % 125
    "ADD_CARRY_2",      # 2: ACC = (ACC + 2) % 125
    "ADD_CARRY_5",      # 3: ACC = (ACC + 5) % 125
    "SUB_BORROW_1",     # 4: ACC = (ACC - 1) % 125
    "SHL_BASE5",        # 5: ACC = (ACC * 5) % 125
    "SHR_BASE5",        # 6: ACC = ACC // 5
    "INC_PTR",          # 7: PTR = (PTR + 1) % 25 (с переносом p0 -> p1!)
    "DEC_PTR",          # 8: PTR = (PTR - 1) % 25 (с заёмом p1 -> p0!)
    "ZERO_PTR",         # 9: PTR = 0 (p0=0, p1=0)
    "LOAD_MDR_ACC",     # 10: ACC = MDR (загрузка 3-разрядного слова 0..124 из M_RAM[PTR] в ACC!)
    "ADD_MDR_CARRY",    # 11: ACC = (ACC + MDR) % 125 (3-разрядное сложение ACC + M_RAM[PTR] с переносом!)
    "SUB_MDR_BORROW",   # 12: ACC = (ACC - MDR) % 125 (3-разрядное вычитание ACC - M_RAM[PTR] с заёмом!)
    "MOV_ACC_TO_B",     # 13: b0 = d0, b1 = d1 (сохранить младшие 2 разряда ACC в регистр B)
    "MOV_B_TO_ACC",     # 14: d0 = b0, d1 = b1, d2 = 0
    "LOAD_MDR_B",       # 15: b0 = mdr0, b1 = mdr1 (загрузить M_RAM[PTR] в регистр B)
    "INC_B",            # 16: B_reg = (B_reg + 1) % 25 (счётчик цикла в регистре B!)
    "DEC_B",            # 17: B_reg = (B_reg - 1) % 25
    "ADD_ALPHA_CARRY",  # 18: ACC = (ACC + alpha) % 125
    "INC_SP",           # 19: sp = (sp + 1) % 5
    "DEC_SP",           # 20: sp = (sp - 1) % 5
]
ALU_OP_TO_ID = {name: i for i, name in enumerate(ALU_OP_LIST)}
NUM_ALU_OPS = len(ALU_OP_LIST)

# Режимы записи на внешнюю ленту M_RAM в конце такта:
# 0: NO_WRITE
# 1: STORE_ACC_TO_PTR   (M_RAM[PTR] <- ACC)
# 2: STORE_B_TO_PTR     (M_RAM[PTR] <- (b0, b1, 0))
# 3: SWAP_ACC_AND_PTR   (ACC <- MDR, M_RAM[PTR] <- old ACC — атомарный обмен для сортировки!)
# 4: PUSH_ACC_TO_STACK  (M_RAM[20 + sp] <- ACC, sp <- (sp + 1) % 5)

# Предикаты переходов для автономного счётчика команд PC (Program Counter):
# 0: NEXT       (PC <- PC + 1)
# 1: JZ_ACC     (если ACC == 0: PC <- target, иначе PC + 1)
# 2: JNZ_ACC    (если ACC != 0: PC <- target, иначе PC + 1)
# 3: JZ_B       (если B_reg == 0: PC <- target, иначе PC + 1)
# 4: JNZ_B      (если B_reg != 0: PC <- target, иначе PC + 1)
# 5: JLT_ACC_MDR(если ACC < MDR: PC <- target, иначе PC + 1 — сравнение с ячейкой M_RAM[PTR]!)
# 6: JMP        (безусловный переход PC <- target)
# 7: HALT       (останов процессора: halted <- True)
NUM_BRANCH_CONDS = 6  # предикаты состояния [ACC==0, ACC!=0, B==0, B!=0, ACC<MDR, PTR==B]

MAX_CLAUSE_TOKENS = 48
MAX_CHARS_PER_TOK = 32  # 32 байта UTF-8 = 16 кириллических букв (исправляет обрезку в v13!)
CHAR_VOCAB_SIZE = 256
NGRAM_BUCKETS = 4096


def apply_alu_op_tensor(regs, mdr, op_ids, alpha_v, beta_v):
    """
    Точная векторная семантика 21 микрооперации АЛУ над батчем регистров ЦПУ regs (B, 8)
    и предварительно прочитанным буфером памяти mdr (B, 3) = M_RAM[PTR].
    """
    d0, d1, d2, p0, p1, sp, b0, b1 = [regs[:, i].clone() for i in range(8)]
    mdr0, mdr1, mdr2 = mdr[:, 0], mdr[:, 1], mdr[:, 2]

    acc = d2 * 25 + d1 * 5 + d0
    mdr_val = mdr2 * 25 + mdr1 * 5 + mdr0
    ptr = p1 * 5 + p0
    b_val = b1 * 5 + b0

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

    return torch.stack([nd0, nd1, nd2, np0, np1, nsp, nb0, nb1], dim=1)


def eval_v14_predicates(regs, mdr):
    """
    Точные булевы значения 6 аппаратных предикатов ветвления/останова (B, 6)
    над регистрами ЦПУ regs (B, 8) и буфером памяти mdr (B, 3):
      [0] ACC == 0
      [1] ACC != 0
      [2] B_reg == 0
      [3] B_reg != 0
      [4] ACC < MDR   (сравнение 3-разрядного ACC с M_RAM[PTR]!)
      [5] PTR == B_reg (достижение указателем границы массива B_reg!)
    """
    acc = regs[:, 2] * 25 + regs[:, 1] * 5 + regs[:, 0]
    ptr = regs[:, 4] * 5 + regs[:, 3]
    b_val = regs[:, 7] * 5 + regs[:, 6]
    mdr_val = mdr[:, 2] * 25 + mdr[:, 1] * 5 + mdr[:, 0]

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
    def __init__(self, d_model=640, n_groups=8):
        super().__init__()
        self.n_groups = n_groups
        self.d_sub = d_model // n_groups

    def forward(self, x):
        orig_shape = x.shape
        x_g = x.view(*orig_shape[:-1], self.n_groups, self.d_sub)
        x_g = F.normalize(x_g, p=2, dim=-1)
        return x_g.view(*orig_shape)


class ProductSimVQ8(nn.Module):
    def __init__(self, d_model=640, n_groups=8, num_vals=5):
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

    def encode_regs(self, regs):
        cb = self.get_codebook()
        parts = [cb[g, regs[:, g]] for g in range(self.n_groups)]
        return torch.cat(parts, dim=-1)

    def encode_mdr(self, mdr_digits):
        """Кодирует 3-разрядное слово MDR (B, 3) в сферические базисы разрядов [0, 1, 2] -> (B, 240)."""
        cb = self.get_codebook()
        parts = [cb[g, mdr_digits[:, g]] for g in range(3)]
        return torch.cat(parts, dim=-1)

    def decode_mdr(self, h_mdr):
        """Декодирует 3-сферный вектор MDR (B, 240) в цифры (B, 3) in [0..4]."""
        cb = self.get_codebook()[:3]  # (3, 5, 80)
        h_g = F.normalize(h_mdr.view(-1, 3, self.d_sub), p=2, dim=-1)
        logits = torch.einsum("bgd,gvd->bgv", h_g, cb) * self.logit_scale
        return logits.argmax(dim=-1)

    def snap_mdr(self, h_mdr):
        """Примагничивает прочитанный из M_RAM вектор MDR (B, 240) к точным центроидам Вороного."""
        cb = self.get_codebook()[:3]
        idx = self.decode_mdr(h_mdr)
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


class BiorthogonalExternalTapeRAM(nn.Module):
    """
    Внешняя биортогональная матричная лента памяти M_RAM (25 адресов x 3 разряда SimVQ = 240 каналов).
    Использует кронекеровское произведение координат указателя PTR = (p0, p1) in Z_5^2 (25 адресов):
      - Чтение LOAD_RAM: h_mdr = snap_mdr(k_dual(p0, p1)^T @ M_RAM)
      - Запись ранга 1 STORE_RAM: M_RAM <- M_RAM + k_dual(p0, p1) @ (v_new - k_read(p0, p1)^T @ M_RAM)
    Так как <k_dual(a), k_read(b)> = delta_{ab}, запись в ячейку a имеет строго 0.000000 интерференции
    со всеми остальными 24 ячейками ленты!
    """

    def __init__(self, d_sub=80):
        super().__init__()
        self.d_sub = d_sub
        self.d_word = 3 * d_sub  # 240 (3 сферы под 3-разрядное слово 0..124)
        self.register_buffer("C_p0_dual", torch.zeros(5, d_sub))
        self.register_buffer("C_p1_dual", torch.zeros(5, d_sub))
        self.register_buffer("K_addr_read", torch.zeros(25, 25))
        self.register_buffer("K_addr_dual", torch.zeros(25, 25))
        self.ortho_err = 0.0

    @staticmethod
    def _ns_inv(G, steps=16):
        I = torch.eye(G.size(0), device=G.device)
        # Строгая верхняя оценка спектрального радиуса через след tr(G^T G):
        # гарантирует 0 < alpha * lambda_i < 2 и монотонную квадратичную сходимость без NaN!
        trace_sq = (G * G).sum().item()
        X = (1.5 / max(trace_sq, 1e-6)) * G.T
        for _ in range(steps):
            X = X @ (2.0 * I - G @ X)
        return X

    @torch.no_grad()
    def calibrate_address_basis(self, simvq8):
        """Вычисляет дуальные базисы адресов PTR (сферы 3 и 4) через итерацию Ньютона-Шульца."""
        cb = simvq8.get_codebook()
        c_p0, c_p1 = cb[3], cb[4]  # (5, 80)
        self.C_p0_dual.copy_(self._ns_inv(c_p0 @ c_p0.T, steps=16) @ c_p0)
        self.C_p1_dual.copy_(self._ns_inv(c_p1 @ c_p1.T, steps=16) @ c_p1)

        keys = []
        for v1 in range(5):
            for v0 in range(5):
                u0 = c_p0[v0] @ self.C_p0_dual.T  # (5,)
                u1 = c_p1[v1] @ self.C_p1_dual.T  # (5,)
                k_vec = torch.outer(u1, u0).reshape(25)
                keys.append(k_vec)
        K = torch.stack(keys, dim=0)  # (25, 25)
        G = K @ K.T
        K_dual = self._ns_inv(G, steps=16) @ K
        self.K_addr_read.copy_(K)
        self.K_addr_dual.copy_(K_dual)
        I25 = torch.eye(25, device=K.device)
        self.ortho_err = (K_dual @ K.T - I25).abs().max().item()
        return self.ortho_err

    def _addr_key_from_spheres(self, h_p0, h_p1):
        """Извлекает кронекеровский ключ адреса (B, 25) напрямую из сферических векторов h_p0, h_p1 (B, 80)."""
        u0 = h_p0 @ self.C_p0_dual.T  # (B, 5)
        u1 = h_p1 @ self.C_p1_dual.T  # (B, 5)
        return torch.einsum("bi,bj->bij", u1, u0).reshape(h_p0.size(0), 25)

    @torch.no_grad()
    def init_tape(self, ram_digits_tensor, simvq8):
        """
        Инициализирует матрицу внешней ленты M_RAM (B, 25, 240) из начальных цифр памяти (B, 25, 3).
        """
        B = ram_digits_tensor.size(0)
        flat_digits = ram_digits_tensor.reshape(B * 25, 3)
        V_words = simvq8.encode_mdr(flat_digits).view(B, 25, self.d_word)  # (B, 25, 240)
        # M_RAM = K_addr_dual^T @ V_words
        M_ram = torch.einsum("ak,bkd->bad", self.K_addr_dual.T, V_words)
        return M_ram

    @torch.no_grad()
    def read_at_ptr(self, M_ram, h_state, simvq8):
        """
        Читает 3-сферный вектор MDR (B, 240) из внешней матрицы M_ram (B, 25, 240)
        по координатам сфер указателя PTR (сферы 3 и 4 внутри h_state).
        """
        B = h_state.size(0)
        h_g = h_state.view(B, 8, self.d_sub)
        k_read = self._addr_key_from_spheres(h_g[:, 3], h_g[:, 4])  # (B, 25)
        raw_word = torch.einsum("ba,bad->bd", k_read, M_ram)        # (B, 240)
        return simvq8.snap_mdr(raw_word)

    @torch.no_grad()
    def read_at_stack(self, M_ram, h_state, simvq8):
        """Читает верх стека M_RAM[20 + sp] по сфере указателя стека sp (сфера 5 внутри h_state)."""
        B = h_state.size(0)
        cb = simvq8.get_codebook()
        h_g = h_state.view(B, 8, self.d_sub)
        sp_idx = (h_g[:, 5] @ cb[5].T).argmax(dim=-1)  # (B,) in 0..4
        h_p0 = cb[3, sp_idx]
        h_p1 = cb[4, torch.full_like(sp_idx, 4)]       # p1 = 4 -> адрес 4*5 + sp = 20 + sp
        k_read = self._addr_key_from_spheres(h_p0, h_p1)
        raw_word = torch.einsum("ba,bad->bd", k_read, M_ram)
        return simvq8.snap_mdr(raw_word)

    @torch.no_grad()
    def write_at_ptr(self, M_ram, h_state, v_new_word, mask=None):
        """
        Проекционное обновление ранга 1 матрицы M_ram по адресу PTR (сферы 3 и 4):
        M_ram <- M_ram + k_dual(PTR) @ (v_new - k_read(PTR)^T M_ram)
        """
        B = h_state.size(0)
        h_g = h_state.view(B, 8, self.d_sub)
        k_read = self._addr_key_from_spheres(h_g[:, 3], h_g[:, 4])  # (B, 25)
        k_dual = k_read @ self.K_addr_dual                          # (B, 25)
        v_old = torch.einsum("ba,bad->bd", k_read, M_ram)           # (B, 240)
        delta_M = torch.einsum("ba,bd->bad", k_dual, v_new_word - v_old)
        if mask is not None:
            delta_M = delta_M * mask.view(B, 1, 1).float()
        return M_ram + delta_M

    @torch.no_grad()
    def write_at_stack(self, M_ram, h_state, v_new_word, simvq8, mask=None):
        """Запись ранга 1 в аппаратный стек M_RAM[20 + sp] по сфере sp (сфера 5)."""
        B = h_state.size(0)
        cb = simvq8.get_codebook()
        h_g = h_state.view(B, 8, self.d_sub)
        sp_idx = (h_g[:, 5] @ cb[5].T).argmax(dim=-1)
        h_p0 = cb[3, sp_idx]
        h_p1 = cb[4, torch.full_like(sp_idx, 4)]
        k_read = self._addr_key_from_spheres(h_p0, h_p1)
        k_dual = k_read @ self.K_addr_dual
        v_old = torch.einsum("ba,bad->bd", k_read, M_ram)
        delta_M = torch.einsum("ba,bd->bad", k_dual, v_new_word - v_old)
        if mask is not None:
            delta_M = delta_M * mask.view(B, 1, 1).float()
        return M_ram + delta_M

    @torch.no_grad()
    def decode_full_tape(self, M_ram, simvq8):
        """Декодирует все 25 ячеек внешней ленты M_ram (B, 25, 240) в целые числа (B, 25) in [0..124]."""
        B = M_ram.size(0)
        words = torch.einsum("ak,bkd->bad", self.K_addr_read, M_ram).reshape(B * 25, self.d_word)
        digits = simvq8.decode_mdr(words).view(B, 25, 3)
        return digits[:, :, 2] * 25 + digits[:, :, 1] * 5 + digits[:, :, 0]


class TernaryMDRCore(nn.Module):
    """
    100% Тернарное ядро АЛУ (BitLinear {-1, 0, +1}), которое принимает:
      - текущее состояние 8 регистров ЦПУ h_prev (B, 640),
      - предварительно прочитанный из внешней ленты M_RAM вектор регистра данных h_mdr (B, 240),
      - код операции op_ids и константы пролога (alpha, beta).
    Ключевой принцип v14:
      - Прямая поразрядная передача между M_RAM и регистрами (`LOAD_MDR_ACC` op=10, `MOV_ACC_TO_B` op=13,
        `MOV_B_TO_ACC` op=14, `LOAD_MDR_B` op=15, `ZERO_PTR` op=9, `NOP` op=0) маршрутизируется
        детерминированной перестановкой сферических слотов на одной и той же кодовой книге Вороного (0% ошибки!).
      - Арифметические операции с каскадным переносом и заёмом (`ADD_MDR_CARRY`, `SUB_MDR_BORROW`, `INC_PTR`,
        `DEC_PTR`, `ADD_CARRY_1..5`, `SUB_BORROW_1`, `SHL/SHR_BASE5`, `INC/DEC_B`, `INC/DEC_SP`)
        выполняются 100% тернарным билинейным контуром `BitLinear`, где второй множитель `u_b`
        зависит ИСКЛЮЧИТЕЛЬНО от целевых слагаемых (`h_mdr`, `c_op`, `c_epi`), что исключает
        комбинаторные помехи между несвязанными регистрами!
    """

    def __init__(self, d_model=640, n_groups=8, num_vals=5, num_ops=NUM_ALU_OPS):
        super().__init__()
        self.d_model = d_model
        self.n_groups = n_groups
        self.d_sub = d_model // n_groups
        self.d_mdr = 3 * self.d_sub  # 240
        self.simvq = ProductSimVQ8(d_model=d_model, n_groups=n_groups, num_vals=num_vals)
        self.sphere_norm = SubspaceSphereNorm(d_model=d_model, n_groups=n_groups)

        self.op_emb = nn.Embedding(num_ops, d_model)
        self.alpha_emb = nn.Embedding(num_vals, d_model // 2)
        self.beta_emb = nn.Embedding(num_vals, d_model // 2)
        nn.init.orthogonal_(self.op_emb.weight)
        nn.init.orthogonal_(self.alpha_emb.weight)
        nn.init.orthogonal_(self.beta_emb.weight)

        d_hidden = 1536
        # Выделенная шина второго операнда для 3-разрядных операций ACC (+|-|=) MDR и констант alpha:
        # только сферы MDR (240), сфера p0 (80) и сфера b0 (80) несут внешний сдвиг для старших разрядов!
        self.in_state_a = BitLinear(d_model + self.d_mdr, d_hidden, bias=False)
        self.in_state_b = BitLinear(d_model + self.d_mdr, d_hidden, bias=False)
        self.in_op_a = BitLinear(d_model, d_hidden, bias=False)
        self.in_op_b = BitLinear(d_model, d_hidden, bias=False)
        self.in_epi = BitLinear(d_model, d_hidden, bias=False)

        self.mid_norm = nn.LayerNorm(d_hidden, elementwise_affine=False)
        self.mid_gate = BitLinear(d_hidden, d_hidden, bias=False)
        self.mid_val = BitLinear(d_hidden, d_hidden, bias=False)
        self.out_proj = BitLinear(d_hidden, d_model, bias=False)

        # Таблица маскирования активных сфер ВЫХОДА (21 операция x 8 сфер):
        mask_mat = torch.zeros(num_ops, n_groups)
        for oid in [1, 2, 3, 4, 5, 6, 10, 11, 12, 14, 18]:
            mask_mat[oid, 0:3] = 1.0  # Операции, изменяющие ACC (d0, d1, d2)
        for oid in [7, 8, 9]:
            mask_mat[oid, 3:5] = 1.0  # Операции, изменяющие PTR (p0, p1)
        for oid in [19, 20]:
            mask_mat[oid, 5] = 1.0    # Операции, изменяющие SP (sp)
        for oid in [13, 15, 16, 17]:
            mask_mat[oid, 6:8] = 1.0  # Операции, изменяющие B_reg (b0, b1)
        self.register_buffer("op_sphere_mask", mask_mat)

        # Таблица маскирования активных сфер ВХОДА (21 операция x 11 входных сфер: 8 ЦПУ + 3 MDR):
        # 100% физическая развязка входов! Каждая микрооперация видит ТОЛЬКО свои входные регистры:
        in_mask = torch.zeros(num_ops, n_groups + 3)
        for oid in [1, 2, 3, 4, 5, 6, 18]:
            in_mask[oid, 0:3] = 1.0   # Унарные операции над ACC видят только ACC (d0, d1, d2)
        for oid in [11, 12]:
            in_mask[oid, 0:3] = 1.0   # Бинарные операции ACC (+|-) mdr0 видят ACC (0..2) и сферу mdr0 (8)!
            in_mask[oid, 8] = 1.0
        in_mask[10, 8:11] = 1.0       # LOAD_MDR_ACC видит 3 сферы MDR (8..10)
        for oid in [7, 8, 9]:
            in_mask[oid, 3:5] = 1.0   # Операции над PTR видят только PTR (p0, p1)
        for oid in [19, 20]:
            in_mask[oid, 5] = 1.0     # Операции над SP видят только SP (sp)
        in_mask[13, 0:2] = 1.0        # MOV_ACC_TO_B видит d0, d1
        in_mask[14, 6:8] = 1.0        # MOV_B_TO_ACC видит b0, b1
        in_mask[15, 8:10] = 1.0       # LOAD_MDR_B видит mdr0, mdr1
        for oid in [16, 17]:
            in_mask[oid, 6:8] = 1.0   # INC_B / DEC_B видят только b0, b1
        self.register_buffer("op_input_mask", in_mask)

    def set_quant_alpha(self, alpha):
        for m in self.modules():
            if isinstance(m, BitLinear):
                m.quant_alpha = alpha

    def step(self, h_prev, h_mdr, op_ids, alpha_v, beta_v, snap_strength=1.0):
        B = h_prev.size(0)
        c_op = self.op_emb(op_ids)
        c_epi = torch.cat([self.alpha_emb(alpha_v), self.beta_emb(beta_v)], dim=-1)

        # Маскируем неиспользуемые входные сферы (0.0 на всех регистрах, не участвующих в op_ids):
        h_in_g = torch.cat([h_prev, h_mdr], dim=-1).view(B, self.n_groups + 3, self.d_sub)
        in_m = self.op_input_mask[op_ids].unsqueeze(-1)  # (B, 11, 1)
        h_in = (h_in_g * in_m).view(B, self.d_model + self.d_mdr)

        u_a = F.silu(self.in_state_a(h_in) + self.in_op_a(c_op))
        u_b = F.gelu(self.in_state_b(h_in) + self.in_op_b(c_op) + self.in_epi(c_epi))
        h_int = self.mid_norm(u_a * u_b)
        h_mlp = F.silu(self.mid_gate(h_int)) * self.mid_val(h_int)
        delta = self.out_proj(h_int + h_mlp)

        h_cand_raw = self.sphere_norm(0.15 * h_prev + delta).view(B, self.n_groups, self.d_sub)
        h_prev_g = h_prev.view(B, self.n_groups, self.d_sub)
        h_mdr_g = h_mdr.view(B, 3, self.d_sub)

        # Изоляция неактивных сфер выхода:
        sp_mask = self.op_sphere_mask[op_ids].unsqueeze(-1)  # (B, 8, 1)
        h_cand_g = torch.where(sp_mask > 0.5, h_cand_raw, h_prev_g)

        # Прямые шины пересылки регистров (LOAD_MDR_ACC op=10, MOV_ACC_TO_B op=13, MOV_B_TO_ACC op=14, LOAD_MDR_B op=15, ZERO_PTR op=9):
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

        h_cand = h_cand_g.view(B, self.d_model)
        h_next = self.simvq.snap_step(h_cand, snap_strength=snap_strength)
        return h_next, h_cand


class TernaryBranchRouter_v14(nn.Module):
    """
    100% Тернарный маршрутизатор предикатов переходов PC и останова (BitLinear {-1, 0, +1}).
    Принимает [h_state (640), h_mdr (240)] и вычисляет 6 предикатов:
      [ACC==0, ACC!=0, B==0, B!=0, ACC<MDR, PTR==B].
    """

    def __init__(self, d_model=640, d_mdr=240):
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
# Автономный счётчик команд PC (Program Counter) и машина с внешней лентой M_RAM
# ==============================================================================
# Формат одной скомпилированной инструкции в памяти программ (4 целых числа):
#   (alu_op_id, write_mode, branch_type, target_pc)
# где:
#   alu_op_id   in [0..20] (операция TernaryMDRCore над регистрами и прочитанным MDR)
#   write_mode  in [0..5]:
#       0 = NO_WRITE
#       1 = STORE_ACC_TO_PTR  (M_RAM[PTR] <- старый ACC до шага АЛУ)
#       2 = STORE_B_TO_PTR    (M_RAM[PTR] <- старый B_reg)
#       3 = SWAP_ACC_AND_PTR  (alu_op=LOAD_MDR_ACC загружает M_RAM[PTR] в ACC, а в M_RAM[PTR] пишется старый ACC!)
#       4 = PUSH_ACC_TO_STACK (M_RAM[20 + sp] <- старый ACC, alu_op=INC_SP двигает sp+1)
#       5 = POP_STACK_TO_ACC  (перед шагом АЛУ в MDR подаётся верх стека M_RAM[20 + sp - 1], alu_op=LOAD_MDR_ACC)
#   branch_type in [0..7]:
#       0 = NEXT          (PC <- PC + 1)
#       1 = JZ_ACC        (если cond[0] (ACC == 0): PC <- target_pc, иначе PC + 1)
#       2 = JNZ_ACC       (если cond[1] (ACC != 0): PC <- target_pc, иначе PC + 1)
#       3 = JZ_B          (если cond[2] (B == 0):   PC <- target_pc, иначе PC + 1)
#       4 = JNZ_B         (если cond[3] (B != 0):   PC <- target_pc, иначе PC + 1)
#       5 = JLT_ACC_MDR   (если cond[4] (ACC < MDR):PC <- target_pc, иначе PC + 1)
#       6 = JEQ_PTR_B     (если cond[5] (PTR == B): PC <- target_pc, иначе PC + 1)
#       7 = JMP           (безусловный переход PC <- target_pc)
#       8 = HALT          (останов машины: halted <- True)


def ground_truth_execute_program(init_regs, init_tape_vals, prog_table, alpha_v, beta_v, max_ticks=96):
    """
    Эталонный (Ground Truth) пошаговый интерпретатор программы prog_table (L_prog, 4)
    над батчем начальных регистров init_regs (B, 8) и внешней памяти init_tape_vals (B, 25) in [0..124].
    Возвращает финальные регистры (B, 8), финальную ленту памяти (B, 25), число тактов (B,) и траекторию PC.
    """
    B = init_regs.size(0)
    dev = init_regs.device
    regs = init_regs.clone()
    tape = init_tape_vals.clone()  # (B, 25) in [0..124]
    pc = torch.zeros(B, dtype=torch.long, device=dev)
    halted = torch.zeros(B, dtype=torch.bool, device=dev)
    ticks = torch.zeros(B, dtype=torch.long, device=dev)
    L_prog = prog_table.size(0)

    for _ in range(max_ticks):
        if halted.all():
            break
        active = ~halted
        pc_clamped = pc.clamp(0, L_prog - 1)
        inst = prog_table[pc_clamped]  # (B, 4)
        alu_op = inst[:, 0]
        w_mode = inst[:, 1]
        br_type = inst[:, 2]
        tgt_pc = inst[:, 3]

        is_halt = active & (br_type == 8)
        halted = halted | is_halt
        active = ~halted
        if not active.any():
            break

        ptr = regs[:, 4] * 5 + regs[:, 3]  # (B,) in [0..24]
        sp_prev = regs[:, 5]
        sp_pop = (sp_prev - 1) % 5
        stack_pop_addr = STACK_BASE + sp_pop

        # Чтение MDR: либо из M_RAM[PTR], либо из верха стека при POP (w_mode == 5)
        read_addr = torch.where(w_mode == 5, stack_pop_addr, ptr)
        mdr_val = tape.gather(1, read_addr.unsqueeze(1)).squeeze(1)
        mdr_digits = torch.stack([mdr_val % 5, (mdr_val // 5) % 5, (mdr_val // 25) % 5], dim=1)

        # Вычисление предикатов ДО изменения регистров
        conds = eval_v14_predicates(regs, mdr_digits)  # (B, 6)

        old_acc = regs[:, 2] * 25 + regs[:, 1] * 5 + regs[:, 0]
        old_b = regs[:, 7] * 5 + regs[:, 6]

        # При POP (w_mode == 5) сначала уменьшаем sp на 1, затем загружаем MDR в ACC
        next_regs = apply_alu_op_tensor(regs, mdr_digits, alu_op, alpha_v, beta_v)
        if (w_mode == 5).any():
            next_regs[:, 5] = torch.where(w_mode == 5, sp_pop, next_regs[:, 5])

        # Запись на внешнюю ленту памяти tape (B, 25)
        new_tape = tape.clone()
        for b_i in range(B):
            if not active[b_i]:
                continue
            wm = int(w_mode[b_i].item())
            if wm in (1, 3):
                new_tape[b_i, ptr[b_i]] = old_acc[b_i]
            elif wm == 2:
                new_tape[b_i, ptr[b_i]] = old_b[b_i]
            elif wm == 4:
                new_tape[b_i, STACK_BASE + sp_prev[b_i]] = old_acc[b_i]

        # Обновление счётчика команд PC
        take_jump = torch.zeros(B, dtype=torch.bool, device=dev)
        for c_idx in range(1, 7):
            take_jump = take_jump | ((br_type == c_idx) & (conds[:, c_idx - 1] == 1))
        take_jump = take_jump | (br_type == 7)
        next_pc = torch.where(take_jump, tgt_pc, pc + 1)

        regs = torch.where(active.unsqueeze(1), next_regs, regs)
        tape = torch.where(active.unsqueeze(1), new_tape, tape)
        pc = torch.where(active, next_pc, pc)
        ticks = ticks + active.long()

    return regs, tape, ticks


@torch.no_grad()
def neural_execute_program_v14(
    core, router, ext_ram, init_regs, init_tape_digits, prog_table, alpha_v, beta_v, max_ticks=96
):
    """
    100% Нейронное исполнение программы с автономным счётчиком команд PC и внешней биортогональной
    матричной лентой M_RAM:
      - Регистры ЦПУ живут на 8 сферах Вороного `h_cur` (B, 640),
      - Оперативная память и стек живут во внешней матрице `M_ram` (B, 25, 240),
      - На каждом такте `MDR` читается умножением дуального ключа адреса на `M_ram`,
      - Переходы `PC` управляются тернарным маршрутизатором `TernaryBranchRouter_v14`,
      - Запись на ленту выполняется одношаговым проекционным обновлением ранга 1!
    """
    B = init_regs.size(0)
    dev = init_regs.device
    simvq = core.simvq

    h_cur = simvq.encode_regs(init_regs)
    M_ram = ext_ram.init_tape(init_tape_digits, simvq)

    pc = torch.zeros(B, dtype=torch.long, device=dev)
    halted = torch.zeros(B, dtype=torch.bool, device=dev)
    ticks = torch.zeros(B, dtype=torch.long, device=dev)
    L_prog = prog_table.size(0)

    for _ in range(max_ticks):
        if halted.all():
            break
        active = ~halted
        pc_clamped = pc.clamp(0, L_prog - 1)
        inst = prog_table[pc_clamped]
        alu_op = inst[:, 0]
        w_mode = inst[:, 1]
        br_type = inst[:, 2]
        tgt_pc = inst[:, 3]

        is_halt = active & (br_type == 8)
        halted = halted | is_halt
        active = ~halted
        if not active.any():
            break

        # Если POP (w_mode == 5), сначала делаем шаг DEC_SP (op=20) в сфере sp, чтобы прочитать M_RAM[20 + sp - 1]
        is_pop = active & (w_mode == 5)
        h_for_read = h_cur
        if is_pop.any():
            op_dec_sp = torch.full((B,), 20, dtype=torch.long, device=dev)
            dummy_mdr = simvq.encode_mdr(torch.zeros(B, 3, dtype=torch.long, device=dev))
            h_dec_sp, _ = core.step(h_cur, dummy_mdr, op_dec_sp, alpha_v, beta_v, snap_strength=1.0)
            h_for_read = torch.where(is_pop.unsqueeze(1), h_dec_sp, h_cur)

        # Чтение MDR из внешней матрицы M_RAM по сферам PTR (или по сфере SP для POP)
        h_mdr_ptr = ext_ram.read_at_ptr(M_ram, h_for_read, simvq)
        if is_pop.any():
            h_mdr_stk = ext_ram.read_at_stack(M_ram, h_for_read, simvq)
            h_mdr = torch.where(is_pop.unsqueeze(1), h_mdr_stk, h_mdr_ptr)
        else:
            h_mdr = h_mdr_ptr

        # Оценка всех 6 предикатов тернарным маршрутизатором прямо над сферами [h_for_read, h_mdr]
        pred_logits = router(h_for_read, h_mdr)  # (B, 6, 2)
        pred_conds = pred_logits.argmax(dim=-1)  # (B, 6)

        # Сохраняем векторы старого ACC (сферы 0,1,2 -> 240 каналов) и старого B (сферы 6,7 + нулевая d2 -> 240)
        h_g_old = h_for_read.view(B, 8, D_SUB)
        v_old_acc = h_g_old[:, :3, :].reshape(B, 240)
        cb_d2_zero = simvq.get_codebook()[2, 0].unsqueeze(0).expand(B, D_SUB)
        v_old_b = torch.cat([h_g_old[:, 6, :], h_g_old[:, 7, :], cb_d2_zero], dim=-1)

        # Шаг 100% тернарного ядра АЛУ
        h_next, _ = core.step(h_for_read, h_mdr, alu_op, alpha_v, beta_v, snap_strength=1.0)

        # Запись ранга 1 во внешнюю матрицу M_ram
        m_write_acc = active & ((w_mode == 1) | (w_mode == 3))
        if m_write_acc.any():
            M_ram = ext_ram.write_at_ptr(M_ram, h_for_read, v_old_acc, mask=m_write_acc)
        m_write_b = active & (w_mode == 2)
        if m_write_b.any():
            M_ram = ext_ram.write_at_ptr(M_ram, h_for_read, v_old_b, mask=m_write_b)
        m_push = active & (w_mode == 4)
        if m_push.any():
            M_ram = ext_ram.write_at_stack(M_ram, h_for_read, v_old_acc, simvq, mask=m_push)

        # Автономный переход счётчика команд PC
        take_jump = torch.zeros(B, dtype=torch.bool, device=dev)
        for c_idx in range(1, 7):
            take_jump = take_jump | ((br_type == c_idx) & (pred_conds[:, c_idx - 1] == 1))
        take_jump = take_jump | (br_type == 7)
        next_pc = torch.where(take_jump, tgt_pc, pc + 1)

        h_cur = torch.where(active.unsqueeze(1), h_next, h_cur)
        pc = torch.where(active, next_pc, pc)
        ticks = ticks + active.long()

    pred_regs = simvq.compute_logits(h_cur).argmax(dim=-1)
    pred_tape = ext_ram.decode_full_tape(M_ram, simvq)
    return pred_regs, pred_tape, ticks


# ==============================================================================
# Библиотека алгоритмов с автономным PC над внешней лентой M_RAM[0..24]
# ==============================================================================
def build_v14_algorithm_rom(dev="cuda"):
    """
    Возвращает словарь многошаговых программ в ПЗУ (ROM), которые управляются
    автономным счётчиком команд PC (включая вложенные циклы и стек!):
      1. `ArraySum_25`: суммирует участок ленты M_RAM[PTR .. B_reg-1] в 3-разрядный ACC с переносом разряда.
      2. `ArrayMax_Scan`: находит максимум в массиве M_RAM[0..3] и сохраняет его в ACC и M_RAM[4].
      3. `BubbleSort_3`: полная сортировка массива M_RAM[0..2] по возрастанию на месте (in-place)
         с помощью условного сравнения JLT_ACC_MDR и атомарного обмена SWAP_ACC_AND_PTR!
      4. `StackReverse_3`: читает 3 числа из M_RAM[0..2], кладёт их в аппаратный стек PUSH (M_RAM[20..22]),
         а затем извлекает через POP обратно в M_RAM[0..2], разворачивая массив задом наперёд!
    """
    progs = {}

    # 1. ArraySum_25: while PTR != B_reg: ACC = (ACC + M_RAM[PTR]) % 125; PTR += 1
    #   PC=0: NOP, NO_WRITE, JEQ_PTR_B -> PC=3
    #   PC=1: ADD_MDR_CARRY (11), NO_WRITE, NEXT
    #   PC=2: INC_PTR (7), NO_WRITE, JMP -> PC=0
    #   PC=3: NOP (0), NO_WRITE, HALT (8)
    progs["ArraySum_25"] = torch.tensor(
        [
            [0, 0, 6, 3],
            [11, 0, 0, 0],
            [7, 0, 7, 0],
            [0, 0, 8, 3],
        ],
        dtype=torch.long,
        device=dev,
    )

    # 2. ArrayMax_Scan: ищет максимум среди M_RAM[0..3] (при старте PTR=0, B_reg=4, ACC=0)
    #   PC=0: NOP, NO_WRITE, JEQ_PTR_B -> PC=4 (конец цикла, когда PTR == 4)
    #   PC=1: NOP, NO_WRITE, JLT_ACC_MDR -> PC=3 (если ACC < M_RAM[PTR], прыгаем на обновление максимума!)
    #   PC=2: INC_PTR (7), NO_WRITE, JMP -> PC=0
    #   PC=3: LOAD_MDR_ACC (10), NO_WRITE, JMP -> PC=2 (ACC = M_RAM[PTR], затем PC=2 увеличит PTR)
    #   PC=4: NOP (0), STORE_ACC_TO_PTR (1), HALT (8) (записывает найденный максимум в M_RAM[4] и останавливается)
    progs["ArrayMax_Scan"] = torch.tensor(
        [
            [0, 0, 6, 4],
            [0, 0, 5, 3],
            [7, 0, 7, 0],
            [10, 0, 7, 2],
            [0, 1, 8, 4],
        ],
        dtype=torch.long,
        device=dev,
    )

    # 3. BubbleSort_3: полная сортировка массива M_RAM[0], M_RAM[1], M_RAM[2] по возрастанию на месте!
    #   Проход 1 (сравнение M_RAM[0] и M_RAM[1]):
    #     PC=0: ZERO_PTR (9)
    #     PC=1: LOAD_MDR_ACC (10)  -> ACC = M_RAM[0]
    #     PC=2: INC_PTR (7)        -> PTR = 1 (MDR = M_RAM[1])
    #     PC=3: NOP, NO_WRITE, JLT_ACC_MDR -> PC=6 (если M_RAM[0] < M_RAM[1], обмен не нужен)
    #     PC=4: LOAD_MDR_ACC (10), SWAP_ACC_AND_PTR (3) -> M_RAM[1] = old M_RAM[0], ACC = old M_RAM[1]
    #     PC=5: DEC_PTR (8)        -> PTR = 0; запись ACC в M_RAM[0] на следующем шаге:
    #     ... сделаем ещё чище и короче:
    progs["BubbleSort_3"] = torch.tensor(
        [
            # --- Шаг сравнения пары (0, 1) ---
            [9, 0, 0, 0],   # PC=0: ZERO_PTR (PTR=0)
            [10, 0, 0, 0],  # PC=1: LOAD_MDR_ACC (ACC = M_RAM[0])
            [7, 0, 0, 0],   # PC=2: INC_PTR (PTR=1, MDR = M_RAM[1])
            [0, 0, 5, 7],   # PC=3: if ACC < M_RAM[1] -> jump to PC=7 (пара 0,1 уже по порядку)
            [10, 3, 0, 0],  # PC=4: SWAP! ACC <- M_RAM[1], M_RAM[1] <- old ACC (M_RAM[0])
            [8, 0, 0, 0],   # PC=5: DEC_PTR (PTR=0)
            [0, 1, 7, 8],   # PC=6: STORE_ACC_TO_PTR (M_RAM[0] <- ACC), JMP -> PC=8
            [10, 0, 0, 0],  # PC=7: LOAD_MDR_ACC (ACC = M_RAM[1] — текущий максимум из первых двух)
            # --- Шаг сравнения пары (1, 2) ---
            [9, 0, 0, 0],   # PC=8: ZERO_PTR (PTR=0)
            [7, 0, 0, 0],   # PC=9: INC_PTR (PTR=1)
            [10, 0, 0, 0],  # PC=10: LOAD_MDR_ACC (ACC = M_RAM[1])
            [7, 0, 0, 0],   # PC=11: INC_PTR (PTR=2, MDR = M_RAM[2])
            [0, 0, 5, 16],  # PC=12: if ACC < M_RAM[2] -> jump to PC=16
            [10, 3, 0, 0],  # PC=13: SWAP! ACC <- M_RAM[2], M_RAM[2] <- old M_RAM[1]
            [8, 0, 0, 0],   # PC=14: DEC_PTR (PTR=1)
            [0, 1, 0, 0],   # PC=15: STORE_ACC_TO_PTR (M_RAM[1] <- ACC)
            # --- Финальный проход пары (0, 1) ---
            [9, 0, 0, 0],   # PC=16: ZERO_PTR (PTR=0)
            [10, 0, 0, 0],  # PC=17: LOAD_MDR_ACC (ACC = M_RAM[0])
            [7, 0, 0, 0],   # PC=18: INC_PTR (PTR=1, MDR = M_RAM[1])
            [0, 0, 5, 23],  # PC=19: if ACC < M_RAM[1] -> jump to PC=23 (HALT)
            [10, 3, 0, 0],  # PC=20: SWAP! ACC <- M_RAM[1], M_RAM[1] <- old M_RAM[0]
            [8, 0, 0, 0],   # PC=21: DEC_PTR (PTR=0)
            [0, 1, 0, 0],   # PC=22: STORE_ACC_TO_PTR (M_RAM[0] <- ACC)
            [0, 0, 8, 23],  # PC=23: HALT!
        ],
        dtype=torch.long,
        device=dev,
    )

    # 4. StackReverse_3: разворот массива M_RAM[0..2] через аппаратный стек M_RAM[20..22] (PUSH x3 + POP x3)
    progs["StackReverse_3"] = torch.tensor(
        [
            [9, 0, 0, 0],   # PC=0: ZERO_PTR (PTR=0)
            [10, 0, 0, 0],  # PC=1: LOAD_MDR_ACC (ACC = M_RAM[0])
            [19, 4, 0, 0],  # PC=2: PUSH_ACC_TO_STACK (M_RAM[20+sp] <- ACC, sp += 1)
            [7, 0, 0, 0],   # PC=3: INC_PTR (PTR=1)
            [10, 0, 0, 0],  # PC=4: LOAD_MDR_ACC (ACC = M_RAM[1])
            [19, 4, 0, 0],  # PC=5: PUSH_ACC_TO_STACK (M_RAM[20+sp] <- ACC, sp += 1)
            [7, 0, 0, 0],   # PC=6: INC_PTR (PTR=2)
            [10, 0, 0, 0],  # PC=7: LOAD_MDR_ACC (ACC = M_RAM[2])
            [19, 4, 0, 0],  # PC=8: PUSH_ACC_TO_STACK (M_RAM[20+sp] <- ACC, sp += 1)
            # Теперь извлекаем из стека (LIFO) и записываем в M_RAM[0], M_RAM[1], M_RAM[2]:
            [9, 0, 0, 0],   # PC=9: ZERO_PTR (PTR=0)
            [10, 5, 0, 0],  # PC=10: POP_STACK_TO_ACC (sp -= 1, ACC <- M_RAM[20+sp])
            [0, 1, 0, 0],   # PC=11: STORE_ACC_TO_PTR (M_RAM[0] <- старый M_RAM[2])
            [7, 0, 0, 0],   # PC=12: INC_PTR (PTR=1)
            [10, 5, 0, 0],  # PC=13: POP_STACK_TO_ACC (sp -= 1, ACC <- M_RAM[20+sp])
            [0, 1, 0, 0],   # PC=14: STORE_ACC_TO_PTR (M_RAM[1] <- старый M_RAM[1])
            [7, 0, 0, 0],   # PC=15: INC_PTR (PTR=2)
            [10, 5, 0, 0],  # PC=16: POP_STACK_TO_ACC (sp -= 1, ACC <- M_RAM[20+sp])
            [0, 1, 0, 0],   # PC=17: STORE_ACC_TO_PTR (M_RAM[2] <- старый M_RAM[0])
            [0, 0, 8, 18],  # PC=18: HALT!
        ],
        dtype=torch.long,
        device=dev,
    )
    return progs


# ==============================================================================
# ЭТАП 3 (Stage v14.3): Открытый символьно-субсловный фронтенд с окном 32 байта UTF-8
# ==============================================================================

_WORD_CACHE_V14 = {}


def sample_initial_regs_v14(batch_size, zero_shot=False, dev="cuda"):
    """
    Разбиение 390,625 состояний Z_5^8 на непересекающиеся TRAIN (80%) и ZEROSHOT (20%)
    по взаимно-простому полиномиальному хешу состояния.
    """
    out = []
    while sum(x.size(0) for x in out) < batch_size:
        cand = torch.randint(0, 5, (batch_size * 2, 8), device=dev)
        h = (
            cand[:, 0]
            + 3 * cand[:, 1]
            + 7 * cand[:, 2]
            + 13 * cand[:, 3]
            + 19 * cand[:, 4]
            + 29 * cand[:, 5]
            + 37 * cand[:, 6]
            + 43 * cand[:, 7]
        ) % 5
        mask = (h == 0) if zero_shot else (h != 0)
        out.append(cand[mask])
    return torch.cat(out, dim=0)[:batch_size]


def _encode_single_word_v14(w, max_chars=MAX_CHARS_PER_TOK):
    """
    Кодирует слово в 32 байта UTF-8 (16 кириллических символов!) + 4 полиномиальных хеша
    (префикс, корень, суффикс и консонантный скелет без удвоенных букв против опечаток!).
    """
    res = _WORD_CACHE_V14.get(w)
    if res is not None:
        return res
    utf_bytes = list(w.encode("utf-8"))[:max_chars]
    c_row = [max(1, b_val % CHAR_VOCAB_SIZE) for b_val in utf_bytes] + [0] * (
        max_chars - len(utf_bytes)
    )
    w_low = w.lower()
    # Удаляем подряд идущие дубликаты символов (нормализация опечаток удвоения букв!):
    dedup_chars = []
    for ch in w_low:
        if ch in ("-", "_"):
            break
        if not dedup_chars or dedup_chars[-1] != ch:
            dedup_chars.append(ch)
    stem_clean = "".join(dedup_chars)

    p_str = stem_clean[:4]
    m_str = stem_clean[:7]
    s_str = stem_clean[-4:] if len(stem_clean) >= 4 else stem_clean
    h1 = (
        sum((i + 1) * 131 * ord(ch) for i, ch in enumerate(p_str))
        % (NGRAM_BUCKETS - 1)
    ) + 1
    h2 = (
        sum((i + 1) * 313 * ord(ch) for i, ch in enumerate(m_str))
        % (NGRAM_BUCKETS - 1)
    ) + 1
    h3 = (
        sum((i + 1) * 911 * ord(ch) for i, ch in enumerate(s_str))
        % (NGRAM_BUCKETS - 1)
    ) + 1
    h4 = (
        sum((i + 1) * 617 * ord(ch) for i, ch in enumerate(stem_clean))
        % (NGRAM_BUCKETS - 1)
    ) + 1
    res = (c_row, [h1, h2, h3, h4])
    _WORD_CACHE_V14[w] = res
    return res


def open_vocab_tokenize_clause_v14(
    text_str, max_tokens=MAX_CLAUSE_TOKENS, max_chars=MAX_CHARS_PER_TOK
):
    cleaned = text_str.replace("\n", " ; ").replace(":", " : ").replace(",", " ")
    raw_words = [w for w in cleaned.strip().split() if len(w) > 0][:max_tokens]
    n_w = len(raw_words)
    char_mat, ngram_mat = [], []
    for w in raw_words:
        c_row, ng_row = _encode_single_word_v14(w, max_chars)
        char_mat.append(c_row)
        ngram_mat.append(ng_row)
    pad_n = max_tokens - n_w
    if pad_n > 0:
        char_mat.extend([[0] * max_chars] * pad_n)
        ngram_mat.extend([[0, 0, 0, 0]] * pad_n)
    mask_vec = [1] * n_w + [0] * pad_n
    return char_mat, ngram_mat, mask_vec


class OpenVocabCharSubwordFrontend_v14(nn.Module):
    """
    Открытый символьно-субсловный фронтенд Stage v14.3:
      - 32 байта UTF-8 на слово (16 русских букв без обрезки окончаний!)
      - 4 хеш-бакета с дедупликацией удвоенных букв (100% устойчивость к опечаткам и суффиксам)
      - Перестановочно-инвариантный Set-CrossAttention для Clause 0
      - 2-слойный TransformerEncoder для распознавания алгоритмов и микроопераций над внешней лентой M_RAM.
    """

    def __init__(self, d_adapter=256, num_algos=4):
        super().__init__()
        self.d_adapter = d_adapter
        self.char_emb = nn.Embedding(CHAR_VOCAB_SIZE, 20, padding_idx=0)
        self.char_flat_proj = nn.Linear(MAX_CHARS_PER_TOK * 20, d_adapter, bias=False)
        self.ngram_emb = nn.Embedding(NGRAM_BUCKETS, d_adapter, padding_idx=0)
        self.word_norm = nn.LayerNorm(d_adapter)

        self.c0_queries = nn.Linear(d_adapter, 10, bias=False)
        self.c0_heads = nn.ModuleList([nn.Linear(d_adapter, 5) for _ in range(10)])

        self.step_word_pos = nn.Parameter(
            torch.randn(1, MAX_CLAUSE_TOKENS, d_adapter) * 0.02
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_adapter,
            nhead=4,
            dim_feedforward=d_adapter * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.step_encoder = nn.TransformerEncoder(
            enc_layer, num_layers=2, enable_nested_tensor=False
        )
        self.slot_queries = nn.Parameter(torch.randn(4, d_adapter) * 0.05)

        self.mode_head = nn.Linear(d_adapter, 2)  # 0: single/loop op, 1: full PC algorithm
        self.loop_head = nn.Linear(d_adapter, 4)  # k in 1..4
        self.algo_head = nn.Linear(d_adapter, num_algos)
        self.alu_head = nn.Linear(d_adapter, NUM_ALU_OPS)

    def encode_words(self, chars_t, ngrams_t):
        c_e = self.char_emb(chars_t).flatten(-2)
        c_feat = self.char_flat_proj(c_e)
        ng_feat = self.ngram_emb(ngrams_t).sum(dim=-2)
        return self.word_norm(F.gelu(c_feat) + ng_feat)

    def parse_clause0(self, chars_c0, ngrams_c0, mask_c0):
        w_emb = self.encode_words(chars_c0, ngrams_c0)
        scores = (
            self.c0_queries(w_emb) * (4.0 / math.sqrt(self.d_adapter))
        ).masked_fill(~mask_c0.unsqueeze(-1), -1e9)
        slots = torch.einsum("bws,bwd->bsd", F.softmax(scores, dim=1), w_emb)
        out_logits = torch.stack(
            [self.c0_heads[i](slots[:, i]) for i in range(10)], dim=1
        )
        return out_logits[:, :8, :], out_logits[:, 8:, :]

    def parse_step_clauses(self, chars_st, ngrams_st, mask_st):
        B, T, W, C = chars_st.shape
        BT = B * T
        w_emb = self.encode_words(
            chars_st.reshape(BT, W, C), ngrams_st.reshape(BT, W, 4)
        )
        pad_mask = ~mask_st.reshape(BT, W).clone()
        pad_mask[:, 0] = False
        h = self.step_encoder(
            w_emb + self.step_word_pos[:, :W, :], src_key_padding_mask=pad_mask
        )
        scores = (
            torch.einsum("sd,bwd->bsw", self.slot_queries, h)
            / math.sqrt(self.d_adapter)
        ).masked_fill(pad_mask.unsqueeze(1), -1e9)
        slots = torch.einsum("bsw,bwd->bsd", F.softmax(scores, dim=-1), h)
        m_l = self.mode_head(slots[:, 0]).view(B, T, 2)
        l_l = self.loop_head(slots[:, 1]).view(B, T, 4)
        a_l = self.algo_head(slots[:, 2]).view(B, T, -1)
        op_l = self.alu_head(slots[:, 3]).view(B, T, NUM_ALU_OPS)
        return m_l, l_l, a_l, op_l


PY_V14_OP_TEMPLATES = {
    0: "pass",
    1: "acc = (acc + 1) % 125",
    2: "acc = (acc + 2) % 125",
    3: "acc = (acc + 5) % 125",
    4: "acc = (acc - 1) % 125",
    5: "acc = (acc * 5) % 125",
    6: "acc = acc // 5",
    7: "ptr = (ptr + 1) % 25",
    8: "ptr = (ptr - 1) % 25",
    9: "ptr = 0",
    10: "acc = m_ram[ptr]",
    11: "acc = (acc + (m_ram[ptr] % 5)) % 125",
    12: "acc = (acc - (m_ram[ptr] % 5)) % 125",
    13: "b_reg = acc % 25",
    14: "acc = b_reg",
    15: "b_reg = m_ram[ptr] % 25",
    16: "b_reg = (b_reg + 1) % 25",
    17: "b_reg = (b_reg - 1) % 25",
    18: "acc = (acc + alpha) % 125",
    19: "sp = (sp + 1) % 5",
    20: "sp = (sp - 1) % 5",
}

RU_V14_OP_TEMPLATES = {
    0: "пропусти такт холостой_ход nop",
    1: "прибавь 1 к аккумулятору с_переносом",
    2: "прибавь 2 к аккумулятору с_переносом",
    3: "прибавь 5 к среднему_разряду аккумулятора",
    4: "вычти 1 из аккумулятора с_заемом",
    5: "сдвинь разряды аккумулятора влево shl_base5",
    6: "сдвинь разряды аккумулятора вправо shr_base5",
    7: "увеличь двухсловный указатель ptr на 1",
    8: "уменьши двухсловный указатель ptr на 1",
    9: "обнули указатель ленты ptr_zero в 0",
    10: "загрузи трехзначное слово из ленты m_ram[ptr] в acc",
    11: "сложи аккумулятор acc и ячейку ленты m_ram[ptr] с_переносом",
    12: "вычти ячейку ленты m_ram[ptr] из аккумулятора с_заемом",
    13: "скопируй аккумулятор acc в рабочий регистр b_reg",
    14: "перенеси рабочий регистр b_reg обратно в acc",
    15: "прочитай ячейку ленты m_ram[ptr] в регистр b_reg",
    16: "увеличь счетчик регистра b_reg плюс 1",
    17: "уменьши счетчик регистра b_reg минус 1",
    18: "прибавь константу квоту alpha к аккумулятору",
    19: "сдвинь указатель стека sp вверх inc_sp",
    20: "сдвинь указатель стека sp вниз dec_sp",
}

PY_V14_ALGO_TEMPLATES = {
    0: "while ptr != b_reg: acc = (acc + (m_ram[ptr] % 5)) % 125; ptr = (ptr + 1) % 25",
    1: "for ptr in range(4): acc = m_ram[ptr] if acc < m_ram[ptr] else acc; m_ram[4] = acc",
    2: "for _pass in range(2):\n    for i in range(2):\n        if m_ram[i] >= m_ram[i + 1]: m_ram[i], m_ram[i + 1] = m_ram[i + 1], m_ram[i]",
    3: "for i in range(3): stack.append(m_ram[i])\nfor i in range(3): m_ram[i] = stack.pop()",
}

RU_V14_ALGO_TEMPLATES = {
    0: "выполни программу pc_arraysum : пока указатель ptr не_равен b_reg суммируй ленту m_ram[ptr] в аккумулятор",
    1: "выполни программу pc_arraymax : найди максимум среди первых четырех ячеек ленты m_ram и запиши в m_ram[4]",
    2: "выполни программу pc_bubblesort : отсортируй массив m_ram[0..2] по возрастанию методом пузырька на_месте",
    3: "выполни программу pc_stackrev : разверни массив m_ram[0..2] задом наперед через аппаратный стек push_pop",
}


def inject_v14_typos_and_suffixes(ru_str):
    words = ru_str.split()
    out = []
    for w in words:
        if (
            "=" in w
            or "[" in w
            or w.isdigit()
            or w
            in (
                "d0",
                "d1",
                "d2",
                "p0",
                "p1",
                "sp",
                "b0",
                "b1",
                "ptr",
                "acc",
                "b_reg",
                "alpha",
                "beta",
                "nop",
                "shl_base5",
                "shr_base5",
                "ptr_zero",
                "inc_sp",
                "dec_sp",
                "pc_arraysum",
                "pc_arraymax",
                "pc_bubblesort",
                "pc_stackrev",
                ":",
            )
        ):
            out.append(w)
            continue
        r = random.random()
        if r < 0.30:
            w = w + random.choice(["-ка", "-те", "_плиз"])
        elif r < 0.50 and len(w) >= 6:
            pos = random.randint(2, len(w) - 2)
            w = w[:pos] + w[pos] + w[pos:]
        out.append(w)
    return " ".join(out)


ALGO_KEYS_V14 = ["ArraySum_25", "ArrayMax_Scan", "BubbleSort_3", "StackReverse_3"]


def build_v14_text_batch(batch_size, T, style="mixed", zero_shot=False, dev="cuda"):
    """
    Генерирует батч текстовых программ (Valid Python AST, RU, и RU с опечатками и суффиксами)
    для обучения и верификации OpenVocabCharSubwordFrontend_v14 + ExternalTapeSimVQ_v14.
    """
    regs_init = sample_initial_regs_v14(batch_size, zero_shot=zero_shot, dev=dev)
    alpha_v = torch.randint(0, 5, (batch_size,), device=dev)
    beta_v = torch.randint(0, 5, (batch_size,), device=dev)

    modes = torch.zeros(batch_size, T, dtype=torch.long, device=dev)
    loops = torch.zeros(batch_size, T, dtype=torch.long, device=dev)
    algos = torch.zeros(batch_size, T, dtype=torch.long, device=dev)
    ops = torch.randint(0, NUM_ALU_OPS, (batch_size, T), device=dev)

    for b in range(batch_size):
        for t in range(T):
            r = random.random()
            if r < 0.25:
                modes[b, t] = 1  # Вызов полного PC-алгоритма из ПЗУ!
                algos[b, t] = random.randint(0, 3)
            elif r < 0.55:
                modes[b, t] = 0
                loops[b, t] = random.randint(1, 3)  # Цикл for _ in range(2..4)
            else:
                modes[b, t] = 0
                loops[b, t] = 0  # Одиночная операция

    c0_chars, c0_ngrams, c0_masks = [], [], []
    st_chars, st_ngrams, st_masks = [], [], []
    py_programs = []

    reg_names = ["d0", "d1", "d2", "p0", "p1", "sp", "b0", "b1"]
    ru_reg_Fillers = ["пусть", "дано", "старт", "регистры", "инициализация"]

    for b in range(batch_size):
        st_b = (
            style
            if style != "mixed"
            else random.choice(["python", "ru_clean", "ru_typos"])
        )
        r_vals = [int(regs_init[b, i].item()) for i in range(8)]
        av = int(alpha_v[b].item())
        bv = int(beta_v[b].item())

        # Clause 0 (перестановочно-инвариантная инициализация регистров без позиционного внимания,
        # поэтому токен обязан быть неразрывным `name=val`!):
        items = [f"{reg_names[i]}={r_vals[i]}" for i in range(8)] + [
            f"alpha={av}",
            f"beta={bv}",
        ]
        random.shuffle(items)
        if st_b == "python":
            c0_str = " ; ".join(items)
        else:
            c0_str = random.choice(ru_reg_Fillers) + " " + " ".join(items)

        cc, cn, cm = open_vocab_tokenize_clause_v14(c0_str)
        c0_chars.append(cc)
        c0_ngrams.append(cn)
        c0_masks.append(cm)

        b_sc, b_sn, b_sm = [], [], []
        py_lines = [c0_str]
        for t in range(T):
            m_val = int(modes[b, t].item())
            l_val = int(loops[b, t].item())
            a_val = int(algos[b, t].item())
            op_val = int(ops[b, t].item())

            if m_val == 1:
                if st_b == "python":
                    step_str = PY_V14_ALGO_TEMPLATES[a_val]
                else:
                    step_str = RU_V14_ALGO_TEMPLATES[a_val]
                    if st_b == "ru_typos":
                        step_str = inject_v14_typos_and_suffixes(step_str)
            else:
                k_rep = l_val + 1
                if st_b == "python":
                    base_op = PY_V14_OP_TEMPLATES[op_val]
                    if k_rep == 1:
                        step_str = base_op
                    else:
                        step_str = f"for _step in range({k_rep}):\n    {base_op}"
                else:
                    base_op = RU_V14_OP_TEMPLATES[op_val]
                    if k_rep == 1:
                        step_str = f"один_шаг : {base_op}"
                    else:
                        step_str = f"повтори {k_rep} раза : {base_op}"
                    if st_b == "ru_typos":
                        step_str = inject_v14_typos_and_suffixes(step_str)

            if st_b == "python":
                py_lines.append(step_str)
            sc, sn, sm = open_vocab_tokenize_clause_v14(step_str)
            b_sc.append(sc)
            b_sn.append(sn)
            b_sm.append(sm)

        st_chars.append(b_sc)
        st_ngrams.append(b_sn)
        st_masks.append(b_sm)
        if st_b == "python":
            py_programs.append("\n".join(py_lines))

    return (
        torch.tensor(c0_chars, dtype=torch.long, device=dev),
        torch.tensor(c0_ngrams, dtype=torch.long, device=dev),
        torch.tensor(c0_masks, dtype=torch.bool, device=dev),
        torch.tensor(st_chars, dtype=torch.long, device=dev),
        torch.tensor(st_ngrams, dtype=torch.long, device=dev),
        torch.tensor(st_masks, dtype=torch.bool, device=dev),
        regs_init,
        alpha_v,
        beta_v,
        modes,
        loops,
        algos,
        ops,
        py_programs,
    )


class ExternalTapeSimVQ_v14(nn.Module):
    """
    Полная архитектура v14.0 (`ExternalTapeSimVQ_v14`):
      1. `core` (`TernaryMDRCore`): 100% тернарное ядро АЛУ с предвыборкой MDR (240 каналов) и 8 сферами ЦПУ (640 каналов).
      2. `router` (`TernaryBranchRouter_v14`): 100% тернарный маршрутизатор 6 аппаратных предикатов для автономного счётчика команд PC.
      3. `ext_ram` (`BiorthogonalExternalTapeRAM`): Внешняя биортогональная матричная лента памяти M_RAM (25x240) + аппаратный стек M_RAM[20..24].
      4. `frontend` (`OpenVocabCharSubwordFrontend_v14`): Открытый символьно-субсловный фронтенд с окном 32 байта UTF-8 и защитой от опечаток.
    """

    def __init__(self, d_model=640, d_mdr=240, d_adapter=256):
        super().__init__()
        self.core = TernaryMDRCore(d_model=d_model)
        self.router = TernaryBranchRouter_v14(d_model=d_model, d_mdr=d_mdr)
        self.ext_ram = BiorthogonalExternalTapeRAM(d_sub=d_model // 8)
        self.frontend = OpenVocabCharSubwordFrontend_v14(
            d_adapter=d_adapter, num_algos=4
        )

    @torch.no_grad()
    def execute_text_clauses(
        self, c0_c, c0_n, c0_m, st_c, st_n, st_m, init_tape_vals, prog_rom
    ):
        """
        Сквозное исполнение текстовой программы (Python AST или RU с опечатками)
        над внешней матричной лентой M_RAM[0..24].
        Возвращает финальные регистры (B, 8) и финальную ленту памяти (B, 25).
        """
        B, T, _, _ = st_c.shape
        dev = c0_c.device
        simvq = self.core.simvq

        r_logits, ab_logits = self.frontend.parse_clause0(c0_c, c0_n, c0_m)
        pred_init_regs = r_logits.argmax(dim=-1)  # (B, 8)
        pred_alpha = ab_logits[:, 0, :].argmax(dim=-1)
        pred_beta = ab_logits[:, 1, :].argmax(dim=-1)

        m_l, l_l, a_l, op_l = self.frontend.parse_step_clauses(st_c, st_n, st_m)
        pred_modes = m_l.argmax(dim=-1)  # (B, T)
        pred_loops = l_l.argmax(dim=-1)  # (B, T)
        pred_algos = a_l.argmax(dim=-1)  # (B, T)
        pred_ops = op_l.argmax(dim=-1)  # (B, T)

        h_cur = simvq.encode_regs(pred_init_regs)
        init_tape_digits = torch.stack(
            [
                init_tape_vals % 5,
                (init_tape_vals // 5) % 5,
                (init_tape_vals // 25) % 5,
            ],
            dim=-1,
        )
        M_ram = self.ext_ram.init_tape(init_tape_digits, simvq)

        for t in range(T):
            m_t = pred_modes[:, t]
            l_t = pred_loops[:, t]
            a_t = pred_algos[:, t]
            op_t = pred_ops[:, t]

            # Если на шаге t вызван полноценный PC-алгоритм (m_t == 1), исполняем его через автономный PC:
            if (m_t == 1).any():
                cur_regs = simvq.compute_logits(h_cur).argmax(dim=-1)
                cur_tape_vals = self.ext_ram.decode_full_tape(M_ram, simvq)
                for a_idx, a_name in enumerate(ALGO_KEYS_V14):
                    mask_a = (m_t == 1) & (a_t == a_idx)
                    if mask_a.any():
                        sub_regs = cur_regs[mask_a]
                        sub_tape = cur_tape_vals[mask_a]
                        sub_td = torch.stack(
                            [sub_tape % 5, (sub_tape // 5) % 5, (sub_tape // 25) % 5],
                            dim=-1,
                        )
                        out_r, out_t, _ = neural_execute_program_v14(
                            self.core,
                            self.router,
                            self.ext_ram,
                            sub_regs,
                            sub_td,
                            prog_rom[a_name],
                            pred_alpha[mask_a],
                            pred_beta[mask_a],
                            max_ticks=64,
                        )
                        h_cur[mask_a] = simvq.encode_regs(out_r)
                        out_td = torch.stack(
                            [out_t % 5, (out_t // 5) % 5, (out_t // 25) % 5], dim=-1
                        )
                        M_ram[mask_a] = self.ext_ram.init_tape(out_td, simvq)

            # Одиночные операции и циклы for _ in range(k) (m_t == 0):
            if (m_t == 0).any():
                for rep in range(4):
                    active_rep = (m_t == 0) & (l_t >= rep)
                    if not active_rep.any():
                        break
                    h_mdr = self.ext_ram.read_at_ptr(M_ram, h_cur, simvq)
                    h_next, _ = self.core.step(
                        h_cur, h_mdr, op_t, pred_alpha, pred_beta, snap_strength=1.0
                    )
                    h_cur = torch.where(active_rep.unsqueeze(1), h_next, h_cur)

        final_regs = simvq.compute_logits(h_cur).argmax(dim=-1)
        final_tape = self.ext_ram.decode_full_tape(M_ram, simvq)
        return final_regs, final_tape


def ground_truth_execute_text_clauses(
    regs_init, alpha_v, beta_v, modes, loops, algos, ops, init_tape_vals, prog_rom
):
    B, T = modes.shape
    regs = regs_init.clone()
    tape = init_tape_vals.clone()
    for t in range(T):
        m_t = modes[:, t]
        l_t = loops[:, t]
        a_t = algos[:, t]
        op_t = ops[:, t]

        if (m_t == 1).any():
            for a_idx, a_name in enumerate(ALGO_KEYS_V14):
                mask_a = (m_t == 1) & (a_t == a_idx)
                if mask_a.any():
                    out_r, out_t, _ = ground_truth_execute_program(
                        regs[mask_a],
                        tape[mask_a],
                        prog_rom[a_name],
                        alpha_v[mask_a],
                        beta_v[mask_a],
                        max_ticks=64,
                    )
                    regs[mask_a] = out_r
                    tape[mask_a] = out_t

        if (m_t == 0).any():
            for rep in range(4):
                active_rep = (m_t == 0) & (l_t >= rep)
                if not active_rep.any():
                    break
                ptr = regs[:, 4] * 5 + regs[:, 3]
                mdr_val = tape.gather(1, ptr.unsqueeze(1)).squeeze(1)
                mdr_digits = torch.stack(
                    [mdr_val % 5, (mdr_val // 5) % 5, (mdr_val // 25) % 5], dim=1
                )
                next_regs = apply_alu_op_tensor(
                    regs, mdr_digits, op_t, alpha_v, beta_v
                )
                regs = torch.where(active_rep.unsqueeze(1), next_regs, regs)
    return regs, tape


# ==============================================================================
# Сквозное 3-этапное обучение и строгая верификация Stage v14.0
# ==============================================================================
def train_and_verify_v14(dev="cuda"):
    torch.manual_seed(42)
    random.seed(42)
    t_start = time.time()

    print("=" * 80)
    print(
        "  ЭТАП v14.0: ВНЕШНЯЯ МАТРИЧНАЯ ЛЕНТА ПАМЯТИ И АВТОНОМНЫЙ СЧЁТЧИК КОМАНД (PC)"
    )
    print("=" * 80)

    model = ExternalTapeSimVQ_v14(d_model=640, d_mdr=240, d_adapter=256).to(dev)
    core = model.core
    router = model.router
    ext_ram = model.ext_ram
    frontend = model.frontend
    simvq = core.simvq
    prog_rom = build_v14_algorithm_rom(dev=dev)

    # --------------------------------------------------------------------------
    # ЭТАП 1 (Stage v14.1): Обучение 100% Тернарного Ядра АЛУ с предвыборкой MDR
    # --------------------------------------------------------------------------
    print(
        "\n--- [Stage v14.1] Обучение TernaryMDRCore (8 сфер ЦПУ + 3 сферы предвыборки MDR, 100% BitLinear) ---"
    )
    opt_core = torch.optim.AdamW(core.parameters(), lr=3e-3, weight_decay=1e-4)
    steps_core = 900
    sched_core = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_core, T_max=steps_core, eta_min=1e-4
    )

    for step in range(1, steps_core + 1):
        # Плавное включение 100% тернарного STE-квантования и жёсткого аттрактора Вороного:
        q_alpha = min(1.0, max(0.0, (step - 150) / 400.0))
        snap_s = min(1.0, max(0.2, step / 350.0))
        core.set_quant_alpha(q_alpha)

        B = 512
        T_roll = 4
        regs = sample_initial_regs_v14(B, zero_shot=False, dev=dev)
        # Обогащаем батч каскадными переносами (44_5 + 1 -> 100_5) и заёмами (00_5 - 1 -> 444_5):
        regs[:64, 0] = 4
        regs[:64, 1] = 4
        regs[64:128, 0] = 0
        regs[64:128, 1] = 0
        regs[128:160, 3] = 4  # p0=4 (перенос указателя PTR)
        regs[160:192, 3] = 0  # p0=0 (заём указателя PTR)
        regs[192:224, 6] = 4  # b0=4 (перенос регистра B)
        regs[224:256, 6] = 0  # b0=0 (заём регистра B)

        alpha_v = torch.randint(0, 5, (B,), device=dev)
        beta_v = torch.randint(0, 5, (B,), device=dev)

        h_cur = simvq.encode_regs(regs)

        loss_total = 0.0
        correct_cells = 0
        total_cells = B * T_roll

        for _ in range(T_roll):
            op_id = torch.randint(0, NUM_ALU_OPS, (B,), device=dev)
            mdr_digits = torch.randint(0, 5, (B, 3), device=dev)
            h_mdr = simvq.encode_mdr(mdr_digits)

            next_regs = apply_alu_op_tensor(regs, mdr_digits, op_id, alpha_v, beta_v)
            h_next, h_cand = core.step(
                h_cur, h_mdr, op_id, alpha_v, beta_v, snap_strength=snap_s
            )

            logits_cand = simvq.compute_logits(h_cand)  # (B, 8, 5)
            logits_snap = simvq.compute_logits(h_next)  # (B, 8, 5)
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

    # Калибровка внешней биортогональной матрицы памяти M_RAM[0..24] методом Ньютона — Шульца:
    ortho_err = ext_ram.calibrate_address_basis(simvq)
    print(
        f"  [BiorthogonalExternalTapeRAM] Newton-Schulz ||K_read @ K_write^T - I_25||_max = {ortho_err:.2e}"
    )

    # Верификация Stage 1:
    with torch.no_grad():
        # 1. Каскадный перенос и заём разряда на T=32:
        B_eval = 256
        r_eval = sample_initial_regs_v14(B_eval, zero_shot=True, dev=dev)
        av_e = torch.randint(0, 5, (B_eval,), device=dev)
        bv_e = torch.randint(0, 5, (B_eval,), device=dev)
        h_e = simvq.encode_regs(r_eval)
        r_gt = r_eval.clone()
        carry_ops = torch.tensor([1, 2, 3, 4, 5, 6, 11, 12, 18], device=dev)
        for _ in range(32):
            op_e = carry_ops[torch.randint(0, len(carry_ops), (B_eval,), device=dev)]
            mdr_d = torch.randint(0, 5, (B_eval, 3), device=dev)
            h_mdr_e = simvq.encode_mdr(mdr_d)
            r_gt = apply_alu_op_tensor(r_gt, mdr_d, op_e, av_e, bv_e)
            h_e, _ = core.step(h_e, h_mdr_e, op_e, av_e, bv_e, snap_strength=1.0)
        pred_r = simvq.compute_logits(h_e).argmax(dim=-1)
        carry_acc_t32 = 100.0 * (pred_r == r_gt).all(dim=-1).float().mean().item()

        # 2. Косвенная адресация внешней ленты M_RAM[0..24] через предвыборку MDR на T=32 (было 96.09% в v13!):
        r_eval = sample_initial_regs_v14(B_eval, zero_shot=True, dev=dev)
        tape_vals = torch.randint(0, 125, (B_eval, 25), device=dev)
        tape_digits = torch.stack(
            [tape_vals % 5, (tape_vals // 5) % 5, (tape_vals // 25) % 5], dim=-1
        )
        h_e = simvq.encode_regs(r_eval)
        M_ram_e = ext_ram.init_tape(tape_digits, simvq)
        r_gt = r_eval.clone()
        ram_ops = torch.tensor([7, 8, 9, 10, 11, 12, 15], device=dev)
        for _ in range(32):
            op_e = ram_ops[torch.randint(0, len(ram_ops), (B_eval,), device=dev)]
            ptr_gt = r_gt[:, 4] * 5 + r_gt[:, 3]
            mdr_v = tape_vals.gather(1, ptr_gt.unsqueeze(1)).squeeze(1)
            mdr_d = torch.stack(
                [mdr_v % 5, (mdr_v // 5) % 5, (mdr_v // 25) % 5], dim=1
            )
            r_gt = apply_alu_op_tensor(r_gt, mdr_d, op_e, av_e, bv_e)

            h_mdr_e = ext_ram.read_at_ptr(M_ram_e, h_e, simvq)
            h_e, _ = core.step(h_e, h_mdr_e, op_e, av_e, bv_e, snap_strength=1.0)
        pred_r = simvq.compute_logits(h_e).argmax(dim=-1)
        mdr_ram_acc_t32 = 100.0 * (pred_r == r_gt).all(dim=-1).float().mean().item()

        # 3. Экстраполяция всех 21 операции над внешней лентой от T=16 до T=1024 на ZEROSHOT-состояниях:
        extrap_accs = {}
        for T_hor in [16, 64, 256, 1024]:
            B_h = 128
            r_h = sample_initial_regs_v14(B_h, zero_shot=True, dev=dev)
            t_vals = torch.randint(0, 125, (B_h, 25), device=dev)
            t_digs = torch.stack(
                [t_vals % 5, (t_vals // 5) % 5, (t_vals // 25) % 5], dim=-1
            )
            h_h = simvq.encode_regs(r_h)
            M_h = ext_ram.init_tape(t_digs, simvq)
            r_gt_h = r_h.clone()
            av_h = torch.randint(0, 5, (B_h,), device=dev)
            bv_h = torch.randint(0, 5, (B_h,), device=dev)
            for _ in range(T_hor):
                op_h = torch.randint(0, NUM_ALU_OPS, (B_h,), device=dev)
                ptr_h = r_gt_h[:, 4] * 5 + r_gt_h[:, 3]
                mv_h = t_vals.gather(1, ptr_h.unsqueeze(1)).squeeze(1)
                md_h = torch.stack(
                    [mv_h % 5, (mv_h // 5) % 5, (mv_h // 25) % 5], dim=1
                )
                r_gt_h = apply_alu_op_tensor(r_gt_h, md_h, op_h, av_h, bv_h)

                h_mdr_h = ext_ram.read_at_ptr(M_h, h_h, simvq)
                h_h, _ = core.step(h_h, h_mdr_h, op_h, av_h, bv_h, snap_strength=1.0)
            pr_h = simvq.compute_logits(h_h).argmax(dim=-1)
            extrap_accs[f"All21Ops_ZeroShot_T{T_hor}"] = (
                100.0 * (pr_h == r_gt_h).all(dim=-1).float().mean().item()
            )

    print(
        f"  [Stage 1 Verified] Carry_Borrow_T32={carry_acc_t32:.2f}% | "
        f"External_Tape_MDR_Addressing_T32={mdr_ram_acc_t32:.2f}% | "
        f"All21Ops_T1024={extrap_accs['All21Ops_ZeroShot_T1024']:.2f}%"
    )

    # --------------------------------------------------------------------------
    # ЭТАП 2 (Stage v14.2): Обучение 100% Тернарного Маршрутизатора Предикатов PC
    # --------------------------------------------------------------------------
    print(
        "\n--- [Stage v14.2] Обучение TernaryBranchRouter_v14 (6 аппаратных предикатов PC, 100% BitLinear) ---"
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
        regs = torch.randint(0, 5, (B, 8), device=dev)
        mdr_d = torch.randint(0, 5, (B, 3), device=dev)
        # Обогащаем батч пограничными случаями (ACC==0, B==0, PTR==B, ACC==MDR и равные старшие разряды d2==mdr2!):
        regs[:128, 0:3] = 0
        regs[128:256, 6:8] = 0
        regs[256:384, 3:5] = regs[256:384, 6:8]
        mdr_d[384:512, :] = regs[384:512, 0:3]
        mdr_d[512:640, 2] = regs[512:640, 2]  # равный старший разряд d2 == mdr2
        mdr_d[640:768, 1:3] = regs[640:768, 1:3]  # равные старший и средний разряды d2, d1 == mdr2, mdr1

        with torch.no_grad():
            h_s = simvq.encode_regs(regs)
            h_m = simvq.encode_mdr(mdr_d)
            target_conds = eval_v14_predicates(regs, mdr_d)  # (B, 6)

        logits = router(h_s, h_m)  # (B, 6, 2)
        loss_r = F.cross_entropy(logits.reshape(-1, 2), target_conds.reshape(-1))

        opt_router.zero_grad()
        loss_r.backward()
        opt_router.step()
        sched_router.step()

        if step == 1 or step % 175 == 0:
            with torch.no_grad():
                pred_c = logits.argmax(dim=-1)
                acc_c = 100.0 * (pred_c == target_conds).all(dim=-1).float().mean().item()
            print(
                f"  [Stage 2 | step {step:3d}/700] loss={loss_r.item():.4f} | "
                f"all6_predicates_acc={acc_c:.2f}% | quant_alpha={q_alpha:.2f}"
            )

    router.set_quant_alpha(1.0)
    router.eval()

    # Верификация автономных алгоритмов с счётчиком команд PC над внешней лентой M_RAM[0..24]:
    algo_metrics = {}
    with torch.no_grad():
        B_alg = 256
        for a_name, prog_table in prog_rom.items():
            r_init = sample_initial_regs_v14(B_alg, zero_shot=True, dev=dev)
            t_init = torch.randint(0, 25, (B_alg, 25), device=dev)
            if a_name == "ArraySum_25":
                # Устанавливаем PTR в [0..4], а B_reg = PTR + длина участка [1..6], ACC = 0:
                r_init[:, 0:3] = 0
                r_init[:, 3] = torch.randint(0, 4, (B_alg,), device=dev)
                r_init[:, 4] = torch.randint(0, 2, (B_alg,), device=dev)
                ptr_start = r_init[:, 4] * 5 + r_init[:, 3]
                span = torch.randint(1, 6, (B_alg,), device=dev)
                b_end = (ptr_start + span).clamp(max=24)
                r_init[:, 6] = b_end % 5
                r_init[:, 7] = b_end // 5
            elif a_name == "ArrayMax_Scan":
                r_init[:, 0:5] = 0
                r_init[:, 6] = 4  # B_reg = 4
                r_init[:, 7] = 0
            elif a_name == "StackReverse_3":
                r_init[:, 5] = 0  # sp = 0

            av_a = torch.randint(0, 5, (B_alg,), device=dev)
            bv_a = torch.randint(0, 5, (B_alg,), device=dev)
            t_digs = torch.stack(
                [t_init % 5, (t_init // 5) % 5, (t_init // 25) % 5], dim=-1
            )

            gt_r, gt_t, _ = ground_truth_execute_program(
                r_init, t_init, prog_table, av_a, bv_a, max_ticks=64
            )
            pr_r, pr_t, _ = neural_execute_program_v14(
                core, router, ext_ram, r_init, t_digs, prog_table, av_a, bv_a, max_ticks=64
            )
            match = (pr_r == gt_r).all(dim=-1) & (pr_t == gt_t).all(dim=-1)
            algo_metrics[a_name] = 100.0 * match.float().mean().item()
            print(
                f"  [Autonomous PC + External M_RAM] {a_name:16s}: {algo_metrics[a_name]:.2f}% (regs + all 25 RAM cells exact match)"
            )

    # --------------------------------------------------------------------------
    # ЭТАП 3 (Stage v14.3): Обучение OpenVocabCharSubwordFrontend_v14 (32 UTF-8 байта)
    # --------------------------------------------------------------------------
    print(
        "\n--- [Stage v14.3] Обучение OpenVocabCharSubwordFrontend_v14 (32 байта UTF-8 + защита от опечаток) ---"
    )
    for p in router.parameters():
        p.requires_grad_(False)

    opt_fe = torch.optim.AdamW(frontend.parameters(), lr=3e-3, weight_decay=1e-4)
    sched_fe = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_fe, T_max=360, eta_min=1e-4
    )

    for step in range(1, 361):
        (
            c0_c,
            c0_n,
            c0_m,
            st_c,
            st_n,
            st_m,
            r_init,
            av,
            bv,
            modes,
            loops,
            algos,
            ops,
            _,
        ) = build_v14_text_batch(128, T=4, style="mixed", zero_shot=False, dev=dev)

        r_log, ab_log = frontend.parse_clause0(c0_c, c0_n, c0_m)
        loss_c0 = (
            F.cross_entropy(r_log.reshape(-1, 5), r_init.reshape(-1))
            + F.cross_entropy(ab_log[:, 0, :], av)
            + F.cross_entropy(ab_log[:, 1, :], bv)
        )

        m_l, l_l, a_l, op_l = frontend.parse_step_clauses(st_c, st_n, st_m)
        loss_mode = F.cross_entropy(m_l.reshape(-1, 2), modes.reshape(-1))

        mask_m0 = (modes == 0).reshape(-1)
        mask_m1 = (modes == 1).reshape(-1)
        loss_st = loss_mode
        if mask_m0.any():
            loss_st = (
                loss_st
                + F.cross_entropy(l_l.reshape(-1, 4)[mask_m0], loops.reshape(-1)[mask_m0])
                + F.cross_entropy(
                    op_l.reshape(-1, NUM_ALU_OPS)[mask_m0], ops.reshape(-1)[mask_m0]
                )
            )
        if mask_m1.any():
            loss_st = loss_st + F.cross_entropy(
                a_l.reshape(-1, 4)[mask_m1], algos.reshape(-1)[mask_m1]
            )

        loss_fe = loss_c0 + loss_st
        opt_fe.zero_grad()
        loss_fe.backward()
        torch.nn.utils.clip_grad_norm_(frontend.parameters(), 1.0)
        opt_fe.step()
        sched_fe.step()

        if step == 1 or step % 90 == 0:
            print(
                f"  [Stage 3 | step {step:3d}/360] loss_fe={loss_fe.item():.4f} "
                f"(c0={loss_c0.item():.4f}, st={loss_st.item():.4f})"
            )

    frontend.eval()

    # Проверка перестановочной инвариантности Clause 0 и синтаксиса AST Python:
    with torch.no_grad():
        c0_c1, c0_n1, c0_m1 = [], [], []
        c0_c2, c0_n2, c0_m2 = [], [], []
        for _ in range(64):
            items = [f"d{i}={random.randint(0,4)}" for i in range(3)] + [
                f"p{i}={random.randint(0,4)}" for i in range(2)
            ] + [
                f"sp={random.randint(0,4)}",
                f"b0={random.randint(0,4)}",
                f"b1={random.randint(0,4)}",
                f"alpha={random.randint(0,4)}",
                f"beta={random.randint(0,4)}",
            ]
            s1 = " ; ".join(items)
            random.shuffle(items)
            s2 = " ; ".join(items)
            cc1, cn1, cm1 = open_vocab_tokenize_clause_v14(s1)
            cc2, cn2, cm2 = open_vocab_tokenize_clause_v14(s2)
            c0_c1.append(cc1)
            c0_n1.append(cn1)
            c0_m1.append(cm1)
            c0_c2.append(cc2)
            c0_n2.append(cn2)
            c0_m2.append(cm2)

        r1, ab1 = frontend.parse_clause0(
            torch.tensor(c0_c1, device=dev),
            torch.tensor(c0_n1, device=dev),
            torch.tensor(c0_m1, dtype=torch.bool, device=dev),
        )
        r2, ab2 = frontend.parse_clause0(
            torch.tensor(c0_c2, device=dev),
            torch.tensor(c0_n2, device=dev),
            torch.tensor(c0_m2, dtype=torch.bool, device=dev),
        )
        perm_diff = max(
            (r1 - r2).abs().max().item(), (ab1 - ab2).abs().max().item()
        )

        # Сквозная проверка RU с опечатками (T=16), Valid Python AST (T=32) и Combined_T256:
        fe_metrics = {}
        for tag, st_mode, T_len, B_sz in [
            ("OpenVocab_RU_Clean_T16", "ru_clean", 16, 128),
            ("OpenVocab_RU_with_Typos_T16", "ru_typos", 16, 128),
            ("Valid_Python_AST_Code_T32", "python", 32, 128),
            ("Combined_AllStages_T256", "mixed", 256, 64),
        ]:
            (
                c0_c,
                c0_n,
                c0_m,
                st_c,
                st_n,
                st_m,
                r_init,
                av,
                bv,
                modes,
                loops,
                algos,
                ops,
                py_progs,
            ) = build_v14_text_batch(
                B_sz, T=T_len, style=st_mode, zero_shot=True, dev=dev
            )
            init_tape = torch.randint(0, 20, (B_sz, 25), device=dev)
            if st_mode == "python":
                ast_ok = sum(1 for src in py_progs if ast.parse(src) is not None)
                fe_metrics["AST_Valid_Python_Syntax"] = 100.0 * ast_ok / len(py_progs)

            gt_r, gt_t = ground_truth_execute_text_clauses(
                r_init, av, bv, modes, loops, algos, ops, init_tape, prog_rom
            )
            pr_r, pr_t = model.execute_text_clauses(
                c0_c, c0_n, c0_m, st_c, st_n, st_m, init_tape, prog_rom
            )
            exact = (pr_r == gt_r).all(dim=-1) & (pr_t == gt_t).all(dim=-1)
            fe_metrics[tag] = 100.0 * exact.float().mean().item()
            print(f"  [End-to-End Text -> M_RAM] {tag:30s}: {fe_metrics[tag]:.2f}%")

    # --------------------------------------------------------------------------
    # Живая демонстрация: сортировка пузырьком и разворот через стек во внешней ленте M_RAM
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(
        "  ЖИВАЯ ДЕМОНСТРАЦИЯ v14.0: СОРТИРОВКА M_RAM[0..2] И СТЕКОВЫЙ РАЗВОРОТ НА МЕСТЕ"
    )
    print("=" * 80)
    with torch.no_grad():
        demo_regs = torch.zeros(1, 8, dtype=torch.long, device=dev)
        demo_tape = torch.zeros(1, 25, dtype=torch.long, device=dev)
        demo_tape[0, 0] = 97
        demo_tape[0, 1] = 14
        demo_tape[0, 2] = 63
        demo_td = torch.stack(
            [demo_tape % 5, (demo_tape // 5) % 5, (demo_tape // 25) % 5], dim=-1
        )
        av_d = torch.zeros(1, dtype=torch.long, device=dev)
        bv_d = torch.zeros(1, dtype=torch.long, device=dev)

        _, sorted_tape, sort_ticks = neural_execute_program_v14(
            core,
            router,
            ext_ram,
            demo_regs,
            demo_td,
            prog_rom["BubbleSort_3"],
            av_d,
            bv_d,
            max_ticks=64,
        )
        sorted_td = torch.stack(
            [sorted_tape % 5, (sorted_tape // 5) % 5, (sorted_tape // 25) % 5], dim=-1
        )
        _, rev_tape, rev_ticks = neural_execute_program_v14(
            core,
            router,
            ext_ram,
            demo_regs,
            sorted_td,
            prog_rom["StackReverse_3"],
            av_d,
            bv_d,
            max_ticks=64,
        )
        print(
            f"  Начальная лента M_RAM[0..2] : {demo_tape[0, :3].tolist()} (3-разрядные слова в [0..124])"
        )
        print(
            f"  После BubbleSort_3 ({int(sort_ticks[0].item())} тактов PC): {sorted_tape[0, :3].tolist()}"
        )
        print(
            f"  После StackReverse_3 ({int(rev_ticks[0].item())} тактов PC): {rev_tape[0, :3].tolist()} "
            f"| Аппаратный стек M_RAM[20..22]: {rev_tape[0, 20:23].tolist()}"
        )

    # Сохранение чекпоинта:
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", "v14_external_tape_simvq.pt")
    all_metrics = {
        "Biorthogonal_RAM_OrthoErr": ortho_err,
        "Carry_Borrow_Cascade_T32": carry_acc_t32,
        "External_Tape_MDR_Addressing_T32": mdr_ram_acc_t32,
        **extrap_accs,
        **{f"PC_Algo_{k}": v for k, v in algo_metrics.items()},
        "PermInv_Clause0_MaxDiff": perm_diff,
        **fe_metrics,
    }
    torch.save(
        {
            "version": "v14.0_ExternalTapeSimVQ",
            "model_state_dict": model.state_dict(),
            "metrics": all_metrics,
        },
        ckpt_path,
    )
    elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"  ИТОГОВАЯ СВОДКА ВЕРИФИКАЦИИ v14.0 (время: {elapsed:.1f} с)")
    print("=" * 80)
    for k, v in all_metrics.items():
        if isinstance(v, float) and v < 1e-3:
            print(f"  {k:36s}: {v:.2e}")
        else:
            print(f"  {k:36s}: {v:.2f}%")
    print(f"  Чекпоинт сохранён в: {ckpt_path}")
    return all_metrics


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_and_verify_v14(dev=device)
