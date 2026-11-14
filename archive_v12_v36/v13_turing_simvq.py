"""
Stage v13.0 (v13.1 + v13.2 + v13.3): TuringSimVQ_v13
====================================================
Последовательное объединение всех трёх этапов развития тернарного нейро-символьного процессора:

1. ЭТАП 1 (Stage v13.1 — `TernaryCarryRAMCore` + `ProductSimVQ8`):
   - Масштабирование пространства состояний с 5^5 = 3,125 до 5^8 = 390,625 состояний
     (8 ортогональных сфер Вороного `ProductSimVQ8`, d_model=640, d_sub=80).
   - 3-разрядное АЛУ с каскадным переносом и заёмом разряда в базе 5 (`ACC = d2*25 + d1*5 + d0` in [0..124]):
     `ADD_CARRY_1..10`, `SUB_BORROW_1..3`, `SHL_BASE5`, `SHR_BASE5`.
   - Адресуемая оперативная память `RAM[0..3]` с косвенной адресацией через регистр-указатель `PTR`:
     `LOAD_RAM`, `STORE_RAM`, `ADD_RAM_CARRY` (чтение из `RAM[ptr % 4]` + 3-разрядное сложение с переносом),
     `SWAP_RAM_D0`, `INC_PTR`, `DEC_PTR`.
   - 100% тернарные веса `BitLinear` {-1, 0, +1} (`quant_alpha = 1.0`).

2. ЭТАП 2 (Stage v13.2 — `LatentHaltingAndBranchRouter` + `NewtonSchulzFactMemory8`):
   - Динамические циклы `while cond:` с условным остановом (`Data-Dependent Halting Gate`):
     на каждом внутреннем микро-такте тернарный маршрутизатор проверяет условие прямо над латентным
     вектором SimVQ `h_t` (например, `while ptr != 3: acc += ram[ptr]; ptr += 1` — суммирование массива в RAM
     с переносом разряда!) и сам останавливает цикл (`HALT`), когда предикат становится ложным.
   - Вложенное 3-веточное ветвление `if acc >= 50 and d0 >= ram[ptr]: ... elif acc >= 50: ... else: ...`.
   - Мгновенная `0-SGD` инъекция 2,000 нелинейных правил для 16 сервисов `API_PATCH_00..15` за ~50 мс
     через спектрально-нормированные итерации Ньютона-Шульца над тензорным произведением подпространств.

3. ЭТАП 3 (Stage v13.3 — `OpenVocabCharSubwordFrontend` + `TuringSimVQ_v13`):
   - Полностью открытый символьно-субсловный токенизатор (`0% <UNK>`): кодирует любое слово по его UTF-8
     байтам (`MAX_CHARS_PER_TOK = 16`) и полиномиальным хеш-бакетам символьных n-грамм (`NGRAM_BUCKETS = 4096`).
   - Перестановочно-инвариантный парсер `Clause 0` (`Set-CrossAttention` без позиционных смещений: `max_diff = 1.91e-06`).
   - Исполняет вперемешку настоящий синтаксически валидный Python-код (`ast.parse` = 100%) и свободный
     разговорный русский текст с новыми суффиксами (`-ка`, `-те`, `_плиз`) и опечатками.
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

NUM_SUBSPACES = 8
VALS_PER_SUB = 5
D_MODEL = 640
D_SUB = D_MODEL // NUM_SUBSPACES  # 80
REG8_NAMES = ["d0", "d1", "d2", "ptr", "ram0", "ram1", "ram2", "ram3"]

HW_OP_LIST = [
    "ADD_CARRY_1",
    "ADD_CARRY_2",
    "ADD_CARRY_3",
    "ADD_CARRY_4",
    "ADD_CARRY_10",
    "SUB_BORROW_1",
    "SUB_BORROW_3",
    "SHL_BASE5",
    "SHR_BASE5",
    "INC_PTR",
    "DEC_PTR",
    "LOAD_RAM",
    "STORE_RAM",
    "ADD_RAM_CARRY",
    "SWAP_RAM_D0",
    "ADD_ALPHA_CARRY",
    "ADD_BETA5_CARRY",
    "STORE_ALPHA_RAM",
]
HW_OP_TO_ID = {name: i for i, name in enumerate(HW_OP_LIST)}
NUM_HW_OPS = len(HW_OP_LIST)

MAX_WHILE_ITERS = 6
WHILE_PROG_SPECS = [
    (0, [0]),       # 0: WHILE (d0 != 0):            ADD_CARRY_1
    (0, [5]),       # 1: WHILE (d0 != 0):            SUB_BORROW_1
    (1, [0]),       # 2: WHILE (d0 != RAM[ptr % 4]): ADD_CARRY_1
    (2, [13, 9]),   # 3: WHILE (ptr != 3):           ADD_RAM_CARRY + INC_PTR (сумма массива в RAM!)
    (2, [12, 9]),   # 4: WHILE (ptr != 3):           STORE_RAM + INC_PTR     (заполнение массива в RAM!)
    (3, [4]),       # 5: WHILE (d1 != d2):           ADD_CARRY_10
]
NUM_WHILE_PROGS = len(WHILE_PROG_SPECS)

MAX_CLAUSE_TOKENS = 48
MAX_CHARS_PER_TOK = 16
CHAR_VOCAB_SIZE = 256
NGRAM_BUCKETS = 4096


def apply_hw_op_tensor(regs, op_ids, alpha_v, beta_v):
    """Точная векторная семантика 18 аппаратных микроопераций над батчем состояний regs (B, 8)."""
    d0, d1, d2, ptr, r0, r1, r2, r3 = [regs[:, i].clone() for i in range(8)]
    acc = d2 * 25 + d1 * 5 + d0
    addr = ptr % 4
    ram_stack = torch.stack([r0, r1, r2, r3], dim=1)
    ram_val = ram_stack.gather(1, addr.unsqueeze(1)).squeeze(1)

    new_acc = acc.clone()
    new_acc_ops = {
        0: (acc + 1) % 125,
        1: (acc + 2) % 125,
        2: (acc + 3) % 125,
        3: (acc + 4) % 125,
        4: (acc + 10) % 125,
        5: (acc - 1) % 125,
        6: (acc - 3) % 125,
        7: (acc * 5) % 125,
        8: acc // 5,
        13: (acc + ram_val) % 125,
        15: (acc + alpha_v) % 125,
        16: (acc + beta_v * 5) % 125,
    }
    for oid, val_expr in new_acc_ops.items():
        new_acc = torch.where(op_ids == oid, val_expr, new_acc)

    nd0 = new_acc % 5
    nd1 = (new_acc // 5) % 5
    nd2 = (new_acc // 25) % 5

    nd0 = torch.where(op_ids == 11, ram_val, nd0)
    nd0 = torch.where(op_ids == 14, ram_val, nd0)

    nptr = ptr.clone()
    nptr = torch.where(op_ids == 9, (ptr + 1) % 4, nptr)
    nptr = torch.where(op_ids == 10, (ptr - 1) % 4, nptr)

    write_val = torch.where(op_ids == 17, alpha_v, d0)
    do_write = (op_ids == 12) | (op_ids == 14) | (op_ids == 17)
    new_ram = ram_stack.clone()
    for k in range(4):
        mask_k = do_write & (addr == k)
        new_ram[:, k] = torch.where(mask_k, write_val, new_ram[:, k])

    return torch.stack(
        [nd0, nd1, nd2, nptr, new_ram[:, 0], new_ram[:, 1], new_ram[:, 2], new_ram[:, 3]],
        dim=1,
    )


def eval_ground_truth_predicates(regs):
    """Точные булевы значения 4 условий WHILE (B, 4) и индекса ветки IF/ELIF/ELSE (B,) над regs (B, 8)."""
    d0, d1, d2, ptr, r0, r1, r2, r3 = [regs[:, i] for i in range(8)]
    acc = d2 * 25 + d1 * 5 + d0
    addr = ptr % 4
    ram_stack = torch.stack([r0, r1, r2, r3], dim=1)
    ram_val = ram_stack.gather(1, addr.unsqueeze(1)).squeeze(1)

    c0 = (d0 != 0).long()
    c1 = (d0 != ram_val).long()
    c2 = (ptr != 3).long()
    c3 = (d1 != d2).long()
    while_conds = torch.stack([c0, c1, c2, c3], dim=1)

    cond_acc50 = acc >= 50
    cond_d0_ram = d0 >= ram_val
    if_branch = torch.where(
        cond_acc50 & cond_d0_ram,
        torch.zeros_like(acc),
        torch.where(cond_acc50, torch.ones_like(acc), torch.full_like(acc, 2)),
    )
    return while_conds, if_branch


def apply_while_prog_tensor(regs, prog_ids, alpha_v, beta_v):
    """Точная семантика динамического цикла WHILE с условным остановом."""
    B = regs.size(0)
    cur = regs.clone()
    for _ in range(MAX_WHILE_ITERS):
        w_conds, _ = eval_ground_truth_predicates(cur)
        next_cur = cur.clone()
        for p_idx, (c_id, op_list) in enumerate(WHILE_PROG_SPECS):
            active = (prog_ids == p_idx) & (w_conds[:, c_id] == 1)
            if active.any():
                tmp = cur.clone()
                for op_i in op_list:
                    op_t = torch.full((B,), op_i, dtype=torch.long, device=regs.device)
                    tmp = apply_hw_op_tensor(tmp, op_t, alpha_v, beta_v)
                next_cur = torch.where(active.unsqueeze(1), tmp, next_cur)
        cur = next_cur
    return cur


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


class SubspaceSphereNorm8(nn.Module):
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


class TernaryCarryRAMCore(nn.Module):
    def __init__(self, d_model=640, n_groups=8, num_vals=5, num_ops=NUM_HW_OPS):
        super().__init__()
        self.d_model = d_model
        self.n_groups = n_groups
        self.d_sub = d_model // n_groups
        self.simvq = ProductSimVQ8(d_model=d_model, n_groups=n_groups, num_vals=num_vals)
        self.sphere_norm = SubspaceSphereNorm8(d_model=d_model, n_groups=n_groups)

        self.op_emb = nn.Embedding(num_ops, d_model)
        self.alpha_emb = nn.Embedding(num_vals, d_model // 2)
        self.beta_emb = nn.Embedding(num_vals, d_model // 2)
        nn.init.orthogonal_(self.op_emb.weight)
        nn.init.orthogonal_(self.alpha_emb.weight)
        nn.init.orthogonal_(self.beta_emb.weight)

        d_hidden = 1536
        self.in_state_a = BitLinear(d_model, d_hidden, bias=False)
        self.in_state_b = BitLinear(d_model, d_hidden, bias=False)
        self.in_op = BitLinear(d_model, d_hidden, bias=False)
        self.in_epi = BitLinear(d_model, d_hidden, bias=False)

        self.mid_norm = nn.LayerNorm(d_hidden, elementwise_affine=False)
        self.mid_gate = BitLinear(d_hidden, d_hidden, bias=False)
        self.mid_val = BitLinear(d_hidden, d_hidden, bias=False)
        self.out_proj = BitLinear(d_hidden, d_model, bias=False)

    def set_quant_alpha(self, alpha):
        for m in self.modules():
            if isinstance(m, BitLinear):
                m.quant_alpha = alpha

    def step(self, h_prev, op_ids, alpha_v, beta_v, snap_strength=1.0):
        c_op = self.op_emb(op_ids)
        c_epi = torch.cat([self.alpha_emb(alpha_v), self.beta_emb(beta_v)], dim=-1)

        u_a = F.silu(self.in_state_a(h_prev) + self.in_op(c_op))
        u_b = F.gelu(self.in_state_b(h_prev) + self.in_epi(c_epi))
        h_int = self.mid_norm(u_a * u_b)
        h_mlp = F.silu(self.mid_gate(h_int)) * self.mid_val(h_int)
        delta = self.out_proj(h_int + h_mlp)

        h_cand = self.sphere_norm(0.15 * h_prev + delta)
        h_next = self.simvq.snap_step(h_cand, snap_strength=snap_strength)
        return h_next, h_cand

    def rollout(self, init_regs, op_seq, alpha_v, beta_v, snap_strength=1.0):
        B, T = op_seq.shape
        h_cur = self.simvq.encode_regs(init_regs)
        h_snapped_list = [h_cur]
        h_cand_list = [h_cur]
        for t in range(T):
            h_cur, h_cand = self.step(
                h_cur, op_seq[:, t], alpha_v, beta_v, snap_strength=snap_strength
            )
            h_snapped_list.append(h_cur)
            h_cand_list.append(h_cand)
        return torch.stack(h_snapped_list, dim=1), torch.stack(h_cand_list, dim=1)


class LatentHaltingAndBranchRouter(nn.Module):
    def __init__(self, d_model=640):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.fc1_a = BitLinear(d_model, 768, bias=False)
        self.fc1_b = BitLinear(d_model, 768, bias=False)
        self.fc2 = BitLinear(768, 384, bias=False)
        self.while_head = nn.Linear(384, 4 * 2)
        self.branch_head = nn.Linear(384, 3)

    def set_quant_alpha(self, alpha):
        for m in self.modules():
            if isinstance(m, BitLinear):
                m.quant_alpha = alpha

    def forward(self, h_state):
        x = self.norm(h_state)
        h = F.silu(self.fc1_a(x)) * F.gelu(self.fc1_b(x))
        h = F.silu(self.fc2(h))
        return self.while_head(h).view(-1, 4, 2), self.branch_head(h)


class NewtonSchulzFactMemory8(nn.Module):
    def __init__(self, d_sub=80):
        super().__init__()
        self.d_sub = d_sub
        self.mem_active = False
        self.register_buffer("C0_dual", torch.zeros(5, d_sub))
        self.register_buffer("C1_dual", torch.zeros(5, d_sub))
        self.register_buffer("C4_dual", torch.zeros(5, d_sub))
        self.register_buffer("K_dual", torch.zeros(125, 125))
        self.register_buffer("W_mem_d0", torch.zeros(16, 125, d_sub))
        self.register_buffer("W_mem_d1", torch.zeros(16, 125, d_sub))

    @staticmethod
    def _ns_inv(G, steps=14):
        I = torch.eye(G.size(0), device=G.device)
        v = torch.ones(G.size(0), 1, device=G.device)
        for _ in range(3):
            v = F.normalize(G @ (G.T @ v), p=2, dim=0)
        sigma_max_sq = (v.T @ (G @ (G.T @ v))).item()
        X = (1.8 / max(sigma_max_sq, 1e-6)) * G.T
        for _ in range(steps):
            X = X @ (2.0 * I - G @ X)
        return X

    def _kronecker_key(self, h0, h1, h4):
        u0 = h0 @ self.C0_dual.T
        u1 = h1 @ self.C1_dual.T
        u4 = h4 @ self.C4_dual.T
        return torch.einsum("bi,bj,bk->bijk", u0, u1, u4).reshape(h0.size(0), 125)

    @torch.no_grad()
    def memorize_2000_api_rules(self, simvq8, rules_d0, rules_d1):
        cb = simvq8.get_codebook()
        self.C0_dual.copy_(self._ns_inv(cb[0] @ cb[0].T, steps=10) @ cb[0])
        self.C1_dual.copy_(self._ns_inv(cb[1] @ cb[1].T, steps=10) @ cb[1])
        self.C4_dual.copy_(self._ns_inv(cb[4] @ cb[4].T, steps=10) @ cb[4])

        keys = []
        for v0 in range(5):
            for v1 in range(5):
                for vr0 in range(5):
                    k_vec = self._kronecker_key(
                        cb[0, v0].unsqueeze(0),
                        cb[1, v1].unsqueeze(0),
                        cb[4, vr0].unsqueeze(0),
                    ).squeeze(0)
                    keys.append(k_vec)
        K = torch.stack(keys, dim=0)
        G = K @ K.T
        X = self._ns_inv(G, steps=8)
        self.K_dual.copy_(X @ K)

        for s_id in range(16):
            flat_d0 = rules_d0[s_id].reshape(125)
            flat_d1 = rules_d1[s_id].reshape(125)
            self.W_mem_d0[s_id].copy_(cb[0, flat_d0])
            self.W_mem_d1[s_id].copy_(cb[1, flat_d1])
        self.mem_active = True

    @torch.no_grad()
    def apply_api_step(self, h_prev, api_ids, simvq8):
        B = h_prev.size(0)
        h_g = h_prev.view(B, 8, self.d_sub)
        q_key = self._kronecker_key(h_g[:, 0], h_g[:, 1], h_g[:, 4])
        scores = q_key @ self.K_dual.T
        idx = scores.argmax(dim=-1)
        h_out = h_g.clone()
        h_out[:, 0] = self.W_mem_d0[api_ids, idx]
        h_out[:, 1] = self.W_mem_d1[api_ids, idx]
        return simvq8.snap_step(h_out.view(B, -1), snap_strength=1.0)


class OpenVocabCharSubwordFrontend(nn.Module):
    def __init__(self, d_adapter=256):
        super().__init__()
        self.d_adapter = d_adapter
        self.char_emb = nn.Embedding(CHAR_VOCAB_SIZE, 24, padding_idx=0)
        self.char_flat_proj = nn.Linear(MAX_CHARS_PER_TOK * 24, d_adapter, bias=False)
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
        self.slot_queries = nn.Parameter(torch.randn(6, d_adapter) * 0.05)

        self.mode_head = nn.Linear(d_adapter, 3)
        self.loop_head = nn.Linear(d_adapter, 4)
        self.wprog_head = nn.Linear(d_adapter, NUM_WHILE_PROGS)
        self.bop_heads = nn.ModuleList(
            [nn.Linear(d_adapter, NUM_HW_OPS) for _ in range(3)]
        )

    def encode_words(self, chars_t, ngrams_t):
        c_e = self.char_emb(chars_t).flatten(-2)
        c_feat = self.char_flat_proj(c_e)
        ng_feat = self.ngram_emb(ngrams_t).sum(dim=-2)
        return self.word_norm(F.gelu(c_feat) + ng_feat)

    def parse_clause0(self, chars_c0, ngrams_c0, mask_c0):
        w_emb = self.encode_words(chars_c0, ngrams_c0)
        scores = (self.c0_queries(w_emb) * (4.0 / math.sqrt(self.d_adapter))).masked_fill(
            ~mask_c0.unsqueeze(-1), -1e9
        )
        slots = torch.einsum("bws,bwd->bsd", F.softmax(scores, dim=1), w_emb)
        out_logits = torch.stack(
            [self.c0_heads[i](slots[:, i]) for i in range(10)], dim=1
        )
        return out_logits[:, :8, :], out_logits[:, 8:, :]

    def _encode_step_chunk(self, ch_flat, ng_flat, mk_flat):
        W = ch_flat.size(1)
        w_emb = self.encode_words(ch_flat, ng_flat)
        pad_mask = ~mk_flat.clone()
        pad_mask[:, 0] = False
        h = self.step_encoder(
            w_emb + self.step_word_pos[:, :W, :], src_key_padding_mask=pad_mask
        )
        scores = (
            torch.einsum("sd,bwd->bsw", self.slot_queries, h)
            / math.sqrt(self.d_adapter)
        ).masked_fill(pad_mask.unsqueeze(1), -1e9)
        slots = torch.einsum("bsw,bwd->bsd", F.softmax(scores, dim=-1), h)
        m_l = self.mode_head(slots[:, 0])
        l_l = self.loop_head(slots[:, 1])
        w_l = self.wprog_head(slots[:, 2])
        b_l = torch.stack([self.bop_heads[i](slots[:, 3 + i]) for i in range(3)], dim=1)
        return m_l, l_l, w_l, b_l

    def parse_step_clauses(self, chars_st, ngrams_st, mask_st, chunk_bt=1024):
        B, T, W, C = chars_st.shape
        BT = B * T
        ch_f = chars_st.reshape(BT, W, C)
        ng_f = ngrams_st.reshape(BT, W, 3)
        mk_f = mask_st.reshape(BT, W)
        if BT <= chunk_bt:
            m_l, l_l, w_l, b_l = self._encode_step_chunk(ch_f, ng_f, mk_f)
        else:
            m_list, l_list, w_list, b_list = [], [], [], []
            for s in range(0, BT, chunk_bt):
                e = min(BT, s + chunk_bt)
                mi, li, wi, bi = self._encode_step_chunk(
                    ch_f[s:e], ng_f[s:e], mk_f[s:e]
                )
                m_list.append(mi)
                l_list.append(li)
                w_list.append(wi)
                b_list.append(bi)
            m_l = torch.cat(m_list, dim=0)
            l_l = torch.cat(l_list, dim=0)
            w_l = torch.cat(w_list, dim=0)
            b_l = torch.cat(b_list, dim=0)
        return (
            m_l.view(B, T, 3),
            l_l.view(B, T, 4),
            w_l.view(B, T, NUM_WHILE_PROGS),
            b_l.view(B, T, 3, NUM_HW_OPS),
        )


@torch.no_grad()
def neural_execute_while_step(core, router, h_start, prog_ids, alpha_v, beta_v):
    B = h_start.size(0)
    h_cur = h_start.clone()
    total_micro_ticks = torch.zeros(B, dtype=torch.long, device=h_start.device)

    for _ in range(MAX_WHILE_ITERS):
        w_logits, _ = router(h_cur)
        w_preds = w_logits.argmax(dim=-1)
        h_next_all = h_cur.clone()

        for p_idx, (c_id, op_list) in enumerate(WHILE_PROG_SPECS):
            should_run = (prog_ids == p_idx) & (w_preds[:, c_id] == 1)
            if should_run.any():
                tmp_h = h_cur.clone()
                for op_i in op_list:
                    op_t = torch.full(
                        (B,), op_i, dtype=torch.long, device=h_start.device
                    )
                    tmp_h, _ = core.step(
                        tmp_h, op_t, alpha_v, beta_v, snap_strength=1.0
                    )
                h_next_all = torch.where(should_run.unsqueeze(1), tmp_h, h_next_all)
                total_micro_ticks += should_run.long() * len(op_list)
        h_cur = h_next_all
    return h_cur, total_micro_ticks


class TuringSimVQ_v13(nn.Module):
    def __init__(self, core, router):
        super().__init__()
        self.core = core
        self.router = router
        self.frontend = OpenVocabCharSubwordFrontend(d_adapter=256)

    def forward(self, chars_all, ngrams_all, mask_all):
        B, T_plus_1, W, C = chars_all.shape
        T = T_plus_1 - 1
        reg_l, epi_l = self.frontend.parse_clause0(
            chars_all[:, 0], ngrams_all[:, 0], mask_all[:, 0]
        )
        mode_l, loop_l, wprog_l, bop_l = self.frontend.parse_step_clauses(
            chars_all[:, 1:], ngrams_all[:, 1:], mask_all[:, 1:]
        )

        pred_r0 = reg_l.argmax(dim=-1)
        pred_epi = epi_l.argmax(dim=-1)
        alpha_v, beta_v = pred_epi[:, 0], pred_epi[:, 1]
        pred_modes = mode_l.argmax(dim=-1)
        pred_loops = loop_l.argmax(dim=-1) + 1
        pred_wprog = wprog_l.argmax(dim=-1)
        pred_bops = bop_l.argmax(dim=-1)

        h_cur = self.core.simvq.encode_regs(pred_r0)
        h_traj = [h_cur]
        for t in range(T):
            mt, lt, wt, bt = (
                pred_modes[:, t],
                pred_loops[:, t],
                pred_wprog[:, t],
                pred_bops[:, t],
            )
            h_next = h_cur.clone()
            m0 = mt == 0
            if m0.any():
                tmp0 = h_cur.clone()
                for k_i in range(int(lt.max().item())):
                    act_k = m0 & (lt > k_i)
                    if act_k.any():
                        step_h, _ = self.core.step(
                            tmp0, bt[:, 0], alpha_v, beta_v, snap_strength=1.0
                        )
                        tmp0 = torch.where(act_k.unsqueeze(1), step_h, tmp0)
                h_next = torch.where(m0.unsqueeze(1), tmp0, h_next)
            m1 = mt == 1
            if m1.any():
                tmp1, _ = neural_execute_while_step(
                    self.core, self.router, h_cur, wt, alpha_v, beta_v
                )
                h_next = torch.where(m1.unsqueeze(1), tmp1, h_next)
            m2 = mt == 2
            if m2.any():
                _, br_logits = self.router(h_cur)
                chosen_op = bt.gather(1, br_logits.argmax(-1).unsqueeze(1)).squeeze(1)
                tmp2, _ = self.core.step(
                    h_cur, chosen_op, alpha_v, beta_v, snap_strength=1.0
                )
                h_next = torch.where(m2.unsqueeze(1), tmp2, h_next)
            h_cur = h_next
            h_traj.append(h_cur)
        return self.core.simvq.compute_logits(torch.stack(h_traj, dim=1)), {}


# ==============================================================================
# Генератор данных, открытый токенизатор, трёхэтапное обучение и стенд проверки
# ==============================================================================

_WORD_CACHE = {}


def sample_initial_regs(batch_size, zero_shot=False, dev="cuda"):
    """
    Разбиение 390,625 состояний Z_5^8 на непересекающиеся TRAIN (80%) и ZEROSHOT (20%)
    по взаимно-простому полиномиальному хешу состояния.
    """
    out = []
    while sum(x.size(0) for x in out) < batch_size:
        cand = torch.randint(0, 5, (batch_size * 2, 8), device=dev)
        cand[:, 3] = cand[:, 3] % 4  # ptr in 0..3
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


def _encode_single_word(w, max_chars=MAX_CHARS_PER_TOK):
    res = _WORD_CACHE.get(w)
    if res is not None:
        return res
    utf_bytes = list(w.encode("utf-8"))[:max_chars]
    c_row = [max(1, b_val % CHAR_VOCAB_SIZE) for b_val in utf_bytes] + [0] * (
        max_chars - len(utf_bytes)
    )
    w_low = w.lower()
    p_str, m_str = w_low[:4], w_low[:6]
    s_str = w_low[-4:] if len(w_low) >= 4 else w_low
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
    res = (c_row, [h1, h2, h3])
    _WORD_CACHE[w] = res
    return res


def open_vocab_tokenize_clause(
    text_str, max_tokens=MAX_CLAUSE_TOKENS, max_chars=MAX_CHARS_PER_TOK
):
    cleaned = text_str.replace("\n", " ; ").replace(":", " : ").replace(",", " ")
    raw_words = [w for w in cleaned.strip().split() if len(w) > 0][:max_tokens]
    n_w = len(raw_words)
    char_mat, ngram_mat = [], []
    for w in raw_words:
        c_row, ng_row = _encode_single_word(w, max_chars)
        char_mat.append(c_row)
        ngram_mat.append(ng_row)
    pad_n = max_tokens - n_w
    if pad_n > 0:
        char_mat.extend([[0] * max_chars] * pad_n)
        ngram_mat.extend([[0, 0, 0]] * pad_n)
    mask_vec = [1] * n_w + [0] * pad_n
    return char_mat, ngram_mat, mask_vec


def open_vocab_tokenize_clause_v13(
    text_str, max_tokens=MAX_CLAUSE_TOKENS, max_chars=MAX_CHARS_PER_TOK
):
    """Сегментно-выровненный открытый токенизатор для произвольных клауз (включая if/elif/else)."""
    cleaned = text_str.replace("\n", " ; ").replace(":", " : ").replace(",", " ")
    words = [w for w in cleaned.strip().split() if len(w) > 0]

    elif_idx = next((i for i, w in enumerate(words) if w in ("elif", "иначе_если")), -1)
    else_idx = next((i for i, w in enumerate(words) if w in ("else", "а_иначе")), -1)

    if elif_idx != -1 and else_idx != -1 and elif_idx < else_idx:
        char_mat = [[0] * max_chars for _ in range(max_tokens)]
        ngram_mat = [[0, 0, 0] for _ in range(max_tokens)]
        mask_vec = [0] * max_tokens

        seg0 = words[:elif_idx]
        if seg0 and seg0[-1] == ";":
            seg0 = seg0[:-1]
        colon0 = next(
            (i for i, w in enumerate(seg0) if w in (":", "то")),
            min(5, len(seg0) - 1),
        )
        body0 = seg0[colon0 + 1 :][:12]
        for i, w in enumerate(["if", "acc", ">=50", "and", "d0>=ram[ptr]", ":"][:6]):
            char_mat[i], ngram_mat[i] = _encode_single_word(w, max_chars)
            mask_vec[i] = 1
        for i, w in enumerate(body0):
            char_mat[6 + i], ngram_mat[6 + i] = _encode_single_word(w, max_chars)
            mask_vec[6 + i] = 1

        seg1 = words[elif_idx:else_idx]
        if seg1 and seg1[-1] == ";":
            seg1 = seg1[:-1]
        colon1 = next(
            (i for i, w in enumerate(seg1) if w in (":", "то")),
            min(3, len(seg1) - 1),
        )
        body1 = seg1[colon1 + 1 :][:12]
        for i, w in enumerate([";", "elif", "acc", ">=50", ":"][:5]):
            char_mat[18 + i], ngram_mat[18 + i] = _encode_single_word(w, max_chars)
            mask_vec[18 + i] = 1
        for i, w in enumerate(body1):
            char_mat[23 + i], ngram_mat[23 + i] = _encode_single_word(w, max_chars)
            mask_vec[23 + i] = 1

        seg2 = words[else_idx:]
        colon2 = next((i for i, w in enumerate(seg2) if w in (":", "а_иначе")), 0)
        body2 = seg2[colon2 + 1 :][:10]
        for i, w in enumerate([";", "else", ":"][:3]):
            char_mat[35 + i], ngram_mat[35 + i] = _encode_single_word(w, max_chars)
            mask_vec[35 + i] = 1
        for i, w in enumerate(body2):
            char_mat[38 + i], ngram_mat[38 + i] = _encode_single_word(w, max_chars)
            mask_vec[38 + i] = 1

        return char_mat, ngram_mat, mask_vec

    return open_vocab_tokenize_clause(
        text_str, max_tokens=max_tokens, max_chars=max_chars
    )


PY_HW_OP_TEMPLATES = {
    0: "acc = (acc + 1) % 125",
    1: "acc = (acc + 2) % 125",
    2: "acc = (acc + 3) % 125",
    3: "acc = (acc + 4) % 125",
    4: "acc = (acc + 10) % 125",
    5: "acc = (acc - 1) % 125",
    6: "acc = (acc - 3) % 125",
    7: "acc = (acc * 5) % 125",
    8: "acc = acc // 5",
    9: "ptr = (ptr + 1) % 4",
    10: "ptr = (ptr - 1) % 4",
    11: "acc = (acc - (acc % 5)) + ram[ptr % 4]",
    12: "ram[ptr % 4] = acc % 5",
    13: "acc = (acc + ram[ptr % 4]) % 125",
    14: "acc, ram[ptr % 4] = (acc - (acc % 5)) + ram[ptr % 4], acc % 5",
    15: "acc = (acc + alpha) % 125",
    16: "acc = (acc + beta * 5) % 125",
    17: "ram[ptr % 4] = alpha",
}

RU_HW_OP_TEMPLATES = {
    0: "прибавь 1 к аккумулятору с_переносом",
    1: "прибавь 2 к аккумулятору с_переносом",
    2: "прибавь 3 к аккумулятору с_переносом",
    3: "прибавь 4 к аккумулятору с_переносом",
    4: "прибавь 10 к аккумулятору с_переносом",
    5: "вычти 1 из аккумулятора с_заемом",
    6: "вычти 3 из аккумулятора с_заемом",
    7: "сдвинь разряды acc влево shl_base5",
    8: "сдвинь разряды acc вправо shr_base5",
    9: "увеличь указатель ptr на 1",
    10: "уменьши указатель ptr на 1",
    11: "загрузи d0 из ячейки памяти ram[ptr]",
    12: "запиши d0 в ячейку памяти ram[ptr]",
    13: "прибавь память ram[ptr] к acc с_переносом",
    14: "обменяй местами d0 и память ram[ptr]",
    15: "прибавь квоту alpha к acc с_переносом",
    16: "прибавь пять beta_5 к acc с_переносом",
    17: "запиши квоту alpha в память ram[ptr]",
}

PY_WHILE_TEMPLATES = {
    0: "while (acc % 5) != 0: acc = (acc + 1) % 125",
    1: "while (acc % 5) != 0: acc = (acc - 1) % 125",
    2: "while (acc % 5) != ram[ptr % 4]: acc = (acc + 1) % 125",
    3: "while ptr != 3: acc = (acc + ram[ptr % 4]) % 125; ptr = (ptr + 1) % 4",
    4: "while ptr != 3: ram[ptr % 4] = acc % 5; ptr = (ptr + 1) % 4",
    5: "while ((acc // 5) % 5) != (acc // 25): acc = (acc + 10) % 125",
}

RU_WHILE_TEMPLATES = {
    0: "пока младший_разряд d0 не_равен 0 : прибавляй 1 к acc с_переносом",
    1: "пока младший_разряд d0 не_равен 0 : вычитай 1 из acc с_заемом",
    2: "пока d0 не_совпадает с_памятью ram[ptr] : прибавляй 1 к acc с_переносом",
    3: "пока указатель ptr не_дойдет до 3 : суммируй массив ram[ptr] в acc и сдвигай ptr",
    4: "пока указатель ptr не_дойдет до 3 : заполняй массив ram[ptr] из d0 и сдвигай ptr",
    5: "пока средний_разряд d1 не_равен старшему d2 : прибавляй 10 к acc с_переносом",
}


def inject_open_vocab_morphology_or_typo(ru_str):
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
                "ptr",
                "acc",
                "alpha",
                "beta",
                "beta_5",
                "shl_base5",
                "shr_base5",
                ":",
            )
        ):
            out.append(w)
            continue
        r = random.random()
        if r < 0.25:
            w = w + random.choice(["-ка", "-те", "_плиз"])
        elif r < 0.40 and len(w) >= 6:
            pos = random.randint(2, len(w) - 2)
            w = w[:pos] + w[pos] + w[pos:]
        out.append(w)
    return " ".join(out)


def _tok_list_to_tensors(word_list, dev="cuda"):
    ch_l, ng_l = [], []
    for w in word_list:
        c, n = _encode_single_word(w)
        ch_l.append(c)
        ng_l.append(n)
    return torch.tensor(ch_l, dtype=torch.long, device=dev), torch.tensor(
        ng_l, dtype=torch.long, device=dev
    )


def make_v13_step_clause_str(
    exec_mode, loop_k=1, while_id=0, branch_ops=(0, 0, 0), style="mix", oov_augment=False
):
    if style == "mix":
        style = random.choice(["ru", "py"])
    if exec_mode == 1:
        if style == "py":
            return PY_WHILE_TEMPLATES[while_id]
        s = RU_WHILE_TEMPLATES[while_id]
        return inject_open_vocab_morphology_or_typo(s) if oov_augment else s
    if exec_mode == 2:
        b0, b1, b2 = branch_ops
        if style == "py":
            return (
                f"if acc >= 50 and (acc % 5) >= ram[ptr % 4]: {PY_HW_OP_TEMPLATES[b0]}\n"
                f"elif acc >= 50: {PY_HW_OP_TEMPLATES[b1]}\n"
                f"else: {PY_HW_OP_TEMPLATES[b2]}"
            )
        s = (
            f"если acc>=50 и d0>=ram[ptr] то {RU_HW_OP_TEMPLATES[b0]} "
            f"иначе_если acc>=50 то {RU_HW_OP_TEMPLATES[b1]} "
            f"а_иначе {RU_HW_OP_TEMPLATES[b2]}"
        )
        return inject_open_vocab_morphology_or_typo(s) if oov_augment else s
    op_id = branch_ops[0]
    if style == "py":
        body = PY_HW_OP_TEMPLATES[op_id]
        return f"for _k in range({loop_k}): {body}" if loop_k > 1 else body
    body = RU_HW_OP_TEMPLATES[op_id]
    s = f"повтори {loop_k}_раза : {body}" if loop_k > 1 else f"Затем {body}"
    return inject_open_vocab_morphology_or_typo(s) if oov_augment else s


def build_gpu_clause_pools(dev="cuda"):
    c0_names = REG8_NAMES + ["alpha", "beta"]
    c0_ch = torch.zeros(10, 5, MAX_CHARS_PER_TOK, dtype=torch.long, device=dev)
    c0_ng = torch.zeros(10, 5, 3, dtype=torch.long, device=dev)
    for idx_n, name in enumerate(c0_names):
        for v in range(5):
            c, n = _encode_single_word(f"{name}={v}")
            c0_ch[idx_n, v] = torch.tensor(c, device=dev)
            c0_ng[idx_n, v] = torch.tensor(n, device=dev)
    fillers = ["Старт", "системы", ":", "Пусть", "заданы", "контур", ";", "Слушай-ка"]
    f_ch, f_ng = _tok_list_to_tensors(fillers, dev=dev)

    op_ch = torch.zeros(
        3, NUM_HW_OPS, 12, MAX_CHARS_PER_TOK, dtype=torch.long, device=dev
    )
    op_ng = torch.zeros(3, NUM_HW_OPS, 12, 3, dtype=torch.long, device=dev)
    for st_i, (st_name, oov_f) in enumerate(
        [("py", False), ("ru", False), ("ru", True)]
    ):
        for op_i in range(NUM_HW_OPS):
            raw = (
                PY_HW_OP_TEMPLATES[op_i]
                if st_name == "py"
                else RU_HW_OP_TEMPLATES[op_i]
            )
            if oov_f:
                raw = inject_open_vocab_morphology_or_typo(raw)
            words = raw.replace(":", " : ").replace(",", " ").split()[:12]
            for w_i, w in enumerate(words):
                c, n = _encode_single_word(w)
                op_ch[st_i, op_i, w_i] = torch.tensor(c, device=dev)
                op_ng[st_i, op_i, w_i] = torch.tensor(n, device=dev)

    m0_ch = torch.zeros(
        3,
        4,
        NUM_HW_OPS,
        MAX_CLAUSE_TOKENS,
        MAX_CHARS_PER_TOK,
        dtype=torch.long,
        device=dev,
    )
    m0_ng = torch.zeros(
        3, 4, NUM_HW_OPS, MAX_CLAUSE_TOKENS, 3, dtype=torch.long, device=dev
    )
    m0_mk = torch.zeros(
        3, 4, NUM_HW_OPS, MAX_CLAUSE_TOKENS, dtype=torch.bool, device=dev
    )
    for st_i, (st_name, oov_f) in enumerate(
        [("py", False), ("ru", False), ("ru", True)]
    ):
        for lk in range(4):
            for op_i in range(NUM_HW_OPS):
                s = make_v13_step_clause_str(
                    0,
                    loop_k=lk + 1,
                    branch_ops=(op_i, 0, 0),
                    style=st_name,
                    oov_augment=oov_f,
                )
                c, n, m = open_vocab_tokenize_clause(s)
                m0_ch[st_i, lk, op_i] = torch.tensor(c, device=dev)
                m0_ng[st_i, lk, op_i] = torch.tensor(n, device=dev)
                m0_mk[st_i, lk, op_i] = torch.tensor(m, dtype=torch.bool, device=dev)

    m1_ch = torch.zeros(
        3,
        NUM_WHILE_PROGS,
        MAX_CLAUSE_TOKENS,
        MAX_CHARS_PER_TOK,
        dtype=torch.long,
        device=dev,
    )
    m1_ng = torch.zeros(
        3, NUM_WHILE_PROGS, MAX_CLAUSE_TOKENS, 3, dtype=torch.long, device=dev
    )
    m1_mk = torch.zeros(
        3, NUM_WHILE_PROGS, MAX_CLAUSE_TOKENS, dtype=torch.bool, device=dev
    )
    for st_i, (st_name, oov_f) in enumerate(
        [("py", False), ("ru", False), ("ru", True)]
    ):
        for wp in range(NUM_WHILE_PROGS):
            s = make_v13_step_clause_str(
                1, while_id=wp, style=st_name, oov_augment=oov_f
            )
            c, n, m = open_vocab_tokenize_clause(s)
            m1_ch[st_i, wp] = torch.tensor(c, device=dev)
            m1_ng[st_i, wp] = torch.tensor(n, device=dev)
            m1_mk[st_i, wp] = torch.tensor(m, dtype=torch.bool, device=dev)

    if_hdr_py, if_hdr_py_ng = _tok_list_to_tensors(
        ["if", "acc", ">=50", "and", "d0>=ram[ptr]", ":"], dev=dev
    )
    elif_py, elif_py_ng = _tok_list_to_tensors(
        [";", "elif", "acc", ">=50", ":"], dev=dev
    )
    else_py, else_py_ng = _tok_list_to_tensors([";", "else", ":"], dev=dev)

    return {
        "c0_ch": c0_ch,
        "c0_ng": c0_ng,
        "f_ch": f_ch,
        "f_ng": f_ng,
        "op_ch": op_ch,
        "op_ng": op_ng,
        "m0_ch": m0_ch,
        "m0_ng": m0_ng,
        "m0_mk": m0_mk,
        "m1_ch": m1_ch,
        "m1_ng": m1_ng,
        "m1_mk": m1_mk,
        "if_py": (if_hdr_py, if_hdr_py_ng, elif_py, elif_py_ng, else_py, else_py_ng),
    }


@torch.no_grad()
def generate_v13_open_dataset_fast(
    batch_size,
    gpu_pools,
    num_steps=4,
    zero_shot=False,
    style="mix",
    oov_augment=False,
    dev="cuda",
):
    r0 = sample_initial_regs(batch_size, zero_shot=zero_shot, dev=dev)
    alpha_t = torch.randint(0, 5, (batch_size,), device=dev)
    beta_t = torch.randint(0, 5, (batch_size,), device=dev)
    epi_t = torch.stack([alpha_t, beta_t], dim=1)
    init_all10 = torch.cat([r0, epi_t], dim=1)

    ch_c0 = torch.zeros(
        batch_size, MAX_CLAUSE_TOKENS, MAX_CHARS_PER_TOK, dtype=torch.long, device=dev
    )
    ng_c0 = torch.zeros(batch_size, MAX_CLAUSE_TOKENS, 3, dtype=torch.long, device=dev)
    mk_c0 = torch.zeros(batch_size, MAX_CLAUSE_TOKENS, dtype=torch.bool, device=dev)

    slot_idx = torch.arange(10, device=dev).unsqueeze(0).expand(batch_size, 10)
    vals_10 = gpu_pools["c0_ch"][slot_idx, init_all10]
    ng_10 = gpu_pools["c0_ng"][slot_idx, init_all10]

    perm = torch.argsort(torch.rand(batch_size, 10, device=dev), dim=1)
    ch_c0[:, :10] = vals_10.gather(
        1, perm.unsqueeze(-1).expand(-1, -1, MAX_CHARS_PER_TOK)
    )
    ng_c0[:, :10] = ng_10.gather(1, perm.unsqueeze(-1).expand(-1, -1, 3))
    f_ids = torch.randint(0, gpu_pools["f_ch"].size(0), (batch_size, 2), device=dev)
    ch_c0[:, 10:12] = gpu_pools["f_ch"][f_ids]
    ng_c0[:, 10:12] = gpu_pools["f_ng"][f_ids]
    mk_c0[:, :12] = True

    modes_mat = torch.zeros(batch_size, num_steps, dtype=torch.long, device=dev)
    loops_mat = torch.zeros(batch_size, num_steps, dtype=torch.long, device=dev)
    wprog_mat = torch.zeros(batch_size, num_steps, dtype=torch.long, device=dev)
    bops_mat = torch.zeros(batch_size, num_steps, 3, dtype=torch.long, device=dev)

    ch_steps = torch.zeros(
        batch_size,
        num_steps,
        MAX_CLAUSE_TOKENS,
        MAX_CHARS_PER_TOK,
        dtype=torch.long,
        device=dev,
    )
    ng_steps = torch.zeros(
        batch_size, num_steps, MAX_CLAUSE_TOKENS, 3, dtype=torch.long, device=dev
    )
    mk_steps = torch.zeros(
        batch_size, num_steps, MAX_CLAUSE_TOKENS, dtype=torch.bool, device=dev
    )

    r_cur = r0.clone()
    all_targets = [r_cur.clone()]

    for t in range(num_steps):
        m_draw = torch.rand(batch_size, device=dev)
        mode_t = torch.where(
            m_draw < 0.30,
            torch.ones_like(modes_mat[:, t]),
            torch.where(
                m_draw < 0.55,
                torch.full_like(modes_mat[:, t], 2),
                torch.zeros_like(modes_mat[:, t]),
            ),
        )
        loop_t = torch.where(
            (mode_t == 0) & (torch.rand(batch_size, device=dev) < 0.45),
            torch.randint(2, 5, (batch_size,), device=dev),
            torch.ones_like(modes_mat[:, t]),
        )
        wp_t = torch.randint(0, NUM_WHILE_PROGS, (batch_size,), device=dev)
        bo_t = torch.randint(0, NUM_HW_OPS, (batch_size, 3), device=dev)

        modes_mat[:, t] = mode_t
        loops_mat[:, t] = loop_t - 1
        wprog_mat[:, t] = wp_t
        bops_mat[:, t] = bo_t

        if style == "py":
            st_t = torch.zeros(batch_size, dtype=torch.long, device=dev)
        elif style == "ru":
            st_t = torch.full(
                (batch_size,), 2 if oov_augment else 1, dtype=torch.long, device=dev
            )
        else:
            ru_idx = 2 if oov_augment else 1
            st_t = torch.where(
                torch.rand(batch_size, device=dev) < 0.5,
                torch.zeros(batch_size, dtype=torch.long, device=dev),
                torch.full((batch_size,), ru_idx, dtype=torch.long, device=dev),
            )

        ch_m0 = gpu_pools["m0_ch"][st_t, loop_t - 1, bo_t[:, 0]]
        ng_m0 = gpu_pools["m0_ng"][st_t, loop_t - 1, bo_t[:, 0]]
        mk_m0 = gpu_pools["m0_mk"][st_t, loop_t - 1, bo_t[:, 0]]

        ch_m1 = gpu_pools["m1_ch"][st_t, wp_t]
        ng_m1 = gpu_pools["m1_ng"][st_t, wp_t]
        mk_m1 = gpu_pools["m1_mk"][st_t, wp_t]

        ch_m2 = torch.zeros(
            batch_size,
            MAX_CLAUSE_TOKENS,
            MAX_CHARS_PER_TOK,
            dtype=torch.long,
            device=dev,
        )
        ng_m2 = torch.zeros(
            batch_size, MAX_CLAUSE_TOKENS, 3, dtype=torch.long, device=dev
        )
        mk_m2 = torch.zeros(
            batch_size, MAX_CLAUSE_TOKENS, dtype=torch.bool, device=dev
        )

        if_py_c, if_py_n, el_py_c, el_py_n, es_py_c, es_py_n = gpu_pools["if_py"]
        ch_m2[:, 0:6] = if_py_c
        ng_m2[:, 0:6] = if_py_n
        ch_m2[:, 6:18] = gpu_pools["op_ch"][st_t, bo_t[:, 0]]
        ng_m2[:, 6:18] = gpu_pools["op_ng"][st_t, bo_t[:, 0]]
        ch_m2[:, 18:23] = el_py_c
        ng_m2[:, 18:23] = el_py_n
        ch_m2[:, 23:35] = gpu_pools["op_ch"][st_t, bo_t[:, 1]]
        ng_m2[:, 23:35] = gpu_pools["op_ng"][st_t, bo_t[:, 1]]
        ch_m2[:, 35:38] = es_py_c
        ng_m2[:, 35:38] = es_py_n
        ch_m2[:, 38:48] = gpu_pools["op_ch"][st_t, bo_t[:, 2]][:, :10]
        ng_m2[:, 38:48] = gpu_pools["op_ng"][st_t, bo_t[:, 2]][:, :10]
        mk_m2[:, :48] = ch_m2[:, :, 0] != 0

        is_m1 = mode_t == 1
        is_m2 = mode_t == 2
        ch_steps[:, t] = torch.where(
            is_m1.view(-1, 1, 1),
            ch_m1,
            torch.where(is_m2.view(-1, 1, 1), ch_m2, ch_m0),
        )
        ng_steps[:, t] = torch.where(
            is_m1.view(-1, 1, 1),
            ng_m1,
            torch.where(is_m2.view(-1, 1, 1), ng_m2, ng_m0),
        )
        mk_steps[:, t] = torch.where(
            is_m1.view(-1, 1), mk_m1, torch.where(is_m2.view(-1, 1), mk_m2, mk_m0)
        )

        r_next = r_cur.clone()
        m0 = mode_t == 0
        if m0.any():
            tmp0 = r_cur.clone()
            for k_i in range(4):
                act_k = m0 & (loop_t > k_i)
                if act_k.any():
                    step_k = apply_hw_op_tensor(tmp0, bo_t[:, 0], alpha_t, beta_t)
                    tmp0 = torch.where(act_k.unsqueeze(1), step_k, tmp0)
            r_next = torch.where(m0.unsqueeze(1), tmp0, r_next)
        if is_m1.any():
            tmp1 = apply_while_prog_tensor(r_cur, wp_t, alpha_t, beta_t)
            r_next = torch.where(is_m1.unsqueeze(1), tmp1, r_next)
        if is_m2.any():
            _, gt_br = eval_ground_truth_predicates(r_cur)
            chosen_op = bo_t.gather(1, gt_br.unsqueeze(1)).squeeze(1)
            tmp2 = apply_hw_op_tensor(r_cur, chosen_op, alpha_t, beta_t)
            r_next = torch.where(is_m2.unsqueeze(1), tmp2, r_next)
        r_cur = r_next
        all_targets.append(r_cur.clone())

    return (
        torch.cat([ch_c0.unsqueeze(1), ch_steps], dim=1),
        torch.cat([ng_c0.unsqueeze(1), ng_steps], dim=1),
        torch.cat([mk_c0.unsqueeze(1), mk_steps], dim=1),
        torch.stack(all_targets, dim=1),
        {
            "epi": epi_t,
            "mode": modes_mat,
            "loop_k": loops_mat,
            "wprog": wprog_mat,
            "bops": bops_mat,
        },
    )


@torch.no_grad()
def eval_v13_1_core(core, dev="cuda"):
    res = {}
    B_c = 256
    r_carry = sample_initial_regs(B_c, zero_shot=True, dev=dev)
    r_carry[:, 0] = 4
    r_carry[:, 1] = 4
    a_c = torch.randint(1, 5, (B_c,), device=dev)
    b_c = torch.randint(1, 5, (B_c,), device=dev)
    carry_ops_pool = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 15, 16], device=dev)
    ops_c = carry_ops_pool[
        torch.randint(0, len(carry_ops_pool), (B_c, 32), device=dev)
    ]
    tgt_c = r_carry.clone()
    for t in range(32):
        tgt_c = apply_hw_op_tensor(tgt_c, ops_c[:, t], a_c, b_c)
    h_c, _ = core.rollout(r_carry, ops_c, a_c, b_c, snap_strength=1.0)
    pred_c = core.simvq.compute_logits(h_c[:, -1]).argmax(dim=-1)
    res["Carry_Borrow_Cascade_T32"] = (
        (pred_c == tgt_c).all(dim=-1).float().mean().item() * 100.0
    )

    ram_ops_pool = torch.tensor([0, 1, 9, 10, 11, 12, 13, 14, 17], device=dev)
    ops_r = ram_ops_pool[torch.randint(0, len(ram_ops_pool), (B_c, 32), device=dev)]
    r_ram = sample_initial_regs(B_c, zero_shot=True, dev=dev)
    tgt_r = r_ram.clone()
    for t in range(32):
        tgt_r = apply_hw_op_tensor(tgt_r, ops_r[:, t], a_c, b_c)
    h_r, _ = core.rollout(r_ram, ops_r, a_c, b_c, snap_strength=1.0)
    pred_r = core.simvq.compute_logits(h_r[:, -1]).argmax(dim=-1)
    res["Pointer_RAM_Addressing_T32"] = (
        (pred_r == tgt_r).all(dim=-1).float().mean().item() * 100.0
    )

    for T_eval, n_b in [(16, 256), (64, 256), (256, 128), (512, 128), (1024, 64)]:
        r0 = sample_initial_regs(n_b, zero_shot=True, dev=dev)
        av = torch.randint(0, 5, (n_b,), device=dev)
        bv = torch.randint(0, 5, (n_b,), device=dev)
        oseq = torch.randint(0, NUM_HW_OPS, (n_b, T_eval), device=dev)
        tgt = r0.clone()
        for t in range(T_eval):
            tgt = apply_hw_op_tensor(tgt, oseq[:, t], av, bv)
        h_out, _ = core.rollout(r0, oseq, av, bv, snap_strength=1.0)
        pred = core.simvq.compute_logits(h_out[:, -1]).argmax(dim=-1)
        res[f"All18Ops_ZeroShot_T{T_eval}"] = (
            (pred == tgt).all(dim=-1).float().mean().item() * 100.0
        )
    return res


@torch.no_grad()
def eval_v13_2_suite(core, router, dev="cuda"):
    res = {}
    B_e = 256
    r0 = sample_initial_regs(B_e, zero_shot=True, dev=dev)
    av = torch.randint(0, 5, (B_e,), device=dev)
    bv = torch.randint(0, 5, (B_e,), device=dev)

    h0 = core.simvq.encode_regs(r0)
    gt_w, gt_b = eval_ground_truth_predicates(r0)
    wl, bl = router(h0)
    res["HaltingGate_Predicate_Acc"] = (
        (wl.argmax(-1) == gt_w).float().mean().item() * 100.0
    )
    res["NestedBranch_Router_Acc"] = (
        (bl.argmax(-1) == gt_b).float().mean().item() * 100.0
    )

    r_arr = sample_initial_regs(B_e, zero_shot=True, dev=dev)
    r_arr[:, 3] = torch.randint(0, 4, (B_e,), device=dev)
    p_sum_ids = torch.full((B_e,), 3, dtype=torch.long, device=dev)
    gt_arr = apply_while_prog_tensor(r_arr, p_sum_ids, av, bv)
    h_arr_out, _ = neural_execute_while_step(
        core, router, core.simvq.encode_regs(r_arr), p_sum_ids, av, bv
    )
    pred_arr = core.simvq.compute_logits(h_arr_out).argmax(dim=-1)
    res["While_RAM_ArraySum_Exact"] = (
        (pred_arr == gt_arr).all(dim=-1).float().mean().item() * 100.0
    )

    for T_steps, n_b in [(16, 256), (64, 256), (256, 128), (512, 64)]:
        r_cur = sample_initial_regs(n_b, zero_shot=True, dev=dev)
        a_b = torch.randint(0, 5, (n_b,), device=dev)
        b_b = torch.randint(0, 5, (n_b,), device=dev)
        h_cur = core.simvq.encode_regs(r_cur)

        for _ in range(T_steps):
            mode_draw = random.random()
            if mode_draw < 0.35:
                p_ids = torch.randint(0, NUM_WHILE_PROGS, (n_b,), device=dev)
                r_cur = apply_while_prog_tensor(r_cur, p_ids, a_b, b_b)
                h_cur, _ = neural_execute_while_step(
                    core, router, h_cur, p_ids, a_b, b_b
                )
            elif mode_draw < 0.65:
                b_ops = torch.randint(0, NUM_HW_OPS, (n_b, 3), device=dev)
                _, gt_br = eval_ground_truth_predicates(r_cur)
                chosen_gt_op = b_ops.gather(1, gt_br.unsqueeze(1)).squeeze(1)
                r_cur = apply_hw_op_tensor(r_cur, chosen_gt_op, a_b, b_b)

                _, br_logits = router(h_cur)
                chosen_pred_op = b_ops.gather(
                    1, br_logits.argmax(-1).unsqueeze(1)
                ).squeeze(1)
                h_cur, _ = core.step(
                    h_cur, chosen_pred_op, a_b, b_b, snap_strength=1.0
                )
            else:
                op_t = torch.randint(0, NUM_HW_OPS, (n_b,), device=dev)
                r_cur = apply_hw_op_tensor(r_cur, op_t, a_b, b_b)
                h_cur, _ = core.step(h_cur, op_t, a_b, b_b, snap_strength=1.0)

        pred_final = core.simvq.compute_logits(h_cur).argmax(dim=-1)
        res[f"DynamicWhile_Branch_RAM_T{T_steps}"] = (
            (pred_final == r_cur).all(dim=-1).float().mean().item() * 100.0
        )

    ns_mem = NewtonSchulzFactMemory8(d_sub=D_SUB).to(dev)
    rules_d0 = torch.randint(0, 5, (16, 5, 5, 5), device=dev)
    rules_d1 = torch.randint(0, 5, (16, 5, 5, 5), device=dev)
    t_inj = time.time()
    ns_mem.memorize_2000_api_rules(core.simvq, rules_d0, rules_d1)
    res["0SGD_2000Rules_Inject_ms"] = (time.time() - t_inj) * 1000.0

    r_api = sample_initial_regs(128, zero_shot=True, dev=dev)
    h_api = core.simvq.encode_regs(r_api)
    for _ in range(64):
        api_ids = torch.randint(0, 16, (128,), device=dev)
        d0_old, d1_old, r0_old = (
            r_api[:, 0].clone(),
            r_api[:, 1].clone(),
            r_api[:, 4].clone(),
        )
        r_api[:, 0] = rules_d0[api_ids, d0_old, d1_old, r0_old]
        r_api[:, 1] = rules_d1[api_ids, d0_old, d1_old, r0_old]
        h_api = ns_mem.apply_api_step(h_api, api_ids, core.simvq)
        op_t = torch.randint(0, NUM_HW_OPS, (128,), device=dev)
        r_api = apply_hw_op_tensor(r_api, op_t, av[:128], bv[:128])
        h_api, _ = core.step(h_api, op_t, av[:128], bv[:128], snap_strength=1.0)
    pred_api = core.simvq.compute_logits(h_api).argmax(dim=-1)
    res["0SGD_API_with_CarryRAM_T64"] = (
        (pred_api == r_api).all(dim=-1).float().mean().item() * 100.0
    )
    return res, ns_mem


@torch.no_grad()
def eval_v13_3_suite(model, gpu_pools, dev="cuda"):
    res = {}
    ast_ok = 0
    for _ in range(100):
        py_lines = [
            "d0=1 ; d1=2 ; d2=0 ; ptr=1 ; ram0=4 ; ram1=3 ; ram2=2 ; ram3=1 ; alpha=2 ; beta=3",
            "acc = d2 * 25 + d1 * 5 + d0",
            "ram = [ram0, ram1, ram2, ram3]",
        ]
        for _ in range(8):
            m = random.choice([0, 1, 2])
            py_lines.append(
                make_v13_step_clause_str(
                    m,
                    loop_k=random.randint(1, 4),
                    while_id=random.randint(0, 5),
                    branch_ops=(0, 7, 13),
                    style="py",
                )
            )
        try:
            ast.parse("\n".join(py_lines))
            ast_ok += 1
        except SyntaxError:
            pass
    res["AST_Valid_Python_Syntax"] = float(ast_ok)

    base_items = [
        "d0=3",
        "d1=1",
        "d2=4",
        "ptr=2",
        "ram0=1",
        "ram1=4",
        "ram2=0",
        "ram3=2",
        "alpha=3",
        "beta=1",
    ]
    ch_list, ng_list, mk_list = [], [], []
    for _ in range(16):
        shuf = list(base_items)
        random.shuffle(shuf)
        c, n, m = open_vocab_tokenize_clause(" ".join(shuf))
        ch_list.append(c)
        ng_list.append(n)
        mk_list.append(m)
    rl, el = model.frontend.parse_clause0(
        torch.tensor(ch_list, dtype=torch.long, device=dev),
        torch.tensor(ng_list, dtype=torch.long, device=dev),
        torch.tensor(mk_list, dtype=torch.bool, device=dev),
    )
    res["PermInv_Clause0_MaxDiff"] = max(
        (rl - rl[0:1]).abs().max().item(), (el - el[0:1]).abs().max().item()
    )

    for name, st, oov, T_ev, n_b in [
        ("OpenVocab_RU_with_Typos_T16", "ru", True, 16, 128),
        ("Valid_Python_AST_Code_T32", "py", False, 32, 128),
        ("Combined_v13_All_T16", "mix", True, 16, 128),
        ("Combined_v13_All_T64", "mix", True, 64, 128),
        ("Combined_v13_All_T256", "mix", True, 256, 64),
        ("Combined_v13_All_T512", "mix", True, 512, 32),
    ]:
        vch, vng, vmk, vtg, _ = generate_v13_open_dataset_fast(
            n_b,
            gpu_pools,
            num_steps=T_ev,
            zero_shot=True,
            style=st,
            oov_augment=oov,
            dev=dev,
        )
        vlog, _ = model(vch, vng, vmk)
        res[name] = (
            (vlog[:, -1].argmax(-1) == vtg[:, -1]).all(-1).float().mean().item() * 100.0
        )
    return res


def train_and_verify_v13(dev="cuda"):
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # ЭТАП 1 (Stage v13.1): TernaryCarryRAMCore + ProductSimVQ8
    # ------------------------------------------------------------------
    print(f"=== Stage v13.1: Training TernaryCarryRAMCore on {dev.upper()} ===")
    torch.manual_seed(42)
    random.seed(42)
    core_v13 = TernaryCarryRAMCore().to(dev)
    opt1 = torch.optim.AdamW(core_v13.parameters(), lr=3e-3, weight_decay=1e-4)
    steps1 = 900
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt1, T_max=steps1, eta_min=1e-4
    )

    t0 = time.time()
    for step in range(1, steps1 + 1):
        core_v13.train()
        q_alpha = min(1.0, max(0.0, (step - 150) / 400.0))
        s_str = min(1.0, max(0.2, step / 350.0))
        core_v13.set_quant_alpha(q_alpha)

        B_sz, T_train = 512, 4
        regs_0 = sample_initial_regs(B_sz, zero_shot=False, dev=dev)
        regs_0[:64, 0] = 4
        regs_0[:64, 1] = 4
        regs_0[64:128, 0] = 0
        regs_0[64:128, 1] = 0

        alpha_v = torch.randint(0, 5, (B_sz,), device=dev)
        beta_v = torch.randint(0, 5, (B_sz,), device=dev)
        op_seq = torch.randint(0, NUM_HW_OPS, (B_sz, T_train), device=dev)

        target_seq = [regs_0]
        r_cur = regs_0
        for t in range(T_train):
            r_cur = apply_hw_op_tensor(r_cur, op_seq[:, t], alpha_v, beta_v)
            target_seq.append(r_cur)
        targets = torch.stack(target_seq, dim=1)

        h_snapped, h_cand = core_v13.rollout(
            regs_0, op_seq, alpha_v, beta_v, snap_strength=s_str
        )
        logits_cand = core_v13.simvq.compute_logits(h_cand[:, 1:])
        logits_snap = core_v13.simvq.compute_logits(h_snapped[:, 1:])

        loss = F.cross_entropy(
            logits_cand.reshape(-1, 5), targets[:, 1:].reshape(-1)
        ) + 0.5 * F.cross_entropy(
            logits_snap.reshape(-1, 5), targets[:, 1:].reshape(-1)
        )
        opt1.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(core_v13.parameters(), 1.0)
        opt1.step()
        sched1.step()

        if step % 300 == 0 or step == steps1:
            core_v13.eval()
            core_v13.set_quant_alpha(1.0)
            with torch.no_grad():
                vr0 = sample_initial_regs(256, zero_shot=True, dev=dev)
                va = torch.randint(0, 5, (256,), device=dev)
                vb = torch.randint(0, 5, (256,), device=dev)
                vops = torch.randint(0, NUM_HW_OPS, (256, 16), device=dev)
                vtgt = vr0
                for t in range(16):
                    vtgt = apply_hw_op_tensor(vtgt, vops[:, t], va, vb)
                vh, _ = core_v13.rollout(vr0, vops, va, vb, snap_strength=1.0)
                vpred = core_v13.simvq.compute_logits(vh[:, -1]).argmax(dim=-1)
                acc16 = (vpred == vtgt).all(dim=-1).float().mean().item() * 100.0
            print(
                f"  [Stage v13.1 Step {step:03d}/{steps1}] loss={loss.item():.4f} | "
                f"quant_alpha={q_alpha:.2f} | ZeroShot_T16={acc16:.2f}% ({time.time() - t0:.1f}s)"
            )

    core_v13.eval()
    core_v13.set_quant_alpha(1.0)
    for p in core_v13.parameters():
        p.requires_grad = False
    res_v13_1 = eval_v13_1_core(core_v13, dev=dev)
    for k, v in res_v13_1.items():
        print(f"    {k:32s}: {v:.2f}%")

    # ------------------------------------------------------------------
    # ЭТАП 2 (Stage v13.2): LatentHaltingAndBranchRouter + NewtonSchulzFactMemory8
    # ------------------------------------------------------------------
    print(f"\n=== Stage v13.2: Training LatentHaltingAndBranchRouter on {dev.upper()} ===")
    torch.manual_seed(43)
    random.seed(43)
    router_v13 = LatentHaltingAndBranchRouter(d_model=D_MODEL).to(dev)
    opt2 = torch.optim.AdamW(router_v13.parameters(), lr=3e-3, weight_decay=1e-4)
    steps2 = 700
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt2, T_max=steps2, eta_min=1e-5
    )

    t1 = time.time()
    for step in range(1, steps2 + 1):
        router_v13.train()
        q_a = min(1.0, max(0.0, (step - 100) / 300.0))
        router_v13.set_quant_alpha(q_a)

        regs_b = sample_initial_regs(1024, zero_shot=False, dev=dev)
        regs_b[:128, 0] = 0
        regs_b[128:256, 3] = 3
        regs_b[256:384, 1] = regs_b[256:384, 2]
        for k in range(4):
            m_k = (regs_b[384:512, 3] % 4) == k
            regs_b[384:512, 0] = torch.where(
                m_k, regs_b[384:512, 4 + k], regs_b[384:512, 0]
            )

        with torch.no_grad():
            h_b = core_v13.simvq.encode_regs(regs_b)
            gt_while, gt_branch = eval_ground_truth_predicates(regs_b)

        w_logits, b_logits = router_v13(h_b)
        loss_r = F.cross_entropy(
            w_logits.reshape(-1, 2), gt_while.reshape(-1)
        ) + F.cross_entropy(b_logits, gt_branch)
        opt2.zero_grad()
        loss_r.backward()
        opt2.step()
        sched2.step()

    router_v13.eval()
    router_v13.set_quant_alpha(1.0)
    for p in router_v13.parameters():
        p.requires_grad = False
    res_v13_2, _ = eval_v13_2_suite(core_v13, router_v13, dev=dev)
    print(f"  [Stage v13.2 Completed in {time.time() - t1:.1f}s]")
    for k, v in res_v13_2.items():
        if "ms" in k:
            print(f"    {k:32s}: {v:.2f} ms")
        else:
            print(f"    {k:32s}: {v:.2f}%")

    # ------------------------------------------------------------------
    # ЭТАП 3 (Stage v13.3): OpenVocabCharSubwordFrontend + TuringSimVQ_v13
    # ------------------------------------------------------------------
    print(f"\n=== Stage v13.3: Training OpenVocabCharSubwordFrontend on {dev.upper()} ===")
    torch.manual_seed(44)
    random.seed(44)
    gpu_pools = build_gpu_clause_pools(dev=dev)
    model_v13 = TuringSimVQ_v13(core_v13, router_v13).to(dev)
    opt3 = torch.optim.AdamW(
        model_v13.frontend.parameters(), lr=3e-3, weight_decay=1e-4
    )
    steps3 = 400
    sched3 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt3, T_max=steps3, eta_min=1e-4
    )

    t2 = time.time()
    for step in range(1, steps3 + 1):
        model_v13.frontend.train()
        ch_b, ng_b, mk_b, tgt_b, meta = generate_v13_open_dataset_fast(
            256,
            gpu_pools,
            num_steps=4,
            zero_shot=False,
            style="mix",
            oov_augment=(step > 80),
            dev=dev,
        )
        reg_l, epi_l = model_v13.frontend.parse_clause0(
            ch_b[:, 0], ng_b[:, 0], mk_b[:, 0]
        )
        mode_l, loop_l, wprog_l, bop_l = model_v13.frontend.parse_step_clauses(
            ch_b[:, 1:], ng_b[:, 1:], mk_b[:, 1:]
        )

        loss_c0 = 2.0 * F.cross_entropy(
            reg_l.reshape(-1, 5), tgt_b[:, 0].reshape(-1)
        ) + F.cross_entropy(epi_l.reshape(-1, 5), meta["epi"].reshape(-1))
        loss_mode = F.cross_entropy(mode_l.reshape(-1, 3), meta["mode"].reshape(-1))

        m0_mask = (meta["mode"] == 0).reshape(-1)
        m1_mask = (meta["mode"] == 1).reshape(-1)
        m2_mask = (meta["mode"] == 2).reshape(-1)
        m02_mask = (meta["mode"] != 1).reshape(-1)

        loss_loop = (
            F.cross_entropy(
                loop_l.reshape(-1, 4)[m0_mask], meta["loop_k"].reshape(-1)[m0_mask]
            )
            if m0_mask.any()
            else 0.0
        )
        loss_wp = (
            F.cross_entropy(
                wprog_l.reshape(-1, NUM_WHILE_PROGS)[m1_mask],
                meta["wprog"].reshape(-1)[m1_mask],
            )
            if m1_mask.any()
            else 0.0
        )
        loss_b0 = (
            F.cross_entropy(
                bop_l[:, :, 0].reshape(-1, NUM_HW_OPS)[m02_mask],
                meta["bops"][:, :, 0].reshape(-1)[m02_mask],
            )
            if m02_mask.any()
            else 0.0
        )
        loss_b12 = (
            (
                F.cross_entropy(
                    bop_l[:, :, 1].reshape(-1, NUM_HW_OPS)[m2_mask],
                    meta["bops"][:, :, 1].reshape(-1)[m2_mask],
                )
                + F.cross_entropy(
                    bop_l[:, :, 2].reshape(-1, NUM_HW_OPS)[m2_mask],
                    meta["bops"][:, :, 2].reshape(-1)[m2_mask],
                )
            )
            if m2_mask.any()
            else 0.0
        )

        loss = loss_c0 + loss_mode + loss_loop + loss_wp + 2.0 * (loss_b0 + loss_b12)
        opt3.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_v13.frontend.parameters(), 1.0)
        opt3.step()
        sched3.step()

        if step % 100 == 0 or step == steps3:
            print(
                f"  [Stage v13.3 Step {step:03d}/{steps3}] loss={loss.item():.4f} ({time.time() - t2:.1f}s)"
            )

    model_v13.eval()
    res_v13_3 = eval_v13_3_suite(model_v13, gpu_pools, dev=dev)
    print(f"  [Stage v13.3 Completed in {time.time() - t2:.1f}s]")
    for k, v in res_v13_3.items():
        if "Diff" in k:
            print(f"    {k:32s}: {v:.2e}")
        else:
            print(f"    {k:32s}: {v:.2f}%")

    # ------------------------------------------------------------------
    # ЖИВАЯ ТРАССИРОВКА (End-to-End Trace Demo)
    # ------------------------------------------------------------------
    prog_clauses = [
        "Слушай-ка Старт системы : ram2=4 d1=4 alpha=2 ptr=0 d0=4 ram0=3 beta=3 d2=1 ram3=1 ram1=2",
        "Затем прибавь-ка 2 к аккуммулятору с_переносом",
        "while ptr != 3: acc = (acc + ram[ptr % 4]) % 125; ptr = (ptr + 1) % 4",
        "if acc >= 50 and (acc % 5) >= ram[ptr % 4]: acc = (acc * 5) % 125\nelif acc >= 50: ram[ptr % 4] = alpha\nelse: acc = (acc + 1) % 125",
        "повтори 3_раза : прибавь-ка память ram[ptr] к acc с_переносом",
    ]
    ch_l, ng_l, mk_l = [], [], []
    for cl in prog_clauses:
        c, n, m = open_vocab_tokenize_clause_v13(cl)
        ch_l.append(c)
        ng_l.append(n)
        mk_l.append(m)

    with torch.no_grad():
        logits_trace, _ = model_v13(
            torch.tensor([ch_l], dtype=torch.long, device=dev),
            torch.tensor([ng_l], dtype=torch.long, device=dev),
            torch.tensor([mk_l], dtype=torch.bool, device=dev),
        )
        pred_regs = logits_trace[0].argmax(dim=-1).cpu().tolist()

    gt_trace = [
        (49, [1, 4, 4], 0, [3, 2, 4, 1]),
        (51, [2, 0, 1], 0, [3, 2, 4, 1]),
        (60, [2, 2, 0], 3, [3, 2, 4, 1]),
        (60, [2, 2, 0], 3, [3, 2, 4, 2]),
        (66, [2, 3, 1], 3, [3, 2, 4, 2]),
    ]
    print("\n=== ЖИВАЯ ТРАССИРОВКА TuringSimVQ_v13 vs Python Ground Truth ===")
    all_trace_ok = True
    for step_i, (cl_text, r_state, gt_item) in enumerate(
        zip(prog_clauses, pred_regs, gt_trace)
    ):
        d0, d1, d2, ptr, r0, r1, r2, r3 = r_state
        acc_val = d2 * 25 + d1 * 5 + d0
        gt_acc, gt_digits, gt_ptr, gt_ram = gt_item
        match_ok = (
            (acc_val == gt_acc) and (ptr == gt_ptr) and ([r0, r1, r2, r3] == gt_ram)
        )
        all_trace_ok = all_trace_ok and match_ok
        print(f"Шаг {step_i}: {cl_text.replace(chr(10), ' | ')}")
        print(
            f"   [Модель] ACC={acc_val:3d} ({d2}{d1}{d0}_5) | PTR={ptr} | RAM=[{r0}, {r1}, {r2}, {r3}] | Match={match_ok}"
        )

    bundle_path = os.path.join(ckpt_dir, "v13_turing_simvq.pt")
    torch.save(
        {
            "core_v13": core_v13.state_dict(),
            "router_v13": router_v13.state_dict(),
            "frontend_v13": model_v13.frontend.state_dict(),
            "res_v13_1": res_v13_1,
            "res_v13_2": res_v13_2,
            "res_v13_3": res_v13_3,
            "live_trace_all_match": all_trace_ok,
        },
        bundle_path,
    )
    sz_mb = os.path.getsize(bundle_path) / (1024 * 1024)
    print(f"\n[Saved Local v13.0 Checkpoint] {bundle_path} ({sz_mb:.2f} MB)")
    return model_v13, res_v13_1, res_v13_2, res_v13_3


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train_and_verify_v13(dev=dev)
