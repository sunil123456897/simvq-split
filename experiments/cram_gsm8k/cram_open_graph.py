"""cram_open_graph.py — Открытый энергетический граф операций `CRAM+z` без BPTT.

Реализует:
  - `CRAMzOperatorGraphEBM(nn.Module)` с выпуклым полем Гершгорина (`m - ||W_zz||_1 >= 4.0`),
    интенсивной (`1/N`) энергией подграфов `psi_seq`, жёстким `Sparsemax`-шлюзом поверх
    скрытых состояний `Calcite-5 System-1` и локальным псевдоправдоподобием Бесага (`0` BPTT).
  - `OpenGraphCRAMSolver`: корпусно-индуцированный стемминг без английских словарей и `re`,
    секционированную эпизодическую память прецедентов `(n_ug, n_uc)`, восходящий
    синтез `Workspace Pair-Merge Forest`, 2-уровневую `5-fold OOF` калибровку
    двойного сертификата `(sigma^2_q <= theta_epi) AND (E_cert <= tau_energy)` и
    исполнение финальных кандидатов на сферическом регистре `SphericalBase10RegisterALU`.
"""

from __future__ import annotations

from collections import Counter
import itertools
import math
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .slot_extractor import (
        ExtractedSlot,
        TrainInducedQuantityExtractor,
        extract_gt_number,
        scan_question_tokens,
    )
    from .spherical_alu import SphericalBase10RegisterALU
except ImportError:
    from slot_extractor import (
        ExtractedSlot,
        TrainInducedQuantityExtractor,
        extract_gt_number,
        scan_question_tokens,
    )
    from spherical_alu import SphericalBase10RegisterALU


MAX_GRAPH_STEPS = 12
DU_HASH = 2048
DP_HASH = 2304
DR_HASH = 640
DC_HASH = 4096
N_META_FEATS = 15
N_RERANK_FEATS = 6
N_CERT_FEATS = 12
D_STEP_PSI = 16
D_Z_FIELD = 16
N_FIELD_GROUPS = 4

_SIG_PROBE = [
    math.pi + 0.713 * (i + 1) + math.sqrt(2.0 + 3.1 * i)
    for i in range(32)
]


def _det_hash(key: str, mod: int) -> int:
    h = 2166136261
    for ch in key:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return h % mod


def _eval_dag_fast(vals: Sequence[float], dag: Sequence[Tuple[int, int, int]]) -> Optional[float]:
    buf = list(vals)
    for op, a, b in dag:
        if a >= len(buf) or b >= len(buf):
            return None
        va, vb = buf[a], buf[b]
        if op == 0:
            rv = va + vb
        elif op == 1:
            rv = va - vb
        elif op == 2:
            rv = va * vb
        else:
            if abs(vb) < 1e-9:
                return None
            rv = va / vb
        if not math.isfinite(rv) or abs(rv) > 1e9:
            return None
        buf.append(rv)
    if dag:
        return float(buf[-1])
    return float(buf[0]) if buf else 0.0


def _dag_rational_sig(dag: Sequence[Tuple[int, int, int]], k_slots: int) -> Optional[float]:
    val = _eval_dag_fast(_SIG_PROBE[: max(k_slots, 1)], dag)
    if val is None or not math.isfinite(val) or abs(val) >= 1e8:
        return None
    return round(val, 4)


def is_connected_minimal_dag(
    dag: Tuple[Tuple[int, int, int], ...],
    k_slots: int,
    max_steps: int = MAX_GRAPH_STEPS,
) -> bool:
    """Проверяет синтаксис DAG, каноничность `+`/`*` и достижимость всех шагов до `s_{T-1}`."""
    if not dag or len(dag) > max_steps:
        return False
    needed: Set[int] = {k_slots + len(dag) - 1}
    for s_idx in range(len(dag) - 1, -1, -1):
        op_x, ax, bx = dag[s_idx]
        n_lim = k_slots + s_idx
        if not (0 <= op_x < 4 and 0 <= ax < n_lim and 0 <= bx < n_lim):
            return False
        if ax == bx and op_x not in (0, 2):
            return False
        if op_x in (0, 2) and ax > bx:
            return False
        if (k_slots + s_idx) not in needed:
            return False
        needed.add(ax)
        needed.add(bx)
    return True


def _prune_to_connected_dag(
    dag: Sequence[Tuple[int, int, int]],
    k_slots: int,
    max_steps: int = MAX_GRAPH_STEPS,
) -> Tuple[Tuple[int, int, int], ...]:
    """Канонизует `+`/`*` и вычищает мёртвые узлы, недостижимые от корня `s_{T-1}`."""
    if not dag or len(dag) > max_steps:
        return ()
    for s_idx, (op_x, ax, bx) in enumerate(dag):
        n_lim = k_slots + s_idx
        if not (0 <= op_x < 4 and 0 <= ax < n_lim and 0 <= bx < n_lim):
            return ()
        if ax == bx and op_x not in (0, 2):
            return ()
    needed: Set[int] = {k_slots + len(dag) - 1}
    for s_idx in range(len(dag) - 1, -1, -1):
        if (k_slots + s_idx) in needed:
            _, ax, bx = dag[s_idx]
            needed.add(ax)
            needed.add(bx)
    remap: Dict[int, int] = {i: i for i in range(k_slots)}
    clean: List[Tuple[int, int, int]] = []
    for s_idx, (op_x, ax, bx) in enumerate(dag):
        if (k_slots + s_idx) not in needed:
            continue
        ra = remap[ax]
        rb = remap[bx]
        sa, sb = (rb, ra) if (op_x in (0, 2) and ra > rb) else (ra, rb)
        remap[k_slots + s_idx] = k_slots + len(clean)
        clean.append((op_x, sa, sb))
    res = tuple(clean)
    return res if is_connected_minimal_dag(res, k_slots, max_steps=max_steps) else ()


class Sparsemax(torch.autograd.Function):
    """Точная евклидова проекция на симплекс с занулением шумового хвоста."""

    @staticmethod
    def forward(ctx, z: torch.Tensor) -> torch.Tensor:
        zs, _ = torch.sort(z, dim=-1, descending=True)
        d = z.size(-1)
        k_arr = torch.arange(1, d + 1, device=z.device, dtype=z.dtype).view(
            [1] * (z.ndim - 1) + [d]
        )
        cs = torch.cumsum(zs, dim=-1)
        valid = (1.0 + k_arr * zs) > cs
        k_max = torch.where(valid, k_arr, torch.ones_like(k_arr)).max(
            dim=-1, keepdim=True
        ).values
        cs_k = torch.gather(cs, -1, k_max.long() - 1)
        tau = (cs_k - 1.0) / k_max
        out = torch.clamp(z - tau, min=0.0)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (out,) = ctx.saved_tensors
        supp = (out > 0).to(grad_output.dtype)
        supp_cnt = supp.sum(dim=-1, keepdim=True).clamp(min=1.0)
        v_hat = (grad_output * supp).sum(dim=-1, keepdim=True) / supp_cnt
        return supp * (grad_output - v_hat)


def _question_focus_and_money(
    q_text: str,
    slots: Sequence[ExtractedSlot],
    extractor: TrainInducedQuantityExtractor,
) -> Tuple[List[str], bool]:
    """Выделяет упорядоченный список слов финального вопроса и признак денежного ответа без хардкода."""
    q_focus = extractor.extract_question_sentence(q_text)
    focus_toks = scan_question_tokens(q_focus)
    q_words = [tk.lower for tk in focus_toks if tk.kind == "WORD"]
    dollar_words = {w for sl in slots if sl.has_dollar for w in sl.right_words[:2]}
    q_wants_money = ("$" in q_focus) or bool(dollar_words & set(q_words))
    return q_words, q_wants_money


def _compute_numeric_slot_features(
    slots: Sequence[ExtractedSlot],
    q_focus_words: Sequence[str],
    extractor: TrainInducedQuantityExtractor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Строит числовые `(K, 16)` и контекстные индексные массивы `(K, 8)`."""
    k = len(slots)
    n_s = max(1, k - 1)
    max_sent = max([sl.sent_id for sl in slots] + [1])
    max_clause = max([sl.clause_id for sl in slots] + [1])

    num_feats = np.zeros((k, 16), dtype=np.float32)
    left_ctx_ids = np.zeros((k, 8), dtype=np.int64)
    right_ctx_ids = np.zeros((k, 8), dtype=np.int64)

    q_focus_set = set(q_focus_words)
    vocab = extractor.ctx_token_to_id
    for i, sl in enumerate(slots):
        av = abs(float(sl.val))
        is_int = 1.0 if abs(sl.val - round(sl.val)) < 1e-4 else 0.0
        post_set = set(sl.right_words)
        overlap = len(post_set & q_focus_set) / float(max(1, len(post_set))) if post_set else 0.0
        num_feats[i] = np.array(
            [
                math.log1p(av) / 10.0,
                1.0 if sl.val >= 0 else -1.0,
                is_int,
                1.0 if 0.0 < av < 1.0 else 0.0,
                1.0 if sl.is_pct else 0.0,
                1.0 if sl.has_dollar else 0.0,
                1.0 if sl.kind == "EXPLICIT" else 0.0,
                1.0 if sl.kind == "INDUCED_WORD" else 0.0,
                1.0 if sl.kind == "CONST" else 0.0,
                1.0 if sl.is_in_question else 0.0,
                i / float(n_s),
                sl.sent_id / float(max_sent),
                sl.clause_id / float(max_clause),
                overlap,
                1.0 if av > 100.0 else 0.0,
                1.0 if abs(av - 1.0) < 1e-4 or abs(av - 2.0) < 1e-4 else 0.0,
            ],
            dtype=np.float32,
        )

        l_bag: List[int] = []
        lw = sl.left_words[-4:]
        for w in lw:
            tid = vocab.get(w)
            if tid is not None:
                l_bag.append(tid + 1)
        for j in range(len(lw) - 1):
            tid_bi = vocab.get("bi:" + lw[j] + "_" + lw[j + 1])
            if tid_bi is not None:
                l_bag.append(tid_bi + 1)
        for pos, tid_b in enumerate(l_bag[:8]):
            left_ctx_ids[i, pos] = tid_b

        r_bag: List[int] = []
        rw = sl.right_words[:5]
        for w in rw:
            tid = vocab.get(w)
            if tid is not None:
                r_bag.append(tid + 1)
        for j in range(len(rw) - 1):
            tid_bi = vocab.get("bi:" + rw[j] + "_" + rw[j + 1])
            if tid_bi is not None:
                r_bag.append(tid_bi + 1)
        for pos, tid_b in enumerate(r_bag[:8]):
            right_ctx_ids[i, pos] = tid_b

    return num_feats, left_ctx_ids, right_ctx_ids


def _stem_by_corpus(word: str, vocab_counts: Counter) -> str:
    """Языково-независимое укорочение словоформ по частотному словарю корпуса."""
    if len(word) >= 5 and vocab_counts.get(word[:-2], 0) >= 5:
        return word[:-2]
    if len(word) >= 4 and vocab_counts.get(word[:-1], 0) >= 5:
        return word[:-1]
    return word


def _to_narrative_skeleton(
    dag: Sequence[Tuple[int, int, int]], slots: Sequence[ExtractedSlot]
) -> Tuple[Tuple[Tuple[int, int, int], ...], List[int], List[int]]:
    """Канонизует DAG с сохранением порядка повествования для земных слотов и констант."""
    k = len(slots)
    used = sorted({x for _, a, b in dag for x in (a, b) if x < k})
    u_gr = [x for x in used if slots[x].kind != "CONST"]
    u_cn = [x for x in used if slots[x].kind == "CONST"]
    order = u_gr + u_cn
    rmap = {old: nw for nw, old in enumerate(order)}
    skel = tuple(
        (
            op,
            rmap[a] if a < k else 100 + (a - k),
            rmap[b] if b < k else 100 + (b - k),
        )
        for op, a, b in dag
    )
    return skel, u_gr, u_cn


class CRAMzOperatorGraphEBM(nn.Module):
    """Интенсивная (1/N) энергетическая модель `CRAM+z` без BPTT и без `dict`/`set`-состояний.

    Объединяет:
      1) Выпуклое поле Гершгорина `z*(Y) = (mI - W_zz)^{-1} W_zy psi_sum` (`m = 5.0`,
         `||W_zz||_1 <= 0.90`, `gershgorin_margin >= 4.10`) с узловой и групповой энергией;
      2) Якорные потенциалы ролей `W_u (9, 2048)`, парных связей `W_p (7, 2304)`,
         корня вопроса `W_r (4, 640)` и сферических векторов `Calcite-5` `W_lm (9, 480)`;
      3) Тензор структурного взаимодействия `K_cram (9, 9, 7)` и жёсткий `Sparsemax`-шлюз
         `lm_norm -> sparse_gate -> Sparsemax -> sparse_role`.
    """

    def __init__(
        self,
        d_model: int = 160,
        d_unary: int = DU_HASH,
        d_pair: int = DP_HASH,
        d_root: int = DR_HASH,
        d_step: int = D_STEP_PSI,
        d_z: int = D_Z_FIELD,
        n_groups: int = N_FIELD_GROUPS,
        group_k: int = 8,
        m_convex: float = 5.0,
        w_zz_l1_cap: float = 0.90,
        step_field_scale: float = 0.03,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_unary = int(d_unary)
        self.d_pair = int(d_pair)
        self.d_root = int(d_root)
        self.d_step = int(d_step)
        self.d_z = int(d_z)
        self.n_groups = int(n_groups)
        self.group_k = int(group_k)
        self.m_convex = float(m_convex)
        self.w_zz_l1_cap = float(w_zz_l1_cap)
        self.step_field_scale = float(step_field_scale)

        # Буферы аналитического замкнутого старта Вудбери для проксимальной регуляризации
        self.register_buffer("Wu0", torch.zeros(9, self.d_unary))
        self.register_buffer("Wp0", torch.zeros(7, self.d_pair))
        self.register_buffer("Wr0", torch.zeros(4, self.d_root))
        self.register_buffer("Wlm0", torch.zeros(9, 3 * self.d_model))
        init_wm = torch.tensor(
            [
                -0.30, -0.45, -0.15, 1.40, 0.45,
                0.50, 1.10, 1.10, 1.40, 1.30,
                1.10, 1.10, -0.30, -0.60, 0.90,
            ],
            dtype=torch.float32,
        )
        self.register_buffer("wm0", init_wm.clone())

        # Якорные потенциалы ролей, связей, корня и Sparsemax-шлюза над Calcite-5
        self.W_u = nn.Parameter(torch.randn(9, self.d_unary) * 0.01)
        self.W_p = nn.Parameter(torch.randn(7, self.d_pair) * 0.01)
        self.W_r = nn.Parameter(torch.randn(4, self.d_root) * 0.01)
        self.W_lm = nn.Parameter(torch.randn(9, 3 * self.d_model) * 0.01)
        self.K_cram = nn.Parameter(torch.zeros(9, 9, 7))
        self.w_meta = nn.Parameter(init_wm.clone())

        self.lm_norm = nn.LayerNorm(3 * self.d_model)
        self.sparse_gate = nn.Linear(3 * self.d_model, 32)
        self.sparse_role = nn.Linear(32, 9, bias=False)
        nn.init.normal_(self.sparse_role.weight, std=2e-4)

        # Интенсивное выпуклое поле Гершгорина CRAM+z по дескрипторам шагов DAG (0 BPTT)
        self.step_proj = nn.Linear(self.d_step, self.d_step)
        self.step_unary_head = nn.Linear(self.d_step, 1, bias=False)
        self.W_zz_raw = nn.Parameter(torch.randn(self.d_z, self.d_z) * 0.03)
        self.W_zy = nn.Parameter(torch.randn(self.d_z, self.d_step) * 0.06)
        self.W_edge = nn.Parameter(torch.randn(self.d_step, self.d_step) * 0.04)
        self.V_groups = nn.Parameter(torch.randn(self.n_groups, self.group_k, self.d_step) * 0.06)
        self.U_z = nn.Linear(self.d_z, self.n_groups)

    @torch.no_grad()
    def set_woodbury_anchors(
        self,
        wu0: np.ndarray,
        wp0: np.ndarray,
        wr0: np.ndarray,
        wlm0: np.ndarray,
    ) -> None:
        """Фиксирует замкнутые веса Вудбери в буферах и инициализирует ими графовые матрицы."""
        dev = self.W_u.device
        for buf_name, param, arr in (
            ("Wu0", self.W_u, wu0),
            ("Wp0", self.W_p, wp0),
            ("Wr0", self.W_r, wr0),
            ("Wlm0", self.W_lm, wlm0),
        ):
            t = torch.as_tensor(arr, dtype=torch.float32, device=dev)
            getattr(self, buf_name).copy_(t)
            param.copy_(t)

    def normalized_W_zz(self) -> torch.Tensor:
        """Возвращает симметричную `W_zz` с нулевой диагональю и `||W_zz||_1 <= 0.90`."""
        sym = 0.5 * (self.W_zz_raw + self.W_zz_raw.t())
        eye = torch.eye(self.d_z, device=sym.device, dtype=sym.dtype)
        sym = sym * (1.0 - eye)
        row_l1 = sym.abs().sum(dim=-1, keepdim=True).clamp(min=1e-6)
        scale = torch.clamp(self.w_zz_l1_cap / row_l1, max=1.0)
        return sym * torch.minimum(scale, scale.t())

    def gershgorin_margin(self) -> float:
        """Возвращает запас строгой диагональной доминантности `m - ||W_zz||_1 >= 4.0`."""
        with torch.no_grad():
            max_l1 = float(self.normalized_W_zz().abs().sum(dim=-1).max().item())
            return float(self.m_convex - max_l1)

    def _bounded_couplings(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Возвращает ограниченные матрицы связей `(M_z, W_zy_b, W_e_b, V_g_b)`."""
        W_zz = self.normalized_W_zz()
        eye = torch.eye(self.d_z, device=W_zz.device, dtype=W_zz.dtype)
        A = self.m_convex * eye - W_zz
        W_zy_b = self.W_zy / torch.clamp(self.W_zy.norm() / 1.4, min=1.0)
        W_e_raw = 0.5 * (self.W_edge + self.W_edge.t())
        W_e_b = W_e_raw / torch.clamp(W_e_raw.norm() / 1.2, min=1.0)
        V_g_b = self.V_groups / torch.clamp(self.V_groups.norm() / 1.4, min=1.0)
        M_z = torch.linalg.solve(A, W_zy_b)
        return M_z, W_zy_b, W_e_b, V_g_b

    def solve_z_star(self, psi_sum: torch.Tensor) -> torch.Tensor:
        """Замкнутое решение `z*(Y) = (mI - W_zz)^{-1} W_zy_b psi_sum` (`0` шагов BPTT)."""
        M_z, _, _, _ = self._bounded_couplings()
        return F.linear(psi_sum, M_z)

    def energy_parts_from_psi_stack(
        self,
        psi_seq: torch.Tensor,
        couplings: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Точная интенсивная (`1/N`) энергия последовательности узлов `psi_seq (N, d_step)`."""
        M_z, W_zy_b, W_e_b, V_g_b = self._bounded_couplings() if couplings is None else couplings
        n_nodes = float(max(1, psi_seq.size(0)))
        psi_scaled = psi_seq / n_nodes
        psi_sum = psi_scaled.sum(dim=0)
        z_star = F.linear(psi_sum, M_z)

        e_unary = -self.step_unary_head(psi_scaled).sum()
        pair_all = torch.einsum("id,df,jf->ij", psi_scaled, W_e_b, psi_scaled)
        e_pair = -0.5 * (pair_all.sum() - torch.diagonal(pair_all).sum())

        v_proj = torch.einsum("gkd,nd->gnk", V_g_b, psi_scaled)
        group_term = v_proj.sum(dim=1).square().sum(dim=-1) - v_proj.square().sum(dim=(1, 2))
        alpha = F.softmax(self.U_z(z_star), dim=-1)
        e_group = -0.5 * (alpha * group_term).sum()
        e_z = -0.5 * (z_star * F.linear(psi_sum, W_zy_b)).sum()
        return e_unary + e_pair + e_group + e_z, e_unary, z_star

    def energy_from_psi_stack(
        self,
        psi_seq: torch.Tensor,
        couplings: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Возвращает `(E_total_intensive, z_star)` для стека узлов `psi_seq (N, d_step)`."""
        e_tot, _, z_star = self.energy_parts_from_psi_stack(psi_seq, couplings=couplings)
        return e_tot, z_star

    def conditional_candidate_energies_no_leak(
        self,
        psi_seq_true: torch.Tensor,
        target_t: int,
        cand_psis: torch.Tensor,
        couplings: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Besag Local Pseudolikelihood (`0 BPTT`, `0` утечка цели `target_t`)."""
        M_z, W_zy_b, W_e_b, V_g_b = self._bounded_couplings() if couplings is None else couplings
        n_nodes = float(max(1, psi_seq_true.size(0)))
        other_psis = torch.cat([psi_seq_true[:target_t], psi_seq_true[target_t + 1 :]], dim=0)
        other_scaled = other_psis / n_nodes
        cand_scaled = cand_psis / n_nodes
        psi_minus_t = other_scaled.sum(dim=0)

        psi_sum_cands = psi_minus_t.unsqueeze(0) + cand_scaled
        z_cands = F.linear(psi_sum_cands, M_z)

        e_unary = -self.step_unary_head(cand_scaled).squeeze(-1)
        e_pair = -F.linear(cand_scaled, W_e_b) @ psi_minus_t

        v_other = torch.einsum("gkd,nd->gnk", V_g_b, other_scaled)
        s_minus_t = v_other.sum(dim=1)
        v_minus_sq = v_other.square().sum(dim=(1, 2))
        base_g = (s_minus_t.square().sum(dim=-1) - v_minus_sq).unsqueeze(0)
        v_cand = torch.einsum("gkd,md->gmk", V_g_b, cand_scaled)
        cross_g = base_g + 2.0 * torch.einsum("gk,gmk->gm", s_minus_t, v_cand).t()
        alpha = F.softmax(self.U_z(z_cands), dim=-1)
        e_group = -0.5 * (alpha * cross_g).sum(dim=-1)
        e_z = -0.5 * (z_cands * F.linear(psi_sum_cands, W_zy_b)).sum(dim=-1)
        return e_unary + e_pair + e_group + e_z

    def conditional_pair_energies_2opt(
        self,
        psi_seq_true: torch.Tensor,
        t1: int,
        t2: int,
        cand_pairs_psi: torch.Tensor,
        couplings: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """2-Opt парный Besag без утечки (`0 BPTT`): оценивает все парные перестановки `(t1, t2)`
        при полном исключении истинных состояний `t1, t2` и замкнутом пересчёте поля `z*`."""
        M_z, W_zy_b, W_e_b, V_g_b = self._bounded_couplings() if couplings is None else couplings
        n_total = int(psi_seq_true.size(0))
        lo, hi = (int(t1), int(t2)) if int(t1) < int(t2) else (int(t2), int(t1))
        if lo == hi or lo < 0 or hi >= n_total:
            raise ValueError(f"Expected distinct valid node indices in 0..{n_total - 1}, got ({t1}, {t2})")
        n_nodes = float(max(2, n_total))
        other_psis = torch.cat(
            [psi_seq_true[:lo], psi_seq_true[lo + 1 : hi], psi_seq_true[hi + 1 :]],
            dim=0,
        )
        other_scaled = other_psis / n_nodes
        cand1 = cand_pairs_psi[:, 0, :] / n_nodes
        cand2 = cand_pairs_psi[:, 1, :] / n_nodes
        cand_pair_sum = cand1 + cand2

        psi_rest = other_scaled.sum(dim=0)
        psi_tot = psi_rest.unsqueeze(0) + cand_pair_sum
        z_cands = F.linear(psi_tot, M_z)

        e_unary = -self.step_unary_head(cand_pair_sum).squeeze(-1)
        e_pair_ext = -F.linear(cand_pair_sum, W_e_b) @ psi_rest
        e_pair_int = -(F.linear(cand1, W_e_b) * cand2).sum(dim=-1)
        e_pair = e_pair_ext + e_pair_int

        if other_scaled.size(0) > 0:
            v_other = torch.einsum("gkd,nd->gnk", V_g_b, other_scaled)
            s_rest = v_other.sum(dim=1)
            v_rest_sq = v_other.square().sum(dim=(1, 2))
        else:
            s_rest = torch.zeros(self.n_groups, self.group_k, device=cand_pairs_psi.device, dtype=cand_pairs_psi.dtype)
            v_rest_sq = torch.zeros(self.n_groups, device=cand_pairs_psi.device, dtype=cand_pairs_psi.dtype)

        v1 = torch.einsum("gkd,md->gmk", V_g_b, cand1)
        v2 = torch.einsum("gkd,md->gmk", V_g_b, cand2)
        v_tot = s_rest.unsqueeze(1) + v1 + v2
        v_cand_sq = v1.square().sum(dim=-1) + v2.square().sum(dim=-1)
        cross_g = (v_tot.square().sum(dim=-1) - v_rest_sq.unsqueeze(1) - v_cand_sq).t()

        alpha = F.softmax(self.U_z(z_cands), dim=-1)
        e_group = -0.5 * (alpha * cross_g).sum(dim=-1)
        e_z = -0.5 * (z_cands * F.linear(psi_tot, W_zy_b)).sum(dim=-1)
        return e_unary + e_pair + e_group + e_z

    def _total_discrete_graph_energy(
        self,
        y_idx: torch.Tensor,
        node_cand_bank: torch.Tensor,
        pair_edges: torch.Tensor,
        unary_node_bias: Optional[torch.Tensor],
        pair_edge_pot: Optional[torch.Tensor],
        couplings: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, float]:
        """Собирает `psi_seq` по индексам `y_idx` и считает точную полную энергию графа ограничений."""
        n_nodes = int(node_cand_bank.size(0))
        node_ar = torch.arange(n_nodes, device=node_cand_bank.device, dtype=torch.long)
        psi_cur = node_cand_bank[node_ar, y_idx]
        e_cram, _ = self.energy_from_psi_stack(psi_cur, couplings=couplings)
        e_tot = float(e_cram.item())
        if unary_node_bias is not None:
            e_tot += float(unary_node_bias[node_ar, y_idx].sum().item())
        if pair_edge_pot is not None and pair_edges.numel() > 0:
            eu = pair_edges[:, 0]
            ev = pair_edges[:, 1]
            edge_ar = torch.arange(pair_edges.size(0), device=node_cand_bank.device, dtype=torch.long)
            e_tot += float(pair_edge_pot[edge_ar, y_idx[eu], y_idx[ev]].sum().item())
        return psi_cur, e_tot

    @staticmethod
    def _build_edge_adjacency_lists(
        n_nodes: int, pair_edges: torch.Tensor
    ) -> Tuple[List[List[Tuple[int, int, int]]], List[Tuple[int, int, int]]]:
        """Строит списки смежности `inc[node] = [(edge_idx, nbr, side)]` без `dict`/`set`."""
        inc: List[List[Tuple[int, int, int]]] = [[] for _ in range(n_nodes)]
        edge_list: List[Tuple[int, int, int]] = []
        if pair_edges.numel() > 0:
            edges_cpu = pair_edges.detach().cpu().tolist()
            for e_idx, uv in enumerate(edges_cpu):
                u = int(uv[0])
                v = int(uv[1])
                if u == v:
                    continue
                inc[u].append((e_idx, v, 0))
                inc[v].append((e_idx, u, 1))
                edge_list.append((e_idx, u, v))
        return inc, edge_list

    @torch.no_grad()
    def _one_flip_sweep(
        self,
        y_cur: torch.Tensor,
        psi_cur: torch.Tensor,
        node_cand_bank: torch.Tensor,
        inc: List[List[Tuple[int, int, int]]],
        unary_node_bias: Optional[torch.Tensor],
        pair_edge_pot: Optional[torch.Tensor],
        couplings: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> bool:
        """Один координатный 1-flip проход Бесага по всем узлам без dict/set."""
        n_nodes = int(node_cand_bank.size(0))
        changed = False
        for t in range(n_nodes):
            e_cands = self.conditional_candidate_energies_no_leak(
                psi_cur, t, node_cand_bank[t], couplings=couplings
            )
            if unary_node_bias is not None:
                e_cands = e_cands + unary_node_bias[t]
            if pair_edge_pot is not None and inc[t]:
                for e_idx, nbr, side in inc[t]:
                    yn = int(y_cur[nbr].item())
                    if side == 0:
                        e_cands = e_cands + pair_edge_pot[e_idx, :, yn]
                    else:
                        e_cands = e_cands + pair_edge_pot[e_idx, yn, :]
            cur_c = int(y_cur[t].item())
            best_c = int(torch.argmin(e_cands).item())
            if float(e_cands[best_c].item()) < float(e_cands[cur_c].item()) - 1e-8:
                y_cur[t] = best_c
                psi_cur[t] = node_cand_bank[t, best_c]
                changed = True
        return changed

    @torch.no_grad()
    def besag_1flip_relax(
        self,
        y_init: torch.Tensor,
        node_cand_bank: torch.Tensor,
        pair_edges: torch.Tensor,
        unary_node_bias: Optional[torch.Tensor] = None,
        pair_edge_pot: Optional[torch.Tensor] = None,
        max_sweeps: int = 6,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Одноузловая координатная релаксация Бесага (`1-flip`, `0 BPTT`)."""
        couplings = self._bounded_couplings()
        n_nodes = int(node_cand_bank.size(0))
        y_cur = y_init.clone().long().to(node_cand_bank.device)
        psi_cur, _ = self._total_discrete_graph_energy(
            y_cur, node_cand_bank, pair_edges, unary_node_bias, pair_edge_pot, couplings=couplings
        )
        inc, _ = self._build_edge_adjacency_lists(n_nodes, pair_edges)

        for _ in range(max_sweeps):
            if not self._one_flip_sweep(
                y_cur, psi_cur, node_cand_bank, inc, unary_node_bias, pair_edge_pot, couplings
            ):
                break

        psi_fin, e_fin = self._total_discrete_graph_energy(
            y_cur, node_cand_bank, pair_edges, unary_node_bias, pair_edge_pot, couplings=couplings
        )
        return y_cur, psi_fin, e_fin

    @torch.no_grad()
    def block_gibbs_2opt_relax(
        self,
        y_init: torch.Tensor,
        node_cand_bank: torch.Tensor,
        pair_edges: torch.Tensor,
        unary_node_bias: Optional[torch.Tensor] = None,
        pair_edge_pot: Optional[torch.Tensor] = None,
        max_sweeps: int = 6,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Детерминированная `0-BPTT 2-Opt Block-Gibbs` релаксация по связанным парам `(u, v)`.
        Чередует 1-узловой шаг и полную совместную оценку пары состояний на каждом ребре."""
        couplings = self._bounded_couplings()
        n_nodes, n_cands, _ = node_cand_bank.shape
        y_cur = y_init.clone().long().to(node_cand_bank.device)
        psi_cur, _ = self._total_discrete_graph_energy(
            y_cur, node_cand_bank, pair_edges, unary_node_bias, pair_edge_pot, couplings=couplings
        )
        inc, edge_list = self._build_edge_adjacency_lists(n_nodes, pair_edges)

        for _ in range(max_sweeps):
            changed = self._one_flip_sweep(
                y_cur, psi_cur, node_cand_bank, inc, unary_node_bias, pair_edge_pot, couplings
            )

            # Парный 2-opt блочный проход по связанным рёбрам (u, v)
            for _, u, v in edge_list:
                b_u = node_cand_bank[u].unsqueeze(1).expand(n_cands, n_cands, self.d_step)
                b_v = node_cand_bank[v].unsqueeze(0).expand(n_cands, n_cands, self.d_step)
                cand_pairs = torch.stack([b_u, b_v], dim=2).reshape(n_cands * n_cands, 2, self.d_step)
                grid_e = self.conditional_pair_energies_2opt(
                    psi_cur, u, v, cand_pairs, couplings=couplings
                ).view(n_cands, n_cands)

                if unary_node_bias is not None:
                    grid_e = grid_e + unary_node_bias[u].unsqueeze(1) + unary_node_bias[v].unsqueeze(0)

                if pair_edge_pot is not None:
                    for e2, nbr, side in inc[u]:
                        if nbr == v:
                            pot_uv = pair_edge_pot[e2] if side == 0 else pair_edge_pot[e2].t()
                            grid_e = grid_e + pot_uv
                        else:
                            yn = int(y_cur[nbr].item())
                            vec_u = pair_edge_pot[e2, :, yn] if side == 0 else pair_edge_pot[e2, yn, :]
                            grid_e = grid_e + vec_u.unsqueeze(1)
                    for e2, nbr, side in inc[v]:
                        if nbr != u:
                            yn = int(y_cur[nbr].item())
                            vec_v = pair_edge_pot[e2, :, yn] if side == 0 else pair_edge_pot[e2, yn, :]
                            grid_e = grid_e + vec_v.unsqueeze(0)

                cur_u = int(y_cur[u].item())
                cur_v = int(y_cur[v].item())
                flat_best = int(torch.argmin(grid_e.reshape(-1)).item())
                best_u = flat_best // n_cands
                best_v = flat_best % n_cands
                if float(grid_e[best_u, best_v].item()) < float(grid_e[cur_u, cur_v].item()) - 1e-8:
                    y_cur[u] = best_u
                    y_cur[v] = best_v
                    psi_cur[u] = node_cand_bank[u, best_u]
                    psi_cur[v] = node_cand_bank[v, best_v]
                    changed = True

            if not changed:
                break

        psi_fin, e_fin = self._total_discrete_graph_energy(
            y_cur, node_cand_bank, pair_edges, unary_node_bias, pair_edge_pot, couplings=couplings
        )
        return y_cur, psi_fin, e_fin

    def _score_psi_batched(
        self,
        psi_pad: torch.Tensor,
        step_mask: torch.Tensor,
        seq_lens: torch.Tensor,
        couplings: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Векторная оценка интенсивной энергии Гершгорина по батчу из `M_tot` графов."""
        M_z, W_zy_b, W_e_b, V_g_b = couplings
        inv_n = 1.0 / seq_lens.clamp(min=1.0).view(-1, 1, 1)
        psi_scaled = psi_pad * step_mask.unsqueeze(-1) * inv_n
        psi_sum = psi_scaled.sum(dim=1)
        z_star = F.linear(psi_sum, M_z)

        e_unary = -self.step_unary_head(psi_scaled).squeeze(-1).sum(dim=-1)
        pair_all = torch.einsum("mid,df,mjf->mij", psi_scaled, W_e_b, psi_scaled)
        pair_diag = torch.diagonal(pair_all, dim1=1, dim2=2).sum(dim=-1)
        e_pair = -0.5 * (pair_all.sum(dim=(1, 2)) - pair_diag)

        v_proj = torch.einsum("gkd,mnd->mgnk", V_g_b, psi_scaled)
        group_term = v_proj.sum(dim=2).square().sum(dim=-1) - v_proj.square().sum(dim=(2, 3))
        alpha = F.softmax(self.U_z(z_star), dim=-1)
        e_group = -0.5 * (alpha * group_term).sum(dim=-1)
        e_z = -0.5 * (z_star * F.linear(psi_sum, W_zy_b)).sum(dim=-1)
        return e_unary + e_pair + e_group + e_z

    def encode_lm_sparsemax(self, lm_cat: torch.Tensor, jitter: float = 0.0) -> torch.Tensor:
        """Считает Sparsemax-разреженный поправочный потенциал слотов поверх скрытых векторов `Calcite-5`."""
        x = lm_cat + jitter * torch.randn_like(lm_cat) if (self.training and jitter > 0.0) else lm_cat
        g_sp = Sparsemax.apply(self.sparse_gate(self.lm_norm(x)))
        return self.sparse_role(g_sp)

    def score_batch(
        self,
        all_uf: torch.Tensor,
        all_pf: torch.Tensor,
        all_rf: torch.Tensor,
        all_lm: torch.Tensor,
        slot_gidx: torch.Tensor,
        slot_roles: torch.Tensor,
        slot_cand_idx: torch.Tensor,
        slot_inv_k: torch.Tensor,
        root_prob_idx: torch.Tensor,
        root_op: torch.Tensor,
        bm_cand: torch.Tensor,
        bp_pair: torch.Tensor,
        bo_op: torch.Tensor,
        bra: torch.Tensor,
        brb: torch.Tensor,
        bwt: torch.Tensor,
        psi_raw_pad: torch.Tensor,
        psi_mask: torch.Tensor,
        psi_lens: torch.Tensor,
        meta_m: torch.Tensor,
        n_cands: int,
        jitter: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Считает `(e_cram, e_unary)` сразу для плоского батча из `n_cands` кандидатных графов."""
        dev = all_uf.device
        u_lex = 2.5 * F.linear(all_uf, self.W_u)
        u_lm = 1.0 * F.linear(all_lm, self.W_lm)
        u_sp = self.encode_lm_sparsemax(all_lm, jitter=jitter)

        u_tot_entries = (u_lex[slot_gidx, slot_roles] + u_lm[slot_gidx, slot_roles] + u_sp[slot_gidx, slot_roles]) * slot_inv_k
        u_un_entries = u_lex[slot_gidx, slot_roles] * slot_inv_k

        eu_full = torch.zeros(n_cands, dtype=torch.float32, device=dev).index_add_(0, slot_cand_idx, u_tot_entries)
        eu_ablate = torch.zeros(n_cands, dtype=torch.float32, device=dev).index_add_(0, slot_cand_idx, u_un_entries)

        r_pot = 1.5 * F.linear(all_rf, self.W_r)
        er = r_pot[root_prob_idx, root_op]

        if bm_cand.numel() > 0:
            p_pot = 2.5 * F.linear(all_pf, self.W_p)
            ee = (p_pot[bp_pair, bo_op] + self.K_cram[bra, brb, bo_op]) * bwt
            ep = torch.zeros(n_cands, dtype=torch.float32, device=dev).index_add_(0, bm_cand, ee)
        else:
            ep = torch.zeros(n_cands, dtype=torch.float32, device=dev)

        couplings = self._bounded_couplings()
        psi_seq = torch.tanh(self.step_proj(psi_raw_pad))
        ez = self.step_field_scale * self._score_psi_batched(psi_seq, psi_mask, psi_lens, couplings)
        em = meta_m @ self.w_meta
        e_cram = eu_full + ep + er + ez + em
        e_unary = eu_ablate + er
        return e_cram, e_unary

    def prox_regularization_loss(self) -> torch.Tensor:
        """Проксимальная L2-регуляризация к аналитическим якорям Вудбери + норма остаточных матриц."""
        prox = (
            3e-4 * (self.W_u - self.Wu0).square().sum()
            + 3e-4 * (self.W_p - self.Wp0).square().sum()
            + 2.5e-4 * (self.W_r - self.Wr0).square().sum()
            + 7.5e-4 * (self.W_lm - self.Wlm0).square().sum()
            + 5e-5 * (self.w_meta - self.wm0).square().sum()
        )
        res_reg = (
            1.5e-4 * self.K_cram.square().sum()
            + 2e-3 * self.sparse_role.weight.square().sum()
            + 5e-4 * (self.sparse_gate.weight.square().sum() + self.lm_norm.weight.square().sum())
            + 3e-4 * (
                self.step_proj.weight.square().sum()
                + self.step_unary_head.weight.square().sum()
                + self.W_zy.square().sum()
                + self.W_edge.square().sum()
                + self.V_groups.square().sum()
                + self.U_z.weight.square().sum()
            )
        )
        return prox + res_reg


class OpenGraphCRAMSolver:
    """Открытый нейро-символический решатель GSM8K (`CRAM+z`, `0 BPTT`, `SphericalALU`, `DualCert`)."""

    def __init__(
        self,
        extractor: TrainInducedQuantityExtractor,
        alu: SphericalBase10RegisterALU,
        d_model: int = 160,
        device: str = "cpu",
        seed: int = 2026,
    ):
        self.extractor = extractor
        self.alu = alu
        self.d_model = int(d_model)
        self.device = device
        self.seed = int(seed)

        self.ebm = CRAMzOperatorGraphEBM(d_model=self.d_model).to(self.device)

        self.vocab_counts: Counter = Counter()
        self.stop_words: Set[str] = set()
        self.keep_vocab: Set[str] = set()
        self.ep_vecs: Dict[Tuple[int, int], np.ndarray] = {}
        self.ep_items: Dict[Tuple[int, int], List[Tuple[Tuple[Tuple[int, int, int], ...], np.ndarray]]] = {}
        self.sk_cnt: Dict[Tuple[int, int], Counter] = {}
        self.sk_tot: Dict[Tuple[int, int], float] = {}

        # 6 признаков пост-калибровки ранжирования + скаляр температуры Больцмана
        self.base_rerank_weights = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.rerank_weights = self.base_rerank_weights.copy()
        self.boltzmann_beta: float = 1.0

        # Двойной анти-галлюцинационный сертификат (u_prob <= theta_epi) AND (E_cert <= tau_energy)
        self.cert_weights = np.zeros(N_CERT_FEATS, dtype=np.float64)
        self.cert_bias: float = 0.0
        self.theta_epi: float = 0.85
        self.tau_energy: float = -0.35

        # Опциональный внешний переранжировщик System-1 (teacher-forced скоринг топ-кандидатов)
        self._sys1_reranker: Optional[object] = None
        self._sys1_use_lut: bool = False

    def attach_system1_reranker(
        self,
        sys1_encoder: Optional[object],
        use_lut: bool = False,
    ) -> None:
        """Прикрепляет `Calcite5MathSystem1` для оценки топ-кандидатов в одиночном `solve_blind`."""
        self._sys1_reranker = sys1_encoder
        self._sys1_use_lut = bool(use_lut)

    def state_dict(self) -> Dict[str, object]:
        """Возвращает полный сериализуемый словарь состояния решателя, памяти прецедентов и порогов."""
        return {
            "ebm_state_dict": {k: v.detach().cpu() for k, v in self.ebm.state_dict().items()},
            "vocab_counts": dict(self.vocab_counts),
            "stop_words": sorted(self.stop_words),
            "keep_vocab": sorted(self.keep_vocab),
            "ep_vecs": {f"{a}_{b}": v for (a, b), v in self.ep_vecs.items()},
            "ep_items": {
                f"{a}_{b}": [(sk, rl.tolist()) for sk, rl in items]
                for (a, b), items in self.ep_items.items()
            },
            "sk_cnt": {f"{a}_{b}": dict(cnt) for (a, b), cnt in self.sk_cnt.items()},
            "sk_tot": {f"{a}_{b}": float(v) for (a, b), v in self.sk_tot.items()},
            "base_rerank_weights": self.base_rerank_weights.tolist(),
            "rerank_weights": self.rerank_weights.tolist(),
            "boltzmann_beta": float(self.boltzmann_beta),
            "cert_weights": self.cert_weights.tolist(),
            "cert_bias": float(self.cert_bias),
            "theta_epi": float(self.theta_epi),
            "tau_energy": float(self.tau_energy),
        }

    def load_state_dict(self, sd: Dict[str, object]) -> None:
        """Восстанавливает обученный `CRAMzOperatorGraphEBM`, эпизодическую память и пороги DualCert."""
        self.ebm.load_state_dict(sd["ebm_state_dict"], strict=True)  # type: ignore[arg-type]
        self.ebm.to(self.device).eval()

        if "vocab_counts" in sd:
            self.vocab_counts = Counter({str(k): int(v) for k, v in dict(sd["vocab_counts"]).items()})  # type: ignore[arg-type]
        if "stop_words" in sd:
            self.stop_words = {str(x) for x in list(sd["stop_words"])}  # type: ignore[arg-type]
        if "keep_vocab" in sd:
            self.keep_vocab = {str(x) for x in list(sd["keep_vocab"])}  # type: ignore[arg-type]

        def _parse_kb(key_s: str) -> Tuple[int, int]:
            a_s, b_s = key_s.split("_")
            return (int(a_s), int(b_s))

        if "ep_vecs" in sd:
            self.ep_vecs = {_parse_kb(k): np.asarray(v, dtype=np.float32) for k, v in dict(sd["ep_vecs"]).items()}  # type: ignore[arg-type]
        if "ep_items" in sd:
            self.ep_items = {
                _parse_kb(k): [
                    (tuple(tuple(int(x) for x in st) for st in sk), np.asarray(rl, dtype=np.int64))
                    for sk, rl in items
                ]
                for k, items in dict(sd["ep_items"]).items()  # type: ignore[arg-type]
            }
        if "sk_cnt" in sd:
            self.sk_cnt = {
                _parse_kb(k): Counter({tuple(tuple(int(x) for x in st) for st in sk): int(c) for sk, c in cnt.items()})
                for k, cnt in dict(sd["sk_cnt"]).items()  # type: ignore[arg-type]
            }
        if "sk_tot" in sd:
            self.sk_tot = {_parse_kb(k): float(v) for k, v in dict(sd["sk_tot"]).items()}  # type: ignore[arg-type]

        if "base_rerank_weights" in sd:
            self.base_rerank_weights = np.asarray(sd["base_rerank_weights"], dtype=np.float64).copy()
        self.rerank_weights = np.asarray(sd.get("rerank_weights", self.base_rerank_weights), dtype=np.float64).copy()
        self.boltzmann_beta = float(sd.get("boltzmann_beta", self.boltzmann_beta))  # type: ignore[arg-type]
        if "cert_weights" in sd:
            cw = np.asarray(sd["cert_weights"], dtype=np.float64).copy()
            if cw.shape == self.cert_weights.shape:
                self.cert_weights = cw
        self.cert_bias = float(sd.get("cert_bias", self.cert_bias))  # type: ignore[arg-type]
        self.theta_epi = float(sd.get("theta_epi", self.theta_epi))  # type: ignore[arg-type]
        self.tau_energy = float(sd.get("tau_energy", self.tau_energy))  # type: ignore[arg-type]

    def _fit_corpus_stats(self, train_items: Sequence[Dict[str, str]]) -> None:
        df_cnt: Counter = Counter()
        voc: Counter = Counter()
        for it in train_items:
            ws = [tk.lower for tk in scan_question_tokens(str(it["question"])) if tk.kind == "WORD"]
            voc.update(ws)
            df_cnt.update(set(ws))
        n_docs = max(1, len(train_items))
        self.vocab_counts = voc
        self.stop_words = {w for w, c in df_cnt.items() if c > 0.085 * n_docs or len(w) <= 2}
        self.keep_vocab = {w for w, c in df_cnt.items() if c >= 5}

    def _norm_tok(self, word: str) -> str:
        sw = _stem_by_corpus(word.lower(), self.vocab_counts)
        return sw if sw in self.keep_vocab else "<UNK>"

    def _featurize_problem(
        self,
        q_text: str,
        slots: Sequence[ExtractedSlot],
        lm_feat: Optional[Dict[str, object]],
    ) -> Dict[str, object]:
        k = len(slots)
        q_ws, q_wants_money = _question_focus_and_money(q_text, slots, self.extractor)
        nf, _, _ = _compute_numeric_slot_features(slots, q_ws, self.extractor)
        q_root_aff = self.extractor.score_question_root_affinity(q_text)
        q_mode = int(np.argmax(q_root_aff))
        saff = self.extractor.score_slot_affinities(slots) if k > 0 else np.zeros((0, 7), dtype=np.float32)
        paff = self.extractor.score_pair_mid_affinities(q_text, slots) if k > 1 else {}

        q_stem_set = {_stem_by_corpus(w, self.vocab_counts) for w in q_ws} - self.stop_words
        slot_nouns = [
            next(
                (
                    _stem_by_corpus(w, self.vocab_counts)
                    for w in s.right_words[:3]
                    if _stem_by_corpus(w, self.vocab_counts) not in self.stop_words
                    and _stem_by_corpus(w, self.vocab_counts) in self.keep_vocab
                ),
                "",
            )
            for s in slots
        ]
        q_unit = next(
            (
                _stem_by_corpus(w, self.vocab_counts)
                for w in q_ws
                if _stem_by_corpus(w, self.vocab_counts) in {u for u in slot_nouns if u}
            ),
            "",
        )

        if lm_feat is None:
            hn = np.zeros((k, self.d_model), dtype=np.float32)
            hp = np.zeros((k, self.d_model), dtype=np.float32)
            hq = np.zeros((self.d_model,), dtype=np.float32)
            u_prob = 1.0
            beams: List[Tuple[float, Tuple[Tuple[int, int, int], ...]]] = []
        else:
            hn_raw = np.asarray(lm_feat["h_num"], dtype=np.float32)  # type: ignore[arg-type]
            hp_raw = np.asarray(lm_feat["h_post"], dtype=np.float32)  # type: ignore[arg-type]
            hq = np.asarray(lm_feat["h_q"], dtype=np.float32)  # type: ignore[arg-type]
            if len(hn_raw) != k:
                hn = np.zeros((k, self.d_model), dtype=np.float32)
                hp = np.zeros((k, self.d_model), dtype=np.float32)
                mc = min(k, len(hn_raw))
                if mc > 0:
                    hn[:mc] = hn_raw[:mc]
                    hp[:mc] = hp_raw[:mc]
            else:
                hn, hp = hn_raw, hp_raw
            u_prob = float(lm_feat.get("u_prob", 1.0))  # type: ignore[arg-type]
            beams = [
                (float(x[0]), x[1]) if isinstance(x[0], (int, float)) else (float(x[1]), x[0])
                for x in lm_feat.get("beam_plans", [])  # type: ignore[arg-type]
            ]

        hq_u = hq / (float(np.linalg.norm(hq)) + 1e-6)
        hn_u = hn / (np.linalg.norm(hn, axis=-1, keepdims=True) + 1e-6) if k > 0 else hn
        hp_u = hp / (np.linalg.norm(hp, axis=-1, keepdims=True) + 1e-6) if k > 0 else hp

        all_toks = scan_question_tokens(q_text)
        cl_words: Dict[int, List[str]] = {}
        for tk in all_toks:
            if tk.kind == "WORD":
                cl_words.setdefault(tk.clause_id, []).append(self._norm_tok(tk.lower))
        q_norm = [self._norm_tok(w) for w in q_ws][-15:]

        Uf = np.zeros((k, DU_HASH), dtype=np.float32)
        hbu = DU_HASH - 36
        vals = [float(s.val) for s in slots]
        ctx_h = np.zeros(DC_HASH, dtype=np.float32)
        hbc = DC_HASH - 48
        gr_pos = 0

        for i, s in enumerate(slots):
            v = Uf[i]
            lw = [self._norm_tok(w) for w in s.left_words]
            rw = [self._norm_tok(w) for w in s.right_words]
            iq = int(s.is_in_question)
            v[_det_hash(f"KD_{s.kind}", hbu)] += 2.2
            q_ov = float(any(_stem_by_corpus(w, self.vocab_counts) in q_stem_set for w in s.right_words[:3]))
            if s.trigger_token:
                v[_det_hash(f"TR_{s.trigger_token}", hbu)] += 2.8
                v[_det_hash(f"TRQ_{q_mode}_{s.trigger_token}", hbu)] += 1.8
            for dist, w in enumerate(reversed(lw)):
                v[_det_hash(f"UL_{iq}_{w}", hbu)] += 1.8 if dist < 2 else 0.9
                v[_det_hash(f"ULD_{dist}_{w}", hbu)] += 1.4
                v[_det_hash(f"ULM_{q_mode}_{w}", hbu)] += 1.3
                if s.kind != "CONST" and gr_pos < 8:
                    ctx_h[_det_hash(f"CL_{gr_pos}_{w}", hbc)] += 1.7 if dist < 2 else 0.8
                    ctx_h[_det_hash(f"CLD_{gr_pos}_{dist}_{w}", hbc)] += 1.2
            for dist, w in enumerate(rw):
                v[_det_hash(f"UR_{iq}_{w}", hbu)] += 1.8 if dist < 2 else 0.9
                v[_det_hash(f"URD_{dist}_{w}", hbu)] += 1.4
                v[_det_hash(f"URM_{q_mode}_{w}", hbu)] += 1.3
                if s.kind != "CONST" and gr_pos < 8:
                    ctx_h[_det_hash(f"CR_{gr_pos}_{w}", hbc)] += 1.7 if dist < 2 else 0.8
                    ctx_h[_det_hash(f"CRD_{gr_pos}_{dist}_{w}", hbc)] += 1.2
            if s.kind != "CONST":
                gr_pos += 1
            if len(lw) >= 2:
                v[_det_hash(f"ULB_{lw[-2]}_{lw[-1]}", hbu)] += 1.5
            if len(rw) >= 2:
                v[_det_hash(f"URB_{rw[0]}_{rw[1]}", hbu)] += 1.5
            if lw and rw:
                v[_det_hash(f"UB_{lw[-1]}_{rw[0]}", hbu)] += 1.6
            for cw in cl_words.get(s.clause_id, ())[:10]:
                v[_det_hash(f"UC_{cw}", hbu)] += 0.55
            for qw in q_norm[-8:]:
                v[_det_hash(f"UQ_{qw}", hbu)] += 0.65
                if q_ov > 0.0:
                    v[_det_hash(f"QOV_{q_wants_money}_{int(s.has_dollar)}_{qw}", hbu)] += 1.8
            unit_match = (slot_nouns[i] != "" and slot_nouns[i] == q_unit) or (s.has_dollar and q_wants_money)
            v[hbu : hbu + 27] = [
                1.0,
                float(unit_match) * 2.5,
                q_ov * 2.4,
                float(q_ov > 0.0 and q_wants_money and not s.has_dollar) * 2.8,
                *(nf[i] * 1.9),
                *(saff[i] * 2.8),
            ]
            v /= max(float(np.linalg.norm(v)), 1e-6)

        for qw in q_norm:
            ctx_h[_det_hash(f"CQ_{qw}", hbc)] += 2.0
        for w1, w2 in zip(q_norm[:-1], q_norm[1:]):
            ctx_h[_det_hash(f"CQB_{w1}_{w2}", hbc)] += 2.6
        ng = sum(1 for s in slots if s.kind != "CONST") or max(1, k)
        ctx_h[hbc : hbc + 14] = [
            ng / 8.0,
            float(q_wants_money) * 2.2,
            float(any(s.is_pct for s in slots)) * 2.4,
            *(q_root_aff * 2.6),
            *(saff.mean(axis=0) * 2.2 if k > 0 else np.zeros(7, dtype=np.float32)),
        ]
        ctx_h /= max(float(np.linalg.norm(ctx_h)), 1e-6)
        ctx = np.concatenate([ctx_h * 0.86, hq_u * 0.51]).astype(np.float32)
        ctx /= max(float(np.linalg.norm(ctx)), 1e-6)

        npair = max(1, k * (k - 1) // 2)
        Pf = np.zeros((npair, DP_HASH), dtype=np.float32)
        hbp = DP_HASH - 32
        pmap: Dict[Tuple[int, int], int] = {}
        pc = 0
        csets = [
            {_stem_by_corpus(w, self.vocab_counts) for w in (*s.left_words, *s.right_words) if _stem_by_corpus(w, self.vocab_counts) not in self.stop_words and len(w) > 2}
            for s in slots
        ]
        for i in range(k):
            for j in range(i + 1, k):
                pmap[(i, j)] = pc
                vp = Pf[pc]
                pc += 1
                si, sj = slots[i], slots[j]
                vi, vj = vals[i], vals[j]
                if si.trigger_token or sj.trigger_token:
                    vp[_det_hash(f"PT_{si.trigger_token}_{sj.trigger_token}", hbp)] += 2.4
                mids = [
                    self._norm_tok(tk.lower)
                    for tk in all_toks
                    if tk.kind == "WORD" and si.b1 <= tk.b0 and tk.b1 <= sj.b0
                ][:10]
                for w in mids:
                    vp[_det_hash(f"M_{w}", hbp)] += 1.8
                for w1, w2 in zip(mids[:-1], mids[1:]):
                    vp[_det_hash(f"MB_{w1}_{w2}", hbp)] += 2.1
                lwi = [self._norm_tok(x) for x in si.left_words[-3:]]
                rwi = [self._norm_tok(x) for x in si.right_words[:3]]
                lwj = [self._norm_tok(x) for x in sj.left_words[-3:]]
                rwj = [self._norm_tok(x) for x in sj.right_words[:3]]
                for w in lwi:
                    vp[_det_hash(f"LI_{w}", hbp)] += 1.5
                    vp[_det_hash(f"QLI_{q_mode}_{w}", hbp)] += 1.3
                for w in rwi:
                    vp[_det_hash(f"RI_{w}", hbp)] += 1.4
                for w in lwj:
                    vp[_det_hash(f"LJ_{w}", hbp)] += 1.8
                    vp[_det_hash(f"QLJ_{q_mode}_{w}", hbp)] += 1.5
                for w in rwj:
                    vp[_det_hash(f"RJ_{w}", hbp)] += 1.7
                if lwi and lwj:
                    vp[_det_hash(f"VP_{lwi[-1]}_{lwj[-1]}", hbp)] += 2.0
                if si.clause_id == sj.clause_id:
                    for w in lwj + rwj[:2]:
                        vp[_det_hash(f"SC_{w}", hbp)] += 2.0
                pm = paff.get((i, j), np.zeros(6, dtype=np.float32))
                lov = min(3.0, float(len(csets[i] & csets[j])))
                vp[hbp : hbp + 26] = [
                    1.0,
                    float(j == i + 1) * 2.4,
                    float(si.clause_id == sj.clause_id) * 2.8,
                    float(si.sent_id == sj.sent_id) * 1.5,
                    float(slot_nouns[i] != "" and slot_nouns[i] == slot_nouns[j]) * 3.2,
                    float(si.has_dollar and sj.has_dollar) * 2.6,
                    float(si.has_dollar != sj.has_dollar) * 2.4,
                    float(vi > vj) * 2.0,
                    float(vj > 1 and vi > 0 and abs(vi % vj) < 1e-4) * 2.6,
                    float(vi > 1 and vj > 0 and abs(vj % vi) < 1e-4) * 2.6,
                    float(si.is_pct) * 2.8,
                    float(sj.is_pct) * 2.8,
                    float(si.kind == "CONST") * 2.5,
                    float(sj.kind == "CONST") * 2.5,
                    lov * 1.8,
                    float(np.dot(hn_u[i], hn_u[j])) * 2.0,
                    float(np.dot(hp_u[i], hp_u[j])) * 2.0,
                    float(np.dot(hp_u[i], hq_u)) * 1.8,
                    float(np.dot(hp_u[j], hq_u)) * 1.8,
                    (j - i) / max(1.0, k - 1.0) * 1.5,
                    *(pm * 2.8),
                ]
                vp /= max(float(np.linalg.norm(vp)), 1e-6)

        Rf = np.zeros(DR_HASH, dtype=np.float32)
        hbr = DR_HASH - 16
        for w in q_norm:
            Rf[_det_hash(f"RQ_{w}", hbr)] += 1.9
        for w1, w2 in zip(q_norm[:-1], q_norm[1:]):
            Rf[_det_hash(f"RQB_{w1}_{w2}", hbr)] += 2.5
        Rf[hbr : hbr + 9] = [
            1.0,
            float(q_wants_money) * 2.5,
            float(any(s.is_pct for s in slots)) * 2.5,
            ng / 8.0,
            float(k > ng) * 1.8,
            *(q_root_aff * 2.8),
        ]
        Rf /= max(float(np.linalg.norm(Rf)), 1e-6)

        if k > 0:
            lm_cat = np.concatenate(
                [hn_u, hp_u, np.repeat(hq_u[None, :], k, axis=0)],
                axis=-1,
            ).astype(np.float32)
        else:
            lm_cat = np.zeros((0, 3 * self.d_model), dtype=np.float32)

        return {
            "Uf": Uf,
            "Pf": Pf,
            "pm": pmap,
            "Rf": Rf,
            "ctx": ctx,
            "vs": vals,
            "lm": lm_cat,
            "beams": beams,
            "up": u_prob,
            "nns": slot_nouns,
            "qm": q_wants_money,
            "qu": q_unit,
            "qra": q_root_aff,
            "ng": ng,
        }

    def _analyze_dag(
        self,
        dag: Sequence[Tuple[int, int, int]],
        slots: Sequence[ExtractedSlot],
        pf: Dict[str, object],
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int, int, float]], int, np.ndarray, np.ndarray]:
        """Нисходящий анализ графа: `9` ролей слотов, `7` направленных связей, стек `psi (T, 16)` и `12` штрафов."""
        k = len(slots)
        t_len = len(dag)
        ng = int(pf["ng"])  # type: ignore[arg-type]
        rl = np.zeros(k, dtype=np.int64)
        pol = [0] * (k + t_len)
        mdir = [0] * (k + t_len)
        hadd = [False] * (k + t_len)
        heads = [-1] * (k + t_len)
        leaves: List[List[int]] = [[] for _ in range(k + t_len)]
        for i in range(k):
            heads[i] = i
            leaves[i] = [i]

        vs = list(pf["vs"])  # type: ignore[arg-type]
        mn = [1 if s.has_dollar else 0 for s in slots]
        pc = [bool(s.is_pct) for s in slots]
        un = list(pf["nns"])  # type: ignore[arg-type]
        cl_ids = [int(s.clause_id) for s in slots]
        qra: np.ndarray = pf["qra"]  # type: ignore[assignment]
        pmap: Dict[Tuple[int, int], int] = pf["pm"]  # type: ignore[assignment]
        q_mon = bool(pf["qm"])
        q_unit = str(pf["qu"])

        acl = mcl = dok = ndv = ugly = 0
        psi_seq = np.zeros((t_len, D_STEP_PSI), dtype=np.float32)

        for t, (op, a, b) in enumerate(dag):
            nd = k + t
            hadd[nd] = (op in (0, 1)) or hadd[a] or hadd[b]
            heads[nd] = heads[a] if heads[a] >= 0 else heads[b]
            leaves[nd] = leaves[a] + [x for x in leaves[b] if x not in leaves[a]]
            cl_ids.append(max(cl_ids[a], cl_ids[b]))

            va, vb = vs[a], vs[b]
            ma, mb = mn[a], mn[b]
            pa, pb = pc[a], pc[b]
            ua, ub = un[a], un[b]
            step_clash = 0
            if op in (0, 1):
                ov = va + vb if op == 0 else va - vb
                if pa != pb or (ma != mb and a < k and b < k and slots[a].kind != "CONST" and slots[b].kind != "CONST"):
                    acl += 1
                    step_clash = 1
                vs.append(ov)
                mn.append(1 if (ma or mb) else 0)
                pc.append(pa and pb)
                un.append(ua or ub)
            elif op == 2:
                ov = va * vb
                if (ma and mb) or (pa and pb) or (ua != "" and ua == ub and not pa and not pb):
                    mcl += 1
                    step_clash = 1
                vs.append(ov)
                mn.append(1 if ma != mb else 0)
                pc.append(False)
                un.append(ua or ub)
            else:
                ndv += 1
                ov = va / vb if abs(vb) > 1e-9 else 1e9
                if abs(vb) > 1e-6 and abs(ov * 4.0 - round(ov * 4.0)) < 1e-3 and ov > 0.0:
                    dok += 1
                vs.append(ov)
                mn.append(1 if (ma and not mb) else 0)
                pc.append(False)
                un.append("" if (ma and mb) else ua)

            step_ugly = int(t < t_len - 1 and abs(ov * 100.0 - round(ov * 100.0)) > 1e-2 and abs(ov * 12.0 - round(ov * 12.0)) > 1e-2)
            ugly += step_ugly
            is_root = float(t == t_len - 1)
            unit_ok_step = float(
                is_root > 0.5 and ((q_mon and mn[-1] == 1) or (not q_mon and q_unit != "" and un[-1] == q_unit))
            )
            psi_seq[t] = [
                float(op == 0),
                float(op == 1),
                float(op == 2),
                float(op == 3),
                is_root,
                (t + 1) / float(max(1, t_len)),
                float(a < k and b < k),
                float(a >= k and b < k),
                float(a < k and b >= k),
                float(a >= k and b >= k),
                float(t > 0 and dag[t - 1][0] == op),
                (1.0 if is_root > 0.5 else 0.3) * float(qra[op]),
                float(cl_ids[a] == cl_ids[b]),
                float(step_clash),
                float(step_ugly),
                unit_ok_step,
            ]

        pol[k + t_len - 1] = 1
        mdir[k + t_len - 1] = 1
        cnt = [0] * k
        br: Dict[Tuple[int, int, int], float] = {}
        rv_op = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4, 5: 6, 6: 5}
        for t in range(t_len - 1, -1, -1):
            p, m = pol[k + t], mdir[k + t]
            op, a, b = dag[t]
            if a < k:
                cnt[a] += 1
            if b < k:
                cnt[b] += 1
            if op == 0:
                pol[a] = pol[b] = p
                mdir[a] = mdir[b] = m
                oc = 0
            elif op == 1:
                pol[a], pol[b] = p, -p
                mdir[a] = mdir[b] = m
                oc = 1 if p >= 0 else 2
                if a < k and p > 0 and rl[a] == 0:
                    rl[a] = 3
            elif op == 2:
                pol[a] = pol[b] = p
                mdir[a] = mdir[b] = m
                oc = 4 if (hadd[a] or hadd[b]) else 3
                if m > 0:
                    if a < k and not hadd[a]:
                        rl[a] = 7 if hadd[b] else 4
                    if b < k and not hadd[b]:
                        rl[b] = 7 if hadd[a] else 4
            else:
                pol[a] = pol[b] = p
                mdir[a], mdir[b] = m, -m
                oc = 5 if m >= 0 else 6
                if a < k and m > 0 and not hadd[a] and rl[a] == 0:
                    rl[a] = 6

            prs = [(u, v) for u in leaves[a] for v in leaves[b] if u != v]
            if prs:
                xh, yh = heads[a], heads[b]
                if xh >= 0 and yh >= 0 and xh != yh:
                    ky = (xh, yh, oc) if xh < yh else (yh, xh, rv_op[oc])
                    br[ky] = br.get(ky, 0.0) + 0.55
                we = 0.45 / float(len(prs))
                for u, v in prs:
                    ky = (u, v, oc) if u < v else (v, u, rv_op[oc])
                    br[ky] = br.get(ky, 0.0) + we

        for i in range(k):
            if cnt[i] >= 2:
                rl[i] = 8
            elif cnt[i] == 1 and rl[i] == 0:
                rl[i] = 5 if mdir[i] < 0 else (2 if pol[i] < 0 else 1)

        dn = float(max(max(1, ng - 1), t_len))
        bonds = [(u, v, pmap[(u, v)], oc, w / dn) for (u, v, oc), w in br.items()]
        ug = sum(1 for i in range(k) if slots[i].kind != "CONST" and rl[i] > 0)
        uc = sum(1 for i in range(k) if slots[i].kind == "CONST" and rl[i] > 0)
        ovf = float(vs[-1])
        mv = max([float(s.val) for s in slots if s.kind != "CONST"] + [1.0])
        uok = 1.0 if ((q_mon and mn[-1] == 1) or (not q_mon and q_unit != "" and un[-1] == q_unit)) else 0.0
        mt = np.array(
            [
                (ng - ug) / max(1.0, float(ng)),
                float(uc) * 0.5,
                max(0.0, t_len - max(1, ng)) / 3.0,
                float(ug <= 1),
                float(abs(ovf - round(ovf)) > 1e-4),
                float(ovf <= 1e-6),
                float(qra[1] > 0.30 and mv >= 10.0 and ovf > mv * 1.05),
                float(acl),
                float(mcl),
                dok / float(max(1, ndv)) if ndv else 0.5,
                uok,
                float(ugly),
            ],
            dtype=np.float32,
        )
        return rl, bonds, int(dag[-1][0]), psi_seq, mt

    def _synthesize_open_pool(
        self,
        slots: Sequence[ExtractedSlot],
        pf: Dict[str, object],
        mask_self: bool = False,
    ) -> List[List[object]]:
        """Открытая сборка кандидатов DAG: 1/2-шаговые графы, Workspace Forest, System-1 лучи и прецеденты."""
        k = len(slots)
        if k <= 1:
            return []
        vals: List[float] = pf["vs"]  # type: ignore[assignment]
        Uf_t = torch.from_numpy(pf["Uf"]).to(self.device)  # type: ignore[arg-type]
        Pf_t = torch.from_numpy(pf["Pf"]).to(self.device)  # type: ignore[arg-type]
        lm_t = torch.from_numpy(pf["lm"]).to(self.device)  # type: ignore[arg-type]
        with torch.no_grad():
            u_pot = (
                2.5 * F.linear(Uf_t, self.ebm.Wu0)
                + 1.0 * F.linear(lm_t, self.ebm.Wlm0)
            ).cpu().numpy()
            p_pot = (2.5 * F.linear(Pf_t, self.ebm.Wp0)).cpu().numpy()

        gr = [i for i, s in enumerate(slots) if s.kind != "CONST"] or list(range(k))
        cn = sorted([i for i, s in enumerate(slots) if s.kind == "CONST"], key=lambda i: u_pot[i, 1:].min())[:2]
        ng = len(gr)
        gr_set = set(gr)
        min_ug = max(1, ng - 1 if ng <= 5 else ng - 2)
        eff_min_ug = max(2, min(5, ng - 3)) if ng >= 8 else min_ug
        pmap: Dict[Tuple[int, int], int] = pf["pm"]  # type: ignore[assignment]
        by_sg: Dict[float, List[object]] = {}

        def _add_dag(d: Sequence[Tuple[int, int, int]], kw: float = 0.0, is_bm: float = 0.0, s1_lp: float = -12.0, in_s1: float = 0.0) -> None:
            cd = _prune_to_connected_dag(d, k, max_steps=MAX_GRAPH_STEPS)
            if not cd:
                return
            ul = {x for _, a, b in cd for x in (a, b) if x < k}
            ug = len(ul & gr_set)
            uc = len(ul) - ug
            if ug < eff_min_ug or uc > 1:
                return
            sg = _dag_rational_sig(cd, k)
            rv = _eval_dag_fast(vals, cd)
            if sg is None or rv is None or not (0.0 < rv < 1e8) or abs(rv * 100.0 - round(rv * 100.0)) > 1.5e-2:
                return
            cur = by_sg.get(sg)
            if cur is None:
                sk, u_g, u_c = _to_narrative_skeleton(cd, slots)
                kb = (len(u_g), len(u_c))
                cadj = max(0, self.sk_cnt.get(kb, Counter()).get(sk, 0) - (1 if mask_self else 0))
                pr_p = (cadj + 0.25) / self.sk_tot.get(kb, 50.0)
                by_sg[sg] = [cd, float(rv), pr_p, kw, is_bm, sg, in_s1, s1_lp]
            else:
                if len(cd) < len(cur[0]):  # type: ignore[arg-type]
                    cur[0] = cd
                cur[3] = float(cur[3]) + kw  # type: ignore[operator]
                cur[4] = max(float(cur[4]), is_bm)  # type: ignore[arg-type]
                cur[6] = max(float(cur[6]), in_s1)  # type: ignore[arg-type]
                cur[7] = max(float(cur[7]), s1_lp)  # type: ignore[arg-type]

        if not mask_self:
            for s1_lp, st_b in pf["beams"]:  # type: ignore[attr-defined]
                _add_dag(st_b, is_bm=0.0, s1_lp=float(s1_lp), in_s1=1.0)

        def _sub_bond_e(op: int, h1: int, l1: Tuple[int, ...], h2: int, l2: Tuple[int, ...], hd1: bool, hd2: bool) -> float:
            prs = [(u, v) for u in l1 for v in l2 if u != v]
            if not prs:
                return 0.4
            def _pe(u: int, v: int) -> float:
                lo, hi = (u, v) if u < v else (v, u)
                if op == 0:
                    oc = 0
                elif op == 1:
                    oc = 1 if u < v else 2
                elif op == 2:
                    oc = 4 if (hd1 or hd2) else 3
                else:
                    oc = 5 if u < v else 6
                return float(p_pot[pmap[(lo, hi)], oc])
            e_head = _pe(h1, h2) if h1 != h2 else 0.0
            return 0.55 * e_head + 0.45 * sum(_pe(u, v) for u, v in prs) / float(len(prs))

        if ng <= 3:
            act_all = gr + cn
            for i in act_all:
                for j in act_all:
                    if i > j:
                        continue
                    for o0 in range(4):
                        if i == j and o0 not in (0, 2):
                            continue
                        firsts = [(o0, i, j)] + ([(o0, j, i)] if o0 in (1, 3) and i != j else [])
                        for s0 in firsts:
                            _add_dag([s0], is_bm=1.0)
                            for c in act_all:
                                for o1 in range(4):
                                    sa = min(c, k) if o1 in (0, 2) else k
                                    sb = max(c, k) if o1 in (0, 2) else c
                                    _add_dag([s0, (o1, sa, sb)], is_bm=1.0)
                                    if o1 in (1, 3):
                                        _add_dag([s0, (o1, c, k)], is_bm=1.0)
        elif ng <= 7:
            seeds: List[Tuple[Tuple[Tuple[int, float, int, Tuple[int, ...], bool], ...], Tuple[Tuple[int, int, int], ...], float]] = [
                (tuple((i, vals[i], i, (i,), False) for i in gr), (), 0.0)
            ]
            if ng <= 6:
                di = min(gr, key=lambda i: u_pot[i, 0])
                seeds.append((tuple((i, vals[i], i, (i,), False) for i in gr if i != di), (), float(u_pot[di, 0]) + 0.4))
                for ci in cn[:1]:
                    seeds.append((tuple((i, vals[i], i, (i,), False) for i in (gr + [ci])), (), float(u_pot[ci, 4]) + 0.45))

            beam = seeds
            done_forest: List[Tuple[float, Tuple[Tuple[int, int, int], ...]]] = []
            for _ in range(6):
                nxt = []
                for ws, d, sc in beam:
                    mw = len(ws)
                    if mw <= 1:
                        continue
                    nid = k + len(d)
                    for ia in range(mw):
                        na, va, ha, la, hda = ws[ia]
                        for ib in range(ia + 1, mw):
                            nb, vb, hb, lb, hdb = ws[ib]
                            rws = tuple(ws[x] for x in range(mw) if x != ia and x != ib)
                            lv = la + tuple(x for x in lb if x not in la)
                            cand_ops = (
                                (0, (na, va, ha, la, hda, nb, vb, hb, lb, hdb)),
                                (1, (na, va, ha, la, hda, nb, vb, hb, lb, hdb)),
                                (1, (nb, vb, hb, lb, hdb, na, va, ha, la, hda)),
                                (2, (na, va, ha, la, hda, nb, vb, hb, lb, hdb)),
                                (3, (na, va, ha, la, hda, nb, vb, hb, lb, hdb)),
                                (3, (nb, vb, hb, lb, hdb, na, va, ha, la, hda)),
                            )
                            for op, (i1, v1, h1, l1, d1, i2, v2, h2, l2, d2) in cand_ops:
                                if (op == 1 and v1 < v2 - 1e-6) or (op == 2 and (abs(v1 - 1.0) < 1e-6 or abs(v2 - 1.0) < 1e-6)) or (op == 3 and (abs(v2) < 1e-6 or abs(v2 - 1.0) < 1e-6)):
                                    continue
                                rv = v1 + v2 if op == 0 else (v1 - v2 if op == 1 else (v1 * v2 if op == 2 else v1 / v2))
                                if not math.isfinite(rv) or abs(rv) > 1e8:
                                    continue
                                sa, sb = (min(i1, i2), max(i1, i2)) if op in (0, 2) else (i1, i2)
                                nd_d = d + ((op, sa, sb),)
                                be = _sub_bond_e(op, h1, l1, h2, l2, d1, d2)
                                nn_t = (nid, rv, h1, lv, (op in (0, 1)) or d1 or d2)
                                ws2 = rws + (nn_t,)
                                if len(ws2) == 1:
                                    done_forest.append((sc + be, nd_d))
                                else:
                                    nxt.append((ws2, nd_d, sc + be))
                                if op in (2, 3) and i1 < k and slots[i1].kind != "CONST" and len(rws) <= 2 and len(d) <= 3:
                                    nxt.append((rws + ((i1, v1, h1, l1, d1), nn_t), nd_d, sc + be + float(u_pot[i1, 8]) * 0.4 + 0.5))
                if not nxt:
                    break
                nxt.sort(key=lambda x: x[2])
                seen_s: Set[Tuple[float, ...]] = set()
                beam = []
                for itm in nxt:
                    sw = tuple(sorted(round(nd[1] * 1.137 + len(nd[3]) * 0.31, 3) for nd in itm[0]))
                    if sw not in seen_s:
                        seen_s.add(sw)
                        beam.append(itm)
                        if len(beam) >= 18:
                            break
            done_forest.sort(key=lambda x: x[0])
            for _, d_fin in done_forest[:24]:
                _add_dag(d_fin, is_bm=1.0)
        else:
            # Снятие обрыва на глубоких задачах (`ng >= 8`): ранжирование по информативности слотов + Iterative Register Folding
            sal_map = {i: float(u_pot[i, 0] - np.min(u_pot[i, 1:])) for i in gr}
            gr_by_sal = sorted(gr, key=lambda i: (sal_map[i], i), reverse=True)
            core_subsets: List[Tuple[List[int], float]] = []
            seen_cores: Set[Tuple[int, ...]] = set()
            for k_sub in (4, 5, 6, 7, 8, min(10, ng)):
                if k_sub <= ng:
                    sub_s = tuple(sorted(gr_by_sal[:k_sub]))
                    if sub_s not in seen_cores:
                        seen_cores.add(sub_s)
                        core_subsets.append((list(sub_s), 0.0))
                    sub_p = tuple(gr[:k_sub])
                    if k_sub <= 7 and sub_p not in seen_cores:
                        seen_cores.add(sub_p)
                        core_subsets.append((list(sub_p), 0.12))
                    sub_t = tuple(gr[-k_sub:])
                    if k_sub <= 7 and sub_t not in seen_cores:
                        seen_cores.add(sub_t)
                        core_subsets.append((list(sub_t), 0.15))

            seeds_hi: List[Tuple[Tuple[Tuple[int, float, int, Tuple[int, ...], bool], ...], Tuple[Tuple[int, int, int], ...], float]] = []
            for sub_gr, base_sc in core_subsets:
                if len(sub_gr) <= 8:
                    seeds_hi.append((tuple((i, vals[i], i, (i,), False) for i in sub_gr), (), base_sc))

            beam_hi = seeds_hi[:6]
            done_hi: List[Tuple[float, Tuple[Tuple[int, int, int], ...]]] = []
            max_rounds_hi = min(MAX_GRAPH_STEPS, max(7, min(len(gr_by_sal), 9)))
            for _ in range(max_rounds_hi):
                nxt_hi = []
                for ws, d, sc in beam_hi:
                    mw = len(ws)
                    if mw <= 1 or len(d) >= MAX_GRAPH_STEPS:
                        continue
                    nid = k + len(d)
                    for ia in range(mw):
                        na, va, ha, la, hda = ws[ia]
                        for ib in range(ia + 1, mw):
                            nb, vb, hb, lb, hdb = ws[ib]
                            rws = tuple(ws[x] for x in range(mw) if x != ia and x != ib)
                            lv = la + tuple(x for x in lb if x not in la)
                            cand_ops = (
                                (0, (na, va, ha, la, hda, nb, vb, hb, lb, hdb)),
                                (1, (na, va, ha, la, hda, nb, vb, hb, lb, hdb)),
                                (1, (nb, vb, hb, lb, hdb, na, va, ha, la, hda)),
                                (2, (na, va, ha, la, hda, nb, vb, hb, lb, hdb)),
                                (3, (na, va, ha, la, hda, nb, vb, hb, lb, hdb)),
                                (3, (nb, vb, hb, lb, hdb, na, va, ha, la, hda)),
                            )
                            for op, (i1, v1, h1, l1, d1, i2, v2, h2, l2, d2) in cand_ops:
                                if (op == 1 and v1 < v2 - 1e-6) or (op == 2 and (abs(v1 - 1.0) < 1e-6 or abs(v2 - 1.0) < 1e-6)) or (op == 3 and (abs(v2) < 1e-6 or abs(v2 - 1.0) < 1e-6)):
                                    continue
                                rv = v1 + v2 if op == 0 else (v1 - v2 if op == 1 else (v1 * v2 if op == 2 else v1 / v2))
                                if (not math.isfinite(rv)) or abs(rv) > 1e8:
                                    continue
                                sa, sb = (min(i1, i2), max(i1, i2)) if op in (0, 2) else (i1, i2)
                                nd_d = d + ((op, sa, sb),)
                                be = _sub_bond_e(op, h1, l1, h2, l2, d1, d2)
                                nn_t = (nid, rv, h1, lv, (op in (0, 1)) or d1 or d2)
                                ws2 = rws + (nn_t,)
                                if len(ws2) == 1:
                                    done_hi.append((sc + be, nd_d))
                                else:
                                    nxt_hi.append((ws2, nd_d, sc + be))
                if not nxt_hi:
                    break
                nxt_hi.sort(key=lambda x: x[2])
                seen_hi: Set[Tuple[float, ...]] = set()
                beam_hi = []
                for itm in nxt_hi:
                    sw = tuple(sorted(round(nd[1] * 1.137 + len(nd[3]) * 0.31, 3) for nd in itm[0]))
                    if sw not in seen_hi:
                        seen_hi.add(sw)
                        beam_hi.append(itm)
                        if len(beam_hi) >= 14:
                            break
            done_hi.sort(key=lambda x: x[0])
            for _, d_fin in done_hi[:24]:
                _add_dag(d_fin, is_bm=1.0)

            # Быстрая жадная регистровая свёртка (`Iterative Register Folding`) для гарантии 100% непустого пула до ng<=16
            for sub_gr, _ in core_subsets:
                work = [(i, vals[i], i, (i,), False) for i in sub_gr]
                fold_steps: List[Tuple[int, int, int]] = []
                while len(work) > 1 and len(fold_steps) < MAX_GRAPH_STEPS:
                    best_fold = None
                    nid = k + len(fold_steps)
                    for ia in range(len(work)):
                        na, va, ha, la, hda = work[ia]
                        for ib in range(ia + 1, len(work)):
                            nb, vb, hb, lb, hdb = work[ib]
                            for op in (0, 1, 2, 3):
                                for (i1, v1, h1, l1, d1, i2, v2, h2, l2, d2) in (
                                    (na, va, ha, la, hda, nb, vb, hb, lb, hdb),
                                    (nb, vb, hb, lb, hdb, na, va, ha, la, hda),
                                )[: (1 if op in (0, 2) else 2)]:
                                    if (op == 1 and v1 < v2 - 1e-6) or (op == 3 and abs(v2) < 1e-6):
                                        continue
                                    rv = v1 + v2 if op == 0 else (v1 - v2 if op == 1 else (v1 * v2 if op == 2 else v1 / v2))
                                    if (not math.isfinite(rv)) or rv <= 0.0 or rv > 1e8:
                                        continue
                                    be = _sub_bond_e(op, h1, l1, h2, l2, d1, d2)
                                    if best_fold is None or be < best_fold[0]:
                                        sa, sb = (min(i1, i2), max(i1, i2)) if op in (0, 2) else (i1, i2)
                                        lv = l1 + tuple(x for x in l2 if x not in l1)
                                        nn_t = (nid, rv, h1, lv, (op in (0, 1)) or d1 or d2)
                                        best_fold = (be, ia, ib, (op, sa, sb), nn_t)
                    if best_fold is None:
                        break
                    _, ia_b, ib_b, st_b, nn_b = best_fold
                    fold_steps.append(st_b)
                    work = [work[x] for x in range(len(work)) if x != ia_b and x != ib_b] + [nn_b]
                    if len(work) == 1:
                        _add_dag(fold_steps, is_bm=1.0)

        buckets: List[Tuple[Tuple[int, int], List[List[int]], int, float]] = [((ng, 0), [gr], 80, 1.0)]
        if 2 <= ng <= 7:
            buckets.append(((ng - 1, 0), [list(c) for c in itertools.combinations(gr, ng - 1)], 44, 0.45))
        if cn:
            buckets.append(((ng, 1), [gr + [ci] for ci in cn], 44, 0.65))
            if 2 <= ng <= 6:
                buckets.append(((ng - 1, 1), [list(c) + [ci] for c in itertools.combinations(gr, ng - 1) for ci in cn[:1]], 24, 0.30))
        if ng >= 8:
            sal_map_b = {i: float(u_pot[i, 0] - np.min(u_pot[i, 1:])) for i in gr}
            gr_by_sal_b = sorted(gr, key=lambda i: (sal_map_b[i], i), reverse=True)
            for k_sub in (8, 7, 6, 5, 4):
                if k_sub < ng:
                    mps = [sorted(gr_by_sal_b[:k_sub]), gr[-k_sub:]]
                    buckets.append(((k_sub, 0), mps, 36, 0.55))
                    if cn:
                        buckets.append(((k_sub, 1), [m + [cn[0]] for m in mps], 24, 0.35))

        q_ctx: np.ndarray = pf["ctx"]  # type: ignore[assignment]
        for kb, mp_list, top_n, b_scale in buckets:
            if kb not in self.ep_vecs or not mp_list:
                continue
            sims = self.ep_vecs[kb] @ q_ctx
            if mask_self:
                sims = np.where(sims > 0.9995, -1e9, sims)
            nn_k = min(top_n, len(sims))
            tnb = np.argpartition(-sims, nn_k - 1)[:nn_k]
            tnb = tnb[np.argsort(-sims[tnb])]
            ts = sims[tnb] * 16.0
            ts -= np.max(ts)
            kw_arr = np.exp(ts)
            kw_arr /= max(float(np.sum(kw_arr)), 1e-8)
            items_b = self.ep_items[kb]
            for nbi, w_nb in zip(tnb, kw_arr):
                sk, src_rl = items_b[int(nbi)]
                if len(mp_list) == 1:
                    best_mps = [mp_list[0]]
                else:
                    scored = sorted((sum(float(u_pot[idx, r]) for idx, r in zip(mp, src_rl)), mp) for mp in mp_list)
                    best_mps = [x[1] for x in scored[:2]]
                wt_each = float(w_nb) * b_scale / float(len(best_mps))
                for mp in best_mps:
                    lifted = []
                    for o, a, b in sk:
                        ra = mp[a] if a < 100 else k + (a - 100)
                        rb = mp[b] if b < 100 else k + (b - 100)
                        sa, sb = (rb, ra) if (o in (0, 2) and ra > rb) else (ra, rb)
                        lifted.append((o, sa, sb))
                    _add_dag(lifted, kw=wt_each)

        return list(by_sg.values())

    def _pack_candidates_cpu(
        self,
        slots: Sequence[ExtractedSlot],
        pf: Dict[str, object],
        cands: Sequence[List[object]],
    ) -> Dict[str, object]:
        k = len(slots)
        n_c = len(cands)
        roles_m = np.zeros((n_c, k), dtype=np.int64)
        rt_m = np.zeros(n_c, dtype=np.int64)
        psi_pad = np.zeros((n_c, MAX_GRAPH_STEPS, D_STEP_PSI), dtype=np.float32)
        psi_mask = np.zeros((n_c, MAX_GRAPH_STEPS), dtype=np.float32)
        psi_lens = np.ones(n_c, dtype=np.float32)
        meta_m = np.zeros((n_c, N_META_FEATS), dtype=np.float32)
        bm: List[int] = []
        bp: List[int] = []
        bo: List[int] = []
        bra: List[int] = []
        brb: List[int] = []
        bwt: List[float] = []

        for m_i, c in enumerate(cands):
            dag: Tuple[Tuple[int, int, int], ...] = c[0]  # type: ignore[assignment]
            rl, bds, rtop, psi_seq, mt = self._analyze_dag(dag, slots, pf)
            t_len = min(len(psi_seq), MAX_GRAPH_STEPS)
            roles_m[m_i] = rl
            rt_m[m_i] = rtop
            psi_pad[m_i, :t_len] = psi_seq[:t_len]
            psi_mask[m_i, :t_len] = 1.0
            psi_lens[m_i] = float(max(1, t_len))
            pr_lp = math.log(float(c[2]) + 1e-5)
            kn_lp = math.log(0.78 * float(c[3]) + 0.22 * float(c[2]) + 1e-4)
            meta_m[m_i] = [pr_lp * 0.25, kn_lp * 0.25, float(c[4]), *mt]
            for u, v, pidx, oc, w in bds:
                bm.append(m_i)
                bp.append(pidx)
                bo.append(oc)
                bra.append(int(rl[u]))
                brb.append(int(rl[v]))
                bwt.append(float(w))

        return {
            "k": k,
            "m": n_c,
            "Uf": pf["Uf"],
            "Pf": pf["Pf"],
            "Rf": pf["Rf"],
            "lm": pf["lm"],
            "roles_m": roles_m,
            "rt_m": rt_m,
            "psi_pad": psi_pad,
            "psi_mask": psi_mask,
            "psi_lens": psi_lens,
            "meta_m": meta_m,
            "bm": np.array(bm, dtype=np.int64),
            "bp": np.array(bp, dtype=np.int64),
            "bo": np.array(bo, dtype=np.int64),
            "bra": np.array(bra, dtype=np.int64),
            "brb": np.array(brb, dtype=np.int64),
            "bwt": np.array(bwt, dtype=np.float32),
        }

    def _collate_packs_to_device(
        self, batch_packs: Sequence[Dict[str, object]]
    ) -> Tuple[Tuple[torch.Tensor, ...], List[int]]:
        """Склеивает произвольный батч задач в единый плотный граф тензоров на `self.device`."""
        uf_l, pf_l, rf_l, lm_l = [], [], [], []
        sg_l, sr_l, sc_l, sinv_l = [], [], [], []
        rp_l, ro_l = [], []
        bm_l, bp_l, bo_l, bra_l, brb_l, bwt_l = [], [], [], [], [], []
        psi_p_l, psi_m_l, psi_ln_l, meta_l = [], [], [], []
        m_counts: List[int] = []

        slot_off = 0
        pair_off = 0
        cand_off = 0
        for p_idx, pk in enumerate(batch_packs):
            k = int(pk["k"])  # type: ignore[arg-type]
            m = int(pk["m"])  # type: ignore[arg-type]
            uf = pk["Uf"]  # type: ignore[assignment]
            pf_arr = pk["Pf"]  # type: ignore[assignment]
            uf_l.append(uf)
            pf_l.append(pf_arr)
            rf_l.append(pk["Rf"])
            lm_l.append(pk["lm"])

            roles_m: np.ndarray = pk["roles_m"]  # type: ignore[assignment]
            s_idx = np.broadcast_to(np.arange(k, dtype=np.int64)[None, :], (m, k)).reshape(-1) + slot_off
            c_idx = np.broadcast_to(np.arange(m, dtype=np.int64)[:, None], (m, k)).reshape(-1) + cand_off
            sg_l.append(s_idx)
            sr_l.append(roles_m.reshape(-1))
            sc_l.append(c_idx)
            sinv_l.append(np.full(m * k, 1.0 / float(max(1, k)), dtype=np.float32))

            rp_l.append(np.full(m, p_idx, dtype=np.int64))
            ro_l.append(pk["rt_m"])

            bm_arr: np.ndarray = pk["bm"]  # type: ignore[assignment]
            if len(bm_arr) > 0:
                bm_l.append(bm_arr + cand_off)
                bp_l.append(pk["bp"] + pair_off)  # type: ignore[operator]
                bo_l.append(pk["bo"])
                bra_l.append(pk["bra"])
                brb_l.append(pk["brb"])
                bwt_l.append(pk["bwt"])

            psi_p_l.append(pk["psi_pad"])
            psi_m_l.append(pk["psi_mask"])
            psi_ln_l.append(pk["psi_lens"])
            meta_l.append(pk["meta_m"])

            slot_off += len(uf)  # type: ignore[arg-type]
            pair_off += len(pf_arr)  # type: ignore[arg-type]
            cand_off += m
            m_counts.append(m)

        dev = self.device
        bm_t = torch.from_numpy(np.concatenate(bm_l)) if bm_l else torch.zeros(0, dtype=torch.long)
        bp_t = torch.from_numpy(np.concatenate(bp_l)) if bp_l else torch.zeros(0, dtype=torch.long)
        bo_t = torch.from_numpy(np.concatenate(bo_l)) if bo_l else torch.zeros(0, dtype=torch.long)
        bra_t = torch.from_numpy(np.concatenate(bra_l)) if bra_l else torch.zeros(0, dtype=torch.long)
        brb_t = torch.from_numpy(np.concatenate(brb_l)) if brb_l else torch.zeros(0, dtype=torch.long)
        bwt_t = torch.from_numpy(np.concatenate(bwt_l)) if bwt_l else torch.zeros(0, dtype=torch.float32)

        tensors = (
            torch.from_numpy(np.concatenate(uf_l, axis=0)).to(dev),
            torch.from_numpy(np.concatenate(pf_l, axis=0)).to(dev),
            torch.from_numpy(np.stack(rf_l, axis=0)).to(dev),
            torch.from_numpy(np.concatenate(lm_l, axis=0)).to(dev),
            torch.from_numpy(np.concatenate(sg_l)).to(dev),
            torch.from_numpy(np.concatenate(sr_l)).to(dev),
            torch.from_numpy(np.concatenate(sc_l)).to(dev),
            torch.from_numpy(np.concatenate(sinv_l)).to(dev),
            torch.from_numpy(np.concatenate(rp_l)).to(dev),
            torch.from_numpy(np.concatenate(ro_l)).to(dev),
            bm_t.to(dev),
            bp_t.to(dev),
            bo_t.to(dev),
            bra_t.to(dev),
            brb_t.to(dev),
            bwt_t.to(dev),
            torch.from_numpy(np.concatenate(psi_p_l, axis=0)).to(dev),
            torch.from_numpy(np.concatenate(psi_m_l, axis=0)).to(dev),
            torch.from_numpy(np.concatenate(psi_ln_l)).to(dev),
            torch.from_numpy(np.concatenate(meta_l, axis=0)).to(dev),
        )
        return tensors, m_counts

    def fit_bptt_free(
        self,
        train_items: Sequence[Dict[str, str]],
        slots_list: Sequence[Sequence[ExtractedSlot]],
        lm_feats_list: Sequence[Dict[str, object]],
        epochs: int = 10,
        batch_size: int = 128,
        lr: float = 0.018,
    ) -> Dict[str, float]:
        """Аналитический старт Вудбери + контрастивная энергия CRAM+z и Besag PL строго за `0` шагов BPTT."""
        t0 = time.perf_counter()
        self._fit_corpus_stats(train_items)
        self.ep_vecs.clear()
        self.ep_items.clear()
        self.sk_cnt.clear()
        self.sk_tot.clear()

        ep_vecs_list: Dict[Tuple[int, int], List[np.ndarray]] = {}
        uX, uY, pX, pY, rX, rY, lmX, lmY = [], [], [], [], [], [], [], []
        parsed: List[Tuple[Sequence[ExtractedSlot], Dict[str, object], float, Optional[float], Optional[np.ndarray]]] = []

        for it, sl, lmf in zip(train_items, slots_list, lm_feats_list):
            if len(sl) < 2 or len(sl) > 22:
                continue
            pf = self._featurize_problem(str(it["question"]), sl, lmf)
            st, _, ok = self.extractor.compile_train_dag(sl, str(it["answer"]))
            stc = _prune_to_connected_dag(st, len(sl), max_steps=MAX_GRAPH_STEPS) if (ok and st) else ()
            sgt = _dag_rational_sig(stc, len(sl)) if stc else None
            psi_true: Optional[np.ndarray] = None
            if stc and sgt is not None:
                rl, bds, rtop, psi_seq, _ = self._analyze_dag(stc, sl, pf)
                psi_true = psi_seq
                for i in range(len(sl)):
                    uX.append(pf["Uf"][i])
                    uY.append(int(rl[i]))
                    lmX.append(pf["lm"][i])
                    lmY.append(int(rl[i]))
                for _, _, pidx, oc, _ in bds:
                    pX.append(pf["Pf"][pidx])
                    pY.append(int(oc))
                rX.append(pf["Rf"])
                rY.append(int(rtop))
                sk, u_gr, u_cn = _to_narrative_skeleton(stc, sl)
                if len(u_cn) <= 1 and 1 <= len(u_gr) <= 10:
                    kb = (len(u_gr), len(u_cn))
                    self.sk_cnt.setdefault(kb, Counter())[sk] += 1
                    ep_vecs_list.setdefault(kb, []).append(pf["ctx"])  # type: ignore[arg-type]
                    self.ep_items.setdefault(kb, []).append((sk, rl[u_gr + u_cn]))
            parsed.append((sl, pf, extract_gt_number(str(it["answer"])), sgt, psi_true))

        self.ep_vecs = {kb: np.stack(vs, axis=0) for kb, vs in ep_vecs_list.items()}
        self.sk_tot = {kb: float(sum(cnt.values()) + 20.0) for kb, cnt in self.sk_cnt.items()}

        UX = np.stack(uX, axis=0).astype(np.float64)
        UY = np.eye(9, dtype=np.float64)[np.array(uY)]
        UY -= UY.mean(axis=0, keepdims=True)
        Wu0 = (-np.linalg.solve(UX.T @ UX + 18.0 * np.eye(DU_HASH), UX.T @ UY).T).astype(np.float32)

        PX = np.stack(pX, axis=0).astype(np.float64)
        PY = np.eye(7, dtype=np.float64)[np.array(pY)]
        PY -= PY.mean(axis=0, keepdims=True)
        Wp0 = (-np.linalg.solve(PX.T @ PX + 18.0 * np.eye(DP_HASH), PX.T @ PY).T).astype(np.float32)

        RX = np.stack(rX, axis=0).astype(np.float64)
        RY = np.eye(4, dtype=np.float64)[np.array(rY)]
        RY -= RY.mean(axis=0, keepdims=True)
        Wr0 = (-np.linalg.solve(RX.T @ RX + 18.0 * np.eye(DR_HASH), RX.T @ RY).T).astype(np.float32)

        LMX = np.stack(lmX, axis=0).astype(np.float64)
        LMY = np.eye(9, dtype=np.float64)[np.array(lmY)]
        LMY -= LMY.mean(axis=0, keepdims=True)
        Wlm0 = (-np.linalg.solve(LMX.T @ LMX + 45.0 * np.eye(3 * self.d_model), LMX.T @ LMY).T).astype(np.float32)

        self.ebm.set_woodbury_anchors(Wu0, Wp0, Wr0, Wlm0)

        train_packs: List[Tuple[Dict[str, object], np.ndarray, Optional[np.ndarray]]] = []
        for sl, pf, gt, sgt, psi_true in parsed:
            cands = self._synthesize_open_pool(sl, pf, mask_self=True)
            if len(cands) < 2:
                continue
            pw = np.zeros(len(cands), dtype=np.float32)
            has_exact = False
            if sgt is not None:
                for i, c in enumerate(cands):
                    if c[5] == sgt:
                        pw[i] = 1.0
                        has_exact = True
            for i, c in enumerate(cands):
                if abs(float(c[1]) - gt) <= 1e-2 and pw[i] == 0.0:
                    pw[i] = 0.15 if has_exact else 1.0
            if float(pw.sum()) <= 0.0:
                continue
            pw /= float(pw.sum())
            train_packs.append((self._pack_candidates_cpu(sl, pf, cands), pw, psi_true))

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.ebm.train()
        anchor_params = [
            self.ebm.W_u,
            self.ebm.W_p,
            self.ebm.W_r,
            self.ebm.W_lm,
            self.ebm.K_cram,
            self.ebm.w_meta,
        ]
        anchor_ids = {id(p) for p in anchor_params}
        neural_params = [p for p in self.ebm.parameters() if id(p) not in anchor_ids]
        opt = torch.optim.Adam(
            [
                {"params": anchor_params, "lr": float(lr)},
                {"params": neural_params, "lr": 1.2e-3},
            ]
        )
        total_steps = max(1, epochs * math.ceil(len(train_packs) / batch_size))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

        final_loss = 0.0
        for ep in range(epochs):
            perm = rng.permutation(len(train_packs))
            ep_loss = 0.0
            for b0 in range(0, len(train_packs), batch_size):
                b_ids = perm[b0 : b0 + batch_size]
                batch_pks = [train_packs[int(ix)][0] for ix in b_ids]
                tensors, m_counts = self._collate_packs_to_device(batch_pks)
                n_tot_c = sum(m_counts)

                opt.zero_grad()
                e_cram, _ = self.ebm.score_batch(*tensors, n_cands=n_tot_c, jitter=0.0)

                loss_pool = torch.tensor(0.0, device=self.device)
                cur = 0
                for ix, m_c in zip(b_ids, m_counts):
                    e_slice = e_cram[cur : cur + m_c]
                    pw_t = torch.from_numpy(train_packs[int(ix)][1]).to(self.device)
                    loss_pool = loss_pool - (pw_t * F.log_softmax(-e_slice, dim=0)).sum()
                    cur += m_c
                loss_pool = loss_pool / float(max(1, len(b_ids)))

                # Локальное псевдоправдоподобие Бесага (0 BPTT, 0 утечка цели) на сэмплах истинных DAG
                loss_pl = torch.tensor(0.0, device=self.device)
                couplings = self.ebm._bounded_couplings()
                pl_cnt = 0
                for ix in b_ids[:12]:
                    psi_np = train_packs[int(ix)][2]
                    if psi_np is None or len(psi_np) < 2:
                        continue
                    t_idx = int(rng.integers(0, len(psi_np)))
                    cands_np = np.repeat(psi_np[t_idx : t_idx + 1], 6, axis=0)
                    for row_i in range(1, 6):
                        alt_op = (int(np.argmax(cands_np[row_i, :4])) + row_i) % 4
                        cands_np[row_i, :4] = 0.0
                        cands_np[row_i, alt_op] = 1.0
                        if row_i >= 4:
                            cands_np[row_i, 13] = 1.0
                    psi_true_t = torch.tanh(self.ebm.step_proj(torch.from_numpy(psi_np).to(self.device)))
                    cand_t = torch.tanh(self.ebm.step_proj(torch.from_numpy(cands_np).to(self.device)))
                    e_cond = self.ebm.conditional_candidate_energies_no_leak(
                        psi_true_t, t_idx, cand_t, couplings=couplings
                    )
                    loss_pl = loss_pl + F.cross_entropy(-e_cond.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=self.device))
                    pl_cnt += 1
                if pl_cnt > 0:
                    loss_pl = 0.05 * (loss_pl / float(pl_cnt))

                loss = loss_pool + loss_pl + self.ebm.prox_regularization_loss()
                loss.backward()
                opt.step()
                sched.step()
                ep_loss += float(loss_pool.item()) * len(b_ids)
            final_loss = ep_loss / float(max(1, len(train_packs)))
            if (ep + 1) in (1, 5, epochs):
                print(
                    f"   [OpenGraph CRAM+z Fit] Ep {ep+1:2d}/{epochs} | "
                    f"Contrastive Loss={final_loss:.4f} | Gershgorin={self.ebm.gershgorin_margin():.2f}",
                    flush=True,
                )

        self.ebm.eval()
        return {
            "trainable_packs": float(len(train_packs)),
            "final_contrastive_loss": final_loss,
            "gershgorin_margin": self.ebm.gershgorin_margin(),
            "train_seconds": time.perf_counter() - t0,
        }

    def _assemble_and_score_candidates(
        self,
        q_text: str,
        slots: Sequence[ExtractedSlot],
        lm_feat: Optional[Dict[str, object]],
    ) -> Tuple[List[Dict[str, object]], float]:
        """Собирает пул открытых DAG, считает `(e_cram, e_unary)` и исполняет на `SphericalALU`."""
        k = len(slots)
        if k == 0:
            return [], 0.0
        pf = self._featurize_problem(q_text, slots, lm_feat)
        slot_vals: List[float] = pf["vs"]  # type: ignore[assignment]

        if k == 1:
            slot_regs = [self.alu.encode(slot_vals[0])]
            reg, _, dr_dag, inv = self.alu.execute_dag_spherical(slot_regs, (), direct_idx=0)
            dv, dr_dec = self.alu.decode_with_drift(reg)
            geo = self.alu.register_geometric_features(reg)
            return [
                {
                    "steps": (),
                    "val": float(dv),
                    "norm_drift": max(dr_dag, dr_dec),
                    "invalid": inv,
                    "geo": geo,
                    "e_cram": 0.0,
                    "e_unary": 0.0,
                    "meta": np.zeros(N_META_FEATS, dtype=np.float32),
                    "in_sys1": 0.0,
                    "sys1_lp": -12.0,
                }
            ], max(dr_dag, dr_dec)

        cands = self._synthesize_open_pool(slots, pf, mask_self=False)
        if not cands:
            return [], 0.0

        pk = self._pack_candidates_cpu(slots, pf, cands)
        tensors, _ = self._collate_packs_to_device([pk])
        with torch.no_grad():
            e_cram_t, e_unary_t = self.ebm.score_batch(*tensors, n_cands=len(cands), jitter=0.0)
        e_cram = e_cram_t.cpu().numpy().astype(np.float64)
        e_unary = e_unary_t.cpu().numpy().astype(np.float64)
        meta_m: np.ndarray = pk["meta_m"]  # type: ignore[assignment]

        tf_map: Dict[Tuple[Tuple[int, int, int], ...], float] = (
            lm_feat.get("cand_tf_lp", {}) if lm_feat is not None else {}  # type: ignore[assignment]
        )
        max_drift = 0.0
        records: List[Dict[str, object]] = []
        for idx, c in enumerate(cands):
            dag: Tuple[Tuple[int, int, int], ...] = c[0]  # type: ignore[assignment]
            has_tf = dag in tf_map
            s1_lp_val = float(tf_map[dag]) if has_tf else float(c[7])
            records.append(
                {
                    "steps": dag,
                    "val": float(c[1]),
                    "norm_drift": 0.0,
                    "invalid": False,
                    "e_cram": float(e_cram[idx]),
                    "e_unary": float(e_unary[idx]),
                    "meta": meta_m[idx],
                    "in_sys1": float(c[6]),
                    "sys1_lp": s1_lp_val,
                    "has_tf": has_tf,
                }
            )
        return records, max_drift

    def _build_rerank_matrix(self, cands: Sequence[Dict[str, object]]) -> np.ndarray:
        e_c = np.array([float(r["e_cram"]) for r in cands], dtype=np.float64)
        e_u = np.array([float(r["e_unary"]) for r in cands], dtype=np.float64)
        in_b = np.array([float(r["in_sys1"]) for r in cands], dtype=np.float64)
        s1_lp = np.array([float(r["sys1_lp"]) for r in cands], dtype=np.float64)
        valid_mask = [float(r["in_sys1"]) > 0.0 or bool(r.get("has_tf", False)) for r in cands]
        if any(valid_mask):
            max_lp = max(float(r["sys1_lp"]) for r, flg in zip(cands, valid_mask) if flg)
            s1_rel = np.maximum(-8.0, s1_lp - max_lp)
        else:
            s1_rel = np.zeros_like(s1_lp)
        meta = np.stack([r["meta"] for r in cands], axis=0).astype(np.float64)  # type: ignore[arg-type]
        return np.stack(
            [
                e_c - float(np.min(e_c)),
                e_u - float(np.min(e_u)),
                -in_b,
                -s1_rel / 6.0,
                meta[:, 5],
                meta[:, 14],
            ],
            axis=-1,
        )

    def enrich_lm_feats_with_system1_tf(
        self,
        items: Sequence[Dict[str, str]],
        slots_list: Sequence[Sequence[ExtractedSlot]],
        lm_feats_list: Sequence[Dict[str, object]],
        sys1_encoder: object,
        top_k: int = 16,
        use_lut: bool = False,
        batch_size: int = 64,
    ) -> List[Dict[str, object]]:
        """Оценивает топ-`top_k` кандидатов `CRAM+z` батчевым teacher-forced скорингом `System-1`."""
        questions: List[str] = []
        top_cands_per_q: List[List[Tuple[Tuple[int, int, int], ...]]] = []
        for it, sl, lmf in zip(items, slots_list, lm_feats_list):
            q_str = str(it["question"])
            questions.append(q_str)
            cands, _ = self._assemble_and_score_candidates(q_str, sl, lmf)
            if len(sl) <= 1 or not cands:
                top_cands_per_q.append([])
                continue
            top_recs = sorted(cands, key=lambda r: float(r["e_cram"]))[: max(1, int(top_k))]
            top_cands_per_q.append([r["steps"] for r in top_recs if r["steps"]])  # type: ignore[misc]

        tf_scores = sys1_encoder.score_candidate_dags_teacher_forced(  # type: ignore[attr-defined]
            questions=questions,
            slots_per_q=slots_list,
            cands_per_q=top_cands_per_q,
            extractor=self.extractor,
            alu=self.alu,
            use_lut=use_lut,
            batch_size=batch_size,
        )
        enriched: List[Dict[str, object]] = []
        for lmf, c_dags, c_scs in zip(lm_feats_list, top_cands_per_q, tf_scores):
            rec = dict(lmf) if lmf is not None else {}
            rec["cand_tf_lp"] = {dag: float(sc) for dag, sc in zip(c_dags, c_scs)}
            enriched.append(rec)
        return enriched

    def _boltzmann_rank(
        self, cands: Sequence[Dict[str, object]], e_vec: np.ndarray
    ) -> Tuple[List[Tuple[float, float]], float, float, float, Dict[str, object]]:
        es = self.boltzmann_beta * (e_vec - float(np.min(e_vec)))
        bw = np.exp(-np.minimum(es, 60.0))
        bw /= max(float(np.sum(bw)), 1e-12)
        val_prob: Dict[float, float] = {}
        val_best_e: Dict[float, float] = {}
        val_best_rec: Dict[float, Dict[str, object]] = {}
        for rec, wv, ev in zip(cands, bw, e_vec):
            kv = round(float(rec["val"]), 2)
            val_prob[kv] = val_prob.get(kv, 0.0) + float(wv)
            if kv not in val_best_e or float(ev) < val_best_e[kv]:
                val_best_e[kv] = float(ev)
                val_best_rec[kv] = rec
        ranked = sorted(val_prob.items(), key=lambda x: x[1], reverse=True)
        top_v = ranked[0][0]
        margin_e = float(val_best_e[ranked[1][0]] - val_best_e[top_v]) if len(ranked) >= 2 else 4.0
        ent = -sum(q * math.log(max(q, 1e-12)) for _, q in ranked)
        return ranked, float(top_v), margin_e, ent, val_best_rec[top_v]

    @staticmethod
    def _cert_feature_vec(
        e_vec: np.ndarray,
        margin_e: float,
        p_top: float,
        entropy_val: float,
        best_rec: Dict[str, object],
        k_slots: int,
    ) -> np.ndarray:
        meta: np.ndarray = best_rec["meta"]  # type: ignore[assignment]
        dag_len = len(best_rec["steps"])  # type: ignore[arg-type]
        return np.array(
            [
                float(np.min(e_vec)),
                float(margin_e),
                float(p_top),
                float(entropy_val),
                float(meta[0]),
                float(meta[1]),
                float(meta[3]),
                float(meta[5]),
                float(meta[10] + meta[11] + meta[14]),
                float(meta[13]),
                dag_len / 5.0,
                float(k_slots) / 6.0,
            ],
            dtype=np.float64,
        )

    def _compute_e_cert(
        self,
        e_vec: np.ndarray,
        margin_e: float,
        p_top: float,
        entropy_val: float,
        best_rec: Dict[str, object],
        k_slots: int,
        w_cert: Optional[np.ndarray] = None,
        b_cert: Optional[float] = None,
    ) -> float:
        cf = self._cert_feature_vec(e_vec, margin_e, p_top, entropy_val, best_rec, k_slots)
        if bool(best_rec.get("invalid", False)):
            return 50.0
        w = self.cert_weights if w_cert is None else w_cert
        b = self.cert_bias if b_cert is None else b_cert
        return float(cf @ w + b)

    @staticmethod
    def _fit_cert_ridge(rows: Sequence[Tuple[float, np.ndarray, int]]) -> Tuple[np.ndarray, float]:
        if len(rows) < 20:
            return np.zeros(N_CERT_FEATS, dtype=np.float64), 0.0
        X = np.stack([r[1] for r in rows], axis=0)
        y = np.array([1.0 - float(r[2]) for r in rows], dtype=np.float64)
        X_aug = np.concatenate([np.ones((len(X), 1), dtype=np.float64), X], axis=1)
        reg = 8.0 * np.eye(1 + N_CERT_FEATS, dtype=np.float64)
        reg[0, 0] = 1e-4
        w_all = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y)
        return w_all[1:].copy(), float(w_all[0])

    def _execute_top_on_alu(
        self,
        slot_regs: Sequence[np.ndarray],
        ranked: Sequence[Tuple[float, float]],
        cands: Sequence[Dict[str, object]],
        e_vec: np.ndarray,
    ) -> Tuple[float, List[float], float, bool]:
        """Исполняет выбранный DAG корня топ-1/3/5 на 12-разрядном сферическом регистре ALU."""
        val_best_idx: Dict[float, int] = {}
        for idx, (rec, ev) in enumerate(zip(cands, e_vec)):
            kv = round(float(rec["val"]), 2)
            if kv not in val_best_idx or float(ev) < float(e_vec[val_best_idx[kv]]):
                val_best_idx[kv] = idx

        top_preds: List[float] = []
        max_drift = 0.0
        top1_invalid = False
        for rank_i, (kv, _) in enumerate(ranked[:5]):
            dag = cands[val_best_idx[kv]]["steps"]  # type: ignore[index]
            reg, _, dr_dag, inv = self.alu.execute_dag_spherical(slot_regs, dag, direct_idx=0)
            dv, dr_dec = self.alu.decode_with_drift(reg)
            geo = self.alu.register_geometric_features(reg)
            max_drift = max(max_drift, dr_dag, dr_dec)
            if rank_i == 0:
                top1_invalid = bool(inv or float(geo[4]) > 0.5)
            top_preds.append(float(dv))
        return (top_preds[0] if top_preds else 0.0), top_preds, max_drift, top1_invalid

    def calibrate_on_split(
        self,
        cal_items: Sequence[Dict[str, str]],
        slots_list: Sequence[Sequence[ExtractedSlot]],
        lm_feats_list: Sequence[Dict[str, object]],
        min_cert_precision: float = 0.38,
        min_cert_coverage: float = 0.085,
    ) -> Dict[str, float]:
        """5-кратная двухуровневая (OOF) калибровка ранжирования и порогов DualCert на `cal`."""
        packed = []
        oracle_hits = 0
        for it, slots, lmf in zip(cal_items, slots_list, lm_feats_list):
            gt = extract_gt_number(str(it["answer"]))
            cands, _ = self._assemble_and_score_candidates(str(it["question"]), slots, lmf)
            u_epi = float(lmf.get("u_prob", 1.0)) if lmf is not None else 1.0  # type: ignore[arg-type]
            if not cands:
                packed.append(([], np.zeros((0, N_RERANK_FEATS), dtype=np.float64), u_epi, gt, len(slots)))
                continue
            rmat = self._build_rerank_matrix(cands)
            if any(abs(float(r["val"]) - gt) <= 1e-2 for r in cands):
                oracle_hits += 1
            packed.append((cands, rmat, u_epi, gt, len(slots)))

        n_cal = max(1, len(cal_items))
        has_sys1_signals = any(
            (lmf is not None) and (bool(lmf.get("cand_tf_lp")) or bool(lmf.get("beam_plans")))
            for lmf in lm_feats_list
        )
        w_grid = [
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        ]
        if has_sys1_signals:
            for w_in in (0.0, 0.15, 0.30):
                for w_lp in (0.0, 0.25, 0.45, 0.70):
                    if w_in == 0.0 and w_lp == 0.0:
                        continue
                    w_grid.append(np.array([1.0, 0.0, w_in, w_lp, 0.0, 0.0], dtype=np.float64))

        n_folds = 5 if len(packed) >= 25 else 1
        base_w = w_grid[0]
        if len(w_grid) > 1 and any(packed[i][0] for i in range(len(packed))):
            wg_hits: List[int] = []
            for wg in w_grid:
                sc_tot = 0
                for cands, rmat, _, gt, _ in packed:
                    if not cands:
                        continue
                    _, top_v, _, _, _ = self._boltzmann_rank(cands, rmat @ wg)
                    sc_tot += int(abs(top_v - gt) <= 1e-2)
                wg_hits.append(sc_tot)
            base_sc = wg_hits[0]
            best_wi = 0
            best_sc = base_sc
            for wi in range(1, len(w_grid)):
                if wg_hits[wi] > best_sc:
                    best_sc = wg_hits[wi]
                    best_wi = wi
            chosen_w = w_grid[best_wi] if best_sc >= base_sc + 2 else base_w
        else:
            chosen_w = base_w
        self.rerank_weights = chosen_w.copy()

        oof_rows: List[Tuple[int, float, np.ndarray, int]] = []
        p1_hits = 0
        p5_hits = 0
        un_hits = 0
        for idx, (cands, rmat, u_epi, gt, k_sl) in enumerate(packed):
            fold = idx % n_folds if n_folds > 1 else 0
            if not cands:
                oof_rows.append((fold, u_epi, np.zeros(N_CERT_FEATS, dtype=np.float64), 0))
                continue
            e_vec = rmat @ self.rerank_weights
            ranked, top_v, margin_e, ent_v, best_rec = self._boltzmann_rank(cands, e_vec)
            h1 = int(abs(top_v - gt) <= 1e-2)
            h5 = int(any(abs(kv - gt) <= 1e-2 for kv, _ in ranked[:5]))
            e_un = np.array([float(r["e_unary"]) for r in cands], dtype=np.float64)
            _, top_u, _, _, _ = self._boltzmann_rank(cands, e_un)
            h_un = int(abs(top_u - gt) <= 1e-2)
            p1_hits += h1
            p5_hits += h5
            un_hits += h_un
            p_top = float(ranked[0][1]) if ranked else 0.0
            cf = self._cert_feature_vec(e_vec, margin_e, p_top, ent_v, best_rec, k_sl)
            oof_rows.append((fold, u_epi, cf, h1))

        # 2-й уровень OOF для калибровки логита сертификата E_cert
        eval_rows: List[Tuple[float, float, int]] = []
        if n_folds > 1:
            for fold in range(n_folds):
                tr_cf = [(u, cf, h) for f, u, cf, h in oof_rows if f != fold]
                val_cf = [(u, cf, h) for f, u, cf, h in oof_rows if f == fold]
                wc_f, bc_f = self._fit_cert_ridge(tr_cf)
                for u, cf, h in val_cf:
                    eval_rows.append((u, float(cf @ wc_f + bc_f), h))
        else:
            all_cf = [(u, cf, h) for _, u, cf, h in oof_rows]
            wc_f, bc_f = self._fit_cert_ridge(all_cf)
            for u, cf, h in all_cf:
                eval_rows.append((u, float(cf @ wc_f + bc_f), h))

        self.cert_weights, self.cert_bias = self._fit_cert_ridge([(u, cf, h) for _, u, cf, h in oof_rows])

        u_vals = sorted(r[0] for r in eval_rows)
        e_vals = sorted(r[1] for r in eval_rows)
        min_n = max(12, int(min_cert_coverage * len(eval_rows))) if len(eval_rows) >= 40 else max(1, len(eval_rows) // 3)
        p1_cal = p1_hits / float(n_cal)
        target_prec = max(min_cert_precision, p1_cal + 0.18)

        best_cov = -1.0
        best_prec = 0.0
        best_pair = (u_vals[min(len(u_vals) - 1, int(0.85 * (len(u_vals) - 1)))], e_vals[min(len(e_vals) - 1, int(0.12 * (len(e_vals) - 1)))])
        for qu in np.linspace(0.45, 0.99, 20):
            tu = u_vals[min(len(u_vals) - 1, int(qu * (len(u_vals) - 1)))]
            for qe in np.linspace(0.04, 0.45, 36):
                te = e_vals[min(len(e_vals) - 1, int(qe * (len(e_vals) - 1)))]
                sub = [h for u, e, h in eval_rows if u <= tu and e <= te]
                if len(sub) < min_n:
                    continue
                pr = sum(sub) / float(len(sub))
                cv = len(sub) / float(len(eval_rows))
                if pr >= target_prec and cv > best_cov:
                    best_cov = cv
                    best_prec = pr
                    best_pair = (float(tu), float(te))

        if best_cov < 0.0:
            scored = []
            for qu in np.linspace(0.45, 0.99, 20):
                tu = u_vals[min(len(u_vals) - 1, int(qu * (len(u_vals) - 1)))]
                for qe in np.linspace(0.04, 0.45, 36):
                    te = e_vals[min(len(e_vals) - 1, int(qe * (len(e_vals) - 1)))]
                    sub = [h for u, e, h in eval_rows if u <= tu and e <= te]
                    if len(sub) >= min_n:
                        scored.append((sum(sub) / float(len(sub)), len(sub) / float(len(eval_rows)), tu, te))
            if scored:
                scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
                best_prec, best_cov, tu0, te0 = scored[0]
                best_pair = (float(tu0), float(te0))

        self.theta_epi, self.tau_energy = best_pair
        return {
            "cal_oracle_pct": 100.0 * oracle_hits / float(n_cal),
            "cal_oof_pass1_pct": 100.0 * p1_cal,
            "cal_pass5_pct": 100.0 * p5_hits / float(n_cal),
            "cal_unary_ablation_pct": 100.0 * un_hits / float(n_cal),
            "theta_epi": self.theta_epi,
            "tau_energy": self.tau_energy,
            "cal_cert_prec_pct": 100.0 * best_prec,
            "cal_cert_cov_pct": 100.0 * max(0.0, best_cov),
            "rerank_weights": [float(x) for x in self.rerank_weights],
        }

    def solve_blind(
        self,
        question_text: str,
        slots: Optional[Sequence[ExtractedSlot]] = None,
        lm_feat: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """Строго слепое решение одного вопроса GSM8K через открытый граф CRAM+z и `SphericalALU`."""
        if "####" in question_text or "<<" in question_text:
            raise ValueError("Anti-Cheat Violation: answer or <<>> tags leaked into solve_blind!")

        if slots is None:
            slots, _, _ = self.extractor.extract_slots(question_text, include_aux_consts=True)
        u_epi = float(lm_feat.get("u_prob", 1.0)) if lm_feat is not None else 1.0  # type: ignore[arg-type]

        cands, init_drift = self._assemble_and_score_candidates(question_text, slots, lm_feat)
        if not cands:
            return {
                "pred_pass1": 0.0,
                "pred_pass3": [0.0],
                "pred_pass5": [0.0],
                "pred_unary_ablation": 0.0,
                "best_steps": (),
                "sigma2_q": u_epi,
                "e_cert": 999.0,
                "dual_certified": False,
                "norm_drift": init_drift,
            }

        if self._sys1_reranker is not None and (lm_feat is None or "cand_tf_lp" not in lm_feat) and len(slots) >= 2:
            top_idx = sorted(range(len(cands)), key=lambda i: float(cands[i]["e_cram"]))[:16]
            top_dags = [cands[i]["steps"] for i in top_idx if cands[i]["steps"]]
            top_i_filt = [i for i in top_idx if cands[i]["steps"]]
            if top_dags:
                tf_sc = self._sys1_reranker.score_candidate_dags_teacher_forced(  # type: ignore[attr-defined]
                    questions=[question_text],
                    slots_per_q=[slots],
                    cands_per_q=[top_dags],  # type: ignore[arg-type]
                    extractor=self.extractor,
                    alu=self.alu,
                    use_lut=self._sys1_use_lut,
                    batch_size=16,
                )[0]
                for idx_c, sc_val in zip(top_i_filt, tf_sc):
                    cands[idx_c]["sys1_lp"] = float(sc_val)
                    cands[idx_c]["has_tf"] = True

        slot_regs = [self.alu.encode(float(s.val)) for s in slots]
        rmat = self._build_rerank_matrix(cands)
        e_vec = rmat @ self.rerank_weights
        ranked, _, margin_e, ent_v, best_rec = self._boltzmann_rank(cands, e_vec)
        p_top = float(ranked[0][1]) if ranked else 0.0

        top1_alu, top5_alu, dr_cram, inv_top1 = self._execute_top_on_alu(slot_regs, ranked, cands, e_vec)

        e_unary_arr = np.array([float(r["e_unary"]) for r in cands], dtype=np.float64)
        rk_un, _, _, _, _ = self._boltzmann_rank(cands, e_unary_arr)
        unary_alu, _, dr_un, _ = self._execute_top_on_alu(slot_regs, rk_un[:1], cands, e_unary_arr)

        if inv_top1:
            best_rec = dict(best_rec)
            best_rec["invalid"] = True
        e_cert = self._compute_e_cert(e_vec, margin_e, p_top, ent_v, best_rec, len(slots))
        certified = bool((u_epi <= self.theta_epi) and (e_cert <= self.tau_energy))
        max_drift = max(init_drift, dr_cram, dr_un)

        return {
            "pred_pass1": top1_alu,
            "pred_pass3": top5_alu[:3],
            "pred_pass5": top5_alu[:5],
            "pred_unary_ablation": unary_alu,
            "best_steps": best_rec["steps"],
            "sigma2_q": u_epi,
            "e_cert": e_cert,
            "dual_certified": certified,
            "norm_drift": max_drift,
        }
