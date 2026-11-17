"""uebr_graph_solver.py — Phase J (`Calcite-6 / UEBR`) System-2 Solver over Frozen Tier-1 (`5.41 MB 2-bit LUT`).

Upgrades `OpenGraphCRAMSolver` (`CRAM+z`) across four orthogonal mathematical axes while keeping
the frozen `95`-layer Tier-1 LUT backbone, `experiments/cram_gsm8k/`, and `simvq_split/` untouched:
  1) `GAQSlotBridge` (`SimVQ` orthogonal basis `C_frozen @ E_vq` + `Householder R_theta`) with
     Woodbury + VarEntropy risk gating `(1 - g_risk) * u_lm + u_gaq` (curing Lazy Identity Collapse).
  2) Compositional DAG propagation + context-conditioned `SPDGraphLaplacianField` `H_psi(Q_p)`
     solved per batch via matrix-RHS Cholesky implicit adjoint (`0-BPTT`, `AMP=False` shield),
     verified with matrix-free `ePC` and regularized by Directional Eikonal distance `d_psi`.
  3) Coupled 3-slot `RLNS-1` Fenchel-Young block training with absorbing Null Slot `0 (Ø)`,
     `2-Opt Block-Gibbs` role marginal inference, and multi-distractor pool seeding.
  4) `PermFlow C(U)` / `Dykstra` + lossless `Birkhoff Extension` QAP slot-to-skeleton matching
     for permutation- and distractor-invariant lifting + `SphericalBase10RegisterALU` execution.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import itertools
import math
from pathlib import Path
import random
import sys
import time
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_CUR_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CUR_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.cram_gsm8k.cram_open_graph import (  # noqa: E402
    DC_HASH,
    DP_HASH,
    DR_HASH,
    DU_HASH,
    D_STEP_PSI,
    D_Z_FIELD,
    MAX_GRAPH_STEPS,
    N_CERT_FEATS,
    N_FIELD_GROUPS,
    N_META_FEATS,
    N_RERANK_FEATS,
    OpenGraphCRAMSolver,
    Sparsemax,
    _compute_numeric_slot_features,
    _dag_rational_sig,
    _eval_dag_fast,
    _prune_to_connected_dag,
    _question_focus_and_money,
    _stem_by_corpus,
    _to_narrative_skeleton,
)
from experiments.cram_gsm8k.slot_extractor import (  # noqa: E402
    ExtractedSlot,
    TrainInducedQuantityExtractor,
    extract_gt_number,
    scan_question_tokens,
)
from experiments.cram_gsm8k.spherical_alu import SphericalBase10RegisterALU  # noqa: E402


K_SLOT_BUF = 24
D_GAQ_TOTAL = 64
D_LAP = 16
N_RLNS_STATES = 9
D_NULL_INV = 32

_ANCHOR_TRAIN_CACHE: Dict[Tuple[object, ...], Dict[str, object]] = {}


class UEBREncodeOut(NamedTuple):
    u_tot: torch.Tensor
    u_tot_rlns: torch.Tensor
    u_lex: torch.Tensor
    q_slots: torch.Tensor
    q_prob: torch.Tensor
    vq_loss: torch.Tensor
    g_prob: torch.Tensor


class UEBRBatchOut(NamedTuple):
    e_tot: torch.Tensor
    e_ablate: torch.Tensor
    u_tot: torch.Tensor
    u_tot_rlns: torch.Tensor
    p_pot: torch.Tensor
    p_pot_rlns: torch.Tensor
    q_slots: torch.Tensor
    q_prob: torch.Tensor
    vq_loss: torch.Tensor
    g_prob: torch.Tensor
    d_psi: torch.Tensor
    z_flat: Optional[torch.Tensor]
    H_psi: Optional[torch.Tensor]
    S_pad: Optional[torch.Tensor]
    Z_pad: Optional[torch.Tensor]


def _compute_null_slot_features(
    q_text: str,
    slots: Sequence[ExtractedSlot],
    pf: Dict[str, object],
    extractor: TrainInducedQuantityExtractor,
    vocab_counts: Counter,
    stop_words: Set[str],
    d_model: int,
) -> np.ndarray:
    """Position-invariant 32-dim Null Ø features: masked `Uf[-36:-9]` (`cols 14:17 = 0`) + 5 discourse/spherical cohesion signals."""
    k = len(slots)
    if k == 0:
        return np.zeros((0, D_NULL_INV), dtype=np.float32)
    hbu = DU_HASH - 36
    uf_arr: np.ndarray = pf["Uf"]  # type: ignore[assignment]
    uf_tail = uf_arr[:, hbu : hbu + 27].copy()
    # Zero non-invariant positional ratios (i/n_s, sent_id/max_sent, clause_id/max_clause)
    uf_tail[:, 14:17] = 0.0

    q_focus = extractor.extract_question_sentence(q_text)
    q_stems: Set[str] = set()
    for tk in scan_question_tokens(q_focus):
        if tk.kind == "WORD" and len(tk.lower) > 2:
            sw = _stem_by_corpus(tk.lower, vocab_counts)
            if sw not in stop_words:
                q_stems.add(sw)

    sent_stems: Dict[int, Set[str]] = {}
    for tk in scan_question_tokens(q_text):
        if tk.kind == "WORD" and len(tk.lower) > 2:
            sw = _stem_by_corpus(tk.lower, vocab_counts)
            if sw not in stop_words:
                sent_stems.setdefault(tk.sent_id, set()).add(sw)
    n_sents = len(sent_stems)

    lm_arr: np.ndarray = pf["lm"]  # type: ignore[assignment]
    hp_u = lm_arr[:, d_model : 2 * d_model]
    hq_u: np.ndarray = pf["hq_u"]  # type: ignore[assignment]
    gr_idx = [i for i, s in enumerate(slots) if s.kind != "CONST"]

    coh = np.zeros((k, 5), dtype=np.float32)
    for i, sl in enumerate(slots):
        if sl.kind == "CONST":
            coh[i] = [0.0, 0.0, 1.0, 0.0, 0.0]
            continue
        my_w = sent_stems.get(sl.sent_id, set())
        if n_sents <= 1:
            s_peer = 1.0
        else:
            oth_w: Set[str] = set()
            for sid, ws in sent_stems.items():
                if sid != sl.sent_id:
                    oth_w |= ws
            s_peer = len(my_w & oth_w) / float(max(1, len(my_w)))
        s_q = len(my_w & q_stems) / float(max(1, len(my_w)))
        u_match = 1.0 if float(uf_arr[i, hbu + 1]) > 1e-5 else 0.0
        iso_f = 1.0 if (n_sents >= 2 and s_peer == 0.0 and s_q == 0.0 and u_match == 0.0 and not sl.is_in_question) else 0.0
        peers = [float(np.dot(hp_u[i], hp_u[j])) for j in gr_idx if j != i]
        p_cos = max(peers) if peers else 0.0
        q_cos = float(np.dot(hp_u[i], hq_u))
        coh[i] = [s_peer, s_q, iso_f, p_cos, q_cos]
    return np.concatenate([uf_tail, coh], axis=-1).astype(np.float32)


# ===========================================================================
# 1. Closed-Form PermFlow C(U), Dykstra, and Lossless Birkhoff Extension QAP
# ===========================================================================


def permflow_tangent_project(U: torch.Tensor) -> torch.Tensor:
    """Closed-form orthogonal projection onto Birkhoff affine tangent hull `T` in `float64`."""
    Ud = U.double()
    row_m = Ud.mean(dim=-1, keepdim=True)
    col_m = Ud.mean(dim=-2, keepdim=True)
    grand_m = Ud.mean(dim=(-2, -1), keepdim=True)
    return (Ud - row_m - col_m + grand_m).to(U.dtype)


def project_birkhoff_dykstra(X: torch.Tensor, iters: int = 24) -> torch.Tensor:
    """Projects `(..., n, n)` onto Birkhoff polytope `D_n` via Dykstra + doubly-stochastic polish."""
    n = X.size(-1)
    Xd = X.double()
    J = torch.full_like(Xd, 1.0 / float(n))
    x = Xd
    p = torch.zeros_like(Xd)
    q = torch.zeros_like(Xd)
    for _ in range(iters):
        y = (x + p).clamp(min=0.0)
        p = x + p - y
        x = J + permflow_tangent_project(y + q - J)
        q = y + q - x
    A = x.clamp(min=1e-12)
    for _ in range(10):
        A = A / A.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        A = A / A.sum(dim=-2, keepdim=True).clamp(min=1e-12)
    return A.to(X.dtype)


def _permflow_tangent_np(U: np.ndarray) -> np.ndarray:
    n = float(U.shape[-1])
    rm = U.sum(axis=-1, keepdims=True) / n
    cm = U.sum(axis=-2, keepdims=True) / n
    gm = U.sum() / (n * n)
    return U - rm - cm + gm


def _project_birkhoff_dykstra_np(X: np.ndarray, iters: int = 6) -> np.ndarray:
    n = int(X.shape[-1])
    j_val = 1.0 / float(n)
    x = np.asarray(X, dtype=np.float64).copy()
    p = np.zeros_like(x)
    q = np.zeros_like(x)
    for _ in range(iters):
        y = np.maximum(x + p, 0.0)
        p = x + p - y
        rm_y = (y + q).sum(axis=-1, keepdims=True) / float(n)
        cm_y = (y + q).sum(axis=-2, keepdims=True) / float(n)
        gm_y = (y + q).sum() / float(n * n)
        x = y + q - rm_y - cm_y + (gm_y + j_val)
        q = y + q - x
    A = np.maximum(x, 1e-12)
    for _ in range(6):
        A /= np.maximum(A.sum(axis=-1, keepdims=True), 1e-12)
        A /= np.maximum(A.sum(axis=-2, keepdims=True), 1e-12)
    return A


def _max_weight_bipartite_matching(weights: List[List[float]]) -> List[int]:
    """Exact O(n^3) Hungarian max-weight matching (`weights[i][j]`) without external SciPy dependency."""
    n = len(weights)
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    inf_val = 1e18
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf_val] * (n + 1)
        used = [False] * (n + 1)
        row_w = weights[i - 1]
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf_val
            j1 = 0
            row_i0 = weights[i0 - 1] if i0 > 0 else row_w
            for j in range(1, n + 1):
                if not used[j]:
                    cur = -row_i0[j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    perm = [0] * n
    for j in range(1, n + 1):
        if p[j] > 0:
            perm[p[j] - 1] = j - 1
    return perm


def birkhoff_extension_round_np(
    A_mat: np.ndarray,
    eval_fn: Callable[[np.ndarray], float],
    S_mat: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float, List[np.ndarray]]:
    """Convex Birkhoff-von Neumann decomposition + lossless rounding (`f(P*) <= F_S(A)`)."""
    R = np.asarray(A_mat, dtype=np.float64).copy()
    n = int(R.shape[0])
    assert R.shape == (n, n), f"Expected square (n, n), got {R.shape}"
    S_arr = R.copy() if S_mat is None else np.asarray(S_mat, dtype=np.float64)

    if float(np.max(np.abs(R.sum(axis=1) - 1.0))) > 1e-4 or float(np.min(R)) < -1e-5:
        R = _project_birkhoff_dykstra_np(R, iters=12)
    R = np.clip(R, 0.0, None)

    alphas: List[float] = []
    perms: List[np.ndarray] = []
    r_mass = 1.0
    max_terms = min(2 * n + 2, 12)
    row_ar = np.arange(n, dtype=np.int64)
    S_list = S_arr.tolist()

    for _ in range(max_terms):
        if r_mass <= 1e-4:
            break
        R_list = R.tolist()
        w_mat = [[S_list[i][j] if R_list[i][j] > 1e-9 else -1e9 for j in range(n)] for i in range(n)]
        perm = _max_weight_bipartite_matching(w_mat)
        alpha = min(R_list[i][perm[i]] for i in range(n))
        if alpha <= 1e-11:
            w_mat_fb = [[S_list[i][j] if R_list[i][j] > 0.0 else -1e9 for j in range(n)] for i in range(n)]
            perm = _max_weight_bipartite_matching(w_mat_fb)
            alpha = min(R_list[i][perm[i]] for i in range(n))
            if alpha <= 1e-12:
                break

        P = np.zeros((n, n), dtype=np.float64)
        P[row_ar, perm] = 1.0
        R[row_ar, perm] = np.maximum(0.0, R[row_ar, perm] - alpha)
        R[R < 1e-13] = 0.0
        alphas.append(float(alpha))
        perms.append(P)
        r_mass -= float(alpha)

    if not perms:
        P_eye = np.eye(n, dtype=np.float64)
        v_eye = float(eval_fn(P_eye))
        return P_eye, v_eye, v_eye, [P_eye]

    a_sum = max(sum(alphas), 1e-12)
    alphas = [a / a_sum for a in alphas]
    scored: List[Tuple[float, np.ndarray]] = []
    ext_val = 0.0
    for a_l, P_l in zip(alphas, perms):
        fv = float(eval_fn(P_l))
        ext_val += a_l * fv
        scored.append((fv, P_l))

    scored.sort(key=lambda x: x[0])
    best_val, best_P = scored[0]
    top_perms = [p_l for _, p_l in scored[:2]]
    return best_P, float(best_val), float(ext_val), top_perms


def solve_qap_permflow_birkhoff(
    C_lin: np.ndarray,
    edges: Sequence[Tuple[int, int, int]],
    pair_mats: Sequence[np.ndarray],
    lam_qap: float = 0.35,
    steps: int = 6,
) -> Tuple[List[List[int]], bool]:
    """Solves slot-to-(skeleton + Null) QAP in `D_n` via NumPy PermFlow + Dykstra + lossless Birkhoff Extension."""
    n = int(C_lin.shape[0])
    if n <= 1:
        return ([[0]] if n == 1 else [[]]), True

    C_f = np.asarray(C_lin, dtype=np.float64)
    logits0 = -C_f * 1.2
    logits0 -= np.max(logits0, axis=-1, keepdims=True)
    e_l = np.exp(np.clip(logits0, -30.0, 0.0))
    A = _project_birkhoff_dykstra_np(e_l / np.maximum(e_l.sum(axis=-1, keepdims=True), 1e-12), iters=5)

    def _eval_and_grad(X_mat: np.ndarray) -> Tuple[float, np.ndarray]:
        val = float(np.sum(C_f * X_mat))
        grad = C_f.copy()
        if lam_qap > 0.0 and edges:
            for e_tup, B_mat in zip(edges, pair_mats):
                u_s, v_s = e_tup[0], e_tup[1]
                col_u = X_mat[:, u_s]
                col_v = X_mat[:, v_s]
                Bv = B_mat @ col_v
                BTu = B_mat.T @ col_u
                val += lam_qap * float(col_u @ Bv)
                grad[:, u_s] += lam_qap * Bv
                grad[:, v_s] += lam_qap * BTu
        return val, grad

    lr = 0.35
    for st in range(min(steps, 6)):
        _, g = _eval_and_grad(A)
        step_dir = _permflow_tangent_np(-g)
        step_sz = lr * (1.0 - 0.5 * st / float(max(1, min(steps, 6))))
        A = _project_birkhoff_dykstra_np(A + step_sz * step_dir, iters=4)

    _, g_fin = _eval_and_grad(A)
    best_P, best_v, ext_v, top_Ps = birkhoff_extension_round_np(
        A,
        eval_fn=lambda P_m: _eval_and_grad(P_m)[0],
        S_mat=-g_fin,
    )
    lossless = bool(best_v <= ext_v + 1e-7)

    out_assigns: List[List[int]] = []
    seen_assigns: Set[Tuple[int, ...]] = set()
    for P_m in top_Ps:
        cols_to_rows = [int(np.argmax(P_m[:, j])) for j in range(n)]
        tup = tuple(cols_to_rows)
        if tup not in seen_assigns:
            seen_assigns.add(tup)
            out_assigns.append(cols_to_rows)
    if not out_assigns:
        out_assigns.append([int(np.argmax(best_P[:, j])) for j in range(n)])
    return out_assigns, lossless


# ===========================================================================
# 2. Sectioned GAQSlotBridge (SimVQ + Householder) & Matrix Cholesky Adjoint
# ===========================================================================


class SectionedGAQBridge(nn.Module):
    """Multi-section Geometry Alignment Quantization (`SimVQ + Householder R_theta`) bridge."""

    def __init__(self, d_in: int, m_sections: int = 4, d_code: int = 16, num_codes: int = 32, seed: int = 31415) -> None:
        super().__init__()
        self.d_in = int(d_in)
        self.m_sections = int(m_sections)
        self.d_code = int(d_code)
        self.num_codes = int(num_codes)
        self.d_out = self.m_sections * self.d_code

        gen = torch.Generator().manual_seed(seed)
        self.register_buffer(
            "C_frozen",
            torch.randn(self.m_sections, self.num_codes, self.d_code, generator=gen) / math.sqrt(float(self.d_code)),
        )
        self.in_norm = nn.LayerNorm(self.d_in)
        self.in_proj = nn.Linear(self.d_in, self.d_out, bias=False)
        eye_stack = torch.eye(self.d_code).unsqueeze(0).repeat(self.m_sections, 1, 1)
        self.E_vq = nn.Parameter(eye_stack + 0.03 * torch.randn(self.m_sections, self.d_code, self.d_code, generator=gen))

    def codebooks(self) -> torch.Tensor:
        return torch.bmm(self.C_frozen, self.E_vq)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n_items = x.size(0)
        if n_items == 0:
            return torch.zeros((0, self.d_out), device=x.device, dtype=x.dtype), torch.zeros((), device=x.device, dtype=x.dtype)
        r = self.in_proj(self.in_norm(x)).view(n_items, self.m_sections, self.d_code)
        c_mats = self.codebooks()  # (M_sec, K, d_code)

        # Squared distances per section: (n_items, M_sec, K)
        r_s = r.transpose(0, 1)  # (M_sec, n_items, d_code)
        dists = torch.cdist(r_s, c_mats, p=2.0)  # (M_sec, n_items, K)
        idx = dists.argmin(dim=-1)  # (M_sec, n_items)
        sec_ar = torch.arange(self.m_sections, device=x.device).unsqueeze(1).expand(-1, n_items)
        c_bar = c_mats[sec_ar, idx].transpose(0, 1).reshape(-1, self.d_code)
        r_flat = r.reshape(-1, self.d_code)

        r_norm = r_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        c_norm = c_bar.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        v = r_flat / r_norm
        u = c_bar / c_norm
        w = (u - v).detach()
        w_sq = (w * w).sum(dim=-1, keepdim=True)
        dx = r_flat - r_flat.detach()
        w_dot_dx = (w * dx).sum(dim=-1, keepdim=True)
        h_dx = torch.where(w_sq > 1e-10, dx - 2.0 * w * (w_dot_dx / w_sq.clamp(min=1e-10)), dx)
        rot_dx = (c_norm / r_norm).detach() * h_dx
        q_hat = (c_bar.detach() + rot_dx).view(n_items, self.d_out)

        vq_loss = F.mse_loss(r_flat.detach(), c_bar) + 0.25 * F.mse_loss(r_flat, c_bar.detach())
        return q_hat, vq_loss


class _ImplicitLaplacianSolveFn(torch.autograd.Function):
    """Exact 0-BPTT matrix-RHS Cholesky adjoint solver for SPD `H_psi @ Z = S_rhs` under AMP shield."""

    @staticmethod
    def forward(ctx, H_psi: torch.Tensor, S_rhs: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(H_psi.device.type, enabled=False):
            work_dtype = torch.float64 if H_psi.dtype == torch.float64 else torch.float32
            H_w = 0.5 * (H_psi.to(work_dtype) + H_psi.to(work_dtype).transpose(-1, -2))
            S_w = S_rhs.to(work_dtype)
            s_col = S_w.unsqueeze(-1) if S_w.dim() == 2 else S_w
            L = torch.linalg.cholesky(H_w)
            Z0 = torch.cholesky_solve(s_col, L)
            out = Z0.squeeze(-1) if S_w.dim() == 2 else Z0
        ctx.save_for_backward(L, Z0)
        ctx.in_dtype = H_psi.dtype
        ctx.squeeze_last = S_rhs.dim() == 2
        return out.to(H_psi.dtype)

    @staticmethod
    def backward(ctx, grad_Z: torch.Tensor):
        L, Z0 = ctx.saved_tensors
        with torch.amp.autocast(L.device.type, enabled=False):
            gz = grad_Z.to(L.dtype)
            gz_col = gz.unsqueeze(-1) if ctx.squeeze_last else gz
            Lam = torch.cholesky_solve(gz_col, L)
            grad_H = -0.5 * (
                torch.matmul(Lam, Z0.transpose(-1, -2))
                + torch.matmul(Z0, Lam.transpose(-1, -2))
            )
            grad_S = Lam.squeeze(-1) if ctx.squeeze_last else Lam
        return grad_H.to(ctx.in_dtype), grad_S.to(ctx.in_dtype)


@torch.no_grad()
def epc_equilibrium_solve(H_psi: torch.Tensor, S_rhs: torch.Tensor, steps: int = 45) -> torch.Tensor:
    """Matrix-inversion-free preconditioned error-space Predictive Coding (`ePC`) relaxation."""
    H_f = 0.5 * (H_psi.float() + H_psi.float().transpose(-1, -2))
    S_f = S_rhs.float()
    diag_inv = 1.0 / torch.diagonal(H_f, dim1=-2, dim2=-1).clamp(min=1e-6).unsqueeze(-1)
    Z = torch.zeros_like(S_f)
    eps = S_f - torch.bmm(H_f, Z)
    R = diag_inv * eps
    P = R.clone()
    rz_old = (eps * R).sum(dim=-2, keepdim=True)
    for _ in range(steps):
        Hp = torch.bmm(H_f, P)
        denom = (P * Hp).sum(dim=-2, keepdim=True).clamp(min=1e-12)
        alpha = rz_old / denom
        Z = Z + alpha * P
        eps = eps - alpha * Hp
        if float(eps.abs().max().item()) < 1e-7:
            break
        R = diag_inv * eps
        rz_new = (eps * R).sum(dim=-2, keepdim=True)
        beta = rz_new / rz_old.clamp(min=1e-12)
        P = R + beta * P
        rz_old = rz_new
    return Z.to(S_rhs.dtype)


class SPDGraphLaplacianField(nn.Module):
    """Input-conditioned SPD Graph Laplacian `H_psi(Q_p)`, compositional DAG tree builder, and Eikonal head."""

    def __init__(self, d_slot: int = D_GAQ_TOTAL, d_lap: int = D_LAP, d_psi_step: int = D_STEP_PSI) -> None:
        super().__init__()
        self.d_slot = int(d_slot)
        self.d_lap = int(d_lap)

        # Cross-attention pooling from problem slot/question codes -> 16 graph nodes (B, 16, 64)
        self.node_bank = nn.Parameter(torch.randn(d_lap, d_slot) * 0.15)
        self.node_k_proj = nn.Linear(d_slot, d_slot, bias=False)
        self.node_v_proj = nn.Linear(d_slot, d_slot, bias=False)
        self.q_ctx_proj = nn.Linear(d_slot, d_slot)

        # SPD Laplacian parameters (Rademacher orthogonal key init)
        self.slot_key = nn.Linear(d_slot, d_slot, bias=False)
        with torch.no_grad():
            rad = (torch.randint(0, 2, (d_slot, d_slot)).float() * 2.0 - 1.0) / math.sqrt(float(d_slot))
            self.slot_key.weight.copy_(rad)
        self.edge_scale = nn.Parameter(torch.tensor(8.0))
        self.edge_bias = nn.Parameter(torch.tensor(-2.5))
        self.mu_raw = nn.Parameter(torch.tensor(-0.3))

        # Compositional DAG propagation over operand slots (a_t, b_t, op_t, psi_t)
        self.leaf_proj = nn.Linear(d_slot, d_lap, bias=False)
        self.op_emb = nn.Embedding(4, d_lap)
        self.step_comp_fc = nn.Sequential(
            nn.Linear(2 * d_lap + d_lap + d_psi_step, d_lap),
            nn.SiLU(),
            nn.Linear(d_lap, d_lap),
        )
        self.src_proj = nn.Linear(d_lap + d_psi_step, d_lap, bias=False)

        # Directional Eikonal distance-to-goal head d_psi(z_c, q_Q) >= 0
        self.dist_fc1 = nn.Linear(d_lap + d_slot, 48)
        self.dist_fc2 = nn.Linear(48, 1)

        # Small near-zero output residual init (preserves 100% live grads while guarding Woodbury Ridge anchors)
        nn.init.normal_(self.src_proj.weight, std=1e-3)
        self.lap_energy_scale = nn.Parameter(torch.tensor(0.05))
        self.lap_dist_scale = nn.Parameter(torch.tensor(0.02))

    def build_laplacian(self, q_slots_pad: torch.Tensor, slot_mask: torch.Tensor, q_prob: torch.Tensor) -> torch.Tensor:
        """Constructs strictly SPD `H_psi(Q_p)` `(B, D_LAP, D_LAP)` from quantized slot & question codes."""
        bsz = q_slots_pad.size(0)
        # Include question code as extra key/value column so pooling always has >= 1 valid entry
        kv_toks = torch.cat([q_slots_pad, q_prob.unsqueeze(1)], dim=1)  # (B, K_BUF+1, d_slot)
        q_ones = torch.ones((bsz, 1), device=slot_mask.device, dtype=torch.bool)
        full_mask = torch.cat([slot_mask, q_ones], dim=1)  # (B, K_BUF+1)

        keys = self.node_k_proj(kv_toks)
        vals = self.node_v_proj(kv_toks)
        attn_logits = torch.matmul(self.node_bank.unsqueeze(0), keys.transpose(-1, -2)) / math.sqrt(float(self.d_slot))
        attn_logits = attn_logits.masked_fill(~full_mask.unsqueeze(1), -1e9)
        attn_w = F.softmax(attn_logits, dim=-1)
        Q_p = torch.bmm(attn_w, vals) + self.q_ctx_proj(q_prob).unsqueeze(1)  # (B, D_LAP, d_slot)

        k_n = F.normalize(self.slot_key(Q_p), dim=-1)
        sim = torch.bmm(k_n, k_n.transpose(-1, -2))
        a_raw = F.softplus(self.edge_scale * sim + self.edge_bias)
        eye = torch.eye(self.d_lap, device=Q_p.device, dtype=Q_p.dtype).unsqueeze(0)
        a_sym = 0.5 * (a_raw + a_raw.transpose(-1, -2)) * (1.0 - eye)
        deg = a_sym.sum(dim=-1)
        lap = torch.diag_embed(deg) - a_sym
        mass = torch.exp(torch.clamp(self.mu_raw, -1.2, 2.0))
        return lap + mass * eye

    def propagate_dag_sources(
        self,
        q_slots_pad: torch.Tensor,
        cand_prob_idx: torch.Tensor,
        dag_op: torch.Tensor,
        dag_a: torch.Tensor,
        dag_b: torch.Tensor,
        dag_len: torch.Tensor,
        psi_raw_pad: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized operand-aware composition over DAG steps (`<= MAX_GRAPH_STEPS`), returning `S_c (N_tot, D_LAP)`."""
        n_tot = int(cand_prob_idx.size(0))
        if n_tot == 0:
            return torch.zeros((0, self.d_lap), device=q_slots_pad.device, dtype=q_slots_pad.dtype)

        leaf_bank = torch.tanh(self.leaf_proj(q_slots_pad))  # (B, K_BUF, D_LAP)
        cand_leaves = leaf_bank[cand_prob_idx]  # (N_tot, K_BUF, D_LAP)
        max_steps = max(1, int(dag_len.max().item()))
        ar_n = torch.arange(n_tot, device=q_slots_pad.device, dtype=torch.long)

        cur_nodes = cand_leaves
        inter_list: List[torch.Tensor] = []
        for t in range(max_steps):
            max_idx = cur_nodes.size(1) - 1
            ia = dag_a[:, t].clamp(min=0, max=max_idx)
            ib = dag_b[:, t].clamp(min=0, max=max_idx)
            va = cur_nodes[ar_n, ia]
            vb = cur_nodes[ar_n, ib]
            oe = self.op_emb(dag_op[:, t].clamp(0, 3))
            st_vec = torch.tanh(self.step_comp_fc(torch.cat([va, vb, oe, psi_raw_pad[:, t]], dim=-1)))
            inter_list.append(st_vec)
            if t + 1 < max_steps:
                cur_nodes = torch.cat([cur_nodes, st_vec.unsqueeze(1)], dim=1)

        inter_stack = torch.stack(inter_list, dim=1)  # (N_tot, max_steps, D_LAP)
        last_idx = (dag_len - 1).clamp(min=0, max=max_steps - 1)
        root_tree = inter_stack[ar_n, last_idx]
        root_psi = psi_raw_pad[ar_n, last_idx]
        return self.src_proj(torch.cat([root_tree, root_psi], dim=-1))

    def distance_field(self, z_c: torch.Tensor, q_prob: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.dist_fc1(torch.cat([z_c, q_prob], dim=-1)))
        return F.softplus(self.dist_fc2(h)).squeeze(-1)

    def directional_eikonal_loss(
        self,
        z_pos: torch.Tensor,
        z_neg: torch.Tensor,
        q_prob_sel: torch.Tensor,
    ) -> torch.Tensor:
        if z_pos.size(0) == 0:
            return torch.zeros((), device=z_pos.device, dtype=z_pos.dtype)
        zp = z_pos.detach()
        zn = z_neg.detach()
        qp = q_prob_sel.detach()
        gamma = torch.rand(zp.size(0), 1, device=zp.device, dtype=zp.dtype) * 0.90 + 0.05
        z_gamma = ((1.0 - gamma) * zp + gamma * zn).requires_grad_(True)
        d_g = self.distance_field(z_gamma, qp)
        grad_z = torch.autograd.grad(d_g.sum(), z_gamma, create_graph=True)[0]
        g_norm = torch.sqrt(grad_z.pow(2).sum(dim=-1) + 1e-8)
        eik_pde = ((g_norm - 1.0) ** 2).mean()
        d_pos = self.distance_field(zp, qp)
        d_neg = self.distance_field(zn, qp)
        bnd_loss = (d_pos ** 2).mean() + F.relu(1.0 - d_neg).pow(2).mean()
        return eik_pde + 0.5 * bnd_loss

    def score_batch(
        self,
        q_slots_pad: torch.Tensor,
        slot_mask: torch.Tensor,
        q_prob: torch.Tensor,
        cand_prob_idx: torch.Tensor,
        cand_local_idx: torch.Tensor,
        m_max: int,
        dag_op: torch.Tensor,
        dag_a: torch.Tensor,
        dag_b: torch.Tensor,
        dag_len: torch.Tensor,
        psi_raw_pad: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solves `H_psi(Q_p) Z* = S_pad` for all `(B, M_max)` candidates in one Cholesky batch."""
        bsz = q_slots_pad.size(0)
        H_psi = self.build_laplacian(q_slots_pad, slot_mask, q_prob)
        s_flat = self.propagate_dag_sources(
            q_slots_pad, cand_prob_idx, dag_op, dag_a, dag_b, dag_len, psi_raw_pad
        )
        S_pad = torch.zeros((bsz, self.d_lap, m_max), device=q_slots_pad.device, dtype=q_slots_pad.dtype)
        if cand_prob_idx.numel() > 0:
            S_pad[cand_prob_idx, :, cand_local_idx] = s_flat

        Z_pad = _ImplicitLaplacianSolveFn.apply(H_psi, S_pad)
        z_flat = Z_pad[cand_prob_idx, :, cand_local_idx] if cand_prob_idx.numel() > 0 else torch.zeros_like(s_flat)
        q_prob_c = q_prob[cand_prob_idx]
        d_psi = self.distance_field(z_flat, q_prob_c)

        e_quad = -0.5 * (z_flat * s_flat).sum(dim=-1) / math.sqrt(float(self.d_lap))
        e_lap = torch.clamp(self.lap_energy_scale, -2.0, 2.0) * e_quad + torch.clamp(self.lap_dist_scale, 0.0, 2.0) * d_psi
        return e_lap, d_psi, z_flat, H_psi, S_pad, Z_pad


# ===========================================================================
# 3. UEBROpenGraphEBM & Coupled 3-Slot RLNS-1 / 2-Opt Block-Gibbs
# ===========================================================================


class UEBROpenGraphEBM(nn.Module):
    """Unified `Calcite-6 / UEBR` Energy Model with exact ablation flags."""

    def __init__(
        self,
        d_model: int = 384,
        use_gaq_bridge: bool = True,
        use_spd_laplacian: bool = True,
        use_rlns_and_null: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.use_gaq_bridge = bool(use_gaq_bridge)
        self.use_spd_laplacian = bool(use_spd_laplacian)
        self.use_rlns_and_null = bool(use_rlns_and_null)

        self.register_buffer("Wu0", torch.zeros(9, DU_HASH))
        self.register_buffer("Wp0", torch.zeros(7, DP_HASH))
        self.register_buffer("Wr0", torch.zeros(4, DR_HASH))
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
        self.register_buffer("tau_risk_floor", torch.tensor(0.45))
        self.register_buffer(
            "wm_pen_mask",
            torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1], dtype=torch.bool),
        )

        self.W_u = nn.Parameter(torch.zeros(9, DU_HASH))
        self.W_p = nn.Parameter(torch.zeros(7, DP_HASH))
        self.W_r = nn.Parameter(torch.zeros(4, DR_HASH))
        self.W_lm = nn.Parameter(torch.zeros(9, 3 * self.d_model))
        self.K_cram = nn.Parameter(torch.zeros(9, 9, 7))
        self.w_meta = nn.Parameter(init_wm.clone())

        self.lm_norm = nn.LayerNorm(3 * self.d_model)
        self.sparse_gate = nn.Linear(3 * self.d_model, 32)
        self.sparse_role = nn.Linear(32, 9, bias=False)
        nn.init.normal_(self.sparse_role.weight, std=2e-4)

        # Baseline Gershgorin field (preserved for A1/A2 and prox compatibility)
        self.d_step = D_STEP_PSI
        self.d_z = D_Z_FIELD
        self.n_groups = N_FIELD_GROUPS
        self.group_k = 4
        self.m_convex = 5.0
        self.w_zz_l1_cap = 0.90
        self.step_field_scale = 0.03
        self.step_proj = nn.Linear(self.d_step, self.d_step)
        self.step_unary_head = nn.Linear(self.d_step, 1, bias=False)
        self.W_zy = nn.Parameter(torch.randn(self.d_z, self.d_step) * 0.06)
        self.W_zz_raw = nn.Parameter(torch.randn(self.d_z, self.d_z) * 0.03)
        self.W_edge = nn.Parameter(torch.randn(self.d_step, self.d_step) * 0.04)
        self.V_groups = nn.Parameter(torch.randn(self.n_groups, self.group_k, self.d_step) * 0.06)
        self.U_z = nn.Linear(self.d_z, self.n_groups)

        # Ablation Axis 2: Sectioned GAQSlotBridge + Risk Gate `(1 - g_risk)`
        if self.use_gaq_bridge or self.use_spd_laplacian:
            self.slot_gaq = SectionedGAQBridge(d_in=3 * self.d_model + 36, m_sections=4, d_code=16, num_codes=32, seed=31415)
            self.q_gaq = SectionedGAQBridge(d_in=self.d_model + 16, m_sections=4, d_code=16, num_codes=32, seed=27182)

        if self.use_gaq_bridge:
            self.gaq_unary_head = nn.Sequential(
                nn.LayerNorm(2 * D_GAQ_TOTAL),
                nn.Linear(2 * D_GAQ_TOTAL, 64),
                nn.SiLU(),
                nn.Linear(64, 9),
            )
            self.gaq_pair_head = nn.Sequential(
                nn.LayerNorm(3 * D_GAQ_TOTAL),
                nn.Linear(3 * D_GAQ_TOTAL, 64),
                nn.SiLU(),
                nn.Linear(64, 7),
            )
            for _out_l in (self.gaq_unary_head[-1], self.gaq_pair_head[-1]):
                nn.init.normal_(_out_l.weight, std=1e-3)
                nn.init.zeros_(_out_l.bias)
            self.risk_gamma_raw = nn.Parameter(torch.tensor(1.5))
            self.risk_tau = nn.Parameter(torch.tensor(0.45))

        # Ablation Axis 3: Absorbing Null Slot prior + RLNS coupling bias
        states = torch.arange(N_RLNS_STATES, dtype=torch.long)
        self.register_buffer("grid2", torch.cartesian_prod(states, states), persistent=False)
        self.register_buffer("grid3", torch.cartesian_prod(states, states, states), persistent=False)
        if self.use_rlns_and_null:
            self.null_prior_net = nn.Sequential(
                nn.Linear(D_NULL_INV, 24),
                nn.SiLU(),
                nn.Linear(24, 1),
            )
            nn.init.normal_(self.null_prior_net[-1].weight, std=1e-3)
            nn.init.zeros_(self.null_prior_net[-1].bias)

        # Ablation Axis 4: Input-dependent SPD Graph Laplacian + Cholesky Adjoint + Eikonal
        if self.use_spd_laplacian:
            self.spd_laplacian = SPDGraphLaplacianField(d_slot=D_GAQ_TOTAL, d_lap=D_LAP, d_psi_step=D_STEP_PSI)

    def set_woodbury_anchors(
        self,
        Wu0: np.ndarray,
        Wp0: np.ndarray,
        Wr0: np.ndarray,
        Wlm0: np.ndarray,
        tau_risk_init: float = 0.45,
    ) -> None:
        with torch.no_grad():
            for nm, arr in (("Wu0", Wu0), ("Wp0", Wp0), ("Wr0", Wr0), ("Wlm0", Wlm0)):
                getattr(self, nm).copy_(torch.from_numpy(arr).to(device=self.W_u.device, dtype=torch.float32))
            self.W_u.copy_(self.Wu0)
            self.W_p.copy_(self.Wp0)
            self.W_r.copy_(self.Wr0)
            self.W_lm.copy_(self.Wlm0)
            if self.use_gaq_bridge:
                self.tau_risk_floor.fill_(float(tau_risk_init))
                self.risk_tau.fill_(float(tau_risk_init))

    def w_meta_effective(self) -> torch.Tensor:
        return torch.where(self.wm_pen_mask, torch.maximum(self.w_meta, 0.40 * self.wm0), self.w_meta)

    def normalized_W_zz(self) -> torch.Tensor:
        sym = 0.5 * (self.W_zz_raw + self.W_zz_raw.t())
        eye = torch.eye(self.d_z, device=sym.device, dtype=sym.dtype)
        sym = sym * (1.0 - eye)
        row_l1 = sym.abs().sum(dim=-1, keepdim=True).clamp(min=1e-6)
        scale = torch.clamp(self.w_zz_l1_cap / row_l1, max=1.0)
        return sym * torch.minimum(scale, scale.t())

    def _bounded_couplings(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        W_zz = self.normalized_W_zz()
        eye = torch.eye(self.d_z, device=W_zz.device, dtype=W_zz.dtype)
        A = self.m_convex * eye - W_zz
        W_zy_b = self.W_zy / torch.clamp(self.W_zy.norm() / 1.4, min=1.0)
        W_e_raw = 0.5 * (self.W_edge + self.W_edge.t())
        W_e_b = W_e_raw / torch.clamp(W_e_raw.norm() / 1.2, min=1.0)
        V_g_b = self.V_groups / torch.clamp(self.V_groups.norm() / 1.4, min=1.0)
        M_z = torch.linalg.solve(A, W_zy_b)
        return M_z, W_zy_b, W_e_b, V_g_b

    def _score_psi_batched(
        self,
        psi_pad: torch.Tensor,
        step_mask: torch.Tensor,
        seq_lens: torch.Tensor,
        couplings: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
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

    def encode_gaq_and_unary(
        self,
        all_uf: torch.Tensor,
        all_lm: torch.Tensor,
        all_rf: torch.Tensor,
        all_hq: torch.Tensor,
        slot_prob_idx: torch.Tensor,
        prob_u_risk: torch.Tensor,
        all_nfeat: Optional[torch.Tensor] = None,
    ) -> UEBREncodeOut:
        """Computes slot & question GAQ codes, blocked linear `W_lm` residuals, and unary role potentials `(N_slots, 9)`."""
        u_lex = 2.5 * F.linear(all_uf, self.W_u)
        u_lm_raw = 1.0 * F.linear(all_lm, self.W_lm)
        u_sp = self.sparse_role(Sparsemax.apply(self.sparse_gate(self.lm_norm(all_lm))))

        if self.use_gaq_bridge or self.use_spd_laplacian:
            # Mask non-invariant positional ratios (cols 14:17 of tail 36) before GAQ bridge
            uf_gaq = all_uf[:, -36:].clone()
            uf_gaq[:, 14:17] = 0.0
            slot_in = torch.cat([all_lm, uf_gaq], dim=-1)
            q_slots, vq_s = self.slot_gaq(slot_in)
            q_in = torch.cat([all_hq, all_rf[:, -16:]], dim=-1)
            q_prob, vq_q = self.q_gaq(q_in)
            vq_loss = vq_s + vq_q
        else:
            q_slots = torch.zeros((all_uf.size(0), D_GAQ_TOTAL), device=all_uf.device, dtype=all_uf.dtype)
            q_prob = torch.zeros((all_rf.size(0), D_GAQ_TOTAL), device=all_rf.device, dtype=all_rf.dtype)
            vq_loss = torch.zeros((), device=all_uf.device, dtype=all_uf.dtype)

        if self.use_gaq_bridge:
            gamma = F.softplus(self.risk_gamma_raw).clamp(min=0.1, max=8.0)
            tau_eff = torch.maximum(self.risk_tau, self.tau_risk_floor - 0.05)
            g_prob = 0.50 * torch.sigmoid(gamma * (prob_u_risk - tau_eff))
            g_slot = g_prob[slot_prob_idx].unsqueeze(-1)
            q_prob_s = q_prob[slot_prob_idx]
            u_gaq = self.gaq_unary_head(torch.cat([q_slots, q_prob_s], dim=-1))
            u_tot = u_lex + (1.0 - g_slot) * u_lm_raw + u_sp + u_gaq
        else:
            g_prob = torch.zeros_like(prob_u_risk)
            u_tot = u_lex + u_lm_raw + u_sp

        if self.use_rlns_and_null:
            if all_nfeat is None:
                uf_tail = all_uf[:, DU_HASH - 36 : DU_HASH - 9].clone()
                uf_tail[:, 14:17] = 0.0
                nfeat_in = F.pad(uf_tail, (0, D_NULL_INV - 27))
            else:
                nfeat_in = all_nfeat
            null_b = self.null_prior_net(nfeat_in).squeeze(-1)
            u_tot = u_tot.clone()
            u_tot[:, 0] = u_tot[:, 0] - 0.15 * null_b

        u_tot_rlns = u_lex.detach() + (u_tot - u_lex)
        return UEBREncodeOut(
            u_tot=u_tot,
            u_tot_rlns=u_tot_rlns,
            u_lex=u_lex,
            q_slots=q_slots,
            q_prob=q_prob,
            vq_loss=vq_loss,
            g_prob=g_prob,
        )

    def rlns_block3_fenchel_young_loss(
        self,
        u_slots: torch.Tensor,
        p_pot: torch.Tensor,
        slot_start: int,
        k_sl: int,
        pair_map_arr: np.ndarray,
        rl_true: torch.Tensor,
        gamma_fy: float = 0.70,
        pair_iu_t: Optional[torch.Tensor] = None,
        pair_iv_t: Optional[torch.Tensor] = None,
        pair_pidx_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Vectorized 0-BPTT coupled 3-slot (or 2-slot) RLNS-1 Fenchel-Young loss with zero `.item()` syncs."""
        dev = u_slots.device
        u_p = u_slots[slot_start : slot_start + k_sl]
        if k_sl < 2 or p_pot.size(0) == 0:
            return torch.zeros((), device=dev, dtype=u_slots.dtype)

        if pair_iu_t is None or pair_iv_t is None or pair_pidx_t is None:
            iu_np, iv_np = np.triu_indices(k_sl, k=1)
            pair_iu_t = torch.as_tensor(iu_np, dtype=torch.long, device=dev)
            pair_iv_t = torch.as_tensor(iv_np, dtype=torch.long, device=dev)
            pair_pidx_t = torch.as_tensor(pair_map_arr[iu_np, iv_np], dtype=torch.long, device=dev)

        pp = p_pot[pair_pidx_t]
        oc_best = torch.argmin(pp, dim=-1)
        p_mean = pp.mean(dim=-1)
        e_edges = self.K_cram[:, :, oc_best].permute(2, 0, 1) + 0.35 * p_mean.view(-1, 1, 1)

        pair_mat = torch.zeros((k_sl, k_sl, N_RLNS_STATES, N_RLNS_STATES), device=dev, dtype=u_slots.dtype)
        pair_mat[pair_iu_t, pair_iv_t] = e_edges
        pair_mat[pair_iv_t, pair_iu_t] = e_edges.transpose(-1, -2)

        if k_sl == 2:
            g0 = self.grid2[:, 0]
            g1 = self.grid2[:, 1]
            e_all = u_p[0, g0] + u_p[1, g1] + pair_mat[0, 1, g0, g1]
            true_idx = rl_true[0] * N_RLNS_STATES + rl_true[1]
        else:
            aff_e = pp.abs().max(dim=-1).values
            best_e = torch.argmax(aff_e)
            b0 = pair_iu_t[best_e]
            b1 = pair_iv_t[best_e]
            aff_mat = torch.zeros((k_sl, k_sl), device=dev, dtype=u_slots.dtype)
            aff_mat[pair_iu_t, pair_iv_t] = aff_e
            aff_mat[pair_iv_t, pair_iu_t] = aff_e
            ar = torch.arange(k_sl, device=dev)
            cand_sc = aff_mat[b0] + aff_mat[b1] + 2.0 * (rl_true == 0).to(u_slots.dtype)
            cand_sc = torch.where((ar == b0) | (ar == b1), torch.full_like(cand_sc, -1e9), cand_sc)
            b2 = torch.argmax(cand_sc)
            blk = torch.stack([b0, b1, b2])
            in_blk = (ar.unsqueeze(0) == blk.unsqueeze(1)).any(dim=0)
            rest_mask = (~in_blk).to(u_slots.dtype)
            rl_exp = rl_true.view(1, k_sl, 1, 1).expand(3, k_sl, N_RLNS_STATES, 1)
            ext_all = torch.gather(pair_mat[blk], 3, rl_exp).squeeze(-1)
            ext_u = u_p[blk] + (ext_all * rest_mask.view(1, k_sl, 1)).sum(dim=1)
            g0 = self.grid3[:, 0]
            g1 = self.grid3[:, 1]
            g2 = self.grid3[:, 2]
            e_all = (
                ext_u[0, g0]
                + ext_u[1, g1]
                + ext_u[2, g2]
                + pair_mat[b0, b1, g0, g1]
                + pair_mat[b0, b2, g0, g2]
                + pair_mat[b1, b2, g1, g2]
            )
            true_idx = rl_true[b0] * 81 + rl_true[b1] * 9 + rl_true[b2]

        logits = -e_all / gamma_fy
        return gamma_fy * (torch.logsumexp(logits, dim=0) - logits[true_idx])

    def score_batch(
        self,
        all_uf: torch.Tensor,
        all_pf: torch.Tensor,
        all_rf: torch.Tensor,
        all_lm: torch.Tensor,
        all_hq: torch.Tensor,
        slot_prob_idx: torch.Tensor,
        prob_u_risk: torch.Tensor,
        slot_offsets: Sequence[int],
        slot_counts: Sequence[int],
        slot_gidx: torch.Tensor,
        slot_roles: torch.Tensor,
        slot_cand_idx: torch.Tensor,
        slot_inv_k: torch.Tensor,
        root_prob_idx: torch.Tensor,
        root_op: torch.Tensor,
        bm_cand: torch.Tensor,
        bp_pair: torch.Tensor,
        bp_u_gidx: torch.Tensor,
        bp_v_gidx: torch.Tensor,
        bo_op: torch.Tensor,
        bra: torch.Tensor,
        brb: torch.Tensor,
        bwt: torch.Tensor,
        psi_raw_pad: torch.Tensor,
        psi_mask: torch.Tensor,
        psi_lens: torch.Tensor,
        meta_m: torch.Tensor,
        cand_prob_idx: torch.Tensor,
        cand_local_idx: torch.Tensor,
        m_max: int,
        dag_op: torch.Tensor,
        dag_a: torch.Tensor,
        dag_b: torch.Tensor,
        dag_len: torch.Tensor,
        all_nfeat: Optional[torch.Tensor] = None,
        n_cands: Optional[int] = None,
    ) -> UEBRBatchOut:
        """Scores a dense batch of open DAG candidates (`n_cands`) cleanly across enabled UEBR axes."""
        dev = all_uf.device
        n_c = int(meta_m.size(0)) if n_cands is None else int(n_cands)
        enc = self.encode_gaq_and_unary(
            all_uf, all_lm, all_rf, all_hq, slot_prob_idx, prob_u_risk, all_nfeat=all_nfeat
        )
        u_tot = enc.u_tot
        u_lex = enc.u_lex
        q_slots = enc.q_slots
        q_prob = enc.q_prob

        u_entries = u_tot[slot_gidx, slot_roles] * slot_inv_k
        u_ablate = u_lex[slot_gidx, slot_roles] * slot_inv_k
        eu_full = torch.zeros(n_c, dtype=torch.float32, device=dev).index_add_(0, slot_cand_idx, u_entries)
        eu_ablate = torch.zeros(n_c, dtype=torch.float32, device=dev).index_add_(0, slot_cand_idx, u_ablate)

        r_pot = 1.5 * F.linear(all_rf, self.W_r)
        er = r_pot[root_prob_idx, root_op]

        p_pot = 2.5 * F.linear(all_pf, self.W_p)
        if bm_cand.numel() > 0:
            edge_base = p_pot[bp_pair, bo_op] + self.K_cram[bra, brb, bo_op]
            if self.use_gaq_bridge:
                bp_prob = slot_prob_idx[bp_u_gidx]
                gaq_pair_in = torch.cat([q_slots[bp_u_gidx], q_slots[bp_v_gidx], q_prob[bp_prob]], dim=-1)
                gaq_pair_logits = self.gaq_pair_head(gaq_pair_in)
                edge_base = edge_base + torch.gather(gaq_pair_logits, 1, bo_op.unsqueeze(-1)).squeeze(-1)
            ep = torch.zeros(n_c, dtype=torch.float32, device=dev).index_add_(0, bm_cand, edge_base * bwt)
        else:
            ep = torch.zeros(n_c, dtype=torch.float32, device=dev)

        couplings = self._bounded_couplings()
        psi_seq = torch.tanh(self.step_proj(psi_raw_pad))
        ez = self.step_field_scale * self._score_psi_batched(psi_seq, psi_mask, psi_lens, couplings)
        em = meta_m @ self.w_meta_effective()

        p_anchor = 2.5 * F.linear(all_pf, self.Wp0)
        p_pot_rlns = p_anchor.detach() + 0.25 * (p_pot - p_anchor)

        e_tot = eu_full + ep + er + ez + em
        d_psi = torch.zeros(n_c, dtype=torch.float32, device=dev)
        z_flat: Optional[torch.Tensor] = None
        H_psi: Optional[torch.Tensor] = None
        S_pad: Optional[torch.Tensor] = None
        Z_pad: Optional[torch.Tensor] = None

        if self.use_spd_laplacian:
            bsz = len(slot_offsets)
            q_pad = torch.zeros((bsz, K_SLOT_BUF, D_GAQ_TOTAL), device=dev, dtype=torch.float32)
            s_mask = torch.zeros((bsz, K_SLOT_BUF), device=dev, dtype=torch.bool)
            for bi, (so, sc) in enumerate(zip(slot_offsets, slot_counts)):
                uc = min(sc, K_SLOT_BUF)
                if uc > 0:
                    q_pad[bi, :uc] = q_slots[so : so + uc]
                    s_mask[bi, :uc] = True
            e_lap, d_psi, z_flat, H_psi, S_pad, Z_pad = self.spd_laplacian.score_batch(
                q_slots_pad=q_pad,
                slot_mask=s_mask,
                q_prob=q_prob,
                cand_prob_idx=cand_prob_idx,
                cand_local_idx=cand_local_idx,
                m_max=max(1, int(m_max)),
                dag_op=dag_op,
                dag_a=dag_a,
                dag_b=dag_b,
                dag_len=dag_len,
                psi_raw_pad=psi_raw_pad,
            )
            e_tot = e_tot + e_lap

        return UEBRBatchOut(
            e_tot=e_tot,
            e_ablate=eu_ablate + er,
            u_tot=u_tot,
            u_tot_rlns=enc.u_tot_rlns,
            p_pot=p_pot,
            p_pot_rlns=p_pot_rlns,
            q_slots=q_slots,
            q_prob=q_prob,
            vq_loss=enc.vq_loss,
            g_prob=enc.g_prob,
            d_psi=d_psi,
            z_flat=z_flat,
            H_psi=H_psi,
            S_pad=S_pad,
            Z_pad=Z_pad,
        )

    def prox_regularization_loss(self) -> torch.Tensor:
        loss = (
            3.5e-3
            * (
                (self.W_u - self.Wu0).square().sum()
                + (self.W_p - self.Wp0).square().sum()
                + (self.W_r - self.Wr0).square().sum()
                + (self.W_lm - self.Wlm0).square().sum()
            )
            + 8.0e-3 * (self.w_meta - self.wm0).square().sum()
            + 1.5e-4 * self.K_cram.square().sum()
            + 1.5e-3 * self.sparse_role.weight.square().sum()
            + 4.0e-4 * (self.sparse_gate.weight.square().sum() + self.lm_norm.weight.square().sum())
            + 2.5e-4 * (
                self.step_proj.weight.square().sum()
                + self.step_unary_head.weight.square().sum()
                + self.W_zy.square().sum()
                + self.W_edge.square().sum()
                + self.V_groups.square().sum()
                + self.U_z.weight.square().sum()
            )
        )
        return loss


# ===========================================================================
# 4. UEBROpenGraphSolver (Full Pipeline over Frozen Tier-1 LUT Features)
# ===========================================================================


class UEBROpenGraphSolver(OpenGraphCRAMSolver):
    """Calcite-6 (`UEBR`) System-2 solver supporting full pipeline + clean axis ablations."""

    def __init__(
        self,
        extractor: TrainInducedQuantityExtractor,
        alu: SphericalBase10RegisterALU,
        d_model: int = 384,
        device: str = "cuda",
        seed: int = 2026,
        use_birkhoff_qap: bool = True,
        use_rlns_and_null: bool = True,
        use_gaq_bridge: bool = True,
        use_spd_laplacian: bool = True,
    ) -> None:
        super().__init__(extractor=extractor, alu=alu, d_model=d_model, device=device, seed=seed)
        self.use_birkhoff_qap = bool(use_birkhoff_qap)
        self.use_rlns_and_null = bool(use_rlns_and_null)
        self.use_gaq_bridge = bool(use_gaq_bridge)
        self.use_spd_laplacian = bool(use_spd_laplacian)

        self.ebm = UEBROpenGraphEBM(
            d_model=self.d_model,
            use_gaq_bridge=self.use_gaq_bridge,
            use_spd_laplacian=self.use_spd_laplacian,
            use_rlns_and_null=self.use_rlns_and_null,
        ).to(self.device)

        # Runtime Phase I / J telemetry counters & (N_CERT_FEATS + 1) cert buffer for roundtrip
        self.cert_weights = np.zeros(N_CERT_FEATS + 1, dtype=np.float64)
        self.birkhoff_total_solves: int = 0
        self.birkhoff_lossless_wins: int = 0
        self.last_epc_err: float = 0.0
        self.last_min_eig: float = 1.0

    def state_dict(self) -> Dict[str, object]:
        base = super().state_dict()
        base.update(
            {
                "uebr_flags": {
                    "use_birkhoff_qap": self.use_birkhoff_qap,
                    "use_rlns_and_null": self.use_rlns_and_null,
                    "use_gaq_bridge": self.use_gaq_bridge,
                    "use_spd_laplacian": self.use_spd_laplacian,
                },
                "birkhoff_total_solves": int(self.birkhoff_total_solves),
                "birkhoff_lossless_wins": int(self.birkhoff_lossless_wins),
                "last_epc_err": float(self.last_epc_err),
                "last_min_eig": float(self.last_min_eig),
            }
        )
        return base

    def load_state_dict(self, sd: Dict[str, object]) -> None:
        """Restores `UEBROpenGraphSolver`, preserving `(N_CERT_FEATS + 1)` `cert_weights` and UEBR counters."""
        super().load_state_dict(sd)
        if "cert_weights" in sd:
            cw = np.asarray(sd["cert_weights"], dtype=np.float64).copy()
            if cw.ndim == 1 and cw.shape[0] in (N_CERT_FEATS, N_CERT_FEATS + 1):
                self.cert_weights = cw
        if "birkhoff_total_solves" in sd:
            self.birkhoff_total_solves = int(sd["birkhoff_total_solves"])  # type: ignore[arg-type]
        if "birkhoff_lossless_wins" in sd:
            self.birkhoff_lossless_wins = int(sd["birkhoff_lossless_wins"])  # type: ignore[arg-type]
        if "last_epc_err" in sd:
            self.last_epc_err = float(sd["last_epc_err"])  # type: ignore[arg-type]
        if "last_min_eig" in sd:
            self.last_min_eig = float(sd["last_min_eig"])  # type: ignore[arg-type]

    def _featurize_problem(
        self,
        q_text: str,
        slots: Sequence[ExtractedSlot],
        lm_feat: Optional[Dict[str, object]],
    ) -> Dict[str, object]:
        pf = super()._featurize_problem(q_text, slots, lm_feat)
        k = len(slots)
        if lm_feat is not None:
            hq = np.asarray(lm_feat.get("h_q", np.zeros((self.d_model,), dtype=np.float32)), dtype=np.float32)
            u_risk = float(lm_feat.get("u_risk", pf["up"]))  # type: ignore[arg-type]
        else:
            hq = np.zeros((self.d_model,), dtype=np.float32)
            u_risk = float(pf["up"])  # type: ignore[arg-type]
        hq_u = hq / (float(np.linalg.norm(hq)) + 1e-6)
        pf["hq_u"] = hq_u.astype(np.float32)
        pf["u_risk"] = u_risk
        pf["k"] = k
        pf["null_inv"] = _compute_null_slot_features(
            q_text=q_text,
            slots=slots,
            pf=pf,
            extractor=self.extractor,
            vocab_counts=self.vocab_counts,
            stop_words=self.stop_words,
            d_model=self.d_model,
        )
        return pf

    @torch.no_grad()
    def _infer_slot_null_marginals(
        self,
        slots: Sequence[ExtractedSlot],
        pf: Dict[str, object],
        u_pot: np.ndarray,
        p_pot: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Runs 2-Opt Block-Gibbs discrete equilibrium over `(0=Ø, 1..8)` roles -> `(best_roles, null_probs)`."""
        k = len(slots)
        if k == 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float64)

        pmap: Dict[Tuple[int, int], int] = pf["pm"]  # type: ignore[assignment]
        K_np = self.ebm.K_cram.detach().cpu().numpy()
        p_mat = np.zeros((k, k, N_RLNS_STATES, N_RLNS_STATES), dtype=np.float64)
        edges: List[Tuple[int, int]] = []
        for i in range(k):
            for j in range(i + 1, k):
                pp = p_pot[pmap[(i, j)]]
                oc_best = int(np.argmin(pp))
                e_ij = K_np[:, :, oc_best] + 0.35 * float(np.mean(pp))
                p_mat[i, j] = e_ij
                p_mat[j, i] = e_ij.T
                edges.append((i, j))

        u_d = u_pot.astype(np.float64)
        r_cur = np.argmin(u_d, axis=-1).astype(np.int64)
        # 2 sweeps of 2-Opt Block-Gibbs over slot pairs + 1-flip unary relaxation
        for _ in range(2):
            for i in range(k):
                e_i = u_d[i].copy()
                for j in range(k):
                    if j != i:
                        e_i += p_mat[i, j, :, r_cur[j]]
                r_cur[i] = int(np.argmin(e_i))
            if k >= 2:
                for u_e, v_e in edges[: min(len(edges), 12)]:
                    eu = u_d[u_e].copy()
                    ev = u_d[v_e].copy()
                    for w_n in range(k):
                        if w_n != u_e and w_n != v_e:
                            eu += p_mat[u_e, w_n, :, r_cur[w_n]]
                            ev += p_mat[v_e, w_n, :, r_cur[w_n]]
                    grid = eu[:, None] + ev[None, :] + p_mat[u_e, v_e]
                    flat_i = int(np.argmin(grid))
                    r_cur[u_e] = flat_i // N_RLNS_STATES
                    r_cur[v_e] = flat_i % N_RLNS_STATES

        null_probs = np.zeros((k,), dtype=np.float64)
        for i in range(k):
            e_i = u_d[i].copy()
            for j in range(k):
                if j != i:
                    e_i += p_mat[i, j, :, r_cur[j]]
            logits = -(e_i - float(np.min(e_i))) / 0.85
            probs = np.exp(np.clip(logits, -30.0, 0.0))
            probs /= max(float(np.sum(probs)), 1e-12)
            null_probs[i] = float(probs[0])
        return r_cur, null_probs

    def _synthesize_open_pool(
        self,
        slots: Sequence[ExtractedSlot],
        pf: Dict[str, object],
        mask_self: bool = False,
        use_anchor_only: bool = False,
    ) -> List[List[object]]:
        """Open candidate synthesis with RLNS Null-Slot `Ø` multi-distractor unlock & Birkhoff Extension QAP."""
        k = len(slots)
        if k <= 1:
            return []

        vals: List[float] = pf["vs"]  # type: ignore[assignment]
        Uf_t = torch.from_numpy(pf["Uf"]).to(self.device)  # type: ignore[arg-type]
        Nf_t = torch.from_numpy(pf["null_inv"]).to(self.device)  # type: ignore[arg-type]
        Pf_t = torch.from_numpy(pf["Pf"]).to(self.device)  # type: ignore[arg-type]
        Rf_t = torch.from_numpy(pf["Rf"]).unsqueeze(0).to(self.device)  # type: ignore[arg-type]
        lm_t = torch.from_numpy(pf["lm"]).to(self.device)  # type: ignore[arg-type]
        hq_t = torch.from_numpy(pf["hq_u"]).unsqueeze(0).to(self.device)  # type: ignore[arg-type]
        s_pidx = torch.zeros(k, dtype=torch.long, device=self.device)
        u_r_t = torch.tensor([float(pf["u_risk"])], dtype=torch.float32, device=self.device)  # type: ignore[arg-type]

        with torch.no_grad():
            u_anchor = (
                2.5 * F.linear(Uf_t, self.ebm.Wu0)
                + 1.0 * F.linear(lm_t, self.ebm.Wlm0)
            ).cpu().numpy()
            p_anchor = (2.5 * F.linear(Pf_t, self.ebm.Wp0)).cpu().numpy()
            if use_anchor_only:
                u_pot = u_anchor
                p_pot = p_anchor
                if self.use_rlns_and_null and k > 0:
                    u_sh = u_anchor - np.min(u_anchor, axis=-1, keepdims=True)
                    p_anc = np.exp(-np.clip(u_sh / 0.85, 0.0, 30.0))
                    p_anc /= np.maximum(p_anc.sum(axis=-1, keepdims=True), 1e-12)
                    null_probs = p_anc[:, 0].astype(np.float64)
                else:
                    null_probs = np.zeros((k,), dtype=np.float64)
            else:
                enc = self.ebm.encode_gaq_and_unary(
                    Uf_t, lm_t, Rf_t, hq_t, s_pidx, u_r_t, all_nfeat=Nf_t
                )
                u_pot = 0.55 * u_anchor + 0.45 * enc.u_tot.cpu().numpy()
                p_pot = 0.65 * p_anchor + 0.35 * (2.5 * F.linear(Pf_t, self.ebm.W_p)).cpu().numpy()
                if self.use_rlns_and_null:
                    _, null_probs = self._infer_slot_null_marginals(slots, pf, u_pot, p_pot)
                else:
                    null_probs = np.zeros((k,), dtype=np.float64)

        pf["null_probs"] = null_probs
        gr = [i for i, s in enumerate(slots) if s.kind != "CONST"] or list(range(k))
        cn = sorted([i for i, s in enumerate(slots) if s.kind == "CONST"], key=lambda i: u_pot[i, 1:].min())[:2]
        ng = len(gr)
        gr_set = set(gr)
        pmap: Dict[Tuple[int, int], int] = pf["pm"]  # type: ignore[assignment]

        if self.use_rlns_and_null:
            eff_min_ug = 1 if ng <= 2 else 2
        else:
            min_ug = max(1, ng - 1 if ng <= 5 else ng - 2)
            eff_min_ug = max(2, min(5, ng - 3)) if ng >= 8 else min_ug

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

        def _run_enum_small(act_list: Sequence[int]) -> None:
            for idx_a in range(len(act_list)):
                i = act_list[idx_a]
                for idx_b in range(idx_a, len(act_list)):
                    j = act_list[idx_b]
                    for o0 in range(4):
                        if i == j and o0 not in (0, 2):
                            continue
                        firsts = [(o0, i, j)] + ([(o0, j, i)] if o0 in (1, 3) and i != j else [])
                        for s0 in firsts:
                            _add_dag([s0], is_bm=1.0)
                            for c in act_list:
                                for o1 in range(4):
                                    sa = min(c, k) if o1 in (0, 2) else k
                                    sb = max(c, k) if o1 in (0, 2) else c
                                    _add_dag([s0, (o1, sa, sb)], is_bm=1.0)
                                    if o1 in (1, 3):
                                        _add_dag([s0, (o1, c, k)], is_bm=1.0)

        def _run_beam_forest(seed_subsets: Sequence[Tuple[List[int], float]], allow_cn: bool = True) -> None:
            seeds_b = []
            for sub_l, base_sc in seed_subsets:
                if len(sub_l) >= 2:
                    seeds_b.append((tuple((i, vals[i], i, (i,), False) for i in sub_l), (), base_sc))
                if allow_cn and cn and 2 <= len(sub_l) <= 5:
                    seeds_b.append((tuple((i, vals[i], i, (i,), False) for i in (list(sub_l) + [cn[0]])), (), base_sc + 0.45))
            if not seeds_b:
                return
            beam = seeds_b[:8]
            done_f: List[Tuple[float, Tuple[Tuple[int, int, int], ...]]] = []
            for _ in range(min(MAX_GRAPH_STEPS, 7)):
                nxt = []
                for ws, d, sc in beam:
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
                                    done_f.append((sc + be, nd_d))
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
            done_f.sort(key=lambda x: x[0])
            for _, d_fin in done_f[:24]:
                _add_dag(d_fin, is_bm=1.0)

        # Rank non-constant slots by non-null active evidence (1 - null_prob) + active potential
        rel_score = {i: (1.0 - float(null_probs[i])) + 0.20 * float(u_pot[i, 0] - np.min(u_pot[i, 1:])) for i in gr}
        gr_ranked = sorted(gr, key=lambda i: (rel_score[i], -i), reverse=True)

        if ng <= 3:
            _run_enum_small(gr + cn)
        elif ng <= 7:
            # RLNS Null-Slot Multi-Distractor Unlock: also include small subsets & drop-1..ng-2!
            seeds_sub: List[Tuple[List[int], float]] = [(list(gr), 0.0)]
            if self.use_rlns_and_null:
                _run_enum_small(gr_ranked[: min(3, ng)] + cn[:1])
                for m_sub in range(max(2, ng - 3), ng):
                    seeds_sub.append((sorted(gr_ranked[:m_sub]), 0.10 * (ng - m_sub)))
                if ng <= 6:
                    di_min = min(gr, key=lambda i: rel_score[i])
                    seeds_sub.append(([i for i in gr if i != di_min], 0.15))
            else:
                di = min(gr, key=lambda i: u_pot[i, 0])
                seeds_sub.append(([i for i in gr if i != di], float(u_pot[di, 0]) + 0.4))
            _run_beam_forest(seeds_sub, allow_cn=(ng <= 6))
        else:
            sal_map = {i: float(rel_score[i]) for i in gr}
            gr_by_sal = sorted(gr, key=lambda i: (sal_map[i], i), reverse=True)
            core_subsets: List[Tuple[List[int], float]] = []
            seen_cores: Set[Tuple[int, ...]] = set()
            for k_sub in (2, 3, 4, 5, 6, 7, 8, min(10, ng)):
                if k_sub <= ng:
                    sub_s = tuple(sorted(gr_by_sal[:k_sub]))
                    if sub_s not in seen_cores:
                        seen_cores.add(sub_s)
                        core_subsets.append((list(sub_s), 0.0))
                    if k_sub <= 7:
                        sub_p = tuple(gr[:k_sub])
                        sub_t = tuple(gr[-k_sub:])
                        for sub_cand, pen in ((sub_p, 0.12), (sub_t, 0.15)):
                            if sub_cand not in seen_cores:
                                seen_cores.add(sub_cand)
                                core_subsets.append((list(sub_cand), pen))
            if self.use_rlns_and_null:
                _run_enum_small(gr_by_sal[:3] + cn[:1])
            _run_beam_forest([cs for cs in core_subsets if 2 <= len(cs[0]) <= 8], allow_cn=False)

            # Fast greedy Iterative Register Folding fallback
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

        # Episodic Precedent Lifting: PermFlow + Birkhoff Extension QAP (when enabled) vs Chronological baseline
        buckets: List[Tuple[Tuple[int, int], int, float]] = [((ng, 0), 80, 1.0)]
        if 2 <= ng <= 7:
            for m_sub in range(max(2, ng - 2 if self.use_rlns_and_null else ng - 1), ng):
                buckets.append(((m_sub, 0), 40, 0.55 if m_sub == ng - 1 else 0.40))
            if cn:
                buckets.append(((ng, 1), 40, 0.65))
                if ng <= 6:
                    buckets.append(((ng - 1, 1), 24, 0.35))
        elif ng >= 8:
            for k_sub in (8, 7, 6, 5, 4, 3, 2):
                if k_sub < ng:
                    buckets.append(((k_sub, 0), 32, 0.50))
                    if cn and k_sub >= 4:
                        buckets.append(((k_sub, 1), 20, 0.35))

        q_ctx: np.ndarray = pf["ctx"]  # type: ignore[assignment]
        qap_cache: Dict[Tuple[int, Tuple[int, ...], int], List[List[int]]] = {}

        # Precompute directional pair cost matrices over gr once when Birkhoff QAP is active
        pair_bank_gr: Dict[int, np.ndarray] = {}
        rv_op_map = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4, 5: 6, 6: 5}
        if self.use_birkhoff_qap and 2 <= ng <= 10:
            for oc in range(7):
                B_oc = np.zeros((ng, ng), dtype=np.float64)
                for ii in range(ng):
                    for jj in range(ng):
                        if ii == jj:
                            B_oc[ii, jj] = 4.0
                        else:
                            gi, gj = gr[ii], gr[jj]
                            if gi < gj:
                                B_oc[ii, jj] = float(p_pot[pmap[(gi, gj)], oc])
                            else:
                                B_oc[ii, jj] = float(p_pot[pmap[(gj, gi)], rv_op_map[oc]])
                pair_bank_gr[oc] = B_oc

        for kb, top_n, b_scale in buckets:
            if kb not in self.ep_vecs or not self.ep_items.get(kb):
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
            m_g, m_c = kb

            # Fallback subset list for baseline chronological lifting
            if not self.use_birkhoff_qap or ng > 10:
                if m_g == ng:
                    mps_raw = [gr] if m_c == 0 else [gr + [ci] for ci in cn[:1]]
                elif ng <= 7:
                    base_g = [list(c) for c in itertools.combinations(gr, m_g)]
                    mps_raw = base_g if m_c == 0 else [c + [cn[0]] for c in base_g if cn]
                else:
                    base_g = [sorted(gr_by_sal[:m_g]), gr[-m_g:]]
                    mps_raw = base_g if m_c == 0 else [c + [cn[0]] for c in base_g if cn]
            else:
                mps_raw = []

            for nb_rank, (nbi, w_nb) in enumerate(zip(tnb, kw_arr)):
                sk, src_rl = items_b[int(nbi)]
                m_tot = m_g + m_c
                weighted_mps: List[Tuple[List[int], float]] = []

                if self.use_birkhoff_qap and 2 <= ng <= 10 and m_g <= ng and (m_c == 0 or cn):
                    mp_chron = list(gr[:m_g]) + ([cn[0]] if m_c > 0 and cn else [])
                    base_wt = float(w_nb) * b_scale
                    if m_g == ng and ng <= 3:
                        if len(mp_chron) == m_tot:
                            weighted_mps.append((mp_chron, base_wt))
                        run_qap = nb_rank < 12
                    else:
                        run_qap = True

                    if run_qap:
                        src_g_tup = tuple(int(x) for x in src_rl[:m_g])
                        root_op_sk = int(sk[-1][0]) if sk else 0
                        cache_key = (m_g, src_g_tup, root_op_sk)
                        if cache_key in qap_cache:
                            row_assigns = qap_cache[cache_key]
                        else:
                            C_lin = np.zeros((ng, ng), dtype=np.float64)
                            for ii, gi in enumerate(gr):
                                for j in range(ng):
                                    if j < m_g:
                                        C_lin[ii, j] = float(u_pot[gi, int(src_g_tup[j])]) + 0.06 * float(abs(ii - j))
                                    else:
                                        # Absorbing Null Slot Ø cost: reward dropping distractors!
                                        C_lin[ii, j] = float(u_pot[gi, 0]) - 0.35 * float(null_probs[gi])
                            sk_edges: List[Tuple[int, int, int]] = []
                            sk_mats: List[np.ndarray] = []
                            for op_s, a_s, b_s in sk:
                                if a_s < m_g and b_s < m_g and a_s != b_s:
                                    oc = 0 if op_s == 0 else (1 if op_s == 1 else (3 if op_s == 2 else 5))
                                    sk_edges.append((a_s, b_s, oc))
                                    sk_mats.append(pair_bank_gr[oc])
                                if len(sk_edges) >= 3:
                                    break
                            row_assigns, lossless_ok = solve_qap_permflow_birkhoff(
                                C_lin=C_lin,
                                edges=sk_edges,
                                pair_mats=sk_mats,
                                lam_qap=0.30,
                                steps=6,
                            )
                            self.birkhoff_total_solves += 1
                            self.birkhoff_lossless_wins += int(lossless_ok)
                            qap_cache[cache_key] = row_assigns

                        r0 = row_assigns[0]
                        mp_qap = [gr[r0[j]] for j in range(m_g)] + ([cn[0]] if m_c > 0 and cn else [])
                        if m_g == ng and ng <= 3:
                            if mp_qap != mp_chron and len(mp_qap) == m_tot:
                                weighted_mps.append((mp_qap, 0.20 * base_wt))
                        elif len(mp_qap) == m_tot:
                            if mp_qap == mp_chron:
                                weighted_mps.append((mp_qap, base_wt))
                            elif m_g < ng:
                                weighted_mps.append((mp_qap, 0.85 * base_wt))
                                if len(mp_chron) == m_tot:
                                    weighted_mps.append((mp_chron, 0.15 * base_wt))
                            else:
                                if len(mp_chron) == m_tot:
                                    weighted_mps.append((mp_chron, 0.55 * base_wt))
                                weighted_mps.append((mp_qap, 0.55 * base_wt))
                else:
                    if len(mps_raw) <= 1:
                        best_mps = list(mps_raw)
                    else:
                        scored = sorted((sum(float(u_pot[idx, r]) for idx, r in zip(mp, src_rl)), mp) for mp in mps_raw)
                        best_mps = [x[1] for x in scored[:2]]
                    if best_mps:
                        wt_e = float(w_nb) * b_scale / float(len(best_mps))
                        weighted_mps = [(mp, wt_e) for mp in best_mps]

                for mp, wt_mp in weighted_mps:
                    lifted = []
                    valid_sk = True
                    for o, a, b in sk:
                        ra = mp[a] if a < 100 else k + (a - 100)
                        rb = mp[b] if b < 100 else k + (b - 100)
                        if ra >= k + len(lifted) or rb >= k + len(lifted):
                            valid_sk = False
                            break
                        sa, sb = (rb, ra) if (o in (0, 2) and ra > rb) else (ra, rb)
                        lifted.append((o, sa, sb))
                    if valid_sk and lifted:
                        _add_dag(lifted, kw=wt_mp)

        return list(by_sg.values())

    def _pack_candidates_cpu(
        self,
        slots: Sequence[ExtractedSlot],
        pf: Dict[str, object],
        cands: Sequence[List[object]],
    ) -> Dict[str, object]:
        pk = super()._pack_candidates_cpu(slots, pf, cands)
        k = len(slots)
        n_c = len(cands)
        gr = [i for i, s in enumerate(slots) if s.kind != "CONST"] or list(range(k))
        gr_set = set(gr)
        ng = len(gr)
        null_probs = pf.get("null_probs", np.zeros((k,), dtype=np.float64))

        dag_op = np.zeros((n_c, MAX_GRAPH_STEPS), dtype=np.int64)
        dag_a = np.zeros((n_c, MAX_GRAPH_STEPS), dtype=np.int64)
        dag_b = np.zeros((n_c, MAX_GRAPH_STEPS), dtype=np.int64)
        dag_len = np.ones((n_c,), dtype=np.int64)

        meta_m: np.ndarray = pk["meta_m"]  # type: ignore[assignment]
        for m_i, c in enumerate(cands):
            dag: Tuple[Tuple[int, int, int], ...] = c[0]  # type: ignore[assignment]
            t_len = min(len(dag), MAX_GRAPH_STEPS)
            dag_len[m_i] = max(1, t_len)
            used_leaves: Set[int] = set()
            for t_idx in range(t_len):
                op_t, a_t, b_t = dag[t_idx]
                dag_op[m_i, t_idx] = int(op_t)
                na = int(a_t) if a_t < k else (K_SLOT_BUF + int(a_t - k))
                nb = int(b_t) if b_t < k else (K_SLOT_BUF + int(b_t - k))
                dag_a[m_i, t_idx] = min(na, K_SLOT_BUF + MAX_GRAPH_STEPS - 1)
                dag_b[m_i, t_idx] = min(nb, K_SLOT_BUF + MAX_GRAPH_STEPS - 1)
                if a_t < k:
                    used_leaves.add(int(a_t))
                if b_t < k:
                    used_leaves.add(int(b_t))

            # Conservative distractor discounting on unused-slot meta penalty strictly when ng >= 4
            if self.use_rlns_and_null and ng >= 4:
                unused_gr = [i for i in gr_set if i not in used_leaves]
                adj_unused = sum(max(0.25, 1.0 - 0.65 * float(null_probs[i])) for i in unused_gr)
                meta_m[m_i, 3] = np.float32(adj_unused / float(max(1, ng)))

        # Build pairwise map array & upper-triangle indices for vectorized RLNS-1
        pmap: Dict[Tuple[int, int], int] = pf["pm"]  # type: ignore[assignment]
        pair_map_arr = np.zeros((k, k), dtype=np.int64)
        for i in range(k):
            for j in range(i + 1, k):
                pair_map_arr[i, j] = pmap[(i, j)]
                pair_map_arr[j, i] = pmap[(i, j)]
        iu_arr, iv_arr = np.triu_indices(k, k=1)

        pk["null_inv"] = pf["null_inv"]
        pk["hq_u"] = pf["hq_u"]
        pk["u_risk"] = float(pf.get("u_risk", 0.5))  # type: ignore[arg-type]
        pk["dag_op"] = dag_op
        pk["dag_a"] = dag_a
        pk["dag_b"] = dag_b
        pk["dag_len"] = dag_len
        pk["pair_map_arr"] = pair_map_arr
        pk["pair_iu"] = iu_arr.astype(np.int64)
        pk["pair_iv"] = iv_arr.astype(np.int64)
        pk["pair_pidx"] = pair_map_arr[iu_arr, iv_arr].astype(np.int64)
        return pk

    def _collate_uebr_to_device(
        self, batch_packs: Sequence[Dict[str, object]]
    ) -> Tuple[Tuple[object, ...], List[int]]:
        dev = self.device
        uf_l, nf_l, pf_l, rf_l, lm_l, hq_l, ur_l = [], [], [], [], [], [], []
        slot_offsets, slot_counts = [], []
        s_pidx_l = []
        sg_l, sr_l, sc_l, sinv_l = [], [], [], []
        rp_l, ro_l = [], []
        bm_l, bp_l, bpu_l, bpv_l, bo_l, bra_l, brb_l, bwt_l = [], [], [], [], [], [], [], []
        psi_p_l, psi_m_l, psi_ln_l, meta_l = [], [], [], []
        cpidx_l, cloc_l, dop_l, da_l, db_l, dlen_l = [], [], [], [], [], []
        m_counts: List[int] = []

        slot_off = 0
        pair_off = 0
        cand_off = 0
        for p_idx, pk in enumerate(batch_packs):
            k = int(pk["k"])  # type: ignore[arg-type]
            m = int(pk["m"])  # type: ignore[arg-type]
            uf_l.append(pk["Uf"])
            nf_l.append(pk["null_inv"])
            pf_l.append(pk["Pf"])
            rf_l.append(pk["Rf"])
            lm_l.append(pk["lm"])
            hq_l.append(pk["hq_u"])
            ur_l.append(float(pk["u_risk"]))  # type: ignore[arg-type]

            slot_offsets.append(slot_off)
            slot_counts.append(k)
            s_pidx_l.append(np.full(k, p_idx, dtype=np.int64))

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
                # Map pairs back to slot_gidx using pk edges?Wait: `_analyze_dag` uses pmap; recover (u,v) via precomputed lookup or add bpu/bpv!
                bo_l.append(pk["bo"])
                bra_l.append(pk["bra"])
                brb_l.append(pk["brb"])
                bwt_l.append(pk["bwt"])
                # Recover (u, v) from flat pair_map_arr cleanly:
                pmap_arr: np.ndarray = pk["pair_map_arr"]  # type: ignore[assignment]
                inv_uv: Dict[int, Tuple[int, int]] = {}
                for ui in range(k):
                    for vj in range(ui + 1, k):
                        inv_uv[int(pmap_arr[ui, vj])] = (ui + slot_off, vj + slot_off)
                bp_arr_local: np.ndarray = pk["bp"]  # type: ignore[assignment]
                u_gl = [inv_uv[int(pid)][0] for pid in bp_arr_local]
                v_gl = [inv_uv[int(pid)][1] for pid in bp_arr_local]
                bpu_l.append(np.array(u_gl, dtype=np.int64))
                bpv_l.append(np.array(v_gl, dtype=np.int64))

            psi_p_l.append(pk["psi_pad"])
            psi_m_l.append(pk["psi_mask"])
            psi_ln_l.append(pk["psi_lens"])
            meta_l.append(pk["meta_m"])

            cpidx_l.append(np.full(m, p_idx, dtype=np.int64))
            cloc_l.append(np.arange(m, dtype=np.int64))
            dop_l.append(pk["dag_op"])
            da_l.append(pk["dag_a"])
            db_l.append(pk["dag_b"])
            dlen_l.append(pk["dag_len"])

            slot_off += k
            pair_off += len(pk["Pf"])  # type: ignore[arg-type]
            cand_off += m
            m_counts.append(m)

        bm_t = torch.from_numpy(np.concatenate(bm_l)) if bm_l else torch.zeros(0, dtype=torch.long)
        bp_t = torch.from_numpy(np.concatenate(bp_l)) if bp_l else torch.zeros(0, dtype=torch.long)
        bpu_t = torch.from_numpy(np.concatenate(bpu_l)) if bpu_l else torch.zeros(0, dtype=torch.long)
        bpv_t = torch.from_numpy(np.concatenate(bpv_l)) if bpv_l else torch.zeros(0, dtype=torch.long)
        bo_t = torch.from_numpy(np.concatenate(bo_l)) if bo_l else torch.zeros(0, dtype=torch.long)
        bra_t = torch.from_numpy(np.concatenate(bra_l)) if bra_l else torch.zeros(0, dtype=torch.long)
        brb_t = torch.from_numpy(np.concatenate(brb_l)) if brb_l else torch.zeros(0, dtype=torch.long)
        bwt_t = torch.from_numpy(np.concatenate(bwt_l)) if bwt_l else torch.zeros(0, dtype=torch.float32)

        m_max = max(m_counts) if m_counts else 1
        args = (
            torch.from_numpy(np.concatenate(uf_l, axis=0)).to(dev),
            torch.from_numpy(np.concatenate(pf_l, axis=0)).to(dev),
            torch.from_numpy(np.stack(rf_l, axis=0)).to(dev),
            torch.from_numpy(np.concatenate(lm_l, axis=0)).to(dev),
            torch.from_numpy(np.stack(hq_l, axis=0)).to(dev),
            torch.from_numpy(np.concatenate(s_pidx_l)).to(dev),
            torch.tensor(ur_l, dtype=torch.float32, device=dev),
            slot_offsets,
            slot_counts,
            torch.from_numpy(np.concatenate(sg_l)).to(dev),
            torch.from_numpy(np.concatenate(sr_l)).to(dev),
            torch.from_numpy(np.concatenate(sc_l)).to(dev),
            torch.from_numpy(np.concatenate(sinv_l)).to(dev),
            torch.from_numpy(np.concatenate(rp_l)).to(dev),
            torch.from_numpy(np.concatenate(ro_l)).to(dev),
            bm_t.to(dev),
            bp_t.to(dev),
            bpu_t.to(dev),
            bpv_t.to(dev),
            bo_t.to(dev),
            bra_t.to(dev),
            brb_t.to(dev),
            bwt_t.to(dev),
            torch.from_numpy(np.concatenate(psi_p_l, axis=0)).to(dev),
            torch.from_numpy(np.concatenate(psi_m_l, axis=0)).to(dev),
            torch.from_numpy(np.concatenate(psi_ln_l)).to(dev),
            torch.from_numpy(np.concatenate(meta_l, axis=0)).to(dev),
            torch.from_numpy(np.concatenate(cpidx_l)).to(dev),
            torch.from_numpy(np.concatenate(cloc_l)).to(dev),
            m_max,
            torch.from_numpy(np.concatenate(dop_l, axis=0)).to(dev),
            torch.from_numpy(np.concatenate(da_l, axis=0)).to(dev),
            torch.from_numpy(np.concatenate(db_l, axis=0)).to(dev),
            torch.from_numpy(np.concatenate(dlen_l)).to(dev),
            torch.from_numpy(np.concatenate(nf_l, axis=0)).to(dev),
        )
        return args, m_counts

    def _fit_null_prior_net(
        self,
        train_items: Sequence[Dict[str, str]],
        slots_list: Sequence[Sequence[ExtractedSlot]],
        lm_feats_list: Sequence[Dict[str, object]],
    ) -> None:
        """Trains decoupled position-invariant `null_prior_net` (`0` BPTT) on clean vs donor-clause premise insertions."""
        if not self.use_rlns_and_null:
            return
        rng = random.Random(self.seed)
        X_n: List[np.ndarray] = []
        y_n: List[float] = []
        donors: List[Tuple[Set[str], str, float, np.ndarray, np.ndarray]] = []
        parsed_q: List[Tuple[List[str], Dict[int, List[int]], str, List[int], Set[str]]] = []

        n_collect = min(2500, len(train_items))
        for it, sl, lmf in zip(train_items[:n_collect], slots_list[:n_collect], lm_feats_list[:n_collect]):
            q_str = str(it["question"]).strip()
            toks = scan_question_tokens(q_str)
            raw_b = q_str.encode("utf-8")
            q_sid = toks[-1].sent_id if toks else 0
            for tk in reversed(toks):
                if tk.kind == "PUNCT" and tk.text == "?":
                    q_sid = tk.sent_id
                    break
            by_sent: Dict[int, List] = {}
            by_clause: Dict[int, List] = {}
            t_words: Set[str] = set()
            for tk in toks:
                by_sent.setdefault(tk.sent_id, []).append(tk)
                by_clause.setdefault(tk.clause_id, []).append(tk)
                if tk.kind == "WORD" and len(tk.lower) >= 3:
                    t_words.add(tk.lower)

            prem_sents: List[str] = []
            q_sents: List[str] = []
            sent_sids: Dict[int, List[int]] = {}
            for sid in sorted(by_sent.keys()):
                stoks = by_sent[sid]
                stxt = raw_b[stoks[0].b0 : stoks[-1].b1].decode("utf-8", errors="ignore").strip()
                if not stxt:
                    continue
                if stxt[-1] not in (".", "?", "!"):
                    stxt = stxt + "."
                s_ids = [si for si, s_obj in enumerate(sl) if s_obj.kind != "CONST" and s_obj.sent_id == sid]
                if sid < q_sid:
                    sent_sids[len(prem_sents)] = s_ids
                    prem_sents.append(stxt)
                else:
                    q_sents.append(stxt)
            q_fin = " ".join(q_sents) if q_sents else q_str
            q_sids = [si for si, s_obj in enumerate(sl) if s_obj.kind != "CONST" and s_obj.sent_id >= q_sid]
            parsed_q.append((prem_sents, sent_sids, q_fin, q_sids, t_words))

            hn = np.asarray(lmf["h_num"], dtype=np.float32)
            hp = np.asarray(lmf["h_post"], dtype=np.float32)
            if hn.ndim != 2 or hn.shape[0] != len(sl) or len(sl) < 2 or len(sl) > K_SLOT_BUF - 2:
                continue
            st, _, ok = self.extractor.compile_train_dag(sl, str(it["answer"]))
            stc = _prune_to_connected_dag(st, len(sl), max_steps=MAX_GRAPH_STEPS) if (ok and st) else ()
            if stc:
                pf = self._featurize_problem(q_str, sl, lmf)
                rl, *_ = self._analyze_dag(stc, sl, pf)
                nf: np.ndarray = pf["null_inv"]  # type: ignore[assignment]
                for i, s_obj in enumerate(sl):
                    if s_obj.kind != "CONST" and rl[i] > 0:
                        X_n.append(nf[i])
                        y_n.append(0.0)
            for _, ctoks in by_clause.items():
                if not ctoks or ctoks[0].sent_id >= q_sid:
                    continue
                cid = ctoks[0].clause_id
                c_sl = [si for si, s_obj in enumerate(sl) if s_obj.kind == "EXPLICIT" and s_obj.clause_id == cid]
                c_words = [tk.lower for tk in ctoks if tk.kind == "WORD"]
                if len(c_sl) == 1 and 3 <= len(c_words) <= 14:
                    ctxt = raw_b[ctoks[0].b0 : ctoks[-1].b1].decode("utf-8", errors="ignore").strip(" ,.;:?!")
                    if len(ctxt) >= 8:
                        si0 = c_sl[0]
                        donors.append(({w for w in c_words if len(w) >= 3}, ctxt, float(sl[si0].val), hn[si0], hp[si0]))

        if not donors or not X_n:
            return

        max_pos = min(1400, len(parsed_q))
        pos_added = 0
        for idx in range(len(parsed_q)):
            if pos_added >= max_pos:
                break
            it, sl, lmf = train_items[idx], slots_list[idx], lm_feats_list[idx]
            hn_o = np.asarray(lmf["h_num"], dtype=np.float32)
            hp_o = np.asarray(lmf["h_post"], dtype=np.float32)
            prem, sent_sids, q_fin, q_sids, t_words = parsed_q[idx]
            if hn_o.ndim != 2 or hn_o.shape[0] != len(sl) or not prem:
                continue
            donor_picked = None
            for _ in range(20):
                dc = donors[rng.randrange(len(donors))]
                if not (dc[0] & t_words):
                    donor_picked = dc
                    break
            if donor_picked is None:
                continue
            _, d_txt, _, d_hn, d_hp = donor_picked
            d_sent = d_txt[0].upper() + d_txt[1:] + "."
            ins_p = rng.randint(0, len(prem))
            parts: List[str] = []
            exp_order: List[int] = []
            for p_idx in range(len(prem) + 1):
                if p_idx == ins_p:
                    parts.append(d_sent)
                    exp_order.append(-1)
                if p_idx < len(prem):
                    parts.append(prem[p_idx])
                    exp_order.extend(sent_sids.get(p_idx, []))
            parts.append(q_fin)
            exp_order.extend(q_sids)
            aug_q = " ".join(p for p in parts if p)
            sl_aug, _, _ = self.extractor.extract_slots(aug_q, include_aux_consts=True)
            gr_aug = [i for i, s_obj in enumerate(sl_aug) if s_obj.kind != "CONST"]
            if len(gr_aug) != len(exp_order) or len(sl_aug) > K_SLOT_BUF - 2:
                continue
            hn_new = np.zeros((len(sl_aug), self.d_model), dtype=np.float32)
            hp_new = np.zeros((len(sl_aug), self.d_model), dtype=np.float32)
            d_slot = -1
            ok_align = True
            for ng_i, old_si in zip(gr_aug, exp_order):
                if old_si == -1:
                    d_slot = ng_i
                    hn_new[ng_i] = d_hn
                    hp_new[ng_i] = d_hp
                elif old_si < len(hn_o):
                    hn_new[ng_i] = hn_o[old_si]
                    hp_new[ng_i] = hp_o[old_si]
                else:
                    ok_align = False
                    break
            if not ok_align or d_slot < 0:
                continue
            lmf_aug = dict(lmf)
            lmf_aug["h_num"] = hn_new
            lmf_aug["h_post"] = hp_new
            lmf_aug["beam_plans"] = []
            pf_aug = self._featurize_problem(aug_q, sl_aug, lmf_aug)
            nf_aug: np.ndarray = pf_aug["null_inv"]  # type: ignore[assignment]
            X_n.append(nf_aug[d_slot])
            y_n.append(1.0)
            pos_added += 1

        if pos_added == 0:
            return
        Xt = torch.from_numpy(np.stack(X_n, axis=0)).to(device=self.device, dtype=torch.float32)
        yt = torch.tensor(y_n, dtype=torch.float32, device=self.device)
        n_pos = max(1.0, float((yt > 0.5).sum().item()))
        n_neg = max(1.0, float((yt < 0.5).sum().item()))
        pos_w = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=self.device)
        self.ebm.null_prior_net.train()
        opt_n = torch.optim.AdamW(self.ebm.null_prior_net.parameters(), lr=5.0e-3, weight_decay=1.0e-4)
        for _ in range(250):
            opt_n.zero_grad()
            logits_n = self.ebm.null_prior_net(Xt).squeeze(-1)
            loss_n = F.binary_cross_entropy_with_logits(logits_n, yt, pos_weight=pos_w)
            loss_n.backward()
            opt_n.step()
        self.ebm.null_prior_net.zero_grad(set_to_none=True)
        self.ebm.null_prior_net.eval()

    def fit_bptt_free(
        self,
        train_items: Sequence[Dict[str, str]],
        slots_list: Sequence[Sequence[ExtractedSlot]],
        lm_feats_list: Sequence[Dict[str, object]],
        epochs: int = 10,
        batch_size: int = 96,
        lr: float = 0.006,
    ) -> Dict[str, float]:
        """0-BPTT analytical Woodbury + GAQ + SPD Cholesky Adjoint + Eikonal + RLNS-1 block Fenchel-Young."""
        t0 = time.perf_counter()
        h_tr = hashlib.sha256()
        h_tr.update(
            f"n={len(train_items)}|dm={self.d_model}|rlns={int(self.use_rlns_and_null)}|qap={int(self.use_birkhoff_qap)}\n".encode("utf-8")
        )
        samp_idxs = list(range(min(8, len(train_items)))) + list(range(max(0, len(train_items) - 8), len(train_items)))
        for si_h in samp_idxs:
            h_tr.update(str(train_items[si_h]["question"]).encode("utf-8"))
            h_tr.update(str(train_items[si_h]["answer"]).encode("utf-8"))
        cache_key = (h_tr.hexdigest(), int(len(train_items)), int(self.d_model), bool(self.use_rlns_and_null), bool(self.use_birkhoff_qap))

        if cache_key in _ANCHOR_TRAIN_CACHE:
            cached = _ANCHOR_TRAIN_CACHE[cache_key]
            self.vocab_counts = Counter(cached["vocab_counts"])  # type: ignore[arg-type]
            self.stop_words = set(cached["stop_words"])  # type: ignore[arg-type]
            self.keep_vocab = set(cached["keep_vocab"])  # type: ignore[arg-type]
            self.ep_vecs = dict(cached["ep_vecs"])  # type: ignore[arg-type]
            self.ep_items = dict(cached["ep_items"])  # type: ignore[arg-type]
            self.sk_cnt = dict(cached["sk_cnt"])  # type: ignore[arg-type]
            self.sk_tot = dict(cached["sk_tot"])  # type: ignore[arg-type]
            Wu0 = cached["Wu0"]  # type: ignore[assignment]
            Wp0 = cached["Wp0"]  # type: ignore[assignment]
            Wr0 = cached["Wr0"]  # type: ignore[assignment]
            Wlm0 = cached["Wlm0"]  # type: ignore[assignment]
            tau_med = float(cached["tau_med"])  # type: ignore[arg-type]
            train_packs: List[Tuple[Dict[str, object], np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]] = cached["train_packs"]  # type: ignore[assignment]
            self.ebm.set_woodbury_anchors(Wu0, Wp0, Wr0, Wlm0, tau_risk_init=tau_med)  # type: ignore[arg-type]
        else:
            self._fit_corpus_stats(train_items)
            self.ep_vecs.clear()
            self.ep_items.clear()
            self.sk_cnt.clear()
            self.sk_tot.clear()

            ep_vecs_list: Dict[Tuple[int, int], List[np.ndarray]] = {}
            uX, uY, pX, pY, rX, rY, lmX, lmY, ur_all = [], [], [], [], [], [], [], [], []
            parsed: List[Tuple[Sequence[ExtractedSlot], Dict[str, object], float, Optional[float], Optional[np.ndarray], Optional[np.ndarray]]] = []

            for it, sl, lmf in zip(train_items, slots_list, lm_feats_list):
                if len(sl) < 2 or len(sl) > K_SLOT_BUF - 2:
                    continue
                pf = self._featurize_problem(str(it["question"]), sl, lmf)
                ur_all.append(float(pf["u_risk"]))  # type: ignore[arg-type]
                st, _, ok = self.extractor.compile_train_dag(sl, str(it["answer"]))
                stc = _prune_to_connected_dag(st, len(sl), max_steps=MAX_GRAPH_STEPS) if (ok and st) else ()
                sgt = _dag_rational_sig(stc, len(sl)) if stc else None
                psi_true: Optional[np.ndarray] = None
                rl_true: Optional[np.ndarray] = None
                if stc and sgt is not None:
                    rl, bds, rtop, psi_seq, _ = self._analyze_dag(stc, sl, pf)
                    psi_true = psi_seq
                    rl_true = rl.copy()
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
                parsed.append((sl, pf, extract_gt_number(str(it["answer"])), sgt, psi_true, rl_true))

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

            tau_med = float(np.percentile(ur_all, 70)) if ur_all else 0.45
            self.ebm.set_woodbury_anchors(Wu0, Wp0, Wr0, Wlm0, tau_risk_init=tau_med)

            # Clean anchor-only pools + strict admissible positive labels
            train_packs = []
            for sl, pf, gt, sgt, psi_true, rl_true in parsed:
                cands = self._synthesize_open_pool(sl, pf, mask_self=True, use_anchor_only=True)
                if len(cands) < 2:
                    continue
                pw = np.zeros(len(cands), dtype=np.float32)
                has_exact = False
                if sgt is not None:
                    for i, c in enumerate(cands):
                        if c[5] == sgt:
                            pw[i] = 1.0
                            has_exact = True
                gr_sl = [i for i, s in enumerate(sl) if s.kind != "CONST"] or list(range(len(sl)))
                gr_set_sl = set(gr_sl)
                ng_sl = len(gr_sl)
                min_ug_sl = max(1, ng_sl - 1 if ng_sl <= 5 else ng_sl - 2)
                b0_min_ug = max(2, min(5, ng_sl - 3)) if ng_sl >= 8 else min_ug_sl
                for i, c in enumerate(cands):
                    if abs(float(c[1]) - gt) <= 1e-2 and pw[i] == 0.0:
                        ul_c = {x for _, a, b in c[0] for x in (a, b) if x < len(sl)}
                        ug_c = len(ul_c & gr_set_sl)
                        uc_c = len(ul_c) - ug_c
                        if ug_c >= b0_min_ug and uc_c == 0:
                            pw[i] = 0.12 if has_exact else 1.0
                if float(pw.sum()) <= 0.0:
                    continue
                pw /= float(pw.sum())
                train_packs.append((self._pack_candidates_cpu(sl, pf, cands), pw, psi_true, rl_true))

            _ANCHOR_TRAIN_CACHE[cache_key] = {
                "vocab_counts": Counter(self.vocab_counts),
                "stop_words": set(self.stop_words),
                "keep_vocab": set(self.keep_vocab),
                "ep_vecs": dict(self.ep_vecs),
                "ep_items": dict(self.ep_items),
                "sk_cnt": dict(self.sk_cnt),
                "sk_tot": dict(self.sk_tot),
                "Wu0": Wu0.copy(),
                "Wp0": Wp0.copy(),
                "Wr0": Wr0.copy(),
                "Wlm0": Wlm0.copy(),
                "tau_med": tau_med,
                "train_packs": train_packs,
            }

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
        opt = torch.optim.AdamW(
            [
                {"params": anchor_params, "lr": float(lr), "weight_decay": 0.0},
                {"params": neural_params, "lr": 3.0e-3, "weight_decay": 1e-4},
            ]
        )
        total_steps = max(1, epochs * math.ceil(max(1, len(train_packs)) / batch_size))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=5e-4)

        final_loss = 0.0
        for ep in range(epochs):
            perm = rng.permutation(len(train_packs))
            ep_loss = 0.0
            for b0 in range(0, len(train_packs), batch_size):
                b_ids = perm[b0 : b0 + batch_size]
                batch_pks = [train_packs[int(ix)][0] for ix in b_ids]
                tensors, m_counts = self._collate_uebr_to_device(batch_pks)
                n_tot_c = sum(m_counts)

                opt.zero_grad()
                batch_out = self.ebm.score_batch(*tensors, n_cands=n_tot_c)
                e_cram = batch_out.e_tot

                loss_pool = torch.tensor(0.0, device=self.device)
                pos_cands: List[int] = []
                neg_cands: List[int] = []
                pos_pidxs: List[int] = []

                cur = 0
                for b_loc, (ix, m_c) in enumerate(zip(b_ids, m_counts)):
                    e_slice = e_cram[cur : cur + m_c]
                    pw_np = train_packs[int(ix)][1]
                    pw_t = torch.from_numpy(pw_np).to(self.device)
                    loss_pool = loss_pool - (pw_t * F.log_softmax(-e_slice, dim=0)).sum()

                    # Collect best positive & hardest negative candidate indices for Directional Eikonal Loss
                    if self.use_spd_laplacian and m_c >= 2:
                        pos_loc = int(np.argmax(pw_np))
                        mask_neg = pw_np == 0.0
                        if np.any(mask_neg):
                            with torch.no_grad():
                                e_neg_chk = e_slice.detach().clone()
                                e_neg_chk[torch.from_numpy(~mask_neg).to(self.device)] = 1e9
                                neg_loc = int(torch.argmin(e_neg_chk).item())
                        else:
                            neg_loc = (pos_loc + 1) % m_c
                        pos_cands.append(cur + pos_loc)
                        neg_cands.append(cur + neg_loc)
                        pos_pidxs.append(b_loc)
                    cur += m_c

                loss_pool = loss_pool / float(max(1, len(b_ids)))

                # Coupled 3-slot RLNS-1 Fenchel-Young loss with absorbing Null Slot 0 (Ø) & Woodbury gradient isolation
                loss_rlns = torch.tensor(0.0, device=self.device)
                if self.use_rlns_and_null:
                    u_tot = batch_out.u_tot_rlns
                    p_pot = batch_out.p_pot_rlns
                    slot_offs = tensors[7]  # type: ignore[index]
                    slot_cnts = tensors[8]  # type: ignore[index]
                    pair_off = 0
                    rlns_cnt = 0
                    for b_loc, ix in enumerate(b_ids):
                        rl_true_np = train_packs[int(ix)][3]
                        k_b = int(slot_cnts[b_loc])  # type: ignore[arg-type]
                        so_b = int(slot_offs[b_loc])  # type: ignore[arg-type]
                        p_len_b = k_b * (k_b - 1) // 2
                        if rl_true_np is not None and k_b >= 2:
                            pk_b = train_packs[int(ix)][0]
                            loss_rlns = loss_rlns + self.ebm.rlns_block3_fenchel_young_loss(
                                u_slots=u_tot,
                                p_pot=p_pot[pair_off : pair_off + p_len_b],
                                slot_start=so_b,
                                k_sl=k_b,
                                pair_map_arr=pk_b["pair_map_arr"],  # type: ignore[arg-type]
                                rl_true=torch.from_numpy(rl_true_np).to(self.device),
                                pair_iu_t=torch.from_numpy(pk_b["pair_iu"]).to(self.device),  # type: ignore[arg-type]
                                pair_iv_t=torch.from_numpy(pk_b["pair_iv"]).to(self.device),  # type: ignore[arg-type]
                                pair_pidx_t=torch.from_numpy(pk_b["pair_pidx"]).to(self.device),  # type: ignore[arg-type]
                            )
                            rlns_cnt += 1
                        pair_off += max(1, p_len_b)
                    if rlns_cnt > 0:
                        loss_rlns = 0.04 * (loss_rlns / float(rlns_cnt))

                loss_eik = torch.tensor(0.0, device=self.device)
                if self.use_spd_laplacian and pos_cands and batch_out.z_flat is not None:
                    z_flat = batch_out.z_flat
                    q_prob = batch_out.q_prob
                    p_idx_t = torch.tensor(pos_cands, dtype=torch.long, device=self.device)
                    n_idx_t = torch.tensor(neg_cands, dtype=torch.long, device=self.device)
                    pb_t = torch.tensor(pos_pidxs, dtype=torch.long, device=self.device)
                    loss_eik = 0.03 * self.ebm.spd_laplacian.directional_eikonal_loss(
                        z_flat[p_idx_t], z_flat[n_idx_t], q_prob[pb_t]
                    )

                loss_vq = 0.04 * batch_out.vq_loss if (self.use_gaq_bridge or self.use_spd_laplacian) else 0.0
                loss = loss_pool + loss_rlns + loss_eik + loss_vq + self.ebm.prox_regularization_loss()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ebm.parameters(), 2.0)
                opt.step()
                sched.step()
                ep_loss += float(loss_pool.item()) * len(b_ids)

            final_loss = ep_loss / float(max(1, len(train_packs)))
            if (ep + 1) in (1, max(1, epochs // 2), epochs):
                print(
                    f"   [UEBR CRAM Fit] Ep {ep+1:2d}/{epochs} | Loss={final_loss:.4f} | "
                    f"QAP_lossless={self.birkhoff_lossless_wins}/{max(1, self.birkhoff_total_solves)}",
                    flush=True,
                )

        self.ebm.eval()
        if self.use_rlns_and_null:
            self._fit_null_prior_net(train_items, slots_list, lm_feats_list)

        # Measure final matrix-free ePC consistency & SPD min_eig on 1 batch
        epc_err = 0.0
        min_eig = 1.0
        if self.use_spd_laplacian and train_packs:
            with torch.no_grad():
                probe_pks = [train_packs[i][0] for i in range(min(8, len(train_packs)))]
                tens_p, mc_p = self._collate_uebr_to_device(probe_pks)
                ex_p = self.ebm.score_batch(*tens_p, n_cands=sum(mc_p))
                if ex_p.H_psi is not None and ex_p.S_pad is not None and ex_p.Z_pad is not None:
                    Z_epc = epc_equilibrium_solve(ex_p.H_psi, ex_p.S_pad, steps=45)
                    epc_err = float((ex_p.Z_pad - Z_epc).abs().max().item())
                    min_eig = float(torch.linalg.eigvalsh(ex_p.H_psi.float()).min().item())
        self.last_epc_err = epc_err
        self.last_min_eig = min_eig

        with torch.no_grad():
            wu_drift = float((self.ebm.W_u - self.ebm.Wu0).norm().item())
            w_eff_pen = self.ebm.w_meta_effective()[self.ebm.wm_pen_mask]
            w_raw_pen = self.ebm.w_meta[self.ebm.wm_pen_mask]
            wm_eff_min = float(w_eff_pen.min().item())
            wm_raw_min = float(w_raw_pen.min().item())

        return {
            "trainable_packs": float(len(train_packs)),
            "final_contrastive_loss": round(final_loss, 4),
            "gershgorin_margin": self.gershgorin_margin(),
            "laplacian_min_eig": round(min_eig, 4),
            "epc_max_err": epc_err,
            "wu_drift_norm": round(wu_drift, 4),
            "wm_eff_min_pen": round(wm_eff_min, 4),
            "wm_raw_min_pen": round(wm_raw_min, 4),
            "train_seconds": round(time.perf_counter() - t0, 2),
        }

    def gershgorin_margin(self) -> float:
        with torch.no_grad():
            max_l1 = float(self.ebm.normalized_W_zz().abs().sum(dim=-1).max().item())
            return float(self.ebm.m_convex - max_l1)

    def _assemble_and_score_candidates(
        self,
        q_text: str,
        slots: Sequence[ExtractedSlot],
        lm_feat: Optional[Dict[str, object]],
        _allow_null_proj: bool = True,
    ) -> Tuple[List[Dict[str, object]], float]:
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
                    "d_psi": 0.0,
                    "meta": np.zeros(N_META_FEATS, dtype=np.float32),
                    "in_sys1": 0.0,
                    "sys1_lp": -12.0,
                }
            ], max(dr_dag, dr_dec)

        # Active-subgraph projection via position-invariant Null Slot Ø gate
        if self.use_rlns_and_null and _allow_null_proj and k >= 3 and lm_feat is not None:
            with torch.no_grad():
                nf_t = torch.from_numpy(pf["null_inv"]).to(self.device)  # type: ignore[arg-type]
                p_null = torch.sigmoid(self.ebm.null_prior_net(nf_t).squeeze(-1)).cpu().numpy()
            gr_idx = [i for i, s in enumerate(slots) if s.kind != "CONST"]
            act_gr = [i for i in gr_idx if float(p_null[i]) <= 0.78]
            if 2 <= len(act_gr) < len(gr_idx):
                keep = act_gr + [i for i, s in enumerate(slots) if s.kind == "CONST"]
                sub_slots = [slots[i] for i in keep]
                hn_o = np.asarray(lm_feat.get("h_num", np.zeros((0, self.d_model))), dtype=np.float32)
                hp_o = np.asarray(lm_feat.get("h_post", np.zeros((0, self.d_model))), dtype=np.float32)
                if hn_o.ndim == 2 and hn_o.shape[0] == k and hp_o.ndim == 2 and hp_o.shape[0] == k:
                    lm_sub = dict(lm_feat)
                    lm_sub["h_num"] = hn_o[keep]
                    lm_sub["h_post"] = hp_o[keep]
                    lm_sub["beam_plans"] = []
                    sub_recs, dr_sub = self._assemble_and_score_candidates(
                        q_text, sub_slots, lm_sub, _allow_null_proj=False
                    )
                    if sub_recs:
                        ks = len(sub_slots)
                        for rec in sub_recs:
                            nd = []
                            for op_r, a_r, b_r in rec["steps"]:  # type: ignore[attr-defined]
                                ra = keep[a_r] if a_r < ks else k + (a_r - ks)
                                rb = keep[b_r] if b_r < ks else k + (b_r - ks)
                                sa, sb = (rb, ra) if (op_r in (0, 2) and ra > rb) else (ra, rb)
                                nd.append((op_r, sa, sb))
                            rec["steps"] = tuple(nd)
                        return sub_recs, dr_sub

        cands = self._synthesize_open_pool(slots, pf, mask_self=False)
        if not cands:
            return [], 0.0

        pk = self._pack_candidates_cpu(slots, pf, cands)
        tensors, _ = self._collate_uebr_to_device([pk])
        with torch.no_grad():
            batch_out = self.ebm.score_batch(*tensors, n_cands=len(cands))
            if self.use_spd_laplacian and batch_out.H_psi is not None:
                eigs = torch.linalg.eigvalsh(batch_out.H_psi.float())
                self.last_min_eig = min(self.last_min_eig, float(eigs.min().item()))
        e_cram = batch_out.e_tot.cpu().numpy().astype(np.float64)
        e_unary = batch_out.e_ablate.cpu().numpy().astype(np.float64)
        d_psi = batch_out.d_psi.cpu().numpy().astype(np.float64)
        meta_m: np.ndarray = pk["meta_m"]  # type: ignore[assignment]

        records: List[Dict[str, object]] = []
        for idx, c in enumerate(cands):
            dag: Tuple[Tuple[int, int, int], ...] = c[0]  # type: ignore[assignment]
            records.append(
                {
                    "steps": dag,
                    "val": float(c[1]),
                    "norm_drift": 0.0,
                    "invalid": False,
                    "e_cram": float(e_cram[idx]),
                    "e_unary": float(e_unary[idx]),
                    "d_psi": float(d_psi[idx]),
                    "meta": meta_m[idx],
                    "in_sys1": float(c[6]),
                    "sys1_lp": float(c[7]),
                    "has_tf": False,
                }
            )
        return records, 0.0

    @staticmethod
    def _cert_feature_vec_uebr(
        e_vec: np.ndarray,
        margin_e: float,
        p_top: float,
        entropy_val: float,
        best_rec: Dict[str, object],
        k_slots: int,
    ) -> np.ndarray:
        base = OpenGraphCRAMSolver._cert_feature_vec(e_vec, margin_e, p_top, entropy_val, best_rec, k_slots)
        d_psi_val = float(best_rec.get("d_psi", 0.0))
        return np.concatenate([base, np.array([d_psi_val], dtype=np.float64)], axis=0)

    @staticmethod
    def _fit_cert_ridge_dyn(rows: Sequence[Tuple[float, np.ndarray, int]]) -> Tuple[np.ndarray, float]:
        if len(rows) < 20:
            dim = int(rows[0][1].shape[0]) if rows else (N_CERT_FEATS + 1)
            return np.zeros(dim, dtype=np.float64), 0.0
        X = np.stack([r[1] for r in rows], axis=0)
        y = np.array([1.0 - float(r[2]) for r in rows], dtype=np.float64)
        X_aug = np.concatenate([np.ones((len(X), 1), dtype=np.float64), X], axis=1)
        reg = 8.0 * np.eye(1 + X.shape[1], dtype=np.float64)
        reg[0, 0] = 1e-4
        w_all = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y)
        return w_all[1:].copy(), float(w_all[0])

    def calibrate_on_split(
        self,
        cal_items: Sequence[Dict[str, str]],
        slots_list: Sequence[Sequence[ExtractedSlot]],
        lm_feats_list: Sequence[Dict[str, object]],
        min_cert_precision: float = 0.38,
        min_cert_coverage: float = 0.085,
    ) -> Dict[str, float]:
        """5-fold OOF DualCert calibration using `u_risk = u_epi + 0.05 * VarEntropy` and `d_psi`."""
        packed = []
        oracle_hits = 0
        for it, slots, lmf in zip(cal_items, slots_list, lm_feats_list):
            gt = extract_gt_number(str(it["answer"]))
            cands, _ = self._assemble_and_score_candidates(str(it["question"]), slots, lmf)
            u_risk = float(lmf.get("u_risk", lmf.get("u_prob", 1.0))) if lmf is not None else 1.0  # type: ignore[arg-type]
            if not cands:
                packed.append(([], np.zeros((0, N_RERANK_FEATS), dtype=np.float64), u_risk, gt, len(slots)))
                continue
            rmat = self._build_rerank_matrix(cands)
            if any(abs(float(r["val"]) - gt) <= 1e-2 for r in cands):
                oracle_hits += 1
            packed.append((cands, rmat, u_risk, gt, len(slots)))

        n_cal = max(1, len(cal_items))
        self.rerank_weights = self.base_rerank_weights.copy()
        n_folds = 5 if len(packed) >= 25 else 1
        oof_rows: List[Tuple[int, float, np.ndarray, int]] = []
        p1_hits = p5_hits = un_hits = 0
        for idx, (cands, rmat, u_risk, gt, k_sl) in enumerate(packed):
            fold = idx % n_folds if n_folds > 1 else 0
            if not cands:
                oof_rows.append((fold, u_risk, np.zeros(N_CERT_FEATS + 1, dtype=np.float64), 0))
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
            cf = self._cert_feature_vec_uebr(e_vec, margin_e, p_top, ent_v, best_rec, k_sl)
            oof_rows.append((fold, u_risk, cf, h1))

        eval_rows: List[Tuple[float, float, int]] = []
        if n_folds > 1:
            for fold in range(n_folds):
                tr_cf = [(u, cf, h) for f, u, cf, h in oof_rows if f != fold]
                val_cf = [(u, cf, h) for f, u, cf, h in oof_rows if f == fold]
                wc_f, bc_f = self._fit_cert_ridge_dyn(tr_cf)
                for u, cf, h in val_cf:
                    eval_rows.append((u, float(cf @ wc_f + bc_f), h))
        else:
            all_cf = [(u, cf, h) for _, u, cf, h in oof_rows]
            wc_f, bc_f = self._fit_cert_ridge_dyn(all_cf)
            for u, cf, h in all_cf:
                eval_rows.append((u, float(cf @ wc_f + bc_f), h))

        self.cert_weights, self.cert_bias = self._fit_cert_ridge_dyn([(u, cf, h) for _, u, cf, h in oof_rows])

        u_vals = sorted(r[0] for r in eval_rows)
        e_vals = sorted(r[1] for r in eval_rows)
        min_n = max(12, int(min_cert_coverage * len(eval_rows))) if len(eval_rows) >= 40 else max(1, len(eval_rows) // 3)
        p1_cal = p1_hits / float(n_cal)
        target_prec = max(min_cert_precision, p1_cal + 0.18)

        best_cov = -1.0
        best_prec = 0.0
        best_pair = (
            u_vals[min(len(u_vals) - 1, int(0.85 * (len(u_vals) - 1)))],
            e_vals[min(len(e_vals) - 1, int(0.12 * (len(e_vals) - 1)))],
        )
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
        if "####" in question_text or "<<" in question_text:
            raise ValueError("Anti-Cheat Violation: answer or <<>> tags leaked into solve_blind!")
        if slots is None:
            slots, _, _ = self.extractor.extract_slots(question_text, include_aux_consts=True)
        u_risk = float(lm_feat.get("u_risk", lm_feat.get("u_prob", 1.0))) if lm_feat is not None else 1.0  # type: ignore[arg-type]

        cands, init_drift = self._assemble_and_score_candidates(question_text, slots, lm_feat)
        if not cands:
            return {
                "pred_pass1": 0.0,
                "pred_pass3": [0.0],
                "pred_pass5": [0.0],
                "pred_unary_ablation": 0.0,
                "best_steps": (),
                "cand_vals": [],
                "sigma2_q": u_risk,
                "u_risk": u_risk,
                "e_cert": 999.0,
                "dual_certified": False,
                "norm_drift": init_drift,
            }

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
        if len(self.cert_weights) == N_CERT_FEATS + 1:
            cf = self._cert_feature_vec_uebr(e_vec, margin_e, p_top, ent_v, best_rec, len(slots))
            e_cert = 50.0 if bool(best_rec.get("invalid", False)) else float(cf @ self.cert_weights + self.cert_bias)
        else:
            e_cert = self._compute_e_cert(e_vec, margin_e, p_top, ent_v, best_rec, len(slots))
        certified = bool((u_risk <= self.theta_epi) and (e_cert <= self.tau_energy))
        max_drift = max(init_drift, dr_cram, dr_un)

        return {
            "pred_pass1": top1_alu,
            "pred_pass3": top5_alu[:3],
            "pred_pass5": top5_alu[:5],
            "pred_unary_ablation": unary_alu,
            "best_steps": best_rec["steps"],
            "cand_vals": [float(r["val"]) for r in cands],
            "sigma2_q": u_risk,
            "u_risk": u_risk,
            "e_cert": e_cert,
            "dual_certified": certified,
            "norm_drift": max_drift,
        }


def verify_matrix_laplacian_gradcheck(seed: int = 2026) -> float:
    """Verifies `_ImplicitLaplacianSolveFn` 3D matrix-RHS Cholesky adjoint against `float64` finite differences."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    B, D, M_c = 2, 6, 4
    A = torch.randn(B, D, D, generator=gen, dtype=torch.float64)
    H = torch.matmul(A, A.transpose(-1, -2)) + 0.8 * torch.eye(D, dtype=torch.float64).unsqueeze(0)
    H_var = H.clone().requires_grad_(True)
    S_var = torch.randn(B, D, M_c, generator=gen, dtype=torch.float64).requires_grad_(True)
    W = torch.randn(B, D, M_c, generator=gen, dtype=torch.float64)

    Z = _ImplicitLaplacianSolveFn.apply(H_var, S_var)
    loss = (Z * W).sum()
    gH_adj, gS_adj = torch.autograd.grad(loss, (H_var, S_var))

    eps = 1e-6
    gH_fd = torch.zeros_like(H)
    gS_fd = torch.zeros_like(S_var)
    with torch.no_grad():
        for b in range(B):
            for r in range(D):
                for c in range(D):
                    dH = torch.zeros_like(H)
                    dH[b, r, c] = eps
                    lp = (_ImplicitLaplacianSolveFn.apply(H + dH, S_var) * W).sum()
                    lm = (_ImplicitLaplacianSolveFn.apply(H - dH, S_var) * W).sum()
                    gH_fd[b, r, c] = (lp - lm) / (2.0 * eps)
                for c in range(M_c):
                    dS = torch.zeros_like(S_var)
                    dS[b, r, c] = eps
                    lp = (_ImplicitLaplacianSolveFn.apply(H, S_var + dS) * W).sum()
                    lm = (_ImplicitLaplacianSolveFn.apply(H, S_var - dS) * W).sum()
                    gS_fd[b, r, c] = (lp - lm) / (2.0 * eps)
    return float(max((gH_adj - gH_fd).abs().max().item(), (gS_adj - gS_fd).abs().max().item()))
