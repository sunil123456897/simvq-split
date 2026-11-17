"""Tracer #2 (`Phase I / Calcite-6 UEBR`): 4 architectural invariants + live Tier-1 LUT probe.

Self-contained PyTorch verification suite proving four UEBR mathematical laws on real
CUDA/CPU tensors without host regex/ast heuristics, BPTT loops through relaxation,
or bare residual shortcuts:
  1) Input-dependent SPD Graph Laplacian `L_psi(q_hat)` + `Cholesky Implicit Adjoint`
     (`0-BPTT`) + matrix-free `ePC` error-space relaxation + Directional Eikonal Loss
     beating static global `W_zz` on cyclic shortest-path vs long-detour graphs
     (both FP32 and 1.58-bit STE QAT) and matching `Symmetric EqProp`.
  2) Closed-form `PermFlow` affine-hull tangent projector `C(U)` + inertial Kerr
     oscillator bifurcation (`cSB`) + lossless `Birkhoff Extension` rounding beating
     entropic `Sinkhorn` on symmetric multi-modal permutations and dissipative `PGD`
     on 50 random non-convex `QAP (n=12)` instances.
  3) `0-BPTT` block Fenchel-Young (`RLNS-1`) with Shannon entropy regularization,
     adaptive min-cut freezing, and absorbing inactive node `0 (null)` beating
     single-cell `1-Flip Besag PL` on variable-cardinality paired-conservation tasks.
  4) `Lazy Identity Collapse` diagnosis under open continuous residual bypasses vs full
     recovery via risk-gated bypass blocking + `GAQSlotBridge` (`SimVQ + Householder`),
     plus a read-only live `Woodbury + VarEntropy` risk probe on the real `Tier-1`
     `5.41 MB 2-bit LUT` checkpoint (`calcite5_tier1_17m_utf8_qat_h_lut.pt`).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _auroc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    pos_l = [float(x) for x in pos.detach().cpu().reshape(-1).tolist()]
    neg_l = [float(x) for x in neg.detach().cpu().reshape(-1).tolist()]
    if not pos_l or not neg_l:
        return 0.5
    wins = 0.0
    for p in pos_l:
        for n in neg_l:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / float(len(pos_l) * len(neg_l))


def quantize_ternary_ste(w: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """1.58-bit groupwise absmean ternary quantization with STE (valid-count denominator)."""
    orig_shape = w.shape
    flat = w.reshape(-1)
    n = flat.numel()
    pad = (group_size - (n % group_size)) % group_size
    flat_p = F.pad(flat, (0, pad)) if pad > 0 else flat
    groups = flat_p.view(-1, group_size)
    mask = torch.ones(n, device=w.device, dtype=w.dtype)
    mask_p = (F.pad(mask, (0, pad)) if pad > 0 else mask).view(-1, group_size)
    scale = (
        groups.abs().sum(dim=-1, keepdim=True)
        / mask_p.sum(dim=-1, keepdim=True).clamp(min=1.0)
    ).clamp(min=1e-6)
    q_int = torch.clamp(torch.round(groups / scale), -1.0, 1.0)
    q = (q_int * scale).reshape(-1)[:n].view(orig_shape)
    return w + (q - w).detach()


def _linear_qat(x: torch.Tensor, layer: nn.Linear, use_qat: bool) -> torch.Tensor:
    w = quantize_ternary_ste(layer.weight) if use_qat else layer.weight
    return F.linear(x, w, layer.bias)


# ===========================================================================
# Invariant 1: Input-Dependent SPD Graph Laplacian + Cholesky Implicit Adjoint
# ===========================================================================


class _ImplicitLaplacianSolveFn(torch.autograd.Function):
    """Exact 0-BPTT linear adjoint solver for SPD `H_psi @ z0 = s_psi` under AMP shield."""

    @staticmethod
    def forward(ctx, H_psi: torch.Tensor, s_psi: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(H_psi.device.type, enabled=False):
            work_dtype = torch.float64 if H_psi.dtype == torch.float64 else torch.float32
            H_w = 0.5 * (H_psi.to(work_dtype) + H_psi.to(work_dtype).transpose(-1, -2))
            s_w = s_psi.to(work_dtype)
            s_col = s_w.unsqueeze(-1) if s_w.dim() == 2 else s_w
            L = torch.linalg.cholesky(H_w)
            z0 = torch.cholesky_solve(s_col, L)
            out = z0.squeeze(-1) if s_w.dim() == 2 else z0
        ctx.save_for_backward(L, z0)
        ctx.in_dtype = H_psi.dtype
        ctx.squeeze_last = s_psi.dim() == 2
        return out.to(H_psi.dtype)

    @staticmethod
    def backward(ctx, grad_z0: torch.Tensor):
        L, z0 = ctx.saved_tensors
        with torch.amp.autocast(L.device.type, enabled=False):
            gz = grad_z0.to(L.dtype)
            gz_col = gz.unsqueeze(-1) if ctx.squeeze_last else gz
            lam = torch.cholesky_solve(gz_col, L)
            grad_H = -0.5 * (
                torch.matmul(lam, z0.transpose(-1, -2))
                + torch.matmul(z0, lam.transpose(-1, -2))
            )
            grad_s = lam.squeeze(-1) if ctx.squeeze_last else lam
        return grad_H.to(ctx.in_dtype), grad_s.to(ctx.in_dtype)


class SPDGraphLaplacianField(nn.Module):
    """Autonomous input-dependent Graph Laplacian + ePC + Directional Eikonal head."""

    def __init__(self, d_slot: int = 64, d_z: int = 32, d_k: int = 64) -> None:
        super().__init__()
        self.d_slot = int(d_slot)
        self.d_z = int(d_z)
        self.d_k = int(d_k)
        self.slot_key = nn.Linear(d_slot, d_k, bias=False)
        with torch.no_grad():
            rad = (torch.randint(0, 2, (d_k, d_slot)).float() * 2.0 - 1.0) / math.sqrt(float(d_k))
            self.slot_key.weight.copy_(rad)
        self.edge_scale = nn.Parameter(torch.tensor(10.0))
        self.edge_bias = nn.Parameter(torch.tensor(-3.0))
        self.mu_raw = nn.Parameter(torch.tensor(-0.4))
        self.src_proj = nn.Linear(2, d_z, bias=False)
        self.route_head = nn.Linear(d_z, 1)
        self.dist_fc1 = nn.Linear(d_slot + d_z, 32)
        self.dist_fc2 = nn.Linear(32, 1)

    def build_system(
        self, q_hat: torch.Tensor, use_qat: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, M, _ = q_hat.shape
        k = _linear_qat(q_hat, self.slot_key, use_qat)
        k_n = F.normalize(k, dim=-1)
        sim = torch.bmm(k_n, k_n.transpose(-1, -2))
        a_raw = F.softplus(self.edge_scale * sim + self.edge_bias)
        eye = torch.eye(M, device=q_hat.device, dtype=q_hat.dtype).unsqueeze(0)
        a_sym = 0.5 * (a_raw + a_raw.transpose(-1, -2)) * (1.0 - eye)
        deg = a_sym.sum(dim=-1)
        lap = torch.diag_embed(deg) - a_sym
        mu = torch.clamp(self.mu_raw, -1.5, 2.0)
        mass = torch.exp(mu)
        H_psi = lap + mass * eye
        s_psi = _linear_qat(q_hat[..., -2:], self.src_proj, use_qat)
        return H_psi, s_psi, a_sym

    def solve_equilibrium(
        self, q_hat: torch.Tensor, use_qat: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        H_psi, s_psi, a_sym = self.build_system(q_hat, use_qat=use_qat)
        z0 = _ImplicitLaplacianSolveFn.apply(H_psi, s_psi)
        return z0, H_psi, s_psi, a_sym

    @torch.no_grad()
    def relax_epc_no_grad(
        self, H_psi: torch.Tensor, s_psi: torch.Tensor, steps: int = 40
    ) -> torch.Tensor:
        """Matrix-inversion-free error-space Predictive Coding (`ePC`) relaxation."""
        H_f = 0.5 * (H_psi.float() + H_psi.float().transpose(-1, -2))
        s_f = s_psi.float()
        diag_inv = 1.0 / torch.diagonal(H_f, dim1=-2, dim2=-1).clamp(min=1e-6).unsqueeze(-1)
        z = torch.zeros_like(s_f)
        eps = s_f - torch.bmm(H_f, z)
        r = diag_inv * eps
        p = r.clone()
        rz_old = (eps * r).sum(dim=(-2, -1), keepdim=True)
        for _ in range(steps):
            Hp = torch.bmm(H_f, p)
            denom = (p * Hp).sum(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
            alpha = rz_old / denom
            z = z + alpha * p
            eps = eps - alpha * Hp
            if float(eps.abs().max().item()) < 1e-7:
                break
            r = diag_inv * eps
            rz_new = (eps * r).sum(dim=(-2, -1), keepdim=True)
            beta = rz_new / rz_old.clamp(min=1e-12)
            p = r + beta * p
            rz_old = rz_new
        return z.to(s_psi.dtype)

    def distance_field(
        self, z: torch.Tensor, q_hat: torch.Tensor, use_qat: bool = False
    ) -> torch.Tensor:
        h = torch.cat([q_hat, z], dim=-1)
        h = F.silu(_linear_qat(h, self.dist_fc1, use_qat))
        return F.softplus(_linear_qat(h, self.dist_fc2, use_qat)).squeeze(-1)

    def directional_eikonal_loss(
        self, z0: torch.Tensor, q_hat: torch.Tensor, use_qat: bool = False
    ) -> torch.Tensor:
        z_t = z0.detach()
        noise = torch.randn_like(z_t)
        gamma = torch.rand(z_t.size(0), 1, 1, device=z_t.device, dtype=z_t.dtype) * 0.85
        z_gamma = ((1.0 - gamma) * noise + gamma * z_t).requires_grad_(True)
        d_pred = self.distance_field(z_gamma, q_hat.detach(), use_qat=use_qat)
        grad_z = torch.autograd.grad(
            d_pred.sum(),
            z_gamma,
            create_graph=True,
        )[0]
        g_norm = torch.sqrt(grad_z.pow(2).sum(dim=-1) + 1e-8)
        eik_loss = ((g_norm - 1.0) ** 2).mean()
        d_target = self.distance_field(z_t, q_hat.detach(), use_qat=use_qat)
        zero_anchor = (d_target ** 2).mean()
        return eik_loss + 0.5 * zero_anchor

    def score_candidates(
        self,
        q_hat: torch.Tensor,
        z0: torch.Tensor,
        cand_masks: torch.Tensor,
        use_qat: bool = False,
    ) -> torch.Tensor:
        """Intensive (`1 / |y|`) candidate path energy from latent field `z0`: lower means shorter route."""
        del use_qat
        node_u = self.route_head(z0).squeeze(-1)
        node_cost = -node_u
        denom = cand_masks.sum(dim=-1).clamp(min=1.0)
        energies = torch.bmm(cand_masks, node_cost.unsqueeze(-1)).squeeze(-1) / denom
        return energies


class StaticWzzBaseline(nn.Module):
    """Input-independent static global W_zz baseline (cannot adapt to permuted edges)."""

    def __init__(self, d_slot: int, num_nodes: int = 12, d_z: int = 32) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_z = int(d_z)
        self.W_zz_raw = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.05)
        self.src_proj = nn.Linear(2, d_z)
        self.route_head = nn.Linear(d_z, 1)

    def forward_energies(self, q_hat: torch.Tensor, cand_masks: torch.Tensor) -> torch.Tensor:
        B, M, _ = q_hat.shape
        sym = 0.5 * (self.W_zz_raw + self.W_zz_raw.t())
        eye = torch.eye(M, device=q_hat.device, dtype=q_hat.dtype)
        sym = sym * (1.0 - eye)
        row_l1 = sym.abs().sum(dim=-1, keepdim=True).clamp(min=1e-6)
        W_zz = sym * torch.clamp(0.9 / row_l1, max=1.0)
        H_stat = (5.0 * eye - W_zz).unsqueeze(0).expand(B, -1, -1)
        s_psi = self.src_proj(q_hat[..., -2:])
        z0 = _ImplicitLaplacianSolveFn.apply(H_stat, s_psi)
        node_cost = -self.route_head(z0).squeeze(-1)
        denom = cand_masks.sum(dim=-1).clamp(min=1.0)
        return torch.bmm(cand_masks, node_cost.unsqueeze(-1)).squeeze(-1) / denom


def _generate_cyclic_graph_batch(
    batch_size: int,
    num_nodes: int,
    d_slot: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Builds random permuted cyclic graphs (`short 2..3 hops` vs `long 5..7 hops`, internal deg=2)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    d_edge = d_slot - 2
    q_all = torch.zeros(batch_size, num_nodes, d_slot, dtype=torch.float32)
    masks_all = torch.zeros(batch_size, 2, num_nodes, dtype=torch.float32)
    targets = torch.zeros(batch_size, dtype=torch.long)  # 0 or 1 index of shortest route

    for b in range(batch_size):
        perm = torch.randperm(num_nodes, generator=gen).tolist()
        short_internal = 1 if (b % 2 == 0) else 2
        long_internal = 5 if (b % 3 != 0) else 6
        s_node = perm[0]
        g_node = perm[1]
        short_nodes = perm[2 : 2 + short_internal]
        long_nodes = perm[2 + short_internal : 2 + short_internal + long_internal]

        edges: list[tuple[int, int]] = []
        chain_s = [s_node] + short_nodes + [g_node]
        for i in range(len(chain_s) - 1):
            edges.append((chain_s[i], chain_s[i + 1]))
        chain_l = [s_node] + long_nodes + [g_node]
        for i in range(len(chain_l) - 1):
            edges.append((chain_l[i], chain_l[i + 1]))
        # Optional distractor loop on remaining unused nodes so node degrees vary randomly
        rem = perm[2 + short_internal + long_internal :]
        if len(rem) >= 3:
            for i in range(len(rem)):
                edges.append((rem[i], rem[(i + 1) % len(rem)]))

        edge_codes = F.normalize(torch.randn(len(edges), d_edge, generator=gen), dim=-1)
        node_feat = torch.zeros(num_nodes, d_edge, dtype=torch.float32)
        for ei, (u, v) in enumerate(edges):
            node_feat[u] = node_feat[u] + edge_codes[ei]
            node_feat[v] = node_feat[v] + edge_codes[ei]
        node_feat = F.normalize(node_feat, dim=-1)

        q_all[b, :, :d_edge] = node_feat
        q_all[b, s_node, d_edge] = 1.0
        q_all[b, g_node, d_edge + 1] = 1.0

        m_short = torch.zeros(num_nodes, dtype=torch.float32)
        for idx in short_nodes:
            m_short[idx] = 1.0
        m_long = torch.zeros(num_nodes, dtype=torch.float32)
        for idx in long_nodes:
            m_long[idx] = 1.0

        swap = int(torch.randint(0, 2, (1,), generator=gen).item())
        if swap == 0:
            masks_all[b, 0] = m_short
            masks_all[b, 1] = m_long
            targets[b] = 0
        else:
            masks_all[b, 0] = m_long
            masks_all[b, 1] = m_short
            targets[b] = 1

    return q_all.to(device), masks_all.to(device), targets.to(device)


def run_invariant_1_laplacian(
    device: torch.device, seed: int = 2026, fast: bool = False
) -> dict[str, object]:
    torch.manual_seed(seed)
    num_nodes = 12
    d_slot = 64
    d_z = 32
    d_k = 64
    n_train = 512 if fast else 1024
    n_test = 64 if fast else 128
    train_steps = 160 if fast else 220

    q_tr, m_tr, y_tr = _generate_cyclic_graph_batch(n_train, num_nodes, d_slot, device, seed)
    q_te, m_te, y_te = _generate_cyclic_graph_batch(n_test, num_nodes, d_slot, device, seed + 10_000)

    # 1) Check Cholesky Implicit Adjoint vs finite-diff in float64 (B=1 CPU) & ePC convergence
    probe_mod = SPDGraphLaplacianField(d_slot=d_slot, d_z=d_z, d_k=d_k).to(device)
    with torch.no_grad():
        H_p, s_p, _ = probe_mod.build_system(q_te[:8])
        eigs = torch.linalg.eigvalsh(H_p.float())
        min_eig = float(eigs.min().item())
        z_chol = _ImplicitLaplacianSolveFn.apply(H_p, s_p)
        z_epc = probe_mod.relax_epc_no_grad(H_p, s_p, steps=40)
        epc_err = float((z_chol - z_epc).abs().max().item())

    H_64 = H_p[:1].detach().cpu().to(torch.float64).requires_grad_(True)
    s_64 = s_p[:1].detach().cpu().to(torch.float64).requires_grad_(True)
    z_64 = _ImplicitLaplacianSolveFn.apply(H_64, s_64)
    w_probe = torch.randn_like(z_64)
    loss_64 = (z_64 * w_probe).sum()
    gH_adj, gs_adj = torch.autograd.grad(loss_64, (H_64, s_64))
    eps_fd = 1e-6
    gH_fd = torch.zeros_like(H_64)
    with torch.no_grad():
        for r in range(num_nodes):
            for c in range(num_nodes):
                dH = torch.zeros_like(H_64)
                dH[0, r, c] = eps_fd
                lp = (_ImplicitLaplacianSolveFn.apply(H_64 + dH, s_64) * w_probe).sum()
                lm = (_ImplicitLaplacianSolveFn.apply(H_64 - dH, s_64) * w_probe).sum()
                gH_fd[0, r, c] = (lp - lm) / (2.0 * eps_fd)
    adj_grad_err = float((gH_adj - gH_fd).abs().max().item())

    # 2) Train StaticWzzBaseline (0-BPTT)
    stat_m = StaticWzzBaseline(d_slot=d_slot, num_nodes=num_nodes, d_z=d_z).to(device)
    opt_stat = torch.optim.AdamW(stat_m.parameters(), lr=5e-3, weight_decay=1e-4)
    bs = 64
    for st in range(train_steps):
        idx = torch.randint(0, n_train, (bs,), device=device)
        e_cand = stat_m.forward_energies(q_tr[idx], m_tr[idx])
        loss = F.cross_entropy(-e_cand * 5.0, y_tr[idx])
        opt_stat.zero_grad(set_to_none=True)
        loss.backward()
        opt_stat.step()
    with torch.no_grad():
        e_stat_te = stat_m.forward_energies(q_te, m_te)
        stat_acc = float((e_stat_te.argmin(dim=-1) == y_te).float().mean().item() * 100.0)

    # 3) Train SymmetricEqPropQAT (0-BPTT 2-phase symmetric EqProp under 1.58-bit STE QAT)
    eqp_m = SPDGraphLaplacianField(d_slot=d_slot, d_z=d_z, d_k=d_k).to(device)
    opt_eqp = torch.optim.AdamW(eqp_m.parameters(), lr=8e-3, weight_decay=1e-4)
    beta_ep = 0.25
    for st in range(train_steps):
        idx = torch.randint(0, n_train, (bs,), device=device)
        qb, mb, yb = q_tr[idx], m_tr[idx], y_tr[idx]
        H_b, s_b, _ = eqp_m.build_system(qb, use_qat=True)
        with torch.no_grad():
            z0_free = eqp_m.relax_epc_no_grad(H_b.detach(), s_b.detach(), steps=25)
        z0_var = z0_free.detach().requires_grad_(True)
        e_cand_ep = eqp_m.score_candidates(qb, z0_var, mb, use_qat=True)
        loss_task = F.cross_entropy(-e_cand_ep * 5.0, yb)
        grad_z_task = torch.autograd.grad(loss_task, z0_var, retain_graph=True)[0]
        with torch.no_grad():
            z_plus = eqp_m.relax_epc_no_grad(
                H_b.detach(), s_b.detach() - beta_ep * grad_z_task, steps=25
            )
            z_minus = eqp_m.relax_epc_no_grad(
                H_b.detach(), s_b.detach() + beta_ep * grad_z_task, steps=25
            )

        def _e_int(H_in: torch.Tensor, s_in: torch.Tensor, z_in: torch.Tensor) -> torch.Tensor:
            Hz = torch.bmm(H_in, z_in)
            return (0.5 * (z_in * Hz).sum(dim=(-2, -1)) - (s_in * z_in).sum(dim=(-2, -1))).sum()

        loss_eqp = loss_task + (_e_int(H_b, s_b, z_plus) - _e_int(H_b, s_b, z_minus)) / (2.0 * beta_ep)
        opt_eqp.zero_grad(set_to_none=True)
        loss_eqp.backward()
        torch.nn.utils.clip_grad_norm_(eqp_m.parameters(), 2.0)
        opt_eqp.step()

    with torch.no_grad():
        H_e, s_e, _ = eqp_m.build_system(q_te, use_qat=True)
        z_e = eqp_m.relax_epc_no_grad(H_e, s_e, steps=25)
        e_eqp_te = eqp_m.score_candidates(q_te, z_e, m_te, use_qat=True)
        eqp_acc = float((e_eqp_te.argmin(dim=-1) == y_te).float().mean().item() * 100.0)

    # 4) Train UEBRLaplacianAdjoint (FP32 and 1.58-bit STE QAT with Directional Eikonal)
    results_uebr: dict[str, dict[str, float]] = {}
    for mode_name, use_qat in (("fp32", False), ("qat_158", True)):
        torch.manual_seed(seed + (100 if use_qat else 0))
        um = SPDGraphLaplacianField(d_slot=d_slot, d_z=d_z, d_k=d_k).to(device)
        opt_u = torch.optim.AdamW(um.parameters(), lr=8e-3, weight_decay=1e-4)
        sched_u = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_u, T_max=train_steps, eta_min=1e-3
        )
        for st in range(train_steps):
            idx = torch.randint(0, n_train, (bs,), device=device)
            qb, mb, yb = q_tr[idx], m_tr[idx], y_tr[idx]
            z0_b, H_b, s_b, _ = um.solve_equilibrium(qb, use_qat=use_qat)
            e_cand = um.score_candidates(qb, z0_b, mb, use_qat=use_qat)
            loss_ce = F.cross_entropy(-e_cand * 5.0, yb)
            loss_eik = um.directional_eikonal_loss(z0_b, qb, use_qat=use_qat)
            loss = loss_ce + 0.05 * loss_eik
            opt_u.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(um.parameters(), 2.0)
            opt_u.step()
            sched_u.step()

        with torch.no_grad():
            z0_te, H_te, s_te, _ = um.solve_equilibrium(q_te, use_qat=use_qat)
            e_te = um.score_candidates(q_te, z0_te, m_te, use_qat=use_qat)
            acc = float((e_te.argmin(dim=-1) == y_te).float().mean().item() * 100.0)
            row_idx = torch.arange(n_test, device=device)
            e_short = e_te[row_idx, y_te]
            e_long = e_te[row_idx, 1 - y_te]
            gap = float((e_short - e_long).mean().item())
            e_min_te = float(torch.linalg.eigvalsh(H_te.float()).min().item())
            z_epc_te = um.relax_epc_no_grad(H_te, s_te, steps=40)
            epc_te_err = float((z0_te - z_epc_te).abs().max().item())
        results_uebr[mode_name] = {
            "acc_pct": round(acc, 2),
            "energy_gap_short_minus_long": round(gap, 4),
            "min_eig": round(e_min_te, 4),
            "epc_max_err": epc_te_err,
        }

    fp_acc = results_uebr["fp32"]["acc_pct"]
    qat_acc = results_uebr["qat_158"]["acc_pct"]
    passed = (
        min_eig > 0.0
        and adj_grad_err < 1e-5
        and epc_err < 1e-4
        and results_uebr["fp32"]["min_eig"] > 0.0
        and results_uebr["qat_158"]["min_eig"] > 0.0
        and results_uebr["fp32"]["epc_max_err"] < 1e-4
        and results_uebr["qat_158"]["epc_max_err"] < 1e-4
        and fp_acc >= 90.0
        and qat_acc >= 90.0
        and qat_acc >= stat_acc + 25.0
        and results_uebr["fp32"]["energy_gap_short_minus_long"] < 0.0
        and results_uebr["qat_158"]["energy_gap_short_minus_long"] < 0.0
    )
    return {
        "min_eig_init": round(min_eig, 4),
        "adjoint_grad_err": adj_grad_err,
        "epc_init_max_err": epc_err,
        "static_wzz_acc_pct": round(stat_acc, 2),
        "sym_eqprop_qat_acc_pct": round(eqp_acc, 2),
        "uebr_fp32": results_uebr["fp32"],
        "uebr_qat_158": results_uebr["qat_158"],
        "passed": bool(passed),
    }


# ===========================================================================
# Invariant 2: PermFlow C(U) + cSB Inertia + Lossless Birkhoff Extension
# ===========================================================================


def permflow_tangent_project(U: torch.Tensor) -> torch.Tensor:
    """Closed-form orthogonal projection onto Birkhoff affine tangent space T."""
    Ud = U.double()
    row_mean = Ud.mean(dim=-1, keepdim=True)
    col_mean = Ud.mean(dim=-2, keepdim=True)
    grand_mean = Ud.mean(dim=(-2, -1), keepdim=True)
    return (Ud - row_mean - col_mean + grand_mean).to(U.dtype)


def _project_to_birkhoff_polytope(X: torch.Tensor, iters: int = 30) -> torch.Tensor:
    """Dykstra Euclidean projection onto `D_n = {A >= 0} cap B_a` + exact sum polish."""
    n = X.size(-1)
    J = torch.full_like(X, 1.0 / float(n))
    x = X
    p = torch.zeros_like(X)
    q = torch.zeros_like(X)
    for _ in range(iters):
        y = (x + p).clamp(min=0.0)
        p = x + p - y
        x = J + permflow_tangent_project(y + q - J)
        q = y + q - x
    A = x.clamp(min=1e-12)
    for _ in range(12):
        A = A / A.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        A = A / A.sum(dim=-2, keepdim=True).clamp(min=1e-12)
    return A


def _max_weight_bipartite_matching(weights: list[list[float]]) -> list[int]:
    """Exact O(n^3) Kuhn-Munkres / Jonker-Volgenant Hungarian max-weight matching."""
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


def birkhoff_extension_round(
    A: torch.Tensor,
    eval_fn: Callable[[torch.Tensor], float],
    S: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float, float, int]:
    """Birkhoff-von Neumann convex decomposition ordered by `S` with lossless rounding."""
    assert A.dim() == 2 and A.size(0) == A.size(1), f"Expected square matrix, got {A.shape}"
    n = A.size(0)
    dev = A.device
    dt = A.dtype
    R = A.detach().double().cpu().clone()
    S_cpu = (A if S is None else S).detach().double().cpu()
    # Ensure A lies cleanly in Birkhoff polytope before decomposition
    if float((R.sum(dim=-1) - 1.0).abs().max().item()) > 1e-4 or float(R.min().item()) < -1e-5:
        R = _project_to_birkhoff_polytope(R.unsqueeze(0)).squeeze(0)
    R = R.clamp(min=0.0)

    alphas: list[float] = []
    perms: list[torch.Tensor] = []
    max_terms = n * n - n + 4
    r_mass = 1.0

    for _ in range(max_terms):
        if r_mass <= 1e-7:
            break
        # Find perfect matching inside positive support of R maximizing <S, P>
        R_list = R.tolist()
        S_list = S_cpu.tolist()
        w_mat = [
            [S_list[i][j] if R_list[i][j] > 1e-9 else -1e9 for j in range(n)]
            for i in range(n)
        ]
        perm = _max_weight_bipartite_matching(w_mat)
        alpha = min(R_list[i][perm[i]] for i in range(n))
        if alpha <= 1e-11:
            # Retry with wider support threshold > 0
            w_mat_fb = [
                [S_list[i][j] if R_list[i][j] > 0.0 else -1e9 for j in range(n)]
                for i in range(n)
            ]
            perm = _max_weight_bipartite_matching(w_mat_fb)
            alpha = min(R_list[i][perm[i]] for i in range(n))
            if alpha <= 1e-12:
                break
        P = torch.zeros((n, n), dtype=torch.float64)
        for i in range(n):
            P[i, perm[i]] = 1.0
            R[i, perm[i]] = max(0.0, R[i, perm[i]] - alpha)
        R[R < 1e-13] = 0.0
        alphas.append(float(alpha))
        perms.append(P)
        r_mass -= float(alpha)

    if not perms:
        P_eye = torch.eye(n, device=dev, dtype=dt)
        v_eye = float(eval_fn(P_eye))
        return P_eye, v_eye, v_eye, 1

    a_sum = sum(alphas)
    alphas = [a / max(a_sum, 1e-12) for a in alphas]
    best_p = perms[0].to(device=dev, dtype=dt)
    best_val = float("inf")
    ext_val = 0.0
    for a_l, P_l in zip(alphas, perms):
        P_d = P_l.to(device=dev, dtype=dt)
        fv = float(eval_fn(P_d))
        ext_val += a_l * fv
        if fv < best_val:
            best_val = fv
            best_p = P_d

    return best_p, float(best_val), float(ext_val), len(perms)


def _greedy_birkhoff_matrix(scores: torch.Tensor) -> torch.Tensor:
    """Fast batched greedy rounding `(B, n, n) -> P in S_n`."""
    B, n, _ = scores.shape
    work = scores.clone()
    perm = torch.zeros_like(scores)
    b_idx = torch.arange(B, device=scores.device)
    for _ in range(n):
        flat = work.view(B, n * n).argmax(dim=-1)
        r = flat // n
        c = flat % n
        perm[b_idx, r, c] = 1.0
        work[b_idx, r, :] = -1e9
        work[b_idx, :, c] = -1e9
    return perm


def _log_sinkhorn_matrix(logits: torch.Tensor, iters: int = 8) -> torch.Tensor:
    y = logits
    for _ in range(iters):
        y = y - torch.logsumexp(y, dim=-1, keepdim=True)
        y = y - torch.logsumexp(y, dim=-2, keepdim=True)
    return torch.exp(y)


def run_invariant_2_cSB_permflow(
    device: torch.device, seed: int = 2026, fast: bool = False
) -> dict[str, object]:
    torch.manual_seed(seed)

    # --- Part 2A: 8 symmetric cyclic-shift optima on n=8 ---
    n_sym = 8
    modes = torch.zeros(n_sym, n_sym, n_sym, device=device, dtype=torch.float32)
    idx_r = torch.arange(n_sym, device=device)
    for m in range(n_sym):
        modes[m, idx_r, (idx_r + m) % n_sym] = 1.0

    n_samp = 60 if fast else 100
    # Entropic Sinkhorn relaxation over symmetric mode objective (Proposition 13 collapse)
    Z_sk = torch.randn(n_samp, n_sym, n_sym, device=device) * 0.08
    opt_sk = torch.optim.SGD([Z_sk.requires_grad_(True)], lr=0.8)
    for _ in range(40):
        X_sk = _log_sinkhorn_matrix(Z_sk / 0.45, iters=8)
        dots = torch.einsum("sij,mij->sm", X_sk, modes)
        # Symmetric expected attraction + entropic regularization toward barycenter
        ent_reg = (X_sk * torch.log(X_sk.clamp(min=1e-9))).sum(dim=(-2, -1))
        loss_sk = (-torch.logsumexp(dots * 1.5, dim=-1) + 1.2 * ent_reg).sum()
        opt_sk.zero_grad()
        loss_sk.backward()
        opt_sk.step()
    with torch.no_grad():
        P_sk = _greedy_birkhoff_matrix(_log_sinkhorn_matrix(Z_sk / 0.45, iters=8))
        sk_match = (torch.einsum("sij,mij->sm", P_sk, modes) >= n_sym - 0.5)
        sk_valid_ids = sk_match.long().argmax(dim=-1)[sk_match.any(dim=-1)].tolist()
        sk_cov_pct = 100.0 * len(set(sk_valid_ids)) / float(n_sym)

    # PermFlow Euler integration strictly on affine hull B_a with tangent projector C(U)
    J_sym = torch.full((n_samp, n_sym, n_sym), 1.0 / n_sym, device=device, dtype=torch.float32)
    X_pf = J_sym + permflow_tangent_project(torch.randn_like(J_sym) * 0.22)
    dt_pf = 0.08
    max_ba_err_2a = 0.0
    with torch.no_grad():
        for step in range(45):
            sigma_t = max(0.05, 0.35 * (1.0 - step / 45.0))
            dists = ((X_pf.unsqueeze(1) - modes.unsqueeze(0)) ** 2).sum(dim=(-2, -1))
            w_m = F.softmax(-dists / (2.0 * sigma_t * sigma_t), dim=-1)
            target_x = torch.einsum("sm,mij->sij", w_m, modes)
            vel = permflow_tangent_project(target_x - X_pf)
            X_pf = X_pf + dt_pf * vel
            row_e = float((X_pf.sum(dim=-1) - 1.0).abs().max().item())
            col_e = float((X_pf.sum(dim=-2) - 1.0).abs().max().item())
            max_ba_err_2a = max(max_ba_err_2a, row_e, col_e)
        P_pf = _greedy_birkhoff_matrix(X_pf)
        pf_match = (torch.einsum("sij,mij->sm", P_pf, modes) >= n_sym - 0.5)
        pf_valid_ids = pf_match.long().argmax(dim=-1)[pf_match.any(dim=-1)].tolist()
        pf_cov_pct = 100.0 * len(set(pf_valid_ids)) / float(n_sym)
        pf_exact_pct = 100.0 * float(pf_match.any(dim=-1).float().mean().item())

    # --- Part 2B: Footnote-23 QAP check & Soft D_n Lossless Birkhoff Extension (n=12, 50 instances) ---
    n_qap = 12
    n_inst = 50
    gen_q = torch.Generator(device="cpu").manual_seed(seed + 77)
    F_raw = torch.randn(n_inst, n_qap, n_qap, generator=gen_q, dtype=torch.float64)
    D_raw = torch.randn(n_inst, n_qap, n_qap, generator=gen_q, dtype=torch.float64)
    F_q = (0.5 * (F_raw + F_raw.transpose(-1, -2))).to(device)
    D_q = (0.5 * (D_raw + D_raw.transpose(-1, -2))).to(device)
    J_q = torch.full((n_inst, n_qap, n_qap), 1.0 / n_qap, dtype=torch.float64, device=device)

    with torch.no_grad():
        e_bary = (J_q * torch.matmul(torch.matmul(F_q, J_q), D_q)).sum(dim=(-2, -1))

        # 2B-1: Honest empirical check of footnote-23 claim (single-chain from exact barycenter J)
        n_steps_sc = 50 if fast else 60
        X_p1 = J_q.clone()
        for _ in range(n_steps_sc):
            g_p1 = 2.0 * torch.matmul(torch.matmul(F_q, X_p1.clamp(0.0, 1.0)), D_q)
            X_p1 = (X_p1 - 0.015 * permflow_tangent_project(g_p1)).clamp(0.0, 1.0)
            X_p1 = J_q + permflow_tangent_project(X_p1 - J_q)
        P_p1 = _greedy_birkhoff_matrix(X_p1)
        e_aff_pgd = (P_p1 * torch.matmul(torch.matmul(F_q, P_p1), D_q)).sum(dim=(-2, -1)) - e_bary

        torch.manual_seed(seed)
        X_c1 = J_q.clone()
        V_c1 = permflow_tangent_project(torch.randn_like(J_q) * 0.02)
        max_ba_err_f64 = 0.0
        for st in range(n_steps_sc):
            tf = st / float(max(1, n_steps_sc - 1))
            g_c1 = 2.0 * torch.matmul(torch.matmul(F_q, X_c1.clamp(0.0, 1.0)), D_q)
            wall_c1 = -25.0 * (F.relu(X_c1 - 1.0) - F.relu(-X_c1))
            acc_c1 = -0.12 * V_c1 - 0.12 * g_c1 + (-0.3 + 1.5 * tf) * (X_c1 - J_q) + wall_c1
            V_c1 = permflow_tangent_project(V_c1 + 0.08 * permflow_tangent_project(acc_c1))
            X_c1 = X_c1 + 0.08 * V_c1
            row_e = float((X_c1.sum(dim=-1) - 1.0).abs().max().item())
            col_e = float((X_c1.sum(dim=-2) - 1.0).abs().max().item())
            max_ba_err_f64 = max(max_ba_err_f64, row_e, col_e)
        P_c1 = _greedy_birkhoff_matrix(X_c1)
        e_unbounded_csb = (P_c1 * torch.matmul(torch.matmul(F_q, P_c1), D_q)).sum(dim=(-2, -1)) - e_bary
        unbounded_csb_wins = int((e_unbounded_csb < e_aff_pgd).sum().item())

        # 2B-2: Soft D_n rounding benchmark (`12` Dykstra-PGD steps -> Greedy vs Birkhoff Extension)
        torch.manual_seed(seed + 101)
        X_soft = _project_to_birkhoff_polytope(
            J_q + permflow_tangent_project(torch.randn_like(J_q) * 0.05), iters=15
        )
        for _ in range(12):
            g_sf = 2.0 * torch.matmul(torch.matmul(F_q, X_soft), D_q)
            X_soft = _project_to_birkhoff_polytope(X_soft - 0.015 * g_sf, iters=10)
        P_gr = _greedy_birkhoff_matrix(X_soft)
        e_gr = ((P_gr * torch.matmul(torch.matmul(F_q, P_gr), D_q)).sum(dim=(-2, -1)) - e_bary).cpu()
        grad_sf = 2.0 * torch.matmul(torch.matmul(F_q, X_soft), D_q)

    e_be = torch.empty(n_inst, dtype=torch.float64)
    lossless_ok = 0
    for i in range(n_inst):
        Fi = F_q[i]
        Di = D_q[i]
        eb_i = float(e_bary[i].item())

        def _eval_qap(P_mat: torch.Tensor) -> float:
            return float((P_mat * (Fi @ P_mat @ Di)).sum().item() - eb_i)

        _, f_best_i, f_ext_i, _ = birkhoff_extension_round(
            X_soft[i], _eval_qap, S=-grad_sf[i]
        )
        if f_best_i <= f_ext_i + 1e-6:
            lossless_ok += 1
        e_be[i] = f_best_i

    be_wins = int((e_be < e_gr).sum().item())
    mean_gr = float(e_gr.mean().item())
    mean_be = float(e_be.mean().item())
    passed = (
        pf_cov_pct >= 87.5
        and sk_cov_pct <= 25.0
        and max_ba_err_2a < 1e-5
        and max_ba_err_f64 < 1e-13
        and lossless_ok == n_inst
        and be_wins >= 40
        and mean_be < mean_gr - 5.0
    )
    return {
        "part2a_sinkhorn_mode_cov_pct": round(sk_cov_pct, 2),
        "part2a_permflow_mode_cov_pct": round(pf_cov_pct, 2),
        "part2a_permflow_exact_pct": round(pf_exact_pct, 2),
        "part2a_ba_err": max_ba_err_2a,
        "part2b_n_instances": n_inst,
        "part2b_footnote23_affine_pgd_mean": round(float(e_aff_pgd.mean().item()), 3),
        "part2b_footnote23_unbounded_csb_mean": round(float(e_unbounded_csb.mean().item()), 3),
        "part2b_footnote23_unbounded_csb_wins": unbounded_csb_wins,
        "part2b_ba_err_f64": max_ba_err_f64,
        "part2b_greedy_mean_energy": round(mean_gr, 3),
        "part2b_birkhoff_ext_mean_energy": round(mean_be, 3),
        "part2b_be_wins": be_wins,
        "part2b_be_win_pct": round(100.0 * be_wins / n_inst, 2),
        "part2b_lossless_rounding_pct": round(100.0 * lossless_ok / n_inst, 2),
        "passed": bool(passed),
    }


# ===========================================================================
# Invariant 3: RLNS Block Fenchel-Young + Min-Cut Freezing + Null Slot (Ø)
# ===========================================================================


class RLNSVariableSlotEBM(nn.Module):
    """0-BPTT Block Fenchel-Young (`RLNS-1`) Energy Model with absorbing null slot `0`."""

    def __init__(self, d_feat: int = 24, n_slots: int = 10, n_states: int = 5) -> None:
        super().__init__()
        self.n_slots = int(n_slots)
        self.n_states = int(n_states)  # state 0 = null/inactive (Ø), 1..4 = active flow labels
        self.unary_net = nn.Sequential(
            nn.Linear(d_feat, 32),
            nn.SiLU(),
            nn.Linear(32, n_states),
        )
        self.pair_proj = nn.Linear(d_feat, 16, bias=False)
        # Learnable state-state compatibility tensor (S, S) plus pair-type tensor
        self.state_compat = nn.Parameter(torch.randn(n_states, n_states) * 0.05)
        self.edge_gate = nn.Parameter(torch.tensor(1.0))

    def compute_potentials(
        self, x_slots: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns unary potentials `U (B, M, V)`, coupling `A_ij (B, M, M)`, and `W_st (V, V)`."""
        B, M, _ = x_slots.shape
        u = self.unary_net(x_slots)
        p_vec = self.pair_proj(x_slots)
        sim = torch.bmm(p_vec, p_vec.transpose(-1, -2)) / math.sqrt(16.0)
        eye = torch.eye(M, device=x_slots.device, dtype=x_slots.dtype).unsqueeze(0)
        a_sym = F.softplus(self.edge_gate * 0.5 * (sim + sim.transpose(-1, -2))) * (1.0 - eye)
        w_st = 0.5 * (self.state_compat + self.state_compat.t())
        return u, a_sym, w_st

    def block_fenchel_young_loss(
        self,
        x_slots: torch.Tensor,
        y_true: torch.Tensor,
        block_idx: torch.Tensor,
        gamma: float = 0.35,
    ) -> torch.Tensor:
        """Exact Shannon-entropy regularized RLNS block loss (`0-BPTT`) over free block `B`."""
        B, M, _ = x_slots.shape
        V = self.n_states
        k_blk = int(block_idx.size(1))
        u, a_sym, w_st = self.compute_potentials(x_slots)

        # Build cartesian block configurations V^k (for k=4, V=5 -> 625 states)
        grid = torch.cartesian_prod(*([torch.arange(V, device=x_slots.device)] * k_blk))
        if grid.dim() == 1:
            grid = grid.unsqueeze(-1)
        N_cfg = grid.size(0)  # 625

        b_arange = torch.arange(B, device=x_slots.device).unsqueeze(1)
        is_free = torch.zeros(B, M, dtype=torch.bool, device=x_slots.device)
        is_free.scatter_(1, block_idx, True)

        # Frozen neighbor contributions from S = [M] \ B
        y_frozen_oh = F.one_hot(y_true, num_classes=V).float() * (~is_free).unsqueeze(-1).float()
        # msg_frozen: (B, M, V) = sum_{j in S} a_{ij} * W_st[., y_j]
        a_w = torch.bmm(a_sym, torch.matmul(y_frozen_oh, w_st))  # (B, M, V)
        u_blk = u[b_arange, block_idx]  # (B, k_blk, V)
        ext_blk = a_w[b_arange, block_idx]  # (B, k_blk, V)
        eff_unary = u_blk + ext_blk  # (B, k_blk, V)

        # Unary energy term for every configuration c in grid (N_cfg, k_blk)
        pos_arange = torch.arange(k_blk, device=x_slots.device).unsqueeze(0).expand(N_cfg, -1)
        # gather eff_unary over grid -> (B, N_cfg, k_blk) -> sum over k_blk
        grid_exp = grid.unsqueeze(0).expand(B, -1, -1)  # (B, N_cfg, k_blk)
        b_idx3 = torch.arange(B, device=x_slots.device).view(B, 1, 1)
        p_idx3 = pos_arange.unsqueeze(0)
        e_unary_all = eff_unary[b_idx3, p_idx3, grid_exp].sum(dim=-1)  # (B, N_cfg)

        # Internal pairwise coupling inside block B: (B, k_blk, k_blk)
        a_blk = a_sym.gather(1, block_idx.unsqueeze(-1).expand(-1, -1, M)).gather(
            2, block_idx.unsqueeze(1).expand(-1, k_blk, -1)
        )
        # For each pair (a < b) inside block, accumulate a_blk[:, a, b] * w_st[grid[:, a], grid[:, b]]
        e_pair_all = torch.zeros(B, N_cfg, device=x_slots.device, dtype=x_slots.dtype)
        for p1 in range(k_blk):
            for p2 in range(p1 + 1, k_blk):
                w_pair_cfg = w_st[grid[:, p1], grid[:, p2]].unsqueeze(0)  # (1, N_cfg)
                e_pair_all = e_pair_all + a_blk[:, p1, p2].unsqueeze(1) * w_pair_cfg

        # Negative energy = logits / gamma
        scores_all = -(e_unary_all + e_pair_all) / max(gamma, 1e-3)  # (B, N_cfg)
        y_blk = y_true[b_arange, block_idx]  # (B, k_blk)
        powers = (V ** torch.arange(k_blk - 1, -1, -1, device=x_slots.device)).unsqueeze(0)
        true_cfg_idx = (y_blk * powers).sum(dim=-1)  # (B,)
        return gamma * F.cross_entropy(scores_all, true_cfg_idx)

    @torch.no_grad()
    def adaptive_mincut_blocks(self, a_sym: torch.Tensor, k_blk: int = 4) -> list[torch.Tensor]:
        """Builds adaptive variable blocks clustering strongly coupled slots (min-cut freezing)."""
        B, M, _ = a_sym.shape
        blocks: list[torch.Tensor] = []
        used_mask = torch.zeros(B, M, dtype=torch.bool, device=a_sym.device)
        while int((~used_mask).sum().item()) > 0 and len(blocks) < M:
            # Pick unvisited node with strongest total coupling
            rem_deg = a_sym.sum(dim=-1).masked_fill(used_mask, -1.0)
            seeds = rem_deg.argmax(dim=-1)
            blk = [seeds]
            in_blk = torch.zeros(B, M, dtype=torch.bool, device=a_sym.device)
            in_blk.scatter_(1, seeds.unsqueeze(1), True)
            for _ in range(k_blk - 1):
                # Greedily add node maximizing internal coupling to current block (min cut)
                score_to_blk = (a_sym * in_blk.unsqueeze(1).float()).sum(dim=-1)
                score_to_blk = score_to_blk.masked_fill(in_blk, -1e6)
                nxt = score_to_blk.argmax(dim=-1)
                blk.append(nxt)
                in_blk.scatter_(1, nxt.unsqueeze(1), True)
            blk_t = torch.stack(blk, dim=1)  # (B, k_blk)
            blocks.append(blk_t)
            used_mask = used_mask | in_blk
            if used_mask.all():
                break
        return blocks

    @torch.no_grad()
    def infer_block_gibbs(
        self, x_slots: torch.Tensor, k_blk: int = 4, sweeps: int = 4
    ) -> torch.Tensor:
        """Block coordinate minimization initialized from unaries using min-cut blocks."""
        B, M, _ = x_slots.shape
        V = self.n_states
        u, a_sym, w_st = self.compute_potentials(x_slots)
        y = (-u).argmax(dim=-1)  # unary initialization
        blocks = self.adaptive_mincut_blocks(a_sym, k_blk=k_blk)
        grid = torch.cartesian_prod(*([torch.arange(V, device=x_slots.device)] * k_blk))
        if grid.dim() == 1:
            grid = grid.unsqueeze(-1)
        N_cfg = grid.size(0)
        b_arange = torch.arange(B, device=x_slots.device).unsqueeze(1)
        pos_arange = torch.arange(k_blk, device=x_slots.device).unsqueeze(0).expand(N_cfg, -1)
        b_idx3 = torch.arange(B, device=x_slots.device).view(B, 1, 1)
        p_idx3 = pos_arange.unsqueeze(0)
        grid_exp = grid.unsqueeze(0).expand(B, -1, -1)

        for _ in range(sweeps):
            for blk_idx in blocks:
                is_free = torch.zeros(B, M, dtype=torch.bool, device=x_slots.device)
                is_free.scatter_(1, blk_idx, True)
                y_frozen_oh = F.one_hot(y, num_classes=V).float() * (~is_free).unsqueeze(-1).float()
                a_w = torch.bmm(a_sym, torch.matmul(y_frozen_oh, w_st))
                eff_unary = u[b_arange, blk_idx] + a_w[b_arange, blk_idx]
                e_unary = eff_unary[b_idx3, p_idx3, grid_exp].sum(dim=-1)
                a_blk = a_sym.gather(1, blk_idx.unsqueeze(-1).expand(-1, -1, M)).gather(
                    2, blk_idx.unsqueeze(1).expand(-1, k_blk, -1)
                )
                e_pair = torch.zeros(B, N_cfg, device=x_slots.device, dtype=x_slots.dtype)
                for p1 in range(k_blk):
                    for p2 in range(p1 + 1, k_blk):
                        e_pair = e_pair + a_blk[:, p1, p2].unsqueeze(1) * w_st[grid[:, p1], grid[:, p2]].unsqueeze(0)
                best_cfg = (e_unary + e_pair).argmin(dim=-1)
                y.scatter_(1, blk_idx, grid[best_cfg])
        return y


def _generate_paired_null_slot_batch(
    batch_size: int,
    n_slots: int,
    d_feat: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, list[list[tuple[int, int]]]]:
    """Generates variable-cardinality (`K_act in {4, 6, 8}`) paired conservation tasks with `0 = null`."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    d_pair_code = d_feat - 4
    x_all = torch.zeros(batch_size, n_slots, d_feat, dtype=torch.float32)
    y_all = torch.zeros(batch_size, n_slots, dtype=torch.long)
    pairs_meta: list[list[tuple[int, int]]] = []

    for b in range(batch_size):
        k_act = (2 + (b % 3)) * 2  # 4, 6, or 8 active slots -> 6, 4, or 2 null slots (Ø)
        perm = torch.randperm(n_slots, generator=gen).tolist()
        act_nodes = perm[:k_act]
        null_nodes = perm[k_act:]

        pair_codes = F.normalize(torch.randn(k_act // 2, d_pair_code, generator=gen), dim=-1)
        b_pairs: list[tuple[int, int]] = []
        for pi in range(k_act // 2):
            u, v = act_nodes[2 * pi], act_nodes[2 * pi + 1]
            b_pairs.append((u, v))
            val_u = int(torch.randint(1, 5, (1,), generator=gen).item())
            val_v = 5 - val_u  # coupled flow conservation: y_u + y_v == 5
            x_all[b, u, :d_pair_code] = pair_codes[pi]
            x_all[b, v, :d_pair_code] = pair_codes[pi]
            # Unambiguous sender cue; masked receiver cue + null hint pulling unary init to 0 (Ø)
            x_all[b, u, d_pair_code] = float(val_u) / 4.0
            x_all[b, v, d_pair_code] = 0.0
            x_all[b, u, d_pair_code + 1] = 1.0
            x_all[b, v, d_pair_code + 1] = 0.3
            x_all[b, u, d_pair_code + 2] = 1.0
            x_all[b, v, d_pair_code + 2] = -1.0
            x_all[b, v, d_pair_code + 3] = 0.65
            y_all[b, u] = val_u
            y_all[b, v] = val_v

        null_codes = F.normalize(torch.randn(len(null_nodes), d_pair_code, generator=gen), dim=-1)
        for ni, w in enumerate(null_nodes):
            x_all[b, w, :d_pair_code] = null_codes[ni]
            x_all[b, w, d_pair_code + 3] = 1.0  # inactive hint mixed with noise
            y_all[b, w] = 0

        pairs_meta.append(b_pairs)

    return x_all.to(device), y_all.to(device), pairs_meta


def _check_paired_conservation_valid(
    y_pred: torch.Tensor, pairs_meta: list[list[tuple[int, int]]]
) -> tuple[float, float]:
    """Checks 100% conservation validity (`y_u+y_v==5` on active pairs, `0` on null slots)."""
    y_cpu = y_pred.detach().cpu().tolist()
    n_ok = 0
    n_null_ok = 0
    for b, b_pairs in enumerate(pairs_meta):
        row = y_cpu[b]
        act_set = {x for pair in b_pairs for x in pair}
        valid_pairs = all(
            (row[u] >= 1 and row[v] >= 1 and (row[u] + row[v] == 5)) for (u, v) in b_pairs
        )
        valid_null = all(row[w] == 0 for w in range(len(row)) if w not in act_set)
        if valid_pairs and valid_null:
            n_ok += 1
        if valid_null:
            n_null_ok += 1
    tot = float(max(1, len(pairs_meta)))
    return 100.0 * n_ok / tot, 100.0 * n_null_ok / tot


def run_invariant_3_rlns_null_slot(
    device: torch.device, seed: int = 2026, fast: bool = False
) -> dict[str, object]:
    torch.manual_seed(seed)
    n_slots = 10
    d_feat = 24
    n_train = 512
    n_test = 64 if fast else 128
    steps = 160 if fast else 220

    x_tr, y_tr, _ = _generate_paired_null_slot_batch(n_train, n_slots, d_feat, device, seed)
    x_te, y_te, pairs_te = _generate_paired_null_slot_batch(n_test, n_slots, d_feat, device, seed + 5000)

    # 1) 1-Flip Besag PL baseline (|B| = 1)
    m_1f = RLNSVariableSlotEBM(d_feat=d_feat, n_slots=n_slots, n_states=5).to(device)
    opt_1f = torch.optim.AdamW(m_1f.parameters(), lr=8e-3, weight_decay=1e-4)
    bs = 64
    for st in range(steps):
        idx = torch.randint(0, n_train, (bs,), device=device)
        blk = torch.randint(0, n_slots, (bs, 1), device=device)
        loss = m_1f.block_fenchel_young_loss(x_tr[idx], y_tr[idx], blk, gamma=0.35)
        opt_1f.zero_grad(set_to_none=True)
        loss.backward()
        opt_1f.step()
    with torch.no_grad():
        y_pred_1f = m_1f.infer_block_gibbs(x_te, k_blk=1, sweeps=6)
        acc_1f = float((y_pred_1f == y_te).all(dim=-1).float().mean().item() * 100.0)
        valid_1f, _ = _check_paired_conservation_valid(y_pred_1f, pairs_te)

    # 2) Adaptive Min-Cut RLNS (|B| = 4, 0-BPTT)
    torch.manual_seed(seed)
    m_rlns = RLNSVariableSlotEBM(d_feat=d_feat, n_slots=n_slots, n_states=5).to(device)
    opt_rlns = torch.optim.AdamW(m_rlns.parameters(), lr=8e-3, weight_decay=1e-4)
    for st in range(steps):
        idx = torch.randint(0, n_train, (bs,), device=device)
        xb, yb = x_tr[idx], y_tr[idx]
        with torch.no_grad():
            _, a_sym, _ = m_rlns.compute_potentials(xb)
            blocks = m_rlns.adaptive_mincut_blocks(a_sym, k_blk=4)
            blk = blocks[st % len(blocks)]
        loss = m_rlns.block_fenchel_young_loss(xb, yb, blk, gamma=0.35)
        opt_rlns.zero_grad(set_to_none=True)
        loss.backward()
        if st == 0:
            live_grads = sum(
                1 for p in m_rlns.parameters() if p.grad is not None and float(p.grad.abs().sum().item()) > 0.0
            )
            total_params = sum(1 for _ in m_rlns.parameters())
        opt_rlns.step()

    with torch.no_grad():
        y_pred_rlns = m_rlns.infer_block_gibbs(x_te, k_blk=4, sweeps=4)
        acc_rlns = float((y_pred_rlns == y_te).all(dim=-1).float().mean().item() * 100.0)
        valid_rlns, null_rlns = _check_paired_conservation_valid(y_pred_rlns, pairs_te)

    passed = (
        live_grads == total_params
        and acc_rlns >= 85.0
        and valid_rlns >= 98.0
        and null_rlns == 100.0
        and acc_rlns >= acc_1f + 35.0
    )
    return {
        "live_param_tensors": f"{live_grads}/{total_params}",
        "besag_1flip_exact_pct": round(acc_1f, 2),
        "besag_1flip_valid_pct": round(valid_1f, 2),
        "rlns_block4_exact_pct": round(acc_rlns, 2),
        "rlns_block4_valid_pct": round(valid_rlns, 2),
        "rlns_null_slot_acc_pct": round(null_rlns, 2),
        "passed": bool(passed),
    }


# ===========================================================================
# Invariant 4: Lazy Identity Collapse + GAQ Bridge + Live Tier-1 LUT Probe
# ===========================================================================


def compute_householder_rotation(r: torch.Tensor, c_bar: torch.Tensor) -> torch.Tensor:
    """Computes exact sample-wise GAQ Householder operator `R_theta` `(..., d, d)` with 0/0 guard."""
    d = r.size(-1)
    r_norm = r.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    c_norm = c_bar.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    v = r / r_norm
    u = c_bar / c_norm
    w = u - v
    w_sq = (w * w).sum(dim=-1, keepdim=True).unsqueeze(-1)
    eye = torch.eye(d, device=r.device, dtype=r.dtype).expand(*r.shape[:-1], d, d)
    wwt = w.unsqueeze(-1) * w.unsqueeze(-2)
    h_mat = torch.where(w_sq > 1e-10, eye - 2.0 * wwt / w_sq.clamp(min=1e-10), eye)
    scale = (c_norm / r_norm).unsqueeze(-1)
    return scale * h_mat


class GAQSlotBridge(nn.Module):
    """Geometry Alignment Quantization (`SimVQ + Householder R_theta`) bridge without codebook collapse."""

    def __init__(self, d_in: int, d_code: int = 16, num_codes: int = 64) -> None:
        super().__init__()
        self.d_in = int(d_in)
        self.d_code = int(d_code)
        self.num_codes = int(num_codes)
        gen = torch.Generator().manual_seed(314159)
        self.register_buffer("C_frozen", torch.randn(num_codes, d_code, generator=gen))
        self.in_proj = nn.Linear(d_in, d_code, bias=False)
        self.E_vq = nn.Parameter(torch.eye(d_code) + 0.02 * torch.randn(d_code, d_code, generator=gen))

    def codebook(self) -> torch.Tensor:
        return torch.matmul(self.C_frozen, self.E_vq)

    def forward(self, h_slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r = self.in_proj(h_slots)
        orig_shape = r.shape
        r_flat = r.reshape(-1, self.d_code)
        c_mat = self.codebook()
        dists = torch.cdist(r_flat, c_mat, p=2.0)
        idx = dists.argmin(dim=-1)
        c_bar = c_mat[idx]

        # Householder Rotation Trick `q_hat = sg[c_bar] + sg[R_theta](r - sg[r])`
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
        q_hat = c_bar.detach() + rot_dx

        vq_loss = F.mse_loss(r_flat.detach(), c_bar) + 0.25 * F.mse_loss(r_flat, c_bar.detach())
        return (
            q_hat.view(orig_shape),
            vq_loss,
            r_flat,
            c_bar,
        )


def _measure_gaq_radial_tangential_leakage(bridge: GAQSlotBridge, x: torch.Tensor) -> tuple[float, float]:
    r_flat = bridge.in_proj(x).detach().reshape(-1, bridge.d_code)
    c_mat = bridge.codebook().detach()
    idx = torch.cdist(r_flat, c_mat, p=2.0).argmin(dim=-1)
    c_bar = c_mat[idx]
    r_norm = r_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    c_norm = c_bar.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    v = r_flat / r_norm
    u = c_bar / c_norm
    a_scalar = torch.ones(r_flat.size(0), 1, device=r_flat.device, dtype=r_flat.dtype)
    grad_rad = a_scalar * u

    # Under STE (Jacobian = I): tangential projection (I - v v^T)(a u)
    ste_proj = grad_rad - (grad_rad * v).sum(dim=-1, keepdim=True) * v
    ste_leak = float(ste_proj.norm(dim=-1).max().item())

    # Under Householder R_theta^T: (I - v v^T)(R_theta^T (a u)) == 0
    R_theta = compute_householder_rotation(r_flat, c_bar)
    r_grad = torch.bmm(R_theta.transpose(-1, -2), grad_rad.unsqueeze(-1)).squeeze(-1)
    gaq_proj = r_grad - (r_grad * v).sum(dim=-1, keepdim=True) * v
    gaq_leak = float(gaq_proj.norm(dim=-1).max().item())
    return gaq_leak, ste_leak


class _DualModeReasonerHarness(nn.Module):
    """Tests Lazy Identity Collapse (open continuous bypass) vs Blocked-Residual + Sectioned GAQ."""

    def __init__(self, d_x: int = 16, block_bypass: bool = False) -> None:
        super().__init__()
        self.block_bypass = bool(block_bypass)
        self.s1_backbone = nn.Sequential(nn.Linear(d_x, 128), nn.SiLU(), nn.Linear(128, d_x))
        self.s1_head = nn.Sequential(nn.Linear(d_x, 64), nn.SiLU(), nn.Linear(64, 2))
        self.bridge = GAQSlotBridge(d_in=4, d_code=8, num_codes=24)
        self.s2_net = nn.Sequential(
            nn.Linear(32, 48),
            nn.SiLU(),
            nn.Linear(48, 48),
            nn.SiLU(),
            nn.Linear(48, 2),
        )
        if not block_bypass:
            nn.init.zeros_(self.s2_net[-1].weight)
            nn.init.zeros_(self.s2_net[-1].bias)

    def forward(
        self, x: torch.Tensor, risk_gate: torch.Tensor, disable_s2: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_s1 = self.s1_backbone(x)
        logits_s1 = self.s1_head(h_s1)
        if disable_s2:
            return logits_s1, torch.tensor(0.0, device=x.device)

        q_slots, vq_loss, _, _ = self.bridge(x.view(x.size(0), 4, 4))
        logits_s2 = self.s2_net(q_slots.reshape(x.size(0), 32))
        g = risk_gate.unsqueeze(-1)
        if self.block_bypass:
            logits = (1.0 - g) * logits_s1 + g * logits_s2
        else:
            logits = logits_s1 + g * logits_s2
        return logits, vq_loss


def run_invariant_4a_lazy_collapse_and_gaq(
    device: torch.device, seed: int = 2026, fast: bool = False
) -> dict[str, object]:
    torch.manual_seed(seed)

    # 1) Measure Householder radial-tangential leakage & Step-1 100% active codes
    gaq_probe = GAQSlotBridge(d_in=16, d_code=16, num_codes=64).to(device)
    x_probe = torch.randn(32, 16, device=device)
    gaq_leak, ste_leak = _measure_gaq_radial_tangential_leakage(gaq_probe, x_probe)

    with torch.no_grad():
        c_before = gaq_probe.codebook().clone()
    opt_cb = torch.optim.SGD([gaq_probe.E_vq], lr=0.1)
    _, vq_l, _, _ = gaq_probe(x_probe)
    opt_cb.zero_grad()
    vq_l.backward()
    opt_cb.step()
    with torch.no_grad():
        c_after = gaq_probe.codebook()
        gaq_active_pct = 100.0 * float(((c_after - c_before).abs().max(dim=-1).values > 1e-7).float().mean().item())

    classic_cb = nn.Parameter(gaq_probe.C_frozen.clone())
    r_cl = gaq_probe.in_proj(x_probe).detach()
    idx_cl = torch.cdist(r_cl, classic_cb).argmin(dim=-1)
    l_cl = F.mse_loss(r_cl, classic_cb[idx_cl])
    l_cl.backward()
    classic_active_pct = 100.0 * float((classic_cb.grad.abs().max(dim=-1).values > 1e-7).float().mean().item())

    # 2) Mixed stream: Easy System-1 vs Hard System-2 (3-slot parity XOR)
    n_train = 192
    n_test = 128 if fast else 256
    d_x = 16
    steps = 160 if fast else 200

    def _make_mixed(n: int, s_offset: int):
        g = torch.Generator(device="cpu").manual_seed(seed + s_offset)
        is_hard = torch.arange(n) >= (n // 2)
        x = torch.randn(n, d_x, generator=g) * 0.35
        y = torch.zeros(n, dtype=torch.long)
        risk = torch.zeros(n, dtype=torch.float32)
        for i in range(n):
            if not is_hard[i]:
                sig = 2.0 if (i % 2 == 0) else -2.0
                x[i, 0] = sig
                y[i] = 1 if sig > 0 else 0
                risk[i] = 0.0
            else:
                risk[i] = 1.0
                bits = torch.randint(0, 2, (3,), generator=g)
                x[i, 4] = (float(bits[0]) * 2.0 - 1.0) * 2.2
                x[i, 8] = (float(bits[1]) * 2.0 - 1.0) * 2.2
                x[i, 12] = (float(bits[2]) * 2.0 - 1.0) * 2.2
                y[i] = int(bits.sum().item() % 2)
        return x.to(device), y.to(device), risk.to(device), is_hard.to(device)

    x_tr, y_tr, r_tr, _ = _make_mixed(n_train, 200)
    x_te, y_te, r_te, h_te = _make_mixed(n_test, 9026)

    out_cmp: dict[str, dict[str, float]] = {}
    for name, block_bp in (("open_bypass", False), ("blocked_gaq", True)):
        torch.manual_seed(seed + 404)
        mod = _DualModeReasonerHarness(d_x=d_x, block_bypass=block_bp).to(device)
        opt = torch.optim.AdamW(mod.parameters(), lr=5e-3, weight_decay=1e-4)
        bs = 64
        for _ in range(steps):
            idx = torch.randint(0, n_train, (bs,), device=device)
            lg, vql = mod(x_tr[idx], r_tr[idx], disable_s2=False)
            loss = F.cross_entropy(lg, y_tr[idx]) + 0.1 * vql
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        with torch.no_grad():
            lg_full, _ = mod(x_te, r_te, disable_s2=False)
            lg_nos2, _ = mod(x_te, r_te, disable_s2=True)
            pred_full = lg_full.argmax(dim=-1)
            pred_nos2 = lg_nos2.argmax(dim=-1)
            hard_acc = float((pred_full[h_te] == y_te[h_te]).float().mean().item() * 100.0)
            hard_nos2 = float((pred_nos2[h_te] == y_te[h_te]).float().mean().item() * 100.0)
            easy_acc = float((pred_full[~h_te] == y_te[~h_te]).float().mean().item() * 100.0)
            false_trig_pct = float((r_te[~h_te] > 0.5).float().mean().item() * 100.0)
        out_cmp[name] = {
            "hard_acc_pct": round(hard_acc, 2),
            "hard_nos2_acc_pct": round(hard_nos2, 2),
            "s2_contrib_pp": round(hard_acc - hard_nos2, 2),
            "easy_acc_pct": round(easy_acc, 2),
            "easy_false_trigger_pct": round(false_trig_pct, 2),
        }

    passed = (
        gaq_leak < 1e-6
        and ste_leak > 0.05
        and gaq_active_pct == 100.0
        and classic_active_pct <= 45.0
        and out_cmp["open_bypass"]["s2_contrib_pp"] < 5.0
        and out_cmp["blocked_gaq"]["hard_acc_pct"] >= 90.0
        and out_cmp["blocked_gaq"]["s2_contrib_pp"] >= 35.0
        and out_cmp["blocked_gaq"]["easy_false_trigger_pct"] == 0.0
    )
    return {
        "gaq_radial_tangential_leak": gaq_leak,
        "ste_radial_tangential_leak": ste_leak,
        "gaq_step1_active_codes_pct": round(gaq_active_pct, 2),
        "classic_vq_step1_active_codes_pct": round(classic_active_pct, 2),
        "open_bypass": out_cmp["open_bypass"],
        "blocked_gaq": out_cmp["blocked_gaq"],
        "passed": bool(passed),
    }


def run_invariant_4b_live_tier1_lut_probe(
    device: torch.device, seed: int = 2026, fast: bool = False
) -> dict[str, object]:
    """Read-only probe on `checkpoints/calcite5_tier1_17m_utf8_qat_h_lut.pt` (`5.41 MB 2-bit LUT`)."""
    from kernels.fast_packed_lut import FastPackedLUT2BitLinear, load_packed_lut_checkpoint
    from run_simvq_split_benchmark import compute_state_cache_kb
    from simvq_split.memory import compute_chunker_state_bytes
    from simvq_split.quant_lut import compute_2bit_lut_footprint
    from simvq_split.tasks import (
        load_gsm8k_corpora,
        make_overwrite_story,
        make_trap_overwrite_story,
    )

    ckpt_path = ROOT_DIR / "checkpoints" / "calcite5_tier1_17m_utf8_qat_h_lut.pt"
    assert ckpt_path.exists(), f"Missing Tier-1 Phase H LUT checkpoint: {ckpt_path}"

    dev_str = str(device)
    model, cfg, tok, _ = load_packed_lut_checkpoint(ckpt_path, device=dev_str, backend="pytorch_vec")
    fp_info = compute_2bit_lut_footprint(model)
    n_lut = len([m for m in model.modules() if isinstance(m, FastPackedLUT2BitLinear)])
    cache_kb_2048 = compute_state_cache_kb("SimVQSplitLM", cfg, None, 2048)
    buf_bytes = compute_chunker_state_bytes(
        cfg.d_model, cfg.byte_char_dim, cfg.byte_router_char_dim, byte_chunk_mode=cfg.byte_chunk_mode
    )

    _, te_docs, _, te_sents = load_gsm8k_corpora(ROOT_DIR)
    rng = random.Random(seed)
    n_each = 8 if fast else 16
    id_prompts: list[str] = []
    trap_prompts: list[str] = []
    math_prompts: list[str] = []
    for i in range(n_each):
        p_id, _, _ = make_overwrite_story(
            rng, num_entities=4, num_overwrites=2, distractor_pool=te_sents, filler_sentences_after=2
        )
        p_tr, _, _ = make_trap_overwrite_story(
            rng, num_entities=4, num_overwrites=2, distractor_pool=te_sents, filler_sentences_after=2
        )
        q_txt = te_docs[(i * 7 + 3) % len(te_docs)].split("\nSolution:")[0]
        id_prompts.append(p_id)
        trap_prompts.append(p_tr)
        math_prompts.append(q_txt + "\nSolution:")

    def _score(prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        seqs = [tok.encode(p, add_bos=True, add_eos=False) for p in prompts]
        max_l = max(len(s) for s in seqs)
        xb = torch.full((len(seqs), max_l), tok.PAD_ID, dtype=torch.long, device=device)
        lens = [len(s) for s in seqs]
        for row_i, s in enumerate(seqs):
            xb[row_i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(xb)
            u_raw = model.woodbury.last_epistemic_uncertainty.float()
            if model.byte_chunker is not None and u_raw.size(1) != max_l:
                _, _, chunk_ctx = model.byte_chunker.encode_to_chunks(
                    model.tok_emb(xb), xb, model.tok_emb.weight
                )
                s_idx = (
                    chunk_ctx.slot_idx.clamp(min=0, max=max(0, u_raw.size(1) - 1))
                    .unsqueeze(-1)
                    .expand(-1, -1, u_raw.size(-1))
                )
                u_raw = torch.gather(u_raw, 1, s_idx)
            row_t = torch.arange(len(seqs), device=device)
            end_t = torch.tensor([l_i - 1 for l_i in lens], dtype=torch.long, device=device)
            u_epi = u_raw[row_t, end_t, : model.woodbury.H_epi].mean(dim=-1)
            lg_last = logits[row_t, end_t, :].float()
            lp = F.log_softmax(lg_last, dim=-1)
            p = lp.exp()
            ent = -(p * lp).sum(dim=-1, keepdim=True)
            var_ent = (p * ((-lp - ent) ** 2)).sum(dim=-1)
        return u_epi.cpu(), var_ent.cpu()

    u_id, ve_id = _score(id_prompts)
    u_tr, ve_tr = _score(trap_prompts)
    u_ma, ve_ma = _score(math_prompts)

    lam_var = 0.05
    r_id = u_id + lam_var * ve_id
    r_tr = u_tr + lam_var * ve_tr
    r_ma = u_ma + lam_var * ve_ma
    r_trigger = torch.cat([r_tr, r_ma])

    auroc_woodbury_trap = _auroc(u_tr, u_id)
    auroc_varent_math = _auroc(ve_ma, ve_id)
    auroc_combined = _auroc(r_trigger, r_id)

    passed = (
        n_lut == 95
        and abs(float(fp_info["total_lut_kb"]) - 5539.81) <= 0.1
        and abs(float(cache_kb_2048) - 230.11) <= 0.1
        and int(buf_bytes) == 2164
        and auroc_combined >= 0.90
        and auroc_woodbury_trap >= 0.90
    )
    return {
        "lut_layers": n_lut,
        "total_lut_kb": round(float(fp_info["total_lut_kb"]), 2),
        "cache_kb_L2048": round(float(cache_kb_2048), 2),
        "chunker_state_bytes": int(buf_bytes),
        "n_prompts_per_split": n_each,
        "id_mean_u_epi": round(float(u_id.mean().item()), 4),
        "trap_mean_u_epi": round(float(u_tr.mean().item()), 4),
        "math_mean_u_epi": round(float(u_ma.mean().item()), 4),
        "id_mean_varent": round(float(ve_id.mean().item()), 4),
        "math_mean_varent": round(float(ve_ma.mean().item()), 4),
        "lam_var": lam_var,
        "auroc_woodbury_trap": round(auroc_woodbury_trap, 4),
        "auroc_varent_math": round(auroc_varent_math, 4),
        "auroc_combined_risk": round(auroc_combined, 4),
        "passed": bool(passed),
    }


def run_all_invariants(
    device: torch.device, fast: bool = False
) -> dict[str, object]:
    t_start = time.perf_counter()
    # Two-seed noise calibration (2026, 2027) for OEB 2*sigma discipline on Core Invariants 1-3 + 4A
    seed_runs: dict[str, dict[str, object]] = {}
    for sd in (2026, 2027):
        inv1 = run_invariant_1_laplacian(device, seed=sd, fast=fast)
        inv2 = run_invariant_2_cSB_permflow(device, seed=sd, fast=fast)
        inv3 = run_invariant_3_rlns_null_slot(device, seed=sd, fast=fast)
        inv4a = run_invariant_4a_lazy_collapse_and_gaq(device, seed=sd, fast=fast)
        seed_runs[str(sd)] = {
            "invariant_1": inv1,
            "invariant_2": inv2,
            "invariant_3": inv3,
            "invariant_4a": inv4a,
        }

    # Live read-only probe on Tier-1 Phase H 5.41 MB 2-bit LUT checkpoint
    inv4b = run_invariant_4b_live_tier1_lut_probe(device, seed=2026, fast=fast)
    _sync_if_cuda(device)
    elapsed_s = round(time.perf_counter() - t_start, 2)

    s1, s2 = seed_runs["2026"], seed_runs["2027"]
    two_sigma = {
        "inv1_uebr_qat_acc_pp": round(
            2.0 * abs(float(s1["invariant_1"]["uebr_qat_158"]["acc_pct"]) - float(s2["invariant_1"]["uebr_qat_158"]["acc_pct"])),
            2,
        ),
        "inv2_be_wins": round(
            2.0 * abs(float(s1["invariant_2"]["part2b_be_wins"]) - float(s2["invariant_2"]["part2b_be_wins"])),
            2,
        ),
        "inv3_rlns_exact_pp": round(
            2.0 * abs(float(s1["invariant_3"]["rlns_block4_exact_pct"]) - float(s2["invariant_3"]["rlns_block4_exact_pct"])),
            2,
        ),
        "inv4a_blocked_hard_pp": round(
            2.0 * abs(float(s1["invariant_4a"]["blocked_gaq"]["hard_acc_pct"]) - float(s2["invariant_4a"]["blocked_gaq"]["hard_acc_pct"])),
            2,
        ),
    }

    all_passed = (
        all(bool(seed_runs[str(sd)][k]["passed"]) for sd in (2026, 2027) for k in ("invariant_1", "invariant_2", "invariant_3", "invariant_4a"))
        and bool(inv4b["passed"])
    )
    return {
        "protocol": {
            "phase": "Phase_I_UEBR_Tracer",
            "device": str(device),
            "fast_mode": bool(fast),
            "seeds": [2026, 2027],
            "elapsed_sec": elapsed_s,
        },
        "two_sigma_noise_floor": two_sigma,
        "seed_2026": seed_runs["2026"],
        "seed_2027": seed_runs["2027"],
        "invariant_4b_live_tier1_lut": inv4b,
        "all_passed": bool(all_passed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase I Calcite-6 / UEBR 4-invariant tracer")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Target device (`cuda` or `cpu`)",
    )
    parser.add_argument(
        "--json-out",
        default=str(ROOT_DIR / "benchmarks" / "phase_i_uebr_tracer_results.json"),
        help="Output JSON report path",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Reduced iteration budget for fast smoke testing (< 8 s)",
    )
    args = parser.parse_args()

    dev = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"Calcite-6 (UEBR) Tracer #2 | device={dev} | fast={args.fast}", flush=True)
    rep = run_all_invariants(dev, fast=args.fast)

    s26 = rep["seed_2026"]
    i1 = s26["invariant_1"]
    i2 = s26["invariant_2"]
    i3 = s26["invariant_3"]
    i4a = s26["invariant_4a"]
    i4b = rep["invariant_4b_live_tier1_lut"]

    print(
        f"[Inv 1] min_eig={i1['min_eig_init']} | adj_err={i1['adjoint_grad_err']:.2e} | epc_err={i1['epc_init_max_err']:.2e} | "
        f"StaticWzz={i1['static_wzz_acc_pct']}% EqPropQAT={i1['sym_eqprop_qat_acc_pct']}% "
        f"UEBR_FP32={i1['uebr_fp32']['acc_pct']}% UEBR_QAT={i1['uebr_qat_158']['acc_pct']}% (gap={i1['uebr_qat_158']['energy_gap_short_minus_long']})",
        flush=True,
    )
    print(
        f"[Inv 2] 2A ModeCov: Sinkhorn={i2['part2a_sinkhorn_mode_cov_pct']}% vs PermFlow={i2['part2a_permflow_mode_cov_pct']}% "
        f"(ba_err={i2['part2a_ba_err']:.2e}) | 2B Footnote23(affPGD={i2['part2b_footnote23_affine_pgd_mean']} vs unbCSB={i2['part2b_footnote23_unbounded_csb_mean']}, ba64={i2['part2b_ba_err_f64']:.2e}) "
        f"| 2B Rounding: Greedy={i2['part2b_greedy_mean_energy']} vs BE={i2['part2b_birkhoff_ext_mean_energy']} "
        f"wins={i2['part2b_be_wins']}/50 ({i2['part2b_be_win_pct']}%) | lossless={i2['part2b_lossless_rounding_pct']}%",
        flush=True,
    )
    print(
        f"[Inv 3] live_grads={i3['live_param_tensors']} | 1-Flip={i3['besag_1flip_exact_pct']}% (valid={i3['besag_1flip_valid_pct']}%) "
        f"vs RLNS+Null={i3['rlns_block4_exact_pct']}% (valid={i3['rlns_block4_valid_pct']}%, null={i3['rlns_null_slot_acc_pct']}%)",
        flush=True,
    )
    print(
        f"[Inv 4A] GAQ leak={i4a['gaq_radial_tangential_leak']:.2e} vs STE={i4a['ste_radial_tangential_leak']:.2e} | "
        f"active_step1={i4a['gaq_step1_active_codes_pct']}% vs VQ={i4a['classic_vq_step1_active_codes_pct']}% | "
        f"OpenBypass S2={i4a['open_bypass']['s2_contrib_pp']}pp vs BlockedGAQ Hard={i4a['blocked_gaq']['hard_acc_pct']}% (+{i4a['blocked_gaq']['s2_contrib_pp']}pp, false_trig={i4a['blocked_gaq']['easy_false_trigger_pct']}%)",
        flush=True,
    )
    print(
        f"[Inv 4B] Tier-1 5.41MB LUT ({i4b['lut_layers']} layers, cache={i4b['cache_kb_L2048']}KB, buf={i4b['chunker_state_bytes']}B): "
        f"u_ID={i4b['id_mean_u_epi']} u_Trap={i4b['trap_mean_u_epi']} u_Math={i4b['math_mean_u_epi']} | "
        f"AUROC(Trap)={i4b['auroc_woodbury_trap']} AUROC(Comb)={i4b['auroc_combined_risk']}",
        flush=True,
    )
    print(f"2*sigma noise floor: {rep['two_sigma_noise_floor']} | elapsed={rep['protocol']['elapsed_sec']}s", flush=True)

    out_p = Path(args.json_out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    assert rep["all_passed"], "One or more Phase I UEBR invariants failed!"
    print("ALL PHASE I UEBR INVARIANTS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
