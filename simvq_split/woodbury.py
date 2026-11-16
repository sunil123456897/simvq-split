"""
Second-Order Epistemic Memory Primitives for `SimVQ-Split` (`Calcite-5`):
- TorchScript chunk scans (`_neural_woodbury_chunk_scan_fp_jit`, `_neural_woodbury_chunk_scan_tern_jit`, `neural_woodbury_chunk_scan_jit`)
- `inspect_woodbury_spd_spectrum` & `NeuralWoodburyLayer`
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .baselines import RMSNorm
from .quant_lut import PackedLUT2BitLinear, quantize_ternary_weights_ls


@torch.jit.script
def _neural_woodbury_chunk_scan_fp_jit(
    q_chunks: torch.Tensor,  # (B, M, C, H, d_k) unit-normalized retrieval token queries
    k_seq: torch.Tensor,  # (B, M, H, d_k) unit-normalized retrieval chunk write keys
    v_seq: torch.Tensor,  # (B, M, H, d_v) chunk write values
    beta_seq: torch.Tensor,  # (B, M, H, 1) sparse write gate in (0, 1)
    gamma_seq: torch.Tensor,  # (B, M, H, 1) retention factor in (0, 1)
    inv_lam: float,
    q_epi_chunks: torch.Tensor
    | None = None,  # optional pure 3-way Hadamard epistemic queries
    k_epi_seq: torch.Tensor
    | None = None,  # optional pure 3-way Hadamard epistemic keys
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Strictly SPD-preserving Second-Order Sherman-Morrison-Woodbury Memory + Analytic Epistemic Sensor (FP core).
    Uses cubic write selectivity `sigmoid((bt - 0.26) * 18)^3` on stop-gradient covariance `P` so
    distractor prose never saturates the 32D key subspace over long OOD contexts (`u_seen < 0.25`
    vs `u_trap > 0.48`), while keeping `100%` smooth live gradients on `M_mat`.
    """
    B, M_len, C, H, d_k = q_chunks.shape
    d_v = v_seq.size(-1)
    BH = B * H
    lam_reg = 1.0 / inv_lam

    qe_src = q_epi_chunks if q_epi_chunks is not None else q_chunks
    ke_src = k_epi_seq if k_epi_seq is not None else k_seq

    M_mat = torch.zeros(BH, d_v, d_k, device=k_seq.device, dtype=torch.float32)
    P0 = (
        torch.eye(d_k, device=k_seq.device, dtype=torch.float32)
        .unsqueeze(0)
        .expand(BH, d_k, d_k)
        * inv_lam
    )
    P = P0.clone()
    outs: list[torch.Tensor] = []
    uncerts: list[torch.Tensor] = []

    q_u = q_chunks.float().permute(1, 0, 3, 4, 2).reshape(M_len, BH, d_k, C).unbind(0)
    k_u = k_seq.float().permute(1, 0, 2, 3).reshape(M_len, BH, d_k, 1).unbind(0)
    qe_u = (
        qe_src.detach()
        .float()
        .permute(1, 0, 3, 4, 2)
        .reshape(M_len, BH, d_k, C)
        .unbind(0)
    )
    ke_u = (
        ke_src.detach().float().permute(1, 0, 2, 3).reshape(M_len, BH, d_k, 1).unbind(0)
    )
    v_u = v_seq.float().permute(1, 0, 2, 3).reshape(M_len, BH, d_v, 1).unbind(0)
    b_u = beta_seq.float().permute(1, 0, 2, 3).reshape(M_len, BH, 1, 1).unbind(0)
    g_u = gamma_seq.float().permute(1, 0, 2, 3).reshape(M_len, BH, 1, 1).unbind(0)

    for m in range(M_len):
        qm = q_u[m]  # (BH, d_k, C)
        read_m = torch.bmm(M_mat, qm)  # (BH, d_v, C)

        qem = qe_u[m]
        pq_m = torch.bmm(P, qem)
        u_m = ((qem * pq_m).sum(dim=1) * lam_reg).clamp(0.0, 1.5)  # (BH, C)

        outs.append(read_m.reshape(B, H, d_v, C).permute(0, 3, 1, 2))
        uncerts.append(u_m.reshape(B, H, C).permute(0, 2, 1))

        kt = k_u[m]
        ket = ke_u[m]
        vt = v_u[m]
        bt = b_u[m]
        gt = g_u[m]

        # Associative biorthogonal Delta/Householder overwrite on M_mat (100% live gradients!)
        pred_v = torch.bmm(M_mat, kt)
        M_mat = gt * M_mat + (1.35 * bt) * torch.bmm(
            vt - gt * pred_v, kt.transpose(1, 2)
        )

        # Stop-gradient SPD Woodbury update gated by sharp cubic write selectivity
        bt_d = bt.detach()
        s_gate = torch.sigmoid((bt_d - 0.26) * 18.0)
        w_cov = s_gate * s_gate * s_gate

        pk = torch.bmm(P, ket)
        quad = (ket * pk).sum(dim=1, keepdim=True).clamp(min=1e-5)
        nov = w_cov * 0.92
        denom = 1.0 + nov * quad
        P_next = P - (nov / denom) * torch.bmm(pk, pk.transpose(1, 2))
        P = 0.5 * (P_next + P_next.transpose(1, 2))

    return (
        torch.cat(outs, dim=1).to(v_seq.dtype),
        torch.cat(uncerts, dim=1).to(v_seq.dtype),
    )


@torch.jit.script
def _neural_woodbury_chunk_scan_tern_jit(
    q_chunks: torch.Tensor,
    k_seq: torch.Tensor,
    v_seq: torch.Tensor,
    beta_seq: torch.Tensor,
    gamma_seq: torch.Tensor,
    inv_lam: float,
    q_epi_chunks: torch.Tensor | None = None,
    k_epi_seq: torch.Tensor | None = None,
    tern_gate: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Ternary-gated Second-Order Sherman-Morrison-Woodbury Memory engaging Quartic Soft-Ceiling
    (`DH = 144.0`) and gentle active-forget SPD floor `rho_rec` under 1.58-bit ternary weights.
    """
    B, M_len, C, H, d_k = q_chunks.shape
    d_v = v_seq.size(-1)
    BH = B * H
    lam_reg = 1.0 / inv_lam
    inv_dh = tern_gate / 144.0
    rho_g = 0.002 * tern_gate
    rho_c = 0.002 * tern_gate

    qe_src = q_epi_chunks if q_epi_chunks is not None else q_chunks
    ke_src = k_epi_seq if k_epi_seq is not None else k_seq

    M_mat = torch.zeros(BH, d_v, d_k, device=k_seq.device, dtype=torch.float32)
    P0 = (
        torch.eye(d_k, device=k_seq.device, dtype=torch.float32)
        .unsqueeze(0)
        .expand(BH, d_k, d_k)
        * inv_lam
    )
    P = P0.clone()
    outs: list[torch.Tensor] = []
    uncerts: list[torch.Tensor] = []

    q_u = q_chunks.float().permute(1, 0, 3, 4, 2).reshape(M_len, BH, d_k, C).unbind(0)
    k_u = k_seq.float().permute(1, 0, 2, 3).reshape(M_len, BH, d_k, 1).unbind(0)
    qe_u = (
        qe_src.detach()
        .float()
        .permute(1, 0, 3, 4, 2)
        .reshape(M_len, BH, d_k, C)
        .unbind(0)
    )
    ke_u = (
        ke_src.detach().float().permute(1, 0, 2, 3).reshape(M_len, BH, d_k, 1).unbind(0)
    )
    v_u = v_seq.float().permute(1, 0, 2, 3).reshape(M_len, BH, d_v, 1).unbind(0)
    b_u = beta_seq.float().permute(1, 0, 2, 3).reshape(M_len, BH, 1, 1).unbind(0)
    g_u = gamma_seq.float().permute(1, 0, 2, 3).reshape(M_len, BH, 1, 1).unbind(0)

    for m in range(M_len):
        qm = q_u[m]
        read_m = torch.bmm(M_mat, qm)

        qem = qe_u[m]
        pq_m = torch.bmm(P, qem)
        u_m = ((qem * pq_m).sum(dim=1) * lam_reg).clamp(0.0, 1.5)

        outs.append(read_m.reshape(B, H, d_v, C).permute(0, 3, 1, 2))
        uncerts.append(u_m.reshape(B, H, C).permute(0, 2, 1))

        kt = k_u[m]
        ket = ke_u[m]
        vt = v_u[m]
        bt = b_u[m]
        gt = g_u[m]

        pred_v = torch.bmm(M_mat, kt)
        err = vt - gt * pred_v
        r_sq = err.pow(2).sum(dim=1, keepdim=True) * inv_dh
        err_eff = err * torch.rsqrt(torch.sqrt(1.0 + r_sq * r_sq))
        M_mat = gt * M_mat + (1.35 * bt) * torch.bmm(err_eff, kt.transpose(1, 2))

        bt_d = bt.detach()
        s_gate = torch.sigmoid((bt_d - 0.26) * 20.0)
        w_cov = s_gate * s_gate * s_gate

        pk = torch.bmm(P, ket)
        quad = (ket * pk).sum(dim=1, keepdim=True).clamp(min=1e-5)
        nov = w_cov * 0.92
        denom = 1.0 + nov * quad
        P_next = P - (nov / denom) * torch.bmm(pk, pk.transpose(1, 2))
        P_mid = 0.5 * (P_next + P_next.transpose(1, 2))
        gt_d = gt.detach()
        rho_rec = ((1.0 - gt_d) * w_cov * rho_g).clamp(0.0, rho_c)
        P = (1.0 - rho_rec) * P_mid + rho_rec * P0
        P = 0.5 * (P + P.transpose(1, 2))

    return (
        torch.cat(outs, dim=1).to(v_seq.dtype),
        torch.cat(uncerts, dim=1).to(v_seq.dtype),
    )


def _wy_fp_block_fwd(
    M0: torch.Tensor,
    q_blk: torch.Tensor,
    k_blk: torch.Tensor,
    v_blk: torch.Tensor,
    b_blk: torch.Tensor,
    g_blk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    WY / Ut-style exact block-parallel forward step for FP Woodbury scan.
    Computes `G` products via float64 `log` + `cumsum` masked strictly before `exp`
    to eliminate `0.0 / 0.0 -> NaN` underflow at arbitrarily small retention `gamma`,
    and shields all `torch.bmm` calls from outer AMP `BFloat16` downcasting.
    """
    with torch.amp.autocast(M0.device.type, enabled=False):
        M0 = M0.float()
        q_blk = q_blk.float()
        k_blk = k_blk.float()
        v_blk = v_blk.float()
        b_blk = b_blk.float()
        g_blk = g_blk.float()
        BH, dv, _ = M0.shape
        blen = k_blk.size(-1)
        C = q_blk.size(-1) // blen
        a = 1.35 * b_blk

        idx = torch.arange(blen, device=M0.device)
        strict_lower_bool = (idx.unsqueeze(1) > idx.unsqueeze(0)).unsqueeze(0)
        log_g = torch.log(g_blk.double().clamp(min=1e-12))
        cs = F.pad(torch.cumsum(log_g, dim=1), (1, 0))
        zero_f64 = torch.zeros(1, device=M0.device, dtype=torch.float64)
        diff_cs = torch.where(
            strict_lower_bool,
            cs[:, :blen].unsqueeze(2) - cs[:, 1 : blen + 1].unsqueeze(1),
            zero_f64,
        )
        L_mj = torch.where(
            strict_lower_bool, torch.exp(diff_cs), zero_f64
        ).to(M0.dtype)
        G_m = torch.exp(cs[:, :blen]).to(M0.dtype)
        G_end = torch.exp(cs[:, blen : blen + 1]).to(M0.dtype)
        d_end = torch.exp(cs[:, blen : blen + 1] - cs[:, 1 : blen + 1]).to(M0.dtype)

        P0_K = torch.bmm(M0, k_blk)
        KK = torch.bmm(k_blk.transpose(1, 2), k_blk)
        ag = a * g_blk
        A_sys = torch.eye(blen, device=M0.device, dtype=M0.dtype).unsqueeze(0) + (
            ag.unsqueeze(2) * L_mj * KK
        )
        R = a.unsqueeze(1) * (v_blk - (g_blk * G_m).unsqueeze(1) * P0_K)
        U_T = torch.linalg.solve_triangular(
            A_sys, R.transpose(1, 2), upper=False, unitriangular=True
        )
        U = U_T.transpose(1, 2)

        P0_Q = torch.bmm(M0, q_blk).view(BH, dv, blen, C)
        read_0 = P0_Q * G_m.view(BH, 1, blen, 1)
        KQ = torch.bmm(k_blk.transpose(1, 2), q_blk).view(BH, blen, blen, C)
        LKQ = (L_mj.transpose(1, 2).unsqueeze(-1) * KQ).reshape(BH, blen, blen * C)
        read_u = torch.bmm(U, LKQ).view(BH, dv, blen, C)
        read_blk = (read_0 + read_u).reshape(BH, dv, blen * C)

        M_end = G_end.unsqueeze(-1) * M0 + torch.bmm(
            U * d_end.unsqueeze(1), k_blk.transpose(1, 2)
        )
        return read_blk, M_end


def block_parallel_wy_fp_scan(
    q_chunks: torch.Tensor,
    k_seq: torch.Tensor,
    v_seq: torch.Tensor,
    beta_seq: torch.Tensor,
    gamma_seq: torch.Tensor,
    inv_lam: float,
    q_epi_chunks: torch.Tensor | None = None,
    k_epi_seq: torch.Tensor | None = None,
    Bc: int = 32,
    checkpointed: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Zero-Triton Block-Parallel WY + Cholesky Block-Woodbury `P` scan (`Bc=32`)
    mathematically equivalent to `_neural_woodbury_chunk_scan_fp_jit`.
    """
    B, M_len, C, H, d_k = q_chunks.shape
    d_v = v_seq.size(-1)
    BH = B * H
    lam_reg = 1.0 / inv_lam
    orig_dtype = v_seq.dtype
    qe_src = q_epi_chunks if q_epi_chunks is not None else q_chunks
    ke_src = k_epi_seq if k_epi_seq is not None else k_seq
    use_ckpt = (
        checkpointed
        if checkpointed is not None
        else (torch.is_grad_enabled() and M_len > Bc)
    )

    with torch.amp.autocast(q_chunks.device.type, enabled=False):
        q_bh = (
            q_chunks.float()
            .permute(0, 3, 1, 2, 4)
            .reshape(BH, M_len * C, d_k)
            .transpose(1, 2)
            .contiguous()
        )
        k_bh = (
            k_seq.float()
            .permute(0, 2, 1, 3)
            .reshape(BH, M_len, d_k)
            .transpose(1, 2)
            .contiguous()
        )
        v_bh = (
            v_seq.float()
            .permute(0, 2, 1, 3)
            .reshape(BH, M_len, d_v)
            .transpose(1, 2)
            .contiguous()
        )
        b_bh = beta_seq.float().squeeze(-1).permute(0, 2, 1).reshape(BH, M_len).contiguous()
        g_bh = gamma_seq.float().squeeze(-1).permute(0, 2, 1).reshape(BH, M_len).contiguous()

        with torch.no_grad():
            qe_bh = (
                qe_src.detach()
                .float()
                .permute(0, 3, 1, 2, 4)
                .reshape(BH, M_len * C, d_k)
                .transpose(1, 2)
                .contiguous()
            )
            ke_bh = (
                ke_src.detach()
                .float()
                .permute(0, 2, 1, 3)
                .reshape(BH, M_len, d_k)
                .transpose(1, 2)
                .contiguous()
            )
            s_gate = torch.sigmoid((b_bh - 0.26) * 18.0)
            nov_sqrt = torch.sqrt((s_gate * s_gate * s_gate) * 0.92)
            eye_dk = torch.eye(d_k, device=k_seq.device, dtype=torch.float32).unsqueeze(0)
            P = eye_dk.expand(BH, d_k, d_k).clone() * inv_lam
            eye_bc = torch.eye(Bc, device=k_seq.device, dtype=torch.float32).unsqueeze(0)
            idx_full = torch.arange(Bc, device=P.device)
            sl_full = (
                (idx_full.unsqueeze(1) > idx_full.unsqueeze(0))
                .float()
                .view(1, Bc, Bc, 1)
            )
            u_blocks: list[torch.Tensor] = []
            for s in range(0, M_len, Bc):
                e = min(M_len, s + Bc)
                blen = e - s
                ke_b = ke_bh[:, :, s:e]
                qe_b = qe_bh[:, :, s * C : e * C]
                ns_b = nov_sqrt[:, s:e]
                ke_s = ke_b * ns_b.unsqueeze(1)
                PKe_s = torch.bmm(P, ke_s)
                PQe = torch.bmm(P, qe_b)

                KPK = torch.bmm(ke_s.transpose(1, 2), PKe_s)
                G_mat = eye_bc[:, :blen, :blen] + KPK
                G_mat = 0.5 * (G_mat + G_mat.transpose(1, 2))
                L_chol = torch.linalg.cholesky(G_mat)

                B_q = torch.bmm(ke_s.transpose(1, 2), PQe)
                X_q = torch.linalg.solve_triangular(L_chol, B_q, upper=False)
                X_sq = X_q.square().view(BH, blen, blen, C)
                corr = (X_sq.transpose(1, 2) * sl_full[:, :blen, :blen]).sum(dim=2)
                qPq0 = (qe_b * PQe).sum(dim=1).view(BH, blen, C)
                u_blocks.append(((qPq0 - corr) * lam_reg).clamp(0.0, 1.5))

                # Stabilized Joseph-form precision update (`A P A^T + K K^T >= 0` guarantees PSD!)
                Z = torch.linalg.solve_triangular(L_chol, PKe_s.transpose(1, 2), upper=False)
                K_T = torch.linalg.solve_triangular(L_chol.transpose(1, 2), Z, upper=True)
                K_gain = K_T.transpose(1, 2)
                A_upd = eye_dk - torch.bmm(K_gain, ke_s.transpose(1, 2))
                P = torch.bmm(
                    torch.bmm(A_upd, P), A_upd.transpose(1, 2)
                ) + torch.bmm(K_gain, K_gain.transpose(1, 2))
                P = 0.5 * (P + P.transpose(1, 2))
            uncerts = (
                torch.cat(u_blocks, dim=1)
                .view(B, H, M_len * C)
                .permute(0, 2, 1)
                .to(orig_dtype)
            )

        M0 = torch.zeros(BH, d_v, d_k, device=k_seq.device, dtype=torch.float32)
        reads: list[torch.Tensor] = []
        for s in range(0, M_len, Bc):
            e = min(M_len, s + Bc)
            if use_ckpt:
                r_blk, M0 = torch.utils.checkpoint.checkpoint(
                    _wy_fp_block_fwd,
                    M0,
                    q_bh[:, :, s * C : e * C],
                    k_bh[:, :, s:e],
                    v_bh[:, :, s:e],
                    b_bh[:, s:e],
                    g_bh[:, s:e],
                    use_reentrant=False,
                )
            else:
                r_blk, M0 = _wy_fp_block_fwd(
                    M0,
                    q_bh[:, :, s * C : e * C],
                    k_bh[:, :, s:e],
                    v_bh[:, :, s:e],
                    b_bh[:, s:e],
                    g_bh[:, s:e],
                )
            reads.append(r_blk)
        out_all = (
            torch.cat(reads, dim=-1)
            .view(B, H, d_v, M_len, C)
            .permute(0, 3, 4, 1, 2)
            .reshape(B, M_len * C, H, d_v)
            .to(orig_dtype)
        )
    return out_all, uncerts


@torch.jit.script
def _tern_p_uncerts_nograd_jit(
    qe_u_all: torch.Tensor,
    ke_u_all: torch.Tensor,
    bt_d_all: torch.Tensor,
    gt_d_all: torch.Tensor,
    inv_lam: float,
    tern_gate: float,
) -> torch.Tensor:
    M_len, BH, d_k, C = qe_u_all.shape
    lam_reg = 1.0 / inv_lam
    rho_g = 0.002 * tern_gate
    rho_c = 0.002 * tern_gate
    P0 = (
        torch.eye(d_k, device=qe_u_all.device, dtype=torch.float32)
        .unsqueeze(0)
        .expand(BH, d_k, d_k)
        * inv_lam
    )
    P = P0.clone()
    qe_u = qe_u_all.unbind(0)
    ke_u = ke_u_all.unbind(0)
    b_u = bt_d_all.unbind(0)
    g_u = gt_d_all.unbind(0)
    uncerts = torch.empty(M_len, BH, C, device=qe_u_all.device, dtype=torch.float32)
    for m in range(M_len):
        qem = qe_u[m]
        pq_m = torch.bmm(P, qem)
        uncerts[m] = ((qem * pq_m).sum(dim=1) * lam_reg).clamp(0.0, 1.5)
        ket = ke_u[m]
        s_gate = torch.sigmoid((b_u[m] - 0.26) * 20.0)
        w_cov = s_gate * s_gate * s_gate
        pk = torch.bmm(P, ket)
        quad = (ket * pk).sum(dim=1, keepdim=True).clamp(min=1e-5)
        nov = w_cov * 0.92
        denom = 1.0 + nov * quad
        P_next = P - (nov / denom) * torch.bmm(pk, pk.transpose(1, 2))
        P_mid = 0.5 * (P_next + P_next.transpose(1, 2))
        rho_rec = ((1.0 - g_u[m]) * w_cov * rho_g).clamp(0.0, rho_c)
        P = (1.0 - rho_rec) * P_mid + rho_rec * P0
        P = 0.5 * (P + P.transpose(1, 2))
    return uncerts


@torch.jit.script
def _tern_m_block_scan_jit(
    M0: torch.Tensor,
    q_blk: torch.Tensor,
    k_blk: torch.Tensor,
    v_blk: torch.Tensor,
    b_blk: torch.Tensor,
    g_blk: torch.Tensor,
    inv_dh: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    blen = q_blk.size(0)
    M_mat = M0
    q_u = q_blk.unbind(0)
    k_u = k_blk.unbind(0)
    v_u = v_blk.unbind(0)
    b_u = b_blk.unbind(0)
    g_u = g_blk.unbind(0)
    outs: list[torch.Tensor] = []
    for m in range(blen):
        outs.append(torch.bmm(M_mat, q_u[m]))
        kt = k_u[m]
        gt = g_u[m]
        err = v_u[m] - gt * torch.bmm(M_mat, kt)
        r_sq = err.pow(2).sum(dim=1, keepdim=True) * inv_dh
        err_eff = err * torch.rsqrt(torch.sqrt(1.0 + r_sq * r_sq))
        M_mat = gt * M_mat + (1.35 * b_u[m]) * torch.bmm(err_eff, kt.transpose(1, 2))
    return torch.stack(outs, dim=0), M_mat


class _CheckpointedTernaryJitFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q_m: torch.Tensor,
        k_m: torch.Tensor,
        v_m: torch.Tensor,
        b_m: torch.Tensor,
        g_m: torch.Tensor,
        inv_dh: float,
        Bc: int,
    ) -> torch.Tensor:
        M_len, BH, d_k, C = q_m.shape
        d_v = v_m.size(2)
        n_blocks = (M_len + Bc - 1) // Bc
        M_starts = torch.empty(
            n_blocks, BH, d_v, d_k, device=q_m.device, dtype=torch.float32
        )
        M_cur = torch.zeros(BH, d_v, d_k, device=q_m.device, dtype=torch.float32)
        out_all = torch.empty(M_len, BH, d_v, C, device=q_m.device, dtype=torch.float32)
        for bi in range(n_blocks):
            s = bi * Bc
            e = min(M_len, s + Bc)
            M_starts[bi].copy_(M_cur)
            o_b, M_cur = _tern_m_block_scan_jit(
                M_cur, q_m[s:e], k_m[s:e], v_m[s:e], b_m[s:e], g_m[s:e], inv_dh
            )
            out_all[s:e].copy_(o_b)
        ctx.save_for_backward(q_m, k_m, v_m, b_m, g_m, M_starts)
        ctx.inv_dh = inv_dh
        ctx.Bc = Bc
        return out_all

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q_m, k_m, v_m, b_m, g_m, M_starts = ctx.saved_tensors
        inv_dh = ctx.inv_dh
        Bc = ctx.Bc
        M_len, BH, d_k, _ = q_m.shape
        d_v = v_m.size(2)
        n_blocks = M_starts.size(0)

        dq = torch.zeros_like(q_m)
        dk = torch.zeros_like(k_m)
        dv = torch.zeros_like(v_m)
        db = torch.zeros_like(b_m)
        dg = torch.zeros_like(g_m)
        g_M = torch.zeros(BH, d_v, d_k, device=q_m.device, dtype=torch.float32)

        for bi in range(n_blocks - 1, -1, -1):
            s = bi * Bc
            e = min(M_len, s + Bc)
            M0 = M_starts[bi].detach().requires_grad_(True)
            qb = q_m[s:e].detach().requires_grad_(True)
            kb = k_m[s:e].detach().requires_grad_(True)
            vb = v_m[s:e].detach().requires_grad_(True)
            bb = b_m[s:e].detach().requires_grad_(True)
            gb = g_m[s:e].detach().requires_grad_(True)
            go_b = grad_out[s:e]
            with torch.enable_grad():
                o_b, M_end = _tern_m_block_scan_jit(M0, qb, kb, vb, bb, gb, inv_dh)
                g_M, g_qb, g_kb, g_vb, g_bb, g_gb = torch.autograd.grad(
                    (o_b, M_end), (M0, qb, kb, vb, bb, gb), (go_b, g_M)
                )
            dq[s:e].copy_(g_qb)
            dk[s:e].copy_(g_kb)
            dv[s:e].copy_(g_vb)
            db[s:e].copy_(g_bb)
            dg[s:e].copy_(g_gb)

        return dq, dk, dv, db, dg, None, None


def block_checkpointed_tern_scan(
    q_chunks: torch.Tensor,
    k_seq: torch.Tensor,
    v_seq: torch.Tensor,
    beta_seq: torch.Tensor,
    gamma_seq: torch.Tensor,
    inv_lam: float,
    q_epi_chunks: torch.Tensor | None = None,
    k_epi_seq: torch.Tensor | None = None,
    tern_gate: float = 1.0,
    Bc: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Block-checkpointed scan (`Bc=32`) for nonlinear quartic `DH=144.0` ternary Woodbury
    memory (`~4x` VRAM reduction vs full unrolled TorchScript scan).
    """
    B, M_len, C, H, d_k = q_chunks.shape
    d_v = v_seq.size(-1)
    BH = B * H
    inv_dh = tern_gate / 144.0
    qe_src = q_epi_chunks if q_epi_chunks is not None else q_chunks
    ke_src = k_epi_seq if k_epi_seq is not None else k_seq

    with torch.no_grad():
        qe_m = qe_src.detach().float().permute(1, 0, 3, 4, 2).reshape(M_len, BH, d_k, C)
        ke_m = ke_src.detach().float().permute(1, 0, 2, 3).reshape(M_len, BH, d_k, 1)
        bm_d = beta_seq.detach().float().permute(1, 0, 2, 3).reshape(M_len, BH, 1, 1)
        gm_d = gamma_seq.detach().float().permute(1, 0, 2, 3).reshape(M_len, BH, 1, 1)
        u_m = _tern_p_uncerts_nograd_jit(qe_m, ke_m, bm_d, gm_d, inv_lam, tern_gate)
        uncerts = (
            u_m.reshape(M_len, B, H, C)
            .permute(1, 0, 3, 2)
            .reshape(B, M_len * C, H)
            .to(v_seq.dtype)
        )

    q_m = q_chunks.float().permute(1, 0, 3, 4, 2).reshape(M_len, BH, d_k, C).contiguous()
    k_m = k_seq.float().permute(1, 0, 2, 3).reshape(M_len, BH, d_k, 1).contiguous()
    v_m = v_seq.float().permute(1, 0, 2, 3).reshape(M_len, BH, d_v, 1).contiguous()
    b_m = beta_seq.float().permute(1, 0, 2, 3).reshape(M_len, BH, 1, 1).contiguous()
    g_m = gamma_seq.float().permute(1, 0, 2, 3).reshape(M_len, BH, 1, 1).contiguous()
    out_m = _CheckpointedTernaryJitFn.apply(q_m, k_m, v_m, b_m, g_m, inv_dh, Bc)
    out_all = (
        out_m.reshape(M_len, B, H, d_v, C)
        .permute(1, 0, 4, 2, 3)
        .reshape(B, M_len * C, H, d_v)
        .to(v_seq.dtype)
    )
    return out_all, uncerts


def neural_woodbury_chunk_scan_jit(
    q_chunks: torch.Tensor,
    k_seq: torch.Tensor,
    v_seq: torch.Tensor,
    beta_seq: torch.Tensor,
    gamma_seq: torch.Tensor,
    inv_lam: float,
    q_epi_chunks: torch.Tensor | None = None,
    k_epi_seq: torch.Tensor | None = None,
    tern_gate: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    M_len = q_chunks.size(1)
    if tern_gate <= 0.0:
        if M_len >= 8:
            return block_parallel_wy_fp_scan(
                q_chunks,
                k_seq,
                v_seq,
                beta_seq,
                gamma_seq,
                inv_lam,
                q_epi_chunks,
                k_epi_seq,
                Bc=32,
                checkpointed=torch.is_grad_enabled() and M_len > 32,
            )
        return _neural_woodbury_chunk_scan_fp_jit(
            q_chunks, k_seq, v_seq, beta_seq, gamma_seq, inv_lam, q_epi_chunks, k_epi_seq
        )
    if torch.is_grad_enabled() and M_len > 32:
        return block_checkpointed_tern_scan(
            q_chunks,
            k_seq,
            v_seq,
            beta_seq,
            gamma_seq,
            inv_lam,
            q_epi_chunks,
            k_epi_seq,
            tern_gate=tern_gate,
            Bc=32,
        )
    return _neural_woodbury_chunk_scan_tern_jit(
        q_chunks, k_seq, v_seq, beta_seq, gamma_seq, inv_lam, q_epi_chunks, k_epi_seq, tern_gate
    )


@torch.no_grad()
def inspect_woodbury_spd_spectrum(
    k_epi_seq: torch.Tensor,
    beta_seq: torch.Tensor,
    gamma_seq: torch.Tensor,
    inv_lam: float,
    sample_every: int = 32,
    tern_gate: float = 0.0,
) -> tuple[float, float, float]:
    """
    Audits exact SPD & symmetry bounds `(0 < min_eig <= max_eig <= inv_lam)`
    of the `neural_woodbury_chunk_scan_jit` precision matrix `P_t` across horizon `M_len`.
    """
    B, M_len, H, d_k = k_epi_seq.shape
    BH = B * H
    rho_g = 0.002 * tern_gate
    rho_c = 0.002 * tern_gate
    slope = 20.0 if rho_g > 0.0 else 18.0
    P0 = (
        torch.eye(d_k, device=k_epi_seq.device, dtype=torch.float32)
        .unsqueeze(0)
        .expand(BH, d_k, d_k)
        * float(inv_lam)
    )
    P = P0.clone()
    ke_u = k_epi_seq.float().permute(1, 0, 2, 3).reshape(M_len, BH, d_k, 1).unbind(0)
    b_u = beta_seq.float().permute(1, 0, 2, 3).reshape(M_len, BH, 1, 1).unbind(0)
    g_u = gamma_seq.float().permute(1, 0, 2, 3).reshape(M_len, BH, 1, 1).unbind(0)

    min_eig = float(inv_lam)
    max_eig = 0.0
    max_sym_err = 0.0
    for m in range(M_len):
        ket = ke_u[m]
        bt_d = b_u[m]
        gt_d = g_u[m]
        s_gate = torch.sigmoid((bt_d - 0.26) * slope)
        w_cov = s_gate * s_gate * s_gate

        pk = torch.bmm(P, ket)
        quad = (ket * pk).sum(dim=1, keepdim=True).clamp(min=1e-5)
        nov = w_cov * 0.92
        denom = 1.0 + nov * quad
        P_next = P - (nov / denom) * torch.bmm(pk, pk.transpose(1, 2))
        P_mid = 0.5 * (P_next + P_next.transpose(1, 2))
        if rho_g > 0.0:
            rho_rec = ((1.0 - gt_d) * w_cov * rho_g).clamp(0.0, rho_c)
            P = (1.0 - rho_rec) * P_mid + rho_rec * P0
            P = 0.5 * (P + P.transpose(1, 2))
        else:
            P = P_mid

        if (m % sample_every == 0) or (m == M_len - 1):
            sym_err = float((P - P.transpose(-1, -2)).abs().max().item())
            eigs = torch.linalg.eigvalsh(P)
            min_eig = min(min_eig, float(eigs.min().item()))
            max_eig = max(max_eig, float(eigs.max().item()))
            max_sym_err = max(max_sym_err, sym_err)
    return min_eig, max_eig, max_sym_err


class NeuralWoodburyLayer(nn.Module):
    """
    Stream C of SimVQ-Split v4: Hybrid `(H-1)+1` Query-Intent Phrase-Latch Woodbury Memory
    with Kurtosis-Controlled 3-Factor Hadamard Sketch, Cubic-Gated Analytic Bayesian Sensor
    (`u_t = lambda * q_t^T P_t q_t`), and Dedicated Continuous Prose Context Head.
    """

    def __init__(
        self,
        d_model: int,
        mem_heads: int = 5,
        mem_dim: int = 32,
        ridge_lambda: float = 0.04,
        chunk_size: int = 4,
        latch_span: int = 7,
    ):
        super().__init__()
        self.d_model = d_model
        self.H = mem_heads
        self.H_epi = max(1, mem_heads - 1)
        self.H_prose = mem_heads - self.H_epi
        self.D = mem_dim
        self.C = chunk_size
        self.L_latch = latch_span  # causal offsets [-6, -5, -4, -3, -2, -1, 0]
        self.epi_dim = self.H_epi * mem_dim
        self.total_dim = mem_heads * mem_dim
        self.inv_lam = 1.0 / ridge_lambda
        self.qat_active_mix: float = 0.0

        # Learnable lexical-vs-contextual blend (init sigmoid(1.7) = 0.85 lexical, 0.15 contextual)
        self.lex_logit = nn.Parameter(torch.tensor(1.70))

        # 3-factor local phrase projections for epistemic heads
        self.f1_proj = nn.Linear(d_model, self.epi_dim, bias=False)
        self.f2_proj = nn.Linear(d_model, self.epi_dim, bias=False)
        self.f3_proj = nn.Linear(d_model, self.epi_dim, bias=False)

        # Dedicated continuous prose context head + query-intent brake
        if self.H_prose > 0:
            prose_dim = self.H_prose * mem_dim
            self.prose_q_proj = nn.Linear(d_model, prose_dim, bias=False)
            self.prose_k_proj = nn.Linear(d_model, prose_dim, bias=False)
            gen = torch.Generator().manual_seed(314159)
            if d_model >= prose_dim:
                q_pr, _ = torch.linalg.qr(torch.randn(d_model, d_model, generator=gen))
                base_pr = q_pr[:prose_dim]
                with torch.no_grad():
                    self.prose_q_proj.weight.copy_(
                        base_pr + torch.randn(prose_dim, d_model, generator=gen) * 0.08
                    )
                    self.prose_k_proj.weight.copy_(
                        base_pr + torch.randn(prose_dim, d_model, generator=gen) * 0.08
                    )
            self.query_intent_proj = nn.Linear(d_model, 1, bias=True)
            nn.init.constant_(self.query_intent_proj.bias, -0.25)
        else:
            self.prose_q_proj = None
            self.prose_k_proj = None
            self.query_intent_proj = None

        # Shared causal subject-latch salience + relative lag prior peaking at offset -3 (Surname)
        self.subj_salience = nn.Linear(d_model, self.H_epi, bias=False)
        nn.init.normal_(self.subj_salience.weight, std=0.02)
        init_lag = torch.tensor([-1.5, -0.5, 0.2, 1.9, -0.5, -1.5, -2.0]).view(
            1, latch_span
        )
        self.subj_lag_bias = nn.Parameter(init_lag.repeat(self.H_epi, 1))

        self.write_pool_score = nn.Linear(d_model, mem_heads, bias=False)
        self.v_proj = nn.Linear(d_model, self.total_dim, bias=False)
        self.beta_proj = nn.Linear(d_model, mem_heads, bias=True)
        self.gamma_proj = nn.Linear(d_model, mem_heads, bias=True)
        with torch.no_grad():
            self.beta_proj.bias.copy_(
                torch.tensor([-1.15] * self.H_epi + [-0.10] * self.H_prose)
            )
            self.gamma_proj.bias.copy_(
                torch.tensor([4.1] * self.H_epi + [2.4] * self.H_prose)
            )

        self.out_norm = RMSNorm(self.total_dim)
        self.conf_thresh = nn.Parameter(torch.full((self.H_epi,), 0.42))
        self.conf_slope = nn.Parameter(torch.full((self.H_epi,), 8.0))
        self.uncert_scale = nn.Parameter(torch.ones(self.H_epi * 3) * 1.5)
        self.out_proj = nn.Linear(self.total_dim + self.H_epi * 3, d_model, bias=False)
        self.last_epistemic_uncertainty: torch.Tensor | None = None
        self.last_gate_loss: torch.Tensor | None = None
        self._tern_cache_key: tuple[int, int] | None = None
        self._tern_cache_val: float = 0.0

    def _tern_gate(self) -> float:
        if self.qat_active_mix > 0.0 or isinstance(self.v_proj, PackedLUT2BitLinear):
            return 1.0
        w = self.v_proj.weight
        key = (int(w.data_ptr()), int(w._version))
        if self._tern_cache_key == key:
            return self._tern_cache_val
        with torch.no_grad():
            w_s = w[0, : min(32, w.size(1))].detach().float().unsqueeze(0)
            w_q, _, _ = quantize_ternary_weights_ls(w_s)
            is_tern = bool(
                ((w_s - w_q).norm() < 1e-3 * w_s.norm().clamp(min=1e-6)).item()
            )
        val = 1.0 if is_tern else 0.0
        self._tern_cache_key = key
        self._tern_cache_val = val
        return val

    def _safe_unit(self, v: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        vf = v.float()
        return vf * (torch.rsqrt(vf.pow(2).sum(dim=-1, keepdim=True) + 1e-5) * scale)

    def _compute_latched_keys(
        self, x: torch.Tensor, h_lex: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D_in = x.shape
        x_u = self._safe_unit(x)
        if h_lex is not None:
            w_lex = torch.sigmoid(self.lex_logit.float())
            e_u = self._safe_unit(h_lex)
            z = self._safe_unit(w_lex * e_u + (1.0 - w_lex) * x_u)
        else:
            z = x_u

        # Damped strictly causal sequence centering (eliminates c_0^2 linear polynomial leak)
        steps = torch.arange(1, T + 1, device=x.device, dtype=torch.float32).view(
            1, T, 1
        )
        c_mean = torch.cumsum(z, dim=1) / (steps + 2.0)
        zc = (z - c_mean).to(x.dtype)

        # Causal left-replicate boundary for [t-3, t-2, t-1, t]
        z_head_rep = zc[:, :1].expand(B, 3, D_in)
        zp = torch.cat([z_head_rep, zc], dim=1)  # (B, T + 3, D_in)
        z0 = zp[:, 3:]  # offset 0
        z12 = zp[:, 2:-1] + zp[:, 1:-2]  # offsets -1, -2
        z3 = zp[:, 0:-3] + 0.25 * zp[:, 1:-2]  # offset -3 (+ light -2 support)

        sc_raw = math.sqrt(self.D)
        f1_r = self._safe_unit(self.f1_proj(z0).view(B, T, self.H_epi, self.D), sc_raw)
        f2_r = self._safe_unit(
            self.f2_proj(z12).view(B, T, self.H_epi, self.D), sc_raw
        )
        f3_r = self._safe_unit(self.f3_proj(z3).view(B, T, self.H_epi, self.D), sc_raw)
        f1_e = torch.tanh(f1_r * 1.2)
        f2_e = torch.tanh(f2_r * 1.2)
        f3_e = torch.tanh(f3_r * 1.2)

        phi_epi = self._safe_unit(f1_e * f2_e * f3_e)
        phi_lin = self._safe_unit(f1_r + f2_r + f3_r)
        phi_rw = self._safe_unit(0.35 * phi_lin + 0.94 * phi_epi)

        # Shared Causal Subject-Head Latch over [t - (L_latch - 1) .. t]
        K = self.L_latch
        sal = self.subj_salience(x).float()  # (B, T, H_epi)
        sal_pad = F.pad(sal, (0, 0, K - 1, 0), value=-1e4)  # (B, T + K - 1, H_epi)
        sal_win = sal_pad.unfold(1, K, 1)  # (B, T, H_epi, K)
        logits = sal_win + self.subj_lag_bias.float().view(1, 1, self.H_epi, K)
        att = F.softmax(logits, dim=-1)  # (B, T, H_epi, K)

        rw_pad = F.pad(phi_rw.reshape(B, T, self.epi_dim), (0, 0, K - 1, 0))
        rw_win = rw_pad.unfold(1, K, 1).view(B, T, self.H_epi, self.D, K)
        latched_rw = self._safe_unit((rw_win * att.unsqueeze(3)).sum(dim=-1)).to(
            x.dtype
        )

        epi_pad = F.pad(phi_epi.reshape(B, T, self.epi_dim), (0, 0, K - 1, 0))
        epi_win = epi_pad.unfold(1, K, 1).view(B, T, self.H_epi, self.D, K)
        latched_epi = self._safe_unit((epi_win * att.unsqueeze(3)).sum(dim=-1)).to(
            x.dtype
        )

        return latched_rw, latched_epi

    def encode_phrase_sketch(
        self, x: torch.Tensor, h_lex: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Returns pure 3-factor Hadamard epistemic phrase sketches `(B, T, H, D)`."""
        _, q_epi_t = self._compute_latched_keys(x, h_lex=h_lex)
        if self.H_prose > 0:
            return torch.cat([q_epi_t, q_epi_t[:, :, : self.H_prose]], dim=2)
        return q_epi_t

    def forward(
        self, x: torch.Tensor, h_lex: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, T, D_in = x.shape
        C = self.C
        q_rw_t, q_epi_t = self._compute_latched_keys(x, h_lex=h_lex)

        if self.H_prose > 0:
            q_prose = self._safe_unit(
                self.prose_q_proj(x).view(B, T, self.H_prose, self.D)
            ).to(x.dtype)
            k_prose = self._safe_unit(
                self.prose_k_proj(x).view(B, T, self.H_prose, self.D)
            ).to(x.dtype)
            q_rw_all = torch.cat([q_rw_t, q_prose], dim=2)
            k_src_all = torch.cat([q_rw_t, k_prose], dim=2)
            q_epi_all = torch.cat([q_epi_t, q_epi_t[:, :, : self.H_prose]], dim=2)
        else:
            q_rw_all = q_rw_t
            k_src_all = q_rw_t
            q_epi_all = q_epi_t

        pad = (C - (T % C)) % C
        if pad > 0:
            x_pad = F.pad(x, (0, 0, 0, pad))
            q_rw_pad = F.pad(q_rw_all, (0, 0, 0, 0, 0, pad))
            k_src_pad = F.pad(k_src_all, (0, 0, 0, 0, 0, pad))
            q_epi_pad = F.pad(q_epi_all, (0, 0, 0, 0, 0, pad))
        else:
            x_pad = x
            q_rw_pad = q_rw_all
            k_src_pad = k_src_all
            q_epi_pad = q_epi_all

        T_pad = x_pad.size(1)
        M_len = T_pad // C

        q_chunks = q_rw_pad.view(B, M_len, C, self.H, self.D)
        k_chunks = k_src_pad.view(B, M_len, C, self.H, self.D)
        q_epi_chunks = q_epi_pad.view(B, M_len, C, self.H, self.D)

        x_chunks = x_pad.view(B, M_len, C, D_in)
        w_scores = F.softmax(self.write_pool_score(x_chunks).float(), dim=2)

        k_seq = self._safe_unit(
            (k_chunks.float() * w_scores.unsqueeze(-1)).sum(dim=2)
        ).to(x.dtype)
        k_epi_seq = self._safe_unit(
            (q_epi_chunks.float() * w_scores.unsqueeze(-1)).sum(dim=2)
        ).to(x.dtype)

        v_tok = self.v_proj(x_chunks).view(B, M_len, C, self.H, self.D)
        v_seq = (v_tok * w_scores.to(x.dtype).unsqueeze(-1)).sum(dim=2)

        z_head = (
            (x_chunks.unsqueeze(3) * w_scores.to(x.dtype).unsqueeze(-1))
            .sum(dim=2)
            .mean(dim=2)
        )
        beta = torch.sigmoid(self.beta_proj(z_head)).unsqueeze(-1)
        gamma = torch.sigmoid(self.gamma_proj(z_head)).unsqueeze(-1)

        t_gate = self._tern_gate()
        read_pad, uncert_pad = neural_woodbury_chunk_scan_jit(
            q_chunks,
            k_seq,
            v_seq,
            beta,
            gamma,
            self.inv_lam,
            q_epi_chunks,
            k_epi_seq,
            t_gate,
        )
        read_seq = read_pad[:, :T].reshape(B, T, self.total_dim)
        uncert_seq = uncert_pad[:, :T]
        self.last_epistemic_uncertainty = uncert_seq

        u_epi = uncert_seq[:, :, : self.H_epi].float()
        conf_gate_epi = torch.sigmoid(
            self.conf_slope.float().view(1, 1, self.H_epi)
            * (self.conf_thresh.float().view(1, 1, self.H_epi) - u_epi)
        )

        if self.H_prose > 0:
            q_intent = torch.sigmoid(self.query_intent_proj(x).float())
            self.last_gate_loss = None
            prose_gate = (1.0 - 0.88 * q_intent).expand(B, T, self.H_prose)
            all_gates = torch.cat([conf_gate_epi, prose_gate], dim=-1).unsqueeze(-1)
        else:
            q_intent = None
            self.last_gate_loss = None
            all_gates = conf_gate_epi.unsqueeze(-1)

        normed_read = (
            self.out_norm(read_seq).view(B, T, self.H, self.D) * all_gates.to(x.dtype)
        ).reshape(B, T, self.total_dim)

        raw_u_feats = torch.cat(
            [
                u_epi - 0.42,
                torch.tanh((u_epi - 0.42) * 7.0),
                F.relu(u_epi - 0.36) * 2.5,
            ],
            dim=-1,
        )
        if q_intent is not None:
            u_feats = q_intent.to(x.dtype) * (
                raw_u_feats.to(x.dtype) * self.uncert_scale.to(x.dtype)
            )
        else:
            u_feats = raw_u_feats.to(x.dtype) * self.uncert_scale.to(x.dtype)
        return self.out_proj(torch.cat([normed_read, u_feats], dim=-1))
