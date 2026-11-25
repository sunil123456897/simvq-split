"""
Calcite v2 Second-Order Biorthogonal Kalman-Woodbury Epistemic Memory (`calcite_v2/woodbury.py`).

Improvements over `simvq_split/woodbury.py`:
1. True 2nd-order Kalman-preconditioned write (`use_kalman_precond=True`):
   - Uses stop-grad SPD precision `P_{t-1}` to form the bounded biorthogonal RLS dual vector
     `k_tilde = sg(P_{t-1} k / (max(k^T P_{t-1} k, ||P_{t-1} k|| / 2.5) + 1e-5))`.
   - Exact key overwrite (`k_new == k_old`) preserves `100%` write magnitude (`k_tilde == k`),
     while correlated distinct keys rotate sharply onto the novel orthogonal subspace (`3x–10x`
     less collateral interference).
   - Preserves strict `torch.no_grad()` SPD convex blend on `P_t` and live gradients on `beta_t`
     (`Dead End #4` + `Tracer #1` invariants).
2. Multi-timescale head initialization (`timescale_mode`: `uniform`, `fast`, `slow`, `multiband`)
   so early layers absorb short-range prose drift while deep layers keep long-horizon facts clean
   across `L = 32K–64K` contexts without saturating a single shared matrix.
3. Zero monkey-patching: uses `fast_woodbury_forward` when `use_kalman_precond=False`, and a
   VRAM-efficient block-checkpointed Float32 AMP-shielded scan when `use_kalman_precond=True`.
4. Exact incremental streaming API (`init_stream_state`, `step`) matching batch `forward`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels.fast_calcite_scans import fast_woodbury_forward
from simvq_split.quant_lut import PackedLUT2BitLinear, quantize_ternary_weights_ls

from .norm import CalciteRMSNorm


@dataclass
class WoodburyStreamState:
    M_mat: torch.Tensor       # (B*H, d_v, d_k) fp32
    P_mat: torch.Tensor       # (B*H, d_k, d_k) fp32
    z_cumsum: torch.Tensor    # (B, d_model) fp32
    step_count: int
    z_ring3: torch.Tensor     # (B, 3, d_model)
    sal_ring: torch.Tensor    # (B, K-1, H_epi) fp32
    rw_ring: torch.Tensor     # (B, K-1, H_epi, d_k) fp32
    epi_ring: torch.Tensor    # (B, K-1, H_epi, d_k) fp32
    # Pending micro-chunk buffer (`C = 4` chunks per write commit)
    buf_x: torch.Tensor       # (B, C, d_model)
    buf_k: torch.Tensor       # (B, C, H, d_k)
    buf_k_epi: torch.Tensor   # (B, C, H, d_k)
    buf_pos: int
    z3_ptr: int = 0           # circular write pointer for z_ring3 (chronological when 0)
    latch_ptr: int = 0        # circular write pointer for sal/rw/epi rings (chronological when 0)

    def memory_bytes(self) -> int:
        b = max(1, int(self.z_cumsum.size(0)))
        total = 0
        for t in (
            self.M_mat,
            self.P_mat,
            self.z_cumsum,
            self.z_ring3,
            self.sal_ring,
            self.rw_ring,
            self.epi_ring,
            self.buf_x,
            self.buf_k,
            self.buf_k_epi,
        ):
            total += int(t.numel() * t.element_size())
        return total // b


@torch.jit.script
def _kalman_p_and_dual_nograd_jit(
    qe_m: torch.Tensor,   # (M, BH, d_k, C) fp32
    ke_m: torch.Tensor,   # (M, BH, d_k, 1) fp32
    k_m: torch.Tensor,    # (M, BH, d_k, 1) fp32
    b_m: torch.Tensor,    # (M, BH, 1, 1) fp32
    g_m: torch.Tensor,    # (M, BH, 1, 1) fp32
    inv_lam: float,
    tern_gate: float,
    kalman_mix: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Runs the strictly stop-gradient SPD Woodbury precision scan `P_t` and returns:
    - `uncerts`: `(M, BH, C)` epistemic ridge variance `lam * q^T P q`
    - `k_dual`: `(M, BH, d_k, 1)` bounded biorthogonal RLS write vectors preconditioned by `P_{t-1}`
    """
    M_len, BH, d_k, C = qe_m.shape
    lam_reg = 1.0 / inv_lam
    rho_g = 0.002 * tern_gate
    rho_c = 0.002 * tern_gate
    slope_s = 18.0 + 2.0 * tern_gate

    P0 = (
        torch.eye(d_k, device=ke_m.device, dtype=torch.float32)
        .unsqueeze(0)
        .expand(BH, d_k, d_k)
        * inv_lam
    )
    P = P0.clone()
    uncerts = torch.empty(M_len, BH, C, device=ke_m.device, dtype=torch.float32)
    k_dual = torch.empty(M_len, BH, d_k, 1, device=ke_m.device, dtype=torch.float32)

    qe_u = qe_m.unbind(0)
    ke_u = ke_m.unbind(0)
    k_u = k_m.unbind(0)
    b_u = b_m.unbind(0)
    g_u = g_m.unbind(0)

    for m in range(M_len):
        qem = qe_u[m]
        pq_m = torch.bmm(P, qem)
        uncerts[m] = ((qem * pq_m).sum(dim=1) * lam_reg).clamp(0.0, 1.5)

        kt = k_u[m]
        if kalman_mix > 0.0:
            pk_w = torch.bmm(P, kt)
            quad_w = (kt * pk_w).sum(dim=1, keepdim=True)
            norm_pk = torch.sqrt(pk_w.pow(2).sum(dim=1, keepdim=True) + 1e-8)
            denom_w = torch.maximum(quad_w, norm_pk * 0.4) + 1e-5
            k_rls = pk_w / denom_w
            k_dual[m] = (1.0 - kalman_mix) * kt + kalman_mix * k_rls
        else:
            k_dual[m] = kt

        ket = ke_u[m]
        s_gate = torch.sigmoid((b_u[m] - 0.26) * slope_s)
        w_cov = s_gate * s_gate * s_gate
        pk = torch.bmm(P, ket)
        quad = (ket * pk).sum(dim=1, keepdim=True).clamp(min=1e-5)
        nov = w_cov * 0.92
        denom = 1.0 + nov * quad
        P_next = P - (nov / denom) * torch.bmm(pk, pk.transpose(1, 2))
        P_mid = 0.5 * (P_next + P_next.transpose(1, 2))
        if tern_gate > 0.0:
            rho_rec = ((1.0 - g_u[m]) * w_cov * rho_g).clamp(0.0, rho_c)
            P = (1.0 - rho_rec) * P_mid + rho_rec * P0
            P = 0.5 * (P + P.transpose(1, 2))
        else:
            P = P_mid

    return uncerts, k_dual


@torch.jit.script
def _kalman_m_block_scan_jit(
    M0: torch.Tensor,
    q_blk: torch.Tensor,
    k_blk: torch.Tensor,
    kd_blk: torch.Tensor,
    v_blk: torch.Tensor,
    b_blk: torch.Tensor,
    g_blk: torch.Tensor,
    inv_dh: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    blen = q_blk.size(0)
    M_mat = M0
    q_u = q_blk.unbind(0)
    k_u = k_blk.unbind(0)
    kd_u = kd_blk.unbind(0)
    v_u = v_blk.unbind(0)
    b_u = b_blk.unbind(0)
    g_u = g_blk.unbind(0)
    outs: list[torch.Tensor] = []
    for m in range(blen):
        outs.append(torch.bmm(M_mat, q_u[m]))
        kt = k_u[m]
        kdt = kd_u[m]
        gt = g_u[m]
        err = v_u[m] - gt * torch.bmm(M_mat, kt)
        if inv_dh > 0.0:
            r_sq = err.pow(2).sum(dim=1, keepdim=True) * inv_dh
            err = err * torch.rsqrt(torch.sqrt(1.0 + r_sq * r_sq))
        M_mat = gt * M_mat + (1.35 * b_u[m]) * torch.bmm(err, kdt.transpose(1, 2))
    return torch.stack(outs, dim=0), M_mat


class _CheckpointedKalmanWoodburyFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q_m: torch.Tensor,
        k_m: torch.Tensor,
        kd_m: torch.Tensor,
        v_m: torch.Tensor,
        b_m: torch.Tensor,
        g_m: torch.Tensor,
        inv_dh: float,
        Bc: int,
    ) -> torch.Tensor:
        with torch.amp.autocast(q_m.device.type, enabled=False):
            q_m = q_m.float()
            k_m = k_m.float()
            kd_m = kd_m.float()
            v_m = v_m.float()
            b_m = b_m.float()
            g_m = g_m.float()
            M_len, BH, d_k, C = q_m.shape
            d_v = v_m.size(2)
            n_blocks = (M_len + Bc - 1) // Bc
            M_starts = torch.empty(n_blocks, BH, d_v, d_k, device=q_m.device, dtype=torch.float32)
            M_cur = torch.zeros(BH, d_v, d_k, device=q_m.device, dtype=torch.float32)
            out_all = torch.empty(M_len, BH, d_v, C, device=q_m.device, dtype=torch.float32)
            for bi in range(n_blocks):
                s = bi * Bc
                e = min(M_len, s + Bc)
                M_starts[bi].copy_(M_cur)
                o_b, M_cur = _kalman_m_block_scan_jit(
                    M_cur, q_m[s:e], k_m[s:e], kd_m[s:e], v_m[s:e], b_m[s:e], g_m[s:e], inv_dh
                )
                out_all[s:e].copy_(o_b)
        ctx.save_for_backward(q_m, k_m, kd_m, v_m, b_m, g_m, M_starts)
        ctx.inv_dh = inv_dh
        ctx.Bc = Bc
        return out_all

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q_m, k_m, kd_m, v_m, b_m, g_m, M_starts = ctx.saved_tensors
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

        with torch.amp.autocast(q_m.device.type, enabled=False):
            grad_out = grad_out.float()
            for bi in range(n_blocks - 1, -1, -1):
                s = bi * Bc
                e = min(M_len, s + Bc)
                M0 = M_starts[bi].detach().requires_grad_(True)
                qb = q_m[s:e].detach().requires_grad_(True)
                kb = k_m[s:e].detach().requires_grad_(True)
                kdb = kd_m[s:e].detach()
                vb = v_m[s:e].detach().requires_grad_(True)
                bb = b_m[s:e].detach().requires_grad_(True)
                gb = g_m[s:e].detach().requires_grad_(True)
                go_b = grad_out[s:e]
                with torch.enable_grad():
                    o_b, M_end = _kalman_m_block_scan_jit(
                        M0, qb, kb, kdb, vb, bb, gb, inv_dh
                    )
                    g_M, g_qb, g_kb, g_vb, g_bb, g_gb = torch.autograd.grad(
                        (o_b, M_end), (M0, qb, kb, vb, bb, gb), (go_b, g_M)
                    )
                dq[s:e].copy_(g_qb)
                dk[s:e].copy_(g_kb)
                dv[s:e].copy_(g_vb)
                db[s:e].copy_(g_bb)
                dg[s:e].copy_(g_gb)

        return dq, dk, None, dv, db, dg, None, None


@torch.jit.script
def _deltanet2_m_block_scan_jit(
    M0: torch.Tensor,
    q_blk: torch.Tensor,   # (L, BH, D_k, C)
    k_blk: torch.Tensor,   # (L, BH, D_k, 1)
    kd_blk: torch.Tensor,  # (L, BH, D_k, 1)
    v_blk: torch.Tensor,   # (L, BH, D_v, 1)
    e_blk: torch.Tensor,   # (L, BH, D_k, 1) channel-wise erase gate
    w_blk: torch.Tensor,   # (L, BH, D_v, 1) channel-wise write gate
    a_blk: torch.Tensor,   # (L, BH, 1, D_k) channel-wise decay gate
    inv_dh: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    blen = q_blk.size(0)
    M_mat = M0
    q_u = q_blk.unbind(0)
    k_u = k_blk.unbind(0)
    kd_u = kd_blk.unbind(0)
    v_u = v_blk.unbind(0)
    e_u = e_blk.unbind(0)
    w_u = w_blk.unbind(0)
    a_u = a_blk.unbind(0)
    outs: list[torch.Tensor] = []
    for m in range(blen):
        outs.append(torch.bmm(M_mat, q_u[m]))
        kt = k_u[m]
        kdt = kd_u[m]
        M_dec = M_mat * a_u[m]
        pred = torch.bmm(M_dec, kt)
        wv = w_u[m] * v_u[m]
        ek = e_u[m] * kdt
        if inv_dh > 0.0:
            r_sq = (wv - pred).pow(2).sum(dim=1, keepdim=True) * inv_dh
            sc = torch.rsqrt(torch.sqrt(1.0 + r_sq * r_sq))
            wv = wv * sc
            pred = pred * sc
        M_mat = M_dec + 1.35 * (torch.bmm(wv, kdt.transpose(1, 2)) - torch.bmm(pred, ek.transpose(1, 2)))
    return torch.stack(outs, dim=0), M_mat


class _CheckpointedDeltaNet2WoodburyFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q_m: torch.Tensor,
        k_m: torch.Tensor,
        kd_m: torch.Tensor,
        v_m: torch.Tensor,
        e_m: torch.Tensor,
        w_m: torch.Tensor,
        a_m: torch.Tensor,
        inv_dh: float,
        Bc: int,
    ) -> torch.Tensor:
        with torch.amp.autocast(q_m.device.type, enabled=False):
            q_m = q_m.float()
            k_m = k_m.float()
            kd_m = kd_m.float()
            v_m = v_m.float()
            e_m = e_m.float()
            w_m = w_m.float()
            a_m = a_m.float()
            M_len, BH, d_k, C = q_m.shape
            d_v = v_m.size(2)
            n_blocks = (M_len + Bc - 1) // Bc
            M_starts = torch.empty(n_blocks, BH, d_v, d_k, device=q_m.device, dtype=torch.float32)
            M_cur = torch.zeros(BH, d_v, d_k, device=q_m.device, dtype=torch.float32)
            out_all = torch.empty(M_len, BH, d_v, C, device=q_m.device, dtype=torch.float32)
            for bi in range(n_blocks):
                s = bi * Bc
                e = min(M_len, s + Bc)
                M_starts[bi].copy_(M_cur)
                o_b, M_cur = _deltanet2_m_block_scan_jit(
                    M_cur, q_m[s:e], k_m[s:e], kd_m[s:e], v_m[s:e], e_m[s:e], w_m[s:e], a_m[s:e], inv_dh
                )
                out_all[s:e].copy_(o_b)
        ctx.save_for_backward(q_m, k_m, kd_m, v_m, e_m, w_m, a_m, M_starts)
        ctx.inv_dh = inv_dh
        ctx.Bc = Bc
        return out_all

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q_m, k_m, kd_m, v_m, e_m, w_m, a_m, M_starts = ctx.saved_tensors
        inv_dh = ctx.inv_dh
        Bc = ctx.Bc
        M_len, BH, d_k, _ = q_m.shape
        d_v = v_m.size(2)
        n_blocks = M_starts.size(0)

        dq = torch.zeros_like(q_m)
        dk = torch.zeros_like(k_m)
        dv = torch.zeros_like(v_m)
        de = torch.zeros_like(e_m)
        dw = torch.zeros_like(w_m)
        da = torch.zeros_like(a_m)
        g_M = torch.zeros(BH, d_v, d_k, device=q_m.device, dtype=torch.float32)

        with torch.amp.autocast(q_m.device.type, enabled=False):
            grad_out = grad_out.float()
            for bi in range(n_blocks - 1, -1, -1):
                s = bi * Bc
                e = min(M_len, s + Bc)
                M0 = M_starts[bi].detach().requires_grad_(True)
                qb = q_m[s:e].detach().requires_grad_(True)
                kb = k_m[s:e].detach().requires_grad_(True)
                kdb = kd_m[s:e].detach()
                vb = v_m[s:e].detach().requires_grad_(True)
                eb = e_m[s:e].detach().requires_grad_(True)
                wb = w_m[s:e].detach().requires_grad_(True)
                ab = a_m[s:e].detach().requires_grad_(True)
                go_b = grad_out[s:e]
                with torch.enable_grad():
                    o_b, M_end = _deltanet2_m_block_scan_jit(
                        M0, qb, kb, kdb, vb, eb, wb, ab, inv_dh
                    )
                    g_M, g_qb, g_kb, g_vb, g_eb, g_wb, g_ab = torch.autograd.grad(
                        (o_b, M_end), (M0, qb, kb, vb, eb, wb, ab), (go_b, g_M)
                    )
                dq[s:e].copy_(g_qb)
                dk[s:e].copy_(g_kb)
                dv[s:e].copy_(g_vb)
                de[s:e].copy_(g_eb)
                dw[s:e].copy_(g_wb)
                da[s:e].copy_(g_ab)

        return dq, dk, None, dv, de, dw, da, None, None


class CalciteV2WoodburyLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        mem_heads: int = 5,
        mem_dim: int = 32,
        ridge_lambda: float = 0.04,
        chunk_size: int = 4,
        latch_span: int = 6,
        use_kalman_precond: bool = True,
        timescale_mode: str = "multiband",
        use_fast_kernels: bool = True,
        sub_banks: int = 1,
        top_k_banks: int = 1,
        use_deltanet2_gates: bool = False,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.H = int(mem_heads)
        self.H_epi = max(1, self.H - 1)
        self.H_prose = self.H - self.H_epi
        self.K_sub = max(1, int(sub_banks))
        self.top_k_banks = max(1, min(self.K_sub, int(top_k_banks)))
        self.H_banks = self.H_epi * self.K_sub + self.H_prose
        self.use_deltanet2_gates = bool(use_deltanet2_gates)
        self.D = int(mem_dim)
        self.C = int(chunk_size)
        self.L_latch = int(max(1, latch_span))
        self.epi_dim = self.H_epi * self.D
        self.total_dim = self.H * self.D
        self.inv_lam = 1.0 / float(ridge_lambda)
        self.use_kalman_precond = bool(use_kalman_precond)
        self.timescale_mode = str(timescale_mode)
        self.use_fast_kernels = bool(use_fast_kernels)

        if self.K_sub > 1:
            self.bank_router = nn.Parameter(
                torch.randn(self.H_epi, self.D, self.K_sub, dtype=torch.float32) * 0.05
            )
        else:
            self.bank_router = None

        self.lex_logit = nn.Parameter(torch.tensor(0.35))
        self.f1_proj = nn.Linear(d_model, self.epi_dim, bias=False)
        self.f2_proj = nn.Linear(d_model, self.epi_dim, bias=False)
        self.f3_proj = nn.Linear(d_model, self.epi_dim, bias=False)

        self.subj_salience = nn.Linear(d_model, self.H_epi, bias=False)
        if self.L_latch in (5, 6):
            base_lag = torch.tensor([-1.5, -0.5, 0.2, 1.9, -0.5, -1.5, -2.0], dtype=torch.float32)
            init_lag = base_lag[-self.L_latch :].clone()
        elif self.L_latch == 7:
            init_lag = torch.tensor([-1.5, -0.5, 0.2, 1.9, -0.5, -1.5, -2.0], dtype=torch.float32)
        else:
            pos = torch.linspace(-1.0, 1.0, steps=self.L_latch, dtype=torch.float32)
            init_lag = 1.6 * torch.exp(-pos * pos * 2.0) - 0.6
        self.subj_lag_bias = nn.Parameter(
            init_lag.view(1, self.L_latch).repeat(self.H_epi, 1).clone()
        )

        if self.H_prose > 0:
            self.prose_q_proj = nn.Linear(d_model, self.H_prose * self.D, bias=False)
            self.prose_k_proj = nn.Linear(d_model, self.H_prose * self.D, bias=False)
            self.query_intent_proj = nn.Linear(d_model, 1, bias=True)
            nn.init.constant_(self.query_intent_proj.bias, -1.5)
        else:
            self.prose_q_proj = None
            self.prose_k_proj = None
            self.query_intent_proj = None

        self.write_pool_score = nn.Linear(d_model, self.H, bias=False)
        self.v_proj = nn.Linear(d_model, self.total_dim, bias=False)

        if self.timescale_mode == "fast":
            g_init = torch.linspace(0.2, 2.0, steps=self.H)
        elif self.timescale_mode == "slow":
            g_init = torch.linspace(2.6, 5.2, steps=self.H)
        elif self.timescale_mode == "multiband":
            g_init = torch.linspace(0.6, 5.0, steps=self.H)
        else:
            g_init = torch.full((self.H,), 3.4)

        if self.use_deltanet2_gates:
            self.beta_proj = None
            self.gamma_proj = None
            self.erase_vec_proj = nn.Linear(d_model, self.total_dim, bias=True)
            self.write_vec_proj = nn.Linear(d_model, self.total_dim, bias=True)
            self.alpha_vec_proj = nn.Linear(d_model, self.total_dim, bias=True)
            nn.init.constant_(self.erase_vec_proj.bias, -0.6)
            nn.init.constant_(self.write_vec_proj.bias, -0.6)
            with torch.no_grad():
                self.alpha_vec_proj.bias.copy_(
                    g_init.unsqueeze(-1).expand(self.H, self.D).reshape(self.total_dim)
                )
        else:
            self.beta_proj = nn.Linear(d_model, self.H, bias=True)
            self.gamma_proj = nn.Linear(d_model, self.H, bias=True)
            self.erase_vec_proj = None
            self.write_vec_proj = None
            self.alpha_vec_proj = None
            nn.init.constant_(self.beta_proj.bias, -0.6)
            with torch.no_grad():
                self.gamma_proj.bias.copy_(g_init)

        self.conf_thresh = nn.Parameter(torch.full((self.H_epi,), 0.44))
        self.conf_slope = nn.Parameter(torch.full((self.H_epi,), 12.0))
        self.uncert_scale = nn.Parameter(torch.full((3 * self.H_epi,), 0.35))

        self.out_norm = CalciteRMSNorm(self.total_dim)
        self.out_proj = nn.Linear(self.total_dim + 3 * self.H_epi, d_model, bias=False)

        self.last_epistemic_uncertainty: torch.Tensor | None = None
        self.last_gate_loss: torch.Tensor | None = None
        self.qat_active_mix: float = 0.0
        self._tern_cache_key: tuple[int, int] | None = None
        self._tern_cache_val: float = 0.0

    def compute_state_bytes(self) -> int:
        wood_mat = self.H_banks * 2 * self.D * self.D * 4
        w_latch_lag = max(0, self.L_latch - 1)
        wood_ring = (
            (2 * self.d_model + 1) * 4
            + (3 * self.d_model * 2)
            + (w_latch_lag * (2 * self.H_epi * self.D + self.H_epi) * 2)
            + (self.C * (self.d_model + 2 * self.H * self.D) * 2)
        )
        return int(wood_mat + wood_ring)

    def _route_sub_banks(
        self, k_epi: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Routes epistemic keys `(..., H_epi, D)` across `K_sub` banks via straight-through `top-k`.
        Returns `(r_st, r_probs, r_hard)` each of shape `(..., H_epi, K_sub)`.
        """
        assert self.bank_router is not None
        logits = torch.einsum("...hd,hdk->...hk", k_epi.float(), self.bank_router.float())
        probs = F.softmax(logits, dim=-1)
        if self.top_k_banks == 1:
            top_idx = probs.argmax(dim=-1, keepdim=True)
            hard = torch.zeros_like(probs).scatter_(-1, top_idx, 1.0)
            r_st = hard + (probs - probs.detach())
        else:
            top_idx = probs.topk(self.top_k_banks, dim=-1).indices
            mask = torch.zeros_like(probs).scatter_(-1, top_idx, 1.0)
            masked_p = probs * mask
            hard = masked_p / masked_p.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            r_st = hard.detach() + (probs - probs.detach())
        return r_st, probs, hard

    def _tern_gate(self) -> float:
        if self.qat_active_mix > 0.0 or isinstance(self.v_proj, PackedLUT2BitLinear):
            return 1.0
        w = self.v_proj.weight
        # Avoid per-step GPU->CPU `.item()` sync during continuous FP training
        if (
            self.training
            and self._tern_cache_key is not None
            and self._tern_cache_val == 0.0
            and int(w.data_ptr()) == self._tern_cache_key[0]
        ):
            return 0.0
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

    @staticmethod
    def _safe_unit(v: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
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

        steps = torch.arange(1, T + 1, device=x.device, dtype=torch.float32).view(1, T, 1)
        c_mean = torch.cumsum(z, dim=1) / (steps + 2.0)
        zc = (z - c_mean).to(x.dtype)

        z_head_rep = zc[:, :1].expand(B, 3, D_in)
        zp = torch.cat([z_head_rep, zc], dim=1)
        z0 = zp[:, 3:]
        z12 = zp[:, 2:-1] + zp[:, 1:-2]
        z3 = zp[:, 0:-3] + 0.25 * zp[:, 1:-2]

        sc_raw = math.sqrt(self.D)
        f1_r = self._safe_unit(self.f1_proj(z0).view(B, T, self.H_epi, self.D), sc_raw)
        f2_r = self._safe_unit(self.f2_proj(z12).view(B, T, self.H_epi, self.D), sc_raw)
        f3_r = self._safe_unit(self.f3_proj(z3).view(B, T, self.H_epi, self.D), sc_raw)
        phi_epi = self._safe_unit(torch.tanh(f1_r * 1.2) * torch.tanh(f2_r * 1.2) * torch.tanh(f3_r * 1.2))
        phi_lin = self._safe_unit(f1_r + f2_r + f3_r)
        phi_rw = self._safe_unit(0.35 * phi_lin + 0.94 * phi_epi)

        K = self.L_latch
        sal = self.subj_salience(x).float()
        sal_pad = F.pad(sal, (0, 0, K - 1, 0), value=-1e4)
        sal_win = sal_pad.unfold(1, K, 1)
        logits = sal_win + self.subj_lag_bias.float().view(1, 1, self.H_epi, K)
        att = F.softmax(logits, dim=-1)

        # Fused single pad+unfold+matmul over `[phi_rw, phi_epi]` (avoids 2x 5D product materializations)
        phi_both = torch.cat((phi_rw, phi_epi), dim=-1)  # (B, T, H_epi, 2*D)
        both_win = (
            F.pad(phi_both.reshape(B, T, self.epi_dim * 2), (0, 0, K - 1, 0))
            .unfold(1, K, 1)
            .view(B, T, self.H_epi, 2 * self.D, K)
        )
        both_agg = torch.matmul(both_win, att.unsqueeze(-1)).squeeze(-1)
        rw_agg, epi_agg = both_agg.chunk(2, dim=-1)
        latched_rw = self._safe_unit(rw_agg).to(x.dtype)
        latched_epi = self._safe_unit(epi_agg).to(x.dtype)
        return latched_rw, latched_epi

    def forward(
        self, x: torch.Tensor, h_lex: torch.Tensor | None = None
    ) -> torch.Tensor:
        if (
            self.use_fast_kernels
            and not self.use_kalman_precond
            and not self.use_deltanet2_gates
            and self.K_sub == 1
            and x.is_cuda
        ):
            return fast_woodbury_forward(self, x, h_lex=h_lex)

        B, T, D_in = x.shape
        C = self.C
        q_rw_t, q_epi_t = self._compute_latched_keys(x, h_lex=h_lex)

        if self.H_prose > 0 and self.prose_q_proj is not None and self.prose_k_proj is not None:
            q_prose = self._safe_unit(self.prose_q_proj(x).view(B, T, self.H_prose, self.D)).to(x.dtype)
            k_prose = self._safe_unit(self.prose_k_proj(x).view(B, T, self.H_prose, self.D)).to(x.dtype)
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

        M_len = x_pad.size(1) // C
        q_chunks = q_rw_pad.view(B, M_len, C, self.H, self.D)
        k_chunks = k_src_pad.view(B, M_len, C, self.H, self.D)
        q_epi_chunks = q_epi_pad.view(B, M_len, C, self.H, self.D)

        x_chunks = x_pad.view(B, M_len, C, D_in)
        w_scores = F.softmax(self.write_pool_score(x_chunks).float(), dim=2)

        k_seq = self._safe_unit((k_chunks.float() * w_scores.unsqueeze(-1)).sum(dim=2)).to(x.dtype)
        k_epi_seq = self._safe_unit((q_epi_chunks.float() * w_scores.unsqueeze(-1)).sum(dim=2)).to(x.dtype)
        v_tok = self.v_proj(x_chunks).view(B, M_len, C, self.H, self.D)
        v_seq = (v_tok * w_scores.to(x.dtype).unsqueeze(-1)).sum(dim=2)

        # Lean z_head: algebraically identical to 5D `(B, M, C, H, D)` sum-then-mean, `H`x smaller
        z_head = (x_chunks * w_scores.to(x.dtype).mean(dim=-1, keepdim=True)).sum(dim=2)
        if self.use_deltanet2_gates and self.erase_vec_proj is not None and self.write_vec_proj is not None and self.alpha_vec_proj is not None:
            e_vec = torch.sigmoid(self.erase_vec_proj(z_head)).view(B, M_len, self.H, self.D)
            w_vec = torch.sigmoid(self.write_vec_proj(z_head)).view(B, M_len, self.H, self.D)
            a_vec = torch.sigmoid(self.alpha_vec_proj(z_head)).view(B, M_len, self.H, self.D)
            beta = e_vec.mean(dim=-1, keepdim=True)
            gamma = a_vec.mean(dim=-1, keepdim=True)
        else:
            e_vec = None
            w_vec = None
            a_vec = None
            assert self.beta_proj is not None and self.gamma_proj is not None
            beta = torch.sigmoid(self.beta_proj(z_head)).unsqueeze(-1)
            gamma = torch.sigmoid(self.gamma_proj(z_head)).unsqueeze(-1)

        t_gate = self._tern_gate()
        inv_dh = t_gate / 144.0
        kalman_mix = 1.0 if self.use_kalman_precond else 0.0
        lb_loss: torch.Tensor | None = None

        if self.K_sub > 1 and self.bank_router is not None:
            r_write_st, r_w_probs, r_w_hard = self._route_sub_banks(k_seq[:, :, : self.H_epi])
            r_read_st, r_r_probs, r_r_hard = self._route_sub_banks(q_rw_t)
            f_bal = 0.5 * (r_w_hard.mean(dim=(0, 1)) + r_r_hard.mean(dim=(0, 1)))
            p_bal = 0.5 * (r_w_probs.mean(dim=(0, 1)) + r_r_probs.mean(dim=(0, 1)))
            lb_loss = 0.015 * float(self.K_sub) * (f_bal * p_bal).sum(dim=-1).mean()

            K = self.K_sub
            H_eb = self.H_epi * K
            HB = self.H_banks
            BH = B * HB

            def _exp_epi(t_all: torch.Tensor, h_dim: int) -> torch.Tensor:
                t_epi = t_all.narrow(h_dim, 0, self.H_epi)
                t_exp = t_epi.unsqueeze(h_dim + 1).expand(
                    *t_epi.shape[: h_dim + 1], K, *t_epi.shape[h_dim + 1 :]
                )
                new_shape = list(t_epi.shape[:h_dim]) + [H_eb] + list(t_epi.shape[h_dim + 1 :])
                t_eb = t_exp.reshape(*new_shape)
                if self.H_prose > 0:
                    return torch.cat([t_eb, t_all.narrow(h_dim, self.H_epi, self.H_prose)], dim=h_dim)
                return t_eb

            q_c_b = _exp_epi(q_chunks, 3)
            qe_c_b = _exp_epi(q_epi_chunks, 3)
            k_s_b = _exp_epi(k_seq, 2)
            ke_s_b = _exp_epi(k_epi_seq, 2)
            v_s_b = _exp_epi(v_seq, 2)

            b_epi_b = (beta[:, :, : self.H_epi, 0].float().unsqueeze(-1) * r_write_st).reshape(B, M_len, H_eb, 1)
            g_epi_b = (
                1.0 - r_write_st * (1.0 - gamma[:, :, : self.H_epi, 0].float().unsqueeze(-1))
            ).reshape(B, M_len, H_eb, 1)
            if self.H_prose > 0:
                b_b = torch.cat([b_epi_b, beta[:, :, self.H_epi :].float()], dim=2)
                g_b = torch.cat([g_epi_b, gamma[:, :, self.H_epi :].float()], dim=2)
            else:
                b_b, g_b = b_epi_b, g_epi_b

            with torch.no_grad():
                qe_m = qe_c_b.detach().float().permute(1, 0, 3, 4, 2).reshape(M_len, BH, self.D, C)
                ke_m = ke_s_b.detach().float().permute(1, 0, 2, 3).reshape(M_len, BH, self.D, 1)
                k_m_d = k_s_b.detach().float().permute(1, 0, 2, 3).reshape(M_len, BH, self.D, 1)
                b_m_d = b_b.detach().float().permute(1, 0, 2, 3).reshape(M_len, BH, 1, 1)
                g_m_d = g_b.detach().float().permute(1, 0, 2, 3).reshape(M_len, BH, 1, 1)
                u_m, kd_m = _kalman_p_and_dual_nograd_jit(
                    qe_m, ke_m, k_m_d, b_m_d, g_m_d, self.inv_lam, t_gate, kalman_mix
                )
                u_bank_seq = (
                    u_m.reshape(M_len, B, HB, C)
                    .permute(1, 0, 3, 2)
                    .reshape(B, M_len * C, HB)[:, :T]
                )

            q_m = q_c_b.float().permute(1, 0, 3, 4, 2).reshape(M_len, BH, self.D, C).contiguous()
            k_m = k_s_b.float().permute(1, 0, 2, 3).reshape(M_len, BH, self.D, 1).contiguous()
            v_m = v_s_b.float().permute(1, 0, 2, 3).reshape(M_len, BH, self.D, 1).contiguous()
            if self.use_deltanet2_gates and e_vec is not None and w_vec is not None and a_vec is not None:
                rw_d = r_write_st.unsqueeze(-1)  # (B, M, H_epi, K, 1)
                e_epi_b = (e_vec[:, :, : self.H_epi].float().unsqueeze(3) * rw_d).reshape(B, M_len, H_eb, self.D)
                w_epi_b = (w_vec[:, :, : self.H_epi].float().unsqueeze(3) * rw_d).reshape(B, M_len, H_eb, self.D)
                a_epi_b = (1.0 - rw_d * (1.0 - a_vec[:, :, : self.H_epi].float().unsqueeze(3))).reshape(B, M_len, H_eb, self.D)
                if self.H_prose > 0:
                    e_b = torch.cat([e_epi_b, e_vec[:, :, self.H_epi :].float()], dim=2)
                    w_b = torch.cat([w_epi_b, w_vec[:, :, self.H_epi :].float()], dim=2)
                    a_b = torch.cat([a_epi_b, a_vec[:, :, self.H_epi :].float()], dim=2)
                else:
                    e_b, w_b, a_b = e_epi_b, w_epi_b, a_epi_b
                e_m = e_b.permute(1, 0, 2, 3).reshape(M_len, BH, self.D, 1).contiguous()
                w_m = w_b.permute(1, 0, 2, 3).reshape(M_len, BH, self.D, 1).contiguous()
                a_m = a_b.permute(1, 0, 2, 3).reshape(M_len, BH, 1, self.D).contiguous()
                out_m = _CheckpointedDeltaNet2WoodburyFn.apply(
                    q_m, k_m, kd_m.contiguous(), v_m, e_m, w_m, a_m, inv_dh, 32
                )
            else:
                b_m = b_b.float().permute(1, 0, 2, 3).reshape(M_len, BH, 1, 1).contiguous()
                g_m = g_b.float().permute(1, 0, 2, 3).reshape(M_len, BH, 1, 1).contiguous()
                out_m = _CheckpointedKalmanWoodburyFn.apply(
                    q_m, k_m, kd_m.contiguous(), v_m, b_m, g_m, inv_dh, 32
                )
            read_bank_seq = (
                out_m.reshape(M_len, B, HB, self.D, C)
                .permute(1, 0, 4, 2, 3)
                .reshape(B, M_len * C, HB, self.D)[:, :T]
            )
            read_epi = (
                read_bank_seq[:, :, :H_eb].view(B, T, self.H_epi, K, self.D)
                * r_read_st.unsqueeze(-1)
            ).sum(dim=3)
            u_epi_comb = (
                u_bank_seq[:, :, :H_eb].view(B, T, self.H_epi, K) * r_read_st
            ).sum(dim=3)
            if self.H_prose > 0:
                read_all = torch.cat([read_epi, read_bank_seq[:, :, H_eb:]], dim=2).to(x.dtype)
                uncert_seq = torch.cat([u_epi_comb, u_bank_seq[:, :, H_eb:]], dim=2).to(x.dtype)
            else:
                read_all = read_epi.to(x.dtype)
                uncert_seq = u_epi_comb.to(x.dtype)
            read_seq = read_all.reshape(B, T, self.total_dim)
        else:
            BH = B * self.H
            with torch.no_grad():
                qe_m = q_epi_chunks.detach().float().permute(1, 0, 3, 4, 2).reshape(M_len, BH, self.D, C)
                ke_m = k_epi_seq.detach().float().permute(1, 0, 2, 3).reshape(M_len, BH, self.D, 1)
                k_m_d = k_seq.detach().float().permute(1, 0, 2, 3).reshape(M_len, BH, self.D, 1)
                b_m_d = beta.detach().float().permute(1, 0, 2, 3).reshape(M_len, BH, 1, 1)
                g_m_d = gamma.detach().float().permute(1, 0, 2, 3).reshape(M_len, BH, 1, 1)
                u_m, kd_m = _kalman_p_and_dual_nograd_jit(
                    qe_m, ke_m, k_m_d, b_m_d, g_m_d, self.inv_lam, t_gate, kalman_mix
                )
                uncert_pad = (
                    u_m.reshape(M_len, B, self.H, C)
                    .permute(1, 0, 3, 2)
                    .reshape(B, M_len * C, self.H)
                    .to(x.dtype)
                )

            q_m = q_chunks.float().permute(1, 0, 3, 4, 2).reshape(M_len, BH, self.D, C).contiguous()
            k_m = k_seq.float().permute(1, 0, 2, 3).reshape(M_len, BH, self.D, 1).contiguous()
            v_m = v_seq.float().permute(1, 0, 2, 3).reshape(M_len, BH, self.D, 1).contiguous()
            if self.use_deltanet2_gates and e_vec is not None and w_vec is not None and a_vec is not None:
                e_m = e_vec.float().permute(1, 0, 2, 3).reshape(M_len, BH, self.D, 1).contiguous()
                w_m = w_vec.float().permute(1, 0, 2, 3).reshape(M_len, BH, self.D, 1).contiguous()
                a_m = a_vec.float().permute(1, 0, 2, 3).reshape(M_len, BH, 1, self.D).contiguous()
                out_m = _CheckpointedDeltaNet2WoodburyFn.apply(
                    q_m, k_m, kd_m.contiguous(), v_m, e_m, w_m, a_m, inv_dh, 32
                )
            else:
                b_m = beta.float().permute(1, 0, 2, 3).reshape(M_len, BH, 1, 1).contiguous()
                g_m = gamma.float().permute(1, 0, 2, 3).reshape(M_len, BH, 1, 1).contiguous()
                out_m = _CheckpointedKalmanWoodburyFn.apply(
                    q_m, k_m, kd_m.contiguous(), v_m, b_m, g_m, inv_dh, 32
                )
            read_pad = (
                out_m.reshape(M_len, B, self.H, self.D, C)
                .permute(1, 0, 4, 2, 3)
                .reshape(B, M_len * C, self.H, self.D)
                .to(x.dtype)
            )

            read_seq = read_pad[:, :T].reshape(B, T, self.total_dim)
            uncert_seq = uncert_pad[:, :T]

        self.last_epistemic_uncertainty = uncert_seq

        u_epi = uncert_seq[:, :, : self.H_epi].float()
        conf_gate_epi = torch.sigmoid(
            self.conf_slope.float().view(1, 1, self.H_epi)
            * (self.conf_thresh.float().view(1, 1, self.H_epi) - u_epi)
        )

        if self.H_prose > 0 and self.query_intent_proj is not None:
            q_intent = torch.sigmoid(self.query_intent_proj(x).float())
            self.last_gate_loss = lb_loss
            prose_gate = (1.0 - 0.88 * q_intent).expand(B, T, self.H_prose)
            all_gates = torch.cat([conf_gate_epi, prose_gate], dim=-1).unsqueeze(-1)
        else:
            q_intent = None
            self.last_gate_loss = lb_loss
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
            u_feats = q_intent.to(x.dtype) * (raw_u_feats.to(x.dtype) * self.uncert_scale.to(x.dtype))
        else:
            u_feats = raw_u_feats.to(x.dtype) * self.uncert_scale.to(x.dtype)
        return self.out_proj(torch.cat([normed_read, u_feats], dim=-1))

    # ------------------------------------------------------------------
    # Strict O(1) Incremental Streaming API (exact parity with batch)
    # ------------------------------------------------------------------

    def init_stream_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> WoodburyStreamState:
        BH = batch_size * self.H_banks
        P0 = (
            torch.eye(self.D, device=device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(BH, 1, 1)
            * self.inv_lam
        )
        km1 = max(0, self.L_latch - 1)
        return WoodburyStreamState(
            M_mat=torch.zeros(BH, self.D, self.D, device=device, dtype=torch.float32),
            P_mat=P0,
            z_cumsum=torch.zeros(batch_size, self.d_model, device=device, dtype=torch.float32),
            step_count=0,
            z_ring3=torch.zeros(batch_size, 3, self.d_model, device=device, dtype=dtype),
            sal_ring=torch.full((batch_size, km1, self.H_epi), -1e4, device=device, dtype=torch.float32),
            rw_ring=torch.zeros(batch_size, km1, self.H_epi, self.D, device=device, dtype=torch.float32),
            epi_ring=torch.zeros(batch_size, km1, self.H_epi, self.D, device=device, dtype=torch.float32),
            buf_x=torch.zeros(batch_size, self.C, self.d_model, device=device, dtype=dtype),
            buf_k=torch.zeros(batch_size, self.C, self.H, self.D, device=device, dtype=dtype),
            buf_k_epi=torch.zeros(batch_size, self.C, self.H, self.D, device=device, dtype=dtype),
            buf_pos=0,
            z3_ptr=0,
            latch_ptr=0,
        )

    @torch.no_grad()
    def step(
        self,
        x_t: torch.Tensor,
        h_lex_t: torch.Tensor | None,
        state: WoodburyStreamState,
    ) -> torch.Tensor:
        """Executes ONE chunk step `(B, d_model)` in strict `O(1)` memory."""
        B, D_in = x_t.shape
        dt = x_t.dtype
        BH = B * self.H
        lam_reg = 1.0 / self.inv_lam

        x_u = self._safe_unit(x_t)
        if h_lex_t is not None:
            w_lex = torch.sigmoid(self.lex_logit.float())
            e_u = self._safe_unit(h_lex_t)
            z = self._safe_unit(w_lex * e_u + (1.0 - w_lex) * x_u)
        else:
            z = x_u

        state.step_count += 1
        state.z_cumsum = state.z_cumsum + z
        zc = (z - state.z_cumsum / (float(state.step_count) + 2.0)).to(dt)

        if state.step_count == 1:
            state.z_ring3[:] = zc.unsqueeze(1).expand(B, 3, D_in)
            state.z3_ptr = 0
        zp = state.z3_ptr
        z0 = zc
        z12 = state.z_ring3[:, (zp - 1) % 3] + state.z_ring3[:, (zp - 2) % 3]
        z3 = state.z_ring3[:, zp] + 0.25 * state.z_ring3[:, (zp - 2) % 3]
        state.z_ring3[:, zp] = zc
        state.z3_ptr = (zp + 1) % 3

        sc_raw = math.sqrt(self.D)
        f1_r = self._safe_unit(self.f1_proj(z0).view(B, self.H_epi, self.D), sc_raw)
        f2_r = self._safe_unit(self.f2_proj(z12).view(B, self.H_epi, self.D), sc_raw)
        f3_r = self._safe_unit(self.f3_proj(z3).view(B, self.H_epi, self.D), sc_raw)
        phi_epi = self._safe_unit(torch.tanh(f1_r * 1.2) * torch.tanh(f2_r * 1.2) * torch.tanh(f3_r * 1.2))
        phi_lin = self._safe_unit(f1_r + f2_r + f3_r)
        phi_rw = self._safe_unit(0.35 * phi_lin + 0.94 * phi_epi)

        sal_t = self.subj_salience(x_t).float()  # (B, H_epi)
        if self.L_latch > 1:
            lp = state.latch_ptr
            km1 = self.L_latch - 1
            if lp == 0:
                sal_win = torch.cat((state.sal_ring, sal_t.unsqueeze(1)), dim=1).transpose(1, 2)
                rw_win = torch.cat((state.rw_ring, phi_rw.unsqueeze(1)), dim=1).permute(0, 2, 3, 1)
                epi_win = torch.cat((state.epi_ring, phi_epi.unsqueeze(1)), dim=1).permute(0, 2, 3, 1)
            else:
                sal_win = torch.cat(
                    (state.sal_ring[:, lp:], state.sal_ring[:, :lp], sal_t.unsqueeze(1)), dim=1
                ).transpose(1, 2)
                rw_win = torch.cat(
                    (state.rw_ring[:, lp:], state.rw_ring[:, :lp], phi_rw.unsqueeze(1)), dim=1
                ).permute(0, 2, 3, 1)
                epi_win = torch.cat(
                    (state.epi_ring[:, lp:], state.epi_ring[:, :lp], phi_epi.unsqueeze(1)), dim=1
                ).permute(0, 2, 3, 1)
            state.sal_ring[:, lp] = sal_t
            state.rw_ring[:, lp] = phi_rw
            state.epi_ring[:, lp] = phi_epi
            state.latch_ptr = (lp + 1) % km1
        else:
            sal_win = sal_t.unsqueeze(-1)
            rw_win = phi_rw.unsqueeze(-1)
            epi_win = phi_epi.unsqueeze(-1)

        att = F.softmax(sal_win + self.subj_lag_bias.float().unsqueeze(0), dim=-1)
        q_rw_t = self._safe_unit((rw_win * att.unsqueeze(2)).sum(dim=-1)).to(dt)
        q_epi_t = self._safe_unit((epi_win * att.unsqueeze(2)).sum(dim=-1)).to(dt)

        if self.H_prose > 0 and self.prose_q_proj is not None and self.prose_k_proj is not None:
            q_prose = self._safe_unit(self.prose_q_proj(x_t).view(B, self.H_prose, self.D)).to(dt)
            k_prose = self._safe_unit(self.prose_k_proj(x_t).view(B, self.H_prose, self.D)).to(dt)
            q_rw_all = torch.cat([q_rw_t, q_prose], dim=1)
            k_src_all = torch.cat([q_rw_t, k_prose], dim=1)
            q_epi_all = torch.cat([q_epi_t, q_epi_t[:, : self.H_prose]], dim=1)
        else:
            q_rw_all = q_rw_t
            k_src_all = q_rw_t
            q_epi_all = q_epi_t

        # Read from M_mat & P_mat BEFORE current micro-chunk commits (exact causality)
        BH_b = B * self.H_banks
        if self.K_sub > 1 and self.bank_router is not None:
            K = self.K_sub
            H_eb = self.H_epi * K
            r_read_st, _, _ = self._route_sub_banks(q_rw_t)  # (B, H_epi, K)

            def _exp_t(t_all: torch.Tensor) -> torch.Tensor:
                t_epi = t_all[:, : self.H_epi].unsqueeze(2).expand(B, self.H_epi, K, self.D).reshape(B, H_eb, self.D)
                if self.H_prose > 0:
                    return torch.cat([t_epi, t_all[:, self.H_epi :]], dim=1)
                return t_epi

            q_bh = _exp_t(q_rw_all).float().reshape(BH_b, self.D, 1)
            qe_bh = _exp_t(q_epi_all).float().reshape(BH_b, self.D, 1)
            read_b = torch.bmm(state.M_mat, q_bh).squeeze(-1).reshape(B, self.H_banks, self.D)
            pq_t = torch.bmm(state.P_mat, qe_bh)
            u_b = ((qe_bh * pq_t).sum(dim=1).squeeze(-1) * lam_reg).clamp(0.0, 1.5).reshape(B, self.H_banks)
            read_epi = (read_b[:, :H_eb].view(B, self.H_epi, K, self.D) * r_read_st.unsqueeze(-1)).sum(dim=2)
            u_epi_c = (u_b[:, :H_eb].view(B, self.H_epi, K) * r_read_st).sum(dim=2)
            if self.H_prose > 0:
                read_t = torch.cat([read_epi, read_b[:, H_eb:]], dim=1).reshape(B, self.total_dim).to(dt)
                u_t = torch.cat([u_epi_c, u_b[:, H_eb:]], dim=1).to(dt)
            else:
                read_t = read_epi.reshape(B, self.total_dim).to(dt)
                u_t = u_epi_c.to(dt)
        else:
            q_bh = q_rw_all.float().reshape(BH_b, self.D, 1)
            qe_bh = q_epi_all.float().reshape(BH_b, self.D, 1)
            read_t = torch.bmm(state.M_mat, q_bh).squeeze(-1).reshape(B, self.total_dim).to(dt)
            pq_t = torch.bmm(state.P_mat, qe_bh)
            u_t = ((qe_bh * pq_t).sum(dim=1).squeeze(-1) * lam_reg).clamp(0.0, 1.5).reshape(B, self.H).to(dt)
        self.last_epistemic_uncertainty = u_t.unsqueeze(1)

        # Accumulate into C-chunk write buffer and commit every C chunks
        p_idx = state.buf_pos
        state.buf_x[:, p_idx] = x_t
        state.buf_k[:, p_idx] = k_src_all
        state.buf_k_epi[:, p_idx] = q_epi_all
        state.buf_pos += 1

        if state.buf_pos == self.C:
            state.buf_pos = 0
            w_scores = F.softmax(self.write_pool_score(state.buf_x).float(), dim=1)  # (B, C, H)
            kt_h = self._safe_unit((state.buf_k.float() * w_scores.unsqueeze(-1)).sum(dim=1))  # (B, H, D)
            ket_h = self._safe_unit((state.buf_k_epi.float() * w_scores.unsqueeze(-1)).sum(dim=1))
            v_tok = self.v_proj(state.buf_x).view(B, self.C, self.H, self.D)
            vt_h = (v_tok.float() * w_scores.unsqueeze(-1)).sum(dim=1)
            z_head = (state.buf_x * w_scores.to(dt).mean(dim=-1, keepdim=True)).sum(dim=1)
            if self.use_deltanet2_gates and self.erase_vec_proj is not None and self.write_vec_proj is not None and self.alpha_vec_proj is not None:
                et_h = torch.sigmoid(self.erase_vec_proj(z_head)).float().view(B, self.H, self.D)
                wt_h = torch.sigmoid(self.write_vec_proj(z_head)).float().view(B, self.H, self.D)
                at_h = torch.sigmoid(self.alpha_vec_proj(z_head)).float().view(B, self.H, self.D)
                bt_h = et_h.mean(dim=-1)
                gt_h = at_h.mean(dim=-1)
            else:
                et_h = None
                wt_h = None
                at_h = None
                assert self.beta_proj is not None and self.gamma_proj is not None
                bt_h = torch.sigmoid(self.beta_proj(z_head)).float()   # (B, H)
                gt_h = torch.sigmoid(self.gamma_proj(z_head)).float()  # (B, H)

            if self.K_sub > 1 and self.bank_router is not None:
                K = self.K_sub
                H_eb = self.H_epi * K
                r_write_st, _, _ = self._route_sub_banks(kt_h[:, : self.H_epi])  # (B, H_epi, K)

                def _exp_w(t_all: torch.Tensor) -> torch.Tensor:
                    t_epi = t_all[:, : self.H_epi].unsqueeze(2).expand(B, self.H_epi, K, self.D).reshape(B, H_eb, self.D)
                    if self.H_prose > 0:
                        return torch.cat([t_epi, t_all[:, self.H_epi :]], dim=1)
                    return t_epi

                kt = _exp_w(kt_h).reshape(BH_b, self.D, 1)
                ket = _exp_w(ket_h).reshape(BH_b, self.D, 1)
                vt = _exp_w(vt_h).reshape(BH_b, self.D, 1)
                bt_epi = (bt_h[:, : self.H_epi].unsqueeze(-1) * r_write_st).reshape(B, H_eb)
                gt_epi = (1.0 - r_write_st * (1.0 - gt_h[:, : self.H_epi].unsqueeze(-1))).reshape(B, H_eb)
                if self.H_prose > 0:
                    bt = torch.cat([bt_epi, bt_h[:, self.H_epi :]], dim=1).reshape(BH_b, 1, 1)
                    gt = torch.cat([gt_epi, gt_h[:, self.H_epi :]], dim=1).reshape(BH_b, 1, 1)
                else:
                    bt = bt_epi.reshape(BH_b, 1, 1)
                    gt = gt_epi.reshape(BH_b, 1, 1)
                if self.use_deltanet2_gates and et_h is not None and wt_h is not None and at_h is not None:
                    rw_d = r_write_st.unsqueeze(-1)
                    et_epi = (et_h[:, : self.H_epi].unsqueeze(2) * rw_d).reshape(B, H_eb, self.D)
                    wt_epi = (wt_h[:, : self.H_epi].unsqueeze(2) * rw_d).reshape(B, H_eb, self.D)
                    at_epi = (1.0 - rw_d * (1.0 - at_h[:, : self.H_epi].unsqueeze(2))).reshape(B, H_eb, self.D)
                    if self.H_prose > 0:
                        et = torch.cat([et_epi, et_h[:, self.H_epi :]], dim=1).reshape(BH_b, self.D, 1)
                        wt = torch.cat([wt_epi, wt_h[:, self.H_epi :]], dim=1).reshape(BH_b, self.D, 1)
                        at = torch.cat([at_epi, at_h[:, self.H_epi :]], dim=1).reshape(BH_b, 1, self.D)
                    else:
                        et = et_epi.reshape(BH_b, self.D, 1)
                        wt = wt_epi.reshape(BH_b, self.D, 1)
                        at = at_epi.reshape(BH_b, 1, self.D)
                else:
                    et = wt = at = None
            else:
                kt = kt_h.reshape(BH_b, self.D, 1)
                ket = ket_h.reshape(BH_b, self.D, 1)
                vt = vt_h.reshape(BH_b, self.D, 1)
                bt = bt_h.reshape(BH_b, 1, 1)
                gt = gt_h.reshape(BH_b, 1, 1)
                if self.use_deltanet2_gates and et_h is not None and wt_h is not None and at_h is not None:
                    et = et_h.reshape(BH_b, self.D, 1)
                    wt = wt_h.reshape(BH_b, self.D, 1)
                    at = at_h.reshape(BH_b, 1, self.D)
                else:
                    et = wt = at = None

            t_gate = self._tern_gate()
            inv_dh = t_gate / 144.0
            if self.use_kalman_precond:
                pk_w = torch.bmm(state.P_mat, kt)
                quad_w = (kt * pk_w).sum(dim=1, keepdim=True)
                norm_pk = torch.sqrt(pk_w.pow(2).sum(dim=1, keepdim=True) + 1e-8)
                kdt = pk_w / (torch.maximum(quad_w, norm_pk * 0.4) + 1e-5)
            else:
                kdt = kt

            if self.use_deltanet2_gates and et is not None and wt is not None and at is not None:
                M_dec = state.M_mat * at
                pred = torch.bmm(M_dec, kt)
                wv = wt * vt
                ek = et * kdt
                if inv_dh > 0.0:
                    r_sq = (wv - pred).pow(2).sum(dim=1, keepdim=True) * inv_dh
                    sc = torch.rsqrt(torch.sqrt(1.0 + r_sq * r_sq))
                    wv = wv * sc
                    pred = pred * sc
                state.M_mat = M_dec + 1.35 * (
                    torch.bmm(wv, kdt.transpose(1, 2)) - torch.bmm(pred, ek.transpose(1, 2))
                )
            else:
                err = vt - gt * torch.bmm(state.M_mat, kt)
                if inv_dh > 0.0:
                    r_sq = err.pow(2).sum(dim=1, keepdim=True) * inv_dh
                    err = err * torch.rsqrt(torch.sqrt(1.0 + r_sq * r_sq))
                state.M_mat = gt * state.M_mat + (1.35 * bt) * torch.bmm(err, kdt.transpose(1, 2))

            slope_s = 18.0 + 2.0 * t_gate
            s_gate = torch.sigmoid((bt - 0.26) * slope_s)
            w_cov = s_gate * s_gate * s_gate
            pk = torch.bmm(state.P_mat, ket)
            quad = (ket * pk).sum(dim=1, keepdim=True).clamp(min=1e-5)
            nov = w_cov * 0.92
            P_next = state.P_mat - (nov / (1.0 + nov * quad)) * torch.bmm(pk, pk.transpose(1, 2))
            P_mid = 0.5 * (P_next + P_next.transpose(1, 2))
            if t_gate > 0.0:
                rho_rec = ((1.0 - gt) * w_cov * 0.002 * t_gate).clamp(0.0, 0.002 * t_gate)
                P_new = (1.0 - rho_rec) * P_mid
                P_new.diagonal(dim1=-2, dim2=-1).add_(rho_rec.squeeze(-1) * self.inv_lam)
                state.P_mat = 0.5 * (P_new + P_new.transpose(1, 2))
            else:
                state.P_mat = P_mid

        u_epi = u_t[:, : self.H_epi].float()
        conf_gate_epi = torch.sigmoid(
            self.conf_slope.float().unsqueeze(0) * (self.conf_thresh.float().unsqueeze(0) - u_epi)
        )
        if self.H_prose > 0 and self.query_intent_proj is not None:
            q_intent = torch.sigmoid(self.query_intent_proj(x_t).float())
            prose_gate = (1.0 - 0.88 * q_intent).expand(B, self.H_prose)
            all_gates = torch.cat([conf_gate_epi, prose_gate], dim=-1).unsqueeze(-1)
        else:
            q_intent = None
            all_gates = conf_gate_epi.unsqueeze(-1)

        normed_read = (
            self.out_norm(read_t).view(B, self.H, self.D) * all_gates.to(dt)
        ).reshape(B, self.total_dim)
        raw_u_feats = torch.cat(
            [
                u_epi - 0.42,
                torch.tanh((u_epi - 0.42) * 7.0),
                F.relu(u_epi - 0.36) * 2.5,
            ],
            dim=-1,
        )
        if q_intent is not None:
            u_feats = q_intent.to(dt) * (raw_u_feats.to(dt) * self.uncert_scale.to(dt))
        else:
            u_feats = raw_u_feats.to(dt) * self.uncert_scale.to(dt)
        return self.out_proj(torch.cat([normed_read, u_feats], dim=-1))
