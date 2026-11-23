"""
Calcite v2 Voronoi-Birkhoff Operator Crystal (`calcite_v2/crystal.py`).

Clean, zero-monkey-patch wrapper over the proven `VoronoiCrystalLayer` geometry with:
1. Direct native v3 CUDA dispatch (`fast_crystal_forward`) when `use_fast_kernels=True` and on CUDA,
   and AMP `Float32` shielded fallback on CPU (no global module patching or `__class__` reassignment).
2. Configurable `W_chunks` (`V1_lean_pruned = 2`) initialized cleanly in `__init__`.
3. Exact incremental streaming API (`init_stream_state`, `step`) in strict `O(1)` memory
   matching `VoronoiCrystalLayer.forward()` bit-for-bit (`< 1e-5`).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels.fast_calcite_scans import fast_crystal_forward
from simvq_split.crystal import (
    VoronoiCrystalLayer,
    _sphere_norm_jit,
    log_sinkhorn_6,
    project_birkhoff_to_perm_jit,
)


@dataclass
class CrystalStreamState:
    s_bh: torch.Tensor         # (B*H, G_h, d_s) fp32
    z_cumsum: torch.Tensor     # (B, d_model) fp32
    step_count: int
    # Refractory gate lags (2 micro-chunks)
    p_f_m1: torch.Tensor       # (B, H) fp32
    p_f_m2: torch.Tensor       # (B, H) fp32
    # Pointer window ring (`W_chunks * C` chunks)
    zc_win: torch.Tensor       # (B, W_len, d_model)
    u_win: torch.Tensor        # (B, W_len, H, d_slot) fp32
    win_valid: torch.Tensor    # (B, W_len) bool
    # Write lag window (4 previous chunks for 5-step window)
    sal_w_ring: torch.Tensor   # (B, 4, H) fp32
    u_w_ring: torch.Tensor     # (B, 4, H, d_slot) fp32
    # Read lag window (3 previous chunks for 4-step window)
    sal_r_ring: torch.Tensor   # (B, 3, H) fp32
    u_r_ring: torch.Tensor     # (B, 3, H, d_slot) fp32
    # Op context ring (3 previous micro-chunk means)
    z_mean_ring: torch.Tensor  # (B, 3, d_model)
    # Current micro-chunk accumulator (`C = 4`)
    buf_x: torch.Tensor        # (B, C, d_model)
    buf_zc: torch.Tensor       # (B, C, d_model)
    buf_u: torch.Tensor        # (B, C, H, d_slot) fp32
    buf_kw: torch.Tensor       # (B, C, H, d_slot) fp32
    buf_pos: int

    def memory_bytes(self) -> int:
        b = max(1, int(self.z_cumsum.size(0)))
        total = 0
        for t in (
            self.s_bh,
            self.z_cumsum,
            self.p_f_m1,
            self.p_f_m2,
            self.zc_win,
            self.u_win,
            self.win_valid,
            self.sal_w_ring,
            self.u_w_ring,
            self.sal_r_ring,
            self.u_r_ring,
            self.z_mean_ring,
            self.buf_x,
            self.buf_zc,
            self.buf_u,
            self.buf_kw,
        ):
            total += int(t.numel() * t.element_size())
        return total // b


class CalciteV2CrystalLayer(VoronoiCrystalLayer):
    def __init__(
        self,
        d_model: int,
        num_spheres: int = 24,
        d_sphere: int = 32,
        codebook_size: int = 32,
        w_chunks: int = 2,
        use_fast_kernels: bool = True,
    ):
        super().__init__(
            d_model=d_model,
            num_spheres=num_spheres,
            d_sphere=d_sphere,
            codebook_size=codebook_size,
        )
        self.use_fast_kernels = bool(use_fast_kernels)
        w_chunks_i = int(max(1, w_chunks))
        if w_chunks_i != self.W_chunks:
            self.W_chunks = w_chunks_i
            w_len = self.W_chunks * self.C
            p_a = torch.linspace(0.6, -0.6, steps=w_len).view(w_len, 1, 1).expand(w_len, self.H, 1)
            p_b = torch.linspace(-0.6, 0.6, steps=w_len).view(w_len, 1, 1).expand(w_len, self.H, 1)
            self.ptr_pos_bias = nn.Parameter(torch.cat((p_a, p_b), dim=-1).clone())

    def compute_state_bytes(self) -> int:
        cryst_lag = max(0, 2 * self.W_chunks - 1)
        return int(
            self.G * self.d_s * 4
            + (self.d_model + 1 + 2 * self.H) * 4
            + 4 * self.H * 2
            + cryst_lag * (self.d_model + self.H * self.d_s) * 2
            + (3 + 1) * self.d_model * 2
        )

    def forward(
        self, x: torch.Tensor, h_lex: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.use_fast_kernels and x.is_cuda:
            return fast_crystal_forward(self, x, h_lex=h_lex)
        with torch.amp.autocast(x.device.type, enabled=False):
            return super().forward(
                x.float(), h_lex=h_lex.float() if h_lex is not None else None
            ).to(x.dtype)

    # ------------------------------------------------------------------
    # Strict O(1) Incremental Streaming API (exact parity with batch)
    # ------------------------------------------------------------------

    def init_stream_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> CrystalStreamState:
        cb, c_norm_t = self.get_codebook_fp32()
        s0_bh = (
            self._prepare_s0(batch_size, cb.to(device), c_norm_t.to(device))
            .reshape(batch_size * self.H, self.G_h, self.d_s)
            .clone()
        )
        w_len = self.W_chunks * self.C
        return CrystalStreamState(
            s_bh=s0_bh,
            z_cumsum=torch.zeros(batch_size, self.d_model, device=device, dtype=torch.float32),
            step_count=0,
            p_f_m1=torch.zeros(batch_size, self.H, device=device, dtype=torch.float32),
            p_f_m2=torch.zeros(batch_size, self.H, device=device, dtype=torch.float32),
            zc_win=torch.zeros(batch_size, w_len, self.d_model, device=device, dtype=dtype),
            u_win=torch.zeros(batch_size, w_len, self.H, self.d_slot, device=device, dtype=torch.float32),
            win_valid=torch.zeros(batch_size, w_len, device=device, dtype=torch.bool),
            sal_w_ring=torch.full((batch_size, 4, self.H), -1e4, device=device, dtype=torch.float32),
            u_w_ring=torch.zeros(batch_size, 4, self.H, self.d_slot, device=device, dtype=torch.float32),
            sal_r_ring=torch.full((batch_size, 3, self.H), -1e4, device=device, dtype=torch.float32),
            u_r_ring=torch.zeros(batch_size, 3, self.H, self.d_slot, device=device, dtype=torch.float32),
            z_mean_ring=torch.zeros(batch_size, 3, self.d_model, device=device, dtype=dtype),
            buf_x=torch.zeros(batch_size, self.C, self.d_model, device=device, dtype=dtype),
            buf_zc=torch.zeros(batch_size, self.C, self.d_model, device=device, dtype=dtype),
            buf_u=torch.zeros(batch_size, self.C, self.H, self.d_slot, device=device, dtype=torch.float32),
            buf_kw=torch.zeros(batch_size, self.C, self.H, self.d_slot, device=device, dtype=torch.float32),
            buf_pos=0,
        )

    @torch.no_grad()
    def step(
        self,
        x_t: torch.Tensor,
        h_lex_t: torch.Tensor | None,
        state: CrystalStreamState,
    ) -> torch.Tensor:
        """Executes ONE chunk step `(B, d_model)` in strict `O(1)` memory."""
        B, D = x_t.shape
        dt = x_t.dtype
        BH = B * self.H
        scale_r = math.sqrt(float(self.d_s))
        sc_d = math.sqrt(float(D))

        # Exact `_encode_tokens` step
        x_u = self._safe_unit(x_t)
        if h_lex_t is not None:
            w_lex = torch.sigmoid(self.lex_logit.float())
            z = self._safe_unit(w_lex * self._safe_unit(h_lex_t) + (1.0 - w_lex) * x_u)
        else:
            z = x_u

        state.step_count += 1
        state.z_cumsum = state.z_cumsum + z
        zc = self._safe_unit(z - state.z_cumsum / (float(state.step_count) + 2.0), sc_d).to(dt)
        u_t = self._safe_unit(self.slot_proj(zc).view(B, self.H, self.d_slot))
        sal_t = self.subj_sal_proj(zc).float()

        # 5-step write-lag key (`kw_vec_tok`)
        sal_w_win = torch.cat([state.sal_w_ring, sal_t.unsqueeze(1)], dim=1).transpose(1, 2)  # (B, H, 5)
        w_lag_att = F.softmax(sal_w_win + self.write_lag_logits.float().unsqueeze(0), dim=-1)
        u_w_win = torch.cat([state.u_w_ring, u_t.unsqueeze(1)], dim=1).permute(0, 2, 3, 1)  # (B, H, ds, 5)
        kw_vec_t = self._safe_unit(
            (u_w_win * w_lag_att.unsqueeze(2)).sum(dim=-1)
            + 0.30 * self._safe_unit(self.w_ctx_proj(x_t).view(B, self.H, self.d_slot))
        )
        state.sal_w_ring[:, :-1] = state.sal_w_ring[:, 1:].clone()
        state.sal_w_ring[:, -1] = sal_t
        state.u_w_ring[:, :-1] = state.u_w_ring[:, 1:].clone()
        state.u_w_ring[:, -1] = u_t

        # Read from current state `s_bh` BEFORE micro-chunk commit (matches `forward`)
        k_slots_u = self._safe_unit(self.K_slots).repeat_interleave(self.R_read, dim=0)
        sal_win = torch.cat([state.sal_r_ring, sal_t.unsqueeze(1)], dim=1)  # (B, 4, H)
        sal_rep = sal_win.repeat_interleave(self.R_read, dim=-1).transpose(1, 2)  # (B, H_r, 4)
        r_lag_att = F.softmax(sal_rep + self.read_lag_logits.float().unsqueeze(0), dim=-1)

        u_win_r = torch.cat([state.u_r_ring, u_t.unsqueeze(1)], dim=1)  # (B, 4, H, d_slot)
        u_rep = u_win_r.repeat_interleave(self.R_read, dim=2).permute(0, 2, 3, 1)  # (B, H_r, d_slot, 4)

        state.sal_r_ring[:, :-1] = state.sal_r_ring[:, 1:].clone()
        state.sal_r_ring[:, -1] = sal_t
        state.u_r_ring[:, :-1] = state.u_r_ring[:, 1:].clone()
        state.u_r_ring[:, -1] = u_t

        qr_vec = self._safe_unit(
            (u_rep * r_lag_att.unsqueeze(2)).sum(dim=-1)
            + 0.30 * self._safe_unit(self.r_ctx_proj(x_t).view(B, self.H_r, self.d_slot))
        )
        q_logits = torch.einsum("bhd,hgd->bhg", qr_vec, k_slots_u) * float(self.slot_temp)
        q_soft = F.softmax(q_logits, dim=-1).to(dt)

        s_h = state.s_bh.view(B, self.H, self.G_h, self.d_s).to(dt)
        s_hr = s_h.repeat_interleave(self.R_read, dim=1)
        ret = torch.einsum("bhg,bhgd->bhd", q_soft, s_hr).reshape(B, self.H_r * self.d_s)
        out = self.out_proj(self.out_norm(ret))

        # Push chunk into micro-chunk accumulator (`C = 4`)
        p_idx = state.buf_pos
        state.buf_x[:, p_idx] = x_t
        state.buf_zc[:, p_idx] = zc
        state.buf_u[:, p_idx] = u_t
        state.buf_kw[:, p_idx] = kw_vec_t
        state.buf_pos += 1

        if state.buf_pos == self.C:
            state.buf_pos = 0
            if self.W_chunks > 1:
                state.zc_win[:, : -self.C] = state.zc_win[:, self.C :].clone()
                state.u_win[:, : -self.C] = state.u_win[:, self.C :].clone()
                state.win_valid[:, : -self.C] = state.win_valid[:, self.C :].clone()
            state.zc_win[:, -self.C :] = state.buf_zc
            state.u_win[:, -self.C :] = state.buf_u
            state.win_valid[:, -self.C :] = True

            s_temp = float(self.slot_temp)
            o_temp = float(self.op_temp)
            g_tau = float(self.gate_temp)
            eff_alpha = 1.0 if (not self.training and self.snap_alpha > 0.0) else (0.85 * float(self.snap_alpha))
            is_hard = eff_alpha >= 0.9999

            k_slots_h = self._safe_unit(self.K_slots)
            cb, c_norm_t = self.get_codebook_fp32()

            # Write-slot selection & Voronoi codebook projection (exact match to `forward_states`)
            w_logit_tok = self.w_eta_proj(state.buf_x).float()  # (B, C, H)
            w_pool = F.softmax(w_logit_tok * 2.0, dim=1)
            w_logit = (w_logit_tok * w_pool).sum(dim=1)  # (B, H)
            kw_vec = self._safe_unit((state.buf_kw * w_pool.unsqueeze(-1)).sum(dim=1))

            k_logits = torch.einsum("bhd,hgd->bhg", kw_vec, k_slots_h) * s_temp
            k_prob = F.softmax(k_logits, dim=-1)
            w_peak = k_prob / k_prob.max(dim=-1, keepdim=True).values.clamp(min=1e-5)

            v_tok = self.w_val_proj(state.buf_x).float().view(B, self.C, self.H, self.d_s)
            v_pool = F.softmax(self.pool_score(state.buf_x).float(), dim=1)  # (B, C, H)
            v_unit = self._safe_unit((v_tok * v_pool.unsqueeze(-1)).sum(dim=1))  # (B, H, d_s)
            v_logits = torch.matmul(v_unit, c_norm_t) * 9.0
            v_probs = F.softmax(v_logits, dim=-1)
            v_soft = _sphere_norm_jit(torch.matmul(v_probs, cb), scale_r)
            v_hard = cb[v_logits.argmax(dim=-1)]
            if is_hard:
                v_ste = v_hard
            elif eff_alpha > 0.0:
                v_ste = _sphere_norm_jit((1.0 - eff_alpha) * v_soft + eff_alpha * v_hard, scale_r)
            else:
                v_ste = v_soft

            w_eta = torch.sigmoid((w_logit + 0.58) * g_tau - 0.58)
            kw_soft = (w_eta.unsqueeze(-1) * w_peak).unsqueeze(-1)
            k_hard = F.one_hot(k_prob.argmax(dim=-1), num_classes=self.G_h).float()
            w_hard_g = (w_eta >= 0.36).float()
            kw_targ = (w_hard_g.unsqueeze(-1) * k_hard).unsqueeze(-1)
            kw_eff = kw_targ if is_hard else ((1.0 - eff_alpha) * kw_soft + eff_alpha * kw_targ)

            # Pointer swap matrix
            ptr_raw = self.op_ptr_proj(state.zc_win).float().view(
                B, self.W_chunks * self.C, self.H, 2
            ) + self.ptr_pos_bias.float().unsqueeze(0)
            ptr_raw = ptr_raw.masked_fill(~state.win_valid.unsqueeze(-1).unsqueeze(-1), -1e4)
            att_a = F.softmax(ptr_raw[..., 0], dim=1)
            att_b = F.softmax(ptr_raw[..., 1] - 4.5 * att_a, dim=1)
            ua_vec = self._safe_unit((state.u_win * att_a.unsqueeze(-1)).sum(dim=1))
            ub_vec = self._safe_unit((state.u_win * att_b.unsqueeze(-1)).sum(dim=1))
            psi_a = torch.einsum("bhd,hgd->bhg", ua_vec, k_slots_h) * s_temp
            psi_b = torch.einsum("bhd,hgd->bhg", ub_vec, k_slots_h) * s_temp
            ws = 1.0 if ((not self.training and self.snap_alpha > 0.0) or self.qat_active_mix > 0.0) else 0.0
            _, _, T_swap = self._compute_ptr_swap(psi_a.unsqueeze(1), psi_b.unsqueeze(1), well_scale=ws)
            T_swap = T_swap.squeeze(1)

            x_ch_mean = state.buf_x.float().mean(dim=1).to(dt)
            z_op_in = torch.cat([state.z_mean_ring, x_ch_mean.unsqueeze(1)], dim=1).reshape(B, 4 * self.d_model)
            state.z_mean_ring[:, :-1] = state.z_mean_ring[:, 1:].clone()
            state.z_mean_ring[:, -1] = x_ch_mean
            z_op = F.silu(self.op_ctx_proj(z_op_in))

            B_mats = log_sinkhorn_6(self.birkhoff_logits.float() * o_temp)
            m_prob = F.softmax(self.op_mode_proj(z_op).float().view(B, self.H, 1 + self.M_glob) * o_temp, dim=-1)
            P_cand = m_prob[..., 0:1].unsqueeze(-1) * T_swap + torch.einsum("bhk,hkij->bhij", m_prob[..., 1:], B_mats)

            p_logit_raw = self.perm_gate_proj(z_op).float()
            p_logit = -4.5 + F.softplus(p_logit_raw + 4.5)
            p_raw = torch.sigmoid((p_logit + 0.49) * g_tau - 0.49)
            p_prev = torch.maximum(state.p_f_m1, state.p_f_m2)
            p_eta = p_raw * (1.0 - 0.92 * torch.sigmoid((p_prev - 0.35) * 12.0))
            state.p_f_m2 = state.p_f_m1
            state.p_f_m1 = p_eta

            eye = self.eye_Gh.float().view(1, 1, self.G_h, self.G_h)
            P_soft = eye + p_eta.unsqueeze(-1).unsqueeze(-1) * (P_cand - eye)
            if eff_alpha > 0.0:
                P_perm_hard = project_birkhoff_to_perm_jit(P_cand.reshape(-1, self.G_h, self.G_h)).view(
                    B, self.H, self.G_h, self.G_h
                )
                p_hard_g = (p_eta >= 0.38).float()
                P_targ = eye + p_hard_g.unsqueeze(-1).unsqueeze(-1) * (P_perm_hard - eye)
                P_eff = P_targ if is_hard else ((1.0 - eff_alpha) * P_soft + eff_alpha * P_targ)
            else:
                P_eff = P_soft

            kw_bh = kw_eff.reshape(BH, self.G_h, 1)
            vs_bh = v_ste.reshape(BH, 1, self.d_s)
            Pe_bh = P_eff.reshape(BH, self.G_h, self.G_h)
            s_w = (1.0 - kw_bh) * state.s_bh + kw_bh * vs_bh
            s_next = torch.bmm(Pe_bh, s_w)
            state.s_bh = s_next if is_hard else _sphere_norm_jit(s_next, scale_r)

        return out
