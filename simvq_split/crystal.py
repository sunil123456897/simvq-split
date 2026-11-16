"""
Operator Crystal Primitives for `SimVQ-Split` (`Calcite-5`):
- TorchScript spherical & Birkhoff-Sinkhorn permutation operators
- `VoronoiCrystalLayer` (2-Head Pointer-Addressed Birkhoff-Voronoi Operator Crystal)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .baselines import RMSNorm
from .quant_lut import PackedLUT2BitLinear


@torch.jit.script
def _sphere_norm_jit(x: torch.Tensor, scale: float) -> torch.Tensor:
    inv = torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
    return x * (inv * scale)


@torch.jit.script
def project_birkhoff_to_perm_jit(scores: torch.Tensor) -> torch.Tensor:
    """
    Projects a batch of score matrices `scores` `(N, G, G)` onto exact `{0, 1}`
    permutation matrices `P_hard in S_G` via confidence-ordered bipartite greedy matching.
    """
    N, G, _ = scores.shape
    P_hard = torch.zeros_like(scores)
    col_avail = torch.ones(N, G, device=scores.device, dtype=scores.dtype)
    _, row_order = scores.max(dim=-1).values.sort(dim=-1, descending=True)

    for step_g in range(G):
        r = row_order[:, step_g]
        row_s = scores.gather(1, r.view(N, 1, 1).expand(N, 1, G)).squeeze(1)
        masked_s = row_s - (1.0 - col_avail) * 1e6
        c = masked_s.argmax(dim=-1)
        col_one_hot = F.one_hot(c, num_classes=G).to(scores.dtype)
        row_one_hot = F.one_hot(r, num_classes=G).to(scores.dtype)
        P_hard = P_hard + row_one_hot.unsqueeze(-1) * col_one_hot.unsqueeze(1)
        col_avail = col_avail * (1.0 - col_one_hot)

    return P_hard


def log_sinkhorn_6(log_scores: torch.Tensor) -> torch.Tensor:
    y = log_scores
    for _ in range(6):
        y = y - torch.logsumexp(y, dim=-1, keepdim=True)
        y = y - torch.logsumexp(y, dim=-2, keepdim=True)
    return torch.exp(y)


@torch.jit.script
def causal_refractory_gate_jit(p_raw: torch.Tensor) -> torch.Tensor:
    M = p_raw.size(1)
    p_u = p_raw.unbind(1)
    z = torch.zeros_like(p_u[0])
    f_m1 = z
    f_m2 = z
    outs: list[torch.Tensor] = []
    for m in range(M):
        prev = torch.maximum(f_m1, f_m2)
        inh = torch.sigmoid((prev - 0.35) * 12.0)
        pm = p_u[m] * (1.0 - 0.92 * inh)
        outs.append(pm)
        f_m2 = f_m1.detach()
        f_m1 = pm.detach()
    return torch.stack(outs, dim=1)


@torch.jit.script
def diag_birkhoff_crystal_scan_jit(
    s0: torch.Tensor,  # (B*H, G_h, d_s)
    kw_eff_seq: torch.Tensor,  # (B*H, M, G_h, 1)
    kw_vg_seq: torch.Tensor,  # (B*H, M, G_h, 1)
    v_ste_seq: torch.Tensor,  # (B*H, M, d_s)
    P_eff_seq: torch.Tensor,  # (B*H, M, G_h, G_h)
    is_hard: bool,
    scale_r: float,
) -> torch.Tensor:
    """
    Executes multi-head diagonal-overwrite Pointer Birkhoff-Voronoi Crystal recurrence in float32.
    """
    M = v_ste_seq.size(1)
    s = s0.float()
    kwe_u = kw_eff_seq.float().unbind(1)
    kwv_u = kw_vg_seq.float().unbind(1)
    vs_u = v_ste_seq.float().unsqueeze(2).unbind(1)
    Pe_u = P_eff_seq.float().unbind(1)
    s_list: list[torch.Tensor] = []
    for m in range(M):
        s_list.append(s)
        vs = vs_u[m]
        vs_d = vs.detach()
        kwe = kwe_u[m]
        s_w = (1.0 - kwe) * s + kwe * vs_d + kwv_u[m] * (vs - vs_d)
        s_next = torch.bmm(Pe_u[m], s_w)
        if is_hard:
            s = s_next
        else:
            s = _sphere_norm_jit(s_next, scale_r)
    return torch.stack(s_list, dim=1)


@torch.jit.script
def _diag_birkhoff_crystal_block_jit(
    s0: torch.Tensor,
    kw_eff_blk: torch.Tensor,
    kw_vg_blk: torch.Tensor,
    v_ste_blk: torch.Tensor,
    P_eff_blk: torch.Tensor,
    is_hard: bool,
    scale_r: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    blen = v_ste_blk.size(1)
    s = s0.float()
    kwe_u = kw_eff_blk.float().unbind(1)
    kwv_u = kw_vg_blk.float().unbind(1)
    vs_u = v_ste_blk.float().unsqueeze(2).unbind(1)
    Pe_u = P_eff_blk.float().unbind(1)
    s_list: list[torch.Tensor] = []
    for m in range(blen):
        s_list.append(s)
        vs = vs_u[m]
        vs_d = vs.detach()
        kwe = kwe_u[m]
        s_w = (1.0 - kwe) * s + kwe * vs_d + kwv_u[m] * (vs - vs_d)
        s_next = torch.bmm(Pe_u[m], s_w)
        if is_hard:
            s = s_next
        else:
            s = _sphere_norm_jit(s_next, scale_r)
    return torch.stack(s_list, dim=1), s


class _CheckpointedCrystalScanFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        s0: torch.Tensor,
        kw_eff_seq: torch.Tensor,
        kw_vg_seq: torch.Tensor,
        v_ste_seq: torch.Tensor,
        P_eff_seq: torch.Tensor,
        is_hard: bool,
        scale_r: float,
        Bc: int,
    ) -> torch.Tensor:
        BH, G_h, d_s = s0.shape
        M_len = v_ste_seq.size(1)
        n_blocks = (M_len + Bc - 1) // Bc
        s_starts = torch.empty(
            n_blocks, BH, G_h, d_s, device=s0.device, dtype=torch.float32
        )
        s_cur = s0.float()
        out_all = torch.empty(
            BH, M_len, G_h, d_s, device=s0.device, dtype=torch.float32
        )
        for bi in range(n_blocks):
            st = bi * Bc
            en = min(M_len, st + Bc)
            s_starts[bi].copy_(s_cur)
            s_blk, s_cur = _diag_birkhoff_crystal_block_jit(
                s_cur,
                kw_eff_seq[:, st:en],
                kw_vg_seq[:, st:en],
                v_ste_seq[:, st:en],
                P_eff_seq[:, st:en],
                is_hard,
                scale_r,
            )
            out_all[:, st:en].copy_(s_blk)
        ctx.save_for_backward(
            s_starts, kw_eff_seq, kw_vg_seq, v_ste_seq, P_eff_seq
        )
        ctx.is_hard = bool(is_hard)
        ctx.scale_r = float(scale_r)
        ctx.Bc = int(Bc)
        return out_all

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        s_starts, kw_eff_seq, kw_vg_seq, v_ste_seq, P_eff_seq = ctx.saved_tensors
        is_hard = ctx.is_hard
        scale_r = ctx.scale_r
        Bc = ctx.Bc
        M_len = v_ste_seq.size(1)
        n_blocks = s_starts.size(0)

        g_kwe = torch.zeros_like(kw_eff_seq, dtype=torch.float32)
        g_kwv = torch.zeros_like(kw_vg_seq, dtype=torch.float32)
        g_vs = torch.zeros_like(v_ste_seq, dtype=torch.float32)
        g_Pe = torch.zeros_like(P_eff_seq, dtype=torch.float32)
        g_s_carry = torch.zeros_like(s_starts[0], dtype=torch.float32)

        for bi in range(n_blocks - 1, -1, -1):
            st = bi * Bc
            en = min(M_len, st + Bc)
            s0_b = s_starts[bi].detach().requires_grad_(True)
            kwe_b = kw_eff_seq[:, st:en].detach().float().requires_grad_(True)
            kwv_b = kw_vg_seq[:, st:en].detach().float().requires_grad_(True)
            vs_b = v_ste_seq[:, st:en].detach().float().requires_grad_(True)
            Pe_b = P_eff_seq[:, st:en].detach().float().requires_grad_(True)
            go_b = grad_out[:, st:en].float()
            with torch.enable_grad():
                s_blk, s_end = _diag_birkhoff_crystal_block_jit(
                    s0_b, kwe_b, kwv_b, vs_b, Pe_b, is_hard, scale_r
                )
                grads = torch.autograd.grad(
                    (s_blk, s_end),
                    (s0_b, kwe_b, kwv_b, vs_b, Pe_b),
                    (go_b, g_s_carry),
                    allow_unused=True,
                )
            g_s_carry = grads[0] if grads[0] is not None else torch.zeros_like(s0_b)
            if grads[1] is not None:
                g_kwe[:, st:en].copy_(grads[1])
            if grads[2] is not None:
                g_kwv[:, st:en].copy_(grads[2])
            if grads[3] is not None:
                g_vs[:, st:en].copy_(grads[3])
            if grads[4] is not None:
                g_Pe[:, st:en].copy_(grads[4])

        return (
            g_s_carry.to(s_starts.dtype),
            g_kwe.to(kw_eff_seq.dtype),
            g_kwv.to(kw_vg_seq.dtype),
            g_vs.to(v_ste_seq.dtype),
            g_Pe.to(P_eff_seq.dtype),
            None,
            None,
            None,
        )


def diag_birkhoff_crystal_scan(
    s0: torch.Tensor,
    kw_eff_seq: torch.Tensor,
    kw_vg_seq: torch.Tensor,
    v_ste_seq: torch.Tensor,
    P_eff_seq: torch.Tensor,
    is_hard: bool,
    scale_r: float,
    Bc: int = 32,
) -> torch.Tensor:
    """
    Dispatches `VoronoiCrystalLayer` scan to `_CheckpointedCrystalScanFn` (`Bc=32`)
    during training when `M_len > Bc` (cutting autograd saved VRAM by `~4x+`), and
    uses `diag_birkhoff_crystal_scan_jit` directly in `eval()` / short sequences.
    """
    M_len = v_ste_seq.size(1)
    if torch.is_grad_enabled() and M_len > Bc:
        return _CheckpointedCrystalScanFn.apply(
            s0.float(),
            kw_eff_seq.float(),
            kw_vg_seq.float(),
            v_ste_seq.float(),
            P_eff_seq.float(),
            bool(is_hard),
            float(scale_r),
            int(Bc),
        )
    return diag_birkhoff_crystal_scan_jit(
        s0, kw_eff_seq, kw_vg_seq, v_ste_seq, P_eff_seq, is_hard, scale_r
    )


class VoronoiCrystalLayer(nn.Module):
    """
    Stream B of SimVQ-Split v4: 2-Head Pointer-Addressed Birkhoff-Voronoi Operator Crystal.
    """

    def __init__(
        self,
        d_model: int,
        num_spheres: int = 24,
        d_sphere: int = 32,
        codebook_size: int = 32,
        chunk_size: int = 2,
        n_global_ops: int = 6,
        read_mult: int = 2,
    ):
        super().__init__()
        assert codebook_size <= d_sphere
        self.d_model = d_model
        self.G = num_spheres
        self.H = 2 if (num_spheres >= 4 and num_spheres % 2 == 0) else 1
        self.G_h = num_spheres // self.H
        assert self.G_h >= 2
        self.d_s = d_sphere
        self.d_slot = d_sphere
        self.K = codebook_size
        self.C = chunk_size
        self.M_glob = n_global_ops
        self.R_read = read_mult
        self.H_r = self.H * self.R_read
        self.W_chunks = 5
        self.snap_alpha: float = 0.0
        self.slot_temp: float = 9.0
        self.op_temp: float = 2.0
        self.gate_temp: float = 1.0
        self.sharp_weight: float = 0.04
        self.qat_active_mix: float = 0.0
        self.last_commit_loss: torch.Tensor | None = None

        gen = torch.Generator().manual_seed(271828)
        q, _ = torch.linalg.qr(torch.randn(d_sphere, d_sphere, generator=gen))
        c_init = q[:codebook_size] * math.sqrt(d_sphere)
        self.register_buffer("C_frozen", c_init)
        self.register_buffer("eye_Gh", torch.eye(self.G_h))

        k_proto = []
        for _ in range(self.H):
            qk, _ = torch.linalg.qr(
                torch.randn(self.d_slot, self.d_slot, generator=gen)
            )
            k_proto.append(qk[: self.G_h])
        self.K_slots = nn.Parameter(torch.stack(k_proto, dim=0))

        self.W_simvq = nn.Parameter(
            torch.eye(d_sphere) + torch.randn(d_sphere, d_sphere, generator=gen) * 0.002
        )
        s0_idx = torch.arange(num_spheres) % codebook_size
        self.s0_param = nn.Parameter(c_init[s0_idx].unsqueeze(0).clone())

        self.lex_logit = nn.Parameter(torch.tensor(1.50))
        self.slot_proj = nn.Linear(d_model, self.H * self.d_slot, bias=False)
        self.w_ctx_proj = nn.Linear(d_model, self.H * self.d_slot, bias=False)
        self.subj_sal_proj = nn.Linear(d_model, self.H, bias=False)
        nn.init.normal_(self.subj_sal_proj.weight, std=0.04)

        w_lag = torch.tensor([-0.6, 0.3, 1.4, 0.1, -1.2]).view(1, 5)
        self.write_lag_logits = nn.Parameter(w_lag.repeat(self.H, 1))
        r_lag = torch.tensor([-0.8, 0.2, 1.5, -0.6]).view(1, 4)
        self.read_lag_logits = nn.Parameter(r_lag.repeat(self.H_r, 1))

        self.pool_score = nn.Linear(d_model, self.H, bias=False)
        self.w_val_proj = nn.Linear(d_model, self.H * d_sphere, bias=False)
        self.w_eta_proj = nn.Linear(d_model, self.H, bias=True)

        w_len = self.W_chunks * self.C
        self.op_ptr_proj = nn.Linear(d_model, self.H * 2, bias=False)
        p_a_bias = (
            torch.linspace(0.6, -0.6, steps=w_len)
            .view(w_len, 1, 1)
            .expand(w_len, self.H, 1)
        )
        p_b_bias = (
            torch.linspace(-0.6, 0.6, steps=w_len)
            .view(w_len, 1, 1)
            .expand(w_len, self.H, 1)
        )
        self.ptr_pos_bias = nn.Parameter(torch.cat([p_a_bias, p_b_bias], dim=-1).clone())

        d_op = max(32, d_model // 2)
        self.op_ctx_proj = nn.Linear(4 * d_model, d_op, bias=False)
        self.op_mode_proj = nn.Linear(d_op, self.H * (1 + self.M_glob), bias=False)
        self.perm_gate_proj = nn.Linear(d_op, self.H, bias=True)
        b_init = torch.eye(self.G_h).view(1, 1, self.G_h, self.G_h) * 0.8 + torch.randn(
            self.H, self.M_glob, self.G_h, self.G_h, generator=gen
        ) * 0.40
        self.birkhoff_logits = nn.Parameter(b_init)

        self.r_ctx_proj = nn.Linear(d_model, self.H_r * self.d_slot, bias=False)
        self.out_norm = RMSNorm(self.H_r * d_sphere)
        self.out_proj = nn.Linear(self.H_r * d_sphere, d_model, bias=False)

        if d_model >= self.H * self.d_slot:
            q_kr, _ = torch.linalg.qr(torch.randn(d_model, d_model, generator=gen))
            base = q_kr[: self.H * self.d_slot]
            with torch.no_grad():
                self.slot_proj.weight.copy_(base)
                self.w_ctx_proj.weight.copy_(
                    torch.randn_like(base, generator=gen) * 0.05
                )
                self.r_ctx_proj.weight.copy_(
                    torch.randn(self.H_r * self.d_slot, d_model, generator=gen) * 0.05
                )

        nn.init.constant_(self.w_eta_proj.bias, -1.4)
        with torch.no_grad():
            if self.H >= 2:
                # Head 0: S_5 operator-ready; Head 1: static Overwrite ledger
                self.perm_gate_proj.bias.copy_(torch.tensor([-1.35, -3.20]))
            else:
                nn.init.constant_(self.perm_gate_proj.bias, -1.6)

    def _compute_ptr_swap(
        self, psi_a: torch.Tensor, psi_b: torch.Tensor, well_scale: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pa0 = F.softmax(psi_a, dim=-1)
        pb0 = F.softmax(psi_b, dim=-1)
        if well_scale > 0.0:
            p_bias = (
                self.perm_gate_proj.bias_fp16
                if isinstance(self.perm_gate_proj, PackedLUT2BitLinear)
                else self.perm_gate_proj.bias
            )
            w_base = 0.598 + 0.12 * torch.tanh(-0.5 * (p_bias.float() + 1.35))
            w = (well_scale * w_base).view(1, 1, self.H, 1)
            pa = F.softmax(psi_a + (0.22 * w) * pb0, dim=-1)
            pb = F.softmax(psi_b + w * pa0, dim=-1)
        else:
            pa = pa0
            pb = pb0
        eye = self.eye_Gh.float().view(1, 1, 1, self.G_h, self.G_h)
        t_swap = (
            eye
            - torch.diag_embed(pa + pb)
            + pa.unsqueeze(-1) * pb.unsqueeze(-2)
            + pb.unsqueeze(-1) * pa.unsqueeze(-2)
        )
        return pa, pb, t_swap

    def _safe_unit(self, v: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        vf = v.float()
        return vf * (torch.rsqrt(vf.pow(2).sum(dim=-1, keepdim=True) + 1e-5) * scale)

    def get_codebook_fp32(self) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.C_frozen.float() @ self.W_simvq.float()
        unit = F.normalize(raw, dim=-1)
        return unit * math.sqrt(self.d_s), unit.t().contiguous()

    def _prepare_s0(
        self, B: int, cb: torch.Tensor, c_norm_t: torch.Tensor
    ) -> torch.Tensor:
        scale_r = math.sqrt(self.d_s)
        s0_u = F.normalize(self.s0_param.float(), dim=-1)
        s0_base = s0_u * scale_r
        hard = cb[torch.matmul(s0_u, c_norm_t).argmax(dim=-1)]
        q_u = hard / scale_r
        u_ste = F.normalize(s0_u + q_u, dim=-1).detach()
        s0_enc = (
            s0_base
            - 2.0 * (s0_base * u_ste).sum(-1, keepdim=True) * u_ste
            + 2.0 * (s0_base * s0_u.detach()).sum(-1, keepdim=True) * q_u.detach()
        )
        return (s0_enc + (hard - s0_enc).detach()).expand(B, self.G, self.d_s)

    def _encode_tokens(
        self, x: torch.Tensor, h_lex: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        sc_d = math.sqrt(D)
        x_u = self._safe_unit(x)
        if h_lex is not None:
            w_lex = torch.sigmoid(self.lex_logit.float())
            z = self._safe_unit(w_lex * self._safe_unit(h_lex) + (1.0 - w_lex) * x_u)
        else:
            z = x_u * (1.0 + 0.0 * self.lex_logit.float())
        steps = torch.arange(1, T + 1, device=x.device, dtype=torch.float32).view(
            1, T, 1
        )
        zc = self._safe_unit(z - torch.cumsum(z, dim=1) / (steps + 2.0), sc_d).to(
            x.dtype
        )
        u_vec = self._safe_unit(self.slot_proj(zc).view(B, T, self.H, self.d_slot))
        sal_subj = self.subj_sal_proj(zc).float()
        return zc, u_vec, sal_subj

    def forward_states(
        self, x: torch.Tensor, h_lex: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, int, int, int]:
        B, T, D = x.shape
        C, H, Gh, ds = self.C, self.H, self.G_h, self.d_slot
        pad = (C - (T % C)) % C
        T_pad = T + pad
        M_len = T_pad // C

        eff_alpha = (
            1.0
            if (not self.training and self.snap_alpha > 0.0)
            else (0.85 * float(self.snap_alpha))
        )
        is_hard = eff_alpha >= 0.9999
        scale_r = math.sqrt(self.d_s)
        s_temp = float(self.slot_temp)
        o_temp = float(self.op_temp)
        g_tau = float(self.gate_temp)

        zc, u_vec, sal_subj = self._encode_tokens(x, h_lex)
        k_slots_u = self._safe_unit(self.K_slots)

        sal_w_win = F.pad(sal_subj, (0, 0, 4, 0), value=-1e4).unfold(1, 5, 1)
        w_lag_att = F.softmax(
            sal_w_win + self.write_lag_logits.float().view(1, 1, H, 5), dim=-1
        )
        u_w_win = (
            F.pad(u_vec.reshape(B, T, H * ds), (0, 0, 4, 0))
            .unfold(1, 5, 1)
            .view(B, T, H, ds, 5)
        )
        kw_vec_tok = self._safe_unit(
            (u_w_win * w_lag_att.unsqueeze(3)).sum(dim=-1)
            + 0.30 * self._safe_unit(self.w_ctx_proj(x).view(B, T, H, ds))
        )

        x_pad = F.pad(x, (0, 0, 0, pad)) if pad > 0 else x
        zc_pad = F.pad(zc, (0, 0, 0, pad)) if pad > 0 else zc
        u_pad = F.pad(u_vec, (0, 0, 0, 0, 0, pad)) if pad > 0 else u_vec
        kw_pad = F.pad(kw_vec_tok, (0, 0, 0, 0, 0, pad)) if pad > 0 else kw_vec_tok

        valid_1d = torch.zeros(1, T_pad, 1, device=x.device, dtype=torch.bool)
        valid_1d[:, :T] = True
        valid_ch = valid_1d.view(1, M_len, C, 1)

        x_ch = x_pad.view(B, M_len, C, D)
        kw_ch = kw_pad.view(B, M_len, C, H, ds)

        cb, c_norm_t = self.get_codebook_fp32()
        s0_bh = self._prepare_s0(B, cb, c_norm_t).reshape(B * H, Gh, self.d_s)

        w_logit_tok = self.w_eta_proj(x_ch).float()
        w_pool = F.softmax(w_logit_tok.masked_fill(~valid_ch, -1e4) * 2.0, dim=2)
        w_logit = (w_logit_tok * w_pool).sum(dim=2)
        kw_vec = self._safe_unit((kw_ch * w_pool.unsqueeze(-1)).sum(dim=2))

        k_logits = torch.einsum("bmhd,hgd->bmhg", kw_vec, k_slots_u) * s_temp
        k_prob = F.softmax(k_logits, dim=-1)
        w_peak = k_prob / k_prob.detach().max(dim=-1, keepdim=True).values.clamp(
            min=1e-5
        )

        v_tok = self.w_val_proj(x_ch).float().view(B, M_len, C, H, self.d_s)
        v_pool = F.softmax(
            self.pool_score(x_ch).float().masked_fill(~valid_ch, -1e4), dim=2
        )
        v_unit = self._safe_unit((v_tok * v_pool.unsqueeze(-1)).sum(dim=2))
        v_logits = torch.matmul(v_unit, c_norm_t) * 9.0
        v_probs = F.softmax(v_logits, dim=-1)
        v_soft = _sphere_norm_jit(torch.matmul(v_probs, cb), scale_r)
        v_hard = cb[v_logits.argmax(dim=-1)]

        if is_hard:
            v_cur = v_hard
        elif eff_alpha > 0.0:
            v_cur = _sphere_norm_jit(
                (1.0 - eff_alpha) * v_soft + eff_alpha * v_hard, scale_r
            )
        else:
            v_cur = v_soft

        vs_u, vq_u = v_soft / scale_r, v_cur / scale_r
        u_ste = F.normalize(vs_u + vq_u, dim=-1).detach()
        v_enc = (
            v_soft
            - 2.0 * (v_soft * u_ste).sum(-1, keepdim=True) * u_ste
            + 2.0 * (v_soft * vs_u.detach()).sum(-1, keepdim=True) * vq_u.detach()
        )
        v_ste = v_enc + (v_cur - v_enc).detach()

        w_eta = torch.sigmoid((w_logit + 0.58) * g_tau - 0.58)
        kw_soft = (w_eta.unsqueeze(-1) * w_peak).unsqueeze(-1)
        k_hard = F.one_hot(k_prob.argmax(dim=-1), num_classes=Gh).float()
        w_hard_g = (w_eta.detach() >= 0.36).float()
        kw_targ = (w_hard_g.unsqueeze(-1) * k_hard).unsqueeze(-1)

        if is_hard:
            kw_eff = kw_soft + (kw_targ - kw_soft).detach()
            kw_vg = torch.maximum(kw_targ, kw_soft).detach()
        elif eff_alpha > 0.0:
            kw_cur = (1.0 - eff_alpha) * kw_soft + eff_alpha * kw_targ
            kw_eff = kw_soft + (kw_cur - kw_soft).detach()
            kw_vg = torch.maximum(kw_cur, kw_soft).detach()
        else:
            kw_eff = kw_soft
            kw_vg = kw_soft.detach()

        eye = self.eye_Gh.float().view(1, 1, 1, Gh, Gh)
        left_pad = (self.W_chunks - 1) * C
        w_len = self.W_chunks * C
        zc_win = F.pad(zc_pad, (0, 0, left_pad, 0)).unfold(1, w_len, C).transpose(2, 3)
        u_win = (
            F.pad(u_pad.reshape(B, T_pad, H * ds), (0, 0, left_pad, 0))
            .unfold(1, w_len, C)
            .transpose(2, 3)
            .view(B, M_len, w_len, H, ds)
        )
        win_valid = (
            F.pad(valid_1d, (0, 0, left_pad, 0))
            .unfold(1, w_len, C)
            .transpose(2, 3)
            .unsqueeze(-1)
        )

        ptr_raw = self.op_ptr_proj(zc_win).float().view(
            B, M_len, w_len, H, 2
        ) + self.ptr_pos_bias.float().view(1, 1, w_len, H, 2)
        ptr_raw = ptr_raw.masked_fill(~win_valid, -1e4)
        att_a = F.softmax(ptr_raw[..., 0], dim=2)
        att_b = F.softmax(ptr_raw[..., 1] - 4.5 * att_a.detach(), dim=2)

        ua_vec = self._safe_unit((u_win * att_a.unsqueeze(-1)).sum(dim=2))
        ub_vec = self._safe_unit((u_win * att_b.unsqueeze(-1)).sum(dim=2))
        psi_a = torch.einsum("bmhd,hgd->bmhg", ua_vec, k_slots_u) * s_temp
        psi_b = torch.einsum("bmhd,hgd->bmhg", ub_vec, k_slots_u) * s_temp
        _, _, T_swap_raw = self._compute_ptr_swap(psi_a, psi_b, well_scale=0.0)
        if (not self.training and self.snap_alpha > 0.0) or self.qat_active_mix > 0.0:
            _, _, T_swap = self._compute_ptr_swap(psi_a, psi_b, well_scale=1.0)
        else:
            T_swap = T_swap_raw

        z_mean = x_ch.float().mean(dim=2).to(x.dtype)
        z_op_in = (
            F.pad(z_mean, (0, 0, 3, 0))
            .unfold(1, 4, 1)
            .permute(0, 1, 3, 2)
            .reshape(B, M_len, 4 * D)
        )
        z_op = F.silu(self.op_ctx_proj(z_op_in))

        B_mats = log_sinkhorn_6(self.birkhoff_logits.float() * o_temp)
        m_prob = F.softmax(
            self.op_mode_proj(z_op).float().view(B, M_len, H, 1 + self.M_glob) * o_temp,
            dim=-1,
        )
        P_cand = m_prob[..., 0:1].unsqueeze(-1) * T_swap + torch.einsum(
            "bmhk,hkij->bmhij", m_prob[..., 1:], B_mats
        )

        p_logit_raw = self.perm_gate_proj(z_op).float()
        p_logit = -4.5 + F.softplus(p_logit_raw + 4.5)
        p_raw = torch.sigmoid((p_logit + 0.49) * g_tau - 0.49)
        p_eta = causal_refractory_gate_jit(p_raw)

        P_soft = eye + p_eta.unsqueeze(-1).unsqueeze(-1) * (P_cand - eye)
        if eff_alpha > 0.0:
            P_perm_hard = project_birkhoff_to_perm_jit(
                P_cand.detach().reshape(-1, Gh, Gh)
            ).view(B, M_len, H, Gh, Gh)
            p_hard_g = (p_eta.detach() >= 0.38).float()
            P_targ = eye + p_hard_g.unsqueeze(-1).unsqueeze(-1) * (P_perm_hard - eye)
            if is_hard:
                P_eff = P_soft + (P_targ - P_soft).detach()
            else:
                P_cur = (1.0 - eff_alpha) * P_soft + eff_alpha * P_targ
                P_eff = P_soft + (P_cur - P_soft).detach()
        else:
            P_eff = P_soft

        wd = w_eta.detach()
        w_sum = wd.sum(dim=(0, 1)).clamp(min=1e-3)
        p_slot_bar = (k_prob * wd.unsqueeze(-1)).sum(dim=(0, 1)) / w_sum.unsqueeze(-1)
        cap_slot = F.relu(p_slot_bar - max(0.20, 1.35 / float(Gh))).pow(2).sum() * 4.0
        p_cb_bar = (v_probs * wd.unsqueeze(-1)).sum(dim=(0, 1)) / w_sum.unsqueeze(-1)
        cap_cb = F.relu(p_cb_bar - max(0.20, 1.35 / float(self.K))).pow(2).sum() * 4.0
        pd = p_eta.detach()
        sharp_k = (wd * (1.0 - k_prob.max(dim=-1).values)).mean()
        sharp_p = (pd * (1.0 - P_cand.max(dim=-1).values.mean(dim=-1))).mean()
        commit_v = (wd * (1.0 - (v_unit * (v_hard / scale_r).detach()).sum(dim=-1))).mean()
        p_bin_d = (p_eta.detach() >= 0.38).float()
        gate_bimodal = (
            float(self.snap_alpha)
            * 0.04
            * ((w_eta - w_hard_g).pow(2).mean() + (p_eta - p_bin_d).pow(2).mean())
        )
        idle_reg = 0.008 * w_eta.mean() + 0.015 * F.relu(p_eta.mean() - 0.06)
        self.last_commit_loss = (
            commit_v
            + cap_slot
            + cap_cb
            + float(self.sharp_weight) * (sharp_k + sharp_p)
            + gate_bimodal
            + idle_reg
        )

        BH = B * H
        s_bh = diag_birkhoff_crystal_scan(
            s0_bh,
            kw_eff.permute(0, 2, 1, 3, 4).reshape(BH, M_len, Gh, 1),
            kw_vg.permute(0, 2, 1, 3, 4).reshape(BH, M_len, Gh, 1),
            v_ste.permute(0, 2, 1, 3).reshape(BH, M_len, self.d_s),
            P_eff.permute(0, 2, 1, 3, 4).reshape(BH, M_len, Gh, Gh),
            is_hard,
            scale_r,
        )
        s_read = (
            s_bh.view(B, H, M_len, Gh, self.d_s)
            .permute(0, 2, 1, 3, 4)
            .reshape(B, M_len, self.G, self.d_s)
        )
        return s_read.to(x.dtype), T, T_pad, M_len

    def forward(
        self, x: torch.Tensor, h_lex: torch.Tensor | None = None
    ) -> torch.Tensor:
        B = x.size(0)
        s_read, T, T_pad, M_len = self.forward_states(x, h_lex=h_lex)
        _, u_vec, sal_subj = self._encode_tokens(x, h_lex)
        k_slots_u = self._safe_unit(self.K_slots).repeat_interleave(self.R_read, dim=0)

        sal_r_win = F.pad(
            sal_subj.repeat_interleave(self.R_read, dim=-1), (0, 0, 3, 0), value=-1e4
        ).unfold(1, 4, 1)
        r_lag_att = F.softmax(
            sal_r_win + self.read_lag_logits.float().view(1, 1, self.H_r, 4), dim=-1
        )
        u_rep = u_vec.repeat_interleave(self.R_read, dim=2)
        u_r_win = (
            F.pad(u_rep.reshape(B, T, self.H_r * self.d_slot), (0, 0, 3, 0))
            .unfold(1, 4, 1)
            .view(B, T, self.H_r, self.d_slot, 4)
        )
        qr_vec = self._safe_unit(
            (u_r_win * r_lag_att.unsqueeze(3)).sum(dim=-1)
            + 0.30 * self._safe_unit(self.r_ctx_proj(x).view(B, T, self.H_r, self.d_slot))
        )
        q_logits = (
            torch.einsum("bthd,hgd->bthg", qr_vec, k_slots_u) * float(self.slot_temp)
        )
        q_soft = F.softmax(q_logits, dim=-1).to(x.dtype)

        s_h = (
            s_read.view(B, M_len, self.H, self.G_h, self.d_s)
            .unsqueeze(2)
            .expand(B, M_len, self.C, self.H, self.G_h, self.d_s)
            .reshape(B, T_pad, self.H, self.G_h, self.d_s)[:, :T]
        )
        s_hr = s_h.repeat_interleave(self.R_read, dim=2)
        ret = torch.einsum("bthg,bthgd->bthd", q_soft, s_hr).reshape(
            B, T, self.H_r * self.d_s
        )
        return self.out_proj(self.out_norm(ret))
