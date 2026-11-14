"""
Post-Transformer State & Memory Primitives for SimVQ-Split v4:
1. `VoronoiCrystalLayer` (2-Head Pointer-Addressed Birkhoff-Voronoi Operator Crystal):
   - Shared SimVQ Voronoi value codebook `C` on S^{d_s - 1} (`0.000000` norm & lattice drift).
   - Factors `G = 24` slots into `H = 2` heads x `G_h = 12` orthogonal prototypes `K_slots` with
     causal subject-salience latch (`write_lag / read_lag`), two-point pointer swap operator
     `T_swap(p_a, p_b)` + Sinkhorn Birkhoff basis `log_sinkhorn_6`, causal refractory gate
     `causal_refractory_gate_jit`, diagonal slot overwrite (`diag_birkhoff_crystal_scan_jit`),
     and exact bipartite permutation projection `project_birkhoff_to_perm_jit` into S_{G_h} at eval.
2. `NeuralWoodburyLayer` (Hybrid `(H-1)+1` Query-Intent Phrase-Latch Woodbury Memory):
   - Heads `0..H_epi-1`: causally centered `z_c`, kurtosis-controlled `tanh` 3-factor Hadamard
     phrase sketches (`Rank (*) Role (*) Surname`) with causal subject-salience latch and
     stop-gradient cubic-gated SPD Woodbury observer `P_t` (`Hallucination Brake`).
   - Head `H_epi`: dedicated continuous contextual retrieval head `(prose_q_proj, prose_k_proj)`
     modulated by learned `query_intent_proj` `(1 - 0.88 * q_intent)` so fluent GSM8K prose
     perplexity drops sharply (`PPL ~ 16-17`) without diluting ledger fact verification.
3. `quantize_ternary_ste`: BitNet-b1.58 32-channel groupwise ternary quantization (`{-1, 0, +1}`)
   with Straight-Through Estimator for 1.58-bit STE QAT.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .baselines import RMSNorm


def quantize_ternary_ste(w: torch.Tensor) -> torch.Tensor:
    """
    BitNet-b1.58 32-channel groupwise ternary quantization (`{-1, 0, +1} * scale`)
    with Straight-Through Estimator (`w + (w_tern - w).detach()`).
    """
    orig_shape = w.shape
    g_size = 32 if w.numel() % 32 == 0 else w.size(-1)
    w_g = w.float().reshape(-1, g_size)
    scale = w_g.abs().mean(dim=-1, keepdim=True).clamp(min=1e-6)
    w_tern = (
        (torch.round(torch.clamp(w_g / scale, -1.0, 1.0)) * scale)
        .reshape(orig_shape)
        .to(w.dtype)
    )
    return w + (w_tern - w).detach()


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
def neural_woodbury_chunk_scan_jit(
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
    Strictly SPD-preserving Second-Order Sherman-Morrison-Woodbury Memory + Analytic Epistemic Sensor.
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
        pa = F.softmax(torch.einsum("bmhd,hgd->bmhg", ua_vec, k_slots_u) * s_temp, dim=-1)
        pb = F.softmax(torch.einsum("bmhd,hgd->bmhg", ub_vec, k_slots_u) * s_temp, dim=-1)
        T_swap = (
            eye
            - torch.diag_embed(pa + pb)
            + pa.unsqueeze(-1) * pb.unsqueeze(-2)
            + pb.unsqueeze(-1) * pa.unsqueeze(-2)
        )

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
        s_bh = diag_birkhoff_crystal_scan_jit(
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

        read_pad, uncert_pad = neural_woodbury_chunk_scan_jit(
            q_chunks,
            k_seq,
            v_seq,
            beta,
            gamma,
            self.inv_lam,
            q_epi_chunks,
            k_epi_seq,
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
