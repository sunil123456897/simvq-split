"""Tracer #1 for Calcite-5 (SimVQ-CRAM v5) architectural invariants.

Self-contained PyTorch verification suite proving four core invariants on real
CUDA/CPU tensors without host regex/string heuristics, BPTT loops in energy
training, or bare RMSNorm bypasses:
  1) Strict O(1) streaming state memory across L = 64 -> 512 -> 2048 under
     surprise-gated dynamic chunking + Omega-Woodbury 2nd-order memory +
     DeltaBirkhoff crystal.
  2) BPTT-free Besag Pseudolikelihood (CRAM Eq. 5) + convex latent z (m > 4)
     beating unrolled BPTT and JFB on latent CRAM+z operator-graph assembly
     in < 3.0 s wall-clock, coupled to a pure-tensor Spherical Base-10 ALU.
  3) Spherical norm drift < 1e-6 and exact S_5 multiset bijection conservation
     across long distractor streams with positive S_G-equivariant id_well > 0
     (vs id_well = 0) and under 1.58-bit STE QAT.
  4) 100.00% rejection of unseen correlated traps and familiar-token
     contradiction traps via the dual certificate (sigma^2_q <= theta, E <= tau).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def quantize_ternary_ste(w: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """1.58-bit groupwise absmean ternary quantization {-1, 0, +1} with STE."""
    orig_shape = w.shape
    flat = w.reshape(-1)
    pad = (group_size - (flat.numel() % group_size)) % group_size
    if pad > 0:
        flat_padded = F.pad(flat, (0, pad))
    else:
        flat_padded = flat
    groups = flat_padded.view(-1, group_size)
    scale = groups.abs().mean(dim=-1, keepdim=True).clamp(min=1e-6)
    q_int = torch.clamp(torch.round(groups / scale), -1.0, 1.0)
    q = (q_int * scale).reshape(-1)[: flat.numel()].view(orig_shape)
    return w + (q - w).detach()


class BitLinear158(nn.Module):
    """SubLN-paired 1.58-bit STE linear projection (BitNet b1.58 style)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_features, elementwise_affine=False)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.use_qat: bool = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = self.norm(x)
        w = quantize_ternary_ste(self.weight) if self.use_qat else self.weight
        return F.linear(x_n, w, self.bias)


def greedy_birkhoff_project(scores: torch.Tensor) -> torch.Tensor:
    """Project (B, G, G) doubly-stochastic logits into exact S_G permutation matrix."""
    B, G, _ = scores.shape
    work = scores.clone()
    perm = torch.zeros_like(scores)
    neg_inf = -1e9
    batch_idx = torch.arange(B, device=scores.device)
    for _ in range(G):
        flat_idx = work.view(B, G * G).argmax(dim=-1)
        row = flat_idx // G
        col = flat_idx % G
        perm[batch_idx, row, col] = 1.0
        work[batch_idx, row, :] = neg_inf
        work[batch_idx, :, col] = neg_inf
    return perm


def log_sinkhorn(logits: torch.Tensor, iters: int = 6) -> torch.Tensor:
    """Differentiable doubly-stochastic Sinkhorn projection in log space."""
    log_p = logits
    for _ in range(iters):
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)
    return log_p.exp()


@dataclass
class Calcite5State:
    """Strictly O(1) streaming state container across arbitrary sequence length L."""

    ring_buf: torch.Tensor          # (B, W, D)
    ring_ptr: torch.Tensor          # (1,) int64
    crystal_slots: torch.Tensor     # (B, G, D_s) unit spheres on S^{D_s-1}
    mem_M: torch.Tensor             # (B, H, D_m, D_m) fast memory matrix
    mem_P: torch.Tensor             # (B, H, D_m, D_m) Woodbury inverse covariance
    chunk_count: torch.Tensor       # (B,) int64

    def total_bytes(self) -> int:
        tensors = (
            self.ring_buf,
            self.ring_ptr,
            self.crystal_slots,
            self.mem_M,
            self.mem_P,
            self.chunk_count,
        )
        return int(sum(t.numel() * t.element_size() for t in tensors))


class EquivariantDeltaCrystal(nn.Module):
    """Compositional r-step Birkhoff-Delta crystal with S_G-equivariant id_well > 0."""

    def __init__(
        self,
        d_model: int,
        num_slots: int = 5,
        d_sphere: int = 16,
        r_steps: int = 2,
        temp: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.d_sphere = d_sphere
        self.r_steps = r_steps
        self.temp = temp

        self.d_k = 8
        self.in_proj = BitLinear158(d_model, r_steps * 2 * num_slots * self.d_k)
        # Positive self-consistency identity well gamma_id -> softplus(gamma_id) > 0
        self.gamma_id = nn.Parameter(torch.tensor(2.0))
        self.gram_scale = nn.Parameter(torch.tensor(0.25))
        self.gate_proj = BitLinear158(d_model, 1)

    def init_slots(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # Canonical orthonormal basis slots on S^{d_sphere-1} (stored in float32)
        eye = torch.eye(self.num_slots, self.d_sphere, device=device, dtype=torch.float32)
        return eye.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

    def step(
        self,
        slots: torch.Tensor,
        chunk_feat: torch.Tensor,
        chunk_gate: torch.Tensor,
        use_id_well: bool = True,
        hard: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Apply r compositional S_G-equivariant Birkhoff permutations without damping."""
        B, G, _ = slots.shape
        qk = self.in_proj(chunk_feat).view(B, self.r_steps, 2, G, self.d_k)
        p_total = torch.eye(G, device=slots.device, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1).clone()
        eye_g = torch.eye(G, device=slots.device, dtype=torch.float32).unsqueeze(0)

        id_well = F.softplus(self.gamma_id).clamp(min=1e-4)
        id_bias = id_well * eye_g if use_id_well else 0.0 * eye_g

        # Gate decides how strongly chunk acts vs pure identity preservation
        action_strength = torch.sigmoid(self.gate_proj(chunk_feat) + 3.0 * (chunk_gate - 0.5))  # (B, 1)

        cur_slots = slots.float()
        scale_k = 1.0 / math.sqrt(self.d_k)
        for r_idx in range(self.r_steps):
            q_r = qk[:, r_idx, 0]  # (B, G, d_k)
            k_r = qk[:, r_idx, 1]  # (B, G, d_k)
            # Bilinear QK^T + S_G-equivariant Gram relation <s_i, s_j>
            gram_s = torch.bmm(cur_slots, cur_slots.transpose(-1, -2))  # (B, G, G)
            pair_logits = torch.bmm(q_r, k_r.transpose(-1, -2)) * scale_k + self.gram_scale * gram_s

            # Adaptive identity dominance modulated by surprise gate; zero mode damping!
            scaled_logits = (pair_logits * action_strength.unsqueeze(-1) + id_bias) / self.temp
            p_soft = log_sinkhorn(scaled_logits, iters=6)
            if hard:
                p_hard = greedy_birkhoff_project(scaled_logits)
                p_step = p_hard + (p_soft - p_soft.detach())
            else:
                p_step = p_soft
            p_total = torch.bmm(p_step, p_total)
            cur_slots = torch.bmm(p_step, cur_slots)
            # Exact unit-sphere normalization prevents numerical float drift across L=2048+
            cur_slots = F.normalize(cur_slots, dim=-1, eps=1e-12)

        return cur_slots, p_total, float(id_well.item())


class OmegaWoodburyMemory(nn.Module):
    """Sliding/decaying 2nd-order Woodbury-MesaNet memory with analytical variance sigma^2_q."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 2,
        d_head: int = 16,
        ridge_lambda: float = 0.08,
        gamma_max: float = 0.995,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.ridge_lambda = ridge_lambda
        self.gamma_max = gamma_max

        self.q_proj = BitLinear158(d_model, n_heads * d_head)
        self.k_proj = BitLinear158(d_model, n_heads * d_head)
        self.v_proj = BitLinear158(d_model, n_heads * d_head)
        self.beta_proj = BitLinear158(d_model, n_heads)
        self.gamma_proj = BitLinear158(d_model, n_heads)
        self.out_proj = BitLinear158(n_heads * d_head, d_model)
        self.out_rms = nn.RMSNorm(d_model)
        self.brake_scale = nn.Parameter(torch.tensor(4.0))

    def init_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        H, D = self.n_heads, self.d_head
        M = torch.zeros(batch_size, H, D, D, device=device, dtype=torch.float32)
        eye = torch.eye(D, device=device, dtype=torch.float32)
        P = (1.0 / self.ridge_lambda) * eye.view(1, 1, D, D).expand(batch_size, H, D, D).clone()
        return M, P

    def query_only(
        self,
        M: torch.Tensor,
        P: torch.Tensor,
        x_causal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read memory + epistemic variance sigma^2_q = lam * q_epi^T P q_epi before update."""
        B = x_causal.shape[0]
        H, D = self.n_heads, self.d_head
        q = F.normalize(self.q_proj(x_causal).view(B, H, D).float(), dim=-1)
        raw_out = torch.einsum("bhij,bhj->bhi", M, q).reshape(B, H * D)

        # Probe P_t in the exact key metric space (k_proj) while keeping P strictly stop-grad
        q_epi = F.normalize(self.k_proj(x_causal).view(B, H, D).float(), dim=-1)
        Pq = torch.einsum("bhij,bhj->bhi", P.detach(), q_epi)
        sigma2 = (q_epi * Pq).sum(dim=-1).mean(dim=-1) * self.ridge_lambda  # (B,)
        sigma2 = sigma2.clamp(0.0, 1.5)

        # Multiplicative epistemic brake integrated WITH RMSNorm (never bare RMSNorm)
        brake = torch.exp(-F.softplus(self.brake_scale) * (sigma2 - 0.35).clamp(min=0.0)).unsqueeze(-1)
        out = self.out_rms(self.out_proj(raw_out)) * brake
        return out, sigma2

    def update(
        self,
        M: torch.Tensor,
        P: torch.Tensor,
        x_causal: torch.Tensor,
        chunk_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Preconditioned 2nd-order Woodbury update (P strictly stop-grad)."""
        B = x_causal.shape[0]
        H, D = self.n_heads, self.d_head
        k = F.normalize(self.k_proj(x_causal).view(B, H, D).float(), dim=-1)
        v = self.v_proj(x_causal).view(B, H, D).float()

        beta = (1.95 * torch.sigmoid(self.beta_proj(x_causal)) * chunk_gate).clamp(1e-4, 1.95)
        gamma = (self.gamma_max * torch.sigmoid(self.gamma_proj(x_causal) + 4.5)).clamp(0.5, self.gamma_max)

        # Predict error with decaying memory: e_t = v_t - gamma * M_{t-1} k_t
        M_decayed = gamma.unsqueeze(-1).unsqueeze(-1) * M
        pred = torch.einsum("bhij,bhj->bhi", M_decayed, k)
        err = v - pred
        # Huber-type soft clipping (MIRAS/Yaad robustness against outlier tokens)
        err_clipped = err / (1.0 + 0.1 * err.abs())

        # Stop-grad inverse covariance P update + preconditioned Kalman gain kappa
        with torch.no_grad():
            kd = k.detach()
            bd = beta.detach().clamp(min=1e-4)
            gd = gamma.detach().clamp(min=0.5)
            # Soft cubic trust gate prevents junk tokens from destroying precision
            trust = torch.sigmoid((bd - 0.15) * 12.0) ** 3
            Pk = torch.einsum("bhij,bhj->bhi", P, kd)
            denom = (1.0 / (bd * trust).clamp(min=1e-5)) + (kd * Pk).sum(dim=-1)
            kappa = Pk / denom.unsqueeze(-1)
            # Decaying / sliding Woodbury precision with guaranteed SPD convex ridge blend
            P_mid = P - torch.einsum("bhi,bhj->bhij", kappa, Pk)
            P_mid = 0.5 * (P_mid + P_mid.transpose(-1, -2))
            eye = torch.eye(D, device=P.device, dtype=torch.float32).view(1, 1, D, D)
            gd_box = gd.unsqueeze(-1).unsqueeze(-1)
            P_next = gd_box * P_mid + (1.0 - gd_box) * (1.0 / self.ridge_lambda) * eye
            P_next = 0.5 * (P_next + P_next.transpose(-1, -2))
            k_Pk_sg = (kd * Pk).sum(dim=-1)
            Pk_sg = Pk.detach()

        # Preconditioned 2nd-order state write M_t = gamma M_{t-1} + err * kappa_write^T
        # Only P_t is stop-grad; beta remains live so memory write gates train cleanly.
        beta_live = beta * (torch.sigmoid((beta - 0.15) * 12.0) ** 3)
        kappa_write = (beta_live.unsqueeze(-1) * Pk_sg) / (1.0 + beta_live * k_Pk_sg).unsqueeze(-1)
        M_next = M_decayed + torch.einsum("bhi,bhj->bhij", err_clipped, kappa_write)
        return M_next, P_next


class Calcite5StreamingCore(nn.Module):
    """Surprise-gated streaming O(1) cortex + DeltaBirkhoff crystal + Omega-Woodbury."""

    def __init__(
        self,
        vocab_size: int = 128,
        d_model: int = 64,
        window_size: int = 12,
        num_slots: int = 5,
        d_sphere: int = 16,
        r_steps: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.local_conv = nn.Conv1d(d_model, d_model, kernel_size=window_size, groups=1)
        self.local_norm = nn.LayerNorm(d_model)
        self.surprise_head = nn.Linear(d_model + 1, 1)
        self.crystal = EquivariantDeltaCrystal(
            d_model=d_model, num_slots=num_slots, d_sphere=d_sphere, r_steps=r_steps
        )
        self.memory = OmegaWoodburyMemory(d_model=d_model, n_heads=2, d_head=16)
        # S_G-equivariant causal readout from crystal slots into token residual stream
        self.cryst_q = nn.Linear(d_model, d_sphere)
        self.cryst_out = nn.Linear(d_sphere, d_model)
        self.cryst_gate = nn.Linear(d_model, 1)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def init_state(self, batch_size: int, device: torch.device) -> Calcite5State:
        M, P = self.memory.init_state(batch_size, device)
        return Calcite5State(
            ring_buf=torch.zeros(batch_size, self.window_size, self.d_model, device=device),
            ring_ptr=torch.zeros(1, dtype=torch.long, device=device),
            crystal_slots=self.crystal.init_slots(batch_size, device),
            mem_M=M,
            mem_P=P,
            chunk_count=torch.zeros(batch_size, dtype=torch.long, device=device),
        )

    def step_token(
        self,
        tok_id: torch.Tensor,
        state: Calcite5State,
        step_idx: int,
        hard_perm: bool = True,
        use_crystal: bool = True,
    ) -> tuple[torch.Tensor, Calcite5State, torch.Tensor, torch.Tensor]:
        B = tok_id.shape[0]
        x = self.embed(tok_id)  # (B, D)

        # Strictly O(1) shift/ring window update (no variable-length KV cache)
        ring_next = torch.cat([state.ring_buf[:, 1:, :], x.unsqueeze(1)], dim=1)
        local_ctx = F.conv1d(
            ring_next.transpose(1, 2),
            self.local_conv.weight,
            self.local_conv.bias,
        ).squeeze(-1)
        # Causal centering BEFORE nonlinear/sketch read
        h_local = self.local_norm(x + F.silu(local_ctx))
        h_centered = h_local - ring_next.mean(dim=1)

        mem_out, sigma2 = self.memory.query_only(state.mem_M, state.mem_P, h_centered)

        # Surprise-gated H-Net / BLT boundary score from sigma^2_q + local residual
        gate_logits = self.surprise_head(torch.cat([h_centered, sigma2.unsqueeze(-1)], dim=-1))
        chunk_gate = torch.sigmoid(gate_logits)  # (B, 1)
        is_chunk = (chunk_gate.squeeze(-1) > 0.5)

        slots_next, _, _ = self.crystal.step(
            state.crystal_slots, h_local, chunk_gate, use_id_well=True, hard=hard_perm
        )
        M_next, P_next = self.memory.update(state.mem_M, state.mem_P, h_centered, chunk_gate)

        # Hard STE gate at inference (zero norm drift); smooth gate during differentiable training
        mask_b = is_chunk.view(B, 1, 1)
        if hard_perm:
            gate_mix = mask_b.float() + (chunk_gate - chunk_gate.detach()).view(B, 1, 1)
        else:
            gate_mix = chunk_gate.view(B, 1, 1)
        slots_out = F.normalize(
            gate_mix * slots_next + (1.0 - gate_mix) * state.crystal_slots,
            dim=-1,
            eps=1e-12,
        )

        # S_G-invariant content-addressed readout wires crystal slots directly into next-token logits
        if use_crystal:
            q_c = F.normalize(self.cryst_q(h_local), dim=-1, eps=1e-12)
            slot_scores = torch.einsum("bd,bgd->bg", q_c, slots_out) * 4.0
            slot_weights = F.softmax(slot_scores, dim=-1)
            cryst_vec = torch.einsum("bg,bgd->bd", slot_weights, slots_out)
            cryst_contrib = self.cryst_out(cryst_vec) * torch.sigmoid(self.cryst_gate(h_local))
        else:
            cryst_contrib = torch.zeros_like(h_local)

        logits = self.lm_head(h_local + mem_out + cryst_contrib)
        next_state = Calcite5State(
            ring_buf=ring_next,
            ring_ptr=torch.tensor([(step_idx + 1) % self.window_size], device=tok_id.device),
            crystal_slots=slots_out,
            mem_M=M_next,
            mem_P=P_next,
            chunk_count=state.chunk_count + is_chunk.long(),
        )
        return logits, next_state, sigma2, chunk_gate


class CRAMzOperatorGraphEBM(nn.Module):
    """CRAM+z latent-convex energy model for open multi-node operator graph assembly.

    No fixed template dictionary: builds an N-node operator DAG (each node selects
    joint state from C=8 op/parent/role labels) coupled via pairwise factor tensor
    W_pair, z-modulated conditional coupling W_gated(alpha(z)) (CRAM+z Eq. 14),
    and strongly convex latent diffusion field z (m - ||W_zz||_2 >= 4.0).
    """

    def __init__(
        self,
        n_nodes: int = 6,
        n_choices: int = 8,
        d_obs: int = 16,
        d_z: int = 16,
        n_z_gates: int = 8,
        m_convex: float = 5.0,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.n_choices = n_choices
        self.d_z = d_z
        self.n_z_gates = n_z_gates
        self.m_convex = m_convex

        self.unary_net = nn.Sequential(
            nn.Linear(d_obs, 48),
            nn.SiLU(),
            nn.Linear(48, n_choices),
        )
        # Learnable global pairwise coupling W_pair (N, C, N, C)
        self.W_pair_raw = nn.Parameter(0.05 * torch.randn(n_nodes, n_choices, n_nodes, n_choices))
        # Factorized z-gated edge couplings (CRAM+z Eq. 14) for 3-body composition in O(BKNR)
        self.V_gated = nn.Parameter(0.15 * torch.randn(n_z_gates, n_nodes, n_choices, 8))
        self.U_z_gate = nn.Linear(d_z, n_z_gates)
        # Convex latent field z parameters: W_zz restricted to spectral norm <= 1.0 < m_convex=5.0
        # Zero global flattened W_zx shortcut: all cross-node communication passes via W_pair & z!
        self.W_zz_raw = nn.Parameter(0.05 * torch.randn(d_z, d_z))
        self.W_zy = nn.Parameter(0.15 * torch.randn(d_z, n_nodes, n_choices))
        self.tau_energy = nn.Parameter(torch.tensor(0.0))

    def _effective_pair(self) -> torch.Tensor:
        N = self.n_nodes
        W = 0.5 * (self.W_pair_raw + self.W_pair_raw.permute(2, 3, 0, 1))
        eye_n = torch.eye(N, device=W.device).view(N, 1, N, 1)
        return W * (1.0 - eye_n)

    def _effective_Wzz(self) -> torch.Tensor:
        S = 0.5 * (self.W_zz_raw + self.W_zz_raw.t())
        # Gershgorin bound: row L1 norm <= 1.0 => spectral radius <= 1.0 without cuSolver SVD
        row_l1 = S.abs().sum(dim=-1).max().clamp(min=1.0)
        return S / row_l1

    def _spd_A_inv(self, device: torch.device) -> torch.Tensor:
        # Factorized 8-term Neumann product (I+S)(I+S^2)(I+S^4)/m in 3 matmuls, err < 3e-6
        W_zz = self._effective_Wzz()
        s1 = W_zz / self.m_convex
        eye = torch.eye(self.d_z, device=device)
        s2 = s1 @ s1
        s4 = s2 @ s2
        return ((eye + s1) @ (eye + s2) @ (eye + s4)) / self.m_convex

    def solve_convex_z(self, y_soft: torch.Tensor, A_inv: torch.Tensor | None = None) -> torch.Tensor:
        """Exact convex latent fixed point z*(y) = (m I - W_zz)^{-1} (W_zy y)."""
        if A_inv is None:
            A_inv = self._spd_A_inv(y_soft.device)
        rhs = torch.einsum("znc,bnc->bz", self.W_zy, y_soft)
        return rhs @ A_inv.t()

    def node_conditional_energies(
        self,
        y_soft: torch.Tensor,
        x_obs: torch.Tensor,
        unary: torch.Tensor | None = None,
        W_pair: torch.Tensor | None = None,
        A_inv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Zero-leak exact conditional energies E(y_i=c | y_{-i}, z*(y_{-i}); x) + z-gated Eq. 14."""
        if unary is None:
            u_raw = self.unary_net(x_obs)  # (B, N, C), purely node-local observations
            unary = u_raw - u_raw.mean(dim=-1, keepdim=True)
        if W_pair is None:
            W_pair = self._effective_pair()
        pair_msg = torch.einsum("icjd,bjd->bic", W_pair, y_soft)

        if A_inv is None:
            A_inv = self._spd_A_inv(y_soft.device)
        zy_per_node = torch.einsum("zic,bic->biz", self.W_zy, y_soft)  # (B, N, d_z)
        r_all = zy_per_node.sum(dim=1)  # (B, d_z)
        r_minus_i = r_all.unsqueeze(1) - zy_per_node  # (B, N, d_z), zero leak of node i
        z_minus_i = torch.einsum("zw,bnw->bnz", A_inv, r_minus_i)
        z_cross = torch.einsum("bnz,znc->bnc", z_minus_i, self.W_zy)
        z_self = 0.5 * torch.einsum("znc,zw,wnc->nc", self.W_zy, A_inv, self.W_zy)

        # CRAM+z Eq. 14: alpha_k(z_{-i}) modulates high-order neighbor interactions for node i
        alpha_z = F.softmax(self.U_z_gate(z_minus_i), dim=-1)  # (B, N, K)
        v_node = torch.einsum("kncr,bnc->bknr", self.V_gated, y_soft)  # (B, K, N, R)
        s_minus_i = v_node.sum(dim=2, keepdim=True) - v_node  # (B, K, N, R), strictly 0 leak
        gated_per_k = torch.einsum("kicr,bkir->bkic", self.V_gated, s_minus_i)
        gated_msg = torch.einsum("bik,bkic->bic", alpha_z, gated_per_k)

        return -(unary + pair_msg + gated_msg + z_cross + z_self.unsqueeze(0))

    def total_energy(self, y_soft: torch.Tensor, x_obs: torch.Tensor) -> torch.Tensor:
        """Exact scalar CRAM+z energy E_theta(y, z*; x) per batch item."""
        z = self.solve_convex_z(y_soft)
        W_zz = self._effective_Wzz()
        u_raw = self.unary_net(x_obs)
        unary = u_raw - u_raw.mean(dim=-1, keepdim=True)
        W_pair = self._effective_pair()
        alpha_z = F.softmax(self.U_z_gate(z), dim=-1)  # (B, K)
        v_node = torch.einsum("kncr,bnc->bknr", self.V_gated, y_soft)  # (B, K, N, R)
        s_k = v_node.sum(dim=2)  # (B, K, R)

        e_unary = -(y_soft * unary).sum(dim=(1, 2))
        e_pair = -0.5 * torch.einsum("bic,icjd,bjd->b", y_soft, W_pair, y_soft)
        e_gated = -0.5 * torch.einsum(
            "bk,bkr->b", alpha_z, s_k.pow(2) - v_node.pow(2).sum(dim=2)
        )
        e_z_quad = 0.5 * self.m_convex * (z * z).sum(dim=-1) - 0.5 * torch.einsum(
            "bz,zw,bw->b", z, W_zz, z
        )
        e_zy = -torch.einsum("bz,zic,bic->b", z, self.W_zy, y_soft)
        return e_unary + e_pair + e_gated + e_z_quad + e_zy

    def pseudolikelihood_loss(self, y_true_idx: torch.Tensor, x_obs: torch.Tensor) -> torch.Tensor:
        """Besag Local Pseudolikelihood (CRAM Eq. 5) with 0 BPTT + DSM perturbation."""
        y_oh = F.one_hot(y_true_idx, num_classes=self.n_choices).float()
        u_raw = self.unary_net(x_obs)
        unary = u_raw - u_raw.mean(dim=-1, keepdim=True)
        W_pair = self._effective_pair()
        A_inv = self._spd_A_inv(y_oh.device)
        cond_e = self.node_conditional_energies(y_oh, x_obs, unary=unary, W_pair=W_pair, A_inv=A_inv)
        pl_loss = F.cross_entropy((-cond_e).reshape(-1, self.n_choices), y_true_idx.reshape(-1))

        # Denoising Score Matching auxiliary loss on 1-node Hamming-corrupted neighbors (0 BPTT)
        B, N, C = y_oh.shape
        flip_mask = (torch.rand(B, N, device=y_oh.device) < 0.25)
        rand_state = F.one_hot(torch.randint(0, C, (B, N), device=y_oh.device), num_classes=C).float()
        y_noisy = torch.where(flip_mask.unsqueeze(-1), rand_state, y_oh)
        cond_e_noisy = self.node_conditional_energies(y_noisy, x_obs, unary=unary, W_pair=W_pair, A_inv=A_inv)
        dsm_loss = F.cross_entropy((-cond_e_noisy).reshape(-1, C), y_true_idx.reshape(-1))
        l1_reg = 5e-4 * (self.W_pair_raw.abs().mean() + self.V_gated.abs().mean())
        return 0.7 * pl_loss + 0.3 * dsm_loss + l1_reg

    def unrolled_relaxation_loss(
        self,
        y_true_idx: torch.Tensor,
        x_obs: torch.Tensor,
        n_steps: int = 12,
        jfb_1step: bool = False,
        step_lr: float = 0.25,
    ) -> torch.Tensor:
        """Unrolled soft relaxation loss (BPTT or 1-step JFB) on the identical EBM."""
        B, N = y_true_idx.shape
        C = self.n_choices
        u_raw = self.unary_net(x_obs)
        unary = u_raw - u_raw.mean(dim=-1, keepdim=True)
        W_pair = self._effective_pair()
        A_inv = self._spd_A_inv(x_obs.device)
        logits_y = torch.zeros(B, N, C, device=x_obs.device)
        for s in range(n_steps):
            if jfb_1step and s < n_steps - 1:
                with torch.no_grad():
                    y_soft = F.softmax(logits_y, dim=-1)
                    cond_e = self.node_conditional_energies(
                        y_soft, x_obs, unary=unary, W_pair=W_pair, A_inv=A_inv
                    )
                    logits_y = (1.0 - step_lr) * logits_y + step_lr * (-cond_e)
                logits_y = logits_y.detach()
            else:
                y_soft = F.softmax(logits_y, dim=-1)
                cond_e = self.node_conditional_energies(
                    y_soft, x_obs, unary=unary, W_pair=W_pair, A_inv=A_inv
                )
                logits_y = (1.0 - step_lr) * logits_y + step_lr * (-cond_e)
        return F.cross_entropy(logits_y.reshape(-1, C), y_true_idx.reshape(-1))

    @torch.no_grad()
    def focused_gibbs_infer(
        self,
        x_obs: torch.Tensor,
        n_sweeps: int = 10,
        restarts: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Focused Node-wise Gibbs sampler with best-energy restart selection."""
        B = x_obs.shape[0]
        N, C = self.n_nodes, self.n_choices
        x_rep = x_obs.unsqueeze(1).expand(B, restarts, N, -1).reshape(B * restarts, N, -1)

        u_raw = self.unary_net(x_rep)
        unary = u_raw - u_raw.mean(dim=-1, keepdim=True)
        W_pair = self._effective_pair()
        A_inv = self._spd_A_inv(x_obs.device)
        y_idx = unary.argmax(dim=-1)
        if restarts > 1:
            probs = F.softmax(unary, dim=-1)
            sampled = torch.multinomial(probs.reshape(-1, C), 1).reshape(B * restarts, N)
            keep_greedy = torch.zeros(B, restarts, 1, dtype=torch.bool, device=x_obs.device)
            keep_greedy[:, 0, :] = True
            y_idx = torch.where(keep_greedy.reshape(B * restarts, 1), y_idx, sampled)

        y_oh = F.one_hot(y_idx, num_classes=C).float()
        for sw in range(n_sweeps):
            temp = max(0.12, 1.6 * (0.72 ** sw))
            for i in range(N):
                cond_e = self.node_conditional_energies(
                    y_oh, x_rep, unary=unary, W_pair=W_pair, A_inv=A_inv
                )[:, i, :]
                if sw < n_sweeps - 2:
                    probs_i = F.softmax(-cond_e / temp, dim=-1)
                    choice_i = torch.multinomial(probs_i, 1).squeeze(-1)
                else:
                    choice_i = (-cond_e).argmax(dim=-1)
                y_oh[:, i, :] = F.one_hot(choice_i, num_classes=C).float()

        energies = self.total_energy(y_oh, x_rep).view(B, restarts)
        best_r = energies.argmin(dim=-1)
        y_best_oh = y_oh.view(B, restarts, N, C)[torch.arange(B, device=x_obs.device), best_r]
        y_best_idx = y_best_oh.argmax(dim=-1)
        best_e = energies.gather(1, best_r.unsqueeze(1)).squeeze(1)
        return y_best_idx, best_e


class SphericalBase10RegisterALU(nn.Module):
    """Pure-tensor 6-digit fixed-point ALU on (S^14)^6 x S^14_sign (zero regex/host math)."""

    def __init__(self, n_digits: int = 6, d_sphere: int = 15, scale_pow: int = 2) -> None:
        super().__init__()
        self.n_digits = n_digits
        self.d_sphere = d_sphere
        self.scale = 10 ** scale_pow  # 100 -> 2 fractional decimal digits
        # Fixed orthogonal unit-sphere codebook for digits 0..9 in R^{15}
        basis = torch.eye(10, d_sphere, dtype=torch.float32)
        self.register_buffer("digit_codebook", basis)
        sign_basis = torch.eye(2, d_sphere, dtype=torch.float32)  # [+, -]
        self.register_buffer("sign_codebook", sign_basis)

    def encode_scaled(self, scaled_int: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode signed scaled integer tensor (B,) onto (S^14)^G x S^14_sign."""
        sign_idx = (scaled_int < 0).long()
        mag = scaled_int.abs()
        digits = []
        rem = mag.clone()
        for _ in range(self.n_digits):
            d = rem % 10
            digits.append(d)
            rem = rem // 10
        d_tensor = torch.stack(digits, dim=-1)  # (B, G), LSD first
        d_sphere = F.normalize(self.digit_codebook[d_tensor], dim=-1, eps=1e-12)
        s_sphere = F.normalize(self.sign_codebook[sign_idx], dim=-1, eps=1e-12)
        return d_sphere, s_sphere

    def decode_scaled(self, d_sphere: torch.Tensor, s_sphere: torch.Tensor) -> torch.Tensor:
        """Exact Voronoi nearest-codebook readout back to signed scaled int."""
        sim_d = torch.einsum("bgd,kd->bgk", d_sphere, self.digit_codebook)
        digits = sim_d.argmax(dim=-1)  # (B, G)
        powers = (10 ** torch.arange(self.n_digits, device=d_sphere.device, dtype=torch.long)).view(1, -1)
        mag = (digits * powers).sum(dim=-1)
        sim_s = torch.einsum("bd,kd->bk", s_sphere, self.sign_codebook)
        is_neg = sim_s.argmax(dim=-1) == 1
        return torch.where(is_neg, -mag, mag)

    def apply_op(
        self,
        a_d: torch.Tensor,
        a_s: torch.Tensor,
        b_d: torch.Tensor,
        b_s: torch.Tensor,
        op_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute tensor ALU op {0:+, 1:-, 2:*, 3:/} purely via codebook decoding & re-encoding."""
        va = self.decode_scaled(a_d, a_s)
        vb = self.decode_scaled(b_d, b_s)
        out_add = va + vb
        out_sub = va - vb
        out_mul = torch.div(va * vb, self.scale, rounding_mode="trunc")
        safe_vb = torch.where(vb == 0, torch.ones_like(vb), vb)
        # Integer-safe signed division preserving rounding toward zero
        out_div = torch.div(va * self.scale, safe_vb, rounding_mode="trunc")
        out = torch.where(
            op_id == 0,
            out_add,
            torch.where(op_id == 1, out_sub, torch.where(op_id == 2, out_mul, out_div)),
        )
        return self.encode_scaled(out)


def generate_coupled_dag_dataset(
    n_samples: int,
    n_nodes: int = 6,
    n_choices: int = 8,
    d_obs: int = 16,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a 6-node open-graph operator assembly DAG with strict local ambiguity.

    Two independent latent roots a, b in {0..7} yield 64 valid graphs:
      y0=a, y1=(a+1)%8, y2=(a+2)%8, y3=b, y4=(b+1)%8, y5=(a+b)%8.
    Nodes 0 & 3 observe ONLY mod 4 (50% Bayes ceiling per node).
    Nodes 2 & 4 observe ONLY // 4  (25% Bayes ceiling per node).
    Node 1 (1-hop bridge) and Node 5 (3-body root op y5 = (a+b)%8) have ZERO obs.
    """
    assert n_nodes == 6 and n_choices == 8
    gen = torch.Generator(device="cpu").manual_seed(seed)
    embed_gen = torch.Generator(device="cpu").manual_seed(202405)

    a = torch.randint(0, n_choices, (n_samples,), generator=gen)
    b = torch.randint(0, n_choices, (n_samples,), generator=gen)
    y0 = a
    y1 = (a + 1) % n_choices
    y2 = (a + 2) % n_choices
    y3 = b
    y4 = (b + 1) % n_choices
    y5 = (a + b) % n_choices
    y_true = torch.stack([y0, y1, y2, y3, y4, y5], dim=1)

    E_n0 = torch.randn(4, d_obs, generator=embed_gen) / math.sqrt(d_obs)
    E_n2 = torch.randn(2, d_obs, generator=embed_gen) / math.sqrt(d_obs)
    E_n3 = torch.randn(4, d_obs, generator=embed_gen) / math.sqrt(d_obs)
    E_n4 = torch.randn(2, d_obs, generator=embed_gen) / math.sqrt(d_obs)

    x_obs = 0.04 * torch.randn(n_samples, n_nodes, d_obs, generator=gen)
    x_obs[:, 0, :] = x_obs[:, 0, :] + E_n0[y0 % 4]
    # Node 1 stays pure noise (hidden bridge)
    x_obs[:, 2, :] = x_obs[:, 2, :] + E_n2[y2 // 4]
    x_obs[:, 3, :] = x_obs[:, 3, :] + E_n3[y3 % 4]
    x_obs[:, 4, :] = x_obs[:, 4, :] + E_n4[y4 // 4]
    # Node 5 stays pure noise (hidden 3-body root op y5 = (a+b)%8)

    op_vals = torch.randint(1, 25, (n_samples, 2), generator=gen) * 100
    return x_obs.to(device), y_true.to(device), op_vals.to(device)


def run_invariant_1_o1_state(device: torch.device) -> bool:
    print("\n=== [Invariant 1] Strict O(1) State Bytes (L = 64 -> 512 -> 2048) ===")
    torch.manual_seed(101)
    core = Calcite5StreamingCore(vocab_size=64, d_model=64, window_size=12, num_slots=5, d_sphere=16, r_steps=2).to(device)
    core.eval()

    lengths = [64, 256, 512, 2048]
    byte_table: dict[int, int] = {}
    max_norm_err = 0.0
    chunk_ratios: dict[int, float] = {}
    min_eig_all = float("inf")
    max_eig_all = 0.0
    max_p_cap = (1.0 / core.memory.ridge_lambda) + 1e-4

    with torch.no_grad():
        for L in lengths:
            state = core.init_state(batch_size=2, device=device)
            toks = torch.randint(0, 64, (2, L), device=device)
            # Run full streaming rollout and check final byte footprint + Woodbury SPD health
            for t in range(L):
                _, state, _, _ = core.step_token(toks[:, t], state, t, hard_perm=True)
            byte_table[L] = state.total_bytes()
            norm_err = (state.crystal_slots.norm(dim=-1) - 1.0).abs().max().item()
            max_norm_err = max(max_norm_err, norm_err)
            chunk_ratios[L] = float(state.chunk_count.float().mean().item()) / float(L)

            eigvals = torch.linalg.eigvalsh(state.mem_P)
            p_min_eig = float(eigvals.min().item())
            p_max_eig = float(eigvals.max().item())
            sym_err = float((state.mem_P - state.mem_P.transpose(-1, -2)).abs().max().item())
            min_eig_all = min(min_eig_all, p_min_eig)
            max_eig_all = max(max_eig_all, p_max_eig)
            assert p_min_eig > 0.0, f"Woodbury P_t lost SPD at L={L}: min_eig={p_min_eig}"
            assert p_max_eig <= max_p_cap, f"Woodbury P_t exceeded ridge cap at L={L}: max_eig={p_max_eig}"
            assert sym_err < 1e-5, f"Woodbury P_t symmetry drift at L={L}: sym_err={sym_err}"
            print(
                f"  L={L:4d} | state_bytes={byte_table[L]:6d} B ({byte_table[L]/1024:.2f} KB) "
                f"| chunk_rate={chunk_ratios[L]:.3f} | norm_err={norm_err:.2e} "
                f"| P_eig=[{p_min_eig:.3f}, {p_max_eig:.3f}]"
            )

    strict_o1 = len(set(byte_table.values())) == 1
    assert strict_o1, f"State bytes grew with L: {byte_table}"
    assert max_norm_err < 1e-6, f"Norm error exceeded 1e-6: {max_norm_err}"
    print(
        f"  PASS: Exact O(1) state invariant = {byte_table[64]} B across L=64..2048; "
        f"P_t SPD eig=[{min_eig_all:.3f}, {max_eig_all:.3f}]."
    )
    return True


def run_invariant_2_pl_vs_bptt_and_alu(device: torch.device) -> bool:
    print("\n=== [Invariant 2] CRAM+z Besag PL (0 BPTT) vs JFB/BPTT (< 3.0 s) + Tensor ALU ===")
    x_tr, y_tr, op_tr = generate_coupled_dag_dataset(512, n_nodes=6, seed=42, device=device)
    x_te, y_te, op_te = generate_coupled_dag_dataset(256, n_nodes=6, seed=999, device=device)

    # Full CUDA kernel warm-up (PL, unrolled BPTT, 1-step JFB, AdamW, Gibbs) before fair timing
    warm = CRAMzOperatorGraphEBM(n_nodes=6, n_choices=8, d_obs=16, d_z=16, n_z_gates=8, m_convex=5.0).to(device)
    warm_opt = torch.optim.AdamW(warm.parameters(), lr=0.07, weight_decay=1e-4)
    for w_mode in ("pl", "bptt", "jfb"):
        warm_opt.zero_grad()
        if w_mode == "pl":
            w_loss = warm.pseudolikelihood_loss(y_tr[:32], x_tr[:32])
        elif w_mode == "bptt":
            w_loss = warm.unrolled_relaxation_loss(y_tr[:32], x_tr[:32], n_steps=16, jfb_1step=False)
        else:
            w_loss = warm.unrolled_relaxation_loss(y_tr[:32], x_tr[:32], n_steps=16, jfb_1step=True)
        w_loss.backward()
        torch.nn.utils.clip_grad_norm_(warm.parameters(), 2.0)
        warm_opt.step()
    warm.eval()
    warm.focused_gibbs_infer(x_te[:16], n_sweeps=4, restarts=4)
    del warm, warm_opt
    _sync_if_cuda(device)

    results: dict[str, tuple[float, float, float]] = {}
    pl_pred_y = None
    pl_budget_s = 2.5

    for mode in ("pl_0bptt", "unrolled_bptt", "jfb_1step"):
        torch.manual_seed(77)
        model = CRAMzOperatorGraphEBM(n_nodes=6, n_choices=8, d_obs=16, d_z=16, n_z_gates=8, m_convex=5.0).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=0.07, weight_decay=1e-4)

        _sync_if_cuda(device)
        t0 = time.perf_counter()
        n_iters = 220
        batch_sz = 256
        iters_done = 0
        for it in range(n_iters):
            if mode != "pl_0bptt" and (time.perf_counter() - t0) >= pl_budget_s:
                break
            idx = (it * batch_sz) % x_tr.shape[0]
            xb = x_tr[idx : idx + batch_sz]
            yb = y_tr[idx : idx + batch_sz]
            opt.zero_grad()
            if mode == "pl_0bptt":
                loss = model.pseudolikelihood_loss(yb, xb)
            elif mode == "unrolled_bptt":
                loss = model.unrolled_relaxation_loss(yb, xb, n_steps=16, jfb_1step=False, step_lr=0.25)
            else:
                loss = model.unrolled_relaxation_loss(yb, xb, n_steps=16, jfb_1step=True, step_lr=0.25)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            iters_done += 1
        _sync_if_cuda(device)
        elapsed = time.perf_counter() - t0
        if mode == "pl_0bptt":
            # Cap baselines strictly at min(pl_time, 2.5s) so contention never grants BPTT extra steps
            pl_budget_s = max(min(elapsed, 2.5), 0.5)

        model.eval()
        pred_y, _ = model.focused_gibbs_infer(x_te, n_sweeps=10, restarts=10)
        if mode == "pl_0bptt":
            pl_pred_y = pred_y
        node_acc = (pred_y == y_te).float().mean().item() * 100.0
        exact_graph_acc = (pred_y == y_te).all(dim=-1).float().mean().item() * 100.0
        results[mode] = (elapsed, node_acc, exact_graph_acc)
        ms_iter = (elapsed * 1000.0) / max(iters_done, 1)
        print(
            f"  mode={mode:13s} | iters={iters_done:3d} | train_time={elapsed:5.3f} s ({ms_iter:5.2f} ms/it) | "
            f"node_acc={node_acc:6.2f}% | exact_graph_acc={exact_graph_acc:6.2f}%"
        )

    pl_time, _, pl_exact = results["pl_0bptt"]
    bptt_time, _, bptt_exact = results["unrolled_bptt"]
    jfb_time, _, jfb_exact = results["jfb_1step"]

    # Pure-tensor Spherical Base-10 ALU verification:
    # (a) Isolated exactness assert (100.00% across +, -, *, / incl. signed values & carries)
    # (b) End-to-end CRAM+z predicted 3-body root op y5 % 4 -> ALU
    assert pl_pred_y is not None
    alu = SphericalBase10RegisterALU(n_digits=6, d_sphere=15, scale_pow=2).to(device)
    a_d, a_s = alu.encode_scaled(op_te[:, 0])
    b_d, b_s = alu.encode_scaled(op_te[:, 1])
    op_pred = pl_pred_y[:, 5] % 4
    op_gold = y_te[:, 5] % 4
    assert set(op_gold.tolist()) == {0, 1, 2, 3}, f"Missing ALU ops in test split: {set(op_gold.tolist())}"

    va, vb = op_te[:, 0], op_te[:, 1]
    gold_val = torch.where(
        op_gold == 0,
        va + vb,
        torch.where(
            op_gold == 1,
            va - vb,
            torch.where(
                op_gold == 2,
                torch.div(va * vb, 100, rounding_mode="trunc"),
                torch.div(va * 100, vb, rounding_mode="trunc"),
            ),
        ),
    )

    iso_d, iso_s = alu.apply_op(a_d, a_s, b_d, b_s, op_gold)
    iso_decoded = alu.decode_scaled(iso_d, iso_s)
    alu_isolated_acc = (iso_decoded == gold_val).float().mean().item() * 100.0

    # Signed multi-digit carry & trunc-division boundary probe
    sig_a = torch.tensor([-3450, 1250, -999, 4321], device=device, dtype=torch.long)
    sig_b = torch.tensor([700, -250, -333, 567], device=device, dtype=torch.long)
    for test_op in range(4):
        op_vec = torch.full((4,), test_op, device=device, dtype=torch.long)
        sd, ss = alu.apply_op(*alu.encode_scaled(sig_a), *alu.encode_scaled(sig_b), op_vec)
        dec_sig = alu.decode_scaled(sd, ss)
        if test_op == 0:
            ref_sig = sig_a + sig_b
        elif test_op == 1:
            ref_sig = sig_a - sig_b
        elif test_op == 2:
            ref_sig = torch.div(sig_a * sig_b, 100, rounding_mode="trunc")
        else:
            ref_sig = torch.div(sig_a * 100, sig_b, rounding_mode="trunc")
        assert torch.equal(dec_sig, ref_sig), f"Signed ALU mismatch on op={test_op}: {dec_sig} vs {ref_sig}"

    out_d, out_s = alu.apply_op(a_d, a_s, b_d, b_s, op_pred)
    alu_decoded = alu.decode_scaled(out_d, out_s)
    alu_e2e_match = (alu_decoded == gold_val).float().mean().item() * 100.0
    alu_norm_err = max(
        (out_d.norm(dim=-1) - 1.0).abs().max().item(),
        (out_s.norm(dim=-1) - 1.0).abs().max().item(),
        (iso_d.norm(dim=-1) - 1.0).abs().max().item(),
        (iso_s.norm(dim=-1) - 1.0).abs().max().item(),
    )
    print(
        f"  Tensor ALU (6-digit Base-10 on S^14): isolated_exact={alu_isolated_acc:6.2f}% "
        f"| e2e_CRAM_acc={alu_e2e_match:6.2f}% | norm_err={alu_norm_err:.2e}"
    )

    assert pl_time < 3.0, f"PL training exceeded 3.0 s: {pl_time:.3f}s"
    assert pl_exact > bptt_exact + 10.0, f"PL ({pl_exact:.2f}%) did not beat BPTT ({bptt_exact:.2f}%)"
    assert pl_exact > jfb_exact + 10.0, f"PL ({pl_exact:.2f}%) did not beat JFB ({jfb_exact:.2f}%)"
    assert alu_isolated_acc == 100.0, f"Isolated ALU failed exact arithmetic: {alu_isolated_acc:.2f}%"
    assert alu_e2e_match >= 80.0, f"End-to-end CRAM->ALU below 80%: {alu_e2e_match:.2f}%"
    assert alu_norm_err < 1e-6, f"ALU spherical norm drift {alu_norm_err} >= 1e-6"
    print(f"  PASS: Besag PL (0 BPTT) won in {pl_time:.3f}s ({pl_exact:.2f}% vs BPTT {bptt_exact:.2f}% / JFB {jfb_exact:.2f}%).")
    return True


def run_invariant_3_crystal_norm_and_id_well(device: torch.device) -> bool:
    print("\n=== [Invariant 3] DeltaBirkhoff Crystal: Norm Drift < 1e-6 & S_5 id_well > 0 ===")
    G = 5
    Ds = 16
    B = 64
    L_stream = 512

    # First check direct algebraic S_G bijection conservation over L=1024 + id_well advantage
    torch.manual_seed(2025)
    crystal = EquivariantDeltaCrystal(d_model=32, num_slots=G, d_sphere=Ds, r_steps=2, temp=0.22).to(device)
    # Train crystal gently for 40 steps to interpret swap signal vs distractor noise
    opt = torch.optim.AdamW(crystal.parameters(), lr=0.025)

    # Synthetic control stream: class 0 -> identity distractor, class 1 -> cyclic shift (0->1->2->3->4->0)
    P_cyc = torch.roll(torch.eye(G, device=device), shifts=1, dims=0)
    P_id = torch.eye(G, device=device)

    for step_idx in range(220):
        # Train last 70 steps with 1.58-bit STE QAT enabled so ternary quantization stays sharp
        use_qat_step = step_idx >= 150
        for m in crystal.modules():
            if isinstance(m, BitLinear158):
                m.use_qat = use_qat_step

        feat_id = torch.randn(B, 32, device=device) * 0.40
        feat_id[:, :4] = -1.5  # distractor polarity across 4 feature channels
        feat_sw = torch.randn(B, 32, device=device) * 0.40
        feat_sw[:, :4] = +1.5  # swap polarity across 4 feature channels

        # Random S_5 slot permutations enforce true equivariance across arbitrary history states
        perm_idx = torch.stack([torch.randperm(G, device=device) for _ in range(B)], dim=0)
        base_slots = crystal.init_slots(B, device)
        slots_rand = torch.gather(base_slots, 1, perm_idx.unsqueeze(-1).expand(-1, -1, Ds))

        g_dist = torch.empty(B, 1, device=device).uniform_(0.10, 0.45)
        g_swap = torch.empty(B, 1, device=device).uniform_(0.85, 0.98)

        _, p_id_pred, _ = crystal.step(slots_rand, feat_id, g_dist, use_id_well=True, hard=False)
        _, p_sw_pred, _ = crystal.step(slots_rand, feat_sw, g_swap, use_id_well=True, hard=False)

        t_id = P_id.unsqueeze(0).expand(B, -1, -1)
        t_sw = P_cyc.unsqueeze(0).expand(B, -1, -1)
        loss = -(t_id * torch.log(p_id_pred.clamp(min=1e-6))).sum(dim=-1).mean() - (
            t_sw * torch.log(p_sw_pred.clamp(min=1e-6))
        ).sum(dim=-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    crystal.eval()

    def eval_stream(use_id_well: bool, qat: bool, length: int = L_stream) -> tuple[float, float, float, float]:
        for m in crystal.modules():
            if isinstance(m, BitLinear158):
                m.use_qat = qat
        with torch.no_grad():
            slots = crystal.init_slots(B, device)
            target_perm = P_id.unsqueeze(0).expand(B, -1, -1).clone()
            id_well_val = 0.0
            max_drift = 0.0
            for t in range(length):
                is_swap = (t % 32 == 5)
                feat = torch.randn(B, 32, device=device) * 0.45
                if is_swap:
                    feat[:, :4] = +1.5
                    gate = torch.full((B, 1), 0.92, device=device)
                    target_perm = torch.bmm(P_cyc.unsqueeze(0).expand(B, -1, -1), target_perm)
                else:
                    feat[:, :4] = -1.5
                    gate = torch.full((B, 1), 0.42, device=device)
                slots, _, id_well_val = crystal.step(
                    slots, feat, gate, use_id_well=use_id_well, hard=True
                )
                drift = (slots.norm(dim=-1) - 1.0).abs().max().item()
                max_drift = max(max_drift, drift)

            target_slots = torch.bmm(target_perm, crystal.init_slots(B, device))
            # Check exact S_5 match AND exact multiset bijection error
            slot_match = (slots.argmax(dim=-1) == target_slots.argmax(dim=-1)).all(dim=-1).float().mean().item() * 100.0
            # Multiset check: row-sum of matched Voronoi weights equals 1 for every slot
            sim = torch.bmm(slots, crystal.init_slots(B, device).transpose(-1, -2))
            voronoi_err = (sim.max(dim=-1).values - 1.0).abs().max().item()
            col_counts = F.one_hot(sim.argmax(dim=-1), num_classes=G).sum(dim=1).float()
            bijection_acc = (col_counts == 1.0).all(dim=-1).float().mean().item() * 100.0
        return slot_match, bijection_acc, max_drift, voronoi_err if use_id_well else voronoi_err

    acc_on, bij_on, drift_on, verr_on = eval_stream(use_id_well=True, qat=False, length=512)
    acc_off, bij_off, drift_off, _ = eval_stream(use_id_well=False, qat=False, length=512)
    acc_qat, bij_qat, drift_qat, verr_qat = eval_stream(use_id_well=True, qat=True, length=512)
    acc_1024, bij_1024, drift_1024, verr_1024 = eval_stream(use_id_well=True, qat=False, length=1024)
    id_well_scalar = float(F.softplus(crystal.gamma_id).item())

    print(f"  L= 512, id_well={id_well_scalar:.3f} > 0, FP32 | S_5 exact={acc_on:6.2f}% | bij={bij_on:6.2f}% | norm_drift={drift_on:.2e} | voronoi_err={verr_on:.2e}")
    print(f"  L= 512, id_well=0.000 (off), FP32 | S_5 exact={acc_off:6.2f}% | bij={bij_off:6.2f}% | norm_drift={drift_off:.2e}")
    print(f"  L= 512, id_well={id_well_scalar:.3f} > 0, QAT  | S_5 exact={acc_qat:6.2f}% | bij={bij_qat:6.2f}% | norm_drift={drift_qat:.2e} | voronoi_err={verr_qat:.2e}")
    print(f"  L=1024, id_well={id_well_scalar:.3f} > 0, FP32 | S_5 exact={acc_1024:6.2f}% | bij={bij_1024:6.2f}% | norm_drift={drift_1024:.2e} | voronoi_err={verr_1024:.2e}")

    assert id_well_scalar > 0.0, "id_well must be strictly positive"
    assert max(drift_on, drift_qat, drift_1024) < 1e-6, "Spherical norm drift >= 1e-6"
    assert bij_on == 100.0 and bij_qat == 100.0 and bij_1024 == 100.0, "Lost S_5 bijection conservation"
    assert acc_on > acc_off, f"Expected id_well > 0 ({acc_on:.2f}%) to outperform id_well=0 ({acc_off:.2f}%)"
    assert acc_qat >= 95.0, f"1.58-bit STE QAT S_5 dropped unexpectedly: {acc_qat:.2f}%"
    print(f"  PASS: id_well={id_well_scalar:.3f} > 0 holds S_5 at {acc_on:.2f}% (vs {acc_off:.2f}% without well), drift < 1e-6.")
    return True


def run_invariant_4_dual_certificate_traps(device: torch.device) -> bool:
    print("\n=== [Invariant 4] 100% Trap Rejection via Dual Certificate (sigma^2_q, E <= tau) ===")
    torch.manual_seed(314)
    # 1) Prepare Omega-Woodbury epistemic covariance P_t trained on known low-rank correlated subspace
    mem = OmegaWoodburyMemory(d_model=32, n_heads=1, d_head=32, ridge_lambda=0.02, gamma_max=0.9999).to(device)
    # 2) Prepare CRAM+z physical energy verifier trained on valid operator graphs
    ebm = CRAMzOperatorGraphEBM(n_nodes=6, n_choices=8, d_obs=16, d_z=16, n_z_gates=8, m_convex=5.0).to(device)

    x_tr, y_tr, _ = generate_coupled_dag_dataset(512, n_nodes=6, seed=42, device=device)
    opt = torch.optim.AdamW(ebm.parameters(), lr=0.07, weight_decay=1e-4)
    for it in range(250):
        idx = (it * 256) % x_tr.shape[0]
        opt.zero_grad()
        ebm.pseudolikelihood_loss(y_tr[idx : idx + 256], x_tr[idx : idx + 256]).backward()
        torch.nn.utils.clip_grad_norm_(ebm.parameters(), 2.0)
        opt.step()
    ebm.eval()
    mem.eval()

    # Construct an orthogonal split: Known subspace dims [0:10], Unseen trap subspace dims [10:32]
    N_seen = 48
    N_probe = 64
    seen_keys = torch.zeros(N_seen, 32, device=device)
    seen_keys[:, :10] = torch.randn(N_seen, 10, device=device)
    unseen_keys = torch.zeros(N_probe, 32, device=device)
    unseen_keys[:, 10:] = torch.randn(N_probe, 22, device=device)  # Orthogonal trap subspace

    with torch.no_grad():
        M, P = mem.init_state(1, device)
        for i in range(N_seen):
            gate = torch.ones(1, 1, device=device)
            M, P = mem.update(M, P, seen_keys[i : i + 1], gate)

        # Evaluate epistemic variance sigma^2_q across:
        # (A) Known solvable probes: seen key + valid DAG observation
        # (B) Unseen entity traps: unseen key (high sigma^2_q) + valid DAG observation (low E!)
        # (C) Contradiction traps: seen key (low sigma^2_q!) + contradictory/corrupted DAG observation (high E!)
        M_p = M.expand(N_probe, -1, -1, -1)
        P_p = P.expand(N_probe, -1, -1, -1)

        known_q = seen_keys[torch.arange(N_probe, device=device) % N_seen] + 0.05 * torch.cat(
            [torch.randn(N_probe, 10, device=device), torch.zeros(N_probe, 22, device=device)], dim=-1
        )
        _, sig_A = mem.query_only(M_p, P_p, known_q)
        _, sig_B = mem.query_only(M_p, P_p, unseen_keys)
        _, sig_C = mem.query_only(M_p, P_p, known_q)  # familiar tokens -> low variance!

        cal_gen = torch.Generator(device="cpu").manual_seed(707)
        cal_noise = torch.randn(N_probe, 10, generator=cal_gen).to(device)
        cal_q = seen_keys[(torch.arange(N_probe, device=device) + 7) % N_seen] + 0.05 * torch.cat(
            [cal_noise, torch.zeros(N_probe, 22, device=device)], dim=-1
        )
        x_cal, y_cal, _ = generate_coupled_dag_dataset(128, n_nodes=6, seed=2026, device=device)
        _, eng_cal = ebm.focused_gibbs_infer(x_cal, n_sweeps=10, restarts=10)
        _, sig_cal = mem.query_only(M_p, P_p, cal_q)

        x_te, y_te, _ = generate_coupled_dag_dataset(N_probe, n_nodes=6, seed=1234, device=device)
        # Normal valid DAG predictions for A & B
        pred_A, eng_A = ebm.focused_gibbs_infer(x_te, n_sweeps=10, restarts=10)
        _, eng_B = ebm.focused_gibbs_infer(x_te, n_sweeps=10, restarts=10)

        # Ungrounded / missing-constraint DAG trap for C: familiar syntax keys (known_q), zero valid clues
        bad_gen = torch.Generator(device="cpu").manual_seed(909)
        x_bad = (0.04 * torch.randn(x_te.shape, generator=bad_gen)).to(device)
        _, eng_C = ebm.focused_gibbs_infer(x_bad, n_sweeps=10, restarts=10)

        # Calibrate thresholds strictly from valid held-out calibration set (zero trap leakage)
        theta_epi = float((sig_cal.max() + 0.08).item())
        tau_energy = float((eng_cal.max() + 2.0).item())

        accept_A = (sig_A <= theta_epi) & (eng_A <= tau_energy)
        accept_B = (sig_B <= theta_epi) & (eng_B <= tau_energy)
        accept_C = (sig_C <= theta_epi) & (eng_C <= tau_energy)

        # Measure single-gate failure rates to prove why BOTH gates are strictly necessary
        energy_only_reject_B = float((eng_B > tau_energy).float().mean().item() * 100.0)
        var_only_reject_C = float((sig_C > theta_epi).float().mean().item() * 100.0)
        dual_accept_A = float(accept_A.float().mean().item() * 100.0)
        dual_reject_B = float((~accept_B).float().mean().item() * 100.0)
        dual_reject_C = float((~accept_C).float().mean().item() * 100.0)
        acc_A_exact = float((pred_A == y_te).all(dim=-1).float().mean().item() * 100.0)

    print(f"  Thresholds calibrated: theta_epi={theta_epi:.4f}, tau_energy={tau_energy:.4f}")
    print(f"  (A) Known Solvable  | sig2=[{sig_A.min():.3f},{sig_A.max():.3f}] E=[{eng_A.min():.2f},{eng_A.max():.2f}] | accept={dual_accept_A:6.2f}% | solve={acc_A_exact:6.2f}%")
    print(f"  (B) Unseen Entity   | sig2=[{sig_B.min():.3f},{sig_B.max():.3f}] E=[{eng_B.min():.2f},{eng_B.max():.2f}] | energy_only_rej={energy_only_reject_B:5.1f}% | dual_rej={dual_reject_B:6.2f}%")
    print(f"  (C) Ungrounded Trap | sig2=[{sig_C.min():.3f},{sig_C.max():.3f}] E=[{eng_C.min():.2f},{eng_C.max():.2f}] | var_only_rej={var_only_reject_C:5.1f}%    | dual_rej={dual_reject_C:6.2f}%")

    assert dual_accept_A == 100.0, f"False rejection on known solvable probes: {dual_accept_A}%"
    assert dual_reject_B == 100.0, f"Missed unseen entity trap: {dual_reject_B}%"
    assert dual_reject_C == 100.0, f"Missed contradiction trap: {dual_reject_C}%"
    assert energy_only_reject_B < 10.0 and var_only_reject_C < 10.0, "Traps should require both gates!"
    print("  PASS: Dual Certificate (sigma^2_q <= theta, E <= tau) hit 100.00% trap rejection & 100.00% valid acceptance.")
    return True


def run_anti_cheat_checks(device: torch.device) -> bool:
    """Zero-Cheat Laws 2, 3, 4 on Calcite5StreamingCore: pure CE live grads, strict causality, crystal ablation."""
    torch.manual_seed(9)
    core = Calcite5StreamingCore(vocab_size=32, d_model=32, window_size=6, num_slots=5, d_sphere=8, r_steps=2).to(device)
    B, T = 4, 10
    toks = torch.randint(0, 32, (B, T), device=device)

    # 1. Zero-Cheat Law 2: Pure next-token cross-entropy ONLY (0 auxiliary loss terms on gate/slots)
    core.train()
    core.zero_grad()
    state = core.init_state(batch_size=B, device=device)
    logits_list = []
    for t in range(T):
        logits_t, state, _, _ = core.step_token(toks[:, t], state, t, hard_perm=False, use_crystal=True)
        logits_list.append(logits_t)
    logits_seq = torch.stack(logits_list, dim=1)  # (B, T, V)
    ce_loss = F.cross_entropy(logits_seq[:, :-1].reshape(-1, 32), toks[:, 1:].reshape(-1))
    ce_loss.backward()
    dead = [n for n, p in core.named_parameters() if p.requires_grad and (p.grad is None or p.grad.abs().sum().item() == 0.0)]
    assert len(dead) == 0, f"Found dead parameters under pure next-token CE: {dead}"

    # 2. Zero-Cheat Law 3: Strict causality (perturbing t >= 5 leaves logits[:, :5] bit-identical)
    core.eval()
    with torch.no_grad():
        toks_pert = toks.clone()
        toks_pert[:, 5:] = (toks_pert[:, 5:] + 7) % 32
        s_orig = core.init_state(batch_size=B, device=device)
        s_pert = core.init_state(batch_size=B, device=device)
        s_ablate = core.init_state(batch_size=B, device=device)
        l_orig, l_pert, l_ablate = [], [], []
        for t in range(T):
            lo, s_orig, _, _ = core.step_token(toks[:, t], s_orig, t, hard_perm=True, use_crystal=True)
            lp, s_pert, _, _ = core.step_token(toks_pert[:, t], s_pert, t, hard_perm=True, use_crystal=True)
            la, s_ablate, _, _ = core.step_token(toks[:, t], s_ablate, t, hard_perm=True, use_crystal=False)
            l_orig.append(lo)
            l_pert.append(lp)
            l_ablate.append(la)
        orig_seq = torch.stack(l_orig, dim=1)
        pert_seq = torch.stack(l_pert, dim=1)
        ablate_seq = torch.stack(l_ablate, dim=1)

    causal_diff = float((orig_seq[:, :5] - pert_seq[:, :5]).abs().max().item())
    ablation_diff = float((orig_seq - ablate_seq).abs().max().item())
    assert causal_diff == 0.0, f"Causality leak detected: max prefix diff={causal_diff}"
    assert ablation_diff > 1e-3, f"Crystal ablation made negligible logit impact: max diff={ablation_diff}"

    print(
        f"\n=== [Anti-Cheat Laws 2/3/4] 0 dead params under pure CE ({sum(1 for _ in core.parameters())} tensors) "
        f"| causal_diff={causal_diff:.1e} | crystal_ablation_diff={ablation_diff:.4f} ==="
    )
    return True


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Calcite-5 (SimVQ-CRAM v5) Tracer #1 | device={device}")
    t_start = time.perf_counter()

    ok1 = run_invariant_1_o1_state(device)
    ok2 = run_invariant_2_pl_vs_bptt_and_alu(device)
    ok3 = run_invariant_3_crystal_norm_and_id_well(device)
    ok4 = run_invariant_4_dual_certificate_traps(device)
    ok5 = run_anti_cheat_checks(device)

    _sync_if_cuda(device)
    total_t = time.perf_counter() - t_start
    assert ok1 and ok2 and ok3 and ok4 and ok5
    print(f"\nALL 4 CORE INVARIANTS + ANTI-CHEAT LAWS (2/3/4) PASSED in {total_t:.2f} s.")


if __name__ == "__main__":
    main()
