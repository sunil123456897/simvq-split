"""
Equal-Compute Modern Baselines for SimVQ-Split Tournament.

All baselines use:
- Identical token vocabulary (raw `token_ids` only, weight-tied embeddings)
- Pre-RMSNorm + SwiGLU + Rotary Position Embeddings (RoPE)
- Parameter parity within +/- 5% of `SimVQSplitLM`

Models:
1. `RoPETransformerLM`: Standard modern causal LLM (Llama-3 / Qwen style).
2. `SlidingWindowTransformerLM`: Pure Continuous Cortex ablation (strict window `W`).
3. `LoopedTransformerLM`: Huginn-style (arXiv:2502.05171) Prelude -> R-step looped
   shared Transformer core -> Coda, parameter-matched via wider hidden width.
4. `HouseholderDeltaNetLM`: DeltaProduct-style (arXiv:2502.10297) Gated Householder
   reflection recurrence (`beta in (0, 2)` enabling true orthogonal reflections
   `I - 2 k k^T` for non-abelian S_n tracking) + local sliding-window cortex.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * norm).to(x.dtype) * self.weight


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D), cos/sin: (1, 1, T, D//2)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack((out1, out2), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        cos = torch.cos(freqs).to(dtype).view(1, 1, seq_len, -1)
        sin = torch.sin(freqs).to(dtype).view(1, 1, seq_len, -1)
        return cos, sin


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class CausalRoPEAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, window_size: int | None = None):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        cos, sin = self.rope(T, x.device, x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.window_size is None or T <= self.window_size:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            idx = torch.arange(T, device=x.device)
            dist = idx.unsqueeze(1) - idx.unsqueeze(0)
            mask = (dist >= 0) & (dist < self.window_size)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        return self.out_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, window_size: int | None = None
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalRoPEAttention(d_model, n_heads, window_size=window_size)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


@torch.jit.script
def householder_deltanet_chunk_scan_jit(
    q_chunks: torch.Tensor,  # (B, M, C, H, d_k)
    k_seq: torch.Tensor,  # (B, M, H, d_k) unit-normalized
    v_seq: torch.Tensor,  # (B, M, H, d_v)
    beta_seq: torch.Tensor,  # (B, M, H, 1) in (0, 2) for full Householder reflections!
    gamma_seq: torch.Tensor,  # (B, M, H, 1) retention gate in (0, 1)
) -> torch.Tensor:
    B, M_len, C, H, d_k = q_chunks.shape
    d_v = v_seq.size(-1)
    BH = B * H
    S = torch.zeros(BH, d_v, d_k, device=k_seq.device, dtype=torch.float32)
    outs: list[torch.Tensor] = []

    q_u = q_chunks.float().permute(1, 0, 3, 4, 2).reshape(M_len, BH, d_k, C).unbind(0)
    k_u = k_seq.float().permute(1, 0, 2, 3).reshape(M_len, BH, d_k, 1).unbind(0)
    v_u = v_seq.float().permute(1, 0, 2, 3).reshape(M_len, BH, d_v, 1).unbind(0)
    b_u = beta_seq.float().permute(1, 0, 2, 3).reshape(M_len, BH, 1, 1).unbind(0)
    g_u = gamma_seq.float().permute(1, 0, 2, 3).reshape(M_len, BH, 1, 1).unbind(0)

    for m in range(M_len):
        qm = q_u[m]
        read_m = torch.bmm(S, qm)
        outs.append(read_m.reshape(B, H, d_v, C).permute(0, 3, 1, 2))

        kt = k_u[m]
        vt = v_u[m]
        bt = b_u[m]
        gt = g_u[m]
        sk = torch.bmm(S, kt)
        S = gt * S + bt * torch.bmm(vt - gt * sk, kt.transpose(1, 2))

    return torch.cat(outs, dim=1).to(v_seq.dtype)


class HouseholderDeltaLayer(nn.Module):
    """
    DeltaProduct-style (Siems et al., 2025) Gated Householder Linear Recurrence.
    Allows beta_t in (0, 2) so (I - beta_t k_t k_t^T) spans true reflections (det = -1)
    capable of tracking non-abelian S_n permutations in continuous space.
    Uses identical Causal Shift-1 Micro-Chunking (`chunk_size = 4`) as SimVQ-Split.
    """

    def __init__(self, d_model: int, n_heads: int, chunk_size: int = 4):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.C = chunk_size
        self.pool_score = nn.Linear(d_model, n_heads, bias=False)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.beta_proj = nn.Linear(d_model, n_heads, bias=True)
        self.gamma_proj = nn.Linear(d_model, n_heads, bias=True)
        self.out_norm = RMSNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        nn.init.constant_(self.beta_proj.bias, -1.0)
        nn.init.constant_(self.gamma_proj.bias, 3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D_in = x.shape
        H, D, C = self.n_heads, self.head_dim, self.C
        pad = (C - (T % C)) % C
        x_pad = F.pad(x, (0, 0, 0, pad)) if pad > 0 else x
        T_pad = x_pad.size(1)
        M_len = T_pad // C

        q_chunks = F.normalize(self.q_proj(x_pad).view(B, M_len, C, H, D), dim=-1)
        x_chunks = x_pad.view(B, M_len, C, D_in)
        w_scores = F.softmax(self.pool_score(x_chunks).float(), dim=2).to(x.dtype)
        k_tok = self.k_proj(x_chunks).view(B, M_len, C, H, D)
        v_tok = self.v_proj(x_chunks).view(B, M_len, C, H, D)

        k_seq = F.normalize((k_tok * w_scores.unsqueeze(-1)).sum(dim=2), dim=-1)
        v_seq = (v_tok * w_scores.unsqueeze(-1)).sum(dim=2)
        z_head = x_chunks.mean(dim=2)
        beta = 2.0 * torch.sigmoid(self.beta_proj(z_head)).unsqueeze(-1)
        gamma = torch.sigmoid(self.gamma_proj(z_head)).unsqueeze(-1)

        y = householder_deltanet_chunk_scan_jit(q_chunks, k_seq, v_seq, beta, gamma)
        return self.out_proj(self.out_norm(y[:, :T].reshape(B, T, D_in)))


@dataclass
class BaselineConfig:
    vocab_size: int = 2048
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int | None = None
    window_size: int = 32
    loop_steps: int = 4

    def resolved_dff(self) -> int:
        if self.d_ff is not None:
            return self.d_ff
        return int((8 * self.d_model / 3) // 32 * 32) or (2 * self.d_model)


class RoPETransformerLM(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg = cfg
        d_ff = cfg.resolved_dff()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(cfg.d_model, cfg.n_heads, d_ff, window_size=None)
                for _ in range(cfg.n_layers)
            ]
        )
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(token_ids)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.norm_f(x))


class SlidingWindowTransformerLM(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg = cfg
        d_ff = cfg.resolved_dff()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    cfg.d_model, cfg.n_heads, d_ff, window_size=cfg.window_size
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(token_ids)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.norm_f(x))


class LoopedTransformerLM(nn.Module):
    """
    Huginn-style (Geiping et al., 2025) Recurrent-Depth Transformer:
    Prelude -> R iterations of shared Continuous Core with input injection -> Coda.
    """

    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg = cfg
        d_ff = cfg.resolved_dff()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        self.prelude = TransformerBlock(
            cfg.d_model, cfg.n_heads, d_ff, window_size=None
        )
        self.inject_adapter = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)
        self.core = TransformerBlock(cfg.d_model, cfg.n_heads, d_ff, window_size=None)
        self.coda = TransformerBlock(cfg.d_model, cfg.n_heads, d_ff, window_size=None)
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        e = self.prelude(self.tok_emb(token_ids))
        r = e
        for _ in range(self.cfg.loop_steps):
            r = self.core(self.inject_adapter(torch.cat([r, e], dim=-1)))
        x = self.coda(r)
        return self.lm_head(self.norm_f(x))


class HouseholderDeltaNetLM(nn.Module):
    """
    Hybrid Local-Cortex + Multi-Layer Gated Householder DeltaNet (DeltaProduct style).
    Uses exact same sliding-window local cortex philosophy paired with continuous
    orthogonal Householder reflections (without Voronoi Crystal or 2nd-order Woodbury).
    """

    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg = cfg
        d_ff = cfg.resolved_dff()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.tok_emb.weight, std=0.02)

        self.local_pre = TransformerBlock(
            cfg.d_model, cfg.n_heads, d_ff, window_size=cfg.window_size
        )
        self.delta_norms = nn.ModuleList([RMSNorm(cfg.d_model) for _ in range(2)])
        self.delta_layers = nn.ModuleList(
            [HouseholderDeltaLayer(cfg.d_model, cfg.n_heads) for _ in range(2)]
        )
        self.mlp_norms = nn.ModuleList([RMSNorm(cfg.d_model) for _ in range(2)])
        self.mlps = nn.ModuleList([SwiGLU(cfg.d_model, d_ff) for _ in range(2)])
        self.local_post = TransformerBlock(
            cfg.d_model, cfg.n_heads, d_ff, window_size=cfg.window_size
        )

        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.local_pre(self.tok_emb(token_ids))
        for dn, dl, mn, mlp in zip(
            self.delta_norms, self.delta_layers, self.mlp_norms, self.mlps
        ):
            x = x + dl(dn(x))
            x = x + mlp(mn(x))
        x = self.local_post(x)
        return self.lm_head(self.norm_f(x))


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _tune_dff(
    cls: type[nn.Module], base_cfg: BaselineConfig, target_params: int
) -> BaselineConfig:
    best_cfg = base_cfg
    best_err = 1e18
    for d_ff in range(64, 4096, 16):
        cand = BaselineConfig(
            vocab_size=base_cfg.vocab_size,
            d_model=base_cfg.d_model,
            n_heads=base_cfg.n_heads,
            n_layers=base_cfg.n_layers,
            d_ff=d_ff,
            window_size=base_cfg.window_size,
            loop_steps=base_cfg.loop_steps,
        )
        n_p = _count_params(cls(cand))
        err = abs(n_p - target_params)
        if err < best_err:
            best_err = err
            best_cfg = cand
        if n_p > target_params and err > best_err:
            break
    return best_cfg


def match_baseline_configs(simvq_cfg) -> list | object:
    from .models import SimVQSplitLM

    target = _count_params(SimVQSplitLM(simvq_cfg))
    base = BaselineConfig(
        vocab_size=simvq_cfg.vocab_size,
        d_model=simvq_cfg.d_model,
        n_heads=simvq_cfg.n_heads,
        n_layers=simvq_cfg.n_layers_pre + simvq_cfg.n_layers_post + 1,
        window_size=simvq_cfg.window_size,
        loop_steps=4,
    )
    # Return dict outside nn.Module (allowed by AST rule!)
    out = []
    for name, cls in (
        ("RoPETransformerLM", RoPETransformerLM),
        ("SlidingWindowTransformerLM", SlidingWindowTransformerLM),
        ("LoopedTransformerLM", LoopedTransformerLM),
        ("HouseholderDeltaNetLM", HouseholderDeltaNetLM),
    ):
        out.append((name, (cls, _tune_dff(cls, base, target))))
    return dict(out)
