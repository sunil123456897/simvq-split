"""
Pure Neural Equal-Parameter Raw-Byte Baselines (`RoPETransformerLM` & `ScaledHybridDeltaSWALM`)
for Phase K Scaling Laws (`5M -> 25M -> 85M`, `vocab_size = 259`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from simvq_split.baselines import (
    BaselineConfig,
    HouseholderDeltaLayer,
    RMSNorm,
    RoPETransformerLM,
    SwiGLU,
    TransformerBlock,
)


def householder_deltanet_wy_scan(
    q_chunks: torch.Tensor,
    k_seq: torch.Tensor,
    v_seq: torch.Tensor,
    beta_seq: torch.Tensor,
    gamma_seq: torch.Tensor,
    Bc: int = 16,
) -> torch.Tensor:
    """
    Exact block-parallel WY factorization of `householder_deltanet_chunk_scan_jit`.
    Reduces sequential Python/kernel steps by `Bc` (`192 -> 12` at `L=768`) via
    batched `torch.linalg.solve_triangular` inside a `Float32` AMP shield.
    """
    B, M_len, C, H, d_k = q_chunks.shape
    d_v = v_seq.size(-1)
    BH = B * H
    dev = k_seq.device
    orig_dtype = v_seq.dtype

    pad_m = (Bc - (M_len % Bc)) % Bc
    M_pad = M_len + pad_m
    n_blk = M_pad // Bc

    with torch.amp.autocast(dev.type, enabled=False):
        q = q_chunks.float().permute(0, 3, 1, 2, 4).reshape(BH, M_len, C, d_k)
        k = k_seq.float().permute(0, 2, 1, 3).reshape(BH, M_len, d_k)
        v = v_seq.float().permute(0, 2, 1, 3).reshape(BH, M_len, d_v)
        b = beta_seq.float().permute(0, 2, 1, 3).reshape(BH, M_len)
        g = gamma_seq.float().permute(0, 2, 1, 3).reshape(BH, M_len)

        if pad_m > 0:
            q = F.pad(q, (0, 0, 0, 0, 0, pad_m))
            k = F.pad(k, (0, 0, 0, pad_m))
            v = F.pad(v, (0, 0, 0, pad_m))
            b = F.pad(b, (0, pad_m))
            g = F.pad(g, (0, pad_m), value=1.0)

        q_b = q.view(BH, n_blk, Bc, C, d_k)
        k_b = k.view(BH, n_blk, Bc, d_k)
        v_b = v.view(BH, n_blk, Bc, d_v)
        b_b = b.view(BH, n_blk, Bc)
        g_b = g.view(BH, n_blk, Bc)

        log_g = torch.log(g_b.double().clamp(min=1e-12))
        log_pref = F.pad(torch.cumsum(log_g, dim=-1), (1, 0), value=0.0)

        tril_strict = torch.tril(torch.ones(Bc, Bc, device=dev, dtype=torch.bool), diagonal=-1)
        diff_L = log_pref[..., 1:, None] - log_pref[..., None, 1:]
        G_L = torch.where(
            tril_strict, torch.exp(diff_L.clamp(max=0.0)), torch.zeros_like(diff_L)
        ).float()

        diff_R = log_pref[..., :-1, None] - log_pref[..., None, 1:]
        G_R = torch.where(
            tril_strict, torch.exp(diff_R.clamp(max=0.0)), torch.zeros_like(diff_R)
        ).float()

        G_j0 = torch.exp(log_pref[..., :-1]).float()
        G_j1_0 = torch.exp(log_pref[..., 1:]).float()
        G_end_0 = torch.exp(log_pref[..., -1]).float()
        G_end_t1 = torch.exp(log_pref[..., -1:] - log_pref[..., 1:]).float()

        kk = torch.matmul(k_b, k_b.transpose(-1, -2))
        eye = torch.eye(Bc, device=dev, dtype=torch.float32)
        L = eye + b_b.unsqueeze(-1) * G_L * kk

        rhs_v = b_b.unsqueeze(-1) * v_b
        rhs_k = (b_b * G_j1_0).unsqueeze(-1) * k_b
        rhs = torch.cat((rhs_v, rhs_k), dim=-1)
        sol = torch.linalg.solve_triangular(L, rhs, upper=False, unitriangular=True)
        U_all = sol[..., :d_v]
        W_all = sol[..., d_v:]

        q_flat = q_b.view(BH, n_blk, Bc * C, d_k)
        qk_all = torch.matmul(q_flat, k_b.transpose(-1, -2)).view(BH, n_blk, Bc, C, Bc)
        qk_weighted = (qk_all * G_R.unsqueeze(3)).view(BH, n_blk, Bc * C, Bc)
        q_scaled = (q_b * G_j0.unsqueeze(-1).unsqueeze(-1)).view(BH, n_blk, Bc * C, d_k)

        S = torch.zeros(BH, d_v, d_k, device=dev, dtype=torch.float32)
        outs = []
        for bi in range(n_blk):
            U_i = U_all[:, bi]
            W_i = W_all[:, bi]
            D_i = U_i - torch.bmm(W_i, S.transpose(1, 2))

            read_i = torch.bmm(q_scaled[:, bi], S.transpose(1, 2)) + torch.bmm(
                qk_weighted[:, bi], D_i
            )
            outs.append(read_i.view(B, H, Bc * C, d_v).permute(0, 2, 1, 3))

            D_dec = D_i * G_end_t1[:, bi].unsqueeze(-1)
            S = G_end_0[:, bi].view(BH, 1, 1) * S + torch.bmm(
                D_dec.transpose(1, 2), k_b[:, bi]
            )

        out = torch.cat(outs, dim=1)[:, : M_len * C]
        return out.to(orig_dtype)


class FastHouseholderDeltaLayer(HouseholderDeltaLayer):
    """Exact block-parallel WY `HouseholderDeltaLayer` keeping zero parameter diff."""

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

        y = householder_deltanet_wy_scan(q_chunks, k_seq, v_seq, beta, gamma, Bc=16)
        return self.out_proj(self.out_norm(y[:, :T].reshape(B, T, D_in)))


@dataclass
class HybridDeltaSWAConfig:
    vocab_size: int = 259
    d_model: int = 256
    n_heads: int = 4
    n_swa_pre: int = 1
    n_delta: int = 3
    n_swa_post: int = 1
    d_ff: int = 928
    window_bytes: int = 36
    chunk_size: int = 4


class ScaledHybridDeltaSWALM(nn.Module):
    """
    Modern Gated DeltaNet (Householder Reflection DeltaProduct) + Sliding-Window Attention Hybrid
    over raw UTF-8 bytes (`vocab_size = 259`), re-using core `simvq_split.baselines` primitives.
    """

    def __init__(self, cfg: HybridDeltaSWAConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.tok_emb.weight, std=0.02)

        self.pre = nn.ModuleList(
            [
                TransformerBlock(
                    cfg.d_model,
                    cfg.n_heads,
                    cfg.d_ff,
                    window_size=cfg.window_bytes,
                )
                for _ in range(cfg.n_swa_pre)
            ]
        )
        self.d_norms = nn.ModuleList([RMSNorm(cfg.d_model) for _ in range(cfg.n_delta)])
        self.deltas = nn.ModuleList(
            [
                FastHouseholderDeltaLayer(cfg.d_model, cfg.n_heads, chunk_size=cfg.chunk_size)
                for _ in range(cfg.n_delta)
            ]
        )
        self.m_norms = nn.ModuleList([RMSNorm(cfg.d_model) for _ in range(cfg.n_delta)])
        self.mlps = nn.ModuleList([SwiGLU(cfg.d_model, cfg.d_ff) for _ in range(cfg.n_delta)])
        self.post = nn.ModuleList(
            [
                TransformerBlock(
                    cfg.d_model,
                    cfg.n_heads,
                    cfg.d_ff,
                    window_size=cfg.window_bytes,
                )
                for _ in range(cfg.n_swa_post)
            ]
        )
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(token_ids)
        for blk in self.pre:
            x = blk(x)
        for dn, dl, mn, mlp in zip(self.d_norms, self.deltas, self.m_norms, self.mlps):
            x = x + dl(dn(x))
            x = x + mlp(mn(x))
        for blk in self.post:
            x = blk(x)
        return self.lm_head(self.norm_f(x))


def compute_hybrid_delta_cache_kb(cfg: HybridDeltaSWAConfig, seq_len: int = 2048) -> float:
    """Exact strictly `O(1)` KV + Delta matrix state cache footprint in KB."""
    del seq_len
    n_swa = cfg.n_swa_pre + cfg.n_swa_post
    local_kv = n_swa * 2 * cfg.window_bytes * cfg.d_model * 2
    head_d = cfg.d_model // cfg.n_heads
    delta_s = cfg.n_delta * cfg.n_heads * head_d * head_d * 4
    return (local_kv + delta_s) / 1024.0


def compute_rope_cache_kb(cfg: BaselineConfig, seq_len: int = 2048) -> float:
    """Exact `O(L)` KV cache footprint of `RoPETransformerLM` at sequence length `seq_len` (BF16)."""
    return (cfg.n_layers * 2 * int(seq_len) * cfg.d_model * 2) / 1024.0


def build_matched_baselines_for_tier(
    tier_name: str, target_params: int
) -> dict[str, dict[str, object]]:
    """
    Constructs `< 1.0%` parameter-matched `RoPETransformerLM` and `ScaledHybridDeltaSWALM`
    configurations for `S1_5M`, `S2_25M`, or `S3_85M`.
    """
    tier_shapes = {
        "S1_5M": (256, 4, 6, (1, 3, 1)),
        "S2_25M": (448, 7, 10, (2, 5, 2)),
        "S3_85M": (768, 12, 12, (2, 7, 2)),
    }
    if tier_name not in tier_shapes:
        raise ValueError(f"Unknown tier `{tier_name}`")
    d_model, n_heads, n_layers_rope, (swa_pre, n_delta, swa_post) = tier_shapes[tier_name]

    # Tune d_ff for RoPETransformerLM
    best_rope_cfg: BaselineConfig | None = None
    best_rope_params = 0
    best_rope_err = float("inf")
    for d_ff in range(256, 4096, 32):
        cand = BaselineConfig(
            vocab_size=259,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers_rope,
            d_ff=d_ff,
            window_size=36,
        )
        np_c = int(sum(p.numel() for p in RoPETransformerLM(cand).parameters() if p.requires_grad))
        err = abs(np_c - target_params)
        if err < best_rope_err:
            best_rope_err = err
            best_rope_cfg = cand
            best_rope_params = np_c
        if np_c > target_params and err > best_rope_err:
            break

    # Tune d_ff for ScaledHybridDeltaSWALM
    best_hyb_cfg: HybridDeltaSWAConfig | None = None
    best_hyb_params = 0
    best_hyb_err = float("inf")
    for d_ff in range(256, 4096, 32):
        cand_h = HybridDeltaSWAConfig(
            vocab_size=259,
            d_model=d_model,
            n_heads=n_heads,
            n_swa_pre=swa_pre,
            n_delta=n_delta,
            n_swa_post=swa_post,
            d_ff=d_ff,
            window_bytes=36,
            chunk_size=4,
        )
        np_h = int(
            sum(p.numel() for p in ScaledHybridDeltaSWALM(cand_h).parameters() if p.requires_grad)
        )
        err_h = abs(np_h - target_params)
        if err_h < best_hyb_err:
            best_hyb_err = err_h
            best_hyb_cfg = cand_h
            best_hyb_params = np_h
        if np_h > target_params and err_h > best_hyb_err:
            break

    assert best_rope_cfg is not None and best_hyb_cfg is not None
    return {
        "RoPETransformerLM": {
            "config": asdict(best_rope_cfg),
            "params": best_rope_params,
            "parity_pct": round(100.0 * (best_rope_params - target_params) / target_params, 3),
            "cache_kb_L512": round(compute_rope_cache_kb(best_rope_cfg, 512), 2),
            "cache_kb_L2048": round(compute_rope_cache_kb(best_rope_cfg, 2048), 2),
        },
        "ScaledHybridDeltaSWALM": {
            "config": asdict(best_hyb_cfg),
            "params": best_hyb_params,
            "parity_pct": round(100.0 * (best_hyb_params - target_params) / target_params, 3),
            "cache_kb_L512": round(compute_hybrid_delta_cache_kb(best_hyb_cfg, 512), 2),
            "cache_kb_L2048": round(compute_hybrid_delta_cache_kb(best_hyb_cfg, 2048), 2),
        },
    }
