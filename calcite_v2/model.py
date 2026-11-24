"""
Calcite v2 Unified System-1 Language Model (`calcite_v2/model.py`).

Supports clean single-axis ablations without monkey-patching or hollow wrappers:
- `topology`: `"hourglass"` (v1 baseline) or `"interleaved"` (fast Woodbury -> Crystal 1 ->
  slow epistemic Woodbury -> Crystal 2 interleaved between local SWA blocks).
- `use_scaled_decoder`: intra-chunk byte-offset embedding + causal 4-byte decoder conv + scaled `byte_d_ff`.
- `use_kalman_precond`: true 2nd-order biorthogonal RLS/Kalman write preconditioning in `Woodbury`.
- `use_latent_feedback`: 1-chunk causal Latent Feedback (`Algorithm 1`, Nov 2026) with zero extra KV cache.
- `use_clean_gates`: removes legacy `-1.8` gate brakes on `Crystal`/`Woodbury` streams.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels.fast_calcite_scans import configure_fast_ternary_qat
from kernels.fast_calcite_swa_chunker import (
    fast_swa_rope_attn_cuda,
    get_cached_rope_cos_sin,
)
from simvq_split.baselines import RMSNorm, SwiGLU, apply_rope
from simvq_split.models import _configure_model_ternary_qat

from .chunker import CalciteV2Chunker
from .crystal import CalciteV2CrystalLayer
from .epistemic_slots import EpistemicSlotMemory
from .woodbury import CalciteV2WoodburyLayer


class LocalSWAAttention(nn.Module):
    """Sliding-Window RoPE Attention with native CUDA custom_op + exact streaming `step()`."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int = 8,
        use_fast_kernels: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.window_size = int(window_size)
        self.use_fast_kernels = bool(use_fast_kernels)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope_for_pos(self, pos: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = float(pos) * self.inv_freq.to(device=device, dtype=torch.float32)
        cos = torch.cos(freqs).to(dtype).view(1, 1, -1)
        sin = torch.sin(freqs).to(dtype).view(1, 1, -1)
        return cos, sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        if (
            self.use_fast_kernels
            and x.is_cuda
            and self.head_dim == 64
            and 1 <= self.window_size <= 16
        ):
            qkv_5d = qkv.view(B, T, 3, self.n_heads, self.head_dim)
            cos, sin = get_cached_rope_cos_sin(
                T, qkv_5d.device, qkv_5d.dtype, head_dim=self.head_dim, base=10000.0
            )
            y = fast_swa_rope_attn_cuda(qkv_5d, cos, sin, self.window_size)
            return self.out_proj(y.view(B, T, C))

        qkv_v = qkv.view(B, T, 3, self.n_heads, self.head_dim)
        q = qkv_v[:, :, 0].transpose(1, 2)
        k = qkv_v[:, :, 1].transpose(1, 2)
        v = qkv_v[:, :, 2].transpose(1, 2)

        t = torch.arange(T, device=x.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device=x.device))
        cos = torch.cos(freqs).to(x.dtype).view(1, 1, T, -1)
        sin = torch.sin(freqs).to(x.dtype).view(1, 1, T, -1)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if T <= self.window_size:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Strict O(T * W) sliding-window attention (zero O(T^2) mask allocation at L=32K–128K!)
            W = self.window_size
            k_pad = F.pad(k, (0, 0, W - 1, 0))
            v_pad = F.pad(v, (0, 0, W - 1, 0))
            k_win = k_pad.unfold(2, W, 1)  # (B, H, T, D, W)
            v_win = v_pad.unfold(2, W, 1)  # (B, H, T, D, W)
            scale = 1.0 / math.sqrt(float(self.head_dim))
            scores = torch.einsum("bhtd,bhtdw->bhtw", q.float(), k_win.float()) * scale
            # Mask left-padded positions for t < W - 1
            t_idx = torch.arange(T, device=x.device).unsqueeze(1)
            w_idx = torch.arange(W, device=x.device).unsqueeze(0)
            valid_w = (t_idx + w_idx) >= (W - 1)  # (T, W)
            scores = scores.masked_fill(~valid_w.view(1, 1, T, W), -1e4)
            probs = F.softmax(scores, dim=-1).to(x.dtype)
            y = torch.einsum("bhtw,bhtdw->bhtd", probs, v_win)
        return self.out_proj(y.transpose(1, 2).contiguous().view(B, T, C))

    @torch.no_grad()
    def step(
        self,
        x_t: torch.Tensor,
        k_ring: torch.Tensor,
        v_ring: torch.Tensor,
        pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Strict O(1) sliding-window attention step over `(B, W, H, D)` rings."""
        B, C = x_t.shape
        W = self.window_size
        qkv = self.qkv(x_t).view(B, 3, self.n_heads, self.head_dim)
        q = qkv[:, 0]
        k = qkv[:, 1]
        v = qkv[:, 2]

        cos, sin = self._rope_for_pos(pos, x_t.device, x_t.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if pos < W:
            k_ring[:, pos] = k
            v_ring[:, pos] = v
            n_valid = pos + 1
            k_use = k_ring[:, :n_valid]
            v_use = v_ring[:, :n_valid]
        else:
            k_ring[:, :-1] = k_ring[:, 1:].clone()
            k_ring[:, -1] = k
            v_ring[:, :-1] = v_ring[:, 1:].clone()
            v_ring[:, -1] = v
            k_use = k_ring
            v_use = v_ring

        # Scaled dot-product attention over `<= W` keys
        scale = 1.0 / math.sqrt(float(self.head_dim))
        scores = torch.einsum("bhd,bwhd->bhw", q.float(), k_use.float()) * scale
        probs = F.softmax(scores, dim=-1).to(x_t.dtype)
        y = torch.einsum("bhw,bwhd->bhd", probs, v_use).reshape(B, C)
        return self.out_proj(y), k_ring, v_ring


class DepthAttnMix(nn.Module):
    """Kimi Attention Residuals (`arXiv:2603.15031`): zero-RAM depth-wise softmax over prior layer states."""

    def __init__(self, d_model: int):
        super().__init__()
        self.k_norm = RMSNorm(d_model)
        self.query = nn.Parameter(torch.randn(d_model, dtype=torch.float32) * 0.02)
        self.mix_logit = nn.Parameter(torch.tensor(-1.5, dtype=torch.float32))

    def forward(self, h_cur: torch.Tensor, hist: list[torch.Tensor]) -> torch.Tensor:
        if len(hist) <= 1:
            return h_cur
        v = torch.stack(hist, dim=-2)
        k = self.k_norm(v)
        scores = torch.einsum("...ld,d->...l", k.float(), self.query.float())
        alpha = F.softmax(scores, dim=-1).to(v.dtype)
        attn_mix = (v * alpha.unsqueeze(-1)).sum(dim=-2)
        g = torch.sigmoid(self.mix_logit.float()).to(h_cur.dtype)
        return (1.0 - g) * h_cur + g * attn_mix


class LocalSWABlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        window_size: int = 8,
        use_fast_kernels: bool = True,
        swa_drop_prob: float = 0.0,
        use_canon: bool = False,
    ):
        super().__init__()
        self.swa_drop_prob = float(max(0.0, min(0.95, swa_drop_prob)))
        self.use_canon = bool(use_canon)
        self.norm1 = RMSNorm(d_model)
        self.attn = LocalSWAAttention(
            d_model, n_heads, window_size=window_size, use_fast_kernels=use_fast_kernels
        )
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)
        if self.use_canon:
            # Allen-Zhu Canon-A (pre-attn) + Canon-C (pre-MLP): 4-tap causal depthwise convs
            self.canon_attn = nn.Conv1d(d_model, d_model, kernel_size=4, groups=d_model, bias=True)
            self.canon_mlp = nn.Conv1d(d_model, d_model, kernel_size=4, groups=d_model, bias=True)
            nn.init.normal_(self.canon_attn.weight, std=0.01)
            nn.init.zeros_(self.canon_attn.bias)
            nn.init.normal_(self.canon_mlp.weight, std=0.01)
            nn.init.zeros_(self.canon_mlp.bias)
        else:
            self.canon_attn = None
            self.canon_mlp = None

    def _apply_attn(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.norm1(x)
        if self.canon_attn is not None:
            x1 = x1 + self.canon_attn(F.pad(x1.transpose(1, 2), (3, 0))).transpose(1, 2)
        attn_out = self.attn(x1)
        if self.training and self.swa_drop_prob > 0.0:
            keep = (
                torch.rand((x.size(0), 1, 1), device=x.device, dtype=torch.float32)
                >= self.swa_drop_prob
            ).to(attn_out.dtype)
            attn_out = attn_out * keep
        return attn_out

    def _apply_mlp(self, x: torch.Tensor) -> torch.Tensor:
        x2 = self.norm2(x)
        if self.canon_mlp is not None:
            x2 = x2 + self.canon_mlp(F.pad(x2.transpose(1, 2), (3, 0))).transpose(1, 2)
        return self.mlp(x2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._apply_attn(x)
        x = x + self._apply_mlp(x)
        return x

    @staticmethod
    def _canon_step(
        x_t: torch.Tensor, conv: nn.Conv1d, ring: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dt = x_t.dtype
        win4 = torch.cat([ring, x_t.unsqueeze(1)], dim=1)  # (B, 4, D)
        ring_next = win4[:, 1:, :].clone()
        w = conv.weight.squeeze(1).to(dt)
        b = conv.bias.to(dt) if conv.bias is not None else None
        out = (win4.transpose(1, 2) * w.unsqueeze(0)).sum(dim=-1)
        if b is not None:
            out = out + b.unsqueeze(0)
        return x_t + out, ring_next

    @torch.no_grad()
    def step(
        self,
        x_t: torch.Tensor,
        k_ring: torch.Tensor,
        v_ring: torch.Tensor,
        pos: int,
        canon_rings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.norm1(x_t)
        if self.canon_attn is not None and canon_rings is not None:
            x1, next_ca = self._canon_step(x1, self.canon_attn, canon_rings[0])
            canon_rings[0].copy_(next_ca)
        attn_out, k_ring, v_ring = self.attn.step(x1, k_ring, v_ring, pos)
        x_t = x_t + attn_out
        x2 = self.norm2(x_t)
        if self.canon_mlp is not None and canon_rings is not None:
            x2, next_cm = self._canon_step(x2, self.canon_mlp, canon_rings[1])
            canon_rings[1].copy_(next_cm)
        x_t = x_t + self.mlp(x2)
        return x_t, k_ring, v_ring


@dataclass
class CalciteV2Config:
    vocab_size: int = 259
    d_model: int = 256
    n_heads: int = 4
    d_ff: int | None = None
    n_layers_pre: int = 2
    n_layers_post: int = 2
    window_size: int = 8
    num_spheres: int = 24
    d_sphere: int = 32
    codebook_size: int = 32
    crystal_w_chunks: int = 2
    mem_heads: int = 5
    mem_dim: int = 32
    ridge_lambda: float = 0.04
    woodbury_latch_span: int = 6
    byte_char_dim: int = 32
    byte_d_ff: int = 256
    byte_router_char_dim: int = 16
    byte_router_hidden: int = 96
    byte_commit_scale: float = 0.45
    router_lock_frac: float = 0.135
    target_chunk_bytes: float = 4.6
    max_chunk_bytes: int = 10
    byte_chunk_mode: str = "utf8"
    max_utf8_bytes: int = 16
    use_static_m_cap: bool = True
    ratio_loss_scale: float = 1.5
    pad_token_id: int = 0
    # Calcite v2 architectural axes (each cleanly ablatable)
    topology: str = "interleaved"          # "hourglass" (v1) | "interleaved" (v2)
    use_scaled_decoder: bool = True        # Upgrade A1: byte_offset_emb + dw_conv + scaled byte_d_ff
    use_kalman_precond: bool = True        # Upgrade C: 2nd-order biorthogonal RLS/Kalman write
    use_latent_feedback: bool = False      # Upgrade D: 1-chunk causal Latent Feedback (Alg. 1)
    latent_feedback_skip_p: float = 0.50   # Probability of 1-pass update during training
    use_clean_gates: bool = True           # Remove legacy -1.8 gate brakes
    use_fast_kernels: bool = True
    # Phase M (Calcite v3) orthogonal ablation axes (all default off for 100% v2 compatibility)
    woodbury_sub_banks: int = 1            # M1: Sparse Routed Woodbury sub-banks per epistemic head (e.g. 4)
    woodbury_top_k_banks: int = 1          # M1: Active sub-banks per write/read (top-1 or top-2)
    use_epistemic_slots: bool = False      # M2: Epistemic L2 Exact Slot Cache driven by q^T P q
    epistemic_slots_n: int = 32            # M2: Number of exact O(1) slots
    epistemic_slots_dim: int = 64          # M2: Key/value dimension per exact slot
    use_sym_byte_swa: bool = False         # M3: Symmetric encoder mixer + 3rd-scale decoder + 6 KB byte-SWA
    byte_swa_window: int = 24              # M3: Sliding byte window size
    byte_swa_dim: int = 64                 # M3: Micro byte-SWA hidden dim
    swa_drop_prob: float = 0.0             # M4: Per-sample SWA Window-Drop probability during training
    # Phase M Stage B (Calcite v3 research radar — all default off for backward compatibility)
    use_attnres: bool = False              # M5: Kimi AttnRes depth-wise softmax routing (+0 RAM)
    use_deltanet2_gates: bool = False      # M6: NVIDIA Gated DeltaNet-2 decoupled erase/write/decay (+0 RAM)
    use_canon_chunks: bool = False         # M7: Allen-Zhu Canon 4-tap convs inside chunk SWA blocks
    use_byte_engram: bool = False          # M8: Pure-tensor hashed 2/3-byte n-gram memory in decoder
    engram_slots: int = 4096               # M8: Hash table slots
    engram_dim: int = 32                   # M8: Engram embedding dim
    use_temporal_residual: bool = False    # TR: Princeton Temporal Residual 2nd carried stream (+1 vec)
    tr_train_passes: int = 4               # TR: Jacobi passes during training
    tr_eval_passes: int = 8                # TR: Jacobi passes during batch eval (exact in streaming)
    tr_backprop_passes: int = 2            # TR: Truncated backprop through last K passes

    def resolved_dff(self) -> int:
        if self.d_ff is not None:
            return int(self.d_ff)
        return int((8 * self.d_model / 3) // 32 * 32) or (2 * self.d_model)


class CalciteV2LM(nn.Module):
    def __init__(self, cfg: CalciteV2Config):
        super().__init__()
        self.cfg = cfg
        self.use_fast_kernels = bool(cfg.use_fast_kernels)
        d_ff = cfg.resolved_dff()

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.tok_emb.weight, std=0.02)

        self.byte_chunker = CalciteV2Chunker(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            d_byte=cfg.byte_char_dim,
            d_rchar=cfg.byte_router_char_dim,
            r_hidden=cfg.byte_router_hidden,
            byte_d_ff=cfg.byte_d_ff,
            target_chunk_bytes=cfg.target_chunk_bytes,
            max_chunk_bytes=cfg.max_chunk_bytes,
            pad_token_id=cfg.pad_token_id,
            byte_chunk_mode=cfg.byte_chunk_mode,
            max_utf8_bytes=cfg.max_utf8_bytes,
            use_static_m_cap=cfg.use_static_m_cap,
            use_scaled_decoder=cfg.use_scaled_decoder,
            use_sym_byte_swa=cfg.use_sym_byte_swa,
            byte_swa_window=cfg.byte_swa_window,
            byte_swa_dim=cfg.byte_swa_dim,
            use_byte_engram=cfg.use_byte_engram,
            engram_slots=cfg.engram_slots,
            engram_dim=cfg.engram_dim,
            use_fast_kernels=cfg.use_fast_kernels,
        )

        # Optional 1-chunk causal Latent Feedback projection
        if cfg.use_latent_feedback:
            self.fb_norm = RMSNorm(cfg.d_model)
            self.fb_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            nn.init.normal_(self.fb_proj.weight, std=0.02)
        else:
            self.fb_norm = None
            self.fb_proj = None

        # Cortex SWA stacks
        n_total_swa = max(2, int(cfg.n_layers_pre + cfg.n_layers_post))
        if cfg.topology == "interleaved":
            # Split SWA evenly across 4 stages: [s0 -> FastWoodbury -> s1 -> Crystal1 -> s2 -> SlowWoodbury -> s3 -> Crystal2]
            n0 = max(1, n_total_swa // 4)
            n1 = max(1, (n_total_swa - n0) // 3)
            n2 = max(1, (n_total_swa - n0 - n1) // 2)
            n3 = max(1, n_total_swa - n0 - n1 - n2)
            self.swa_stages = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            LocalSWABlock(
                                cfg.d_model,
                                cfg.n_heads,
                                d_ff,
                                cfg.window_size,
                                cfg.use_fast_kernels,
                                swa_drop_prob=cfg.swa_drop_prob,
                                use_canon=cfg.use_canon_chunks,
                            )
                            for _ in range(cnt)
                        ]
                    )
                    for cnt in (n0, n1, n2, n3)
                ]
            )
            self.cortex_pre = self.swa_stages[0]
            self.cortex_post = self.swa_stages[3]

            # Fast Woodbury (early prose/syntax) + Slow Epistemic Woodbury (deep facts)
            fast_heads = max(2, cfg.mem_heads // 2)
            self.woodbury_fast_norm = RMSNorm(cfg.d_model)
            self.woodbury_fast = CalciteV2WoodburyLayer(
                d_model=cfg.d_model,
                mem_heads=fast_heads,
                mem_dim=cfg.mem_dim,
                ridge_lambda=cfg.ridge_lambda,
                latch_span=cfg.woodbury_latch_span,
                use_kalman_precond=cfg.use_kalman_precond,
                timescale_mode="fast",
                use_fast_kernels=cfg.use_fast_kernels,
            )
            self.woodbury_fast_gate = nn.Linear(cfg.d_model, 1, bias=True)
            nn.init.constant_(self.woodbury_fast_gate.bias, 0.0 if cfg.use_clean_gates else -0.5)

            self.cryst1_norm = RMSNorm(cfg.d_model)
            self.crystal = CalciteV2CrystalLayer(
                d_model=cfg.d_model,
                num_spheres=cfg.num_spheres,
                d_sphere=cfg.d_sphere,
                codebook_size=cfg.codebook_size,
                w_chunks=cfg.crystal_w_chunks,
                use_fast_kernels=cfg.use_fast_kernels,
            )
            self.cryst1_gate = nn.Linear(cfg.d_model, 1, bias=True)
            nn.init.constant_(self.cryst1_gate.bias, 0.0 if cfg.use_clean_gates else -1.8)

            self.split_norm = RMSNorm(cfg.d_model)
            self.woodbury = CalciteV2WoodburyLayer(
                d_model=cfg.d_model,
                mem_heads=cfg.mem_heads,
                mem_dim=cfg.mem_dim,
                ridge_lambda=cfg.ridge_lambda,
                latch_span=cfg.woodbury_latch_span,
                use_kalman_precond=cfg.use_kalman_precond,
                timescale_mode="slow",
                use_fast_kernels=cfg.use_fast_kernels,
                sub_banks=cfg.woodbury_sub_banks,
                top_k_banks=cfg.woodbury_top_k_banks,
                use_deltanet2_gates=cfg.use_deltanet2_gates,
            )
            self.woodbury_slow_gate = nn.Linear(cfg.d_model, 1, bias=True)
            nn.init.constant_(self.woodbury_slow_gate.bias, 0.3 if cfg.use_clean_gates else 0.4)

            self.mid_mlp_norm = RMSNorm(cfg.d_model)
            self.mid_mlp = SwiGLU(cfg.d_model, d_ff)

            self.cryst2_norm = RMSNorm(cfg.d_model)
            self.crystal_2 = CalciteV2CrystalLayer(
                d_model=cfg.d_model,
                num_spheres=cfg.num_spheres,
                d_sphere=cfg.d_sphere,
                codebook_size=cfg.codebook_size,
                w_chunks=cfg.crystal_w_chunks,
                use_fast_kernels=cfg.use_fast_kernels,
            )
            self.cryst2_gate = nn.Linear(cfg.d_model, 1, bias=True)
            nn.init.constant_(self.cryst2_gate.bias, 0.0 if cfg.use_clean_gates else -1.8)
            self.stream_gate = None
        else:
            # Clean hourglass v1 topology (exact control for ablations)
            self.cortex_pre = nn.ModuleList(
                [
                    LocalSWABlock(
                        cfg.d_model,
                        cfg.n_heads,
                        d_ff,
                        cfg.window_size,
                        cfg.use_fast_kernels,
                        swa_drop_prob=cfg.swa_drop_prob,
                        use_canon=cfg.use_canon_chunks,
                    )
                    for _ in range(cfg.n_layers_pre)
                ]
            )
            self.cortex_post = nn.ModuleList(
                [
                    LocalSWABlock(
                        cfg.d_model,
                        cfg.n_heads,
                        d_ff,
                        cfg.window_size,
                        cfg.use_fast_kernels,
                        swa_drop_prob=cfg.swa_drop_prob,
                        use_canon=cfg.use_canon_chunks,
                    )
                    for _ in range(cfg.n_layers_post)
                ]
            )
            self.swa_stages = None
            self.woodbury_fast = None
            self.woodbury_fast_norm = None
            self.woodbury_fast_gate = None
            self.cryst1_norm = None
            self.cryst1_gate = None
            self.woodbury_slow_gate = None

            self.split_norm = RMSNorm(cfg.d_model)
            self.crystal = CalciteV2CrystalLayer(
                d_model=cfg.d_model,
                num_spheres=cfg.num_spheres,
                d_sphere=cfg.d_sphere,
                codebook_size=cfg.codebook_size,
                w_chunks=cfg.crystal_w_chunks,
                use_fast_kernels=cfg.use_fast_kernels,
            )
            self.woodbury = CalciteV2WoodburyLayer(
                d_model=cfg.d_model,
                mem_heads=cfg.mem_heads,
                mem_dim=cfg.mem_dim,
                ridge_lambda=cfg.ridge_lambda,
                latch_span=cfg.woodbury_latch_span,
                use_kalman_precond=cfg.use_kalman_precond,
                timescale_mode="multiband" if cfg.use_clean_gates else "uniform",
                use_fast_kernels=cfg.use_fast_kernels,
                sub_banks=cfg.woodbury_sub_banks,
                top_k_banks=cfg.woodbury_top_k_banks,
                use_deltanet2_gates=cfg.use_deltanet2_gates,
            )
            self.stream_gate = nn.Linear(cfg.d_model, 2, bias=True)
            with torch.no_grad():
                if cfg.use_clean_gates:
                    self.stream_gate.bias.copy_(torch.tensor([0.0, 0.3]))
                else:
                    self.stream_gate.bias.copy_(torch.tensor([-1.8, 0.4]))

            self.mid_mlp_norm = RMSNorm(cfg.d_model)
            self.mid_mlp = SwiGLU(cfg.d_model, d_ff)
            self.cryst2_norm = RMSNorm(cfg.d_model)
            self.crystal_2 = CalciteV2CrystalLayer(
                d_model=cfg.d_model,
                num_spheres=cfg.num_spheres,
                d_sphere=cfg.d_sphere,
                codebook_size=cfg.codebook_size,
                w_chunks=cfg.crystal_w_chunks,
                use_fast_kernels=cfg.use_fast_kernels,
            )
            self.cryst2_gate = nn.Linear(cfg.d_model, 1, bias=True)
            nn.init.constant_(self.cryst2_gate.bias, 0.0 if cfg.use_clean_gates else -1.8)

        # Optional Epistemic L2 Exact Slot Cache driven by Woodbury's `q^T P q` novelty
        if cfg.use_epistemic_slots:
            self.epistemic_slots = EpistemicSlotMemory(
                d_model=cfg.d_model,
                n_slots=cfg.epistemic_slots_n,
                slot_dim=cfg.epistemic_slots_dim,
            )
            self.epistemic_slots_gate = nn.Linear(cfg.d_model, 1, bias=True)
            nn.init.constant_(self.epistemic_slots_gate.bias, -0.5)
        else:
            self.epistemic_slots = None
            self.epistemic_slots_gate = None

        # Optional Kimi AttnRes depth-wise softmax routing (+0 persistent RAM, excluded from 2-bit LUT)
        if cfg.topology == "interleaved" and (cfg.use_attnres or cfg.use_temporal_residual):
            raise ValueError(
                "use_attnres and use_temporal_residual are supported only with topology='hourglass'"
            )
        if cfg.use_attnres:
            n_mix = 4 + int(cfg.n_layers_post)
            self.attnres_mixers = nn.ModuleList([DepthAttnMix(cfg.d_model) for _ in range(n_mix)])
        else:
            self.attnres_mixers = None

        # Optional Princeton Temporal Residual (TR): 2nd carried stream B (+1 vec O(1) RAM, exact step-0 identity)
        if cfg.use_temporal_residual:
            n_tr = 2 * int(cfg.n_layers_pre) + 3 + 2 * int(cfg.n_layers_post)
            self.tr_rA = nn.Parameter(torch.ones(n_tr, dtype=torch.float32))
            self.tr_rB = nn.Parameter(torch.zeros(n_tr, dtype=torch.float32))
            self.tr_sB = nn.Parameter(torch.ones(n_tr, dtype=torch.float32))
            self.tr_phi = RMSNorm(cfg.d_model)
            with torch.no_grad():
                self.tr_phi.weight.fill_(0.35)
        else:
            self.tr_rA = None
            self.tr_rB = None
            self.tr_sB = None
            self.tr_phi = None

        # Residual init scaling when clean gates are open (keeps Gate-2 init std <= 1.25)
        if cfg.use_clean_gates:
            with torch.no_grad():
                for mod in (self.crystal, self.crystal_2, self.woodbury, self.woodbury_fast):
                    if mod is not None:
                        mod.out_proj.weight.mul_(0.35)

        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self._qat_handles: list = []
        self.ternary_qat_enabled: bool = False
        self.qat_mix: float = 1.0

    def _all_crystals(self) -> tuple[CalciteV2CrystalLayer, ...]:
        return tuple(c for c in (self.crystal, self.crystal_2) if c is not None)

    def _all_woodburys(self) -> tuple[CalciteV2WoodburyLayer, ...]:
        return tuple(w for w in (self.woodbury_fast, self.woodbury) if w is not None)

    def _all_swa_blocks(self) -> list[LocalSWABlock]:
        if self.swa_stages is not None:
            return [blk for stg in self.swa_stages for blk in stg]
        return list(self.cortex_pre) + list(self.cortex_post)

    def set_fast_kernels(self, enabled: bool) -> None:
        flag = bool(enabled)
        self.use_fast_kernels = flag
        self.byte_chunker.use_fast_kernels = flag
        for c in self._all_crystals():
            c.use_fast_kernels = flag
        for w in self._all_woodburys():
            w.use_fast_kernels = flag
        for blk in self._all_swa_blocks():
            blk.attn.use_fast_kernels = flag
        if self.ternary_qat_enabled:
            self.set_ternary_qat(True)

    def set_snap_alpha(self, alpha: float) -> None:
        for c in self._all_crystals():
            c.snap_alpha = float(alpha)

    def set_qat_mix(self, mix: float) -> None:
        m = float(max(0.0, min(1.0, mix)))
        self.qat_mix = m
        act = m if self.ternary_qat_enabled else 0.0
        for w in self._all_woodburys():
            w.qat_active_mix = act
        for c in self._all_crystals():
            c.qat_active_mix = act

    def set_ternary_qat(self, enabled: bool) -> None:
        flag = bool(enabled)
        self.ternary_qat_enabled = flag
        if self.use_fast_kernels:
            configure_fast_ternary_qat(self, flag)
        else:
            configure_fast_ternary_qat(self, False)
            _configure_model_ternary_qat(self, self._qat_handles, flag)
        act = float(self.qat_mix) if flag else 0.0
        for w in self._all_woodburys():
            w.qat_active_mix = act
        for c in self._all_crystals():
            c.qat_active_mix = act

    def set_crystal_schedule(self, step: int, total_steps: int) -> None:
        raw_frac = float(step) / float(max(1, total_steps))
        if self.cfg.router_lock_frac > 0.0 and not self.byte_chunker.router_locked:
            lock_f = float(self.cfg.router_lock_frac)
            if raw_frac < lock_f:
                frac = 0.0
            else:
                self.byte_chunker.lock_router(True)
                frac = (raw_frac - lock_f) / max(1e-4, 1.0 - lock_f)
        else:
            frac = raw_frac

        s_temp = 9.0 if frac <= 0.35 else min(13.0, 9.0 + 4.0 * (frac - 0.35) / 0.30)
        o_temp = 2.0 if frac <= 0.35 else min(5.0, 2.0 + 3.0 * (frac - 0.35) / 0.30)
        g_tau = 1.0 if frac <= 0.40 else min(1.8, 1.0 + 0.8 * (frac - 0.40) / 0.25)
        sw = 0.04 if frac <= 0.35 else min(0.35, 0.04 + 0.31 * (frac - 0.35) / 0.30)
        alpha = 0.0 if frac <= 0.50 else min(1.0, (frac - 0.50) / 0.22)
        for c in self._all_crystals():
            c.slot_temp = s_temp
            c.op_temp = o_temp
            c.gate_temp = g_tau
            c.sharp_weight = sw
        self.set_snap_alpha(alpha)

    def get_commitment_loss(self) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.tok_emb.weight.device)
        c_scale = float(self.cfg.byte_commit_scale) if self.byte_chunker.router_locked else 1.0
        for c in self._all_crystals():
            if c.last_commit_loss is not None:
                loss = loss + c_scale * c.last_commit_loss
        for w in self._all_woodburys():
            if w.last_gate_loss is not None:
                loss = loss + w.last_gate_loss
        if self.byte_chunker.last_ratio_loss is not None:
            loss = loss + float(self.cfg.ratio_loss_scale) * self.byte_chunker.last_ratio_loss
        return loss

    def _forward_core(
        self,
        h: torch.Tensor,
        e_lex: torch.Tensor,
        ablate_crystal: bool = False,
        ablate_woodbury: bool = False,
        tr_b_init: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.topology == "interleaved" and self.swa_stages is not None:
            # Stage 0: Early SWA -> Fast Woodbury
            for blk in self.swa_stages[0]:
                h = blk(h)
            if not ablate_woodbury and self.woodbury_fast is not None and self.woodbury_fast_norm is not None and self.woodbury_fast_gate is not None:
                wf_in = self.woodbury_fast_norm(h)
                wf_g = 2.0 * torch.sigmoid(self.woodbury_fast_gate(wf_in))
                h = h + wf_g * self.woodbury_fast(wf_in, h_lex=e_lex)

            # Stage 1: Mid-low SWA -> Crystal 1
            for blk in self.swa_stages[1]:
                h = blk(h)
            if not ablate_crystal and self.cryst1_norm is not None and self.cryst1_gate is not None:
                c1_in = self.cryst1_norm(h)
                c1_g = 2.0 * torch.sigmoid(self.cryst1_gate(c1_in))
                h = h + c1_g * self.crystal(c1_in, h_lex=e_lex)

            # Stage 2: Mid-high SWA -> Slow Epistemic Woodbury + Mid MLP
            for blk in self.swa_stages[2]:
                h = blk(h)
            if not ablate_woodbury and self.woodbury_slow_gate is not None:
                ws_in = self.split_norm(h)
                ws_g = 2.0 * torch.sigmoid(self.woodbury_slow_gate(ws_in))
                h = h + ws_g * self.woodbury(ws_in, h_lex=e_lex)
                if self.epistemic_slots is not None and self.epistemic_slots_gate is not None:
                    es_g = 2.0 * torch.sigmoid(self.epistemic_slots_gate(ws_in))
                    h = h + es_g * self.epistemic_slots(ws_in, self.woodbury.last_epistemic_uncertainty)
            h = h + self.mid_mlp(self.mid_mlp_norm(h))

            # Stage 3: Crystal 2 -> Top SWA
            if not ablate_crystal:
                c2_in = self.cryst2_norm(h)
                c2_g = 2.0 * torch.sigmoid(self.cryst2_gate(c2_in))
                h = h + c2_g * self.crystal_2(c2_in, h_lex=e_lex)
            for blk in self.swa_stages[3]:
                h = blk(h)
            return (h, tr_b_init) if tr_b_init is not None else h

        # Clean hourglass topology (with optional AttnRes + Temporal Residual)
        use_ar = self.attnres_mixers is not None
        use_tr = (
            tr_b_init is not None
            and self.tr_rA is not None
            and self.tr_rB is not None
            and self.tr_sB is not None
        )
        b_stream = tr_b_init
        tr_idx = 0

        def _tr_step(a_cur: torch.Tensor, b_cur: torch.Tensor | None, fn) -> tuple[torch.Tensor, torch.Tensor | None]:
            nonlocal tr_idx
            if use_tr and b_cur is not None and self.tr_rA is not None and self.tr_rB is not None and self.tr_sB is not None:
                u = self.tr_rA[tr_idx].to(a_cur.dtype) * a_cur + self.tr_rB[tr_idx].to(a_cur.dtype) * b_cur
                y = fn(u)
                a_next = a_cur + y
                b_next = b_cur + self.tr_sB[tr_idx].to(a_cur.dtype) * y
                tr_idx += 1
                return a_next, b_next
            return a_cur + fn(a_cur), b_cur

        hist: list[torch.Tensor] = [h] if use_ar else []
        for blk in self.cortex_pre:
            if use_tr:
                h, b_stream = _tr_step(h, b_stream, blk._apply_attn)
                h, b_stream = _tr_step(h, b_stream, blk._apply_mlp)
            else:
                h = blk(h)
            if use_ar:
                hist.append(h)

        def _long_mem_fn(u_in: torch.Tensor) -> torch.Tensor:
            h_split = self.split_norm(u_in)
            delta_long = torch.zeros_like(u_in)
            if not ablate_crystal and not ablate_woodbury and self.stream_gate is not None:
                gates = 2.0 * torch.sigmoid(self.stream_gate(h_split))
                w_out = self.woodbury(h_split, h_lex=e_lex)
                delta_long = (
                    gates[..., 0:1] * self.crystal(h_split, h_lex=e_lex)
                    + gates[..., 1:2] * w_out
                )
                if self.epistemic_slots is not None and self.epistemic_slots_gate is not None:
                    es_g = 2.0 * torch.sigmoid(self.epistemic_slots_gate(h_split))
                    delta_long = delta_long + es_g * self.epistemic_slots(
                        h_split, self.woodbury.last_epistemic_uncertainty
                    )
            else:
                if not ablate_crystal:
                    delta_long = delta_long + self.crystal(h_split, h_lex=e_lex)
                if not ablate_woodbury:
                    delta_long = delta_long + self.woodbury(h_split, h_lex=e_lex)
                    if self.epistemic_slots is not None and self.epistemic_slots_gate is not None:
                        es_g = 2.0 * torch.sigmoid(self.epistemic_slots_gate(h_split))
                        delta_long = delta_long + es_g * self.epistemic_slots(
                            h_split, self.woodbury.last_epistemic_uncertainty
                        )
            return delta_long

        if use_ar and self.attnres_mixers is not None:
            h_long_in = self.attnres_mixers[0](h, hist)
            if use_tr and b_stream is not None and self.tr_rA is not None and self.tr_rB is not None and self.tr_sB is not None:
                u = self.tr_rA[tr_idx].to(h.dtype) * h_long_in + self.tr_rB[tr_idx].to(h.dtype) * b_stream
                y = _long_mem_fn(u)
                h = h + y
                b_stream = b_stream + self.tr_sB[tr_idx].to(h.dtype) * y
                tr_idx += 1
            else:
                h = h + _long_mem_fn(h_long_in)
            hist.append(h)
        else:
            h, b_stream = _tr_step(h, b_stream, _long_mem_fn)

        def _mid_mlp_fn(u_in: torch.Tensor) -> torch.Tensor:
            return self.mid_mlp(self.mid_mlp_norm(u_in))

        if use_ar and self.attnres_mixers is not None:
            h_mid_in = self.attnres_mixers[1](h, hist)
            if use_tr and b_stream is not None and self.tr_rA is not None and self.tr_rB is not None and self.tr_sB is not None:
                u = self.tr_rA[tr_idx].to(h.dtype) * h_mid_in + self.tr_rB[tr_idx].to(h.dtype) * b_stream
                y = _mid_mlp_fn(u)
                h = h + y
                b_stream = b_stream + self.tr_sB[tr_idx].to(h.dtype) * y
                tr_idx += 1
            else:
                h = h + _mid_mlp_fn(h_mid_in)
            hist.append(h)
        else:
            h, b_stream = _tr_step(h, b_stream, _mid_mlp_fn)

        def _c2_fn(u_in: torch.Tensor) -> torch.Tensor:
            if ablate_crystal:
                return torch.zeros_like(u_in)
            c2_in = self.cryst2_norm(u_in)
            return (2.0 * torch.sigmoid(self.cryst2_gate(c2_in))) * self.crystal_2(c2_in, h_lex=e_lex)

        if use_ar and self.attnres_mixers is not None:
            h_c2_in = self.attnres_mixers[2](h, hist)
            if use_tr and b_stream is not None and self.tr_rA is not None and self.tr_rB is not None and self.tr_sB is not None:
                u = self.tr_rA[tr_idx].to(h.dtype) * h_c2_in + self.tr_rB[tr_idx].to(h.dtype) * b_stream
                y = _c2_fn(u)
                h = h + y
                b_stream = b_stream + self.tr_sB[tr_idx].to(h.dtype) * y
                tr_idx += 1
            else:
                h = h + _c2_fn(h_c2_in)
            hist.append(h)
        else:
            h, b_stream = _tr_step(h, b_stream, _c2_fn)

        for pi, blk in enumerate(self.cortex_post):
            if use_ar and self.attnres_mixers is not None:
                h = self.attnres_mixers[3 + pi](h, hist)
            if use_tr:
                h, b_stream = _tr_step(h, b_stream, blk._apply_attn)
                h, b_stream = _tr_step(h, b_stream, blk._apply_mlp)
            else:
                h = blk(h)
            if use_ar:
                hist.append(h)

        if use_ar and self.attnres_mixers is not None:
            h = self.attnres_mixers[-1](h, hist)
        if tr_b_init is not None:
            assert b_stream is not None
            return h, b_stream
        return h

    def forward(
        self,
        token_ids: torch.Tensor,
        ablate_crystal: bool = False,
        ablate_woodbury: bool = False,
        feedback_passes: int | None = None,
        tr_passes: int | None = None,
        tr_backprop_passes: int | None = None,
    ) -> torch.Tensor:
        e_tok = self.tok_emb(token_ids)
        z_in, h_lex_chunk, ctx = self.byte_chunker.encode_to_chunks(
            e_tok, token_ids, self.tok_emb.weight
        )

        if self.cfg.use_temporal_residual and self.tr_phi is not None:
            n_p = int(
                tr_passes
                if tr_passes is not None
                else (self.cfg.tr_train_passes if self.training else self.cfg.tr_eval_passes)
            )
            n_p = max(1, n_p)
            bp_p = int(
                tr_backprop_passes
                if tr_backprop_passes is not None
                else self.cfg.tr_backprop_passes
            )
            bp_p = max(1, min(n_p, bp_p))
            b_init = torch.zeros_like(z_in)
            z_hat = z_in
            for p in range(n_p):
                need_grad = self.training and (p >= n_p - bp_p)
                if need_grad:
                    z_hat, b_out = self._forward_core(
                        z_in,
                        h_lex_chunk,
                        ablate_crystal=ablate_crystal,
                        ablate_woodbury=ablate_woodbury,
                        tr_b_init=b_init,
                    )
                else:
                    with torch.no_grad():
                        z_hat, b_out = self._forward_core(
                            z_in,
                            h_lex_chunk,
                            ablate_crystal=ablate_crystal,
                            ablate_woodbury=ablate_woodbury,
                            tr_b_init=b_init,
                        )
                if p < n_p - 1:
                    if z_in.size(1) > 1:
                        b_shift = F.pad(b_out[:, :-1], (0, 0, 1, 0))
                        b_init = self.tr_phi(b_shift) if need_grad else self.tr_phi(b_shift).detach()
                    else:
                        b_init = torch.zeros_like(z_in)
            h = self.byte_chunker.decode_from_chunks(z_hat, ctx)
            return self.lm_head(self.norm_f(h))

        if self.cfg.use_latent_feedback and self.fb_norm is not None and self.fb_proj is not None:
            if feedback_passes is not None:
                n_passes = max(1, int(feedback_passes))
            elif self.training:
                n_passes = 2 if bool(torch.rand((), device=z_in.device).item() > self.cfg.latent_feedback_skip_p) else 1
            else:
                n_passes = 2

            z_in_eff = z_in
            if n_passes >= 2 and z_in.size(1) > 1:
                for _ in range(n_passes - 1):
                    with torch.no_grad():
                        z_prev = self._forward_core(
                            z_in_eff,
                            h_lex_chunk,
                            ablate_crystal=ablate_crystal,
                            ablate_woodbury=ablate_woodbury,
                        )
                    z_shift = F.pad(z_prev[:, :-1], (0, 0, 1, 0))
                    z_in_eff = z_in + self.fb_proj(self.fb_norm(z_shift))
        else:
            z_in_eff = z_in

        z_hat = self._forward_core(
            z_in_eff,
            h_lex_chunk,
            ablate_crystal=ablate_crystal,
            ablate_woodbury=ablate_woodbury,
        )
        h = self.byte_chunker.decode_from_chunks(z_hat, ctx)
        return self.lm_head(self.norm_f(h))


def compute_v2_state_cache_kb(cfg: CalciteV2Config, seq_len: int = 2048) -> float:
    """Exact strictly `O(1)` streaming state footprint in KB for `CalciteV2LM`."""
    del seq_len
    n_loc = cfg.n_layers_pre + cfg.n_layers_post
    local_kv = n_loc * 2 * cfg.window_size * cfg.d_model * 2
    if cfg.use_canon_chunks:
        local_kv += n_loc * 2 * 3 * cfg.d_model * 2  # 2x 3-chunk causal Canon rings per SWA block

    h_cryst = 2 if (cfg.num_spheres >= 4 and cfg.num_spheres % 2 == 0) else 1
    cryst_lag = max(0, 2 * int(cfg.crystal_w_chunks) - 1)
    cryst_stage = (
        cfg.num_spheres * cfg.d_sphere * 4
        + (cfg.d_model + 1 + 2 * h_cryst) * 4
        + 4 * h_cryst * 2
        + cryst_lag * (cfg.d_model + h_cryst * cfg.d_sphere) * 2
        + (3 + 1) * cfg.d_model * 2
    )
    cryst_b = 2 * cryst_stage

    def _wood_bytes(mem_heads: int, sub_banks: int = 1) -> int:
        h_epi = max(1, mem_heads - 1)
        h_prose = mem_heads - h_epi
        h_banks = h_epi * max(1, int(sub_banks)) + h_prose
        wood_mat = h_banks * 2 * cfg.mem_dim * cfg.mem_dim * 4
        w_latch_lag = max(0, int(cfg.woodbury_latch_span) - 1)
        wood_ring = (
            (2 * cfg.d_model + 1) * 4
            + (3 * cfg.d_model * 2)
            + (w_latch_lag * (2 * h_epi * cfg.mem_dim + h_epi) * 2)
            + (4 * (cfg.d_model + 2 * mem_heads * cfg.mem_dim) * 2)
        )
        return wood_mat + wood_ring

    wood_b = _wood_bytes(cfg.mem_heads, cfg.woodbury_sub_banks)
    if cfg.topology == "interleaved":
        wood_b += _wood_bytes(max(2, cfg.mem_heads // 2), 1)

    slots_b = (
        int(cfg.epistemic_slots_n * (2 * cfg.epistemic_slots_dim * 4 + 4) + 8)
        if cfg.use_epistemic_slots
        else 0
    )

    chunker_b = (
        7 * cfg.byte_char_dim * 2
        + 3 * cfg.byte_router_char_dim * 2
        + 32 * 2
        + 4
        + 2 * cfg.d_model * 4
        + (4 * 4 if cfg.byte_chunk_mode == "utf8" else 4)
        + (9 * cfg.d_model * 2 if cfg.use_scaled_decoder else 0)
        + (
            (18 * cfg.d_model * 2 + 2 * cfg.byte_swa_window * cfg.byte_swa_dim * 2)
            if cfg.use_sym_byte_swa
            else 0
        )
        + (24 if cfg.use_byte_engram else 0)
    )
    fb_b = cfg.d_model * 2 if cfg.use_latent_feedback else 0
    tr_b = cfg.d_model * 2 if cfg.use_temporal_residual else 0
    return (local_kv + cryst_b + wood_b + slots_b + chunker_b + fb_b + tr_b) / 1024.0


def build_calcite_v2_tier_config(
    tier_name: str,
    topology: str = "interleaved",
    use_scaled_decoder: bool = True,
    use_kalman_precond: bool = True,
    use_latent_feedback: bool = False,
    use_clean_gates: bool = True,
    window_size: int = 8,
    crystal_w_chunks: int = 2,
    woodbury_latch_span: int = 6,
    woodbury_sub_banks: int = 1,
    woodbury_top_k_banks: int = 1,
    use_epistemic_slots: bool = False,
    epistemic_slots_n: int = 32,
    epistemic_slots_dim: int = 64,
    use_sym_byte_swa: bool = False,
    byte_swa_window: int = 24,
    byte_swa_dim: int = 64,
    swa_drop_prob: float = 0.0,
    use_attnres: bool = False,
    use_deltanet2_gates: bool = False,
    use_canon_chunks: bool = False,
    use_byte_engram: bool = False,
    engram_slots: int = 4096,
    engram_dim: int = 32,
    use_temporal_residual: bool = False,
    tr_train_passes: int = 4,
    tr_eval_passes: int = 8,
    tr_backprop_passes: int = 2,
) -> CalciteV2Config:
    """
    Parameter-matched (< 2% vs Phase K tiers) `CalciteV2Config` for `S1_5M`, `S2_25M`, `S3_85M`.
    Scales `byte_char_dim`, `byte_d_ff`, and `num_spheres`/`mem_dim` cleanly with width.
    """
    specs = {
        # (d_model, n_heads, n_pre, n_post, mem_heads, mem_dim, byte_char_dim, byte_d_ff, num_spheres)
        "S1_5M": (256, 4, 2, 2, 5, 32, 32, 256 if use_scaled_decoder else 128, 24),
        "S2_25M": (448, 7, 4, 4, 7, 32, 48 if use_scaled_decoder else 32, 448 if use_scaled_decoder else 128, 24),
        "S3_85M": (768, 12, 5, 5, 8, 48, 64 if use_scaled_decoder else 32, 768 if use_scaled_decoder else 128, 32 if use_scaled_decoder else 24),
    }
    if tier_name not in specs:
        raise ValueError(f"Unknown tier `{tier_name}`: expected {tuple(specs.keys())}")
    d_model, n_heads, n_pre, n_post, mem_heads, mem_dim, b_dim, b_dff, n_sph = specs[tier_name]
    return CalciteV2Config(
        vocab_size=259,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=None,
        n_layers_pre=n_pre,
        n_layers_post=n_post,
        window_size=int(window_size),
        num_spheres=n_sph,
        d_sphere=32,
        codebook_size=32,
        crystal_w_chunks=int(crystal_w_chunks),
        mem_heads=mem_heads,
        mem_dim=mem_dim,
        ridge_lambda=0.04,
        woodbury_latch_span=int(woodbury_latch_span),
        byte_char_dim=b_dim,
        byte_d_ff=b_dff,
        byte_router_char_dim=16,
        byte_router_hidden=96,
        byte_commit_scale=0.45,
        router_lock_frac=0.135,
        target_chunk_bytes=4.6,
        max_chunk_bytes=10,
        byte_chunk_mode="utf8",
        max_utf8_bytes=16,
        use_static_m_cap=True,
        ratio_loss_scale=1.5,
        pad_token_id=0,
        topology=topology,
        use_scaled_decoder=use_scaled_decoder,
        use_kalman_precond=use_kalman_precond,
        use_latent_feedback=use_latent_feedback,
        use_clean_gates=use_clean_gates,
        woodbury_sub_banks=int(woodbury_sub_banks),
        woodbury_top_k_banks=int(woodbury_top_k_banks),
        use_epistemic_slots=bool(use_epistemic_slots),
        epistemic_slots_n=int(epistemic_slots_n),
        epistemic_slots_dim=int(epistemic_slots_dim),
        use_sym_byte_swa=bool(use_sym_byte_swa),
        byte_swa_window=int(byte_swa_window),
        byte_swa_dim=int(byte_swa_dim),
        swa_drop_prob=float(swa_drop_prob),
        use_attnres=bool(use_attnres),
        use_deltanet2_gates=bool(use_deltanet2_gates),
        use_canon_chunks=bool(use_canon_chunks),
        use_byte_engram=bool(use_byte_engram),
        engram_slots=int(engram_slots),
        engram_dim=int(engram_dim),
        use_temporal_residual=bool(use_temporal_residual),
        tr_train_passes=int(tr_train_passes),
        tr_eval_passes=int(tr_eval_passes),
        tr_backprop_passes=int(tr_backprop_passes),
    )
