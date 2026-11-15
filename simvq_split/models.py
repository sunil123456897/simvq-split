"""
SimVQ-Split Language Model (`SimVQSplitLM`).

Orthogonal 3-Stream Post-Transformer Architecture:
- Stream A (`Continuous Cortex`): Local Sliding-Window RoPE Attention (`W = 32`, O(1) KV-cache)
  + SwiGLU MLPs. Preserves continuous natural-language semantics and perplexity without
  any global attention shortcut.
- Stream B (`Voronoi Crystal`): Factored G-sphere SimVQ Voronoi attractor state machine
  with Householder Rotation-Trick STE (`0.000000` norm/phase drift across arbitrary length).
- Stream C (`Neural Woodbury Memory`): Self-addressed Second-Order Sherman-Morrison-Woodbury
  dual-key memory eliminating correlated-key overwrite interference.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from .baselines import RMSNorm, SwiGLU, TransformerBlock
from .memory import (
    NeuralWoodburyLayer,
    VoronoiCrystalLayer,
    quantize_ternary_ste,
)


def _make_ternary_qat_hook(parent_model: nn.Module):
    def _ternary_qat_forward_hook(
        mod: nn.Module, inputs: tuple, _output: torch.Tensor
    ) -> torch.Tensor:
        x = inputs[0]
        mix = float(getattr(parent_model, "qat_mix", 1.0))
        w_tern = quantize_ternary_ste(mod.weight)
        if mix < 0.9999:
            w_q = ((1.0 - mix) * mod.weight + mix * w_tern).to(x.dtype)
        else:
            w_q = w_tern.to(x.dtype)
        b = mod.bias.to(x.dtype) if mod.bias is not None else None
        return F.linear(x, w_q, b)

    return _ternary_qat_forward_hook


def _configure_model_ternary_qat(
    model: nn.Module, handles: list, enabled: bool
) -> None:
    for h in handles:
        h.remove()
    handles.clear()
    if enabled:
        hook_fn = _make_ternary_qat_hook(model)
        for _, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and mod is not model.lm_head:
                handles.append(mod.register_forward_hook(hook_fn))


@dataclass
class SimVQSplitConfig:
    vocab_size: int = 2048
    d_model: int = 256
    n_heads: int = 8
    d_ff: int | None = None
    n_layers_pre: int = 2
    n_layers_post: int = 1
    window_size: int = 32
    num_spheres: int = 24
    d_sphere: int = 32
    codebook_size: int = 32
    mem_heads: int = 5
    mem_dim: int = 32
    ridge_lambda: float = 0.04
    use_crystal: bool = True
    use_woodbury: bool = True

    def resolved_dff(self) -> int:
        if self.d_ff is not None:
            return self.d_ff
        return int((8 * self.d_model / 3) // 32 * 32) or (2 * self.d_model)


class SimVQSplitLM(nn.Module):
    def __init__(self, cfg: SimVQSplitConfig):
        super().__init__()
        self.cfg = cfg
        d_ff = cfg.resolved_dff()

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.tok_emb.weight, std=0.02)

        # Lower Continuous Cortex: strictly local window W
        self.cortex_pre = nn.ModuleList(
            [
                TransformerBlock(
                    cfg.d_model,
                    cfg.n_heads,
                    d_ff,
                    window_size=cfg.window_size,
                )
                for _ in range(cfg.n_layers_pre)
            ]
        )

        # Mid-network Orthogonal Split Bottleneck (O(1) state, unbounded horizon)
        self.split_norm = RMSNorm(cfg.d_model)
        self.crystal = (
            VoronoiCrystalLayer(
                d_model=cfg.d_model,
                num_spheres=cfg.num_spheres,
                d_sphere=cfg.d_sphere,
                codebook_size=cfg.codebook_size,
            )
            if cfg.use_crystal
            else None
        )
        self.woodbury = (
            NeuralWoodburyLayer(
                d_model=cfg.d_model,
                mem_heads=cfg.mem_heads,
                mem_dim=cfg.mem_dim,
                ridge_lambda=cfg.ridge_lambda,
            )
            if cfg.use_woodbury
            else None
        )
        if cfg.use_crystal and cfg.use_woodbury:
            self.stream_gate = nn.Linear(cfg.d_model, 2, bias=True)
            with torch.no_grad():
                self.stream_gate.bias.copy_(torch.tensor([-1.8, 0.4]))
        elif cfg.use_crystal:
            self.stream_gate = nn.Linear(cfg.d_model, 1, bias=True)
            nn.init.constant_(self.stream_gate.bias, -2.2)
        else:
            self.stream_gate = None

        self.mid_mlp_norm = RMSNorm(cfg.d_model)
        self.mid_mlp = SwiGLU(cfg.d_model, d_ff)

        # Second-stage Bijective Operator Crystal for non-abelian multi-step composition parity
        self.cryst2_norm = RMSNorm(cfg.d_model) if cfg.use_crystal else None
        self.crystal_2 = (
            VoronoiCrystalLayer(
                d_model=cfg.d_model,
                num_spheres=cfg.num_spheres,
                d_sphere=cfg.d_sphere,
                codebook_size=cfg.codebook_size,
            )
            if cfg.use_crystal
            else None
        )
        self.cryst2_gate = (
            nn.Linear(cfg.d_model, 1, bias=True) if cfg.use_crystal else None
        )
        if self.cryst2_gate is not None:
            nn.init.constant_(
                self.cryst2_gate.bias, -1.8 if cfg.use_woodbury else -2.2
            )

        # Upper Continuous Cortex: strictly local window W composing retrieved state into fluent prose
        self.cortex_post = nn.ModuleList(
            [
                TransformerBlock(
                    cfg.d_model,
                    cfg.n_heads,
                    d_ff,
                    window_size=cfg.window_size,
                )
                for _ in range(cfg.n_layers_post)
            ]
        )

        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self._qat_handles: list = []
        self.ternary_qat_enabled: bool = False
        self.qat_mix: float = 1.0

    def set_snap_alpha(self, alpha: float) -> None:
        if self.crystal is not None:
            self.crystal.snap_alpha = float(alpha)
        if self.crystal_2 is not None:
            self.crystal_2.snap_alpha = float(alpha)

    def set_qat_mix(self, mix: float) -> None:
        m = float(max(0.0, min(1.0, mix)))
        self.qat_mix = m
        act = m if self.ternary_qat_enabled else 0.0
        if self.woodbury is not None:
            self.woodbury.qat_active_mix = act
        if self.crystal is not None:
            self.crystal.qat_active_mix = act
        if self.crystal_2 is not None:
            self.crystal_2.qat_active_mix = act

    def set_crystal_schedule(self, step: int, total_steps: int) -> None:
        frac = float(step) / float(max(1, total_steps))
        s_temp = 9.0 if frac <= 0.35 else min(13.0, 9.0 + 4.0 * (frac - 0.35) / 0.30)
        o_temp = 2.0 if frac <= 0.35 else min(5.0, 2.0 + 3.0 * (frac - 0.35) / 0.30)
        g_tau = 1.0 if frac <= 0.40 else min(1.8, 1.0 + 0.8 * (frac - 0.40) / 0.25)
        sw = 0.04 if frac <= 0.35 else min(0.35, 0.04 + 0.31 * (frac - 0.35) / 0.30)
        alpha = 0.0 if frac <= 0.50 else min(1.0, (frac - 0.50) / 0.22)
        for cryst in (self.crystal, self.crystal_2):
            if cryst is not None:
                cryst.slot_temp = s_temp
                cryst.op_temp = o_temp
                cryst.gate_temp = g_tau
                cryst.sharp_weight = sw
        self.set_snap_alpha(alpha)

    def set_ternary_qat(self, enabled: bool) -> None:
        flag = bool(enabled)
        self.ternary_qat_enabled = flag
        _configure_model_ternary_qat(self, self._qat_handles, flag)
        act = float(self.qat_mix) if flag else 0.0
        if self.woodbury is not None:
            self.woodbury.qat_active_mix = act
        if self.crystal is not None:
            self.crystal.qat_active_mix = act
        if self.crystal_2 is not None:
            self.crystal_2.qat_active_mix = act

    def get_commitment_loss(self) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.tok_emb.weight.device)
        if self.crystal is not None and self.crystal.last_commit_loss is not None:
            loss = loss + self.crystal.last_commit_loss
        if self.crystal_2 is not None and self.crystal_2.last_commit_loss is not None:
            loss = loss + self.crystal_2.last_commit_loss
        if self.woodbury is not None and self.woodbury.last_gate_loss is not None:
            loss = loss + self.woodbury.last_gate_loss
        return loss

    def forward(
        self,
        token_ids: torch.Tensor,
        ablate_crystal: bool = False,
        ablate_woodbury: bool = False,
    ) -> torch.Tensor:
        e_tok = self.tok_emb(token_ids)
        h = e_tok
        for blk in self.cortex_pre:
            h = blk(h)

        h_split = self.split_norm(h)
        delta_long = torch.zeros_like(h)
        if (
            self.stream_gate is not None
            and self.crystal is not None
            and self.woodbury is not None
            and not ablate_crystal
            and not ablate_woodbury
        ):
            gates = 2.0 * torch.sigmoid(self.stream_gate(h_split))
            c_out = self.crystal(h_split, h_lex=e_tok)
            w_out = self.woodbury(h_split, h_lex=e_tok)
            delta_long = gates[..., 0:1] * c_out + gates[..., 1:2] * w_out
        else:
            if self.crystal is not None and not ablate_crystal:
                c_out = self.crystal(h_split, h_lex=e_tok)
                if self.stream_gate is not None and self.woodbury is None:
                    c_out = (2.0 * torch.sigmoid(self.stream_gate(h_split))) * c_out
                delta_long = delta_long + c_out
            if self.woodbury is not None and not ablate_woodbury:
                delta_long = delta_long + self.woodbury(h_split, h_lex=e_tok)

        h = h + delta_long
        h = h + self.mid_mlp(self.mid_mlp_norm(h))
        if (
            self.crystal_2 is not None
            and self.cryst2_norm is not None
            and not ablate_crystal
        ):
            c2_in = self.cryst2_norm(h)
            c2_out = self.crystal_2(c2_in, h_lex=e_tok)
            if self.cryst2_gate is not None and (
                self.woodbury is None or not ablate_woodbury
            ):
                c2_out = (2.0 * torch.sigmoid(self.cryst2_gate(c2_in))) * c2_out
            h = h + c2_out

        for blk in self.cortex_post:
            h = blk(h)

        return self.lm_head(self.norm_f(h))
