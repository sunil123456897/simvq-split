"""
Hybrid `Muon + AdamW` Optimizer (`scripts/muon_opt.py`) for `Calcite v3`.

Applies Keller Jordan's 5-step BF16 Newton-Schulz orthogonalized momentum (`Muon`)
to hidden 2D weight matrices (`SWA`, `SwiGLU`, `chunker` Linear layers) while keeping
`AdamW(fused=True)` on `Woodbury` (`wd=0.0, 0.35x LR`), embeddings, `lm_head`, 1D norms/biases,
and small matrices — preserving all `Calcite` stability invariants.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional

import torch
import torch.nn as nn

from kernels.fast_calcite_scans import clear_fast_qat_cache


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """5-step quintic Newton-Schulz orthogonalization (Keller Jordan / Moonlight)."""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.is_cuda else G.float()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class MuonAdamWHybrid:
    """
    Drop-in optimizer wrapper combining `Muon` (2D hidden matrices) and `AdamW`
    (embeddings, `Woodbury`, 1D params). Exposes `param_groups`, `zero_grad`, `step`,
    `state_dict`, and `load_state_dict`.
    """

    def __init__(
        self,
        model: nn.Module,
        base_lr: float = 1.8e-3,
        muon_lr_mult: float = 6.0,
        weight_decay: float = 0.04,
        momentum: float = 0.95,
        ns_steps: int = 5,
        device: str = "cuda",
        woodbury_lr_mult: float = 0.35,
        protect_crystal: bool = False,
        crystal_lr_mult: float = 0.05,
    ):
        muon_p: List[nn.Parameter] = []
        decay_p: List[nn.Parameter] = []
        no_decay_p: List[nn.Parameter] = []
        wood_p: List[nn.Parameter] = []
        cryst_p: List[nn.Parameter] = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if protect_crystal and (name.startswith("crystal.") or name.startswith("crystal_2.") or ".crystal." in name):
                cryst_p.append(p)
            elif "woodbury" in name:
                wood_p.append(p)
            elif "epistemic_slots" in name or "byte_induct_cache" in name:
                no_decay_p.append(p)
            elif p.ndim < 2 or "norm" in name or "bias" in name or "emb" in name or "lm_head" in name or "router_" in name:
                no_decay_p.append(p)
            elif p.ndim == 2 and min(p.shape) >= 32:
                muon_p.append(p)
            else:
                decay_p.append(p)

        self.muon_momentum = float(momentum)
        self.ns_steps = int(ns_steps)
        self.muon_bufs: dict[int, torch.Tensor] = {}

        adam_groups = []
        if decay_p:
            adam_groups.append({"params": decay_p, "lr": base_lr, "weight_decay": weight_decay, "lr_mult": 1.0, "is_muon": False})
        if no_decay_p:
            adam_groups.append({"params": no_decay_p, "lr": base_lr, "weight_decay": 0.0, "lr_mult": 1.0, "is_muon": False})
        if wood_p:
            adam_groups.append({"params": wood_p, "lr": base_lr * woodbury_lr_mult, "weight_decay": 0.0, "lr_mult": float(woodbury_lr_mult), "is_muon": False})
        if cryst_p:
            adam_groups.append({"params": cryst_p, "lr": base_lr * crystal_lr_mult, "weight_decay": 0.0, "lr_mult": float(crystal_lr_mult), "is_muon": False})

        use_fused = device == "cuda" and torch.cuda.is_available()
        self.adamw: Optional[torch.optim.AdamW] = (
            torch.optim.AdamW(adam_groups, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
            if adam_groups
            else None
        )
        self.muon_group = (
            {
                "params": muon_p,
                "lr": base_lr * muon_lr_mult,
                "weight_decay": weight_decay,
                "lr_mult": float(muon_lr_mult),
                "is_muon": True,
            }
            if muon_p
            else None
        )
        self.param_groups = []
        if self.adamw is not None:
            self.param_groups.extend(self.adamw.param_groups)
        if self.muon_group is not None:
            self.param_groups.append(self.muon_group)

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=set_to_none)
        if self.muon_group is not None:
            for p in self.muon_group["params"]:
                if set_to_none:
                    p.grad = None
                elif p.grad is not None:
                    p.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        if self.adamw is not None:
            self.adamw.step()
        if self.muon_group is not None:
            lr = float(self.muon_group["lr"])
            wd = float(self.muon_group["weight_decay"])
            beta = self.muon_momentum
            for p in self.muon_group["params"]:
                g = p.grad
                if g is None:
                    continue
                pid = id(p)
                buf = self.muon_bufs.get(pid)
                if buf is None:
                    buf = torch.zeros_like(g)
                    self.muon_bufs[pid] = buf
                buf.mul_(beta).add_(g)
                # Nesterov lookahead momentum before Newton-Schulz
                g_nest = g.add(buf, alpha=beta)
                orth = zeropower_via_newtonschulz5(g_nest, steps=self.ns_steps).to(p.dtype)
                # Moonlight aspect-ratio RMS scaling: 0.2 * sqrt(max(rows, cols))
                scale = 0.2 * math.sqrt(float(max(p.size(0), p.size(1))))
                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(orth, alpha=-lr * scale)
        clear_fast_qat_cache()

    def state_dict(self) -> dict:
        muon_params = self.muon_group["params"] if self.muon_group is not None else []
        muon_state = {
            idx: self.muon_bufs[id(p)].detach().clone()
            for idx, p in enumerate(muon_params)
            if id(p) in self.muon_bufs
        }
        muon_hp = (
            {k: v for k, v in self.muon_group.items() if k != "params"}
            if self.muon_group is not None
            else None
        )
        return {
            "adamw": self.adamw.state_dict() if self.adamw is not None else None,
            "muon_bufs": muon_state,
            "muon_group": muon_hp,
            "muon_momentum": self.muon_momentum,
            "ns_steps": self.ns_steps,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if self.adamw is not None and state_dict.get("adamw") is not None:
            self.adamw.load_state_dict(state_dict["adamw"])
        self.muon_momentum = float(state_dict.get("muon_momentum", self.muon_momentum))
        self.ns_steps = int(state_dict.get("ns_steps", self.ns_steps))
        self.muon_bufs.clear()
        if self.muon_group is not None:
            saved_hp = state_dict.get("muon_group")
            if isinstance(saved_hp, dict):
                for k, v in saved_hp.items():
                    if k != "params":
                        self.muon_group[k] = v
            saved_bufs = state_dict.get("muon_bufs") or {}
            for idx, p in enumerate(self.muon_group["params"]):
                if idx in saved_bufs:
                    self.muon_bufs[id(p)] = saved_bufs[idx].to(device=p.device, dtype=p.dtype).clone()
