"""
Hybrid `Muon + AdamW` Optimizer (`scripts/muon_opt.py`) for `Calcite v3`.

Applies Keller Jordan's 5-step BF16 Newton-Schulz orthogonalized momentum (`Muon`)
to hidden 2D weight matrices (`SWA`, `SwiGLU`, `chunker` Linear layers) while keeping
`AdamW(fused=True)` on `Woodbury` (`wd=0.0, 0.35x LR`), embeddings, `lm_head`, 1D norms/biases,
and small matrices — preserving all `Calcite` stability invariants.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from kernels.fast_calcite_scans import clear_fast_qat_cache


# Precomputed minimax odd-quintic Polar Express coefficients (Amsel et al. 2025, arXiv:2505.16932)
# solved via exact Remez exchange with BF16 safety factor `1.01` on intermediate steps so every
# intermediate singular value stays strictly `<= 1.0` in BF16 without overshoot/divergence:
#   - `quintic5`: original Keller Jordan / Moonlight 5-step fixed polynomial (`norm_scale=1.0`).
#   - `polar4`: 4-step Remez minimax (`l0=0.005, safety=1.01, norm_scale=1.01`), strictly tighter
#               than `quintic5` across random & power-law decaying spectra in 20% fewer matmuls.
#   - `polar3`: 3-step Remez minimax (`l0=0.02, safety=1.01, norm_scale=1.01`), 40% fewer matmuls.
_QUINTIC5_COEFFS: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315)

POLAR_EXPRESS_SCHEDULES: Dict[str, Tuple[float, Tuple[Tuple[float, float, float], ...]]] = {
    "quintic5": (1.0, (_QUINTIC5_COEFFS,) * 5),
    "polar4": (
        1.01,
        (
            (4.11994, -11.73209, 8.60225),
            (4.16114, -12.08758, 9.04105),
            (3.90953, -10.13406, 7.31008),
            (3.33717, -5.50456, 3.28451),
        ),
    ),
    "polar3": (
        1.01,
        (
            (4.11994, -11.73209, 8.60225),
            (3.92288, -10.23199, 7.39609),
            (3.39683, -5.71430, 3.44404),
        ),
    ),
}

VALID_NS_SCHEDULES: Tuple[str, ...] = tuple(POLAR_EXPRESS_SCHEDULES.keys())


def _resolve_ns_schedule(
    ns_schedule: str, steps: int
) -> Tuple[float, Tuple[Tuple[float, float, float], ...]]:
    if ns_schedule not in POLAR_EXPRESS_SCHEDULES:
        raise ValueError(
            f"Unknown ns_schedule={ns_schedule!r}; expected one of {VALID_NS_SCHEDULES}"
        )
    if ns_schedule == "quintic5":
        return 1.0, (_QUINTIC5_COEFFS,) * int(steps)
    return POLAR_EXPRESS_SCHEDULES[ns_schedule]


class _HookHandle:
    """Removable handle returned by `MuonAdamWHybrid.register_step_post_hook`."""

    def __init__(self, hooks: Dict[int, Callable], hook_id: int):
        self._hooks = hooks
        self._hook_id = hook_id

    def remove(self) -> None:
        self._hooks.pop(self._hook_id, None)


@torch.no_grad()
def zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    ns_schedule: str = "quintic5",
) -> torch.Tensor:
    """Quintic Newton-Schulz / Polar Express orthogonalization (`quintic5` | `polar4` | `polar3`)."""
    assert G.ndim == 2
    norm_scale, coeffs = _resolve_ns_schedule(ns_schedule, steps)
    X = G.bfloat16() if G.is_cuda else G.float()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    denom = (X.norm() * norm_scale + eps) if norm_scale != 1.0 else (X.norm() + eps)
    X = X / denom
    for a, b, c in coeffs:
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


@torch.no_grad()
def zeropower_via_newtonschulz5_batched(
    G_stack: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    ns_schedule: str = "quintic5",
) -> torch.Tensor:
    """Batched quintic Newton-Schulz / Polar Express orthogonalization via `torch.bmm`."""
    assert G_stack.ndim == 3
    norm_scale, coeffs = _resolve_ns_schedule(ns_schedule, steps)
    X = G_stack.bfloat16() if G_stack.is_cuda else G_stack.float()
    transposed = X.size(1) > X.size(2)
    if transposed:
        X = X.transpose(1, 2).contiguous()
    norms = X.norm(dim=(1, 2), keepdim=True)
    denom = (norms * norm_scale + eps) if norm_scale != 1.0 else (norms + eps)
    X = X / denom
    for a, b, c in coeffs:
        A = torch.bmm(X, X.transpose(1, 2))
        B = b * A + c * torch.bmm(A, A)
        X = a * X + torch.bmm(B, X)
    if transposed:
        X = X.transpose(1, 2)
    return X


class MuonAdamWHybrid:
    """
    Drop-in optimizer wrapper combining shape-bucketed batched `Muon` (2D hidden matrices,
    supporting `quintic5` default and Polar Express `polar4`/`polar3` minimax schedules)
    and `AdamW` (embeddings, `Woodbury`, 1D params). Exposes `param_groups`, `trainable_params`,
    `register_step_post_hook`, `zero_grad`, `step`, `state_dict`, and `load_state_dict`.
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
        ns_schedule: str = "quintic5",
    ):
        if ns_schedule not in POLAR_EXPRESS_SCHEDULES:
            raise ValueError(
                f"Unknown ns_schedule={ns_schedule!r}; expected one of {VALID_NS_SCHEDULES}"
            )
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
        self.ns_schedule = str(ns_schedule)
        self.muon_bufs: dict[int, torch.Tensor] = {}
        self._step_post_hooks: Dict[int, Callable] = {}
        self._next_hook_id: int = 0

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
        self.param_groups: List[dict] = []
        self.trainable_params: List[nn.Parameter] = []
        self._sync_param_groups()
        self._muon_buckets: List[Tuple[int, int, List[Tuple[nn.Parameter, bool]]]] = []
        self._rebuild_muon_buckets()

    def _sync_param_groups(self) -> None:
        """Re-binds `self.param_groups` and `self.trainable_params` to live `adamw`/`muon_group` dicts."""
        live_groups: List[dict] = []
        if self.adamw is not None:
            live_groups.extend(self.adamw.param_groups)
        if self.muon_group is not None:
            live_groups.append(self.muon_group)
        self.param_groups[:] = live_groups
        self.trainable_params = [p for g in self.param_groups for p in g["params"]]

    def _rebuild_muon_buckets(self) -> None:
        """Groups 2D Muon parameters by canonical `(min(r, c), max(r, c), dtype, device)`."""
        self._muon_buckets = []
        if self.muon_group is None:
            return
        buckets: Dict[Tuple[int, int, torch.dtype, torch.device], List[Tuple[nn.Parameter, bool]]] = {}
        for p in self.muon_group["params"]:
            r, c = int(p.size(0)), int(p.size(1))
            r_s, c_s = (c, r) if r > c else (r, c)
            key = (r_s, c_s, p.dtype, p.device)
            buckets.setdefault(key, []).append((p, r > c))
        # Largest buckets first for better GPU occupancy.
        for (r_s, c_s, _dt, _dev), items in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
            self._muon_buckets.append((r_s, c_s, items))

    def register_step_post_hook(self, hook: Callable[["MuonAdamWHybrid"], None]) -> _HookHandle:
        """Registers a callback executed strictly AFTER both `AdamW` and `Muon` updates."""
        hid = self._next_hook_id
        self._next_hook_id += 1
        self._step_post_hooks[hid] = hook
        return _HookHandle(self._step_post_hooks, hid)

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=set_to_none)
        if self.muon_group is not None:
            if set_to_none:
                for p in self.muon_group["params"]:
                    p.grad = None
            else:
                grads = [p.grad for p in self.muon_group["params"] if p.grad is not None]
                if grads:
                    torch._foreach_zero_(grads)

    @torch.no_grad()
    def step(self) -> None:
        if self.adamw is not None:
            self.adamw.step()
        if self.muon_group is not None:
            lr = float(self.muon_group["lr"])
            wd = float(self.muon_group["weight_decay"])
            beta = self.muon_momentum
            # Guard against external mutation of `muon_group["params"]` or device transfers.
            n_cached = sum(len(items) for _, _, items in self._muon_buckets)
            if n_cached != len(self.muon_group["params"]):
                self._rebuild_muon_buckets()

            # Group active tensors by `(dtype, device)` for fused `_foreach_*` momentum/WD.
            dev_groups: Dict[
                Tuple[torch.dtype, torch.device],
                Tuple[List[nn.Parameter], List[torch.Tensor], List[torch.Tensor]],
            ] = {}
            bucket_active: List[Tuple[int, List[Tuple[nn.Parameter, bool, torch.Tensor]]]] = []

            for _r_s, c_s, items in self._muon_buckets:
                active_items: List[Tuple[nn.Parameter, bool, torch.Tensor]] = []
                for p, is_tr in items:
                    g = p.grad
                    if g is None:
                        continue
                    pid = id(p)
                    buf = self.muon_bufs.get(pid)
                    if buf is None:
                        buf = torch.zeros_like(g)
                        self.muon_bufs[pid] = buf
                    key = (p.dtype, p.device)
                    dg = dev_groups.get(key)
                    if dg is None:
                        dg = ([], [], [])
                        dev_groups[key] = dg
                    dg[0].append(p)
                    dg[1].append(g)
                    dg[2].append(buf)
                    active_items.append((p, is_tr, g))
                if active_items:
                    bucket_active.append((c_s, active_items))

            # Vectorized momentum + Nesterov lookahead + decoupled weight decay.
            g_nest_map: Dict[int, torch.Tensor] = {}
            wd_factor = 1.0 - lr * wd
            for _key, (p_list, g_list, b_list) in dev_groups.items():
                torch._foreach_mul_(b_list, beta)
                torch._foreach_add_(b_list, g_list)
                nests = torch._foreach_add(g_list, b_list, alpha=beta)
                if wd > 0.0:
                    torch._foreach_mul_(p_list, wd_factor)
                for p_it, gn_it in zip(p_list, nests):
                    g_nest_map[id(p_it)] = gn_it

            # Shape-bucketed batched `torch.bmm` Newton-Schulz / Polar Express + fused parameter update.
            for c_s, active_items in bucket_active:
                # Moonlight aspect-ratio RMS scaling: 0.2 * sqrt(max(rows, cols))
                alpha = -lr * 0.2 * math.sqrt(float(c_s))
                if len(active_items) == 1:
                    p0, _tr0, _g0 = active_items[0]
                    orth = zeropower_via_newtonschulz5(
                        g_nest_map[id(p0)],
                        steps=self.ns_steps,
                        ns_schedule=self.ns_schedule,
                    ).to(p0.dtype)
                    p0.add_(orth, alpha=alpha)
                else:
                    p_list = [p for p, _tr, _g in active_items]
                    tr_list = [tr for _p, tr, _g in active_items]
                    stack_in = torch.stack(
                        [
                            g_nest_map[id(p)].T if tr else g_nest_map[id(p)]
                            for p, tr in zip(p_list, tr_list)
                        ],
                        dim=0,
                    )
                    orth_stack = zeropower_via_newtonschulz5_batched(
                        stack_in,
                        steps=self.ns_steps,
                        ns_schedule=self.ns_schedule,
                    ).to(p_list[0].dtype)
                    orth_list = [
                        (o.T if tr else o) for o, tr in zip(orth_stack.unbind(0), tr_list)
                    ]
                    torch._foreach_add_(p_list, orth_list, alpha=alpha)

        # Gate 3.5 invariant: invalidate FastTernaryQAT cache strictly AFTER both AdamW and Muon mutate weights.
        clear_fast_qat_cache()
        for hook in list(self._step_post_hooks.values()):
            hook(self)

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
            "ns_schedule": self.ns_schedule,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if self.adamw is not None and state_dict.get("adamw") is not None:
            prev_meta = [
                (float(g.get("lr_mult", 1.0)), bool(g.get("is_muon", False)))
                for g in self.adamw.param_groups
            ]
            self.adamw.load_state_dict(state_dict["adamw"])
            for g, (lr_m, is_m) in zip(self.adamw.param_groups, prev_meta):
                g.setdefault("lr_mult", lr_m)
                g.setdefault("is_muon", is_m)
        self.muon_momentum = float(state_dict.get("muon_momentum", self.muon_momentum))
        self.ns_steps = int(state_dict.get("ns_steps", self.ns_steps))
        saved_sched = str(state_dict.get("ns_schedule", self.ns_schedule))
        if saved_sched not in POLAR_EXPRESS_SCHEDULES:
            raise ValueError(
                f"Unknown ns_schedule={saved_sched!r} in state_dict; expected one of {VALID_NS_SCHEDULES}"
            )
        self.ns_schedule = saved_sched
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
        # `adamw.load_state_dict()` replaces `adamw.param_groups` with fresh dicts; re-bind!
        self._sync_param_groups()
        self._rebuild_muon_buckets()
