"""
Experiment Velocity Engine (`scripts/experiment_velocity.py`) for `Calcite v3.1`.

Deep, zero-bloat workflow primitives enabling 5x–15x faster scientific iteration without
relaxing any `Calcite` invariant:
1. `WSDLRSchedule` + `save_plateau_checkpoint` / `load_plateau_checkpoint`:
   Warmup -> Stable Plateau (checkpointed with model, optimizer, RNG, config, and Law-8a
   Stage-1A `ByteInductionCache` isolation check) -> Decay (`cosine` or `1-sqrt`).
2. `run_tiered_eval(model, tier=0|1|2, ...)`:
   - `Tier-0` (~5–10s fast screen): `Val(8b)`, `OW6(16w/dom)`, `S5_k2(32)`, `OV_L(32)`, `Needle_1500(8)`.
   - `Tier-1` (~25–40s promoted check): `Val(32b)`, `L2048`, `OW6(full)`, `S5/OV_L/Trap_L(60)`, `Needle L512..8K(8)`.
   - `Tier-2`: full closure battery (`evaluate_full_model` + `L=32K` + full Needle & Overwrite ladder).
3. `OEBEarlyStopper`:
   Mid-run statistical early-kill gate against a control `Val BPB` curve with explicit
   `disabled_for_axis` guard for pure cognitive/needle axes.
4. `configure_colab_fast_cache(drive_cache_dir)`:
   Sets persistent `TORCH_EXTENSIONS_DIR`, `TORCHINDUCTOR_CACHE_DIR`, and `TRITON_CACHE_DIR`
   to avoid cold JIT/Inductor rebuilds across Colab sessions.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from kernels.fast_calcite_scans import unwrap_compiled_state_dict


# ---------------------------------------------------------------------------
# 1. Warmup-Stable-Decay (WSD) LR Schedule & Plateau Checkpoint Manager
# ---------------------------------------------------------------------------


class WSDLRSchedule:
    """
    Warmup -> Stable Plateau -> Decay (`cosine` or `1-sqrt`) learning-rate schedule.

    Allows saving ONE reusable plateau checkpoint at `step == stable_end` and branching
    multiple fast `decay_steps` ablations directly from the plateau.
    """

    VALID_DECAY_MODES: Tuple[str, ...] = ("cosine", "1-sqrt")

    def __init__(
        self,
        base_lr: float,
        min_lr: float,
        total_steps: int,
        warmup_steps: int,
        decay_steps: int,
        decay_mode: str = "cosine",
    ):
        if decay_mode not in self.VALID_DECAY_MODES:
            raise ValueError(
                f"Unknown decay_mode={decay_mode!r}; expected one of {self.VALID_DECAY_MODES}"
            )
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(0, min(self.total_steps, int(warmup_steps)))
        self.decay_steps = max(0, min(self.total_steps - self.warmup_steps, int(decay_steps)))
        self.stable_end = self.total_steps - self.decay_steps
        self.decay_mode = str(decay_mode)

    def lr_at(self, step: int) -> float:
        s = int(step)
        if self.warmup_steps > 0 and s <= self.warmup_steps:
            return self.base_lr * (float(max(1, s)) / float(self.warmup_steps))
        if self.decay_steps <= 0 or s <= self.stable_end:
            return self.base_lr
        prog = min(1.0, max(0.0, float(s - self.stable_end) / float(self.decay_steps)))
        if self.decay_mode == "1-sqrt":
            return self.min_lr + (self.base_lr - self.min_lr) * (1.0 - math.sqrt(prog))
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * prog))

    def scale_at(self, step: int) -> float:
        return self.lr_at(step) / max(1e-12, self.base_lr)

    def apply(self, opt: Any, step: int) -> float:
        lr_s = self.lr_at(step)
        sc = lr_s / max(1e-12, self.base_lr)
        for pg in opt.param_groups:
            pg["lr"] = self.base_lr * float(pg.get("lr_mult", 1.0)) * sc
        return lr_s

    def is_plateau_save_step(self, step: int) -> bool:
        return int(step) == self.stable_end

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "stable_end": self.stable_end,
            "decay_mode": self.decay_mode,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WSDLRSchedule":
        return cls(
            base_lr=float(data["base_lr"]),
            min_lr=float(data["min_lr"]),
            total_steps=int(data["total_steps"]),
            warmup_steps=int(data["warmup_steps"]),
            decay_steps=int(data["decay_steps"]),
            decay_mode=str(data.get("decay_mode", "cosine")),
        )


def _cpu_clone_tree(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _cpu_clone_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cpu_clone_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_cpu_clone_tree(v) for v in obj)
    return obj


def save_plateau_checkpoint(
    path: Union[Path, str],
    model: nn.Module,
    opt: Any,
    step: int,
    config: Any = None,
    *,
    rng_py: Optional[random.Random] = None,
    rng_np: Optional[np.random.Generator] = None,
    schedule: Optional[WSDLRSchedule] = None,
    stage: str = "",
    allow_bic_in_s1a: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Saves a WSD plateau checkpoint (`model` unwrapped + `optimizer` + `rng` + `step` + `config`).

    Enforces `Law 8a`: refuses saving a Stage-1A (`S1A`) plateau with `ByteInductionCache`
    enabled unless `allow_bic_in_s1a=True` is explicitly passed.
    """
    cfg_dict = asdict(config) if (config is not None and is_dataclass(config)) else (
        dict(config) if isinstance(config, dict) else {}
    )
    if (
        stage.upper() == "S1A"
        and bool(cfg_dict.get("use_byte_induct_cache", False))
        and not allow_bic_in_s1a
    ):
        raise ValueError(
            "Law 8a violation: `ByteInductionCache` must stay isolated during Stage-1A S_5 grokking!"
        )

    out_p = Path(path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    rng_blob: Dict[str, Any] = {
        "torch_cpu": torch.get_rng_state(),
        "python_global": random.getstate(),
        "numpy_global": np.random.get_state(),
    }
    if torch.cuda.is_available():
        try:
            rng_blob["torch_cuda_all"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    if rng_py is not None:
        rng_blob["rng_py"] = rng_py.getstate()
    if rng_np is not None:
        rng_blob["rng_np"] = rng_np.bit_generator.state

    payload: Dict[str, Any] = {
        "format": "calcite_wsd_plateau_v1",
        "step": int(step),
        "stage": str(stage),
        "config": cfg_dict,
        "schedule": schedule.to_dict() if schedule is not None else None,
        "model_state_dict": _cpu_clone_tree(unwrap_compiled_state_dict(model)),
        "optimizer_state_dict": _cpu_clone_tree(opt.state_dict()) if opt is not None else None,
        "rng_state": rng_blob,
        "extra": dict(extra or {}),
    }
    torch.save(payload, out_p)
    return out_p


def load_plateau_checkpoint(
    path: Union[Path, str],
    model: nn.Module,
    opt: Any = None,
    *,
    rng_py: Optional[random.Random] = None,
    rng_np: Optional[np.random.Generator] = None,
    strict: bool = True,
    restore_rng: bool = True,
    preserve_ns_schedule: bool = True,
) -> Dict[str, Any]:
    """
    Restores a WSD plateau checkpoint into `model` (and optionally `opt` and RNG generators).

    When `preserve_ns_schedule=True`, branching from a `quintic5` plateau into a `polar4`/`polar3`
    decay keeps the caller's newly configured `opt.ns_schedule` intact while restoring all
    momentum buffers.
    """
    p = Path(path)
    try:
        blob = torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(p, map_location="cpu")

    sd = blob["model_state_dict"]
    # Support both raw and `torch.compile`-wrapped target models transparently.
    cur_keys = list(model.state_dict().keys())
    if any("._orig_mod." in k for k in cur_keys) and not any("._orig_mod." in k for k in sd):
        mapped: Dict[str, torch.Tensor] = {}
        for k_cur in cur_keys:
            k_clean = k_cur.replace("._orig_mod.", ".")
            if k_clean in sd:
                mapped[k_cur] = sd[k_clean]
        model.load_state_dict(mapped, strict=strict)
    else:
        model.load_state_dict(sd, strict=strict)

    if opt is not None and blob.get("optimizer_state_dict") is not None:
        caller_ns = getattr(opt, "ns_schedule", None)
        opt.load_state_dict(blob["optimizer_state_dict"])
        if preserve_ns_schedule and caller_ns is not None and hasattr(opt, "ns_schedule"):
            opt.ns_schedule = caller_ns

    if restore_rng and isinstance(blob.get("rng_state"), dict):
        rb = blob["rng_state"]
        if rb.get("torch_cpu") is not None:
            torch.set_rng_state(rb["torch_cpu"])
        if rb.get("python_global") is not None:
            random.setstate(rb["python_global"])
        if rb.get("numpy_global") is not None:
            np.random.set_state(rb["numpy_global"])
        if torch.cuda.is_available() and rb.get("torch_cuda_all") is not None:
            try:
                torch.cuda.set_rng_state_all(rb["torch_cuda_all"])
            except Exception:
                pass
        if rng_py is not None and rb.get("rng_py") is not None:
            rng_py.setstate(rb["rng_py"])
        if rng_np is not None and rb.get("rng_np") is not None:
            rng_np.bit_generator.state = rb["rng_np"]

    return {
        "step": int(blob.get("step", 0)),
        "stage": str(blob.get("stage", "")),
        "config": blob.get("config") or {},
        "schedule": blob.get("schedule"),
        "extra": blob.get("extra") or {},
    }


# ---------------------------------------------------------------------------
# 2. 3-Tier Evaluation Ladder (`Tier-0` Fast Screen -> `Tier-1` -> `Tier-2`)
# ---------------------------------------------------------------------------


def _eval_subset_suite(
    model: nn.Module,
    suite: list,
    pad_id: int,
    n_limit: int,
    batch_size: int,
    device: str,
) -> Dict[str, object]:
    from simvq_split.eval_utils import evaluate_suite_batched

    sub = suite[: min(int(n_limit), len(suite))]
    ex_pct, tk_pct, k_ok, n_tot = evaluate_suite_batched(
        model, sub, pad_id=pad_id, batch_size=batch_size, device=device
    )
    return {
        "exact_pct": round(float(ex_pct), 2),
        "token_pct": round(float(tk_pct), 2),
        "ok": int(k_ok),
        "n": int(n_tot),
    }


@torch.inference_mode()
def run_tiered_eval(
    model: nn.Module,
    tier: int,
    val_mm: np.ndarray,
    eval_wins: Dict[str, List[List[int]]],
    *,
    tok: Any = None,
    s5_suites: Optional[Dict[str, list]] = None,
    ov_suites: Optional[Dict[str, list]] = None,
    trap_suites: Optional[Dict[str, list]] = None,
    device: str = "cuda",
    seq_len: int = 512,
    eval_bs: int = 16,
    cog_bs: Optional[int] = None,
    smoke: bool = False,
) -> Dict[str, object]:
    """
    3-Tier progressive evaluation ladder for fast ablation screening:
      - `tier=0` (~5–10s): `Val(8b)`, `OW6(16w/dom)`, `S5_k2(32)`, `OV_L(32)`, `Needle_1500(8)`.
      - `tier=1` (~25–40s): `Val(32b)`, `L=2048`, `OW6(full)`, `S5/OV_L/Trap_L(60)`, `Needle L512..8K(8)`.
      - `tier=2`: full closure eval (`evaluate_full_model` + `L=32K` + full Needle + Overwrite).
    """
    if tier not in (0, 1, 2):
        raise ValueError(f"Unsupported eval tier={tier}; expected 0, 1, or 2")
    from experiments.scaling_phase_k.eval_suite import (
        eval_length_extrapolation_bpb,
        eval_openweb_6domains_any_byte_lm,
        eval_val_memmap_bpb,
    )
    from run_simvq_split_benchmark import harmonic_mean
    from scripts.colab_phase_l_v2_runner import (
        eval_long_context_stress,
        evaluate_full_model,
        inject_l2048_from_extrap,
    )
    from scripts.run_phase_n_mqar_probe_5m import eval_ood_verbatim_ladder
    from scripts.run_phase_n_v31_lm_check_5m import eval_overwrite_needle_ood

    model.eval()
    if hasattr(model, "set_snap_alpha"):
        model.set_snap_alpha(1.0)
    t0 = time.perf_counter()
    bs = max(2, int(eval_bs))
    c_bs = max(1, int(cog_bs if cog_bs is not None else bs))
    pad_id = int(getattr(tok, "PAD_ID", 0)) if tok is not None else 0
    has_cog = bool(tok is not None and s5_suites and ov_suites)

    if tier == 0:
        val_bpb = float(
            eval_val_memmap_bpb(
                model, val_mm, seq_len=seq_len, n_batches=4 if smoke else 8, batch_size=min(16, bs), device=device
            )
        )
        ow_sub = {k: v[: (4 if smoke else 16)] for k, v in eval_wins.items()}
        ow = eval_openweb_6domains_any_byte_lm(model, ow_sub, device=device, batch_size=bs)
        ndl_hs = (256,) if smoke else (1500,)
        ndl = eval_ood_verbatim_ladder(
            model,
            val_mm,
            device=device,
            haystacks=ndl_hs,
            n_samples=4 if smoke else 8,
            val_len=4 if smoke else 8,
            seed=9901,
        )
        out: Dict[str, object] = {
            "tier": 0,
            "val_bpb": round(val_bpb, 4),
            "ow6_mean_bpb": round(float(ow["mean_openweb_bpb"]), 4),
            "openweb_6": ow,
            "ood_verbatim_ladder": ndl,
        }
        if has_cog and s5_suites is not None and ov_suites is not None:
            n_c = 4 if smoke else 32
            s5_k2 = _eval_subset_suite(model, s5_suites["k2"], pad_id, n_c, c_bs, device)
            ov_l = _eval_subset_suite(model, ov_suites["ood_long"], pad_id, n_c, c_bs, device)
            out["cognitive"] = {"s5_k2": s5_k2, "ov_ood_long": ov_l}
            out["s5_k2_exact_pct"] = s5_k2["exact_pct"]
            out["ov_long_exact_pct"] = ov_l["exact_pct"]
        out["elapsed_s"] = round(time.perf_counter() - t0, 2)
        return out

    if tier == 1:
        val_bpb = float(
            eval_val_memmap_bpb(
                model, val_mm, seq_len=seq_len, n_batches=4 if smoke else 32, batch_size=bs, device=device
            )
        )
        ext = eval_length_extrapolation_bpb(
            model,
            val_mm,
            device=device,
            l_in=seq_len,
            l_ood=1024 if smoke else 2048,
            n_batches=2 if smoke else 12,
            batch_size_ood=min(4, bs),
            reuse_l_in_bpb=val_bpb,
        )
        ow = eval_openweb_6domains_any_byte_lm(model, eval_wins, device=device, batch_size=bs)
        ndl_hs = (256,) if smoke else (512, 1500, 4096, 8192)
        ndl = eval_ood_verbatim_ladder(
            model,
            val_mm,
            device=device,
            haystacks=ndl_hs,
            n_samples=4 if smoke else 8,
            val_len=4 if smoke else 8,
            seed=9901,
        )
        out = {
            "tier": 1,
            "val_bpb": round(val_bpb, 4),
            "extrap_4x": ext,
            "ow6_mean_bpb": round(float(ow["mean_openweb_bpb"]), 4),
            "openweb_6": ow,
            "ood_verbatim_ladder": ndl,
        }
        if has_cog and s5_suites is not None and ov_suites is not None:
            n_c = 4 if smoke else 60
            s5_res = {
                k: _eval_subset_suite(model, s5_suites[k], pad_id, n_c, c_bs, device)
                for k in ("k2", "k4", "k8", "k16")
                if k in s5_suites
            }
            s5_sum = round(sum(float(v["exact_pct"]) for v in s5_res.values()), 2)
            ov_l = _eval_subset_suite(model, ov_suites["ood_long"], pad_id, n_c, c_bs, device)
            cog_d: Dict[str, object] = {"s5": s5_res, "s5_sum": s5_sum, "ov_ood_long": ov_l}
            if trap_suites and "ood_long" in trap_suites:
                tr_l = _eval_subset_suite(model, trap_suites["ood_long"], pad_id, n_c, c_bs, device)
                cog_d["trap_ood_long"] = tr_l
                cog_d["selective_f1_ood_long"] = round(
                    harmonic_mean(float(ov_l["exact_pct"]), float(tr_l["exact_pct"])), 2
                )
            out["cognitive"] = cog_d
        out["elapsed_s"] = round(time.perf_counter() - t0, 2)
        return out

    # Tier 2: Full closure evaluation
    if has_cog and s5_suites is not None and ov_suites is not None and trap_suites is not None:
        full_ev = evaluate_full_model(
            model,
            val_mm,
            eval_wins,
            tok,
            s5_suites,
            ov_suites,
            trap_suites,
            device=device,
            quick=smoke,
            eval_bs=bs,
            cog_bs=c_bs,
        )
        long_ctx = inject_l2048_from_extrap(
            eval_long_context_stress(
                model, val_mm, device=device, lengths=(256, 512) if smoke else (8192, 32768)
            ),
            None if smoke else full_ev.get("extrap_4x"),
        )
        val_bpb = float(full_ev["val_bpb"])
        ow = full_ev["openweb_6"]
    else:
        val_bpb = float(
            eval_val_memmap_bpb(
                model, val_mm, seq_len=seq_len, n_batches=4 if smoke else 24, batch_size=bs, device=device
            )
        )
        ow = eval_openweb_6domains_any_byte_lm(model, eval_wins, device=device, batch_size=bs)
        long_ctx = eval_long_context_stress(
            model, val_mm, device=device, lengths=(256, 512) if smoke else (2048, 8192, 32768)
        )
        full_ev = None

    ndl_hs = (256,) if smoke else (512, 1500, 4096, 8192, 32768)
    ndl = eval_ood_verbatim_ladder(
        model,
        val_mm,
        device=device,
        haystacks=ndl_hs,
        n_samples=4 if smoke else 16,
        val_len=4 if smoke else 8,
        seed=9901,
    )
    ov_ndl = eval_overwrite_needle_ood(
        model, val_mm, device=device, haystack_bytes=256 if smoke else 1500, n_samples=4 if smoke else 16
    )
    out = {
        "tier": 2,
        "val_bpb": round(val_bpb, 4),
        "ow6_mean_bpb": round(float(ow["mean_openweb_bpb"]), 4),
        "ow6_domains": {k: round(float(v["byte_bpb"]), 4) for k, v in ow["regimes"].items()},
        "long_context": long_ctx,
        "ood_verbatim_ladder": ndl,
        "overwrite_needle_1500b": ov_ndl,
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    if full_ev is not None:
        out["full_eval"] = full_ev
    return out


# ---------------------------------------------------------------------------
# 3. Online Early-Kill Gate (`OEBEarlyStopper`)
# ---------------------------------------------------------------------------


class OEBEarlyStopper:
    """
    Mid-run early-stopping gate comparing candidate `val_bpb` against a control curve
    at configured checkpoint steps (default margin `+0.025 BPB` > `4*sigma`).

    Set `disabled_for_axis=True` for pure cognitive or verbatim-needle axes where
    mid-run LM `val_bpb` is not the primary objective.
    """

    def __init__(
        self,
        control_curve: Dict[int, float],
        margin_bpb: float = 0.025,
        disabled_for_axis: bool = False,
        axis_name: str = "",
        min_step: int = 0,
    ):
        self.control_curve: Dict[int, float] = {int(k): float(v) for k, v in control_curve.items()}
        self.margin_bpb = float(margin_bpb)
        self.disabled_for_axis = bool(disabled_for_axis)
        self.axis_name = str(axis_name)
        self.min_step = int(min_step)
        self.killed: bool = False
        self.kill_step: Optional[int] = None
        self.kill_reason: str = ""
        self.history: List[Dict[str, Any]] = []

    def should_check(self, step: int) -> bool:
        if self.disabled_for_axis or self.killed:
            return False
        s = int(step)
        return (s >= self.min_step) and (s in self.control_curve)

    def check(self, step: int, val_bpb: float) -> Tuple[bool, str]:
        s = int(step)
        v = float(val_bpb)
        if self.disabled_for_axis:
            msg = f"bypassed at step={s} (disabled_for_axis={self.axis_name or True})"
            self.history.append({"step": s, "val_bpb": v, "killed": False, "reason": msg})
            return False, msg
        ctrl = self.control_curve.get(s)
        if ctrl is None:
            msg = f"no control anchor at step={s}"
            self.history.append({"step": s, "val_bpb": v, "killed": False, "reason": msg})
            return False, msg
        if not math.isfinite(v):
            self.killed = True
            self.kill_step = s
            self.kill_reason = f"KILL at step={s}: non-finite val_bpb={v}"
            self.history.append({"step": s, "val_bpb": v, "ctrl_bpb": ctrl, "killed": True, "reason": self.kill_reason})
            return True, self.kill_reason
        delta = v - ctrl
        if delta > self.margin_bpb:
            self.killed = True
            self.kill_step = s
            self.kill_reason = (
                f"KILL at step={s}: val_bpb={v:.4f} vs ctrl={ctrl:.4f} "
                f"(delta={delta:+.4f} > +{self.margin_bpb:.4f})"
            )
            self.history.append(
                {"step": s, "val_bpb": v, "ctrl_bpb": ctrl, "delta": round(delta, 5), "killed": True, "reason": self.kill_reason}
            )
            return True, self.kill_reason
        msg = (
            f"PASS at step={s}: val_bpb={v:.4f} vs ctrl={ctrl:.4f} "
            f"(delta={delta:+.4f} <= +{self.margin_bpb:.4f})"
        )
        self.history.append(
            {"step": s, "val_bpb": v, "ctrl_bpb": ctrl, "delta": round(delta, 5), "killed": False, "reason": msg}
        )
        return False, msg

    def summary(self) -> Dict[str, Any]:
        return {
            "killed": self.killed,
            "kill_step": self.kill_step,
            "kill_reason": self.kill_reason,
            "margin_bpb": self.margin_bpb,
            "disabled_for_axis": self.disabled_for_axis,
            "axis_name": self.axis_name,
            "history": list(self.history),
        }


# ---------------------------------------------------------------------------
# 4. Persistent Colab CUDA/Inductor/Triton Cache Configurator
# ---------------------------------------------------------------------------


def configure_colab_fast_cache(drive_cache_dir: Union[Path, str]) -> Dict[str, str]:
    """
    Configures persistent `TORCH_EXTENSIONS_DIR`, `TORCHINDUCTOR_CACHE_DIR`, and
    `TRITON_CACHE_DIR` under `drive_cache_dir` so cold Colab runtimes reuse compiled
    CUDA extensions and Triton kernels instead of rebuilding from scratch.
    """
    base = Path(drive_cache_dir)
    ext_dir = base / "torch_extensions"
    ind_dir = base / "inductor_cache"
    tri_dir = base / "triton_cache"
    for d in (ext_dir, ind_dir, tri_dir):
        d.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(ext_dir)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(ind_dir)
    os.environ["TRITON_CACHE_DIR"] = str(tri_dir)
    return {
        "TORCH_EXTENSIONS_DIR": str(ext_dir),
        "TORCHINDUCTOR_CACHE_DIR": str(ind_dir),
        "TRITON_CACHE_DIR": str(tri_dir),
    }


__all__ = [
    "OEBEarlyStopper",
    "WSDLRSchedule",
    "configure_colab_fast_cache",
    "load_plateau_checkpoint",
    "run_tiered_eval",
    "save_plateau_checkpoint",
]
