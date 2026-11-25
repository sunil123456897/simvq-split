"""
Unified Evaluation Suite for Phase K Raw-UTF8 Scaling Comparison (`Calcite`, `RoPE`, `HybridDeltaSWA`).

Evaluates all architectures on identical byte streams (`vocab_size = 259`):
1. Held-out `val_memmap_bpb` (`L = 512`) and `4x` OOD length extrapolation (`L = 512 -> 2048`, tail vs head).
2. Six disjoint multilingual Open-Web & Code regimes (`GSM8K_EN`, `Russian_UTF8`, `Python_Code`,
   `ARC_JSON`, `CJK_UTF8`, `Emoji_4Byte`) + dynamic `mid_utf8` boundary safety check.
3. Unseen cognitive suites (`S_5` `k=2,4,8,16`, `Overwrite`, `Trap`, `Selective F1`).
4. Gate-2 initialization std & `qat_mix=0` parity guard before any training run.
"""

from __future__ import annotations

import copy
import ctypes
import math
import os
from pathlib import Path
import queue
import statistics
import sys
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.scaling_phase_k.lean_system1 import (
    CalciteLeanSystem1LM,
    LeanSystem1Config,
)
from experiments.v5_proto.probe_phase_b_byte_hnet import measure_crystal_drift_at_L
from kernels.fast_packed_lut import pack_fast_model_2bit_lut
from run_simvq_split_benchmark import harmonic_mean
from simvq_split.eval_utils import evaluate_suite_batched, wilson_ci_95
from simvq_split.quant_lut import PackedLUT2BitLinear
from simvq_split.tokenizer import ByteBPETokenizer


def load_lean_packed_lut_checkpoint(
    path: Path,
    device: str = "cpu",
    backend: str = "auto",
) -> Tuple[CalciteLeanSystem1LM, LeanSystem1Config, ByteBPETokenizer, dict]:
    """Loads a saved Phase K `CalciteLeanSystem1LM` `2-bit LUT` checkpoint into `FastPackedLUT2BitLinear`."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    cfg = LeanSystem1Config(**payload["simvq_split_config"])
    tok = ByteBPETokenizer.from_dict(payload["tokenizer"])
    skel = CalciteLeanSystem1LM(cfg).eval()
    if skel.byte_chunker is not None:
        skel.byte_chunker.lock_router(True)
        skel.set_crystal_schedule(1, 1)
    else:
        skel.set_snap_alpha(1.0)
    lut_m = pack_fast_model_2bit_lut(skel, backend=backend)
    lut_m.load_state_dict(payload["lut_state_dict"], strict=True)
    lut_m = lut_m.to(device).eval()
    if lut_m.byte_chunker is not None:
        lut_m.set_crystal_schedule(1, 1)
    else:
        lut_m.set_snap_alpha(1.0)
    return lut_m, cfg, tok, payload


def has_usable_cuda() -> bool:
    return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)


def sync_if_cuda(device: str) -> None:
    if device == "cuda" and has_usable_cuda():
        torch.cuda.synchronize()


def is_lut_model(model: nn.Module) -> bool:
    return any(isinstance(m, PackedLUT2BitLinear) for m in model.modules())


def _available_ram_mb() -> Optional[float]:
    """Returns currently available host RAM in MiB, or `None` if unavailable."""
    if sys.platform == "win32":
        try:
            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            st = _MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return float(st.ullAvailPhys) / (1024.0 * 1024.0)
        except Exception:
            return None
    elif os.path.isfile("/proc/meminfo"):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return float(line.split()[1]) / 1024.0
        except Exception:
            return None
    return None


def load_utf8_corpus_array(
    path: Union[Path, str],
    max_ram_mb: int = 2048,
    headroom_mb: int = 1024,
) -> np.ndarray:
    """
    Loads a `uint16` UTF-8 corpus binary into RAM via sequential `np.fromfile` when the
    file fits within `max_ram_mb` and host RAM has `>= headroom_mb` free, eliminating
    random-seek HDD `np.memmap` stalls (`~960 ms -> 0.08 ms/batch`). Falls back cleanly
    to read-only `np.memmap` when memory is constrained.
    """
    p = Path(path)
    size_mb = float(p.stat().st_size) / (1024.0 * 1024.0)
    avail_mb = _available_ram_mb()
    fits_limit = size_mb <= float(max_ram_mb)
    fits_host = (avail_mb is None) or (avail_mb >= size_mb + float(headroom_mb))
    if fits_limit and fits_host:
        return np.fromfile(str(p), dtype=np.uint16)
    return np.memmap(str(p), dtype=np.uint16, mode="r")


def encode_utf8_fast_np(
    tok: ByteBPETokenizer,
    text: str,
    add_bos: bool = False,
    add_eos: bool = False,
) -> np.ndarray:
    """Fast raw-UTF8 byte encoding (`merges=[]`) returning `np.int64` without Python byte loops."""
    if getattr(tok, "merges", None):
        return np.asarray(tok.encode(text, add_bos=add_bos, add_eos=add_eos), dtype=np.int64)
    raw = text.encode("utf-8")
    byte_off = int(getattr(tok, "BYTE_OFFSET", 3))
    n = len(raw)
    total = n + int(add_bos) + int(add_eos)
    out = np.empty(total, dtype=np.int64)
    pos = 0
    if add_bos:
        out[0] = int(tok.BOS_ID)
        pos = 1
    if n > 0:
        out[pos : pos + n] = np.frombuffer(raw, dtype=np.uint8).astype(np.int64) + byte_off
        pos += n
    if add_eos:
        out[pos] = int(tok.EOS_ID)
    return out


def encode_utf8_fast(
    tok: ByteBPETokenizer,
    text: str,
    add_bos: bool = False,
    add_eos: bool = False,
) -> List[int]:
    """Drop-in fast equivalent of `ByteBPETokenizer.encode` for raw-UTF8 (`merges=[]`) tokenizers."""
    if getattr(tok, "merges", None):
        return tok.encode(text, add_bos=add_bos, add_eos=add_eos)
    return encode_utf8_fast_np(tok, text, add_bos=add_bos, add_eos=add_eos).tolist()


def _pin_tree(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.pin_memory() if obj.device.type == "cpu" else obj
    if isinstance(obj, tuple):
        return tuple(_pin_tree(x) for x in obj)
    if isinstance(obj, list):
        return [_pin_tree(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _pin_tree(v) for k, v in obj.items()}
    return obj


def _to_device_non_blocking(obj: Any, device: str) -> Any:
    if isinstance(obj, torch.Tensor):
        if str(obj.device) == device:
            return obj
        return obj.to(device=device, non_blocking=(device == "cuda"))
    if isinstance(obj, tuple):
        return tuple(_to_device_non_blocking(x, device) for x in obj)
    if isinstance(obj, list):
        return [_to_device_non_blocking(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_device_non_blocking(v, device) for k, v in obj.items()}
    return obj


class BatchPrefetcher:
    """
    Single-worker background CPU batch prefetcher with `pin_memory()` and async
    `non_blocking=True` H2D transfer on the consumer (main) thread.

    Preserves exact deterministic step order of `sample_fn()` and shuts down cleanly
    as a context manager without leaking daemon threads.
    """

    def __init__(
        self,
        sample_fn: Callable[[], Any],
        device: str,
        total_steps: Optional[int] = None,
        prefetch_depth: int = 4,
        enabled: bool = True,
    ):
        self._sample_fn = sample_fn
        self._device = device
        self._total_steps = total_steps
        self._pin = bool(device == "cuda" and has_usable_cuda())
        self._enabled = bool(enabled and prefetch_depth > 0)
        self._produced = 0
        self._stop = threading.Event()
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(prefetch_depth)))
        self._sentinel = object()
        self._thread: Optional[threading.Thread] = None
        self._closed = False
        if self._enabled:
            self._thread = threading.Thread(
                target=self._worker,
                name="calcite-batch-prefetcher",
                daemon=True,
            )
            self._thread.start()

    def _worker(self) -> None:
        try:
            step = 0
            while not self._stop.is_set():
                if self._total_steps is not None and step >= self._total_steps:
                    break
                item = self._sample_fn()
                if self._pin:
                    item = _pin_tree(item)
                while not self._stop.is_set():
                    try:
                        self._queue.put(("ok", item), timeout=0.05)
                        break
                    except queue.Full:
                        continue
                step += 1
        except BaseException as exc:
            try:
                self._queue.put(("err", exc), timeout=1.0)
            except queue.Full:
                pass
        finally:
            try:
                self._queue.put(("end", self._sentinel), timeout=0.5)
            except queue.Full:
                pass

    def next_batch(self) -> Any:
        if not self._enabled:
            if self._total_steps is not None and self._produced >= self._total_steps:
                raise StopIteration
            self._produced += 1
            item = self._sample_fn()
            if self._pin:
                item = _pin_tree(item)
            return _to_device_non_blocking(item, self._device)

        kind, payload = self._queue.get()
        if kind == "err":
            self.close()
            raise payload
        if kind == "end":
            self.close()
            raise StopIteration
        return _to_device_non_blocking(payload, self._device)

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        return self.next_batch()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def __enter__(self) -> "BatchPrefetcher":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def sample_memmap_batch(
    mm: np.ndarray,
    batch_size: int,
    seq_len: int,
    rng: np.random.Generator,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Samples `uint16` byte sequences aligned to UTF-8 codepoint boundaries."""
    max_start = len(mm) - seq_len - 6
    starts = rng.integers(0, max_start, size=batch_size)
    rows = np.empty((batch_size, seq_len + 1), dtype=np.int64)
    for r_idx, st in enumerate(starts):
        s = int(st)
        for _ in range(3):
            tok_v = int(mm[s])
            if tok_v >= 3 and ((tok_v - 3) & 0xC0) == 0x80:
                s += 1
            else:
                break
        rows[r_idx] = mm[s : s + seq_len + 1]
    t = torch.from_numpy(rows)
    if device == "cuda" and has_usable_cuda():
        t = t.pin_memory().to(device=device, non_blocking=True)
    elif device != "cpu":
        t = t.to(device=device)
    return t[:, :-1], t[:, 1:]


@torch.no_grad()
def check_gate2_block_std_parity(model_fp: nn.Module, device: str) -> Dict[str, object]:
    """
    Gate 2 micro-probe: checks finite block `std` at init, exact `qat_mix=0` logit parity (`< 1e-5`),
    and bounded `std` ratio (`[0.5, 1.5]`) at `qat_mix=1.0` for `CalciteLeanSystem1LM`.
    """
    torch.manual_seed(2026)
    vocab_sz = int(getattr(getattr(model_fp, "cfg", None), "vocab_size", 259))
    x_probe = torch.randint(3, vocab_sz, (4, 192), device=device)
    has_calcite = hasattr(model_fp, "cortex_pre") and hasattr(model_fp, "set_ternary_qat")

    def _capture_stds(m: nn.Module) -> Tuple[torch.Tensor, Dict[str, float]]:
        stds: Dict[str, float] = {}
        hooks = []
        watch: List[Tuple[str, nn.Module]] = []
        if has_calcite:
            watch.extend(
                [
                    ("cortex_pre", m.cortex_pre[-1]),
                    ("mid_mlp", m.mid_mlp),
                    ("cortex_post", m.cortex_post[-1]),
                    ("norm_f", m.norm_f),
                ]
            )
            if getattr(m, "crystal", None) is not None:
                watch.append(("crystal", m.crystal))
            if getattr(m, "woodbury", None) is not None:
                watch.append(("woodbury", m.woodbury))
        else:
            watch.append(("norm_f", m.norm_f))

        for bname, mod in watch:
            def _make_hk(k: str):
                def _hk(_mod, _inp, out):
                    t = out[0] if isinstance(out, tuple) else out
                    assert torch.isfinite(t).all(), f"Non-finite activation in `{k}`!"
                    stds[k] = float(t.detach().float().std().item())
                return _hk
            hooks.append(mod.register_forward_hook(_make_hk(bname)))

        try:
            logits = m(x_probe)
            assert torch.isfinite(logits).all(), "Non-finite logits in Gate 2!"
        finally:
            for h in hooks:
                h.remove()
        return logits.detach(), stds

    if has_calcite:
        fp_eval = copy.deepcopy(model_fp).to(device).eval()
        if getattr(fp_eval, "byte_chunker", None) is not None:
            fp_eval.byte_chunker.lock_router(True)
        fp_eval.set_crystal_schedule(1, 1)
        fp_eval.set_snap_alpha(1.0)
        lg_fp, std_fp = _capture_stds(fp_eval)
    else:
        prev_train = model_fp.training
        model_fp.eval()
        lg_fp, std_fp = _capture_stds(model_fp)
        model_fp.train(prev_train)
        return {
            "mix0_max_abs_diff": 0.0,
            "block_std_fp": {k: round(v, 5) for k, v in std_fp.items()},
            "block_std_ratio_mix1": {k: 1.0 for k in std_fp},
            "gate2_passed": True,
        }

    qat_m = copy.deepcopy(fp_eval).to(device).eval()
    qat_m.set_qat_mix(0.0)
    qat_m.set_ternary_qat(True)
    lg_mix0, _ = _capture_stds(qat_m)
    mix0_diff = float((lg_fp.float() - lg_mix0.float()).abs().max().item())
    assert mix0_diff < 1e-5, f"Gate 2 qat_mix=0 parity failed: diff={mix0_diff}"

    qat_m.set_qat_mix(1.0)
    _, std_mix1 = _capture_stds(qat_m)
    ratios: Dict[str, float] = {}
    for k, sf in std_fp.items():
        r = std_mix1[k] / max(1e-6, sf)
        ratios[k] = round(r, 4)
        assert 0.5 <= r <= 1.5, (
            f"Gate 2 std jump > 1.5x in `{k}`: ratio={r:.4f} (fp={sf:.4f}, qat={std_mix1[k]:.4f})"
        )

    del fp_eval, qat_m
    return {
        "mix0_max_abs_diff": round(mix0_diff, 8),
        "block_std_fp": {k: round(v, 5) for k, v in std_fp.items()},
        "block_std_ratio_mix1": ratios,
        "gate2_passed": True,
    }


@torch.inference_mode()
def eval_val_memmap_bpb(
    model: nn.Module,
    val_mm: np.ndarray,
    seq_len: int,
    n_batches: int,
    batch_size: int,
    device: str,
) -> float:
    model.eval()
    if hasattr(model, "set_snap_alpha"):
        model.set_snap_alpha(1.0)
    rng = np.random.default_rng(42)
    nlls: List[torch.Tensor] = []
    use_amp = (device == "cuda") and has_usable_cuda() and (not is_lut_model(model))
    for _ in range(n_batches):
        xb, yb = sample_memmap_batch(val_mm, batch_size, seq_len, rng, device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            lg = model(xb)
        ce = F.cross_entropy(lg.float().transpose(1, 2), yb, ignore_index=0)
        nlls.append(ce.detach())
    mean_nll = float(torch.stack(nlls).mean().item())
    return round(mean_nll / math.log(2.0), 4)


@torch.inference_mode()
def eval_length_extrapolation_bpb(
    model: nn.Module,
    val_mm: np.ndarray,
    device: str,
    l_in: int = 512,
    l_ood: int = 2048,
    n_batches: int = 16,
    batch_size_ood: int = 4,
    reuse_l_in_bpb: Optional[float] = None,
) -> Dict[str, float]:
    """
    Evaluates in-distribution `L = l_in` BPB vs `4x` OOD `L = l_ood` BPB, splitting
    OOD positions into head `[0..l_in)` and tail `[l_ood//2..l_ood)` to expose RoPE collapse.
    Pass `reuse_l_in_bpb` to reuse a freshly computed `L = l_in` validation BPB cleanly.
    """
    model.eval()
    if hasattr(model, "set_snap_alpha"):
        model.set_snap_alpha(1.0)
    use_amp = (device == "cuda") and has_usable_cuda() and (not is_lut_model(model))
    inv_ln2 = 1.0 / math.log(2.0)

    if reuse_l_in_bpb is not None:
        bpb_in = float(reuse_l_in_bpb)
    else:
        rng_in = np.random.default_rng(142)
        in_nlls: List[torch.Tensor] = []
        for _ in range(n_batches):
            xb, yb = sample_memmap_batch(val_mm, max(4, batch_size_ood * 2), l_in, rng_in, device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                lg = model(xb)
            ce = F.cross_entropy(lg.float().transpose(1, 2), yb, ignore_index=0)
            in_nlls.append(ce.detach())
        bpb_in = float(torch.stack(in_nlls).mean().item()) * inv_ln2

    rng_ood = np.random.default_rng(242)
    ood_all_nlls: List[torch.Tensor] = []
    ood_head_nlls: List[torch.Tensor] = []
    ood_tail_nlls: List[torch.Tensor] = []
    tail_start = l_ood // 2
    for _ in range(n_batches):
        xb, yb = sample_memmap_batch(val_mm, batch_size_ood, l_ood, rng_ood, device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            lg = model(xb)
        tok_nll = F.cross_entropy(lg.float().transpose(1, 2), yb, ignore_index=0, reduction="none")
        ood_all_nlls.append(tok_nll.mean().detach())
        ood_head_nlls.append(tok_nll[:, :l_in].mean().detach())
        ood_tail_nlls.append(tok_nll[:, tail_start:].mean().detach())

    bpb_ood = float(torch.stack(ood_all_nlls).mean().item()) * inv_ln2
    bpb_head = float(torch.stack(ood_head_nlls).mean().item()) * inv_ln2
    bpb_tail = float(torch.stack(ood_tail_nlls).mean().item()) * inv_ln2
    return {
        f"bpb_L{l_in}": round(bpb_in, 4),
        f"bpb_L{l_ood}": round(bpb_ood, 4),
        f"bpb_L{l_ood}_head": round(bpb_head, 4),
        f"bpb_L{l_ood}_tail": round(bpb_tail, 4),
        "extrap_delta_bpb": round(bpb_ood - bpb_in, 4),
        "tail_minus_head_bpb": round(bpb_tail - bpb_head, 4),
    }


@torch.inference_mode()
def eval_openweb_6domains_any_byte_lm(
    model: nn.Module,
    eval_wins: Dict[str, List[List[int]]],
    device: str,
    batch_size: int = 16,
) -> Dict[str, object]:
    """Evaluates all 6 disjoint UTF-8 open-web domains for any byte-level LM (single-pass chunk capture)."""
    model.eval()
    if hasattr(model, "set_snap_alpha"):
        model.set_snap_alpha(1.0)
    use_amp = (device == "cuda") and has_usable_cuda() and (not is_lut_model(model))
    bc = getattr(model, "byte_chunker", None)
    regimes_out: Dict[str, Dict[str, object]] = {}

    captured_ctx: List[Any] = []
    orig_enc = None
    if bc is not None and hasattr(bc, "encode_to_chunks"):
        orig_enc = bc.encode_to_chunks

        def _capturing_enc(*args, **kwargs):
            res = orig_enc(*args, **kwargs)
            if isinstance(res, tuple) and len(res) >= 3:
                captured_ctx.append(res[2])
            return res

        bc.encode_to_chunks = _capturing_enc

    try:
        for r_name, wins in eval_wins.items():
            x_cpu = torch.tensor([w[:-1] for w in wins], dtype=torch.long)
            y_cpu = torch.tensor([w[1:] for w in wins], dtype=torch.long)
            x = x_cpu.to(device=device, non_blocking=(device == "cuda"))
            y = y_cpu.to(device=device, non_blocking=(device == "cuda"))
            nlls: List[torch.Tensor] = []
            bms: List[torch.Tensor] = []
            vms: List[torch.Tensor] = []
            for s in range(0, x.size(0), batch_size):
                xb = x[s : s + batch_size]
                yb = y[s : s + batch_size]
                if bc is not None:
                    captured_ctx.clear()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    lg = model(xb)
                    if bc is not None and not captured_ctx and orig_enc is not None:
                        _, _, ctx_fb = orig_enc(model.tok_emb(xb), xb)
                        captured_ctx.append(ctx_fb)
                if bc is not None and captured_ctx:
                    ctx = captured_ctx[-1]
                    bms.append(ctx.b_mask.detach().cpu())
                    vms.append(ctx.valid_mask.detach().cpu())
                nll = F.cross_entropy(
                    lg.float().transpose(1, 2), yb, ignore_index=0, reduction="none"
                ).mean(dim=1)
                nlls.append(nll.detach().cpu())

            bpb = round(float(torch.cat(nlls).mean().item()) / math.log(2.0), 4)
            if bc is not None and bms:
                b_mask = torch.cat(bms, dim=0)
                v_mask = torch.cat(vms, dim=0)
                is_byte = x_cpu >= 3
                raw_v = (x_cpu - 3).clamp(min=0)
                is_cont = is_byte & (torch.bitwise_and(raw_v, 0xC0) == 0x80)
                is_cp = is_byte & (~is_cont)
                n_bytes = int(is_byte.sum().item())
                n_cps = int(is_cp.sum().item())
                n_chunks = int(b_mask.sum().item())
                b_int = b_mask[:, 1:] & v_mask[:, 1:]
                n_int = int(b_int.sum().item())
                n_mid = int((b_int & is_cont[:, 1:]).sum().item())
                regimes_out[r_name] = {
                    "byte_bpb": bpb,
                    "bytes_per_chunk": round(n_bytes / max(1, n_chunks), 3),
                    "cp_per_chunk": round(n_cps / max(1, n_chunks), 3),
                    "mid_utf8_split_pct": round(100.0 * n_mid / max(1, n_int), 4),
                    "m_mean": round(float(b_mask.sum(dim=1).float().mean().item()), 1),
                    "m_max": int(b_mask.sum(dim=1).max().item()),
                }
            else:
                regimes_out[r_name] = {
                    "byte_bpb": bpb,
                    "bytes_per_chunk": 1.0,
                    "cp_per_chunk": 1.0,
                    "mid_utf8_split_pct": 0.0,
                }
    finally:
        if bc is not None and orig_enc is not None:
            bc.encode_to_chunks = orig_enc

    return {
        "regimes": regimes_out,
        "max_mid_utf8_split_pct": max(float(v["mid_utf8_split_pct"]) for v in regimes_out.values()),
        "mean_openweb_bpb": round(
            sum(float(v["byte_bpb"]) for v in regimes_out.values()) / len(regimes_out), 4
        ),
    }


@torch.inference_mode()
def eval_cognitive_suites_any_byte_lm(
    model: nn.Module,
    tok: ByteBPETokenizer,
    s5_suites: Dict[str, list],
    ov_suites: Dict[str, list],
    trap_suites: Dict[str, list],
    device: str,
    batch_size: int = 32,
) -> Dict[str, object]:
    """Evaluates unseen `S_5`, `Overwrite`, `Trap`, and `Selective F1` + optional crystal drift."""
    model.eval()
    if hasattr(model, "set_snap_alpha"):
        model.set_snap_alpha(1.0)

    def _eval_group(suites: Dict[str, list]) -> Dict[str, Dict[str, object]]:
        out: Dict[str, Dict[str, object]] = {}
        for k, suite in suites.items():
            ex_pct, tk_pct, k_ok, n_tot = evaluate_suite_batched(
                model, suite, pad_id=tok.PAD_ID, batch_size=batch_size, device=device
            )
            lo, hi = wilson_ci_95(k_ok, n_tot)
            out[k] = {
                "exact_pct": round(ex_pct, 2),
                "token_pct": round(tk_pct, 2),
                "ok": int(k_ok),
                "n": int(n_tot),
                "wilson_95": [round(lo, 2), round(hi, 2)],
            }
        return out

    s5_res = _eval_group(s5_suites)
    ov_res = _eval_group(ov_suites)
    trap_res = _eval_group(trap_suites)
    s5_sum = round(sum(float(s5_res[k]["exact_pct"]) for k in ("k2", "k4", "k8", "k16")), 2)
    sel_in = round(
        harmonic_mean(float(ov_res["in_dist"]["exact_pct"]), float(trap_res["in_dist"]["exact_pct"])),
        2,
    )
    sel_long = round(
        harmonic_mean(float(ov_res["ood_long"]["exact_pct"]), float(trap_res["ood_long"]["exact_pct"])),
        2,
    )
    out: Dict[str, object] = {
        "s5": s5_res,
        "s5_sum": s5_sum,
        "overwrite": ov_res,
        "trap": trap_res,
        "selective_f1_in": sel_in,
        "selective_f1_ood_long": sel_long,
    }
    if hasattr(model, "crystal") and getattr(model, "crystal", None) is not None:
        out["crystal_drift_L2048"] = measure_crystal_drift_at_L(model, tok.vocab_size, device, length=2048)
    return out


@torch.inference_mode()
def measure_infer_latency_ms(
    model: nn.Module,
    vocab_size: int,
    device: str,
    batch_size: int = 8,
    seq_len: int = 512,
    warmup: int = 2,
    repeats: int = 6,
) -> float:
    model.eval()
    xb = torch.randint(3, vocab_size, (batch_size, seq_len), device=device)
    use_amp = (device == "cuda") and has_usable_cuda() and (not is_lut_model(model))
    for _ in range(warmup):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            _ = model(xb)
    sync_if_cuda(device)
    times: List[float] = []
    for _ in range(repeats):
        sync_if_cuda(device)
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            _ = model(xb)
        sync_if_cuda(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    return round(statistics.median(times), 2)
