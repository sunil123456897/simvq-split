"""
scripts/colab_phase_g_runner.py — Исполняющий конвейер Фазы G в Google Colab (`A100` / `L4`)
и локальной верификации:

Стадии CLI (`--stage`):
  1. `track1`: Компиляция нативных ядер `kernels/fast_packed_lut.py` (`cuda_ext` `load_inline` +
     `cpu_ext` OpenMP), проверка числового паритета (`max_abs_diff < 1e-4` на `1.48M` и
     `Tier-1 17.49M`, включая `row_aligned=False` `woodbury.out_proj`) и замеров скорости
     `reference (Python)` vs `pytorch_vec` vs `cuda_ext` (`SMEM T-MAC LUT`).
  2. `build-corpus`: Сборка строго чистого UTF-8 бинарного потока (`~1.80 GB` train + `24 MB` val
     в `/content/data/` `uint16`) из открытых корпусов (`FineWeb-Edu`, `FineMath` /
     `OpenWebMath`, русская и китайская Википедия, код Python, 4-байтовый UTF-8 и `train[:6000]`
     математика + `SphericalBase10RegisterALU` трассы) со строгой фильтрацией 45-символьных
     нормализованных подстрок из `GSM8K Cal + BlindDev + Sealed Test` (`0` утечек).
  3. `train-tier1`: Предобучение `Calcite-5 Tier-1` (`17,488,961` параметров, `vocab=259`,
     `d_model=384, n_layers_pre=4, n_layers_post=4, window=12`, `byte_chunk_mode="utf8"`,
     `use_static_m_cap=True`) в `BF16` с калибровкой темпа и периодическим сохранением
     чекпоинта, затем двухгрупповой `STE QAT` -> `5,539.81 KB` (`5.41 MB`) `2-bit LUT`
     (`95` слоёв `16.00x`), и прогон `Track 2` (удерживаемая многоязычная проза, `6` веб-режимов
     Фазы E при `0.00%` разрывов UTF-8 и когнитивные наборы `S_5 / Overwrite / Trap`).
  4. `eval-cram`: Стыковка обученного `Tier-1` (`d_model=384`) с `SphericalBase10RegisterALU + CRAM+z`
     на `GSM8K` (`train[:6000]` -> `Cal[6000:6736]` -> `BlindDev[6736:]` -> однократный
     `Sealed Test` `N=1,319`, ускоренный через `cuda_ext`!).
  5. `all`: Выполняет все стадии последовательно и собирает итоговый
     `benchmarks/phase_g_colab_scaleup_results.json` + чекпоинты `lut.pt` (~6.8 MB) и `bf16.pt`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import glob
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import sys
import time
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.cram_gsm8k.cram_open_graph import OpenGraphCRAMSolver  # noqa: E402
from experiments.cram_gsm8k.slot_extractor import (  # noqa: E402
    TrainInducedQuantityExtractor,
)
from experiments.cram_gsm8k.spherical_alu import SphericalBase10RegisterALU  # noqa: E402
from experiments.cram_gsm8k.system1_encoder import (  # noqa: E402
    Calcite5MathSystem1,
    MAX_SLOTS,
    encode_interleaved_alu_trace,
)
from experiments.v5_proto.probe_phase_b_byte_hnet import (  # noqa: E402
    compute_bpe_to_byte_scale,
    evaluate_byte_prose_and_ratio,
    measure_crystal_drift_at_L,
)
from kernels.fast_packed_lut import (  # noqa: E402
    accelerate_lut_model,
    get_lut_backend_info,
    pack_fast_model_2bit_lut,
    save_packed_lut_checkpoint,
)
from run_phase_b_byte_benchmark import _eval_cognitive_suites_b  # noqa: E402
from run_phase_e_fast_utf8_benchmark import (  # noqa: E402
    build_disjoint_openweb_corpora,
    evaluate_openweb_regimes,
)
from run_phase_f_utf8_cram_benchmark import (  # noqa: E402
    SingleShotTestVault,
    evaluate_phase_f_split,
)
from run_simvq_split_benchmark import compute_state_cache_kb, count_params  # noqa: E402
from simvq_split.memory import compute_chunker_state_bytes  # noqa: E402
from simvq_split.models import SimVQSplitConfig, SimVQSplitLM  # noqa: E402
from simvq_split.quant_lut import (  # noqa: E402
    PackedLUT2BitLinear,
    compute_2bit_lut_footprint,
    pack_model_2bit_lut,
    quantize_ternary_weights_ls,
)
from simvq_split.tasks import (  # noqa: E402
    build_eval_overwrite_suite,
    build_eval_s5_suite,
    build_eval_trap_suite,
    load_gsm8k_corpora,
    make_overwrite_story,
    make_s5_story,
    make_trap_overwrite_story,
)
from simvq_split.tokenizer import ByteBPETokenizer  # noqa: E402


TIER1_CONFIG_DICT: Dict[str, object] = {
    "vocab_size": 259,
    "d_model": 384,
    "n_heads": 6,
    "d_ff": None,
    "n_layers_pre": 4,
    "n_layers_post": 4,
    "window_size": 12,
    "num_spheres": 24,
    "d_sphere": 32,
    "codebook_size": 32,
    "mem_heads": 5,
    "mem_dim": 32,
    "ridge_lambda": 0.04,
    "use_crystal": True,
    "use_woodbury": True,
    "use_byte_chunker": True,
    "byte_char_dim": 32,
    "byte_d_ff": 128,
    "byte_router_char_dim": 16,
    "byte_router_hidden": 96,
    "byte_commit_scale": 0.45,
    "router_lock_frac": 0.135,
    "target_chunk_bytes": 4.6,
    "max_chunk_bytes": 10,
    "byte_chunk_mode": "utf8",
    "max_utf8_bytes": 16,
    "use_static_m_cap": True,
    "ratio_loss_scale": 1.5,
    "pad_token_id": 0,
}


def tier1_config() -> SimVQSplitConfig:
    return SimVQSplitConfig(**TIER1_CONFIG_DICT)  # type: ignore[arg-type]


def _normalize_text_for_leak_check(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def build_heldout_substring_blocklist(repo_root: Path, win_len: int = 45) -> set[str]:
    """Строит хеш-набор 45-символьных подстрок всех отложенных сплитов для нулевой утечки."""
    block: set[str] = set()

    def _add_doc(txt: str) -> None:
        norm = _normalize_text_for_leak_check(txt)
        if len(norm) < win_len:
            if len(norm) >= 24:
                block.add(norm)
            return
        for i in range(0, len(norm) - win_len + 1, 15):
            block.add(norm[i : i + win_len])

    tr_p = repo_root / "benchmarks" / "gsm8k" / "train.jsonl"
    te_p = repo_root / "benchmarks" / "gsm8k" / "test.jsonl"
    with open(tr_p, "r", encoding="utf-8") as f:
        tr_rows = [json.loads(line) for line in f if line.strip()]
    for obj in tr_rows[6000:]:  # Cal + BlindDev
        _add_doc(str(obj["question"]))
    with open(te_p, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            _add_doc(str(obj["question"]))

    arc_ev = sorted(
        glob.glob(str(repo_root / "benchmarks" / "arc_agi" / "ARC-AGI-master" / "data" / "evaluation" / "*.json"))
    )
    for fp in arc_ev[:40]:
        _add_doc(Path(fp).read_text(encoding="utf-8")[:400])
    return block


_SHORT_BLOCK_CACHE: tuple[int, int, tuple[str, ...]] = (0, -1, ())


def is_leaked_doc(text: str, blocklist: set[str], win_len: int = 45) -> bool:
    global _SHORT_BLOCK_CACHE
    norm = _normalize_text_for_leak_check(text)
    if len(norm) < 24:
        return False
    if norm in blocklist:
        return True
    blk_id, blk_len, short_terms = _SHORT_BLOCK_CACHE
    cur_len = len(blocklist)
    if blk_id != id(blocklist) or blk_len != cur_len:
        short_terms = tuple(s for s in blocklist if len(s) < win_len)
        _SHORT_BLOCK_CACHE = (id(blocklist), cur_len, short_terms)
    if short_terms and any(s in norm for s in short_terms):
        return True
    if len(norm) < win_len:
        return False
    for i in range(0, len(norm) - win_len + 1):
        if norm[i : i + win_len] in blocklist:
            return True
    return False


@torch.no_grad()
def _bench_cuda_op(mod, x: torch.Tensor, device: str, warmup: int = 3, repeats: int = 12) -> float:
    has_cuda = device == "cuda" and torch.cuda.is_available()
    for _ in range(warmup):
        _ = mod(x)
    if has_cuda:
        torch.cuda.synchronize()
    times: List[float] = []
    for _ in range(repeats):
        if has_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = mod(x)
        if has_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return round(statistics.median(times), 3)


def run_stage_track1(repo_root: Path, out_dir: Path, device: str) -> Dict[str, object]:
    """Track 1: компиляция CUDA/CPU ядер, проверка точности (<1e-4) и ускорения."""
    print("\n=== [Stage 1 / Track 1] Native C++/CUDA PackedLUT2BitLinear Benchmark ===", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    binfo = get_lut_backend_info()
    print("Backend status:", binfo, flush=True)

    # 1. Layer-level parity & speed (row_aligned=True и row_aligned=False)
    torch.manual_seed(2026)
    layer_results: Dict[str, Dict[str, object]] = {}
    for lname, in_f, out_f, bias_on in (
        ("mlp_384_to_1024_aligned", 384, 1024, False),
        ("mlp_1024_to_384_aligned", 1024, 384, False),
        ("woodbury_172_to_384_unaligned", 172, 384, True),
    ):
        lin = nn.Linear(in_f, out_f, bias=bias_on)
        with torch.no_grad():
            wt, _, _ = quantize_ternary_weights_ls(lin.weight.data)
            lin.weight.data.copy_(wt)
        ref_l = PackedLUT2BitLinear.from_linear(lin).to(device).eval()
        vec_l = accelerate_lut_model(ref_l, backend="pytorch_vec").to(device).eval()
        x = torch.randn(16, 128, in_f, device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
        x_f32 = x.float()

        with torch.no_grad():
            y_ref_f32 = ref_l(x_f32)
            y_vec_f32 = vec_l(x_f32)
            vec_diff = float((y_ref_f32 - y_vec_f32).abs().max().item())
            y_ref_bf = ref_l(x).float()
            cuda_diff = -1.0
            cuda_bf16_diff = -1.0
            cuda_ms = -1.0
            if device == "cuda" and binfo["cuda_ext_available"]:
                cu_l = accelerate_lut_model(ref_l, backend="cuda_ext").to(device).eval()
                y_cu_f32 = cu_l(x_f32)
                cuda_diff = float((y_ref_f32 - y_cu_f32).abs().max().item())
                cuda_bf16_diff = float((y_ref_bf - cu_l(x).float()).abs().max().item())
                assert cuda_diff < 1e-4, f"CUDA FP32 parity failed on {lname}: {cuda_diff}"
                assert cuda_bf16_diff <= 0.03125, f"CUDA BF16 parity exceeded 1 ULP on {lname}: {cuda_bf16_diff}"
                cuda_ms = _bench_cuda_op(cu_l, x, device)
            assert vec_diff < 1e-4, f"pytorch_vec parity failed on {lname}: {vec_diff}"

        ref_ms = _bench_cuda_op(ref_l, x, device)
        vec_ms = _bench_cuda_op(vec_l, x, device)
        fp16_l = lin.to(device).to(x.dtype).eval()
        fp16_ms = _bench_cuda_op(fp16_l, x, device)

        speedup_cuda = round(ref_ms / cuda_ms, 2) if cuda_ms > 0 else 0.0
        layer_results[lname] = {
            "in_features": in_f,
            "out_features": out_f,
            "row_aligned": bool(ref_l.row_aligned),
            "vec_max_abs_diff": round(vec_diff, 8),
            "cuda_max_abs_diff": round(cuda_diff, 8) if cuda_diff >= 0 else None,
            "cuda_bf16_max_abs_diff": round(cuda_bf16_diff, 8) if cuda_bf16_diff >= 0 else None,
            "ref_py_ms": ref_ms,
            "pytorch_vec_ms": vec_ms,
            "cuda_ext_ms": cuda_ms if cuda_ms > 0 else None,
            "fp16_cublas_ms": fp16_ms,
            "cuda_speedup_vs_ref": speedup_cuda,
        }
        print(
            f"  [Layer {lname}] ref={ref_ms:.3f}ms | vec={vec_ms:.3f}ms (diff={vec_diff:.2e}) | "
            f"cuda={cuda_ms:.3f}ms (f32={cuda_diff:.2e}, bf16={cuda_bf16_diff:.4f}, {speedup_cuda:.2f}x)",
            flush=True,
        )

    # 2. Full-model end-to-end parity & speedup: Base (1.48M) и Tier-1 (17.49M)
    model_results: Dict[str, Dict[str, object]] = {}
    base_ckpt_p = repo_root / "checkpoints" / "simvq_split_utf8_openweb.pt"
    ckpt_b = torch.load(base_ckpt_p, map_location="cpu", weights_only=False)
    cfg_base = SimVQSplitConfig(**ckpt_b["simvq_split_config"])
    m_base = SimVQSplitLM(cfg_base).eval()
    m_base.load_state_dict(ckpt_b["simvq_split_ternary_qat_state_dict"], strict=True)
    m_base.byte_chunker.lock_router(True)
    m_base.set_crystal_schedule(1, 1)

    cfg_t1 = tier1_config()
    m_t1 = SimVQSplitLM(cfg_t1).eval()
    m_t1.byte_chunker.lock_router(True)
    m_t1.set_crystal_schedule(1, 1)
    with torch.no_grad():
        for nm, mod in m_t1.named_modules():
            if isinstance(mod, nn.Linear) and nm != "lm_head":
                wt, _, _ = quantize_ternary_weights_ls(mod.weight.data)
                mod.weight.data.copy_(wt)

    for tag, qat_src in (("Base_1.48M_548KB", m_base), ("Tier1_17.49M_5.41MB", m_t1)):
        ref_m = pack_model_2bit_lut(qat_src).to(device).eval()
        fp_info = compute_2bit_lut_footprint(ref_m)
        vec_m = accelerate_lut_model(ref_m, backend="pytorch_vec").to(device).eval()
        cu_m = (
            accelerate_lut_model(ref_m, backend="cuda_ext").to(device).eval()
            if (device == "cuda" and binfo["cuda_ext_available"])
            else None
        )

        # Parity check at B=4, L=512
        xb_par = torch.randint(3, 259, (4, 512), device=device)
        with torch.no_grad():
            y_ref = ref_m(xb_par).float()
            y_vec = vec_m(xb_par).float()
            d_vec = float((y_ref - y_vec).abs().max().item())
            d_cu = -1.0
            if cu_m is not None:
                y_cu = cu_m(xb_par).float()
                d_cu = float((y_ref - y_cu).abs().max().item())
                assert d_cu < 1e-4, f"Full model CUDA parity failed for {tag}: {d_cu}"
            assert d_vec < 1e-4, f"Full model vec parity failed for {tag}: {d_vec}"

        shapes_bench = ((1, 512), (16, 256), (16, 768), (16, 2048))
        qat_dev = qat_src.to(device).eval()
        lat_map: Dict[str, Dict[str, object]] = {}
        for bs, slen in shapes_bench:
            xb = torch.randint(3, 259, (bs, slen), device=device)
            ms_ref = _bench_cuda_op(ref_m, xb, device, warmup=1, repeats=4)
            ms_vec = _bench_cuda_op(vec_m, xb, device, warmup=1, repeats=4)
            ms_cu = (
                _bench_cuda_op(cu_m, xb, device, warmup=2, repeats=8)
                if cu_m is not None
                else -1.0
            )
            ms_fp16 = _bench_cuda_op(qat_dev, xb, device, warmup=2, repeats=6)
            sp_cu = round(ms_ref / ms_cu, 2) if ms_cu > 0 else 0.0
            lat_map[f"B{bs}_L{slen}"] = {
                "ref_py_ms": ms_ref,
                "pytorch_vec_ms": ms_vec,
                "cuda_ext_ms": ms_cu if ms_cu > 0 else None,
                "qat_fp32_ms": ms_fp16,
                "cuda_speedup_vs_ref": sp_cu,
                "cuda_bytes_per_sec": round((bs * slen) / (ms_cu / 1000.0), 1) if ms_cu > 0 else None,
            }
            print(
                f"  [{tag} B={bs:2d} L={slen:4d}] ref={ms_ref:7.2f}ms | vec={ms_vec:7.2f}ms | "
                f"cuda={ms_cu:6.2f}ms ({sp_cu:5.2f}x)",
                flush=True,
            )

        # CPU C++ OpenMP micro-benchmark at B=1, L=256
        cpu_res: Dict[str, object] = {}
        if binfo["cpu_ext_available"]:
            ref_cpu = pack_model_2bit_lut(qat_src.cpu()).cpu().eval()
            c_ext = accelerate_lut_model(ref_cpu, backend="cpu_ext").cpu().eval()
            xb_c = torch.randint(3, 259, (1, 256))
            with torch.no_grad():
                d_cpu = float((ref_cpu(xb_c).float() - c_ext(xb_c).float()).abs().max().item())
            ms_c_ref = _bench_cuda_op(ref_cpu, xb_c, "cpu", warmup=1, repeats=2)
            ms_c_ext = _bench_cuda_op(c_ext, xb_c, "cpu", warmup=1, repeats=3)
            cpu_res = {
                "cpu_max_abs_diff": round(d_cpu, 8),
                "cpu_ref_py_ms": ms_c_ref,
                "cpu_ext_ms": ms_c_ext,
                "cpu_speedup_vs_ref": round(ms_c_ref / max(1e-3, ms_c_ext), 2),
            }
            print(
                f"  [{tag} CPU B=1 L=256] ref={ms_c_ref:.2f}ms | cpu_ext={ms_c_ext:.2f}ms "
                f"({cpu_res['cpu_speedup_vs_ref']:.2f}x, diff={d_cpu:.2e})",
                flush=True,
            )

        model_results[tag] = {
            "lut_footprint": fp_info,
            "vec_max_abs_diff": round(d_vec, 8),
            "cuda_max_abs_diff": round(d_cu, 8) if d_cu >= 0 else None,
            "latencies": lat_map,
            "cpu_b1_l256": cpu_res,
        }
        qat_src.cpu()
        del ref_m, vec_m, cu_m
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    res = {
        "backend_info": binfo,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "layers": layer_results,
        "models": model_results,
    }
    (out_dir / "track1_cuda_lut.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def _encode_doc_uint16(text: str) -> np.ndarray:
    b = text.encode("utf-8", errors="ignore")
    if not b:
        return np.empty(0, dtype=np.uint16)
    arr = np.empty(len(b) + 2, dtype=np.uint16)
    arr[0] = 1  # BOS
    arr[1:-1] = np.frombuffer(b, dtype=np.uint8).astype(np.uint16) + 3
    arr[-1] = 2  # EOS
    return arr


def _stream_hf_parquet_docs(
    repo_candidates: Sequence[tuple[str, str | None]],
    text_columns: Sequence[str],
    target_bytes: int,
    blocklist: set[str],
    domain_name: str,
) -> tuple[List[np.ndarray], int, str]:
    """Скачивает несблокированные Parquet-шарды через `huggingface_hub` + `pyarrow` без утечки теста."""
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    api = HfApi()
    chosen_repo = "none"
    chunks: List[np.ndarray] = []
    collected_bytes = 0
    scrubbed = 0

    for repo_id, prefix_hint in repo_candidates:
        try:
            files = [
                f
                for f in api.list_repo_files(repo_id=repo_id, repo_type="dataset")
                if f.endswith(".parquet") and (prefix_hint is None or prefix_hint in f)
            ]
            files.sort()
            if not files:
                continue
            print(f"  [{domain_name}] Using {repo_id} ({len(files)} parquet shards available)", flush=True)
            chosen_repo = repo_id
            for rel_p in files[:8]:
                if collected_bytes >= target_bytes:
                    break
                local_pq = hf_hub_download(repo_id=repo_id, filename=rel_p, repo_type="dataset")
                pf = pq.ParquetFile(local_pq)
                avail_cols = [c for c in text_columns if c in pf.schema_arrow.names]
                if not avail_cols:
                    # Fallback to first string column
                    for field in pf.schema_arrow:
                        if "string" in str(field.type).lower() or "utf8" in str(field.type).lower():
                            avail_cols.append(field.name)
                            break
                if not avail_cols:
                    continue
                for batch in pf.iter_batches(batch_size=2048, columns=avail_cols):
                    py_dict = batch.to_pydict()
                    n_rows = len(py_dict[avail_cols[0]])
                    for i in range(n_rows):
                        parts = [str(py_dict[c][i]) for c in avail_cols if py_dict[c][i]]
                        txt = "\n".join(parts).strip()
                        if len(txt) < 80:
                            continue
                        if len(txt) > 24000:
                            txt = txt[:24000]
                        if is_leaked_doc(txt, blocklist):
                            scrubbed += 1
                            continue
                        enc = _encode_doc_uint16(txt)
                        chunks.append(enc)
                        collected_bytes += int(enc.size)
                        if collected_bytes >= target_bytes:
                            break
                    if collected_bytes >= target_bytes:
                        break
                # Remove huge cached parquet shard to keep /content cache tidy
                try:
                    Path(local_pq).unlink(missing_ok=True)
                except OSError:
                    pass
            if collected_bytes >= int(0.45 * target_bytes):
                break
        except Exception as exc:
            print(f"  [{domain_name}] Candidate {repo_id} fallback ({type(exc).__name__}: {exc})", flush=True)
            continue
    return chunks, scrubbed, f"{chosen_repo} ({collected_bytes / (1024**2):.1f} MB, scrubbed={scrubbed})"


def run_stage_build_corpus(repo_root: Path, data_dir: Path, total_gb: float = 1.80) -> Dict[str, object]:
    """Собирает `~1.80 GB` непрерывный UTF-8 предобучающий корпус + `24 MB` held-out val."""
    print(f"\n=== [Stage 2 / Corpus] Building {total_gb:.2f} GB Zero-Leakage UTF-8 Stream ===", flush=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    train_bin = data_dir / "train_utf8.u16.bin"
    val_bin = data_dir / "val_utf8.u16.bin"
    meta_p = data_dir / "corpus_meta.json"
    if train_bin.exists() and val_bin.exists() and meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if abs(meta.get("train_tokens_mb", 0) - total_gb * 1024) < 256:
            print("Corpus already built; skipping rebuild.", flush=True)
            return meta

    t0 = time.perf_counter()
    blocklist = build_heldout_substring_blocklist(repo_root)
    print(f"Loaded {len(blocklist):,} held-out 45-char leakage hashes (Cal + BlindDev + SealedTest + Eval).", flush=True)

    target_total = int(total_gb * 1024 * 1024 * 1024)
    val_total = 24 * 1024 * 1024

    # 1. English Edu Prose (~44% -> ~792 MB)
    en_docs, sc_en, s_en = _stream_hf_parquet_docs(
        [
            ("HuggingFaceFW/fineweb-edu", "sample/10BT/"),
            ("wikimedia/wikipedia", "20231101.en/"),
        ],
        ["text"],
        int(0.44 * (target_total + val_total)),
        blocklist,
        "1_FineWeb_EN",
    )

    # 2. Math & Reasoning (~25% -> ~450 MB) + GSM8K train[:6000] & Interleaved ALU traces (~36 MB)
    math_docs, sc_math, s_math = _stream_hf_parquet_docs(
        [
            ("HuggingFaceTB/finemath", "finemath-4plus"),
            ("open-web-math/open-web-math", None),
            ("meta-math/MetaMathQA", None),
        ],
        ["text", "query", "response", "problem", "solution"],
        int(0.24 * (target_total + val_total)),
        blocklist,
        "2_FineMath",
    )

    # Local zero-leak train[:6000] GSM8K + ALU interleaved traces + clean synthetic arithmetic
    tr_p = repo_root / "benchmarks" / "gsm8k" / "train.jsonl"
    with open(tr_p, "r", encoding="utf-8") as f:
        tr_train6000 = [json.loads(line) for line in f if line.strip()][:6000]
    ext = TrainInducedQuantityExtractor()
    ext.fit(tr_train6000)
    alu = SphericalBase10RegisterALU(d_sub=15, num_digits=12, seed=3301)

    local_math_bytes = 0
    local_math_target = 32 * 1024 * 1024
    rng_m = random.Random(7026)
    gsm_pool_enc: List[np.ndarray] = []
    for it in tr_train6000:
        q_str = str(it["question"]).strip()
        a_str = str(it["answer"]).strip()
        gsm_pool_enc.append(_encode_doc_uint16(f"Problem: {q_str}\nSolution: {a_str}"))
        slots, _, _ = ext.extract_slots(q_str, include_aux_consts=True)
        usable_n = min(len(slots), MAX_SLOTS)
        if usable_n >= 1:
            steps, _, ok = ext.compile_train_dag(slots[:usable_n], a_str)
            if ok and steps:
                tr_str, _, _, inv = encode_interleaved_alu_trace(
                    steps, [float(sl.val) for sl in slots[:usable_n]], alu, usable_n
                )
                if (not inv) and tr_str:
                    s_desc = " ".join(f"{chr(65+i)}={slots[i].val:g}" for i in range(usable_n))
                    gsm_pool_enc.append(_encode_doc_uint16(f"Problem: {q_str}\nSlots: {s_desc}\nTrace: {tr_str}"))
    while local_math_bytes < local_math_target and gsm_pool_enc:
        arr = rng_m.choice(gsm_pool_enc)
        math_docs.append(arr)
        local_math_bytes += int(arr.size)

    # 3. Multilingual UTF-8: Russian (~15.5% -> ~280 MB), CJK (~3.0% -> ~55 MB), Emoji/Math Symbols (~1.0%)
    ru_docs, sc_ru, s_ru = _stream_hf_parquet_docs(
        [("wikimedia/wikipedia", "20231101.ru/")],
        ["text"],
        int(0.155 * (target_total + val_total)),
        blocklist,
        "3_Russian_Wiki",
    )
    zh_docs, sc_zh, s_zh = _stream_hf_parquet_docs(
        [("wikimedia/wikipedia", "20231101.zh/")],
        ["text"],
        int(0.03 * (target_total + val_total)),
        blocklist,
        "4_CJK_Wiki",
    )

    # Disjoint 4-byte UTF-8 technical notes & Unicode math/emoji (none from Phase E eval lists)
    em_palette = list("🌍🌌🔭🛸🛰🧪🧬🧲🧭📡🔩🧰📐📏🖊📒🗂📅🕒🔑🔋💡🧮🧱")
    math_syms = list("∑∏∫√∞≈≠≤≥∈∉∪∩→⇒∀∃∂∇λΔΩ")
    u4_docs: List[np.ndarray] = []
    u4_bytes = 0
    rng_u4 = random.Random(8026)
    while u4_bytes < 16 * 1024 * 1024:
        e_seq = "".join(rng_u4.choice(em_palette) for _ in range(rng_u4.randint(4, 12)))
        m_seq = " ".join(rng_u4.choice(math_syms) for _ in range(rng_u4.randint(3, 8)))
        txt = f"Telemetry {e_seq} :: state={rng_u4.randint(10, 999)} | eq: {m_seq}\n{e_seq}"
        arr = _encode_doc_uint16(txt)
        u4_docs.append(arr)
        u4_bytes += int(arr.size)

    # 4. Python Code & Structured JSON (~11.5% -> ~207 MB)
    code_target = int(0.115 * (target_total + val_total))
    code_docs, sc_code, s_code = _stream_hf_parquet_docs(
        [
            ("bigcode/the-stack-smol", "data/python/"),
            ("iamtarun/python_code_instructions_18k_alpaca", None),
            ("FlyTech/python-codes-25k", None),
        ],
        ["content", "code", "output", "text"],
        int(0.75 * code_target),
        blocklist,
        "5_Python_Code",
    )
    # Add Colab site-packages & stdlib Python sources + training-only ARC JSON to reach code_target cleanly
    py_paths: List[Path] = []
    for sp_dir in sys.path:
        p = Path(sp_dir)
        if p.is_dir():
            for cand in p.glob("*/*.py"):
                if "test" not in str(cand).lower() and "cram_gsm8k" not in str(cand).lower():
                    py_paths.append(cand)
                    if len(py_paths) > 12000:
                        break
    arc_tr = sorted(
        Path(x)
        for x in glob.glob(
            str(repo_root / "benchmarks" / "arc_agi" / "ARC-AGI-master" / "data" / "training" / "*.json")
        )
    )
    rng_c = random.Random(8126)
    rng_c.shuffle(py_paths)
    cur_code_b = sum(int(d.size) for d in code_docs)
    for fp in arc_tr + py_paths:
        if cur_code_b >= code_target:
            break
        try:
            txt = fp.read_text(encoding="utf-8", errors="ignore")[:20000]
            if len(txt) < 60 or is_leaked_doc(txt, blocklist):
                continue
            arr = _encode_doc_uint16(txt)
            code_docs.append(arr)
            cur_code_b += int(arr.size)
        except OSError:
            continue

    # Split each domain deterministically into disjoint Train vs Val (last ~2% -> val, rest -> train)
    all_pools = [
        ("en_edu", en_docs),
        ("math", math_docs),
        ("ru_wiki", ru_docs),
        ("zh_wiki", zh_docs),
        ("u4_sym", u4_docs),
        ("code_json", code_docs),
    ]
    rng_mix = random.Random(2026)
    train_chunks: List[np.ndarray] = []
    val_chunks: List[np.ndarray] = []
    domain_stats: Dict[str, Dict[str, float]] = {}
    for dname, pool in all_pools:
        if not pool:
            domain_stats[dname] = {"docs": 0, "train_mb": 0.0, "val_mb": 0.0}
            continue
        rng_mix.shuffle(pool)
        n_val = max(1, min(len(pool) // 40, 2500))
        v_p = pool[:n_val]
        t_p = pool[n_val:]
        val_chunks.extend(v_p)
        train_chunks.extend(t_p)
        domain_stats[dname] = {
            "docs": len(pool),
            "train_mb": round(sum(int(a.size) for a in t_p) / (1024**2), 2),
            "val_mb": round(sum(int(a.size) for a in v_p) / (1024**2), 2),
        }

    rng_mix.shuffle(train_chunks)
    rng_mix.shuffle(val_chunks)

    def _write_bin(path: Path, arrs: Sequence[np.ndarray], max_elems: int) -> int:
        written = 0
        with open(path, "wb") as f:
            buf: List[np.ndarray] = []
            buf_n = 0
            for a in arrs:
                if written + buf_n >= max_elems:
                    break
                buf.append(a)
                buf_n += int(a.size)
                if buf_n >= 4 * 1024 * 1024:
                    cat = np.concatenate(buf)
                    take = min(int(cat.size), max_elems - written)
                    f.write(cat[:take].tobytes())
                    written += take
                    buf.clear()
                    buf_n = 0
            if buf and written < max_elems:
                cat = np.concatenate(buf)
                take = min(int(cat.size), max_elems - written)
                f.write(cat[:take].tobytes())
                written += take
        return written

    n_val_toks = _write_bin(val_bin, val_chunks, val_total)
    n_train_toks = _write_bin(train_bin, train_chunks, target_total)
    total_scrubbed = sc_en + sc_math + sc_ru + sc_zh + sc_code

    meta = {
        "train_bin_path": str(train_bin),
        "val_bin_path": str(val_bin),
        "train_tokens": int(n_train_toks),
        "train_tokens_mb": round(n_train_toks / (1024**2), 2),
        "val_tokens": int(n_val_toks),
        "val_tokens_mb": round(n_val_toks / (1024**2), 2),
        "heldout_blocklist_hashes": len(blocklist),
        "scrubbed_doc_count": int(total_scrubbed),
        "sources": {
            "en": s_en,
            "math": s_math,
            "ru": s_ru,
            "zh": s_zh,
            "code": s_code,
        },
        "domain_stats": domain_stats,
        "build_seconds": round(time.perf_counter() - t0, 2),
    }
    meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"Corpus DONE in {meta['build_seconds']:.1f}s: train={meta['train_tokens_mb']:.1f} MB "
        f"({n_train_toks:,} byte tokens) | val={meta['val_tokens_mb']:.1f} MB | scrubbed={total_scrubbed}",
        flush=True,
    )
    return meta


class CognitiveEpisodeGenerator:
    """Генерирует <BOS>-выровненные учебные эпизоды `S_5 / Overwrite / Trap` только из `tr_sents`."""

    def __init__(self, tr_sents: Sequence[str], tok: ByteBPETokenizer, seq_len: int, seed: int = 9026):
        self.tr_sents = list(tr_sents)
        self.tok = tok
        self.seq_len = int(seq_len)
        self.rng = random.Random(seed)
        self._buf_x: List[List[int]] = []
        self._buf_y: List[List[int]] = []

    def sample_batch(self, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        xb = torch.full((batch_size, self.seq_len), self.tok.PAD_ID, dtype=torch.long)
        yb = torch.full((batch_size, self.seq_len), self.tok.PAD_ID, dtype=torch.long)
        tw = self.seq_len + 1
        for i in range(batch_size):
            u = self.rng.random()
            while True:
                if u < 0.56:
                    k = self.rng.choice((0, 1, 2, 3, 4))
                    d_r = (2, 2) if k == 0 else ((1, 1) if k == 1 else (0, 1))
                    p, t, _ = make_s5_story(self.rng, k, self.tr_sents, d_r)
                elif u < 0.86:
                    p, t, _ = make_overwrite_story(
                        self.rng, self.rng.randint(3, 5), self.rng.randint(1, 2), self.tr_sents, 2, 0.25
                    )
                else:
                    p, t, _ = make_trap_overwrite_story(
                        self.rng, self.rng.randint(3, 5), self.rng.randint(1, 2), self.tr_sents, 2, 0.25
                    )
                p_ids = self.tok.encode(p, add_bos=True, add_eos=False)
                f_ids = self.tok.encode(p + t, add_bos=True, add_eos=True)
                if len(f_ids) <= tw and f_ids[: len(p_ids)] == p_ids:
                    L = len(f_ids)
                    xb[i, : L - 1] = torch.tensor(f_ids[:-1], dtype=torch.long)
                    p_len = len(p_ids)
                    yb[i, p_len - 1 : L - 1] = torch.tensor(f_ids[p_len:], dtype=torch.long)
                    break
        return xb.to(device), yb.to(device)


def _sample_memmap_batch(
    mm: np.memmap,
    batch_size: int,
    seq_len: int,
    rng: np.random.Generator,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Сэмплирует батч из `uint16` memmap с выравниванием старта на границу символа UTF-8."""
    max_start = len(mm) - seq_len - 6
    starts = rng.integers(0, max_start, size=batch_size)
    rows = np.empty((batch_size, seq_len + 1), dtype=np.int64)
    for r_idx, st in enumerate(starts):
        s = int(st)
        # Skip orphaned UTF-8 continuation bytes 0x80..0xBF (token id = byte + 3)
        for _ in range(3):
            tok_v = int(mm[s])
            if tok_v >= 3 and ((tok_v - 3) & 0xC0) == 0x80:
                s += 1
            else:
                break
        rows[r_idx] = mm[s : s + seq_len + 1]
    t = torch.from_numpy(rows).to(device=device, dtype=torch.long)
    return t[:, :-1], t[:, 1:]


@torch.no_grad()
def eval_val_memmap_bpb(
    model: nn.Module,
    val_mm: np.memmap,
    seq_len: int,
    n_batches: int,
    batch_size: int,
    device: str,
) -> float:
    model.eval()
    rng = np.random.default_rng(42)
    nlls: List[float] = []
    use_amp = device == "cuda" and torch.cuda.is_available()
    is_lut = any(isinstance(m, PackedLUT2BitLinear) for m in model.modules())
    for _ in range(n_batches):
        xb, yb = _sample_memmap_batch(val_mm, batch_size, seq_len, rng, device)
        if use_amp and not is_lut:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                lg = model(xb)
        else:
            lg = model(xb)
        ce = F.cross_entropy(lg.float().transpose(1, 2), yb, ignore_index=0)
        nlls.append(float(ce.item()))
    return round(float(statistics.mean(nlls)) / math.log(2.0), 4)


def run_stage_train_tier1(
    repo_root: Path,
    data_dir: Path,
    ckpt_dir: Path,
    out_dir: Path,
    device: str,
    max_steps: int = 12000,
    batch_size: int = 32,
    seq_len: int = 1536,
    qat_steps: int = 600,
    max_train_minutes: float = 95.0,
) -> Dict[str, object]:
    """Обучает `Calcite-5 Tier-1` (`17.49M`, BF16 -> `5.41 MB` 2-bit LUT) и снимает полные метрики."""
    print(
        f"\n=== [Stage 3 / Train Tier-1 17.49M] B={batch_size} L={seq_len} "
        f"max_steps={max_steps} qat={qat_steps} budget={max_train_minutes:.0f}m ===",
        flush=True,
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_mm = np.memmap(data_dir / "train_utf8.u16.bin", dtype=np.uint16, mode="r")
    val_mm = np.memmap(data_dir / "val_utf8.u16.bin", dtype=np.uint16, mode="r")

    tok = ByteBPETokenizer(merges=[])
    tr_docs, te_docs, tr_sents, te_sents = load_gsm8k_corpora(repo_root)
    byte_per_bpe = compute_bpe_to_byte_scale(repo_root, te_docs, tr_docs, tr_sents)

    cfg = tier1_config()
    torch.manual_seed(100)
    model = SimVQSplitLM(cfg).to(device)
    n_params = count_params(model)
    cache_kb = round(compute_state_cache_kb("SimVQSplitLM", cfg, None, 2048), 2)
    print(f"Tier-1 params={n_params:,} | O(1) cache@2048={cache_kb:.2f} KB", flush=True)

    # Speed & VRAM calibration (8 warmup steps) to fit time budget cleanly
    opt_cal = torch.optim.AdamW(model.parameters(), lr=1.2e-3, betas=(0.9, 0.95), weight_decay=0.02)
    rng_np = np.random.default_rng(2026)
    use_amp = device == "cuda" and torch.cuda.is_available()
    t_cal_list: List[float] = []
    for cs in range(8):
        xb, yb = _sample_memmap_batch(train_mm, batch_size, seq_len, rng_np, device)
        model.train()
        model.set_crystal_schedule(cs + 1, 1000)
        opt_cal.zero_grad(set_to_none=True)
        if use_amp:
            torch.cuda.synchronize()
        ts = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            lg = model(xb)
            ce = F.cross_entropy(lg.transpose(1, 2), yb, ignore_index=tok.PAD_ID)
            loss = ce + 0.04 * model.get_commitment_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_cal.step()
        if use_amp:
            torch.cuda.synchronize()
        if cs >= 2:
            t_cal_list.append(time.perf_counter() - ts)
    del opt_cal
    sec_per_step = statistics.median(t_cal_list)
    peak_vram_gb = (torch.cuda.max_memory_allocated() / (1024**3)) if use_amp else 0.0

    # Reserve ~15 min for QAT + evals
    fp_budget_sec = max(600.0, (max_train_minutes - 15.0) * 60.0)
    est_steps = int(fp_budget_sec / max(0.05, sec_per_step))
    total_steps = max(2500, min(max_steps, est_steps))
    print(
        f"Calibration: {sec_per_step*1000:.1f} ms/step | peak_VRAM={peak_vram_gb:.2f} GB | "
        f"scheduled total_steps={total_steps:,} (~{total_steps * sec_per_step / 60.0:.1f} min, "
        f"{total_steps * batch_size * seq_len / 1e9:.2f} GB bytes seen)",
        flush=True,
    )

    # Fresh init for clean training trajectory + resume support if interrupted
    latest_ckpt = ckpt_dir / "tier1_resume_latest.pt"
    torch.manual_seed(100)
    model = SimVQSplitLM(cfg).to(device)
    base_lr = 1.2e-3
    min_lr = 1.2e-4
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.95), weight_decay=0.02)
    start_step = 1
    if latest_ckpt.exists():
        st_pay = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        if int(st_pay.get("step", 0)) < total_steps and st_pay.get("total_steps") == total_steps:
            model.load_state_dict(st_pay["model"], strict=True)
            opt.load_state_dict(st_pay["opt"])
            start_step = int(st_pay["step"]) + 1
            print(f"Resumed Tier-1 FP training from step {start_step}/{total_steps}", flush=True)

    cog_gen = CognitiveEpisodeGenerator(tr_sents, tok, seq_len=min(768, seq_len), seed=9026)
    rng_py = random.Random(2026 + start_step)
    t_fp0 = time.perf_counter()
    warmup_steps = max(200, min(500, total_steps // 16))
    bytes_seen = int((start_step - 1) * batch_size * seq_len)

    loss_log: List[Dict[str, float]] = []
    for step in range(start_step, total_steps + 1):
        model.train()
        model.set_crystal_schedule(step, total_steps)
        if step <= warmup_steps:
            lr = base_lr * (step / float(warmup_steps))
        else:
            prog = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr

        # 88% full open-web/math/RU/CJK/code L=seq_len stream + 12% t=0 cognitive episodes (zero-leak)
        # 82% full open-web/math/RU/CJK/code L=seq_len stream + 18% t=0 cognitive episodes (zero-leak)
        if rng_py.random() < 0.82:
            xb, yb = _sample_memmap_batch(train_mm, batch_size, seq_len, rng_np, device)
            bytes_seen += int(batch_size * seq_len)
        else:
            xb, yb = cog_gen.sample_batch(batch_size, device)
            bytes_seen += int((xb != tok.PAD_ID).sum().item())

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            lg = model(xb)
            t_ce = F.cross_entropy(
                lg.transpose(1, 2), yb, ignore_index=tok.PAD_ID, reduction="none"
            )
            denom = (yb != tok.PAD_ID).sum(dim=1).clamp(min=1)
            ce_loss = (t_ce.sum(dim=1) / denom).mean()
            loss = ce_loss + 0.04 * model.get_commitment_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        opt.step()

        if step in (1, 100, 500) or step % 1000 == 0 or step == total_steps:
            bch = model.byte_chunker.last_chunk_ratio if model.byte_chunker is not None else 1.0
            lck = int(model.byte_chunker.router_locked) if model.byte_chunker is not None else 0
            alpha = model.crystal.snap_alpha if model.crystal is not None else 0.0
            el_m = (time.perf_counter() - t_fp0) / 60.0
            print(
                f"  [Tier-1 FP | {el_m:5.1f}m] step {step:5d}/{total_steps} lr={lr:.2e} "
                f"alpha={alpha:.2f} lock={lck} b/ch={bch:.2f} ce={ce_loss.item():.4f} "
                f"bytes={bytes_seen/1e9:.3f}GB",
                flush=True,
            )
            loss_log.append(
                {
                    "step": int(step),
                    "ce": round(float(ce_loss.item()), 4),
                    "bytes_gb": round(bytes_seen / 1e9, 4),
                    "b_per_chunk": round(float(bch), 3),
                }
            )
        if step % 1500 == 0 and step < total_steps:
            torch.save(
                {"step": step, "total_steps": total_steps, "model": model.state_dict(), "opt": opt.state_dict()},
                latest_ckpt,
            )

    fp_train_min = round((time.perf_counter() - t_fp0) / 60.0, 2)
    model.set_crystal_schedule(total_steps, total_steps)
    model.set_snap_alpha(1.0)
    if model.byte_chunker is not None:
        model.byte_chunker.lock_router(True)
    model.eval()
    fp_sd_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Two-Group 1.58-bit STE QAT -> 5.41 MB 2-bit uint8 LUT
    print(f"\n=== [Stage 3B] Two-Group STE QAT ({qat_steps} steps) -> 5.41 MB 2-bit LUT ===", flush=True)
    t_q0 = time.perf_counter()
    qat_model = SimVQSplitLM(cfg).to(device)
    qat_model.load_state_dict({k: v.clone() for k, v in fp_sd_cpu.items()}, strict=True)
    qat_model.set_crystal_schedule(total_steps, total_steps)
    qat_model.byte_chunker.lock_router(True)
    qat_model.set_snap_alpha(1.0)
    qat_model.set_qat_mix(0.30)
    qat_model.set_ternary_qat(True)

    c_params = [p for nm, p in qat_model.named_parameters() if p.requires_grad and not nm.startswith("byte_chunker.")]
    b_params = [p for nm, p in qat_model.named_parameters() if p.requires_grad and nm.startswith("byte_chunker.")]
    all_qp = c_params + b_params
    CORE_LR, CHUNK_LR = 1.2e-4, 6.0e-4
    opt_q = torch.optim.AdamW(
        [{"params": c_params, "lr": CORE_LR}, {"params": b_params, "lr": CHUNK_LR}],
        weight_decay=0.0,
    )
    warm_q = min(40, max(1, qat_steps // 6))
    ramp_q = min(180, max(1, qat_steps // 3))
    for qs in range(1, qat_steps + 1):
        qat_model.train()
        qat_model.set_qat_mix(min(1.0, 0.30 + 0.70 * (qs / float(ramp_q))))
        mult = (qs / float(warm_q)) if qs <= warm_q else (
            0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * (qs - warm_q) / max(1, qat_steps - warm_q)))
        )
        opt_q.param_groups[0]["lr"] = CORE_LR * mult
        opt_q.param_groups[1]["lr"] = CHUNK_LR * mult
        if rng_py.random() < 0.85:
            xb, yb = _sample_memmap_batch(train_mm, batch_size, seq_len, rng_np, device)
        else:
            xb, yb = cog_gen.sample_batch(batch_size, device)
        opt_q.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            lg = qat_model(xb)
            t_ce = F.cross_entropy(
                lg.transpose(1, 2), yb, ignore_index=tok.PAD_ID, reduction="none"
            )
            denom = (yb != tok.PAD_ID).sum(dim=1).clamp(min=1)
            loss = (t_ce.sum(dim=1) / denom).mean() + 0.04 * qat_model.get_commitment_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_qp, 1.0)
        opt_q.step()
        if qs in (1, qat_steps // 2, qat_steps):
            print(
                f"  [Tier-1 STE QAT] step {qs:3d}/{qat_steps} mix={qat_model.qat_mix:.2f} "
                f"loss={loss.item():.4f}",
                flush=True,
            )

    qat_model.set_ternary_qat(False)
    qat_model.set_qat_mix(1.0)
    with torch.no_grad():
        for nm, mod in qat_model.named_modules():
            if isinstance(mod, nn.Linear) and nm != "lm_head":
                if qat_model.byte_chunker.is_locked_router_module(mod):
                    continue
                wt, _, _ = quantize_ternary_weights_ls(mod.weight.data)
                mod.weight.data.copy_(wt)
    qat_model.eval()
    qat_sd_cpu = {k: v.detach().cpu().clone() for k, v in qat_model.state_dict().items()}
    qat_min = round((time.perf_counter() - t_q0) / 60.0, 2)

    # Pack into FastPackedLUT2BitLinear (`cuda_ext` if on CUDA!)
    binfo = get_lut_backend_info()
    lut_backend = "cuda_ext" if (device == "cuda" and binfo["cuda_ext_available"]) else "auto"
    lut_model = pack_fast_model_2bit_lut(qat_model.cpu(), backend=lut_backend).to(device).eval()
    lut_model.set_crystal_schedule(1, 1)
    lut_fp_info = compute_2bit_lut_footprint(lut_model)
    print(f"Packed Tier-1 2-bit LUT: {lut_fp_info} (backend={lut_backend})", flush=True)

    # Save BF16 + Physical LUT checkpoints immediately
    bf16_payload = {
        "format": "calcite5_tier1_utf8_bf16_v1",
        "simvq_split_config": dict(cfg.__dict__),
        "tokenizer": tok.to_dict(),
        "simvq_split_state_dict": {k: v.to(torch.bfloat16) for k, v in fp_sd_cpu.items()},
        "simvq_split_ternary_qat_state_dict": {k: v.to(torch.bfloat16) for k, v in qat_sd_cpu.items()},
        "train_meta": {
            "total_steps": total_steps,
            "bytes_seen": bytes_seen,
            "fp_train_minutes": fp_train_min,
            "qat_minutes": qat_min,
        },
    }
    bf16_ckpt_p = ckpt_dir / "calcite5_tier1_17m_utf8_bf16.pt"
    lut_ckpt_p = ckpt_dir / "calcite5_tier1_17m_utf8_lut.pt"
    torch.save(bf16_payload, bf16_ckpt_p)
    save_packed_lut_checkpoint(
        lut_ckpt_p,
        lut_model.cpu(),
        cfg,
        tok,
        metadata={"params": n_params, "bytes_seen": bytes_seen, "total_steps": total_steps},
    )
    lut_model = lut_model.to(device).eval()

    # Evaluations: Val memmap + 6 OpenWeb regimes + GSM8K te_docs[:500] + Cognitive suites + Base 1.48M baseline!
    _, eval_wins, worst_b, clean_b = build_disjoint_openweb_corpora(
        repo_root, tok, tr_docs, te_docs, seq_len=768
    )
    N_EVAL = 200
    s5_suites = {
        "k2": build_eval_s5_suite(tok, te_sents, 2, N_EVAL, (0, 1), 3100),
        "k4": build_eval_s5_suite(tok, te_sents, 4, N_EVAL, (0, 1), 3200),
        "k8": build_eval_s5_suite(tok, te_sents, 8, N_EVAL, (1, 1), 3300),
        "k16": build_eval_s5_suite(tok, te_sents, 16, N_EVAL, (1, 2), 3400),
    }
    ov_suites = {
        "in_dist": build_eval_overwrite_suite(tok, te_sents, 4, 2, 2, N_EVAL, 4100),
        "ood_med": build_eval_overwrite_suite(tok, te_sents, 7, 4, 10, N_EVAL, 4200),
        "ood_long": build_eval_overwrite_suite(tok, te_sents, 9, 5, 18, N_EVAL, 4300),
    }
    trap_suites = {
        "in_dist": build_eval_trap_suite(tok, te_sents, 4, 2, 2, N_EVAL, 5100),
        "ood_long": build_eval_trap_suite(tok, te_sents, 9, 5, 18, N_EVAL, 5300),
    }

    def _eval_full_sys1(tag_name: str, eval_mod: nn.Module) -> Dict[str, object]:
        val_bpb = eval_val_memmap_bpb(eval_mod, val_mm, seq_len=min(768, seq_len), n_batches=32, batch_size=16, device=device)
        ow = evaluate_openweb_regimes(eval_mod, eval_wins, worst_b, clean_b, device)
        gsm = evaluate_byte_prose_and_ratio(eval_mod, tok, te_docs[:500], byte_per_bpe, seq_len=768, device=device)
        s5, ov, trp, f1_in, f1_lg = _eval_cognitive_suites_b(eval_mod, tok, s5_suites, ov_suites, trap_suites, device)
        drift = measure_crystal_drift_at_L(eval_mod, tok.vocab_size, device, length=2048)
        s5_sum = round(sum(s5[k]["exact_pct"] for k in ("k2", "k4", "k8", "k16")), 2)
        print(
            f"  [Eval {tag_name}] val_bpb={val_bpb:.4f} | ow_bpb={ow['mean_openweb_bpb']:.4f} "
            f"(mid_utf8={ow['max_mid_utf8_split_pct']:.2f}%) | gsm_bpb={gsm['byte_bpb']:.4f} "
            f"(EqPPL={gsm['bpe_equiv_ppl']:.2f}) | S5_sum={s5_sum:.1f}% | SelF1_long={f1_lg:.1f}%",
            flush=True,
        )
        return {
            "val_memmap_bpb": val_bpb,
            "openweb": ow,
            "gsm8k_te500": gsm,
            "s5": s5,
            "s5_sum": s5_sum,
            "overwrite": ov,
            "trap": trp,
            "selective_f1_in": f1_in,
            "selective_f1_ood_long": f1_lg,
            "crystal_drift_L2048": drift,
        }

    # Evaluate Base 1.48M Phase E reference on the EXACT same val_mm + suites for honest scaling delta
    ckpt_e = torch.load(repo_root / "checkpoints" / "simvq_split_utf8_openweb.pt", map_location="cpu", weights_only=False)
    cfg_e = SimVQSplitConfig(**ckpt_e["simvq_split_config"])
    base_lut_m = SimVQSplitLM(cfg_e).eval()
    base_lut_m.load_state_dict(ckpt_e["simvq_split_ternary_qat_state_dict"], strict=True)
    base_lut_m.byte_chunker.lock_router(True)
    base_lut_m.set_crystal_schedule(1, 1)
    base_lut_fast = pack_fast_model_2bit_lut(base_lut_m, backend=lut_backend).to(device).eval()
    base_eval = _eval_full_sys1("Base_1.48M_548KB_LUT", base_lut_fast)
    del base_lut_m, base_lut_fast

    t1_fp_eval = _eval_full_sys1("Tier1_17.49M_FP", model)
    t1_qat_eval = _eval_full_sys1("Tier1_17.49M_QAT", qat_model)
    t1_lut_eval = _eval_full_sys1("Tier1_17.49M_5.41MB_LUT", lut_model)

    res = {
        "model_spec": {
            "params": n_params,
            "config": TIER1_CONFIG_DICT,
            "state_cache_kb_L2048": cache_kb,
            "chunker_state_bytes": int(
                compute_chunker_state_bytes(
                    cfg.d_model,
                    cfg.byte_char_dim,
                    cfg.byte_router_char_dim,
                    byte_chunk_mode=cfg.byte_chunk_mode,
                )
            ),
            "lut_footprint": lut_fp_info,
        },
        "training": {
            "total_steps": total_steps,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "bytes_seen": bytes_seen,
            "sec_per_step": round(sec_per_step, 4),
            "peak_vram_gb": round(peak_vram_gb, 2),
            "fp_train_minutes": fp_train_min,
            "qat_steps": qat_steps,
            "qat_minutes": qat_min,
            "loss_log": loss_log,
            "bf16_ckpt_bytes": int(bf16_ckpt_p.stat().st_size),
            "lut_ckpt_bytes": int(lut_ckpt_p.stat().st_size),
        },
        "evaluations": {
            "base_1_48m_lut": base_eval,
            "tier1_fp": t1_fp_eval,
            "tier1_qat": t1_qat_eval,
            "tier1_lut_5_41mb": t1_lut_eval,
        },
    }
    (out_dir / "track2_tier1_system1.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    del model, qat_model, lut_model
    gc.collect()
    if use_amp:
        torch.cuda.empty_cache()
    return res


def run_stage_eval_cram(
    repo_root: Path,
    ckpt_dir: Path,
    out_dir: Path,
    device: str,
    skip_sealed_test: bool = False,
    reextract_cram: bool = False,
) -> Dict[str, object]:
    """Track 3: Доадаптация `Tier-1` (`d_model=384`) на `GSM8K + SphericalALU + CRAM+z`."""
    print("\n=== [Stage 4 / Eval CRAM+z] GSM8K + Interleaved SphericalALU (Tier-1 17.49M / 5.41 MB) ===", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    bf16_ckpt_p = ckpt_dir / "calcite5_tier1_17m_utf8_bf16.pt"
    assert bf16_ckpt_p.exists(), f"Missing trained Tier-1 checkpoint: {bf16_ckpt_p}"

    tr_p = repo_root / "benchmarks" / "gsm8k" / "train.jsonl"
    te_p = repo_root / "benchmarks" / "gsm8k" / "test.jsonl"
    with open(tr_p, "r", encoding="utf-8") as f:
        all_tr = [json.loads(line) for line in f if line.strip()]
    train_items = all_tr[:6000]
    cal_items = all_tr[6000:6736]
    blind_items = all_tr[6736:]
    print(f"GSM8K splits: train={len(train_items)} | cal={len(cal_items)} | blind_dev={len(blind_items)}", flush=True)

    ext = TrainInducedQuantityExtractor()
    ext_stats = ext.fit(train_items)
    alu = SphericalBase10RegisterALU(d_sub=15, num_digits=12, seed=3301)
    alu_diag = alu.verify_crystal_invariants(n_pairs=200)

    sys1 = Calcite5MathSystem1(bf16_ckpt_p, device=device, seed=2026)
    adapted_ckpt = ckpt_dir / "tier1_cram_adapted.pt"
    if adapted_ckpt.is_file() and (not reextract_cram):
        print(f"   [System-1 Tier-1] Loading cached adapted checkpoint: {adapted_ckpt}", flush=True)
        sys1.load_adapted(adapted_ckpt)
        ck_meta = torch.load(adapted_ckpt, map_location="cpu", weights_only=False)
        assert sys1.lut_model is not None
        lut_meta = ck_meta.get("lut_footprint", compute_2bit_lut_footprint(sys1.lut_model))
        fit_stats = {
            "cached": 1.0,
            "n_compiled_train": float(ck_meta.get("n_samples", 4443.0)),
            "final_loss": float(ck_meta.get("final_loss", 0.0)),
            "lut_total_kb": float(lut_meta["total_lut_kb"]),
            "num_lut_layers": float(lut_meta["num_lut_layers"]),
            "train_seconds": float(ck_meta["train_seconds"]) if "train_seconds" in ck_meta else 0.0,
        }
    else:
        fit_stats = sys1.fit_on_train_split(
            train_items=train_items,
            extractor=ext,
            epochs=16,
            batch_size=48,
            lr=4.0e-4,
            qat_steps=240,
            save_ckpt_path=adapted_ckpt,
            use_interleaved_alu=True,
            alu=alu,
        )
        fit_stats["cached"] = 0.0
    print(f"System-1 Tier-1 adapted on train[:6000]: {fit_stats}", flush=True)

    # Accelerate 5.41 MB 2-bit LUT with native CUDA C++ kernel!
    binfo = get_lut_backend_info()
    lut_backend = "cuda_ext" if (device == "cuda" and binfo["cuda_ext_available"]) else "auto"
    assert sys1.lut_model is not None
    sys1.lut_model = accelerate_lut_model(sys1.lut_model, backend=lut_backend, inplace=True).to(device).eval()

    d_model = int(sys1.cfg.d_model)
    slots_tr = [ext.extract_slots(str(it["question"]), include_aux_consts=True)[0] for it in train_items]
    slots_cal = [ext.extract_slots(str(it["question"]), include_aux_consts=True)[0] for it in cal_items]
    slots_bld = [ext.extract_slots(str(it["question"]), include_aux_consts=True)[0] for it in blind_items]

    q_tr = [str(it["question"]) for it in train_items]
    q_cal = [str(it["question"]) for it in cal_items]
    q_bld = [str(it["question"]) for it in blind_items]

    # Precompute FP + LUT base features (single-pass beam_width=0)
    t_f0 = time.perf_counter()
    f_tr_fp = sys1.extract_features_and_beams(q_tr, slots_tr, ext, batch_size=32, beam_width=0, use_lut=False, alu=alu)
    f_cal_fp = sys1.extract_features_and_beams(q_cal, slots_cal, ext, batch_size=32, beam_width=0, use_lut=False, alu=alu)
    f_bld_fp = sys1.extract_features_and_beams(q_bld, slots_bld, ext, batch_size=32, beam_width=0, use_lut=False, alu=alu)
    f_cal_lut = sys1.extract_features_and_beams(q_cal, slots_cal, ext, batch_size=32, beam_width=0, use_lut=True, alu=alu)
    f_bld_lut = sys1.extract_features_and_beams(q_bld, slots_bld, ext, batch_size=32, beam_width=0, use_lut=True, alu=alu)
    print(f"Extracted Cal/BlindDev FP + {lut_backend} LUT features in {time.perf_counter() - t_f0:.1f}s", flush=True)

    solver = OpenGraphCRAMSolver(ext, alu, d_model=d_model, device=device, seed=2026)
    cram_stats = solver.fit_bptt_free(
        train_items,
        slots_tr,
        f_tr_fp,
        epochs=12,
        batch_size=128,
        lr=0.018,
    )
    print(f"CRAM+z fit_bptt_free (0 BPTT, d_model={d_model}): {cram_stats}", flush=True)

    # 1) FP branch: batched TF enrichment + DualCert calibration on Cal + Cal/BlindDev(FP)
    f_cal_fp = solver.enrich_lm_feats_with_system1_tf(
        cal_items, slots_cal, f_cal_fp, sys1, top_k=6, use_lut=False, batch_size=48
    )
    f_bld_fp = solver.enrich_lm_feats_with_system1_tf(
        blind_items, slots_bld, f_bld_fp, sys1, top_k=6, use_lut=False, batch_size=48
    )
    cal_fp_stats = solver.calibrate_on_split(cal_items, slots_cal, f_cal_fp)
    res_cal_fp = evaluate_phase_f_split(solver, cal_items, slots_cal, f_cal_fp, "Cal_Tier1_FP")
    res_bld_fp = evaluate_phase_f_split(solver, blind_items, slots_bld, f_bld_fp, "BlindDev_Tier1_FP")

    # 2) 5.41 MB 2-bit LUT branch (`cuda_ext`): separate DualCert calibration on Cal + Cal/BlindDev(LUT)
    f_cal_lut = solver.enrich_lm_feats_with_system1_tf(
        cal_items, slots_cal, f_cal_lut, sys1, top_k=6, use_lut=True, batch_size=32
    )
    f_bld_lut = solver.enrich_lm_feats_with_system1_tf(
        blind_items, slots_bld, f_bld_lut, sys1, top_k=6, use_lut=True, batch_size=32
    )
    cal_lut_stats = solver.calibrate_on_split(cal_items, slots_cal, f_cal_lut)
    res_cal_lut = evaluate_phase_f_split(solver, cal_items, slots_cal, f_cal_lut, "Cal_Tier1_2Bit_LUT")
    res_bld_lut = evaluate_phase_f_split(solver, blind_items, slots_bld, f_bld_lut, "BlindDev_Tier1_2Bit_LUT")

    print(
        f"-> Cal: FP={res_cal_fp['pass1_pct']:.2f}% / LUT={res_cal_lut['pass1_pct']:.2f}% | "
        f"BlindDev: FP={res_bld_fp['pass1_pct']:.2f}% (Pass@5={res_bld_fp['pass5_pct']:.2f}%) / "
        f"LUT={res_bld_lut['pass1_pct']:.2f}% (Pass@5={res_bld_lut['pass5_pct']:.2f}%)",
        flush=True,
    )

    vault = SingleShotTestVault(te_p, max_calls=1)
    sealed_out: Dict[str, object] = {}
    if not skip_sealed_test:
        t_te0 = time.perf_counter()
        sealed_items = vault.load_sealed_items()
        assert vault.calls_used == 1 and len(sealed_items) == 1319
        q_te = [str(it["question"]) for it in sealed_items]
        slots_te = [ext.extract_slots(q, include_aux_consts=True)[0] for q in q_te]
        f_te_fp = sys1.extract_features_and_beams(q_te, slots_te, ext, batch_size=32, beam_width=0, use_lut=False, alu=alu)
        f_te_lut = sys1.extract_features_and_beams(q_te, slots_te, ext, batch_size=32, beam_width=0, use_lut=True, alu=alu)
        f_te_fp = solver.enrich_lm_feats_with_system1_tf(
            sealed_items, slots_te, f_te_fp, sys1, top_k=6, use_lut=False, batch_size=48
        )
        res_te_fp = evaluate_phase_f_split(solver, sealed_items, slots_te, f_te_fp, "SealedTest_Tier1_FP")
        f_te_lut = solver.enrich_lm_feats_with_system1_tf(
            sealed_items, slots_te, f_te_lut, sys1, top_k=6, use_lut=True, batch_size=32
        )
        res_te_lut = evaluate_phase_f_split(solver, sealed_items, slots_te, f_te_lut, "SealedTest_Tier1_2Bit_LUT")
        print(
            f"-> Sealed Test (1/1 in {time.perf_counter()-t_te0:.1f}s): "
            f"FP={res_te_fp['pass1_pct']:.2f}% (Pass@5={res_te_fp['pass5_pct']:.2f}%) | "
            f"5.41MB LUT={res_te_lut['pass1_pct']:.2f}% (Pass@5={res_te_lut['pass5_pct']:.2f}%)",
            flush=True,
        )
        sealed_out = {
            "vault_calls_used": int(vault.calls_used),
            "fp": res_te_fp,
            "lut_2bit": res_te_lut,
        }

    cram_res = {
        "alu_invariants": alu_diag,
        "system1_fit": fit_stats,
        "cram_fit": {**ext_stats, **cram_stats},
        "cal_calibration": {"fp": cal_fp_stats, "lut_2bit": cal_lut_stats},
        "cal": {"fp": res_cal_fp, "lut_2bit": res_cal_lut},
        "blind_dev": {"fp": res_bld_fp, "lut_2bit": res_bld_lut},
        "sealed_test": sealed_out,
    }
    (out_dir / "track3_cram_gsm8k.json").write_text(json.dumps(cram_res, indent=2), encoding="utf-8")
    return cram_res


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calcite-5 Phase G Runner")
    parser.add_argument(
        "--stage",
        choices=("track1", "build-corpus", "train-tier1", "eval-cram", "all"),
        default="all",
    )
    parser.add_argument("--data-dir", type=str, default="/content/data")
    parser.add_argument("--ckpt-dir", type=str, default="/content/phase_g_ckpts")
    parser.add_argument("--out-dir", type=str, default="/content/phase_g_out")
    parser.add_argument("--total-gb", type=float, default=1.80)
    parser.add_argument("--max-steps", type=int, default=14000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--qat-steps", type=int, default=600)
    parser.add_argument("--max-train-minutes", type=float, default=95.0)
    parser.add_argument("--skip-sealed-test", action="store_true")
    parser.add_argument("--reextract-cram", action="store_true")
    args = parser.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {}
    if args.stage in ("track1", "all"):
        summary["track1_cuda_lut"] = run_stage_track1(ROOT_DIR, out_dir, dev)
    elif (out_dir / "track1_cuda_lut.json").exists():
        summary["track1_cuda_lut"] = json.loads((out_dir / "track1_cuda_lut.json").read_text(encoding="utf-8"))

    if args.stage in ("build-corpus", "all"):
        summary["corpus"] = run_stage_build_corpus(ROOT_DIR, data_dir, total_gb=args.total_gb)
    elif (data_dir / "corpus_meta.json").exists():
        summary["corpus"] = json.loads((data_dir / "corpus_meta.json").read_text(encoding="utf-8"))

    if args.stage in ("train-tier1", "all"):
        summary["track2_tier1"] = run_stage_train_tier1(
            ROOT_DIR,
            data_dir,
            ckpt_dir,
            out_dir,
            dev,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            qat_steps=args.qat_steps,
            max_train_minutes=args.max_train_minutes,
        )
    elif (out_dir / "track2_tier1_system1.json").exists():
        summary["track2_tier1"] = json.loads((out_dir / "track2_tier1_system1.json").read_text(encoding="utf-8"))

    if args.stage in ("eval-cram", "all"):
        summary["track3_cram_gsm8k"] = run_stage_eval_cram(
            ROOT_DIR,
            ckpt_dir,
            out_dir,
            dev,
            skip_sealed_test=args.skip_sealed_test,
            reextract_cram=args.reextract_cram,
        )
    elif (out_dir / "track3_cram_gsm8k.json").exists():
        summary["track3_cram_gsm8k"] = json.loads((out_dir / "track3_cram_gsm8k.json").read_text(encoding="utf-8"))

    if summary:
        final_p = out_dir / "phase_g_colab_scaleup_results.json"
        final_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSaved combined Phase G JSON -> {final_p} ({final_p.stat().st_size:,} B)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
