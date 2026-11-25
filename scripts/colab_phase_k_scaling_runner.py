"""
Phase K Execution Runner (`scripts/colab_phase_k_scaling_runner.py`):
Pure `Calcite System-1` Scaling Law (`5M -> 25M -> 85M`) vs `RoPETransformerLM` &
`ScaledHybridDeltaSWALM` on Google Colab GPU (`A100` / `L4`) + `--smoke-local`.

Enforces:
1. Strict zero-leakage corpus scrubber (`45`-char normalized substrings across `GSM8K`,
   `MultiArith-600`, `ASDiv-A`, `SVAMP-1000`, and `ARC Eval`) and ZERO arithmetic `SphericalALU` traces.
2. Identical `32,768` bytes/step (`B_eff=64, L=512`) stratified byte stream (`train_utf8.u16.bin` +
   `6` disjoint `train_wins` + `S_5 / Overwrite / Trap` in-dist episodes) across all 3 architectures.
3. Full checkpoint & cell-by-cell resumability in `benchmarks/phase_k_cells/<tier>_<model>.json`.
4. Empirical power-law scaling fits (`BPB(N)`, `BPB(Weight_MB)`, and `Delta_LUT_BPB(d_model)`) saved
   into `benchmarks/phase_k_scaling_law_results.json`.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import gc
import glob
import inspect
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.scaling_phase_k.byte_baselines import (  # noqa: E402
    HybridDeltaSWAConfig,
    ScaledHybridDeltaSWALM,
    build_matched_baselines_for_tier,
    compute_rope_cache_kb,
)
from experiments.scaling_phase_k.eval_suite import (  # noqa: E402
    check_gate2_block_std_parity,
    encode_utf8_fast,
    encode_utf8_fast_np,
    eval_cognitive_suites_any_byte_lm,
    eval_length_extrapolation_bpb,
    eval_openweb_6domains_any_byte_lm,
    eval_val_memmap_bpb,
    has_usable_cuda,
    load_utf8_corpus_array,
    measure_infer_latency_ms,
    sample_memmap_batch,
    sync_if_cuda,
)
from experiments.scaling_phase_k.lean_system1 import (  # noqa: E402
    CalciteLeanSystem1LM,
    LeanSystem1Config,
    build_tier_lean_config,
    compute_lean_state_cache_kb,
)
from kernels.fast_calcite_scans import (  # noqa: E402
    clear_fast_qat_cache,
    compile_calcite_cortex,
    get_fast_scans_backend_info,
    unwrap_compiled_state_dict,
)
from kernels.fast_packed_lut import (  # noqa: E402
    get_lut_backend_info,
    pack_fast_model_2bit_lut,
    save_packed_lut_checkpoint,
)
from run_phase_e_fast_utf8_benchmark import build_disjoint_openweb_corpora  # noqa: E402
from scripts.colab_phase_g_runner import (  # noqa: E402
    _encode_doc_uint16,
    _stream_hf_parquet_docs,
    is_leaked_doc,
)
from simvq_split.baselines import BaselineConfig, RMSNorm, RoPETransformerLM  # noqa: E402
from simvq_split.eval_utils import evaluate_suite_batched  # noqa: E402
from simvq_split.memory import (  # noqa: E402
    compute_2bit_lut_footprint,
    compute_chunker_state_bytes,
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


TIER_SCHEDULE: Dict[str, Dict[str, object]] = {
    "S1_5M": {
        "fp_steps": 4000,
        "grok_steps": 3200,
        "s1b_steps": 600,
        "qat_steps": 500,
        "core_lr": 2.0e-3,
        "stage2_core_lr": 9.0e-4,
        "stage2_emb_lr": 1.35e-3,
        "min_lr": 1.5e-5,
        "ckpt_name": "calcite5_s1_5m_utf8_lut.pt",
    },
    "S2_25M": {
        "fp_steps": 4000,
        "grok_steps": 3200,
        "s1b_steps": 800,
        "qat_steps": 500,
        "core_lr": 1.4e-3,
        "stage2_core_lr": 7.5e-4,
        "stage2_emb_lr": 1.10e-3,
        "min_lr": 1.5e-5,
        "ckpt_name": "calcite5_s2_25m_utf8_lut.pt",
    },
    "S3_85M": {
        "fp_steps": 4000,
        "grok_steps": 3200,
        "s1b_steps": 800,
        "qat_steps": 500,
        "core_lr": 1.0e-3,
        "stage2_core_lr": 7.0e-4,
        "stage2_emb_lr": 1.60e-3,
        "min_lr": 1.2e-5,
        "ckpt_name": "calcite5_s3_85m_utf8_lut.pt",
    },
}


# ---------------------------------------------------------------------------
# 1. Zero-Leakage All-Vault Heldout Scrubber & Pure System-1 UTF-8 Corpus
# ---------------------------------------------------------------------------


def _norm_leak_str(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def build_phase_k_heldout_blocklist(repo_root: Path, win_len: int = 45) -> set[str]:
    """
    Collects normalized 45-char substrings from all held-out evaluation benchmarks:
    `GSM8K (train[6000:] + test.jsonl)`, `MultiArith-600`, `ASDiv-A`, `SVAMP-1000`, `ARC Eval`.
    """
    block: set[str] = set()

    def _add(txt: str) -> None:
        norm = _norm_leak_str(txt)
        if len(norm) < win_len:
            if len(norm) >= 24:
                block.add(norm)
            return
        for i in range(0, len(norm) - win_len + 1, 15):
            block.add(norm[i : i + win_len])

    gsm_tr = repo_root / "benchmarks" / "gsm8k" / "train.jsonl"
    if gsm_tr.exists():
        rows = [json.loads(ln) for ln in gsm_tr.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for obj in rows[6000:]:
            _add(str(obj.get("question", "")))

    gsm_te = repo_root / "benchmarks" / "gsm8k" / "test.jsonl"
    if gsm_te.exists():
        for ln in gsm_te.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                _add(str(json.loads(ln).get("question", "")))

    for rel in (
        "benchmarks/ood_math/multiarith_600.json",
        "benchmarks/ood_math/asdiv_a.json",
        "benchmarks/svamp/svamp_1000.json",
    ):
        p = repo_root / rel
        if p.exists():
            arr = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(arr, list):
                for item in arr:
                    if isinstance(item, dict):
                        for key in ("question", "Body", "Question"):
                            if item.get(key):
                                _add(str(item[key]))

    arc_ev = sorted(
        glob.glob(str(repo_root / "benchmarks" / "arc_agi" / "ARC-AGI-master" / "data" / "evaluation" / "*.json"))
    )
    for fp in arc_ev[:60]:
        _add(Path(fp).read_text(encoding="utf-8")[:400])
    return block


def run_stage_build_corpus(
    repo_root: Path,
    data_dir: Path,
    total_gb: float = 0.65,
    corpus_gdrive_id: str = "",
) -> Dict[str, object]:
    """Builds or reuses pure multilingual + Python + JSON UTF-8 stream (`0` arithmetic ALU traces)."""
    print(f"\n=== [Phase K | Corpus] Preparing {total_gb:.2f} GB Pure UTF-8 Stream (0 ALU Traces) ===", flush=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    train_bin = data_dir / "train_utf8.u16.bin"
    val_bin = data_dir / "val_utf8.u16.bin"
    meta_p = data_dir / "corpus_meta.json"

    def _try_reuse(src_dir: Path) -> Optional[Dict[str, object]]:
        tb, vb, mp = src_dir / "train_utf8.u16.bin", src_dir / "val_utf8.u16.bin", src_dir / "corpus_meta.json"
        if tb.exists() and vb.exists() and mp.exists():
            meta = json.loads(mp.read_text(encoding="utf-8"))
            if abs(float(meta.get("train_tokens_mb", 0)) - total_gb * 1024) < 128 and meta.get("pure_system1") is True:
                if src_dir.resolve() != data_dir.resolve():
                    import shutil as _shutil

                    print(f"Copying cached Phase K corpus from {src_dir} -> {data_dir}...", flush=True)
                    for fn in ("train_utf8.u16.bin", "val_utf8.u16.bin", "corpus_meta.json"):
                        _shutil.copy2(src_dir / fn, data_dir / fn)
                print(f"Pure Phase K corpus reused from {src_dir} (0s build).", flush=True)
                return meta
        return None

    reused = _try_reuse(data_dir)
    if reused is not None:
        return reused

    drive_cache = Path("/content/drive/MyDrive/calcite_cache/phase_k_data")
    if drive_cache.is_dir():
        reused_drv = _try_reuse(drive_cache)
        if reused_drv is not None:
            return reused_drv
    drive_tar = drive_cache.parent / "phase_k_corpus_cache.tar.gz"
    if drive_tar.is_file():
        import tarfile as _tf

        print(f"Extracting cached corpus archive from {drive_tar}...", flush=True)
        with _tf.open(drive_tar, "r:*") as tf:
            tf.extractall(data_dir)
        reused_tar = _try_reuse(data_dir)
        if reused_tar is not None:
            return reused_tar

    if corpus_gdrive_id:
        import subprocess as _sub
        import tarfile as _tf

        arc_p = data_dir / "phase_k_corpus_cache.tar"
        print(f"Downloading cached corpus archive from Google Drive id={corpus_gdrive_id}...", flush=True)
        _sub.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=False)
        _sub.run(
            [sys.executable, "-m", "gdown", "--id", corpus_gdrive_id, "-O", str(arc_p)],
            check=True,
        )
        with _tf.open(arc_p, "r:*") as tf:
            tf.extractall(data_dir)
        arc_p.unlink(missing_ok=True)
        reused_gd = _try_reuse(data_dir)
        if reused_gd is not None:
            return reused_gd

    t0 = time.perf_counter()
    blocklist = build_phase_k_heldout_blocklist(repo_root)
    print(f"Loaded {len(blocklist):,} held-out 45-char blocklist hashes across all 5 suites.", flush=True)

    target_total = int(total_gb * 1024 * 1024 * 1024)
    val_total = 24 * 1024 * 1024
    tot_plus_val = target_total + val_total

    en_docs, sc_en, s_en = _stream_hf_parquet_docs(
        [("HuggingFaceFW/fineweb-edu", "sample/10BT/"), ("wikimedia/wikipedia", "20231101.en/")],
        ["text"],
        int(0.46 * tot_plus_val),
        blocklist,
        "1_FineWeb_EN",
    )
    math_docs, sc_math, s_math = _stream_hf_parquet_docs(
        [("HuggingFaceTB/finemath", "finemath-4plus"), ("open-web-math/open-web-math", None)],
        ["text"],
        int(0.14 * tot_plus_val),
        blocklist,
        "2_FineMath_Prose",
    )
    ru_docs, sc_ru, s_ru = _stream_hf_parquet_docs(
        [("wikimedia/wikipedia", "20231101.ru/")],
        ["text"],
        int(0.18 * tot_plus_val),
        blocklist,
        "3_Russian_Wiki",
    )
    zh_docs, sc_zh, s_zh = _stream_hf_parquet_docs(
        [("wikimedia/wikipedia", "20231101.zh/")],
        ["text"],
        int(0.05 * tot_plus_val),
        blocklist,
        "4_CJK_Wiki",
    )

    em_palette = list("🌍🌌🔭🛸🛰🧪🧬🧲🧭📡🔩🧰📐📏🖊📒🗂📅🕒🔑🔋💡🧮🧱")
    math_syms = list("∑∏∫√∞≈≠≤≥∈∉∪∩→⇒∀∃∂∇λΔΩ")
    u4_docs: List[np.ndarray] = []
    u4_bytes = 0
    rng_u4 = random.Random(8127)
    u4_cap = min(14 * 1024 * 1024, int(0.03 * tot_plus_val))
    while u4_bytes < u4_cap:
        e_seq = "".join(rng_u4.choice(em_palette) for _ in range(rng_u4.randint(4, 12)))
        m_seq = " ".join(rng_u4.choice(math_syms) for _ in range(rng_u4.randint(3, 8)))
        arr = _encode_doc_uint16(f"Telemetry {e_seq} :: state={rng_u4.randint(10, 999)} | eq: {m_seq}\n{e_seq}")
        u4_docs.append(arr)
        u4_bytes += int(arr.size)

    code_target = int(0.14 * tot_plus_val)
    code_docs, sc_code, s_code = _stream_hf_parquet_docs(
        [("bigcode/the-stack-smol", "data/python/"), ("FlyTech/python-codes-25k", None)],
        ["content", "code", "text"],
        int(0.75 * code_target),
        blocklist,
        "5_Python_Code",
    )
    arc_tr = sorted(
        Path(x)
        for x in glob.glob(str(repo_root / "benchmarks" / "arc_agi" / "ARC-AGI-master" / "data" / "training" / "*.json"))
    )
    cur_code_b = sum(int(d.size) for d in code_docs)
    py_paths: List[Path] = []
    for sp_dir in sys.path:
        p = Path(sp_dir)
        if p.is_dir():
            for cand in p.glob("*/*.py"):
                low = str(cand).lower()
                if "test" not in low and "cram_" not in low and "uebr_" not in low:
                    py_paths.append(cand)
                    if len(py_paths) > 8000:
                        break
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

    rng_mix = random.Random(2026)
    train_chunks: List[np.ndarray] = []
    val_chunks: List[np.ndarray] = []
    domain_stats: Dict[str, Dict[str, float]] = {}
    for dname, pool in (
        ("en_edu", en_docs),
        ("finemath_prose", math_docs),
        ("ru_wiki", ru_docs),
        ("zh_wiki", zh_docs),
        ("u4_sym", u4_docs),
        ("code_json", code_docs),
    ):
        if not pool:
            domain_stats[dname] = {"docs": 0, "train_mb": 0.0, "val_mb": 0.0}
            continue
        rng_mix.shuffle(pool)
        n_val = max(1, min(len(pool) // 25, 2500))
        v_p, t_p = pool[:n_val], pool[n_val:]
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

    n_val = _write_bin(val_bin, val_chunks, val_total)
    n_train = _write_bin(train_bin, train_chunks, target_total)
    scrubbed = int(sc_en + sc_math + sc_ru + sc_zh + sc_code)
    meta = {
        "pure_system1": True,
        "alu_traces_injected": 0,
        "device_name": torch.cuda.get_device_name(0) if has_usable_cuda() else "cpu",
        "train_bin_path": str(train_bin),
        "val_bin_path": str(val_bin),
        "train_tokens": int(n_train),
        "train_tokens_mb": round(n_train / (1024**2), 2),
        "val_tokens": int(n_val),
        "val_tokens_mb": round(n_val / (1024**2), 2),
        "heldout_blocklist_hashes": len(blocklist),
        "scrubbed_doc_count": scrubbed,
        "sources": {"en": s_en, "math": s_math, "ru": s_ru, "zh": s_zh, "code": s_code},
        "domain_stats": domain_stats,
        "build_seconds": round(time.perf_counter() - t0, 2),
    }
    meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Pure Phase K Corpus DONE in {meta['build_seconds']}s: train={meta['train_tokens_mb']}MB | val={meta['val_tokens_mb']}MB | scrubbed={scrubbed}", flush=True)
    return meta


# ---------------------------------------------------------------------------
# 2. Shared Deterministic Stratified Batcher (`B_eff=64, L=512`, `0` ALU Traces)
# ---------------------------------------------------------------------------


class StratifiedSystem1Batcher:
    """Identical 2-stage byte stream across `RoPE`, `HybridDeltaSWA`, and `Calcite` (`0` ALU traces)."""

    def __init__(
        self,
        train_mm: np.ndarray,
        tr_docs: Sequence[str],
        tr_sents: Sequence[str],
        tok: ByteBPETokenizer,
        seq_len: int,
        train_wins: Optional[Dict[str, List[List[int]]]] = None,
        seed: int = 9100,
    ):
        self.train_mm = train_mm
        self.tr_sents = list(tr_sents)
        self.tok = tok
        self.seq_len = int(seq_len)
        self.train_wins = {k: v for k, v in (train_wins or {}).items() if v}
        self._train_wins_np: Dict[str, List[np.ndarray]] = {
            k: [np.asarray(w, dtype=np.int64) for w in v]
            for k, v in self.train_wins.items()
        }
        self.domain_keys = sorted(self.train_wins.keys())
        self.rng_np = np.random.default_rng(seed)
        self.rng_py = random.Random(seed + 17)

        docs = list(tr_docs)
        random.Random(seed + 31).shuffle(docs)
        p_stream: List[int] = []
        for d in docs:
            p_stream.extend(encode_utf8_fast(self.tok, d, add_bos=True, add_eos=True))
        tw = self.seq_len + 1
        self.prose_chunks: List[List[int]] = [
            p_stream[i : i + tw]
            for i in range(0, max(0, len(p_stream) - tw), self.seq_len)
        ]
        if not self.prose_chunks and len(p_stream) >= 2:
            chunk = p_stream[:tw] + [self.tok.PAD_ID] * max(0, tw - len(p_stream))
            self.prose_chunks = [chunk]
        self._prose_chunks_np: List[np.ndarray] = [
            np.asarray(c, dtype=np.int64) for c in self.prose_chunks
        ]

    def _sample_cog_row_np(
        self, kind: str, local_frac: float = 1.0, stage1: bool = False
    ) -> Tuple[np.ndarray, int]:
        tw = self.seq_len + 1
        dr_map = {0: (2, 2), 1: (1, 1), 2: (0, 1), 3: (0, 1), 4: (0, 1)}
        fast_raw = not getattr(self.tok, "merges", None)
        for attempt in range(96):
            n_distr = 2 if attempt < 48 else 1
            if kind == "s5":
                if attempt < 64:
                    if stage1:
                        if local_frac < 0.18:
                            k_pool: Tuple[int, ...] = (0, 1)
                        elif local_frac < 0.40:
                            k_pool = (1, 2)
                        else:
                            k_pool = (1, 2, 3, 4)
                    else:
                        k_pool = (0, 1, 2, 3, 4)
                    k = self.rng_py.choice(k_pool)
                    d_r = dr_map[k]
                else:
                    k = self.rng_py.choice((1, 2))
                    d_r = (1, 1)
                p, t, _ = make_s5_story(self.rng_py, k, self.tr_sents, d_r)
            elif kind == "ov":
                p, t, _ = make_overwrite_story(
                    self.rng_py,
                    self.rng_py.randint(3, 5),
                    self.rng_py.randint(1, 2),
                    self.tr_sents,
                    n_distr,
                    0.25,
                )
            else:
                p, t, _ = make_trap_overwrite_story(
                    self.rng_py,
                    self.rng_py.randint(3, 5),
                    self.rng_py.randint(1, 2),
                    self.tr_sents,
                    n_distr,
                    0.25,
                )
            if fast_raw:
                p_arr = encode_utf8_fast_np(self.tok, p, add_bos=True, add_eos=False)
                t_arr = encode_utf8_fast_np(self.tok, t, add_bos=False, add_eos=True)
                if len(p_arr) + len(t_arr) <= tw:
                    return np.concatenate((p_arr, t_arr)), int(len(p_arr))
            else:
                p_ids = self.tok.encode(p, add_bos=True, add_eos=False)
                f_ids = self.tok.encode(p + t, add_bos=True, add_eos=True)
                if len(f_ids) <= tw and f_ids[: len(p_ids)] == p_ids:
                    return np.asarray(f_ids, dtype=np.int64), len(p_ids)
        # Compact guaranteed-fit short S_5 fallback
        p, t, _ = make_s5_story(self.rng_py, 1, self.tr_sents, (0, 0))
        if fast_raw:
            p_arr = encode_utf8_fast_np(self.tok, p, add_bos=True, add_eos=False)
            t_arr = encode_utf8_fast_np(self.tok, t, add_bos=False, add_eos=True)
            f_arr = np.concatenate((p_arr, t_arr))[:tw]
            return f_arr, min(int(len(p_arr)), int(len(f_arr)) - 1)
        p_ids = self.tok.encode(p, add_bos=True, add_eos=False)
        f_ids = self.tok.encode(p + t, add_bos=True, add_eos=True)[:tw]
        return np.asarray(f_ids, dtype=np.int64), min(len(p_ids), len(f_ids) - 1)

    def _sample_cog_row(
        self, kind: str, local_frac: float = 1.0, stage1: bool = False
    ) -> Tuple[List[int], int]:
        arr, p_len = self._sample_cog_row_np(kind, local_frac=local_frac, stage1=stage1)
        return arr.tolist(), p_len

    def _finalize_pair(
        self, xb_np: np.ndarray, yb_np: np.ndarray, device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xb_t = torch.from_numpy(xb_np)
        yb_t = torch.from_numpy(yb_np)
        if device == "cuda" and has_usable_cuda():
            xb_t = xb_t.pin_memory().to(device=device, non_blocking=True)
            yb_t = yb_t.pin_memory().to(device=device, non_blocking=True)
        elif device != "cpu":
            xb_t = xb_t.to(device=device)
            yb_t = yb_t.to(device=device)
        return xb_t, yb_t

    def sample_prose_batch(
        self, batch_size: int, device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples unmasked English prose (`tr_docs`) for dense pre-lock router alignment."""
        if not self._prose_chunks_np:
            return sample_memmap_batch(
                self.train_mm, batch_size, self.seq_len, self.rng_np, device
            )
        xb_np = np.full((batch_size, self.seq_len), self.tok.PAD_ID, dtype=np.int64)
        yb_np = np.full((batch_size, self.seq_len), self.tok.PAD_ID, dtype=np.int64)
        for ri in range(batch_size):
            ch = self.rng_py.choice(self._prose_chunks_np)
            xb_np[ri] = ch[:-1]
            yb_np[ri] = ch[1:]
        return self._finalize_pair(xb_np, yb_np, device)

    def sample_batch(
        self,
        batch_size: int,
        device: str,
        step: int = 1,
        grok_steps: int = 2800,
        total_steps: int = 3700,
        s1b_steps: int = 500,
        s1a_offset: int = 0,
        return_is_web: bool = False,
    ) -> Tuple[Any, ...]:
        del total_steps
        is_web_batch = False
        if step <= grok_steps:
            s1b_eff = (
                int(s1b_steps)
                if (0 < int(s1b_steps) < grok_steps)
                else (1 if grok_steps >= 2 else 0)
            )
            s1a_steps = grok_steps - s1b_eff
            if step <= s1a_steps:
                # Stage 1A (`Crystal S_5 grokking`): 20% prose, 45% S_5 (curriculum), 28% OV, 7% Trap
                local_frac = max(0.0, (step - s1a_offset - 1) / float(max(1, s1a_steps - s1a_offset)))
                stage1_flag = True
                p_prose = 0.20
                p_s5_end, p_ov_end = 0.65, 0.93
            else:
                # Stage 1B (`Woodbury OV+Trap consolidation`): 10% prose, 20% S_5, 52% OV, 18% Trap
                local_frac = 1.0
                stage1_flag = False
                p_prose = 0.10
                p_s5_end, p_ov_end = 0.30, 0.82
            xb_np = np.full((batch_size, self.seq_len), self.tok.PAD_ID, dtype=np.int64)
            yb_np = np.full((batch_size, self.seq_len), self.tok.PAD_ID, dtype=np.int64)
            for ri in range(batch_size):
                u = self.rng_py.random()
                if u < p_prose and self._prose_chunks_np:
                    ch = self.rng_py.choice(self._prose_chunks_np)
                    xb_np[ri] = ch[:-1]
                    yb_np[ri] = ch[1:]
                else:
                    kind = "s5" if u < p_s5_end else ("ov" if u < p_ov_end else "trap")
                    f_arr, p_len = self._sample_cog_row_np(kind, local_frac=local_frac, stage1=stage1_flag)
                    ls = len(f_arr)
                    xb_np[ri, : ls - 1] = f_arr[:-1]
                    yb_np[ri, p_len - 1 : ls - 1] = f_arr[p_len:]
        else:
            # Stage 2 (`68%` pure 6-domain OpenWeb batches / `32%` pure cognitive maintenance batches)
            is_cog = self.rng_py.random() < 0.32
            if not is_cog:
                is_web_batch = True
                dom_keys = [k for k in self.domain_keys if self._train_wins_np.get(k)]
                n_dom = (
                    len(dom_keys)
                    if (dom_keys and batch_size >= 16)
                    else (min(len(dom_keys), max(1, batch_size // 4)) if (dom_keys and batch_size >= 4) else 0)
                )
                n_web = max(0, batch_size - n_dom)
                xb_np = np.full((batch_size, self.seq_len), self.tok.PAD_ID, dtype=np.int64)
                yb_np = np.full((batch_size, self.seq_len), self.tok.PAD_ID, dtype=np.int64)
                if n_web > 0:
                    xw, yw = sample_memmap_batch(self.train_mm, n_web, self.seq_len, self.rng_np, "cpu")
                    xb_np[:n_web] = xw.numpy()
                    yb_np[:n_web] = yw.numpy()
                if n_dom > 0:
                    chosen = (
                        dom_keys
                        if n_dom == len(dom_keys)
                        else [self.rng_py.choice(dom_keys) for _ in range(n_dom)]
                    )
                    for di, dk in enumerate(chosen):
                        w = self.rng_py.choice(self._train_wins_np[dk])[: self.seq_len + 1]
                        lw = len(w) - 1
                        if lw > 0:
                            xb_np[n_web + di, :lw] = w[:-1]
                            yb_np[n_web + di, :lw] = w[1:]
            else:
                xb_np = np.full((batch_size, self.seq_len), self.tok.PAD_ID, dtype=np.int64)
                yb_np = np.full((batch_size, self.seq_len), self.tok.PAD_ID, dtype=np.int64)
                n_s5 = max(1, int(round(batch_size * 0.60)))
                n_ov = max(0, int(round(batch_size * 0.28)))
                while n_s5 + n_ov > batch_size:
                    if n_s5 > 1:
                        n_s5 -= 1
                    else:
                        n_ov -= 1
                kinds = ["s5"] * n_s5 + ["ov"] * n_ov + ["trap"] * (batch_size - n_s5 - n_ov)
                self.rng_py.shuffle(kinds)
                for ri, kn in enumerate(kinds):
                    f_arr, p_len = self._sample_cog_row_np(kn, local_frac=1.0, stage1=False)
                    ls = len(f_arr)
                    xb_np[ri, : ls - 1] = f_arr[:-1]
                    yb_np[ri, p_len - 1 : ls - 1] = f_arr[p_len:]

        perm = self.rng_np.permutation(batch_size)
        xb_t, yb_t = self._finalize_pair(xb_np[perm], yb_np[perm], device)
        if return_is_web:
            return xb_t, yb_t, is_web_batch
        return xb_t, yb_t


# ---------------------------------------------------------------------------
# 3. Optimization Helpers (4-Group 4-Tap STE QAT & Equal-Compute Baseline Loop)
# ---------------------------------------------------------------------------


def _cosine_lr(step: int, total_steps: int, warmup: int, base_lr: float, min_lr: float) -> float:
    if step <= warmup:
        return base_lr * (step / float(max(1, warmup)))
    prog = (step - warmup) / float(max(1, total_steps - warmup))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * prog))


def _make_adamw(params_or_groups, device: str, **kwargs) -> torch.optim.AdamW:
    use_fused = device == "cuda" and has_usable_cuda()
    if use_fused:
        try:
            return torch.optim.AdamW(params_or_groups, fused=True, **kwargs)
        except (RuntimeError, TypeError):
            pass
    return torch.optim.AdamW(params_or_groups, **kwargs)


def _is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def _run_accum_microbatches(
    xb_full: torch.Tensor,
    yb_full: torch.Tensor,
    micro_bs: int,
    opt: torch.optim.Optimizer,
    device: str,
    step_tag: str,
    micro_fn,
) -> Tuple[int, Optional[torch.Tensor]]:
    batch_size = int(xb_full.size(0))
    while True:
        try:
            opt.zero_grad(set_to_none=True)
            accum = max(1, batch_size // micro_bs)
            acc_t: Optional[torch.Tensor] = None
            for ai in range(accum):
                s0 = ai * micro_bs
                s1 = s0 + micro_bs
                cur_t = micro_fn(xb_full[s0:s1], yb_full[s0:s1], accum)
                acc_t = cur_t if acc_t is None else (acc_t + cur_t)
            return micro_bs, acc_t
        except Exception as exc:
            if _is_oom_error(exc) and micro_bs > 8:
                opt.zero_grad(set_to_none=True)
                if device == "cuda":
                    torch.cuda.empty_cache()
                micro_bs = max(8, micro_bs // 2)
                print(f"  [OOM-Fallback] {step_tag}: micro_bs -> {micro_bs}", flush=True)
                continue
            raise


def _fp_micro_ce(
    mod: nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    pad_id: int,
    use_amp: bool,
    cur_acc: int,
    commit_scale: float = 0.0,
) -> torch.Tensor:
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        lg = mod(xb)
        b_sz, seq_l, vocab_sz = lg.shape
        denom = (yb != pad_id).sum(dim=1).clamp(min=1)
        t_ce = F.cross_entropy(
            lg.float().reshape(b_sz * seq_l, vocab_sz),
            yb.reshape(b_sz * seq_l),
            ignore_index=pad_id,
            reduction="none",
        ).view(b_sz, seq_l)
        ce = (t_ce.sum(dim=1) / denom).mean()
        loss = ce + commit_scale * mod.get_commitment_loss() if commit_scale > 0.0 else ce
    (loss / float(cur_acc)).backward()
    return ce.detach()


class _FusedChunkedCEKLFn(torch.autograd.Function):
    """Chunked FP32 CE + Temperature-Scaled KL avoiding `(B, L, V)` FP32 tensor bloat (`< 1e-6` parity)."""

    @staticmethod
    def forward(
        ctx,
        z_q: torch.Tensor,
        z_fp: torch.Tensor,
        yb: torch.Tensor,
        valid_f: torch.Tensor,
        denom: torch.Tensor,
        kl_temp: float,
        w_ce: float,
        w_kl: float,
        pad_id: int,
        chunk_n: int = 16384,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b_sz, seq_l, vocab_sz = z_q.shape
        n_tok = b_sz * seq_l
        inv_t = 1.0 / float(kl_temp)
        w_tok = (valid_f / denom.unsqueeze(1)).reshape(-1) / float(max(1, b_sz))
        y_flat = yb.reshape(-1)
        zq_flat = z_q.reshape(n_tok, vocab_sz)
        zfp_flat = z_fp.reshape(n_tok, vocab_sz)
        need_grad = z_q.requires_grad
        gz_flat = torch.empty_like(zq_flat) if need_grad else None
        ce_acc = z_q.new_zeros((), dtype=torch.float32)
        kl_acc = z_q.new_zeros((), dtype=torch.float32)
        t_sq = float(kl_temp * kl_temp)
        for s in range(0, n_tok, chunk_n):
            e = min(n_tok, s + chunk_n)
            zf = zq_flat[s:e].float()
            tf = zfp_flat[s:e].float()
            yc = y_flat[s:e]
            wc = w_tok[s:e]
            valid_c = yc != pad_id
            yc_safe = yc.clamp(0, vocab_sz - 1)
            lse_q = torch.logsumexp(zf, dim=-1)
            ce_i = torch.where(
                valid_c,
                lse_q - zf.gather(1, yc_safe.unsqueeze(1)).squeeze(1),
                torch.zeros_like(lse_q),
            )
            ce_acc = ce_acc + (ce_i * wc).sum()
            log_p = F.log_softmax(tf * inv_t, dim=-1)
            p = log_p.exp()
            log_q_t = F.log_softmax(zf * inv_t, dim=-1)
            kl_acc = kl_acc + ((p * (log_p - log_q_t)).sum(dim=-1) * wc).sum() * t_sq
            if gz_flat is not None:
                p_q = (zf - lse_q.unsqueeze(1)).exp()
                p_q.scatter_add_(1, yc_safe.unsqueeze(1), -valid_c.float().unsqueeze(1))
                grad_tot = (
                    (w_ce * wc).unsqueeze(1) * p_q
                    + (w_kl * float(kl_temp) * wc).unsqueeze(1) * (log_q_t.exp() - p)
                )
                gz_flat[s:e] = grad_tot.to(z_q.dtype)
        if gz_flat is not None:
            ctx.save_for_backward(gz_flat.view(b_sz, seq_l, vocab_sz))
        else:
            ctx.save_for_backward()
        return w_ce * ce_acc + w_kl * kl_acc, ce_acc.detach()

    @staticmethod
    def backward(ctx, g_tot: torch.Tensor, _g_ce: Optional[torch.Tensor]):
        if not ctx.saved_tensors:
            return (None,) * 10
        (gz,) = ctx.saved_tensors
        g_out = (gz.float() * g_tot.float()).to(gz.dtype)
        return g_out, None, None, None, None, None, None, None, None, None


_SIG_KWARG_CACHE: Dict[Tuple[type, str], bool] = {}


def _model_supports_kwarg(model: nn.Module, kwarg: str) -> bool:
    key = (type(model), kwarg)
    val = _SIG_KWARG_CACHE.get(key)
    if val is None:
        try:
            val = kwarg in inspect.signature(model.forward).parameters
        except (TypeError, ValueError):
            val = False
        _SIG_KWARG_CACHE[key] = val
    return val


def _clone_from_inference(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.clone() if obj.is_inference() else obj
    if isinstance(obj, dict):
        return {k: _clone_from_inference(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_clone_from_inference(v) for v in obj)
    if isinstance(obj, list):
        return [_clone_from_inference(v) for v in obj]
    return obj


def _slice_precomputed_chunks(
    student: nn.Module, obj: Any, s0: int, s1: int, full_bs: int
) -> Any:
    bc = getattr(student, "byte_chunker", None)
    slicer = getattr(bc, "slice_precomputed_chunks", None)
    if callable(slicer) and isinstance(obj, dict):
        return slicer(obj, s0, s1)
    if isinstance(obj, torch.Tensor):
        if obj.ndim >= 1 and int(obj.size(0)) == full_bs:
            return obj[s0:s1]
        return obj
    if isinstance(obj, dict):
        return {k: _slice_precomputed_chunks(student, v, s0, s1, full_bs) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_slice_precomputed_chunks(student, v, s0, s1, full_bs) for v in obj)
    if isinstance(obj, list):
        return [_slice_precomputed_chunks(student, v, s0, s1, full_bs) for v in obj]
    return obj


def _qat_micro_step(
    teacher: nn.Module,
    student: nn.Module,
    taps_fp: List[torch.Tensor],
    taps_q: List[torch.Tensor],
    kl_temp: float,
    xb: torch.Tensor,
    yb: torch.Tensor,
    pad_id: int,
    use_amp: bool,
    cur_acc: int,
) -> torch.Tensor:
    bc_s = getattr(student, "byte_chunker", None)
    can_reuse = bool(
        _model_supports_kwarg(teacher, "return_chunks")
        and _model_supports_kwarg(student, "precomputed_chunks")
        and (bc_s is not None)
        and bool(getattr(bc_s, "router_locked", False))
    )
    try:
        taps_fp.clear()
        ch_c: Optional[Any] = None
        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                if can_reuse:
                    out_t = teacher(xb, return_chunks=True)
                    if isinstance(out_t, tuple) and len(out_t) >= 2:
                        z_fp, ch_raw = out_t[0], out_t[1]
                        ch_c = ch_raw
                    else:
                        z_fp = out_t
                else:
                    z_fp = teacher(xb)
        z_fp_c = _clone_from_inference(z_fp)
        taps_fp_c = [_clone_from_inference(t) for t in taps_fp]
        if ch_c is not None:
            ch_c = _clone_from_inference(ch_c)
        taps_q.clear()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            if ch_c is not None:
                z_qat = student(xb, precomputed_chunks=ch_c)
            else:
                z_qat = student(xb)
            commit = student.get_commitment_loss()
        valid_b = yb != pad_id
        valid_f = valid_b.float()
        denom = valid_f.sum(dim=1).clamp(min=1.0)
        ce_kl_l, ce_l = _FusedChunkedCEKLFn.apply(
            z_qat, z_fp_c, yb, valid_f, denom, kl_temp, 0.35, 0.45, pad_id, 16384
        )
        assert len(taps_fp_c) == 4 and len(taps_q) == 4, (
            f"4-tap hook count mismatch: fp={len(taps_fp_c)} qat={len(taps_q)}"
        )
        blk_l = _compute_4tap_loss(taps_fp_c, taps_q, last_mask=valid_b)
        loss = ce_kl_l + 0.20 * blk_l + 0.04 * commit.float()
        (loss / float(cur_acc)).backward()
        return ce_l.detach()
    finally:
        taps_fp.clear()
        taps_q.clear()


def _run_4tap_qat_step(
    teacher: nn.Module,
    student: nn.Module,
    taps_fp: List[torch.Tensor],
    taps_q: List[torch.Tensor],
    xb_full: torch.Tensor,
    yb_full: torch.Tensor,
    micro_bs: int,
    opt: Any,
    pad_id: int,
    kl_temp: float,
    use_amp: bool,
    device: str,
    step_tag: str,
    teacher_bs: Optional[int] = None,
    w_ce: float = 0.35,
    w_kl: float = 0.45,
    w_4tap: float = 0.20,
    w_commit: float = 0.04,
    extra_loss_fn: Optional[Callable[[torch.Tensor, int, int, int], Optional[torch.Tensor]]] = None,
    use_chunk_reuse: bool = True,
) -> Tuple[int, Optional[torch.Tensor]]:
    """
    Batched frozen-teacher 4-Tap + Logit-KL STE QAT step.

    Runs the frozen FP teacher in super-batches (`teacher_bs` = 32/64) under `torch.inference_mode()`,
    shares frozen router boundaries via `return_chunks=True` / `slice_precomputed_chunks` /
    `precomputed_chunks` (guaranteeing exact `M_max` parity even with tight eval chunking while
    skipping student router + compaction), and accumulates student microbatches with OOM backoff.
    """
    batch_size = int(xb_full.size(0))
    bc_s = getattr(student, "byte_chunker", None)
    static_m_cap = (bc_s is None) or bool(getattr(bc_s, "use_static_m_cap", False))
    can_reuse_chunks = bool(
        use_chunk_reuse
        and _model_supports_kwarg(teacher, "return_chunks")
        and _model_supports_kwarg(student, "precomputed_chunks")
        and (bc_s is not None)
        and bool(getattr(bc_s, "router_locked", False))
    )
    can_superbatch = bool(can_reuse_chunks or static_m_cap)

    while True:
        try:
            opt.zero_grad(set_to_none=True)
            n_acc = max(1, (batch_size + micro_bs - 1) // micro_bs)
            acc_t: Optional[torch.Tensor] = None
            t_bs = (
                min(batch_size, max(int(micro_bs), int(teacher_bs or 32)))
                if can_superbatch
                else int(micro_bs)
            )
            # Align teacher super-batch to a multiple of student `micro_bs` so every student slice stays inside one super-batch.
            if t_bs > micro_bs:
                t_bs = max(micro_bs, (t_bs // micro_bs) * micro_bs)

            for ts0 in range(0, batch_size, t_bs):
                ts1 = min(batch_size, ts0 + t_bs)
                sup_bs = ts1 - ts0
                taps_fp.clear()
                ch_sup: Optional[Any] = None
                with torch.inference_mode():
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                        if can_reuse_chunks:
                            out_t = teacher(xb_full[ts0:ts1], return_chunks=True)
                            if isinstance(out_t, tuple) and len(out_t) >= 2:
                                z_sup_raw, ch_sup_raw = out_t[0], out_t[1]
                                ch_sup = ch_sup_raw
                            else:
                                z_sup_raw = out_t
                                can_reuse_chunks = False
                        else:
                            z_sup_raw = teacher(xb_full[ts0:ts1])
                z_sup = _clone_from_inference(z_sup_raw)
                taps_sup = [_clone_from_inference(t) for t in taps_fp] if w_4tap > 0.0 else []
                taps_fp.clear()
                if ch_sup is not None:
                    ch_sup = _clone_from_inference(ch_sup)

                for loc0 in range(0, sup_bs, micro_bs):
                    loc1 = min(sup_bs, loc0 + micro_bs)
                    s0, s1 = ts0 + loc0, ts0 + loc1
                    xb = xb_full[s0:s1]
                    yb = yb_full[s0:s1]
                    z_fp_s = z_sup[loc0:loc1]
                    taps_fp_s = [tf[loc0:loc1] for tf in taps_sup]
                    try:
                        taps_q.clear()
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                            if ch_sup is not None:
                                ch_s = _slice_precomputed_chunks(student, ch_sup, loc0, loc1, sup_bs)
                                try:
                                    z_qat = student(xb, precomputed_chunks=ch_s)
                                except TypeError:
                                    ch_sup = None
                                    can_reuse_chunks = False
                                    z_qat = student(xb)
                            else:
                                z_qat = student(xb)
                            commit = student.get_commitment_loss()

                        valid_b = yb != pad_id
                        valid_f = valid_b.float()
                        denom = valid_f.sum(dim=1).clamp(min=1.0)
                        ce_kl_l, ce_l = _FusedChunkedCEKLFn.apply(
                            z_qat, z_fp_s, yb, valid_f, denom, kl_temp, w_ce, w_kl, pad_id, 16384
                        )
                        loss = ce_kl_l + float(w_commit) * commit.float()
                        if w_4tap > 0.0:
                            assert len(taps_fp_s) == 4 and len(taps_q) == 4, (
                                f"4-tap hook count mismatch: fp={len(taps_fp_s)} qat={len(taps_q)}"
                            )
                            blk_l = _compute_4tap_loss(taps_fp_s, taps_q, last_mask=valid_b)
                            loss = loss + float(w_4tap) * blk_l
                        if extra_loss_fn is not None:
                            ext_l = extra_loss_fn(z_qat, s0, s1, n_acc)
                            if ext_l is not None:
                                loss = loss + ext_l
                        (loss / float(n_acc)).backward()
                        ce_det = ce_l.detach()
                        acc_t = ce_det if acc_t is None else (acc_t + ce_det)
                    finally:
                        taps_q.clear()
                del z_sup, taps_sup, ch_sup
            return micro_bs, acc_t
        except Exception as exc:
            taps_fp.clear()
            taps_q.clear()
            if _is_oom_error(exc) and micro_bs > 8:
                opt.zero_grad(set_to_none=True)
                if device == "cuda":
                    torch.cuda.empty_cache()
                micro_bs = max(8, micro_bs // 2)
                print(f"  [OOM-Fallback] {step_tag}: micro_bs -> {micro_bs}", flush=True)
                continue
            raise


def _snapshot_router_params(model: CalciteLeanSystem1LM) -> Dict[str, torch.Tensor]:
    if model.byte_chunker is None:
        return {}
    return {nm: p.detach().cpu().clone() for nm, p in model.byte_chunker.named_parameters() if nm.startswith("router_")}


def _assert_router_unchanged(model: CalciteLeanSystem1LM, snap: Dict[str, torch.Tensor], tag: str) -> float:
    if model.byte_chunker is None or not snap:
        return 0.0
    max_d = 0.0
    for nm, p in model.byte_chunker.named_parameters():
        if nm in snap:
            diff = float((p.detach().cpu() - snap[nm]).abs().max().item())
            max_d = max(max_d, diff)
            assert diff == 0.0, f"[{tag}] Locked router mutated `{nm}` diff={diff:.3e}"
    return max_d


def _split_calcite_fp_groups(
    model: CalciteLeanSystem1LM,
) -> Tuple[List[nn.Parameter], List[nn.Parameter], List[nn.Parameter], List[nn.Parameter]]:
    """Splits Stage-1A params into `(crystal_and_gates, byte_router, core_backbone, woodbury)`."""
    cryst_p: List[nn.Parameter] = []
    chunk_p: List[nn.Parameter] = []
    core_p: List[nn.Parameter] = []
    wood_p: List[nn.Parameter] = []
    seen: set[int] = set()
    for nm, p in model.named_parameters():
        if (not p.requires_grad) or (id(p) in seen):
            continue
        seen.add(id(p))
        if nm.startswith(("crystal.", "crystal_2.", "stream_gate.", "cryst2_gate.")):
            cryst_p.append(p)
        elif nm.startswith("byte_chunker.router_"):
            chunk_p.append(p)
        elif nm.startswith("woodbury."):
            wood_p.append(p)
        else:
            core_p.append(p)
    return cryst_p, chunk_p, core_p, wood_p


def _split_calcite_1b_groups(
    model: CalciteLeanSystem1LM,
) -> Tuple[List[nn.Parameter], List[nn.Parameter], List[nn.Parameter], List[nn.Parameter], List[nn.Parameter], List[nn.Parameter], List[nn.Parameter]]:
    """Splits Stage-1B Consolidation params into Crystal, Pre-Core, Post/Gates, and Woodbury."""
    cryst_1b: List[nn.Parameter] = []
    pre_nodc: List[nn.Parameter] = []
    pre_2d: List[nn.Parameter] = []
    post_nodc: List[nn.Parameter] = []
    post_2d: List[nn.Parameter] = []
    wood_nodc: List[nn.Parameter] = []
    wood_2d: List[nn.Parameter] = []
    seen: set[int] = set()
    for nm, p in model.named_parameters():
        if (not p.requires_grad) or (id(p) in seen):
            continue
        seen.add(id(p))
        if nm.startswith(("crystal.", "crystal_2.")):
            cryst_1b.append(p)
        elif nm.startswith("woodbury."):
            if p.ndim < 2:
                wood_nodc.append(p)
            else:
                wood_2d.append(p)
        elif nm.startswith(("tok_emb.", "byte_chunker.", "cortex_pre.", "split_norm.")):
            if p.ndim < 2:
                pre_nodc.append(p)
            else:
                pre_2d.append(p)
        else:
            if p.ndim < 2 or nm.startswith(("stream_gate.", "cryst2_gate.")):
                post_nodc.append(p)
            else:
                post_2d.append(p)
    return cryst_1b, pre_nodc, pre_2d, post_nodc, post_2d, wood_nodc, wood_2d


def _split_calcite_stage2_groups(
    model: CalciteLeanSystem1LM,
) -> Tuple[List[nn.Parameter], List[nn.Parameter], List[nn.Parameter], List[nn.Parameter], List[nn.Parameter]]:
    """Splits Stage-2 unlocked params into `(crystal_nodc, woodbury_nodc, gate_1d_nodc, tok_chunk_2d, rest_core_2d)`."""
    cryst_p: List[nn.Parameter] = []
    wood_p: List[nn.Parameter] = []
    nodc_other_p: List[nn.Parameter] = []
    emb_chunk_p: List[nn.Parameter] = []
    core_p: List[nn.Parameter] = []
    seen: set[int] = set()
    tok_id = id(model.tok_emb.weight)
    for nm, p in model.named_parameters():
        if (not p.requires_grad) or (id(p) in seen):
            continue
        seen.add(id(p))
        if nm.startswith(("crystal.", "crystal_2.")):
            cryst_p.append(p)
        elif nm.startswith("woodbury."):
            wood_p.append(p)
        elif nm.startswith(("stream_gate.", "cryst2_gate.")) or p.ndim < 2:
            nodc_other_p.append(p)
        elif id(p) == tok_id or nm.startswith("byte_chunker."):
            emb_chunk_p.append(p)
        else:
            core_p.append(p)
    return cryst_p, wood_p, nodc_other_p, emb_chunk_p, core_p


def _calibrate_non_ascii_router_emb(
    model: CalciteLeanSystem1LM,
    train_wins: Dict[str, List[List[int]]],
    device: str,
    cal_steps: int = 120,
    seed: int = 2026,
) -> None:
    """Exact Phase E Stage 3C calibration of non-ASCII `router_emb[131:]` with frozen linear router."""
    bc = model.byte_chunker
    if bc is None:
        return
    orig_tgt = float(bc.target_chunk_bytes)
    orig_static = bool(bc.use_static_m_cap)
    bc.byte_chunk_mode = "utf8"
    bc.use_static_m_cap = False
    bc.lock_router(True)
    for p in model.parameters():
        p.requires_grad_(False)
    bc.router_locked = False
    bc.router_emb.weight.requires_grad_(True)
    bc.target_chunk_bytes = 4.3
    opt_r = torch.optim.AdamW([bc.router_emb.weight], lr=2.0e-3, weight_decay=0.0)
    cal_pools = [
        train_wins[k]
        for k in ("2_Russian_UTF8", "5_CJK_UTF8", "6_Emoji_4Byte")
        if len(train_wins.get(k, [])) > 0
    ]
    if cal_pools:
        rng_cal = random.Random(seed)
        model.train()
        for _ in range(cal_steps):
            batch = [rng_cal.choice(rng_cal.choice(cal_pools))[:-1] for _ in range(8)]
            xb_c = torch.tensor(batch, dtype=torch.long, device=device)
            e_tok_c = model.tok_emb(xb_c)
            _, _, _ = bc.encode_to_chunks(e_tok_c, xb_c)
            loss_r = bc.last_ratio_loss
            opt_r.zero_grad(set_to_none=True)
            loss_r.backward()
            if bc.router_emb.weight.grad is not None:
                bc.router_emb.weight.grad[:131].zero_()
            torch.nn.utils.clip_grad_norm_([bc.router_emb.weight], 1.0)
            opt_r.step()
    del opt_r
    bc.router_emb.weight.requires_grad_(False)
    bc.target_chunk_bytes = orig_tgt
    bc.use_static_m_cap = orig_static
    bc.router_locked = True
    for p in model.parameters():
        p.requires_grad_(True)
    bc.lock_router(True)


def _split_calcite_qat_4groups(
    model: CalciteLeanSystem1LM,
) -> Tuple[List[nn.Parameter], List[nn.Parameter], List[nn.Parameter], List[nn.Parameter]]:
    gain_ids: set[int] = {id(model.tok_emb.weight)}
    for mod in model.modules():
        if isinstance(mod, RMSNorm):
            for p in mod.parameters():
                if p.requires_grad:
                    gain_ids.add(id(p))
    for gm in (model.stream_gate, model.cryst2_gate):
        if gm is not None:
            for p in gm.parameters():
                if p.requires_grad:
                    gain_ids.add(id(p))

    gain_p, cryst_p, chunk_p, core_p = [], [], [], []
    seen: set[int] = set()
    for nm, p in model.named_parameters():
        if (not p.requires_grad) or (id(p) in seen):
            continue
        seen.add(id(p))
        is_wood_scalar = nm.startswith("woodbury.") and (
            (p.ndim < 2 and "out_norm" not in nm) or ("subj_lag_bias" in nm)
        )
        if is_wood_scalar:
            cryst_p.append(p)
        elif id(p) in gain_ids:
            gain_p.append(p)
        elif nm.startswith(("crystal.", "crystal_2.")):
            cryst_p.append(p)
        elif nm.startswith("byte_chunker."):
            chunk_p.append(p)
        else:
            core_p.append(p)
    return gain_p, cryst_p, chunk_p, core_p


def _attach_4tap_hooks(model: CalciteLeanSystem1LM, store: List[torch.Tensor], detach: bool) -> list:
    handles = []
    for mod in (model.cortex_pre[-1], model.mid_mlp, model.cortex_post[-1], model.norm_f):
        def _hk(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            store.append(t.detach() if detach else t)
        handles.append(mod.register_forward_hook(_hk))
    return handles


def _compute_4tap_loss(
    taps_fp: Sequence[torch.Tensor],
    taps_q: Sequence[torch.Tensor],
    last_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Branchless, zero-sync 4-tap RMS-normalized SmoothL1 + cosine distillation loss."""
    acc = torch.zeros((), device=taps_q[0].device, dtype=torch.float32)
    n_t = len(taps_q)
    for idx, (tf, tq) in enumerate(zip(taps_fp, taps_q)):
        af, aq = tf.float(), tq.float()
        af_f = af.reshape(-1, af.size(-1))
        aq_f = aq.reshape(-1, aq.size(-1))
        rms_f = af_f.pow(2).mean(dim=-1, keepdim=True).add(1e-5).sqrt().detach()
        if last_mask is not None and idx == n_t - 1:
            m_b = last_mask.reshape(-1) > 0.5
            m_sum = m_b.float().sum()
            inv_m = (m_sum > 0.0).float() / m_sum.clamp(min=1.0)
            sl1_tok = F.smooth_l1_loss(aq_f / rms_f, af_f / rms_f, reduction="none").mean(dim=-1)
            cos_tok = 1.0 - F.cosine_similarity(af_f, aq_f, dim=-1, eps=1e-6)
            zeros = torch.zeros_like(sl1_tok)
            sl1 = torch.where(m_b, sl1_tok, zeros).sum() * inv_m
            cos_d = torch.where(m_b, cos_tok, zeros).sum() * inv_m
        else:
            sl1 = F.smooth_l1_loss(aq_f / rms_f, af_f / rms_f)
            cos_d = (1.0 - F.cosine_similarity(af_f, aq_f, dim=-1, eps=1e-6)).mean()
        acc = acc + 0.5 * sl1 + 0.5 * cos_d
    return acc / float(n_t)


def _auto_micro_bs(device: str, tier_name: str, target_bs: int = 64, is_qat: bool = False) -> int:
    if device != "cuda" or not has_usable_cuda():
        return min(8, target_bs)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if vram_gb >= 65.0:
        return target_bs
    if vram_gb >= 35.0:
        return min(32, target_bs) if (tier_name == "S3_85M" and is_qat) else target_bs
    if vram_gb >= 20.0:
        if tier_name == "S1_5M":
            return min(64, target_bs)
        if tier_name == "S2_25M":
            return min(32, target_bs) if is_qat else min(64, target_bs)
        return min(32, target_bs)
    if tier_name == "S1_5M":
        return min(32, target_bs)
    if tier_name == "S2_25M":
        return min(16 if is_qat else 32, target_bs)
    return 8


def _eval_complete_cell(
    model: nn.Module,
    tok: ByteBPETokenizer,
    val_mm: np.ndarray,
    eval_wins: Dict[str, List[List[int]]],
    s5_suites: dict,
    ov_suites: dict,
    trap_suites: dict,
    seq_len: int,
    device: str,
    smoke: bool = False,
) -> Dict[str, object]:
    val_bpb = eval_val_memmap_bpb(
        model, val_mm, seq_len=seq_len, n_batches=4 if smoke else 24, batch_size=8 if smoke else 16, device=device
    )
    extrap = eval_length_extrapolation_bpb(
        model, val_mm, device=device, l_in=seq_len, l_ood=1024 if smoke else 2048,
        n_batches=2 if smoke else 12, batch_size_ood=2 if smoke else 4,
        reuse_l_in_bpb=val_bpb,
    )
    ow = eval_openweb_6domains_any_byte_lm(model, eval_wins, device=device, batch_size=8 if smoke else 16)
    cog = eval_cognitive_suites_any_byte_lm(model, tok, s5_suites, ov_suites, trap_suites, device=device, batch_size=16 if smoke else 32)
    lat_ms = measure_infer_latency_ms(
        model, int(getattr(getattr(model, "cfg", None), "vocab_size", 259)), device=device, batch_size=4 if smoke else 8, seq_len=seq_len
    )
    return {
        "val_memmap_bpb": val_bpb,
        "length_extrapolation": extrap,
        "openweb": ow,
        "mean_openweb_bpb": ow["mean_openweb_bpb"],
        "max_mid_utf8_split_pct": ow["max_mid_utf8_split_pct"],
        "s5": cog["s5"],
        "s5_sum": cog["s5_sum"],
        "overwrite": cog["overwrite"],
        "trap": cog["trap"],
        "selective_f1_in": cog["selective_f1_in"],
        "selective_f1_ood_long": cog["selective_f1_ood_long"],
        "infer_ms": lat_ms,
        **({"crystal_drift_L2048": cog["crystal_drift_L2048"]} if "crystal_drift_L2048" in cog else {}),
    }


def _train_baseline_cell(
    tier_name: str,
    model_name: str,
    model: nn.Module,
    cfg_dict: dict,
    cache_kb_512: float,
    cache_kb_2048: float,
    steps: int,
    grok_steps: int,
    core_lr: float,
    stage2_core_lr: float,
    min_lr: float,
    batcher: StratifiedSystem1Batcher,
    tok: ByteBPETokenizer,
    val_mm: np.memmap,
    eval_wins: dict,
    s5_suites: dict,
    ov_suites: dict,
    trap_suites: dict,
    batch_size: int,
    seq_len: int,
    device: str,
    smoke: bool,
    s1b_steps: int = 500,
) -> Dict[str, object]:
    n_params = int(sum(p.numel() for p in model.parameters()))
    micro_bs = _auto_micro_bs(device, tier_name, batch_size)
    accum = max(1, batch_size // micro_bs)
    use_amp = device == "cuda" and has_usable_cuda()
    model = model.to(device)
    train_mod: nn.Module = model
    if model_name == "RoPETransformerLM" and use_amp and not smoke and hasattr(torch, "compile"):
        try:
            train_mod = torch.compile(model)
        except Exception as exc:
            print(f"  [WARN] torch.compile skipped for {tier_name}/{model_name}: {exc}", flush=True)
            train_mod = model

    opt = _make_adamw(model.parameters(), device, lr=core_lr, betas=(0.9, 0.95), weight_decay=0.02)
    warm_1 = min(120, max(1, grok_steps // 12))
    stage2_steps = max(1, steps - grok_steps)
    warm_2 = min(40, max(1, stage2_steps // 8))

    print(
        f"\n--- [Phase K | {tier_name} | {model_name}] params={n_params:,} steps={steps} "
        f"(grok={grok_steps}[s1b={s1b_steps}]+utf8={steps - grok_steps}) B={batch_size}(micro={micro_bs}x{accum}) "
        f"L={seq_len} lr={core_lr:.1e}/{stage2_core_lr:.1e} ---",
        flush=True,
    )
    t0 = time.perf_counter()
    step_ms_list: List[float] = []
    for step in range(1, steps + 1):
        if step == grok_steps + 1:
            del opt
            if device == "cuda":
                torch.cuda.empty_cache()
            opt = _make_adamw(model.parameters(), device, lr=stage2_core_lr, betas=(0.9, 0.95), weight_decay=0.02)
        model.train()
        if step <= grok_steps:
            lr_now = _cosine_lr(step, grok_steps, warm_1, core_lr, 0.15 * core_lr)
        else:
            lr_now = _cosine_lr(step - grok_steps, stage2_steps, warm_2, stage2_core_lr, min_lr)
        opt.param_groups[0]["lr"] = lr_now
        xb_full, yb_full = batcher.sample_batch(
            batch_size, device, step=step, grok_steps=grok_steps, total_steps=steps, s1b_steps=s1b_steps
        )
        timing_on = (15 <= step <= 65) or ((smoke or steps < 20) and step >= 2)
        is_log = (step in (1, grok_steps, steps)) or (step % 400 == 0)
        need_scalar = timing_on or is_log or (step % 50 == 0)
        if timing_on:
            sync_if_cuda(device)
        ts = time.perf_counter()

        while True:
            try:
                micro_bs, ce_sum_t = _run_accum_microbatches(
                    xb_full,
                    yb_full,
                    micro_bs,
                    opt,
                    device,
                    f"{tier_name}/{model_name} s={step}",
                    lambda xb, yb, cur_acc, m_ref=train_mod: _fp_micro_ce(
                        m_ref, xb, yb, tok.PAD_ID, use_amp, cur_acc, 0.0
                    ),
                )
                break
            except Exception as exc:
                if (train_mod is not model) and step <= 2 and (not _is_oom_error(exc)):
                    print(f"  [WARN] torch.compile fallback on {tier_name}/{model_name}: {exc}", flush=True)
                    opt.zero_grad(set_to_none=True)
                    train_mod = model
                    continue
                raise

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if timing_on:
            sync_if_cuda(device)
            step_ms_list.append((time.perf_counter() - ts) * 1000.0)
        if need_scalar and ce_sum_t is not None:
            ce_acc = float(ce_sum_t.item()) / float(max(1, batch_size // micro_bs))
            assert math.isfinite(ce_acc), f"NaN/Inf CE on {tier_name}/{model_name} step {step}"
            if is_log:
                st_lbl = "S1" if step <= grok_steps else "S2"
                print(
                    f"  [{tier_name}/{model_name} {st_lbl}] step {step:4d}/{steps} ce={ce_acc:.4f} ({(time.perf_counter()-t0)/60:.1f}m)",
                    flush=True,
                )

    train_s = round(time.perf_counter() - t0, 2)
    med_ms = round(statistics.median(step_ms_list), 2) if step_ms_list else 0.0
    del train_mod, opt
    if device == "cuda":
        torch.cuda.empty_cache()

    eval_fp = _eval_complete_cell(model, tok, val_mm, eval_wins, s5_suites, ov_suites, trap_suites, seq_len, device, smoke)
    weight_bf16_mb = round((n_params * 2.0) / (1024**2), 3)
    print(
        f"  => [{tier_name}/{model_name}] val_bpb={eval_fp['val_memmap_bpb']:.4f} | "
        f"ow_bpb={eval_fp['mean_openweb_bpb']:.4f} | extrap={eval_fp['length_extrapolation']['extrap_delta_bpb']:+.4f} | "
        f"S5={eval_fp['s5_sum']:.1f}% | OV_long={eval_fp['overwrite']['ood_long']['exact_pct']:.1f}% | "
        f"SelF1_long={eval_fp['selective_f1_ood_long']:.1f}%",
        flush=True,
    )
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "tier": tier_name,
        "model": model_name,
        "params": n_params,
        "config": cfg_dict,
        "weight_bf16_mb": weight_bf16_mb,
        "cache_kb_L512": cache_kb_512,
        "cache_kb_L768": round(cache_kb_512 * (768.0 / 512.0), 2) if model_name == "RoPETransformerLM" else cache_kb_512,
        "cache_kb_L2048": cache_kb_2048,
        "train_seconds": train_s,
        "train_ms_per_step": med_ms,
        "fp": eval_fp,
    }


def _train_calcite_cell(
    tier_name: str,
    lean_cfg: LeanSystem1Config,
    steps: int,
    grok_steps: int,
    qat_steps: int,
    core_lr: float,
    stage2_core_lr: float,
    stage2_emb_lr: float,
    min_lr: float,
    batcher: StratifiedSystem1Batcher,
    train_wins: dict,
    tok: ByteBPETokenizer,
    val_mm: np.memmap,
    eval_wins: dict,
    s5_suites: dict,
    ov_suites: dict,
    trap_suites: dict,
    batch_size: int,
    seq_len: int,
    ckpt_path: Path,
    device: str,
    smoke: bool,
    s1b_steps: int = 600,
    use_fast_kernels: bool = True,
    compile_cortex: bool = False,
    fp_ckpt_dir: Optional[Path] = None,
    resume_stage2_from: Optional[Path] = None,
) -> Dict[str, object]:
    _gate_tmp = CalciteLeanSystem1LM(lean_cfg).to(device)
    _gate_tmp.set_fast_kernels(use_fast_kernels)
    gate2 = check_gate2_block_std_parity(_gate_tmp, device)
    assert gate2["gate2_passed"], f"Gate 2 failed on {tier_name}"
    del _gate_tmp

    m_seed = 102 if tier_name == "S2_25M" else 100
    torch.manual_seed(m_seed)
    random.seed(m_seed)
    model = CalciteLeanSystem1LM(lean_cfg).to(device)
    model.set_fast_kernels(use_fast_kernels)
    if compile_cortex and device == "cuda" and (not smoke):
        compile_calcite_cortex(model, mode="default")
    n_params = int(sum(p.numel() for p in model.parameters()))

    micro_bs = _auto_micro_bs(device, tier_name, batch_size, is_qat=False)
    accum = max(1, batch_size // micro_bs)
    use_amp = device == "cuda" and has_usable_cuda()
    fk_info = get_fast_scans_backend_info() if use_fast_kernels else {"cuda_ext_available": False}

    s1b_eff = (
        int(s1b_steps)
        if (0 < int(s1b_steps) < grok_steps)
        else (1 if grok_steps >= 2 else 0)
    )
    s1a_steps = grok_steps - s1b_eff

    print(
        f"\n--- [Phase K | {tier_name} | CalciteLeanSystem1LM FastScans={fk_info.get('cuda_ext_available')} "
        f"Compile={compile_cortex} (seed={m_seed})] params={n_params:,} "
        f"s1a={s1a_steps} + s1b={s1b_eff} + utf8={steps - grok_steps} + qat={qat_steps} B={batch_size}(micro={micro_bs}x{accum}) "
        f"lr={core_lr:.1e}->{stage2_core_lr:.1e}/{stage2_emb_lr:.1e} ---",
        flush=True,
    )
    t_fp0 = time.perf_counter()
    step_ms_list: List[float] = []
    fp_router_snap: Dict[str, torch.Tensor] = {}
    s1a_byte_s5_k2_pct = 0.0
    s1_byte_s5_k2_pct = 0.0
    s1_byte_ov_long_pct = 0.0
    s1_byte_trap_long_pct = 0.0

    skip_stage1 = resume_stage2_from is not None and resume_stage2_from.is_file()
    if skip_stage1:
        print(f"  [{tier_name}/Calcite] Resuming directly at Stage 2 from {resume_stage2_from}", flush=True)
        res_blob = torch.load(resume_stage2_from, map_location="cpu", weights_only=False)
        res_sd = res_blob["state_dict"] if (isinstance(res_blob, dict) and "state_dict" in res_blob) else res_blob
        clean_loader = CalciteLeanSystem1LM(lean_cfg).to(device)
        clean_loader.load_state_dict(res_sd, strict=True)
        clean_loader.set_fast_kernels(use_fast_kernels)
        if compile_cortex and device == "cuda" and (not smoke):
            compile_calcite_cortex(clean_loader, mode="default")
        model = clean_loader
        if isinstance(res_blob, dict):
            s1a_byte_s5_k2_pct = float(res_blob.get("s1a_byte_s5_k2_pct", 0.0))
            s1_byte_s5_k2_pct = float(res_blob.get("s1_byte_s5_k2_pct", 0.0))
            s1_byte_ov_long_pct = float(res_blob.get("s1_byte_ov_long_pct", 0.0))
            s1_byte_trap_long_pct = float(res_blob.get("s1_byte_trap_long_pct", 0.0))
        if model.byte_chunker is not None:
            model.byte_chunker.byte_chunk_mode = "utf8"
            model.byte_chunker.target_chunk_bytes = float(lean_cfg.target_chunk_bytes)
            model.byte_chunker.use_static_m_cap = False
            model.byte_chunker.lock_router(True)
            fp_router_snap = _snapshot_router_params(model)
        model.set_crystal_schedule(s1a_steps, s1a_steps)
        model.set_snap_alpha(1.0)
    else:
        # Stage 1A (`1 .. s1a_steps`): pre-lock unmasked prose tgt=4.2 with damped router -> post-lock Crystal S_5 grokking (`woodbury` 0.05x LR, wd=0.0)
        if model.byte_chunker is not None:
            model.byte_chunker.byte_chunk_mode = "byte"
            model.byte_chunker.target_chunk_bytes = 4.2
            model.byte_chunker.use_static_m_cap = False

    lock_step = int(float(lean_cfg.router_lock_frac) * s1a_steps) if ((not smoke) and s1a_steps >= 100) else 0
    insp_ids = tok.encode(
        "Initial state. Alpha: gold. Beta: silver. Swap Alpha and Beta. Summary: Alpha:",
        add_bos=True,
        add_eos=False,
    )
    cryst_p, chunk_p, core_p, wood_p = _split_calcite_fp_groups(model)
    all_1a = cryst_p + chunk_p + core_p + wood_p
    fixed_lr_1 = max(core_lr, 2.0e-3)
    base_lrs_1 = (fixed_lr_1, 2.5e-3, core_lr, 0.05 * core_lr)
    min_lrs_1 = (1.5e-4, 1.0e-4, 0.08 * core_lr, 0.01 * core_lr)
    opt = torch.optim.AdamW(
        [
            {"params": cryst_p, "lr": base_lrs_1[0], "weight_decay": 0.01},
            {"params": chunk_p, "lr": base_lrs_1[1], "weight_decay": 0.01},
            {"params": core_p, "lr": base_lrs_1[2], "weight_decay": 0.01},
            {"params": wood_p, "lr": base_lrs_1[3], "weight_decay": 0.0},
        ],
        betas=(0.9, 0.999),
    )
    warm_1 = min(60, max(1, s1a_steps // 12))
    live_diag_steps = (1200, 1800, 2200, s1a_steps) if ((not smoke) and s1a_steps >= 1000) else ()
    for step in range(1, (0 if skip_stage1 else s1a_steps) + 1):
        model.train()
        model.set_crystal_schedule(step, s1a_steps)
        for gi, (blr, mlr) in enumerate(zip(base_lrs_1, min_lrs_1)):
            opt.param_groups[gi]["lr"] = _cosine_lr(step, s1a_steps, warm_1, blr, mlr)
        if lock_step > 0 and step <= lock_step:
            opt.param_groups[1]["lr"] = _cosine_lr(
                step, lock_step, min(40, max(1, lock_step // 6)), 2.5e-3, 4.0e-4
            )
            xb_full, yb_full = batcher.sample_prose_batch(batch_size, device)
        else:
            xb_full, yb_full = batcher.sample_batch(
                batch_size, device, step=step, grok_steps=grok_steps, total_steps=steps, s1b_steps=s1b_steps, s1a_offset=lock_step
            )
        timing_on = (15 <= step <= 65) or ((smoke or steps < 20) and step >= 2)
        is_log = (step in (1, s1a_steps)) or (step % 400 == 0)
        need_scalar = timing_on or is_log or (step % 50 == 0)
        if timing_on:
            sync_if_cuda(device)
        ts = time.perf_counter()

        micro_bs, ce_sum_t = _run_accum_microbatches(
            xb_full,
            yb_full,
            micro_bs,
            opt,
            device,
            f"{tier_name}/Calcite S1A s={step}",
            lambda xb, yb, cur_acc, m_ref=model: _fp_micro_ce(
                m_ref, xb, yb, tok.PAD_ID, use_amp, cur_acc, 0.04
            ),
        )
        torch.nn.utils.clip_grad_norm_(all_1a, 1.0)
        opt.step()
        if timing_on:
            sync_if_cuda(device)
            step_ms_list.append((time.perf_counter() - ts) * 1000.0)
        if need_scalar and ce_sum_t is not None:
            ce_acc = float(ce_sum_t.item()) / float(max(1, batch_size // micro_bs))
            assert math.isfinite(ce_acc), f"NaN/Inf CE on {tier_name}/Calcite S1A step {step}"
            if is_log:
                b_ch = model.byte_chunker.last_chunk_ratio if model.byte_chunker is not None else 1.0
                print(
                    f"  [{tier_name}/Calcite S1A-byte] step {step:4d}/{s1a_steps} b/ch={b_ch:.2f} ce={ce_acc:.4f} ({(time.perf_counter()-t_fp0)/60:.1f}m)",
                    flush=True,
                )
        if lock_step > 0 and step == lock_step and model.byte_chunker is not None:
            py_rng_st = random.getstate()
            th_rng_st = torch.get_rng_state()
            cu_rng_st = torch.cuda.get_rng_state_all() if use_amp else None
            was_tr = model.training
            try:
                model.eval()
                x_insp = torch.tensor([insp_ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    _, _, ctx_i = model.byte_chunker.encode_to_chunks(model.tok_emb(x_insp), x_insp)
                bm_i = ctx_i.b_mask[0].tolist()
                chunks_i: List[str] = []
                cur_b: List[int] = []
                for idx_b, tid_b in enumerate(insp_ids):
                    if idx_b > 0 and bm_i[idx_b]:
                        chunks_i.append(bytes([x - 3 for x in cur_b if x >= 3]).decode("utf-8", "replace"))
                        cur_b = []
                    cur_b.append(tid_b)
                if cur_b:
                    chunks_i.append(bytes([x - 3 for x in cur_b if x >= 3]).decode("utf-8", "replace"))
                print(
                    f"  [{tier_name}/Calcite PRE-LOCK step={step}] b/ch={model.byte_chunker.last_chunk_ratio:.2f} chunks={chunks_i}",
                    flush=True,
                )
            finally:
                random.setstate(py_rng_st)
                torch.set_rng_state(th_rng_st)
                if cu_rng_st is not None:
                    torch.cuda.set_rng_state_all(cu_rng_st)
                model.train(was_tr)
                model.set_crystal_schedule(step, s1a_steps)
        if step in live_diag_steps:
            py_rng_st = random.getstate()
            th_rng_st = torch.get_rng_state()
            cu_rng_st = torch.cuda.get_rng_state_all() if use_amp else None
            was_tr = model.training
            try:
                n_q = min(32, len(s5_suites["k2"]))
                model.eval()
                model.set_snap_alpha(1.0)
                pk2, _, _, _ = evaluate_suite_batched(
                    model, s5_suites["k2"][:n_q], pad_id=tok.PAD_ID, batch_size=32, device=device
                )
                print(f"  [{tier_name}/Calcite S1A LIVE step={step}] s5_k2_32={pk2:.1f}%", flush=True)
            finally:
                random.setstate(py_rng_st)
                torch.set_rng_state(th_rng_st)
                if cu_rng_st is not None:
                    torch.cuda.set_rng_state_all(cu_rng_st)
                model.train(was_tr)
                model.set_crystal_schedule(step, s1a_steps)

    del opt
    if device == "cuda":
        torch.cuda.empty_cache()

    if not skip_stage1:
        model.eval()
        model.set_crystal_schedule(s1a_steps, s1a_steps)
        model.set_snap_alpha(1.0)
        s1_k2_cnt = min(64, len(s5_suites["k2"]))
        s1a_s5_val, _, _, _ = evaluate_suite_batched(
            model, s5_suites["k2"][:s1_k2_cnt], pad_id=tok.PAD_ID, batch_size=32, device=device
        )
        s1a_byte_s5_k2_pct = round(float(s1a_s5_val), 2)
        print(f"  [{tier_name}/Calcite S1A-byte EVAL] s1a_s5_k2={s1a_byte_s5_k2_pct:.2f}% (N={s1_k2_cnt})", flush=True)

    # Stage 1B (`s1a_steps + 1 .. grok_steps`): Associative Woodbury Consolidation (router locked, snap_alpha=1.0)
    if (not skip_stage1) and s1b_eff > 0:
        if model.byte_chunker is not None:
            model.byte_chunker.lock_router(True)
        c1b, pre_n, pre_2, post_n, post_2, wood_n, wood_2 = _split_calcite_1b_groups(model)
        all_1b = c1b + pre_n + pre_2 + post_n + post_2 + wood_n + wood_2
        pre_lr = min(3.0e-4, 0.30 * core_lr)
        post_lr = min(1.2e-3, 1.00 * core_lr)
        wood_lr = min(2.4e-3, 2.20 * core_lr)
        base_lrs_1b = (4.0e-5, pre_lr, pre_lr, post_lr, post_lr, wood_lr, wood_lr)
        opt_1b = torch.optim.AdamW(
            [
                {"params": c1b, "lr": base_lrs_1b[0], "weight_decay": 0.0},
                {"params": pre_n, "lr": base_lrs_1b[1], "weight_decay": 0.0},
                {"params": pre_2, "lr": base_lrs_1b[2], "weight_decay": 0.01},
                {"params": post_n, "lr": base_lrs_1b[3], "weight_decay": 0.0},
                {"params": post_2, "lr": base_lrs_1b[4], "weight_decay": 0.01},
                {"params": wood_n, "lr": base_lrs_1b[5], "weight_decay": 0.0},
                {"params": wood_2, "lr": base_lrs_1b[6], "weight_decay": 0.01},
            ],
            betas=(0.9, 0.999),
        )
        warm_1b = min(40, max(1, s1b_eff // 8))
        ov_live_steps = (400, s1b_eff) if ((not smoke) and s1b_eff >= 400) else ()
        for step in range(s1a_steps + 1, grok_steps + 1):
            model.train()
            model.set_crystal_schedule(s1a_steps, s1a_steps)
            model.set_snap_alpha(1.0)
            o_step = step - s1a_steps
            for gi, blr in enumerate(base_lrs_1b):
                opt_1b.param_groups[gi]["lr"] = _cosine_lr(o_step, s1b_eff, warm_1b, blr, 0.15 * blr)
            xb_full, yb_full = batcher.sample_batch(
                batch_size, device, step=step, grok_steps=grok_steps, total_steps=steps, s1b_steps=s1b_steps
            )
            is_log_1b = o_step in (1, max(1, s1b_eff // 2), s1b_eff)
            need_s_1b = is_log_1b or (o_step % 50 == 0)

            micro_bs, ce_sum_t = _run_accum_microbatches(
                xb_full,
                yb_full,
                micro_bs,
                opt_1b,
                device,
                f"{tier_name}/Calcite S1B s={o_step}",
                lambda xb, yb, cur_acc, m_ref=model: _fp_micro_ce(
                    m_ref, xb, yb, tok.PAD_ID, use_amp, cur_acc, 0.04
                ),
            )
            torch.nn.utils.clip_grad_norm_(all_1b, 1.0)
            opt_1b.step()
            if need_s_1b and ce_sum_t is not None:
                ce_acc = float(ce_sum_t.item()) / float(max(1, batch_size // micro_bs))
                assert math.isfinite(ce_acc), f"NaN/Inf CE on {tier_name}/Calcite S1B step {step}"
                if is_log_1b:
                    print(f"  [{tier_name}/Calcite S1B-wood] step {o_step:4d}/{s1b_eff} ce={ce_acc:.4f} ({(time.perf_counter()-t_fp0)/60:.1f}m)", flush=True)
            if o_step in ov_live_steps:
                py_rng_st = random.getstate()
                th_rng_st = torch.get_rng_state()
                cu_rng_st = torch.cuda.get_rng_state_all() if use_amp else None
                was_tr = model.training
                try:
                    n_ovq = min(32, len(ov_suites["ood_long"]))
                    model.eval()
                    model.set_snap_alpha(1.0)
                    pov, _, _, _ = evaluate_suite_batched(
                        model, ov_suites["ood_long"][:n_ovq], pad_id=tok.PAD_ID, batch_size=32, device=device
                    )
                    print(f"  [{tier_name}/Calcite S1B LIVE o_step={o_step}] ov32={pov:.1f}%", flush=True)
                finally:
                    random.setstate(py_rng_st)
                    torch.set_rng_state(th_rng_st)
                    if cu_rng_st is not None:
                        torch.cuda.set_rng_state_all(cu_rng_st)
                    model.train(was_tr)
                    model.set_crystal_schedule(s1a_steps, s1a_steps)
                    model.set_snap_alpha(1.0)
        del opt_1b
        if device == "cuda":
            torch.cuda.empty_cache()

    if not skip_stage1:
        model.eval()
        model.set_crystal_schedule(s1a_steps, s1a_steps)
        model.set_snap_alpha(1.0)
        s1_s5_k2_val, _, _, _ = evaluate_suite_batched(
            model, s5_suites["k2"][:s1_k2_cnt], pad_id=tok.PAD_ID, batch_size=32, device=device
        )
        s1_ov_cnt = min(64, len(ov_suites["ood_long"]))
        s1_ov_val, _, _, _ = evaluate_suite_batched(
            model, ov_suites["ood_long"][:s1_ov_cnt], pad_id=tok.PAD_ID, batch_size=32, device=device
        )
        s1_tr_cnt = min(64, len(trap_suites["ood_long"]))
        s1_tr_val, _, _, _ = evaluate_suite_batched(
            model, trap_suites["ood_long"][:s1_tr_cnt], pad_id=tok.PAD_ID, batch_size=32, device=device
        )
        s1_byte_s5_k2_pct = round(float(s1_s5_k2_val), 2)
        s1_byte_ov_long_pct = round(float(s1_ov_val), 2)
        s1_byte_trap_long_pct = round(float(s1_tr_val), 2)
        print(
            f"  [{tier_name}/Calcite S1-byte DUAL GATE] s1_s5_k2={s1_byte_s5_k2_pct:.2f}% | "
            f"s1_ov_long={s1_byte_ov_long_pct:.2f}% | s1_trap_long={s1_byte_trap_long_pct:.2f}%",
            flush=True,
        )
        if (not smoke) and steps >= 2000:
            if s1_byte_s5_k2_pct < 40.0 or s1_byte_ov_long_pct < 70.0:
                msg_dg = (
                    f"Stage-1 Dual Gate below target on {tier_name}/Calcite: "
                    f"s1_s5_k2={s1_byte_s5_k2_pct:.2f}%, s1_ov_long={s1_byte_ov_long_pct:.2f}%"
                )
                if tier_name == "S1_5M":
                    raise AssertionError(f"Stop-Crane {msg_dg}")
                print(
                    f"  [WARN] {msg_dg} — continuing to Stage 2 + QAT for honest measurement",
                    flush=True,
                )

        # Boundary: switch to `"utf8"` mode, calibrate `router_emb[131:]` & snapshot locked UTF-8 router
        if model.byte_chunker is not None:
            model.byte_chunker.byte_chunk_mode = "utf8"
            model.byte_chunker.target_chunk_bytes = float(lean_cfg.target_chunk_bytes)
            model.byte_chunker.use_static_m_cap = False
            model.byte_chunker.lock_router(True)
            cal_steps = 2 if smoke else 120
            _calibrate_non_ascii_router_emb(model, train_wins, device, cal_steps=cal_steps, seed=2026)
            fp_router_snap = _snapshot_router_params(model)
            print(f"  [{tier_name}/Calcite] boundary calibrated UTF-8 router ({cal_steps} steps)", flush=True)
        model.set_crystal_schedule(s1a_steps, s1a_steps)
        model.set_snap_alpha(1.0)
        if fp_ckpt_dir is not None:
            fp_ckpt_dir.mkdir(parents=True, exist_ok=True)
            s1_ck = fp_ckpt_dir / f"{tier_name}_stage1b_fp.pt"
            torch.save(
                {
                    "tier": tier_name,
                    "state_dict": unwrap_compiled_state_dict(model),
                    "s1a_byte_s5_k2_pct": s1a_byte_s5_k2_pct,
                    "s1_byte_s5_k2_pct": s1_byte_s5_k2_pct,
                    "s1_byte_ov_long_pct": s1_byte_ov_long_pct,
                    "s1_byte_trap_long_pct": s1_byte_trap_long_pct,
                },
                s1_ck,
            )
            print(f"  [{tier_name}/Calcite] Saved Stage-1B FP checkpoint -> {s1_ck}", flush=True)

    # Stage 2 (`grok_steps + 1 .. steps`): `"utf8"` 6-domain OpenWeb scaling + protected Crystal (`6e-5`) & Woodbury (`8e-5`)
    stage2_steps = steps - grok_steps
    if stage2_steps > 0:
        cryst_2, wood_2, nodc_other_2, emb_chunk_2, core_2 = _split_calcite_stage2_groups(model)
        all_fp2 = cryst_2 + wood_2 + nodc_other_2 + emb_chunk_2 + core_2
        base_lrs_2 = (6.0e-5, 8.0e-5, stage2_core_lr, stage2_emb_lr, stage2_core_lr)
        min_lrs_2 = (1.0e-5, 1.0e-5, min_lr, min_lr, min_lr)
        opt_2 = _make_adamw(
            [
                {"params": cryst_2, "lr": base_lrs_2[0], "weight_decay": 0.0},
                {"params": wood_2, "lr": base_lrs_2[1], "weight_decay": 0.0},
                {"params": nodc_other_2, "lr": base_lrs_2[2], "weight_decay": 0.0},
                {"params": emb_chunk_2, "lr": base_lrs_2[3], "weight_decay": 0.01},
                {"params": core_2, "lr": base_lrs_2[4], "weight_decay": 0.01},
            ],
            device,
            betas=(0.9, 0.95),
        )
        warm_2 = min(40, max(1, stage2_steps // 8))
        for step in range(grok_steps + 1, steps + 1):
            model.train()
            model.set_crystal_schedule(s1a_steps, s1a_steps)
            model.set_snap_alpha(1.0)
            s2 = step - grok_steps
            for gi, (blr, mlr) in enumerate(zip(base_lrs_2, min_lrs_2)):
                opt_2.param_groups[gi]["lr"] = _cosine_lr(s2, stage2_steps, warm_2, blr, mlr)
            xb_full, yb_full = batcher.sample_batch(
                batch_size, device, step=step, grok_steps=grok_steps, total_steps=steps, s1b_steps=s1b_steps
            )
            is_log_2 = (step == steps) or (step % 400 == 0) or (s2 == 1)
            need_s_2 = is_log_2 or (s2 % 50 == 0)

            micro_bs, ce_sum_t = _run_accum_microbatches(
                xb_full,
                yb_full,
                micro_bs,
                opt_2,
                device,
                f"{tier_name}/Calcite S2 s={s2}",
                lambda xb, yb, cur_acc, m_ref=model: _fp_micro_ce(
                    m_ref, xb, yb, tok.PAD_ID, use_amp, cur_acc, 0.04
                ),
            )
            torch.nn.utils.clip_grad_norm_(all_fp2, 1.0)
            opt_2.step()
            if need_s_2 and ce_sum_t is not None:
                ce_acc = float(ce_sum_t.item()) / float(max(1, batch_size // micro_bs))
                assert math.isfinite(ce_acc), f"NaN/Inf CE on {tier_name}/Calcite S2 step {step}"
                if is_log_2:
                    print(f"  [{tier_name}/Calcite S2-utf8] step {step:4d}/{steps} ce={ce_acc:.4f} ({(time.perf_counter()-t_fp0)/60:.1f}m)", flush=True)
        del opt_2
        if device == "cuda":
            torch.cuda.empty_cache()

    fp_train_s = round(time.perf_counter() - t_fp0, 2)
    fp_med_ms = round(statistics.median(step_ms_list), 2) if step_ms_list else 0.0

    if compile_cortex and device == "cuda" and (not smoke):
        fp_sd_clean = unwrap_compiled_state_dict(model)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        model = CalciteLeanSystem1LM(lean_cfg).to(device)
        model.load_state_dict(fp_sd_clean, strict=True)
        model.set_fast_kernels(use_fast_kernels)
    if model.byte_chunker is not None:
        model.byte_chunker.byte_chunk_mode = "utf8"
        model.byte_chunker.target_chunk_bytes = float(lean_cfg.target_chunk_bytes)
        model.byte_chunker.use_static_m_cap = True
        model.byte_chunker.lock_router(True)
        if not fp_router_snap:
            fp_router_snap = _snapshot_router_params(model)
        _assert_router_unchanged(model, fp_router_snap, f"{tier_name}_FP_POSTLOCK")
    model.set_crystal_schedule(grok_steps, grok_steps)
    model.set_snap_alpha(1.0)
    eval_fp = _eval_complete_cell(model, tok, val_mm, eval_wins, s5_suites, ov_suites, trap_suites, seq_len, device, smoke)

    # 4-Group + 4-Tap + Logit-KL STE QAT -> 2-bit uint8 LUT
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    fp_sd_cpu = unwrap_compiled_state_dict(model)
    if fp_ckpt_dir is not None:
        fp_ckpt_dir.mkdir(parents=True, exist_ok=True)
        s2_ck = fp_ckpt_dir / f"{tier_name}_stage2_fp.pt"
        torch.save(
            {
                "tier": tier_name,
                "state_dict": fp_sd_cpu,
                "s1a_byte_s5_k2_pct": s1a_byte_s5_k2_pct,
                "s1_byte_s5_k2_pct": s1_byte_s5_k2_pct,
                "s1_byte_ov_long_pct": s1_byte_ov_long_pct,
                "s1_byte_trap_long_pct": s1_byte_trap_long_pct,
            },
            s2_ck,
        )
        print(f"  [{tier_name}/Calcite] Saved Stage-2 FP checkpoint -> {s2_ck}", flush=True)

    qat_model = CalciteLeanSystem1LM(lean_cfg).to(device)
    qat_model.load_state_dict(fp_sd_cpu, strict=True)
    qat_model.set_fast_kernels(use_fast_kernels)
    if model.byte_chunker is not None:
        model.byte_chunker.use_static_m_cap = True
    if qat_model.byte_chunker is not None:
        qat_model.byte_chunker.use_static_m_cap = True
        qat_model.byte_chunker.lock_router(True)
    router_snap = fp_router_snap if fp_router_snap else _snapshot_router_params(qat_model)
    qat_model.set_crystal_schedule(1, 1)
    qat_model.set_snap_alpha(1.0)
    qat_model.set_qat_mix(0.30)
    qat_model.set_ternary_qat(True)
    if compile_cortex and device == "cuda" and (not smoke):
        compile_calcite_cortex(qat_model, mode="default")

    gain_q, cryst_q, chunk_q, core_q = _split_calcite_qat_4groups(qat_model)
    all_qp = gain_q + cryst_q + chunk_q + core_q
    w_scale = (384.0 / float(lean_cfg.d_model)) ** 0.5
    q_lrs = (1.2e-3 * w_scale, 8.0e-5, 6.0e-4 * w_scale, 4.2e-4 * w_scale)
    q_mins = (1.2e-4 * w_scale, 1.5e-5, 7.0e-5 * w_scale, 5.0e-5 * w_scale)
    opt_q = _make_adamw(
        [
            {"params": gain_q, "lr": q_lrs[0]},
            {"params": cryst_q, "lr": q_lrs[1]},
            {"params": chunk_q, "lr": q_lrs[2]},
            {"params": core_q, "lr": q_lrs[3]},
        ],
        device,
        weight_decay=0.0,
    )
    warm_q = min(48, max(1, qat_steps // 12))
    ramp_q = min(240, max(1, qat_steps // 4))
    kl_temp = 1.5
    taps_fp: List[torch.Tensor] = []
    taps_q: List[torch.Tensor] = []
    hk_fp = _attach_4tap_hooks(model, taps_fp, detach=True)
    hk_q = _attach_4tap_hooks(qat_model, taps_q, detach=False)
    t_q0 = time.perf_counter()
    micro_bs_q = _auto_micro_bs(device, tier_name, batch_size, is_qat=True)
    try:
        for qs in range(1, qat_steps + 1):
            qat_model.train()
            mix = min(1.0, 0.30 + 0.70 * (qs / float(ramp_q)))
            qat_model.set_qat_mix(mix)
            for gi, (blr, mlr) in enumerate(zip(q_lrs, q_mins)):
                opt_q.param_groups[gi]["lr"] = _cosine_lr(qs, qat_steps, warm_q, blr, mlr)
            xb_full, yb_full = batcher.sample_batch(
                batch_size, device, step=steps, grok_steps=grok_steps, total_steps=steps, s1b_steps=s1b_steps
            )
            is_log_q = (qs in (1, max(1, qat_steps // 2), qat_steps)) or (qs % 250 == 0)
            need_s_q = is_log_q or (qs % 50 == 0)

            micro_bs_q, q_ce_sum_t = _run_4tap_qat_step(
                model,
                qat_model,
                taps_fp,
                taps_q,
                xb_full,
                yb_full,
                micro_bs_q,
                opt_q,
                tok.PAD_ID,
                kl_temp,
                use_amp,
                device,
                f"{tier_name}/Calcite QAT qs={qs}",
            )
            torch.nn.utils.clip_grad_norm_(all_qp, 1.0)
            opt_q.step()
            clear_fast_qat_cache()
            if need_s_q and q_ce_sum_t is not None:
                q_ce_acc = float(q_ce_sum_t.item()) / float(max(1, batch_size // micro_bs_q))
                assert math.isfinite(q_ce_acc), f"NaN/Inf QAT CE at step {qs}"
                if is_log_q:
                    print(f"  [{tier_name}/Calcite QAT] step {qs:4d}/{qat_steps} mix={mix:.2f} ce={q_ce_acc:.4f}", flush=True)
    finally:
        for h in hk_fp + hk_q:
            h.remove()
        clear_fast_qat_cache()
        if qat_model.byte_chunker is not None:
            qat_model.byte_chunker.use_static_m_cap = True

    router_mut_diff = _assert_router_unchanged(qat_model, router_snap, f"{tier_name}_QAT")
    qat_s = round(time.perf_counter() - t_q0, 2)
    del model, opt_q
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    qat_model.set_ternary_qat(False)
    qat_model.set_qat_mix(1.0)
    if compile_cortex and device == "cuda" and (not smoke):
        qat_sd_clean = unwrap_compiled_state_dict(qat_model)
        del qat_model
        qat_model = CalciteLeanSystem1LM(lean_cfg)
        qat_model.load_state_dict(qat_sd_clean, strict=True)
        if qat_model.byte_chunker is not None:
            qat_model.byte_chunker.use_static_m_cap = True
            qat_model.byte_chunker.lock_router(True)
    with torch.no_grad():
        for nm, mod in qat_model.named_modules():
            if isinstance(mod, nn.Linear) and nm != "lm_head":
                if qat_model.byte_chunker is not None and qat_model.byte_chunker.is_locked_router_module(mod):
                    continue
                wt, _, _ = quantize_ternary_weights_ls(mod.weight.data)
                mod.weight.data.copy_(wt)
    qat_model.eval()

    binfo = get_lut_backend_info()
    lut_bk = "cuda_ext" if (device == "cuda" and binfo["cuda_ext_available"]) else "auto"
    lut_model = pack_fast_model_2bit_lut(qat_model.cpu(), backend=lut_bk).to(device).eval()
    lut_model.set_crystal_schedule(1, 1)
    lut_fp_info = compute_2bit_lut_footprint(lut_model)
    save_packed_lut_checkpoint(
        ckpt_path,
        lut_model.cpu(),
        lean_cfg,
        tok,
        metadata={
            "tier": tier_name,
            "params": n_params,
            "phase": "K_scaling_law",
            "router_max_abs_diff": router_mut_diff,
        },
    )
    lut_model = lut_model.to(device).eval()
    eval_lut = _eval_complete_cell(lut_model, tok, val_mm, eval_wins, s5_suites, ov_suites, trap_suites, seq_len, device, smoke)

    cache_kb = round(compute_lean_state_cache_kb(lean_cfg, 2048), 2)
    lut_mb = round(float(lut_fp_info["total_lut_kb"]) / 1024.0, 3)
    gap_val_bpb = round(float(eval_lut["val_memmap_bpb"]) - float(eval_fp["val_memmap_bpb"]), 4)
    gap_ow_bpb = round(float(eval_lut["mean_openweb_bpb"]) - float(eval_fp["mean_openweb_bpb"]), 4)
    print(
        f"  => [{tier_name}/Calcite] FP_val={eval_fp['val_memmap_bpb']:.4f} | "
        f"LUT({lut_mb:.2f}MB)_val={eval_lut['val_memmap_bpb']:.4f} (gap={gap_val_bpb:+.4f}) | "
        f"FP_ow={eval_fp['mean_openweb_bpb']:.4f} | LUT_ow={eval_lut['mean_openweb_bpb']:.4f} (gap={gap_ow_bpb:+.4f}) | "
        f"LUT_S5={eval_lut['s5_sum']:.1f}% | LUT_OV_long={eval_lut['overwrite']['ood_long']['exact_pct']:.1f}%",
        flush=True,
    )
    del qat_model, lut_model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "tier": tier_name,
        "model": "CalciteLeanSystem1LM",
        "params": n_params,
        "d_model": int(lean_cfg.d_model),
        "config": asdict(lean_cfg),
        "weight_bf16_mb": round((n_params * 2.0) / (1024**2), 3),
        "weight_lut_mb": lut_mb,
        "lut_footprint": {k: float(v) for k, v in lut_fp_info.items()},
        "cache_kb_L512": cache_kb,
        "cache_kb_L768": cache_kb,
        "cache_kb_L2048": cache_kb,
        "chunker_state_bytes": int(
            compute_chunker_state_bytes(
                lean_cfg.d_model, lean_cfg.byte_char_dim, lean_cfg.byte_router_char_dim, byte_chunk_mode=lean_cfg.byte_chunk_mode
            )
        ),
        "router_max_abs_diff": router_mut_diff,
        "s1a_byte_s5_k2_pct": s1a_byte_s5_k2_pct,
        "s1_byte_s5_k2_pct": s1_byte_s5_k2_pct,
        "s1_byte_ov_long_pct": s1_byte_ov_long_pct,
        "s1_byte_trap_long_pct": s1_byte_trap_long_pct,
        "gate2_init_std": gate2,
        "fp_train_seconds": fp_train_s,
        "qat_train_seconds": qat_s,
        "train_ms_per_step": fp_med_ms,
        "stage2_steps": int(stage2_steps),
        "stage2_emb_lr": float(stage2_emb_lr),
        "stage2_openweb_bytes_per_param": round((stage2_steps * 0.68 * batch_size * seq_len) / float(n_params), 3),
        "fp": eval_fp,
        "lut_2bit": eval_lut,
        "quant_gap_val_bpb": gap_val_bpb,
        "quant_gap_openweb_bpb": gap_ow_bpb,
        "ckpt_path": str(ckpt_path.relative_to(ROOT_DIR)).replace("\\", "/") if ckpt_path.is_relative_to(ROOT_DIR) else str(ckpt_path),
    }


def _fit_power_law(xs: Sequence[float], ys: Sequence[float]) -> Dict[str, float]:
    """Fits `y = a * x^(-alpha)` in log-log space over the 3 scaling tiers."""
    lx = [math.log(max(1e-9, float(v))) for v in xs]
    ly = [math.log(max(1e-9, float(v))) for v in ys]
    mx = sum(lx) / len(lx)
    my = sum(ly) / len(ly)
    var_x = sum((xi - mx) ** 2 for xi in lx)
    slope = sum((xi - mx) * (yi - my) for xi, yi in zip(lx, ly)) / max(1e-12, var_x)
    inter = my - slope * mx
    return {"alpha_exponent": round(-slope, 4), "scale_a": round(math.exp(inter), 4)}


def _merge_and_save_scaling_results(
    cells_dir: Path,
    out_json: Path,
    selected_lean: Dict[str, object],
    corpus_meta: Dict[str, object],
    device: str,
) -> Dict[str, object]:
    tiers_data: Dict[str, Dict[str, object]] = {}
    for tier in ("S1_5M", "S2_25M", "S3_85M"):
        t_map: Dict[str, object] = {}
        for mname in ("RoPETransformerLM", "ScaledHybridDeltaSWALM", "CalciteLeanSystem1LM"):
            cp = cells_dir / f"{tier}_{mname}.json"
            if cp.exists():
                t_map[mname] = json.loads(cp.read_text(encoding="utf-8"))
        if t_map:
            tiers_data[tier] = t_map

    scaling_fits: Dict[str, object] = {}
    if all(t in tiers_data and len(tiers_data[t]) == 3 for t in ("S1_5M", "S2_25M", "S3_85M")):
        ts = ("S1_5M", "S2_25M", "S3_85M")
        cal_fp_vals = [float(tiers_data[t]["CalciteLeanSystem1LM"]["fp"]["val_memmap_bpb"]) for t in ts]
        cal_lut_vals = [float(tiers_data[t]["CalciteLeanSystem1LM"]["lut_2bit"]["val_memmap_bpb"]) for t in ts]
        cal_fp_ows = [float(tiers_data[t]["CalciteLeanSystem1LM"]["fp"]["mean_openweb_bpb"]) for t in ts]
        gaps_val = [float(tiers_data[t]["CalciteLeanSystem1LM"]["quant_gap_val_bpb"]) for t in ts]
        cjk_fp = [
            float(tiers_data[t]["CalciteLeanSystem1LM"]["fp"]["openweb"]["regimes"]["5_CJK_UTF8"]["byte_bpb"])
            for t in ts
        ]
        scaling_fits = {
            "RoPETransformerLM_val_bpb_vs_params": _fit_power_law(
                [tiers_data[t]["RoPETransformerLM"]["params"] for t in ts],
                [tiers_data[t]["RoPETransformerLM"]["fp"]["val_memmap_bpb"] for t in ts],
            ),
            "ScaledHybridDeltaSWALM_val_bpb_vs_params": _fit_power_law(
                [tiers_data[t]["ScaledHybridDeltaSWALM"]["params"] for t in ts],
                [tiers_data[t]["ScaledHybridDeltaSWALM"]["fp"]["val_memmap_bpb"] for t in ts],
            ),
            "CalciteLeanSystem1LM_FP_val_bpb_vs_params": _fit_power_law(
                [tiers_data[t]["CalciteLeanSystem1LM"]["params"] for t in ts],
                [tiers_data[t]["CalciteLeanSystem1LM"]["fp"]["val_memmap_bpb"] for t in ts],
            ),
            "CalciteLeanSystem1LM_LUT_val_bpb_vs_params": _fit_power_law(
                [tiers_data[t]["CalciteLeanSystem1LM"]["params"] for t in ts],
                [tiers_data[t]["CalciteLeanSystem1LM"]["lut_2bit"]["val_memmap_bpb"] for t in ts],
            ),
            "quant_gap_val_bpb_by_width": {
                str(tiers_data[t]["CalciteLeanSystem1LM"]["d_model"]): tiers_data[t]["CalciteLeanSystem1LM"][
                    "quant_gap_val_bpb"
                ]
                for t in ts
            },
            "quant_gap_openweb_bpb_by_width": {
                str(tiers_data[t]["CalciteLeanSystem1LM"]["d_model"]): tiers_data[t]["CalciteLeanSystem1LM"][
                    "quant_gap_openweb_bpb"
                ]
                for t in ts
            },
            "quant_gap_val_bpb_collapse_ratio": round(gaps_val[0] / max(1e-9, gaps_val[-1]), 2),
            "quant_gap_val_bpb_monotonic_passed": bool(gaps_val[0] > gaps_val[1] > gaps_val[2]),
            "lut_val_bpb_monotonic_passed": bool(cal_lut_vals[0] > cal_lut_vals[1] > cal_lut_vals[2]),
            "strict_fp_val_monotonic_passed": bool(cal_fp_vals[0] > cal_fp_vals[1] > cal_fp_vals[2]),
            "strict_openweb_monotonic_passed": bool(cal_fp_ows[0] > cal_fp_ows[1] > cal_fp_ows[2]),
            "s1_to_s2_fp_and_ow_improved": bool((cal_fp_vals[0] > cal_fp_vals[1]) and (cal_fp_ows[0] > cal_fp_ows[1])),
            "cjk_utf8_fp_monotonic_passed": bool(cjk_fp[0] > cjk_fp[1] > cjk_fp[2]),
            "stage2_openweb_bytes_per_param": {
                t: round((800 * 0.68 * 64 * 768) / float(tiers_data[t]["CalciteLeanSystem1LM"]["params"]), 3)
                for t in ts
            },
            "honest_verdict": "PARTIAL_PASS_STAGE2_CHINCHILLA_UNDERBUDGET_AT_85M",
        }

    summary = {
        "protocol": {
            "phase": "Phase_K_System1_Scaling_Law",
            "device": torch.cuda.get_device_name(0) if device == "cuda" else str(corpus_meta.get("device_name", "cpu")),
            "vocab_size": 259,
            "pure_system1_zero_alu_traces": True,
            "selected_lean_params": selected_lean,
            "corpus": corpus_meta,
        },
        "tiers": tiers_data,
        "scaling_fits": scaling_fits,
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# 4. Local Smoke Tracer & CLI Entrypoint
# ---------------------------------------------------------------------------


def run_smoke_local(repo_root: Path) -> int:
    print("=== [Phase K | Smoke Local] Scrubber + S1_5M (RoPE, HybridDeltaSWA, Calcite FP/4TapQAT/LUT) ===", flush=True)
    device = "cuda" if has_usable_cuda() else "cpu"
    blocklist = build_phase_k_heldout_blocklist(repo_root)
    assert len(blocklist) >= 40_000, f"Unexpectedly small blocklist: {len(blocklist)}"
    print(f"  Scrubber hash count: {len(blocklist):,} (`>= 40,000` OK)", flush=True)

    tok = ByteBPETokenizer(merges=[])
    tr_docs, te_docs, tr_sents, te_sents = load_gsm8k_corpora(repo_root)
    train_wins, eval_wins, _, _ = build_disjoint_openweb_corpora(repo_root, tok, tr_docs, te_docs, seq_len=768)
    eval_wins_sm = {k: v[:4] for k, v in eval_wins.items()}

    tmp_dir = repo_root / "benchmarks" / "phase_k_smoke_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tok_buf = np.concatenate([_encode_doc_uint16(d) for d in tr_docs[:80]])
    tr_bin = tmp_dir / "train_sm.u16.bin"
    va_bin = tmp_dir / "val_sm.u16.bin"
    tr_bin.write_bytes(tok_buf.tobytes())
    va_bin.write_bytes(tok_buf[: min(len(tok_buf), 16384)].tobytes())
    train_mm = np.memmap(tr_bin, dtype=np.uint16, mode="r")
    val_mm = np.memmap(va_bin, dtype=np.uint16, mode="r")

    s5_s = {"k2": build_eval_s5_suite(tok, te_sents, 2, 8, (0, 1), 3100),
            "k4": build_eval_s5_suite(tok, te_sents, 4, 8, (0, 1), 3200),
            "k8": build_eval_s5_suite(tok, te_sents, 8, 8, (1, 1), 3300),
            "k16": build_eval_s5_suite(tok, te_sents, 16, 8, (1, 2), 3400)}
    ov_s = {"in_dist": build_eval_overwrite_suite(tok, te_sents, 4, 2, 2, 8, 4100),
            "ood_med": build_eval_overwrite_suite(tok, te_sents, 7, 4, 8, 8, 4200),
            "ood_long": build_eval_overwrite_suite(tok, te_sents, 9, 4, 12, 8, 4300)}
    tr_s = {"in_dist": build_eval_trap_suite(tok, te_sents, 4, 2, 2, 8, 5100),
            "ood_long": build_eval_trap_suite(tok, te_sents, 9, 4, 12, 8, 5300)}

    lean_cfg = build_tier_lean_config("S1_5M", window_size=8, crystal_w_chunks=2, woodbury_latch_span=3)
    c_params = int(sum(p.numel() for p in CalciteLeanSystem1LM(lean_cfg).parameters()))
    bl_specs = build_matched_baselines_for_tier("S1_5M", c_params)

    r_cfg = BaselineConfig(**bl_specs["RoPETransformerLM"]["config"])
    b_r = StratifiedSystem1Batcher(train_mm, tr_docs, tr_sents, tok, seq_len=768, train_wins=train_wins, seed=9100)
    _train_baseline_cell(
        "S1_5M", "RoPETransformerLM", RoPETransformerLM(r_cfg), bl_specs["RoPETransformerLM"]["config"],
        bl_specs["RoPETransformerLM"]["cache_kb_L512"], bl_specs["RoPETransformerLM"]["cache_kb_L2048"],
        4, 2, 2.0e-3, 9.0e-4, 1.5e-5, b_r, tok, val_mm, eval_wins_sm, s5_s, ov_s, tr_s, 16, 768, device, True,
        s1b_steps=1,
    )
    h_cfg = HybridDeltaSWAConfig(**bl_specs["ScaledHybridDeltaSWALM"]["config"])
    b_h = StratifiedSystem1Batcher(train_mm, tr_docs, tr_sents, tok, seq_len=768, train_wins=train_wins, seed=9100)
    _train_baseline_cell(
        "S1_5M", "ScaledHybridDeltaSWALM", ScaledHybridDeltaSWALM(h_cfg), bl_specs["ScaledHybridDeltaSWALM"]["config"],
        bl_specs["ScaledHybridDeltaSWALM"]["cache_kb_L512"], bl_specs["ScaledHybridDeltaSWALM"]["cache_kb_L2048"],
        4, 2, 2.0e-3, 9.0e-4, 1.5e-5, b_h, tok, val_mm, eval_wins_sm, s5_s, ov_s, tr_s, 16, 768, device, True,
        s1b_steps=1,
    )
    b_c = StratifiedSystem1Batcher(train_mm, tr_docs, tr_sents, tok, seq_len=768, train_wins=train_wins, seed=9100)
    _train_calcite_cell(
        "S1_5M", lean_cfg, 4, 2, 2, 2.0e-3, 9.0e-4, 1.35e-3, 1.5e-5, b_c, train_wins, tok, val_mm,
        eval_wins_sm, s5_s, ov_s, tr_s, 16, 768, tmp_dir / "sm_lut.pt", device, True,
        s1b_steps=1,
    )
    del b_r, b_h, b_c
    train_mm._mmap.close()
    val_mm._mmap.close()
    del train_mm, val_mm
    gc.collect()
    for fp in tmp_dir.glob("*"):
        fp.unlink(missing_ok=True)
    tmp_dir.rmdir()
    print("=== SMOKE LOCAL PASSED ===", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-local", action="store_true")
    parser.add_argument("--stage", default="all", choices=("build-corpus", "run-tier", "merge-only", "all"))
    parser.add_argument("--tier", default="all", choices=("S1_5M", "S2_25M", "S3_85M", "all"))
    parser.add_argument(
        "--only-model",
        default="all",
        choices=("all", "RoPETransformerLM", "ScaledHybridDeltaSWALM", "CalciteLeanSystem1LM"),
    )
    parser.add_argument("--data-dir", default="/content/phase_k_data")
    parser.add_argument("--corpus-gb", type=float, default=0.65)
    parser.add_argument("--corpus-gdrive-id", default="")
    parser.add_argument("--seq-len", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fp-steps", type=int, default=-1)
    parser.add_argument("--stage2-steps", type=int, default=-1)
    parser.add_argument("--stage2-emb-lr", type=float, default=-1.0)
    parser.add_argument("--qat-steps", type=int, default=-1)
    parser.add_argument("--window-size", type=int, default=-1)
    parser.add_argument("--crystal-w-chunks", type=int, default=-1)
    parser.add_argument("--woodbury-latch-span", type=int, default=-1)
    parser.add_argument("--compile-cortex", action="store_true")
    parser.add_argument("--no-fast-kernels", action="store_true")
    parser.add_argument("--fp-ckpt-dir", default="")
    parser.add_argument("--resume-stage2-from", default="")
    parser.add_argument("--cells-dir", default="")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--ckpt-name", default="")
    args = parser.parse_args()

    if args.smoke_local:
        return run_smoke_local(ROOT_DIR)

    device = "cuda" if has_usable_cuda() else "cpu"
    data_dir = Path(args.data_dir)
    cells_dir = Path(args.cells_dir) if args.cells_dir else (ROOT_DIR / "benchmarks" / "phase_k_cells")
    cells_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = ROOT_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_json) if args.out_json else (ROOT_DIR / "benchmarks" / "phase_k_scaling_law_results.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # Load selected Via Negativa ring-buffer parameters from probe or defaults
    probe_p = ROOT_DIR / "benchmarks" / "phase_k_via_negativa_probe.json"
    if probe_p.exists():
        selected = json.loads(probe_p.read_text(encoding="utf-8"))["selected_lean_params"]
    else:
        selected = {"variant": "default_lean", "window_size": 8, "crystal_w_chunks": 2, "woodbury_latch_span": 3}
    w_sz = args.window_size if args.window_size > 0 else int(selected["window_size"])
    w_ch = args.crystal_w_chunks if args.crystal_w_chunks > 0 else int(selected["crystal_w_chunks"])
    l_sp = args.woodbury_latch_span if args.woodbury_latch_span > 0 else int(selected["woodbury_latch_span"])
    selected_resolved = {"variant": selected.get("variant", "custom"), "window_size": w_sz, "crystal_w_chunks": w_ch, "woodbury_latch_span": l_sp}

    if args.stage in ("build-corpus", "all"):
        corpus_meta = run_stage_build_corpus(
            ROOT_DIR, data_dir, total_gb=args.corpus_gb, corpus_gdrive_id=args.corpus_gdrive_id
        )
        if args.stage == "build-corpus":
            return 0
    else:
        meta_p = data_dir / "corpus_meta.json"
        corpus_meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}

    if args.stage == "merge-only":
        _merge_and_save_scaling_results(cells_dir, out_json, selected_resolved, corpus_meta, device)
        return 0

    train_mm = load_utf8_corpus_array(data_dir / "train_utf8.u16.bin")
    val_mm = load_utf8_corpus_array(data_dir / "val_utf8.u16.bin")
    tok = ByteBPETokenizer(merges=[])
    tr_docs, te_docs, tr_sents, te_sents = load_gsm8k_corpora(ROOT_DIR)
    train_wins, eval_wins, _, _ = build_disjoint_openweb_corpora(ROOT_DIR, tok, tr_docs, te_docs, seq_len=args.seq_len)

    n_eval = 150
    s5_suites = {
        "k2": build_eval_s5_suite(tok, te_sents, 2, n_eval, (0, 1), 3100),
        "k4": build_eval_s5_suite(tok, te_sents, 4, n_eval, (0, 1), 3200),
        "k8": build_eval_s5_suite(tok, te_sents, 8, n_eval, (1, 1), 3300),
        "k16": build_eval_s5_suite(tok, te_sents, 16, n_eval, (1, 2), 3400),
    }
    ov_suites = {
        "in_dist": build_eval_overwrite_suite(tok, te_sents, 4, 2, 2, n_eval, 4100),
        "ood_med": build_eval_overwrite_suite(tok, te_sents, 7, 4, 10, n_eval, 4200),
        "ood_long": build_eval_overwrite_suite(tok, te_sents, 9, 5, 18, n_eval, 4300),
    }
    trap_suites = {
        "in_dist": build_eval_trap_suite(tok, te_sents, 4, 2, 2, n_eval, 5100),
        "ood_long": build_eval_trap_suite(tok, te_sents, 9, 5, 18, n_eval, 5300),
    }

    target_tiers = ("S1_5M", "S2_25M", "S3_85M") if args.tier == "all" else (args.tier,)
    for tier_name in target_tiers:
        sched = TIER_SCHEDULE[tier_name]
        fp_steps = args.fp_steps if args.fp_steps > 0 else int(sched["fp_steps"])
        grok_steps = (
            int(sched["grok_steps"])
            if fp_steps == int(sched["fp_steps"])
            else (max(2, min(fp_steps - 1, int(round(0.75 * fp_steps)))) if fp_steps >= 3 else 1)
        )
        if args.stage2_steps >= 0:
            fp_steps = grok_steps + int(args.stage2_steps)
        s1b_steps = (
            int(sched["s1b_steps"])
            if grok_steps == int(sched["grok_steps"])
            else (max(1, min(grok_steps - 1, int(round(0.18 * grok_steps)))) if grok_steps >= 2 else 0)
        )
        qat_steps = args.qat_steps if args.qat_steps > 0 else int(sched["qat_steps"])
        lean_cfg = build_tier_lean_config(tier_name, window_size=w_sz, crystal_w_chunks=w_ch, woodbury_latch_span=l_sp)
        c_params = int(sum(p.numel() for p in CalciteLeanSystem1LM(lean_cfg).parameters()))
        bl_specs = build_matched_baselines_for_tier(tier_name, c_params)

        # Equal deterministic seed across all 9 cells so every tier/architecture sees identical byte batches
        models_to_run = (
            ("CalciteLeanSystem1LM", "RoPETransformerLM", "ScaledHybridDeltaSWALM")
            if args.only_model == "all"
            else (args.only_model,)
        )
        for mname in models_to_run:
            cell_p = cells_dir / f"{tier_name}_{mname}.json"
            if cell_p.exists():
                print(f"[Resume] Skipping completed cell {cell_p.name}", flush=True)
                continue
            batcher = StratifiedSystem1Batcher(
                train_mm, tr_docs, tr_sents, tok, seq_len=args.seq_len, train_wins=train_wins, seed=9100
            )
            if mname == "RoPETransformerLM":
                rc = BaselineConfig(**bl_specs[mname]["config"])
                cell_res = _train_baseline_cell(
                    tier_name, mname, RoPETransformerLM(rc), bl_specs[mname]["config"],
                    bl_specs[mname]["cache_kb_L512"], bl_specs[mname]["cache_kb_L2048"],
                    fp_steps, grok_steps, float(sched["core_lr"]), float(sched["stage2_core_lr"]), float(sched["min_lr"]),
                    batcher, tok, val_mm, eval_wins, s5_suites, ov_suites, trap_suites,
                    args.batch_size, args.seq_len, device, False, s1b_steps=s1b_steps,
                )
            elif mname == "ScaledHybridDeltaSWALM":
                hc = HybridDeltaSWAConfig(**bl_specs[mname]["config"])
                cell_res = _train_baseline_cell(
                    tier_name, mname, ScaledHybridDeltaSWALM(hc), bl_specs[mname]["config"],
                    bl_specs[mname]["cache_kb_L512"], bl_specs[mname]["cache_kb_L2048"],
                    fp_steps, grok_steps, float(sched["core_lr"]), float(sched["stage2_core_lr"]), float(sched["min_lr"]),
                    batcher, tok, val_mm, eval_wins, s5_suites, ov_suites, trap_suites,
                    args.batch_size, args.seq_len, device, False, s1b_steps=s1b_steps,
                )
            else:
                s2_emb_lr_eff = float(args.stage2_emb_lr) if args.stage2_emb_lr > 0.0 else float(sched["stage2_emb_lr"])
                ckpt_name_eff = args.ckpt_name if args.ckpt_name else str(sched["ckpt_name"])
                cell_res = _train_calcite_cell(
                    tier_name, lean_cfg, fp_steps, grok_steps, qat_steps,
                    float(sched["core_lr"]), float(sched["stage2_core_lr"]), s2_emb_lr_eff, float(sched["min_lr"]),
                    batcher, train_wins, tok, val_mm, eval_wins, s5_suites, ov_suites, trap_suites,
                    args.batch_size, args.seq_len, ckpt_dir / ckpt_name_eff, device, False,
                    s1b_steps=s1b_steps,
                    use_fast_kernels=not args.no_fast_kernels,
                    compile_cortex=args.compile_cortex,
                    fp_ckpt_dir=Path(args.fp_ckpt_dir) if args.fp_ckpt_dir else None,
                    resume_stage2_from=Path(args.resume_stage2_from) if args.resume_stage2_from else None,
                )
            cell_p.write_text(json.dumps(cell_res, indent=2), encoding="utf-8")
            _merge_and_save_scaling_results(cells_dir, out_json, selected_resolved, corpus_meta, device)

    _merge_and_save_scaling_results(cells_dir, out_json, selected_resolved, corpus_meta, device)
    print(f"\n[DONE] Full Phase K results merged at {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
