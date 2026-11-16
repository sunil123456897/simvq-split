"""simvq_cram_adapter.py — Чистый адаптер `SimVQSplitLM v5 -> Sparsemax -> CRAM+z` (Фаза C).

Инварианты:
  - Нулевой `re`, `ast`, `inspect`, нулевой `eval`/`exec` и нулевые зашитые словари.
  - Точная побайтовая синхронизация числовых слотов `ExtractedSlot(b0, b1)` с `ByteBPETokenizer`.
  - Поддержка режимов `fp` (`simvq_split_state_dict`), `qat` (`simvq_split_ternary_qat_state_dict`)
    и `lut` (`PackedLUT2BitLinear` `uint8`, `16.00x` сжатие матриц) со `strict=True`.
  - Валидация `--verify-alignment` работает автономно без импорта `torch` и без открытия `test.jsonl`.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_EXP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXP_DIR.parent.parent
for _cand_p in (str(_REPO_ROOT), str(_EXP_DIR)):
    if _cand_p not in sys.path:
        sys.path.insert(0, _cand_p)

try:
    from .slot_extractor import (
        ExtractedSlot,
        TrainInducedQuantityExtractor,
    )
except ImportError:
    from slot_extractor import (
        ExtractedSlot,
        TrainInducedQuantityExtractor,
    )


def _load_byte_bpe_tokenizer_class():
    """Загружает `ByteBPETokenizer` напрямую из `simvq_split/tokenizer.py` без импорта `torch`."""
    tok_file = _REPO_ROOT / "simvq_split" / "tokenizer.py"
    if tok_file.is_file():
        spec = importlib.util.spec_from_file_location("_simvq_tokenizer_standalone", str(tok_file))
        if spec is not None and spec.loader is not None:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.ByteBPETokenizer
    from simvq_split.tokenizer import ByteBPETokenizer
    return ByteBPETokenizer


ByteBPETokenizer = _load_byte_bpe_tokenizer_class()

ADAPTER_KNOBS: Dict[str, float | int] = {
    "d_model": 160,
    "extract_batch_size": 96,
    "lut_batch_size": 32,
    "lut_seq_batch": 60,
    "post_num_token_window": 5,
    "sys1_epochs": 18,
    "sys1_batch_size": 48,
    "sys1_lr": 1.0e-3,
    "qat_steps": 240,
    "cram_epochs": 10,
    "cram_batch_size": 128,
    "cram_lr": 0.018,
    "seed": 2026,
}

_PROTECTED_LEGACY_SLICE = (_REPO_ROOT / "benchmarks" / "phase_c_v5_calibration_slice.json").resolve()


def _locate_train_jsonl() -> Path:
    """Находит локальный `benchmarks/gsm8k/train.jsonl` без открытия слепого теста."""
    candidates = [
        Path("benchmarks/gsm8k/train.jsonl"),
        _REPO_ROOT / "benchmarks" / "gsm8k" / "train.jsonl",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError("Не найден обучающий файл `benchmarks/gsm8k/train.jsonl`.")


def sparsemax_numpy(z: np.ndarray) -> np.ndarray:
    """Точная проекция на вероятностный симплекс (Sparsemax) в NumPy по последней оси."""
    z_arr = np.asarray(z, dtype=np.float64)
    z_shift = z_arr - np.max(z_arr, axis=-1, keepdims=True)
    zs = np.sort(z_shift, axis=-1)[..., ::-1]
    d = z_arr.shape[-1]
    k_vec = np.arange(1, d + 1, dtype=np.float64)
    cs = np.cumsum(zs, axis=-1)
    valid = (1.0 + k_vec * zs) > cs
    k_max = np.sum(valid, axis=-1, keepdims=True)
    idx = np.maximum(k_max - 1, 0)
    cs_k = np.take_along_axis(cs, idx, axis=-1)
    tau = (cs_k - 1.0) / np.maximum(k_max.astype(np.float64), 1.0)
    return np.maximum(z_shift - tau, 0.0)


def align_slots_to_bpe_tokens(
    q_text: str,
    slots: Sequence[ExtractedSlot],
    tok,
    post_window: int = 5,
) -> Tuple[List[int], List[Tuple[int, int]], List[Tuple[int, int]], int]:
    """Синхронизирует UTF-8 байтовые интервалы `ExtractedSlot(b0, b1)` с токенами `ByteBPETokenizer`."""
    token_ids: List[int] = tok.encode(q_text, add_bos=True, add_eos=True)
    n_toks = len(token_ids)
    q_tok_idx = max(1, n_toks - 2)

    tok_bytes: List[Tuple[int, int]] = []
    b_cur = 0
    for tid in token_ids:
        raw_b = tok.id_to_bytes.get(int(tid), b"")
        tok_bytes.append((b_cur, b_cur + len(raw_b)))
        b_cur += len(raw_b)

    num_spans: List[Tuple[int, int]] = []
    post_spans: List[Tuple[int, int]] = []
    for sl in slots:
        if sl.b1 > sl.b0:
            ov = [
                ti
                for ti, (tb0, tb1) in enumerate(tok_bytes)
                if 0 < ti < n_toks - 1 and tb1 > sl.b0 and tb0 < sl.b1
            ]
        else:
            ov = []

        if ov:
            t_s, t_e = ov[0], ov[-1] + 1
        else:
            t_s, t_e = q_tok_idx, min(n_toks - 1, q_tok_idx + 1)

        p_s = min(t_e, q_tok_idx)
        p_e = min(t_e + post_window, max(p_s + 1, n_toks - 1))
        if p_e <= p_s:
            p_s, p_e = t_s, t_e
        num_spans.append((t_s, t_e))
        post_spans.append((p_s, p_e))

    return token_ids, num_spans, post_spans, q_tok_idx


def align_slots_to_utf8_bytes(
    question_text: str,
    slots: Sequence[ExtractedSlot],
    bos_offset: int = 1,
) -> List[Tuple[int, int]]:
    """Напрямую сопоставляет `ExtractedSlot(b0, b1)` позициям сырого UTF-8 потока (`vocab=259`, `+<bos>`)."""
    raw_len = len(question_text.encode("utf-8"))
    q_idx = bos_offset + max(0, raw_len - 1)
    spans: List[Tuple[int, int]] = []
    for sl in slots:
        if 0 <= sl.b0 < sl.b1 <= raw_len:
            spans.append((bos_offset + int(sl.b0), bos_offset + int(sl.b1)))
        else:
            spans.append((q_idx, q_idx + 1))
    return spans


def extract_tournament_checkpoint_features(
    ckpt_path: Path,
    items: Sequence[Dict[str, str]],
    extractor: TrainInducedQuantityExtractor,
    slots_per_q: Optional[Sequence[Sequence[ExtractedSlot]]] = None,
    weight_mode: str = "fp",
    device: str = "cpu",
    batch_size: int = 96,
    post_window: int = 5,
    beam_width: int = 0,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    """Извлекает скрытые векторы `SimVQSplitLM` и `u_prob` из турнирного чекпоинта (`fp`/`qat`/`lut`)."""
    import torch

    try:
        from .system1_encoder import Calcite5MathSystem1
    except ImportError:
        from system1_encoder import Calcite5MathSystem1
    from simvq_split.models import SimVQSplitConfig, SimVQSplitLM
    from simvq_split.quant_lut import compute_2bit_lut_footprint, pack_model_2bit_lut

    mode = weight_mode.lower().strip()
    if mode not in ("fp", "qat", "lut"):
        raise ValueError(f"Unsupported weight_mode={weight_mode!r}; expected 'fp', 'qat', or 'lut'.")

    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(ckpt_path, map_location="cpu")

    sys1 = Calcite5MathSystem1(ckpt_path, device=device, seed=int(ADAPTER_KNOBS["seed"]))
    lut_stats: Dict[str, float] = {}

    if mode in ("qat", "lut"):
        sd_key = "simvq_split_ternary_qat_state_dict"
        cfg = SimVQSplitConfig(**payload["simvq_split_config"])
        qat_cpu = SimVQSplitLM(cfg)
        qat_cpu.load_state_dict(payload[sd_key], strict=True)
        qat_cpu.eval()
        if mode == "qat":
            sys1.model = qat_cpu.to(device).eval()
            sys1.model.set_snap_alpha(1.0)
        else:
            lut_m = pack_model_2bit_lut(qat_cpu)
            lut_stats = {k: float(v) for k, v in compute_2bit_lut_footprint(lut_m).items()}
            sys1.lut_model = lut_m.to(device).eval()
            sys1.lut_model.set_snap_alpha(1.0)

    questions = [str(it["question"]) for it in items]
    if slots_per_q is None:
        eff_slots = [extractor.extract_slots(q, include_aux_consts=True)[0] for q in questions]
    else:
        eff_slots = [list(sl) for sl in slots_per_q]

    feats = sys1.extract_features_and_beams(
        questions=questions,
        slots_per_q=eff_slots,
        extractor=extractor,
        batch_size=int(batch_size),
        post_window=int(post_window),
        beam_width=int(beam_width),
        use_lut=(mode == "lut"),
    )
    return feats, lut_stats


def verify_alignment_and_sparsemax_without_torch() -> None:
    """Автономно проверяет без `torch` и без чтения `test.jsonl`:
      1) байтовую привязку `ByteBPETokenizer` к слотам на всех `7,473` задачах `train.jsonl` (> 99.9%),
      2) точность аналитического якобиана `sparsemax_numpy` (< 1e-7).
    """
    train_path = _locate_train_jsonl()
    with open(train_path, "r", encoding="utf-8") as f:
        all_tr = [json.loads(line) for line in f if line.strip()]

    ext = TrainInducedQuantityExtractor()
    ext.fit(all_tr[:6000])
    tok = ByteBPETokenizer.train_from_iterator([str(it["question"]) for it in all_tr[:1500]], vocab_size=1536)
    utf8_tok = ByteBPETokenizer(merges=[])
    assert int(utf8_tok.vocab_size) == 259, f"Expected raw UTF-8 vocab_size=259, got {utf8_tok.vocab_size}"

    checked_slots = 0
    explicit_total = 0
    explicit_digit_matches = 0
    utf8_digit_matches = 0

    for it in all_tr:
        q_str = str(it["question"])
        slots, _, _ = ext.extract_slots(q_str, include_aux_consts=True)
        t_ids, n_sp, p_sp, q_idx = align_slots_to_bpe_tokens(q_str, slots, tok, post_window=5)
        u_ids, u_sp, u_post, u_q_idx = align_slots_to_bpe_tokens(q_str, slots, utf8_tok, post_window=16)
        u_direct = align_slots_to_utf8_bytes(q_str, slots, bos_offset=1)
        assert len(n_sp) == len(slots) == len(p_sp)
        assert len(u_sp) == len(slots) == len(u_post) == len(u_direct)
        assert 0 < q_idx < len(t_ids) and 0 < u_q_idx < len(u_ids)
        for sl, (s0, s1), (p0, p1), (us0, us1), (ud0, ud1) in zip(slots, n_sp, p_sp, u_sp, u_direct):
            assert 0 <= s0 < s1 <= len(t_ids)
            assert 0 <= p0 < p1 <= len(t_ids)
            assert 0 <= us0 < us1 <= len(u_ids)
            if sl.b1 > sl.b0:
                assert (us0, us1) == (ud0, ud1), f"UTF-8 span mismatch: {(us0, us1)} != {(ud0, ud1)}"
            checked_slots += 1
            if sl.kind == "EXPLICIT":
                explicit_total += 1
                decoded_piece = tok.decode(t_ids[s0:s1])
                if any("0" <= ch <= "9" for ch in decoded_piece):
                    explicit_digit_matches += 1
                utf8_piece = utf8_tok.decode(u_ids[us0:us1])
                if any("0" <= ch <= "9" for ch in utf8_piece):
                    utf8_digit_matches += 1

    match_rate = 100.0 * explicit_digit_matches / max(1, explicit_total)
    utf8_rate = 100.0 * utf8_digit_matches / max(1, explicit_total)
    assert match_rate > 99.9, f"Потеря точности BPE-привязки: {match_rate:.2f}%"
    assert utf8_rate == 100.0, f"Потеря точности UTF-8 привязки: {utf8_rate:.2f}%"

    rng = np.random.default_rng(2026)
    z = rng.standard_normal((8, 32))
    grad_out = rng.standard_normal((8, 32))
    p = sparsemax_numpy(z)
    supp = (p > 0.0).astype(np.float64)
    v_hat = np.sum(grad_out * supp, axis=-1, keepdims=True) / np.maximum(np.sum(supp, axis=-1, keepdims=True), 1.0)
    grad_analytic = supp * (grad_out - v_hat)

    eps = 1e-6
    grad_num = np.zeros_like(z)
    for r in range(z.shape[0]):
        for c in range(z.shape[1]):
            zp = z.copy()
            zp[r, c] += eps
            zm = z.copy()
            zm[r, c] -= eps
            grad_num[r, c] = np.sum((sparsemax_numpy(zp) - sparsemax_numpy(zm)) * grad_out) / (2.0 * eps)

    max_grad_err = float(np.max(np.abs(grad_analytic - grad_num)))
    assert max_grad_err < 1e-7, f"Ошибка якобиана Sparsemax: {max_grad_err}"

    print(
        f"[VERIFY OK] Проверено {len(all_tr)} задач train.jsonl ({checked_slots} слотов, test.jsonl закрыт)!\n"
        f"            Точность BPE-привязки явных чисел: {explicit_digit_matches}/{explicit_total} ({match_rate:.2f}%)\n"
        f"            Точность UTF-8 привязки явных чисел: {utf8_digit_matches}/{explicit_total} ({utf8_rate:.2f}%)\n"
        f"            Ошибка аналитического якобиана Sparsemax: {max_grad_err:.2e}"
    )


def run_pytorch_adapter_pipeline(
    ckpt_arg: str = "checkpoints/simvq_split_tournament.pt",
    force_reextract: bool = False,
    weight_mode: str = "fp",
    save_json: Optional[str] = None,
) -> Dict[str, object]:
    """Прогоняет конвейер Фазы C на валидационном срезе `train[:6000] / cal[6000:6736] / dev[6736:]`."""
    import torch

    try:
        from .cram_open_graph import OpenGraphCRAMSolver
        from .gsm8k_cram_solver import evaluate_gsm8k_split
        from .spherical_alu import SphericalBase10RegisterALU
        from .system1_encoder import Calcite5MathSystem1
    except ImportError:
        from cram_open_graph import OpenGraphCRAMSolver
        from gsm8k_cram_solver import evaluate_gsm8k_split
        from spherical_alu import SphericalBase10RegisterALU
        from system1_encoder import Calcite5MathSystem1

    if save_json:
        out_check = (Path(save_json) if Path(save_json).is_absolute() else (_REPO_ROOT / save_json)).resolve()
        if out_check == _PROTECTED_LEGACY_SLICE:
            raise ValueError(f"Запрещено перезаписывать эталонный файл {out_check}")

    mode = weight_mode.lower().strip()
    if mode not in ("fp", "qat", "lut"):
        raise ValueError(f"Unsupported weight_mode={weight_mode!r}; expected 'fp', 'qat', or 'lut'.")

    t0 = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_path = _locate_train_jsonl()
    with open(train_path, "r", encoding="utf-8") as f:
        train_all = [json.loads(line) for line in f if line.strip()]

    tr_items = train_all[:6000]
    cal_items = train_all[6000:6736]
    dev_items = train_all[6736:]

    ext = TrainInducedQuantityExtractor()
    ext.fit(tr_items)
    tr_slots = [ext.extract_slots(str(it["question"]), include_aux_consts=True)[0] for it in tr_items]
    cal_slots = [ext.extract_slots(str(it["question"]), include_aux_consts=True)[0] for it in cal_items]
    dev_slots = [ext.extract_slots(str(it["question"]), include_aux_consts=True)[0] for it in dev_items]

    ckpt_path = Path(ckpt_arg) if Path(ckpt_arg).is_absolute() else (_REPO_ROOT / ckpt_arg)
    sys1 = Calcite5MathSystem1(ckpt_path, device=device, seed=int(ADAPTER_KNOBS["seed"]))
    sys1_ckpt = _EXP_DIR / "checkpoints" / "phase_c_sys1_dev.pt"
    if sys1_ckpt.is_file() and not force_reextract:
        sys1.load_adapted(sys1_ckpt)
    else:
        sys1.fit_on_train_split(
            tr_items,
            ext,
            epochs=int(ADAPTER_KNOBS["sys1_epochs"]),
            batch_size=int(ADAPTER_KNOBS["sys1_batch_size"]),
            lr=float(ADAPTER_KNOBS["sys1_lr"]),
            qat_steps=int(ADAPTER_KNOBS["qat_steps"]),
            save_ckpt_path=sys1_ckpt,
        )

    tr_feats = sys1.extract_features_and_beams([str(it["question"]) for it in tr_items], tr_slots, ext, beam_width=0, use_lut=False)
    if mode == "qat":
        try:
            sys1_payload = torch.load(sys1_ckpt, map_location="cpu", weights_only=False)
        except TypeError:
            sys1_payload = torch.load(sys1_ckpt, map_location="cpu")
        sys1.model.load_state_dict(sys1_payload["qat_state_dict"], strict=True)
        sys1.model.to(device).eval()
        sys1.model.set_snap_alpha(1.0)

    use_lut = mode == "lut"
    cal_feats = sys1.extract_features_and_beams([str(it["question"]) for it in cal_items], cal_slots, ext, use_lut=use_lut)
    dev_feats = sys1.extract_features_and_beams([str(it["question"]) for it in dev_items], dev_slots, ext, use_lut=use_lut)

    alu = SphericalBase10RegisterALU(d_sub=15, num_digits=12, seed=3301)
    solver = OpenGraphCRAMSolver(ext, alu, d_model=int(ADAPTER_KNOBS["d_model"]), device=device, seed=2026)
    fit_stats = solver.fit_bptt_free(
        tr_items,
        tr_slots,
        tr_feats,
        epochs=int(ADAPTER_KNOBS["cram_epochs"]),
        batch_size=int(ADAPTER_KNOBS["cram_batch_size"]),
        lr=float(ADAPTER_KNOBS["cram_lr"]),
    )
    cal_stats = solver.calibrate_on_split(cal_items, cal_slots, cal_feats)
    dev_res = evaluate_gsm8k_split(solver, dev_items, dev_slots, dev_feats, tag=f"BlindDev_{mode.upper()}")

    summary: Dict[str, object] = {
        "mode": mode,
        "ckpt": str(ckpt_path),
        "fit": fit_stats,
        "cal": cal_stats,
        "dev": dev_res,
        "elapsed_sec": time.perf_counter() - t0,
    }
    if save_json:
        out_p = Path(save_json) if Path(save_json).is_absolute() else (_REPO_ROOT / save_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        existing: Dict[str, object] = {}
        if out_p.is_file():
            with open(out_p, "r", encoding="utf-8") as jf:
                existing = json.load(jf)
        existing[mode] = summary
        with open(out_p, "w", encoding="utf-8") as jf:
            json.dump(existing, jf, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Адаптер Calcite-5 SimVQSplitLM v5 -> CRAM+z (Фаза C).")
    parser.add_argument("--ckpt", type=str, default="checkpoints/simvq_split_tournament.pt")
    parser.add_argument("--reextract", action="store_true")
    parser.add_argument("--qat", action="store_true")
    parser.add_argument("--lut", action="store_true")
    parser.add_argument("--save-json", type=str, default="")
    parser.add_argument("--verify-alignment", action="store_true")
    args = parser.parse_args()

    if args.verify_alignment:
        verify_alignment_and_sparsemax_without_torch()
        return

    weight_mode = "lut" if args.lut else ("qat" if args.qat else "fp")
    summary = run_pytorch_adapter_pipeline(
        ckpt_arg=args.ckpt,
        force_reextract=args.reextract,
        weight_mode=weight_mode,
        save_json=args.save_json or None,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


__all__ = [
    "ADAPTER_KNOBS",
    "ByteBPETokenizer",
    "align_slots_to_bpe_tokens",
    "align_slots_to_utf8_bytes",
    "extract_tournament_checkpoint_features",
    "run_pytorch_adapter_pipeline",
    "sparsemax_numpy",
    "verify_alignment_and_sparsemax_without_torch",
]


if __name__ == "__main__":
    main()
