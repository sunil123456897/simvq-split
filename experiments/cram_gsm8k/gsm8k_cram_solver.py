"""gsm8k_cram_solver.py — Чистый фасад Остерхаута для открытого решателя CRAM+z.

Объединяет изолированные модули Фазы C в единый интерфейс:
  - `spherical_alu.py`: 12-разрядный сферический регистровый АЛУ `(S^14)^12 x S^14_sign`;
  - `slot_extractor.py`: посимвольный сканер величин без `re` и data-driven экстрактор;
  - `cram_open_graph.py`: энергетическая модель `CRAMzOperatorGraphEBM` и решатель
    `OpenGraphCRAMSolver` (`0` шагов BPTT, поле Гершгорина, двойной сертификат `DualCert`).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

_EXP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXP_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from .cram_open_graph import (
        CRAMzOperatorGraphEBM,
        DC_HASH,
        DP_HASH,
        DR_HASH,
        DU_HASH,
        D_STEP_PSI,
        D_Z_FIELD,
        N_FIELD_GROUPS,
        OpenGraphCRAMSolver,
    )
    from .slot_extractor import (
        ExtractedSlot,
        TrainInducedQuantityExtractor,
        extract_answer_tags,
        extract_gt_number,
    )
    from .spherical_alu import (
        SphericalBase10FixedPointALU,
        SphericalBase10RegisterALU,
    )
except ImportError:
    from cram_open_graph import (
        CRAMzOperatorGraphEBM,
        DC_HASH,
        DP_HASH,
        DR_HASH,
        DU_HASH,
        D_STEP_PSI,
        D_Z_FIELD,
        N_FIELD_GROUPS,
        OpenGraphCRAMSolver,
    )
    from slot_extractor import (
        ExtractedSlot,
        TrainInducedQuantityExtractor,
        extract_answer_tags,
        extract_gt_number,
    )
    from spherical_alu import (
        SphericalBase10FixedPointALU,
        SphericalBase10RegisterALU,
    )


TUNING_KNOBS: Dict[str, float | int] = {
    "d_model": 160,
    "d_ctx": DC_HASH,
    "d_unary": DU_HASH,
    "d_pair": DP_HASH,
    "d_root": DR_HASH,
    "d_step_psi": D_STEP_PSI,
    "d_z_field": D_Z_FIELD,
    "n_field_groups": N_FIELD_GROUPS,
    "m_convex": 5.0,
    "w_zz_l1_cap": 0.90,
    "gershgorin_min_margin": 4.0,
    "step_field_scale": 0.03,
    "woodbury_ridge_lam": 18.0,
    "cram_epochs": 10,
    "cram_batch_size": 128,
    "cram_lr": 0.018,
    "boltzmann_beta": 1.0,
    "min_cert_precision": 0.38,
    "min_cert_coverage": 0.085,
    "seed": 2026,
}


def wilson_95_ci(k_ok: int, n_tot: int) -> Tuple[float, float]:
    """Возвращает 95% доверительный интервал Уилсона в процентах `(low, high)`."""
    if n_tot <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = k_ok / float(n_tot)
    denom = 1.0 + (z * z) / float(n_tot)
    center = (p + (z * z) / (2.0 * float(n_tot))) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p)) / float(n_tot) + (z * z) / (4.0 * float(n_tot * n_tot)))
    return (max(0.0, (center - half) * 100.0), min(100.0, (center + half) * 100.0))


def locate_gsm8k_files() -> Tuple[str, str]:
    """Находит пути `(train.jsonl, test.jsonl)` локального датасета GSM8K."""
    candidates = [
        (Path("benchmarks/gsm8k/train.jsonl"), Path("benchmarks/gsm8k/test.jsonl")),
        (
            _REPO_ROOT / "benchmarks" / "gsm8k" / "train.jsonl",
            _REPO_ROOT / "benchmarks" / "gsm8k" / "test.jsonl",
        ),
    ]
    for tr_p, te_p in candidates:
        if tr_p.is_file() and te_p.is_file():
            return str(tr_p), str(te_p)
    raise FileNotFoundError("Не найдены файлы benchmarks/gsm8k/{train,test}.jsonl.")


class CramGSM8KSolver(OpenGraphCRAMSolver):
    """Фасад открытого решателя GSM8K поверх `OpenGraphCRAMSolver` (`0` шагов BPTT)."""

    def __init__(
        self,
        extractor: Optional[TrainInducedQuantityExtractor] = None,
        alu: Optional[SphericalBase10RegisterALU] = None,
        knobs: Optional[Dict[str, float | int]] = None,
        d_model: Optional[int] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.knobs = dict(TUNING_KNOBS)
        if knobs:
            self.knobs.update(knobs)
        eff_seed = int(self.knobs["seed"] if seed is None else seed)
        eff_d_model = int(self.knobs["d_model"] if d_model is None else d_model)
        eff_alu = alu if alu is not None else SphericalBase10RegisterALU(d_sub=15, num_digits=12, seed=3301)
        eff_ext = extractor if extractor is not None else TrainInducedQuantityExtractor()
        if device is None:
            import torch
            eff_dev = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            eff_dev = device
        super().__init__(
            extractor=eff_ext,
            alu=eff_alu,
            d_model=eff_d_model,
            device=eff_dev,
            seed=eff_seed,
        )

    def fit(
        self,
        train_items: Sequence[Dict[str, str]],
        slots_list: Optional[Sequence[Sequence[ExtractedSlot]]] = None,
        lm_feats_list: Optional[Sequence[Optional[Dict[str, object]]]] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
    ) -> Dict[str, float]:
        """Обучает data-driven экстрактор (при необходимости) и поле `CRAM+z` строго на `train_items`."""
        if not self.extractor.fitted:
            self.extractor.fit(train_items)
        if slots_list is None:
            slots_list = [
                self.extractor.extract_slots(str(it["question"]), include_aux_consts=True)[0]
                for it in train_items
            ]
        eff_lm: Sequence[Optional[Dict[str, object]]] = (
            [None] * len(train_items) if lm_feats_list is None else lm_feats_list
        )
        return self.fit_bptt_free(
            train_items=train_items,
            slots_list=slots_list,
            lm_feats_list=eff_lm,  # type: ignore[arg-type]
            epochs=int(self.knobs["cram_epochs"] if epochs is None else epochs),
            batch_size=int(self.knobs["cram_batch_size"] if batch_size is None else batch_size),
            lr=float(self.knobs["cram_lr"] if lr is None else lr),
        )


def evaluate_gsm8k_split(
    solver: OpenGraphCRAMSolver,
    items: Sequence[Dict[str, str]],
    slots_list: Optional[Sequence[Sequence[ExtractedSlot]]] = None,
    lm_feats_list: Optional[Sequence[Optional[Dict[str, object]]]] = None,
    tag: str = "Eval",
    include_oracle: bool = False,
) -> Dict[str, object]:
    """Строго слепая оценка: передаёт в `solve_blind` только `it['question']`."""
    n = len(items)
    p1 = p3 = p5 = unary_p1 = cert_tot = cert_ok = oracle_hits = 0
    max_drift = 0.0
    by_steps: Dict[str, List[int]] = {
        "2_steps": [0, 0, 0, 0],
        "3_steps": [0, 0, 0, 0],
        "4plus_steps": [0, 0, 0, 0],
    }

    for idx, it in enumerate(items):
        q_only = str(it["question"])
        ans_raw = str(it["answer"])
        gt = extract_gt_number(ans_raw)
        n_st = max(1, len(extract_answer_tags(ans_raw)))
        sl = None if slots_list is None else slots_list[idx]
        lmf = None if lm_feats_list is None else lm_feats_list[idx]

        out = solver.solve_blind(q_only, slots=sl, lm_feat=lmf)
        max_drift = max(max_drift, float(out["norm_drift"]))

        h1 = abs(float(out["pred_pass1"]) - gt) <= 1e-2
        h_un = abs(float(out["pred_unary_ablation"]) - gt) <= 1e-2
        h3 = any(abs(float(v) - gt) <= 1e-2 for v in out["pred_pass3"])  # type: ignore[attr-defined]
        h5 = any(abs(float(v) - gt) <= 1e-2 for v in out["pred_pass5"])  # type: ignore[attr-defined]
        p1 += int(h1)
        unary_p1 += int(h_un)
        p3 += int(h3)
        p5 += int(h5)

        if bool(out["dual_certified"]):
            cert_tot += 1
            cert_ok += int(h1)

        if include_oracle:
            eff_sl = (
                solver.extractor.extract_slots(q_only, include_aux_consts=True)[0]
                if sl is None
                else sl
            )
            cands, _ = solver._assemble_and_score_candidates(q_only, eff_sl, lmf)
            if any(abs(float(c["val"]) - gt) <= 1e-2 for c in cands):
                oracle_hits += 1

        bucket = "2_steps" if n_st <= 2 else ("3_steps" if n_st == 3 else "4plus_steps")
        by_steps[bucket][0] += 1
        by_steps[bucket][1] += int(h1)
        by_steps[bucket][2] += int(h3)
        by_steps[bucket][3] += int(h5)

    p1_pct = 100.0 * p1 / max(1, n)
    un_pct = 100.0 * unary_p1 / max(1, n)
    cov_pct = 100.0 * cert_tot / max(1, n)
    prec_pct = (100.0 * cert_ok / cert_tot) if cert_tot > 0 else 0.0
    res: Dict[str, object] = {
        "tag": tag,
        "n": n,
        "pass1": p1,
        "pass1_pct": p1_pct,
        "pass1_ci95": wilson_95_ci(p1, n),
        "unary_ablation_pass1": unary_p1,
        "unary_ablation_pass1_pct": un_pct,
        "pass3": p3,
        "pass3_pct": 100.0 * p3 / max(1, n),
        "pass5": p5,
        "pass5_pct": 100.0 * p5 / max(1, n),
        "cert_tot": cert_tot,
        "cert_ok": cert_ok,
        "cert_cov_pct": cov_pct,
        "cert_prec_pct": prec_pct,
        "max_norm_drift": max_drift,
        "gershgorin_margin": solver.ebm.gershgorin_margin(),
        "by_steps": by_steps,
    }
    if include_oracle:
        res["oracle_hits"] = oracle_hits
        res["oracle_pct"] = 100.0 * oracle_hits / max(1, n)
    return res


def run_blind_gsm8k_experiment(device: Optional[str] = None) -> Dict[str, object]:
    """Запускает открытый конвейер `CRAM+z` без LM на валидации `train[:6000] / cal[6000:6736] / dev[6736:]`.

    Холостой тест `test.jsonl` остаётся запечатанным и не открывается из фасада по умолчанию.
    """
    train_path, _ = locate_gsm8k_files()
    with open(train_path, "r", encoding="utf-8") as f:
        train_all = [json.loads(line) for line in f if line.strip()]

    tr_items = train_all[:6000]
    cal_items = train_all[6000:6736]
    dev_items = train_all[6736:]

    t0 = time.perf_counter()
    solver = CramGSM8KSolver(device=device)
    alu_diag = solver.alu.verify_crystal_invariants()
    fit_stats = solver.fit(tr_items)
    cal_slots = [solver.extractor.extract_slots(str(it["question"]), include_aux_consts=True)[0] for it in cal_items]
    dev_slots = [solver.extractor.extract_slots(str(it["question"]), include_aux_consts=True)[0] for it in dev_items]
    cal_stats = solver.calibrate_on_split(
        cal_items,
        cal_slots,
        [None] * len(cal_items),  # type: ignore[list-item]
        min_cert_precision=float(solver.knobs["min_cert_precision"]),
        min_cert_coverage=float(solver.knobs["min_cert_coverage"]),
    )
    dev_res = evaluate_gsm8k_split(solver, dev_items, slots_list=dev_slots, tag="BlindDev_NoLM")
    return {
        "alu_diag": alu_diag,
        "fit": fit_stats,
        "cal": cal_stats,
        "dev": dev_res,
        "elapsed_sec": time.perf_counter() - t0,
    }


__all__ = [
    "CRAMzOperatorGraphEBM",
    "CramGSM8KSolver",
    "OpenGraphCRAMSolver",
    "SphericalBase10FixedPointALU",
    "SphericalBase10RegisterALU",
    "TUNING_KNOBS",
    "TrainInducedQuantityExtractor",
    "evaluate_gsm8k_split",
    "extract_answer_tags",
    "extract_gt_number",
    "locate_gsm8k_files",
    "run_blind_gsm8k_experiment",
    "wilson_95_ci",
]


if __name__ == "__main__":
    summary = run_blind_gsm8k_experiment()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
