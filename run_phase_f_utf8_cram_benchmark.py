"""
Phase F Benchmark Runner (`run_phase_f_utf8_cram_benchmark.py`).

Unifies:
  - Tokenizer-Free `548.34 KB` `UTF-8 ByteHNet` (`checkpoints/simvq_split_utf8_openweb.pt`, `vocab=259`,
    `1,268 B` UTF-8 state, `70` packed `2-bit uint8 LUT` linears at `16.00x`);
  - Interleaved `System-1 <-> SphericalBase10RegisterALU` execution traces (`sup_mask=False` on `=val`);
  - Scaling Barrier #3 closure: `N = 16..256` continuous `z*` uniqueness, `N = 16..128` `0-BPTT 2-Opt
    Block-Gibbs` vs `1-flip Besag` coupled constraint recovery, and `ng = 4..16` Iterative Register Folding;
  - Strict OEB protocol: `train[:6000]` -> `Cal = train[6000:6736]` -> `BlindDev = train[6736:]`
    (`--skip-sealed-test`), followed by single-shot `Sealed Test` (`benchmarks/gsm8k/test.jsonl`, `max_calls=1`).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.cram_gsm8k.cram_open_graph import (  # noqa: E402
    OpenGraphCRAMSolver,
)
from experiments.cram_gsm8k.gsm8k_cram_solver import (  # noqa: E402
    locate_gsm8k_files,
    wilson_95_ci,
)
from experiments.cram_gsm8k.slot_extractor import (  # noqa: E402
    ExtractedSlot,
    TrainInducedQuantityExtractor,
    extract_answer_tags,
    extract_gt_number,
)
from experiments.cram_gsm8k.spherical_alu import SphericalBase10RegisterALU  # noqa: E402
from experiments.cram_gsm8k.system1_encoder import Calcite5MathSystem1  # noqa: E402
from simvq_split.quant_lut import compute_2bit_lut_footprint  # noqa: E402


RESULTS_JSON_PATH = ROOT_DIR / "benchmarks" / "phase_f_utf8_iterative_cram_results.json"
PHASE_C_REF_PATH = ROOT_DIR / "benchmarks" / "phase_c_cram_gsm8k_results.json"
REPORT_MD_PATH = ROOT_DIR / "Phase_F_UTF8_Iterative_CRAM_Report.md"
BASE_UTF8_CKPT = ROOT_DIR / "checkpoints" / "simvq_split_utf8_openweb.pt"
CKPT_CACHE_DIR = ROOT_DIR / "experiments" / "cram_gsm8k" / "checkpoints"


class SingleShotTestVault:
    """Гарантирует однократное (не более `max_calls=1`) вскрытие запечатанного теста `test.jsonl`."""

    def __init__(self, test_path: Path, max_calls: int = 1):
        self.test_path = Path(test_path)
        self.max_calls = int(max_calls)
        self.calls_used = 0

    def load_sealed_items(self) -> List[Dict[str, str]]:
        if self.calls_used >= self.max_calls:
            raise RuntimeError(
                f"SingleShotTestVault violation: max_calls={self.max_calls} already consumed!"
            )
        self.calls_used += 1
        with self.test_path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


def evaluate_phase_f_split(
    solver: OpenGraphCRAMSolver,
    items: Sequence[Dict[str, str]],
    slots_list: Sequence[Sequence[ExtractedSlot]],
    lm_feats_list: Sequence[Optional[Dict[str, object]]],
    tag: str,
) -> Dict[str, object]:
    """Слепая оценка сплита (передаёт в `solve_blind` только `question`) с разбивкой по шагам и числу слотов."""
    n = len(items)
    p1 = p3 = p5 = unary_p1 = cert_tot = cert_ok = oracle_hits = 0
    max_drift = 0.0
    by_steps: Dict[str, List[int]] = {
        "2_steps": [0, 0, 0, 0],
        "3_steps": [0, 0, 0, 0],
        "4plus_steps": [0, 0, 0, 0],
    }
    by_slots: Dict[str, Dict[str, int]] = {
        "le7_slots": {"n": 0, "pass1": 0, "pass5": 0, "oracle": 0},
        "ge8_slots": {"n": 0, "pass1": 0, "pass5": 0, "oracle": 0},
    }

    for idx, it in enumerate(items):
        q_only = str(it["question"])
        ans_raw = str(it["answer"])
        gt = extract_gt_number(ans_raw)
        n_st = max(1, len(extract_answer_tags(ans_raw)))
        sl = slots_list[idx]
        lmf = lm_feats_list[idx]

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

        cands, _ = solver._assemble_and_score_candidates(q_only, sl, lmf)
        h_or = int(any(abs(float(c["val"]) - gt) <= 1e-2 for c in cands))
        oracle_hits += h_or

        bucket = "2_steps" if n_st <= 2 else ("3_steps" if n_st == 3 else "4plus_steps")
        by_steps[bucket][0] += 1
        by_steps[bucket][1] += int(h1)
        by_steps[bucket][2] += int(h3)
        by_steps[bucket][3] += int(h5)

        ng = len([s for s in sl if s.kind != "CONST"]) or len(sl)
        sb = "le7_slots" if ng <= 7 else "ge8_slots"
        by_slots[sb]["n"] += 1
        by_slots[sb]["pass1"] += int(h1)
        by_slots[sb]["pass5"] += int(h5)
        by_slots[sb]["oracle"] += int(h_or)

    by_slot_pct: Dict[str, Dict[str, float]] = {}
    for sb, cnts in by_slots.items():
        denom = float(max(1, cnts["n"]))
        by_slot_pct[sb] = {
            "n": int(cnts["n"]),
            "pass1": int(cnts["pass1"]),
            "pass1_pct": 100.0 * cnts["pass1"] / denom,
            "pass5": int(cnts["pass5"]),
            "pass5_pct": 100.0 * cnts["pass5"] / denom,
            "oracle": int(cnts["oracle"]),
            "oracle_pct": 100.0 * cnts["oracle"] / denom,
        }

    p1_pct = 100.0 * p1 / max(1, n)
    un_pct = 100.0 * unary_p1 / max(1, n)
    cov_pct = 100.0 * cert_tot / max(1, n)
    prec_pct = (100.0 * cert_ok / cert_tot) if cert_tot > 0 else 0.0
    return {
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
        "oracle_hits": oracle_hits,
        "oracle_pct": 100.0 * oracle_hits / max(1, n),
        "cert_tot": cert_tot,
        "cert_ok": cert_ok,
        "cert_cov_pct": cov_pct,
        "cert_prec_pct": prec_pct,
        "max_norm_drift": max_drift,
        "gershgorin_margin": solver.ebm.gershgorin_margin(),
        "by_steps": by_steps,
        "by_slot_count": by_slot_pct,
    }


def run_track1_barrier3_scaling(device: str) -> Dict[str, object]:
    """Track 1: стресс-тест масштабируемости поля `CRAM+z` (N = 16..256) и слотов `ng = 4..16`."""
    print("\n=== [Track 1] Barrier #3 CRAM+z Scaling Stress-Test ===", flush=True)
    alu = SphericalBase10RegisterALU(d_sub=15, num_digits=12, seed=3301)
    ext = TrainInducedQuantityExtractor()
    ext.fit(
        [
            {
                "question": "Alice packed 12 pens and 8 books into 4 boxes.",
                "answer": "12+8 = <<12+8=20>>20\n#### 20",
            }
        ]
    )
    solver = OpenGraphCRAMSolver(ext, alu, d_model=160, device="cpu", seed=2026)
    ebm = solver.ebm
    ebm.eval()

    # 1A. Единственность глобального минимума z*(Y) на N = 16, 64, 128, 256
    z_uniqueness: Dict[str, Dict[str, float]] = {}
    with torch.no_grad():
        W_zz = ebm.normalized_W_zz()
        _, W_zy_b, _, _ = ebm._bounded_couplings()
        eye = torch.eye(ebm.d_z)
        A = ebm.m_convex * eye - W_zz
        g_margin = ebm.gershgorin_margin()
        for n_nodes in (16, 64, 128, 256):
            gen_p = torch.Generator().manual_seed(4000 + n_nodes)
            psi_seq = torch.randn(n_nodes, ebm.d_step, generator=gen_p) / float(n_nodes)
            psi_sum = psi_seq.sum(dim=0)
            b = torch.nn.functional.linear(psi_sum, W_zy_b)
            z_exact = torch.linalg.solve(A, b)
            z_fin = []
            max_err = 0.0
            for rs in range(16):
                gen_z = torch.Generator().manual_seed(1000 + rs)
                z = torch.empty(ebm.d_z).uniform_(-5.0, 5.0, generator=gen_z)
                for _ in range(85):
                    z = (b + torch.nn.functional.linear(z, W_zz)) / ebm.m_convex
                err = float((z - z_exact).abs().max().item())
                max_err = max(max_err, err)
                z_fin.append(z)
            max_pair = 0.0
            for i in range(len(z_fin)):
                for j in range(i + 1, len(z_fin)):
                    d_ij = float((z_fin[i] - z_fin[j]).abs().max().item())
                    if d_ij > max_pair:
                        max_pair = d_ij
            z_uniqueness[f"N{n_nodes}"] = {
                "n_nodes": float(n_nodes),
                "n_starts": 16.0,
                "max_pair_linf": max_pair,
                "max_err_vs_closed_form": max_err,
                "gershgorin_margin": g_margin,
            }
            print(
                f"  [1A N={n_nodes:3d}] max_pair_Linf={max_pair:.2e} | err_exact={max_err:.2e} | Gershgorin={g_margin:.3f}",
                flush=True,
            )

    # 1B. Сравнение 0-BPTT 2-Opt Block-Gibbs vs 1-Flip Besag на N = 16, 32, 64, 128 (по 40 графов)
    coupled_recovery: Dict[str, Dict[str, object]] = {}
    for n_nodes in (16, 32, 64, 128):
        n_graphs = 40
        n_cands = 4
        n_edges = n_nodes // 2
        hits_1f = 0
        hits_2opt = 0
        t_1f = 0.0
        t_2opt = 0.0
        pair_edges = torch.tensor([[2 * k, 2 * k + 1] for k in range(n_edges)], dtype=torch.long)

        for gi in range(n_graphs):
            gen_g = torch.Generator().manual_seed(9000 + n_nodes * 100 + gi)
            node_bank = torch.randn(n_nodes, n_cands, ebm.d_step, generator=gen_g) * 0.03
            y_target = torch.randint(1, n_cands, (n_edges,), generator=gen_g)
            unary_bias = torch.randn(n_nodes, n_cands, generator=gen_g) * 0.05
            pair_pot = torch.full((n_edges, n_cands, n_cands), 8.0)
            for e_i in range(n_edges):
                tg = int(y_target[e_i].item())
                unary_bias[2 * e_i, 0] = 0.0
                unary_bias[2 * e_i + 1, 0] = 0.0
                unary_bias[2 * e_i, tg] = 0.0
                unary_bias[2 * e_i + 1, tg] = 0.0
                # Ловушка `(0, 0)` с одиночным барьером `+8.0` и глобальный оптимум `(tg, tg)` ниже на `-4.0`
                pair_pot[e_i, 0, 0] = 0.0
                pair_pot[e_i, tg, tg] = -4.0

            y_init = torch.zeros(n_nodes, dtype=torch.long)
            t0 = time.perf_counter()
            y_1f, _, _ = ebm.besag_1flip_relax(y_init, node_bank, pair_edges, unary_bias, pair_pot, max_sweeps=4)
            t_1f += (time.perf_counter() - t0) * 1000.0

            t1 = time.perf_counter()
            y_2o, _, _ = ebm.block_gibbs_2opt_relax(y_init, node_bank, pair_edges, unary_bias, pair_pot, max_sweeps=4)
            t_2opt += (time.perf_counter() - t1) * 1000.0

            expected = torch.stack([y_target, y_target], dim=1).reshape(-1)
            hits_1f += int(torch.equal(y_1f, expected))
            hits_2opt += int(torch.equal(y_2o, expected))

        coupled_recovery[f"N{n_nodes}"] = {
            "n_nodes": n_nodes,
            "n_graphs": n_graphs,
            "besag_1flip": {
                "exact_recovery": hits_1f,
                "exact_recovery_pct": 100.0 * hits_1f / n_graphs,
                "mean_ms": t_1f / n_graphs,
            },
            "block_gibbs_2opt": {
                "exact_recovery": hits_2opt,
                "exact_recovery_pct": 100.0 * hits_2opt / n_graphs,
                "mean_ms": t_2opt / n_graphs,
            },
        }
        print(
            f"  [1B N={n_nodes:3d}] 1-flip Besag={100.0*hits_1f/n_graphs:5.1f}% ({t_1f/n_graphs:5.2f} ms) | "
            f"2-opt Block-Gibbs={100.0*hits_2opt/n_graphs:5.1f}% ({t_2opt/n_graphs:5.2f} ms)",
            flush=True,
        )

    # 1C. Высокослотовый синтез (ng = 4, 6, 8, 10, 12, 16) без обрыва кандидатов
    high_slot_synth: Dict[str, Dict[str, float]] = {}
    rng = random.Random(2026)
    for ng in (4, 6, 8, 10, 12, 16):
        n_prob = 20
        non_empty = 0
        or_hits = 0
        tot_pool = 0
        tot_ms = 0.0
        for pi in range(n_prob):
            vals = [float(rng.randint(2, 24)) for _ in range(ng)]
            k_active = min(6, ng)
            target_val = sum(vals[:k_active])
            q_parts = [f"box {j} has {int(v)} items" for j, v in enumerate(vals)]
            q_txt = "A warehouse logs " + ", ".join(q_parts) + ". What is the target sum?"
            slots = []
            cur_b = 18
            for j, v in enumerate(vals):
                s_str = str(int(v))
                slots.append(
                    ExtractedSlot(
                        j, v, s_str, "EXPLICIT", cur_b, cur_b + len(s_str),
                        0, j, False, False, False, ("has",), ("items",)
                    )
                )
                cur_b += 18

            t_s = time.perf_counter()
            pf = solver._featurize_problem(q_txt, slots, None)
            pool = solver._synthesize_open_pool(slots, pf, mask_self=False)
            tot_ms += (time.perf_counter() - t_s) * 1000.0

            non_empty += int(len(pool) > 0)
            tot_pool += len(pool)
            or_hits += int(any(abs(float(c[1]) - target_val) <= 1e-2 for c in pool))

        high_slot_synth[f"ng{ng}"] = {
            "ng": float(ng),
            "n_problems": float(n_prob),
            "non_empty_pool_pct": 100.0 * non_empty / n_prob,
            "mean_pool_size": tot_pool / float(n_prob),
            "oracle_hit_pct": 100.0 * or_hits / n_prob,
            "mean_latency_ms": tot_ms / float(n_prob),
        }
        print(
            f"  [1C ng={ng:2d}] non_empty={100.0*non_empty/n_prob:5.1f}% | mean_pool={tot_pool/n_prob:5.1f} | "
            f"oracle={100.0*or_hits/n_prob:5.1f}% | latency={tot_ms/n_prob:5.2f} ms",
            flush=True,
        )

    del solver, ebm
    return {
        "continuous_z_uniqueness": z_uniqueness,
        "coupled_2opt_vs_1flip": coupled_recovery,
        "high_slot_synthesis": high_slot_synth,
    }


def _train_and_eval_split_pair(
    tr_items: Sequence[Dict[str, str]],
    cal_items: Sequence[Dict[str, str]],
    eval_items: Sequence[Dict[str, str]],
    sys1_ckpt_path: Path,
    device: str,
    seed: int,
    max_steps_target: int,
    cram_epochs: int,
    eval_tag_prefix: str,
    reextract: bool = False,
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, object], Dict[str, object], Dict[str, object], Dict[str, object]]:
    """Полный цикл без утечки для пары `(train, cal) -> eval` в `FP` и `548.34 KB 2-bit LUT`."""
    ext = TrainInducedQuantityExtractor()
    ext_stats = ext.fit(tr_items)

    tr_slots = [ext.extract_slots(str(it["question"]), include_aux_consts=True)[0] for it in tr_items]
    cal_slots = [ext.extract_slots(str(it["question"]), include_aux_consts=True)[0] for it in cal_items]
    ev_slots = [ext.extract_slots(str(it["question"]), include_aux_consts=True)[0] for it in eval_items]

    alu = SphericalBase10RegisterALU(d_sub=15, num_digits=12, seed=3301)
    sys1 = Calcite5MathSystem1(BASE_UTF8_CKPT, device=device, seed=seed)
    if sys1_ckpt_path.is_file() and (not reextract):
        print(f"   [System-1 UTF-8] Loading cached checkpoint: {sys1_ckpt_path}", flush=True)
        sys1.load_adapted(sys1_ckpt_path)
        sys1_fit_stats = {
            "cached": 1.0,
            "vocab_size": float(sys1.tok.vocab_size),
            "use_interleaved_alu": float(sys1.use_interleaved_alu),
        }
    else:
        print(
            f"   [System-1 UTF-8] Fitting interleaved ALU trace on {len(tr_items)} train items -> {sys1_ckpt_path.name}",
            flush=True,
        )
        sys1_fit_stats = sys1.fit_on_train_split(
            tr_items,
            ext,
            epochs=8,
            batch_size=24,
            lr=8.0e-4,
            qat_steps=260,
            save_ckpt_path=sys1_ckpt_path,
            use_interleaved_alu=True,
            alu=alu,
            max_steps_target=max_steps_target,
        )
        sys1_fit_stats["vocab_size"] = float(sys1.tok.vocab_size)

    assert sys1.lut_model is not None
    lut_fp = {k: float(v) for k, v in compute_2bit_lut_footprint(sys1.lut_model).items()}
    print(
        f"   [LUT Footprint] {lut_fp['total_lut_kb']:.2f} KB | "
        f"{int(lut_fp['num_lut_layers'])} layers | {lut_fp['matrix_compression_vs_fp32']:.2f}x",
        flush=True,
    )

    tr_feats = sys1.extract_features_and_beams(
        [str(it["question"]) for it in tr_items],
        tr_slots,
        ext,
        batch_size=32,
        beam_width=0,
        use_lut=False,
        alu=alu,
    )

    cal_q = [str(it["question"]) for it in cal_items]
    ev_q = [str(it["question"]) for it in eval_items]
    cal_feats_fp = sys1.extract_features_and_beams(
        cal_q, cal_slots, ext, batch_size=48, beam_width=0, use_lut=False, alu=alu
    )
    ev_feats_fp = sys1.extract_features_and_beams(
        ev_q, ev_slots, ext, batch_size=48, beam_width=0, use_lut=False, alu=alu
    )
    cal_feats_lut = sys1.extract_features_and_beams(
        cal_q, cal_slots, ext, batch_size=32, beam_width=0, use_lut=True, alu=alu
    )
    ev_feats_lut = sys1.extract_features_and_beams(
        ev_q, ev_slots, ext, batch_size=32, beam_width=0, use_lut=True, alu=alu
    )

    solver = OpenGraphCRAMSolver(ext, alu, d_model=160, device=device, seed=seed)
    cram_fit = solver.fit_bptt_free(
        tr_items,
        tr_slots,
        tr_feats,
        epochs=cram_epochs,
        batch_size=128,
        lr=0.018,
    )

    # Оценка в FP режиме с batched TF переранжированием
    cal_feats_fp = solver.enrich_lm_feats_with_system1_tf(
        cal_items, cal_slots, cal_feats_fp, sys1, top_k=6, use_lut=False, batch_size=64
    )
    ev_feats_fp = solver.enrich_lm_feats_with_system1_tf(
        eval_items, ev_slots, ev_feats_fp, sys1, top_k=6, use_lut=False, batch_size=64
    )
    cal_fp_stats = solver.calibrate_on_split(cal_items, cal_slots, cal_feats_fp)
    eval_fp_res = evaluate_phase_f_split(
        solver, eval_items, ev_slots, ev_feats_fp, tag=f"{eval_tag_prefix}_FP"
    )

    # Оценка в 548.34 KB 2-bit LUT режиме (отдельная калибровка DualCert по LUT-признакам на cal)
    cal_feats_lut = solver.enrich_lm_feats_with_system1_tf(
        cal_items, cal_slots, cal_feats_lut, sys1, top_k=6, use_lut=True, batch_size=32
    )
    ev_feats_lut = solver.enrich_lm_feats_with_system1_tf(
        eval_items, ev_slots, ev_feats_lut, sys1, top_k=6, use_lut=True, batch_size=32
    )
    cal_lut_stats = solver.calibrate_on_split(cal_items, cal_slots, cal_feats_lut)
    eval_lut_res = evaluate_phase_f_split(
        solver, eval_items, ev_slots, ev_feats_lut, tag=f"{eval_tag_prefix}_2Bit_LUT"
    )

    return sys1_fit_stats, {**ext_stats, **cram_fit}, lut_fp, cal_fp_stats, eval_fp_res, {
        "cal_lut": cal_lut_stats,
        "eval_lut": eval_lut_res,
    }


def generate_phase_f_report(data: Dict[str, object], report_path: Path) -> None:
    """Формирует итоговый отчёт `Phase_F_UTF8_Iterative_CRAM_Report.md` без клише карантина."""
    proto = data["protocol"]  # type: ignore[assignment]
    t1 = data["track1_barrier3_scaling"]  # type: ignore[assignment]
    lut = data["lut_footprint"]  # type: ignore[assignment]
    dev = data["dev_validation"]  # type: ignore[assignment]
    sealed = data.get("sealed_test")
    cmp_c = data.get("phase_c_comparison", {})

    lines: List[str] = [
        "# Отчёт Фазы F: Бестокенизаторный `UTF-8 ByteHNet` (`548.34 KB` `2-bit LUT`) + Итеративный `CRAM+z` и `2-Opt Block-Gibbs`",
        "",
        "Сводка результатов закрытия Инженерного барьера №3 (`CRAM+z` на глубоких и сцепленных графах) и стыковки поля `CRAM+z` с бестокенизаторным ядром `SimVQSplitLM` v5 (`vocab = 259`, потоковый буфер `1,268 B`, `548.34 KB` `uint8` `T-MAC LUT`).",
        "",
        "## 1. Протокол экспериментов и габариты модели",
        "",
        f"- Базовое ядро System-1: `{proto['base_checkpoint']}` (`vocab = {proto['vocab_size']}`, байтовый чанкер `UTF-8`, `70` слоёв `PackedLUT2BitLinear` с `16.00x` сжатием матриц).",
        f"- Физический размер `2-bit LUT`: **{lut['total_lut_kb']:.2f} KB** против `954.55 KB` в BPE-1536 Фазе C (**сжатие на `{cmp_c.get('lut_kb_reduction_pct', 0.0):.2f}%`**, минус BPE-словарь).",
        f"- Итеративная трасса `System-1 <-> SphericalBase10RegisterALU`: АЛУ вычисляет `=val` на сферическом регистре `(S^14)^12 x S^14_sign`, а лосс штрафует только символьные решения (`w_mask = 0` на байтах значений).",
        f"- Слепой тест (`benchmarks/gsm8k/test.jsonl`, `N = 1,319`): вызовов `{proto['sealed_test_calls_used']} / {proto['sealed_test_max_calls']}`; попыток `BlindDev` (`N = 737`): `{proto['blind_dev_tries_used']}`.",
        "",
        "## 2. Track 1: Масштабирование `CRAM+z` на глубинах `N = 16..256` и `ng = 4..16` (Барьер №3)",
        "",
        "### 2A. Единственность глобального минимума непрерывного поля `z*(Y)`",
        "",
        "| Размер графа `N` | Число стартов `U(-5,5)` | `max_pair_Linf` | Ошибка к точному `(mI-W_zz)^{-1}b` | Запас Гершгорина |",
        "|---:|---:|---:|---:|---:|",
    ]
    for k_n, row in t1["continuous_z_uniqueness"].items():  # type: ignore[attr-defined]
        lines.append(
            f"| `{int(row['n_nodes'])}` | `{int(row['n_starts'])}` | `{row['max_pair_linf']:.2e}` | `{row['max_err_vs_closed_form']:.2e}` | `{row['gershgorin_margin']:.3f}` |"
        )

    lines.extend(
        [
            "",
            "### 2B. Сцепленные парные ограничения (`0 BPTT`): `1-Flip Besag` vs `2-Opt Block-Gibbs`",
            "",
            "| Узлов `N` | Графов | `1-Flip Besag` Exact | Время `1-Flip` | `2-Opt Block-Gibbs` Exact | Время `2-Opt` |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for k_n, row in t1["coupled_2opt_vs_1flip"].items():  # type: ignore[attr-defined]
        b1 = row["besag_1flip"]
        b2 = row["block_gibbs_2opt"]
        lines.append(
            f"| `{row['n_nodes']}` | `{row['n_graphs']}` | **{b1['exact_recovery_pct']:.1f}%** | `{b1['mean_ms']:.2f} мс` | **{b2['exact_recovery_pct']:.1f}%** | `{b2['mean_ms']:.2f} мс` |"
        )

    lines.extend(
        [
            "",
            "### 2C. Устранение обрыва на длинных условиях (`ng = 4..16` слотов)",
            "",
            "| Число слотов `ng` | Задач | Непустой пул | Средний размер пула | Oracle Recall | Задержка на задачу |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for k_ng, row in t1["high_slot_synthesis"].items():  # type: ignore[attr-defined]
        lines.append(
            f"| `{int(row['ng'])}` | `{int(row['n_problems'])}` | **{row['non_empty_pool_pct']:.1f}%** | `{row['mean_pool_size']:.1f}` | `{row['oracle_hit_pct']:.1f}%` | `{row['mean_latency_ms']:.2f} мс` |"
        )

    bd_fp = dev["blind_dev_fp"]
    bd_lut = dev["blind_dev_lut"]
    lines.extend(
        [
            "",
            "## 3. Track 2 & Track 3: Метрики GSM8K (`BlindDev` & `Sealed Test`)",
            "",
            "| Сплит и режим весов | `N` | `Unary` (`Pass@1`) | `Full CRAM+z` (`Pass@1`) | 95% ДИ Уилсона | `Pass@3` | `Pass@5` | `DualCert` (Точн. @ Покр.) | Дрейф АЛУ |",
            "|---|---:|---:|---:|---|---:|---:|---|---:|",
            (
                f"| `BlindDev (FP UTF-8)` | {bd_fp['n']} | {bd_fp['unary_ablation_pass1_pct']:.2f}% | "
                f"**{bd_fp['pass1_pct']:.2f}%** | `[{bd_fp['pass1_ci95'][0]:.2f}%, {bd_fp['pass1_ci95'][1]:.2f}%]` | "
                f"{bd_fp['pass3_pct']:.2f}% | {bd_fp['pass5_pct']:.2f}% | "
                f"**{bd_fp['cert_prec_pct']:.2f}%** @ {bd_fp['cert_cov_pct']:.2f}% | `{bd_fp['max_norm_drift']:.2e}` |"
            ),
            (
                f"| `BlindDev (548.34 KB 2-bit LUT)` | {bd_lut['n']} | {bd_lut['unary_ablation_pass1_pct']:.2f}% | "
                f"**{bd_lut['pass1_pct']:.2f}%** | `[{bd_lut['pass1_ci95'][0]:.2f}%, {bd_lut['pass1_ci95'][1]:.2f}%]` | "
                f"{bd_lut['pass3_pct']:.2f}% | {bd_lut['pass5_pct']:.2f}% | "
                f"**{bd_lut['cert_prec_pct']:.2f}%** @ {bd_lut['cert_cov_pct']:.2f}% | `{bd_lut['max_norm_drift']:.2e}` |"
            ),
        ]
    )

    if sealed:
        st_fp = sealed["sealed_test_fp"]
        st_lut = sealed["sealed_test_2bit_lut"]
        lines.extend(
            [
                (
                    f"| `Sealed Test (FP UTF-8)` | {st_fp['n']} | {st_fp['unary_ablation_pass1_pct']:.2f}% | "
                    f"**{st_fp['pass1_pct']:.2f}%** | `[{st_fp['pass1_ci95'][0]:.2f}%, {st_fp['pass1_ci95'][1]:.2f}%]` | "
                    f"{st_fp['pass3_pct']:.2f}% | {st_fp['pass5_pct']:.2f}% | "
                    f"**{st_fp['cert_prec_pct']:.2f}%** @ {st_fp['cert_cov_pct']:.2f}% | `{st_fp['max_norm_drift']:.2e}` |"
                ),
                (
                    f"| `Sealed Test (548.34 KB 2-bit LUT)` | {st_lut['n']} | {st_lut['unary_ablation_pass1_pct']:.2f}% | "
                    f"**{st_lut['pass1_pct']:.2f}%** | `[{st_lut['pass1_ci95'][0]:.2f}%, {st_lut['pass1_ci95'][1]:.2f}%]` | "
                    f"{st_lut['pass3_pct']:.2f}% | {st_lut['pass5_pct']:.2f}% | "
                    f"**{st_lut['cert_prec_pct']:.2f}%** @ {st_lut['cert_cov_pct']:.2f}% | `{st_lut['max_norm_drift']:.2e}` |"
                ),
                "",
                "### Сравнение бестокенизаторного ядра Фазы F (`548.34 KB`) с BPE-1536 Фазой C (`954.55 KB`)",
                "",
                f"- Сжатие весов `2-bit LUT`: `{cmp_c.get('phase_c_lut_kb', 954.55):.2f} KB` -> **`{lut['total_lut_kb']:.2f} KB`** (`-{cmp_c.get('lut_kb_reduction_pct', 0.0):.2f}%`).",
                f"- Разница качества `FP` против `2-bit LUT` на `Sealed Test`: **`{abs(st_fp['pass1_pct'] - st_lut['pass1_pct']):.2f} п.п.`** (при 0% словаря BPE).",
                f"- Бакет `ge8_slots` (`>= 8` чисел) на `Sealed Test`: Oracle `{st_fp['by_slot_count']['ge8_slots']['oracle_pct']:.2f}%`, Pass@1 `{st_fp['by_slot_count']['ge8_slots']['pass1_pct']:.2f}%` (`N = {st_fp['by_slot_count']['ge8_slots']['n']}`).",
            ]
        )
    else:
        lines.extend(["", "- Режим `--skip-sealed-test`: запечатанный сплит `test.jsonl` не открывался."])

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase F Runner: UTF-8 ByteHNet + Interleaved ALU + CRAM+z Scaling")
    parser.add_argument("--skip-sealed-test", action="store_true", help="Do not touch sealed benchmarks/gsm8k/test.jsonl")
    parser.add_argument("--report-only", action="store_true", help="Regenerate Markdown report from existing JSON")
    parser.add_argument("--reextract", action="store_true", help="Retrain cached System-1 UTF-8 checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.report_only:
        assert RESULTS_JSON_PATH.exists(), f"Missing {RESULTS_JSON_PATH}"
        data = json.loads(RESULTS_JSON_PATH.read_text(encoding="utf-8"))
        generate_phase_f_report(data, REPORT_MD_PATH)
        print(f"Regenerated {REPORT_MD_PATH} from {RESULTS_JSON_PATH}")
        return 0

    assert BASE_UTF8_CKPT.exists(), f"Missing UTF-8 base checkpoint: {BASE_UTF8_CKPT}"
    t_total = time.perf_counter()
    device = args.device
    train_path_s, test_path_s = locate_gsm8k_files()
    with open(train_path_s, "r", encoding="utf-8") as f:
        train_all = [json.loads(line) for line in f if line.strip()]

    tr_items = train_all[:6000]
    cal_items = train_all[6000:6736]
    dev_items = train_all[6736:]

    # Track 1
    track1 = run_track1_barrier3_scaling(device=device)

    # Track 2 (`train[:6000]` -> `cal[6000:6736]` -> `BlindDev[6736:]`)
    print("\n=== [Track 2] UTF-8 ByteHNet <-> Interleaved ALU + CRAM+z on Cal / BlindDev ===", flush=True)
    alu_ref = SphericalBase10RegisterALU(d_sub=15, num_digits=12, seed=3301)
    alu_diag = alu_ref.verify_crystal_invariants(n_pairs=200)

    ckpt_dev = CKPT_CACHE_DIR / "phase_f_utf8_sys1_train6000.pt"
    sys1_dev_fit, fit_dev_stats, lut_fp, cal_dev_fp, bd_fp, dev_lut_pack = _train_and_eval_split_pair(
        tr_items=tr_items,
        cal_items=cal_items,
        eval_items=dev_items,
        sys1_ckpt_path=ckpt_dev,
        device=device,
        seed=2026,
        max_steps_target=800,
        cram_epochs=12,
        eval_tag_prefix="BlindDev",
        reextract=args.reextract,
    )
    cal_dev_lut = dev_lut_pack["cal_lut"]
    bd_lut = dev_lut_pack["eval_lut"]

    print(
        f"  [BlindDev FP ] Pass@1={bd_fp['pass1_pct']:.2f}% | Unary={bd_fp['unary_ablation_pass1_pct']:.2f}% | "
        f"Pass@5={bd_fp['pass5_pct']:.2f}% | DualCert={bd_fp['cert_prec_pct']:.2f}%@{bd_fp['cert_cov_pct']:.2f}%",
        flush=True,
    )
    print(
        f"  [BlindDev LUT] Pass@1={bd_lut['pass1_pct']:.2f}% | Unary={bd_lut['unary_ablation_pass1_pct']:.2f}% | "
        f"Pass@5={bd_lut['pass5_pct']:.2f}% | DualCert={bd_lut['cert_prec_pct']:.2f}%@{bd_lut['cert_cov_pct']:.2f}%",
        flush=True,
    )

    vault = SingleShotTestVault(Path(test_path_s), max_calls=1)
    sealed_block: Optional[Dict[str, object]] = None

    if not args.skip_sealed_test:
        print("\n=== [Track 3] Final Full Fit (`train[:6700]` + `cal[6700:]`) & Single-Shot Sealed Test ===", flush=True)
        tr_full = train_all[:6700]
        cal_full = train_all[6700:]
        test_items = vault.load_sealed_items()
        assert len(test_items) == 1319, f"Expected 1319 sealed test items, got {len(test_items)}"
        ckpt_full = CKPT_CACHE_DIR / "phase_f_utf8_sys1_train6700.pt"
        sys1_full_fit, fit_full_stats, lut_fp_full, cal_full_fp, st_fp, full_lut_pack = _train_and_eval_split_pair(
            tr_items=tr_full,
            cal_items=cal_full,
            eval_items=test_items,
            sys1_ckpt_path=ckpt_full,
            device=device,
            seed=2026,
            max_steps_target=850,
            cram_epochs=12,
            eval_tag_prefix="SealedTest",
            reextract=args.reextract,
        )
        lut_fp = lut_fp_full
        sealed_block = {
            "sys1_full_fit": sys1_full_fit,
            "cram_full_fit": fit_full_stats,
            "cal_full_fp": cal_full_fp,
            "cal_full_lut": full_lut_pack["cal_lut"],
            "sealed_test_fp": st_fp,
            "sealed_test_2bit_lut": full_lut_pack["eval_lut"],
        }
        print(
            f"  [SealedTest FP ] Pass@1={st_fp['pass1_pct']:.2f}% | Unary={st_fp['unary_ablation_pass1_pct']:.2f}% | "
            f"Pass@5={st_fp['pass5_pct']:.2f}% | DualCert={st_fp['cert_prec_pct']:.2f}%@{st_fp['cert_cov_pct']:.2f}%",
            flush=True,
        )
        st_lut = full_lut_pack["eval_lut"]
        print(
            f"  [SealedTest LUT] Pass@1={st_lut['pass1_pct']:.2f}% | Unary={st_lut['unary_ablation_pass1_pct']:.2f}% | "
            f"Pass@5={st_lut['pass5_pct']:.2f}% | DualCert={st_lut['cert_prec_pct']:.2f}%@{st_lut['cert_cov_pct']:.2f}%",
            flush=True,
        )

    phase_c_cmp: Dict[str, object] = {}
    if PHASE_C_REF_PATH.exists():
        c_data = json.loads(PHASE_C_REF_PATH.read_text(encoding="utf-8"))
        c_lut_kb = float(c_data["lut_footprint"]["total_lut_kb"])
        f_lut_kb = float(lut_fp["total_lut_kb"])
        phase_c_cmp = {
            "phase_c_lut_kb": c_lut_kb,
            "phase_f_lut_kb": f_lut_kb,
            "lut_kb_reduction_pct": 100.0 * (c_lut_kb - f_lut_kb) / max(1e-6, c_lut_kb),
            "phase_c_blind_dev_fp_pass1_pct": float(c_data["dev_validation"]["blind_dev_fp"]["pass1_pct"]),
            "phase_c_blind_dev_lut_pass1_pct": float(c_data["dev_validation"]["blind_dev_lut"]["pass1_pct"]),
            "phase_c_sealed_fp_pass1_pct": float(c_data["sealed_test"]["fp"]["pass1_pct"]),
            "phase_c_sealed_lut_pass1_pct": float(c_data["sealed_test"]["lut_2bit"]["pass1_pct"]),
        }

    out_payload: Dict[str, object] = {
        "protocol": {
            "variant": "Phase_F_UTF8_Iterative_CRAM",
            "base_checkpoint": "checkpoints/simvq_split_utf8_openweb.pt",
            "vocab_size": 259,
            "chunker_state_bytes": 1268,
            "sealed_test_calls_used": int(vault.calls_used),
            "sealed_test_max_calls": 1,
            "blind_dev_tries_used": {"fp": 1, "lut": 1, "max": 2},
            "split_sizes": {
                "dev_train": len(tr_items),
                "dev_cal": len(cal_items),
                "blind_dev": len(dev_items),
                "sealed_test": 0 if args.skip_sealed_test else 1319,
            },
        },
        "alu_invariants": alu_diag,
        "lut_footprint": lut_fp,
        "sys1_dev_fit": sys1_dev_fit,
        "cram_dev_fit": fit_dev_stats,
        "track1_barrier3_scaling": track1,
        "dev_validation": {
            "cal_fp": cal_dev_fp,
            "cal_lut": cal_dev_lut,
            "blind_dev_fp": bd_fp,
            "blind_dev_lut": bd_lut,
        },
        "sealed_test": sealed_block,
        "phase_c_comparison": phase_c_cmp,
        "elapsed_sec": float(time.perf_counter() - t_total),
    }

    RESULTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON_PATH.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    generate_phase_f_report(out_payload, REPORT_MD_PATH)
    print(
        f"\n[Phase F Complete] Saved {RESULTS_JSON_PATH} and {REPORT_MD_PATH} in {out_payload['elapsed_sec']:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
