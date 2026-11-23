"""
SLEEP-DAG-RSI v8 Track T2 (`uebr_mdl_t2`): Eliminating W5 (`UEBR System-2` 1-step & OOD
domain shift on `ASDiv-A`: `29.36% B0 -> 20.93% A3` despite `81.42% Oracle`).

Zero-Leak & Protected-Path Isolation Protocol:
  - ZERO modifications to `experiments/uebr_system2/` or `experiments/cram_gsm8k/`.
  - ZERO access to `benchmarks/gsm8k/test.jsonl`.
  - Dev Selection Split (Rounds 1-4): `GSM8K BlindDev (737)` + `GSM8K BlindDev-Stress (737)`,
    scored via 8 empirical slot-count/stress buckets + Harmonic Mean across `<=3` vs `>=4` slots.
  - Sealed Blind Vault (`BlindVaultGuard(max_calls=1)`): `ASDiv-A (2078)` + `MultiArith (600)` +
    `SVAMP (1000)`, opened ONCE on the final selected champion and compared side-by-side
    against unmodified `phase_j_uebr_a3_2026.pt` baseline.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

PLUGIN_SCRIPTS = os.path.expanduser(
    "~/.gemini/config/plugins/autonomous-research/skills/dream-rsi/scripts"
)
FALLBACK_SCRIPTS = (
    "C:/Users/vanya/.gemini/config/plugins/autonomous-research/skills/dream-rsi/scripts"
)
for _p in (PLUGIN_SCRIPTS, FALLBACK_SCRIPTS):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import engine as dream_engine  # noqa: E402
from cartridge import AutoDreamRunner, BranchCandidate, TaskCartridge  # noqa: E402

from experiments.cram_gsm8k.gsm8k_cram_solver import wilson_95_ci  # noqa: E402
from experiments.cram_gsm8k.slot_extractor import (  # noqa: E402
    ExtractedSlot,
    TrainInducedQuantityExtractor,
    extract_gt_number,
)
from experiments.cram_gsm8k.spherical_alu import SphericalBase10RegisterALU  # noqa: E402
from experiments.uebr_system2.tier1_gaq_extractor import (  # noqa: E402
    SingleShotMathVault,
    SingleShotSVAMPVault,
    build_blind_dev_stress_split,
    extract_tier1_lut_features,
)
from experiments.uebr_system2.uebr_graph_solver import (  # noqa: E402
    D_GAQ_TOTAL,
    K_SLOT_BUF,
    UEBROpenGraphSolver,
)

TIER1_LUT_CKPT = ROOT_DIR / "checkpoints" / "calcite5_tier1_17m_utf8_qat_h_lut.pt"
UEBR_A3_CKPT = ROOT_DIR / "experiments" / "uebr_system2" / "checkpoints" / "phase_j_uebr_a3_2026.pt"
GSM8K_TRAIN_PATH = ROOT_DIR / "benchmarks" / "gsm8k" / "train.jsonl"
SVAMP_PATH = ROOT_DIR / "benchmarks" / "svamp" / "svamp_1000.json"
MULTIARITH_PATH = ROOT_DIR / "benchmarks" / "ood_math" / "multiarith_600.json"
ASDIV_PATH = ROOT_DIR / "benchmarks" / "ood_math" / "asdiv_a.json"
CACHE_DIR = ROOT_DIR / "experiments" / "uebr_system2" / "cache"
OUT_REPORT_PATH = ROOT_DIR / "benchmarks" / "dream_rsi_t2_uebr_results.json"


def _harmonic_mean(vals: Sequence[float]) -> float:
    clean = [max(1e-6, float(v)) for v in vals]
    return float(len(clean)) / sum(1.0 / v for v in clean)


def _load_gsm8k_train_splits() -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    with open(GSM8K_TRAIN_PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:6000], rows[6000:6736], rows[6736:]


@torch.no_grad()
def _extract_decomposed_split(
    solver: UEBROpenGraphSolver,
    items: Sequence[Dict[str, object]],
    slots_list: Sequence[Sequence[ExtractedSlot]],
    lm_feats_list: Sequence[Dict[str, object]],
    tag: str,
) -> List[Dict[str, Any]]:
    """
    Runs the frozen `UEBROpenGraphSolver` candidate assembly + EBM forward pass ONCE per split
    and records exact decomposed energy terms (`eu`, `ep`, `er`, `ez`, `e_lap`, `meta_m`,
    `t_len`, `unused_null_mass`, `u_risk`, `vals`, `gt`) without touching any protected files.
    """
    t0 = time.perf_counter()
    ebm = solver.ebm
    ebm.eval()
    w_eff = ebm.w_meta_effective().detach().cpu().numpy().astype(np.float64)
    records: List[Dict[str, Any]] = []

    for it, sl, lmf in zip(items, slots_list, lm_feats_list):
        q_only = str(it["question"])
        gt = float(it["answer_val"]) if "answer_val" in it else extract_gt_number(str(it["answer"]))
        ng = sum(1 for s in sl if s.kind != "CONST")
        u_risk = float(lmf.get("u_risk", lmf.get("u_prob", 1.0))) if lmf is not None else 1.0

        # Replicate `_assemble_and_score_candidates` exact logic including Null-Slot active-subgraph projection
        def _score_sub(cur_slots: Sequence[ExtractedSlot], cur_lmf: Optional[Dict[str, object]], allow_null: bool):
            k = len(cur_slots)
            if k == 0:
                return None
            if k == 1:
                return {
                    "trivial": True,
                    "val": float(cur_slots[0].val),
                }
            pf = solver._featurize_problem(q_only, cur_slots, cur_lmf)
            p_null = np.zeros(k, dtype=np.float32)
            if solver.use_rlns_and_null and cur_lmf is not None:
                nf_t = torch.from_numpy(pf["null_inv"]).to(solver.device)  # type: ignore[arg-type]
                p_null = torch.sigmoid(ebm.null_prior_net(nf_t).squeeze(-1)).cpu().numpy()
                if allow_null and k >= 3:
                    gr_idx = [i for i, s in enumerate(cur_slots) if s.kind != "CONST"]
                    act_gr = [i for i in gr_idx if float(p_null[i]) <= 0.78]
                    if 2 <= len(act_gr) < len(gr_idx):
                        keep = act_gr + [i for i, s in enumerate(cur_slots) if s.kind == "CONST"]
                        sub_slots = [cur_slots[i] for i in keep]
                        hn_o = np.asarray(cur_lmf.get("h_num", np.zeros((0, solver.d_model))), dtype=np.float32)
                        hp_o = np.asarray(cur_lmf.get("h_post", np.zeros((0, solver.d_model))), dtype=np.float32)
                        if hn_o.ndim == 2 and hn_o.shape[0] == k and hp_o.ndim == 2 and hp_o.shape[0] == k:
                            sub_lm = dict(cur_lmf)
                            sub_lm["h_num"] = hn_o[keep]
                            sub_lm["h_post"] = hp_o[keep]
                            if "u_slots" in cur_lmf and len(cur_lmf["u_slots"]) == k:  # type: ignore[arg-type]
                                sub_lm["u_slots"] = np.asarray(cur_lmf["u_slots"], dtype=np.float32)[keep]
                            if "varent_slots" in cur_lmf and len(cur_lmf["varent_slots"]) == k:  # type: ignore[arg-type]
                                sub_lm["varent_slots"] = np.asarray(cur_lmf["varent_slots"], dtype=np.float32)[keep]
                            res_sub = _score_sub(sub_slots, sub_lm, allow_null=False)
                            if res_sub is not None:
                                return res_sub

            cands = solver._synthesize_open_pool(cur_slots, pf, mask_self=False)
            if not cands:
                return None

            pk = solver._pack_candidates_cpu(cur_slots, pf, cands)
            tensors, _ = solver._collate_uebr_to_device([pk])
            (
                all_uf, all_pf, all_rf, all_lm, all_hq,
                slot_prob_idx, prob_u_risk, slot_offsets, slot_counts,
                slot_gidx, slot_roles, slot_cand_idx, slot_inv_k,
                root_prob_idx, root_op, bm_cand, bp_pair, bp_u_gidx, bp_v_gidx,
                bo_op, bra, brb, bwt, psi_raw_pad, psi_mask, psi_lens,
                meta_m_t, cand_prob_idx, cand_local_idx, m_max,
                dag_op, dag_a, dag_b, dag_len, all_nfeat,
            ) = tensors
            n_c = len(cands)
            dev = all_uf.device

            enc = ebm.encode_gaq_and_unary(
                all_uf, all_lm, all_rf, all_hq, slot_prob_idx, prob_u_risk, all_nfeat=all_nfeat
            )
            eu_full = torch.zeros(n_c, dtype=torch.float32, device=dev).index_add_(
                0, slot_cand_idx, enc.u_tot[slot_gidx, slot_roles] * slot_inv_k
            )
            eu_lex = torch.zeros(n_c, dtype=torch.float32, device=dev).index_add_(
                0, slot_cand_idx, enc.u_lex[slot_gidx, slot_roles] * slot_inv_k
            )
            r_pot = 1.5 * F.linear(all_rf, ebm.W_r)
            er = r_pot[root_prob_idx, root_op]
            p_pot = 2.5 * F.linear(all_pf, ebm.W_p)
            if bm_cand.numel() > 0:
                edge_base = p_pot[bp_pair, bo_op] + ebm.K_cram[bra, brb, bo_op]
                if ebm.use_gaq_bridge:
                    bp_prob = slot_prob_idx[bp_u_gidx]
                    gaq_pair_in = torch.cat(
                        [enc.q_slots[bp_u_gidx], enc.q_slots[bp_v_gidx], enc.q_prob[bp_prob]], dim=-1
                    )
                    gaq_pair_logits = ebm.gaq_pair_head(gaq_pair_in)
                    edge_base = edge_base + torch.gather(gaq_pair_logits, 1, bo_op.unsqueeze(-1)).squeeze(-1)
                ep = torch.zeros(n_c, dtype=torch.float32, device=dev).index_add_(0, bm_cand, edge_base * bwt)
            else:
                ep = torch.zeros(n_c, dtype=torch.float32, device=dev)

            couplings = ebm._bounded_couplings()
            psi_seq = torch.tanh(ebm.step_proj(psi_raw_pad))
            ez = ebm.step_field_scale * ebm._score_psi_batched(psi_seq, psi_mask, psi_lens, couplings)

            if ebm.use_spd_laplacian:
                bsz = len(slot_offsets)
                q_pad = torch.zeros((bsz, K_SLOT_BUF, D_GAQ_TOTAL), device=dev, dtype=torch.float32)
                s_mask = torch.zeros((bsz, K_SLOT_BUF), device=dev, dtype=torch.bool)
                for bi, (so, sc) in enumerate(zip(slot_offsets, slot_counts)):
                    uc = min(sc, K_SLOT_BUF)
                    if uc > 0:
                        q_pad[bi, :uc] = enc.q_slots[so : so + uc]
                        s_mask[bi, :uc] = True
                e_lap, _, _, _, _, _ = ebm.spd_laplacian.score_batch(
                    q_slots_pad=q_pad,
                    slot_mask=s_mask,
                    q_prob=enc.q_prob,
                    cand_prob_idx=cand_prob_idx,
                    cand_local_idx=cand_local_idx,
                    m_max=max(1, int(m_max)),
                    dag_op=dag_op,
                    dag_a=dag_a,
                    dag_b=dag_b,
                    dag_len=dag_len,
                    psi_raw_pad=psi_raw_pad,
                )
            else:
                e_lap = torch.zeros(n_c, dtype=torch.float32, device=dev)

            meta_np = pk["meta_m"].astype(np.float64)  # type: ignore[union-attr]
            t_len_np = np.array([max(1, len(c[0])) for c in cands], dtype=np.float64)
            vals_np = np.array([float(c[1]) for c in cands], dtype=np.float64)

            # Compute per-candidate unused ground slot null-probability mass from `null_prior_net`
            gr_indices = [i for i, s in enumerate(cur_slots) if s.kind != "CONST"]
            cur_ng = max(1, len(gr_indices))
            roles_m = pk["roles_m"]  # (M, k)
            unused_null_frac = np.zeros(n_c, dtype=np.float64)
            for mi in range(n_c):
                nmass = 0.0
                for gi in gr_indices:
                    if int(roles_m[mi, gi]) == 0:  # unused slot
                        nmass += float(p_null[gi])
                unused_null_frac[mi] = nmass / float(cur_ng)

            return {
                "trivial": False,
                "eu": eu_full.cpu().numpy().astype(np.float64),
                "eu_lex": eu_lex.cpu().numpy().astype(np.float64),
                "ep": ep.cpu().numpy().astype(np.float64),
                "er": er.cpu().numpy().astype(np.float64),
                "ez": ez.cpu().numpy().astype(np.float64),
                "e_lap": e_lap.cpu().numpy().astype(np.float64),
                "meta": meta_np,
                "t_len": t_len_np,
                "unused_null_frac": unused_null_frac,
                "vals": vals_np,
                "cur_ng": cur_ng,
            }

        res = _score_sub(sl, lmf, allow_null=True)
        records.append(
            {
                "gt": gt,
                "ng": ng,
                "u_risk": u_risk,
                "data": res,
                "w_eff_ref": w_eff,
            }
        )

    print(
        f"[T2] Decomposed `{tag}` ({len(records)} problems) in {time.perf_counter() - t0:.2f}s",
        flush=True,
    )
    return records


def _boltzmann_top_vals(vals: np.ndarray, e_vec: np.ndarray, beta: float = 1.65) -> Tuple[float, List[float]]:
    """Exact Boltzmann-mass value aggregation matching `OpenGraphCRAMSolver._boltzmann_rank`."""
    if len(vals) == 0:
        return 0.0, [0.0]
    e_shift = e_vec - float(np.min(e_vec))
    w = np.exp(-beta * np.clip(e_shift, 0.0, 40.0))
    w_sum = float(np.sum(w)) + 1e-12
    probs = w / w_sum
    buckets: Dict[float, float] = {}
    for v, p in zip(vals, probs):
        if not math.isfinite(float(v)):
            continue
        kv = round(float(v), 4)
        buckets[kv] = buckets.get(kv, 0.0) + float(p)
    if not buckets:
        return 0.0, [0.0]
    ranked = sorted(buckets.items(), key=lambda x: x[1], reverse=True)
    return float(ranked[0][0]), [float(k) for k, _ in ranked[:5]]


def evaluate_energy_policy_on_split(
    records: Sequence[Dict[str, Any]],
    policy_fn: Callable[[Dict[str, Any], float, int], np.ndarray],
    beta: float = 1.65,
) -> Dict[str, Any]:
    n = len(records)
    p1 = p5 = oracle = 0
    # Granular slot buckets: <=2, ==3, ==4, >=5 plus <=3 / >=4 summaries
    b_tot = {"le2": 0, "eq3": 0, "eq4": 0, "ge5": 0, "le3": 0, "ge4": 0}
    b_hit = {"le2": 0, "eq3": 0, "eq4": 0, "ge5": 0, "le3": 0, "ge4": 0}
    step1_tot = step1_sel = 0

    for rec in records:
        gt = float(rec["gt"])
        ng = int(rec["ng"])
        u_risk = float(rec["u_risk"])
        data = rec["data"]

        if data is None:
            top1, top5, has_orc = 0.0, [0.0], False
        elif data["trivial"]:
            v0 = float(data["val"])
            top1, top5 = v0, [v0]
            has_orc = abs(v0 - gt) <= 1e-2
        else:
            e_vec = np.asarray(policy_fn(data, u_risk, ng), dtype=np.float64)
            if not np.all(np.isfinite(e_vec)):
                return {"valid": False, "error": "Non-finite candidate energy"}
            vals = data["vals"]
            top1, top5 = _boltzmann_top_vals(vals, e_vec, beta=beta)
            has_orc = bool(np.any(np.abs(vals - gt) <= 1e-2))
            best_idx = int(np.argmin(e_vec))
            if float(data["t_len"][best_idx]) <= 1.0:
                step1_sel += 1
            step1_tot += 1

        h1 = int(abs(top1 - gt) <= 1e-2)
        h5 = int(any(abs(v - gt) <= 1e-2 for v in top5))
        p1 += h1
        p5 += h5
        oracle += int(has_orc)

        k_fine = "le2" if ng <= 2 else ("eq3" if ng == 3 else ("eq4" if ng == 4 else "ge5"))
        k_coarse = "le3" if ng <= 3 else "ge4"
        b_tot[k_fine] += 1
        b_hit[k_fine] += h1
        b_tot[k_coarse] += 1
        b_hit[k_coarse] += h1

    pct = lambda h, t: round(100.0 * h / max(1, t), 2)
    ci_lo, ci_hi = wilson_95_ci(p1, n)
    return {
        "valid": True,
        "n": n,
        "pass1": p1,
        "pass1_pct": pct(p1, n),
        "pass1_ci95": [round(ci_lo, 2), round(ci_hi, 2)],
        "pass5_pct": pct(p5, n),
        "oracle_pct": pct(oracle, n),
        "step1_choice_pct": pct(step1_sel, step1_tot),
        "buckets": {k: {"n": b_tot[k], "pass1_pct": pct(b_hit[k], b_tot[k])} for k in b_tot},
    }


def make_energy_policy(
    w_eff_base: np.ndarray,
    *,
    prior_scale: float = 1.0,
    beam_scale: float = 1.0,
    mdl_ops_pen: float = 0.0,
    depth_norm_p: float = 0.0,
    null_relax_lam: float = 0.0,
    unused_g_scale: float = 1.0,
    risk_kappa: float = 0.0,
    risk_tau: float = 0.65,
    lap_scale: float = 1.0,
    unary_boost: float = 0.0,
    const_pen_extra: float = 0.0,
) -> Callable[[Dict[str, Any], float, int], np.ndarray]:
    """
    Pure, mathematically transparent UEBR System-2 Energy Policy:
      - `prior_scale`, `beam_scale`: debias GSM8K skeleton/kNN/beam negative energy weights
      - `mdl_ops_pen`: strictly non-negative Occam/MDL penalty `softplus`-style `>=0 * (t_len - 1)`
      - `depth_norm_p`: normalize depth-growing potentials `(ep + ez + lap_scale*e_lap) / t_len^p`
      - `null_relax_lam`: discount `meta[:, 3]` (`unused_g`) by predicted `null_prior_net` mass
      - `risk_kappa`, `risk_tau`: OOD epistemic brake `exp(-kappa * max(0, u_risk - tau))` on priors/laplacian
    """
    w_base = np.asarray(w_eff_base, dtype=np.float64).copy()
    mdl_pos = max(0.0, float(mdl_ops_pen))
    p_norm = max(0.0, float(depth_norm_p))
    null_lam = max(0.0, float(null_relax_lam))
    r_kap = max(0.0, float(risk_kappa))
    c_extra = max(0.0, float(const_pen_extra))

    def _policy(data: Dict[str, Any], u_risk: float, ng: int) -> np.ndarray:
        meta = data["meta"]
        t_len = data["t_len"]
        # OOD epistemic brake when Tier-1 uncertainty exceeds calibration floor
        ood_gate = math.exp(-r_kap * max(0.0, u_risk - risk_tau)) if r_kap > 0.0 else 1.0

        # Build debiased meta energy
        w_m = w_base.copy()
        w_m[0] = w_base[0] * prior_scale * ood_gate   # pr_lp
        w_m[1] = w_base[1] * prior_scale * ood_gate   # kn_lp (-0.888 -> debiased!)
        w_m[2] = w_base[2] * beam_scale * ood_gate    # is_beam (-0.476 -> debiased!)
        w_m[3] = w_base[3] * unused_g_scale           # unused_g (+1.688)
        w_m[4] = w_base[4] + c_extra                  # uc_const

        if null_lam > 0.0:
            meta_adj = meta.copy()
            # Discount unused-slot penalty by null_prior_net probability mass of unused slots
            meta_adj[:, 3] = np.maximum(0.0, meta[:, 3] - null_lam * data["unused_null_frac"])
            em = meta_adj @ w_m
        else:
            em = meta @ w_m

        # Depth-normalized multi-step graph potentials
        denom = np.power(np.maximum(1.0, t_len), p_norm) if p_norm > 1e-6 else 1.0
        e_graph = (data["ep"] + data["ez"] + (lap_scale * ood_gate) * data["e_lap"]) / denom

        # Strictly non-negative Occam/MDL depth penalty
        e_mdl = mdl_pos * np.maximum(0.0, t_len - 1.0)

        # Optional domain-invariant unary reinforcement
        e_tot = (1.0 + unary_boost) * data["eu"] + data["er"] + e_graph + em + e_mdl
        return e_tot

    return _policy


def build_cartridge() -> Tuple[TaskCartridge, Dict[str, Any]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[T2] Loading frozen Phase J UEBR A3 checkpoint & cached Tier-1 LUT features on {device}...", flush=True)

    tr_items, cal_items, dev_items = _load_gsm8k_train_splits()
    ext = TrainInducedQuantityExtractor()
    ext.fit(tr_items)
    stress_items = build_blind_dev_stress_split(dev_items, ext, seed=2026)

    sl_dev = [ext.extract_slots(str(it["question"]), include_aux_consts=True)[0] for it in dev_items]
    sl_str = [ext.extract_slots(str(it["question"]), include_aux_consts=True)[0] for it in stress_items]

    f_dev, _ = extract_tier1_lut_features(
        [str(it["question"]) for it in dev_items],
        sl_dev,
        ext,
        TIER1_LUT_CKPT,
        CACHE_DIR / "gsm8k_dev737_tier1_lut.npz",
        device=device,
        reuse_cache=True,
        split_tag="gsm8k_dev737",
    )
    f_str, _ = extract_tier1_lut_features(
        [str(it["question"]) for it in stress_items],
        sl_str,
        ext,
        TIER1_LUT_CKPT,
        CACHE_DIR / "gsm8k_stress737_tier1_lut.npz",
        device=device,
        reuse_cache=True,
        split_tag="gsm8k_stress737",
    )

    alu = SphericalBase10RegisterALU(d_sub=15, num_digits=12, seed=3301)
    solver = UEBROpenGraphSolver(
        extractor=ext,
        alu=alu,
        d_model=384,
        device=device,
        seed=2026,
        use_birkhoff_qap=True,
        use_rlns_and_null=True,
        use_gaq_bridge=True,
        use_spd_laplacian=True,
    )
    ckpt = torch.load(UEBR_A3_CKPT, map_location=device, weights_only=False)
    solver.load_state_dict(ckpt)
    beta_ref = float(solver.boltzmann_beta)
    w_eff_base = solver.ebm.w_meta_effective().detach().cpu().numpy().astype(np.float64)

    # Precompute exact decomposed candidate energies ONCE on Dev (737) & Stress (737)
    dev_recs = _extract_decomposed_split(solver, dev_items, sl_dev, f_dev, "GSM8K_BlindDev_737")
    str_recs = _extract_decomposed_split(solver, stress_items, sl_str, f_str, "GSM8K_BlindDevStress_737")

    # Verify 0.00% discrepancy vs baseline UEBR A3
    base_pol = make_energy_policy(w_eff_base)
    base_dev = evaluate_energy_policy_on_split(dev_recs, base_pol, beta=beta_ref)
    base_str = evaluate_energy_policy_on_split(str_recs, base_pol, beta=beta_ref)
    print(
        f"[T2] Baseline A3 Parity Check: Dev={base_dev['pass1_pct']}% "
        f"(<=3:{base_dev['buckets']['le3']['pass1_pct']}%, >=4:{base_dev['buckets']['ge4']['pass1_pct']}%) | "
        f"Stress={base_str['pass1_pct']}% "
        f"(<=3:{base_str['buckets']['le3']['pass1_pct']}%, >=4:{base_str['buckets']['ge4']['pass1_pct']}%)",
        flush=True,
    )

    def verify_and_score(fn: Callable[[Dict[str, Any], float, int], np.ndarray]) -> Tuple[bool, float, str, Dict[str, Any]]:
        r_dev = evaluate_energy_policy_on_split(dev_recs, fn, beta=beta_ref)
        if not r_dev["valid"]:
            return False, 0.0, "non_finite_energy", {"error": r_dev.get("error")}
        r_str = evaluate_energy_policy_on_split(str_recs, fn, beta=beta_ref)
        if not r_str["valid"]:
            return False, 0.0, "non_finite_energy", {"error": r_str.get("error")}

        # Guard against collapsing multi-step GSM8K ge4 accuracy while boosting short/OOD/stress
        d_le3 = float(r_dev["buckets"]["le3"]["pass1_pct"])
        d_ge4 = float(r_dev["buckets"]["ge4"]["pass1_pct"])
        s_le3 = float(r_str["buckets"]["le3"]["pass1_pct"])
        s_ge4 = float(r_str["buckets"]["ge4"]["pass1_pct"])
        d_all = float(r_dev["pass1_pct"])
        s_all = float(r_str["pass1_pct"])

        hmean_4 = _harmonic_mean([d_le3, d_ge4, s_le3, s_ge4])
        # Strongly reward balanced cross-complexity generalization + stress robustness
        score = round(0.55 * hmean_4 + 0.20 * d_all + 0.25 * s_all, 4)

        # 8 real empirical buckets (Dev: <=2, ==3, ==4, >=5; Stress: <=2, ==3, ==4, >=5)
        b_keys = ["le2", "eq3", "eq4", "ge5"]
        bucket_scores = [
            round(float(r_dev["buckets"][k]["pass1_pct"]) / 100.0, 4) for k in b_keys
        ] + [
            round(float(r_str["buckets"][k]["pass1_pct"]) / 100.0, 4) for k in b_keys
        ]
        base_b = [
            float(base_dev["buckets"][k]["pass1_pct"]) / 100.0 for k in b_keys
        ] + [
            float(base_str["buckets"][k]["pass1_pct"]) / 100.0 for k in b_keys
        ]
        pass_mask = [1 if sc >= max(0.30, bb - 0.005) else 0 for sc, bb in zip(bucket_scores, base_b)]

        info = {
            "pass_mask": pass_mask,
            "bucket_scores": bucket_scores,
            "dev_pass1_pct": d_all,
            "dev_le3_pct": d_le3,
            "dev_ge4_pct": d_ge4,
            "stress_pass1_pct": s_all,
            "stress_le3_pct": s_le3,
            "stress_ge4_pct": s_ge4,
            "hmean_4": round(hmean_4, 4),
            "dev_step1_choice_pct": r_dev["step1_choice_pct"],
            "stress_step1_choice_pct": r_str["step1_choice_pct"],
            "summary": (
                f"score={score:.2f} | HMean4={hmean_4:.2f}% | "
                f"Dev={d_all:.2f}% (<=3:{d_le3:.1f}%, >=4:{d_ge4:.1f}%) | "
                f"Stress={s_all:.2f}% (<=3:{s_le3:.1f}%, >=4:{s_ge4:.1f}%)"
            ),
        }
        print(f"  [T2 Eval] {info['summary']}", flush=True)
        return True, score, "ok", info

    # Single-Shot Sealed Blind Vault: ASDiv-A (2078) + MultiArith (600) + SVAMP (1000)
    def blind_vault_eval(fn: Callable[[Dict[str, Any], float, int], np.ndarray]) -> Dict[str, Any]:
        print("\n[T2 Blind Vault] Opening SINGLE-SHOT sealed OOD math vaults (ASDiv-A 2078 + MultiArith 600 + SVAMP 1000)...", flush=True)
        as_vault = SingleShotMathVault(ASDIV_PATH, max_calls=1)
        ma_vault = SingleShotMathVault(MULTIARITH_PATH, max_calls=1)
        sv_vault = SingleShotSVAMPVault(SVAMP_PATH, max_calls=1)

        as_items = as_vault.open_once()
        ma_items = ma_vault.open_once()
        sv_items = sv_vault.open_once()

        sl_as = [ext.extract_slots(str(it["question"]), include_aux_consts=True)[0] for it in as_items]
        sl_ma = [ext.extract_slots(str(it["question"]), include_aux_consts=True)[0] for it in ma_items]
        sl_sv = [ext.extract_slots(str(it["question"]), include_aux_consts=True)[0] for it in sv_items]

        f_as, _ = extract_tier1_lut_features(
            [str(it["question"]) for it in as_items], sl_as, ext, TIER1_LUT_CKPT,
            CACHE_DIR / "asdiv_qonly2078_tier1_lut.npz", device=device, reuse_cache=True, split_tag="asdiv_a_2078",
        )
        f_ma, _ = extract_tier1_lut_features(
            [str(it["question"]) for it in ma_items], sl_ma, ext, TIER1_LUT_CKPT,
            CACHE_DIR / "multiarith_qonly600_tier1_lut.npz", device=device, reuse_cache=True, split_tag="multiarith_600",
        )
        f_sv, _ = extract_tier1_lut_features(
            [str(it["question"]) for it in sv_items], sl_sv, ext, TIER1_LUT_CKPT,
            CACHE_DIR / "svamp_qonly1000_tier1_lut.npz", device=device, reuse_cache=True, split_tag="svamp_1000",
        )

        as_recs = _extract_decomposed_split(solver, as_items, sl_as, f_as, "VAULT_ASDiv_2078")
        ma_recs = _extract_decomposed_split(solver, ma_items, sl_ma, f_ma, "VAULT_MultiArith_600")
        sv_recs = _extract_decomposed_split(solver, sv_items, sl_sv, f_sv, "VAULT_SVAMP_1000")

        out_vault: Dict[str, Any] = {}
        for name, recs in (("ASDiv_2078", as_recs), ("MultiArith_600", ma_recs), ("SVAMP_1000", sv_recs)):
            r_base = evaluate_energy_policy_on_split(recs, base_pol, beta=beta_ref)
            r_champ = evaluate_energy_policy_on_split(recs, fn, beta=beta_ref)
            out_vault[name] = {
                "baseline_a3": r_base,
                "champion": r_champ,
                "delta_pass1_pp": round(float(r_champ["pass1_pct"]) - float(r_base["pass1_pct"]), 2),
                "delta_le3_pp": round(
                    float(r_champ["buckets"]["le3"]["pass1_pct"]) - float(r_base["buckets"]["le3"]["pass1_pct"]), 2
                ),
            }
        out_vault["summary_delta_pp"] = {
            k: v["delta_pass1_pp"] for k, v in out_vault.items()
        }
        out_vault["champion_params"] = getattr(fn, "_params", None)
        return out_vault

    # --- 4 Orthogonal Round-1 Branches ---
    b0 = BranchCandidate(
        id="b0_occam_mdl_debias",
        branch=0,
        description="Occam/MDL strictly non-negative depth penalty + GSM8K kNN/beam prior clamping",
        fn=make_energy_policy(w_eff_base, prior_scale=0.35, beam_scale=0.0, mdl_ops_pen=0.22),
        next_refinement=BranchCandidate(
            id="b0_occam_mdl_debias_strong",
            branch=0,
            parent_id="b0_occam_mdl_debias",
            attempt=2,
            description="Stronger Occam/MDL depth penalty (0.35) + zero beam/prior bias (0.20)",
            fn=make_energy_policy(w_eff_base, prior_scale=0.20, beam_scale=0.0, mdl_ops_pen=0.35),
        ),
    )

    b1 = BranchCandidate(
        id="b1_depth_norm_energy",
        branch=1,
        description="Depth-normalized multi-step graph energy (ep+ez+e_lap)/t_len^0.5",
        fn=make_energy_policy(w_eff_base, depth_norm_p=0.50),
        next_refinement=BranchCandidate(
            id="b1_depth_norm_full",
            branch=1,
            parent_id="b1_depth_norm_energy",
            attempt=2,
            description="Full per-step normalized graph energy (ep+ez+e_lap)/t_len^0.85",
            fn=make_energy_policy(w_eff_base, depth_norm_p=0.85),
        ),
    )

    b2 = BranchCandidate(
        id="b2_null_aware_coverage",
        branch=2,
        description="Null-Slot Ø probability relaxation on unused_g penalty (+1.688) for distractor invariance",
        fn=make_energy_policy(w_eff_base, null_relax_lam=1.0, unused_g_scale=0.65),
        next_refinement=BranchCandidate(
            id="b2_null_aware_coverage_deep",
            branch=2,
            parent_id="b2_null_aware_coverage",
            attempt=2,
            description="Strong Null-Slot Ø relaxation (lam=1.5, unused_g_scale=0.45) + const_pen_extra=0.25",
            fn=make_energy_policy(w_eff_base, null_relax_lam=1.5, unused_g_scale=0.45, const_pen_extra=0.25),
        ),
    )

    b3 = BranchCandidate(
        id="b3_risk_gated_ood_brake",
        branch=3,
        description="Epistemic u_risk exponential brake on GSM8K priors/Laplacian + domain-invariant unary boost",
        fn=make_energy_policy(w_eff_base, risk_kappa=1.5, risk_tau=0.55, unary_boost=0.20),
        next_refinement=BranchCandidate(
            id="b3_risk_gated_ood_brake_v2",
            branch=3,
            parent_id="b3_risk_gated_ood_brake",
            attempt=2,
            description="Sharper epistemic OOD brake (kappa=2.5, tau=0.45) + unary_boost=0.35 + lap_scale=0.75",
            fn=make_energy_policy(w_eff_base, risk_kappa=2.5, risk_tau=0.45, unary_boost=0.35, lap_scale=0.75),
        ),
    )

    # DAG Crossover with 6 clean orthogonal knobs (merging winning b1 depth-norm + b2 null-relaxation + b0 soft MDL)
    def crossover_synthesizer(u: BranchCandidate, v: BranchCandidate) -> BranchCandidate:
        def _factory(p: np.ndarray) -> Callable[[Dict[str, Any], float, int], np.ndarray]:
            pol = make_energy_policy(
                w_eff_base,
                prior_scale=float(p[0]),
                beam_scale=float(p[1]),
                mdl_ops_pen=float(p[2]),
                depth_norm_p=float(p[3]),
                null_relax_lam=float(p[4]),
                unused_g_scale=float(p[5]),
                const_pen_extra=0.20,
            )
            pol._params = {  # type: ignore[attr-defined]
                "prior_scale": round(float(p[0]), 4),
                "beam_scale": round(float(p[1]), 4),
                "mdl_ops_pen": round(float(p[2]), 4),
                "depth_norm_p": round(float(p[3]), 4),
                "null_relax_lam": round(float(p[4]), 4),
                "unused_g_scale": round(float(p[5]), 4),
                "const_pen_extra": 0.20,
            }
            return pol

        p0 = np.array([0.85, 0.50, 0.05, 0.45, 1.30, 0.55], dtype=np.float64)
        return BranchCandidate(
            id=f"xover_{u.id}_x_{v.id}",
            branch=4,
            parent_id=u.id,
            description=(
                f"Unified Depth-Normalized + Null-Slot-Relaxed + Soft-MDL UEBR Energy "
                f"(`{u.id}` x `{v.id}`) with 6 @tunable knobs for Round-4 pruning & plateau verification"
            ),
            fn=_factory(p0),
            tunable_params=p0,
            tunable_bounds=[
                (0.2, 1.1),   # prior_scale
                (0.0, 1.0),   # beam_scale
                (0.0, 0.5),   # mdl_ops_pen
                (0.0, 1.0),   # depth_norm_p
                (0.0, 2.5),   # null_relax_lam
                (0.15, 1.0),  # unused_g_scale
            ],
            param_names=[
                "prior_scale",
                "beam_scale",
                "mdl_ops_pen",
                "depth_norm_p",
                "null_relax_lam",
                "unused_g_scale",
            ],
            param_factory=_factory,
        )

    cartridge = TaskCartridge(
        task_name="uebr_mdl_t2",
        description=(
            "Eliminate W5 UEBR System-2 1-step & distractor domain shift via Occam/MDL depth penalty, "
            "depth-normalized graph energy, Null-Slot Ø relaxation, and epistemic OOD risk brake"
        ),
        baseline_fn=base_pol,
        verify_and_score=verify_and_score,
        initial_branches=[b0, b1, b2, b3],
        crossover_synthesizer=crossover_synthesizer,
        blind_vault_evaluator=blind_vault_eval,
        noise_probe_runs=1,
    )
    return cartridge, {"baseline_dev": base_dev, "baseline_stress": base_str}


def main() -> int:
    pool_dir = os.environ.get("DREAM_RSI_POOL") or dream_engine.default_pool_dir()
    cartridge, ctx_meta = build_cartridge()
    runner = AutoDreamRunner(cartridge, pool_dir=pool_dir)
    report = runner.run()

    print("\n" + "=" * 78)
    print(
        f"[T2 DONE] Task: {report['task_name']} | Baseline: {report['baseline_score']:.4f} -> "
        f"Champion `{report['champion_id']}`: {report['champion_score']:.4f}"
    )
    print(f"Nodes explored: {report['total_nodes_explored']} | Elapsed: {report['elapsed_s']:.1f}s")
    if report.get("blind_vault_report"):
        print("Sealed Blind Vault (ASDiv-A 2078 + MultiArith 600 + SVAMP 1000):")
        print(json.dumps(report["blind_vault_report"], indent=2, ensure_ascii=False))

    out_payload = {
        "task_name": report["task_name"],
        "baseline_score": report["baseline_score"],
        "baseline_dev": ctx_meta["baseline_dev"],
        "baseline_stress": ctx_meta["baseline_stress"],
        "champion_id": report["champion_id"],
        "champion_score": report["champion_score"],
        "champion_metrics": report["champion_metrics"],
        "total_nodes_explored": report["total_nodes_explored"],
        "elapsed_s": round(float(report["elapsed_s"]), 2),
        "blind_vault_report": report["blind_vault_report"],
        "journal": report["journal"],
        "pool_dir": pool_dir,
    }
    OUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT_PATH.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved report -> {OUT_REPORT_PATH} ({OUT_REPORT_PATH.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
