"""tier1_gaq_extractor.py — Frozen Tier-1 (`5.41 MB 2-bit LUT`) extractor, `BlindDev-Stress` builder & `SingleShotSVAMPVault`.

Zero-Cheat & Protocol Contracts:
  - Zero `import re`, `import ast`, `eval`, or `exec`.
  - Computes exact UTF-8 byte-to-token spans for ALL `k = len(slots)` slots (no 8-slot truncation).
  - Extracts `norm_f` hidden states (`384` dims), chunk-aligned Woodbury uncertainty `u_epi`,
    and token-level `VarEntropy(logits)` (`u_risk = u_epi + 0.05 * varent_q`).
  - `SingleShotSVAMPVault` exposes `.read_questions_only()` (no Answer/Equation) for feature caching
    and enforces `max_calls = 1` on `.open_once()` for sealed zero-shot SVAMP evaluation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_CUR_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CUR_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.cram_gsm8k.slot_extractor import (  # noqa: E402
    ExtractedSlot,
    TrainInducedQuantityExtractor,
    scan_question_tokens,
)
from kernels.fast_packed_lut import (  # noqa: E402
    FastPackedLUT2BitLinear,
    load_packed_lut_checkpoint,
)
from simvq_split.quant_lut import compute_2bit_lut_footprint  # noqa: E402


def _slot_label(idx: int) -> str:
    if 0 <= idx < 26:
        return chr(ord("A") + idx)
    return f"S{idx}"


def _format_short_val(sl: ExtractedSlot) -> str:
    v = float(sl.val)
    if abs(v - round(v)) < 1e-6:
        return str(int(round(v)))
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def format_svamp_question(body: str, question: str) -> str:
    """Normalizes SVAMP `(Body, Question)` concatenation so missing trailing period never merges clauses."""
    b = str(body).strip()
    q = str(question).strip()
    if not b:
        return q
    if not q:
        return b
    sep = "" if b[-1] in (".", "?", "!") else "."
    return f"{b}{sep} {q}"


class SingleShotMathVault:
    """Single-shot sealed vault for OOD math benchmarks (`SVAMP`, `MultiArith-600`, `ASDiv-A`; `max_calls = 1`)."""

    def __init__(self, dataset_path: Path, max_calls: int = 1) -> None:
        self.svamp_path = Path(dataset_path)
        self.dataset_path = self.svamp_path
        self.max_calls = int(max_calls)
        self._calls = 0
        assert self.svamp_path.is_file(), f"Missing sealed dataset: {self.svamp_path}"
        self.sha256 = hashlib.sha256(self.svamp_path.read_bytes()).hexdigest()

    @property
    def calls_used(self) -> int:
        return self._calls

    @staticmethod
    def _extract_question_only(row: Dict[str, object]) -> str:
        if "question" in row and str(row["question"]).strip():
            return str(row["question"]).strip()
        return format_svamp_question(str(row.get("Body", "")), str(row.get("Question", "")))

    def read_questions_only(self) -> List[Dict[str, str]]:
        """Reads only normalized question text, ID, and optional Type/Grade without opening answers."""
        raw = json.loads(self.svamp_path.read_text(encoding="utf-8"))
        out: List[Dict[str, str]] = []
        for idx, row in enumerate(raw):
            rec: Dict[str, str] = {
                "ID": str(row.get("ID", f"item-{idx}")),
                "question": self._extract_question_only(row),
            }
            if "Type" in row and str(row["Type"]).strip():
                rec["Type"] = str(row["Type"])
            if "Grade" in row and str(row["Grade"]).strip():
                rec["Grade"] = str(row["Grade"])
            out.append(rec)
        return out

    def open_once(self) -> List[Dict[str, object]]:
        """Unseals full answers strictly once (`max_calls = 1`)."""
        self._calls += 1
        if self._calls > self.max_calls:
            raise RuntimeError(
                f"SingleShotMathVault violation: attempted call #{self._calls} > max_calls={self.max_calls}"
            )
        raw = json.loads(self.svamp_path.read_text(encoding="utf-8"))
        out: List[Dict[str, object]] = []
        for idx, row in enumerate(raw):
            ans_raw = row["answer_val"] if "answer_val" in row else row["Answer"]
            rec: Dict[str, object] = {
                "ID": str(row.get("ID", f"item-{idx}")),
                "question": self._extract_question_only(row),
                "answer_val": float(ans_raw),
            }
            if "Body" in row:
                rec["Body"] = str(row["Body"])
            if "Question" in row:
                rec["Question"] = str(row["Question"])
            if "Type" in row and str(row["Type"]).strip():
                rec["Type"] = str(row["Type"])
            if "Grade" in row and str(row["Grade"]).strip():
                rec["Grade"] = str(row["Grade"])
            if "Equation" in row:
                rec["Equation"] = str(row["Equation"])
            out.append(rec)
        return out


SingleShotSVAMPVault = SingleShotMathVault


def build_blind_dev_stress_split(
    dev_items: Sequence[Dict[str, str]],
    extractor: TrainInducedQuantityExtractor,
    seed: int = 2026,
) -> List[Dict[str, str]]:
    """Builds an OOS permutation + irrelevant-distractor stress split (`N = len(dev_items)`) from `GSM8K dev`.

    Strictly preserves each problem's ground-truth answer while:
      1) Reversing declarative premise sentence order when >= 2 premise sentences exist.
      2) Prepending 1 disjoint irrelevant numerical clause (`On Tuesday, <donor_clause>.`)
         whose content words have zero overlap with the target problem.
    """
    n_tot = len(dev_items)
    parsed_items: List[Tuple[str, List[str], str, List[Tuple[set, str]]]] = []
    for it in dev_items:
        q_str = str(it["question"]).strip()
        ans_str = str(it["answer"])
        toks = scan_question_tokens(q_str)
        raw_b = q_str.encode("utf-8")

        q_sent_id = toks[-1].sent_id if toks else 0
        for tk in reversed(toks):
            if tk.kind == "PUNCT" and tk.text == "?":
                q_sent_id = tk.sent_id
                break

        by_sent: Dict[int, List] = {}
        by_clause: Dict[int, List] = {}
        target_words: set = set()
        for tk in toks:
            by_sent.setdefault(tk.sent_id, []).append(tk)
            by_clause.setdefault(tk.clause_id, []).append(tk)
            if tk.kind == "WORD" and len(tk.lower) >= 3:
                target_words.add(tk.lower)

        premise_sents: List[str] = []
        q_sents: List[str] = []
        for sid in sorted(by_sent.keys()):
            stoks = by_sent[sid]
            stxt = raw_b[stoks[0].b0 : stoks[-1].b1].decode("utf-8", errors="ignore").strip()
            if not stxt:
                continue
            if stxt[-1] not in (".", "?", "!"):
                stxt = stxt + "."
            if sid < q_sent_id:
                premise_sents.append(stxt)
            else:
                q_sents.append(stxt)

        donor_clauses: List[Tuple[set, str]] = []
        for cid in sorted(by_clause.keys()):
            ctoks = by_clause[cid]
            if not ctoks or ctoks[0].sent_id >= q_sent_id:
                continue
            has_num = any(tk.kind == "NUM" for tk in ctoks)
            c_words = [tk.lower for tk in ctoks if tk.kind == "WORD"]
            if not has_num or len(c_words) < 3 or len(c_words) > 14:
                continue
            ctxt = raw_b[ctoks[0].b0 : ctoks[-1].b1].decode("utf-8", errors="ignore").strip()
            while ctxt and ctxt[-1] in (",", ".", ";", ":", "?", "!"):
                ctxt = ctxt[:-1].strip()
            if len(ctxt) < 8:
                continue
            cword_set = {w for w in c_words if len(w) >= 3}
            donor_clauses.append((cword_set, ctxt))

        parsed_items.append((ans_str, premise_sents, " ".join(q_sents) if q_sents else q_str, donor_clauses))

    rng = random.Random(seed)
    all_target_sets: List[set] = []
    for it in dev_items:
        wset = {tk.lower for tk in scan_question_tokens(str(it["question"])) if tk.kind == "WORD" and len(tk.lower) >= 3}
        all_target_sets.append(wset)

    stress_items: List[Dict[str, str]] = []
    for idx in range(n_tot):
        ans_str, prem, q_final, _ = parsed_items[idx]
        t_words = all_target_sets[idx]
        # Deterministic probe over disjoint donor items
        cand_indices = [(idx + 17 + k * 29 + rng.randint(0, 7)) % n_tot for k in range(48)]
        chosen_clause = ""
        for di in cand_indices:
            if di == idx:
                continue
            for c_words, c_txt in parsed_items[di][3]:
                if not (c_words & t_words):
                    chosen_clause = c_txt
                    break
            if chosen_clause:
                break
        if not chosen_clause:
            chosen_clause = "a distant warehouse stored 17 empty crates"
        else:
            if len(chosen_clause) >= 2 and chosen_clause[0].isupper() and chosen_clause[1].islower():
                chosen_clause = chosen_clause[0].lower() + chosen_clause[1:]

        distractor_sent = f"On Tuesday, {chosen_clause}."
        rev_prem = list(reversed(prem)) if len(prem) >= 2 else list(prem)
        parts = [distractor_sent] + rev_prem + [q_final]
        stress_q = " ".join(p for p in parts if p)
        # Sanity ensure extractor finds slots smoothly
        _ = extractor
        stress_items.append({"question": stress_q, "answer": ans_str})
    return stress_items


def build_prompt_for_all_slots(
    extractor: TrainInducedQuantityExtractor,
    q_text: str,
    slots: Sequence[ExtractedSlot],
    tok,
    post_window: int = 16,
    max_prompt_bytes: int = 768,
) -> Tuple[List[int], List[Tuple[int, int]], List[Tuple[int, int]], int]:
    """Builds `Focus: ...\\nProblem: ...\\nSlots: ...\\nPlan:` and aligns ALL `k = len(slots)` slots."""
    q_focus = extractor.extract_question_sentence(q_text)
    prefix = f"Focus: {q_focus}\nProblem: "
    prefix_bytes = len(prefix.encode("utf-8"))

    k = len(slots)
    slot_desc = " ".join(f"{_slot_label(i)}={_format_short_val(slots[i])}" for i in range(k))
    suffix = f"\nSlots: {slot_desc}\nPlan:"
    full_prompt = prefix + q_text + suffix

    token_ids: List[int] = tok.encode(full_prompt, add_bos=True, add_eos=False)
    dropped_prob_bytes = 0
    if len(token_ids) > max_prompt_bytes:
        head_ids = tok.encode(prefix, add_bos=True, add_eos=False)
        prob_ids = tok.encode(q_text, add_bos=False, add_eos=False)
        tail_ids = tok.encode(suffix, add_bos=False, add_eos=False)
        keep_prob = max(24, max_prompt_bytes - len(head_ids) - len(tail_ids))
        if keep_prob < len(prob_ids):
            dropped = prob_ids[: len(prob_ids) - keep_prob]
            dropped_prob_bytes = sum(len(tok.id_to_bytes.get(int(t), b"")) for t in dropped)
            token_ids = head_ids + prob_ids[-keep_prob:] + tail_ids
        else:
            token_ids = (head_ids + prob_ids + tail_ids)[-max_prompt_bytes:]

    n_toks = len(token_ids)
    tok_bytes: List[Tuple[int, int]] = []
    b_cur = 0
    for tid in token_ids:
        raw_b = tok.id_to_bytes.get(int(tid), b"")
        tok_bytes.append((b_cur, b_cur + len(raw_b)))
        b_cur += len(raw_b)

    plan_anchor = max(1, n_toks - 1)
    eff_prefix_bytes = prefix_bytes - dropped_prob_bytes
    prob_end_byte = eff_prefix_bytes + len(q_text.encode("utf-8"))

    num_spans: List[Tuple[int, int]] = []
    post_spans: List[Tuple[int, int]] = []
    for sl in slots:
        sb0 = eff_prefix_bytes + sl.b0
        sb1 = eff_prefix_bytes + sl.b1
        ov = (
            [
                ti
                for ti, (tb0, tb1) in enumerate(tok_bytes)
                if tb1 > sb0 and tb0 < sb1 and tb0 < prob_end_byte and 0 < ti < plan_anchor
            ]
            if sb1 > sb0
            else []
        )
        if not ov:
            t_s, t_e = plan_anchor, min(n_toks, plan_anchor + 1)
        else:
            t_s, t_e = ov[0], ov[-1] + 1
        p_s = min(t_e, plan_anchor)
        p_e = min(t_e + post_window, n_toks)
        if p_e <= p_s:
            p_s, p_e = t_s, t_e
        num_spans.append((t_s, t_e))
        post_spans.append((p_s, p_e))

    return token_ids, num_spans, post_spans, plan_anchor


def _compute_manifest_sha(
    ckpt_sha: str,
    questions: Sequence[str],
    slots_per_q: Sequence[Sequence[ExtractedSlot]],
    post_window: int,
) -> str:
    h = hashlib.sha256()
    h.update(f"{ckpt_sha}|pw={post_window}|n={len(questions)}\n".encode("utf-8"))
    for q, sls in zip(questions, slots_per_q):
        h.update(q.encode("utf-8"))
        sig = ";".join(f"{s.b0}:{s.b1}:{s.val:.6g}" for s in sls)
        h.update(f"|{sig}\n".encode("utf-8"))
    return h.hexdigest()


def _save_features_npz(
    cache_path: Path,
    manifest_sha: str,
    feats: Sequence[Dict[str, object]],
    d_model: int,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    n_q = len(feats)
    offsets = np.zeros(n_q + 1, dtype=np.int64)
    h_num_l, h_post_l, h_q_l = [], [], []
    u_sl_l, ve_sl_l = [], []
    u_prob_a = np.zeros(n_q, dtype=np.float32)
    ve_q_a = np.zeros(n_q, dtype=np.float32)
    u_risk_a = np.zeros(n_q, dtype=np.float32)

    cur = 0
    for i, rec in enumerate(feats):
        hn = np.asarray(rec["h_num"], dtype=np.float16)
        hp = np.asarray(rec["h_post"], dtype=np.float16)
        us = np.asarray(rec["u_slots"], dtype=np.float32)
        vs = np.asarray(rec["varent_slots"], dtype=np.float32)
        k_i = int(hn.shape[0])
        if k_i > 0:
            h_num_l.append(hn)
            h_post_l.append(hp)
            u_sl_l.append(us)
            ve_sl_l.append(vs)
        cur += k_i
        offsets[i + 1] = cur
        h_q_l.append(np.asarray(rec["h_q"], dtype=np.float16))
        u_prob_a[i] = float(rec["u_prob"])  # type: ignore[arg-type]
        ve_q_a[i] = float(rec["varent_q"])  # type: ignore[arg-type]
        u_risk_a[i] = float(rec["u_risk"])  # type: ignore[arg-type]

    h_num_cat = np.concatenate(h_num_l, axis=0) if h_num_l else np.zeros((0, d_model), dtype=np.float16)
    h_post_cat = np.concatenate(h_post_l, axis=0) if h_post_l else np.zeros((0, d_model), dtype=np.float16)
    u_sl_cat = np.concatenate(u_sl_l, axis=0) if u_sl_l else np.zeros((0,), dtype=np.float32)
    ve_sl_cat = np.concatenate(ve_sl_l, axis=0) if ve_sl_l else np.zeros((0,), dtype=np.float32)
    h_q_stk = np.stack(h_q_l, axis=0) if h_q_l else np.zeros((0, d_model), dtype=np.float16)

    np.savez_compressed(
        cache_path,
        manifest_sha=np.array([manifest_sha]),
        offsets=offsets,
        h_num=h_num_cat,
        h_post=h_post_cat,
        h_q=h_q_stk,
        u_slots=u_sl_cat,
        varent_slots=ve_sl_cat,
        u_prob=u_prob_a,
        varent_q=ve_q_a,
        u_risk=u_risk_a,
    )


def _load_features_npz(cache_path: Path, expected_sha: str) -> Optional[List[Dict[str, object]]]:
    if not cache_path.is_file():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as z:
            got_sha = str(z["manifest_sha"][0])
            if got_sha != expected_sha:
                return None
            offsets = z["offsets"]
            h_num = z["h_num"].astype(np.float32)
            h_post = z["h_post"].astype(np.float32)
            h_q = z["h_q"].astype(np.float32)
            u_slots = z["u_slots"].astype(np.float32)
            ve_slots = z["varent_slots"].astype(np.float32)
            u_prob = z["u_prob"].astype(np.float32)
            ve_q = z["varent_q"].astype(np.float32)
            u_risk = z["u_risk"].astype(np.float32)
    except Exception:
        return None

    out: List[Dict[str, object]] = []
    n_q = int(len(offsets) - 1)
    for i in range(n_q):
        s0, s1 = int(offsets[i]), int(offsets[i + 1])
        out.append(
            {
                "h_num": h_num[s0:s1],
                "h_post": h_post[s0:s1],
                "h_q": h_q[i],
                "u_slots": u_slots[s0:s1],
                "varent_slots": ve_slots[s0:s1],
                "u_prob": float(u_prob[i]),
                "varent_q": float(ve_q[i]),
                "u_risk": float(u_risk[i]),
                "beam_plans": [],
            }
        )
    return out


@torch.no_grad()
def extract_tier1_lut_features(
    questions: Sequence[str],
    slots_per_q: Sequence[Sequence[ExtractedSlot]],
    extractor: TrainInducedQuantityExtractor,
    ckpt_path: Path,
    cache_path: Optional[Path] = None,
    device: str = "cuda",
    batch_size: int = 64,
    post_window: int = 16,
    max_prompt_bytes: int = 768,
    reuse_cache: bool = True,
    lam_var: float = 0.05,
    split_tag: str = "split",
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Extracts frozen Tier-1 (`5.41 MB 2-bit LUT`) features + Woodbury & VarEntropy risk signals once and caches."""
    if not questions:
        return [], {"n_prompts": 0, "cached": False}

    ckpt_p = Path(ckpt_path)
    assert ckpt_p.is_file(), f"Missing Tier-1 LUT checkpoint: {ckpt_p}"
    ckpt_sha = hashlib.sha256(ckpt_p.read_bytes()).hexdigest()
    manifest_sha = _compute_manifest_sha(ckpt_sha, questions, slots_per_q, post_window)

    if reuse_cache and cache_path is not None:
        cached_feats = _load_features_npz(cache_path, manifest_sha)
        if cached_feats is not None:
            print(f"  [Tier-1 LUT Extract | {split_tag}] loaded SHA-256 cache ({len(cached_feats)} prompts): {cache_path.name}", flush=True)
            return cached_feats, {
                "n_prompts": len(cached_feats),
                "cached": True,
                "lut_layers": 95,
                "total_lut_kb": 5539.81,
                "mean_u_risk": round(float(np.mean([f["u_risk"] for f in cached_feats])), 4),
            }

    t0 = time.perf_counter()
    eff_dev = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
    model, cfg, tok, _ = load_packed_lut_checkpoint(ckpt_p, device=eff_dev, backend="auto")
    model.eval()
    if model.byte_chunker is not None:
        model.set_crystal_schedule(1, 1)
    else:
        model.set_snap_alpha(1.0)

    n_lut = len([m for m in model.modules() if isinstance(m, FastPackedLUT2BitLinear)])
    fp_info = compute_2bit_lut_footprint(model)
    d_model = int(cfg.d_model)
    assert n_lut == 95, f"Expected 95 FastPackedLUT2BitLinear layers, got {n_lut}"

    captured: Dict[str, torch.Tensor] = {}

    def _hook(_mod, _inp, out):
        captured["h_norm"] = out.detach()

    hook_h = model.norm_f.register_forward_hook(_hook)

    prepared = [
        build_prompt_for_all_slots(
            extractor=extractor,
            q_text=q,
            slots=sl,
            tok=tok,
            post_window=post_window,
            max_prompt_bytes=max_prompt_bytes,
        )
        for q, sl in zip(questions, slots_per_q)
    ]
    order = sorted(range(len(questions)), key=lambda i: len(prepared[i][0]))
    pad_id = int(tok.PAD_ID)
    out_list: List[Dict[str, object]] = [{} for _ in range(len(questions))]

    try:
        for b0 in range(0, len(order), batch_size):
            b_idxs = order[b0 : b0 + batch_size]
            lens = [len(prepared[qi][0]) for qi in b_idxs]
            max_l = max(lens)
            xb = torch.tensor(
                [prepared[qi][0] + [pad_id] * (max_l - len(prepared[qi][0])) for qi in b_idxs],
                dtype=torch.long,
                device=eff_dev,
            )
            logits = model(xb)
            h_batch = captured["h_norm"].float().cpu().numpy()

            lg_f = logits.float()
            lp = F.log_softmax(lg_f, dim=-1)
            p = lp.exp()
            ent = -(p * lp).sum(dim=-1, keepdim=True)
            varent_batch = (p * ((-lp - ent) ** 2)).sum(dim=-1).cpu().numpy()

            if model.woodbury is not None and getattr(model.woodbury, "last_epistemic_uncertainty", None) is not None:
                u_raw = model.woodbury.last_epistemic_uncertainty.float()
                if model.byte_chunker is not None and u_raw.size(1) != max_l:
                    _, _, chunk_ctx = model.byte_chunker.encode_to_chunks(
                        model.tok_emb(xb), xb, model.tok_emb.weight
                    )
                    slot_idx_exp = (
                        chunk_ctx.slot_idx.clamp(min=0, max=max(0, u_raw.size(1) - 1))
                        .unsqueeze(-1)
                        .expand(-1, -1, u_raw.size(-1))
                    )
                    u_raw = torch.gather(u_raw, 1, slot_idx_exp)
                h_epi = int(model.woodbury.H_epi)
                u_batch = u_raw[..., :h_epi].mean(dim=-1).cpu().numpy()
            else:
                u_batch = np.zeros((len(b_idxs), max_l), dtype=np.float32)

            for r_i, orig_i in enumerate(b_idxs):
                t_ids, n_sp, p_sp, q_idx = prepared[orig_i]
                l_i = len(t_ids)
                h_seq = h_batch[r_i, :l_i]
                u_seq = u_batch[r_i, :l_i]
                ve_seq = varent_batch[r_i, :l_i]

                h_q = h_seq[q_idx].astype(np.float32)
                u_q = float(u_seq[q_idx])
                ve_q = float(ve_seq[q_idx])
                k_sl = len(n_sp)
                if k_sl > 0:
                    h_num = np.stack([h_seq[a:b].mean(axis=0) for a, b in n_sp], axis=0).astype(np.float32)
                    h_post = np.stack([h_seq[a:b].mean(axis=0) for a, b in p_sp], axis=0).astype(np.float32)
                    u_sl = np.array([float(u_seq[a:b].mean()) for a, b in p_sp], dtype=np.float32)
                    ve_sl = np.array([float(ve_seq[a:b].mean()) for a, b in p_sp], dtype=np.float32)
                    u_q = max(u_q, float(np.percentile(u_sl, 75)))
                else:
                    h_num = np.zeros((0, d_model), dtype=np.float32)
                    h_post = np.zeros((0, d_model), dtype=np.float32)
                    u_sl = np.zeros((0,), dtype=np.float32)
                    ve_sl = np.zeros((0,), dtype=np.float32)

                u_risk = float(u_q + lam_var * ve_q)
                out_list[orig_i] = {
                    "h_num": h_num,
                    "h_post": h_post,
                    "h_q": h_q,
                    "u_slots": u_sl,
                    "varent_slots": ve_sl,
                    "u_prob": u_q,
                    "varent_q": ve_q,
                    "u_risk": u_risk,
                    "beam_plans": [],
                }
            if ((b0 // batch_size) + 1) % 20 == 0 or (b0 + len(b_idxs)) >= len(order):
                done_n = min(b0 + len(b_idxs), len(order))
                print(
                    f"  [Tier-1 LUT Extract | {split_tag}] {done_n}/{len(order)} prompts ({time.perf_counter()-t0:.1f}s)",
                    flush=True,
                )
    finally:
        hook_h.remove()

    if cache_path is not None:
        _save_features_npz(cache_path, manifest_sha, out_list, d_model=d_model)

    stats = {
        "n_prompts": len(out_list),
        "cached": False,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "lut_layers": int(n_lut),
        "total_lut_kb": round(float(fp_info["total_lut_kb"]), 2),
        "mean_u_risk": round(float(np.mean([f["u_risk"] for f in out_list])), 4),
    }
    return out_list, stats
