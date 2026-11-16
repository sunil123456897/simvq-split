"""
system1_encoder.py — Доменный кодер `Calcite-5 System-1` на базе нетронутого `SimVQSplitLM v5`.

Архитектурные инварианты:
  - Импортирует `SimVQSplitLM`, `SimVQSplitConfig`, `quantize_ternary_weights_ls` и
    `pack_model_2bit_lut` из `simvq_split/` без каких-либо правок внутри `simvq_split/`.
  - Принимает вопрос в question-primed формате:
      `Focus: <финальный_вопрос>\\nProblem: <q_text>\\nSlots: A=<v0> B=<v1> ...\\nPlan:`
    чтобы `VoronoiCrystalLayer` и `NeuralWoodburyLayer` держали целевой вопрос в 89.0 KB
    O(1) кэше при чтении каждого числового слота и порождали открытые DAG-планы.
  - Дообучается ТОЛЬКО на `train_split` (0% доступа к `test.jsonl`) чистым `F.cross_entropy`
    + `get_commitment_loss()`, без BPTT по графу поиска, с прогрессивным 1.58-bit STE QAT
    для отдельной LUT-версии (`PackedLUT2BitLinear` uint8 16.0x).
"""

from __future__ import annotations

import math
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from simvq_split.models import SimVQSplitConfig, SimVQSplitLM
from simvq_split.quant_lut import compute_2bit_lut_footprint, pack_model_2bit_lut, quantize_ternary_weights_ls
from simvq_split.tokenizer import ByteBPETokenizer

try:
    from .slot_extractor import ExtractedSlot, ID_TO_OP, OP_TO_ID, TrainInducedQuantityExtractor
except ImportError:
    from slot_extractor import ExtractedSlot, ID_TO_OP, OP_TO_ID, TrainInducedQuantityExtractor


MAX_SLOTS = 20
MAX_INTERMEDIATE = 12


def slot_symbol(idx: int, n_init: int) -> str:
    """Кодирует индекс регистра: `0..n_init-1 -> A..T`, `n_init.. -> a..l`."""
    if idx < n_init:
        return chr(ord("A") + idx)
    return chr(ord("a") + (idx - n_init))


def symbol_to_index(ch: str, n_init: int) -> int:
    code = ord(ch)
    if ord("A") <= code < ord("A") + n_init:
        return code - ord("A")
    if ord("a") <= code < ord("a") + MAX_INTERMEDIATE:
        return n_init + (code - ord("a"))
    return -1


def encode_dag_dsl(steps: Sequence[Tuple[int, int, int]], n_init: int) -> str:
    """Кодирует бинарные шаги DAG в компактный 4-символьный DSL: `B+C;A-a;b*D`."""
    parts: List[str] = []
    for op_id, a_i, b_i in steps:
        if a_i < 0 or b_i < 0 or a_i >= n_init + len(parts) or b_i >= n_init + len(parts):
            return ""
        parts.append(f"{slot_symbol(a_i, n_init)}{ID_TO_OP[op_id]}{slot_symbol(b_i, n_init)}")
    return ";".join(parts)


def parse_dag_dsl(dsl: str, n_init: int) -> Optional[List[Tuple[int, int, int]]]:
    """Обратный декодер DSL `B+C;A-a;b*D` -> `[(op_id, a_i, b_i), ...]`."""
    s = dsl.strip()
    if not s:
        return None
    chunks = s.split(";")
    steps: List[Tuple[int, int, int]] = []
    for step_idx, ch in enumerate(chunks):
        if len(ch) != 3:
            return None
        op_id = OP_TO_ID.get(ch[1], -1)
        a_i = symbol_to_index(ch[0], n_init)
        b_i = symbol_to_index(ch[2], n_init)
        max_valid = n_init + step_idx
        if op_id < 0 or not (0 <= a_i < max_valid) or not (0 <= b_i < max_valid):
            return None
        steps.append((op_id, a_i, b_i))
    return steps


def format_short_val(sl: ExtractedSlot) -> str:
    v = float(sl.val)
    if abs(v - round(v)) < 1e-6:
        return str(int(round(v)))
    return f"{v:.4f}".rstrip("0").rstrip(".")


def build_system1_prompt(
    extractor: TrainInducedQuantityExtractor,
    q_text: str,
    slots: Sequence[ExtractedSlot],
    tok: ByteBPETokenizer,
    post_window: int = 5,
    max_prompt_tokens: int = 240,
) -> Tuple[List[int], List[Tuple[int, int]], List[Tuple[int, int]], int]:
    """Формирует question-primed вход `Focus: ...\\nProblem: ...\\nSlots: ...\\nPlan:`
    с точным выравниванием байтовых спанов слотов внутри `<Problem>`."""
    q_focus = extractor.extract_question_sentence(q_text)
    prefix = f"Focus: {q_focus}\nProblem: "
    prefix_bytes = len(prefix.encode("utf-8"))

    usable_n = min(len(slots), MAX_SLOTS)
    slot_desc = " ".join(f"{slot_symbol(i, usable_n)}={format_short_val(slots[i])}" for i in range(usable_n))
    suffix = f"\nSlots: {slot_desc}\nPlan:"
    full_prompt = prefix + q_text + suffix

    token_ids = tok.encode(full_prompt, add_bos=True, add_eos=False)
    dropped_prob_bytes = 0
    if len(token_ids) > max_prompt_tokens:
        head_ids = tok.encode(prefix, add_bos=True, add_eos=False)
        prob_ids = tok.encode(q_text, add_bos=False, add_eos=False)
        tail_ids = tok.encode(suffix, add_bos=False, add_eos=False)
        keep_prob = max(24, max_prompt_tokens - len(head_ids) - len(tail_ids))
        dropped = prob_ids[: max(0, len(prob_ids) - keep_prob)]
        dropped_prob_bytes = sum(len(tok.id_to_bytes.get(int(t), b"")) for t in dropped)
        token_ids = head_ids + prob_ids[-keep_prob:] + tail_ids

    n_toks = len(token_ids)
    tok_bytes: List[Tuple[int, int]] = []
    b_cur = 0
    for tid in token_ids:
        raw_b = tok.id_to_bytes.get(int(tid), b"")
        tok_bytes.append((b_cur, b_cur + len(raw_b)))
        b_cur += len(raw_b)

    # Позиция финального токена `Plan:` предсказывает 0-й символ открытого DAG
    plan_anchor = max(1, n_toks - 1)

    eff_prefix_bytes = prefix_bytes - dropped_prob_bytes
    prob_end_byte = eff_prefix_bytes + len(q_text.encode("utf-8"))
    num_spans: List[Tuple[int, int]] = []
    post_spans: List[Tuple[int, int]] = []
    for sl in slots[:usable_n]:
        sb0 = eff_prefix_bytes + sl.b0
        sb1 = eff_prefix_bytes + sl.b1
        ov = [
            ti
            for ti, (tb0, tb1) in enumerate(tok_bytes)
            if tb1 > sb0 and tb0 < sb1 and tb0 < prob_end_byte and 0 < ti < plan_anchor
        ]
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


def _byte_token_id(ch: str) -> int:
    return ByteBPETokenizer.BYTE_OFFSET + ord(ch)


class Calcite5MathSystem1:
    """Обёртка доменной адаптации и батчевого инференса `SimVQSplitLM v5` на `train_split`."""

    def __init__(
        self,
        base_ckpt_path: Path,
        device: str = "cuda",
        seed: int = 2026,
    ):
        self.base_ckpt_path = Path(base_ckpt_path)
        self.device = device
        self.seed = int(seed)

        try:
            payload = torch.load(self.base_ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(self.base_ckpt_path, map_location="cpu")
        self.cfg = SimVQSplitConfig(**payload["simvq_split_config"])
        self.tok = ByteBPETokenizer.from_dict(payload["tokenizer"])
        self.model = SimVQSplitLM(self.cfg)
        self.model.load_state_dict(payload["simvq_split_state_dict"], strict=True)
        self.model.to(self.device)
        self.model.set_snap_alpha(1.0)
        self.lut_model: Optional[SimVQSplitLM] = None

    def fit_on_train_split(
        self,
        train_items: Sequence[Dict[str, str]],
        extractor: TrainInducedQuantityExtractor,
        epochs: int = 18,
        batch_size: int = 48,
        lr: float = 1.0e-3,
        qat_steps: int = 240,
        save_ckpt_path: Optional[Path] = None,
    ) -> Dict[str, float]:
        """Дообучает `SimVQSplitLM v5` в BF16 на скомпилированных `(Focus + Problem + Slots) -> Plan`
        примерах из `train_items`, а затем формирует отдельную 1.58-bit STE QAT -> `PackedLUT2BitLinear` копию."""
        t0 = time.perf_counter()
        rng = random.Random(self.seed)
        torch.manual_seed(self.seed)

        samples: List[Tuple[List[int], List[int], int]] = []
        max_len = 280
        for it in train_items:
            q_str = str(it["question"])
            slots, _, _ = extractor.extract_slots(q_str, include_aux_consts=True)
            usable_n = min(len(slots), MAX_SLOTS)
            if usable_n < 1:
                continue
            steps, _, ok = extractor.compile_train_dag(slots[:usable_n], str(it["answer"]))
            if not ok or not steps or len(steps) > 8:
                continue
            dsl = encode_dag_dsl(steps, usable_n)
            if not dsl:
                continue
            p_ids, _, _, _ = build_system1_prompt(extractor, q_str, slots, self.tok, max_prompt_tokens=240)
            plan_ids = [_byte_token_id(ch) for ch in dsl] + [self.tok.EOS_ID]
            seq = p_ids + plan_ids
            if len(seq) > max_len:
                continue
            samples.append((seq[:-1], seq[1:], max(0, len(p_ids) - 1)))

        if not samples:
            raise RuntimeError("No compilable training DAGs found for System-1 domain adaptation!")

        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.02)
        total_steps = max(1, epochs * ((len(samples) + batch_size - 1) // batch_size))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
        use_amp = self.device == "cuda" and torch.cuda.is_available()

        def _make_batch(batch_items: Sequence[Tuple[List[int], List[int], int]]):
            m_len = max(len(x[0]) for x in batch_items)
            xb_l = torch.full((len(batch_items), m_len), self.tok.PAD_ID, dtype=torch.long)
            yb_l = torch.full((len(batch_items), m_len), -100, dtype=torch.long)
            wm_l = torch.zeros((len(batch_items), m_len), dtype=torch.float32)
            for ri, (xs, ys, p_split) in enumerate(batch_items):
                li = len(xs)
                xb_l[ri, :li] = torch.tensor(xs, dtype=torch.long)
                yb_l[ri, :li] = torch.tensor(ys, dtype=torch.long)
                wm_l[ri, :p_split] = 0.02
                wm_l[ri, p_split:li] = 2.50
            return xb_l.to(self.device), yb_l.to(self.device), wm_l.to(self.device)

        step_counter = 0
        last_loss = 0.0
        self.model.set_ternary_qat(False)
        for ep in range(1, epochs + 1):
            self.model.train()
            rng.shuffle(samples)
            ep_loss = 0.0
            ep_n = 0
            for b0 in range(0, len(samples), batch_size):
                batch = samples[b0 : b0 + batch_size]
                xb, yb, w_mask = _make_batch(batch)
                step_counter += 1
                self.model.set_crystal_schedule(min(step_counter, total_steps), total_steps)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    logits = self.model(xb)
                    ce_tok = F.cross_entropy(
                        logits.reshape(-1, self.cfg.vocab_size),
                        yb.reshape(-1),
                        ignore_index=-100,
                        reduction="none",
                    ).view_as(yb)
                    loss = (ce_tok * w_mask).sum() / w_mask.sum().clamp(min=1.0)
                    loss = loss + 0.06 * self.model.get_commitment_loss()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                ep_loss += float(loss.item()) * len(batch)
                ep_n += len(batch)
            last_loss = ep_loss / max(1, ep_n)
            if ep in (1, max(1, epochs // 2), epochs):
                print(f"   [System-1 FP Adapt] ep {ep:2d}/{epochs} | weighted_loss={last_loss:.4f}")

        self.model.eval()
        self.model.set_snap_alpha(1.0)

        # Отдельная прогрессивная 1.58-bit STE QAT-копия -> PackedLUT2BitLinear (`uint8`)
        qat_model = SimVQSplitLM(self.cfg).to(self.device)
        qat_model.load_state_dict({k: v.detach().clone() for k, v in self.model.state_dict().items()}, strict=True)
        qat_model.set_crystal_schedule(total_steps, total_steps)
        qat_model.set_snap_alpha(1.0)
        qat_model.set_qat_mix(0.30)
        qat_model.set_ternary_qat(True)
        q_opt = torch.optim.AdamW(qat_model.parameters(), lr=1.5e-4, weight_decay=0.0)
        for q_step in range(1, qat_steps + 1):
            qat_model.train()
            mix = min(1.0, 0.30 + 0.70 * (q_step / 90.0)) if q_step <= 90 else 1.0
            qat_model.set_qat_mix(mix)
            batch = rng.sample(samples, min(batch_size, len(samples)))
            xb, yb, w_mask = _make_batch(batch)
            q_opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = qat_model(xb)
                ce_tok = F.cross_entropy(
                    logits.reshape(-1, self.cfg.vocab_size),
                    yb.reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view_as(yb)
                loss = (ce_tok * w_mask).sum() / w_mask.sum().clamp(min=1.0)
                loss = loss + 0.04 * qat_model.get_commitment_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qat_model.parameters(), 1.0)
            q_opt.step()
            if q_step in (qat_steps // 2, qat_steps):
                print(f"   [System-1 QAT Tail] step {q_step:3d}/{qat_steps} mix={mix:.2f} | loss={float(loss.item()):.4f}")

        qat_model.set_ternary_qat(False)
        qat_model.set_qat_mix(1.0)
        with torch.no_grad():
            for m_name, mod in qat_model.named_modules():
                if isinstance(mod, nn.Linear) and mod is not qat_model.lm_head and "lm_head" not in m_name:
                    w_tern, _, _ = quantize_ternary_weights_ls(mod.weight.data)
                    mod.weight.data.copy_(w_tern)
        qat_model.eval()
        qat_model.set_snap_alpha(1.0)

        qat_sd = {k: v.detach().cpu() for k, v in qat_model.state_dict().items()}
        qat_clean = SimVQSplitLM(self.cfg)
        qat_clean.load_state_dict(qat_sd, strict=True)
        qat_clean.eval()
        lut_m = pack_model_2bit_lut(qat_clean).to(self.device).eval()
        lut_m.set_snap_alpha(1.0)
        self.lut_model = lut_m
        lut_fp_info = compute_2bit_lut_footprint(lut_m)

        if save_ckpt_path is not None:
            save_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "fp_state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                    "qat_state_dict": qat_sd,
                    "d_model": self.cfg.d_model,
                    "n_samples": len(samples),
                    "final_loss": last_loss,
                    "lut_footprint": lut_fp_info,
                },
                save_ckpt_path,
            )

        return {
            "n_compiled_train": float(len(samples)),
            "final_loss": float(last_loss),
            "lut_total_kb": float(lut_fp_info["total_lut_kb"]),
            "train_seconds": float(time.perf_counter() - t0),
        }

    def load_adapted(self, ckpt_path: Path) -> None:
        try:
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(payload["fp_state_dict"], strict=True)
        self.model.to(self.device).eval()
        self.model.set_snap_alpha(1.0)

        qat_m = SimVQSplitLM(self.cfg)
        qat_m.load_state_dict(payload["qat_state_dict"], strict=True)
        qat_m.eval()
        lut_m = pack_model_2bit_lut(qat_m).to(self.device).eval()
        lut_m.set_snap_alpha(1.0)
        self.lut_model = lut_m

    def _active_model(self, use_lut: bool) -> SimVQSplitLM:
        if use_lut:
            assert self.lut_model is not None, "Call fit_on_train_split or load_adapted before LUT inference"
            return self.lut_model
        return self.model

    @torch.no_grad()
    def extract_features_and_beams(
        self,
        questions: Sequence[str],
        slots_per_q: Sequence[Sequence[ExtractedSlot]],
        extractor: TrainInducedQuantityExtractor,
        batch_size: int = 96,
        post_window: int = 5,
        beam_width: int = 5,
        max_beam_steps: int = 4,
        use_lut: bool = False,
    ) -> List[Dict[str, object]]:
        """Плотный length-sorted батчевый прогон с каузальным правым паддингом."""
        if not questions:
            return []
        m = self._active_model(use_lut=use_lut)
        m.eval()
        m.set_snap_alpha(1.0)
        d_model = int(self.cfg.d_model)
        pad_id = int(self.tok.PAD_ID)

        captured: Dict[str, torch.Tensor] = {}

        def _hook(_mod, _inp, out):
            captured["h_norm"] = out.detach()

        h_handle = m.norm_f.register_forward_hook(_hook)

        prepared = []
        for q_str, slots in zip(questions, slots_per_q):
            t_ids, n_sp, p_sp, q_idx = build_system1_prompt(extractor, q_str, slots, self.tok, post_window=post_window)
            prepared.append((t_ids, n_sp, p_sp, q_idx, min(len(slots), MAX_SLOTS)))

        order = sorted(range(len(questions)), key=lambda i: len(prepared[i][0]))
        op_tids = [_byte_token_id(op) for op in ID_TO_OP]
        semi_tid = _byte_token_id(";")
        eos_tid = int(self.tok.EOS_ID)
        use_amp = self.device == "cuda" and torch.cuda.is_available() and (not use_lut)

        out_list: List[Dict[str, object]] = [{} for _ in range(len(questions))]

        eff_bs = min(batch_size, 32) if use_lut else batch_size
        try:
            for b_start in range(0, len(order), eff_bs):
                b_idxs = order[b_start : b_start + eff_bs]
                lens = [len(prepared[qi][0]) for qi in b_idxs]
                max_len = max(lens)
                xb_rows = [prepared[qi][0] + [pad_id] * (max_len - len(prepared[qi][0])) for qi in b_idxs]
                xb = torch.tensor(xb_rows, dtype=torch.long, device=self.device)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    logits = m(xb)

                h_batch = captured["h_norm"].float().cpu().numpy()
                row_t = torch.arange(len(b_idxs), device=self.device)
                end_t = torch.tensor([l_i - 1 for l_i in lens], dtype=torch.long, device=self.device)
                last_lp = F.log_softmax(logits[row_t, end_t, :].float(), dim=-1).cpu().numpy()
                if m.woodbury is not None and getattr(m.woodbury, "last_epistemic_uncertainty", None) is not None:
                    u_batch = m.woodbury.last_epistemic_uncertainty.float().mean(dim=-1).cpu().numpy()
                else:
                    u_batch = np.zeros((len(b_idxs), max_len), dtype=np.float32)

                for row_i, orig_i in enumerate(b_idxs):
                    t_ids, n_sp, p_sp, q_idx, n_init = prepared[orig_i]
                    l_i = len(t_ids)
                    k_slots = len(n_sp)
                    h_seq = h_batch[row_i, :l_i]
                    u_seq = u_batch[row_i, :l_i]
                    h_q = h_seq[q_idx].astype(np.float32)
                    u_q = float(u_seq[q_idx])
                    if k_slots > 0:
                        h_num = np.stack([h_seq[a:b].mean(axis=0) for a, b in n_sp], axis=0).astype(np.float32)
                        h_post = np.stack([h_seq[a:b].mean(axis=0) for a, b in p_sp], axis=0).astype(np.float32)
                        u_slots = np.array([float(u_seq[a:b].mean()) for a, b in p_sp], dtype=np.float32)
                        u_q = max(u_q, float(np.percentile(u_slots, 75)))
                    else:
                        h_num = np.zeros((0, d_model), dtype=np.float32)
                        h_post = np.zeros((0, d_model), dtype=np.float32)
                        u_slots = np.zeros((0,), dtype=np.float32)

                    out_list[orig_i] = {
                        "h_num": h_num,
                        "h_post": h_post,
                        "h_q": h_q,
                        "u_slots": u_slots,
                        "u_prob": np.float32(u_q),
                        "first_lp": last_lp[row_i],
                        "prompt_ids": t_ids,
                        "n_init": n_init,
                        "beam_plans": [],
                    }

            if beam_width > 0:
                eligible = [qi for qi in order if prepared[qi][4] >= 2]
                self._decode_length_bucket(
                    m,
                    eligible,
                    out_list,
                    op_tids,
                    semi_tid,
                    eos_tid,
                    beam_width,
                    max_beam_steps,
                    use_amp,
                    seq_batch=60 if use_lut else max(batch_size * 2, 192),
                )
        finally:
            h_handle.remove()

        for rec in out_list:
            rec.pop("first_lp", None)
            rec.pop("prompt_ids", None)
        return out_list

    def _decode_length_bucket(
        self,
        m: SimVQSplitLM,
        q_indices: List[int],
        out_list: List[Dict[str, object]],
        op_tids: Sequence[int],
        semi_tid: int,
        eos_tid: int,
        beam_width: int,
        max_steps: int,
        use_amp: bool,
        seq_batch: int = 192,
    ) -> None:
        """Плотный length-sorted constrained beam search с правым паддингом и early-stop отсечением."""
        if not q_indices:
            return
        pad_id = int(self.tok.PAD_ID)
        q_chunk_sz = max(8, seq_batch // max(1, beam_width))
        for chunk_s in range(0, len(q_indices), q_chunk_sz):
            chunk = q_indices[chunk_s : chunk_s + q_chunk_sz]
            beams_map: Dict[int, List[Tuple[float, List[int], List[Tuple[int, int, int]]]]] = {}
            fin_map: Dict[int, List[Tuple[float, List[Tuple[int, int, int]]]]] = {qi: [] for qi in chunk}

            for qi in chunk:
                n_init = int(out_list[qi]["n_init"])
                lp0: np.ndarray = out_list[qi]["first_lp"]  # type: ignore[assignment]
                a_opts = sorted(
                    ((float(lp0[_byte_token_id(slot_symbol(a, n_init))]), a) for a in range(n_init)),
                    reverse=True,
                )[:beam_width]
                beams_map[qi] = [
                    (sc, [_byte_token_id(slot_symbol(a, n_init))], []) for sc, a in a_opts
                ]

            def _batch_step_logits(items_list: List[Tuple[int, List[int]]]) -> np.ndarray:
                raw_seqs = [list(out_list[qi]["prompt_ids"]) + suf for qi, suf in items_list]  # type: ignore[arg-type]
                max_s = max(len(s) for s in raw_seqs)
                padded = [s + [pad_id] * (max_s - len(s)) for s in raw_seqs]
                seqs = torch.tensor(padded, dtype=torch.long, device=self.device)
                row_t = torch.arange(len(raw_seqs), device=self.device)
                end_t = torch.tensor([len(s) - 1 for s in raw_seqs], dtype=torch.long, device=self.device)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    lg = m(seqs)[row_t, end_t, :].float()
                return F.log_softmax(lg, dim=-1).cpu().numpy()

            for st_idx in range(max_steps):
                if st_idx > 0:
                    flat = [(qi, suf) for qi in chunk for _, suf, _ in beams_map[qi]]
                    if not flat:
                        break
                    lps = _batch_step_logits(flat)
                    ptr = 0
                    for qi in chunk:
                        if not beams_map[qi]:
                            continue
                        n_init_q = int(out_list[qi]["n_init"])
                        n_avail = n_init_q + st_idx
                        nxt = []
                        for sc, suf, sts in beams_map[qi]:
                            row_lp = lps[ptr]; ptr += 1
                            top_a = sorted(
                                ((float(row_lp[_byte_token_id(slot_symbol(a, n_init_q))]), a) for a in range(n_avail)),
                                reverse=True,
                            )[:3]
                            for lpa, a in top_a:
                                tid_a = _byte_token_id(slot_symbol(a, n_init_q))
                                nxt.append((sc + lpa, suf + [tid_a], sts))
                        nxt.sort(key=lambda x: x[0], reverse=True)
                        beams_map[qi] = nxt[:beam_width]

                # Оператор
                flat_op = [(qi, suf) for qi in chunk for _, suf, _ in beams_map[qi]]
                if not flat_op:
                    break
                lps_op = _batch_step_logits(flat_op)
                ptr = 0
                op_map: Dict[int, List[Tuple[float, List[int], List[Tuple[int, int, int]], int]]] = {}
                for qi in chunk:
                    nxt = []
                    for sc, suf, sts in beams_map[qi]:
                        row_lp = lps_op[ptr]; ptr += 1
                        top_o = sorted(((float(row_lp[op_tids[o]]), o) for o in range(4)), reverse=True)[:2]
                        for lpo, o in top_o:
                            nxt.append((sc + lpo, suf + [op_tids[o]], sts, o))
                    nxt.sort(key=lambda x: x[0], reverse=True)
                    op_map[qi] = nxt[:beam_width]

                # arg2
                flat_a2 = [(qi, suf) for qi in chunk for _, suf, _, _ in op_map[qi]]
                if not flat_a2:
                    break
                lps_a2 = _batch_step_logits(flat_a2)
                ptr = 0
                a2_map: Dict[int, List[Tuple[float, List[int], List[Tuple[int, int, int]]]]] = {}
                for qi in chunk:
                    n_init = int(out_list[qi]["n_init"])
                    n_avail = n_init + st_idx
                    nxt = []
                    for sc, suf, sts, o in op_map[qi]:
                        row_lp = lps_a2[ptr]; ptr += 1
                        a1_idx = symbol_to_index(chr(suf[-2] - ByteBPETokenizer.BYTE_OFFSET), n_init)
                        top_a2 = sorted(
                            (
                                (float(row_lp[_byte_token_id(slot_symbol(a2, n_init))]), a2)
                                for a2 in range(n_avail)
                                if (
                                    (o in (0, 2) and (a2 > a1_idx or (a2 == a1_idx and a1_idx < n_init)))
                                    or (o in (1, 3) and a2 != a1_idx)
                                )
                            ),
                            reverse=True,
                        )[:3]
                        for lpa2, a2 in top_a2:
                            tid_a2 = _byte_token_id(slot_symbol(a2, n_init))
                            nxt.append((sc + lpa2, suf + [tid_a2], sts + [(o, a1_idx, a2)]))
                    nxt.sort(key=lambda x: x[0], reverse=True)
                    a2_map[qi] = nxt[:beam_width]

                # Конец (`eos`) или продолжение (`;`) с GNMT-нормировкой длины
                flat_end = [(qi, suf) for qi in chunk for _, suf, _ in a2_map[qi]]
                if not flat_end:
                    break
                lps_end = _batch_step_logits(flat_end)
                ptr = 0
                for qi in chunk:
                    nxt = []
                    for sc, suf, sts in a2_map[qi]:
                        row_lp = lps_end[ptr]; ptr += 1
                        lp_norm = math.pow((5.0 + 4.0 * len(sts)) / 9.0, 0.15)
                        fin_sc = (sc + float(row_lp[eos_tid])) / lp_norm
                        fin_map[qi].append((fin_sc, sts))
                        cont_sc = sc + float(row_lp[semi_tid])
                        nxt.append((cont_sc, suf + [semi_tid], sts))
                    nxt.sort(key=lambda x: x[0], reverse=True)
                    beams_map[qi] = nxt[:beam_width]

            for qi in chunk:
                fin = sorted(fin_map[qi], key=lambda x: x[0], reverse=True)
                seen = set()
                uniq = []
                for sc, sts in fin:
                    key = tuple(sts)
                    if key not in seen:
                        seen.add(key)
                        uniq.append((float(sc), key))
                    if len(uniq) >= beam_width:
                        break
                out_list[qi]["beam_plans"] = uniq
