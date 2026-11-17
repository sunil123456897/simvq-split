"""
scripts/colab_phase_h_qat_runner.py — Исполняющий конвейер Фазы H (Попытка №2: OEB 2/2)
в Google Colab (`A100`) и локальной верификации (`--smoke-local` на `RTX 5060`):

Закрывает два остаточных разрыва `Calcite-5 Tier-1` (`17,488,961` параметров, `5,539.81 KB` = `5.41 MB`
`2-bit LUT`, `95` слоёв `16.00x`, потоковый буфер `2,164 B`, кэш `O(1)` = `230.11 KB` @ `L=2048`)
без единого байта правок в защищённых `simvq_split/` и `experiments/cram_gsm8k/`:

Исправления Попытки №2 против Попытки №1:
1. Стратифицированный батч (`B=64, L=1024`) включает все 6 строго непересекающихся тренировочных
   доменов `train_wins` (`EN/RU/Code/ARC_JSON/CJK/Emoji_4Byte`) + поток `train_utf8.u16.bin` +
   когнитивные эпизоды (`S_5/Overwrite/Trap` только на `tr_sents`, `k<=4`, `0` OOD-утечки).
2. 4-групповой `AdamW` для `STE QAT`: отдельный быстрый `LR = 1.2e-3 -> 1.5e-4` для `FP16`-шкал
   `RMSNorm.weight`, вентилей `stream_gate/cryst2_gate` и `tok_emb`, компенсирующий кубическую
   усадку амплитуды `SwiGLU` (`0.84^3`), плюс `LR = 4.5e-4 -> 6.0e-5` для тернарных ядер.
3. 4-точечная поблочная дистилляция `(cortex_pre[-1], mid_mlp, cortex_post[-1], norm_f)` по амплитуде
   (`smooth_l1` относительно `rms_fp`) и направлению (`1 - cos`) вместе с `CE + Logit KL`.
4. `Stage 3` (`CRAM+z`, `B=64`, `1120` шагов QAT) с парным `solver_lut.fit_bptt_free(train[:6000], f_tr_lut)`.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

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
    _byte_token_id,
    build_system1_prompt,
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
from scripts.colab_phase_g_runner import (  # noqa: E402
    TIER1_CONFIG_DICT,
    _sample_memmap_batch,
    eval_val_memmap_bpb,
    run_stage_build_corpus,
    tier1_config,
)
from simvq_split.baselines import RMSNorm  # noqa: E402
from simvq_split.memory import compute_chunker_state_bytes  # noqa: E402
from simvq_split.models import SimVQSplitConfig, SimVQSplitLM  # noqa: E402
from simvq_split.quant_lut import (  # noqa: E402
    compute_2bit_lut_footprint,
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


# ---------------------------------------------------------------------------
# Guardrails, Router Snapshots & Multi-Tap Block Capture
# ---------------------------------------------------------------------------


def _sync_if_cuda(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _snapshot_router_params(model: SimVQSplitLM) -> Dict[str, torch.Tensor]:
    snap: Dict[str, torch.Tensor] = {}
    if model.byte_chunker is None:
        return snap
    for nm, p in model.byte_chunker.named_parameters():
        if nm.startswith("router_"):
            snap[nm] = p.detach().cpu().clone()
    return snap


def _assert_router_unchanged(model: SimVQSplitLM, snap: Dict[str, torch.Tensor], tag: str) -> float:
    if model.byte_chunker is None or not snap:
        return 0.0
    max_diff = 0.0
    for nm, p in model.byte_chunker.named_parameters():
        if nm in snap:
            diff = float((p.detach().cpu() - snap[nm]).abs().max().item())
            max_diff = max(max_diff, diff)
            assert diff == 0.0, f"[{tag}] Locked router parameter `{nm}` mutated by {diff:.3e}!"
    return max_diff


def _attach_tap_hooks(model: SimVQSplitLM, store: List[torch.Tensor], detach: bool) -> list:
    tap_mods = (model.cortex_pre[-1], model.mid_mlp, model.cortex_post[-1], model.norm_f)
    handles = []
    for mod in tap_mods:
        def _hk(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            store.append(t.detach() if detach else t)
        handles.append(mod.register_forward_hook(_hk))
    return handles


def _compute_tap_loss(
    taps_fp: Sequence[torch.Tensor],
    taps_q: Sequence[torch.Tensor],
    last_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert len(taps_fp) == len(taps_q) and len(taps_q) == 4
    acc = torch.zeros((), device=taps_q[0].device, dtype=torch.float32)
    n_taps = len(taps_q)
    for idx, (tf, tq) in enumerate(zip(taps_fp, taps_q)):
        af = tf.float()
        aq = tq.float()
        if last_mask is not None and idx == n_taps - 1:
            m = last_mask.reshape(-1) > 0.5
            if int(m.sum().item()) == 0:
                continue
            af_flat = af.reshape(-1, af.size(-1))[m]
            aq_flat = aq.reshape(-1, aq.size(-1))[m]
        else:
            af_flat = af.reshape(-1, af.size(-1))
            aq_flat = aq.reshape(-1, aq.size(-1))
        rms_f = af_flat.pow(2).mean(dim=-1, keepdim=True).add(1e-5).sqrt().detach()
        sl1 = F.smooth_l1_loss(aq_flat / rms_f, af_flat / rms_f)
        cos_d = (1.0 - F.cosine_similarity(af_flat, aq_flat, dim=-1, eps=1e-6)).mean()
        acc = acc + (0.5 * sl1 + 0.5 * cos_d)
    return acc / float(n_taps)


def _measure_block_alignment(
    model_fp: SimVQSplitLM,
    model_qat: SimVQSplitLM,
    device: str,
) -> Dict[str, Dict[str, float]]:
    """Измеряет поблочное выравнивание (`cos` и отношение `std`) между FP и QAT моделями."""
    torch.manual_seed(0)
    xb = torch.randint(3, model_fp.cfg.vocab_size, (4, 256), device=device)
    use_amp = device == "cuda" and torch.cuda.is_available()

    def _capture(m: SimVQSplitLM) -> Dict[str, torch.Tensor]:
        outs: Dict[str, torch.Tensor] = {}
        watch: List[Tuple[str, nn.Module]] = []
        if m.byte_chunker is not None:
            watch.append(("enc_mlp", m.byte_chunker.enc_mlp))
        watch.extend(
            [
                ("pre3", m.cortex_pre[-1]),
                ("mid_mlp", m.mid_mlp),
                ("post0", m.cortex_post[0]),
                ("norm_f", m.norm_f),
            ]
        )
        handles = []
        for bname, mod in watch:
            def _make_hk(k: str):
                def _hk(_mod, _inp, out):
                    t = out[0] if isinstance(out, tuple) else out
                    outs[k] = t.detach().float()
                return _hk
            handles.append(mod.register_forward_hook(_make_hk(bname)))
        prev_mode = m.training
        m.eval()
        try:
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    _ = m(xb)
        finally:
            for h in handles:
                h.remove()
            m.train(prev_mode)
        return outs

    fp_outs = _capture(model_fp)
    qat_outs = _capture(model_qat)
    res: Dict[str, Dict[str, float]] = {}
    for k, ft in fp_outs.items():
        qt = qat_outs[k]
        ff = ft.reshape(-1, ft.size(-1))
        qf = qt.reshape(-1, qt.size(-1))
        cos_v = float(F.cosine_similarity(ff, qf, dim=-1, eps=1e-6).mean().item())
        sf = float(ft.std().item())
        sq = float(qt.std().item())
        res[k] = {
            "cos": round(cos_v, 4),
            "std_f": round(sf, 4),
            "std_q": round(sq, 4),
            "std_ratio": round(sq / max(1e-6, sf), 4),
        }
    return res


def check_gate2_block_std_parity(model_fp: SimVQSplitLM, device: str) -> Dict[str, object]:
    """Gate 2: проверяет сохранение масштаба `std` по блокам и нулевой сдвиг при `qat_mix=0`."""
    torch.manual_seed(2026)
    x_probe = torch.randint(3, model_fp.cfg.vocab_size, (4, 192), device=device)

    def _capture_stds(m: SimVQSplitLM) -> Tuple[torch.Tensor, Dict[str, float]]:
        stds: Dict[str, float] = {}
        hooks = []
        watch: List[Tuple[str, nn.Module]] = [
            ("cortex_pre", m.cortex_pre[-1]),
            ("mid_mlp", m.mid_mlp),
            ("cortex_post", m.cortex_post[-1]),
            ("norm_f", m.norm_f),
        ]
        if m.crystal is not None:
            watch.append(("crystal", m.crystal))
        if m.woodbury is not None:
            watch.append(("woodbury", m.woodbury))
        for bname, mod in watch:
            def _make_hk(k: str):
                def _hk(_mod, _inp, out):
                    t = out[0] if isinstance(out, tuple) else out
                    assert torch.isfinite(t).all(), f"Non-finite activation in block `{k}`!"
                    stds[k] = float(t.detach().float().std().item())
                return _hk
            hooks.append(mod.register_forward_hook(_make_hk(bname)))
        try:
            with torch.no_grad():
                logits = m(x_probe)
            assert torch.isfinite(logits).all(), "Non-finite logits in Gate 2!"
        finally:
            for h in hooks:
                h.remove()
        return logits.detach(), stds

    model_fp.eval()
    lg_fp, std_fp = _capture_stds(model_fp)

    qat_m = copy.deepcopy(model_fp).to(device).eval()
    if qat_m.byte_chunker is not None:
        qat_m.byte_chunker.lock_router(True)
    qat_m.set_crystal_schedule(1, 1)
    qat_m.set_snap_alpha(1.0)
    qat_m.set_qat_mix(0.0)
    qat_m.set_ternary_qat(True)
    lg_mix0, _ = _capture_stds(qat_m)
    mix0_diff = float((lg_fp.float() - lg_mix0.float()).abs().max().item())
    assert mix0_diff < 1e-5, f"Gate 2 mix=0 parity failed: diff={mix0_diff}"

    qat_m.set_qat_mix(1.0)
    _, std_mix1 = _capture_stds(qat_m)
    ratios: Dict[str, float] = {}
    for k, sf in std_fp.items():
        r = std_mix1[k] / max(1e-6, sf)
        ratios[k] = round(r, 4)
        assert 0.5 <= r <= 1.5, f"Gate 2 std jump > 1.5x in `{k}`: ratio={r:.4f} (fp={sf:.4f}, qat={std_mix1[k]:.4f})"

    del qat_m
    return {
        "mix0_max_abs_diff": round(mix0_diff, 8),
        "block_std_fp": {k: round(v, 5) for k, v in std_fp.items()},
        "block_std_ratio_mix1": ratios,
        "gate2_passed": True,
    }


def _cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float) -> float:
    if step <= warmup_steps:
        return base_lr * (step / float(max(1, warmup_steps)))
    prog = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * prog))


def _split_tier1_fp_groups(
    model: SimVQSplitLM,
) -> Tuple[List[nn.Parameter], List[nn.Parameter], List[nn.Parameter]]:
    cryst_p: List[nn.Parameter] = []
    chunk_p: List[nn.Parameter] = []
    core_p: List[nn.Parameter] = []
    for nm, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if nm.startswith(("crystal.", "crystal_2.")):
            cryst_p.append(p)
        elif nm.startswith("byte_chunker."):
            chunk_p.append(p)
        else:
            core_p.append(p)
    return cryst_p, core_p, chunk_p


def _split_qat_4groups(
    model: SimVQSplitLM,
) -> Tuple[List[nn.Parameter], List[nn.Parameter], List[nn.Parameter], List[nn.Parameter]]:
    """4 группы оптимизации для QAT: (gain_emb_p, cryst_p, chunk_p, core_p).
    Все `RMSNorm.weight` (хранимые в FP16) + `tok_emb` + `stream_gate/cryst2_gate` получают
    повышенный шаг для быстрой компенсации кубической усадки амплитуды `SwiGLU`."""
    gain_emb_ids: set[int] = {id(model.tok_emb.weight)}
    for mod in model.modules():
        if isinstance(mod, RMSNorm):
            for p in mod.parameters():
                if p.requires_grad:
                    gain_emb_ids.add(id(p))
    for gate_mod in (model.stream_gate, model.cryst2_gate):
        if gate_mod is not None:
            for p in gate_mod.parameters():
                if p.requires_grad:
                    gain_emb_ids.add(id(p))

    gain_emb_p: List[nn.Parameter] = []
    cryst_p: List[nn.Parameter] = []
    chunk_p: List[nn.Parameter] = []
    core_p: List[nn.Parameter] = []
    seen: set[int] = set()
    for nm, p in model.named_parameters():
        if (not p.requires_grad) or (id(p) in seen):
            continue
        seen.add(id(p))
        if id(p) in gain_emb_ids:
            gain_emb_p.append(p)
        elif nm.startswith(("crystal.", "crystal_2.")):
            cryst_p.append(p)
        elif nm.startswith("byte_chunker."):
            chunk_p.append(p)
        else:
            core_p.append(p)
    return gain_emb_p, cryst_p, chunk_p, core_p


# ---------------------------------------------------------------------------
# Stratified Web + 6 Disjoint Train Regimes + In-Distribution Cognitive Batcher
# ---------------------------------------------------------------------------


class StratifiedPhaseHBatcher:
    """Формирует стратифицированный батч (`B=64`: `38` поток `train_utf8.u16.bin` +
    `6` по 1 из каждого тренировочного домена `train_wins` + `20` когнитивных эпизодов
    строго в in-distribution границах `tr_sents`, `k <= 4`)."""

    def __init__(
        self,
        train_mm: np.memmap,
        tr_sents: Sequence[str],
        tok: ByteBPETokenizer,
        seq_len: int,
        train_wins: Optional[Dict[str, List[List[int]]]] = None,
        seed: int = 9026,
    ):
        self.train_mm = train_mm
        self.tr_sents = list(tr_sents)
        self.tok = tok
        self.seq_len = int(seq_len)
        self.cog_seq_len = min(768, self.seq_len)
        self.train_wins = {k: v for k, v in (train_wins or {}).items() if v}
        self.domain_keys = sorted(self.train_wins.keys())
        self.rng_np = np.random.default_rng(seed)
        self.rng_py = random.Random(seed + 17)

    def _sample_one_cognitive(self, kind: str) -> Tuple[List[int], List[int], int]:
        tw = self.cog_seq_len + 1
        while True:
            if kind == "s5":
                k = self.rng_py.choice((0, 1, 2, 3, 4))
                d_r = (2, 2) if k == 0 else ((1, 1) if k == 1 else (0, 1))
                p, t, _ = make_s5_story(self.rng_py, k, self.tr_sents, d_r)
            elif kind == "ov":
                p, t, _ = make_overwrite_story(
                    self.rng_py,
                    self.rng_py.randint(3, 5),
                    self.rng_py.randint(1, 2),
                    self.tr_sents,
                    2,
                    0.25,
                )
            else:
                p, t, _ = make_trap_overwrite_story(
                    self.rng_py,
                    self.rng_py.randint(3, 5),
                    self.rng_py.randint(1, 2),
                    self.tr_sents,
                    2,
                    0.25,
                )
            p_ids = self.tok.encode(p, add_bos=True, add_eos=False)
            f_ids = self.tok.encode(p + t, add_bos=True, add_eos=True)
            if len(f_ids) <= tw and f_ids[: len(p_ids)] == p_ids:
                return p_ids, f_ids, len(p_ids)

    def sample_batch(self, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if batch_size >= 64 and self.domain_keys:
            n_dom = len(self.domain_keys)
            n_cog = 20
            n_web = batch_size - n_dom - n_cog
        elif batch_size >= 12 and self.domain_keys:
            n_dom = len(self.domain_keys)
            n_cog = max(2, int(round(20.0 * batch_size / 64.0)))
            n_web = max(1, batch_size - n_dom - n_cog)
        else:
            n_dom = 1 if (self.domain_keys and batch_size >= 4) else 0
            n_cog = max(1, batch_size // 3)
            n_web = max(1, batch_size - n_dom - n_cog)

        parts_x: List[torch.Tensor] = []
        parts_y: List[torch.Tensor] = []

        if n_web > 0:
            xb_web, yb_web = _sample_memmap_batch(self.train_mm, n_web, self.seq_len, self.rng_np, device)
            parts_x.append(xb_web)
            parts_y.append(yb_web)

        if n_dom > 0 and self.domain_keys:
            xb_dom = torch.full((n_dom, self.seq_len), self.tok.PAD_ID, dtype=torch.long)
            yb_dom = torch.full((n_dom, self.seq_len), self.tok.PAD_ID, dtype=torch.long)
            chosen_keys = (
                self.domain_keys
                if n_dom == len(self.domain_keys)
                else [self.rng_py.choice(self.domain_keys) for _ in range(n_dom)]
            )
            for di, dk in enumerate(chosen_keys):
                w = self.rng_py.choice(self.train_wins[dk])
                w_use = w[: self.seq_len + 1]
                lw = len(w_use) - 1
                if lw > 0:
                    xb_dom[di, :lw] = torch.tensor(w_use[:-1], dtype=torch.long)
                    yb_dom[di, :lw] = torch.tensor(w_use[1:], dtype=torch.long)
            parts_x.append(xb_dom.to(device))
            parts_y.append(yb_dom.to(device))

        if n_cog > 0:
            xb_cog = torch.full((n_cog, self.seq_len), self.tok.PAD_ID, dtype=torch.long)
            yb_cog = torch.full((n_cog, self.seq_len), self.tok.PAD_ID, dtype=torch.long)
            n_s5 = max(1, int(round(n_cog * (11.0 / 20.0))))
            n_ov = max(0, int(round(n_cog * (6.0 / 20.0))))
            while n_s5 + n_ov > n_cog:
                if n_s5 > 1:
                    n_s5 -= 1
                else:
                    n_ov -= 1
            n_tr = n_cog - n_s5 - n_ov
            kinds = ["s5"] * n_s5 + ["ov"] * n_ov + ["trap"] * n_tr
            self.rng_py.shuffle(kinds)
            for ri, k_name in enumerate(kinds):
                _, f_ids, p_len = self._sample_one_cognitive(k_name)
                l_seq = len(f_ids)
                xb_cog[ri, : l_seq - 1] = torch.tensor(f_ids[:-1], dtype=torch.long)
                yb_cog[ri, p_len - 1 : l_seq - 1] = torch.tensor(f_ids[p_len:], dtype=torch.long)
            parts_x.append(xb_cog.to(device))
            parts_y.append(yb_cog.to(device))

        xb = torch.cat(parts_x, dim=0)
        yb = torch.cat(parts_y, dim=0)
        perm = torch.randperm(xb.size(0), device=device)
        return xb[perm], yb[perm]


# ---------------------------------------------------------------------------
# Stage 1 + Stage 2: Balanced FP Polish & Multi-Tap Block + Logit KL STE QAT
# ---------------------------------------------------------------------------


def run_stage_polish_and_qat(
    repo_root: Path,
    seed_bf16_ckpt: Path,
    data_dir: Path,
    ckpt_dir: Path,
    out_dir: Path,
    device: str,
    polish_steps: int = 1900,
    qat_steps: int = 1600,
    batch_size: int = 64,
    seq_len: int = 1024,
    kl_temp: float = 1.5,
    smoke: bool = False,
) -> Dict[str, object]:
    """Выполняет `Stage 1` (балансировка FP) и `Stage 2` (4-Tap + Logit-KL QAT -> `5.41 MB 2-bit LUT`)."""
    print(
        f"\n=== [Phase H | Stage 1+2] Seed={seed_bf16_ckpt.name} | "
        f"polish={polish_steps} + 4tap_kl_qat={qat_steps} | B={batch_size} L={seq_len} ===",
        flush=True,
    )
    assert seed_bf16_ckpt.exists(), f"Missing seed BF16 checkpoint: {seed_bf16_ckpt}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_mm = np.memmap(data_dir / "train_utf8.u16.bin", dtype=np.uint16, mode="r")
    val_mm = np.memmap(data_dir / "val_utf8.u16.bin", dtype=np.uint16, mode="r")

    payload = torch.load(seed_bf16_ckpt, map_location="cpu", weights_only=False)
    cfg = SimVQSplitConfig(**payload["simvq_split_config"])
    tok = ByteBPETokenizer.from_dict(payload["tokenizer"])
    tr_docs, te_docs, tr_sents, te_sents = load_gsm8k_corpora(repo_root)
    byte_per_bpe = compute_bpe_to_byte_scale(repo_root, te_docs, tr_docs, tr_sents)
    train_wins, eval_wins, worst_b, clean_b = build_disjoint_openweb_corpora(
        repo_root, tok, tr_docs, te_docs, seq_len=768
    )

    m_fp = SimVQSplitLM(cfg).to(device)
    m_fp.load_state_dict(payload["simvq_split_state_dict"], strict=True)
    if m_fp.byte_chunker is not None:
        m_fp.byte_chunker.lock_router(True)
    m_fp.set_crystal_schedule(1, 1)
    m_fp.set_snap_alpha(1.0)

    gate2_diag = check_gate2_block_std_parity(m_fp, device)
    print(f"  [Gate 2] Passed: ratios={gate2_diag['block_std_ratio_mix1']}", flush=True)
    router_snap = _snapshot_router_params(m_fp)

    batcher = StratifiedPhaseHBatcher(
        train_mm, tr_sents, tok, seq_len=seq_len, train_wins=train_wins, seed=9026
    )
    use_amp = device == "cuda" and torch.cuda.is_available()

    # Stage 1: Stratified FP Polish (includes 6 disjoint openweb train domains)
    cryst_fp, core_fp, chunk_fp = _split_tier1_fp_groups(m_fp)
    all_fp_p = cryst_fp + core_fp + chunk_fp
    fp_lrs = (5.0e-5, 1.8e-4, 2.5e-4)
    fp_min_lrs = (1.0e-5, 3.0e-5, 4.0e-5)
    opt_fp = torch.optim.AdamW(
        [
            {"params": cryst_fp, "lr": fp_lrs[0]},
            {"params": core_fp, "lr": fp_lrs[1]},
            {"params": chunk_fp, "lr": fp_lrs[2]},
        ],
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    warm_fp = min(50, max(1, polish_steps // 8))
    t_p0 = time.perf_counter()
    fp_log: List[Dict[str, float]] = []
    for step in range(1, polish_steps + 1):
        m_fp.train()
        m_fp.set_crystal_schedule(1, 1)
        m_fp.set_snap_alpha(1.0)
        for gi, (blr, mlr) in enumerate(zip(fp_lrs, fp_min_lrs)):
            opt_fp.param_groups[gi]["lr"] = _cosine_lr(step, polish_steps, warm_fp, blr, mlr)

        xb, yb = batcher.sample_batch(batch_size, device)
        opt_fp.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            lg = m_fp(xb)
            t_ce = F.cross_entropy(lg.transpose(1, 2), yb, ignore_index=tok.PAD_ID, reduction="none")
            denom = (yb != tok.PAD_ID).sum(dim=1).clamp(min=1)
            ce_loss = (t_ce.sum(dim=1) / denom).mean()
            loss = ce_loss + 0.04 * m_fp.get_commitment_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_fp_p, 1.0)
        opt_fp.step()

        if step in (1, max(1, polish_steps // 2), polish_steps) or step % 200 == 0:
            _sync_if_cuda(device)
            el_m = (time.perf_counter() - t_p0) / 60.0
            ce_v = float(ce_loss.item())
            assert math.isfinite(ce_v), f"Non-finite Stage 1 CE at step {step}"
            print(f"  [Stage 1 FP Polish | {el_m:4.1f}m] step {step:4d}/{polish_steps} ce={ce_v:.4f}", flush=True)
            fp_log.append({"step": step, "ce": round(ce_v, 4), "minutes": round(el_m, 2)})

    _assert_router_unchanged(m_fp, router_snap, "Stage1_FP_Polish")
    polish_min = round((time.perf_counter() - t_p0) / 60.0, 2)
    m_fp.eval()
    for p in m_fp.parameters():
        p.requires_grad_(False)
    fp_sd_cpu = {k: v.detach().cpu().clone() for k, v in m_fp.state_dict().items()}
    del opt_fp

    # Stage 2: 4-Group + 4-Tap Multi-Block & Logit-KL Distilled STE QAT
    print(
        f"\n=== [Stage 2 4-Tap KL-QAT] {qat_steps} steps (CE=0.35, KL=0.45, Blk=0.20, T={kl_temp}) -> 5.41 MB 2-bit LUT ===",
        flush=True,
    )
    t_q0 = time.perf_counter()
    qat_model = SimVQSplitLM(cfg).to(device)
    qat_model.load_state_dict({k: v.clone() for k, v in fp_sd_cpu.items()}, strict=True)
    if qat_model.byte_chunker is not None:
        qat_model.byte_chunker.lock_router(True)
    qat_model.set_crystal_schedule(1, 1)
    qat_model.set_snap_alpha(1.0)
    qat_model.set_qat_mix(0.30)
    qat_model.set_ternary_qat(True)

    gain_q, cryst_q, chunk_q, core_q = _split_qat_4groups(qat_model)
    all_qp = gain_q + cryst_q + chunk_q + core_q
    q_lrs = (1.2e-3, 1.0e-4, 6.0e-4, 4.5e-4)
    q_min_lrs = (1.5e-4, 1.5e-5, 8.0e-5, 6.0e-5)
    opt_q = torch.optim.AdamW(
        [
            {"params": gain_q, "lr": q_lrs[0]},
            {"params": cryst_q, "lr": q_lrs[1]},
            {"params": chunk_q, "lr": q_lrs[2]},
            {"params": core_q, "lr": q_lrs[3]},
        ],
        weight_decay=0.0,
    )
    warm_q = min(64, max(1, qat_steps // 16))
    ramp_q = min(320, max(1, qat_steps // 5))
    qat_log: List[Dict[str, float]] = []

    taps_fp: List[torch.Tensor] = []
    taps_q: List[torch.Tensor] = []
    hk_fp = _attach_tap_hooks(m_fp, taps_fp, detach=True)
    hk_q = _attach_tap_hooks(qat_model, taps_q, detach=False)

    try:
        for qs in range(1, qat_steps + 1):
            qat_model.train()
            mix = min(1.0, 0.30 + 0.70 * (qs / float(ramp_q)))
            qat_model.set_qat_mix(mix)
            for gi, (blr, mlr) in enumerate(zip(q_lrs, q_min_lrs)):
                opt_q.param_groups[gi]["lr"] = _cosine_lr(qs, qat_steps, warm_q, blr, mlr)

            xb, yb = batcher.sample_batch(batch_size, device)
            taps_fp.clear()
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    z_fp = m_fp(xb)
                log_p_fp = F.log_softmax(z_fp.float() / kl_temp, dim=-1)

            taps_q.clear()
            opt_q.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                z_qat = qat_model(xb)
                commit = qat_model.get_commitment_loss()

            valid_bool = yb != tok.PAD_ID
            valid_f = valid_bool.float()
            denom = valid_f.sum(dim=1).clamp(min=1.0)
            z_qat_f = z_qat.float()
            t_ce = F.cross_entropy(z_qat_f.transpose(1, 2), yb, ignore_index=tok.PAD_ID, reduction="none")
            ce_loss = (t_ce.sum(dim=1) / denom).mean()

            log_p_q = F.log_softmax(z_qat_f / kl_temp, dim=-1)
            kl_tok = F.kl_div(log_p_q, log_p_fp, reduction="none", log_target=True).sum(dim=-1) * (kl_temp * kl_temp)
            kl_loss = ((kl_tok * valid_f).sum(dim=1) / denom).mean()

            blk_loss = _compute_tap_loss(taps_fp, taps_q, last_mask=valid_bool)
            loss = 0.35 * ce_loss + 0.45 * kl_loss + 0.20 * blk_loss + 0.04 * commit.float()
            loss.backward()
            taps_fp.clear()
            taps_q.clear()
            torch.nn.utils.clip_grad_norm_(all_qp, 1.0)
            opt_q.step()

            if qs in (1, max(1, qat_steps // 2), qat_steps) or qs % 400 == 0:
                _sync_if_cuda(device)
                el_m = (time.perf_counter() - t_q0) / 60.0
                ce_v, kl_v, bk_v = float(ce_loss.item()), float(kl_loss.item()), float(blk_loss.item())
                assert math.isfinite(ce_v) and math.isfinite(kl_v) and math.isfinite(bk_v), f"NaN at step {qs}"
                print(
                    f"  [Stage 2 KL+4Tap | {el_m:4.1f}m] step {qs:4d}/{qat_steps} mix={mix:.2f} "
                    f"ce={ce_v:.4f} kl={kl_v:.4f} blk={bk_v:.4f}",
                    flush=True,
                )
                qat_log.append(
                    {"step": qs, "mix": round(mix, 3), "ce": round(ce_v, 4), "kl": round(kl_v, 4), "blk": round(bk_v, 4)}
                )
    finally:
        for h in hk_fp + hk_q:
            h.remove()

    router_mut_diff = _assert_router_unchanged(qat_model, router_snap, "Stage2_KL_QAT")
    qat_model.set_ternary_qat(False)
    qat_model.set_qat_mix(1.0)
    with torch.no_grad():
        for nm, mod in qat_model.named_modules():
            if isinstance(mod, nn.Linear) and nm != "lm_head":
                if qat_model.byte_chunker is not None and qat_model.byte_chunker.is_locked_router_module(mod):
                    continue
                wt, _, _ = quantize_ternary_weights_ls(mod.weight.data)
                mod.weight.data.copy_(wt)
    qat_model.eval()
    qat_sd_cpu = {k: v.detach().cpu().clone() for k, v in qat_model.state_dict().items()}
    qat_min = round((time.perf_counter() - t_q0) / 60.0, 2)
    del opt_q

    binfo = get_lut_backend_info()
    lut_backend = "cuda_ext" if (device == "cuda" and binfo["cuda_ext_available"]) else "auto"
    lut_model = pack_fast_model_2bit_lut(qat_model.cpu(), backend=lut_backend).to(device).eval()
    lut_model.set_crystal_schedule(1, 1)
    lut_fp_info = compute_2bit_lut_footprint(lut_model)
    assert int(lut_fp_info["num_lut_layers"]) == 95
    assert abs(float(lut_fp_info["total_lut_kb"]) - 5539.81) <= 0.1
    print(f"  [Packed Phase H 2-bit LUT] {lut_fp_info} (backend={lut_backend})", flush=True)

    # Save BF16 + Physical LUT checkpoints
    bf16_ckpt_p = ckpt_dir / "calcite5_tier1_17m_utf8_qat_h_bf16.pt"
    lut_ckpt_p = ckpt_dir / "calcite5_tier1_17m_utf8_qat_h_lut.pt"
    bf16_payload = {
        "format": "calcite5_tier1_utf8_qat_h_bf16_v2",
        "simvq_split_config": dict(cfg.__dict__),
        "tokenizer": tok.to_dict(),
        "simvq_split_state_dict": {k: v.to(torch.bfloat16) for k, v in fp_sd_cpu.items()},
        "simvq_split_ternary_qat_state_dict": {k: v.to(torch.bfloat16) for k, v in qat_sd_cpu.items()},
        "train_meta": {
            "attempt": 2,
            "seed_ckpt": seed_bf16_ckpt.name,
            "polish_steps": polish_steps,
            "qat_steps": qat_steps,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "kl_temp": kl_temp,
            "polish_minutes": polish_min,
            "qat_minutes": qat_min,
            "router_max_abs_diff": router_mut_diff,
        },
    }
    n_params = int(sum(p.numel() for p in m_fp.parameters()))
    torch.save(bf16_payload, bf16_ckpt_p)
    save_packed_lut_checkpoint(
        lut_ckpt_p,
        lut_model.cpu(),
        cfg,
        tok,
        metadata={
            "params": n_params,
            "polish_steps": polish_steps,
            "qat_steps": qat_steps,
            "phase": "H_attempt2",
        },
    )
    lut_model = lut_model.to(device).eval()

    # Evaluations: identical windows & OOD suites as Phase G
    n_eval = 20 if smoke else 200
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

    def _eval_sys1(tag_name: str, eval_mod: nn.Module) -> Dict[str, object]:
        val_bpb = eval_val_memmap_bpb(
            eval_mod, val_mm, seq_len=min(768, seq_len), n_batches=4 if smoke else 32, batch_size=8 if smoke else 16, device=device
        )
        ow = evaluate_openweb_regimes(eval_mod, eval_wins, worst_b, clean_b, device)
        gsm_docs = te_docs[:40] if smoke else te_docs[:500]
        gsm = evaluate_byte_prose_and_ratio(eval_mod, tok, gsm_docs, byte_per_bpe, seq_len=min(768, seq_len), device=device)
        s5, ov, trp, f1_in, f1_lg = _eval_cognitive_suites_b(eval_mod, tok, s5_suites, ov_suites, trap_suites, device)
        drift = measure_crystal_drift_at_L(eval_mod, tok.vocab_size, device, length=2048)
        s5_sum = round(sum(s5[k]["exact_pct"] for k in ("k2", "k4", "k8", "k16")), 2)
        print(
            f"  [Eval {tag_name}] val_bpb={val_bpb:.4f} | ow_bpb={ow['mean_openweb_bpb']:.4f} "
            f"(mid_utf8={ow['max_mid_utf8_split_pct']:.2f}%) | gsm_bpb={gsm['byte_bpb']:.4f} "
            f"(EqPPL={gsm['bpe_equiv_ppl']:.2f}) | S5_sum={s5_sum:.1f}% | "
            f"OV_long={ov['ood_long']['exact_pct']:.1f}% | SelF1_long={f1_lg:.1f}%",
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

    h_fp_eval = _eval_sys1("PhaseH_Tier1_FP", m_fp)
    qat_model = qat_model.to(device).eval()
    h_qat_eval = _eval_sys1("PhaseH_Tier1_QAT", qat_model)
    h_lut_eval = _eval_sys1("PhaseH_Tier1_5.41MB_LUT", lut_model)
    blk_v2 = _measure_block_alignment(m_fp, qat_model, device)

    lut_val = float(h_lut_eval["val_memmap_bpb"])
    fp_val = float(h_fp_eval["val_memmap_bpb"])
    bpb_gap = round(lut_val - fp_val, 4)
    lut_ow = float(h_lut_eval["openweb"]["mean_openweb_bpb"])
    ow_gap = round(lut_ow - float(h_fp_eval["openweb"]["mean_openweb_bpb"]), 4)
    lut_ov_long = float(h_lut_eval["overwrite"]["ood_long"]["exact_pct"])

    stop_crane_ok = (
        smoke
        or (
            float(h_lut_eval["openweb"]["max_mid_utf8_split_pct"]) == 0.0
            and lut_val <= 1.930
            and bpb_gap <= 0.175
            and lut_ow < 2.8682
            and float(h_lut_eval["s5_sum"]) >= 118.0
            and lut_ov_long >= 85.0
        )
    )
    beats_phase_g_all_axes = bool(
        smoke
        or (
            float(h_lut_eval["openweb"]["max_mid_utf8_split_pct"]) == 0.0
            and lut_val < 1.9879
            and lut_ow < 2.8682
            and float(h_lut_eval["gsm8k_te500"]["byte_bpb"]) < 1.9059
            and float(h_lut_eval["s5_sum"]) >= 115.0
            and lut_ov_long >= 83.5
        )
    )

    cache_kb = round(compute_state_cache_kb("SimVQSplitLM", cfg, None, 2048), 2)
    chunker_bytes = int(
        compute_chunker_state_bytes(
            cfg.d_model, cfg.byte_char_dim, cfg.byte_router_char_dim, byte_chunk_mode=cfg.byte_chunk_mode
        )
    )
    res = {
        "model_spec": {
            "params": n_params,
            "config": TIER1_CONFIG_DICT,
            "state_cache_kb_L2048": cache_kb,
            "chunker_state_bytes": chunker_bytes,
            "lut_footprint": {k: float(v) for k, v in lut_fp_info.items()},
        },
        "gate2_init_std": gate2_diag,
        "training": {
            "attempt": 2,
            "seed_bf16_ckpt": seed_bf16_ckpt.name,
            "polish_steps": polish_steps,
            "qat_steps": qat_steps,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "kl_temp": kl_temp,
            "router_max_abs_diff": router_mut_diff,
            "polish_minutes": polish_min,
            "qat_minutes": qat_min,
            "fp_log": fp_log,
            "qat_log": qat_log,
            "bf16_ckpt_bytes": int(bf16_ckpt_p.stat().st_size),
            "lut_ckpt_bytes": int(lut_ckpt_p.stat().st_size),
        },
        "gap_closure": {
            "val_memmap_bpb_gap_lut_minus_fp": bpb_gap,
            "mean_openweb_bpb_gap_lut_minus_fp": ow_gap,
            "stop_crane_passed": bool(stop_crane_ok),
            "beats_phase_g_all_axes": bool(beats_phase_g_all_axes),
        },
        "evaluations": {
            "tier1_h_fp": h_fp_eval,
            "tier1_h_qat": h_qat_eval,
            "tier1_h_lut_5_41mb": h_lut_eval,
        },
        "block_alignment_v2": blk_v2,
    }
    (out_dir / "stage12_system1_qat_closure.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    del m_fp, qat_model, lut_model
    gc.collect()
    if use_amp:
        torch.cuda.empty_cache()

    if not beats_phase_g_all_axes:
        raise RuntimeError(
            f"Stop-Crane triggered after Stage 2 (Attempt #2): val_lut={lut_val:.4f}, gap={bpb_gap:.4f}, "
            f"ow_lut={lut_ow:.4f}, s5_lut={h_lut_eval['s5_sum']:.1f}%, ov_long={lut_ov_long:.1f}%"
        )
    return res


# ---------------------------------------------------------------------------
# Stage 3: GSM8K + SphericalALU + 0-BPTT CRAM+z 4-Tap + KL STE QAT
# ---------------------------------------------------------------------------


def _compile_interleaved_alu_samples(
    train_items: Sequence[Dict[str, str]],
    extractor: TrainInducedQuantityExtractor,
    tok: ByteBPETokenizer,
    alu: SphericalBase10RegisterALU,
    is_byte_hnet: bool,
) -> List[Tuple[List[int], List[int], List[float]]]:
    samples: List[Tuple[List[int], List[int], List[float]]] = []
    max_len = 864 if is_byte_hnet else 280
    max_dag_steps = 10 if is_byte_hnet else 8
    for it in train_items:
        q_str = str(it["question"])
        slots, _, _ = extractor.extract_slots(q_str, include_aux_consts=True)
        usable_n = min(len(slots), MAX_SLOTS)
        if usable_n < 1:
            continue
        steps, _, ok = extractor.compile_train_dag(slots[:usable_n], str(it["answer"]))
        if not ok or not steps or len(steps) > max_dag_steps:
            continue
        dsl, sup_mask, _, inv = encode_interleaved_alu_trace(
            steps,
            [float(sl.val) for sl in slots[:usable_n]],
            alu,
            usable_n,
        )
        if inv or (not dsl) or len(sup_mask) != len(dsl):
            continue
        p_ids, _, _, _ = build_system1_prompt(extractor, q_str, slots, tok)
        plan_ids = [_byte_token_id(ch) for ch in dsl] + [tok.EOS_ID]
        plan_w = [2.50 if flg else 0.0 for flg in sup_mask] + [2.50]
        seq = p_ids + plan_ids
        if len(seq) > max_len:
            continue
        shift_w = ([0.02] * max(0, len(p_ids) - 1)) + plan_w
        samples.append((seq[:-1], seq[1:], shift_w))
    return samples


def train_kl_cosine_cram_qat(
    sys1: Calcite5MathSystem1,
    samples: Sequence[Tuple[List[int], List[int], List[float]]],
    adapted_ckpt: Path,
    fp_train_loss: float,
    fp_train_sec: float,
    qat_steps: int = 1120,
    batch_size: int = 64,
    kl_temp: float = 1.5,
    seed: int = 2026,
) -> Dict[str, float]:
    """Доучивает `qat_model` поверх `sys1.model` (`4` группы LR, `Masked CE + KL + 4-Tap Block`)."""
    t_q0 = time.perf_counter()
    device = sys1.device
    use_amp = device == "cuda" and torch.cuda.is_available()
    rng = random.Random(seed + 101)
    torch.manual_seed(seed + 101)

    teacher = sys1.model.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    qat_model = SimVQSplitLM(sys1.cfg).to(device)
    qat_model.load_state_dict({k: v.detach().clone() for k, v in teacher.state_dict().items()}, strict=True)
    if qat_model.byte_chunker is not None:
        sys1._restore_locked_router_weights(qat_model)
    qat_model.set_crystal_schedule(1, 1)
    qat_model.set_snap_alpha(1.0)
    qat_model.set_qat_mix(0.30)
    qat_model.set_ternary_qat(True)
    router_snap = _snapshot_router_params(qat_model)

    gain_q, cryst_q, chunk_q, core_q = _split_qat_4groups(qat_model)
    all_qp = gain_q + cryst_q + chunk_q + core_q
    q_lrs = (1.2e-3, 1.0e-4, 6.0e-4, 4.5e-4)
    q_min_lrs = (1.5e-4, 1.5e-5, 8.0e-5, 6.0e-5)
    q_opt = torch.optim.AdamW(
        [
            {"params": gain_q, "lr": q_lrs[0]},
            {"params": cryst_q, "lr": q_lrs[1]},
            {"params": chunk_q, "lr": q_lrs[2]},
            {"params": core_q, "lr": q_lrs[3]},
        ],
        weight_decay=0.0,
    )

    taps_fp: List[torch.Tensor] = []
    taps_q: List[torch.Tensor] = []
    hk_fp = _attach_tap_hooks(teacher, taps_fp, detach=True)
    hk_q = _attach_tap_hooks(qat_model, taps_q, detach=False)

    def _make_batch(batch_items: Sequence[Tuple[List[int], List[int], List[float]]]):
        m_len = max(len(x[0]) for x in batch_items)
        xb_l = torch.full((len(batch_items), m_len), sys1.tok.PAD_ID, dtype=torch.long)
        yb_l = torch.full((len(batch_items), m_len), -100, dtype=torch.long)
        wm_l = torch.zeros((len(batch_items), m_len), dtype=torch.float32)
        for ri, (xs, ys, sw) in enumerate(batch_items):
            li = len(xs)
            xb_l[ri, :li] = torch.tensor(xs, dtype=torch.long)
            yb_l[ri, :li] = torch.tensor(ys, dtype=torch.long)
            wm_l[ri, :li] = torch.tensor(sw, dtype=torch.float32)
        return xb_l.to(device), yb_l.to(device), wm_l.to(device)

    warm_q = min(60, max(1, qat_steps // 12))
    ramp_q = min(280, max(1, qat_steps // 4))
    last_loss = 0.0
    try:
        for qs in range(1, qat_steps + 1):
            qat_model.train()
            mix = min(1.0, 0.30 + 0.70 * (qs / float(ramp_q)))
            qat_model.set_qat_mix(mix)
            for gi, (blr, mlr) in enumerate(zip(q_lrs, q_min_lrs)):
                q_opt.param_groups[gi]["lr"] = _cosine_lr(qs, qat_steps, warm_q, blr, mlr)

            batch = rng.sample(list(samples), min(batch_size, len(samples)))
            xb, yb, w_mask = _make_batch(batch)

            taps_fp.clear()
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    z_fp = teacher(xb)
                log_p_fp = F.log_softmax(z_fp.float() / kl_temp, dim=-1)

            taps_q.clear()
            q_opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                z_qat = qat_model(xb)
                commit = qat_model.get_commitment_loss()

            z_qat_f = z_qat.float()
            ce_tok = F.cross_entropy(
                z_qat_f.reshape(-1, sys1.cfg.vocab_size), yb.reshape(-1), ignore_index=-100, reduction="none"
            ).view_as(yb)
            ce_loss = (ce_tok * w_mask).sum() / w_mask.sum().clamp(min=1.0)

            not_pad = yb != -100
            kl_w = torch.where(not_pad, torch.where(w_mask > 0.5, 1.0, torch.where(w_mask > 0.0, 0.15, 0.0)), 0.0)
            log_p_q = F.log_softmax(z_qat_f / kl_temp, dim=-1)
            kl_tok = F.kl_div(log_p_q, log_p_fp, reduction="none", log_target=True).sum(dim=-1) * (kl_temp * kl_temp)
            kl_loss = (kl_tok * kl_w).sum() / kl_w.sum().clamp(min=1.0)

            blk_loss = _compute_tap_loss(taps_fp, taps_q, last_mask=not_pad)
            loss = 0.35 * ce_loss + 0.45 * kl_loss + 0.20 * blk_loss + 0.04 * commit.float()
            loss.backward()
            taps_fp.clear()
            taps_q.clear()
            torch.nn.utils.clip_grad_norm_(all_qp, 1.0)
            q_opt.step()
            last_loss = float(loss.item())

            if qs in (1, max(1, qat_steps // 2), qat_steps) or qs % 280 == 0:
                print(
                    f"  [Stage 3B CRAM 4Tap+KL] step {qs:4d}/{qat_steps} mix={mix:.2f} "
                    f"ce={ce_loss.item():.4f} kl={kl_loss.item():.4f} blk={blk_loss.item():.4f}",
                    flush=True,
                )
    finally:
        for h in hk_fp + hk_q:
            h.remove()

    router_diff = _assert_router_unchanged(qat_model, router_snap, "Stage3B_CRAM_QAT")
    qat_model.set_ternary_qat(False)
    qat_model.set_qat_mix(1.0)
    with torch.no_grad():
        for m_name, mod in qat_model.named_modules():
            if isinstance(mod, nn.Linear) and mod is not qat_model.lm_head and "lm_head" not in m_name:
                if qat_model.byte_chunker is not None and qat_model.byte_chunker.is_locked_router_module(mod):
                    continue
                w_tern, _, _ = quantize_ternary_weights_ls(mod.weight.data)
                mod.weight.data.copy_(w_tern)
    if qat_model.byte_chunker is not None:
        sys1._restore_locked_router_weights(qat_model)
    qat_model.eval()
    qat_model.set_snap_alpha(1.0)

    qat_sd = {k: v.detach().cpu() for k, v in qat_model.state_dict().items()}
    fp_sd = {k: v.detach().cpu() for k, v in teacher.state_dict().items()}
    binfo = get_lut_backend_info()
    lut_backend = "cuda_ext" if (device == "cuda" and binfo["cuda_ext_available"]) else "auto"
    lut_m = pack_fast_model_2bit_lut(qat_model.cpu(), backend=lut_backend).to(device).eval()
    lut_m.set_crystal_schedule(1, 1)
    lut_fp_info = compute_2bit_lut_footprint(lut_m)

    adapted_ckpt.parent.mkdir(parents=True, exist_ok=True)
    tot_sec = fp_train_sec + (time.perf_counter() - t_q0)
    torch.save(
        {
            "fp_state_dict": fp_sd,
            "qat_state_dict": qat_sd,
            "d_model": sys1.cfg.d_model,
            "n_samples": len(samples),
            "final_loss": float(fp_train_loss),
            "qat_kl_cos_loss": float(last_loss),
            "cram_qat_steps": int(qat_steps),
            "lut_footprint": lut_fp_info,
            "use_interleaved_alu": True,
            "simvq_split_config": dict(sys1.cfg.__dict__),
            "train_seconds": float(tot_sec),
        },
        adapted_ckpt,
    )
    sys1.load_adapted(adapted_ckpt)
    sys1.lut_model = accelerate_lut_model(sys1.lut_model, backend=lut_backend, inplace=True).to(device).eval()
    return {
        "n_compiled_train": float(len(samples)),
        "fp_final_loss": float(fp_train_loss),
        "qat_kl_cos_final_loss": float(last_loss),
        "cram_qat_steps": float(qat_steps),
        "lut_total_kb": float(lut_fp_info["total_lut_kb"]),
        "num_lut_layers": float(lut_fp_info["num_lut_layers"]),
        "router_max_abs_diff": float(router_diff),
        "train_seconds": float(tot_sec),
    }


def run_stage_eval_cram_h(
    repo_root: Path,
    ckpt_dir: Path,
    out_dir: Path,
    device: str,
    cram_qat_steps: int = 1120,
    skip_sealed_test: bool = False,
    reextract_cram: bool = False,
    smoke: bool = False,
) -> Dict[str, object]:
    """Track 3 Фазы H: `GSM8K + SphericalALU + 4-Tap KL QAT + парный solver_lut` (`0 BPTT`)."""
    print(f"\n=== [Phase H | Stage 3 CRAM+z] qat_steps={cram_qat_steps} (smoke={smoke}) ===", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    bf16_ckpt_p = ckpt_dir / "calcite5_tier1_17m_utf8_qat_h_bf16.pt"
    assert bf16_ckpt_p.exists(), f"Missing Phase H BF16 checkpoint: {bf16_ckpt_p}"

    tr_p = repo_root / "benchmarks" / "gsm8k" / "train.jsonl"
    te_p = repo_root / "benchmarks" / "gsm8k" / "test.jsonl"
    with open(tr_p, "r", encoding="utf-8") as f:
        all_tr = [json.loads(line) for line in f if line.strip()]
    train_items = all_tr[:64] if smoke else all_tr[:6000]
    cal_items = all_tr[6000:6032] if smoke else all_tr[6000:6736]
    blind_items = all_tr[6736:6768] if smoke else all_tr[6736:]

    ext = TrainInducedQuantityExtractor()
    ext_stats = ext.fit(all_tr[:6000])
    alu = SphericalBase10RegisterALU(d_sub=15, num_digits=12, seed=3301)
    alu_diag = alu.verify_crystal_invariants(n_pairs=200)

    sys1 = Calcite5MathSystem1(bf16_ckpt_p, device=device, seed=2026)
    adapted_ckpt = ckpt_dir / "tier1_h_cram_adapted.pt"
    binfo = get_lut_backend_info()
    lut_backend = "cuda_ext" if (device == "cuda" and binfo["cuda_ext_available"]) else "auto"

    if adapted_ckpt.is_file() and (not reextract_cram) and (not smoke):
        print(f"  [System-1 H] Loading cached adapted ckpt: {adapted_ckpt}", flush=True)
        sys1.load_adapted(adapted_ckpt)
        ck_meta = torch.load(adapted_ckpt, map_location="cpu", weights_only=False)
        assert sys1.lut_model is not None
        sys1.lut_model = accelerate_lut_model(sys1.lut_model, backend=lut_backend, inplace=True).to(device).eval()
        lut_meta = ck_meta.get("lut_footprint", compute_2bit_lut_footprint(sys1.lut_model))
        fit_stats = {
            "cached": 1.0,
            "n_compiled_train": float(ck_meta.get("n_samples", 4443.0)),
            "fp_final_loss": float(ck_meta.get("final_loss", 0.0)),
            "qat_kl_cos_final_loss": float(ck_meta.get("qat_kl_cos_loss", 0.0)),
            "cram_qat_steps": float(ck_meta.get("cram_qat_steps", cram_qat_steps)),
            "lut_total_kb": float(lut_meta["total_lut_kb"]),
            "num_lut_layers": float(lut_meta["num_lut_layers"]),
            "train_seconds": float(ck_meta.get("train_seconds", 0.0)),
        }
    else:
        fp_stats = sys1.fit_on_train_split(
            train_items=train_items,
            extractor=ext,
            epochs=1 if smoke else 16,
            batch_size=16 if smoke else 64,
            lr=4.0e-4,
            qat_steps=0,
            save_ckpt_path=None,
            use_interleaved_alu=True,
            alu=alu,
        )
        samples = _compile_interleaved_alu_samples(train_items, ext, sys1.tok, alu, sys1.is_byte_hnet)
        fit_stats = train_kl_cosine_cram_qat(
            sys1,
            samples,
            adapted_ckpt=adapted_ckpt,
            fp_train_loss=float(fp_stats["final_loss"]),
            fp_train_sec=float(fp_stats["train_seconds"]),
            qat_steps=4 if smoke else cram_qat_steps,
            batch_size=8 if smoke else 64,
            kl_temp=1.5,
            seed=2026,
        )
        fit_stats["cached"] = 0.0

    d_model = int(sys1.cfg.d_model)
    slots_tr = [ext.extract_slots(str(it["question"]), include_aux_consts=True)[0] for it in train_items]
    slots_cal = [ext.extract_slots(str(it["question"]), include_aux_consts=True)[0] for it in cal_items]
    slots_bld = [ext.extract_slots(str(it["question"]), include_aux_consts=True)[0] for it in blind_items]
    q_tr = [str(it["question"]) for it in train_items]
    q_cal = [str(it["question"]) for it in cal_items]
    q_bld = [str(it["question"]) for it in blind_items]

    f_tr_fp = sys1.extract_features_and_beams(q_tr, slots_tr, ext, batch_size=32, beam_width=0, use_lut=False, alu=alu)
    f_tr_lut = sys1.extract_features_and_beams(q_tr, slots_tr, ext, batch_size=32, beam_width=0, use_lut=True, alu=alu)
    f_cal_fp = sys1.extract_features_and_beams(q_cal, slots_cal, ext, batch_size=32, beam_width=0, use_lut=False, alu=alu)
    f_bld_fp = sys1.extract_features_and_beams(q_bld, slots_bld, ext, batch_size=32, beam_width=0, use_lut=False, alu=alu)
    f_cal_lut = sys1.extract_features_and_beams(q_cal, slots_cal, ext, batch_size=32, beam_width=0, use_lut=True, alu=alu)
    f_bld_lut = sys1.extract_features_and_beams(q_bld, slots_bld, ext, batch_size=32, beam_width=0, use_lut=True, alu=alu)

    cram_ep = 2 if smoke else 12
    solver_fp = OpenGraphCRAMSolver(ext, alu, d_model=d_model, device=device, seed=2026)
    cram_stats_fp = solver_fp.fit_bptt_free(train_items, slots_tr, f_tr_fp, epochs=cram_ep, batch_size=128, lr=0.018)

    solver_lut = OpenGraphCRAMSolver(ext, alu, d_model=d_model, device=device, seed=2026)
    cram_stats_lut = solver_lut.fit_bptt_free(train_items, slots_tr, f_tr_lut, epochs=cram_ep, batch_size=128, lr=0.018)

    # 1) FP branch
    f_cal_fp_e = solver_fp.enrich_lm_feats_with_system1_tf(cal_items, slots_cal, f_cal_fp, sys1, top_k=6, use_lut=False, batch_size=48)
    f_bld_fp_e = solver_fp.enrich_lm_feats_with_system1_tf(blind_items, slots_bld, f_bld_fp, sys1, top_k=6, use_lut=False, batch_size=48)
    cal_fp_stats = solver_fp.calibrate_on_split(cal_items, slots_cal, f_cal_fp_e)
    res_cal_fp = evaluate_phase_f_split(solver_fp, cal_items, slots_cal, f_cal_fp_e, "Cal_Tier1H_FP")
    res_bld_fp = evaluate_phase_f_split(solver_fp, blind_items, slots_bld, f_bld_fp_e, "BlindDev_Tier1H_FP")

    # 2) Ablation: LUT System-1 scored by solver_fp (calibrated fairly via clone on cal_lut)
    solver_fp_lutcal = OpenGraphCRAMSolver(ext, alu, d_model=d_model, device=device, seed=2026)
    solver_fp_lutcal.load_state_dict(solver_fp.state_dict())
    f_cal_lut_on_fp = solver_fp_lutcal.enrich_lm_feats_with_system1_tf(
        cal_items, slots_cal, copy.deepcopy(f_cal_lut), sys1, top_k=6, use_lut=True, batch_size=32
    )
    f_bld_lut_on_fp = solver_fp_lutcal.enrich_lm_feats_with_system1_tf(
        blind_items, slots_bld, copy.deepcopy(f_bld_lut), sys1, top_k=6, use_lut=True, batch_size=32
    )
    solver_fp_lutcal.calibrate_on_split(cal_items, slots_cal, f_cal_lut_on_fp)
    res_bld_lut_on_fp = evaluate_phase_f_split(solver_fp_lutcal, blind_items, slots_bld, f_bld_lut_on_fp, "BlindDev_Tier1H_LUT_on_FPSolver")

    # 3) Co-adapted 5.41 MB 2-bit LUT branch (`solver_lut`)
    f_cal_lut_e = solver_lut.enrich_lm_feats_with_system1_tf(cal_items, slots_cal, f_cal_lut, sys1, top_k=6, use_lut=True, batch_size=32)
    f_bld_lut_e = solver_lut.enrich_lm_feats_with_system1_tf(blind_items, slots_bld, f_bld_lut, sys1, top_k=6, use_lut=True, batch_size=32)
    cal_lut_stats = solver_lut.calibrate_on_split(cal_items, slots_cal, f_cal_lut_e)
    res_cal_lut = evaluate_phase_f_split(solver_lut, cal_items, slots_cal, f_cal_lut_e, "Cal_Tier1H_2Bit_LUT")
    res_bld_lut = evaluate_phase_f_split(solver_lut, blind_items, slots_bld, f_bld_lut_e, "BlindDev_Tier1H_2Bit_LUT")

    bld_gap_pp = round(abs(float(res_bld_fp["pass1_pct"]) - float(res_bld_lut["pass1_pct"])), 3)
    print(
        f"-> BlindDev: FP={res_bld_fp['pass1_pct']:.2f}% | "
        f"LUT(on FP solver)={res_bld_lut_on_fp['pass1_pct']:.2f}% | "
        f"LUT(on LUT solver)={res_bld_lut['pass1_pct']:.2f}% (gap={bld_gap_pp:.2f} pp)",
        flush=True,
    )

    cram_gate_ok = smoke or (float(res_bld_lut["pass1_pct"]) > 16.15 and bld_gap_pp < 1.90)
    vault = SingleShotTestVault(te_p, max_calls=1)
    sealed_out: Dict[str, object] = {}
    if (not skip_sealed_test) and (not smoke):
        if not cram_gate_ok:
            sealed_out = {"vault_calls_used": 0, "blocked_by_stop_crane": True}
        else:
            t_te0 = time.perf_counter()
            sealed_items = vault.load_sealed_items()
            assert vault.calls_used == 1 and len(sealed_items) == 1319
            q_te = [str(it["question"]) for it in sealed_items]
            slots_te = [ext.extract_slots(q, include_aux_consts=True)[0] for q in q_te]
            f_te_fp = sys1.extract_features_and_beams(q_te, slots_te, ext, batch_size=32, beam_width=0, use_lut=False, alu=alu)
            f_te_lut = sys1.extract_features_and_beams(q_te, slots_te, ext, batch_size=32, beam_width=0, use_lut=True, alu=alu)
            f_te_fp = solver_fp.enrich_lm_feats_with_system1_tf(sealed_items, slots_te, f_te_fp, sys1, top_k=6, use_lut=False, batch_size=48)
            res_te_fp = evaluate_phase_f_split(solver_fp, sealed_items, slots_te, f_te_fp, "SealedTest_Tier1H_FP")
            f_te_lut = solver_lut.enrich_lm_feats_with_system1_tf(sealed_items, slots_te, f_te_lut, sys1, top_k=6, use_lut=True, batch_size=32)
            res_te_lut = evaluate_phase_f_split(solver_lut, sealed_items, slots_te, f_te_lut, "SealedTest_Tier1H_2Bit_LUT")
            print(
                f"-> Sealed Test (1/1 in {time.perf_counter()-t_te0:.1f}s): "
                f"FP={res_te_fp['pass1_pct']:.2f}% (Pass@5={res_te_fp['pass5_pct']:.2f}%) | "
                f"5.41MB LUT={res_te_lut['pass1_pct']:.2f}% (Pass@5={res_te_lut['pass5_pct']:.2f}%)",
                flush=True,
            )
            sealed_out = {"vault_calls_used": int(vault.calls_used), "fp": res_te_fp, "lut_2bit": res_te_lut}

    cram_res = {
        "alu_invariants": alu_diag,
        "system1_fit": fit_stats,
        "cram_fit_fp": {**ext_stats, **cram_stats_fp},
        "cram_fit_lut": {**ext_stats, **cram_stats_lut},
        "cal_calibration": {"fp": cal_fp_stats, "lut_2bit": cal_lut_stats},
        "cal": {"fp": res_cal_fp, "lut_2bit": res_cal_lut},
        "blind_dev": {"fp": res_bld_fp, "lut_on_fp_solver": res_bld_lut_on_fp, "lut_2bit": res_bld_lut},
        "blind_dev_stop_crane_passed": bool(cram_gate_ok),
        "sealed_test": sealed_out,
    }
    (out_dir / "stage3_cram_qat_closure.json").write_text(json.dumps(cram_res, indent=2), encoding="utf-8")
    if (not skip_sealed_test) and (not smoke) and (not cram_gate_ok):
        raise RuntimeError(
            f"Stop-Crane triggered on BlindDev before SealedTest: LUT={res_bld_lut['pass1_pct']:.2f}% (need >16.15%), gap={bld_gap_pp:.2f}pp (need <1.90)"
        )
    return cram_res


# ---------------------------------------------------------------------------
# Local Smoke Micro-Corpus & Combined Summary Assembly
# ---------------------------------------------------------------------------


def _build_local_smoke_corpus(repo_root: Path, smoke_data_dir: Path) -> None:
    smoke_data_dir.mkdir(parents=True, exist_ok=True)
    tr_b = smoke_data_dir / "train_utf8.u16.bin"
    val_b = smoke_data_dir / "val_utf8.u16.bin"
    if tr_b.exists() and val_b.exists():
        return
    tr_docs, _, _, _ = load_gsm8k_corpora(repo_root)
    tok = ByteBPETokenizer(merges=[])
    sample_txt = "\n".join(tr_docs[:250]) + "\nМногоязычный тест кириллицы 🌍🔭🧪 def add(x, y): return x + y\n"
    arr = np.array([tok.BOS_ID] + list(sample_txt.encode("utf-8")) + [tok.EOS_ID], dtype=np.uint16)
    tr_arr = np.tile(arr, 12)
    val_arr = np.tile(arr[:2048], 4)
    tr_arr.tofile(tr_b)
    val_arr.tofile(val_b)


def assemble_phase_h_summary(repo_root: Path, s12: Dict[str, object], s3: Dict[str, object]) -> Dict[str, object]:
    phase_g_p = repo_root / "benchmarks" / "phase_g_colab_scaleup_results.json"
    phase_g_ref: Dict[str, object] = {}
    if phase_g_p.exists():
        g_data = json.loads(phase_g_p.read_text(encoding="utf-8"))
        t2_g = g_data["track2_tier1"]["evaluations"]
        t3_g = g_data["track3_cram_gsm8k"]
        phase_g_ref = {
            "base_1_48m_lut": t2_g["base_1_48m_lut"],
            "phase_g_tier1_fp": t2_g["tier1_fp"],
            "phase_g_tier1_lut": t2_g["tier1_lut_5_41mb"],
            "phase_g_cram_blind_dev": t3_g["blind_dev"],
            "phase_g_cram_sealed_test": t3_g["sealed_test"],
        }
    return {
        "protocol": {
            "variant": "Phase_H_Tier1_QAT_Closure",
            "vocab_size": 259,
            "chunker_state_bytes": s12["model_spec"]["chunker_state_bytes"],  # type: ignore[index]
            "state_cache_kb_L2048": s12["model_spec"]["state_cache_kb_L2048"],  # type: ignore[index]
            "sealed_test_calls_used": int(s3.get("sealed_test", {}).get("vault_calls_used", 0)),  # type: ignore[union-attr]
            "sealed_test_max_calls": 1,
        },
        "phase_g_reference": phase_g_ref,
        "system1_qat_closure": s12,
        "track3_cram_gsm8k": s3,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Calcite-5 Phase H QAT Closure Runner")
    parser.add_argument(
        "--stage",
        choices=("build-corpus", "polish-and-qat", "eval-cram", "all"),
        default="all",
    )
    parser.add_argument("--seed-bf16-ckpt", type=str, default=str(ROOT_DIR / "checkpoints" / "calcite5_tier1_17m_utf8_bf16.pt"))
    parser.add_argument("--data-dir", type=str, default="/content/data")
    parser.add_argument("--ckpt-dir", type=str, default=str(ROOT_DIR / "checkpoints"))
    parser.add_argument("--out-dir", type=str, default=str(ROOT_DIR / "benchmarks" / "phase_h_out"))
    parser.add_argument("--total-gb", type=float, default=1.80)
    parser.add_argument("--polish-steps", type=int, default=1900)
    parser.add_argument("--qat-steps", type=int, default=1600)
    parser.add_argument("--cram-qat-steps", type=int, default=1120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--skip-sealed-test", action="store_true")
    parser.add_argument("--reextract-cram", action="store_true")
    parser.add_argument("--smoke-local", action="store_true", help="Быстрый локальный Трассер №1 на RTX 5060 (<60с)")
    args = parser.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seed_ckpt = Path(args.seed_bf16_ckpt)
    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.out_dir)

    if args.smoke_local:
        out_dir = ROOT_DIR / "benchmarks" / "phase_h_smoke_out"
        ckpt_dir = out_dir / "ckpts"
        data_dir = out_dir / "smoke_data"
        _build_local_smoke_corpus(ROOT_DIR, data_dir)
        s12 = run_stage_polish_and_qat(
            ROOT_DIR,
            seed_ckpt,
            data_dir,
            ckpt_dir,
            out_dir,
            dev,
            polish_steps=4,
            qat_steps=4,
            batch_size=4,
            seq_len=512,
            smoke=True,
        )
        s3 = run_stage_eval_cram_h(
            ROOT_DIR,
            ckpt_dir,
            out_dir,
            dev,
            cram_qat_steps=4,
            skip_sealed_test=True,
            reextract_cram=True,
            smoke=True,
        )
        comb = assemble_phase_h_summary(ROOT_DIR, s12, s3)
        smk_p = out_dir / "phase_h_smoke_results.json"
        smk_p.write_text(json.dumps(comb, indent=2), encoding="utf-8")
        print(f"\n=== LOCAL SMOKE TRACER #1 PASSED -> {smk_p} ({smk_p.stat().st_size:,} B) ===", flush=True)
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in ("build-corpus", "all"):
        if not (data_dir / "train_utf8.u16.bin").exists() or args.stage == "build-corpus":
            run_stage_build_corpus(ROOT_DIR, data_dir, total_gb=args.total_gb)

    s12: Dict[str, object] = {}
    if args.stage in ("polish-and-qat", "all"):
        s12 = run_stage_polish_and_qat(
            ROOT_DIR,
            seed_ckpt,
            data_dir,
            ckpt_dir,
            out_dir,
            dev,
            polish_steps=args.polish_steps,
            qat_steps=args.qat_steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            smoke=False,
        )
    elif (out_dir / "stage12_system1_qat_closure.json").exists():
        s12 = json.loads((out_dir / "stage12_system1_qat_closure.json").read_text(encoding="utf-8"))

    s3: Dict[str, object] = {}
    if args.stage in ("eval-cram", "all"):
        s3 = run_stage_eval_cram_h(
            ROOT_DIR,
            ckpt_dir,
            out_dir,
            dev,
            cram_qat_steps=args.cram_qat_steps,
            skip_sealed_test=args.skip_sealed_test,
            reextract_cram=args.reextract_cram,
            smoke=False,
        )
    elif (out_dir / "stage3_cram_qat_closure.json").exists():
        s3 = json.loads((out_dir / "stage3_cram_qat_closure.json").read_text(encoding="utf-8"))

    if s12 and s3:
        comb = assemble_phase_h_summary(ROOT_DIR, s12, s3)
        final_p = out_dir / "phase_h_tier1_qat_closure_results.json"
        final_p.write_text(json.dumps(comb, indent=2), encoding="utf-8")
        print(f"\nSaved combined Phase H JSON -> {final_p} ({final_p.stat().st_size:,} B)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
