"""
Zero-Cheat Phase H Verification Suite (`tests/test_phase_h_qat_closure.py`).

Enforces 5 deterministic Phase H invariants (`h1..h5`):
  h1: 4-group QAT parameter splitting (FP16 `RMSNorm` gains + embeddings separated without overlap),
      4-tap multi-block capture (`cortex_pre[-1]`, `mid_mlp`, `cortex_post[-1]`, `norm_f`), live
      gradients under ternary STE QAT, and bitwise locked-router immutability (`max_diff == 0.0`).
  h2: Direct loading and `0.0` roundtrip parity of `checkpoints/calcite5_tier1_17m_utf8_qat_h_lut.pt`
      (`17,488,961` params, `95` `FastPackedLUT2BitLinear` layers = `5,539.81 KB`, `16.00x`,
      `230.11 KB` `O(1)` cache @ `L=2048`, `2,164 B` constant chunker buffer, `0.00%` mid-UTF8 cuts).
  h3: System-1 Phase H `5.41 MB LUT` strictly outperforms Phase G LUT on every core benchmark
      (`val_memmap_bpb`, `mean_openweb_bpb`, `gsm8k_te500`, `s5_sum >= 115%`, `overwrite.ood_long`)
      at `0.00%` mid-UTF8 cuts while honestly recording the `val_memmap` `0.2007 BPB` grid plateau.
  h4: `GSM8K + SphericalALU + 0-BPTT CRAM+z` closure (`vault_calls_used == 1`, `n == 1319`,
      `ALU == 100%`, `drift < 1e-6`, `Gershgorin >= 4.0`, `BlindDev LUT >= 17%`, `Sealed LUT >= 14%`).
  h5: AST hygiene of `scripts/colab_phase_h_qat_runner.py` and strict `0`-byte `git diff HEAD` on
      protected `simvq_split/`, `experiments/cram_gsm8k/`, Phase G LUT ckpt, and A–G benchmark JSONs.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys

import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from kernels.fast_packed_lut import (  # noqa: E402
    FastPackedLUT2BitLinear,
    load_packed_lut_checkpoint,
    save_packed_lut_checkpoint,
)
from run_simvq_split_benchmark import compute_state_cache_kb, count_params  # noqa: E402
from scripts.colab_phase_h_qat_runner import (  # noqa: E402
    _assert_router_unchanged,
    _attach_tap_hooks,
    _compute_tap_loss,
    _measure_block_alignment,
    _snapshot_router_params,
    _split_qat_4groups,
)
from simvq_split.baselines import RMSNorm  # noqa: E402
from simvq_split.memory import compute_chunker_state_bytes  # noqa: E402
from simvq_split.models import SimVQSplitConfig, SimVQSplitLM  # noqa: E402
from simvq_split.quant_lut import PackedLUT2BitLinear, compute_2bit_lut_footprint  # noqa: E402

PHASE_H_LUT_CKPT = ROOT_DIR / "checkpoints" / "calcite5_tier1_17m_utf8_qat_h_lut.pt"
PHASE_H_JSON = ROOT_DIR / "benchmarks" / "phase_h_tier1_qat_closure_results.json"


def test_h1_4group_4tap_distill_and_router_lock():
    torch.manual_seed(2026)
    cfg = SimVQSplitConfig(
        vocab_size=259,
        d_model=64,
        n_heads=2,
        d_ff=128,
        n_layers_pre=1,
        n_layers_post=1,
        window_size=8,
        num_spheres=4,
        d_sphere=16,
        codebook_size=8,
        mem_heads=2,
        mem_dim=16,
        use_byte_chunker=True,
        byte_char_dim=16,
        byte_d_ff=64,
        byte_router_char_dim=8,
        byte_router_hidden=32,
        byte_chunk_mode="utf8",
        use_static_m_cap=True,
    )
    teacher = SimVQSplitLM(cfg).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = SimVQSplitLM(cfg)
    student.load_state_dict(teacher.state_dict(), strict=True)
    teacher.byte_chunker.lock_router(True)
    student.byte_chunker.lock_router(True)
    teacher.set_crystal_schedule(1, 1)
    student.set_crystal_schedule(1, 1)
    teacher.set_snap_alpha(1.0)
    student.set_snap_alpha(1.0)
    student.set_qat_mix(1.0)
    student.set_ternary_qat(True)

    snap = _snapshot_router_params(student)
    assert len(snap) > 0
    gain_q, cryst_q, chunk_q, core_q = _split_qat_4groups(student)
    all_groups = gain_q + cryst_q + chunk_q + core_q
    assert len(all_groups) == len({id(p) for p in all_groups})
    assert all(p.requires_grad for p in all_groups)
    # Every RMSNorm trainable gain parameter must reside in gain_q
    gain_ids = {id(p) for p in gain_q}
    for mod in student.modules():
        if isinstance(mod, RMSNorm) and mod.weight.requires_grad:
            assert id(mod.weight) in gain_ids

    xb = torch.tensor([[1] + list(b"Check 4-tap block distil. \xf0\x9f\x9a\x80") + [2]], dtype=torch.long)
    mask = torch.ones_like(xb, dtype=torch.bool)
    mask[:, -1] = False
    taps_f, taps_q = [], []
    hk_f = _attach_tap_hooks(teacher, taps_f, detach=True)
    hk_q = _attach_tap_hooks(student, taps_q, detach=False)
    try:
        with torch.no_grad():
            _ = teacher(xb)
        _ = student(xb)
        assert len(taps_f) == 4 and len(taps_q) == 4
        loss = _compute_tap_loss(taps_f, taps_q, last_mask=mask)
        assert torch.isfinite(loss) and float(loss.item()) >= 0.0
        loss.backward()
    finally:
        for h in hk_f + hk_q:
            h.remove()

    assert student.norm_f.weight.grad is not None and float(student.norm_f.weight.grad.abs().sum().item()) > 0.0
    assert _assert_router_unchanged(student, snap, "test_h1") == 0.0

    blk = _measure_block_alignment(teacher, student, "cpu")
    assert tuple(blk.keys()) == ("enc_mlp", "pre3", "mid_mlp", "post0", "norm_f")
    for k_blk, stats in blk.items():
        assert -1.0 <= float(stats["cos"]) <= 1.0 and float(stats["std_ratio"]) > 0.0, f"Bad stats `{k_blk}`: {stats}"


def test_h2_phase_h_lut_checkpoint_roundtrip_and_utf8_invariants(tmp_path: Path):
    assert PHASE_H_LUT_CKPT.exists(), f"Missing Phase H LUT checkpoint: {PHASE_H_LUT_CKPT}"
    lut_m, cfg, tok, _ = load_packed_lut_checkpoint(PHASE_H_LUT_CKPT, device="cpu", backend="pytorch_vec")
    assert count_params(SimVQSplitLM(cfg)) == 17_488_961
    fast_layers = [m for m in lut_m.modules() if isinstance(m, FastPackedLUT2BitLinear)]
    assert len(fast_layers) == 95 and all(isinstance(m, PackedLUT2BitLinear) for m in fast_layers)

    fp = compute_2bit_lut_footprint(lut_m)
    assert abs(float(fp["total_lut_kb"]) - 5539.81) <= 0.1
    assert abs(float(fp["matrix_compression_vs_fp32"]) - 16.0) < 1e-4

    cache_64 = compute_state_cache_kb("SimVQSplitLM", cfg, None, 64)
    cache_2048 = compute_state_cache_kb("SimVQSplitLM", cfg, None, 2048)
    assert cache_64 == cache_2048 and abs(cache_2048 - 230.11) <= 0.1
    buf_b = compute_chunker_state_bytes(
        cfg.d_model, cfg.byte_char_dim, cfg.byte_router_char_dim, byte_chunk_mode=cfg.byte_chunk_mode
    )
    assert buf_b == 2164

    tmp_ckpt = tmp_path / "phase_h_rt.pt"
    save_packed_lut_checkpoint(tmp_ckpt, lut_m, cfg, tok, metadata={"phase": "H"})
    lut_rt, _, _, _ = load_packed_lut_checkpoint(tmp_ckpt, device="cpu", backend="pytorch_vec")
    multi = (
        "Проверка UTF-8 без разрывов. "
        "多语言字节流测试与符号对齐。🚀🔭🧪 "
        "def solve(x): return x * 2 + 1\n"
    ) * 3
    xb = torch.tensor([tok.encode(multi, add_bos=True, add_eos=True)[:512]], dtype=torch.long)
    with torch.no_grad():
        y1 = lut_m(xb)
        y2 = lut_rt(xb)
        _, _, ctx = lut_m.byte_chunker.encode_to_chunks(lut_m.tok_emb(xb), xb)
    assert float((y1 - y2).abs().max().item()) == 0.0
    is_cont = (xb >= 3) & (torch.bitwise_and((xb - 3).clamp(min=0), 0xC0) == 0x80)
    assert int(((ctx.b_mask[:, 1:] & ctx.valid_mask[:, 1:]) & is_cont[:, 1:]).sum().item()) == 0


def test_h3_system1_closure_beats_phase_g_json():
    assert PHASE_H_JSON.exists(), f"Missing {PHASE_H_JSON}"
    d = json.loads(PHASE_H_JSON.read_text(encoding="utf-8"))
    s12 = d["system1_qat_closure"]
    gref = d["phase_g_reference"]
    h_fp = s12["evaluations"]["tier1_h_fp"]
    h_lut = s12["evaluations"]["tier1_h_lut_5_41mb"]
    g_fp = gref["phase_g_tier1_fp"]
    g_lut = gref["phase_g_tier1_lut"]

    assert s12["gap_closure"]["beats_phase_g_all_axes"] is True
    assert s12["gap_closure"]["stop_crane_passed"] is False  # honest record of +0.2007 vs 0.175
    assert "block_alignment_v2" in s12 and float(s12["block_alignment_v2"]["mid_mlp"]["std_ratio"]) >= 0.75
    assert float(h_lut["openweb"]["max_mid_utf8_split_pct"]) == 0.0
    assert float(h_lut["val_memmap_bpb"]) < float(g_lut["val_memmap_bpb"])
    assert float(h_lut["openweb"]["mean_openweb_bpb"]) < float(g_fp["openweb"]["mean_openweb_bpb"])
    assert float(h_lut["openweb"]["mean_openweb_bpb"]) < float(g_lut["openweb"]["mean_openweb_bpb"]) - 0.30
    assert float(h_lut["gsm8k_te500"]["byte_bpb"]) < float(g_lut["gsm8k_te500"]["byte_bpb"]) - 0.12
    assert float(h_lut["s5_sum"]) >= 115.0 and float(h_fp["s5_sum"]) >= 130.0
    assert float(h_lut["overwrite"]["ood_long"]["exact_pct"]) > float(g_lut["overwrite"]["ood_long"]["exact_pct"]) + 5.0
    assert float(s12["training"]["router_max_abs_diff"]) == 0.0


def test_h4_cram_closure_single_shot_and_invariants():
    assert PHASE_H_JSON.exists(), f"Missing {PHASE_H_JSON}"
    d = json.loads(PHASE_H_JSON.read_text(encoding="utf-8"))
    t3 = d["track3_cram_gsm8k"]
    alu = t3["alu_invariants"]
    for op in ("add_acc_pct", "sub_acc_pct", "mul_acc_pct", "div_acc_pct"):
        assert float(alu[op]) == 100.0
    assert float(alu["max_norm_drift"]) < 1e-6
    assert float(t3["cram_fit_fp"]["gershgorin_margin"]) >= 4.0
    assert float(t3["cram_fit_lut"]["gershgorin_margin"]) >= 4.0

    bd = t3["blind_dev"]
    assert float(bd["lut_on_fp_solver"]["pass1_pct"]) >= 18.0
    assert float(bd["lut_2bit"]["pass1_pct"]) >= 17.0

    st = t3["sealed_test"]
    assert int(st["vault_calls_used"]) == 1
    assert int(st["fp"]["n"]) == 1319 and int(st["lut_2bit"]["n"]) == 1319
    assert float(st["lut_2bit"]["pass1_pct"]) >= 14.0
    assert float(st["lut_2bit"]["pass1_pct"]) >= float(d["phase_g_reference"]["phase_g_cram_sealed_test"]["lut_2bit"]["pass1_pct"]) + 1.0


def test_h5_ast_hygiene_and_protected_zero_diff():
    runner_p = ROOT_DIR / "scripts" / "colab_phase_h_qat_runner.py"
    src = runner_p.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(runner_p))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"eval", "exec", "compile"}

    for prot in (
        "simvq_split/",
        "experiments/cram_gsm8k/",
        "checkpoints/calcite5_tier1_17m_utf8_lut.pt",
        "tests/test_anti_cheat.py",
        "tests/test_cram_phase_c.py",
        "tests/test_cram_phase_f.py",
        "tests/test_phase_g_lut_and_scale.py",
        "benchmarks/simvq_split_results.json",
        "benchmarks/phase_b_byte_chunker_results.json",
        "benchmarks/phase_c_cram_gsm8k_results.json",
        "benchmarks/phase_e_fast_utf8_results.json",
        "benchmarks/phase_f_utf8_iterative_cram_results.json",
        "benchmarks/phase_g_colab_scaleup_results.json",
    ):
        chk = subprocess.run(
            ["git", "diff", "--exit-code", "HEAD", "--", prot],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
        )
        assert chk.returncode == 0, f"Protected artifact modified `{prot}`:\n{chk.stdout}\n{chk.stderr}"
