"""
Zero-Cheat Phase G Verification Suite (`tests/test_phase_g_lut_and_scale.py`).

Enforces 5 deterministic Phase G scaling & native kernel invariants:
  g1: `FastPackedLUT2BitLinear` numerical parity (`< 1e-5` FP32) on both `row_aligned=True`
      (`384 -> 1024`) and `row_aligned=False` (`172 -> 384`, `bias=True`) layers, subclassing
      `PackedLUT2BitLinear` and preserving `.bias_fp16` (`Law 15`).
  g2: Exact `2-bit uint8 LUT` footprints for `Base 1.48M` (`548.34 KB`, `70` layers) and
      `Tier-1 17.49M` (`5,539.81 KB`, `95` layers, `16.00x` matrix compression), plus direct
      loading and zero-diff save/load roundtrip of `checkpoints/calcite5_tier1_17m_utf8_lut.pt`.
  g3: `Tier-1` `O(1)` state cache (`230.11 KB` @ `L=2048`), constant chunker streaming buffer
      (`2,164 B` for `d_model=384` across `L=64..2048`), and `0.00%` mid-UTF8 cuts on multilingual
      UTF-8 prose with the trained `calcite5_tier1_17m_utf8_lut.pt` model.
  g4: Corpus 45-char leakage scrubber catches whitespace/case-perturbed substrings from held-out
      `GSM8K` splits while passing clean prose.
  g5: AST hygiene of `kernels/fast_packed_lut.py` and strict `0`-byte `git diff HEAD` on
      protected `simvq_split/` and `experiments/cram_gsm8k/`.
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
    accelerate_lut_model,
    get_lut_backend_info,
    load_packed_lut_checkpoint,
    pack_fast_model_2bit_lut,
    save_packed_lut_checkpoint,
)
from run_simvq_split_benchmark import compute_state_cache_kb, count_params  # noqa: E402
from scripts.colab_phase_g_runner import (  # noqa: E402
    build_heldout_substring_blocklist,
    is_leaked_doc,
    tier1_config,
)
from simvq_split.memory import compute_chunker_state_bytes  # noqa: E402
from simvq_split.models import SimVQSplitConfig, SimVQSplitLM  # noqa: E402
from simvq_split.quant_lut import (  # noqa: E402
    PackedLUT2BitLinear,
    compute_2bit_lut_footprint,
    pack_model_2bit_lut,
    quantize_ternary_weights_ls,
)
from simvq_split.tokenizer import ByteBPETokenizer  # noqa: E402


TIER1_LUT_CKPT = ROOT_DIR / "checkpoints" / "calcite5_tier1_17m_utf8_lut.pt"
BANNED_KERNEL_IMPORTS = {"re", "pickle", "marshal", "shelve"}
BANNED_KERNEL_CALLS = {"eval", "exec", "compile"}


def test_g1_fast_packed_lut_parity_and_law15():
    torch.manual_seed(2026)
    binfo = get_lut_backend_info()
    backends = ["pytorch_vec"]
    if binfo["cpu_ext_available"]:
        backends.append("cpu_ext")
    if torch.cuda.is_available() and binfo["cuda_ext_available"]:
        backends.append("cuda_ext")

    for in_f, out_f, use_bias, expect_aligned in (
        (384, 1024, False, True),
        (172, 384, True, False),
    ):
        lin = nn.Linear(in_f, out_f, bias=use_bias)
        with torch.no_grad():
            wq, _, _ = quantize_ternary_weights_ls(lin.weight.data)
            lin.weight.data.copy_(wq)
            if lin.bias is not None:
                lin.bias.data.normal_(0.0, 0.25)

        ref = PackedLUT2BitLinear.from_linear(lin).eval()
        assert isinstance(ref, PackedLUT2BitLinear)
        if use_bias:
            assert hasattr(ref, "bias_fp16") and ref.bias_fp16 is not None

        x_cpu = torch.randn(4, 32, in_f, dtype=torch.float32)
        with torch.no_grad():
            y_ref = ref(x_cpu)

        for bk in backends:
            fast = FastPackedLUT2BitLinear.from_packed(ref, backend=bk).eval()
            assert isinstance(fast, PackedLUT2BitLinear)
            assert fast.row_aligned == expect_aligned
            if use_bias:
                assert fast.bias_fp16 is not None
                assert torch.equal(fast.bias_fp16.cpu(), ref.bias_fp16.cpu())

            dev = "cuda" if bk == "cuda_ext" else "cpu"
            fast_d = fast.to(dev)
            with torch.no_grad():
                y_fast = fast_d(x_cpu.to(dev)).float().cpu()
            diff = float((y_ref.float() - y_fast).abs().max().item())
            assert diff < 1e-5, f"{bk} diff={diff} for ({in_f}->{out_f})"

            # Ensure `.float()` casting never corrupts internal scale buffers
            fast_f32 = fast_d.float()
            with torch.no_grad():
                y_f32 = fast_f32(x_cpu.to(dev)).float().cpu()
            assert float((y_ref.float() - y_f32).abs().max().item()) < 1e-5

            # Verify post-forward `load_state_dict` invalidates cached scales via post-hook
            lin2 = nn.Linear(in_f, out_f, bias=use_bias)
            with torch.no_grad():
                wq2, _, _ = quantize_ternary_weights_ls(lin2.weight.data * 2.5)
                lin2.weight.data.copy_(wq2)
                if lin2.bias is not None:
                    lin2.bias.data.normal_(0.0, 0.5)
            ref2 = PackedLUT2BitLinear.from_linear(lin2).eval()
            fast_d.load_state_dict(ref2.state_dict(), strict=True)
            with torch.no_grad():
                y_new = fast_d(x_cpu.to(dev)).float().cpu()
                y_ref2 = ref2(x_cpu).float().cpu()
            assert float((y_ref2 - y_new).abs().max().item()) < 1e-5
            assert float((y_fast - y_new).abs().max().item()) > 1e-3


def test_g2_tier1_lut_footprint_and_checkpoint_roundtrip(tmp_path: Path):
    torch.manual_seed(2026)
    cfg_t1 = tier1_config()
    m_t1 = SimVQSplitLM(cfg_t1).eval()
    assert count_params(m_t1) == 17_488_961
    m_t1.byte_chunker.lock_router(True)
    m_t1.set_crystal_schedule(1, 1)

    with torch.no_grad():
        for nm, mod in m_t1.named_modules():
            if isinstance(mod, nn.Linear) and nm != "lm_head":
                if not m_t1.byte_chunker.is_locked_router_module(mod):
                    wt, _, _ = quantize_ternary_weights_ls(mod.weight.data)
                    mod.weight.data.copy_(wt)

    lut_skel = pack_fast_model_2bit_lut(m_t1, backend="pytorch_vec").eval()
    fp_t1 = compute_2bit_lut_footprint(lut_skel)
    assert int(fp_t1["num_lut_layers"]) == 95
    assert abs(float(fp_t1["matrix_compression_vs_fp32"]) - 16.0) < 1e-4
    assert abs(float(fp_t1["total_lut_kb"]) - 5539.81) <= 0.1

    base_ckpt = torch.load(
        ROOT_DIR / "checkpoints" / "simvq_split_utf8_openweb.pt",
        map_location="cpu",
        weights_only=False,
    )
    cfg_b = SimVQSplitConfig(**base_ckpt["simvq_split_config"])
    m_b = SimVQSplitLM(cfg_b).eval()
    m_b.load_state_dict(base_ckpt["simvq_split_ternary_qat_state_dict"], strict=True)
    m_b.byte_chunker.lock_router(True)
    m_b.set_crystal_schedule(1, 1)
    fp_b = compute_2bit_lut_footprint(pack_model_2bit_lut(m_b))
    assert int(fp_b["num_lut_layers"]) == 70
    assert abs(float(fp_b["total_lut_kb"]) - 548.34) <= 0.1

    assert TIER1_LUT_CKPT.exists(), f"Missing Colab Tier-1 LUT checkpoint: {TIER1_LUT_CKPT}"
    lut_m, loaded_cfg, tok, _ = load_packed_lut_checkpoint(
        TIER1_LUT_CKPT, device="cpu", backend="pytorch_vec"
    )
    assert count_params(SimVQSplitLM(loaded_cfg)) == 17_488_961
    fast_layers = [m for m in lut_m.modules() if isinstance(m, FastPackedLUT2BitLinear)]
    assert len(fast_layers) == 95
    assert all(isinstance(m, PackedLUT2BitLinear) for m in fast_layers)
    assert abs(float(compute_2bit_lut_footprint(lut_m)["total_lut_kb"]) - 5539.81) <= 0.1

    # Zero-diff save/load roundtrip on temp checkpoint
    tmp_ckpt = tmp_path / "roundtrip_tier1_lut.pt"
    save_packed_lut_checkpoint(tmp_ckpt, lut_m, loaded_cfg, tok, metadata={"test": 1})
    lut_rt, _, _, _ = load_packed_lut_checkpoint(tmp_ckpt, device="cpu", backend="pytorch_vec")
    xb = torch.tensor([[1] + list("Check Tier-1 5.41 MB roundtrip.".encode("utf-8"))], dtype=torch.long)
    with torch.no_grad():
        y1 = lut_m(xb)
        y2 = lut_rt(xb)
    assert float((y1 - y2).abs().max().item()) == 0.0

    # Leaf module accelerate_lut_model contract
    lin = nn.Linear(64, 32, bias=True)
    packed_leaf = PackedLUT2BitLinear.from_linear(lin)
    acc_leaf = accelerate_lut_model(packed_leaf, backend="pytorch_vec")
    assert isinstance(acc_leaf, FastPackedLUT2BitLinear)


def test_g3_tier1_utf8_zero_split_and_cache_invariants():
    assert TIER1_LUT_CKPT.exists(), f"Missing {TIER1_LUT_CKPT}"
    lut_m, cfg, tok, _ = load_packed_lut_checkpoint(
        TIER1_LUT_CKPT, device="cpu", backend="pytorch_vec"
    )
    cache_kb_64 = compute_state_cache_kb("SimVQSplitLM", cfg, None, 64)
    cache_kb_2048 = compute_state_cache_kb("SimVQSplitLM", cfg, None, 2048)
    assert cache_kb_64 == cache_kb_2048, f"Cache must be O(1): {cache_kb_64} vs {cache_kb_2048}"
    assert abs(cache_kb_2048 - 230.11) <= 0.1, f"Unexpected Tier-1 cache_kb@2048: {cache_kb_2048}"

    # Verify constant 2,164 B (`d_model=384`) chunker streaming buffer across L=64..2048
    exp_st_b = compute_chunker_state_bytes(
        cfg.d_model,
        cfg.byte_char_dim,
        cfg.byte_router_char_dim,
        byte_chunk_mode=cfg.byte_chunk_mode,
    )
    assert exp_st_b == 2164
    obs_bytes: list[int] = []
    with torch.no_grad():
        for seq_L in (64, 256, 2048):
            ids = tok.encode(("Проверка русского текста, 漢字 и 🚀!\n" * 60), add_bos=True)[:seq_L]
            xs = torch.tensor([ids], dtype=torch.long)
            embs = lut_m.tok_emb(xs)
            ps, bs, _, st_b = lut_m.byte_chunker.forward_streaming_check(embs, xs)
            obs_bytes.append(int(st_b))
            _, _, ctx = lut_m.byte_chunker.encode_to_chunks(embs, xs)
            assert torch.equal(ctx.b_mask, bs)
            assert int(st_b) == exp_st_b == 2164
            assert (ctx.p_prob - ps).abs().max().item() < 5e-3
    assert len(set(obs_bytes)) == 1

    # Verify 0.00% mid-UTF8 cuts on multilingual sample (RU + CJK + 4-byte Emoji + Python)
    multi = (
        "Алиса посчитала 125 яблок и разделила на 5 корзин. "
        "数学计算必须保持高精度与零泄漏。🚀✨🔥 "
        "def solve(x): return (x + 12) // 3\n"
    ) * 4
    x_ids = torch.tensor([tok.encode(multi, add_bos=True, add_eos=True)[:512]], dtype=torch.long)
    with torch.no_grad():
        _, _, ctx_u = lut_m.byte_chunker.encode_to_chunks(lut_m.tok_emb(x_ids), x_ids)
    is_byte = x_ids >= 3
    raw_v = (x_ids - 3).clamp(min=0)
    is_cont = is_byte & (torch.bitwise_and(raw_v, 0xC0) == 0x80)
    b_int = ctx_u.b_mask[:, 1:] & ctx_u.valid_mask[:, 1:]
    n_mid = int((b_int & is_cont[:, 1:]).sum().item())
    assert n_mid == 0, f"Detected {n_mid} illegal mid-UTF8 splits in Tier-1!"


def test_g4_corpus_45char_leakage_scrubber():
    blocklist = build_heldout_substring_blocklist(ROOT_DIR)
    assert len(blocklist) >= 30_000, f"Blocklist unexpectedly small: {len(blocklist)}"

    te_p = ROOT_DIR / "benchmarks" / "gsm8k" / "test.jsonl"
    with te_p.open("r", encoding="utf-8") as f:
        sample_q = str(json.loads(next(ln for ln in f if ln.strip()))["question"])
    perturbed = "Intro noise.\n  " + "   ".join(sample_q.upper().split()) + "\nFooter text."
    assert is_leaked_doc(perturbed, blocklist), "45-char scrubber missed whitespace/case-perturbed test item!"

    clean = (
        "На кухне стоял старый медный чайник, за окном падал мягкий ноябрьский снег, "
        "а кот тихо спал на подоконнике рядом с учебником по линейной алгебре."
    )
    assert not is_leaked_doc(clean, blocklist), "45-char scrubber false-positived on clean prose!"

    # Also verify 24..44-char blocklist entries remain matchable inside longer text
    short_blk = {"short heldout needle thirty chars!"}
    assert is_leaked_doc("Prefix noise. Short heldout needle thirty chars! Suffix noise.", short_blk)


def test_g5_core_and_cram_untouched_ast():
    k_path = ROOT_DIR / "kernels" / "fast_packed_lut.py"
    src = k_path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename="fast_packed_lut.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in BANNED_KERNEL_IMPORTS
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in BANNED_KERNEL_IMPORTS
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in BANNED_KERNEL_CALLS

    for prot in ("simvq_split/", "experiments/cram_gsm8k/"):
        chk = subprocess.run(
            ["git", "diff", "--exit-code", "HEAD", "--", prot],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
        )
        assert chk.returncode == 0, f"Protected core `{prot}` modified:\n{chk.stdout}\n{chk.stderr}"
