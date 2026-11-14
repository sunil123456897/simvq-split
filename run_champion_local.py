"""
Автономный локальный загрузчик, распаковщик и стенд проверки тернарных чемпионов:
1. `c4_b0_a4_coprime_lock` (Победитель Dream-RSI Cycle 4: S=99.61%, T=64: 99.61%, T=1024: 95.31%)
2. `m_nl_transfer` (Билингвальная RU/EN + Python Code модель с мгновенной 0-SGD инъекцией 2000 правил API)
"""

import copy
import io
import lzma
import os
import random
import time
import torch
import torch.nn as nn

from simvq_coprime_gpt import (
    MAX_CLAUSE_WORDS,
    NL_OOD_RULES,
    NUM_FACTS,
    NUM_REGS,
    NUM_SRV_OOD,
    NUM_SRV_TRAIN,
    PAD_ID,
    REG_NAMES,
    ZEROSHOT_INIT_STATES,
    UnifiedLoopedTransformer_v104,
    apply_op_logic,
    generate_stream_dataset,
    make_clause_tokens,
    state_to_regs,
)


def unpack_trits_uint8(d):
    v = d["u8"].to(torch.int16)
    t0 = v % 3
    v = v // 3
    t1 = v % 3
    v = v // 3
    t2 = v % 3
    v = v // 3
    t3 = v % 3
    v = v // 3
    t4 = v % 3
    flat = torch.stack([t0, t1, t2, t3, t4], dim=-1).reshape(-1)[: d["orig_len"]] - 1
    return flat.to(torch.int8).view(*d["shape"])


def unpack_ternary_checkpoint_v2(mod, packed):
    mod_cpu = mod.cpu()
    sd = mod_cpu.state_dict()
    for k, v in packed["other"].items():
        if k in sd:
            sd[k].copy_(v.to(sd[k].dtype))
    for name, bdata in packed["bitlinear"].items():
        sign_t = (
            unpack_trits_uint8(bdata["trits"]) if "trits" in bdata else bdata["sign"]
        )
        mean_abs_sign = sign_t.abs().float().mean().clamp(min=1e-6)
        w_exact = sign_t.float() * (bdata["scale"].float() / mean_abs_sign)
        sd[name + ".weight"].copy_(w_exact)
        if "bias" in bdata and (name + ".bias") in sd:
            sd[name + ".bias"].copy_(bdata["bias"])
    mod_cpu.load_state_dict(sd)
    mod_cpu.set_quant_alpha(1.0)
    mod_cpu.snap_strength = 1.0
    return mod_cpu


import inspect

def load_local_champions(
    bundle_path=os.path.join(
        os.path.dirname(__file__), "checkpoints", "simvq_champions_compact.pt.xz"
    ),
    device="cpu",
):
    if not os.path.exists(bundle_path) and bundle_path.endswith(".xz") and os.path.exists(bundle_path[:-3]):
        bundle_path = bundle_path[:-3]

    with open(bundle_path, "rb") as f:
        file_bytes = f.read()

    try:
        raw_bytes = lzma.decompress(file_bytes)
    except lzma.LZMAError:
        raw_bytes = file_bytes

    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    bundle = torch.load(io.BytesIO(raw_bytes), **load_kwargs)

    vocab_base = bundle["vocab_base"]
    vocab_nl = bundle["vocab_nl"]

    m_c4 = UnifiedLoopedTransformer_v104(
        len(vocab_base), d_model=640, n_heads=8, mode="simvq_ours"
    )
    unpack_ternary_checkpoint_v2(m_c4, bundle["c4_core"])

    m_nl = copy.deepcopy(m_c4)
    # m_nl uses its aligned multilingual token embeddings + 0-SGD M_fast memory with
    # unbounded segment mask (force_full_mask=True) and base core weights:
    m_nl.chunker.force_full_mask = True
    m_nl.chunker.epi_temp_scale = 1.0
    m_nl.wy1 = 0.35
    m_nl.wy2 = 0.45
    m_nl.rot_w = 0.25
    new_emb = nn.Embedding(len(vocab_nl), m_nl.d_model)
    with torch.no_grad():
        new_emb.weight.copy_(bundle["nl_tok_emb"].to(new_emb.weight.dtype))
    m_nl.tok_emb = new_emb

    return m_c4.to(device).eval(), m_nl.to(device).eval(), vocab_base, vocab_nl


def make_nl_code_clause(
    op_type, val, w2i_nl, alpha_val=0, beta_val=0, mode="mix_ru_py"
):
    if mode == "mix_ru_py":
        mode = random.choice(["ru", "py_code", "en1"])
    end_tok = "\n" if mode == "py_code" else "."
    if op_type == "init":
        r_toks = [f"{REG_NAMES[g]}={val[g]}" for g in range(NUM_REGS)] + [
            f"alpha={alpha_val}",
            f"beta={beta_val}",
        ]
        if mode == "ru":
            base = ["Вопрос:", "Инженер", "запускает", "контур"] + r_toks
        elif mode == "py_code":
            base = ["def", "trace():", "init"] + r_toks
        else:
            base = ["Question:", "Alice", "starts", "with"] + r_toks
        return [w2i_nl[w] for w in base[: MAX_CLAUSE_WORDS - 1] + [end_tok]]
    v_s = str(val)
    if mode == "ru":
        if op_type == "s5_cycle5":
            base = ["Затем", "дежурный", "вращает", "кольцо", "s5_cycle5"]
        elif op_type == "s5_coxeter":
            base = ["Затем", "агент", "меняет", "пары", "s5_coxeter"]
        elif op_type == "s5_alt3":
            base = ["Далее", "дежурный", "сдвигает", "тройку", "s5_alt3"]
        elif op_type == "add_ace":
            base = ["Затем", "агент", "прибавляет", v_s, "a,c,e"]
        elif op_type == "add_bd":
            base = ["Далее", "балансировщик", "увеличивает", "b,d", "by", v_s]
        elif op_type == "add_alpha":
            base = ["Затем", "агент", "прибавляет", "квоту_альфа", "b,d"]
        elif op_type == "add_beta":
            base = ["Далее", "балансировщик", "прибавляет", "политику_бета", "a,c,e"]
        elif op_type == "if_s5":
            base = ["Если", "a>=b", "s5_cycle5", v_s, "иначе", "s5_coxeter"]
        elif op_type == "ca_rule110":
            base = ["Затем", "балансировщик", "обновляет", "каскад", "ca_rule110"]
        elif op_type == "srv":
            base = [
                "Затем",
                "агент",
                "вызывает",
                "микросервис",
                f"API_PATCH_{val:02d}",
            ]
    elif mode == "py_code":
        if op_type == "s5_cycle5":
            base = ["pipeline.rotate_ring()", "s5_cycle5", "(", "registers", ")"]
        elif op_type == "s5_coxeter":
            base = ["pipeline.swap_coxeter()", "s5_coxeter", "(", "registers", ")"]
        elif op_type == "s5_alt3":
            base = ["pipeline.shift_alt3()", "s5_alt3", "(", "registers", ")"]
        elif op_type == "add_ace":
            base = ["a,c,e+=", v_s, "%", "5"]
        elif op_type == "add_bd":
            base = ["b,d+=", v_s, "%", "5"]
        elif op_type == "add_alpha":
            base = ["b,d+=", "quota_alpha", "const_alpha"]
        elif op_type == "add_beta":
            base = ["a,c,e+=", "policy_beta", "const_beta"]
        elif op_type == "if_s5":
            base = ["if", "a>=b:", "s5_cycle5", v_s, "else:", "s5_coxeter"]
        elif op_type == "ca_rule110":
            base = ["balancer.cascade_rule110()", "ca_rule110", "(", "cell_ring", ")"]
        elif op_type == "srv":
            base = ["gateway.call_api(", f"API_PATCH_{val:02d}", ")"]
    else:
        if op_type == "srv":
            base = ["Then", "she", "queries", "server", f"API_PATCH_{val:02d}"]
            return [w2i_nl[w] for w in base[: MAX_CLAUSE_WORDS - 1] + [end_tok]]
        return make_clause_tokens(
            op_type, val, alpha_val, beta_val, mode="en1", allow_filler=False
        )
    return [w2i_nl[w] for w in base[: MAX_CLAUSE_WORDS - 1] + [end_tok]]


def generate_nl_code_dataset(
    num_samples,
    w2i_nl,
    num_steps=2,
    fact_range=(0, NUM_SRV_TRAIN),
    init_pool=ZEROSHOT_INIT_STATES,
    srv_prob=0.25,
    nl_ood=False,
    style_mode="mix_ru_py",
    dev="cpu",
):
    max_stream_len = (num_steps + 1) * MAX_CLAUSE_WORDS
    X_list, Y_list = [], []
    f_min, f_max = fact_range
    ops_pool = [
        "s5_cycle5",
        "s5_coxeter",
        "s5_alt3",
        "add_ace",
        "add_bd",
        "if_s5",
        "ca_rule110",
        "add_alpha",
        "add_beta",
    ]
    for _ in range(num_samples):
        s0 = random.choice(init_pool)
        av, bv = random.randint(0, 4), random.randint(0, 4)
        regs = state_to_regs(s0)
        cur_style = (
            random.choice(["ru", "py_code", "en1"])
            if style_mode == "mix_ru_py"
            else style_mode
        )
        stream_toks = list(
            make_nl_code_clause("init", regs, w2i_nl, av, bv, mode=cur_style)
        )
        y_sample = [list(regs)]
        for _ in range(1, num_steps + 1):
            step_style = (
                random.choice(["ru", "py_code", "en1"])
                if style_mode == "mix_ru_py"
                else style_mode
            )
            if random.random() >= srv_prob:
                op_t = random.choice(ops_pool)
                val = (
                    0
                    if op_t
                    in (
                        "s5_cycle5",
                        "s5_coxeter",
                        "s5_alt3",
                        "ca_rule110",
                        "add_alpha",
                        "add_beta",
                    )
                    else (
                        random.choice([1, 2])
                        if op_t == "if_s5"
                        else random.randint(0, 2)
                    )
                )
            else:
                op_t, val = "srv", random.randint(f_min, f_max - 1)
            regs = apply_op_logic(regs, op_t, val, av, bv, nl_ood=nl_ood)
            stream_toks.extend(
                make_nl_code_clause(op_t, val, w2i_nl, av, bv, mode=step_style)
            )
            y_sample.append(list(regs))
        stream_toks.extend([PAD_ID] * (max_stream_len - len(stream_toks)))
        X_list.append(stream_toks)
        Y_list.append(y_sample)
    return torch.tensor(X_list, dtype=torch.long, device=dev), torch.tensor(
        Y_list, dtype=torch.long, device=dev
    )


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    m_c4, m_nl, vocab_base, vocab_nl = load_local_champions(device=dev)
    w2i_nl = {w: i for i, w in enumerate(vocab_nl)}
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    torch.save(m_c4.state_dict(), os.path.join(ckpt_dir, "c4_b0_a4_coprime_lock.pt"))
    torch.save(
        {"state_dict": m_nl.state_dict(), "vocab_nl": vocab_nl},
        os.path.join(ckpt_dir, "m_nl_transfer_ru_en_py.pt"),
    )
    print(
        f"[Local Champions Loaded & Unpacked in {time.time() - t0:.2f}s on {dev.upper()}]"
    )
    print(
        "  Saved: checkpoints/c4_b0_a4_coprime_lock.pt & checkpoints/m_nl_transfer_ru_en_py.pt"
    )

    torch.manual_seed(999)
    random.seed(999)
    # 1. Verify c4_b0_a4_coprime_lock locally on T=16
    bx_base, by_base = generate_stream_dataset(
        64, num_steps=16, init_pool=ZEROSHOT_INIT_STATES
    )
    bx_base, by_base = bx_base.to(dev), by_base.to(dev)
    with torch.no_grad():
        logits_base, _ = m_c4(bx_base, num_steps=16)
    acc_base16 = (logits_base[:, -1].argmax(dim=-1) == by_base[:, -1]).all(
        dim=-1
    ).float().mean().item() * 100.0

    # 2. Verify m_nl_transfer locally on Mixed RU+EN+Python (T=16)
    bx_nl, by_nl = generate_nl_code_dataset(
        64, w2i_nl, num_steps=16, style_mode="mix_ru_py", dev=dev
    )
    with torch.no_grad():
        logits_nl, _ = m_nl(bx_nl, num_steps=16)
    acc_nl16 = (logits_nl[:, -1].argmax(dim=-1) == by_nl[:, -1]).all(
        dim=-1
    ).float().mean().item() * 100.0

    # 3. Verify 0-SGD API Injection of 2,000 rules locally on Mixed RU+EN+Python
    H_target_c = m_nl.simvq.get_codebook().float().clone()
    api_tok_ids = torch.tensor(
        [w2i_nl[f"API_PATCH_{s_id:02d}"] for s_id in range(NUM_SRV_TRAIN, NUM_FACTS)],
        dtype=torch.long,
        device=dev,
    )
    with torch.no_grad():
        api_key_vecs = m_nl.tok_emb(api_tok_ids)
    nl_srcs, nl_tables = [], []
    for i in range(NUM_SRV_OOD):
        p_src, tbl = NL_OOD_RULES[NUM_SRV_TRAIN + i]
        nl_srcs.append(list(p_src))
        nl_tables.append(tbl)
    t_inj = time.time()
    m_nl.fast_op_mem.memorize_bilinear_operators(
        api_key_vecs,
        torch.tensor(nl_srcs, dtype=torch.long, device=dev),
        torch.stack(nl_tables, dim=0).to(dev),
        H_target_c,
    )
    inj_ms = (time.time() - t_inj) * 1000.0
    bx_ood, by_ood = generate_nl_code_dataset(
        64,
        w2i_nl,
        num_steps=8,
        fact_range=(NUM_SRV_TRAIN, NUM_FACTS),
        srv_prob=0.8,
        nl_ood=True,
        style_mode="mix_ru_py",
        dev=dev,
    )
    with torch.no_grad():
        logits_ood, _ = m_nl(bx_ood, num_steps=8)
    acc_ood8 = (logits_ood[:, -1].argmax(dim=-1) == by_ood[:, -1]).all(
        dim=-1
    ).float().mean().item() * 100.0

    # 4. Verify Auto-Dream-RSI Cycle 5 Long-Horizon OOS (T=1024 & m_nl OOD-2000 T=128)
    torch.manual_seed(9005)
    random.seed(9005)
    bx_1024, by_1024 = generate_stream_dataset(
        32, num_steps=1024, init_pool=ZEROSHOT_INIT_STATES, device=dev
    )
    with torch.no_grad():
        logits_1024, _ = m_c4(bx_1024, num_steps=1024)
    acc_base1024 = (logits_1024[:, -1].argmax(dim=-1) == by_1024[:, -1]).all(
        dim=-1
    ).float().mean().item() * 100.0

    torch.manual_seed(2026)
    random.seed(2026)
    bx_ood128, by_ood128 = generate_nl_code_dataset(
        64,
        w2i_nl,
        num_steps=128,
        fact_range=(NUM_SRV_TRAIN, NUM_FACTS),
        srv_prob=0.5,
        nl_ood=True,
        style_mode="mix_ru_py",
        dev=dev,
    )
    with torch.no_grad():
        logits_ood128, _ = m_nl(bx_ood128, num_steps=128)
    acc_ood128 = (logits_ood128[:, -1].argmax(dim=-1) == by_ood128[:, -1]).all(
        dim=-1
    ).float().mean().item() * 100.0

    print(
        f"  [Local Smoke Test] c4_b0_a4_coprime_lock (T=16): {acc_base16:.2f}% | "
        f"OOS (T=1024): {acc_base1024:.2f}% | "
        f"m_nl_transfer RU+EN+PY (T=16): {acc_nl16:.2f}% | "
        f"0-SGD 2000 API Rules (T=8): {acc_ood8:.2f}%, (T=128): {acc_ood128:.2f}% (injected in {inj_ms:.1f} ms)"
    )
    assert (
        acc_base16 == 100.0
        and acc_base1024 == 100.0
        and acc_nl16 == 100.0
        and acc_ood8 == 100.0
        and acc_ood128 == 100.0
    ), "Regression detected in SimVQCoprimeGPT local champions!"
