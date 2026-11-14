"""
Stage v12.0: FreeFormBranchingSimVQGPT
======================================
Расширение замороженного тернарного чемпиона `c4_b0_a4_coprime_lock` (S5 |>< Z5^5)
до свободного нешаблонного русского языка и ветвящегося/циклического Python-кода:
1. Permutation-Invariant Clause 0: произвольный порядок (`random.shuffle`) инициализации
   регистров `a..e`, констант `alpha, beta` и объявления подпрограммы `def_proc(...)`
   среди разговорного русского текста и Python kwargs (`MAX_V12_CLAUSE_WORDS = 20`).
2. Свободный нешаблонный русский язык (разговорные вставки, синонимы, инверсия порядка слов).
3. Циклы (`for _k in range(k):` / `повтори k раза подряд`, k in {1, 2, 3, 4}):
   динамическая развёртка тернарного ядра + SimVQ-аттрактора на k тактов внутри одного шага
   (включая вызовы 0-SGD инъектированных микросервисов `API_PATCH_08..23` внутри цикла).
4. Вложенные ветвления (`if a>=b and c<d: OP1 elif a>=b: OP2 else: OP3` /
   `если одновременно a>=b и c<d ... иначе_если a>=b ... а_иначе ...`).
5. Стек вызовов подпрограмм (`def_proc(proc_rot_alpha | proc_cox_beta)` в Clause 0 +
   `call_stack.invoke()` / `вызови_из_стека` на шаге t): 2-тактовая композитная развёртка
   через замороженное тернарное ядро и SimVQ.
"""

import os
import time
import math
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from simvq_coprime_gpt import (
    NUM_STATES,
    NUM_REGS,
    VALS_PER_REG,
    REG_NAMES,
    NUM_FACTS,
    NUM_SRV_TRAIN,
    NUM_SRV_OOD,
    NL_OOD_RULES,
    TRAIN_INIT_STATES,
    ZEROSHOT_INIT_STATES,
    state_to_regs,
    apply_op_logic,
    make_clause_tokens,
    VOCAB,
)
from run_champion_local import load_local_champions

MAX_V12_CLAUSE_WORDS = 20
PAD_ID = 0

# --- Словарь v12.0 (базовый + нешаблонный RU + Python + ветвления/циклы/стек) ---
V12_EXTRA_TOKENS = [
    "\n",
    "(",
    ")",
    ":",
    ",",
    ";",
    "%",
    "5",
    "def",
    "trace():",
    "init",
    "trace_init(",
    "registers",
    "cell_ring",
    "const_alpha",
    "const_beta",
    "quota_alpha",
    "policy_beta",
    "pipeline.rotate_ring()",
    "pipeline.swap_coxeter()",
    "pipeline.shift_alt3()",
    "balancer.cascade_rule110()",
    "gateway.call_api(",
    "a,c,e+=",
    "b,d+=",
    "if",
    "elif",
    "else:",
    "and",
    "a>=b:",
    "c<d:",
    "a>=b",
    "c<d",
    "for",
    "_k",
    "in",
    "range(1):",
    "range(2):",
    "range(3):",
    "range(4):",
    "def_proc(",
    "proc_rot_alpha",
    "proc_cox_beta",
    "call_stack.invoke()",
    "Вопрос:",
    "Инженер",
    "запускает",
    "контур",
    "Старт:",
    "Система",
    "начинает",
    "работу",
    "Пусть",
    "пусть",
    "сначала",
    "заданы",
    "параметры",
    "на_входе",
    "пожалуйста",
    "сейчас",
    "в_контуре",
    "установим",
    "значения",
    "для_запуска",
    "регистры",
    "равны",
    "Затем",
    "Далее",
    "После_этого",
    "Теперь",
    "Следом",
    "На_очередном_шаге",
    "дежурный",
    "агент",
    "балансировщик",
    "оператор",
    "контроллер",
    "планировщик",
    "ядро",
    "вращает",
    "прокручивает",
    "сдвигает_по_кругу",
    "кольцо",
    "регистров",
    "меняет",
    "переставляет",
    "инвертирует",
    "пары",
    "соседей",
    "сдвигает",
    "перемешивает",
    "циклически_меняет",
    "тройку",
    "первых_регистров",
    "прибавляет",
    "добавляет",
    "увеличивает",
    "смещает",
    "на",
    "к",
    "регистрам",
    "по_модулю_5",
    "квоту_альфа",
    "константу_альфа",
    "политику_бета",
    "константу_бета",
    "обновляет",
    "пересчитывает",
    "прогоняет",
    "каскад",
    "автомата",
    "вызывает",
    "запрашивает",
    "дергает",
    "обращается_в",
    "микросервис",
    "шлюз",
    "повтори",
    "выполни_в_цикле",
    "прокрути",
    "1_раз",
    "2_раза",
    "3_раза",
    "4_раза",
    "один_раз",
    "дважды",
    "трижды",
    "четырежды",
    "подряд",
    "шага_подряд",
    "Если",
    "если",
    "одновременно",
    "и",
    "то",
    "иначе_если",
    "а_иначе",
    "иначе",
    "выполни",
    "определи_процедуру",
    "запиши_в_стек",
    "вызови_из_стека",
    "запусти_процедуру",
    "выполни_подпрограмму",
    "0",
    "1",
    "2",
    "3",
    "4",
    "a,c,e",
    "b,d",
] + [f"API_PATCH_{i:02d}" for i in range(NUM_FACTS)]

VOCAB_V12 = list(VOCAB)
for _t in V12_EXTRA_TOKENS:
    if _t not in VOCAB_V12:
        VOCAB_V12.append(_t)
W2I_V12 = {w: i for i, w in enumerate(VOCAB_V12)}

SINGLE_OPS = [
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
S5_BRANCH_OPS = ["s5_cycle5", "s5_coxeter", "s5_alt3"]
PROC_NAMES = ["proc_rot_alpha", "proc_cox_beta"]

# 22 канонические операции ядра c4_b0_a4_coprime_lock
CANON_OP_LIST = (
    [("s5_cycle5", 0), ("s5_coxeter", 0), ("s5_alt3", 0)]
    + [("add_ace", v) for v in range(3)]
    + [("add_bd", v) for v in range(3)]
    + [("add_alpha", 0), ("add_beta", 0)]
    + [("if_s5", 1), ("if_s5", 2)]
    + [("ca_rule110", 0)]
    + [("srv", s) for s in range(NUM_SRV_TRAIN)]
)
CANON_OP_TO_ID = {op_val: i for i, op_val in enumerate(CANON_OP_LIST)}
NUM_CANON_OPS = len(CANON_OP_LIST)


def apply_op_v12(
    regs,
    op_type,
    val,
    alpha_val=0,
    beta_val=0,
    proc_id=0,
    loop_k=1,
    branch_ops=None,
    nl_ood=False,
):
    """Точная математическая семантика шага v12.0 над S5 |>< Z5^5."""
    out = list(regs)
    for _ in range(loop_k):
        if op_type == "nested_if":
            b0, b1, b2 = branch_ops
            if out[0] >= out[1] and out[2] < out[3]:
                chosen_op = b0
            elif out[0] >= out[1]:
                chosen_op = b1
            else:
                chosen_op = b2
            out = apply_op_logic(out, chosen_op, 0, alpha_val, beta_val, nl_ood=nl_ood)
        elif op_type == "call_proc":
            if proc_id == 0:
                out = apply_op_logic(
                    out, "s5_cycle5", 0, alpha_val, beta_val, nl_ood=nl_ood
                )
                out = apply_op_logic(
                    out, "add_alpha", 0, alpha_val, beta_val, nl_ood=nl_ood
                )
            else:
                out = apply_op_logic(
                    out, "s5_coxeter", 0, alpha_val, beta_val, nl_ood=nl_ood
                )
                out = apply_op_logic(
                    out, "add_beta", 0, alpha_val, beta_val, nl_ood=nl_ood
                )
        else:
            out = apply_op_logic(out, op_type, val, alpha_val, beta_val, nl_ood=nl_ood)
    return out


def _pad_clause(tok_names, end_tok="."):
    trimmed = tok_names[: MAX_V12_CLAUSE_WORDS - 1] + [end_tok]
    ids = [W2I_V12[w] for w in trimmed]
    if len(ids) < MAX_V12_CLAUSE_WORDS:
        ids.extend([PAD_ID] * (MAX_V12_CLAUSE_WORDS - len(ids)))
    return ids


def make_v12_init_clause(
    regs, alpha_val, beta_val, proc_id=0, style="free_ru", permute=True
):
    if style == "mix":
        style = random.choice(["free_ru", "py_code"])
    end_tok = "\n" if style == "py_code" else "."
    assign_toks = [f"{REG_NAMES[g]}={regs[g]}" for g in range(NUM_REGS)] + [
        f"alpha={alpha_val}",
        f"beta={beta_val}",
    ]
    proc_tok = PROC_NAMES[proc_id]
    if permute:
        random.shuffle(assign_toks)

    if style == "py_code":
        idx = random.randint(0, len(assign_toks)) if permute else len(assign_toks)
        items = assign_toks[:idx] + ["def_proc(", proc_tok, ")"] + assign_toks[idx:]
        prefix = random.choice(
            [
                ["def", "trace():", "init"],
                ["trace_init("],
                ["init", "registers", ":"],
            ]
        )
        return _pad_clause(prefix + items, end_tok=end_tok)
    else:
        ru_openers = [
            ["Вопрос:", "пожалуйста", "сначала", "установим"],
            ["Старт:", "пусть", "сейчас", "в_контуре", "заданы"],
            ["Инженер", "запускает", "контур", "на_входе"],
            ["Система", "начинает", "работу", "регистры", "равны"],
            ["Пусть", "для_запуска", "сейчас", "установим", "значения"],
        ]
        proc_phrases = [
            ["определи_процедуру", proc_tok],
            ["запиши_в_стек", proc_tok],
        ]
        opener = list(random.choice(ru_openers))
        p_phrase = list(random.choice(proc_phrases))
        idx = random.randint(0, len(assign_toks)) if permute else len(assign_toks)
        body = assign_toks[:idx] + p_phrase + assign_toks[idx:]
        fillers = ["пожалуйста", "сейчас", "в_контуре", "на_входе", "равны"]
        if permute and len(opener) + len(body) + 2 < MAX_V12_CLAUSE_WORDS:
            f_pos = random.randint(0, len(body))
            body.insert(f_pos, random.choice(fillers))
        return _pad_clause(opener + body, end_tok=end_tok)


def _make_single_op_phrase_ru(op_type, val):
    conn = random.choice(
        ["Затем", "Далее", "После_этого", "Теперь", "Следом", "На_очередном_шаге"]
    )
    subj = random.choice(
        [
            "дежурный",
            "агент",
            "балансировщик",
            "оператор",
            "контроллер",
            "планировщик",
            "ядро",
        ]
    )
    polite = (
        [random.choice(["пожалуйста", "сейчас", "в_контуре"])]
        if random.random() < 0.5
        else []
    )
    v_s = str(val)

    if op_type == "s5_cycle5":
        v = random.choice(["вращает", "прокручивает", "сдвигает_по_кругу"])
        templates = [
            [conn] + polite + [subj, v, "кольцо", "s5_cycle5"],
            [conn, "кольцо", "регистров", "s5_cycle5", v, subj] + polite,
            [conn] + polite + [v, "s5_cycle5", "кольцо", "регистров"],
        ]
    elif op_type == "s5_coxeter":
        v = random.choice(["меняет", "переставляет", "инвертирует"])
        templates = [
            [conn] + polite + [subj, v, "пары", "s5_coxeter"],
            [conn, "пары", "соседей", "s5_coxeter", v, subj] + polite,
            [conn] + polite + [v, "пары", "s5_coxeter", "в_контуре"],
        ]
    elif op_type == "s5_alt3":
        v = random.choice(["сдвигает", "перемешивает", "циклически_меняет"])
        templates = [
            [conn] + polite + [subj, v, "тройку", "s5_alt3"],
            [conn, "тройку", "первых_регистров", "s5_alt3", v, subj],
            [conn] + polite + [v, "s5_alt3", "тройку", "регистров"],
        ]
    elif op_type == "add_ace":
        v = random.choice(["прибавляет", "добавляет", "увеличивает", "смещает"])
        templates = [
            [conn] + polite + [subj, v, "на", v_s, "к", "a,c,e"],
            [conn, "регистры", "a,c,e", subj, v, "на", v_s, "по_модулю_5"],
            [conn] + polite + [v, "a,c,e", "на", v_s],
        ]
    elif op_type == "add_bd":
        v = random.choice(["прибавляет", "добавляет", "увеличивает", "смещает"])
        templates = [
            [conn] + polite + [subj, v, "b,d", "на", v_s],
            [conn, "к", "регистрам", "b,d", v, v_s, "по_модулю_5"],
            [conn] + polite + [v, "на", v_s, "регистры", "b,d"],
        ]
    elif op_type == "add_alpha":
        v = random.choice(["прибавляет", "добавляет", "смещает"])
        c_tok = random.choice(["квоту_альфа", "константу_альфа"])
        templates = [
            [conn] + polite + [subj, v, c_tok, "к", "b,d"],
            [conn, "регистры", "b,d", v, "на", c_tok] + polite,
            [conn] + polite + [v, "b,d", c_tok, "по_модулю_5"],
        ]
    elif op_type == "add_beta":
        v = random.choice(["прибавляет", "добавляет", "смещает"])
        c_tok = random.choice(["политику_бета", "константу_бета"])
        templates = [
            [conn] + polite + [subj, v, c_tok, "к", "a,c,e"],
            [conn, "регистры", "a,c,e", v, "на", c_tok] + polite,
            [conn] + polite + [v, "a,c,e", c_tok, "по_модулю_5"],
        ]
    elif op_type == "if_s5":
        templates = [
            ["Если", "a>=b", "то", "s5_cycle5", v_s, "иначе", "s5_coxeter"] + polite,
            [
                conn,
                "если",
                "a>=b",
                "выполни",
                "s5_cycle5",
                v_s,
                "а_иначе",
                "s5_coxeter",
            ],
        ]
    elif op_type == "ca_rule110":
        v = random.choice(["обновляет", "пересчитывает", "прогоняет"])
        templates = [
            [conn] + polite + [subj, v, "каскад", "ca_rule110"],
            [conn, "каскад", "автомата", "ca_rule110", v, subj] + polite,
            [conn] + polite + [v, "ca_rule110", "каскад", "регистров"],
        ]
    elif op_type == "srv":
        v = random.choice(["вызывает", "запрашивает", "дергает", "обращается_в"])
        n = random.choice(["микросервис", "шлюз"])
        api_t = f"API_PATCH_{val:02d}"
        templates = [
            [conn] + polite + [subj, v, n, api_t],
            [conn, n, api_t, v, subj] + polite,
            [conn] + polite + [v, api_t, "в_контуре", n],
        ]
    else:
        raise ValueError(f"Unknown single op_type: {op_type}")
    return random.choice(templates)


def _make_single_op_phrase_py(op_type, val):
    v_s = str(val)
    if op_type == "s5_cycle5":
        return ["pipeline.rotate_ring()", "s5_cycle5", "(", "registers", ")"]
    elif op_type == "s5_coxeter":
        return ["pipeline.swap_coxeter()", "s5_coxeter", "(", "registers", ")"]
    elif op_type == "s5_alt3":
        return ["pipeline.shift_alt3()", "s5_alt3", "(", "registers", ")"]
    elif op_type == "add_ace":
        return ["a,c,e+=", v_s, "%", "5"]
    elif op_type == "add_bd":
        return ["b,d+=", v_s, "%", "5"]
    elif op_type == "add_alpha":
        return ["b,d+=", "quota_alpha", "const_alpha"]
    elif op_type == "add_beta":
        return ["a,c,e+=", "policy_beta", "const_beta"]
    elif op_type == "if_s5":
        return ["if", "a>=b:", "s5_cycle5", v_s, "else:", "s5_coxeter"]
    elif op_type == "ca_rule110":
        return ["balancer.cascade_rule110()", "ca_rule110", "(", "cell_ring", ")"]
    elif op_type == "srv":
        return ["gateway.call_api(", f"API_PATCH_{val:02d}", ")"]
    else:
        raise ValueError(f"Unknown py op_type: {op_type}")


def make_v12_step_clause(op_type, val=0, loop_k=1, branch_ops=None, style="free_ru"):
    if style == "mix":
        style = random.choice(["free_ru", "py_code"])
    end_tok = "\n" if style == "py_code" else "."

    if op_type == "nested_if":
        b0, b1, b2 = branch_ops
        if style == "py_code":
            toks = ["if", "a>=b", "and", "c<d:", b0, "elif", "a>=b:", b1, "else:", b2]
        else:
            conn = random.choice(["Затем", "Далее", "Теперь", "Следом"])
            polite = ["пожалуйста"] if random.random() < 0.4 else []
            toks = (
                [conn]
                + polite
                + [
                    "если",
                    "одновременно",
                    "a>=b",
                    "и",
                    "c<d",
                    "то",
                    b0,
                    "иначе_если",
                    "a>=b",
                    "то",
                    b1,
                    "а_иначе",
                    b2,
                ]
            )
        return _pad_clause(toks, end_tok=end_tok)

    if op_type == "call_proc":
        if style == "py_code":
            toks = (
                ["for", "_k", "in", f"range({loop_k}):", "call_stack.invoke()"]
                if loop_k > 1
                else ["call_stack.invoke()"]
            )
        else:
            conn = random.choice(["Затем", "Далее", "После_этого", "Теперь", "Следом"])
            subj = random.choice(["дежурный", "агент", "контроллер", "оператор"])
            call_v = random.choice(
                ["вызови_из_стека", "запусти_процедуру", "выполни_подпрограмму"]
            )
            if loop_k > 1:
                lv = random.choice(["повтори", "выполни_в_цикле", "прокрути"])
                lk_tok = random.choice(
                    [
                        f"{loop_k}_раза" if loop_k > 1 else "1_раз",
                        ["один_раз", "дважды", "трижды", "четырежды"][loop_k - 1],
                    ]
                )
                toks = [conn, subj, lv, lk_tok, "подряд", ":", call_v]
            else:
                polite = random.choice([["пожалуйста"], ["сейчас"], ["в_контуре"], []])
                toks = [conn] + polite + [subj, call_v]
        return _pad_clause(toks, end_tok=end_tok)

    if style == "py_code":
        body = _make_single_op_phrase_py(op_type, val)
        toks = ["for", "_k", "in", f"range({loop_k}):"] + body if loop_k > 1 else body
    else:
        body = _make_single_op_phrase_ru(op_type, val)
        if loop_k > 1:
            conn = random.choice(["Затем", "Далее", "Теперь", "Следом"])
            lv = random.choice(["повтори", "выполни_в_цикле", "прокрути"])
            lk_tok = random.choice(
                [
                    f"{loop_k}_раза" if loop_k > 1 else "1_раз",
                    ["один_раз", "дважды", "трижды", "четырежды"][loop_k - 1],
                ]
            )
            tail = random.choice(["подряд", "шага_подряд"])
            clean_body = body[1:] if len(body) > 1 else body
            if random.random() < 0.5:
                toks = [conn, lv, lk_tok, tail, ":"] + clean_body
            else:
                toks = [conn] + clean_body + [lv, lk_tok, tail]
        else:
            toks = body
    return _pad_clause(toks, end_tok=end_tok)


def make_canonical_base_clause(op_type, val, alpha_val=0, beta_val=0):
    return make_clause_tokens(
        op_type, val, alpha_val, beta_val, mode="en1", allow_filler=False
    )


def generate_v12_dataset(
    num_samples,
    num_steps=4,
    fact_range=(0, NUM_SRV_TRAIN),
    init_pool=ZEROSHOT_INIT_STATES,
    style_mode="mix",
    permute_init=True,
    p_loop=0.25,
    p_nested_if=0.20,
    p_call_proc=0.20,
    p_srv=0.15,
    nl_ood=False,
    dev="cpu",
):
    X_list, Y_list = [], []
    meta_proc_id = []
    meta_epi = []  # (B, 2): [alpha, beta]
    meta_mode = []  # (B, T): 0=single, 1=nested_if, 2=call_proc
    meta_loop_k = []  # (B, T): 0..3 для k=1..4
    meta_branch_ops = []  # (B, T, 3): индексы в CANON_OP_LIST (0..21)

    f_min, f_max = fact_range
    for _ in range(num_samples):
        s0 = random.choice(init_pool)
        av, bv = random.randint(0, 4), random.randint(0, 4)
        proc_id = random.randint(0, 1)
        regs = state_to_regs(s0)

        init_style = (
            random.choice(["free_ru", "py_code"]) if style_mode == "mix" else style_mode
        )
        c0_toks = make_v12_init_clause(
            regs, av, bv, proc_id=proc_id, style=init_style, permute=permute_init
        )
        stream_toks = list(c0_toks)
        y_sample = [list(regs)]

        sample_modes, sample_loops, sample_b_ops = [], [], []
        for _ in range(num_steps):
            step_style = (
                random.choice(["free_ru", "py_code"])
                if style_mode == "mix"
                else style_mode
            )
            r_draw = random.random()
            if r_draw < p_nested_if:
                op_t = "nested_if"
                val = 0
                loop_k = 1
                branch_ops = random.sample(S5_BRANCH_OPS, 3)
                mode_id = 1
                b_ids = [CANON_OP_TO_ID[(b_op, 0)] for b_op in branch_ops]
            elif r_draw < p_nested_if + p_call_proc:
                op_t = "call_proc"
                val = 0
                loop_k = random.choice([1, 2]) if random.random() < p_loop else 1
                branch_ops = None
                mode_id = 2
                b_ids = [0, 0, 0]
            else:
                mode_id = 0
                branch_ops = None
                loop_k = random.choice([2, 3, 4]) if random.random() < p_loop else 1
                if random.random() < p_srv and f_max > f_min:
                    op_t = "srv"
                    val = random.randint(f_min, f_max - 1)
                    c_id = CANON_OP_TO_ID[("srv", min(val, NUM_SRV_TRAIN - 1))]
                else:
                    op_t = random.choice(SINGLE_OPS)
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
                    c_id = CANON_OP_TO_ID[(op_t, val)]
                b_ids = [c_id, c_id, c_id]

            regs = apply_op_v12(
                regs,
                op_t,
                val,
                alpha_val=av,
                beta_val=bv,
                proc_id=proc_id,
                loop_k=loop_k,
                branch_ops=branch_ops,
                nl_ood=nl_ood,
            )
            step_toks = make_v12_step_clause(
                op_t, val=val, loop_k=loop_k, branch_ops=branch_ops, style=step_style
            )
            stream_toks.extend(step_toks)
            y_sample.append(list(regs))
            sample_modes.append(mode_id)
            sample_loops.append(loop_k - 1)
            sample_b_ops.append(b_ids)

        X_list.append(stream_toks)
        Y_list.append(y_sample)
        meta_proc_id.append(proc_id)
        meta_epi.append([av, bv])
        meta_mode.append(sample_modes)
        meta_loop_k.append(sample_loops)
        meta_branch_ops.append(sample_b_ops)

    X_t = torch.tensor(X_list, dtype=torch.long, device=dev)
    Y_t = torch.tensor(Y_list, dtype=torch.long, device=dev)
    meta = {
        "proc_id": torch.tensor(meta_proc_id, dtype=torch.long, device=dev),
        "epi": torch.tensor(meta_epi, dtype=torch.long, device=dev),
        "mode": torch.tensor(meta_mode, dtype=torch.long, device=dev),
        "loop_k": torch.tensor(meta_loop_k, dtype=torch.long, device=dev),
        "branch_ops": torch.tensor(meta_branch_ops, dtype=torch.long, device=dev),
    }
    return X_t, Y_t, meta


class FreeFormClauseSemanticAdapter(nn.Module):
    """
    Нейронный парсер свободных русских и Python клауз (MAX_V12_CLAUSE_WORDS = 20):
    1. Для Clause 0: строго перестановочно-инвариантный (Set-Transformer без позиционного смещения!)
       8-слотовый кросс-аттеншн экстрактор:
       - слоты 0..4 -> начальные значения регистров `a..e` (5x5 логитов),
       - слоты 5..6 -> константы пролога `alpha, beta` (2x5 логитов),
       - слот 7     -> объявленная подпрограмма стека `proc_id` (2 логита).
    2. Для Clauses 1..T: позиционно-осведомлённый 2-слойный бидирекциональный энкодер +
       5-слотовый кросс-аттеншн, предсказывающий:
       - `mode_logits` (0=single_op, 1=nested_if, 2=call_proc),
       - `loop_logits` (кратность цикла k in {1, 2, 3, 4}),
       - `branch_op_logits` (3 ветки x 22 канонических оператора ядра).
    """

    def __init__(self, vocab_size, d_adapter=256):
        super().__init__()
        self.d_adapter = d_adapter
        self.syn_emb = nn.Embedding(vocab_size, d_adapter)
        nn.init.normal_(self.syn_emb.weight, std=0.05)

        # Clause 0: строго без позиционных эмбеддингов (100% математическая перестановочная инвариантность!)
        self.c0_slot_queries = nn.Linear(d_adapter, 8, bias=False)
        self.c0_reg_heads = nn.ModuleList(
            [nn.Linear(d_adapter, VALS_PER_REG) for _ in range(NUM_REGS)]
        )
        self.c0_epi_heads = nn.ModuleList(
            [nn.Linear(d_adapter, VALS_PER_REG) for _ in range(2)]
        )
        self.c0_proc_head = nn.Linear(d_adapter, 2)

        # Clauses 1..T: позиционные эмбеддинги внутри клаузы для различения веток if / elif / else
        self.step_word_pos = nn.Parameter(
            torch.randn(1, MAX_V12_CLAUSE_WORDS, d_adapter) * 0.02
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_adapter,
            nhead=4,
            dim_feedforward=d_adapter * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.step_encoder = nn.TransformerEncoder(
            enc_layer, num_layers=2, enable_nested_tensor=False
        )

        self.slot_queries = nn.Parameter(torch.randn(5, d_adapter) * 0.05)
        self.mode_head = nn.Sequential(nn.LayerNorm(d_adapter), nn.Linear(d_adapter, 3))
        self.loop_head = nn.Sequential(nn.LayerNorm(d_adapter), nn.Linear(d_adapter, 4))
        self.branch_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_adapter), nn.Linear(d_adapter, NUM_CANON_OPS)
                )
                for _ in range(3)
            ]
        )

    def parse_clause0(self, idx_clause0):
        """
        Строго перестановочно-инвариантный парсинг Clause 0:
        f(pi(idx_clause0)) == f(idx_clause0) для любой перестановки pi!
        """
        mask = idx_clause0 != PAD_ID
        x0 = self.syn_emb(idx_clause0)  # (B, W, D) — без позиционных смещений!
        scores = self.c0_slot_queries(x0) * (
            4.0 / math.sqrt(self.d_adapter)
        )  # (B, W, 8)
        scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
        attn = F.softmax(scores, dim=1)  # (B, W, 8)
        slots = torch.einsum("bws,bwd->bsd", attn, x0)  # (B, 8, D)

        reg_logits = torch.stack(
            [self.c0_reg_heads[g](slots[:, g]) for g in range(NUM_REGS)], dim=1
        )  # (B, 5, 5)
        epi_logits = torch.stack(
            [self.c0_epi_heads[e](slots[:, 5 + e]) for e in range(2)], dim=1
        )  # (B, 2, 5)
        proc_logits = self.c0_proc_head(slots[:, 7])  # (B, 2)
        return reg_logits, epi_logits, proc_logits

    def parse_step_clauses(self, idx_steps):
        B, T, W = idx_steps.shape
        flat_idx = idx_steps.reshape(B * T, W)
        pad_mask = flat_idx == PAD_ID
        safe_pad_mask = pad_mask.clone()
        safe_pad_mask[:, 0] = False

        x = self.syn_emb(flat_idx) + self.step_word_pos[:, :W, :]
        h = self.step_encoder(x, src_key_padding_mask=safe_pad_mask)

        scores = torch.einsum("sd,bwd->bsw", self.slot_queries, h) / math.sqrt(
            self.d_adapter
        )
        scores = scores.masked_fill(safe_pad_mask.unsqueeze(1), -1e9)
        attn = F.softmax(scores, dim=-1)
        slots = torch.einsum("bsw,bwd->bsd", attn, h)  # (B*T, 5, D)

        mode_logits = self.mode_head(slots[:, 0]).view(B, T, 3)
        loop_logits = self.loop_head(slots[:, 1]).view(B, T, 4)
        branch_op_logits = torch.stack(
            [
                self.branch_heads[b_i](slots[:, 2 + b_i]).view(B, T, NUM_CANON_OPS)
                for b_i in range(3)
            ],
            dim=2,
        )  # (B, T, 3, 22)
        return mode_logits, loop_logits, branch_op_logits


class StatePredicateRouter(nn.Module):
    """Нейронный маршрутизатор вложенных предикатов `nested_if` над состоянием SimVQ `h_prev`."""

    def __init__(self, d_model=640):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )

    def forward(self, h_state):
        return self.net(h_state.squeeze(1))


class FreeFormBranchingSimVQGPT(nn.Module):
    """
    Архитектура Stage v12.0 (`FreeFormBranchingSimVQGPT`):
    - 100% замороженное тернарное ядро `c4_b0_a4_coprime_lock` (`core_block`, `simvq`, `fast_op_mem`).
    - Перестановочно-инвариантный парсер `Clause 0` и контекстный парсер `Clauses 1..T`
      с притяжением команд к канонической кодовой книге операторов замороженного ядра.
    - Динамическая развёртка циклов (`loop_k in {1..4}`), вложенных ветвлений (`nested_if`)
      и стека вызовов подпрограмм (`call_proc`) со стабилизацией `SimVQ` после каждого микро-такта.
    """

    def __init__(self, base_c4_model, vocab_v12):
        super().__init__()
        self.core = copy.deepcopy(base_c4_model)
        for p in self.core.parameters():
            p.requires_grad = False
        self.core.eval()

        self.vocab_v12 = list(vocab_v12)
        self.w2i_v12 = {w: i for i, w in enumerate(self.vocab_v12)}
        self.d_model = self.core.d_model
        self.d_sub = self.core.d_sub

        self.tok_emb_v12 = nn.Embedding(len(self.vocab_v12), self.d_model)
        self._init_api_patch_embeddings()

        self.adapter = FreeFormClauseSemanticAdapter(len(self.vocab_v12), d_adapter=256)
        self.predicate_router = StatePredicateRouter(d_model=self.d_model)
        self._build_canonical_core_codebooks()

    def _init_api_patch_embeddings(self):
        base_w2i = {w: i for i, w in enumerate(VOCAB)}
        with torch.no_grad():
            nn.init.normal_(self.tok_emb_v12.weight, std=0.02)
            for w, idx_v12 in self.w2i_v12.items():
                if w in base_w2i:
                    self.tok_emb_v12.weight[idx_v12].copy_(
                        self.core.tok_emb.weight[base_w2i[w]]
                    )
            for s_id in range(NUM_FACTS):
                api_str = f"API_PATCH_{s_id:02d}"
                srv_str = f"SRV_{s_id:02d}"
                if api_str in self.w2i_v12 and srv_str in base_w2i:
                    self.tok_emb_v12.weight[self.w2i_v12[api_str]].copy_(
                        self.core.tok_emb.weight[base_w2i[srv_str]]
                    )
        self.tok_emb_v12.weight.requires_grad = False

    def _build_canonical_core_codebooks(self):
        """
        Извлекает из замороженного чанкера `c4_b0_a4_coprime_lock`:
        1. `epi_codebook` (5, 5, 2, d_model): точные векторы пролога `ep_pro` для всех (alpha, beta).
        2. `canon_cmd_codebook` (22, d_model): точные векторы команд `x_cmd_raw` для 22 базовых операций.
        3. `proc_cmd_table` (2, 2, d_model): точные векторы команд для 2 подпрограмм стека.
        """
        dev = next(self.core.parameters()).device
        n_epi = self.core.chunker.n_epi_slots
        with torch.no_grad():
            # 1. Таблица эпизодического пролога (5 x 5 x n_epi x d_model)
            epi_table = torch.zeros(
                VALS_PER_REG, VALS_PER_REG, n_epi, self.d_model, device=dev
            )
            c_dummy_step = make_canonical_base_clause("s5_cycle5", 0, 0, 0)
            for av in range(VALS_PER_REG):
                for bv in range(VALS_PER_REG):
                    c_init = make_canonical_base_clause("init", (0, 0, 0, 0, 0), av, bv)
                    s_toks = c_init + c_dummy_step
                    s_toks = s_toks + [PAD_ID] * (24 - len(s_toks))
                    stream = torch.tensor([s_toks], dtype=torch.long, device=dev)
                    _, _, ep_pro_raw, _, _ = self.core.chunker(
                        stream, self.core.tok_emb(stream), num_clauses=2
                    )
                    epi_table[av, bv].copy_(ep_pro_raw[0] * 0.05)

            # 2. Кодовая книга 22 канонических команд ядра
            cmd_table = torch.zeros(NUM_CANON_OPS, self.d_model, device=dev)
            c_init0 = make_canonical_base_clause("init", (0, 0, 0, 0, 0), 0, 0)
            for op_idx, (op_name, op_val) in enumerate(CANON_OP_LIST):
                c_step = make_canonical_base_clause(op_name, op_val, 0, 0)
                s_toks = c_init0 + c_step
                s_toks = s_toks + [PAD_ID] * (24 - len(s_toks))
                stream = torch.tensor([s_toks], dtype=torch.long, device=dev)
                x_cmd_raw, _, _, _, _ = self.core.chunker(
                    stream, self.core.tok_emb(stream), num_clauses=2
                )
                cmd_table[op_idx].copy_(x_cmd_raw[0, 1])

            # 3. Таблица 2-шаговых подпрограмм стека
            idx_rot = CANON_OP_TO_ID[("s5_cycle5", 0)]
            idx_alp = CANON_OP_TO_ID[("add_alpha", 0)]
            idx_cox = CANON_OP_TO_ID[("s5_coxeter", 0)]
            idx_bet = CANON_OP_TO_ID[("add_beta", 0)]
            proc_cmds = torch.stack(
                [
                    torch.stack([cmd_table[idx_rot], cmd_table[idx_alp]], dim=0),
                    torch.stack([cmd_table[idx_cox], cmd_table[idx_bet]], dim=0),
                ],
                dim=0,
            )

        self.register_buffer("epi_codebook", epi_table)
        self.register_buffer("canon_cmd_codebook", cmd_table)
        self.register_buffer("proc_cmd_table", proc_cmds)

    def _single_micro_tick(self, h_prev, c_raw_step, ep_pro, clause_toks=None):
        c_scaled = c_raw_step * 0.05
        h_slot = self.core.subspace_sphere_norm(c_scaled)
        h_pair = torch.cat([h_prev, h_slot], dim=1)
        c_pair = torch.cat([c_scaled, c_scaled], dim=1)
        delta_t = self.core.core_step_t(h_pair, c_pair, ep_pro)
        h_cand = self.core.rotational_step_with_consensus(h_slot, delta_t)

        if self.core.fast_op_mem.mem_active and clause_toks is not None:
            w_embs_clause = self.tok_emb_v12(clause_toks)
            mask_clause = clause_toks != PAD_ID
            h_cand = self.core.fast_op_mem.apply_in_circuit_step(
                w_embs_clause,
                mask_clause,
                h_prev,
                h_cand,
                self.core.subspace_sphere_norm,
            )
        h_next = self.core.simvq.snap_step(h_cand, snap_strength=1.0)
        return h_next, h_cand

    def forward(self, idx_stream, num_steps):
        B = idx_stream.size(0)
        T_clauses = num_steps + 1
        clauses = idx_stream.view(B, T_clauses, MAX_V12_CLAUSE_WORDS)
        idx_c0 = clauses[:, 0, :]
        idx_steps = clauses[:, 1:, :]

        # 1. Строго перестановочно-инвариантный парсинг Clause 0
        reg_logits, epi_logits, proc_logits = self.adapter.parse_clause0(idx_c0)
        pred_regs = reg_logits.argmax(dim=-1)  # (B, 5)
        pred_epi = epi_logits.argmax(dim=-1)  # (B, 2)
        pred_proc_id = proc_logits.argmax(dim=-1)  # (B,)

        cb = self.core.simvq.get_codebook()  # (5, 5, 128)
        h0 = torch.cat(
            [cb[g, pred_regs[:, g]] for g in range(NUM_REGS)], dim=-1
        ).unsqueeze(1)  # (B, 1, 640)
        ep_pro = self.epi_codebook[pred_epi[:, 0], pred_epi[:, 1]]  # (B, 2, 640)

        # 2. Контекстный парсинг Clauses 1..T с притяжением к канонической кодовой книге команд
        mode_logits, loop_logits, branch_op_logits = self.adapter.parse_step_clauses(
            idx_steps
        )
        pred_modes = mode_logits.argmax(dim=-1)  # (B, T)
        pred_loops = loop_logits.argmax(dim=-1) + 1  # (B, T)
        pred_branch_ops = branch_op_logits.argmax(dim=-1)  # (B, T, 3)
        cmd_branches = self.canon_cmd_codebook[pred_branch_ops]  # (B, T, 3, 640)

        h_states = [h0]
        h_cur = h0
        batch_proc_cmds = self.proc_cmd_table[pred_proc_id]  # (B, 2, 640)

        for t in range(num_steps):
            mode_t = pred_modes[:, t]
            loop_t = pred_loops[:, t]
            max_k = int(loop_t.max().item())
            clause_t_toks = idx_steps[:, t, :]

            for k_iter in range(max_k):
                active_mask = (loop_t > k_iter).view(B, 1, 1)
                if not active_mask.any():
                    break

                branch_logits = self.predicate_router(h_cur)
                chosen_branch = branch_logits.argmax(dim=-1)
                b_idx = chosen_branch.view(B, 1, 1).expand(B, 1, self.d_model)
                cmd_nested = torch.gather(cmd_branches[:, t, :, :], dim=1, index=b_idx)

                cmd_single = cmd_branches[:, t, 0:1, :]
                cmd_proc_s1 = batch_proc_cmds[:, 0:1, :]
                cmd_proc_s2 = batch_proc_cmds[:, 1:2, :]

                is_nested = (mode_t == 1).view(B, 1, 1)
                is_proc = (mode_t == 2).view(B, 1, 1)

                c_step1 = torch.where(
                    is_proc, cmd_proc_s1, torch.where(is_nested, cmd_nested, cmd_single)
                )
                h_after_1, _ = self._single_micro_tick(
                    h_cur, c_step1, ep_pro, clause_toks=clause_t_toks
                )
                if is_nested.any() or is_proc.any():
                    h_after_1_nomem, _ = self._single_micro_tick(
                        h_cur, c_step1, ep_pro, clause_toks=None
                    )
                    h_after_1 = torch.where(
                        is_nested | is_proc, h_after_1_nomem, h_after_1
                    )

                if is_proc.any():
                    h_after_2, _ = self._single_micro_tick(
                        h_after_1, cmd_proc_s2, ep_pro, clause_toks=None
                    )
                    h_step_out = torch.where(is_proc, h_after_2, h_after_1)
                else:
                    h_step_out = h_after_1

                h_cur = torch.where(active_mask, h_step_out, h_cur)

            h_states.append(h_cur)

        h_all = torch.cat(h_states, dim=1)
        logits = self.core.simvq.compute_logits(h_all)
        return logits, {
            "reg_logits": reg_logits,
            "epi_logits": epi_logits,
            "proc_logits": proc_logits,
            "mode_logits": mode_logits,
            "loop_logits": loop_logits,
            "branch_op_logits": branch_op_logits,
        }


def train_v12_model(model, dev="cuda", steps=400, batch_size=128, lr=3e-3):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-4)

    t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        bx, by, meta = generate_v12_dataset(
            batch_size,
            num_steps=3,
            fact_range=(0, NUM_FACTS),
            init_pool=TRAIN_INIT_STATES,
            style_mode="mix",
            permute_init=True,
            p_loop=0.35,
            p_nested_if=0.25,
            p_call_proc=0.20,
            p_srv=0.20,
            dev=dev,
        )
        clauses = bx.view(batch_size, 4, MAX_V12_CLAUSE_WORDS)
        idx_c0 = clauses[:, 0, :]
        idx_steps = clauses[:, 1:, :]

        # 1. Clause 0: перестановочно-инвариантные регистры a..e, константы alpha, beta и подпрограмма proc_id
        reg_logits, epi_logits, proc_logits = model.adapter.parse_clause0(idx_c0)
        loss_c0_regs = F.cross_entropy(
            reg_logits.reshape(-1, VALS_PER_REG), by[:, 0, :].reshape(-1)
        )
        loss_c0_epi = F.cross_entropy(
            epi_logits.reshape(-1, VALS_PER_REG), meta["epi"].reshape(-1)
        )
        loss_proc = F.cross_entropy(proc_logits, meta["proc_id"])

        # 2. Clauses 1..T: mode, loop_k, and branch operators
        mode_logits, loop_logits, branch_op_logits = model.adapter.parse_step_clauses(
            idx_steps
        )
        loss_mode = F.cross_entropy(
            mode_logits.reshape(-1, 3), meta["mode"].reshape(-1)
        )
        loss_loop = F.cross_entropy(
            loop_logits.reshape(-1, 4), meta["loop_k"].reshape(-1)
        )

        # Маска для веток команд: на single_op (mode==0) и nested_if (mode==1) обучаем слот 0;
        # на nested_if (mode==1) обучаем также слоты 1 и 2
        b_targets = meta["branch_ops"]  # (B, T, 3)
        active_b0 = (meta["mode"] != 2).reshape(-1)
        loss_b0 = F.cross_entropy(
            branch_op_logits[:, :, 0, :].reshape(-1, NUM_CANON_OPS)[active_b0],
            b_targets[:, :, 0].reshape(-1)[active_b0],
        )
        active_b12 = (meta["mode"] == 1).reshape(-1)
        if active_b12.any():
            loss_b1 = F.cross_entropy(
                branch_op_logits[:, :, 1, :].reshape(-1, NUM_CANON_OPS)[active_b12],
                b_targets[:, :, 1].reshape(-1)[active_b12],
            )
            loss_b2 = F.cross_entropy(
                branch_op_logits[:, :, 2, :].reshape(-1, NUM_CANON_OPS)[active_b12],
                b_targets[:, :, 2].reshape(-1)[active_b12],
            )
            loss_cmd = loss_b0 + loss_b1 + loss_b2
        else:
            loss_cmd = loss_b0

        # 3. StatePredicateRouter на всех 3125 состояниях SimVQ
        with torch.no_grad():
            s_idx = torch.randint(0, NUM_STATES, (256,), device=dev)
            a_v = (s_idx // 625) % 5
            b_v = (s_idx // 125) % 5
            c_v = (s_idx // 25) % 5
            d_v = (s_idx // 5) % 5
            e_v = s_idx % 5
            cond_ab = a_v >= b_v
            cond_cd = c_v < d_v
            branch_tgt = torch.where(
                cond_ab & cond_cd,
                torch.zeros_like(s_idx),
                torch.where(cond_ab, torch.ones_like(s_idx), torch.full_like(s_idx, 2)),
            )
            cb = model.core.simvq.get_codebook().float()
            h_codebook = torch.cat(
                [cb[0, a_v], cb[1, b_v], cb[2, c_v], cb[3, d_v], cb[4, e_v]], dim=-1
            ).unsqueeze(1)
        pred_branch_logits = model.predicate_router(h_codebook)
        loss_router = F.cross_entropy(pred_branch_logits, branch_tgt)

        loss = (
            2.0 * loss_c0_regs
            + 2.0 * loss_c0_epi
            + loss_proc
            + loss_mode
            + loss_loop
            + 2.0 * loss_cmd
            + loss_router
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        opt.step()
        sched.step()

        if step % 100 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                vx, vy, _ = generate_v12_dataset(
                    128,
                    num_steps=4,
                    init_pool=ZEROSHOT_INIT_STATES,
                    style_mode="mix",
                    permute_init=True,
                    p_srv=0.0,
                    dev=dev,
                )
                v_logits, _ = model(vx, num_steps=4)
                acc_s0 = (v_logits[:, 0].argmax(dim=-1) == vy[:, 0]).all(
                    dim=-1
                ).float().mean().item() * 100.0
                acc_s4 = (v_logits[:, -1].argmax(dim=-1) == vy[:, -1]).all(
                    dim=-1
                ).float().mean().item() * 100.0
            print(
                f"  [v12 Train Step {step:03d}/{steps}] "
                f"loss={loss.item():.4f} (c0_r={loss_c0_regs.item():.4f}, cmd={loss_cmd.item():.4f}, rtr={loss_router.item():.4f}) | "
                f"Clause0_PermInv={acc_s0:.2f}% | T=4_Full_Exact={acc_s4:.2f}% ({time.time() - t0:.1f}s)"
            )
    return model


@torch.no_grad()
def eval_v12_suite(model, dev="cuda"):
    """Полный стенд проверки всех режимов Stage v12.0 на Zero-Shot начальных состояниях."""
    model.eval()
    results = {}

    # 1. Strict Mathematical Permutation Invariance Check on Clause 0
    s_test = ZEROSHOT_INIT_STATES[0]
    regs_test = state_to_regs(s_test)
    c0_perm1 = make_v12_init_clause(
        regs_test, 3, 2, proc_id=1, style="free_ru", permute=True
    )
    non_pad = [t for t in c0_perm1 if t != PAD_ID]
    perm_batch = []
    for _ in range(16):
        shuf = list(non_pad)
        random.shuffle(shuf)
        shuf.extend([PAD_ID] * (MAX_V12_CLAUSE_WORDS - len(shuf)))
        perm_batch.append(shuf)
    perm_t = torch.tensor(perm_batch, dtype=torch.long, device=dev)
    reg_p, epi_p, proc_p = model.adapter.parse_clause0(perm_t)
    max_r_diff = (reg_p - reg_p[0:1]).abs().max().item()
    max_e_diff = (epi_p - epi_p[0:1]).abs().max().item()
    max_p_diff = (proc_p - proc_p[0:1]).abs().max().item()
    results["perm_invariance_max_diff"] = max(max_r_diff, max_e_diff, max_p_diff)

    def _run_acc(n_samples, n_steps, **kwargs):
        bx, by, _ = generate_v12_dataset(
            n_samples,
            num_steps=n_steps,
            init_pool=ZEROSHOT_INIT_STATES,
            dev=dev,
            **kwargs,
        )
        logits, _ = model(bx, num_steps=n_steps)
        return (logits[:, -1].argmax(dim=-1) == by[:, -1]).all(
            dim=-1
        ).float().mean().item() * 100.0

    # 2. Специализированные стресс-тесты каждого нового механизма v12.0 (N=256)
    results["FreeRU_Permuted_Init_T16"] = _run_acc(
        256,
        16,
        style_mode="free_ru",
        permute_init=True,
        p_loop=0.0,
        p_nested_if=0.0,
        p_call_proc=0.0,
        p_srv=0.0,
    )
    results["Loops_k1to4_RU_PY_T16"] = _run_acc(
        256,
        16,
        style_mode="mix",
        permute_init=True,
        p_loop=0.85,
        p_nested_if=0.0,
        p_call_proc=0.0,
        p_srv=0.0,
    )
    results["Nested_If_Elif_Else_T16"] = _run_acc(
        256,
        16,
        style_mode="mix",
        permute_init=True,
        p_loop=0.0,
        p_nested_if=0.85,
        p_call_proc=0.0,
        p_srv=0.0,
    )
    results["CallStack_Subroutines_T16"] = _run_acc(
        256,
        16,
        style_mode="mix",
        permute_init=True,
        p_loop=0.30,
        p_nested_if=0.0,
        p_call_proc=0.85,
        p_srv=0.0,
    )

    # 3. Комбинированный поток (Free RU + Python + Permuted Clause 0 + Loops + Nested If + Call Stack)
    for T_eval, n_s in [(16, 256), (64, 256), (128, 128), (256, 128), (512, 64)]:
        results[f"Combined_All_v12_T{T_eval}"] = _run_acc(
            n_s,
            T_eval,
            style_mode="mix",
            permute_init=True,
            p_loop=0.30,
            p_nested_if=0.25,
            p_call_proc=0.20,
            p_srv=0.0,
        )

    # 4. Live 0-SGD API Injection 2000 правил ВНУТРИ ЦИКЛОВ (`for _k in range(k): gateway.call_api(...)` / RU)
    H_target_c = model.core.simvq.get_codebook().float().clone()
    api_tok_ids = torch.tensor(
        [
            model.w2i_v12[f"API_PATCH_{s_id:02d}"]
            for s_id in range(NUM_SRV_TRAIN, NUM_FACTS)
        ],
        dtype=torch.long,
        device=dev,
    )
    api_key_vecs = model.tok_emb_v12(api_tok_ids)
    nl_srcs, nl_tables = [], []
    for i in range(NUM_SRV_OOD):
        p_src, tbl = NL_OOD_RULES[NUM_SRV_TRAIN + i]
        nl_srcs.append(list(p_src))
        nl_tables.append(tbl)
    t_inj = time.time()
    model.core.fast_op_mem.memorize_bilinear_operators(
        api_key_vecs,
        torch.tensor(nl_srcs, dtype=torch.long, device=dev),
        torch.stack(nl_tables, dim=0).to(dev),
        H_target_c,
    )
    results["0SGD_API_inject_ms"] = (time.time() - t_inj) * 1000.0
    results["0SGD_API_in_Loops_T16"] = _run_acc(
        256,
        16,
        fact_range=(NUM_SRV_TRAIN, NUM_FACTS),
        style_mode="mix",
        permute_init=True,
        p_loop=0.50,
        p_nested_if=0.15,
        p_call_proc=0.15,
        p_srv=0.70,
        nl_ood=True,
    )
    results["0SGD_API_in_Loops_T64"] = _run_acc(
        128,
        64,
        fact_range=(NUM_SRV_TRAIN, NUM_FACTS),
        style_mode="mix",
        permute_init=True,
        p_loop=0.40,
        p_nested_if=0.15,
        p_call_proc=0.15,
        p_srv=0.60,
        nl_ood=True,
    )
    return results


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    random.seed(42)

    print(f"=== Stage v12.0: FreeFormBranchingSimVQGPT on {dev.upper()} ===")
    m_c4, _, _, _ = load_local_champions(device=dev)
    v12_model = FreeFormBranchingSimVQGPT(m_c4, VOCAB_V12).to(dev)

    n_total = sum(p.numel() for p in v12_model.parameters())
    n_train = sum(p.numel() for p in v12_model.parameters() if p.requires_grad)
    print(
        f"  Vocab v12: {len(VOCAB_V12)} tokens | Total params: {n_total:,} | "
        f"Trainable adapter params: {n_train:,} (Core 100% Frozen)"
    )

    v12_model = train_v12_model(v12_model, dev=dev, steps=400, batch_size=128, lr=3e-3)

    print("\n=== Running Stage v12.0 Zero-Shot Evaluation Suite ===")
    res = eval_v12_suite(v12_model, dev=dev)
    for k, v in res.items():
        if "diff" in k:
            print(f"  {k:32s}: {v:.2e}")
        elif "ms" in k:
            print(f"  {k:32s}: {v:.2f} ms")
        else:
            print(f"  {k:32s}: {v:.2f}%")

    ckpt_path = os.path.join(
        os.path.dirname(__file__), "checkpoints", "v12_freeform_branching.pt"
    )
    torch.save(
        {
            "adapter_state_dict": v12_model.adapter.state_dict(),
            "predicate_router_state_dict": v12_model.predicate_router.state_dict(),
            "tok_emb_v12": v12_model.tok_emb_v12.weight.detach()
            .cpu()
            .to(torch.float16),
            "vocab_v12": VOCAB_V12,
            "eval_results": res,
        },
        ckpt_path,
    )
    sz_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    print(f"\n[Saved Local v12.0 Checkpoint] {ckpt_path} ({sz_mb:.2f} MB)")
