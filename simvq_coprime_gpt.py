"""
Zero-Crutch SimVQCoprimeGPT Core Architecture (v10.7 / v11.0)
- BitLinear ternary {-1, 0, +1} recurrent weights (1.58-bit STE)
- SimVQSubspaceAttractor_v104: Factored G x K Voronoi repeater with rotation-trick STE
- InCircuitBilinearFastMemory_v104: 6-shift bilinear Gram solver for 0-SGD OOD operator injection
- ZeroMaskStreamChunker_v104: Learned boundary detection & Clause-0 state/episodic extraction
- DualCircuitCausalAttention_v104: Circuit-1 Markov d in {0,1} + Circuit-2 Episodic Prologue
- Exact O(T) 2-step Markov window execution (bit-exact with full O(T^3) prefix under strict_d1=True)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_REGS, VALS_PER_REG = (5, 5)
NUM_STATES = VALS_PER_REG**NUM_REGS
NUM_SRV_TRAIN, NUM_SRV_OOD = (8, 16)
NUM_FACTS = NUM_SRV_TRAIN + NUM_SRV_OOD
MAX_CLAUSE_WORDS, NUM_EPI_SLOTS = (14, 4)
S5_PERMS = [
    (1, 2, 3, 4, 0),
    (1, 0, 3, 4, 2),
    (1, 2, 0, 3, 4),
    (4, 3, 2, 1, 0),
    (2, 0, 4, 1, 3),
    (3, 4, 0, 1, 2),
    (0, 2, 3, 4, 1),
    (1, 0, 4, 2, 3),
]


def make_valid_s5_srv_rule(idx):
    return (
        S5_PERMS[idx % len(S5_PERMS)],
        tuple(((idx * (2 * g + 1) + g + 1) % 5 for g in range(NUM_REGS))),
    )


def make_nonlinear_s5_ood_rule(i):
    pair_src = S5_PERMS[(i + 2) % len(S5_PERMS)]
    sh = tuple(((i * (g + 2) + g + 1) % 5 for g in range(NUM_REGS)))
    table = torch.zeros((NUM_REGS, VALS_PER_REG, VALS_PER_REG), dtype=torch.long)
    for g_out in range(NUM_REGS):
        d_val = sh[g_out]
        for v1 in range(VALS_PER_REG):
            for v2 in range(VALS_PER_REG):
                table[g_out, v1, v2] = (
                    (v1 * 2 + v2 + d_val) % 5
                    if g_out % 2 == 0
                    else (v1 + d_val + 2) % 5
                    if v1 >= v2
                    else (v1 + 3 * v2 + d_val) % 5
                )
    return (pair_src, table)


FACT_RULES = {i: make_valid_s5_srv_rule(i) for i in range(NUM_FACTS)}
NL_OOD_RULES = {
    NUM_SRV_TRAIN + i: make_nonlinear_s5_ood_rule(i) for i in range(NUM_SRV_OOD)
}
REG_NAMES = ["a", "b", "c", "d", "e"]
VOCAB = (
    [f"{r}={v}" for r in REG_NAMES for v in range(VALS_PER_REG)]
    + [f"alpha={v}" for v in range(VALS_PER_REG)]
    + [f"beta={v}" for v in range(VALS_PER_REG)]
    + [
        "<pad>",
        ".",
        "\n",
        "Question:",
        "def",
        "trace():",
        "Alice",
        "Emma",
        "starts",
        "begins",
        "with",
        "init",
        "registers",
        "please",
        "note",
        "that",
        "#_comment:",
        "check_sync",
        "noop_tag",
        "step_ok",
        "Then",
        "Next",
        "she",
        "system",
        "applies",
        "s5_cycle5",
        "s5_coxeter",
        "s5_alt3",
        "ca_rule110",
        "cell_ring",
        "step",
        "shift",
        "adds",
        "increments",
        "a,c,e",
        "b,d",
        "by",
        "0",
        "1",
        "2",
        "3",
        "4",
        "const_alpha",
        "const_beta",
        "a,c,e+=",
        "b,d+=",
        "queries",
        "loads",
        "rule",
        "from",
        "server",
        "apply_srv",
        "(",
        ")",
        "%",
        "5",
        "If",
        "When",
        "if",
        "else",
        "else:",
        "a>=b",
        "a>=b:",
        "c<d",
        "c<d:",
    ]
    + [f"SRV_{i:02d}" for i in range(NUM_FACTS)]
)
VOCAB_SIZE = len(VOCAB)
w2i = {w: i for i, w in enumerate(VOCAB)}
PAD_ID, END_IDS = (w2i["<pad>"], (w2i["."], w2i["\n"]))
import random


def state_to_regs(s):
    out = []
    for _ in range(NUM_REGS):
        out.append(s % VALS_PER_REG)
        s //= VALS_PER_REG
    return tuple(out)


def apply_op_logic(regs, op_type, val=0, alpha_val=0, beta_val=0, nl_ood=False):
    a, b, c, d, e = regs
    if op_type == "s5_cycle5":
        a, b, c, d, e = (b, c, d, e, a)
    elif op_type == "s5_coxeter":
        a, b, c, d, e = (b, a, d, e, c)
    elif op_type == "s5_alt3":
        a, b, c, d, e = (b, c, a, (d + 1) % 5, (e + 2) % 5)
    elif op_type == "add_ace":
        a, c, e = ((a + val) % 5, (c + val) % 5, (e + val) % 5)
    elif op_type == "add_bd":
        b, d = ((b + val) % 5, (d + val) % 5)
    elif op_type == "add_alpha":
        b, d = ((b + alpha_val) % 5, (d + alpha_val) % 5)
    elif op_type == "add_beta":
        a, c, e = ((a + beta_val) % 5, (c + beta_val) % 5, (e + beta_val) % 5)
    elif op_type == "if_s5":
        if a >= b:
            a, b, c, d, e = ((b + val) % 5, c, d, e, (a + val) % 5)
        else:
            a, b, c, d, e = (b, a, d, e, (c + val) % 5)
    elif op_type == "ca_rule110":
        old_r = (a, b, c, d, e)
        bits = [r % 2 for r in old_r]
        new_r = []
        for i in range(NUM_REGS):
            L, M, R = (
                bits[(i - 1) % NUM_REGS],
                bits[i],
                bits[(i + 1) % NUM_REGS],
            )
            r110_bit = 110 >> (L << 2 | M << 1 | R) & 1
            new_r.append((old_r[i] + r110_bit + 1) % 5)
        a, b, c, d, e = tuple(new_r)
    elif op_type == "srv":
        old_r = (a, b, c, d, e)
        if nl_ood and val in NL_OOD_RULES:
            pair_src, table = NL_OOD_RULES[val]
            a, b, c, d, e = tuple(
                (
                    int(
                        table[
                            g,
                            old_r[pair_src[g]],
                            old_r[(pair_src[g] + 1) % NUM_REGS],
                        ].item()
                    )
                    for g in range(NUM_REGS)
                )
            )
        else:
            perm, sh = FACT_RULES[val]
            a, b, c, d, e = tuple(
                ((old_r[perm[g]] + sh[g]) % 5 for g in range(NUM_REGS))
            )
    return (a, b, c, d, e)


COMMENT_CHUNKS = [
    [],
    ["#_comment:", "check_sync"],
    ["#_comment:", "noop_tag", "step_ok"],
    ["please", "note", "that"],
]


def make_clause_tokens(
    op_type, val, alpha_val=0, beta_val=0, mode="mix", allow_filler=True
):
    if mode == "mix":
        mode = random.choice(["en1", "en2", "py"])
    filler = random.choice(COMMENT_CHUNKS) if allow_filler else []
    end_tok = "\n" if mode == "py" else "."
    if op_type == "init":
        r_toks = [f"{REG_NAMES[g]}={val[g]}" for g in range(NUM_REGS)] + [
            f"alpha={alpha_val}",
            f"beta={beta_val}",
        ]
        base = (
            ["Question:", "Alice", "starts", "with"] + r_toks
            if mode == "en1"
            else ["Question:", "Emma", "begins", "with"] + r_toks
            if mode == "en2"
            else ["def", "trace():", "init"] + r_toks
        )
        return [
            w2i[w] for w in (base + filler[:1])[: MAX_CLAUSE_WORDS - 1] + [end_tok]
        ]
    v_s = str(val)
    if op_type == "s5_cycle5":
        base = (
            ["Then", "she", "applies", "s5_cycle5", "step"]
            if mode != "py"
            else ["s5_cycle5", "(", "registers", ")"]
        )
    elif op_type == "s5_coxeter":
        base = (
            ["Then", "system", "applies", "s5_coxeter", "step"]
            if mode != "py"
            else ["s5_coxeter", "(", "registers", ")"]
        )
    elif op_type == "s5_alt3":
        base = (
            ["Next", "she", "applies", "s5_alt3", "step"]
            if mode != "py"
            else ["s5_alt3", "(", "registers", ")"]
        )
    elif op_type == "add_ace":
        base = (
            ["Then", "she", "adds", v_s, "a,c,e"]
            if mode != "py"
            else ["a,c,e+=", v_s, "%", "5"]
        )
    elif op_type == "add_bd":
        base = (
            ["Next", "system", "increments", "b,d", "by", v_s]
            if mode != "py"
            else ["b,d+=", v_s, "%", "5"]
        )
    elif op_type == "add_alpha":
        base = (
            ["Then", "she", "adds", "const_alpha", "b,d"]
            if mode != "py"
            else ["b,d+=", "const_alpha"]
        )
    elif op_type == "add_beta":
        base = (
            ["Next", "system", "adds", "const_beta", "a,c,e"]
            if mode != "py"
            else ["a,c,e+=", "const_beta"]
        )
    elif op_type == "if_s5":
        base = (
            ["If", "a>=b", "s5_cycle5", v_s, "else", "s5_coxeter"]
            if mode != "py"
            else ["if", "a>=b:", "s5_cycle5", v_s, "else:", "s5_coxeter"]
        )
    elif op_type == "ca_rule110":
        base = (
            ["Then", "system", "applies", "ca_rule110", "cell_ring"]
            if mode != "py"
            else ["ca_rule110", "(", "cell_ring", ")"]
        )
    elif op_type == "srv":
        srv_s = f"SRV_{val:02d}"
        base = (
            ["Then", "she", "queries", "server", srv_s]
            if mode != "py"
            else ["apply_srv", "(", srv_s, ")"]
        )
    return [w2i[w] for w in (base + filler)[: MAX_CLAUSE_WORDS - 1] + [end_tok]]


TRAIN_INIT_STATES = [
    s for s in range(NUM_STATES) if sum(state_to_regs(s)) % 5 != 4
]
ZEROSHOT_INIT_STATES = [
    s for s in range(NUM_STATES) if sum(state_to_regs(s)) % 5 == 4
]


def generate_stream_dataset(
    num_samples,
    num_steps=2,
    fact_range=(0, NUM_SRV_TRAIN),
    init_pool=TRAIN_INIT_STATES,
    srv_prob=0.25,
    nl_ood=False,
    noop_anchor_prob=0.18,
    pure_s5_ca_only=False,
    device="cpu",
):
    max_stream_len = (num_steps + 1) * MAX_CLAUSE_WORDS
    X_list, Y_list = ([], [])
    f_min, f_max = fact_range
    ops_pool = (
        ["s5_cycle5", "s5_coxeter", "s5_alt3", "if_s5", "ca_rule110"]
        if pure_s5_ca_only
        else [
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
    )
    for _ in range(num_samples):
        style = random.choice(["en1", "en2", "py", "mix"])
        s0 = random.choice(init_pool)
        alpha_val, beta_val = (random.randint(0, 4), random.randint(0, 4))
        regs = state_to_regs(s0)
        stream_toks = list(
            make_clause_tokens("init", regs, alpha_val, beta_val, mode=style)
        )
        y_sample = [list(regs)]
        for _ in range(1, num_steps + 1):
            if not pure_s5_ca_only and random.random() < noop_anchor_prob:
                op_t, val = (random.choice(["add_ace", "add_bd"]), 0)
            elif pure_s5_ca_only or random.random() >= srv_prob:
                op_t = random.choice(ops_pool)
                if op_t in (
                    "s5_cycle5",
                    "s5_coxeter",
                    "s5_alt3",
                    "ca_rule110",
                    "add_alpha",
                    "add_beta",
                ):
                    val = 0
                elif op_t == "if_s5":
                    val = random.choice([1, 2])
                else:
                    val = random.randint(0, 2)
            else:
                op_t, val = ("srv", random.randint(f_min, f_max - 1))
            regs = apply_op_logic(
                regs, op_t, val, alpha_val, beta_val, nl_ood=nl_ood
            )
            stream_toks.extend(
                make_clause_tokens(op_t, val, alpha_val, beta_val, mode=style)
            )
            y_sample.append(list(regs))
        stream_toks.extend([PAD_ID] * (max_stream_len - len(stream_toks)))
        X_list.append(stream_toks)
        Y_list.append(y_sample)
    return (
        torch.tensor(X_list, dtype=torch.long, device=device),
        torch.tensor(Y_list, dtype=torch.long, device=device),
    )



class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.quant_alpha = 0.0

    def forward(self, x):
        w = self.weight
        if self.quant_alpha > 0.0:
            scale = w.abs().mean().clamp(min=1e-06)
            w_tern = torch.round(torch.clamp(w / scale, -1.0, 1.0)) * scale
            w = (1.0 - self.quant_alpha) * w + self.quant_alpha * (
                w + (w_tern - w).detach()
            )
        return F.linear(x, w, self.bias)


class SimVQSubspaceAttractor_v104(nn.Module):
    def __init__(self, d_model, num_regs=NUM_REGS, vals_per_reg=VALS_PER_REG):
        super().__init__()
        self.num_regs, self.vals_per_reg = (num_regs, vals_per_reg)
        self.d_sub = d_model // num_regs
        c_init = torch.zeros(num_regs, vals_per_reg, self.d_sub)
        for g in range(num_regs):
            q, _ = torch.linalg.qr(torch.randn(self.d_sub, self.d_sub))
            c_init[g] = q[:vals_per_reg] * math.sqrt(self.d_sub)
        self.register_buffer("C_frozen", c_init)
        eye_g = torch.eye(self.d_sub).unsqueeze(0).repeat(num_regs, 1, 1)
        self.W_simvq = nn.Parameter(eye_g + torch.randn_like(eye_g) * 0.002)
        self.logit_scale = nn.Parameter(torch.ones(num_regs, 1) * 6.5)

    def get_codebook(self):
        return F.normalize(torch.bmm(self.C_frozen, self.W_simvq), dim=-1) * math.sqrt(
            self.d_sub
        )

    def compute_logits(self, h):
        B, T, _ = h.shape
        h_normed = F.normalize(h.view(B, T, self.num_regs, self.d_sub), dim=-1)
        c_normed = F.normalize(self.get_codebook(), dim=-1)
        scale = self.logit_scale.clamp(2.0, 14.0).view(1, 1, self.num_regs, 1)
        return torch.einsum("btgd,gkd->btgk", h_normed, c_normed) * scale

    def rotation_trick_ste(self, z, z_hat):
        e, q = (F.normalize(z, dim=-1), F.normalize(z_hat, dim=-1))
        u = F.normalize(e + q, dim=-1).detach()
        e_det, q_det = (e.detach(), q.detach())
        return (
            z
            - 2.0 * (z * u).sum(dim=-1, keepdim=True) * u
            + 2.0 * (z * e_det).sum(dim=-1, keepdim=True) * q_det
        )

    def snap_step(self, h_step, snap_strength):
        if snap_strength <= 0.0:
            return h_step
        B, T, D = h_step.shape
        h_sub = h_step.view(B, T, self.num_regs, self.d_sub)
        logits = self.compute_logits(h_step)
        codebook = self.get_codebook().to(h_step.dtype)
        pred_k = logits.argmax(dim=-1)
        g_idx = torch.arange(self.num_regs, device=h_step.device).view(
            1, 1, self.num_regs
        )
        hard_anchors = codebook[g_idx, pred_k]
        if not self.training:
            return hard_anchors.reshape(B, T, D)
        with torch.no_grad():
            probs_sharp = F.softmax(logits.float() * 1.5, dim=-1)
            top2 = torch.topk(probs_sharp, k=2, dim=-1).values
            reg_margin = (top2[..., 0] - top2[..., 1]).to(h_step.dtype).unsqueeze(-1)
        conf = ((reg_margin - 0.3) / 0.35).clamp(0.0, 1.0)
        snap_gate = 0.9 * snap_strength * conf
        target_sub = F.normalize(
            (1.0 - snap_gate) * h_sub.detach() + snap_gate * hard_anchors.detach(),
            dim=-1,
        ) * math.sqrt(self.d_sub)
        return self.rotation_trick_ste(h_sub, target_sub).reshape(B, T, D)


class InCircuitBilinearFastMemory_v104(nn.Module):
    def __init__(
        self,
        d_model,
        num_regs=NUM_REGS,
        vals_per_reg=VALS_PER_REG,
        max_ood_rules=NUM_SRV_OOD,
    ):
        super().__init__()
        self.d_model, self.num_regs, self.vals_per_reg = (
            d_model,
            num_regs,
            vals_per_reg,
        )
        self.d_sub = d_model // num_regs
        self.shifts = (7, 23, 47, 73, 101, 113)
        self.d_pair = len(self.shifts) * self.d_sub
        self.d_key = num_regs * self.d_pair
        self.register_buffer("M_fast", torch.zeros(d_model, self.d_key))
        self.register_buffer("S_momentum", torch.zeros(d_model, self.d_key))
        self.register_buffer(
            "H_post_sub", torch.zeros(num_regs, vals_per_reg, self.d_sub)
        )
        self.register_buffer("mem_cmd_keys", torch.zeros(max_ood_rules, d_model))
        self.register_buffer("mem_srv_signs", torch.zeros(max_ood_rules, self.d_key))
        k_ortho = torch.zeros(num_regs, vals_per_reg, self.d_sub)
        gen_k = torch.Generator().manual_seed(314159)
        for g in range(num_regs):
            q, _ = torch.linalg.qr(torch.randn(self.d_sub, self.d_sub, generator=gen_k))
            k_ortho[g] = F.normalize(q[:vals_per_reg], dim=-1)
        self.register_buffer("K_ortho_basis", k_ortho)
        self.mem_active = False
        self.last_ortho_err = 0.0

    def make_pair_features(self, h1, h2):
        return F.normalize(
            torch.cat([h1 * torch.roll(h2, s, -1) for s in self.shifts], dim=-1),
            dim=-1,
        )

    @torch.no_grad()
    def memorize_bilinear_operators(
        self,
        cmd_key_vecs,
        pair_src_tensor,
        pair_tables_tensor,
        h_target_sub,
        mu_momentum=0.25,
        lr_write=0.85,
        n_passes=6,
    ):
        self.M_fast.zero_()
        self.S_momentum.zero_()
        self.mem_active = True
        self.H_post_sub.copy_(h_target_sub.to(self.H_post_sub.dtype))
        dev = cmd_key_vecs.device
        N_srv, G, K, d_sub, d_pair = (
            cmd_key_vecs.size(0),
            self.num_regs,
            self.vals_per_reg,
            self.d_sub,
            self.d_pair,
        )
        K2 = K * K
        self.mem_cmd_keys[:N_srv].copy_(F.normalize(cmd_key_vecs.float(), dim=-1))
        srv_signs = torch.empty((N_srv, self.d_key), device=dev, dtype=torch.float32)
        for i in range(N_srv):
            gen = torch.Generator(device=dev).manual_seed(10007 + i * 7919)
            srv_signs[i] = torch.where(
                torch.randn(self.d_key, generator=gen, device=dev) >= 0.0, 1.0, -1.0
            )
        self.mem_srv_signs[:N_srv].copy_(srv_signs)
        key_norm = F.normalize(self.K_ortho_basis.float(), dim=-1)
        pair_basis = torch.zeros((G, K2, d_pair), device=dev, dtype=torch.float32)
        for p_in in range(G):
            g1, g2 = (p_in, (p_in + 1) % G)
            s1 = key_norm[g1].unsqueeze(1).expand(K, K, d_sub).reshape(K2, d_sub)
            s2 = key_norm[g2].unsqueeze(0).expand(K, K, d_sub).reshape(K2, d_sub)
            pair_basis[p_in] = self.make_pair_features(s1, s2)
        inv_pair_src = torch.argsort(pair_src_tensor, dim=-1)
        K_read_sub = (
            srv_signs.view(N_srv, G, d_pair).permute(1, 0, 2).unsqueeze(2)
            * pair_basis.unsqueeze(1)
        ).reshape(G, N_srv * K2, d_pair)
        K_read_d = K_read_sub.double()
        eye_d = torch.eye(N_srv * K2, device=dev, dtype=torch.float64).unsqueeze(0)
        K_write_d = torch.linalg.solve(
            torch.bmm(K_read_d, K_read_d.transpose(1, 2)) + 1e-06 * eye_d, K_read_d
        )
        self.last_ortho_err = (
            (torch.bmm(K_write_d, K_read_d.transpose(1, 2)) - eye_d).abs().max().item()
        )
        K_write_sub = K_write_d.float()
        K_read_full = torch.zeros(
            (G, N_srv * K2, self.d_key), device=dev, dtype=torch.float32
        )
        K_write_full = torch.zeros(
            (G, N_srv * K2, self.d_key), device=dev, dtype=torch.float32
        )
        for p_in in range(G):
            K_read_full[p_in, :, p_in * d_pair : (p_in + 1) * d_pair] = K_read_sub[p_in]
            K_write_full[p_in, :, p_in * d_pair : (p_in + 1) * d_pair] = K_write_sub[
                p_in
            ]
        h_tgt_f = h_target_sub.float()
        V_target_full = torch.zeros(
            (G, N_srv, K2, G, d_sub), device=dev, dtype=torch.float32
        )
        s_idx = torch.arange(N_srv, device=dev).unsqueeze(1).expand(N_srv, K2)
        uv_idx = torch.arange(K2, device=dev).unsqueeze(0).expand(N_srv, K2)
        flat_tables = pair_tables_tensor.reshape(N_srv, G, K2)
        for p_in in range(G):
            g_out = inv_pair_src[:, p_in].unsqueeze(1).expand(N_srv, K2)
            val_out = flat_tables[s_idx, g_out, uv_idx]
            V_target_full[p_in, s_idx, uv_idx, g_out, :] = h_tgt_f[g_out, val_out]
        K_read_all, K_write_all, V_flat_all = (
            K_read_full.reshape(-1, self.d_key),
            K_write_full.reshape(-1, self.d_key),
            V_target_full.reshape(-1, self.d_model),
        )
        M, S = (self.M_fast.float(), self.S_momentum.float())
        for _ in range(n_passes):
            S = mu_momentum * S + lr_write * (
                (V_flat_all - K_read_all @ M.T).T @ K_write_all
            )
            M = M + S
        self.M_fast.copy_(M.to(self.M_fast.dtype))
        self.S_momentum.copy_(S.to(self.S_momentum.dtype))

    @torch.no_grad()
    def apply_in_circuit_step(
        self,
        word_embs,
        clause_t_mask,
        h_prev_step,
        h_cand_step,
        subspace_sphere_norm_fn,
    ):
        if not self.mem_active:
            return h_cand_step
        tok_sim = (
            F.normalize(word_embs.float(), dim=-1) @ self.mem_cmd_keys.T
        ).masked_fill(~clause_t_mask.unsqueeze(-1), -1.0)
        sim = tok_sim.max(dim=1).values.unsqueeze(1)
        gamma = torch.sigmoid((sim.max(dim=-1).values - 0.85) * 80.0)
        if gamma.max().item() < 0.001:
            return h_cand_step
        srv_signs = (F.softmax(sim * 60.0, dim=-1) @ self.mem_srv_signs).to(
            h_cand_step.dtype
        )
        B, dtype = (h_prev_step.size(0), h_prev_step.dtype)
        state_norm = F.normalize(self.H_post_sub.to(dtype), dim=-1)
        h_prev_sub = F.normalize(
            h_prev_step.view(B, 1, self.num_regs, self.d_sub), dim=-1
        )
        cos_sim = torch.einsum("btgd,gkd->btgk", h_prev_sub, state_norm)
        key_norm = F.normalize(self.K_ortho_basis.to(dtype), dim=-1)
        g_idx = torch.arange(self.num_regs, device=h_prev_step.device).view(
            1, 1, self.num_regs
        )
        clean_sub_in = key_norm[g_idx, cos_sim.argmax(dim=-1)]
        pair_feats = [
            self.make_pair_features(
                clean_sub_in[:, :, p_in], clean_sub_in[:, :, (p_in + 1) % self.num_regs]
            )
            for p_in in range(self.num_regs)
        ]
        k_bound = srv_signs * torch.cat(pair_feats, dim=-1)
        h_mem_pre = subspace_sphere_norm_fn(
            k_bound @ self.M_fast.to(h_cand_step.dtype).T
        )
        g_3d = gamma.to(h_cand_step.dtype).unsqueeze(-1)
        return subspace_sphere_norm_fn((1.0 - g_3d) * h_cand_step + g_3d * h_mem_pre)


class ZeroMaskStreamChunker_v104(nn.Module):
    def __init__(self, d_model, n_slots=4, n_epi_slots=NUM_EPI_SLOTS):
        super().__init__()
        self.d_model, self.d_sub, self.n_slots, self.n_epi_slots = (
            d_model,
            d_model // NUM_REGS,
            n_slots,
            n_epi_slots,
        )
        self.boundary_vec = nn.Parameter(torch.randn(1, d_model))
        self.boundary_scale = nn.Parameter(torch.tensor(10.0))
        self.boundary_bias = nn.Parameter(torch.tensor(-3.0))
        self.word_pos = nn.Parameter(torch.randn(MAX_CLAUSE_WORDS, d_model) * 0.01)
        self.slot_queries = nn.Linear(d_model, n_slots, bias=False)
        self.slot_proj = nn.Linear(n_slots * d_model, d_model, bias=False)
        self.init_reg_queries = nn.Linear(d_model, NUM_REGS, bias=False)
        self.init_reg_proj = nn.Linear(d_model, self.d_sub, bias=False)
        self.epi_slot_queries = nn.Linear(d_model, n_epi_slots, bias=False)
        self.ln_out = nn.LayerNorm(d_model)
        self.ln_epi = nn.LayerNorm(d_model)
        # Auto-Dream-RSI Cycle 5 (Lean-DAG-RSI + Neuro-Symbolic Split):
        # 1. `force_full_mask = True` removes the `num_clauses <= 66` truncation so `M_fast`
        #    (`InCircuitBilinearFastMemory_v104`) stays active at all horizons T=128..1024.
        # 2. `epi_temp_scale = 1.32` sharpens Clause 0 episodic extraction against filler tokens.
        self.force_full_mask = True
        self.epi_temp_scale = 1.32

    def forward(self, idx_stream, word_embs, num_clauses):
        B, L, D = word_embs.shape
        valid_tok = idx_stream != PAD_ID
        true_is_end = (idx_stream == END_IDS[0]) | (idx_stream == END_IDS[1])
        w_norm = F.normalize(word_embs.float(), dim=-1)
        b_norm = F.normalize(self.boundary_vec.float(), dim=-1)
        boundary_logits = (w_norm @ b_norm.T).squeeze(
            -1
        ) * self.boundary_scale + self.boundary_bias
        b_loss = F.binary_cross_entropy_with_logits(
            boundary_logits[valid_tok], true_is_end[valid_tok].float()
        )
        is_end = true_is_end if self.training else (boundary_logits > 0.0) & valid_tok
        seg_id = torch.cumsum(is_end.long(), dim=1) - is_end.long()
        is_start = torch.zeros_like(is_end)
        is_start[:, 0] = True
        is_start[:, 1:] = is_end[:, :-1]
        pos = torch.arange(L, device=idx_stream.device).unsqueeze(0).expand(B, L)
        tok_rel_pos = (pos - torch.cummax(pos * is_start.long(), dim=1).values).clamp(
            0, MAX_CLAUSE_WORDS - 1
        )
        x = word_embs + self.word_pos[tok_rel_pos]
        raw_scores = self.slot_queries(x) * (6.0 / math.sqrt(self.n_slots))
        slots_list = []
        safe_seg_masks = []
        keep_full_mask = bool(getattr(self, "force_full_mask", True)) or (
            num_clauses <= 66
        )
        clause0_mask = None
        c_step = 64
        for c0 in range(0, num_clauses, c_step):
            c1 = min(num_clauses, c0 + c_step)
            t_arange = torch.arange(c0, c1, device=idx_stream.device).view(
                1, c1 - c0, 1
            )
            seg_m = (seg_id.unsqueeze(1) == t_arange) & valid_tok.unsqueeze(1)
            safe_m = torch.where(
                seg_m.any(dim=-1, keepdim=True),
                seg_m,
                seg_m | (pos.unsqueeze(1) == 0),
            )
            if c0 == 0:
                clause0_mask = safe_m[:, 0]
            att_c = F.softmax(
                raw_scores.float()
                .unsqueeze(1)
                .masked_fill(~safe_m.unsqueeze(-1), -1000000000.0),
                dim=2,
            ).to(word_embs.dtype)
            slots_list.append(torch.einsum("btls,bld->btsd", att_c, x))
            if keep_full_mask:
                safe_seg_masks.append(safe_m)
        slots = torch.cat(slots_list, dim=1) if len(slots_list) > 1 else slots_list[0]
        safe_seg_mask = (
            (
                torch.cat(safe_seg_masks, dim=1)
                if len(safe_seg_masks) > 1
                else safe_seg_masks[0]
            )
            if keep_full_mask
            else None
        )
        x_cmd = self.ln_out(
            self.slot_proj(slots.reshape(B, num_clauses, self.n_slots * D))
            + slots.mean(dim=2)
        )
        init_scores = self.init_reg_queries(x) * (6.0 / math.sqrt(NUM_REGS))
        init_attn = F.softmax(
            init_scores.float().masked_fill(~clause0_mask.unsqueeze(-1), -1000000000.0),
            dim=1,
        ).to(word_embs.dtype)
        h0_raw = self.init_reg_proj(torch.einsum("blg,bld->bgd", init_attn, x)).reshape(
            B, 1, D
        )
        epi_scale = float(getattr(self, "epi_temp_scale", 1.32))
        epi_scores = (
            self.epi_slot_queries(x) * epi_scale * (6.0 / math.sqrt(self.n_epi_slots))
        )
        epi_attn = F.softmax(
            epi_scores.float().masked_fill(~clause0_mask.unsqueeze(-1), -1000000000.0),
            dim=1,
        ).to(word_embs.dtype)
        episodic_prologue = self.ln_epi(torch.einsum("blp,bld->bpd", epi_attn, x))
        return (x_cmd, h0_raw, episodic_prologue, safe_seg_mask, b_loss)


class DualCircuitCausalAttention_v104(nn.Module):
    def __init__(
        self, d_model, n_heads=8, max_rel_dist=160, use_bitnet=True, strict_d1=True
    ):
        super().__init__()
        self.n_heads, self.head_dim, self.strict_d1 = (
            n_heads,
            d_model // n_heads,
            strict_d1,
        )
        self.h1, self.h2 = (6, n_heads - 6)
        d1, d2 = (self.h1 * self.head_dim, self.h2 * self.head_dim)
        Lin = BitLinear if use_bitnet else nn.Linear
        self.q_proj, self.k_proj, self.v_proj = (
            Lin(d_model, d_model),
            Lin(d_model, d_model),
            Lin(d_model, d_model),
        )
        self.out_proj1 = Lin(d1, d_model)
        self.out_proj2 = Lin(d2, d_model)
        self.rel_bias = nn.Parameter(torch.zeros(self.h1, max_rel_dist))
        with torch.no_grad():
            for h in range(self.h1):
                for d in range(max_rel_dist):
                    self.rel_bias[h, d] = 4.5 if d == 1 else -3.5 * abs(d - 1)
        self.k_ep, self.v_ep = (Lin(d_model, d2), Lin(d_model, d2))
        self.k_slot_addr = nn.Parameter(
            torch.randn(1, self.h2, NUM_EPI_SLOTS, self.head_dim) * 0.5
        )
        self.k_null = nn.Parameter(torch.randn(1, self.h2, 1, self.head_dim) * 0.02)
        self.v_null = nn.Parameter(torch.zeros(1, self.h2, 1, self.head_dim))

    def forward(self, x_state, x_cmd, episodic_prologue):
        B, T, C = x_state.shape
        q_all = (
            self.q_proj(x_state + x_cmd)
            .view(B, T, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k_all = (
            self.k_proj(x_state).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        )
        v_all = (
            self.v_proj(x_state).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        )
        q1, k1, v1 = (q_all[:, : self.h1], k_all[:, : self.h1], v_all[:, : self.h1])
        att1 = q1 @ k1.transpose(-2, -1) * (1.0 / math.sqrt(self.head_dim))
        idx = torch.arange(T, device=x_state.device)
        raw_dist = idx.unsqueeze(1) - idx.unsqueeze(0)
        att1 = att1 + self.rel_bias[:, raw_dist.clamp(0, self.rel_bias.size(1) - 1)].to(
            att1.dtype
        ).unsqueeze(0)
        causal_mask = (
            (raw_dist >= 0) & (raw_dist <= 1) if self.strict_d1 else raw_dist >= 0
        ).view(1, 1, T, T)
        y1 = (
            F.softmax(att1.float().masked_fill(~causal_mask, float("-inf")), dim=-1).to(
                v1.dtype
            )
            @ v1
        )
        y1_out = self.out_proj1(
            y1.transpose(1, 2).contiguous().view(B, T, self.h1 * self.head_dim)
        )
        P_slots = episodic_prologue.size(1)
        q2 = q_all[:, self.h1 :]
        k_pro = self.k_ep(episodic_prologue).view(
            B, P_slots, self.h2, self.head_dim
        ).transpose(1, 2) + self.k_slot_addr[:, :, :P_slots, :].to(q2.dtype)
        v_pro = (
            self.v_ep(episodic_prologue)
            .view(B, P_slots, self.h2, self.head_dim)
            .transpose(1, 2)
        )
        k2 = torch.cat([k_pro, self.k_null.expand(B, -1, -1, -1).to(q2.dtype)], dim=2)
        v2 = torch.cat([v_pro, self.v_null.expand(B, -1, -1, -1).to(q2.dtype)], dim=2)
        y2 = (
            F.softmax(
                (q2 @ k2.transpose(-2, -1) * (1.0 / math.sqrt(self.head_dim))).float(),
                dim=-1,
            ).to(v2.dtype)
            @ v2
        )
        y2_out = self.out_proj2(
            y2.transpose(1, 2).contiguous().view(B, T, self.h2 * self.head_dim)
        )
        return (y1_out, y2_out)


class UnifiedLoopedTransformer_v104(nn.Module):
    def __init__(self, vocab_size, d_model=640, n_heads=8, mode="simvq_ours"):
        super().__init__()
        self.d_model, self.d_sub, self.mode = (d_model, d_model // NUM_REGS, mode)
        self.snap_strength = 0.0
        self.fast_local_d1 = mode != "standard"
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        self.chunker = ZeroMaskStreamChunker_v104(d_model, n_slots=4)
        self.simvq = SimVQSubspaceAttractor_v104(d_model, NUM_REGS, VALS_PER_REG)
        self.fast_op_mem = InCircuitBilinearFastMemory_v104(
            d_model, NUM_REGS, VALS_PER_REG
        )
        strict_d1 = mode != "standard"
        Lin = BitLinear if mode != "standard" else nn.Linear
        self.ln_state, self.ln_cmd, self.ln_ep = (
            nn.LayerNorm(d_model),
            nn.LayerNorm(d_model),
            nn.LayerNorm(d_model),
        )
        self.attn = DualCircuitCausalAttention_v104(
            d_model, n_heads, use_bitnet=mode != "standard", strict_d1=strict_d1
        )
        self.ln_mlp = nn.LayerNorm(d_model)
        self.ln_ep_out = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            Lin(d_model * 2, 6 * d_model), nn.GELU(), Lin(6 * d_model, d_model)
        )
        self.angle_proj = Lin(d_model, d_model // 2)
        self.ln_standard_out = nn.LayerNorm(d_model)
        # Auto-Dream-RSI Cycle 5 Neuro-Symbolic Split parameters:
        self.wy1 = 0.37
        self.wy2 = 0.61
        self.rot_w = 0.18

    def subspace_sphere_norm(self, h):
        orig = h.shape
        return (
            F.normalize(h.view(*orig[:-1], NUM_REGS, self.d_sub), dim=-1)
            * math.sqrt(self.d_sub)
        ).reshape(orig)

    def rotational_step_with_consensus(self, h_cmd_slot, delta_feat):
        raw_angles = torch.tanh(self.angle_proj(delta_feat)) * 0.2
        B, T, half_d = raw_angles.shape
        grouped = raw_angles.view(B, T, half_d // 4, 4)
        clean_angles = (
            0.75 * grouped + 0.25 * grouped.median(dim=-1, keepdim=True).values
        ).view(B, T, half_d)
        cos_a, sin_a = (torch.cos(clean_angles), torch.sin(clean_angles))
        h_rot = torch.stack(
            [
                h_cmd_slot[..., 0::2] * cos_a - h_cmd_slot[..., 1::2] * sin_a,
                h_cmd_slot[..., 0::2] * sin_a + h_cmd_slot[..., 1::2] * cos_a,
            ],
            dim=-1,
        ).flatten(-2)
        rw = float(getattr(self, "rot_w", 0.18))
        return self.subspace_sphere_norm(
            rw * h_rot + (1.0 - rw) * self.subspace_sphere_norm(delta_feat)
        )

    def core_step_t(self, h_prefix, c_prefix, ep_pro):
        h_n, c_n, ep_n = (
            self.ln_state(h_prefix),
            self.ln_cmd(c_prefix),
            self.ln_ep(ep_pro),
        )
        y1_all, y2_all = self.attn(h_n, c_n, ep_n)
        wy1 = float(getattr(self, "wy1", 0.37))
        wy2 = float(getattr(self, "wy2", 0.61))
        ctx_in = (
            h_n[:, -2:-1]
            + wy1 * self.ln_mlp(y1_all[:, -1:])
            + wy2 * self.ln_ep_out(y2_all[:, -1:])
        )
        return self.mlp(torch.cat([ctx_in, c_n[:, -1:]], dim=-1))

    def set_quant_alpha(self, alpha):
        for m in self.modules():
            if isinstance(m, BitLinear):
                m.quant_alpha = alpha

    def forward(
        self,
        idx_stream,
        num_steps,
        init_target=None,
        final_target=None,
        inject_noise_std=0.0,
        return_hidden=False,
    ):
        T_clauses = num_steps + 1
        word_embs = self.tok_emb(idx_stream)
        x_cmd_raw, h0_raw, ep_pro_raw, safe_seg_mask, b_loss = self.chunker(
            idx_stream, word_embs, T_clauses
        )
        x_cmd, ep_pro = (x_cmd_raw * 0.05, ep_pro_raw * 0.05)
        eff_snap = (
            (1.0 if not self.training else self.snap_strength)
            if self.mode == "simvq_ours"
            else 0.0
        )
        if self.mode == "standard":
            h0_pre = self.ln_standard_out(h0_raw)
            h0 = h0_pre
            h = torch.cat([h0, self.ln_standard_out(x_cmd[:, 1:])], dim=1)
        else:
            h0_pre = self.subspace_sphere_norm(h0_raw)
            h0 = self.simvq.snap_step(h0_pre, snap_strength=eff_snap)
            h = torch.cat([h0, self.subspace_sphere_norm(x_cmd[:, 1:])], dim=1)
        if not self.training:
            h = h.clone()
        h_last_pre = h[:, -1:]
        use_local_2 = (not self.training) and self.fast_local_d1 and self.attn.strict_d1
        use_mfast = (
            (self.mode == "simvq_ours")
            and self.fast_op_mem.mem_active
            and (safe_seg_mask is not None)
        )
        for m in range(num_steps):
            step_t = m + 1
            if use_local_2:
                delta_t = self.core_step_t(
                    h[:, step_t - 1 : step_t + 1],
                    x_cmd[:, step_t - 1 : step_t + 1],
                    ep_pro,
                )
            else:
                delta_t = self.core_step_t(
                    h[:, : step_t + 1], x_cmd[:, : step_t + 1], ep_pro
                )
            if self.mode == "standard":
                h_t_cand = self.ln_standard_out(h[:, step_t : step_t + 1] + delta_t)
                if inject_noise_std > 0.0:
                    h_t_cand = self.ln_standard_out(
                        h_t_cand + torch.randn_like(h_t_cand) * inject_noise_std
                    )
                h_t_next = h_t_cand
            else:
                h_t_cand = self.rotational_step_with_consensus(
                    h[:, step_t : step_t + 1], delta_t
                )
                if inject_noise_std > 0.0:
                    h_t_cand = self.subspace_sphere_norm(
                        h_t_cand + torch.randn_like(h_t_cand) * inject_noise_std
                    )
                if use_mfast:
                    h_t_cand = self.fast_op_mem.apply_in_circuit_step(
                        word_embs,
                        safe_seg_mask[:, step_t],
                        h[:, step_t - 1 : step_t],
                        h_t_cand,
                        self.subspace_sphere_norm,
                    )
                h_t_next = self.simvq.snap_step(h_t_cand, snap_strength=eff_snap)
            if step_t == num_steps:
                h_last_pre = h_t_cand
            if not self.training:
                h[:, step_t : step_t + 1] = h_t_next
            else:
                h = torch.cat([h[:, :step_t], h_t_next, h[:, step_t + 1 :]], dim=1)
        logits = self.simvq.compute_logits(h)
        loss = None
        if init_target is not None and final_target is not None:
            logits_init_pre = self.simvq.compute_logits(h0_pre)[:, 0]
            logits_final_pre = self.simvq.compute_logits(h_last_pre)[:, 0]
            loss_init = F.cross_entropy(
                logits_init_pre.reshape(-1, VALS_PER_REG), init_target.reshape(-1)
            )
            loss_final = F.cross_entropy(
                logits_final_pre.reshape(-1, VALS_PER_REG), final_target.reshape(-1)
            )
            loss = loss_final + 0.75 * loss_init + 0.1 * b_loss
        if return_hidden:
            return (logits, loss, h)
        return (logits, loss)
