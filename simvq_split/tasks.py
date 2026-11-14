"""
Multi-Track Benchmark Tasks & Data Loaders for SimVQ-Split vs Post-Transformer Baselines.

All tasks emit raw UTF-8 strings tokenized by the single shared `ByteBPETokenizer`
into `token_ids: torch.LongTensor`. Models are trained ONLY with standard next-token
cross-entropy (`F.cross_entropy(logits[:, :-1], targets[:, 1:])`) — zero task IDs,
zero clause boundary labels, zero auxiliary state targets.

Tracks:
- Track A (`Real English Prose Perplexity`): Unmodified English texts from official
  `benchmarks/gsm8k/train.jsonl` (7,473 docs) and held-out `benchmarks/gsm8k/test.jsonl` (1,319 docs).
- Track B (`Non-Abelian S_5 State Tracking in Natural Prose`): 5-vessel permutation
  stories interleaved with real English sentences from GSM8K (train uses `train.jsonl`
  sentences, eval/OOD uses `test.jsonl` sentences). Trained ONLY at short horizons
  `k in {1, 2, 3, 4}` (`L <= 160` tokens); blind-evaluated at `k in {4, 8, 16, 32}`
  up to `L ~ 900+` tokens (up to 25x local Cortex receptive field).
- Track C (`Correlated-Key Counterfactual In-Context Overwrite`): Multi-entity stories
  with overlapping multi-token titles ("Senior Archive Officer Lin" vs "Junior Archive
  Officer Lin"), mid-stream reassignments, and real English distractor prose.
  Trained at `L <= 160` tokens; blind-evaluated up to `L ~ 900+` tokens.
- Track E (`Correlated Unseen Trap / Hallucination Brake`): Same multi-entity dispatch
  ledgers as Track C, queried on a correlated unmentioned officer whose rank, role, and
  surname words appeared across other ledger officers -> target `" Unknown."`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
from typing import List, Tuple
import torch

from .tokenizer import ByteBPETokenizer


SLOTS = ("Alpha", "Beta", "Gamma", "Delta", "Epsilon")
GEMS = ("ruby", "amber", "jade", "cobalt", "onyx")

# Non-abelian generators of S_5
S5_OPS = (
    (
        "swap_01",
        (1, 0, 2, 3, 4),
        (
            "Next, the curator swaps Alpha and Beta.",
            "Then Alpha and Beta exchange their stones.",
        ),
    ),
    (
        "swap_12",
        (0, 2, 1, 3, 4),
        (
            "Next, the curator swaps Beta and Gamma.",
            "Then Beta and Gamma exchange their stones.",
        ),
    ),
    (
        "swap_23",
        (0, 1, 3, 2, 4),
        (
            "Next, the curator swaps Gamma and Delta.",
            "Then Gamma and Delta exchange their stones.",
        ),
    ),
    (
        "swap_34",
        (0, 1, 2, 4, 3),
        (
            "Next, the curator swaps Delta and Epsilon.",
            "Then Delta and Epsilon exchange their stones.",
        ),
    ),
    (
        "rot_left",
        (1, 2, 3, 4, 0),
        (
            "Afterwards, every pedestal shifts one step left in a cycle.",
            "Then the five pedestals rotate left by one slot.",
        ),
    ),
    (
        "rot_right",
        (4, 0, 1, 2, 3),
        (
            "Afterwards, every pedestal shifts one step right in a cycle.",
            "Then the five pedestals rotate right by one slot.",
        ),
    ),
    (
        "reverse_all",
        (4, 3, 2, 1, 0),
        (
            "Suddenly, the entire row of five pedestals reverses order.",
            "Then the keeper flips the full order from Alpha through Epsilon.",
        ),
    ),
)

# Correlated multi-token entity names (high subword overlap across rank/division/surname)
RANKS = ("Senior", "Junior", "Chief")
ROLES = ("Archive Officer", "Vault Warden", "Signal Envoy")
SURNAMES = ("Lin", "Cho", "Patel", "Novak")
CITIES = (
    "Lisbon",
    "Prague",
    "Kyoto",
    "Ottawa",
    "Bergen",
    "Verona",
    "bruges",
    "Galway",
    "Hobart",
    "quito",
)
# Normalize capitalization cleanly
CITIES = (
    "Lisbon",
    "Prague",
    "Kyoto",
    "Ottawa",
    "Bergen",
    "Verona",
    "Bruges",
    "Galway",
    "Hobart",
    "Quito",
)

ALL_ENTITIES = tuple(
    f"{rk} {rl} {sn}" for rk in RANKS for rl in ROLES for sn in SURNAMES
)


def load_gsm8k_corpora(
    repo_root: Path,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Returns:
      train_docs: full question + answer texts from gsm8k/train.jsonl
      test_docs: full question + answer texts from gsm8k/test.jsonl (strictly disjoint)
      train_sentences: clean narrative sentences from train_docs questions for distractor prose
      test_sentences: clean narrative sentences from test_docs questions for unseen eval distractor prose
    """
    train_path = repo_root / "benchmarks" / "gsm8k" / "train.jsonl"
    test_path = repo_root / "benchmarks" / "gsm8k" / "test.jsonl"
    assert train_path.exists() and test_path.exists(), (
        f"Missing GSM8K files at {train_path}"
    )

    def _read(path: Path) -> Tuple[List[str], List[str]]:
        docs: List[str] = []
        sents: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                q = obj["question"].strip()
                a = (
                    re.sub(r"<<[^>]+>>", "", obj["answer"])
                    .replace("\n#### ", " Final answer: ")
                    .strip()
                )
                docs.append(f"Problem: {q}\nSolution: {a}")
                for s in re.split(r"(?<=[.!?])\s+", q):
                    s_clean = s.strip()
                    if 25 <= len(s_clean) <= 95 and not s_clean.endswith("?"):
                        sents.append(s_clean)
        return docs, sents

    tr_docs, tr_sents = _read(train_path)
    te_docs, te_sents = _read(test_path)
    return tr_docs, te_docs, tr_sents, te_sents


@dataclass
class EvalExample:
    prompt_ids: List[int]
    target_ids: List[int]
    total_len: int
    meta: str


def make_s5_story(
    rng: random.Random,
    num_ops: int,
    distractor_pool: List[str],
    distractors_per_op: Tuple[int, int] = (1, 2),
) -> Tuple[str, str, Tuple[str, ...]]:
    """
    Returns `(prompt_prefix, target_suffix, final_state_tuple)`.
    Uses natural space after colon (`Alpha: ruby`) so BPE gem tokens match write-time
    gem tokens under tied input/output embeddings across all models (`5! = 120` states,
    `0.83%` exact-match random chance).
    """
    state = list(GEMS)
    rng.shuffle(state)

    parts: List[str] = [
        f"Vault log: At the start, Alpha holds {state[0]}, Beta holds {state[1]}, "
        f"Gamma holds {state[2]}, Delta holds {state[3]}, and Epsilon holds {state[4]}."
    ]
    # Add initial prose distractor so even op 1 / query is well past local window
    for _ in range(rng.randint(distractors_per_op[0], distractors_per_op[1])):
        parts.append(rng.choice(distractor_pool))

    for _ in range(num_ops):
        _, perm, templates = rng.choice(S5_OPS)
        state = [state[p] for p in perm]
        parts.append(rng.choice(templates))
        n_d = rng.randint(distractors_per_op[0], distractors_per_op[1])
        for _ in range(n_d):
            parts.append(rng.choice(distractor_pool))

    prompt = " ".join(parts) + " Final vault summary:"
    target = (
        f" Alpha: {state[0]}, Beta: {state[1]}, Gamma: {state[2]}, "
        f"Delta: {state[3]}, Epsilon: {state[4]}."
    )
    return prompt, target, tuple(state)


def make_overwrite_story(
    rng: random.Random,
    num_entities: int,
    num_overwrites: int,
    distractor_pool: List[str],
    filler_sentences_after: int = 2,
    distractor_prob: float = 0.4,
) -> Tuple[str, str, str]:
    """
    Generates a correlated-key associative narrative with mid-stream counterfactual
    reassignments and distractor prose.
    """
    # Deliberately sample correlated entities sharing surname or role
    anchor = rng.choice(ALL_ENTITIES)
    anchor_words = set(anchor.split())
    correlated = [
        e
        for e in ALL_ENTITIES
        if e != anchor and len(anchor_words.intersection(e.split())) >= 1
    ]
    rng.shuffle(correlated)
    chosen = [anchor] + correlated[: num_entities - 1]
    rng.shuffle(chosen)

    assignment = {}
    events: List[str] = ["Dispatch ledger:"]
    for ent in chosen:
        city = rng.choice(CITIES)
        assignment[ent] = city
        events.append(f"Record: {ent} is stationed in {city}.")
        if rng.random() < distractor_prob:
            events.append(rng.choice(distractor_pool))

    overwritten_entities: List[str] = []
    for _ in range(num_overwrites):
        ent = rng.choice(chosen)
        old_c = assignment[ent]
        new_c = rng.choice([c for c in CITIES if c != old_c])
        assignment[ent] = new_c
        overwritten_entities.append(ent)
        if rng.random() < distractor_prob:
            events.append(rng.choice(distractor_pool))
        events.append(f"Update: {ent} relocated to {new_c}.")

    for _ in range(filler_sentences_after):
        events.append(rng.choice(distractor_pool))

    query_ent = (
        rng.choice(overwritten_entities) if overwritten_entities else rng.choice(chosen)
    )
    ans_city = assignment[query_ent]
    prompt = " ".join(events) + f" Audit query: {query_ent} is currently in"
    target = f" {ans_city}."
    return prompt, target, ans_city


def make_trap_overwrite_story(
    rng: random.Random,
    num_entities: int,
    num_overwrites: int,
    distractor_pool: List[str],
    filler_sentences_after: int = 2,
    distractor_prob: float = 0.4,
) -> Tuple[str, str, str]:
    """
    Generates an identical dispatch ledger to `make_overwrite_story`, then queries
    a correlated UNSEEN trap officer sharing >= 2 title/surname words with ledger
    officers. Ground-truth answer is `" Unknown."` (`Hallucination Brake`).
    """
    anchor = rng.choice(ALL_ENTITIES)
    anchor_words = set(anchor.split())
    correlated = [
        e
        for e in ALL_ENTITIES
        if e != anchor and len(anchor_words.intersection(e.split())) >= 1
    ]
    rng.shuffle(correlated)
    chosen = [anchor] + correlated[: num_entities - 1]
    rng.shuffle(chosen)
    chosen_set = set(chosen)

    assignment = {}
    events: List[str] = ["Dispatch ledger:"]
    for ent in chosen:
        city = rng.choice(CITIES)
        assignment[ent] = city
        events.append(f"Record: {ent} is stationed in {city}.")
        if rng.random() < distractor_prob:
            events.append(rng.choice(distractor_pool))

    for _ in range(num_overwrites):
        ent = rng.choice(chosen)
        old_c = assignment[ent]
        new_c = rng.choice([c for c in CITIES if c != old_c])
        assignment[ent] = new_c
        if rng.random() < distractor_prob:
            events.append(rng.choice(distractor_pool))
        events.append(f"Update: {ent} relocated to {new_c}.")

    for _ in range(filler_sentences_after):
        events.append(rng.choice(distractor_pool))

    used_words = {w for e in chosen for w in e.split()}
    hard_traps = [
        e
        for e in ALL_ENTITIES
        if e not in chosen_set and len(used_words.intersection(e.split())) >= 3
    ]
    if not hard_traps:
        hard_traps = [
            e
            for e in ALL_ENTITIES
            if e not in chosen_set and len(used_words.intersection(e.split())) >= 2
        ]
    trap_ent = rng.choice(hard_traps)
    prompt = " ".join(events) + f" Audit query: {trap_ent} is currently in"
    target = " Unknown."
    return prompt, target, "Unknown"


def build_mixed_lm_chunks(
    tokenizer: ByteBPETokenizer,
    train_docs: List[str],
    train_sents: List[str],
    seq_len: int = 192,
    num_prose_docs: int = 6000,
    num_s5_stories: int = 9000,
    num_overwrite_stories: int = 9000,
    seed: int = 42,
) -> torch.Tensor:
    """
    Builds a single unified next-token LM training tensor `(N_seqs, seq_len + 1)`
    combining:
    - Real GSM8K English prose documents (Track A)
    - Short-horizon S_5 prose stories `k in {1, 2, 3, 4}` fitting within `seq_len` (Track B)
    - Short-horizon correlated-key overwrite stories fitting within `seq_len` (Track C)

    Every story starts at index 0 of a sequence (padded with PAD_ID on the right) or
    is packed cleanly for continuous prose, so no story's target is truncated mid-answer.
    """
    rng = random.Random(seed)
    rows: List[List[int]] = []
    target_w = seq_len + 1

    # 1. Pack continuous real English prose chunks
    prose_subset = list(train_docs)
    rng.shuffle(prose_subset)
    token_stream: List[int] = []
    for doc in prose_subset[:num_prose_docs]:
        token_stream.extend(tokenizer.encode(doc, add_bos=True, add_eos=True))
    for i in range(0, len(token_stream) - target_w, seq_len):
        rows.append(token_stream[i : i + target_w])

    # 2. Short-horizon S_5 stories (k in {1, 2, 3, 4}, strictly <= seq_len + 1 tokens)
    added_s5 = 0
    attempts = 0
    while added_s5 < num_s5_stories and attempts < num_s5_stories * 4:
        attempts += 1
        k = rng.choice((1, 2, 3, 4))
        d_range = (1, 2) if k <= 2 else (0, 1)
        prompt, target, _ = make_s5_story(
            rng, k, train_sents, distractors_per_op=d_range
        )
        ids = tokenizer.encode(prompt + target, add_bos=True, add_eos=True)
        if len(ids) <= target_w:
            ids = ids + [tokenizer.PAD_ID] * (target_w - len(ids))
            rows.append(ids)
            added_s5 += 1

    # 3. Short-horizon Overwrite stories (3..5 correlated entities, 1..2 overwrites)
    added_ov = 0
    attempts = 0
    while added_ov < num_overwrite_stories and attempts < num_overwrite_stories * 4:
        attempts += 1
        n_ent = rng.randint(3, 5)
        n_ov = rng.randint(1, 2)
        prompt, target, _ = make_overwrite_story(
            rng,
            n_ent,
            n_ov,
            train_sents,
            filler_sentences_after=2,
            distractor_prob=0.30,
        )
        ids = tokenizer.encode(prompt + target, add_bos=True, add_eos=True)
        if len(ids) <= target_w:
            ids = ids + [tokenizer.PAD_ID] * (target_w - len(ids))
            rows.append(ids)
            added_ov += 1

    rng.shuffle(rows)
    return torch.tensor(rows, dtype=torch.long)


def build_eval_s5_suite(
    tokenizer: ByteBPETokenizer,
    test_sents: List[str],
    num_ops: int,
    n_samples: int = 200,
    distractors_per_op: Tuple[int, int] = (1, 2),
    seed: int = 1000,
) -> List[EvalExample]:
    rng = random.Random(seed + num_ops * 97)
    out: List[EvalExample] = []
    for _ in range(n_samples):
        prompt, target, state = make_s5_story(
            rng, num_ops, test_sents, distractors_per_op=distractors_per_op
        )
        p_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        full_ids = tokenizer.encode(prompt + target, add_bos=True, add_eos=False)
        t_ids = full_ids[len(p_ids) :]
        # Ensure lossless prefix alignment across BPE boundary
        if full_ids[: len(p_ids)] != p_ids:
            p_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
            t_ids = tokenizer.encode(target, add_bos=False, add_eos=False)
        out.append(EvalExample(p_ids, t_ids, len(p_ids) + len(t_ids), ",".join(state)))
    return out


def build_eval_overwrite_suite(
    tokenizer: ByteBPETokenizer,
    test_sents: List[str],
    num_entities: int,
    num_overwrites: int,
    filler_after: int,
    n_samples: int = 200,
    seed: int = 2000,
) -> List[EvalExample]:
    rng = random.Random(seed + num_entities * 31 + filler_after * 131)
    out: List[EvalExample] = []
    for _ in range(n_samples):
        prompt, target, city = make_overwrite_story(
            rng,
            num_entities,
            num_overwrites,
            test_sents,
            filler_sentences_after=filler_after,
        )
        p_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        full_ids = tokenizer.encode(prompt + target, add_bos=True, add_eos=False)
        if full_ids[: len(p_ids)] == p_ids:
            t_ids = full_ids[len(p_ids) :]
        else:
            t_ids = tokenizer.encode(target, add_bos=False, add_eos=False)
        out.append(EvalExample(p_ids, t_ids, len(p_ids) + len(t_ids), city))
    return out


def build_eval_trap_suite(
    tokenizer: ByteBPETokenizer,
    test_sents: List[str],
    num_entities: int,
    num_overwrites: int,
    filler_after: int,
    n_samples: int = 200,
    seed: int = 5000,
) -> List[EvalExample]:
    rng = random.Random(seed + num_entities * 43 + filler_after * 157)
    out: List[EvalExample] = []
    for _ in range(n_samples):
        prompt, target, ans = make_trap_overwrite_story(
            rng,
            num_entities,
            num_overwrites,
            test_sents,
            filler_sentences_after=filler_after,
        )
        p_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        full_ids = tokenizer.encode(prompt + target, add_bos=True, add_eos=False)
        if full_ids[: len(p_ids)] == p_ids:
            t_ids = full_ids[len(p_ids) :]
        else:
            t_ids = tokenizer.encode(target, add_bos=False, add_eos=False)
        out.append(EvalExample(p_ids, t_ids, len(p_ids) + len(t_ids), ans))
    return out
