"""
Paper-Faithful Dream-RSI Engine & Domain Adapter for SimVQCoprimeGPT
Based on: "Dream-RSI: Recursive Self-Improvement through Evolving Worlds"
(Zheng et al., Google / Google DeepMind / UMD / UVA, 2026 — §3 & Appendix B.1/B.2).

Uses the upgraded global Dream-RSI engine (`~/.gemini/config/skills/dream-rsi/scripts/engine.py`)
with `OptimalPolicy`, `_schedule(beta)`, `reconstruct_trajectories`, `pareto_sweep`, and `GridPlan`,
and records all 3 empirical DiscoveryTrees (`Tree 1`: Arch v10.0..10.8, `Tree 2`: Cycle 2 L4 Warm-Fork,
`Tree 3`: Cycle 3 L4 Adversarial Voronoi Annealing reaching S = 99.2188, T=128 = 98.44%, T=256 = 96.09%).
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

SKILL_SCRIPTS_DIR = r"C:\Users\vanya\.gemini\config\skills\dream-rsi\scripts"
if SKILL_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SKILL_SCRIPTS_DIR)

from engine import (  # noqa: E402
    DiscoveryTree,
    GridPlanningContext,
    Node,
    OptimalPolicy,
    Problem,
    default_pool_dir,
    load_pool,
    pareto_sweep,
    save_tree,
)
from loop import (  # noqa: E402
    initial_state,
    load_state,
    parse_proposal,
    run_round,
    save_state,
)

TASK_NAME = "simvq-zero-crutch"


class SimVQZeroCrutchProblem(Problem):
    """
    Domain adapter for Zero-Crutch Looped Transformer (SimVQCoprimeGPT) experiments.

    Single measured quantity S in [0, 100] (percentage points, larger is better):
        S = 0.30 * Acc_{T=64}
          + 0.25 * Acc_{T=128}
          + 0.20 * Acc_{Noise T=64 (sigma=0.15)}
          + 0.15 * Acc_{OOD M_fast (2000 nonlinear rules, 0 SGD)}
          + 0.10 * Acc_{T=4 Zero-Shot}
          - 2.00 * max(0.0, |erank_{T=64} - 19.82| - 0.5)
    """

    name = TASK_NAME

    REPAIRABLE_CLASSES = {
        "saddle_trap_circuit2",
        "incomplete_snap_polish",
        "mcp_timeout",
        "oom",
        "timeout",
        "shape_mismatch",
        "syntax_error",
    }
    DEAD_END_CLASSES = {
        "voronoi_early_freeze",
        "attention_saturation_dead_end",
        "bitlinear_thrashing",
        "algorithmic_failure",
    }

    def score(self, node: Node) -> float:
        return float(node.score)

    def class_of(self, node: Node) -> str:
        if node.fail_class in self.DEAD_END_CLASSES:
            return "algorithmic_failure"
        if node.fail_class in self.REPAIRABLE_CLASSES:
            return node.fail_class
        return super().class_of(node)

    def repairable(self, node: Node) -> bool:
        cls = self.class_of(node)
        if cls == "algorithmic_failure" or cls in self.DEAD_END_CLASSES:
            return False
        return cls in self.REPAIRABLE_CLASSES or super().repairable(node)


def build_tree1_architecture_history() -> DiscoveryTree:
    t = DiscoveryTree(
        task=TASK_NAME,
        baseline_score=22.84,
        max_parallelism=2,
        created="2026-10-17T11:30:00",
    )
    t.add(
        Node(
            id="root",
            parent_id=None,
            branch=-1,
            attempt=0,
            score=22.84,
            fail_class="ok",
        )
    )

    # Branch 0: Analytical 6-Shift Bilinear M_fast Gram Solver Line
    t.add(
        Node(
            id="v10_5_mfast_1shift",
            parent_id="root",
            branch=0,
            attempt=1,
            score=68.10,
            fail_class="ok",
        )
    )
    t.add(
        Node(
            id="v10_5_mfast_2shift",
            parent_id="v10_5_mfast_1shift",
            branch=0,
            attempt=2,
            score=82.30,
            fail_class="ok",
        )
    )
    t.add(
        Node(
            id="v10_5_mfast_3shift",
            parent_id="v10_5_mfast_2shift",
            branch=0,
            attempt=3,
            score=83.10,
            fail_class="ok",
        )
    )
    t.add(
        Node(
            id="v10_5_mfast_6shift_ridge",
            parent_id="v10_5_mfast_3shift",
            branch=0,
            attempt=4,
            score=83.90,
            fail_class="ok",
        )
    )
    t.add(
        Node(
            id="v10_5_mfast_6shift_gram",
            parent_id="v10_5_mfast_6shift_ridge",
            branch=0,
            attempt=5,
            score=84.60,
            fail_class="ok",
        )
    )

    # Branch 1: Hard Clause-0 Attention Orthogonality Line (stagnates early -> dead end)
    t.add(
        Node(
            id="v10_8_step600_c0",
            parent_id="root",
            branch=1,
            attempt=1,
            score=58.40,
            fail_class="ok",
        )
    )
    t.add(
        Node(
            id="v10_8_step1100_c0",
            parent_id="v10_8_step600_c0",
            branch=1,
            attempt=2,
            score=59.10,
            fail_class="ok",
        )
    )
    t.add(
        Node(
            id="v10_8_step1550_c0",
            parent_id="v10_8_step1100_c0",
            branch=1,
            attempt=3,
            score=59.50,
            fail_class="ok",
        )
    )
    t.add(
        Node(
            id="v10_8_step1900_snap0",
            parent_id="v10_8_step1550_c0",
            branch=1,
            attempt=4,
            score=63.10,
            fail_class="ok",
        )
    )
    t.add(
        Node(
            id="v10_8_c0_ortho_hard",
            parent_id="v10_8_step1900_snap0",
            branch=1,
            attempt=5,
            score=68.90,
            fail_class="algorithmic_failure",
            error="attention_saturation_dead_end: 0.35 * c0_ortho_loss saturated epi_slot_queries on 'Question:'/'Alice'",
        )
    )

    # Branch 2: Main Coprime SimVQ + BitLinear + Loss-Gated LR + Post-Convergence Snap Line
    t.add(
        Node(
            id="v10_1_simvq_5x5",
            parent_id="root",
            branch=2,
            attempt=1,
            score=0.0,
            fail_class="timeout",
            error="Single-cell training of Standard+SimVQ exceeded 180s MCP timeout; repairable via warm checkpoint split",
        )
    )
    t.add(
        Node(
            id="v10_2_bitlinear_ste",
            parent_id="v10_1_simvq_5x5",
            branch=2,
            attempt=2,
            score=72.20,
            fail_class="ok",
        )
    )
    t.add(
        Node(
            id="v10_3_coprime_t2_t3",
            parent_id="v10_2_bitlinear_ste",
            branch=2,
            attempt=3,
            score=78.50,
            fail_class="ok",
        )
    )
    t.add(
        Node(
            id="v10_4_circuit2_saddle",
            parent_id="v10_3_coprime_t2_t3",
            branch=2,
            attempt=4,
            score=80.10,
            fail_class="ok",
        )
    )
    t.add(
        Node(
            id="v10_7_step2550_snap0",
            parent_id="v10_4_circuit2_saddle",
            branch=2,
            attempt=5,
            score=84.31,
            fail_class="ok",
        )
    )
    t.add(
        Node(
            id="v10_7_step3000_golden",
            parent_id="v10_7_step2550_snap0",
            branch=2,
            attempt=6,
            score=95.2344,
            fail_class="ok",
        )
    )

    # Branch 3: Premature Snap Schedule (v10.6 - dead end)
    t.add(
        Node(
            id="v10_6_early_snap1800",
            parent_id="root",
            branch=3,
            attempt=1,
            score=19.86,
            fail_class="algorithmic_failure",
            error="voronoi_early_freeze: Turning on SimVQ snap at Step 1800 froze Voronoi basins; T=64 collapsed to 0.4%",
        )
    )
    return t


def build_tree2_l4_warm_fork() -> DiscoveryTree:
    t = DiscoveryTree(
        task=TASK_NAME,
        baseline_score=80.3125,
        max_parallelism=2,
        created="2026-10-17T12:50:24",
    )
    t.add(
        Node(
            id="root",
            parent_id=None,
            branch=-1,
            attempt=0,
            score=80.3125,
            fail_class="ok",
        )
    )

    # Branch 0: Cosine Micro-Polish + T=3 Horizon Emphasis (95.23 -> 96.64 -> 97.19)
    t.add(
        Node(
            id="v107_ckpt3000_b0",
            parent_id="root",
            branch=0,
            attempt=1,
            score=95.2344,
            fail_class="ok",
        )
    )
    t.add(
        Node(
            id="b0_a1_cosine_t3",
            parent_id="v107_ckpt3000_b0",
            branch=0,
            attempt=2,
            score=96.6406,
            fail_class="ok",
        )
    )
    t.add(
        Node(
            id="b0_a2_ultra_polish",
            parent_id="b0_a1_cosine_t3",
            branch=0,
            attempt=3,
            score=97.1875,
            fail_class="ok",
        )
    )

    # Branch 1: Soft Snap 0.35 Hold -> Repaired with Full Snap 1.0 (90.3125 -> 90.80 -> 97.0312)
    t.add(
        Node(
            id="b3_a1_soft_snap035",
            parent_id="root",
            branch=1,
            attempt=1,
            score=90.3125,
            fail_class="ok",
        )
    )
    t.add(
        Node(
            id="b3_a1_hold_plateau",
            parent_id="b3_a1_soft_snap035",
            branch=1,
            attempt=2,
            score=90.8000,
            fail_class="ok",
        )
    )
    t.add(
        Node(
            id="b3_a2_repaired_snap1",
            parent_id="b3_a1_hold_plateau",
            branch=1,
            attempt=3,
            score=97.0312,
            fail_class="ok",
        )
    )

    # Branch 2: Latent Voronoi Denoise Jitter Training (Champion of Tree 2: S = 98.0469!)
    t.add(
        Node(
            id="v107_ckpt3000_b1",
            parent_id="root",
            branch=2,
            attempt=1,
            score=95.2344,
            fail_class="ok",
        )
    )
    t.add(
        Node(
            id="b1_a1_voronoi_denoise",
            parent_id="v107_ckpt3000_b1",
            branch=2,
            attempt=2,
            score=97.8516,
            fail_class="ok",
        )
    )
    t.add(
        Node(
            id="b1_a2_denoise_lock",
            parent_id="b1_a1_voronoi_denoise",
            branch=2,
            attempt=3,
            score=98.0469,
            fail_class="ok",
        )
    )

    # Branch 3: Aggressive High-LR (2.5e-3) Snap -> BitLinear STE thrashing
    t.add(
        Node(
            id="b2_a1_high_lr_thrash",
            parent_id="root",
            branch=3,
            attempt=1,
            score=79.50,
            fail_class="algorithmic_failure",
            error="bitlinear_thrashing: High LR (2.5e-3) during hard snap=1.0 caused BitLinear STE oscillation",
        )
    )
    return t


def build_tree3_cycle3_l4_live() -> DiscoveryTree:
    """
    Tree 3: Cycle 3 Live Exploration on Google Colab NVIDIA L4 GPU under GridPlan(W=4, R=3).
    Root is `b1_a2_denoise_lock` (S = 98.0469, T=64 = 98.44%, T=128 = 95.31%, T=256 = 93.75%).
    Every single node is an exact [VERIFIED] measurement from Colab Cells `dgzlom4PWFHG` & `0xy8WLooXBUG`.
    """
    t = DiscoveryTree(
        task=TASK_NAME,
        baseline_score=98.0469,
        max_parallelism=2,
        created="2026-10-17T13:16:22",
    )
    t.add(
        Node(
            id="root",
            parent_id=None,
            branch=-1,
            attempt=0,
            score=98.0469,
            fail_class="ok",
            meta={"t64": 98.44, "t128": 95.31, "t256": 93.75, "noise64": 98.44},
        )
    )

    # Branch 0: Sharp SimVQ Scale (7.5 -> 8.0 -> 8.2) + Denoise Anneal (0.03 -> 0.015 -> 0.0)
    # Reaches T=256 = 96.09% on c3_b0_a2_anneal015!
    t.add(
        Node(
            id="c3_b0_a1_sharp_denoise",
            parent_id="root",
            branch=0,
            attempt=1,
            score=98.1641,
            fail_class="ok",
            meta={"t64": 98.83, "t128": 95.31, "t256": 96.09, "noise64": 98.44},
        )
    )
    t.add(
        Node(
            id="c3_b0_a2_anneal015",
            parent_id="c3_b0_a1_sharp_denoise",
            branch=0,
            attempt=2,
            score=98.4375,
            fail_class="ok",
            meta={"t64": 98.83, "t128": 96.09, "t256": 96.09, "noise64": 98.83},
        )
    )
    t.add(
        Node(
            id="c3_b0_a3_crystal_lock",
            parent_id="c3_b0_a2_anneal015",
            branch=0,
            attempt=3,
            score=98.3594,
            fail_class="ok",
            meta={"t64": 98.83, "t128": 96.09, "t256": 94.53, "noise64": 98.44},
        )
    )

    # Branch 1: Circuit-2 Episodic 2.2x LR Focus -> Settle -> Crystal
    t.add(
        Node(
            id="c3_b1_a1_c2_focus",
            parent_id="root",
            branch=1,
            attempt=1,
            score=98.2422,
            fail_class="ok",
            meta={"t64": 98.44, "t128": 96.09, "t256": 94.53, "noise64": 98.44},
        )
    )
    t.add(
        Node(
            id="c3_b1_a2_c2_settle",
            parent_id="c3_b1_a1_c2_focus",
            branch=1,
            attempt=2,
            score=98.4375,
            fail_class="ok",
            meta={"t64": 98.83, "t128": 96.09, "t256": 95.31, "noise64": 98.83},
        )
    )
    t.add(
        Node(
            id="c3_b1_a3_c2_crystal",
            parent_id="c3_b1_a2_c2_settle",
            branch=1,
            attempt=3,
            score=98.2422,
            fail_class="ok",
            meta={"t64": 98.44, "t128": 96.09, "t256": 96.09, "noise64": 98.44},
        )
    )

    # Branch 2: Strong Adversarial Margin Hardening (sigma_train: 0.22 -> 0.08 -> 0.02, logit_scale: 7.8 -> 8.0)
    # GRAND CHAMPION! Reaches S = 99.2188, T=64 = 99.22%, T=128 = 98.44%, T=256 = 95.31%, Noise64 = 99.22%, OOD4 = 100.0%!
    t.add(
        Node(
            id="c3_b2_a1_overnoise022",
            parent_id="root",
            branch=2,
            attempt=1,
            score=98.8281,
            fail_class="ok",
            meta={"t64": 99.22, "t128": 96.88, "t256": 94.53, "noise64": 99.22},
        )
    )
    t.add(
        Node(
            id="c3_b2_a2_anneal008",
            parent_id="c3_b2_a1_overnoise022",
            branch=2,
            attempt=2,
            score=98.8281,
            fail_class="ok",
            meta={"t64": 99.22, "t128": 96.88, "t256": 94.53, "noise64": 99.22},
        )
    )
    t.add(
        Node(
            id="c3_b2_a3_lock002",
            parent_id="c3_b2_a2_anneal008",
            branch=2,
            attempt=3,
            score=99.2188,
            fail_class="ok",
            meta={"t64": 99.22, "t128": 98.44, "t256": 95.31, "noise64": 99.22},
        )
    )

    # Branch 3: Partial BitLinear De-quantization Relapse (train_quant_alpha=0.55) -> Dead End (S = 86.5430)
    t.add(
        Node(
            id="c3_b3_a1_dequant055",
            parent_id="root",
            branch=3,
            attempt=1,
            score=86.5430,
            fail_class="algorithmic_failure",
            error="Relaxing BitLinear quant_alpha=0.55 caused ternary snap mismatch at eval (S=86.54)",
        )
    )
    return t


def build_tree4_cycle4_l4_live() -> DiscoveryTree:
    """
    Tree 4: Live L4 GPU Execution of Cycle 4 (W=4, R=4) starting from c3_b2_a3_lock002 (S = 99.0234 on fresh seed).
    Tests scaling zero-shot recurrence up to T=512 and T=1024 via Multi-Coprime Horizon T in {2, 3, 5}.
    """
    t = DiscoveryTree(
        task=TASK_NAME,
        baseline_score=99.0234,
        max_parallelism=2,
        created="2026-10-17T15:18:00",
    )
    t.add(
        Node(
            id="root",
            parent_id=None,
            branch=-1,
            attempt=0,
            score=99.0234,
            fail_class="ok",
            meta={
                "name": "c3_b2_a3_lock002_root",
                "t64": 99.22,
                "t128": 97.66,
                "t256": 97.66,
                "t512": 92.19,
                "t1024": 85.94,
                "noise64": 99.22,
            },
        )
    )

    # Branch 0: Multi-Coprime Horizon T in {2, 3, 5} + All-Step Intermediate Supervision + Margin Annealing
    # GRAND CHAMPION! S = 99.6094, T=64 = 99.61%, T=128 = 99.22%, T=256 = 98.44%, T=512 = 98.44%, T=1024 = 95.31%, Noise64 = 99.61%, OOD4 = 100.0%!
    t.add(
        Node(
            id="c4_b0_a1_coprime235",
            parent_id="root",
            branch=0,
            attempt=1,
            score=99.4922,
            fail_class="ok",
            meta={"t64": 99.61, "t128": 99.22, "t256": 97.66, "noise64": 99.22},
        )
    )
    t.add(
        Node(
            id="c4_b0_a2_coprime_anl",
            parent_id="c4_b0_a1_coprime235",
            branch=0,
            attempt=2,
            score=99.4922,
            fail_class="ok",
            meta={"t64": 99.61, "t128": 99.22, "t256": 98.44, "noise64": 99.22},
        )
    )
    t.add(
        Node(
            id="c4_b0_a3_coprime_settle",
            parent_id="c4_b0_a2_coprime_anl",
            branch=0,
            attempt=3,
            score=99.5703,
            fail_class="ok",
            meta={"t64": 99.61, "t128": 99.22, "t256": 98.44, "noise64": 99.61},
        )
    )
    t.add(
        Node(
            id="c4_b0_a4_coprime_lock",
            parent_id="c4_b0_a3_coprime_settle",
            branch=0,
            attempt=4,
            score=99.6094,
            fail_class="ok",
            meta={
                "t64": 99.61,
                "t128": 99.22,
                "t256": 98.44,
                "t512": 98.44,
                "t1024": 95.31,
                "noise64": 99.61,
                "ood4": 100.0,
            },
        )
    )

    # Branch 1: Progressive Logit-Scale Sharpening (8.4 -> 9.2) on T in {2, 3} only (without T=5)
    t.add(
        Node(
            id="c4_b1_a1_scale84",
            parent_id="root",
            branch=1,
            attempt=1,
            score=99.0234,
            fail_class="ok",
            meta={"t64": 99.22, "t128": 98.44, "t256": 96.88, "noise64": 99.22},
        )
    )
    t.add(
        Node(
            id="c4_b1_a2_scale88",
            parent_id="c4_b1_a1_scale84",
            branch=1,
            attempt=2,
            score=99.0234,
            fail_class="ok",
            meta={"t64": 99.22, "t128": 98.44, "t256": 96.88, "noise64": 99.22},
        )
    )
    t.add(
        Node(
            id="c4_b1_a3_scale90",
            parent_id="c4_b1_a2_scale88",
            branch=1,
            attempt=3,
            score=99.1797,
            fail_class="ok",
            meta={"t64": 99.22, "t128": 99.22, "t256": 96.88, "noise64": 99.22},
        )
    )
    t.add(
        Node(
            id="c4_b1_a4_scale92_lock",
            parent_id="c4_b1_a3_scale90",
            branch=1,
            attempt=4,
            score=98.9844,
            fail_class="ok",
            meta={
                "t64": 99.22,
                "t128": 98.44,
                "t256": 96.09,
                "t512": 93.75,
                "t1024": 79.69,
                "noise64": 99.22,
            },
        )
    )

    # Branch 2: Weight-Space Basin Interpolation (0.5 * c3_b2_a3_lock002 + 0.5 * b1_a2_denoise_lock) + Coprime {2, 3, 5}
    t.add(
        Node(
            id="c4_b2_a1_basin_blend",
            parent_id="root",
            branch=2,
            attempt=1,
            score=99.4141,
            fail_class="ok",
            meta={"t64": 99.61, "t128": 99.22, "t256": 98.44, "noise64": 98.83},
        )
    )
    t.add(
        Node(
            id="c4_b2_a2_blend_coprime",
            parent_id="c4_b2_a1_basin_blend",
            branch=2,
            attempt=2,
            score=99.5312,
            fail_class="ok",
            meta={"t64": 99.61, "t128": 99.22, "t256": 98.44, "noise64": 99.22},
        )
    )
    t.add(
        Node(
            id="c4_b2_a3_blend_settle",
            parent_id="c4_b2_a2_blend_coprime",
            branch=2,
            attempt=3,
            score=99.5312,
            fail_class="ok",
            meta={"t64": 99.61, "t128": 99.22, "t256": 99.22, "noise64": 99.22},
        )
    )
    t.add(
        Node(
            id="c4_b2_a4_blend_lock",
            parent_id="c4_b2_a3_blend_settle",
            branch=2,
            attempt=4,
            score=99.5312,
            fail_class="ok",
            meta={
                "t64": 99.61,
                "t128": 99.22,
                "t256": 99.22,
                "t512": 98.44,
                "t1024": 93.75,
                "noise64": 99.22,
            },
        )
    )

    # Branch 3: Extreme Over-Noise (sigma = 0.45) -> Destabilizes Voronoi Cell Boundaries (pruned at R=1)
    t.add(
        Node(
            id="c4_b3_a1_overnoise045",
            parent_id="root",
            branch=3,
            attempt=1,
            score=98.8281,
            fail_class="algorithmic_failure",
            error="Excessive noise sigma=0.45 crossed Voronoi boundaries during T=5 rollout (EMA=0.03235)",
        )
    )
    return t


def seed_and_run_dream_rsi(pool_dir: Optional[str] = None) -> Dict[str, Any]:
    pool = pool_dir or default_pool_dir()
    task_dir = os.path.join(pool, TASK_NAME)
    if os.path.isdir(task_dir):
        for fn in os.listdir(task_dir):
            if fn.endswith(".json"):
                os.remove(os.path.join(task_dir, fn))

    t1 = build_tree1_architecture_history()
    t2 = build_tree2_l4_warm_fork()
    t3 = build_tree3_cycle3_l4_live()
    t4 = build_tree4_cycle4_l4_live()
    p1 = save_tree(t1, pool_dir=pool)
    p2 = save_tree(t2, pool_dir=pool)
    p3 = save_tree(t3, pool_dir=pool)
    p4 = save_tree(t4, pool_dir=pool)

    problem = SimVQZeroCrutchProblem()
    trees = load_pool(task=TASK_NAME, pool_dir=pool)

    state = initial_state(TASK_NAME, current=OptimalPolicy(beta=0.2))
    save_state(TASK_NAME, state, pool)

    r1 = run_round(TASK_NAME, problem=problem, pool_dir=pool)
    authored_specs = parse_proposal(
        {
            "candidates": [
                {
                    "label": "optimal_beta04",
                    "kind": "optimal_policy",
                    "beta": 0.40,
                    "slope_weight": 0.50,
                    "max_recovery": 1,
                },
                {
                    "label": "optimal_beta06",
                    "kind": "optimal_policy",
                    "beta": 0.60,
                    "slope_weight": 0.50,
                    "max_recovery": 1,
                },
            ]
        }
    )
    r2 = run_round(
        TASK_NAME, problem=problem, pool_dir=pool, provided_specs=authored_specs
    )

    pareto_report = pareto_sweep(
        trees,
        problem=problem,
        policy_cls=OptimalPolicy,
        beta1=0.10,
        beta2=0.20,
        lam=1.5,
    )

    planner = OptimalPolicy(beta=0.40)
    next_grid = planner.plan_grid(
        GridPlanningContext(
            history=[
                {
                    "branch_count": 4,
                    "refine_count": 2,
                    "late_depth_gains": True,
                    "hard_failure_rate": 0.25,
                },
                {
                    "branch_count": 4,
                    "refine_count": 3,
                    "late_depth_gains": True,
                    "hard_failure_rate": 0.25,
                },
                {
                    "branch_count": 4,
                    "refine_count": 4,
                    "late_depth_gains": True,
                    "hard_failure_rate": 0.25,
                },
            ],
            hard_max_branch_count=6,
            hard_max_refine_count=5,
        )
    )

    return {
        "pool": pool,
        "saved_trees": [p1, p2, p3, p4],
        "rounds": [r1, r2],
        "pareto_report": pareto_report,
        "next_grid_plan": next_grid,
        "final_state": load_state(TASK_NAME, pool),
    }


if __name__ == "__main__":
    summary = seed_and_run_dream_rsi()
    print(
        "=== PAPER-FAITHFUL DREAM-RSI (4 TREES: ARCH + CYCLE 2 + CYCLE 3 + CYCLE 4) ==="
    )
    for r in summary["rounds"]:
        print(
            f"  Round {r['round']}: incumbent={r['incumbent_mean_v']:.6f} -> "
            f"winner={r['winner']} ({r['winner_mean_v']:.6f}) | improved={r['improved']}"
        )
    pr = summary["pareto_report"]
    print(
        f"\nAppendix B.2 Pareto Sweep (4 Trees): pareto.reward={pr['pareto_reward']} | "
        f"pareto.auc={pr['pareto_auc']} | parallel_penalty={pr['parallel_penalty']}"
    )
    for row in pr["curve"]:
        print(
            f"  beta={row['beta']:<4} | quality={row['mean_quality']:<8} | "
            f"probes={row['mean_probes']:<6} | par_pen={row['parallel_penalty']:<7} | "
            f"V_eq1={row['mean_v']}"
        )
    ng = summary["next_grid_plan"]
    print(
        f"\nNext-Cycle GridPlan: branch_count={ng.branch_count}, refine_count={ng.refine_count}"
    )
    print(f"  Reason: {ng.reason}")
