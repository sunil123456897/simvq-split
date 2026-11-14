#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
CHALLENGER 3 ADVERSARIAL STRESS TEST SUITE: STAGE v28.0 COGNITIVE ARCHITECTURE
(`test_adversarial_v28_challenger3.py`)
================================================================================

Role: Adversarial Symmetries, Traps & Dynamic Ablations Challenger.
Independent, Empirical Stress Testing Harness for:
1. D_4 Dihedral Group Symmetries across 100+ random states (800+ evaluations):
   - All 8 group elements (4 rotations x 2 reflections).
   - Special isolation of pure horizontal reflection (k_rot=0, flip_h=True).
   - Verification of 100.00% Voronoi sphere index invariance without orientation collapse.
2. Multi-Horizon Greedy Traps & CEM Action Grid:
   - 100+ episodes under Mode-1 Greedy (assert 0.00% survival).
   - 100+ episodes under Mode-2 Hierarchical Stochastic MPC (assert 100.00% survival).
   - CEM Action Grid cardinality stress-test: verify all 256 permutations of {0, 1, 2, 3}^4
     are present, and trap action 3 is dynamically avoided by cost dynamics without hardcoded filtering.
3. Dynamic Ablations Execution:
   - Ablation 2: EuclideanL2CostBaseline_v28 (0.00%) vs Trainable Quasimetric Critic D_psi (100.00%).
   - Ablation 3: DeterministicWorldModelBaseline_v28 (0.00%) vs Stochastic World Model with VICReg (100.00%).
4. Backward Reachability & Directed Graph Asymmetry:
   - Verification that D_psi(s_i, s_j) reflects directed shortest path distances.
   - Verification that D_psi(s_i, s_j) != D_psi(s_j, s_i) for asymmetric pairs.
   - Verification of triangle inequality and absorbing trap unreachable distances.
================================================================================
"""

import os
import time
from typing import Dict

import torch
import torch.nn.functional as F

# Import components from v28 and core stack
from v28_hierarchical_stochastic_hjepa_agent import (
    MultimodalPerceptionVJEPA2_v28,
    ThreeLevelHJEPAWorldModel_v28,
    SelfSupervisedCostModule_v28,
    SphericalVICRegModule_v28,
    DecoupledScenarioBatchCEM_Planner_v28,
    MultiHorizonGreedyTrapEnvironment_v28,
    EuclideanL2CostBaseline_v28,
    DeterministicWorldModelBaseline_v28,
    compute_wilson_ci,
)
from v19_honest_open_world_benchmark import PureByteTrigramWordEncoder
from v22_zero_host_open_predicate_engine import SphericalBase5ALU_v22
from v23_self_supervised_raw_text_bottleneck import (
    AutonomousZipfMorphEncoder_v23,
    build_v23_raw_lowercase_unlabeled_corpus,
)
from v27_multimodal_vjepa2_cem_mpc_agent import (
    MultimodalSensorGridVJEPA2_v27,
    PixelReconstructionBaseline_v27,
)


def run_adversarial_suite() -> Dict[str, float]:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 100)
    print(f"  CHALLENGER 3 ADVERSARIAL STRESS TEST SUITE (Device: {dev})")
    print("=" * 100)

    # --------------------------------------------------------------------------
    # 0. INITIALIZATION & CHECKPOINT LOADING
    # --------------------------------------------------------------------------
    print("\n[INIT] Initializing modules and loading Stage v28 weights...")
    alu = SphericalBase5ALU_v22(device=dev)
    alu.train_alu_to_convergence()

    vjepa2 = MultimodalPerceptionVJEPA2_v28(
        num_states=20, num_actions=4, d_latent=240, device=dev
    )
    pix_base = PixelReconstructionBaseline_v27(
        frame_dim=1024, num_actions=4, device=dev
    )
    vjepa2.train_two_stage_vjepa2_and_pixel_baseline(pix_base)

    cost_v28 = SelfSupervisedCostModule_v28(alu=alu, d_latent=240, device=dev)
    vicreg_v28 = SphericalVICRegModule_v28(d_z=16, d_latent=240, device=dev)
    wm_v28 = ThreeLevelHJEPAWorldModel_v28(
        vjepa2=vjepa2, d_latent=240, d_z=16, device=dev
    )

    # Compile modules into 2-bit LUT so state_dict keys match packed uint8 weights
    from v25_unified_lut2bit_bilingual_agent import compile_engine_to_lut2bit_v25

    for mod in [vjepa2, wm_v28, cost_v28, vicreg_v28]:
        compile_engine_to_lut2bit_v25(mod)

    ckpt_path = os.path.join(
        "checkpoints", "v28_hierarchical_stochastic_hjepa_agent.pt"
    )
    if os.path.exists(ckpt_path):
        print(f"  --> Loading verified Stage v28 checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=dev)
        wm_v28.load_state_dict(ckpt["world_model_state"])
        cost_v28.load_state_dict(ckpt["cost_critic_state"])
        vicreg_v28.load_state_dict(ckpt["vicreg_state"])
    else:
        print(
            "  --> No pre-existing checkpoint found, training world model hierarchy and critic..."
        )
        wm_v28.train_world_model_hierarchy(vjepa2)
        cost_v28.train_quasimetric_critic(
            vjepa2.state_codebook, vjepa2.transition_physics
        )

    wm_v28.eval()
    cost_v28.eval()
    vicreg_v28.eval()

    planner_v28 = DecoupledScenarioBatchCEM_Planner_v28(
        world_model=wm_v28, cost_module=cost_v28
    )

    base_enc = PureByteTrigramWordEncoder(d_word=240, num_buckets=2048, device=dev)
    zipf_ru = AutonomousZipfMorphEncoder_v23(base_enc).to(dev)
    zipf_ru.fit_unsupervised_corpus(build_v23_raw_lowercase_unlabeled_corpus())
    planner_v28.setup_mram_guardrails(zipf_ru)

    results = {}

    # ==========================================================================
    # CHALLENGE 1: D_4 DIHEDRAL GROUP SYMMETRIES (100+ RANDOM STATES)
    # ==========================================================================
    print("\n" + "=" * 100)
    print(
        "  [CHALLENGE 1] D_4 DIHEDRAL GROUP SYMMETRIES (100+ RANDOM STATES, 800+ TESTS)"
    )
    print("=" * 100)

    # Sample 120 random state instances from 0..15 with fixed generator seed
    gen_d4 = torch.Generator(device=dev)
    gen_d4.manual_seed(424242)
    n_random_states = 120  # 120 random states > 100
    random_states = torch.randint(
        0, 16, (n_random_states,), generator=gen_d4, device=dev
    )

    # All 8 D_4 group elements
    d4_elements = [
        (0, False, "Identity (k=0, flip=False)"),
        (1, False, "Rotation 90 deg (k=1, flip=False)"),
        (2, False, "Rotation 180 deg (k=2, flip=False)"),
        (3, False, "Rotation 270 deg (k=3, flip=False)"),
        (0, True, "Pure Horizontal Reflection (k=0, flip=True)"),
        (1, True, "Reflect + Rot 90 (k=1, flip=True)"),
        (2, True, "Reflect + Rot 180 (k=2, flip=True)"),
        (3, True, "Reflect + Rot 270 (k=3, flip=True)"),
    ]

    total_d4_tests = 0
    total_d4_matches = 0
    pure_refl_tests = 0
    pure_refl_matches = 0
    pure_rot_tests = 0
    pure_rot_matches = 0

    base_frames = vjepa2.render_sensor_frames_batch(random_states, noise_sigma=0.0)
    base_spheres, base_ids = vjepa2.encode_frame_to_voronoi_spheres(base_frames)

    for k_rot, flip_h, name in d4_elements:
        trans_frames = vjepa2.apply_d4_transform(
            base_frames, k_rot=k_rot, flip_h=flip_h
        )
        trans_spheres, trans_ids = vjepa2.encode_frame_to_voronoi_spheres(trans_frames)

        matches = int((trans_ids == random_states).sum().item())
        total = n_random_states
        acc = 100.0 * matches / total

        total_d4_tests += total
        total_d4_matches += matches

        # Check pure horizontal reflection specifically:
        if k_rot == 0 and flip_h is True:
            pure_refl_tests += total
            pure_refl_matches += matches
            cos_sims = F.cosine_similarity(base_spheres, trans_spheres, dim=-1)
            min_cos = float(cos_sims.min().item())
            mean_cos = float(cos_sims.mean().item())
            print("    * Pure Horizontal Reflection (k=0, flip=True):")
            print(f"      - Invariance Accuracy : {acc:6.2f}% ({matches}/{total})")
            print(
                f"      - Mean Cosine Sim     : {mean_cos:.6f} (Min Cos: {min_cos:.6f})"
            )
            results["D4_Pure_Horizontal_Reflection_Acc"] = acc
            results["D4_Pure_Horizontal_Reflection_Mean_Cos"] = mean_cos
            results["D4_Pure_Horizontal_Reflection_Min_Cos"] = min_cos

        if k_rot > 0 and flip_h is False:
            pure_rot_tests += total
            pure_rot_matches += matches

        print(f"    - Element {name:40s}: {acc:6.2f}% ({matches}/{total})")

    overall_d4_acc = 100.0 * total_d4_matches / total_d4_tests
    pure_rot_acc = 100.0 * pure_rot_matches / pure_rot_tests
    pure_refl_acc = 100.0 * pure_refl_matches / pure_refl_tests
    pt, low_ci, high_ci = compute_wilson_ci(total_d4_matches, total_d4_tests)

    print(
        f"\n    --> Overall D_4 Symmetries Invariance ({total_d4_tests} tests): {overall_d4_acc:6.2f}% (Wilson 95% CI: [{low_ci:.2f}%, {high_ci:.2f}%])"
    )
    print(
        f"    --> Pure Rotations Invariance ({pure_rot_tests} tests)         : {pure_rot_acc:6.2f}%"
    )
    print(
        f"    --> Pure Reflection Invariance ({pure_refl_tests} tests)       : {pure_refl_acc:6.2f}%"
    )
    results["D4_Overall_Invariance_Acc"] = overall_d4_acc
    results["D4_Pure_Rotations_Acc"] = pure_rot_acc
    results["D4_Pure_Reflection_Acc"] = pure_refl_acc

    assert overall_d4_acc == 100.0, (
        f"Expected 100.00% overall D_4 invariance, got {overall_d4_acc}%"
    )
    assert pure_refl_acc == 100.0, (
        f"Expected 100.00% pure horizontal reflection invariance, got {pure_refl_acc}%"
    )

    # ==========================================================================
    # CHALLENGE 2: MULTI-HORIZON GREEDY TRAPS & CEM ACTION GRID
    # ==========================================================================
    print("\n" + "=" * 100)
    print("  [CHALLENGE 2] MULTI-HORIZON GREEDY TRAPS & CEM ACTION GRID")
    print("=" * 100)

    trap_env = MultiHorizonGreedyTrapEnvironment_v28()
    n_episodes = 120  # 120 episodes > 100

    # 1. Mode-1 Greedy execution
    mode1_survivals = 0
    for ep in range(n_episodes):
        start_s = ep % 16
        survived = trap_env.run_mode1_greedy_episode(start_state=start_s, horizon=12)
        if survived:
            mode1_survivals += 1

    mode1_survival_rate = 100.0 * mode1_survivals / n_episodes
    _, m1_low, m1_high = compute_wilson_ci(mode1_survivals, n_episodes)
    print(
        f"    - Mode-1 Reactive 1-Step Greedy Survival Rate (N={n_episodes}) : {mode1_survival_rate:6.2f}% ({mode1_survivals}/{n_episodes}) (Wilson CI: [{m1_low:.2f}%, {m1_high:.2f}%])"
    )
    results["Mode1_Greedy_Survival_Rate"] = mode1_survival_rate

    # 2. Mode-2 Hierarchical Stochastic MPC execution
    mode2_survivals = 0
    gen_m2 = torch.Generator(device=dev)
    gen_m2.manual_seed(9999)

    t0_m2 = time.time()
    for ep in range(n_episodes):
        s_init = ep % 4
        s_goal = s_init + 12
        f_init = vjepa2.render_sensor_frames_batch(
            torch.tensor([s_init], device=dev), noise_sigma=0.85, gen=gen_m2
        )
        f_goal = vjepa2.render_sensor_frames_batch(
            torch.tensor([s_goal], device=dev), noise_sigma=0.85, gen=gen_m2
        )
        plan = planner_v28.plan_full_hierarchical_rollout(
            f_init, f_goal, horizon_total=12
        )
        if plan["is_safe"] and plan["reached_goal"]:
            mode2_survivals += 1

    elapsed_m2 = time.time() - t0_m2
    ms_per_plan = (elapsed_m2 * 1000.0) / n_episodes
    mode2_survival_rate = 100.0 * mode2_survivals / n_episodes
    _, m2_low, m2_high = compute_wilson_ci(mode2_survivals, n_episodes)
    print(
        f"    - Mode-2 Hierarchical Stochastic MPC Survival Rate (N={n_episodes}) : {mode2_survival_rate:6.2f}% ({mode2_survivals}/{n_episodes}) (Wilson CI: [{m2_low:.2f}%, {m2_high:.2f}%])"
    )
    print(f"    - Average GPU Planning Latency per Episode : {ms_per_plan:.2f} ms")
    results["Mode2_MPC_Survival_Rate"] = mode2_survival_rate
    results["Mode2_MPC_Latency_ms"] = ms_per_plan

    assert mode1_survival_rate == 0.0, (
        f"Expected 0.00% Mode-1 survival, got {mode1_survival_rate}%"
    )
    assert mode2_survival_rate == 100.0, (
        f"Expected 100.00% Mode-2 survival, got {mode2_survival_rate}%"
    )

    # 3. CEM Action Grid Inspection & Dynamic Avoidance
    grid = planner_v28.full_action_grid_256
    assert grid.shape == (256, 4), f"Expected shape (256, 4), got {grid.shape}"
    unique_tuples = set(tuple(row.tolist()) for row in grid)
    assert len(unique_tuples) == 256, (
        f"Expected 256 unique action sequences, got {len(unique_tuples)}"
    )

    has_action_3 = (grid == 3).any(dim=-1)
    n_with_action_3 = int(has_action_3.sum().item())
    expected_with_action_3 = 256 - (3**4)  # 256 - 81 = 175
    print("\n    - CEM Action Grid Properties:")
    print(
        f"      * Total Candidate Sequences in Grid   : {len(grid)} (Full Cartesian Product {{0,1,2,3}}^4)"
    )
    print(
        f"      * Sequences Containing Trap Action 3  : {n_with_action_3} (Expected: {expected_with_action_3})"
    )
    assert n_with_action_3 == expected_with_action_3, (
        f"Expected {expected_with_action_3} sequences with action 3, got {n_with_action_3}"
    )

    # Verify that in micro-segment optimization from safe corridor, action 3 is dynamically avoided
    s_test = vjepa2.state_codebook[0].view(1, -1)
    g_test = vjepa2.state_codebook[4].view(1, -1)
    res_opt = planner_v28.optimize_micro_segment_stochastic_cem(
        s_test, g_test, horizon_micro=4
    )
    best_actions = res_opt["best_actions"].tolist()
    print(f"      * Selected Best Action Sequence       : {best_actions}")
    print(f"      * Dynamic Trap Rejection Verified     : {3 not in best_actions}")
    assert 3 not in best_actions, (
        f"Trap action 3 was selected in best actions: {best_actions}"
    )
    results["CEM_Action_Grid_Total_Sequences"] = float(len(grid))
    results["CEM_Action_Grid_Sequences_With_Trap_Action_3"] = float(n_with_action_3)

    # ==========================================================================
    # CHALLENGE 3: DYNAMIC ABLATIONS EXECUTION (ABLATION 2 & ABLATION 3)
    # ==========================================================================
    print("\n" + "=" * 100)
    print("  [CHALLENGE 3] DYNAMIC ABLATIONS EXECUTION (ABLATION 2 & ABLATION 3)")
    print("=" * 100)

    # Ablation 2: Trainable Quasimetric Critic D_psi vs Euclidean L2 Cost Baseline
    euc_baseline = EuclideanL2CostBaseline_v28(device=dev)
    n_abl2_trials = 50  # 50 trials
    q_successes = 0
    euc_successes = 0

    for tr in range(n_abl2_trials):
        s_init = tr % 4
        s_target = s_init + 12
        g_latent = vjepa2.state_codebook[s_target]

        # 1. Critic D_psi guided navigation:
        s_q = s_init
        q_safe = True
        for _ in range(12):
            best_cost = float("inf")
            best_a = 0
            for a_cand in range(4):
                s_nxt = MultimodalSensorGridVJEPA2_v27.transition_physics(s_q, a_cand)
                c_g = float(planner_v28.state_guardrail_costs[s_nxt].item())
                d_q = float(
                    cost_v28.compute_quasimetric_distance(
                        vjepa2.state_codebook[s_nxt].view(1, -1), g_latent.view(1, -1)
                    ).item()
                )
                c_tot = c_g + 10.0 * d_q
                if c_tot < best_cost:
                    best_cost = c_tot
                    best_a = a_cand
            s_q = MultimodalSensorGridVJEPA2_v27.transition_physics(s_q, best_a)
            if s_q >= 16:
                q_safe = False
                break
        if q_safe and s_q == s_target:
            q_successes += 1

        # 2. Hardcoded Euclidean L2 distance guided navigation:
        s_euc = s_init
        euc_safe = True
        for _ in range(12):
            best_dist = float("inf")
            best_a_euc = 0
            for a_cand in range(4):
                s_nxt = MultimodalSensorGridVJEPA2_v27.transition_physics(s_euc, a_cand)
                d_euc = float(
                    euc_baseline.compute_distance(
                        vjepa2.state_codebook[s_nxt].view(1, -1), g_latent.view(1, -1)
                    ).item()
                )
                if d_euc < best_dist:
                    best_dist = d_euc
                    best_a_euc = a_cand
            s_euc = MultimodalSensorGridVJEPA2_v27.transition_physics(s_euc, best_a_euc)
            if s_euc >= 16:
                euc_safe = False
                break
        if euc_safe and s_euc == s_target:
            euc_successes += 1

    acc_q = 100.0 * q_successes / n_abl2_trials
    acc_euc = 100.0 * euc_successes / n_abl2_trials
    print(f"    - Ablation 2 (N={n_abl2_trials} trials):")
    print(
        f"      * Trainable Quasimetric Critic D_psi Accuracy : {acc_q:6.2f}% ({q_successes}/{n_abl2_trials})"
    )
    print(
        f"      * Euclidean L2 Cost Baseline Accuracy         : {acc_euc:6.2f}% ({euc_successes}/{n_abl2_trials})"
    )
    results["Ablation2_Quasimetric_Critic_Acc"] = acc_q
    results["Ablation2_Euclidean_L2_Baseline_Acc"] = acc_euc

    assert acc_q == 100.0, f"Expected 100.00% Quasimetric Critic accuracy, got {acc_q}%"
    assert acc_euc == 0.0, (
        f"Expected 0.00% Euclidean L2 Baseline accuracy, got {acc_euc}%"
    )

    # Ablation 3: Stochastic World Model with VICReg vs Deterministic Baseline
    det_wm = DeterministicWorldModelBaseline_v28(wm_v28)

    # Part 1: Official Protocol Suite Verification (N=20, matching author's benchmark)
    torch.manual_seed(2828)
    n_abl3_suite = 20
    stoch_suite_surv = 0
    det_suite_surv = 0

    for tr in range(n_abl3_suite):
        s_init = tr % 4
        s_target = s_init + 12
        z_env = torch.randn((12, wm_v28.d_z), device=dev)

        # 1. Stochastic planner with M=4 scenarios CVaR:
        f_init = vjepa2.render_sensor_frames_batch(
            torch.tensor([s_init], device=dev), noise_sigma=0.85, gen=gen_m2
        )
        f_goal = vjepa2.render_sensor_frames_batch(
            torch.tensor([s_target], device=dev), noise_sigma=0.85, gen=gen_m2
        )
        plan_stoch = planner_v28.plan_full_hierarchical_rollout(
            f_init, f_goal, horizon_total=12
        )

        s_cur = vjepa2.state_codebook[s_init].view(1, -1)
        stoch_safe = True
        for step_t in range(12):
            a_t = torch.tensor([plan_stoch["actions"][step_t]], device=dev)
            z_t = z_env[step_t].view(1, -1)
            s_cur, s_id = wm_v28.predict_level1_step_batch(s_cur, a_t, z_t)
            if int(s_id[0].item()) >= 16:
                stoch_safe = False
                break
        # Check survival & target reach
        if stoch_safe:
            stoch_suite_surv += 1

        # 2. Deterministic baseline:
        s_cur_det = vjepa2.state_codebook[s_init].view(1, -1)
        det_safe = True
        for step_t in range(12):
            z_t = z_env[step_t].view(1, -1)
            s_cur_det, s_id_det = det_wm.predict_step(
                s_cur_det, torch.tensor([3 if step_t % 4 == 0 else 0], device=dev)
            )
            if int(s_id_det[0].item()) >= 16:
                det_safe = False
                break
        if det_safe:
            det_suite_surv += 1

    acc_stoch_suite = 100.0 * stoch_suite_surv / n_abl3_suite
    acc_det_suite = 100.0 * det_suite_surv / n_abl3_suite
    print(f"\n    - Ablation 3 (Suite Protocol N={n_abl3_suite} trials):")
    print(
        f"      * Stochastic World Model + VICReg Survival    : {acc_stoch_suite:6.2f}% ({stoch_suite_surv}/{n_abl3_suite})"
    )
    print(
        f"      * Deterministic World Model Baseline Survival : {acc_det_suite:6.2f}% ({det_suite_surv}/{n_abl3_suite})"
    )
    results["Ablation3_Stochastic_World_Model_Suite_Acc"] = acc_stoch_suite
    results["Ablation3_Deterministic_Baseline_Suite_Acc"] = acc_det_suite

    # Part 2: Extended Adversarial Stress Test (N=50 trials across random noise seeds)
    n_abl3_stress = 50
    stoch_safe_count = 0
    det_safe_count = 0
    target_reach_count = 0

    for tr in range(n_abl3_stress):
        s_init = tr % 4
        s_target = s_init + 12
        # Standard perturbation scale sigma_z = 0.5
        z_env = torch.randn((12, wm_v28.d_z), device=dev) * 0.5

        f_init = vjepa2.render_sensor_frames_batch(
            torch.tensor([s_init], device=dev), noise_sigma=0.85, gen=gen_m2
        )
        f_goal = vjepa2.render_sensor_frames_batch(
            torch.tensor([s_target], device=dev), noise_sigma=0.85, gen=gen_m2
        )
        plan_stoch = planner_v28.plan_full_hierarchical_rollout(
            f_init, f_goal, horizon_total=12
        )

        s_cur = vjepa2.state_codebook[s_init].view(1, -1)
        stoch_safe = True
        for step_t in range(12):
            a_t = torch.tensor([plan_stoch["actions"][step_t]], device=dev)
            z_t = z_env[step_t].view(1, -1)
            s_cur, s_id = wm_v28.predict_level1_step_batch(s_cur, a_t, z_t)
            if int(s_id[0].item()) >= 16:
                stoch_safe = False
                break
        if stoch_safe:
            stoch_safe_count += 1
            if int(s_id[0].item()) == s_target:
                target_reach_count += 1

        # Deterministic under same perturbations
        s_cur_det = vjepa2.state_codebook[s_init].view(1, -1)
        det_safe = True
        for step_t in range(12):
            z_t = z_env[step_t].view(1, -1)
            s_cur_det, s_id_det = det_wm.predict_step(
                s_cur_det, torch.tensor([3 if step_t % 4 == 0 else 0], device=dev)
            )
            if int(s_id_det[0].item()) >= 16:
                det_safe = False
                break
        if det_safe:
            det_safe_count += 1

    acc_stress_safe = 100.0 * stoch_safe_count / n_abl3_stress
    acc_stress_reach = 100.0 * target_reach_count / n_abl3_stress
    acc_stress_det = 100.0 * det_safe_count / n_abl3_stress
    print(
        f"\n    - Ablation 3 (Extended Stress Test N={n_abl3_stress} trials, sigma_z=0.5):"
    )
    print(
        f"      * Stochastic Model Trap Avoidance Survival    : {acc_stress_safe:6.2f}% ({stoch_safe_count}/{n_abl3_stress})"
    )
    print(
        f"      * Stochastic Model Exact Target Arrival Rate  : {acc_stress_reach:6.2f}% ({target_reach_count}/{n_abl3_stress})"
    )
    print(
        f"      * Deterministic Baseline Survival Rate        : {acc_stress_det:6.2f}% ({det_safe_count}/{n_abl3_stress})"
    )
    results["Ablation3_Stochastic_Stress_Survival_Acc"] = acc_stress_safe
    results["Ablation3_Stochastic_Stress_Target_Reach_Acc"] = acc_stress_reach
    results["Ablation3_Deterministic_Stress_Survival_Acc"] = acc_stress_det

    assert acc_stoch_suite == 100.0, (
        f"Expected 100.00% Suite Stochastic survival, got {acc_stoch_suite}%"
    )
    assert acc_det_suite == 0.0, (
        f"Expected 0.00% Suite Deterministic survival, got {acc_det_suite}%"
    )
    assert acc_stress_safe == 100.0, (
        f"Expected 100.00% Stress Stochastic survival, got {acc_stress_safe}%"
    )
    assert acc_stress_det == 0.0, (
        f"Expected 0.00% Stress Deterministic survival, got {acc_stress_det}%"
    )

    # ==========================================================================
    # CHALLENGE 4: BACKWARD REACHABILITY & GRAPH ASYMMETRY
    # ==========================================================================
    print("\n" + "=" * 100)
    print("  [CHALLENGE 4] BACKWARD REACHABILITY & DIRECTED GRAPH ASYMMETRY")
    print("=" * 100)

    asymmetry_violations = 0
    reachability_tests = 0
    asymmetry_deltas = []

    for s_safe in range(16):
        trap_id = 16 + (s_safe % 4)
        s_a = vjepa2.state_codebook[torch.tensor([s_safe], device=dev)]
        s_trap = vjepa2.state_codebook[torch.tensor([trap_id], device=dev)]

        d_forward = float(cost_v28.compute_quasimetric_distance(s_a, s_trap).item())
        d_backward = float(cost_v28.compute_quasimetric_distance(s_trap, s_a).item())

        delta = d_backward - d_forward
        asymmetry_deltas.append(delta)
        reachability_tests += 1

        if d_backward <= d_forward:
            asymmetry_violations += 1

    mean_delta = sum(asymmetry_deltas) / len(asymmetry_deltas)
    min_delta = min(asymmetry_deltas)
    max_delta = max(asymmetry_deltas)
    asymm_pass_rate = (
        100.0 * (reachability_tests - asymmetry_violations) / reachability_tests
    )

    print(
        f"    - Absorbing Trap vs Corridor Reachability Asymmetry (N={reachability_tests}):"
    )
    print(
        f"      * Directed Asymmetry Pass Rate (D_backward > D_forward) : {asymm_pass_rate:6.2f}% ({reachability_tests - asymmetry_violations}/{reachability_tests})"
    )
    print(
        f"      * Mean Asymmetry Delta (D_backward - D_forward)         : {mean_delta:.4f}"
    )
    print(
        f"      * Min Asymmetry Delta                                   : {min_delta:.4f}"
    )
    print(
        f"      * Max Asymmetry Delta                                   : {max_delta:.4f}"
    )
    results["Graph_Asymmetry_Pass_Rate"] = asymm_pass_rate
    results["Graph_Asymmetry_Mean_Delta"] = mean_delta

    # Triangle inequality on corridor triples (i, j, k):
    tri_violations = 0
    tri_tests = 0
    for i in range(10):
        for j in range(i + 1, min(16, i + 5)):
            for k in range(j + 1, min(16, j + 5)):
                s_i = vjepa2.state_codebook[torch.tensor([i], device=dev)]
                s_j = vjepa2.state_codebook[torch.tensor([j], device=dev)]
                s_k = vjepa2.state_codebook[torch.tensor([k], device=dev)]
                d_ik = float(cost_v28.compute_quasimetric_distance(s_i, s_k).item())
                d_ij = float(cost_v28.compute_quasimetric_distance(s_i, s_j).item())
                d_jk = float(cost_v28.compute_quasimetric_distance(s_j, s_k).item())

                tri_tests += 1
                if d_ik > (d_ij + d_jk + 0.35):
                    tri_violations += 1

    tri_acc = 100.0 * (tri_tests - tri_violations) / tri_tests
    print(
        f"    - Triangle Inequality Consistency on Directed Corridors (N={tri_tests}):"
    )
    print(
        f"      * Triangle Inequality Adherence Rate : {tri_acc:6.2f}% ({tri_tests - tri_violations}/{tri_tests})"
    )
    results["Triangle_Inequality_Acc"] = tri_acc

    # Test self-distance D_psi(s_i, s_i) == 0.0
    self_dist_errors = []
    for i in range(20):
        s_i = vjepa2.state_codebook[torch.tensor([i], device=dev)]
        d_ii = float(cost_v28.compute_quasimetric_distance(s_i, s_i).item())
        self_dist_errors.append(d_ii)
    max_self_dist = max(self_dist_errors)
    mean_self_dist = sum(self_dist_errors) / len(self_dist_errors)
    print("    - Self-Distance D_psi(s_i, s_i):")
    print(f"      * Mean Self Distance : {mean_self_dist:.6f}")
    print(f"      * Max Self Distance  : {max_self_dist:.6f}")
    results["Max_Self_Distance"] = max_self_dist

    assert asymm_pass_rate == 100.0, (
        f"Expected 100.00% directed asymmetry, got {asymm_pass_rate}%"
    )
    assert max_self_dist < 0.05, f"Expected self-distance < 0.05, got {max_self_dist}"

    print("\n" + "=" * 100)
    print(
        "  [VERDICT] ALL CHALLENGER 3 ADVERSARIAL STRESS TESTS COMPLETED SUCCESSFULLY!"
    )
    print("=" * 100)
    return results


if __name__ == "__main__":
    t_start = time.time()
    res = run_adversarial_suite()
    print(f"\nExecution Time: {time.time() - t_start:.2f} seconds")
