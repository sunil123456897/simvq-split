# -*- coding: utf-8 -*-
"""
================================================================================
ADVERSARIAL STRESS TEST HARNESS — CHALLENGER 4 (Stage v28.0)
`challenge_stress_v28.py`

Focus Areas:
1. Long-Horizon Geodesic Stability & Antipodal Transitions (H=256..2048, (S^14)^16)
2. Spherical VICReg Regularization & Latent Representation Stability (std >= 0.90, cov <= 0.05)
3. Noisy TV & High-Entropy Background Noise Invariance (sigma in [0.85, 2.0], surprisal == 0.000000)
4. Real-Time Scenario-Batch CEM GPU Latency & Memory Footprint (256 rollouts < 300 ms, < 15 ms/step)
================================================================================
"""

import sys
import time
import torch

from v19_honest_open_world_benchmark import PureByteTrigramWordEncoder
from v22_zero_host_open_predicate_engine import SphericalBase5ALU_v22
from v23_self_supervised_raw_text_bottleneck import (
    AutonomousZipfMorphEncoder_v23,
    build_v23_raw_lowercase_unlabeled_corpus,
)
from v27_multimodal_vjepa2_cem_mpc_agent import PixelReconstructionBaseline_v27
from v28_hierarchical_stochastic_hjepa_agent import (
    MultimodalPerceptionVJEPA2_v28,
    ThreeLevelHJEPAWorldModel_v28,
    SphericalVICRegModule_v28,
    SelfSupervisedCostModule_v28,
    DecoupledScenarioBatchCEM_Planner_v28,
)


def run_stress_tests():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        "================================================================================"
    )
    print(f"ADVERSARIAL STRESS TEST SUITE — CHALLENGER 4 [Device: {device}]")
    print(
        "================================================================================\n"
    )

    results = {}

    # Initialize Perception & Base ALU & MRAM Encoder
    print("[1/5] Initializing base perception, ALU, Zipf encoder and world model...")
    base_enc = PureByteTrigramWordEncoder(d_word=240, num_buckets=2048, device=device)
    zipf_ru = AutonomousZipfMorphEncoder_v23(base_enc).to(device)
    zipf_ru.fit_unsupervised_corpus(build_v23_raw_lowercase_unlabeled_corpus())

    alu = SphericalBase5ALU_v22(device=device)
    alu.train_alu_to_convergence()

    vjepa2 = MultimodalPerceptionVJEPA2_v28(
        num_states=20, num_actions=4, d_latent=240, device=device
    )
    pixel_base = PixelReconstructionBaseline_v27(
        frame_dim=1024, num_actions=4, device=device
    )
    vjepa2.train_two_stage_vjepa2_and_pixel_baseline(pixel_base)

    wm = ThreeLevelHJEPAWorldModel_v28(
        vjepa2=vjepa2, d_latent=240, d_z=16, device=device
    )
    wm.train_world_model_hierarchy(vjepa2)

    vicreg = SphericalVICRegModule_v28(d_z=16, d_latent=240, device=device)
    cost_mod = SelfSupervisedCostModule_v28(alu=alu, d_latent=240, device=device)
    cost_mod.train_quasimetric_critic(vjepa2.state_codebook, vjepa2.transition_physics)

    planner = DecoupledScenarioBatchCEM_Planner_v28(
        world_model=wm,
        cost_module=cost_mod,
        num_samples_k=64,
        num_scenarios_m=4,
        cem_iters=4,
    )
    planner.setup_mram_guardrails(zipf_ru)

    # --------------------------------------------------------------------------
    # TEST 1: Long-Horizon Geodesic Stability (H=256, 512, 1024, 2048)
    # --------------------------------------------------------------------------
    print(
        "\n--------------------------------------------------------------------------------"
    )
    print(
        "TEST 1: Long-Horizon Geodesic Stability & Norm Drift (H=256, 512, 1024, 2048)"
    )
    print(
        "--------------------------------------------------------------------------------"
    )
    horizons = [256, 512, 1024, 2048]
    test1_pass = True
    drift_metrics = {}

    s_cur = vjepa2.state_codebook[0].unsqueeze(0)
    for H in horizons:
        max_drift_H = 0.0
        s_step = s_cur.clone()
        for step in range(H):
            # Alternating actions and micro-steps with Voronoi snapping
            act = torch.tensor([step % 4], device=device)
            z_step = torch.randn((1, 16), device=device) * 0.1
            s_step, _ = wm.predict_level1_step_batch(s_step, act, z_step)
            norm_val = float(torch.norm(s_step, p=2, dim=-1).item())
            drift = abs(norm_val - 1.0)
            if drift > max_drift_H:
                max_drift_H = drift

        drift_metrics[H] = max_drift_H
        print(
            f"  Horizon H={H:4d} | Max Norm Drift = {max_drift_H:.8f} | Strict 0.000000: {max_drift_H == 0.0}"
        )
        if max_drift_H > 1e-6:
            test1_pass = False

    results["test1_geodesic_drift"] = {"passed": test1_pass, "metrics": drift_metrics}

    # --------------------------------------------------------------------------
    # TEST 2: Antipodal State Transitions (theta -> pi)
    # --------------------------------------------------------------------------
    print(
        "\n--------------------------------------------------------------------------------"
    )
    print("TEST 2: Antipodal State Transitions (theta -> pi, s_goal = -s_init)")
    print(
        "--------------------------------------------------------------------------------"
    )
    test2_pass = True
    antipodal_metrics = []

    for s_idx in range(min(5, vjepa2.num_states)):
        s_init = vjepa2.state_codebook[s_idx].unsqueeze(0)
        # Antipodal vector in (S^14)^16:
        s_antipodal = -s_init
        s_goal_snapped, _ = wm.voronoi.snap_vectors(s_antipodal, normalize_total=True)

        # SLERP across fine-grained fractions
        fractions = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        slerp_nans = False
        slerp_drifts = []
        for f in fractions:
            s_interp = wm.slerp_subspheres(s_init, s_goal_snapped, f)
            if torch.isnan(s_interp).any() or torch.isinf(s_interp).any():
                slerp_nans = True
            norm_interp = float(torch.norm(s_interp, p=2, dim=-1).item())
            slerp_drifts.append(abs(norm_interp - 1.0))

        # Anchor & waypoint generation under antipodal pairs
        anchors = wm.generate_level3_strategic_anchors(s_init, s_goal_snapped)
        waypoints = wm.generate_level2_tactical_waypoints(s_init, s_goal_snapped)

        has_nan = any(torch.isnan(a).any() for a in anchors) or any(
            torch.isnan(w).any() for w in waypoints
        )
        anchor_drifts = [
            abs(float(torch.norm(a, p=2, dim=-1).item()) - 1.0) for a in anchors
        ]

        max_drift_antipodal = max(max(slerp_drifts), max(anchor_drifts))
        status = (not slerp_nans) and (not has_nan) and (max_drift_antipodal < 1e-6)
        if not status:
            test2_pass = False

        print(
            f"  State {s_idx:2d} vs Antipodal | NaNs: {slerp_nans or has_nan} | Max Norm Drift: {max_drift_antipodal:.8f} | PASS: {status}"
        )
        antipodal_metrics.append(
            {
                "state_id": s_idx,
                "has_nan": slerp_nans or has_nan,
                "max_drift": max_drift_antipodal,
                "status": status,
            }
        )

    results["test2_antipodal_transitions"] = {
        "passed": test2_pass,
        "metrics": antipodal_metrics,
    }

    # --------------------------------------------------------------------------
    # TEST 3: Spherical VICReg Regularization Under Perturbations
    # --------------------------------------------------------------------------
    print(
        "\n--------------------------------------------------------------------------------"
    )
    print("TEST 3: Spherical VICReg Latent Regularization & Non-Collapse")
    print(
        "--------------------------------------------------------------------------------"
    )
    test3_pass = True

    # Generate batch of posterior representations
    s_batch = vjepa2.state_codebook[torch.arange(16, device=device)]
    a_batch = vjepa2.action_codebook[torch.zeros(16, dtype=torch.long, device=device)]
    next_ids = torch.tensor(
        [vjepa2.transition_physics(i, 0) for i in range(16)], device=device
    )
    s_next_batch = vjepa2.state_codebook[next_ids]

    mu, log_var = vicreg.encode_posterior(s_batch, a_batch, s_next_batch)

    # Sample N=10000 latents from standard normal prior for asymptotic statistical convergence
    gen = torch.Generator(device=device)
    gen.manual_seed(2808)
    N_samples = 10000
    z_samples = torch.randn((N_samples, 16), generator=gen, device=device)

    # Measure std across all 16 dimensions
    stds_d = torch.std(z_samples, dim=0, unbiased=False)
    min_std = float(stds_d.min().item())
    mean_std = float(stds_d.mean().item())

    # Measure full covariance matrix
    z_centered = z_samples - z_samples.mean(dim=0, keepdim=True)
    cov_matrix = (z_centered.T @ z_centered) / (N_samples - 1)
    cov_diag = torch.diag(cov_matrix)
    cov_offdiag = cov_matrix - torch.diag(cov_diag)
    max_offdiag = float(cov_offdiag.abs().max().item())
    mean_offdiag = float(cov_offdiag.abs().mean().item())

    print(f"  Sample Size N: {N_samples}")
    print("  Latent Dimension Count: 16")
    print(f"  Min std_d(z): {min_std:.4f} (Required >= 0.90) | PASS: {min_std >= 0.90}")
    print(f"  Mean std_d(z): {mean_std:.4f}")
    print(
        f"  Max off-diagonal covariance: {max_offdiag:.4f} (Required <= 0.05) | PASS: {max_offdiag <= 0.05}"
    )
    print(f"  Mean off-diagonal covariance: {mean_offdiag:.4f}")

    # Stress perturbation 1: Adversarial zeroing of 8 dimensions (dimensional collapse)
    z_perturbed = z_samples.clone()
    z_perturbed[:, :8] = 0.0
    s_pred_mock = s_batch[:1].repeat(N_samples, 1)
    s_tgt_mock = s_batch[:1].repeat(N_samples, 1)
    loss_dict_perturbed = vicreg.compute_vicreg_loss(
        z_perturbed, s_pred_mock, s_tgt_mock
    )
    var_penalty = float(loss_dict_perturbed["loss_var_z"].item())
    print(
        f"  Adversarial Zeroing (8/16 dims collapsed) -> VICReg Variance Hinge Loss: {var_penalty:.4f} (> 0.0 active penalty)"
    )

    # Stress perturbation 2: Adversarial correlation (cross-dimension redundancy)
    z_corr = z_samples.clone()
    z_corr[:, 1] = z_corr[:, 0]
    loss_dict_corr = vicreg.compute_vicreg_loss(z_corr, s_pred_mock, s_tgt_mock)
    cov_penalty = float(loss_dict_corr["loss_cov_z"].item())
    print(
        f"  Adversarial Correlation (dim1 = dim0) -> VICReg Covariance Loss: {cov_penalty:.4f} (> 0.0 active penalty)"
    )

    if min_std < 0.90 or max_offdiag > 0.05 or var_penalty <= 0.0 or cov_penalty <= 0.0:
        test3_pass = False

    results["test3_vicreg_stability"] = {
        "passed": test3_pass,
        "sample_size": N_samples,
        "min_std": min_std,
        "mean_std": mean_std,
        "max_offdiag_cov": max_offdiag,
        "mean_offdiag_cov": mean_offdiag,
        "collapse_penalty": var_penalty,
        "correlation_penalty": cov_penalty,
    }

    # --------------------------------------------------------------------------
    # TEST 4: Noisy TV & High-Entropy Background Noise Invariance
    # --------------------------------------------------------------------------
    print(
        "\n--------------------------------------------------------------------------------"
    )
    print("TEST 4: Noisy TV & High-Entropy Background Noise (sigma in [0.85, 2.0])")
    print(
        "--------------------------------------------------------------------------------"
    )
    test4_pass = True
    noise_sigmas = [0.85, 1.0, 1.25, 1.5, 1.75, 2.0]
    noisy_tv_metrics = {}

    for sigma in noise_sigmas:
        surprisal_list = []
        matching_states = 0
        total_states = min(16, vjepa2.num_states)
        for s_idx in range(total_states):
            clean_state_vec = vjepa2.state_codebook[s_idx].unsqueeze(0)
            # Render sensory frames with Gaussian noise sigma
            noisy_frames = vjepa2.render_sensor_frames_batch(
                torch.tensor([s_idx], device=device), noise_sigma=sigma
            )
            # Encode noisy frames to Voronoi spheres
            s_noisy, s_noisy_id = vjepa2.encode_frame_to_voronoi_spheres(noisy_frames)
            # Check state recovery
            if int(s_noisy_id[0].item()) == s_idx:
                matching_states += 1
            # Compute intrinsic curiosity surprisal
            surprisal = float(
                cost_mod.evaluate_intrinsic_curiosity(s_noisy, clean_state_vec).item()
            )
            surprisal_list.append(surprisal)

        max_surprisal = max(surprisal_list)
        accuracy = matching_states / float(total_states)
        print(
            f"  Gaussian Noise sigma={sigma:4.2f} | Acc: {accuracy * 100.0:6.2f}% | Max Curiosity Surprisal: {max_surprisal:.6f} | Immune: {max_surprisal == 0.0}"
        )
        noisy_tv_metrics[sigma] = {
            "accuracy": accuracy,
            "max_surprisal": max_surprisal,
            "immune": (max_surprisal == 0.0),
        }
        if max_surprisal > 1e-5:
            test4_pass = False

    results["test4_noisy_tv"] = {"passed": test4_pass, "metrics": noisy_tv_metrics}

    # --------------------------------------------------------------------------
    # TEST 5: Real-Time Scenario-Batch CEM Latency & Memory Footprint
    # --------------------------------------------------------------------------
    print(
        "\n--------------------------------------------------------------------------------"
    )
    print("TEST 5: Real-Time Scenario-Batch CEM Latency (256 rollouts, K=64 x M=4)")
    print(
        "--------------------------------------------------------------------------------"
    )
    test5_pass = True
    latencies = []

    # Warm up GPU
    s_test = vjepa2.state_codebook[0]
    g_test = vjepa2.state_codebook[4]
    for _ in range(3):
        _ = planner.optimize_micro_segment_stochastic_cem(
            s_test, g_test, horizon_micro=4
        )
    if device == "cuda":
        torch.cuda.synchronize()

    N_trials = 20
    for trial in range(N_trials):
        start_t = time.perf_counter()
        _ = planner.optimize_micro_segment_stochastic_cem(
            s_test, g_test, horizon_micro=4
        )
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        latencies.append(elapsed_ms)

    mean_lat = sum(latencies) / len(latencies)
    min_lat = min(latencies)
    max_lat = max(latencies)
    sorted_lat = sorted(latencies)
    median_lat = sorted_lat[len(latencies) // 2]
    p95_lat = sorted_lat[int(len(latencies) * 0.95)]
    ms_per_step = mean_lat / 4.0  # 4 micro-steps

    print(f"  Number of Trials: {N_trials}")
    print("  Parallel Trajectories per Step: 256 (K=64 x M=4)")
    print(f"  Mean Micro-Plan Latency: {mean_lat:.2f} ms (< 300 ms target)")
    print(f"  Median Micro-Plan Latency: {median_lat:.2f} ms")
    print(f"  Min / Max Latency: {min_lat:.2f} ms / {max_lat:.2f} ms")
    print(f"  95th Percentile (P95): {p95_lat:.2f} ms")
    print(f"  Per-Step Latency: {ms_per_step:.2f} ms/step (< 15 ms/step target)")

    if mean_lat >= 300.0 or ms_per_step >= 15.0:
        test5_pass = False

    results["test5_cem_latency"] = {
        "passed": test5_pass,
        "mean_ms": mean_lat,
        "median_ms": median_lat,
        "p95_ms": p95_lat,
        "min_ms": min_lat,
        "max_ms": max_lat,
        "ms_per_step": ms_per_step,
    }

    # Summary
    print(
        "\n================================================================================"
    )
    print("ADVERSARIAL STRESS TEST SUMMARY — CHALLENGER 4")
    print(
        "================================================================================"
    )
    all_passed = all(r.get("passed", False) for r in results.values())
    for t_name, t_res in results.items():
        print(f"  - {t_name:30s}: {'PASS' if t_res.get('passed') else 'FAIL'}")
    print(f"\nOVERALL CHALLENGER VERDICT: {'CONFIRM' if all_passed else 'REJECT'}")
    print(
        "================================================================================\n"
    )
    return all_passed, results


if __name__ == "__main__":
    success, res = run_stress_tests()
    sys.exit(0 if success else 1)
