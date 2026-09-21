from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from flow_grpo.speech_nft_core import (
    PerConditionStatTracker,
    SpeechMixedSamplingState,
    aggregate_reward_advantages,
    build_training_timestep_grid,
    build_all_ode_deterministic_mask,
    build_eval_deterministic_mask,
    compute_advantage_sign_flip_counts,
    compute_gd2po_group_keep_ratios,
    compute_low_std_group_keep_mask,
    compute_multi_reward_pure_nft_terms,
    compute_pure_nft_terms,
    compute_reward_advantage_conflict_stats,
    compute_reward_advantage_pairwise_stats,
    compute_reward_advantage_snr_keep_mask,
    compute_reward_conflict_filter_diagnostics,
    diffusion_nft_decay,
    expand_sampled_time,
    masked_normalize_aggregated_advantages,
    masked_rms_scale_aggregated_advantages,
    normalize_reward_weights,
    reconstruct_x0_from_flow,
    select_timestep_indices,
    summarize_advantage_sign_flip_counts,
)


class DummyFlowModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.1))

    def prepare_condition(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=torch.float32)

    def predict_flow(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        text,
        time: torch.Tensor,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
    ) -> torch.Tensor:
        del text, drop_audio_cond, drop_text
        t = time.reshape(-1, 1, 1).to(x.dtype)
        return self.scale * x + (1.0 - self.scale) * cond + t


class CountingDummyFlowModel(DummyFlowModel):
    def __init__(self):
        super().__init__()
        self.predict_flow_calls = 0

    def predict_flow(self, *args, **kwargs) -> torch.Tensor:
        self.predict_flow_calls += 1
        return super().predict_flow(*args, **kwargs)


def test_diffusion_nft_decay_matches_expected_points():
    assert diffusion_nft_decay(0, 1) == 0.0
    assert abs(diffusion_nft_decay(100, 1) - 0.1) < 1e-8
    assert abs(diffusion_nft_decay(1000, 1) - 0.5) < 1e-8
    assert diffusion_nft_decay(0, 2) == 0.0
    assert abs(diffusion_nft_decay(75, 2) - 0.0) < 1e-8


def test_reward_weight_normalization_supports_registry_and_defaults():
    weights = normalize_reward_weights(
        registry=["dnsmos", "speaker_similarity"],
        weights={"dnsmos": 0.4, "speaker_similarity": 0.6},
    )
    assert weights["dnsmos"] == 0.4
    assert weights["speaker_similarity"] == 0.6
    assert weights["nisqa"] == 0.0
    assert weights["speechbertscore"] == 0.0


def test_reward_advantage_conflict_stats_tracks_sign_and_snr_conflicts():
    stats = compute_reward_advantage_conflict_stats(
        {
            "dnsmos": torch.tensor([1.0, 1.0, 0.0, -1.0]),
            "speechbertscore": torch.tensor([2.0, -1.0, 0.0, -2.0]),
            "speaker_similarity": torch.tensor([3.0, -1.0, 0.0, 1.0]),
        },
        ["dnsmos", "speechbertscore", "speaker_similarity"],
        weights={"dnsmos": 1.0, "speechbertscore": 1.0, "speaker_similarity": 1.0},
    )

    assert stats["reward_adv_conflict_branch_count"] == 3
    assert stats["reward_adv_conflict_sample_count"] == 2
    assert stats["reward_adv_conflict_ratio"] == pytest.approx(0.5)
    assert stats["reward_adv_consensus_ratio"] == pytest.approx(0.25)
    assert stats["reward_adv_neutral_ratio"] == pytest.approx(0.25)
    assert stats["reward_adv_snr_mean"] == pytest.approx((1.0 + 1.0 / 3.0 + 0.0 + 0.5) / 4.0)
    assert stats["reward_adv_snr_retained_ratio_tau_0_5"] == pytest.approx(0.25)
    assert stats["reward_adv_snr_retained_ratio_tau_0_8"] == pytest.approx(0.25)


def test_reward_advantage_pairwise_stats_identifies_agreement_and_lone_dissent():
    stats = compute_reward_advantage_pairwise_stats(
        {
            "dnsmos": torch.tensor([1.0, 1.0, -1.0, 0.0]),
            "speaker_similarity": torch.tensor([1.0, -1.0, -1.0, 0.0]),
            "speechbertscore": torch.tensor([1.0, -1.0, 1.0, 0.0]),
        },
        ["dnsmos", "speaker_similarity", "speechbertscore"],
        weights={"dnsmos": 1.0, "speaker_similarity": 1.0, "speechbertscore": 1.0},
    )

    assert stats["reward_adv_pair_valid_ratio_dnsmos_speaker_similarity"] == pytest.approx(0.75)
    assert stats["reward_adv_pair_agreement_dnsmos_speaker_similarity"] == pytest.approx(2.0 / 3.0)
    assert stats["reward_adv_pair_agreement_dnsmos_speechbertscore"] == pytest.approx(1.0 / 3.0)
    assert stats["reward_adv_pair_agreement_speaker_similarity_speechbertscore"] == pytest.approx(2.0 / 3.0)
    assert stats["reward_adv_lone_dissent_ratio_dnsmos"] == pytest.approx(0.25)
    assert stats["reward_adv_lone_dissent_ratio_speaker_similarity"] == pytest.approx(0.0)
    assert stats["reward_adv_lone_dissent_ratio_speechbertscore"] == pytest.approx(0.25)
    assert stats["reward_adv_abs_contribution_ratio_dnsmos"] == pytest.approx(0.25)
    assert stats["reward_adv_abs_contribution_ratio_speaker_similarity"] == pytest.approx(0.25)
    assert stats["reward_adv_abs_contribution_ratio_speechbertscore"] == pytest.approx(0.25)
    assert -1.0 <= stats["reward_adv_pair_correlation_dnsmos_speaker_similarity"] <= 1.0


def test_reward_conflict_filter_diagnostics_tracks_all_six_three_reward_patterns():
    branch_names = ["dnsmos", "speaker_similarity", "speechbertscore"]
    stats = compute_reward_conflict_filter_diagnostics(
        {
            "dnsmos": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            "speaker_similarity": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0],
            "speechbertscore": [30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
        },
        {
            "dnsmos": [1.0, 2.0, -1.0, -1.0, -2.0, 1.0, 1.0],
            "speaker_similarity": [1.0, -1.0, 2.0, -1.0, 1.0, -2.0, 1.0],
            "speechbertscore": [1.0, -1.0, -1.0, 2.0, 1.0, 1.0, -2.0],
        },
        branch_names,
        [True, False, False, False, False, False, False],
        weights={name: 1.0 / 3.0 for name in branch_names},
    )

    assert stats["reward_filter_diagnostic_kept_count"] == 1
    assert stats["reward_filter_diagnostic_filtered_count"] == 6
    assert stats["reward_filter_kept_raw_mean_dnsmos"] == pytest.approx(10.0)
    assert stats["reward_filter_filtered_raw_mean_dnsmos"] == pytest.approx(13.5)
    assert stats["reward_filter_raw_delta_dnsmos"] == pytest.approx(3.5)
    assert stats["reward_filter_kept_raw_std_dnsmos"] == pytest.approx(0.0)
    assert stats["reward_filter_filtered_raw_std_dnsmos"] == pytest.approx(np.std([11, 12, 13, 14, 15, 16]))
    assert stats["reward_filter_filtered_adv_mean_dnsmos"] == pytest.approx(0.0)
    assert stats["reward_filter_filtered_adv_abs_mean_dnsmos"] == pytest.approx(8.0 / 6.0)
    for polarity in ("pos_only", "neg_only"):
        for branch_name in branch_names:
            prefix = f"reward_filter_pattern_{polarity}_{branch_name}"
            assert stats[f"{prefix}_count"] == 1
            assert stats[f"{prefix}_ratio"] == pytest.approx(1.0 / 6.0)
            assert stats[f"{prefix}_total_count"] == 1
            assert stats[f"{prefix}_kept_count"] == 0
            assert stats[f"{prefix}_filtered_count"] == 1
            assert stats[f"{prefix}_kept_ratio"] == pytest.approx(0.0)
            assert stats[f"{prefix}_filtered_ratio"] == pytest.approx(1.0)
            assert stats[f"{prefix}_snr_mean"] == pytest.approx(0.0)
    assert stats["reward_filter_pattern_pos_only_dnsmos_abs_adv_dnsmos"] == pytest.approx(2.0)
    assert stats["reward_filter_pattern_pos_only_dnsmos_abs_adv_speaker_similarity"] == pytest.approx(1.0)
    assert stats["reward_filter_pattern_covered_ratio"] == pytest.approx(1.0)


def test_reward_conflict_filter_diagnostics_reports_pattern_kept_ratio_before_filtering():
    stats = compute_reward_conflict_filter_diagnostics(
        {
            "dnsmos": [3.0, 3.1, 3.2],
            "speaker_similarity": [0.8, 0.81, 0.82],
            "speechbertscore": [0.7, 0.71, 0.72],
        },
        {
            "dnsmos": [2.0, 3.0, 1.0],
            "speaker_similarity": [-1.0, -0.1, 1.0],
            "speechbertscore": [-1.0, -0.1, 1.0],
        },
        ["dnsmos", "speaker_similarity", "speechbertscore"],
        [False, True, True],
    )

    prefix = "reward_filter_pattern_pos_only_dnsmos"
    assert stats[f"{prefix}_total_count"] == 2
    assert stats[f"{prefix}_kept_count"] == 1
    assert stats[f"{prefix}_filtered_count"] == 1
    assert stats[f"{prefix}_kept_ratio"] == pytest.approx(0.5)
    assert stats[f"{prefix}_filtered_ratio"] == pytest.approx(0.5)
    assert stats[f"{prefix}_kept_snr_mean"] == pytest.approx(2.8 / 3.2)
    assert stats[f"{prefix}_filtered_snr_mean"] == pytest.approx(0.0)


def test_reward_advantage_snr_keep_mask_filters_candidates_below_tau():
    keep_mask, stats = compute_reward_advantage_snr_keep_mask(
        {
            "dnsmos": torch.tensor([1.0, 1.0, 1.0, -1.0]),
            "speechbertscore": torch.tensor([2.0, -1.0, -0.6, -2.0]),
            "speaker_similarity": torch.tensor([3.0, -1.0, -0.6, 1.0]),
        },
        ["dnsmos", "speechbertscore", "speaker_similarity"],
        tau=0.2,
    )

    assert keep_mask.tolist() == [True, True, False, True]
    assert stats["reward_conflict_filter_tau"] == pytest.approx(0.2)
    assert stats["reward_conflict_filter_kept_sample_count"] == 3
    assert stats["reward_conflict_filter_kept_sample_ratio"] == pytest.approx(0.75)
    assert stats["reward_conflict_filter_filtered_sample_count"] == 1


def test_reward_advantage_snr_keep_mask_keeps_neutral_and_zero_tau_boundary():
    keep_mask, _ = compute_reward_advantage_snr_keep_mask(
        {
            "a": torch.tensor([0.0, 1.0, 1.0]),
            "b": torch.tensor([0.0, 2.0, -1.0]),
        },
        ["a", "b"],
        tau=0.0,
    )

    assert keep_mask.tolist() == [True, True, True]


def test_reward_advantage_snr_keep_mask_keeps_all_neutral_samples_at_high_tau():
    keep_mask, stats = compute_reward_advantage_snr_keep_mask(
        {
            "a": torch.zeros(2),
            "b": torch.zeros(2),
        },
        ["a", "b"],
        tau=0.8,
    )

    assert keep_mask.tolist() == [True, True]
    assert stats["reward_conflict_filter_conflict_sample_count"] == 0
    assert stats["reward_conflict_filter_filtered_sample_count"] == 0


def test_gd2po_aggregate_reward_advantages_uses_normalized_uniform_weights():
    aggregated = aggregate_reward_advantages(
        {
            "dnsmos": [1.0, -1.0, 0.0],
            "speaker_similarity": [0.5, 0.5, -1.0],
            "speechbertscore": [-0.5, 1.0, 1.0],
        },
        ["dnsmos", "speaker_similarity", "speechbertscore"],
        weights={"dnsmos": 1.0, "speaker_similarity": 1.0, "speechbertscore": 1.0},
    )

    assert np.allclose(aggregated, np.asarray([1.0 / 3.0, 1.0 / 6.0, 0.0], dtype=np.float32))


def test_gd2po_group_keep_ratios_match_per_query_retained_fraction():
    ratios, stats = compute_gd2po_group_keep_ratios(
        ["utt-a", "utt-a", "utt-a", "utt-a", "utt-b", "utt-b"],
        [True, False, True, False, True, True],
    )

    assert np.allclose(ratios, np.asarray([0.5, 0.5, 0.5, 0.5, 1.0, 1.0], dtype=np.float32))
    assert stats["gd2po_group_keep_ratio_enabled"] is True
    assert stats["gd2po_group_keep_ratio_mean"] == pytest.approx(2.0 / 3.0)
    assert stats["gd2po_group_keep_ratio_min"] == pytest.approx(0.5)
    assert stats["gd2po_group_keep_ratio_max"] == pytest.approx(1.0)


def test_gd2po_group_keep_ratio_is_applied_before_masked_normalization():
    aggregated = np.asarray([2.0, -2.0, 1.0, 3.0], dtype=np.float32)
    keep_mask = np.asarray([True, False, True, True])
    ratios, _ = compute_gd2po_group_keep_ratios(
        ["utt-a", "utt-a", "utt-b", "utt-b"],
        keep_mask,
    )

    weighted = aggregated * ratios
    normalized = masked_normalize_aggregated_advantages(weighted, keep_mask, eps=1e-8)
    expected_kept = np.asarray([1.0, 1.0, 3.0], dtype=np.float32)
    expected_kept = (expected_kept - expected_kept.mean()) / expected_kept.std()

    assert np.allclose(normalized[keep_mask], expected_kept)
    assert normalized[1] == pytest.approx(0.0)


def test_masked_normalize_aggregated_advantages_uses_only_retained_samples():
    normalized = masked_normalize_aggregated_advantages(
        [1.0, 100.0, 3.0, -100.0],
        [True, False, True, False],
        eps=1e-8,
    )

    assert np.allclose(normalized, np.asarray([-1.0, 0.0, 1.0, 0.0], dtype=np.float32))
    assert float(np.mean(normalized[[0, 2]])) == pytest.approx(0.0)
    assert float(np.std(normalized[[0, 2]])) == pytest.approx(1.0)


def test_masked_normalize_aggregated_advantages_handles_empty_and_flat_retained_sets():
    assert np.allclose(
        masked_normalize_aggregated_advantages([1.0, 2.0], [False, False]),
        np.zeros(2, dtype=np.float32),
    )
    assert np.allclose(
        masked_normalize_aggregated_advantages([2.0, 2.0, 9.0], [True, True, False]),
        np.zeros(3, dtype=np.float32),
    )


def test_masked_normalize_aggregated_advantages_accepts_global_moments():
    normalized = masked_normalize_aggregated_advantages(
        [1.0, 3.0],
        [True, True],
        eps=1e-8,
        moments=[4.0, 16.0, 84.0],  # Global retained values: [1, 3, 5, 7].
    )

    assert np.allclose(
        normalized,
        np.asarray([-3.0 / np.sqrt(5.0), -1.0 / np.sqrt(5.0)], dtype=np.float32),
    )


def test_masked_rms_scale_preserves_signs_and_uses_only_retained_samples():
    scaled = masked_rms_scale_aggregated_advantages(
        [0.2, 100.0, 0.4, -100.0, -0.8],
        [True, False, True, False, True],
        eps=1e-8,
    )
    expected_rms = np.sqrt(np.mean(np.square([0.2, 0.4, -0.8])))

    assert np.allclose(scaled[[0, 2, 4]], np.asarray([0.2, 0.4, -0.8]) / expected_rms)
    assert scaled[1] == pytest.approx(0.0)
    assert scaled[3] == pytest.approx(0.0)
    assert np.array_equal(np.sign(scaled[[0, 2, 4]]), np.asarray([1.0, 1.0, -1.0]))


def test_masked_rms_scale_accepts_global_moments():
    scaled = masked_rms_scale_aggregated_advantages(
        [1.0, 3.0],
        [True, True],
        eps=1e-8,
        moments=[4.0, 16.0, 84.0],  # Global retained values: [1, 3, 5, 7].
    )

    assert np.allclose(scaled, np.asarray([1.0, 3.0]) / np.sqrt(21.0))


def test_speech_mixed_sampling_state_progressive_overlap_and_roll_back():
    state = SpeechMixedSamplingState(
        steps=6,
        group_size=2,
        overlap=True,
        overlap_step=1,
        roll_back=True,
    )

    assert state.get_current_sde_indices() == [0, 1]
    assert state.get_current_deterministic_mask() == [False, False, True, True, True, True]

    state.update_iteration()
    assert state.get_current_sde_indices() == [1, 2]
    assert state.get_current_deterministic_mask() == [True, False, False, True, True, True]

    for _ in range(3):
        state.update_iteration()
    assert state.get_current_sde_indices() == [4, 5]

    state.update_iteration()
    assert state.get_current_sde_indices() == [0, 1]


def test_speech_mixed_sampling_state_respects_update_interval():
    state = SpeechMixedSamplingState(
        steps=8,
        group_size=3,
        overlap=True,
        overlap_step=1,
        update_interval=6,
        roll_back=True,
    )

    for _ in range(5):
        assert state.get_current_sde_indices() == [0, 1, 2]
        state.update_iteration()

    assert state.get_current_sde_indices() == [0, 1, 2]
    state.update_iteration()
    assert state.get_current_sde_indices() == [1, 2, 3]


def test_build_all_ode_deterministic_mask_is_validation_default():
    assert build_all_ode_deterministic_mask(4) == [True, True, True, True]


def test_build_eval_deterministic_mask_can_reuse_target_rollout_path():
    assert build_eval_deterministic_mask(
        17,
        use_rollout_schedule=True,
        use_rollout_deterministic_mask=True,
        rollout_deterministic_mask=[False] * 16 + [True],
    ) == [False] * 16 + [True]


def test_build_eval_deterministic_mask_keeps_default_validation_all_ode():
    assert build_eval_deterministic_mask(
        32,
        use_rollout_schedule=False,
        use_rollout_deterministic_mask=False,
        rollout_deterministic_mask=[False] * 16 + [True],
    ) == [True] * 32


def test_build_eval_deterministic_mask_rejects_schedule_mismatch():
    with pytest.raises(ValueError, match="requires.*use_rollout_schedule=True"):
        build_eval_deterministic_mask(
            17,
            use_rollout_schedule=False,
            use_rollout_deterministic_mask=True,
            rollout_deterministic_mask=[False] * 16 + [True],
        )


def test_select_timestep_indices_respects_window_and_fixed_budget():
    timesteps = torch.linspace(0.0, 1.0, steps=33)
    indices = select_timestep_indices(
        timesteps,
        timestep_window=[0.0, 0.5],
        timesteps_per_batch=8,
        rng=random.Random(0),
    )

    assert len(indices) == 8
    assert all(0.0 <= float(timesteps[index]) < 0.5 for index in indices)


def test_select_timestep_indices_clamps_budget_to_window_size():
    timesteps = torch.linspace(0.0, 1.0, steps=33)
    indices = select_timestep_indices(
        timesteps,
        timestep_window=[0.75, 1.0],
        timesteps_per_batch=64,
        rng=random.Random(0),
    )

    assert len(indices) == 9
    assert all(0.75 <= float(timesteps[index]) <= 1.0 for index in indices)


def test_training_timestep_grid_defaults_to_rollout_grid():
    rollout_grid = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

    training_grid = build_training_timestep_grid(rollout_grid, timestep_grid_steps=0)

    assert torch.allclose(training_grid, rollout_grid)


def test_training_timestep_grid_independent_early32_from_train64():
    rollout_grid = torch.linspace(0.0, 1.0, steps=32)
    training_grid = build_training_timestep_grid(rollout_grid, timestep_grid_steps=64)
    indices = select_timestep_indices(
        training_grid,
        timestep_window=[0.0, 0.5],
        timesteps_per_batch=32,
        rng=random.Random(0),
    )

    assert int(training_grid.numel()) == 64
    assert len(indices) == 32
    assert len(set(indices)) == 32
    assert all(0.0 <= float(training_grid[index]) < 0.5 for index in indices)


def test_training_timestep_grid_independent_early_quarter_selects_32_from_128():
    rollout_grid = torch.linspace(0.0, 1.0, steps=32)
    training_grid = build_training_timestep_grid(rollout_grid, timestep_grid_steps=128)
    indices = select_timestep_indices(
        training_grid,
        timestep_window=[0.0, 0.25],
        timesteps_per_batch=32,
        rng=random.Random(0),
    )

    assert int(training_grid.numel()) == 128
    assert len(indices) == 32
    assert len(set(indices)) == 32
    assert all(0.0 <= float(training_grid[index]) < 0.25 for index in indices)


def test_per_condition_stat_tracker_group_centered_advantages():
    tracker = PerConditionStatTracker(global_std=False)
    conditions = ["a", "a", "a", "b", "b", "b"]
    rewards = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

    advantages = tracker.update(conditions, rewards)
    a_adv = advantages[np.array(conditions) == "a"]
    b_adv = advantages[np.array(conditions) == "b"]

    assert abs(float(a_adv.mean())) < 1e-6
    assert abs(float(b_adv.mean())) < 1e-6
    stats = tracker.get_last_stats()
    assert stats["num_groups"] == 2
    assert stats["reward_std_mean"] > 0


def test_low_std_group_filter_masks_high_mean_low_std_group():
    conditions = ["a", "a", "a", "b", "b", "b"]
    rewards = [0.96, 0.961, 0.959, 0.5, 0.7, 0.9]

    keep_mask, stats = compute_low_std_group_keep_mask(
        conditions,
        rewards,
        mean_threshold=0.95,
        std_threshold=0.01,
    )

    assert keep_mask.tolist() == [False, False, False, True, True, True]
    assert stats["low_std_filter_group_count"] == 1
    assert stats["low_std_filter_sample_count"] == 3


def test_compute_pure_nft_terms_is_finite():
    torch.manual_seed(0)
    batch = 4
    n = 10
    d = 6

    current = DummyFlowModel()
    old = DummyFlowModel()
    ref = DummyFlowModel()

    noisy_mel = torch.randn(batch, n, d)
    generated_mel = torch.randn(batch, n, d)
    advantages = torch.tensor([0.5, -1.0, 0.2, 0.8], dtype=torch.float32)
    time = torch.rand(batch)

    terms = compute_pure_nft_terms(
        current_model=current,
        old_model=old,
        ref_model=ref,
        noisy_mel=noisy_mel,
        generated_mel=generated_mel,
        advantages=advantages,
        beta_mix=0.5,
        adv_clip_max=5.0,
        kl_coef=0.01,
        time=time,
        cond_type="wotext",
    )

    assert torch.isfinite(terms["loss_total"]).item()
    assert torch.isfinite(terms["loss_rl"]).item()
    assert torch.isfinite(terms["loss_kl"]).item()
    assert terms["per_sample_policy"].shape[0] == batch


def test_compute_pure_nft_terms_reuses_noise_for_matching_kl():
    torch.manual_seed(0)
    batch = 4
    n = 10
    d = 6

    current = DummyFlowModel()
    old = DummyFlowModel()
    ref = DummyFlowModel()

    noisy_mel = torch.randn(batch, n, d)
    generated_mel = torch.randn(batch, n, d)
    time = torch.rand(batch)
    noise = torch.randn_like(generated_mel)

    terms_a = compute_pure_nft_terms(
        current_model=current,
        old_model=old,
        ref_model=ref,
        noisy_mel=noisy_mel,
        generated_mel=generated_mel,
        advantages=torch.tensor([0.5, -1.0, 0.2, 0.8], dtype=torch.float32),
        beta_mix=0.5,
        adv_clip_max=5.0,
        kl_coef=0.01,
        time=time,
        noise=noise,
        cond_type="wotext",
    )
    terms_b = compute_pure_nft_terms(
        current_model=current,
        old_model=old,
        ref_model=ref,
        noisy_mel=noisy_mel,
        generated_mel=generated_mel,
        advantages=torch.tensor([-0.2, 0.7, -1.5, 1.2], dtype=torch.float32),
        beta_mix=0.5,
        adv_clip_max=5.0,
        kl_coef=0.01,
        time=time,
        noise=noise,
        cond_type="wotext",
    )

    assert torch.allclose(terms_a["noise"], terms_b["noise"])
    assert torch.allclose(terms_a["loss_kl"], terms_b["loss_kl"])


def test_compute_pure_nft_terms_reparameterizes_halfway_target():
    batch = 2
    noise = torch.tensor([[[0.0]], [[2.0]]])
    midpoint = torch.tensor([[[4.0]], [[6.0]]])
    time = torch.tensor([0.125, 0.375])

    terms = compute_pure_nft_terms(
        current_model=DummyFlowModel(),
        old_model=DummyFlowModel(),
        ref_model=DummyFlowModel(),
        noisy_mel=torch.zeros_like(midpoint),
        generated_mel=midpoint,
        advantages=torch.zeros(batch),
        beta_mix=1.0,
        adv_clip_max=5.0,
        kl_coef=0.01,
        time=time,
        noise=noise,
        cond_type="wotext",
        target_time=0.5,
    )

    progress = (time / 0.5).reshape(batch, 1, 1)
    expected_x_t = (1.0 - progress) * noise + progress * midpoint
    assert torch.allclose(terms["x_t"], expected_x_t)
    assert terms["target_time"].item() == pytest.approx(0.5)


def test_reconstruct_x0_from_flow_uses_halfway_endpoint_distance():
    x_t = torch.tensor([[[1.0]]])
    flow = torch.tensor([[[4.0]]])
    reconstructed = reconstruct_x0_from_flow(
        x_t,
        flow,
        time=torch.tensor([0.25]),
        target_time=0.5,
    )

    assert torch.allclose(reconstructed, torch.tensor([[[2.0]]]))


@pytest.mark.parametrize("target_time", [0.5, 1.0])
@pytest.mark.parametrize("target_value", [4.0, 1e-7])
def test_nft_endpoint_time_weight_scales_both_branches_and_preserves_kl(target_time, target_value):
    current, old, ref = DummyFlowModel(), DummyFlowModel(), DummyFlowModel()
    with torch.no_grad():
        current.scale.fill_(0.2)
        ref.scale.fill_(0.3)
    # Include t=0 (also exercises the adaptive denominator floor for tiny targets)
    # and the largest time in the existing 32-point midpoint training grid.
    time = torch.tensor([0.0, 0.25, 15.0 / 31.0])
    target = torch.full((3, 1, 1), target_value)
    terms = compute_pure_nft_terms(
        current_model=current, old_model=old, ref_model=ref,
        noisy_mel=torch.zeros_like(target), generated_mel=target,
        advantages=torch.tensor([-5.0, 0.0, 5.0]),
        beta_mix=1.0, adv_clip_max=5.0, kl_coef=0.01,
        time=time, noise=torch.zeros_like(target), target_time=target_time,
    )
    x = (time / target_time).reshape(-1, 1, 1) * target
    flow_cur = current.scale * x + time.reshape(-1, 1, 1)
    flow_old = old.scale.detach() * x + time.reshape(-1, 1, 1)
    flow_ref = ref.scale.detach() * x + time.reshape(-1, 1, 1)
    factor = (1.0 - time) / (0.5 - time) if target_time == 0.5 else torch.ones_like(time)
    expected_branches = []
    for name, flow in [("positive", flow_cur), ("negative", 2.0 * flow_old - flow_cur)]:
        error = x + (target_time - time).reshape(-1, 1, 1) * flow - target
        denominator = error.detach().abs().mean(dim=(1, 2)).clamp(min=1e-5)
        expected = error.square().mean(dim=(1, 2)) / denominator * factor
        torch.testing.assert_close(terms[f"per_sample_{name}_loss"], expected)
        expected_branches.append(expected)
    p = torch.tensor([0.0, 0.5, 1.0])
    expected_kl = (flow_cur - flow_ref).square().mean()
    expected_total = 5.0 * (p * expected_branches[0] + (1.0 - p) * expected_branches[1]).mean() + 0.01 * expected_kl
    torch.testing.assert_close(terms["loss_kl"], expected_kl)
    torch.testing.assert_close(terms["loss_total"], expected_total)
    actual_grad = torch.autograd.grad(terms["loss_total"], current.scale)[0]
    expected_grad = torch.autograd.grad(expected_total, current.scale)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_midpoint_time_reweighting_rejects_endpoint():
    target = torch.ones(1, 1, 1)
    with pytest.raises(ValueError, match="training time < 0.5"):
        compute_pure_nft_terms(
            current_model=DummyFlowModel(), old_model=DummyFlowModel(), ref_model=DummyFlowModel(),
            noisy_mel=target, generated_mel=target, advantages=torch.zeros(1),
            beta_mix=1.0, adv_clip_max=5.0, kl_coef=0.01,
            time=torch.tensor([0.5]), target_time=0.5,
        )


def test_compute_pure_nft_terms_rejects_time_after_target():
    target = torch.zeros(2, 1, 1)
    with pytest.raises(ValueError, match="NFT training time"):
        compute_pure_nft_terms(
            current_model=DummyFlowModel(),
            old_model=DummyFlowModel(),
            ref_model=DummyFlowModel(),
            noisy_mel=torch.zeros_like(target),
            generated_mel=target,
            advantages=torch.zeros(2),
            beta_mix=1.0,
            adv_clip_max=5.0,
            kl_coef=0.01,
            time=torch.tensor([0.25, 0.75]),
            cond_type="wotext",
            target_time=0.5,
        )


def test_multi_reward_pure_nft_terms_shares_forward_and_matches_separate_losses():
    torch.manual_seed(0)
    batch = 4
    noisy_mel = torch.randn(batch, 10, 6)
    generated_mel = torch.randn(batch, 10, 6)
    time = torch.rand(batch)
    noise = torch.randn_like(generated_mel)
    advantages_by_branch = {
        "dnsmos": torch.tensor([0.5, -1.0, 0.2, 0.8]),
        "speaker_similarity": torch.tensor([-0.2, 0.7, -1.5, 1.2]),
        "speechbertscore": torch.tensor([1.1, -0.4, 0.3, -0.9]),
    }
    branch_weights = {
        "dnsmos": 0.2,
        "speaker_similarity": 0.3,
        "speechbertscore": 0.5,
    }

    shared_current = CountingDummyFlowModel()
    shared_old = CountingDummyFlowModel()
    shared_ref = CountingDummyFlowModel()
    shared_terms = compute_multi_reward_pure_nft_terms(
        current_model=shared_current,
        old_model=shared_old,
        ref_model=shared_ref,
        noisy_mel=noisy_mel,
        generated_mel=generated_mel,
        advantages_by_branch=advantages_by_branch,
        branch_weights=branch_weights,
        beta_mix=0.5,
        adv_clip_max=5.0,
        kl_coef=0.01,
        time=time,
        noise=noise,
        cond_type="wotext",
    )
    shared_terms["loss_total"].backward()

    separate_current = DummyFlowModel()
    separate_old = DummyFlowModel()
    separate_ref = DummyFlowModel()
    separate_loss_total = torch.zeros(())
    separate_loss_rl = torch.zeros(())
    separate_loss_kl = torch.zeros(())
    for branch_name, advantages in advantages_by_branch.items():
        terms = compute_pure_nft_terms(
            current_model=separate_current,
            old_model=separate_old,
            ref_model=separate_ref,
            noisy_mel=noisy_mel,
            generated_mel=generated_mel,
            advantages=advantages,
            beta_mix=0.5,
            adv_clip_max=5.0,
            kl_coef=0.01,
            time=time,
            noise=noise,
            cond_type="wotext",
        )
        weight = branch_weights[branch_name]
        separate_loss_total = separate_loss_total + weight * terms["loss_total"]
        separate_loss_rl = separate_loss_rl + weight * terms["loss_rl"]
        separate_loss_kl = separate_loss_kl + weight * terms["loss_kl"]
    separate_loss_total.backward()

    assert shared_current.predict_flow_calls == 1
    assert shared_old.predict_flow_calls == 1
    assert shared_ref.predict_flow_calls == 1
    assert torch.allclose(shared_terms["loss_total"], separate_loss_total)
    assert torch.allclose(shared_terms["loss_rl"], separate_loss_rl)
    assert torch.allclose(shared_terms["loss_kl"], separate_loss_kl)
    assert torch.allclose(shared_current.scale.grad, separate_current.scale.grad)


def test_compute_pure_nft_terms_respects_train_mask():
    torch.manual_seed(0)
    batch = 4
    n = 10
    d = 6

    current = DummyFlowModel()
    old = DummyFlowModel()
    ref = DummyFlowModel()

    noisy_mel = torch.randn(batch, n, d)
    generated_mel = torch.randn(batch, n, d)
    advantages = torch.tensor([0.5, -1.0, 0.2, 0.8], dtype=torch.float32)
    time = torch.rand(batch)
    noise = torch.randn_like(generated_mel)

    terms = compute_pure_nft_terms(
        current_model=current,
        old_model=old,
        ref_model=ref,
        noisy_mel=noisy_mel,
        generated_mel=generated_mel,
        advantages=advantages,
        train_mask=torch.zeros(batch),
        beta_mix=0.5,
        adv_clip_max=5.0,
        kl_coef=0.01,
        time=time,
        noise=noise,
        cond_type="wotext",
    )

    assert torch.allclose(terms["loss_total"], torch.zeros_like(terms["loss_total"]))
    assert torch.allclose(terms["loss_rl"], torch.zeros_like(terms["loss_rl"]))
    assert torch.allclose(terms["loss_kl"], torch.zeros_like(terms["loss_kl"]))
    assert terms["kept_sample_count"].item() == 0


def test_compute_pure_nft_terms_all_mask_normalization_preserves_gd2po_retained_ratio():
    torch.manual_seed(0)
    batch = 4
    noisy_mel = torch.randn(batch, 10, 6)
    generated_mel = torch.randn(batch, 10, 6)
    advantages = torch.tensor([0.5, -1.0, 0.2, 0.8], dtype=torch.float32)
    train_mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
    time = torch.rand(batch)
    noise = torch.randn_like(generated_mel)
    common_kwargs = {
        "current_model": DummyFlowModel(),
        "old_model": DummyFlowModel(),
        "ref_model": DummyFlowModel(),
        "noisy_mel": noisy_mel,
        "generated_mel": generated_mel,
        "advantages": advantages,
        "train_mask": train_mask,
        "beta_mix": 0.5,
        "adv_clip_max": 5.0,
        "kl_coef": 0.01,
        "time": time,
        "noise": noise,
        "cond_type": "wotext",
    }

    kept_terms = compute_pure_nft_terms(**common_kwargs, train_mask_normalization="kept")
    all_terms = compute_pure_nft_terms(**common_kwargs, train_mask_normalization="all")
    retained_ratio = train_mask.mean()

    assert all_terms["train_mask_normalization_denom"].item() == batch
    assert torch.allclose(all_terms["loss_rl"], kept_terms["loss_rl"] * retained_ratio)
    assert torch.allclose(all_terms["loss_kl"], kept_terms["loss_kl"] * retained_ratio)
    assert torch.allclose(all_terms["loss_total"], kept_terms["loss_total"] * retained_ratio)


def test_advantage_sign_flip_stats_only_count_retained_direction_reversals():
    counts = compute_advantage_sign_flip_counts(
        pre_normalization_advantages=[0.5, 0.2, -0.4, -0.1, 0.0, 1.0],
        post_normalization_advantages=[-0.5, 0.1, 0.3, -0.2, 1.0, -1.0],
        keep_mask=[True, True, True, True, True, False],
    )
    stats = summarize_advantage_sign_flip_counts(counts)

    assert counts.tolist() == [5.0, 2.0, 2.0, 1.0, 1.0]
    assert stats["gd2po_post_norm_sign_flip_ratio"] == pytest.approx(2.0 / 5.0)
    assert stats["gd2po_post_norm_positive_to_negative_ratio"] == pytest.approx(1.0 / 2.0)
    assert stats["gd2po_post_norm_negative_to_positive_ratio"] == pytest.approx(1.0 / 2.0)


def test_advantage_sign_flip_stats_are_zero_without_retained_samples():
    counts = compute_advantage_sign_flip_counts([1.0], [-1.0], [False])
    stats = summarize_advantage_sign_flip_counts(counts)

    assert all(value == 0.0 for value in stats.values())


def test_expand_sampled_time_accepts_scalar_independent_time():
    expanded = expand_sampled_time(0.375, batch_size=3, device=torch.device("cpu"))

    assert expanded.shape == (3,)
    assert torch.allclose(expanded, torch.full((3,), 0.375))


def test_quarter_window_has_sixteen_candidates_on_sixty_four_point_grid():
    grid = build_training_timestep_grid([0.0, 1.0], timestep_grid_steps=64)
    selected = select_timestep_indices(
        grid,
        timestep_window=[0.0, 0.25],
        timestep_fraction=1.0,
        timesteps_per_batch=16,
        rng=random.Random(0),
    )

    assert len(selected) == 16
    assert all(0.0 <= float(grid[index]) < 0.25 for index in selected)
