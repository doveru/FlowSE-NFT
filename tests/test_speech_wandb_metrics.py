from __future__ import annotations

import sys
import types
from types import SimpleNamespace

_BACKEND_STUB = types.ModuleType("flow_grpo.speech_backend_adapter")
_BACKEND_STUB.SpeechBackendAdapter = object
_BACKEND_STUB.set_seed = lambda seed: None
sys.modules.setdefault("flow_grpo.speech_backend_adapter", _BACKEND_STUB)

_EMA_STUB = types.ModuleType("flow_grpo.ema")
_EMA_STUB.EMAModuleWrapper = object
sys.modules.setdefault("flow_grpo.ema", _EMA_STUB)

from flow_grpo.speech_orchestrator import compute_speech_best_metric, flatten_speech_wandb_metrics


def test_flatten_speech_wandb_metrics_keeps_raw_breakdowns_and_groups_gd2po_metrics():
    payload = {
        "global_step": 7,
        "train_reward_breakdown_raw": {
            "dnsmos_ovrl": 3.5,
            "speaker_similarity": 0.91,
            "speechbertscore": 0.87,
            "reward_component_speaker_similarity_raw": 0.91,
        },
        "train_reward_breakdown_norm": {
            "reward_component_dnsmos_norm": 2.2,
            "reward_component_speaker_similarity_norm": 5.1,
            "reward_component_speechbertscore_norm": 4.7,
        },
        "val_num_samples": 2,
        "val_reward_breakdown_raw": {
            "speaker_similarity": 0.9,
            "speechbertscore": 0.85,
        },
        "val_reward_breakdown_norm": {
            "reward_component_speaker_similarity_norm": 4.8,
            "reward_component_speechbertscore_norm": 4.4,
        },
        "train_loss_kl_weighted": 0.00003,
        "reward_adv_conflict_ratio": 0.42,
        "reward_adv_consensus_ratio": 0.55,
        "reward_adv_snr_mean": 0.73,
        "reward_adv_snr_retained_ratio_tau_0_2": 0.92,
        "reward_adv_snr_retained_ratio_tau_0_5": 0.81,
        "reward_adv_snr_retained_ratio_tau_0_8": 0.37,
        "reward_conflict_filter_tau": 0.2,
        "reward_conflict_filter_kept_sample_ratio": 0.92,
        "reward_conflict_filter_filtered_sample_ratio": 0.08,
        "train_mask_kept_sample_ratio": 0.92,
        "gd2po_enabled": True,
        "gd2po_advantage_mean": 0.03,
        "gd2po_advantage_std": 0.81,
        "gd2po_post_normalization_enabled": True,
        "gd2po_advantage_pre_norm_mean": 0.03,
        "gd2po_advantage_pre_norm_std": 0.58,
        "gd2po_group_keep_ratio_enabled": True,
        "gd2po_group_keep_ratio_mean": 0.82,
        "gd2po_group_keep_ratio_min": 0.5,
        "gd2po_group_keep_ratio_max": 1.0,
        "gd2po_group_keep_ratio_std": 0.11,
        "gd2po_advantage_post_norm_kept_mean": 0.0,
        "gd2po_advantage_post_norm_kept_std": 1.0,
        "gd2po_post_norm_sign_flip_ratio": 0.24,
        "gd2po_post_norm_positive_to_negative_ratio": 0.31,
        "gd2po_post_norm_negative_to_positive_ratio": 0.17,
        "reward_adv_pair_agreement_dnsmos_speaker_similarity": 0.43,
        "reward_adv_pair_correlation_dnsmos_speaker_similarity": -0.18,
        "reward_adv_lone_dissent_ratio_dnsmos": 0.36,
        "reward_adv_abs_contribution_ratio_dnsmos": 0.34,
        "reward_filter_filtered_adv_abs_mean_dnsmos": 1.12,
        "reward_filter_pattern_pos_only_dnsmos_ratio": 0.18,
        "reward_filter_filtered_raw_std_dnsmos": 0.24,
        "reward_filter_pattern_pos_only_dnsmos_kept_ratio": 0.15,
    }

    metrics = flatten_speech_wandb_metrics(payload)

    assert metrics["train_reward_breakdown_raw/speaker_similarity"] == 0.91
    assert metrics["train_reward_breakdown_raw/speechbertscore"] == 0.87
    assert metrics["train_reward_breakdown_raw/reward_component_speaker_similarity_raw"] == 0.91
    assert metrics["val_reward_breakdown_raw/speaker_similarity"] == 0.9
    assert metrics["val_reward_breakdown_raw/speechbertscore"] == 0.85
    assert metrics["train_loss_kl_weighted"] == 0.00003
    assert metrics["GD2PO/reward_adv_conflict_ratio"] == 0.42
    assert metrics["GD2PO/reward_adv_snr_mean"] == 0.73
    assert metrics["GD2PO/reward_conflict_filter_kept_sample_ratio"] == 0.92
    assert metrics["GD2PO/gd2po_advantage_pre_norm_std"] == 0.58
    assert metrics["GD2PO/gd2po_group_keep_ratio_mean"] == 0.82
    assert metrics["GD2PO/gd2po_group_keep_ratio_std"] == 0.11
    assert "GD2PO/gd2po_group_keep_ratio_enabled" not in metrics
    assert "GD2PO/gd2po_group_keep_ratio_min" not in metrics
    assert "GD2PO/gd2po_group_keep_ratio_max" not in metrics
    assert metrics["GD2PO/gd2po_post_norm_sign_flip_ratio"] == 0.24
    assert metrics["GD2PO/gd2po_post_norm_positive_to_negative_ratio"] == 0.31
    assert metrics["GD2PO/gd2po_post_norm_negative_to_positive_ratio"] == 0.17
    assert metrics["GD2PO/reward_adv_pair_agreement_dnsmos_speaker_similarity"] == 0.43
    assert "GD2PO/reward_adv_consensus_ratio" not in metrics
    assert "GD2PO/reward_adv_pair_correlation_dnsmos_speaker_similarity" not in metrics
    assert "GD2PO/reward_adv_lone_dissent_ratio_dnsmos" not in metrics
    assert "GD2PO/reward_adv_abs_contribution_ratio_dnsmos" not in metrics
    assert "GD2PO/reward_filter_filtered_adv_abs_mean_dnsmos" not in metrics
    assert "GD2PO/reward_filter_pattern_pos_only_dnsmos_ratio" not in metrics
    assert not any(key.startswith("train/") for key in metrics)
    assert not any(key.startswith("val/") for key in metrics)
    assert not any(key.startswith("train_reward_breakdown_norm/") for key in metrics)
    assert not any(key.startswith("val_reward_breakdown_norm/") for key in metrics)


def test_flatten_speech_wandb_metrics_only_emits_enabled_mode_metrics():
    metrics = flatten_speech_wandb_metrics(
        {
            "mixed_sampling_enabled": True,
            "mixed_sde_step_count": 8,
            "gd2po_enabled": False,
            "gd2po_advantage_std": 0.81,
        }
    )

    assert metrics["mixed_sampling_enabled"] is True
    assert metrics["mixed_sde_step_count"] == 8
    assert "gd2po_enabled" not in metrics
    assert "gd2po_advantage_std" not in metrics


def test_compute_speech_best_metric_uses_stable_multi_reward_norm_breakdown():
    config = SimpleNamespace(
        best_metric="stable_multi_reward",
        speech=SimpleNamespace(
            reward=SimpleNamespace(
                registry=["dnsmos", "speaker_similarity", "speechbertscore"],
                weights={"dnsmos": 0.6, "speaker_similarity": 1.0, "speechbertscore": 1.0},
                primary_keys={
                    "dnsmos": "dnsmos_ovrl",
                    "speaker_similarity": "speaker_similarity",
                    "speechbertscore": "speechbertscore",
                },
            )
        ),
    )
    eval_metrics = {
        "reward_mean": 7279.0,
        "reward_breakdown_norm": {
            "dnsmos_ovrl": 0.593726,
            "speaker_similarity": 0.987444,
            "speechbertscore": 0.842606,
        },
    }

    name, value = compute_speech_best_metric(eval_metrics, config)

    assert name == "val_stable_multi_reward"
    assert abs(value - (0.6 * 0.593726 + 0.987444 + 0.842606)) < 1e-8
