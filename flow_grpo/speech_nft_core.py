
from __future__ import annotations


import random as _random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


SUPPORTED_SPEECH_REWARDS = (
    "dnsmos",
    "nisqa",
    "speechbertscore",
    "speaker_similarity",
)


def compute_reward_advantage_conflict_stats(
    branch_advantages: dict[str, Any],
    branch_names: list[str] | tuple[str, ...],
    *,
    weights: dict[str, float] | None = None,
    sign_eps: float = 1e-8,
    snr_eps: float = 1e-8,
    snr_thresholds: tuple[float, ...] = (0.5, 0.8),
) -> dict[str, float | int]:
    """Summarize cross-reward advantage sign conflicts and SNR consistency."""
    branch_names = [str(name) for name in branch_names]
    if len(branch_names) < 2:
        return {
            "reward_adv_conflict_branch_count": int(len(branch_names)),
            "reward_adv_conflict_sample_count": 0,
            "reward_adv_conflict_ratio": 0.0,
            "reward_adv_consensus_ratio": 0.0,
            "reward_adv_neutral_ratio": 0.0,
            "reward_adv_snr_mean": 0.0,
            "reward_adv_snr_min": 0.0,
        }

    arrays: list[np.ndarray] = []
    for branch_name in branch_names:
        if branch_name not in branch_advantages:
            raise KeyError(f"Missing branch advantages for {branch_name!r}.")
        value = branch_advantages[branch_name]
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().to(torch.float32).numpy().reshape(-1)
        else:
            array = np.asarray(value, dtype=np.float32).reshape(-1)
        arrays.append(array)

    sample_counts = {int(array.shape[0]) for array in arrays}
    if len(sample_counts) != 1:
        raise ValueError("All reward branch advantage arrays must have the same length.")
    sample_count = sample_counts.pop()
    threshold_stats = {
        f"reward_adv_snr_retained_ratio_tau_{str(float(threshold)).replace('.', '_')}": 0.0
        for threshold in snr_thresholds
    }
    if sample_count == 0:
        return {
            "reward_adv_conflict_branch_count": int(len(branch_names)),
            "reward_adv_conflict_sample_count": 0,
            "reward_adv_conflict_ratio": 0.0,
            "reward_adv_consensus_ratio": 0.0,
            "reward_adv_neutral_ratio": 0.0,
            "reward_adv_snr_mean": 0.0,
            "reward_adv_snr_min": 0.0,
            **threshold_stats,
        }

    stacked = np.stack(arrays, axis=1).astype(np.float64, copy=False)
    signs = np.zeros_like(stacked, dtype=np.int8)
    signs[stacked > float(sign_eps)] = 1
    signs[stacked < -float(sign_eps)] = -1
    has_positive = np.any(signs > 0, axis=1)
    has_negative = np.any(signs < 0, axis=1)
    has_non_neutral = np.any(signs != 0, axis=1)
    conflict_mask = has_positive & has_negative
    consensus_mask = has_non_neutral & ~conflict_mask
    neutral_mask = ~has_non_neutral

    if weights is None:
        weight_array = np.ones(len(branch_names), dtype=np.float64)
    else:
        weight_array = np.asarray([float(weights.get(name, 1.0)) for name in branch_names], dtype=np.float64)
    weighted = stacked * weight_array.reshape(1, -1)
    snr = np.abs(np.sum(weighted, axis=1)) / (np.sum(np.abs(weighted), axis=1) + float(snr_eps))

    stats: dict[str, float | int] = {
        "reward_adv_conflict_branch_count": int(len(branch_names)),
        "reward_adv_conflict_sample_count": int(np.count_nonzero(conflict_mask)),
        "reward_adv_conflict_ratio": float(np.mean(conflict_mask)),
        "reward_adv_consensus_ratio": float(np.mean(consensus_mask)),
        "reward_adv_neutral_ratio": float(np.mean(neutral_mask)),
        "reward_adv_snr_mean": float(np.mean(snr)),
        "reward_adv_snr_min": float(np.min(snr)),
    }
    for threshold in snr_thresholds:
        key = f"reward_adv_snr_retained_ratio_tau_{str(float(threshold)).replace('.', '_')}"
        stats[key] = float(np.mean(snr > float(threshold)))
    return stats


def compute_reward_advantage_pairwise_stats(
    branch_advantages: dict[str, Any],
    branch_names: list[str] | tuple[str, ...],
    *,
    weights: dict[str, float] | None = None,
    sign_eps: float = 1e-8,
    corr_eps: float = 1e-12,
    contribution_eps: float = 1e-8,
) -> dict[str, float]:
    """Measure pairwise agreement, correlation, lone dissent, and absolute contribution."""
    branch_names = [str(name) for name in branch_names]
    if len(branch_names) < 2:
        return {}

    arrays: list[np.ndarray] = []
    for branch_name in branch_names:
        if branch_name not in branch_advantages:
            raise KeyError(f"Missing branch advantages for {branch_name!r}.")
        value = branch_advantages[branch_name]
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().to(torch.float32).numpy().reshape(-1)
        else:
            array = np.asarray(value, dtype=np.float32).reshape(-1)
        arrays.append(array.astype(np.float64, copy=False))

    sample_counts = {int(array.shape[0]) for array in arrays}
    if len(sample_counts) != 1:
        raise ValueError("All reward branch advantage arrays must have the same length.")
    sample_count = sample_counts.pop()
    if sample_count == 0:
        return {}

    stacked = np.stack(arrays, axis=1)
    signs = np.zeros_like(stacked, dtype=np.int8)
    signs[stacked > float(sign_eps)] = 1
    signs[stacked < -float(sign_eps)] = -1
    stats: dict[str, float] = {}

    for left_idx, left_name in enumerate(branch_names):
        for right_idx in range(left_idx + 1, len(branch_names)):
            right_name = branch_names[right_idx]
            pair_name = f"{left_name}_{right_name}"
            valid = (signs[:, left_idx] != 0) & (signs[:, right_idx] != 0)
            valid_count = int(np.count_nonzero(valid))
            stats[f"reward_adv_pair_valid_ratio_{pair_name}"] = float(valid_count / sample_count)
            if valid_count > 0:
                agreement = signs[valid, left_idx] == signs[valid, right_idx]
                stats[f"reward_adv_pair_agreement_{pair_name}"] = float(np.mean(agreement))
                stats[f"reward_adv_pair_disagreement_{pair_name}"] = float(1.0 - np.mean(agreement))
            else:
                stats[f"reward_adv_pair_agreement_{pair_name}"] = 0.0
                stats[f"reward_adv_pair_disagreement_{pair_name}"] = 0.0

            left_values = stacked[:, left_idx]
            right_values = stacked[:, right_idx]
            if float(np.std(left_values)) <= float(corr_eps) or float(np.std(right_values)) <= float(corr_eps):
                correlation = 0.0
            else:
                correlation = float(np.corrcoef(left_values, right_values)[0, 1])
                if not np.isfinite(correlation):
                    correlation = 0.0
            stats[f"reward_adv_pair_correlation_{pair_name}"] = correlation

    for branch_idx, branch_name in enumerate(branch_names):
        other_indices = [idx for idx in range(len(branch_names)) if idx != branch_idx]
        other_signs = signs[:, other_indices]
        others_all_positive = np.all(other_signs > 0, axis=1)
        others_all_negative = np.all(other_signs < 0, axis=1)
        lone_dissent = (
            ((signs[:, branch_idx] < 0) & others_all_positive)
            | ((signs[:, branch_idx] > 0) & others_all_negative)
        )
        stats[f"reward_adv_lone_dissent_ratio_{branch_name}"] = float(np.mean(lone_dissent))

    if weights is None:
        weight_array = np.ones(len(branch_names), dtype=np.float64)
    else:
        weight_array = np.asarray([float(weights.get(name, 1.0)) for name in branch_names], dtype=np.float64)
    weighted_abs = np.abs(stacked * weight_array.reshape(1, -1))
    contribution_denom = np.sum(weighted_abs, axis=1, keepdims=True) + float(contribution_eps)
    contribution = weighted_abs / contribution_denom
    for branch_idx, branch_name in enumerate(branch_names):
        stats[f"reward_adv_abs_contribution_ratio_{branch_name}"] = float(
            np.mean(contribution[:, branch_idx])
        )
    return stats


def compute_reward_conflict_filter_diagnostics(
    branch_scores: dict[str, Any],
    branch_advantages: dict[str, Any],
    branch_names: list[str] | tuple[str, ...],
    conflict_keep_mask: Any,
    *,
    weights: dict[str, float] | None = None,
    sign_eps: float = 1e-8,
    snr_eps: float = 1e-8,
) -> dict[str, float | int]:
    """Describe raw rewards, advantages, and sign patterns selected by the SNR filter."""
    branch_names = [str(name) for name in branch_names]
    if len(branch_names) < 2:
        return {}

    score_arrays: list[np.ndarray] = []
    advantage_arrays: list[np.ndarray] = []
    for branch_name in branch_names:
        if branch_name not in branch_scores:
            raise KeyError(f"Missing branch scores for {branch_name!r}.")
        if branch_name not in branch_advantages:
            raise KeyError(f"Missing branch advantages for {branch_name!r}.")
        score_value = branch_scores[branch_name]
        advantage_value = branch_advantages[branch_name]
        if isinstance(score_value, torch.Tensor):
            score_array = score_value.detach().cpu().to(torch.float32).numpy().reshape(-1)
        else:
            score_array = np.asarray(score_value, dtype=np.float32).reshape(-1)
        if isinstance(advantage_value, torch.Tensor):
            advantage_array = advantage_value.detach().cpu().to(torch.float32).numpy().reshape(-1)
        else:
            advantage_array = np.asarray(advantage_value, dtype=np.float32).reshape(-1)
        score_arrays.append(score_array.astype(np.float64, copy=False))
        advantage_arrays.append(advantage_array.astype(np.float64, copy=False))

    sample_counts = {
        *(int(array.shape[0]) for array in score_arrays),
        *(int(array.shape[0]) for array in advantage_arrays),
    }
    if len(sample_counts) != 1:
        raise ValueError("All reward score and advantage arrays must have the same length.")
    sample_count = sample_counts.pop()
    keep_mask = np.asarray(conflict_keep_mask, dtype=bool).reshape(-1)
    if keep_mask.shape[0] != sample_count:
        raise ValueError("`conflict_keep_mask` must match the reward sample count.")
    if sample_count == 0:
        return {}

    filtered_mask = ~keep_mask
    kept_count = int(np.count_nonzero(keep_mask))
    filtered_count = int(np.count_nonzero(filtered_mask))
    score_stacked = np.stack(score_arrays, axis=1)
    advantage_stacked = np.stack(advantage_arrays, axis=1)

    def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        return float(np.mean(values[mask]))

    def masked_std(values: np.ndarray, mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        return float(np.std(values[mask]))

    stats: dict[str, float | int] = {
        "reward_filter_diagnostic_kept_count": kept_count,
        "reward_filter_diagnostic_filtered_count": filtered_count,
    }
    for branch_idx, branch_name in enumerate(branch_names):
        scores = score_stacked[:, branch_idx]
        advantages = advantage_stacked[:, branch_idx]
        kept_raw_mean = masked_mean(scores, keep_mask)
        filtered_raw_mean = masked_mean(scores, filtered_mask)
        stats[f"reward_filter_kept_raw_mean_{branch_name}"] = kept_raw_mean
        stats[f"reward_filter_filtered_raw_mean_{branch_name}"] = filtered_raw_mean
        stats[f"reward_filter_raw_delta_{branch_name}"] = filtered_raw_mean - kept_raw_mean
        stats[f"reward_filter_kept_raw_std_{branch_name}"] = masked_std(scores, keep_mask)
        stats[f"reward_filter_filtered_raw_std_{branch_name}"] = masked_std(scores, filtered_mask)
        stats[f"reward_filter_kept_adv_mean_{branch_name}"] = masked_mean(advantages, keep_mask)
        stats[f"reward_filter_filtered_adv_mean_{branch_name}"] = masked_mean(advantages, filtered_mask)
        stats[f"reward_filter_kept_adv_abs_mean_{branch_name}"] = masked_mean(
            np.abs(advantages), keep_mask
        )
        stats[f"reward_filter_filtered_adv_abs_mean_{branch_name}"] = masked_mean(
            np.abs(advantages), filtered_mask
        )

    if len(branch_names) != 3:
        return stats

    signs = np.zeros_like(advantage_stacked, dtype=np.int8)
    signs[advantage_stacked > float(sign_eps)] = 1
    signs[advantage_stacked < -float(sign_eps)] = -1
    if weights is None:
        weight_array = np.ones(len(branch_names), dtype=np.float64)
    else:
        weight_array = np.asarray([float(weights.get(name, 1.0)) for name in branch_names], dtype=np.float64)
    weighted = advantage_stacked * weight_array.reshape(1, -1)
    snr = np.abs(np.sum(weighted, axis=1)) / (np.sum(np.abs(weighted), axis=1) + float(snr_eps))

    covered_mask = np.zeros(sample_count, dtype=bool)
    for focal_idx, focal_name in enumerate(branch_names):
        other_indices = [idx for idx in range(len(branch_names)) if idx != focal_idx]
        for polarity_name, focal_sign, other_sign in (
            ("pos_only", 1, -1),
            ("neg_only", -1, 1),
        ):
            all_pattern_mask = (
                (signs[:, focal_idx] == focal_sign)
                & np.all(signs[:, other_indices] == other_sign, axis=1)
            )
            kept_pattern_mask = keep_mask & all_pattern_mask
            filtered_pattern_mask = filtered_mask & all_pattern_mask
            covered_mask |= filtered_pattern_mask
            total_pattern_count = int(np.count_nonzero(all_pattern_mask))
            kept_pattern_count = int(np.count_nonzero(kept_pattern_mask))
            filtered_pattern_count = int(np.count_nonzero(filtered_pattern_mask))
            pattern_prefix = f"reward_filter_pattern_{polarity_name}_{focal_name}"
            # Backward-compatible count/ratio describe this pattern's share within filtered samples.
            stats[f"{pattern_prefix}_count"] = filtered_pattern_count
            stats[f"{pattern_prefix}_ratio"] = (
                float(filtered_pattern_count / filtered_count) if filtered_count > 0 else 0.0
            )
            stats[f"{pattern_prefix}_total_count"] = total_pattern_count
            stats[f"{pattern_prefix}_kept_count"] = kept_pattern_count
            stats[f"{pattern_prefix}_filtered_count"] = filtered_pattern_count
            stats[f"{pattern_prefix}_kept_ratio"] = (
                float(kept_pattern_count / total_pattern_count) if total_pattern_count > 0 else 0.0
            )
            stats[f"{pattern_prefix}_filtered_ratio"] = (
                float(filtered_pattern_count / total_pattern_count) if total_pattern_count > 0 else 0.0
            )
            stats[f"{pattern_prefix}_snr_mean"] = masked_mean(snr, filtered_pattern_mask)
            stats[f"{pattern_prefix}_kept_snr_mean"] = masked_mean(snr, kept_pattern_mask)
            stats[f"{pattern_prefix}_filtered_snr_mean"] = masked_mean(snr, filtered_pattern_mask)
            for branch_idx, branch_name in enumerate(branch_names):
                stats[f"{pattern_prefix}_abs_adv_{branch_name}"] = masked_mean(
                    np.abs(advantage_stacked[:, branch_idx]), filtered_pattern_mask
                )
    stats["reward_filter_pattern_covered_ratio"] = (
        float(np.count_nonzero(covered_mask) / filtered_count) if filtered_count > 0 else 0.0
    )
    return stats


def compute_reward_advantage_snr_keep_mask(
    branch_advantages: dict[str, Any],
    branch_names: list[str] | tuple[str, ...],
    *,
    tau: float,
    weights: dict[str, float] | None = None,
    snr_eps: float = 1e-8,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Keep candidates whose weighted cross-reward advantage SNR is above ``tau``."""
    branch_names = [str(name) for name in branch_names]
    if len(branch_names) < 2:
        raise ValueError("Reward conflict filtering requires at least two reward branches.")

    tau = float(tau)
    if not np.isfinite(tau) or tau < 0.0 or tau > 1.0:
        raise ValueError("Reward conflict filter `tau` must be finite and within [0, 1].")
    snr_eps = float(snr_eps)
    if not np.isfinite(snr_eps) or snr_eps <= 0.0:
        raise ValueError("Reward conflict filter `snr_eps` must be finite and positive.")

    arrays: list[np.ndarray] = []
    for branch_name in branch_names:
        if branch_name not in branch_advantages:
            raise KeyError(f"Missing branch advantages for {branch_name!r}.")
        value = branch_advantages[branch_name]
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().to(torch.float32).numpy().reshape(-1)
        else:
            array = np.asarray(value, dtype=np.float32).reshape(-1)
        arrays.append(array)

    sample_counts = {int(array.shape[0]) for array in arrays}
    if len(sample_counts) != 1:
        raise ValueError("All reward branch advantage arrays must have the same length.")
    sample_count = sample_counts.pop()
    if sample_count == 0:
        return np.zeros(0, dtype=bool), {
            "reward_conflict_filter_enabled": True,
            "reward_conflict_filter_tau": tau,
            "reward_conflict_filter_sample_count": 0,
            "reward_conflict_filter_kept_sample_count": 0,
            "reward_conflict_filter_kept_sample_ratio": 0.0,
            "reward_conflict_filter_filtered_sample_count": 0,
            "reward_conflict_filter_filtered_sample_ratio": 0.0,
            "reward_conflict_filter_conflict_sample_count": 0,
            "reward_conflict_filter_conflict_sample_ratio": 0.0,
        }

    stacked = np.stack(arrays, axis=1).astype(np.float64, copy=False)
    if weights is None:
        weight_array = np.ones(len(branch_names), dtype=np.float64)
    else:
        weight_array = np.asarray([float(weights.get(name, 1.0)) for name in branch_names], dtype=np.float64)
    weighted = stacked * weight_array.reshape(1, -1)
    snr = np.abs(np.sum(weighted, axis=1)) / (np.sum(np.abs(weighted), axis=1) + snr_eps)
    positive_mask = stacked > 1e-8
    negative_mask = stacked < -1e-8
    conflict_mask = np.any(positive_mask, axis=1) & np.any(negative_mask, axis=1)
    filtered_mask = conflict_mask & (snr < tau)
    keep_mask = ~filtered_mask
    kept_sample_count = int(np.count_nonzero(keep_mask))
    filtered_sample_count = int(sample_count - kept_sample_count)
    conflict_sample_count = int(np.count_nonzero(conflict_mask))

    return keep_mask, {
        "reward_conflict_filter_enabled": True,
        "reward_conflict_filter_tau": tau,
        "reward_conflict_filter_sample_count": int(sample_count),
        "reward_conflict_filter_kept_sample_count": kept_sample_count,
        "reward_conflict_filter_kept_sample_ratio": float(kept_sample_count / sample_count),
        "reward_conflict_filter_filtered_sample_count": filtered_sample_count,
        "reward_conflict_filter_filtered_sample_ratio": float(filtered_sample_count / sample_count),
        "reward_conflict_filter_conflict_sample_count": conflict_sample_count,
        "reward_conflict_filter_conflict_sample_ratio": float(conflict_sample_count / sample_count),
    }


def aggregate_reward_advantages(
    branch_advantages: dict[str, Any],
    branch_names: list[str] | tuple[str, ...],
    *,
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Aggregate independently normalized reward advantages into one GDPO advantage."""
    branch_names = [str(name) for name in branch_names]
    if len(branch_names) < 2:
        raise ValueError("GD2PO advantage aggregation requires at least two reward branches.")

    arrays: list[np.ndarray] = []
    for branch_name in branch_names:
        if branch_name not in branch_advantages:
            raise KeyError(f"Missing branch advantages for {branch_name!r}.")
        value = branch_advantages[branch_name]
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().to(torch.float32).numpy().reshape(-1)
        else:
            array = np.asarray(value, dtype=np.float32).reshape(-1)
        arrays.append(array)

    sample_counts = {int(array.shape[0]) for array in arrays}
    if len(sample_counts) != 1:
        raise ValueError("All reward branch advantage arrays must have the same length.")
    if weights is None:
        weight_array = np.ones(len(branch_names), dtype=np.float64)
    else:
        weight_array = np.asarray([float(weights.get(name, 0.0)) for name in branch_names], dtype=np.float64)
    if not np.all(np.isfinite(weight_array)) or np.any(weight_array < 0.0):
        raise ValueError("GD2PO reward weights must be finite and non-negative.")
    weight_sum = float(np.sum(weight_array))
    if weight_sum <= 0.0:
        raise ValueError("GD2PO requires at least one positive reward weight.")

    normalized_weights = weight_array / weight_sum
    stacked = np.stack(arrays, axis=1).astype(np.float64, copy=False)
    return np.sum(stacked * normalized_weights.reshape(1, -1), axis=1).astype(np.float32)


def compute_gd2po_group_keep_ratios(
    group_ids: list[str] | tuple[str, ...] | np.ndarray,
    keep_mask: Any,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Return the official GD2PO per-query retained fraction for every candidate."""
    group_id_array = np.asarray(group_ids, dtype=object).reshape(-1)
    keep_array = np.asarray(keep_mask, dtype=bool).reshape(-1)
    if group_id_array.shape != keep_array.shape:
        raise ValueError("`group_ids` and `keep_mask` must have the same length.")

    group_keep_ratios = np.zeros(keep_array.shape[0], dtype=np.float32)
    for group_id in dict.fromkeys(group_id_array.tolist()):
        group_mask = group_id_array == group_id
        group_size = int(np.count_nonzero(group_mask))
        if group_size > 0:
            group_keep_ratios[group_mask] = float(np.count_nonzero(keep_array[group_mask])) / float(group_size)

    if group_keep_ratios.size:
        ratio_mean = float(np.mean(group_keep_ratios))
        ratio_min = float(np.min(group_keep_ratios))
        ratio_max = float(np.max(group_keep_ratios))
        ratio_std = float(np.std(group_keep_ratios))
    else:
        ratio_mean = ratio_min = ratio_max = ratio_std = 0.0

    return group_keep_ratios, {
        "gd2po_group_keep_ratio_enabled": True,
        "gd2po_group_keep_ratio_mean": ratio_mean,
        "gd2po_group_keep_ratio_min": ratio_min,
        "gd2po_group_keep_ratio_max": ratio_max,
        "gd2po_group_keep_ratio_std": ratio_std,
    }


def compute_advantage_sign_flip_counts(
    pre_normalization_advantages: Any,
    post_normalization_advantages: Any,
    keep_mask: Any,
    *,
    sign_eps: float = 1e-8,
) -> np.ndarray:
    """Count retained-sample sign changes caused by post-normalization.

    Returns ``[kept, pre_positive, pre_negative, positive_to_negative,
    negative_to_positive]`` so callers can sum the counts across DDP ranks
    before converting them to ratios.
    """
    if float(sign_eps) < 0.0:
        raise ValueError("`sign_eps` must be non-negative.")

    def as_numpy(value: Any, *, dtype: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().astype(dtype, copy=False).reshape(-1)
        return np.asarray(value, dtype=dtype).reshape(-1)

    pre = as_numpy(pre_normalization_advantages, dtype=np.float64)
    post = as_numpy(post_normalization_advantages, dtype=np.float64)
    kept = as_numpy(keep_mask, dtype=bool)
    if pre.shape != post.shape or pre.shape != kept.shape:
        raise ValueError("Pre/post advantages and `keep_mask` must have the same shape.")
    if not np.all(np.isfinite(pre)) or not np.all(np.isfinite(post)):
        raise ValueError("Pre/post advantages must contain only finite values.")

    eps = float(sign_eps)
    pre_positive = kept & (pre > eps)
    pre_negative = kept & (pre < -eps)
    positive_to_negative = pre_positive & (post < -eps)
    negative_to_positive = pre_negative & (post > eps)
    return np.asarray(
        [
            np.count_nonzero(kept),
            np.count_nonzero(pre_positive),
            np.count_nonzero(pre_negative),
            np.count_nonzero(positive_to_negative),
            np.count_nonzero(negative_to_positive),
        ],
        dtype=np.float64,
    )


def summarize_advantage_sign_flip_counts(counts: Any) -> dict[str, float]:
    """Convert globally aggregated post-normalization sign counts to ratios."""
    count_array = np.asarray(counts, dtype=np.float64).reshape(-1)
    if count_array.shape != (5,):
        raise ValueError("`counts` must contain five sign-flip counters.")
    if not np.all(np.isfinite(count_array)) or np.any(count_array < 0.0):
        raise ValueError("Sign-flip counters must be finite and non-negative.")

    kept, pre_positive, pre_negative, positive_to_negative, negative_to_positive = count_array.tolist()
    sign_flips = positive_to_negative + negative_to_positive
    return {
        "gd2po_post_norm_sign_flip_ratio": float(sign_flips / kept) if kept > 0.0 else 0.0,
        "gd2po_post_norm_positive_to_negative_ratio": (
            float(positive_to_negative / pre_positive) if pre_positive > 0.0 else 0.0
        ),
        "gd2po_post_norm_negative_to_positive_ratio": (
            float(negative_to_positive / pre_negative) if pre_negative > 0.0 else 0.0
        ),
    }


def masked_normalize_aggregated_advantages(
    advantages: Any,
    keep_mask: Any,
    *,
    eps: float = 1e-4,
    moments: Any | None = None,
) -> np.ndarray:
    """Normalize aggregated advantages using retained samples only.

    Filtered samples are set to zero after computing the retained-sample
    statistics, matching GD2PO's filter-then-normalize ordering.
    """
    if float(eps) <= 0.0:
        raise ValueError("`eps` must be positive.")

    if isinstance(advantages, torch.Tensor):
        advantage_array = advantages.detach().cpu().to(torch.float32).numpy().reshape(-1)
    else:
        advantage_array = np.asarray(advantages, dtype=np.float32).reshape(-1)
    if isinstance(keep_mask, torch.Tensor):
        keep_array = keep_mask.detach().cpu().to(torch.bool).numpy().reshape(-1)
    else:
        keep_array = np.asarray(keep_mask, dtype=bool).reshape(-1)
    if advantage_array.shape != keep_array.shape:
        raise ValueError("`advantages` and `keep_mask` must have the same shape.")
    if not np.all(np.isfinite(advantage_array)):
        raise ValueError("`advantages` must contain only finite values.")

    normalized = np.zeros_like(advantage_array, dtype=np.float32)
    if not np.any(keep_array):
        return normalized

    retained = advantage_array[keep_array].astype(np.float64, copy=False)
    if moments is None:
        retained_count = float(retained.size)
        retained_sum = float(np.sum(retained, dtype=np.float64))
        retained_sum_sq = float(np.sum(np.square(retained), dtype=np.float64))
    else:
        moment_array = np.asarray(moments, dtype=np.float64).reshape(-1)
        if moment_array.shape != (3,):
            raise ValueError("`moments` must contain [count, sum, sum_sq].")
        if not np.all(np.isfinite(moment_array)) or moment_array[0] < 0.0:
            raise ValueError("`moments` must be finite and have a non-negative count.")
        retained_count, retained_sum, retained_sum_sq = moment_array.tolist()
    if retained_count <= 0.0:
        return normalized

    retained_mean = retained_sum / retained_count
    retained_variance = max(retained_sum_sq / retained_count - retained_mean**2, 0.0)
    retained_std = float(np.sqrt(retained_variance))
    normalized[keep_array] = ((retained - retained_mean) / (retained_std + float(eps))).astype(np.float32)
    return normalized


def masked_rms_scale_aggregated_advantages(
    advantages: Any,
    keep_mask: Any,
    *,
    eps: float = 1e-4,
    moments: Any | None = None,
) -> np.ndarray:
    """Scale retained aggregate advantages by global RMS without centering.

    Unlike masked whitening, RMS scaling preserves every retained sample's
    original sign while keeping the aggregate advantage magnitude stable.
    Filtered samples are returned as zero.
    """
    if float(eps) <= 0.0:
        raise ValueError("`eps` must be positive.")

    if isinstance(advantages, torch.Tensor):
        advantage_array = advantages.detach().cpu().to(torch.float32).numpy().reshape(-1)
    else:
        advantage_array = np.asarray(advantages, dtype=np.float32).reshape(-1)
    if isinstance(keep_mask, torch.Tensor):
        keep_array = keep_mask.detach().cpu().to(torch.bool).numpy().reshape(-1)
    else:
        keep_array = np.asarray(keep_mask, dtype=bool).reshape(-1)
    if advantage_array.shape != keep_array.shape:
        raise ValueError("`advantages` and `keep_mask` must have the same shape.")
    if not np.all(np.isfinite(advantage_array)):
        raise ValueError("`advantages` must contain only finite values.")

    scaled = np.zeros_like(advantage_array, dtype=np.float32)
    if not np.any(keep_array):
        return scaled

    retained = advantage_array[keep_array].astype(np.float64, copy=False)
    if moments is None:
        retained_count = float(retained.size)
        retained_sum_sq = float(np.sum(np.square(retained), dtype=np.float64))
    else:
        moment_array = np.asarray(moments, dtype=np.float64).reshape(-1)
        if moment_array.shape != (3,):
            raise ValueError("`moments` must contain [count, sum, sum_sq].")
        if not np.all(np.isfinite(moment_array)) or moment_array[0] < 0.0:
            raise ValueError("`moments` must be finite and have a non-negative count.")
        retained_count = float(moment_array[0])
        retained_sum_sq = float(moment_array[2])
    if retained_count <= 0.0:
        return scaled

    retained_rms = float(np.sqrt(max(retained_sum_sq / retained_count, 0.0)))
    scaled[keep_array] = (retained / (retained_rms + float(eps))).astype(np.float32)
    return scaled


@dataclass
class RolloutBatch:

    x0_target: torch.Tensor
    condition: torch.Tensor
    timesteps: torch.Tensor
    candidate_ids: list[str]
    group_ids: list[str]
    reward_dict: dict[str, Any]
    reward_avg: torch.Tensor
    noisy_mel: torch.Tensor
    generated_mel: torch.Tensor
    utt_ids: list[str]
    source_utt_ids: list[str]
    advantages: torch.Tensor | None = None
    train_mask: torch.Tensor | None = None
    sde_timestep_mask: torch.Tensor | None = None
    reward_branch_advantages: dict[str, torch.Tensor] | None = None
    nft_target_time: float = 1.0
    trajectory_final_similarity: dict[str, torch.Tensor] | None = None
    trajectory_information_gain: dict[str, torch.Tensor] | None = None
    trajectory_embedding_kinds: dict[str, str] | None = None


@dataclass
class SpeechMixedSamplingState:
    """MixGRPO-style progressive timestep window for speech rollout sampling."""

    steps: int
    group_size: int
    strategy: str = "progressive"
    overlap: bool = True
    overlap_step: int = 1
    update_interval: int = 1
    roll_back: bool = True
    cur_timestep: int = 0
    cur_iter_in_interval: int = 0

    def __post_init__(self):
        self.steps = int(self.steps)
        self.group_size = int(self.group_size)
        self.overlap_step = int(self.overlap_step)
        self.update_interval = int(self.update_interval)
        self.cur_timestep = int(self.cur_timestep)
        self.cur_iter_in_interval = int(self.cur_iter_in_interval)
        self.strategy = str(self.strategy)

        if self.steps <= 0:
            raise ValueError("`steps` must be positive.")
        if self.group_size <= 0:
            raise ValueError("`group_size` must be positive.")
        if self.overlap_step <= 0:
            raise ValueError("`overlap_step` must be positive.")
        if self.update_interval <= 0:
            raise ValueError("`update_interval` must be positive.")
        if self.strategy != "progressive":
            raise ValueError("Speech mixed sampling currently supports only strategy='progressive'.")

        self.group_size = min(self.group_size, self.steps)
        self.cur_timestep = min(max(self.cur_timestep, 0), self.max_start_timestep)
        self.cur_iter_in_interval = max(self.cur_iter_in_interval, 0)

    @property
    def max_start_timestep(self) -> int:
        return max(self.steps - self.group_size, 0)

    def get_current_sde_indices(self) -> list[int]:
        end = min(self.cur_timestep + self.group_size, self.steps)
        return list(range(self.cur_timestep, end))

    def get_current_deterministic_mask(self) -> list[bool]:
        mask = [True] * self.steps
        for index in self.get_current_sde_indices():
            mask[index] = False
        return mask

    def update_iteration(self) -> None:
        self.cur_iter_in_interval += 1
        if self.cur_iter_in_interval < self.update_interval:
            return
        self.cur_iter_in_interval = 0

        stride = self.overlap_step if self.overlap else self.group_size
        self.cur_timestep += max(1, int(stride))
        if self.cur_timestep > self.max_start_timestep:
            self.cur_timestep = 0 if self.roll_back else self.max_start_timestep

    def get_current_stats(self) -> dict[str, Any]:
        indices = self.get_current_sde_indices()
        return {
            "mixed_sampling_enabled": True,
            "mixed_sde_step_count": int(len(indices)),
            "mixed_ode_step_count": int(self.steps - len(indices)),
            "mixed_sde_timestep_start": int(indices[0]) if indices else -1,
            "mixed_sde_timestep_end": int(indices[-1]) if indices else -1,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in (
                "steps", "group_size", "strategy", "overlap", "overlap_step",
                "update_interval", "roll_back", "cur_timestep", "cur_iter_in_interval",
            )
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for name in ("steps", "group_size", "strategy", "overlap", "overlap_step", "update_interval", "roll_back"):
            if state[name] != getattr(self, name):
                raise ValueError(f"Mixed sampling configuration mismatch for {name}: {state[name]} != {getattr(self, name)}")
        timestep = int(state["cur_timestep"])
        iteration = int(state["cur_iter_in_interval"])
        if not 0 <= timestep <= self.max_start_timestep or not 0 <= iteration < self.update_interval:
            raise ValueError("Invalid mixed sampling progress in checkpoint.")
        self.cur_timestep = timestep
        self.cur_iter_in_interval = iteration


def build_all_ode_deterministic_mask(num_steps: int) -> list[bool]:
    num_steps = int(num_steps)
    if num_steps <= 0:
        raise ValueError("`num_steps` must be positive.")
    return [True] * num_steps


def build_eval_deterministic_mask(
    num_steps: int,
    *,
    use_rollout_schedule: bool,
    use_rollout_deterministic_mask: bool,
    rollout_deterministic_mask,
) -> list[bool]:
    """Resolve validation to all-ODE or the rollout's exact per-step SDE/ODE path."""
    if not bool(use_rollout_deterministic_mask):
        return build_all_ode_deterministic_mask(num_steps)
    if not bool(use_rollout_schedule):
        raise ValueError(
            "`speech.eval.use_rollout_deterministic_mask=True` requires "
            "`speech.eval.use_rollout_schedule=True`."
        )
    mask = [bool(value) for value in rollout_deterministic_mask]
    if len(mask) != int(num_steps):
        raise ValueError(
            "Validation was configured to reuse the rollout deterministic mask, "
            f"but its length is {len(mask)} instead of eval steps {int(num_steps)}."
        )
    return mask


class PerConditionStatTracker:

    def __init__(self, global_std: bool = False, eps: float = 1e-4):
        self.global_std = bool(global_std)
        self.eps = float(eps)
        self.last_update: dict[str, Any] = {
            "num_groups": 0,
            "group_size_mean": 0.0,
            "zero_std_ratio": 0.0,
            "reward_std_mean": 0.0,
            "all_zero_advantage_ratio": 0.0,
            "advantage_mean": 0.0,
            "advantage_std": 0.0,
        }

    def update(self, conditions: list[str], rewards: list[float] | np.ndarray) -> np.ndarray:
        condition_array = np.asarray(list(conditions), dtype=object)
        reward_array = np.asarray(rewards, dtype=np.float64).reshape(-1)
        if condition_array.shape[0] != reward_array.shape[0]:
            raise ValueError("`conditions` and `rewards` must have the same length.")

        unique_conditions = np.unique(condition_array)
        advantages = np.zeros_like(reward_array, dtype=np.float64)
        reward_std_values = []
        group_sizes = []
        global_std_value = float(np.std(reward_array))

        for condition in unique_conditions:
            mask = condition_array == condition
            condition_rewards = reward_array[mask]
            condition_mean = float(np.mean(condition_rewards))
            condition_std = float(np.std(condition_rewards))
            std_for_advantage = (global_std_value if self.global_std else condition_std) + self.eps
            # advantage = (reward - group_mean) / std
            advantages[mask] = (condition_rewards - condition_mean) / std_for_advantage
            reward_std_values.append(condition_std)
            group_sizes.append(int(condition_rewards.shape[0]))

        zero_std_ratio = float(np.mean(np.asarray(reward_std_values) <= self.eps)) if reward_std_values else 0.0
        reward_std_mean = float(np.mean(reward_std_values)) if reward_std_values else 0.0
        all_zero_advantage_ratio = float(np.mean(np.abs(advantages) <= self.eps)) if advantages.size else 0.0

        self.last_update = {
            "num_groups": int(len(unique_conditions)),
            "group_size_mean": float(np.mean(group_sizes)) if group_sizes else 0.0,
            "zero_std_ratio": zero_std_ratio,
            "reward_std_mean": reward_std_mean,
            "all_zero_advantage_ratio": all_zero_advantage_ratio,
            "advantage_mean": float(np.mean(advantages)) if advantages.size else 0.0,
            "advantage_std": float(np.std(advantages)) if advantages.size else 0.0,
        }
        return advantages

    def get_last_stats(self) -> dict[str, Any]:
        return dict(self.last_update)


def compute_low_std_group_keep_mask(
    conditions: list[str],
    rewards: list[float] | np.ndarray,
    *,
    mean_threshold: float,
    std_threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    condition_array = np.asarray(list(conditions), dtype=object)
    reward_array = np.asarray(rewards, dtype=np.float64).reshape(-1)
    if condition_array.shape[0] != reward_array.shape[0]:
        raise ValueError("`conditions` and `rewards` must have the same length.")
    if std_threshold < 0:
        raise ValueError("`std_threshold` must be non-negative.")

    keep_mask = np.ones(reward_array.shape[0], dtype=bool)
    group_means: list[float] = []
    group_stds: list[float] = []
    filtered_group_count = 0
    filtered_sample_count = 0

    unique_conditions = np.unique(condition_array)
    for condition in unique_conditions:
        group_mask = condition_array == condition
        group_rewards = reward_array[group_mask]
        group_mean = float(np.mean(group_rewards))
        group_std = float(np.std(group_rewards))
        group_means.append(group_mean)
        group_stds.append(group_std)

        should_filter = (
            group_rewards.shape[0] >= 2
            and group_mean > float(mean_threshold)
            and group_std < float(std_threshold)
        )
        if should_filter:
            keep_mask[group_mask] = False
            filtered_group_count += 1
            filtered_sample_count += int(np.count_nonzero(group_mask))

    total_groups = int(len(unique_conditions))
    total_samples = int(reward_array.shape[0])
    kept_sample_count = total_samples - filtered_sample_count
    stats = {
        "low_std_filter_enabled": True,
        "low_std_filter_mean_threshold": float(mean_threshold),
        "low_std_filter_std_threshold": float(std_threshold),
        "low_std_filter_group_count": int(filtered_group_count),
        "low_std_filter_group_ratio": (
            float(filtered_group_count / total_groups) if total_groups > 0 else 0.0
        ),
        "low_std_filter_sample_count": int(filtered_sample_count),
        "low_std_filter_sample_ratio": (
            float(filtered_sample_count / total_samples) if total_samples > 0 else 0.0
        ),
        "low_std_filter_kept_sample_count": int(kept_sample_count),
        "low_std_filter_kept_sample_ratio": (
            float(kept_sample_count / total_samples) if total_samples > 0 else 0.0
        ),
        "low_std_filter_reward_mean_mean": float(np.mean(group_means)) if group_means else 0.0,
        "low_std_filter_reward_std_mean": float(np.mean(group_stds)) if group_stds else 0.0,
    }
    return keep_mask, stats

def diffusion_nft_decay(step: int, decay_type: int) -> float:
    if decay_type == 0:
        flat = 0
        uprate = 0.0
        uphold = 0.0
    elif decay_type == 1:
        flat = 0
        uprate = 0.001
        uphold = 0.5
    elif decay_type == 2:
        flat = 75
        uprate = 0.0075
        uphold = 0.999
    else:
        raise ValueError(f"Unsupported decay_type: {decay_type!r}")

    if step < flat:
        return 0.0
    decay = (step - flat) * uprate
    return float(min(decay, uphold))


def normalize_reward_weights(
    registry: list[str],
    weights: dict[str, float] | None,
) -> dict[str, float]:
    registry_set = {item.strip().lower() for item in registry}
    invalid = sorted(registry_set - set(SUPPORTED_SPEECH_REWARDS))
    if invalid:
        raise ValueError(f"Unsupported reward registry entries: {invalid}")

    resolved = {metric: 0.0 for metric in SUPPORTED_SPEECH_REWARDS}
    if weights:
        for metric, value in weights.items():
            metric_key = str(metric).strip().lower()
            if metric_key not in resolved:
                raise ValueError(f"Unsupported reward weight metric: {metric}")
            resolved[metric_key] = float(value)

    for metric in registry_set:
        if metric not in resolved:
            resolved[metric] = 1.0

    for metric in registry_set:
        if metric in SUPPORTED_SPEECH_REWARDS and metric not in (weights or {}):
            resolved[metric] = 1.0
    return resolved


def build_default_timestep_schedule(
    batch_size: int,
    num_timesteps: int,
    device: torch.device | str,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("`batch_size` must be positive")
    if num_timesteps <= 0:
        raise ValueError("`num_timesteps` must be positive")
    base = torch.linspace(999.0, 0.0, steps=num_timesteps, device=device)
    base = torch.round(base).to(torch.long)
    return base.unsqueeze(0).repeat(batch_size, 1)


def select_timestep_indices(
    timestep_values: torch.Tensor | np.ndarray | list[float],
    *,
    timestep_fraction: float = 1.0,
    timestep_window: list[float] | tuple[float, float] | None = None,
    timesteps_per_batch: int | None = None,
    rng: Any = None,
) -> list[int]:
    """
    Select rollout-aligned timestep indices for inner NFT updates.

    `timestep_window` limits the candidate time range in continuous flow time:
    t near 0 is noise-like, t near 1 is close to the generated sample.
    `timesteps_per_batch` fixes the compute budget; when unset, the legacy
    `timestep_fraction` behavior is used inside the selected window.
    """
    if isinstance(timestep_values, torch.Tensor):
        values = timestep_values.detach().cpu().to(torch.float32).numpy()
    else:
        values = np.asarray(timestep_values, dtype=np.float32)
    values = values.reshape(-1)
    total_count = int(values.shape[0])
    if total_count <= 0:
        raise ValueError("`timestep_values` must contain at least one timestep.")
    if not (0.0 < float(timestep_fraction) <= 1.0):
        raise ValueError("`timestep_fraction` must be in (0, 1].")

    if timestep_window is None:
        candidate_indices = list(range(total_count))
    else:
        if len(timestep_window) != 2:
            raise ValueError("`timestep_window` must contain exactly two values: [start, end].")
        start = float(timestep_window[0])
        end = float(timestep_window[1])
        if not (0.0 <= start < end <= 1.0):
            raise ValueError("`timestep_window` must satisfy 0.0 <= start < end <= 1.0.")
        finite = np.isfinite(values)
        if end >= 1.0:
            mask = finite & (values >= start) & (values <= end)
        else:
            mask = finite & (values >= start) & (values < end)
        candidate_indices = np.nonzero(mask)[0].astype(int).tolist()
        if not candidate_indices:
            raise ValueError(
                f"`timestep_window` {list(timestep_window)} selected no timesteps from "
                f"available range [{float(np.nanmin(values)):.6f}, {float(np.nanmax(values)):.6f}]."
            )

    if timesteps_per_batch is None or int(timesteps_per_batch) == 0:
        train_count = max(1, int(len(candidate_indices) * float(timestep_fraction)))
    else:
        train_count = int(timesteps_per_batch)
        if train_count < 0:
            raise ValueError("`timesteps_per_batch` must be non-negative.")
    train_count = min(train_count, len(candidate_indices))

    selected = list(candidate_indices)
    (rng or _random).shuffle(selected)
    return selected[:train_count]


def build_training_timestep_grid(
    rollout_timestep_values: torch.Tensor | np.ndarray | list[float],
    timestep_grid_steps: int = 0,
) -> torch.Tensor:
    """Return rollout-aligned timesteps or an independent uniform training grid."""
    if isinstance(rollout_timestep_values, torch.Tensor):
        rollout_values = rollout_timestep_values.detach().to(dtype=torch.float32).reshape(-1)
        device = rollout_values.device
    else:
        rollout_values = torch.as_tensor(rollout_timestep_values, dtype=torch.float32).reshape(-1)
        device = rollout_values.device
    if int(timestep_grid_steps) <= 0:
        if rollout_values.numel() <= 0:
            raise ValueError("`rollout_timestep_values` must contain at least one timestep.")
        return rollout_values
    grid_steps = int(timestep_grid_steps)
    return torch.linspace(0.0, 1.0, steps=grid_steps, device=device, dtype=torch.float32)


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _to_tensor(values, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=dtype)
    return torch.as_tensor(values, device=device, dtype=dtype)


def expand_sampled_time(
    sampled_time,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert a scalar or per-sample time vector into a [B] tensor."""
    if batch_size <= 0:
        raise ValueError("`batch_size` must be positive.")
    time = _to_tensor(sampled_time, device=device, dtype=dtype).reshape(-1)
    if time.numel() == 1:
        return time.expand(int(batch_size))
    if time.numel() != int(batch_size):
        raise ValueError(f"`sampled_time` must be scalar or match batch size {batch_size}, got {time.numel()} values.")
    return time


def select_nft_target_sample(
    generated_sample: torch.Tensor,
    trajectory: torch.Tensor,
    time_grid: torch.Tensor,
    target_time: float | None,
) -> tuple[torch.Tensor, float]:
    """Select one rollout endpoint for NFT and allow the remaining trajectory to be released."""
    if target_time is None:
        # Preserve the original NFT convention: the sampler output is treated as t=1.
        return generated_sample, 1.0

    target_time = float(target_time)
    if not (0.0 < target_time <= 1.0):
        raise ValueError("`nft_target_time` must be in (0, 1].")
    if trajectory.ndim < 2:
        raise ValueError(f"Expected trajectory shaped [T,B,...], got {tuple(trajectory.shape)}.")

    grid = torch.as_tensor(time_grid, device=trajectory.device, dtype=torch.float32).reshape(-1)
    if int(trajectory.shape[0]) != int(grid.numel()):
        raise ValueError(
            "Trajectory/time-grid length mismatch: "
            f"trajectory={int(trajectory.shape[0])}, time_grid={int(grid.numel())}."
        )
    matches = torch.nonzero(
        torch.isclose(grid, torch.tensor(target_time, device=grid.device), atol=1e-6, rtol=0.0),
        as_tuple=False,
    ).reshape(-1)
    if matches.numel() != 1:
        raise ValueError(
            f"`nft_target_time={target_time}` must occur exactly once in the rollout time grid; "
            f"found {int(matches.numel())} matches in {grid.detach().cpu().tolist()}."
        )
    # Clone the selected state so the returned tensor does not keep the stacked
    # trajectory's entire backing storage alive after rollout.
    return trajectory[int(matches.item())].clone(), target_time


def _mean_over_nonbatch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 1:
        return tensor
    reduce_dims = tuple(range(1, tensor.ndim))
    return tensor.mean(dim=reduce_dims)


def _time_to_broadcast(time: torch.Tensor, ndim: int) -> torch.Tensor:
    if ndim < 1:
        raise ValueError("`ndim` must be positive.")
    view_shape = (time.shape[0],) + (1,) * (ndim - 1)
    return time.reshape(view_shape)


def build_advantage_weights(
    advantages,
    mode: str = "shifted_clip",
    adv_clip_max: float = 5.0,
) -> torch.Tensor:
    if adv_clip_max <= 0:
        raise ValueError("`adv_clip_max` must be positive.")

    if isinstance(advantages, torch.Tensor):
        device = advantages.device
    else:
        device = torch.device("cpu")
    advantages = _to_tensor(advantages, device=device, dtype=torch.float32).reshape(-1)
    clipped = torch.clamp(advantages, -adv_clip_max, adv_clip_max)

    if mode == "shifted_clip":
        weights = (clipped / adv_clip_max) / 2.0 + 0.5
        return torch.clamp(weights, 0.0, 1.0)
    if mode == "positive_only":
        weights = torch.clamp(clipped, 0.0, adv_clip_max) / adv_clip_max
        return torch.clamp(weights, 0.0, 1.0)
    if mode == "binary":
        return (clipped > 0).to(torch.float32)
    raise ValueError(f"Unsupported advantage mode: {mode}")


def reconstruct_x0_from_flow(
    x_t: torch.Tensor,
    flow_pred: torch.Tensor,
    time: torch.Tensor,
    target_time: float = 1.0,
) -> torch.Tensor:
    target_time = float(target_time)
    if not (0.0 < target_time <= 1.0):
        raise ValueError("`target_time` must be in (0, 1].")
    t = _time_to_broadcast(time, x_t.ndim).to(device=x_t.device, dtype=x_t.dtype)
    return x_t + (target_time - t) * flow_pred


def compute_adaptive_reconstruction_loss(
    x0_prediction: torch.Tensor,
    x0_target: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    diff = x0_prediction - x0_target
    weight_factor = diff.detach().to(torch.float64).abs().mean(dim=tuple(range(1, diff.ndim)), keepdim=True)
    weight_factor = torch.clamp(weight_factor, min=float(eps)).to(device=diff.device, dtype=diff.dtype)
    per_sample_loss = _mean_over_nonbatch((diff * diff) / weight_factor)
    return per_sample_loss, weight_factor


def compute_pure_nft_terms(
    current_model,
    old_model,
    ref_model,
    *,
    noisy_mel: torch.Tensor,
    generated_mel: torch.Tensor,
    advantages,
    train_mask: torch.Tensor | None = None,
    train_mask_normalization: str = "kept",
    beta_mix: float,
    adv_clip_max: float,
    kl_coef: float,
    time: torch.Tensor,
    noise: torch.Tensor | None = None,
    cond_type: str = "wotext",
    adv_weight_mode: str = "shifted_clip",
    target_time: float = 1.0,
) -> dict[str, torch.Tensor]:
    if cond_type != "wotext":
        raise ValueError(f"Speech NFT currently supports only `wotext`, got cond_type={cond_type!r}")
    if beta_mix <= 0:
        raise ValueError("`beta_mix` must be positive.")

    current = _unwrap_model(current_model)
    old = _unwrap_model(old_model)
    ref = _unwrap_model(ref_model)

    x0_target = current.prepare_condition(generated_mel)
    cond = current.prepare_condition(noisy_mel)

    if noise is None:
        noise = torch.randn_like(x0_target)
    else:
        noise = _to_tensor(noise, device=x0_target.device, dtype=x0_target.dtype)
        if noise.shape != x0_target.shape:
            try:
                noise = current.prepare_condition(noise)
            except Exception:
                pass
        if noise.shape != x0_target.shape:
            raise ValueError(
                f"`noise` must have the same shape as generated target, got noise={tuple(noise.shape)} "
                f"target={tuple(x0_target.shape)}"
            )

    target_time = float(target_time)
    if not (0.0 < target_time <= 1.0):
        raise ValueError("`target_time` must be in (0, 1].")

    time = _to_tensor(time, device=x0_target.device, dtype=x0_target.dtype).reshape(-1)
    if time.shape[0] != x0_target.shape[0]:
        raise ValueError("`time` batch size must match `generated_mel` batch size")

    if bool(torch.any(time < -1e-6).item()) or bool(torch.any(time > target_time + 1e-6).item()):
        raise ValueError(
            f"NFT training time must stay within [0, target_time={target_time}], got "
            f"[{float(time.min().item()):.6f}, {float(time.max().item()):.6f}]."
        )

    if target_time == 0.5 and bool(torch.any(time >= target_time).item()):
        raise ValueError("Midpoint NFT time reweighting requires training time < 0.5.")

    t = time[:, None, None]
    progress = t / target_time
    x_t = (1.0 - progress) * noise + progress * x0_target

    r = build_advantage_weights(advantages, mode=adv_weight_mode, adv_clip_max=adv_clip_max)
    r = r.to(device=x_t.device, dtype=x_t.dtype)
    one_minus_r = 1.0 - r

    batch_size = x_t.shape[0]
    text_batch = [" "] * batch_size

    from flow_grpo.speech_diagnostics import diagnostic_stage
    diagnostic_stage("forward.current.begin")
    flow_cur = current.predict_flow(
        x=x_t,
        cond=cond,
        text=text_batch,
        time=time,
        drop_audio_cond=False,
        drop_text=True,
    )
    diagnostic_stage("forward.current.end")
    with torch.no_grad():
        diagnostic_stage("forward.old.begin")
        flow_old = old.predict_flow(
            x=x_t,
            cond=cond,
            text=text_batch,
            time=time,
            drop_audio_cond=False,
            drop_text=True,
        )
        diagnostic_stage("forward.old.end")
        diagnostic_stage("forward.ref.begin")
        flow_ref = ref.predict_flow(
            x=x_t,
            cond=cond,
            text=text_batch,
            time=time,
            drop_audio_cond=False,
            drop_text=True,
        )

        diagnostic_stage("forward.ref.end")

    positive_flow = beta_mix * flow_cur + (1.0 - beta_mix) * flow_old
    negative_flow = (1.0 + beta_mix) * flow_old - beta_mix * flow_cur

    x0_pos = reconstruct_x0_from_flow(x_t, positive_flow, time, target_time=target_time)
    x0_neg = reconstruct_x0_from_flow(x_t, negative_flow, time, target_time=target_time)

    per_sample_positive_loss, weight_factor_pos = compute_adaptive_reconstruction_loss(
        x0_prediction=x0_pos,
        x0_target=x0_target,
    )
    per_sample_negative_loss, weight_factor_neg = compute_adaptive_reconstruction_loss(
        x0_prediction=x0_neg,
        x0_target=x0_target,
    )

    if target_time == 0.5:
        time_reweight = (1.0 - time) / (target_time - time)
        per_sample_positive_loss = per_sample_positive_loss * time_reweight
        per_sample_negative_loss = per_sample_negative_loss * time_reweight

    per_sample_policy = (
        (r * per_sample_positive_loss + one_minus_r * per_sample_negative_loss)
        / beta_mix
        * adv_clip_max
    )

    if train_mask is None:
        sample_weight = torch.ones_like(per_sample_policy)
    else:
        sample_weight = _to_tensor(train_mask, device=x_t.device, dtype=per_sample_policy.dtype).reshape(-1)
        if sample_weight.shape[0] != per_sample_policy.shape[0]:
            raise ValueError("`train_mask` batch size must match `generated_mel` batch size")
        sample_weight = (sample_weight > 0).to(dtype=per_sample_policy.dtype)
    kept_sample_count = sample_weight.sum()
    train_mask_normalization = str(train_mask_normalization).strip().lower()
    if train_mask_normalization == "kept":
        sample_denom = torch.clamp(kept_sample_count, min=1.0)
    elif train_mask_normalization == "all":
        sample_denom = torch.as_tensor(
            float(per_sample_policy.shape[0]),
            device=x_t.device,
            dtype=per_sample_policy.dtype,
        ).clamp(min=1.0)
    else:
        raise ValueError("`train_mask_normalization` must be either 'kept' or 'all'.")
    filtered_sample_count = torch.as_tensor(
        float(per_sample_policy.shape[0]), device=x_t.device, dtype=per_sample_policy.dtype
    ) - kept_sample_count

    loss_policy = (per_sample_policy * sample_weight).sum() / sample_denom

    per_sample_kl = _mean_over_nonbatch((flow_cur - flow_ref) ** 2)
    loss_kl = (per_sample_kl * sample_weight).sum() / sample_denom

    loss_total = loss_policy + float(kl_coef) * loss_kl

    per_sample_old_deviate = _mean_over_nonbatch((flow_cur - flow_old) ** 2)
    old_deviate = (per_sample_old_deviate * sample_weight).sum() / sample_denom

    return {
        "loss_total": loss_total,
        "loss_rl": loss_policy,
        "loss_policy": loss_policy,
        "loss_kl": loss_kl,
        "kl_coef": torch.as_tensor(float(kl_coef), device=loss_total.device, dtype=loss_total.dtype),
        "per_sample_rl": per_sample_policy,
        "per_sample_policy": per_sample_policy,
        "per_sample_kl": per_sample_kl,
        "per_sample_positive_loss": per_sample_positive_loss,
        "per_sample_negative_loss": per_sample_negative_loss,
        "adv_weight": r,
        "train_mask": sample_weight,
        "kept_sample_count": kept_sample_count.detach(),
        "filtered_sample_count": filtered_sample_count.detach(),
        "train_mask_normalization_denom": sample_denom.detach(),
        "old_deviate": old_deviate,
        "per_sample_old_deviate": per_sample_old_deviate,
        "time": time,
        "noise": noise,
        "x_t": x_t,
        "x0_target": x0_target,
        "positive_flow": positive_flow,
        "negative_flow": negative_flow,
        "weight_factor_pos": weight_factor_pos,
        "weight_factor_neg": weight_factor_neg,
        "target_time": torch.as_tensor(target_time, device=x_t.device, dtype=x_t.dtype),
    }


def compute_multi_reward_pure_nft_terms(
    current_model,
    old_model,
    ref_model,
    *,
    noisy_mel: torch.Tensor,
    generated_mel: torch.Tensor,
    advantages_by_branch: dict[str, Any],
    branch_weights: dict[str, float],
    train_mask: torch.Tensor | None = None,
    beta_mix: float,
    adv_clip_max: float,
    kl_coef: float,
    time: torch.Tensor,
    noise: torch.Tensor | None = None,
    cond_type: str = "wotext",
    adv_weight_mode: str = "shifted_clip",
    target_time: float = 1.0,
) -> dict[str, Any]:
    """Compute several reward-branch NFT losses from one current/old/ref forward pass."""
    branch_names = list(advantages_by_branch)
    if not branch_names:
        raise ValueError("`advantages_by_branch` must not be empty.")
    missing_weights = [name for name in branch_names if name not in branch_weights]
    if missing_weights:
        raise KeyError(f"Missing loss weights for reward branches: {missing_weights}")
    extra_weights = [name for name in branch_weights if name not in advantages_by_branch]
    if extra_weights:
        raise KeyError(f"Missing advantages for reward branches: {extra_weights}")
    for branch_name in branch_names:
        if float(branch_weights[branch_name]) < 0.0:
            raise ValueError(f"Reward branch {branch_name!r} loss weight must be non-negative.")

    # One regular NFT call prepares the batch and computes all model-dependent terms.
    # The selected first-branch advantage only affects its cheap final per-sample reduction.
    shared = compute_pure_nft_terms(
        current_model=current_model,
        old_model=old_model,
        ref_model=ref_model,
        noisy_mel=noisy_mel,
        generated_mel=generated_mel,
        advantages=advantages_by_branch[branch_names[0]],
        train_mask=train_mask,
        beta_mix=beta_mix,
        adv_clip_max=adv_clip_max,
        kl_coef=kl_coef,
        time=time,
        noise=noise,
        cond_type=cond_type,
        adv_weight_mode=adv_weight_mode,
        target_time=target_time,
    )

    sample_weight = shared["train_mask"]
    sample_denom = torch.clamp(sample_weight.sum(), min=1.0)
    per_sample_positive_loss = shared["per_sample_positive_loss"]
    per_sample_negative_loss = shared["per_sample_negative_loss"]

    branch_terms: dict[str, dict[str, torch.Tensor]] = {}
    weighted_loss_rl = torch.zeros_like(shared["loss_rl"])
    weighted_per_sample_policy = torch.zeros_like(shared["per_sample_policy"])
    weighted_advantage_weight = torch.zeros_like(shared["adv_weight"])
    total_branch_weight = 0.0

    for branch_name in branch_names:
        branch_advantage_weight = build_advantage_weights(
            advantages_by_branch[branch_name],
            mode=adv_weight_mode,
            adv_clip_max=adv_clip_max,
        ).to(device=per_sample_positive_loss.device, dtype=per_sample_positive_loss.dtype)
        if branch_advantage_weight.shape[0] != per_sample_positive_loss.shape[0]:
            raise ValueError(
                f"Reward branch {branch_name!r} advantage batch size must match `generated_mel` batch size"
            )
        per_sample_policy = (
            (
                branch_advantage_weight * per_sample_positive_loss
                + (1.0 - branch_advantage_weight) * per_sample_negative_loss
            )
            / float(beta_mix)
            * float(adv_clip_max)
        )
        loss_rl = (per_sample_policy * sample_weight).sum() / sample_denom
        loss_total = loss_rl + shared["kl_coef"] * shared["loss_kl"]
        branch_terms[branch_name] = {
            "loss_total": loss_total,
            "loss_rl": loss_rl,
            "loss_policy": loss_rl,
            "loss_kl": shared["loss_kl"],
            "kl_coef": shared["kl_coef"],
            "per_sample_rl": per_sample_policy,
            "per_sample_policy": per_sample_policy,
            "adv_weight": branch_advantage_weight,
        }

        branch_weight = float(branch_weights[branch_name])
        weighted_loss_rl = weighted_loss_rl + branch_weight * loss_rl
        weighted_per_sample_policy = weighted_per_sample_policy + branch_weight * per_sample_policy
        weighted_advantage_weight = weighted_advantage_weight + branch_weight * branch_advantage_weight
        total_branch_weight += branch_weight

    weighted_loss_kl = shared["loss_kl"] * total_branch_weight
    weighted_loss_total = weighted_loss_rl + shared["kl_coef"] * weighted_loss_kl
    result = dict(shared)
    result.update(
        {
            "loss_total": weighted_loss_total,
            "loss_rl": weighted_loss_rl,
            "loss_policy": weighted_loss_rl,
            "loss_kl": weighted_loss_kl,
            "per_sample_rl": weighted_per_sample_policy,
            "per_sample_policy": weighted_per_sample_policy,
            "adv_weight": weighted_advantage_weight,
            "branch_weight_sum": torch.as_tensor(
                total_branch_weight,
                device=weighted_loss_total.device,
                dtype=weighted_loss_total.dtype,
            ),
            "branch_terms": branch_terms,
        }
    )
    return result
