from __future__ import annotations

from types import SimpleNamespace

import pytest

ml_collections = pytest.importorskip("ml_collections")
del ml_collections

import config.nft as nft_config
from flow_grpo.speech_backend_adapter import SpeechBackendAdapter


def test_speech_lora_defaults_exist_and_are_valid():
    cfg = nft_config.speech_wotext_pure_nft()
    lora = cfg.speech.model.lora

    assert bool(lora.enabled) is True
    assert str(lora.strategy) in {"multi_model", "shared_adapter"}
    assert int(lora.r) > 0
    assert int(lora.alpha) > 0
    assert float(lora.dropout) >= 0.0
    assert str(lora.bias) == "none"
    assert str(lora.init_lora_weights) in {"gaussian", "loftq", "eva", "olora", "pissa"}
    assert isinstance(lora.target_modules, list)
    assert set(lora.target_modules) >= {"to_q", "to_k", "to_v", "to_out.0"}
    assert bool(lora.save_adapter_only) is True


def test_speech_multi_reward_config_uses_uniform_reward_branches(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "6")
    cfg = nft_config.speech_wotext_multi_reward()
    reward = cfg.speech.reward
    train = cfg.speech.train
    branches = SpeechBackendAdapter._reward_branches_from_config(train)

    assert cfg.run_name == (
        "speech_nft_wotext_multi_reward_gd2po_tau00_early025_train32_"
        "groups48_candidates24_clip3_equal111"
    )
    assert cfg.save_dir == (
        "logs/nft/speech/wotext_multi_reward_gd2po_tau00_early025_train32_"
        "groups48_candidates24_clip3_equal111"
    )
    assert int(cfg.speech.layout.num_groups) == 48
    assert int(cfg.speech.rollout.num_candidates) == 24
    assert int(cfg.speech.layout.num_groups) * int(cfg.speech.rollout.num_candidates) == 1152
    assert int(cfg.speech.data.train_batch_size) == 1
    assert int(cfg.speech.train.rollout_batches_per_epoch) == 8
    assert int(cfg.speech.train.gradient_accumulation_steps) == 8
    assert list(reward.registry) == [
        "dnsmos",
        "speaker_similarity",
        "speechbertscore",
    ]
    assert dict(reward.weights) == {
        "dnsmos": 1.0,
        "speaker_similarity": 1.0,
        "speechbertscore": 1.0,
    }
    assert reward.normalization == "batch_std"
    assert reward.primary_keys["dnsmos"] == "dnsmos_ovrl"
    assert reward.speaker_model_type == "eres2net"
    assert str(reward.speaker_model_path).endswith("/eres2net")
    assert str(reward.speaker_code_path).endswith("/3D-Speaker")
    assert train.multi_reward_update_mode == "gd2po"
    assert bool(train.reward_conflict_filter.enabled) is True
    assert float(train.reward_conflict_filter.tau) == pytest.approx(0.0)
    assert bool(train.reward_conflict_filter.post_normalize) is True
    assert str(train.reward_conflict_filter.post_normalize_mode) == "masked_whiten"
    assert bool(train.reward_conflict_filter.apply_group_keep_ratio) is True
    assert float(train.reward_conflict_filter.post_normalize_eps) == pytest.approx(1e-4)
    assert float(cfg.speech.loss.adv_clip_max) == 3.0
    assert [branch["name"] for branch in branches] == ["dnsmos", "speaker_similarity", "speechbertscore"]
    assert [branch["metric_key"] for branch in branches] == ["dnsmos_ovrl", "speaker_similarity", "speechbertscore"]
    assert all(branch["score_section"] == "raw" for branch in branches)
    assert [float(branch["loss_weight"]) for branch in branches] == pytest.approx([1.0, 1.0, 1.0])
    assert list(train.timestep_window) == [0.0, 0.25]
    assert int(train.timesteps_per_batch) == 32
    assert int(train.timestep_grid_steps) == 128
    assert all(int(branch["timesteps_per_batch"]) == 32 for branch in branches)
    assert all(int(branch["timestep_grid_steps"]) == 128 for branch in branches)
    assert all(float(branch["timestep_fraction"]) == 1.0 for branch in branches)
    assert all(list(branch["timestep_window"]) == [0.0, 0.25] for branch in branches)


@pytest.mark.parametrize(
    ("factory", "expected_mode", "expected_enabled", "run_suffix"),
    [
        (nft_config.speech_wotext_multi_reward_no_post_norm, "none", False, "post_none"),
    ],
)
def test_speech_multi_reward_post_normalization_variants(
    factory,
    expected_mode,
    expected_enabled,
    run_suffix,
):
    cfg = factory()
    conflict_filter = cfg.speech.train.reward_conflict_filter

    assert str(conflict_filter.post_normalize_mode) == expected_mode
    assert bool(conflict_filter.post_normalize) is expected_enabled
    assert str(cfg.run_name).endswith(run_suffix)
    assert str(cfg.save_dir).endswith(run_suffix)


def test_speech_multi_reward_no_post_norm_uses_clip5_and_dnsmos_weight211():
    cfg = nft_config.speech_wotext_multi_reward_no_post_norm()
    branches = SpeechBackendAdapter._reward_branches_from_config(cfg.speech.train)

    assert cfg.run_name == (
        "speech_nft_wotext_multi_reward_gd2po_tau00_early025_train32_"
        "groups48_candidates24_clip5_weight211_post_none"
    )
    assert cfg.save_dir == (
        "logs/nft/speech/wotext_multi_reward_gd2po_tau00_early025_train32_"
        "groups48_candidates24_clip5_weight211_post_none"
    )
    assert float(cfg.speech.loss.adv_clip_max) == 5.0
    assert float(cfg.speech.train.reward_conflict_filter.tau) == pytest.approx(0.0)
    assert str(cfg.speech.train.reward_conflict_filter.post_normalize_mode) == "none"
    assert dict(cfg.speech.reward.weights) == {
        "dnsmos": 2.0,
        "speaker_similarity": 1.0,
        "speechbertscore": 1.0,
    }
    assert [float(branch["loss_weight"]) for branch in branches] == pytest.approx([2.0, 1.0, 1.0])


def test_speech_multi_reward_rms_tau03_uses_clip5_and_dnsmos_weight211():
    cfg = nft_config.speech_wotext_multi_reward_rms_scale_tau03_weight211()
    branches = SpeechBackendAdapter._reward_branches_from_config(cfg.speech.train)
    assert list(cfg.speech.train.timestep_window) == [0.0, 0.25]
    assert int(cfg.speech.train.timesteps_per_batch) == 32
    assert int(cfg.speech.train.timestep_grid_steps) == 128
    assert all(list(branch["timestep_window"]) == [0.0, 0.25] for branch in branches)
    assert all(int(branch["timesteps_per_batch"]) == 32 for branch in branches)
    assert all(int(branch["timestep_grid_steps"]) == 128 for branch in branches)

    assert cfg.run_name == (
        "speech_nft_wotext_multi_reward_gd2po_tau03_early025_train32_"
        "groups48_candidates24_clip5_weight211_post_rms_scale"
    )
    assert cfg.save_dir == (
        "logs/nft/speech/wotext_multi_reward_gd2po_tau03_early025_train32_"
        "groups48_candidates24_clip5_weight211_post_rms_scale"
    )
    assert float(cfg.speech.loss.adv_clip_max) == 5.0
    assert float(cfg.speech.train.reward_conflict_filter.tau) == pytest.approx(0.3)
    assert bool(cfg.speech.train.reward_conflict_filter.post_normalize) is True
    assert str(cfg.speech.train.reward_conflict_filter.post_normalize_mode) == "rms_scale"
    assert dict(cfg.speech.reward.weights) == {
        "dnsmos": 2.0,
        "speaker_similarity": 1.0,
        "speechbertscore": 1.0,
    }
    assert [float(branch["loss_weight"]) for branch in branches] == pytest.approx([2.0, 1.0, 1.0])


def test_speech_multi_reward_early32_updates_each_uniform_branch_window():
    cfg = nft_config.speech_wotext_multi_reward_early32_train64()
    train = cfg.speech.train
    branches = SpeechBackendAdapter._reward_branches_from_config(train)

    assert cfg.run_name == "speech_nft_wotext_multi_reward_gd2po_tau00_early32_train64_clip3_equal111"
    assert cfg.save_dir == "logs/nft/speech/wotext_multi_reward_gd2po_tau00_early32_train64_clip3_equal111"
    assert int(cfg.speech.rollout.steps) == 32
    assert train.multi_reward_update_mode == "gd2po"
    assert list(train.timestep_window) == [0.0, 0.5]
    assert int(train.timesteps_per_batch) == 32
    assert int(train.timestep_grid_steps) == 64
    assert len(branches) == 3
    assert all(list(branch["timestep_window"]) == [0.0, 0.5] for branch in branches)
    assert all(int(branch["timesteps_per_batch"]) == 32 for branch in branches)
    assert all(int(branch["timestep_grid_steps"]) == 64 for branch in branches)
    assert [float(branch["loss_weight"]) for branch in branches] == pytest.approx([1.0, 1.0, 1.0])


def test_speech_dnsmos_early32_train64_uses_independent_training_grid():
    cfg = nft_config.speech_wotext_pure_nft_early32_train64()
    reward = cfg.speech.reward
    train = cfg.speech.train

    assert cfg.run_name == "speech_nft_wotext_pure_early32_train64"
    assert cfg.save_dir == "logs/nft/speech/wotext_pure_early32_train64"
    assert list(reward.registry) == ["dnsmos"]
    assert dict(reward.weights) == {"dnsmos": 1.0}
    assert int(cfg.speech.rollout.steps) == 32
    assert list(train.timestep_window) == [0.0, 0.5]
    assert int(train.timesteps_per_batch) == 32
    assert int(train.timestep_grid_steps) == 64


def test_speech_dnsmos_rollout_baseline_keeps_original_uniform_32_steps():
    cfg = nft_config.speech_wotext_dnsmos_early16_train32_rollout32_baseline()
    rollout = cfg.speech.rollout
    train = cfg.speech.train

    assert list(cfg.speech.reward.registry) == ["dnsmos"]
    assert int(rollout.steps) == 32
    assert list(rollout.time_grid) == []
    assert list(rollout.deterministic_mask) == []
    assert bool(cfg.speech.eval.use_rollout_schedule) is True
    assert int(cfg.speech.eval.steps) == 32
    assert list(train.timestep_window) == [0.0, 0.5]
    assert int(train.timesteps_per_batch) == 16
    assert int(train.timestep_grid_steps) == 32


def test_default_validation_mask_remains_all_ode():
    cfg = nft_config.speech_wotext_dnsmos_early16_train32_rollout32_baseline()
    backend = object.__new__(SpeechBackendAdapter)
    backend.speech_conf = cfg.speech
    backend.eval_rollout_engine = SimpleNamespace(steps=32)

    assert backend._eval_deterministic_mask() == [True] * 32


@pytest.mark.parametrize(
    "config_name,expected_window",
    [
        ("speech_wotext_speechbert_all16", [0.0, 1.0]),
        ("speech_wotext_speechbert_early16", [0.0, 0.5]),
        ("speech_wotext_speechbert_late16", [0.5, 1.0]),
    ],
)
def test_speechbert_only_timestep_window_configs(config_name, expected_window):
    cfg = getattr(nft_config, config_name)()
    reward = cfg.speech.reward
    train = cfg.speech.train

    assert list(reward.registry) == ["speechbertscore"]
    assert dict(reward.weights) == {"speechbertscore": 1.0}
    assert reward.normalization == "raw_linear"
    assert reward.primary_keys["speechbertscore"] == "speechbertscore"
    assert list(train.timestep_window) == expected_window
    assert int(train.timesteps_per_batch) == 16
    assert list(train.reward_branches) == []
    assert reward.speechbert_model_path


@pytest.mark.parametrize(
    "config_name,expected_dnsmos_window",
    [
        ("speech_wotext_speechbert_base_dnsmos_all16", [0.0, 1.0]),
        ("speech_wotext_speechbert_base_dnsmos_early16", [0.0, 0.5]),
        ("speech_wotext_speechbert_base_dnsmos_late16", [0.5, 1.0]),
    ],
)
def test_speechbert_base_dnsmos_gain_branch_configs(config_name, expected_dnsmos_window):
    cfg = getattr(nft_config, config_name)()
    reward = cfg.speech.reward
    branches = list(cfg.speech.train.reward_branches)

    assert list(reward.registry) == ["speechbertscore", "dnsmos"]
    assert dict(reward.weights) == {"speechbertscore": 1.0, "dnsmos": 0.1}
    assert len(branches) == 2
    assert branches[0].name == "speechbertscore_base"
    assert branches[0].metric_key == "speechbertscore"
    assert list(branches[0].timestep_window) == [0.0, 1.0]
    assert int(branches[0].timesteps_per_batch) == 0
    assert float(branches[0].loss_weight) == 1.0
    assert branches[1].name == "dnsmos_gain"
    assert branches[1].metric_key == "dnsmos_avg"
    assert list(branches[1].timestep_window) == expected_dnsmos_window
    assert int(branches[1].timesteps_per_batch) == 16
    assert float(branches[1].loss_weight) == 0.1


@pytest.mark.parametrize(
    "config_name,expected_speechbert_window",
    [
        ("speech_wotext_dnsmos_base_speechbert_all16", [0.0, 1.0]),
        ("speech_wotext_dnsmos_base_speechbert_early16", [0.0, 0.5]),
        ("speech_wotext_dnsmos_base_speechbert_late16", [0.5, 1.0]),
    ],
)
def test_dnsmos_base_speechbert_gain_branch_configs(config_name, expected_speechbert_window):
    cfg = getattr(nft_config, config_name)()
    reward = cfg.speech.reward
    branches = list(cfg.speech.train.reward_branches)

    assert list(reward.registry) == ["dnsmos", "speechbertscore"]
    assert dict(reward.weights) == {"dnsmos": 0.25, "speechbertscore": 1.0}
    assert len(branches) == 2
    assert branches[0].name == "dnsmos_base"
    assert branches[0].metric_key == "dnsmos_avg"
    assert list(branches[0].timestep_window) == [0.0, 1.0]
    assert int(branches[0].timesteps_per_batch) == 0
    assert float(branches[0].loss_weight) == 0.25
    assert branches[1].name == "speechbertscore_gain"
    assert branches[1].metric_key == "speechbertscore"
    assert list(branches[1].timestep_window) == expected_speechbert_window
    assert int(branches[1].timesteps_per_batch) == 16
    assert float(branches[1].loss_weight) == 1.0
