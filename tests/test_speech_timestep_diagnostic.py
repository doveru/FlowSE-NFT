from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from flow_grpo.speech_nft_core import RolloutBatch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_speech_timestep_rewards.py"
_SPEC = spec_from_file_location("diagnose_speech_timestep_rewards_module", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Unable to load diagnose_speech_timestep_rewards module for tests.")
_MODULE = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

RewardMetricSpec = _MODULE.RewardMetricSpec
add_within_reward_sensitivity_statistics = _MODULE.add_within_reward_sensitivity_statistics
add_trajectory_information_gain_ranks = _MODULE.add_trajectory_information_gain_ranks
build_probe_time_grid = _MODULE.build_probe_time_grid
compute_gradient_snapshot = _MODULE.compute_gradient_snapshot
compute_metric_advantages = _MODULE.compute_metric_advantages
compute_trajectory_css_rows = _MODULE.compute_trajectory_css_rows
collect_rollouts = _MODULE.collect_rollouts
configure_diagnostic_config = _MODULE.configure_diagnostic_config
make_result_row = _MODULE.make_result_row
parse_metric_specs = _MODULE.parse_metric_specs
parse_timestep_window = _MODULE.parse_timestep_window
prepare_diagnostic_checkpoint_config = _MODULE.prepare_diagnostic_checkpoint_config
resolve_metric_specs_for_config = _MODULE.resolve_metric_specs_for_config
slice_source_batch = _MODULE.slice_source_batch


def test_diagnostic_cli_requires_explicit_config(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["diagnose", "--checkpoint", "example.pt"])
    with pytest.raises(SystemExit) as error:
        _MODULE.parse_args()
    assert error.value.code == 2


def test_diagnostic_cli_accepts_explicit_config(monkeypatch):
    name = "config/nft.py:speech_wotext_dnsmos_early16_train32_rollout32_baseline"
    monkeypatch.setattr(sys, "argv", ["diagnose", "--checkpoint", "example.pt", "--config", name])
    assert _MODULE.parse_args().config == name


def test_parse_metric_specs_maps_supported_metrics_to_raw_keys():
    specs = parse_metric_specs("dnsmos,speaker_similarity,speechbertscore,nisqa,dnsmos")

    assert [(spec.name, spec.metric_key) for spec in specs] == [
        ("dnsmos", "dnsmos_avg"),
        ("speaker_similarity", "speaker_similarity"),
        ("speechbertscore", "speechbertscore"),
        ("nisqa", "nisqa_mos"),
    ]


def test_metric_specs_follow_selected_config_primary_keys():
    config = SimpleNamespace(
        speech=SimpleNamespace(
            reward=SimpleNamespace(primary_keys={"dnsmos": "dnsmos_ovrl"}),
        )
    )

    specs = resolve_metric_specs_for_config(parse_metric_specs("dnsmos,speechbertscore"), config)

    assert [(spec.name, spec.metric_key) for spec in specs] == [
        ("dnsmos", "dnsmos_ovrl"),
        ("speechbertscore", "speechbertscore"),
    ]


def test_configure_diagnostic_config_overrides_reward_registry_and_limits():
    config = SimpleNamespace(
        speech=SimpleNamespace(
            reward=SimpleNamespace(speechbert_model_path=None),
            rollout=SimpleNamespace(collect_trajectory_similarity=False, num_candidates=24),
            train=SimpleNamespace(rollout_batches_per_epoch=8, reward_branches=["old"]),
            eval=SimpleNamespace(limit_batches=0, num_candidates=4),
        )
    )
    specs = parse_metric_specs("speechbertscore,nisqa")

    configure_diagnostic_config(
        config,
        specs,
        limit_train_batches=2,
        limit_val_batches=3,
        num_candidates=7,
    )

    assert config.speech.reward.registry == ["speechbertscore", "nisqa"]
    assert config.speech.reward.weights == {"speechbertscore": 1.0, "nisqa": 1.0}
    assert config.speech.reward.primary_keys == {
        "speechbertscore": "speechbertscore",
        "nisqa": "nisqa_mos",
    }
    assert config.speech.reward.speechbert_model_path
    assert config.speech.train.rollout_batches_per_epoch == 2
    assert config.speech.train.reward_branches == []
    assert config.speech.eval.limit_batches == 3
    assert config.speech.eval.num_candidates == 1
    assert config.speech.rollout.num_candidates == 7
    assert config.speech.rollout.collect_trajectory_similarity is True


def test_slice_source_batch_slices_all_batch_aligned_fields():
    batch = {
        "utt_id": ["a", "b"],
        "source_utt_id": ["sa", "sb"],
        "noisy_wav": torch.arange(12).reshape(2, 2, 3),
        "clean_num_samples": torch.tensor([3, 3]),
        "metadata": "shared",
    }

    sliced = slice_source_batch(batch, 1)

    assert sliced["utt_id"] == ["a"]
    assert sliced["source_utt_id"] == ["sa"]
    assert sliced["noisy_wav"].shape == (1, 2, 3)
    assert sliced["clean_num_samples"].tolist() == [3]
    assert sliced["metadata"] == "shared"


def test_collect_rollouts_limits_exact_source_count_across_batches():
    class FakeBackend:
        def __init__(self):
            self.batch_sizes = []

        @staticmethod
        def _rollout_seed_offset(*, outer_epoch, batch_index):
            return outer_epoch + batch_index

        def rollout(self, *, batch, **kwargs):
            del kwargs
            batch_size = len(batch["utt_id"])
            self.batch_sizes.append(batch_size)
            return SimpleNamespace(candidate_ids=["candidate"] * (batch_size * 3))

    data_loader = [
        {"utt_id": ["a", "b"], "value": torch.tensor([1, 2])},
        {"utt_id": ["c", "d"], "value": torch.tensor([3, 4])},
    ]
    backend = FakeBackend()
    config = SimpleNamespace(
        speech=SimpleNamespace(
            train=SimpleNamespace(rollout_batches_per_epoch=1),
        )
    )

    batches = collect_rollouts(
        backend=backend,
        old_policy=object(),
        data_loader=data_loader,
        config=config,
        legacy_batch_limit=1,
        limit_source_samples=3,
    )

    assert len(batches) == 2
    assert backend.batch_sizes == [2, 1]
    assert sum(len(batch.candidate_ids) for batch in batches) == 9


def test_make_result_row_has_required_schema_and_delta():
    row = make_result_row(
        spec=RewardMetricSpec(name="dnsmos", metric_key="dnsmos_avg"),
        timestep_index=7,
        timestep_value=0.25,
        grad_norm=1.5,
        alignment_to_all=0.75,
        one_step_before=3.8,
        one_step_after=3.9,
        loss_total=0.1,
        loss_rl=0.08,
        loss_kl=0.02,
    )

    assert set(row) == {
        "metric",
        "metric_key",
        "timestep_index",
        "timestep_value",
        "grad_norm",
        "alignment_to_all",
        "one_step_before",
        "one_step_after",
        "one_step_delta",
        "loss_total",
        "loss_rl",
        "loss_kl",
        "cross_reward_delta",
    }
    assert row["metric"] == "dnsmos"
    assert row["metric_key"] == "dnsmos_avg"
    assert row["one_step_delta"] == row["one_step_after"] - row["one_step_before"]
    assert row["cross_reward_delta"] == {}


def test_compute_metric_advantages_uses_requested_raw_metric():
    rollout_batch = RolloutBatch(
        x0_target=torch.zeros(4, 2, 2),
        condition=torch.zeros(4, 2, 2),
        timesteps=torch.zeros(4, 3),
        candidate_ids=["a0", "a1", "b0", "b1"],
        group_ids=["a", "a", "b", "b"],
        reward_dict={
            "raw": {
                "dnsmos_avg": torch.tensor([1.0, 3.0, 2.0, 4.0]),
            }
        },
        reward_avg=torch.zeros(4),
        noisy_mel=torch.zeros(4, 2, 2),
        generated_mel=torch.zeros(4, 2, 2),
        utt_ids=["a0", "a1", "b0", "b1"],
        source_utt_ids=["a", "a", "b", "b"],
    )

    advantages = compute_metric_advantages(
        [rollout_batch],
        RewardMetricSpec(name="dnsmos", metric_key="dnsmos_avg"),
        global_std=False,
    )

    assert len(advantages) == 1
    assert advantages[0].shape == (4,)
    assert torch.isclose(advantages[0][:2].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(advantages[0][2:].mean(), torch.tensor(0.0), atol=1e-6)
    assert rollout_batch.train_mask is not None


def test_prepare_diagnostic_checkpoint_keeps_lora_enabled_for_base_checkpoint():
    config = SimpleNamespace(
        speech=SimpleNamespace(
            model=SimpleNamespace(
                init_checkpoint="old.pt",
                lora=SimpleNamespace(enabled=True),
            )
        )
    )

    info = prepare_diagnostic_checkpoint_config(
        config,
        {
            "path": "/tmp/pretrain.pt.tar",
            "kind": "base_model_checkpoint",
        },
    )

    assert config.speech.model.init_checkpoint == "/tmp/pretrain.pt.tar"
    assert config.speech.model.lora.enabled is True
    assert info["used_as_init_checkpoint"] is True
    assert info["lora_kept_enabled_for_probe"] is True


def test_gradient_snapshot_reuses_shared_noise_for_every_timestep():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    recorded_noises = []

    class FakeBackend:
        def compute_rl_kl_loss(self, **kwargs):
            recorded_noises.append(kwargs["noise"].clone())
            sampled_time = torch.as_tensor(kwargs["sampled_time"], dtype=torch.float32).mean()
            loss = parameter * (sampled_time + 1.0)
            return {
                "loss_total": loss,
                "loss_rl": loss,
                "loss_kl": loss * 0.0,
            }

    rollout_batch = RolloutBatch(
        x0_target=torch.zeros(2, 2, 2),
        condition=torch.zeros(2, 2, 2),
        timesteps=torch.tensor([[0.25, 0.75], [0.25, 0.75]]),
        candidate_ids=["a0", "a1"],
        group_ids=["a", "a"],
        reward_dict={"raw": {"dnsmos_avg": torch.tensor([1.0, 2.0])}},
        reward_avg=torch.zeros(2),
        noisy_mel=torch.zeros(2, 2, 2),
        generated_mel=torch.zeros(2, 2, 2),
        utt_ids=["a0", "a1"],
        source_utt_ids=["a", "a"],
    )
    shared_noise = torch.randn(2, 2, 2)

    compute_gradient_snapshot(
        backend=FakeBackend(),
        current_model=object(),
        old_model=object(),
        ref_model=object(),
        rollout_batches=[rollout_batch],
        advantages_by_batch=[torch.tensor([-1.0, 1.0])],
        timestep_values=[0.25, 0.75],
        config=SimpleNamespace(),
        named_trainable_parameters=[("weight", parameter)],
        shared_noises_by_batch=[shared_noise],
    )

    assert len(recorded_noises) == 2
    assert all(torch.equal(noise, shared_noise) for noise in recorded_noises)


def test_probe_grid_uses_independent_grid_and_remaining_window():
    rollout_batch = RolloutBatch(
        x0_target=torch.zeros(1, 2, 2),
        condition=torch.zeros(1, 2, 2),
        timesteps=torch.tensor([[0.0, 0.5, 1.0]]),
        candidate_ids=["a0"],
        group_ids=["a"],
        reward_dict={"raw": {"dnsmos_avg": torch.tensor([1.0])}},
        reward_avg=torch.zeros(1),
        noisy_mel=torch.zeros(1, 2, 2),
        generated_mel=torch.zeros(1, 2, 2),
        utt_ids=["a0"],
        source_utt_ids=["a"],
        nft_target_time=1.0,
    )

    full_grid, probe_points, target_time = build_probe_time_grid(
        [rollout_batch],
        configured_grid_steps=5,
        requested_grid_steps=0,
        timestep_window=parse_timestep_window("0.25,1.0"),
    )

    assert full_grid == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert probe_points == [(1, 0.25), (2, 0.5), (3, 0.75), (4, 1.0)]
    assert target_time == 1.0


def test_sensitivity_statistics_are_normalized_and_ranked_per_reward():
    rows = [
        {"metric": "dnsmos", "one_step_delta": 0.1},
        {"metric": "dnsmos", "one_step_delta": 0.3},
        {"metric": "speaker_similarity", "one_step_delta": -0.01},
        {"metric": "speaker_similarity", "one_step_delta": 0.01},
    ]

    add_within_reward_sensitivity_statistics(rows)

    assert [row["sensitivity_rank"] for row in rows] == [2, 1, 2, 1]
    assert [row["positive_sensitivity"] for row in rows] == [True, True, False, True]
    assert abs(rows[0]["one_step_delta_zscore"] + 1.0) < 1e-12
    assert abs(rows[1]["one_step_delta_zscore"] - 1.0) < 1e-12
    assert abs(rows[2]["one_step_delta_zscore"] + 1.0) < 1e-12
    assert abs(rows[3]["one_step_delta_zscore"] - 1.0) < 1e-12


def test_information_gain_rank_filters_near_zero_numerical_changes():
    rows = [
        {
            "reward_sensitivity": {
                "speechbertscore": {
                    "mean_information_gain": 1e-3,
                    "absolute_advantage_correlation": 0.2,
                }
            }
        },
        {
            "reward_sensitivity": {
                "speechbertscore": {
                    "mean_information_gain": 1e-8,
                    "absolute_advantage_correlation": 0.99,
                }
            }
        },
    ]

    add_trajectory_information_gain_ranks(
        rows,
        [RewardMetricSpec(name="speechbertscore", metric_key="speechbertscore")],
        relative_threshold=1e-3,
        absolute_threshold=1e-7,
    )

    reliable = rows[0]["reward_sensitivity"]["speechbertscore"]
    numerical_noise = rows[1]["reward_sensitivity"]["speechbertscore"]
    assert reliable["information_gain_rank"] == 1
    assert reliable["sensitivity_rank"] == 1
    assert reliable["information_gain_above_threshold"] is True
    assert numerical_noise["information_gain_rank"] is None
    assert numerical_noise["sensitivity_rank"] is None
    assert numerical_noise["information_gain_above_threshold"] is False


def test_trajectory_css_rows_correlate_information_gain_with_reward_advantage():
    rollout_batch = RolloutBatch(
        x0_target=torch.zeros(4, 2, 2),
        condition=torch.zeros(4, 2, 2),
        timesteps=torch.tensor([[0.0, 0.5, 1.0]]).repeat(4, 1),
        candidate_ids=["a0", "a1", "a2", "a3"],
        group_ids=["a", "a", "a", "a"],
        reward_dict={"raw": {"dnsmos_avg": torch.tensor([0.0, 1.0, 2.0, 3.0])}},
        reward_avg=torch.zeros(4),
        noisy_mel=torch.zeros(4, 2, 2),
        generated_mel=torch.zeros(4, 2, 2),
        utt_ids=["a0", "a1", "a2", "a3"],
        source_utt_ids=["a", "a", "a", "a"],
        trajectory_final_similarity={
            "dnsmos": torch.tensor(
                [
                    [0.0, 0.5, 1.0],
                    [0.1, 0.5, 1.0],
                    [0.2, 0.5, 1.0],
                    [0.3, 0.5, 1.0],
                ]
            )
        },
        trajectory_information_gain={
            "dnsmos": torch.tensor(
                [
                    [0.0, 0.25],
                    [1.0, 0.25],
                    [2.0, 0.25],
                    [3.0, 0.25],
                ]
            )
        },
        trajectory_embedding_kinds={"dnsmos": "dnsmos_ovrl_sig_bak_output_embedding"},
    )

    time_grid, rows = compute_trajectory_css_rows(
        [rollout_batch],
        [RewardMetricSpec(name="dnsmos", metric_key="dnsmos_avg")],
        timestep_window=(0.0, 1.0),
        global_advantage_std=False,
    )

    assert time_grid == [0.0, 0.5, 1.0]
    assert len(rows) == 2
    first = rows[0]["reward_sensitivity"]["dnsmos"]
    second = rows[1]["reward_sensitivity"]["dnsmos"]
    assert first["mean_final_similarity"] == pytest.approx(0.15)
    assert first["mean_information_gain"] == pytest.approx(1.5)
    assert first["information_gain_rank"] == 1
    assert second["information_gain_rank"] == 2
    assert abs(first["information_gain_advantage_correlation"] - 1.0) < 1e-6
    assert second["information_gain_advantage_correlation"] == 0.0
    assert first["sensitivity_rank"] == 1
