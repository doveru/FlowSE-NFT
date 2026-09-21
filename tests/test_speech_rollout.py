from __future__ import annotations

import pytest
import torch

from flow_grpo.speech_flowse.rollout import (
    compute_reward_embedding_cosine,
    compute_trajectory_final_cosine,
    predict_final_state_from_flow,
)
from flow_grpo.speech_nft_core import select_nft_target_sample


def test_trajectory_final_cosine_matches_css_definition():
    trajectory = torch.tensor(
        [
            [[[0.0, 1.0]]],
            [[[1.0, 1.0]]],
            [[[1.0, 0.0]]],
        ]
    )
    final_state = torch.tensor([[[1.0, 0.0]]])

    similarity, information_gain = compute_trajectory_final_cosine(
        trajectory,
        final_state,
    )

    expected_similarity = torch.tensor([[0.0, 2**-0.5, 1.0]])
    expected_gain = torch.abs(expected_similarity[:, 1:] - expected_similarity[:, :-1])
    assert torch.allclose(similarity, expected_similarity, atol=1e-6)
    assert torch.allclose(information_gain, expected_gain, atol=1e-6)


def test_reward_embedding_cosine_is_computed_independently_per_reward():
    final_embeddings = {
        "speaker_similarity": torch.tensor([[1.0, 0.0]]),
        "speechbertscore": torch.tensor([[0.0, 1.0]]),
    }
    current_embeddings = {
        "speaker_similarity": torch.tensor([[1.0, 0.0]]),
        "speechbertscore": torch.tensor([[1.0, 0.0]]),
    }

    similarities = compute_reward_embedding_cosine(current_embeddings, final_embeddings)

    assert similarities["speaker_similarity"].item() == pytest.approx(1.0)
    assert similarities["speechbertscore"].item() == pytest.approx(0.0)


def test_predict_final_state_from_flow_uses_guided_flow_and_endpoint_distance():
    class FakeModel:
        def predict_flow(self, *, drop_audio_cond, **kwargs):
            del kwargs
            value = 1.0 if not drop_audio_cond else 0.25
            return torch.full((1, 1, 1), value)

    predicted = predict_final_state_from_flow(
        FakeModel(),
        torch.tensor([[[2.0]]]),
        torch.zeros(1, 1, 1),
        [" "],
        time_value=0.25,
        target_time=1.0,
        cfg_strength=1.0,
        drop_text=True,
    )

    # guided flow = 1 + (1 - 0.25) = 1.75; remaining distance = 0.75
    assert torch.allclose(predicted, torch.tensor([[[3.3125]]]))


def test_select_nft_target_sample_preserves_original_final_target_by_default():
    generated = torch.full((2, 1, 1), 99.0)
    trajectory = torch.zeros(3, 2, 1, 1)

    selected, target_time = select_nft_target_sample(
        generated,
        trajectory,
        torch.tensor([0.0, 0.5, 1.0]),
        target_time=None,
    )

    assert selected is generated
    assert target_time == pytest.approx(1.0)


def test_select_nft_target_sample_returns_halfway_state_without_full_trajectory_cache():
    generated = torch.full((2, 1, 1), 99.0)
    trajectory = torch.arange(18.0).reshape(18, 1, 1, 1).expand(-1, 2, -1, -1)
    time_grid = torch.tensor([index / 32.0 for index in range(17)] + [1.0])

    selected, target_time = select_nft_target_sample(
        generated,
        trajectory,
        time_grid,
        target_time=0.5,
    )

    assert selected.shape == generated.shape
    assert torch.all(selected == 16.0)
    assert target_time == pytest.approx(0.5)
    assert selected.untyped_storage().data_ptr() != trajectory.untyped_storage().data_ptr()


def test_select_nft_target_sample_requires_target_on_rollout_grid():
    generated = torch.zeros(1, 1, 1)
    trajectory = torch.zeros(3, 1, 1, 1)
    with pytest.raises(ValueError, match="must occur exactly once"):
        select_nft_target_sample(
            generated,
            trajectory,
            torch.tensor([0.0, 0.25, 1.0]),
            target_time=0.5,
        )
