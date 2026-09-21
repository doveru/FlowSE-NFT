from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
import torch

_REWARDS_PATH = Path(__file__).resolve().parents[1] / "flow_grpo" / "speech_flowse" / "rewards.py"
_SPEC = spec_from_file_location("speech_rewards_module", _REWARDS_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Unable to load speech rewards module for tests.")
_MODULE = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

SpeechRewardPipeline = _MODULE.SpeechRewardPipeline
ERes2NetSpeakerSimilarityReward = _MODULE.ERes2NetSpeakerSimilarityReward
_merge_reward_config = _MODULE._merge_reward_config


class _FakeScorer:
    def __init__(self, scores):
        self.scores = scores

    def score_batch(self, clean_paths, candidate_paths, failed_metrics):
        del clean_paths, candidate_paths, failed_metrics
        return {key: list(value) for key, value in self.scores.items()}


class _FakeEmbeddingScorer:
    def __init__(self, embeddings):
        self.embeddings = torch.as_tensor(embeddings, dtype=torch.float32)

    def extract_embedding_batch(self, wavs, sample_rate):
        del wavs, sample_rate
        return self.embeddings.clone()


def test_default_reward_config_is_single_dnsmos_triplet_mean():
    config = _merge_reward_config(None)

    assert config["registry"] == ["dnsmos"]
    assert config["weights"]["dnsmos"] == 1.0
    assert config["weights"]["speechbertscore"] == 0.0
    assert config["weights"]["speaker_similarity"] == 0.0
    assert config["normalization"] == "raw_linear"
    assert config["primary_keys"]["dnsmos"] == "dnsmos_avg"
    assert config["raw_scales"]["dnsmos"] == 1.0


def test_reward_aligned_embeddings_reuse_speech_encoders():
    pipeline = object.__new__(SpeechRewardPipeline)
    pipeline.scorers = {
        "speaker_similarity": _FakeEmbeddingScorer([[1.0, 2.0], [3.0, 4.0]]),
        "speechbertscore": _FakeEmbeddingScorer([[5.0, 6.0], [7.0, 8.0]]),
    }

    embeddings = pipeline.extract_reward_aligned_embeddings(
        [torch.zeros(16), torch.ones(16)],
        sample_rate=16000,
        metric_names=["speaker_similarity", "speechbertscore"],
    )

    assert torch.equal(embeddings["speaker_similarity"], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert torch.equal(embeddings["speechbertscore"], torch.tensor([[5.0, 6.0], [7.0, 8.0]]))


def test_dnsmos_output_coordinates_form_finite_quality_embedding(tmp_path):
    scorer = _FakeScorer(
        {
            "dnsmos_ovrl": [1.0, 5.0],
            "dnsmos_sig": [3.0, 3.0],
            "dnsmos_bak": [5.0, 1.0],
        }
    )
    scorer.metric_name = "dnsmos"
    scorer.output_keys = ("dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak")
    pipeline = object.__new__(SpeechRewardPipeline)
    pipeline.scorers = {"dnsmos": scorer}
    pipeline.tmp_root = tmp_path

    embeddings = pipeline.extract_reward_aligned_embeddings(
        [torch.zeros(16), torch.ones(16)],
        sample_rate=16000,
        metric_names=["dnsmos"],
    )["dnsmos"]

    assert embeddings.shape == (2, 6)
    assert torch.allclose(embeddings[0], torch.tensor([0.0, 0.5, 1.0, 1.0, 0.5, 0.0]))
    assert torch.allclose(embeddings[1], torch.tensor([1.0, 0.5, 0.0, 0.0, 0.5, 1.0]))


def test_raw_linear_reward_formula_uses_single_dnsmos_triplet_mean():
    pipeline = object.__new__(SpeechRewardPipeline)
    pipeline.reward_config = _merge_reward_config(
        {
            "registry": ["dnsmos"],
            "weights": {
                "dnsmos": 1.0,
            },
            "normalization": "raw_linear",
            "primary_keys": {
                "dnsmos": "dnsmos_avg",
            },
            "raw_scales": {
                "dnsmos": 1.0,
            },
        }
    )
    pipeline.active_metrics = ["dnsmos"]
    pipeline.scorers = {
        "dnsmos": _FakeScorer(
            {
                "dnsmos_ovrl": [2.0, 3.0, 4.0],
                "dnsmos_sig": [4.0, 3.0, 2.0],
                "dnsmos_bak": [3.0, 3.0, 3.0],
            }
        ),
    }

    result = pipeline._score_candidate(
        clean_dir=Path("/tmp/clean"),
        candidate_dir=Path("/tmp/candidate"),
        filenames=["a.wav", "b.wav", "c.wav"],
        failed_metrics={},
    )

    expected = np.asarray([3.0, 3.0, 3.0], dtype=np.float32)

    assert np.allclose(result["raw"]["dnsmos_avg"], expected)
    assert np.allclose(result["raw"]["reward_component_dnsmos_raw"], expected)
    assert np.allclose(result["avg"], expected, atol=1e-6)
    assert "dnsmos_avg" in result["raw"]
    assert "reward_component_dnsmos_norm" in result["norm"]
    assert "speechbertscore" not in result["raw"]
    assert "speaker_similarity" not in result["raw"]


def test_batch_std_multi_reward_formula_matches_flowse_grpo_paper():
    pipeline = object.__new__(SpeechRewardPipeline)
    pipeline.reward_config = _merge_reward_config(
        {
            "registry": [
                "dnsmos",
                "speaker_similarity",
                "speechbertscore",
            ],
            "weights": {
                "dnsmos": 0.6,
                "speaker_similarity": 1.0,
                "speechbertscore": 1.0,
            },
            "normalization": "batch_std",
            "primary_keys": {
                "dnsmos": "dnsmos_ovrl",
                "speaker_similarity": "speaker_similarity",
                "speechbertscore": "speechbertscore",
            },
        }
    )
    pipeline.active_metrics = [
        "dnsmos",
        "speaker_similarity",
        "speechbertscore",
    ]
    pipeline.scorers = {
        "dnsmos": _FakeScorer(
            {
                "dnsmos_ovrl": [5.0, 1.0],
                "dnsmos_sig": [1.0, 5.0],
                "dnsmos_bak": [3.0, 3.0],
            }
        ),
        "speaker_similarity": _FakeScorer({"speaker_similarity": [0.5, 0.1]}),
        "speechbertscore": _FakeScorer({"speechbertscore": [0.8, 0.4]}),
    }

    result = pipeline._score_candidate(
        clean_dir=Path("/tmp/clean"),
        candidate_dir=Path("/tmp/candidate"),
        filenames=["a.wav", "b.wav"],
        failed_metrics={},
    )

    expected = np.asarray([8.0, 2.8], dtype=np.float32)

    assert np.allclose(result["avg"], expected, atol=1e-6)
    assert np.allclose(result["norm"]["reward_component_dnsmos_norm"], [2.5, 0.5])
    assert np.allclose(result["norm"]["reward_component_speaker_similarity_norm"], [2.5, 0.5])
    assert np.allclose(result["norm"]["reward_component_speechbertscore_norm"], [4.0, 2.0])


def test_zero_weight_metric_can_be_scored_without_affecting_reward():
    pipeline = object.__new__(SpeechRewardPipeline)
    pipeline.reward_config = _merge_reward_config(
        {
            "registry": ["dnsmos", "speaker_similarity"],
            "scoring_registry": ["dnsmos", "speaker_similarity"],
            "weights": {"dnsmos": 1.0, "speaker_similarity": 0.0},
            "normalization": "raw_linear",
            "primary_keys": {"dnsmos": "dnsmos_avg", "speaker_similarity": "speaker_similarity"},
        }
    )
    pipeline.reward_metrics = ["dnsmos"]
    pipeline.active_metrics = ["dnsmos"]
    pipeline.scoring_metrics = ["dnsmos", "speaker_similarity"]
    pipeline.scorers = {
        "dnsmos": _FakeScorer(
            {
                "dnsmos_ovrl": [3.0, 4.0],
                "dnsmos_sig": [3.0, 4.0],
                "dnsmos_bak": [3.0, 4.0],
            }
        ),
        "speaker_similarity": _FakeScorer({"speaker_similarity": [0.97, 0.96]}),
    }

    result = pipeline._score_candidate(
        clean_dir=Path("/tmp/clean"),
        candidate_dir=Path("/tmp/candidate"),
        filenames=["a.wav", "b.wav"],
        failed_metrics={},
        metric_names=["dnsmos", "speaker_similarity"],
    )

    assert np.allclose(result["avg"], [3.0, 4.0])
    assert np.allclose(result["raw"]["speaker_similarity"], [0.97, 0.96])
    assert "reward_component_speaker_similarity_norm" not in result["norm"]


def test_eres2net_checkpoint_path_accepts_model_directory(tmp_path):
    checkpoint = tmp_path / ERes2NetSpeakerSimilarityReward.checkpoint_name
    checkpoint.write_bytes(b"checkpoint")

    resolved = ERes2NetSpeakerSimilarityReward._resolve_checkpoint_path(tmp_path)

    assert resolved == checkpoint.resolve()


def test_eres2net_code_path_requires_speakerlab_package(tmp_path):
    try:
        ERes2NetSpeakerSimilarityReward._prepare_speakerlab_import(tmp_path)
    except FileNotFoundError as exc:
        assert "speakerlab" in str(exc)
    else:
        raise AssertionError("Expected a missing speakerlab package to fail.")
