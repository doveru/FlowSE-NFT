
from __future__ import annotations


import contextlib
import importlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch


DEFAULT_VOICE_EVAL_ROOT = Path(os.environ.get("VOICE_EVALUATION_ROOT") or (Path(__file__).resolve().parents[3] / "voice_evaluation"))
DEFAULT_REWARD_WEIGHTS = {
    "dnsmos": 1.0,
    "nisqa": 0.0,
    "speechbertscore": 0.0,
    "speaker_similarity": 0.0,
}
DEFAULT_REWARD_REGISTRY = ["dnsmos"]
SUPPORTED_REWARD_METRICS = (
    "dnsmos",
    "nisqa",
    "speechbertscore",
    "speaker_similarity",
)
REWARD_PRIMARY_NORM_KEY = {
    "dnsmos": "dnsmos_avg",
    "nisqa": "nisqa_mos",
    "speechbertscore": "speechbertscore",
    "speaker_similarity": "speaker_similarity",
}
SUPPORTED_REWARD_PRIMARY_KEYS = {
    "dnsmos": {"dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak", "dnsmos_avg"},
    "nisqa": {"nisqa_mos"},
    "speechbertscore": {"speechbertscore"},
    "speaker_similarity": {"speaker_similarity"},
}
DEFAULT_HF_CACHE_CANDIDATES = [
    "~/.cache/huggingface",
    "~/.cache/huggingface/hub",
]


def _merge_reward_config(reward_config: dict[str, Any] | None) -> dict[str, Any]:
    merged = {
        "weights": dict(DEFAULT_REWARD_WEIGHTS),
        "registry": list(DEFAULT_REWARD_REGISTRY),
        "scoring_registry": list(DEFAULT_REWARD_REGISTRY),
        "normalization": "raw_linear",
        "batch_zscore_eps": 1e-6,
        "primary_keys": dict(REWARD_PRIMARY_NORM_KEY),
        "raw_scales": {"dnsmos": 1.0},
    }
    if reward_config is None:
        return merged

    weights = reward_config.get("weights", {})
    if not isinstance(weights, dict):
        raise TypeError("`reward_config['weights']` must be a dict when provided.")
    for metric_name, metric_weight in weights.items():
        metric_key = str(metric_name).strip().lower()
        if metric_key not in SUPPORTED_REWARD_METRICS:
            raise ValueError(f"Unsupported reward metric in weights: {metric_name!r}")
        merged["weights"][metric_key] = float(metric_weight)

    registry = reward_config.get("registry")
    if registry is not None:
        if not isinstance(registry, (list, tuple)):
            raise TypeError("`reward_config['registry']` must be a list/tuple when provided.")
        normalized_registry: list[str] = []
        seen: set[str] = set()
        for metric_name in registry:
            metric_key = str(metric_name).strip().lower()
            if metric_key not in SUPPORTED_REWARD_METRICS:
                raise ValueError(f"Unsupported reward metric in registry: {metric_name!r}")
            if metric_key in seen:
                continue
            seen.add(metric_key)
            normalized_registry.append(metric_key)
        merged["registry"] = normalized_registry

    scoring_registry = reward_config.get("scoring_registry")
    if scoring_registry is None:
        merged["scoring_registry"] = list(merged["registry"])
    else:
        if not isinstance(scoring_registry, (list, tuple)):
            raise TypeError("`reward_config['scoring_registry']` must be a list/tuple when provided.")
        normalized_scoring_registry: list[str] = []
        seen_scoring_metrics: set[str] = set()
        for metric_name in scoring_registry:
            metric_key = str(metric_name).strip().lower()
            if metric_key not in SUPPORTED_REWARD_METRICS:
                raise ValueError(f"Unsupported reward metric in scoring_registry: {metric_name!r}")
            if metric_key in seen_scoring_metrics:
                continue
            seen_scoring_metrics.add(metric_key)
            normalized_scoring_registry.append(metric_key)
        merged["scoring_registry"] = normalized_scoring_registry

    normalization = str(reward_config.get("normalization", merged["normalization"])).strip().lower()
    if normalization not in {"fixed_range", "raw_linear", "batch_zscore", "batch_std"}:
        raise ValueError(
            "`reward_config['normalization']` must be one of "
            "{'fixed_range', 'raw_linear', 'batch_zscore', 'batch_std'}."
        )
    merged["normalization"] = normalization
    merged["batch_zscore_eps"] = float(reward_config.get("batch_zscore_eps", merged["batch_zscore_eps"]))

    primary_keys = reward_config.get("primary_keys", {})
    if not isinstance(primary_keys, dict):
        raise TypeError("`reward_config['primary_keys']` must be a dict when provided.")
    for metric_name, primary_key in primary_keys.items():
        metric_key = str(metric_name).strip().lower()
        if metric_key not in SUPPORTED_REWARD_METRICS:
            raise ValueError(f"Unsupported reward metric in primary_keys: {metric_name!r}")
        primary_key_str = str(primary_key).strip().lower()
        if primary_key_str not in SUPPORTED_REWARD_PRIMARY_KEYS[metric_key]:
            raise ValueError(
                f"Unsupported primary key {primary_key!r} for reward metric {metric_name!r}. "
                f"Supported={sorted(SUPPORTED_REWARD_PRIMARY_KEYS[metric_key])}"
            )
        merged["primary_keys"][metric_key] = primary_key_str

    raw_scales = reward_config.get("raw_scales", {})
    if not isinstance(raw_scales, dict):
        raise TypeError("`reward_config['raw_scales']` must be a dict when provided.")
    normalized_scales: dict[str, float] = {}
    valid_scale_keys = set(SUPPORTED_REWARD_METRICS)
    for keys in SUPPORTED_REWARD_PRIMARY_KEYS.values():
        valid_scale_keys.update(keys)
    for scale_name, scale_value in raw_scales.items():
        scale_key = str(scale_name).strip().lower()
        if scale_key not in valid_scale_keys:
            raise ValueError(f"Unsupported reward raw scale key: {scale_name!r}")
        normalized_scales[scale_key] = float(scale_value)
    merged["raw_scales"] = normalized_scales
    return merged


def _resolve_device(device: str | torch.device) -> torch.device:
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def _common_hf_cache_dirs(hf_cache_dir: str | Path | None) -> list[Path]:
    candidates: list[str | Path] = []
    if hf_cache_dir:
        candidates.append(hf_cache_dir)
    for env_name in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        env_value = os.environ.get(env_name)
        if env_value:
            candidates.append(env_value)
    candidates.extend(DEFAULT_HF_CACHE_CANDIDATES)

    resolved: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        path = Path(candidate).expanduser().resolve()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            resolved.append(path)
    return resolved


def _resolve_optional_model_path(model_path: str | Path | None) -> Path | None:
    if model_path is None:
        return None
    resolved = Path(model_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Model path not found: {resolved}")
    return resolved


def _load_pretrained_with_fallbacks(
    loader,
    repo_ids: list[str],
    *,
    model_label: str,
    model_path: str | Path | None = None,
    hf_cache_dir: str | Path | None = None,
    local_files_only: bool | None = None,
):
    resolved_model_path = _resolve_optional_model_path(model_path)
    errors: list[str] = []

    if resolved_model_path is not None:
        try:
            return loader(str(resolved_model_path), local_files_only=True)
        except Exception as exc:
            errors.append(f"local path {resolved_model_path}: {exc}")

    cache_dirs = _common_hf_cache_dirs(hf_cache_dir)
    for cache_dir in cache_dirs:
        for repo_id in repo_ids:
            try:
                return loader(repo_id, cache_dir=str(cache_dir), local_files_only=True)
            except Exception as exc:
                errors.append(f"cache_dir={cache_dir}, repo={repo_id}: {exc}")

    allow_remote = local_files_only is False
    if allow_remote:
        for repo_id in repo_ids:
            try:
                kwargs: dict[str, Any] = {}
                if hf_cache_dir is not None:
                    kwargs["cache_dir"] = str(Path(hf_cache_dir).expanduser().resolve())
                return loader(repo_id, local_files_only=False, **kwargs)
            except Exception as exc:
                errors.append(f"remote repo={repo_id}: {exc}")

    hint_parts = [
        f"Unable to load {model_label} in offline mode.",
        "Provide a local model directory, or point `hf_cache_dir` to an existing Hugging Face cache.",
    ]
    if not allow_remote:
        hint_parts.append("If this machine has internet access, rerun with `local_files_only=False`.")
    error_preview = "\n".join(errors[:6])
    raise RuntimeError(" ".join(hint_parts) + (f"\nTried:\n{error_preview}" if error_preview else ""))


def _ensure_voice_eval_root(voice_eval_root: str | Path, active_metrics: list[str] | tuple[str, ...]) -> Path:
    root = Path(voice_eval_root).expanduser().resolve()
    active_set = {str(metric_name).strip().lower() for metric_name in active_metrics}
    required_paths = []
    if "dnsmos" in active_set:
        required_paths.extend(
            [
                root / "evaluation" / "dnsmos.py",
                root / "evaluation" / "DNSMOS" / "model_v8.onnx",
                root / "evaluation" / "DNSMOS" / "sig_bak_ovr.onnx",
            ]
        )
    if "nisqa" in active_set:
        required_paths.extend(
            [
                root / "evaluation" / "nisqa_utils.py",
                root / "evaluation" / "NISQA" / "nisqa.tar",
            ]
        )
    if not root.exists():
        raise FileNotFoundError(f"voice_evaluation root not found: {root}")
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required reward resource not found: {path}")
    return root


def _ensure_voice_eval_on_path(voice_eval_root: Path) -> None:
    root = str(voice_eval_root)
    if root not in sys.path:
        sys.path.insert(0, root)


@contextlib.contextmanager
def patch_onnxruntime_inference_session(device: str | torch.device):
    if importlib.util.find_spec("onnxruntime") is None:
        yield
        return

    import onnxruntime as ort

    available = set(ort.get_available_providers())
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        providers: list[Any] = []
        if "CUDAExecutionProvider" in available:
            providers.append(
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": int(0 if torch_device.index is None else torch_device.index),
                    },
                )
            )
        if "CPUExecutionProvider" in available:
            providers.append("CPUExecutionProvider")
    else:
        providers = [provider for provider in ("CPUExecutionProvider",) if provider in available]
    if not providers:
        providers = list(available)

    original_inference_session = ort.InferenceSession

    def patched_inference_session(*args, **kwargs):
        if kwargs.get("providers") is None:
            kwargs["providers"] = providers
        return original_inference_session(*args, **kwargs)

    ort.InferenceSession = patched_inference_session
    try:
        yield
    finally:
        ort.InferenceSession = original_inference_session


def _safe_filename(index: int, utt_id: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", utt_id).strip("._")
    sanitized = sanitized or f"sample_{index:06d}"
    return f"{index:06d}_{sanitized}.wav"


def _to_waveform_list(wavs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
    if isinstance(wavs, torch.Tensor):
        if wavs.ndim == 3:
            # [B,1,T] -> [B,T]
            return [item.squeeze(0).detach().cpu().to(torch.float32).contiguous() for item in wavs]
        if wavs.ndim == 2:
            # [B,T]
            return [item.detach().cpu().to(torch.float32).contiguous() for item in wavs]
        raise ValueError(f"Expected wav tensor with shape [B,1,T] or [B,T], got {tuple(wavs.shape)}")

    if not isinstance(wavs, (list, tuple)):
        raise TypeError("Waveforms must be a tensor or a list/tuple of tensors.")

    result = []
    for wav in wavs:
        if not isinstance(wav, torch.Tensor):
            wav = torch.as_tensor(wav)
        if wav.ndim == 2:
            if wav.shape[0] != 1:
                wav = wav.mean(dim=0, keepdim=True)
            wav = wav.squeeze(0)
        elif wav.ndim != 1:
            raise ValueError(f"Expected waveform item with shape [T] or [1,T], got {tuple(wav.shape)}")
        result.append(wav.detach().cpu().to(torch.float32).contiguous())
    return result


def _write_waveforms(directory: Path, utt_ids: list[str], wavs: list[torch.Tensor], sample_rate: int) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    filenames: list[str] = []
    for index, (utt_id, wav) in enumerate(zip(utt_ids, wavs)):
        filename = _safe_filename(index, utt_id)
        wav_np = wav.numpy().astype(np.float32, copy=False)
        sf.write(directory / filename, wav_np, sample_rate)
        filenames.append(filename)
    return filenames


def _load_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    wav, sample_rate = sf.read(path, dtype="float32")
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    return wav, sample_rate


def _resample_if_needed(wav: np.ndarray, source_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if source_sample_rate == target_sample_rate:
        return wav.astype(np.float32, copy=False)
    import soxr

    return soxr.resample(wav, source_sample_rate, target_sample_rate).astype(np.float32, copy=False)


def _align_candidate_to_reference(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    if len(candidate) < len(reference):
        candidate = np.pad(candidate, (0, len(reference) - len(candidate)), mode="constant")
    return candidate[: len(reference)]


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(result):
        return float("nan")
    return result


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _normalize_metric(metric_name: str, value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    if metric_name == "nisqa_mos":
        return _clamp01((value - 1.0) / 4.0)
    if metric_name in {"dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak", "dnsmos_avg"}:
        return _clamp01((value - 1.0) / 4.0)
    if metric_name == "speechbertscore":
        return _clamp01(value)
    if metric_name == "speaker_similarity":
        return _clamp01((value + 1.0) / 2.0)
    raise KeyError(f"Unsupported metric normalization: {metric_name}")


def _nanmean_triplet(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(array).any():
        return float("nan")
    return float(np.nanmean(array))


def _batch_zscore(values: list[float], eps: float = 1e-6) -> list[float]:
    array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return [0.0 for _ in values]
    mean = float(np.mean(array[finite]))
    std = float(np.std(array[finite]))
    if not math.isfinite(std) or std <= float(eps):
        return [0.0 for _ in values]
    filled = np.where(finite, array, mean)
    normalized = (filled - mean) / std
    return [_safe_float(float(value)) for value in normalized]


def _batch_std_scale(values: list[float], eps: float = 1e-6) -> list[float]:
    """
    Scale a reward component by its batch standard deviation without centering.
    This matches the FlowSE-GRPO multi-metric reward definition.
    """
    array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return [0.0 for _ in values]
    std = float(np.std(array[finite]))
    if not math.isfinite(std) or std <= float(eps):
        return [0.0 for _ in values]
    normalized = np.where(finite, array / std, 0.0)
    return [_safe_float(float(value)) for value in normalized]


def _component_raw_scale(metric_name: str, metric_key: str, raw_scales: dict[str, float]) -> float:
    if metric_key in raw_scales:
        return float(raw_scales[metric_key])
    if metric_name in raw_scales:
        return float(raw_scales[metric_name])
    return 1.0


class BaseReward:
    metric_name: str
    output_keys: tuple[str, ...]

    def score(self, clean_path: Path, candidate_path: Path) -> dict[str, float]:
        raise NotImplementedError

    def score_batch(
        self,
        clean_paths: list[Path],
        candidate_paths: list[Path],
        failed_metrics: dict[str, int],
    ) -> dict[str, list[float]]:
        batch_scores = {key: [] for key in self.output_keys}
        for clean_path, candidate_path in zip(clean_paths, candidate_paths):
            try:
                sample_scores = self.score(clean_path, candidate_path)
            except Exception:
                failed_metrics[self.metric_name] += 1
                for key in self.output_keys:
                    batch_scores[key].append(float("nan"))
                continue

            for key in self.output_keys:
                batch_scores[key].append(_safe_float(sample_scores.get(key, float("nan"))))
        return batch_scores


class DNSMOSReward(BaseReward):
    metric_name = "dnsmos"
    output_keys = ("dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak")

    def __init__(self, voice_eval_root: Path, device: torch.device):
        dnsmos_device = os.environ.get("SPEECH_DNSMOS_DEVICE", "cpu").strip().lower()
        if dnsmos_device == "cuda":
            dnsmos_torch_device = device
        else:
            dnsmos_torch_device = torch.device("cpu")
        _ensure_voice_eval_on_path(voice_eval_root)
        module = importlib.import_module("evaluation.dnsmos")
        primary_model_path = voice_eval_root / "evaluation" / "DNSMOS" / "sig_bak_ovr.onnx"
        p808_model_path = voice_eval_root / "evaluation" / "DNSMOS" / "model_v8.onnx"
        with patch_onnxruntime_inference_session(str(dnsmos_torch_device)):
            self.compute_score = module.ComputeScore(str(primary_model_path), str(p808_model_path))
        self.sampling_rate = int(module.SAMPLING_RATE)

    def score(self, clean_path: Path, candidate_path: Path) -> dict[str, float]:
        del clean_path
        clip_scores = self.compute_score(str(candidate_path), self.sampling_rate)
        return {
            "dnsmos_ovrl": clip_scores["OVRL"],
            "dnsmos_sig": clip_scores["SIG"],
            "dnsmos_bak": clip_scores["BAK"],
        }


class NISQAReward(BaseReward):
    metric_name = "nisqa"
    output_keys = ("nisqa_mos",)

    def __init__(self, voice_eval_root: Path, device: torch.device):
        _ensure_voice_eval_on_path(voice_eval_root)
        nisqa_utils = importlib.import_module("evaluation.nisqa_utils")
        model_path = voice_eval_root / "evaluation" / "NISQA" / "nisqa.tar"
        self.model = nisqa_utils.load_nisqa_model(str(model_path), device=str(device))
        self.predict_nisqa = nisqa_utils.predict_nisqa

    def score(self, clean_path: Path, candidate_path: Path) -> dict[str, float]:
        del clean_path
        score = self.predict_nisqa(self.model, str(candidate_path))
        return {"nisqa_mos": score["mos_pred"]}


class SpeechBERTScoreReward(BaseReward):
    metric_name = "speechbertscore"
    output_keys = ("speechbertscore",)

    def __init__(
        self,
        device: torch.device,
        hf_cache_dir: str | Path | None = None,
        model_path: str | Path | None = None,
        local_files_only: bool | None = None,
    ):
        import torchaudio
        from transformers import HubertModel

        self.device = device
        self.hubert = _load_pretrained_with_fallbacks(
            HubertModel.from_pretrained,
            ["facebook/hubert-base-ls960"],
            model_label="SpeechBERTScore HuBERT model",
            model_path=model_path,
            hf_cache_dir=hf_cache_dir,
            local_files_only=local_files_only,
        )
        self.hubert.eval().to(device)
        self.layer = 8
        self.target_sample_rate = 16000
        self.resampler_cls = torchaudio.transforms.Resample

    def _process(self, wav: np.ndarray, sample_rate: int) -> torch.Tensor:
        wav_tensor = torch.from_numpy(wav).unsqueeze(0).to(self.device).float()
        if sample_rate != self.target_sample_rate:
            resampler = self.resampler_cls(orig_freq=sample_rate, new_freq=self.target_sample_rate).to(self.device)
            wav_tensor = resampler(wav_tensor)
        with torch.no_grad():
            hidden_states = self.hubert(wav_tensor, output_hidden_states=True).hidden_states
        return hidden_states[self.layer].squeeze(0)

    def _process_batch(self, wavs: list[np.ndarray]) -> tuple[torch.Tensor, list[int]]:
        if not wavs:
            return torch.empty((0, 0, 0), device=self.device), []

        lengths = [int(len(wav)) for wav in wavs]
        max_length = max(lengths)
        batch = torch.zeros((len(wavs), max_length), device=self.device, dtype=torch.float32)
        attention_mask = torch.zeros((len(wavs), max_length), device=self.device, dtype=torch.long)
        for index, wav in enumerate(wavs):
            wav_tensor = torch.from_numpy(wav).to(self.device).float()
            batch[index, : wav_tensor.numel()] = wav_tensor
            attention_mask[index, : wav_tensor.numel()] = 1

        with torch.no_grad():
            hidden_states = self.hubert(
                batch,
                attention_mask=attention_mask,
                output_hidden_states=True,
            ).hidden_states[self.layer]
        feat_lengths = self.hubert._get_feat_extract_output_lengths(
            torch.as_tensor(lengths, device=self.device, dtype=torch.long)
        )
        return hidden_states, [int(length.item()) for length in feat_lengths]

    def extract_embedding_batch(self, wavs: list[np.ndarray], sample_rate: int) -> torch.Tensor:
        """Return one HuBERT layer-8 embedding per waveform for trajectory diagnostics."""
        resampled = [
            _resample_if_needed(wav, int(sample_rate), self.target_sample_rate)
            for wav in wavs
        ]
        hidden_states, feature_lengths = self._process_batch(resampled)
        pooled = []
        for index, feature_length in enumerate(feature_lengths):
            if feature_length <= 0:
                raise ValueError("HuBERT produced an empty feature sequence.")
            pooled.append(hidden_states[index, :feature_length].mean(dim=0))
        return torch.stack(pooled, dim=0)

    def score(self, clean_path: Path, candidate_path: Path) -> dict[str, float]:
        reference, reference_sample_rate = _load_audio_mono(clean_path)
        candidate, candidate_sample_rate = _load_audio_mono(candidate_path)
        candidate = _resample_if_needed(candidate, candidate_sample_rate, reference_sample_rate)
        candidate = _align_candidate_to_reference(reference, candidate)

        reference_features = self._process(reference, reference_sample_rate)
        candidate_features = self._process(candidate, reference_sample_rate)

        similarity = torch.matmul(candidate_features, reference_features.T)
        similarity = similarity / (
            torch.norm(candidate_features, dim=1, keepdim=True) * torch.norm(reference_features, dim=1).unsqueeze(0)
        )
        precision = torch.max(similarity, dim=1)[0].mean().item()
        return {"speechbertscore": precision}

    def score_batch(
        self,
        clean_paths: list[Path],
        candidate_paths: list[Path],
        failed_metrics: dict[str, int],
    ) -> dict[str, list[float]]:
        batch_scores = {key: [float("nan")] * len(clean_paths) for key in self.output_keys}
        reference_wavs: list[np.ndarray] = []
        candidate_wavs: list[np.ndarray] = []
        valid_indices: list[int] = []

        for index, (clean_path, candidate_path) in enumerate(zip(clean_paths, candidate_paths, strict=True)):
            try:
                reference, reference_sample_rate = _load_audio_mono(clean_path)
                candidate, candidate_sample_rate = _load_audio_mono(candidate_path)
                if reference_sample_rate != self.target_sample_rate:
                    reference = _resample_if_needed(reference, reference_sample_rate, self.target_sample_rate)
                    reference_sample_rate = self.target_sample_rate
                candidate = _resample_if_needed(candidate, candidate_sample_rate, reference_sample_rate)
                candidate = _align_candidate_to_reference(reference, candidate)
            except Exception:
                failed_metrics[self.metric_name] += 1
                continue

            reference_wavs.append(reference.astype(np.float32, copy=False))
            candidate_wavs.append(candidate.astype(np.float32, copy=False))
            valid_indices.append(index)

        if not valid_indices:
            return batch_scores

        try:
            reference_features, reference_lengths = self._process_batch(reference_wavs)
            candidate_features, candidate_lengths = self._process_batch(candidate_wavs)
            for local_index, output_index in enumerate(valid_indices):
                reference_feature = reference_features[local_index, : reference_lengths[local_index]]
                candidate_feature = candidate_features[local_index, : candidate_lengths[local_index]]
                similarity = torch.matmul(candidate_feature, reference_feature.T)
                similarity = similarity / (
                    torch.norm(candidate_feature, dim=1, keepdim=True)
                    * torch.norm(reference_feature, dim=1).unsqueeze(0)
                ).clamp_min(1e-8)
                batch_scores["speechbertscore"][output_index] = _safe_float(torch.max(similarity, dim=1)[0].mean().item())
        except Exception:
            failed_metrics[self.metric_name] += len(valid_indices)
        return batch_scores


class SpeakerSimilarityReward(BaseReward):
    metric_name = "speaker_similarity"
    output_keys = ("speaker_similarity",)

    def __init__(
        self,
        device: torch.device,
        hf_cache_dir: str | Path | None = None,
        model_path: str | Path | None = None,
        local_files_only: bool | None = None,
    ):
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

        self.device = device
        repo_ids = ["microsoft/wavlm-base-plus-sv", "wavlm-base-plus-sv"]
        self.feature_extractor = _load_pretrained_with_fallbacks(
            Wav2Vec2FeatureExtractor.from_pretrained,
            repo_ids,
            model_label="speaker feature extractor",
            model_path=model_path,
            hf_cache_dir=hf_cache_dir,
            local_files_only=local_files_only,
        )
        self.model = _load_pretrained_with_fallbacks(
            WavLMForXVector.from_pretrained,
            repo_ids,
            model_label="speaker similarity WavLM model",
            model_path=model_path,
            hf_cache_dir=hf_cache_dir,
            local_files_only=local_files_only,
        )

        self.model.eval().to(device)
        self.target_sample_rate = 16000

    def _extract_embedding(self, wav: np.ndarray) -> torch.Tensor:
        inputs = self.feature_extractor([wav], padding=True, return_tensors="pt", sampling_rate=self.target_sample_rate)
        for key, value in inputs.items():
            inputs[key] = value.to(self.device)
        with torch.no_grad():
            embeddings = self.model(**inputs).embeddings
        return embeddings[0]

    def _extract_embedding_batch(self, wavs: list[np.ndarray]) -> torch.Tensor:
        if not wavs:
            return torch.empty((0, 0), device=self.device)
        inputs = self.feature_extractor(wavs, padding=True, return_tensors="pt", sampling_rate=self.target_sample_rate)
        for key, value in inputs.items():
            inputs[key] = value.to(self.device)
        with torch.no_grad():
            embeddings = self.model(**inputs).embeddings
        return embeddings

    def extract_embedding_batch(self, wavs: list[np.ndarray], sample_rate: int) -> torch.Tensor:
        """Return the same WavLM x-vector representation used by the reward."""
        resampled = [
            _resample_if_needed(wav, int(sample_rate), self.target_sample_rate)
            for wav in wavs
        ]
        return self._extract_embedding_batch(resampled)

    def score(self, clean_path: Path, candidate_path: Path) -> dict[str, float]:
        reference, reference_sample_rate = _load_audio_mono(clean_path)
        candidate, candidate_sample_rate = _load_audio_mono(candidate_path)
        reference = _resample_if_needed(reference, reference_sample_rate, self.target_sample_rate)
        candidate = _resample_if_needed(candidate, candidate_sample_rate, self.target_sample_rate)
        candidate = _align_candidate_to_reference(reference, candidate)

        reference_embedding = self._extract_embedding(reference)
        candidate_embedding = self._extract_embedding(candidate)
        score = torch.nn.functional.cosine_similarity(reference_embedding, candidate_embedding, dim=-1).item()
        return {"speaker_similarity": score}

    def score_batch(
        self,
        clean_paths: list[Path],
        candidate_paths: list[Path],
        failed_metrics: dict[str, int],
    ) -> dict[str, list[float]]:
        batch_scores = {key: [float("nan")] * len(clean_paths) for key in self.output_keys}
        reference_wavs: list[np.ndarray] = []
        candidate_wavs: list[np.ndarray] = []
        valid_indices: list[int] = []

        for index, (clean_path, candidate_path) in enumerate(zip(clean_paths, candidate_paths, strict=True)):
            try:
                reference, reference_sample_rate = _load_audio_mono(clean_path)
                candidate, candidate_sample_rate = _load_audio_mono(candidate_path)
                reference = _resample_if_needed(reference, reference_sample_rate, self.target_sample_rate)
                candidate = _resample_if_needed(candidate, candidate_sample_rate, self.target_sample_rate)
                candidate = _align_candidate_to_reference(reference, candidate)
            except Exception:
                failed_metrics[self.metric_name] += 1
                continue

            reference_wavs.append(reference.astype(np.float32, copy=False))
            candidate_wavs.append(candidate.astype(np.float32, copy=False))
            valid_indices.append(index)

        if not valid_indices:
            return batch_scores

        try:
            reference_embeddings = self._extract_embedding_batch(reference_wavs)
            candidate_embeddings = self._extract_embedding_batch(candidate_wavs)
            scores = torch.nn.functional.cosine_similarity(reference_embeddings, candidate_embeddings, dim=-1)
            for local_index, output_index in enumerate(valid_indices):
                batch_scores["speaker_similarity"][output_index] = _safe_float(scores[local_index].item())
        except Exception:
            failed_metrics[self.metric_name] += len(valid_indices)
        return batch_scores


class ERes2NetSpeakerSimilarityReward(BaseReward):
    """Speaker similarity based on the official 3D-Speaker ERes2Net model."""

    metric_name = "speaker_similarity"
    output_keys = ("speaker_similarity",)
    checkpoint_name = "pretrained_eres2net_aug.ckpt"
    target_sample_rate = 16000
    max_inference_batch_size = 16

    def __init__(
        self,
        device: torch.device,
        model_path: str | Path | None,
        code_path: str | Path | None = None,
    ):
        if model_path is None:
            raise ValueError("ERes2Net requires `speaker_model_path` to point to its checkpoint or model directory.")

        self.device = device
        self.checkpoint_path = self._resolve_checkpoint_path(model_path)
        self.code_path = self._prepare_speakerlab_import(code_path)

        try:
            model_module = importlib.import_module("speakerlab.models.eres2net.ERes2Net_huge")
            model_class = getattr(model_module, "ERes2Net")
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "Unable to import the ERes2Net model from 3D-Speaker. "
                "Set `speaker_code_path` to the cloned 3D-Speaker repository."
            ) from exc

        self.model = model_class(feat_dim=80, embedding_size=192)
        try:
            state_dict = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(self.checkpoint_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if not isinstance(state_dict, dict):
            raise TypeError(f"Unsupported ERes2Net checkpoint format: {type(state_dict).__name__}")
        if state_dict and all(str(key).startswith("module.") for key in state_dict):
            state_dict = {str(key)[7:]: value for key, value in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval().to(self.device)

        try:
            import torchaudio.compliance.kaldi as kaldi
        except ImportError as exc:
            raise RuntimeError("ERes2Net speaker similarity requires torchaudio.") from exc
        self.kaldi = kaldi

    @classmethod
    def _resolve_checkpoint_path(cls, model_path: str | Path) -> Path:
        path = Path(model_path).expanduser().resolve()
        checkpoint = path if path.is_file() else path / cls.checkpoint_name
        if not checkpoint.is_file():
            raise FileNotFoundError(f"ERes2Net checkpoint not found: {checkpoint}")
        return checkpoint

    @staticmethod
    def _prepare_speakerlab_import(code_path: str | Path | None) -> Path | None:
        if code_path is None:
            return None
        root = Path(code_path).expanduser().resolve()
        if not (root / "speakerlab").is_dir():
            raise FileNotFoundError(f"3D-Speaker `speakerlab` package not found under: {root}")
        root_string = str(root)
        if root_string not in sys.path:
            sys.path.insert(0, root_string)
        return root

    def _extract_feature(self, wav: np.ndarray) -> torch.Tensor:
        waveform = torch.from_numpy(np.asarray(wav, dtype=np.float32)).reshape(1, -1)
        min_samples = 400
        if waveform.shape[-1] < min_samples:
            waveform = torch.nn.functional.pad(waveform, (0, min_samples - waveform.shape[-1]))
        feature = self.kaldi.fbank(
            waveform,
            num_mel_bins=80,
            sample_frequency=self.target_sample_rate,
            dither=0.0,
        )
        return feature - feature.mean(dim=0, keepdim=True)

    def _extract_embedding_batch(self, wavs: list[np.ndarray]) -> torch.Tensor:
        if not wavs:
            return torch.empty((0, 192), device=self.device)

        features = [self._extract_feature(wav) for wav in wavs]
        frame_groups: dict[int, list[int]] = defaultdict(list)
        for index, feature in enumerate(features):
            frame_groups[int(feature.shape[0])].append(index)

        embeddings: list[torch.Tensor | None] = [None] * len(features)
        with torch.no_grad():
            for indices in frame_groups.values():
                for start in range(0, len(indices), self.max_inference_batch_size):
                    chunk_indices = indices[start : start + self.max_inference_batch_size]
                    batch = torch.stack([features[index] for index in chunk_indices]).to(self.device)
                    batch_embeddings = self.model(batch)
                    for index, embedding in zip(chunk_indices, batch_embeddings):
                        embeddings[index] = embedding

        if any(embedding is None for embedding in embeddings):
            raise RuntimeError("ERes2Net failed to produce all requested embeddings.")
        return torch.stack([embedding for embedding in embeddings if embedding is not None])

    def extract_embedding_batch(self, wavs: list[np.ndarray], sample_rate: int) -> torch.Tensor:
        """Return the same ERes2Net representation used by the speaker reward."""
        resampled = [
            _resample_if_needed(wav, int(sample_rate), self.target_sample_rate)
            for wav in wavs
        ]
        return self._extract_embedding_batch(resampled)

    def _load_aligned_wavs(self, clean_path: Path, candidate_path: Path) -> tuple[np.ndarray, np.ndarray]:
        reference, reference_sample_rate = _load_audio_mono(clean_path)
        candidate, candidate_sample_rate = _load_audio_mono(candidate_path)
        reference = _resample_if_needed(reference, reference_sample_rate, self.target_sample_rate)
        candidate = _resample_if_needed(candidate, candidate_sample_rate, self.target_sample_rate)
        candidate = _align_candidate_to_reference(reference, candidate)
        return reference.astype(np.float32, copy=False), candidate.astype(np.float32, copy=False)

    def score(self, clean_path: Path, candidate_path: Path) -> dict[str, float]:
        reference, candidate = self._load_aligned_wavs(clean_path, candidate_path)
        embeddings = self._extract_embedding_batch([reference, candidate])
        score = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=-1).item()
        return {"speaker_similarity": score}

    def score_batch(
        self,
        clean_paths: list[Path],
        candidate_paths: list[Path],
        failed_metrics: dict[str, int],
    ) -> dict[str, list[float]]:
        batch_scores = {key: [float("nan")] * len(clean_paths) for key in self.output_keys}
        reference_wavs: list[np.ndarray] = []
        candidate_wavs: list[np.ndarray] = []
        valid_indices: list[int] = []

        for index, (clean_path, candidate_path) in enumerate(zip(clean_paths, candidate_paths, strict=True)):
            try:
                reference, candidate = self._load_aligned_wavs(clean_path, candidate_path)
            except Exception:
                failed_metrics[self.metric_name] += 1
                continue
            reference_wavs.append(reference)
            candidate_wavs.append(candidate)
            valid_indices.append(index)

        if not valid_indices:
            return batch_scores

        try:
            reference_embeddings = self._extract_embedding_batch(reference_wavs)
            candidate_embeddings = self._extract_embedding_batch(candidate_wavs)
            scores = torch.nn.functional.cosine_similarity(reference_embeddings, candidate_embeddings, dim=-1)
            for local_index, output_index in enumerate(valid_indices):
                batch_scores["speaker_similarity"][output_index] = _safe_float(scores[local_index].item())
        except Exception:
            failed_metrics[self.metric_name] += len(valid_indices)
        return batch_scores


class SpeechRewardPipeline:
    def __init__(
        self,
        device: str | torch.device,
        voice_eval_root: str | Path = DEFAULT_VOICE_EVAL_ROOT,
        reward_config: dict[str, Any] | None = None,
        hf_cache_dir: str | Path | None = None,
        speechbert_model_path: str | Path | None = None,
        speaker_model_path: str | Path | None = None,
        local_files_only: bool | None = None,
        tmp_root: str | Path | None = None,
        sample_rate: int = 16000,
        speaker_model_type: str = "wavlm",
        speaker_code_path: str | Path | None = None,
    ):
        self.device = _resolve_device(device)
        self.reward_config = _merge_reward_config(reward_config)
        self.sample_rate = sample_rate
        self.tmp_root = Path(tmp_root) if tmp_root is not None else Path(tempfile.gettempdir()) / "flowse_rewards"
        self.tmp_root = self.tmp_root.expanduser().resolve()
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self.reward_metrics = [
            metric_name
            for metric_name in self.reward_config["registry"]
            if float(self.reward_config["weights"].get(metric_name, 0.0)) > 0.0
        ]
        if not self.reward_metrics:
            raise ValueError(
                "No active reward metrics. Ensure at least one metric in `reward_config['registry']` has positive weight."
            )
        self.active_metrics = list(self.reward_metrics)
        self.scoring_metrics = list(self.reward_config["scoring_registry"])
        for metric_name in self.reward_metrics:
            if metric_name not in self.scoring_metrics:
                self.scoring_metrics.append(metric_name)
        self.voice_eval_root = _ensure_voice_eval_root(voice_eval_root, active_metrics=self.scoring_metrics)

        scorers: dict[str, Any] = {}
        if "dnsmos" in self.scoring_metrics:
            scorers["dnsmos"] = DNSMOSReward(self.voice_eval_root, self.device)
        if "nisqa" in self.scoring_metrics:
            scorers["nisqa"] = NISQAReward(self.voice_eval_root, self.device)
        if "speechbertscore" in self.scoring_metrics:
            scorers["speechbertscore"] = SpeechBERTScoreReward(
                self.device,
                hf_cache_dir=hf_cache_dir,
                model_path=speechbert_model_path,
                local_files_only=local_files_only,
            )
        if "speaker_similarity" in self.scoring_metrics:
            normalized_speaker_model_type = str(speaker_model_type).strip().lower()
            if normalized_speaker_model_type == "wavlm":
                scorers["speaker_similarity"] = SpeakerSimilarityReward(
                    self.device,
                    hf_cache_dir=hf_cache_dir,
                    model_path=speaker_model_path,
                    local_files_only=local_files_only,
                )
            elif normalized_speaker_model_type == "eres2net":
                scorers["speaker_similarity"] = ERes2NetSpeakerSimilarityReward(
                    self.device,
                    model_path=speaker_model_path,
                    code_path=speaker_code_path,
                )
            else:
                raise ValueError(
                    f"Unsupported speaker similarity model type: {speaker_model_type!r}. "
                    "Expected 'wavlm' or 'eres2net'."
                )
        self.scorers = scorers

    def _build_run_dir(self, split: str | None) -> Path:
        split_part = re.sub(r"[^A-Za-z0-9._-]+", "_", split or "default").strip("._") or "default"
        run_dir = self.tmp_root / f"{split_part}_{uuid.uuid4().hex[:12]}"
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def _score_candidate(
        self,
        clean_dir: Path,
        candidate_dir: Path,
        filenames: list[str],
        failed_metrics: dict[str, int],
        metric_names: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        clean_paths = [clean_dir / filename for filename in filenames]
        candidate_paths = [candidate_dir / filename for filename in filenames]

        selected_metrics = list(self.scorers) if metric_names is None else [str(name) for name in metric_names]
        unknown_metrics = [name for name in selected_metrics if name not in self.scorers]
        if unknown_metrics:
            raise KeyError(f"Requested reward scorers are not initialized: {unknown_metrics}")

        raw_scores: dict[str, list[float]] = {}
        for metric_name in selected_metrics:
            scorer = self.scorers[metric_name]
            raw_scores.update(scorer.score_batch(clean_paths, candidate_paths, failed_metrics))

        if "dnsmos" in selected_metrics:
            raw_scores["dnsmos_avg"] = [
                _nanmean_triplet([ovrl, sig, bak])
                for ovrl, sig, bak in zip(
                    raw_scores["dnsmos_ovrl"],
                    raw_scores["dnsmos_sig"],
                    raw_scores["dnsmos_bak"],
                )
            ]

        norm_scores = {
            metric_name: [_normalize_metric(metric_name, value) for value in values]
            for metric_name, values in raw_scores.items()
        }

        weights = self.reward_config["weights"]
        primary_keys = self.reward_config["primary_keys"]
        raw_scales = self.reward_config["raw_scales"]
        normalization = str(self.reward_config["normalization"])
        batch_zscore_eps = float(self.reward_config["batch_zscore_eps"])

        reward_metrics = getattr(self, "reward_metrics", self.active_metrics)
        selected_reward_metrics = [metric_name for metric_name in reward_metrics if metric_name in selected_metrics]
        component_norm_scores: dict[str, list[float]] = {}
        for metric_name in selected_reward_metrics:
            metric_key = primary_keys[metric_name]
            if metric_key not in raw_scores:
                raise KeyError(f"Reward metric {metric_name!r} primary key {metric_key!r} missing from raw scores.")
            scale = _component_raw_scale(metric_name, metric_key, raw_scales)
            raw_component_key = f"reward_component_{metric_name}_raw"
            norm_component_key = f"reward_component_{metric_name}_norm"
            component_raw = [
                _safe_float(float(value) * scale) if math.isfinite(float(value)) else float("nan")
                for value in raw_scores[metric_key]
            ]
            if normalization == "batch_zscore":
                component_norm = _batch_zscore(component_raw, eps=batch_zscore_eps)
            elif normalization == "batch_std":
                component_norm = _batch_std_scale(component_raw, eps=batch_zscore_eps)
            elif normalization == "raw_linear":
                component_norm = [
                    _safe_float(float(value)) if math.isfinite(float(value)) else 0.0
                    for value in component_raw
                ]
            else:
                if scale == 1.0:
                    component_norm = list(norm_scores[metric_key])
                else:
                    component_norm = [
                        _normalize_metric(metric_key, value / scale) * scale if math.isfinite(value) else 0.0
                        for value in component_raw
                    ]
            raw_scores[raw_component_key] = component_raw
            norm_scores[norm_component_key] = component_norm
            component_norm_scores[metric_name] = component_norm

        avg_rewards = []
        for index in range(len(filenames)):
            avg_reward = 0.0
            for metric_name in selected_reward_metrics:
                avg_reward += float(weights[metric_name]) * float(component_norm_scores[metric_name][index])
            avg_rewards.append(_safe_float(avg_reward) if math.isfinite(avg_reward) else 0.0)

        return {
            "raw": raw_scores,
            "norm": norm_scores,
            "avg": avg_rewards,
        }

    def _build_failed_metrics(self) -> defaultdict[str, int]:
        failed_metrics = defaultdict(int)
        for metric_name in self.scorers:
            failed_metrics[metric_name] = 0
        return failed_metrics

    def _build_meta(
        self,
        run_dir: Path,
        failed_metrics: dict[str, int],
        sample_rate: int,
        keep_tmp: bool,
    ) -> dict[str, Any]:
        return {
            "tmp_dir": str(run_dir),
            "tmp_dir_kept": bool(keep_tmp),
            "device": str(self.device),
            "sample_rate": sample_rate,
            "voice_eval_root": str(self.voice_eval_root),
            "weights": dict(self.reward_config["weights"]),
            "registry": list(self.reward_config["registry"]),
            "active_metrics": list(self.active_metrics),
            "scoring_metrics": list(getattr(self, "scoring_metrics", self.scorers)),
            "failed_metrics": dict(failed_metrics),
        }

    def trajectory_embedding_kinds(
        self,
        metric_names: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, str]:
        """Describe the reward-aligned representation used by trajectory CSS."""
        selected = list(self.scorers) if metric_names is None else [str(name) for name in metric_names]
        kinds: dict[str, str] = {}
        for metric_name in selected:
            if metric_name == "speechbertscore":
                kinds[metric_name] = "hubert_layer8_mean_pooled_embedding"
            elif metric_name == "speaker_similarity":
                scorer_name = type(self.scorers[metric_name]).__name__.lower()
                kinds[metric_name] = f"{scorer_name}_embedding"
            elif metric_name == "dnsmos":
                kinds[metric_name] = "dnsmos_ovrl_sig_bak_output_embedding"
            elif metric_name == "nisqa":
                kinds[metric_name] = "nisqa_mos_output_embedding"
            else:
                raise KeyError(f"No trajectory embedding is defined for reward metric {metric_name!r}.")
        return kinds

    def extract_reward_aligned_embeddings(
        self,
        wavs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        *,
        sample_rate: int,
        metric_names: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extract one fixed-size, reward-aligned representation per waveform.

        Speaker similarity reuses its exact speaker encoder, while SpeechBERTScore
        mean-pools the same HuBERT layer-8 features used by the reward. DNSMOS and
        NISQA do not expose hidden activations through their current inference
        interfaces, so their normalized output coordinates and complementary
        coordinates form the representation used for cosine similarity.
        """
        waveform_list = _to_waveform_list(wavs)
        if not waveform_list:
            raise ValueError("At least one waveform is required for reward-aligned embeddings.")

        selected = list(self.scorers) if metric_names is None else [str(name) for name in metric_names]
        unknown = [name for name in selected if name not in self.scorers]
        if unknown:
            raise KeyError(f"Requested reward scorers are not initialized: {unknown}")

        waveform_arrays = [wav.numpy().astype(np.float32, copy=False) for wav in waveform_list]
        embeddings: dict[str, torch.Tensor] = {}

        for metric_name in selected:
            scorer = self.scorers[metric_name]
            if metric_name in {"speechbertscore", "speaker_similarity"}:
                embedding = scorer.extract_embedding_batch(waveform_arrays, int(sample_rate))
                embedding = embedding.detach().to(torch.float32).reshape(len(waveform_arrays), -1)
                if not bool(torch.isfinite(embedding).all().item()):
                    raise RuntimeError(f"{metric_name} produced non-finite trajectory embeddings.")
                embeddings[metric_name] = embedding.cpu().contiguous()

        output_metrics = [name for name in selected if name in {"dnsmos", "nisqa"}]
        if output_metrics:
            run_dir = self._build_run_dir("trajectory_embedding")
            failed_metrics = self._build_failed_metrics()
            try:
                waveform_dir = run_dir / "waveform"
                utt_ids = [f"trajectory_{index:06d}" for index in range(len(waveform_list))]
                filenames = _write_waveforms(waveform_dir, utt_ids, waveform_list, int(sample_rate))
                paths = [waveform_dir / filename for filename in filenames]
                for metric_name in output_metrics:
                    scorer = self.scorers[metric_name]
                    scores = scorer.score_batch(paths, paths, failed_metrics)
                    coordinates = torch.tensor(
                        [scores[key] for key in scorer.output_keys],
                        dtype=torch.float32,
                    ).transpose(0, 1).contiguous()
                    if not bool(torch.isfinite(coordinates).all().item()):
                        raise RuntimeError(
                            f"{metric_name} failed to produce finite trajectory quality outputs."
                        )
                    normalized = torch.clamp((coordinates - 1.0) / 4.0, 0.0, 1.0)
                    embeddings[metric_name] = torch.cat(
                        [normalized, 1.0 - normalized],
                        dim=-1,
                    ).contiguous()
            finally:
                if run_dir.exists():
                    shutil.rmtree(run_dir, ignore_errors=True)

        return embeddings

    def score_candidate_batch(
        self,
        clean_wavs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        candidate_wavs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        utt_ids: list[str],
        sample_rate: int = 16000,
        split: str | None = None,
        keep_tmp: bool = False,
        metric_names: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        clean_list = _to_waveform_list(clean_wavs)
        candidate_list = _to_waveform_list(candidate_wavs)

        batch_size = len(utt_ids)
        if len(clean_list) != batch_size or len(candidate_list) != batch_size:
            raise ValueError("Waveform batch sizes must match `utt_ids`.")

        run_dir = self._build_run_dir(split)
        failed_metrics = self._build_failed_metrics()

        try:
            clean_dir = run_dir / "clean"
            candidate_dir = run_dir / "candidate"
            filenames = _write_waveforms(clean_dir, utt_ids, clean_list, sample_rate)
            _write_waveforms(candidate_dir, utt_ids, candidate_list, sample_rate)

            result = self._score_candidate(
                clean_dir,
                candidate_dir,
                filenames,
                failed_metrics,
                metric_names=metric_names,
            )
            result["meta"] = self._build_meta(
                run_dir=run_dir,
                failed_metrics=failed_metrics,
                sample_rate=sample_rate,
                keep_tmp=keep_tmp,
            )
            return result
        finally:
            if not keep_tmp and run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)

    def score_triplet_batch(
        self,
        clean_wavs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        noisy_wavs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        enhanced_wavs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        utt_ids: list[str],
        sample_rate: int = 16000,
        split: str | None = None,
        keep_tmp: bool = False,
    ) -> dict[str, Any]:
        clean_list = _to_waveform_list(clean_wavs)
        noisy_list = _to_waveform_list(noisy_wavs)
        enhanced_list = _to_waveform_list(enhanced_wavs)

        batch_size = len(utt_ids)
        if len(clean_list) != batch_size or len(noisy_list) != batch_size or len(enhanced_list) != batch_size:
            raise ValueError("Waveform batch sizes must match `utt_ids`.")

        run_dir = self._build_run_dir(split)
        failed_metrics = self._build_failed_metrics()

        try:
            clean_dir = run_dir / "clean"
            noisy_dir = run_dir / "noisy"
            enhanced_dir = run_dir / "enhanced"
            filenames = _write_waveforms(clean_dir, utt_ids, clean_list, sample_rate)
            _write_waveforms(noisy_dir, utt_ids, noisy_list, sample_rate)
            _write_waveforms(enhanced_dir, utt_ids, enhanced_list, sample_rate)

            result = {
                "clean": self._score_candidate(clean_dir, clean_dir, filenames, failed_metrics),
                "noisy": self._score_candidate(clean_dir, noisy_dir, filenames, failed_metrics),
                "enhanced": self._score_candidate(clean_dir, enhanced_dir, filenames, failed_metrics),
                "meta": self._build_meta(
                    run_dir=run_dir,
                    failed_metrics=failed_metrics,
                    sample_rate=sample_rate,
                    keep_tmp=keep_tmp,
                ),
            }
            return result
        finally:
            if not keep_tmp and run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)


def build_reward_fn(
    device: str | torch.device,
    voice_eval_root: str | Path = DEFAULT_VOICE_EVAL_ROOT,
    reward_config: dict[str, Any] | None = None,
    hf_cache_dir: str | Path | None = None,
    speechbert_model_path: str | Path | None = None,
    speaker_model_path: str | Path | None = None,
    local_files_only: bool | None = None,
    tmp_root: str | Path | None = None,
    sample_rate: int = 16000,
    speaker_model_type: str = "wavlm",
    speaker_code_path: str | Path | None = None,
) -> SpeechRewardPipeline:
    return SpeechRewardPipeline(
        device=device,
        voice_eval_root=voice_eval_root,
        reward_config=reward_config,
        hf_cache_dir=hf_cache_dir,
        speechbert_model_path=speechbert_model_path,
        speaker_model_path=speaker_model_path,
        speaker_model_type=speaker_model_type,
        speaker_code_path=speaker_code_path,
        local_files_only=local_files_only,
        tmp_root=tmp_root,
        sample_rate=sample_rate,
    )


def score_triplet_batch(
    clean_wavs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    noisy_wavs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    enhanced_wavs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    utt_ids: list[str],
    sample_rate: int = 16000,
    split: str | None = None,
    *,
    reward_fn: SpeechRewardPipeline | None = None,
    device: str | torch.device = "cpu",
    voice_eval_root: str | Path = DEFAULT_VOICE_EVAL_ROOT,
    reward_config: dict[str, Any] | None = None,
    hf_cache_dir: str | Path | None = None,
    speechbert_model_path: str | Path | None = None,
    speaker_model_path: str | Path | None = None,
    speaker_model_type: str = "wavlm",
    speaker_code_path: str | Path | None = None,
    local_files_only: bool | None = None,
    tmp_root: str | Path | None = None,
    keep_tmp: bool = False,
) -> dict[str, Any]:
    if reward_fn is None:
        reward_fn = build_reward_fn(
            device=device,
            voice_eval_root=voice_eval_root,
            reward_config=reward_config,
            hf_cache_dir=hf_cache_dir,
            speechbert_model_path=speechbert_model_path,
            speaker_model_path=speaker_model_path,
            speaker_model_type=speaker_model_type,
            speaker_code_path=speaker_code_path,
            local_files_only=local_files_only,
            tmp_root=tmp_root,
            sample_rate=sample_rate,
        )
    return reward_fn.score_triplet_batch(
        clean_wavs=clean_wavs,
        noisy_wavs=noisy_wavs,
        enhanced_wavs=enhanced_wavs,
        utt_ids=utt_ids,
        sample_rate=sample_rate,
        split=split,
        keep_tmp=keep_tmp,
    )


def score_candidate_batch(
    clean_wavs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    candidate_wavs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    utt_ids: list[str],
    sample_rate: int = 16000,
    split: str | None = None,
    *,
    reward_fn: SpeechRewardPipeline | None = None,
    device: str | torch.device = "cpu",
    voice_eval_root: str | Path = DEFAULT_VOICE_EVAL_ROOT,
    reward_config: dict[str, Any] | None = None,
    hf_cache_dir: str | Path | None = None,
    speechbert_model_path: str | Path | None = None,
    speaker_model_path: str | Path | None = None,
    speaker_model_type: str = "wavlm",
    speaker_code_path: str | Path | None = None,
    local_files_only: bool | None = None,
    tmp_root: str | Path | None = None,
    keep_tmp: bool = False,
) -> dict[str, Any]:
    if reward_fn is None:
        reward_fn = build_reward_fn(
            device=device,
            voice_eval_root=voice_eval_root,
            reward_config=reward_config,
            hf_cache_dir=hf_cache_dir,
            speechbert_model_path=speechbert_model_path,
            speaker_model_path=speaker_model_path,
            speaker_model_type=speaker_model_type,
            speaker_code_path=speaker_code_path,
            local_files_only=local_files_only,
            tmp_root=tmp_root,
            sample_rate=sample_rate,
        )
    return reward_fn.score_candidate_batch(
        clean_wavs=clean_wavs,
        candidate_wavs=candidate_wavs,
        utt_ids=utt_ids,
        sample_rate=sample_rate,
        split=split,
        keep_tmp=keep_tmp,
    )
