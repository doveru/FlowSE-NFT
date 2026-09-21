from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torchaudio.functional as F_audio
from vocos import Vocos

from flow_grpo.speech_paths import resolve_project_path

from .model import CFM, DiT
from .model.model_utils import get_tokenizer


EPS = torch.finfo(torch.float32).eps


def build_tokenizer(model_conf: dict[str, Any]) -> tuple[dict[str, int] | None, int]:
    tokenizer = model_conf["tokenizer"]
    tokenizer_path = resolve_project_path(model_conf["tokenizer_path"])
    vocab_char_map, vocab_size = get_tokenizer(
        str(tokenizer_path),
        tokenizer=tokenizer,
        tokenizer_path=str(tokenizer_path),
    )
    return vocab_char_map, vocab_size


def build_model(
    model_conf: dict[str, Any],
    vocab_char_map: dict[str, int] | None,
    vocab_size: int,
    device: torch.device | str | None = None,
) -> CFM:
    nnet = CFM(
        transformer=DiT(
            **model_conf["arch"],
            text_num_embeds=vocab_size,
            mel_dim=model_conf["mel_spec"]["n_mel_channels"],
        ),
        audio_drop_prob=model_conf.get("audio_drop_prob", 0.0),
        cond_drop_prob=model_conf.get("cond_drop_prob", 0.0),
        mel_spec_kwargs=model_conf["mel_spec"],
        vocab_char_map=vocab_char_map,
    ).eval()

    if device is not None:
        nnet = nnet.to(device)
    return nnet


def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str,
    model: torch.nn.Module | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    checkpoint_path = resolve_project_path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    return checkpoint


def load_vocoder(
    model_conf: dict[str, Any],
    device: torch.device | str,
    hf_cache_dir: str | None = None,
) -> torch.nn.Module:
    vocoder_conf = model_conf["vocoder"]
    vocoder_name = model_conf["mel_spec"].get("mel_spec_type", "vocos")
    is_local = vocoder_conf.get("is_local", False)
    local_path = resolve_project_path(vocoder_conf.get("local_path", ""))

    if vocoder_name == "vocos":
        if is_local:
            config_path = local_path / "config.yaml"
            model_path = local_path / "pytorch_model.bin"
        else:
            from huggingface_hub import hf_hub_download

            repo_id = "charactr/vocos-mel-24khz"
            config_path = Path(
                hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="config.yaml")
            )
            model_path = Path(
                hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="pytorch_model.bin")
            )

        vocoder = Vocos.from_hparams(str(config_path))
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        from vocos.feature_extractors import EncodecFeatures

        if isinstance(vocoder.feature_extractor, EncodecFeatures):
            encodec_parameters = {
                "feature_extractor.encodec." + key: value
                for key, value in vocoder.feature_extractor.encodec.state_dict().items()
            }
            state_dict.update(encodec_parameters)
        vocoder.load_state_dict(state_dict)
        return vocoder.eval().to(device)

    if vocoder_name == "bigvgan":
        try:
            from third_party.BigVGAN import bigvgan
        except ImportError as exc:
            raise ImportError(
                "BigVGAN is not available. Initialize the submodule before using bigvgan vocoder."
            ) from exc

        if is_local:
            vocoder = bigvgan.BigVGAN.from_pretrained(str(local_path), use_cuda_kernel=False)
        else:
            from huggingface_hub import snapshot_download

            downloaded_path = snapshot_download(
                repo_id="nvidia/bigvgan_v2_24khz_100band_256x",
                cache_dir=hf_cache_dir,
            )
            vocoder = bigvgan.BigVGAN.from_pretrained(downloaded_path, use_cuda_kernel=False)

        vocoder.remove_weight_norm()
        return vocoder.eval().to(device)

    raise ValueError(f"Unsupported vocoder type: {vocoder_name}")


def sample_enhance(
    model: CFM,
    cond: torch.Tensor,
    text: str | list[str],
    *,
    cond_type: str = "noisy",
    input_sample_rate: int | None = None,
    target_sample_rate: int | None = None,
    steps: int = 32,
    cfg_strength: float = 1.0,
    generator: torch.Generator | None = None,
    noise_init: torch.Tensor | None = None,
    solver: str = "dpm2",
    deterministic: bool = True,
    noise_level: float = 0.7,
    sigma_min: float | None = None,
    sigma_max: float = 1.0,
    deterministic_mask=None,
    sampling_time_grid=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if input_sample_rate is not None and target_sample_rate is not None and input_sample_rate != target_sample_rate:
        cond = F_audio.resample(cond, input_sample_rate, target_sample_rate)

    if isinstance(text, str):
        text_batch = [text]
    else:
        text_batch = text

    sample_kwargs = {
        "cond": cond,
        "steps": steps,
        "cfg_strength": cfg_strength,
        "generator": generator,
        "noise_init": noise_init,
        "solver": solver,
        "deterministic": deterministic,
        "noise_level": noise_level,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "deterministic_mask": deterministic_mask,
        "sampling_time_grid": sampling_time_grid,
        "return_time_grid": True,
    }

    if cond_type == "noisy":
        sample_kwargs["text"] = text_batch
        return model.sample(**sample_kwargs)
    if cond_type == "wotext":
        sample_kwargs["text"] = [" "] * cond.shape[0]
        sample_kwargs["drop_text"] = True
        return model.sample(**sample_kwargs)

    raise ValueError(f"Unsupported cond_type: {cond_type}")


def normalize_waveform(waveform: torch.Tensor, target_level: float = -25) -> torch.Tensor:
    rms = waveform.pow(2).mean(dim=-1, keepdim=True).sqrt()
    scalar = (10 ** (target_level / 20)) / (rms + EPS)
    return waveform * scalar


def decode_mel_to_wav(
    vocoder: torch.nn.Module,
    mel: torch.Tensor,
    *,
    vocoder_sample_rate: int,
    output_sample_rate: int,
    normalize_output: bool = True,
    target_num_samples: int | list[int] | torch.Tensor | None = None,
) -> torch.Tensor:
    mel = mel.transpose(-1, -2).to(torch.float32)
    waveform = vocoder.decode(mel)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    if normalize_output:
        waveform = normalize_waveform(waveform)

    if output_sample_rate != vocoder_sample_rate:
        waveform = F_audio.resample(waveform, vocoder_sample_rate, output_sample_rate)

    if target_num_samples is not None:
        if isinstance(target_num_samples, int):
            target_lengths = [target_num_samples] * waveform.shape[0]
        elif isinstance(target_num_samples, torch.Tensor):
            target_lengths = target_num_samples.tolist()
        else:
            target_lengths = list(target_num_samples)

        adjusted = []
        for wav, target_length in zip(waveform, target_lengths, strict=True):
            current_length = wav.shape[-1]
            if current_length < target_length:
                wav = torch.nn.functional.pad(wav, (0, target_length - current_length))
            else:
                wav = wav[:target_length]
            adjusted.append(wav)
        waveform = torch.stack(adjusted, dim=0)

    return waveform
