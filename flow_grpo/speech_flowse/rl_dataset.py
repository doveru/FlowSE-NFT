from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as F_audio
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from flow_grpo.speech_paths import resolve_project_path, resolve_audio_path

try:
    from .model.modules import MelSpec
except ModuleNotFoundError:
    mel_basis_cache: dict[str, torch.Tensor] = {}
    hann_window_cache: dict[str, torch.Tensor] = {}

    def get_bigvgan_mel_spectrogram(
        waveform,
        n_fft=1024,
        n_mel_channels=100,
        target_sample_rate=24000,
        hop_length=256,
        win_length=1024,
        fmin=0,
        fmax=None,
        center=False,
    ):
        from librosa.filters import mel as librosa_mel_fn

        device = waveform.device
        key = f"{n_fft}_{n_mel_channels}_{target_sample_rate}_{hop_length}_{win_length}_{fmin}_{fmax}_{device}"
        if key not in mel_basis_cache:
            mel = librosa_mel_fn(
                sr=target_sample_rate,
                n_fft=n_fft,
                n_mels=n_mel_channels,
                fmin=fmin,
                fmax=fmax,
            )
            mel_basis_cache[key] = torch.from_numpy(mel).float().to(device)
            hann_window_cache[key] = torch.hann_window(win_length).to(device)

        mel_basis = mel_basis_cache[key]
        hann_window = hann_window_cache[key]
        padding = (n_fft - hop_length) // 2
        waveform = F.pad(waveform.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)
        spec = torch.stft(
            waveform,
            n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=hann_window,
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
        mel_spec = torch.matmul(mel_basis, spec)
        return torch.log(torch.clamp(mel_spec, min=1e-5))

    def get_vocos_mel_spectrogram(
        waveform,
        n_fft=1024,
        n_mel_channels=100,
        target_sample_rate=24000,
        hop_length=256,
        win_length=1024,
    ):
        mel_stft = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mel_channels,
            power=1,
            center=True,
            normalized=False,
            norm=None,
        ).to(waveform.device)
        if waveform.ndim == 3:
            waveform = waveform.squeeze(1)
        mel = mel_stft(waveform)
        return mel.clamp(min=1e-5).log()

    class MelSpec(torch.nn.Module):
        def __init__(
            self,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mel_channels=100,
            target_sample_rate=24000,
            mel_spec_type="vocos",
        ):
            super().__init__()
            self.n_fft = n_fft
            self.hop_length = hop_length
            self.win_length = win_length
            self.n_mel_channels = n_mel_channels
            self.target_sample_rate = target_sample_rate
            if mel_spec_type not in {"vocos", "bigvgan"}:
                raise ValueError(f"Unsupported mel_spec_type: {mel_spec_type}")
            self.extractor = get_vocos_mel_spectrogram if mel_spec_type == "vocos" else get_bigvgan_mel_spectrogram
            self.register_buffer("dummy", torch.tensor(0), persistent=False)

        def forward(self, wav):
            if self.dummy.device != wav.device:
                self.to(wav.device)
            return self.extractor(
                waveform=wav,
                n_fft=self.n_fft,
                n_mel_channels=self.n_mel_channels,
                target_sample_rate=self.target_sample_rate,
                hop_length=self.hop_length,
                win_length=self.win_length,
            )


DEFAULT_MEL_SPEC_CONF = {
    "n_fft": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "n_mel_channels": 100,
    "target_sample_rate": 24000,
    "mel_spec_type": "vocos",
}


def read_manifest(manifest_path: str | Path, data_root: str | Path | None = None) -> list[dict]:
    manifest_path = resolve_project_path(manifest_path)
    entries = []
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {manifest_path}:{line_number}") from exc
            for key in ("noisy_path", "clean_path"):
                if not isinstance(entry.get(key), str) or not entry[key].strip():
                    raise ValueError(f"Missing or invalid {key} at {manifest_path}:{line_number}")
                entry[key] = str(resolve_audio_path(entry[key], data_root))
            entries.append(entry)
    if not entries:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return entries


def load_first_channel_audio(path: str | Path, sample_rate: int) -> torch.Tensor:
    waveform, source_sample_rate = torchaudio.load(str(path))
    waveform = waveform[:1].to(torch.float32).contiguous()
    if source_sample_rate != sample_rate:
        waveform = F_audio.resample(waveform, source_sample_rate, sample_rate)
    return waveform


def pad_or_trim(waveform: torch.Tensor, target_num_samples: int) -> torch.Tensor:
    num_samples = waveform.shape[-1]
    if num_samples < target_num_samples:
        return F.pad(waveform, (0, target_num_samples - num_samples))
    if num_samples > target_num_samples:
        return waveform[..., :target_num_samples]
    return waveform


def pad_stack_2d(tensors: list[torch.Tensor]) -> torch.Tensor:
    max_length = max(tensor.shape[-1] for tensor in tensors)
    padded = [F.pad(tensor, (0, max_length - tensor.shape[-1])) for tensor in tensors]
    return torch.stack(padded, dim=0)


def pad_stack_3d(tensors: list[torch.Tensor]) -> torch.Tensor:
    max_length = max(tensor.shape[-1] for tensor in tensors)
    padded = [F.pad(tensor, (0, max_length - tensor.shape[-1])) for tensor in tensors]
    return torch.stack(padded, dim=0)


class RLDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        sample_rate: int = 16000,
        mel_spec_conf: dict | None = None,
        pad_to_chunk: bool = True,
        data_root: str | Path | None = None,
    ):
        super().__init__()
        self.manifest_path = resolve_project_path(manifest_path)
        self.entries = read_manifest(self.manifest_path, data_root=data_root)
        self.sample_rate = sample_rate
        self.pad_to_chunk = pad_to_chunk
        self.mel_spec_conf = dict(DEFAULT_MEL_SPEC_CONF if mel_spec_conf is None else mel_spec_conf)
        self.mel_spectrogram = MelSpec(**self.mel_spec_conf)
        self.mel_sample_rate = self.mel_spec_conf["target_sample_rate"]
        self.chunk_num_samples = max(
            int(round((entry["chunk_end_sec"] - entry["chunk_start_sec"]) * self.sample_rate))
            for entry in self.entries
        )
        if self.chunk_num_samples <= 0:
            raise ValueError(f"Manifest contains invalid chunk durations: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.entries)

    def _segment_samples(self, entry: dict) -> tuple[int, int]:
        start_sample = int(round(entry["chunk_start_sec"] * self.sample_rate))
        end_sample = int(round(entry["chunk_end_sec"] * self.sample_rate))
        if end_sample < start_sample:
            raise ValueError(f"Invalid segment in manifest entry: {entry}")
        return start_sample, end_sample

    def _slice_audio(self, waveform: torch.Tensor, start_sample: int, end_sample: int) -> torch.Tensor:
        return waveform[..., start_sample:end_sample].contiguous()

    def _to_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        mel_waveform = waveform
        if self.sample_rate != self.mel_sample_rate:
            mel_waveform = F_audio.resample(mel_waveform, self.sample_rate, self.mel_sample_rate)
        mel_spec = self.mel_spectrogram(mel_waveform)
        return mel_spec.squeeze(0).contiguous()

    def __getitem__(self, index: int) -> dict:
        entry = self.entries[index]
        start_sample, end_sample = self._segment_samples(entry)

        noisy_wav = load_first_channel_audio(entry["noisy_path"], self.sample_rate)
        clean_wav = load_first_channel_audio(entry["clean_path"], self.sample_rate)

        noisy_wav = self._slice_audio(noisy_wav, start_sample, end_sample)
        clean_wav = self._slice_audio(clean_wav, start_sample, end_sample)

        usable_num_samples = min(noisy_wav.shape[-1], clean_wav.shape[-1])
        noisy_wav = noisy_wav[..., :usable_num_samples]
        clean_wav = clean_wav[..., :usable_num_samples]

        if self.pad_to_chunk:
            noisy_wav = pad_or_trim(noisy_wav, self.chunk_num_samples)
            clean_wav = pad_or_trim(clean_wav, self.chunk_num_samples)

        raw_text = entry.get("text", "") or ""
        sample = {
            "utt_id": entry["utt_id"],
            "source_utt_id": entry.get("source_utt_id", Path(entry["noisy_path"]).stem),
            "split": entry.get("split", ""),
            "noisy_path": entry["noisy_path"],
            "clean_path": entry["clean_path"],
            "raw_text": raw_text,
            "text": [],
            "noisy_wav": noisy_wav,
            "clean_wav": clean_wav,
            "noisy_mel": self._to_mel(noisy_wav),
            "clean_mel": self._to_mel(clean_wav),
            "noisy_num_samples": usable_num_samples,
            "clean_num_samples": usable_num_samples,
            "chunk_start_sec": float(entry["chunk_start_sec"]),
            "chunk_end_sec": float(entry["chunk_end_sec"]),
        }
        return sample


def rl_collate_fn(batch: list[dict]) -> dict:
    if not batch:
        raise ValueError("Empty batch is not supported.")

    return {
        "utt_id": [item["utt_id"] for item in batch],
        "source_utt_id": [item["source_utt_id"] for item in batch],
        "split": [item["split"] for item in batch],
        "noisy_path": [item["noisy_path"] for item in batch],
        "clean_path": [item["clean_path"] for item in batch],
        "raw_text": [item["raw_text"] for item in batch],
        "text": [item["text"] for item in batch],
        "noisy_wav": pad_stack_3d([item["noisy_wav"] for item in batch]),
        "clean_wav": pad_stack_3d([item["clean_wav"] for item in batch]),
        "noisy_mel": pad_stack_2d([item["noisy_mel"] for item in batch]),
        "clean_mel": pad_stack_2d([item["clean_mel"] for item in batch]),
        "noisy_num_samples": torch.tensor([item["noisy_num_samples"] for item in batch], dtype=torch.long),
        "clean_num_samples": torch.tensor([item["clean_num_samples"] for item in batch], dtype=torch.long),
        "chunk_start_sec": torch.tensor([item["chunk_start_sec"] for item in batch], dtype=torch.float32),
        "chunk_end_sec": torch.tensor([item["chunk_end_sec"] for item in batch], dtype=torch.float32),
    }


def make_rl_loader(
    manifest_path: str | Path,
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
    distributed: bool = False,
    sample_rate: int = 16000,
    mel_spec_conf: dict | None = None,
    pad_to_chunk: bool = True,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    drop_last: bool = False,
    data_root: str | Path | None = None,
):
    dataset = RLDataset(
        manifest_path=manifest_path,
        data_root=data_root,
        sample_rate=sample_rate,
        mel_spec_conf=mel_spec_conf,
        pad_to_chunk=pad_to_chunk,
    )

    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
    elif shuffle:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=rl_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=drop_last,
    )
    return sampler, loader
