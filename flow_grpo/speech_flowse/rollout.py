from __future__ import annotations


import hashlib
from typing import Any

import torch
import torch.nn.functional as F
import torchaudio.functional as F_audio
from tqdm import tqdm

from flow_grpo.speech_nft_core import select_nft_target_sample
from flow_grpo.speech_diagnostics import diagnostic_stage

from .runtime import decode_mel_to_wav, sample_enhance


def compute_trajectory_final_cosine(
    trajectory: torch.Tensor,
    final_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-sample CSS similarity and adjacent information gain.

    The full trajectory is used only while it already exists inside rollout.
    Returned tensors contain scalar summaries shaped [B,T] and [B,T-1].
    """
    if trajectory.ndim < 3:
        raise ValueError(f"Expected trajectory shaped [T,B,...], got {tuple(trajectory.shape)}.")
    if final_state.ndim != trajectory.ndim - 1:
        raise ValueError(
            f"Expected final state with {trajectory.ndim - 1} dimensions, got {final_state.ndim}."
        )
    if tuple(trajectory.shape[1:]) != tuple(final_state.shape):
        raise ValueError(
            f"Trajectory/final-state shape mismatch: trajectory={tuple(trajectory.shape)}, "
            f"final={tuple(final_state.shape)}."
        )

    batch_size = int(trajectory.shape[1])
    final_flat = final_state.detach().to(torch.float32).reshape(batch_size, -1)
    # Convert one state at a time so diagnostics do not create a second
    # float32-sized copy of the complete trajectory.
    similarity = torch.stack(
        [
            F.cosine_similarity(
                state.detach().to(torch.float32).reshape(batch_size, -1),
                final_flat,
                dim=-1,
                eps=1e-8,
            )
            for state in trajectory.unbind(dim=0)
        ],
        dim=1,
    ).contiguous()
    information_gain = torch.abs(similarity[:, 1:] - similarity[:, :-1])
    return similarity, information_gain


def compute_reward_embedding_cosine(
    embeddings: dict[str, torch.Tensor],
    final_embeddings: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compare a batch of reward-aligned embeddings with final-output embeddings."""
    if set(embeddings) != set(final_embeddings):
        raise ValueError("Current and final reward embedding dictionaries must have identical keys.")

    similarities: dict[str, torch.Tensor] = {}
    for metric_name, embedding in embeddings.items():
        final_embedding = final_embeddings[metric_name]
        if embedding.shape != final_embedding.shape:
            raise ValueError(
                f"Reward embedding shape mismatch for {metric_name!r}: "
                f"current={tuple(embedding.shape)}, final={tuple(final_embedding.shape)}."
            )
        similarities[metric_name] = F.cosine_similarity(
            embedding.to(torch.float32),
            final_embedding.to(torch.float32),
            dim=-1,
            eps=1e-8,
        ).cpu().contiguous()
    return similarities


def predict_final_state_from_flow(
    model: torch.nn.Module,
    state: torch.Tensor,
    cond_mel: torch.Tensor,
    text: list[str],
    *,
    time_value: float,
    target_time: float,
    cfg_strength: float,
    drop_text: bool,
) -> torch.Tensor:
    """Linearly project one sampled state to the rollout endpoint with the model flow."""
    if float(time_value) > float(target_time) + 1e-8:
        raise ValueError("Trajectory state time cannot be after the requested final time.")
    batch_size = int(state.shape[0])
    time = torch.full(
        (batch_size,),
        float(time_value),
        device=state.device,
        dtype=state.dtype,
    )
    conditional_flow = model.predict_flow(
        x=state,
        cond=cond_mel,
        text=text,
        time=time,
        drop_audio_cond=False,
        drop_text=drop_text,
    )
    if float(cfg_strength) < 1e-5:
        guided_flow = conditional_flow
    else:
        unconditional_flow = model.predict_flow(
            x=state,
            cond=cond_mel,
            text=text,
            time=time,
            drop_audio_cond=True,
            drop_text=True,
        )
        guided_flow = conditional_flow + (conditional_flow - unconditional_flow) * float(cfg_strength)
    return state + (float(target_time) - float(time_value)) * guided_flow


def _repeat_tensor(tensor: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats <= 0:
        raise ValueError("`repeats` must be positive.")
    return tensor.repeat_interleave(repeats, dim=0)


def _repeat_list(values: list[Any], repeats: int) -> list[Any]:
    if repeats <= 0:
        raise ValueError("`repeats` must be positive.")
    return [value for value in values for _ in range(repeats)]


def _resolve_split_name(split_values: list[str]) -> str:
    for value in split_values:
        if value:
            return str(value)
    return "default"


def _ensure_waveform_batch_2d(waveform: torch.Tensor, field_name: str) -> torch.Tensor:
    if waveform.ndim == 3:
        if waveform.shape[1] != 1:
            raise ValueError(f"`{field_name}` must have shape [B,1,T] when 3D, got {tuple(waveform.shape)}")
        return waveform.squeeze(1).contiguous()
    if waveform.ndim == 2:
        return waveform.contiguous()
    raise ValueError(f"`{field_name}` must have shape [B,1,T] or [B,T], got {tuple(waveform.shape)}")


def _prepare_cond_mel(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    input_sample_rate: int,
    target_sample_rate: int,
) -> torch.Tensor:
    cond = waveform
    if input_sample_rate != target_sample_rate:
        cond = F_audio.resample(cond, input_sample_rate, target_sample_rate)
    cond = model.mel_spec(cond)
    return cond.permute(0, 2, 1).contiguous()


def _build_noise_init(cond_mel: torch.Tensor, seeds: torch.Tensor) -> torch.Tensor:
    if cond_mel.ndim != 3:
        raise ValueError(f"`cond_mel` must have shape [B,N,D], got {tuple(cond_mel.shape)}")

    noise_parts = []
    for index, seed in enumerate(seeds.tolist()):
        generator = torch.Generator(device=cond_mel.device)
        generator.manual_seed(int(seed))
        noise = torch.randn(
            (1, cond_mel.shape[1], cond_mel.shape[2]),
            device=cond_mel.device,
            dtype=cond_mel.dtype,
            generator=generator,
        )
        noise_parts.append(noise)
    return torch.cat(noise_parts, dim=0)


def _resolve_deterministic_mask(
    deterministic_mask,
    *,
    steps: int,
    default_deterministic: bool,
) -> list[bool]:
    if steps <= 0:
        raise ValueError("`steps` must be positive.")
    if deterministic_mask is None:
        return [bool(default_deterministic)] * int(steps)

    mask = torch.as_tensor(deterministic_mask, dtype=torch.bool).reshape(-1)
    if int(mask.numel()) != int(steps):
        raise ValueError(f"`deterministic_mask` must have length {steps}, got {int(mask.numel())}.")
    return [bool(value) for value in mask.tolist()]


def _normalize_generated_mel(mel: torch.Tensor, mel_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    if mel.ndim != 3:
        raise ValueError(f"Expected generated mel with shape [B,N,D] or [B,D,N], got {tuple(mel.shape)}")

    if mel.shape[-1] == mel_dim:
        mel_for_decode = mel.contiguous()
        mel_for_output = mel.transpose(1, 2).contiguous()
        return mel_for_decode, mel_for_output

    if mel.shape[1] == mel_dim:
        mel_for_output = mel.contiguous()
        mel_for_decode = mel.transpose(1, 2).contiguous()
        return mel_for_decode, mel_for_output

    raise ValueError(
        f"Unable to infer mel dimension from shape {tuple(mel.shape)} with expected mel_dim={mel_dim}"
    )


def _trim_waveforms(waveforms: torch.Tensor, lengths: torch.Tensor) -> list[torch.Tensor]:
    trimmed: list[torch.Tensor] = []
    for waveform, length in zip(waveforms, lengths, strict=True):
        target_length = int(length.item())
        trimmed.append(waveform[..., :target_length].detach().cpu().contiguous())
    return trimmed


def _cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().contiguous()


def _stable_candidate_seed(base_seed: int, seed_offset: int, candidate_id: str) -> int:
    payload = f"{int(base_seed)}::{int(seed_offset)}::{candidate_id}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**31 - 1)


class SpeechRolloutEngine:
    def __init__(
        self,
        old_model: torch.nn.Module,
        vocoder: torch.nn.Module,
        reward_fn,
        *,
        cond_type: str = "wotext",
        input_sample_rate: int = 16000,
        vocoder_sample_rate: int = 24000,
        output_sample_rate: int = 16000,
        steps: int = 32,
        cfg_strength: float = 1.0,
        base_seed: int = 1234,
        solver: str = "dpm2",
        deterministic: bool = True,
        noise_level: float = 0.7,
        sigma_min: float | None = None,
        sigma_max: float = 1.0,
        time_grid=None,
        default_deterministic_mask=None,
        nft_target_time: float | None = None,
        collect_trajectory_similarity: bool = False,
    ):
        if cond_type != "wotext":
            raise ValueError(
                f"Speech NFT rollout is fixed to `wotext`; got cond_type={cond_type!r}."
            )

        self.model = old_model.eval()
        self.vocoder = vocoder.eval()
        self.reward_fn = reward_fn
        self.cond_type = cond_type
        self.input_sample_rate = int(input_sample_rate)
        self.vocoder_sample_rate = int(vocoder_sample_rate)
        self.output_sample_rate = int(output_sample_rate)
        self.steps = int(steps)
        self.cfg_strength = float(cfg_strength)
        self.solver = str(solver)
        self.deterministic = bool(deterministic)
        self.noise_level = float(noise_level)
        self.sigma_min = None if sigma_min is None else float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.time_grid = None if time_grid is None else [float(value) for value in time_grid]
        if self.time_grid is not None and len(self.time_grid) != self.steps + 1:
            raise ValueError(
                f"`time_grid` must contain steps + 1 = {self.steps + 1} values, got {len(self.time_grid)}."
            )
        self.default_deterministic_mask = (
            None
            if default_deterministic_mask is None
            else _resolve_deterministic_mask(
                default_deterministic_mask,
                steps=self.steps,
                default_deterministic=self.deterministic,
            )
        )
        self.nft_target_time = None if nft_target_time is None else float(nft_target_time)
        if self.nft_target_time is not None and not (0.0 < self.nft_target_time <= 1.0):
            raise ValueError("`nft_target_time` must be in (0, 1].")
        self.collect_trajectory_similarity = bool(collect_trajectory_similarity)
        self.base_seed = int(base_seed)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def rollout_batch(
        self,
        batch: dict[str, Any],
        num_candidates: int,
        seed_offset: int = 0,
        keep_reward_tmp: bool = False,
        keep_waveforms: bool = True,
        deterministic_mask=None,
        reward_metric_names=None,
    ) -> dict[str, Any]:
        prepared_batch = self.prepare_rollout_batch(
            batch,
            num_candidates=num_candidates,
            seed_offset=seed_offset,
            keep_waveforms=keep_waveforms,
            deterministic_mask=deterministic_mask,
            trajectory_metric_names=reward_metric_names,
        )
        diagnostic_stage("rollout.reward.begin", seed_offset=seed_offset)
        result = self.score_prepared_batch(
            prepared_batch,
            keep_reward_tmp=keep_reward_tmp,
            reward_metric_names=reward_metric_names,
        )
        diagnostic_stage("rollout.reward.end", seed_offset=seed_offset)
        return result

    @torch.no_grad()
    def prepare_rollout_batch(
        self,
        batch: dict[str, Any],
        num_candidates: int,
        seed_offset: int = 0,
        keep_waveforms: bool = True,
        deterministic_mask=None,
        trajectory_metric_names=None,
    ) -> dict[str, Any]:
        if num_candidates <= 0:
            raise ValueError("`num_candidates` must be positive.")

        required_keys = {
            "utt_id",
            "source_utt_id",
            "split",
            "noisy_wav",
            "clean_wav",
            "clean_num_samples",
        }
        missing = sorted(required_keys - set(batch))
        if missing:
            raise KeyError(f"Missing rollout batch keys: {missing}")

        batch_size = len(batch["utt_id"])
        if batch_size == 0:
            raise ValueError("Empty rollout batch is not supported.")

        resolved_deterministic_mask = (
            self.default_deterministic_mask if deterministic_mask is None else deterministic_mask
        )
        deterministic_mask_values = _resolve_deterministic_mask(
            resolved_deterministic_mask,
            steps=self.steps,
            default_deterministic=self.deterministic,
        )
        sde_timestep_mask = [not value for value in deterministic_mask_values]

        repeated_utt_ids = _repeat_list(list(batch["utt_id"]), num_candidates)
        repeated_source_utt_ids = _repeat_list(list(batch["source_utt_id"]), num_candidates)
        repeated_split = _repeat_list(list(batch["split"]), num_candidates)
        repeated_raw_text = _repeat_list(list(batch.get("raw_text", [""] * batch_size)), num_candidates)
        repeated_text = [" "] * (batch_size * num_candidates)

        candidate_index = torch.arange(num_candidates, dtype=torch.long).repeat(batch_size)
        candidate_ids = [
            f"{utt_id}__cand_{candidate_idx:03d}"
            for utt_id, candidate_idx in zip(repeated_utt_ids, candidate_index.tolist(), strict=True)
        ]
        seeds = torch.tensor(
            [
                _stable_candidate_seed(self.base_seed, seed_offset, candidate_id)
                for candidate_id in candidate_ids
            ],
            dtype=torch.long,
        )

        noisy_wav = _repeat_tensor(batch["noisy_wav"].to(self.device), num_candidates)
        noisy_mel = _repeat_tensor(batch["noisy_mel"], num_candidates).to(torch.float32)
        clean_wav = _repeat_tensor(batch["clean_wav"], num_candidates).to(torch.float32)
        clean_num_samples = _repeat_tensor(batch["clean_num_samples"], num_candidates).to(torch.long)

        noisy_cond = _ensure_waveform_batch_2d(noisy_wav, "noisy_wav")
        cond_mel = _prepare_cond_mel(
            self.model,
            noisy_cond,
            input_sample_rate=self.input_sample_rate,
            target_sample_rate=self.vocoder_sample_rate,
        )
        noise_init = _build_noise_init(cond_mel, seeds.to(device=cond_mel.device))

        diagnostic_stage("rollout.sample.begin", seed_offset=seed_offset)
        generated_sample, trajectory, ode_time_grid = sample_enhance(
            self.model,
            noisy_cond,
            repeated_text,
            cond_type=self.cond_type,
            input_sample_rate=self.input_sample_rate,
            target_sample_rate=self.vocoder_sample_rate,
            steps=self.steps,
            cfg_strength=self.cfg_strength,
            noise_init=noise_init,
            solver=self.solver,
            deterministic=self.deterministic,
            noise_level=self.noise_level,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            deterministic_mask=deterministic_mask_values,
            sampling_time_grid=self.time_grid,
        )
        diagnostic_stage("rollout.sample.end", seed_offset=seed_offset)
        generated_mel_for_decode, generated_mel = _normalize_generated_mel(
            generated_sample,
            mel_dim=int(self.model.num_channels),
        )
        nft_target_sample, effective_nft_target_time = select_nft_target_sample(
            generated_sample,
            trajectory,
            ode_time_grid,
            self.nft_target_time,
        )
        if nft_target_sample is generated_sample:
            nft_target_mel = generated_mel
        else:
            _, nft_target_mel = _normalize_generated_mel(
                nft_target_sample,
                mel_dim=int(self.model.num_channels),
            )

        generated_num_samples = clean_num_samples.clone()
        diagnostic_stage("rollout.decode.begin", seed_offset=seed_offset)
        generated_wav = decode_mel_to_wav(
            self.vocoder,
            generated_mel_for_decode,
            vocoder_sample_rate=self.vocoder_sample_rate,
            output_sample_rate=self.output_sample_rate,
            target_num_samples=generated_num_samples.tolist(),
        )

        generated_wav_cpu = _cpu_tensor(generated_wav)
        generated_num_samples_cpu = _cpu_tensor(generated_num_samples)
        diagnostic_stage("rollout.decode.end", seed_offset=seed_offset)

        trajectory_final_similarity: dict[str, torch.Tensor] | None = None
        trajectory_information_gain: dict[str, torch.Tensor] | None = None
        trajectory_embedding_kinds: dict[str, str] | None = None
        if self.collect_trajectory_similarity:
            final_waveforms = _trim_waveforms(generated_wav_cpu, generated_num_samples_cpu)
            final_embeddings = self.reward_fn.extract_reward_aligned_embeddings(
                final_waveforms,
                sample_rate=self.output_sample_rate,
                metric_names=trajectory_metric_names,
            )
            trajectory_embedding_kinds = self.reward_fn.trajectory_embedding_kinds(
                trajectory_metric_names
            )
            per_metric_similarities: dict[str, list[torch.Tensor]] = {
                metric_name: [] for metric_name in final_embeddings
            }
            endpoint_time = float(ode_time_grid[-1].item())
            trajectory_states = trajectory.unbind(dim=0)
            for state_index, (state, time_value_tensor) in enumerate(
                tqdm(
                    zip(trajectory_states, ode_time_grid, strict=True),
                    total=len(trajectory_states),
                    desc="Reward-aligned states",
                    unit="state",
                    position=1,
                    leave=False,
                    dynamic_ncols=True,
                    miniters=1,
                    disable=(
                        torch.distributed.is_available()
                        and torch.distributed.is_initialized()
                        and torch.distributed.get_rank() != 0
                    ),
                )
            ):
                time_value = float(time_value_tensor.item())
                if state_index == len(trajectory_states) - 1:
                    current_embeddings = final_embeddings
                else:
                    predicted_final_state = predict_final_state_from_flow(
                        self.model,
                        state,
                        cond_mel,
                        repeated_text,
                        time_value=time_value,
                        target_time=endpoint_time,
                        cfg_strength=self.cfg_strength,
                        drop_text=(self.cond_type == "wotext"),
                    )
                    predicted_mel_for_decode, _ = _normalize_generated_mel(
                        predicted_final_state,
                        mel_dim=int(self.model.num_channels),
                    )
                    predicted_wav = decode_mel_to_wav(
                        self.vocoder,
                        predicted_mel_for_decode,
                        vocoder_sample_rate=self.vocoder_sample_rate,
                        output_sample_rate=self.output_sample_rate,
                        target_num_samples=generated_num_samples.tolist(),
                    )
                    endpoint_waveforms = _trim_waveforms(
                        _cpu_tensor(predicted_wav),
                        generated_num_samples_cpu,
                    )
                    current_embeddings = self.reward_fn.extract_reward_aligned_embeddings(
                        endpoint_waveforms,
                        sample_rate=self.output_sample_rate,
                        metric_names=trajectory_metric_names,
                    )
                current_similarities = compute_reward_embedding_cosine(
                    current_embeddings,
                    final_embeddings,
                )
                for metric_name, similarity in current_similarities.items():
                    per_metric_similarities[metric_name].append(similarity)

            trajectory_final_similarity = {
                metric_name: torch.stack(values, dim=1).contiguous()
                for metric_name, values in per_metric_similarities.items()
            }
            trajectory_information_gain = {
                metric_name: torch.abs(similarity[:, 1:] - similarity[:, :-1]).contiguous()
                for metric_name, similarity in trajectory_final_similarity.items()
            }

        # The NFT endpoint was cloned when necessary and trajectory diagnostics
        # retain only scalar summaries, so the full trajectory can now be freed.
        del trajectory
        prepared = {
            "utt_id": repeated_utt_ids,
            "source_utt_id": repeated_source_utt_ids,
            "split": repeated_split,
            "raw_text": repeated_raw_text,
            "text": repeated_text,
            "candidate_id": candidate_ids,
            "candidate_index": candidate_index.cpu(),
            "seed": seeds.cpu(),
            "noisy_mel": noisy_mel,
            "generated_mel": _cpu_tensor(nft_target_mel),
            "nft_target_time": float(effective_nft_target_time),
            "ode_time_grid": ode_time_grid.detach().cpu().to(torch.float32).contiguous(),
            "deterministic_mask": torch.tensor(deterministic_mask_values, dtype=torch.bool),
            "sde_timestep_mask": torch.tensor(sde_timestep_mask, dtype=torch.bool),
            "batch_size": batch_size,
            "num_candidates": num_candidates,
            "cond_type": self.cond_type,
            "_reward_payload": {
                "clean_wavs": _trim_waveforms(clean_wav, clean_num_samples),
                "candidate_wavs": _trim_waveforms(generated_wav_cpu, generated_num_samples_cpu),
                "utt_ids": candidate_ids,
                "sample_rate": self.output_sample_rate,
                "split": _resolve_split_name(repeated_split),
            },
        }
        if trajectory_final_similarity is not None:
            prepared["trajectory_final_similarity"] = {
                metric_name: _cpu_tensor(similarity)
                for metric_name, similarity in trajectory_final_similarity.items()
            }
            prepared["trajectory_information_gain"] = {
                metric_name: _cpu_tensor(information_gain)
                for metric_name, information_gain in trajectory_information_gain.items()
            }
            prepared["trajectory_embedding_kinds"] = dict(trajectory_embedding_kinds or {})
        if keep_waveforms:
            prepared["generated_wav"] = generated_wav_cpu
            prepared["generated_num_samples"] = generated_num_samples_cpu
        return prepared

    def score_prepared_batch(
        self,
        prepared_batch: dict[str, Any],
        *,
        keep_reward_tmp: bool = False,
        reward_metric_names=None,
    ) -> dict[str, Any]:
        reward_payload = prepared_batch.pop("_reward_payload", None)
        if reward_payload is None:
            raise KeyError("Prepared rollout batch is missing `_reward_payload`.")

        reward_result = self.reward_fn.score_candidate_batch(
            clean_wavs=reward_payload["clean_wavs"],
            candidate_wavs=reward_payload["candidate_wavs"],
            utt_ids=reward_payload["utt_ids"],
            sample_rate=int(reward_payload["sample_rate"]),
            split=reward_payload["split"],
            keep_tmp=keep_reward_tmp,
            metric_names=reward_metric_names,
        )
        reward_breakdown = {
            "raw": {
                metric_name: torch.tensor(values, dtype=torch.float32)
                for metric_name, values in reward_result["raw"].items()
            },
            "norm": {
                metric_name: torch.tensor(values, dtype=torch.float32)
                for metric_name, values in reward_result["norm"].items()
            },
        }

        prepared_batch["reward"] = torch.tensor(reward_result["avg"], dtype=torch.float32)
        prepared_batch["reward_breakdown"] = reward_breakdown
        return prepared_batch


def build_rollout_engine(
    old_model: torch.nn.Module,
    vocoder: torch.nn.Module,
    reward_fn,
    *,
    cond_type: str = "wotext",
    input_sample_rate: int = 16000,
    vocoder_sample_rate: int = 24000,
    output_sample_rate: int = 16000,
    steps: int = 32,
    cfg_strength: float = 1.0,
    base_seed: int = 1234,
    solver: str = "dpm2",
    deterministic: bool = True,
    noise_level: float = 0.7,
    sigma_min: float | None = None,
    sigma_max: float = 1.0,
    time_grid=None,
    default_deterministic_mask=None,
    nft_target_time: float | None = None,
    collect_trajectory_similarity: bool = False,
) -> SpeechRolloutEngine:
    return SpeechRolloutEngine(
        old_model=old_model,
        vocoder=vocoder,
        reward_fn=reward_fn,
        cond_type=cond_type,
        input_sample_rate=input_sample_rate,
        vocoder_sample_rate=vocoder_sample_rate,
        output_sample_rate=output_sample_rate,
        steps=steps,
        cfg_strength=cfg_strength,
        base_seed=base_seed,
        solver=solver,
        deterministic=deterministic,
        noise_level=noise_level,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        time_grid=time_grid,
        default_deterministic_mask=default_deterministic_mask,
        nft_target_time=nft_target_time,
        collect_trajectory_similarity=collect_trajectory_similarity,
    )
