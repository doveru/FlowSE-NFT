from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from flow_nft.speech_paths import PROJECT_ROOT, resolve_project_path

from flow_nft.speech_flowse import policy_snapshot as flowse_policy_snapshot
from flow_nft.speech_flowse import rewards as flowse_rewards
from flow_nft.speech_flowse import rl_dataset as flowse_dataset
from flow_nft.speech_flowse import rollout as flowse_rollout
from flow_nft.speech_flowse import runtime as flowse_runtime
from flow_nft.speech_nft_core import (
    PerConditionStatTracker,
    RolloutBatch,
    SpeechMixedSamplingState,
    aggregate_reward_advantages,
    build_eval_deterministic_mask,
    compute_advantage_sign_flip_counts,
    compute_gd2po_group_keep_ratios,
    compute_low_std_group_keep_mask,
    compute_multi_reward_pure_nft_terms,
    compute_pure_nft_terms,
    compute_reward_advantage_conflict_stats,
    compute_reward_advantage_pairwise_stats,
    compute_reward_conflict_filter_diagnostics,
    compute_reward_advantage_snr_keep_mask,
    expand_sampled_time,
    masked_normalize_aggregated_advantages,
    masked_rms_scale_aggregated_advantages,
    normalize_reward_weights,
    summarize_advantage_sign_flip_counts,
)


@dataclass
class SpeechModelBundle:

    current_model: torch.nn.Module
    old_model: torch.nn.Module
    ref_model: torch.nn.Module
    vocoder: torch.nn.Module
    lora_enabled: bool = False
    lora_strategy: str = "disabled"
    lora_target_modules: tuple[str, ...] = ()
    trainable_param_count: int = 0
    total_param_count: int = 0


class SpeechBackendAdapter:

    def __init__(self, config, device: torch.device, rank: int, world_size: int):
        self.config = config
        self.device = device
        self.rank = int(rank)
        self.world_size = int(world_size)

        self.project_root = PROJECT_ROOT
        self.speech_conf = config.speech

        self.snapshot_manager = None
        self.train_rollout_engine = None
        self.eval_rollout_engine = None
        self.reward_fn = None
        self.train_reward_metrics: list[str] = []
        self.eval_reward_metrics: list[str] = []
        self.object_collective_group = None
        self._lora_enabled = False
        self._lora_strategy = "disabled"
        self._lora_target_modules: tuple[str, ...] = ()
        self.mixed_sampling_state: SpeechMixedSamplingState | None = None

    def _resolve_path(self, path_like: str | Path) -> Path:
        return resolve_project_path(path_like)

    @staticmethod
    def _count_parameters(model: torch.nn.Module) -> tuple[int, int]:
        total = 0
        trainable = 0
        for parameter in model.parameters():
            numel = int(parameter.numel())
            total += numel
            if bool(parameter.requires_grad):
                trainable += numel
        return trainable, total

    @staticmethod
    def _filter_lora_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        filtered = {}
        for key, value in state_dict.items():
            if "lora_" not in key:
                continue
            if not isinstance(value, torch.Tensor):
                continue
            filtered[key] = value.detach().cpu().clone()
        return filtered

    @staticmethod
    def _load_partial_state_dict(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> int:
        if not state_dict:
            return 0
        own_state = model.state_dict()
        loaded = 0
        with torch.no_grad():
            for key, value in state_dict.items():
                if key not in own_state:
                    continue
                target = own_state[key]
                if not isinstance(target, torch.Tensor):
                    continue
                target.copy_(value.to(device=target.device, dtype=target.dtype))
                loaded += 1
        return loaded

    def _resolve_lora_config(self) -> tuple[bool, Any]:
        model_conf = self.speech_conf.model
        lora_conf = getattr(model_conf, "lora", None)
        if lora_conf is None:
            return False, None
        enabled = bool(getattr(lora_conf, "enabled", False))
        return enabled, lora_conf

    def _resolve_lora_targets(self, transformer: torch.nn.Module, configured_targets: list[str]) -> tuple[str, ...]:
        linear_modules = {
            name for name, module in transformer.named_modules() if isinstance(module, torch.nn.Linear)
        }
        if not linear_modules:
            raise RuntimeError("No linear modules found in transformer; cannot apply LoRA.")

        resolved: list[str] = []
        for target in configured_targets:
            if any(name.endswith(target) for name in linear_modules):
                resolved.append(target)
        if not resolved:
            candidates_preview = ", ".join(sorted(list(linear_modules))[:20])
            raise ValueError(
                "None of configured LoRA target modules matched transformer linear layers. "
                f"Configured={configured_targets}. Available preview={candidates_preview}"
            )
        return tuple(resolved)

    @staticmethod
    def _freeze_non_lora_params(model: torch.nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.requires_grad_(True)
            else:
                parameter.requires_grad_(False)

    def _to_plain_dict(self, node: Any) -> Any:
        if hasattr(node, "to_dict"):
            return node.to_dict()
        if isinstance(node, dict):
            return {key: self._to_plain_dict(value) for key, value in node.items()}
        return node

    def _apply_lora_to_model(self, base_model: torch.nn.Module, lora_conf) -> tuple[torch.nn.Module, tuple[str, ...], str]:
        try:
            from peft import LoraConfig, PeftModel, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "LoRA is enabled but `peft` is not installed. "
                "Install project dependencies with `pip install -r requirements.txt` (peft==0.19.1)."
            ) from exc

        strategy = str(getattr(lora_conf, "strategy", "multi_model")).strip().lower()
        if strategy not in {"multi_model", "shared_adapter"}:
            raise ValueError("`speech.model.lora.strategy` must be one of {'multi_model', 'shared_adapter'}.")

        configured_targets = list(getattr(lora_conf, "target_modules", ["to_q", "to_k", "to_v", "to_out.0"]))
        resolved_targets = self._resolve_lora_targets(base_model.transformer, configured_targets)
        lora_cfg = LoraConfig(
            r=int(getattr(lora_conf, "r", 32)),
            lora_alpha=int(getattr(lora_conf, "alpha", 64)),
            lora_dropout=float(getattr(lora_conf, "dropout", 0.0)),
            bias=str(getattr(lora_conf, "bias", "none")),
            init_lora_weights=str(getattr(lora_conf, "init_lora_weights", "gaussian")),
            target_modules=list(resolved_targets),
        )

        lora_path = getattr(lora_conf, "lora_path", None)
        if lora_path:
            lora_path_resolved = self._resolve_path(str(lora_path))
            base_model.transformer = PeftModel.from_pretrained(
                base_model.transformer,
                str(lora_path_resolved),
                is_trainable=True,
            )
            base_model.transformer.set_adapter("default")
        else:
            base_model.transformer = get_peft_model(base_model.transformer, lora_cfg)

        if strategy == "shared_adapter":
            transformer = base_model.transformer
            peft_config = getattr(transformer, "peft_config", {})
            if "old" not in peft_config:
                transformer.add_adapter("old", copy.deepcopy(peft_config["default"]))
            transformer.set_adapter("default")

        self._freeze_non_lora_params(base_model)
        return base_model, resolved_targets, strategy

    @staticmethod
    def _adapter_export_payload_from_snapshot(snapshot_manager) -> dict[str, Any]:
        if hasattr(snapshot_manager, "export_adapter_state"):
            payload = snapshot_manager.export_adapter_state()
            if isinstance(payload, dict):
                return payload
        return {}

    @staticmethod
    def _checkpoint_lora_dir(checkpoint_path: Path) -> Path:
        return checkpoint_path.parent / f"{checkpoint_path.stem}.lora"

    def save_lora_checkpoint_artifacts(
        self,
        checkpoint_path: Path,
        model_bundle: SpeechModelBundle,
        *,
        outer_epoch: int,
        global_step: int,
        best_eval_reward: float,
        best_metric: str = "val_reward_mean",
    ) -> Path | None:
        if not model_bundle.lora_enabled:
            return None
        save_adapter_only = bool(getattr(self.speech_conf.model.lora, "save_adapter_only", True))
        if not save_adapter_only:
            return None

        lora_dir = self._checkpoint_lora_dir(checkpoint_path)
        lora_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "format": "speech_nft_lora_adapter_v1",
            "strategy": model_bundle.lora_strategy,
            "outer_epoch": int(outer_epoch),
            "global_step": int(global_step),
            "best_eval_reward": float(best_eval_reward),
            "best_metric": str(best_metric),
            "target_modules": list(model_bundle.lora_target_modules),
            "trainable_param_count": int(model_bundle.trainable_param_count),
            "total_param_count": int(model_bundle.total_param_count),
        }
        if model_bundle.lora_strategy == "shared_adapter":
            payload.update(self._adapter_export_payload_from_snapshot(self.snapshot_manager))
        else:
            payload["current_lora_state_dict"] = self._filter_lora_state_dict(
                model_bundle.current_model.transformer.state_dict()
            )
        torch.save(payload, lora_dir / "adapter.pt")
        return lora_dir

    def load_lora_resume(
        self,
        resume_dir: Path,
        model_bundle: SpeechModelBundle,
        *,
        current_best_metric: str = "val_reward_mean",
    ) -> tuple[int, int, float]:
        adapter_file = resume_dir / "adapter.pt"
        if not adapter_file.exists():
            raise FileNotFoundError(f"LoRA resume directory missing adapter payload: {adapter_file}")

        payload = torch.load(adapter_file, map_location=self.device)
        payload_strategy = str(payload.get("strategy", model_bundle.lora_strategy))
        if payload_strategy != model_bundle.lora_strategy:
            raise ValueError(
                "LoRA strategy mismatch between resume payload and current config: "
                f"payload={payload_strategy}, config={model_bundle.lora_strategy}"
            )

        if model_bundle.lora_strategy == "shared_adapter":
            if not hasattr(self.snapshot_manager, "load_adapter_state"):
                raise TypeError("Snapshot manager does not support adapter-state loading.")
            self.snapshot_manager.load_adapter_state(payload)
        else:
            state_dict = payload.get("current_lora_state_dict", {})
            loaded = self._load_partial_state_dict(model_bundle.current_model.transformer, state_dict)
            if loaded <= 0:
                raise RuntimeError("No LoRA parameters were loaded from adapter payload.")
            self.snapshot_manager.hard_sync_old()
            self.snapshot_manager.freeze_auxiliary_models()

        start_epoch = int(payload.get("outer_epoch", -1)) + 1
        global_step = int(payload.get("global_step", 0))
        saved_best_metric = str(payload.get("best_metric", "val_reward_mean"))
        if saved_best_metric != str(current_best_metric):
            best_eval_reward = float("-inf")
        else:
            best_eval_reward = float(payload.get("best_eval_reward", float("-inf")))
        return start_epoch, global_step, best_eval_reward

    def cleanup_lora_checkpoint_artifacts(self, checkpoint_path: Path) -> None:
        lora_dir = self._checkpoint_lora_dir(checkpoint_path)
        if not lora_dir.exists():
            return
        try:
            for file in lora_dir.glob("*"):
                if file.is_file():
                    file.unlink()
            lora_dir.rmdir()
        except OSError:
            pass

    def _build_reward_fn(self):
        reward_conf = self.speech_conf.reward

        registry = [str(metric).strip().lower() for metric in reward_conf.registry]
        weights = normalize_reward_weights(registry, self._to_plain_dict(reward_conf.weights))
        train_registry = [metric for metric in registry if float(weights.get(metric, 0.0)) > 0.0]
        if not train_registry:
            raise ValueError("No enabled speech reward metrics. Ensure reward.registry has at least one positive-weight metric.")
        configured_eval_registry = getattr(reward_conf, "eval_registry", None)
        eval_registry = (
            list(registry)
            if not configured_eval_registry
            else list(dict.fromkeys(str(metric).strip().lower() for metric in configured_eval_registry))
        )
        for metric in train_registry:
            if metric not in eval_registry:
                eval_registry.append(metric)
        scoring_registry = list(dict.fromkeys([*train_registry, *eval_registry]))
        self.train_reward_metrics = train_registry
        self.eval_reward_metrics = eval_registry

        reward_config = {
            "weights": weights,
            "registry": registry,
            "scoring_registry": scoring_registry,
        }
        for optional_key in ("normalization", "batch_zscore_eps", "primary_keys", "raw_scales"):
            if hasattr(reward_conf, optional_key):
                reward_config[optional_key] = self._to_plain_dict(getattr(reward_conf, optional_key))

        reward_device_name = str(getattr(reward_conf, "device", "train")).strip().lower()
        if reward_device_name in {"", "train", "same", "model", "cuda", "gpu"}:
            reward_device = self.device
        elif reward_device_name == "cpu":
            reward_device = torch.device("cpu")
        else:
            reward_device = torch.device(reward_device_name)
            if reward_device.type == "cuda" and reward_device.index is None:
                reward_device = self.device

        self.reward_fn = flowse_rewards.build_reward_fn(
            device=reward_device,
            voice_eval_root=str(self._resolve_path(reward_conf.voice_eval_root)),
            reward_config=reward_config,
            hf_cache_dir=(None if reward_conf.hf_cache_dir is None else str(self._resolve_path(reward_conf.hf_cache_dir))),
            speechbert_model_path=(
                None
                if reward_conf.speechbert_model_path is None
                else str(self._resolve_path(reward_conf.speechbert_model_path))
            ),
            speaker_model_path=(
                None
                if reward_conf.speaker_model_path is None
                else str(self._resolve_path(reward_conf.speaker_model_path))
            ),
            speaker_model_type=str(getattr(reward_conf, "speaker_model_type", "wavlm")),
            speaker_code_path=(
                None
                if getattr(reward_conf, "speaker_code_path", None) is None
                else str(self._resolve_path(reward_conf.speaker_code_path))
            ),
            local_files_only=False if bool(reward_conf.allow_remote_hf) else True,
            tmp_root=(None if reward_conf.tmp_root is None else str(self._resolve_path(reward_conf.tmp_root))),
            sample_rate=int(self.speech_conf.data.sample_rate),
        )

    @staticmethod
    def _reward_branches_from_config(train_conf) -> list[dict[str, Any]]:
        branch_configs = getattr(train_conf, "reward_branches", None)
        if not branch_configs:
            return []
        branches: list[dict[str, Any]] = []
        for branch_conf in branch_configs:
            if not hasattr(branch_conf, "items"):
                raise TypeError("Each `speech.train.reward_branches` entry must be mapping-like.")
            branch = {str(key): value for key, value in branch_conf.items()}
            name = str(branch.get("name", "")).strip()
            metric_key = str(branch.get("metric_key", "")).strip()
            if not name:
                raise ValueError("Each reward branch must define a non-empty `name`.")
            if not metric_key:
                raise ValueError(f"Reward branch {name!r} must define `metric_key`.")
            branch["name"] = name
            branch["metric_key"] = metric_key
            branch["score_section"] = str(branch.get("score_section", "raw")).strip().lower()
            if branch["score_section"] not in {"raw", "norm"}:
                raise ValueError(f"Reward branch {name!r} has unsupported score_section={branch['score_section']!r}.")
            branch["score_scale"] = float(branch.get("score_scale", 1.0))
            branches.append(branch)
        return branches

    def build_models_and_policies(self, config, device: torch.device) -> SpeechModelBundle:
        del config

        model_conf = self._to_plain_dict(self.speech_conf.model.nnet_conf)
        vocab_char_map, vocab_size = flowse_runtime.build_tokenizer(model_conf)
        base_model = flowse_runtime.build_model(model_conf, vocab_char_map, vocab_size, device=device)

        init_checkpoint = self._resolve_path(self.speech_conf.model.init_checkpoint)
        flowse_runtime.load_checkpoint(init_checkpoint, device=device, model=base_model)
        vocoder = flowse_runtime.load_vocoder(model_conf, device=device)

        lora_enabled, lora_conf = self._resolve_lora_config()
        lora_strategy = "disabled"
        lora_targets: tuple[str, ...] = ()
        if lora_enabled:
            base_model, lora_targets, lora_strategy = self._apply_lora_to_model(base_model, lora_conf)
        self._lora_enabled = bool(lora_enabled)
        self._lora_strategy = lora_strategy
        self._lora_target_modules = lora_targets

        snapshot_conf = flowse_policy_snapshot.PolicySnapshotConfig(**self._to_plain_dict(self.speech_conf.snapshot))
        if lora_enabled and lora_strategy == "shared_adapter":
            self.snapshot_manager = flowse_policy_snapshot.build_lora_adapter_snapshot_manager(
                base_model,
                snapshot_conf,
                current_adapter_name="default",
                old_adapter_name="old",
            )
        else:
            self.snapshot_manager = flowse_policy_snapshot.build_policy_snapshot_manager(base_model, snapshot_conf)

        self._build_reward_fn()
        rollout_conf = self.speech_conf.rollout
        rollout_solver = str(getattr(rollout_conf, "solver", "dpm2"))
        rollout_deterministic = bool(getattr(rollout_conf, "deterministic", True))
        rollout_noise_level = float(getattr(rollout_conf, "noise_level", 0.7))
        rollout_sigma_min_cfg = getattr(rollout_conf, "sigma_min", None)
        rollout_sigma_min = None if rollout_sigma_min_cfg is None else float(rollout_sigma_min_cfg)
        rollout_sigma_max = float(getattr(rollout_conf, "sigma_max", 1.0))
        rollout_steps = int(getattr(rollout_conf, "steps", 32))
        nft_target_time_cfg = getattr(self.speech_conf.train, "nft_target_time", None)
        nft_target_time = None if nft_target_time_cfg is None else float(nft_target_time_cfg)
        rollout_time_grid_values = list(getattr(rollout_conf, "time_grid", []))
        rollout_time_grid = rollout_time_grid_values or None
        rollout_mask_values = list(getattr(rollout_conf, "deterministic_mask", []))
        rollout_default_deterministic_mask = rollout_mask_values or None
        eval_conf = self.speech_conf.eval
        eval_use_rollout_schedule = bool(getattr(eval_conf, "use_rollout_schedule", True))
        if eval_use_rollout_schedule:
            eval_steps = rollout_steps
            eval_time_grid = rollout_time_grid
        else:
            eval_steps = int(getattr(eval_conf, "steps", 32))
            if eval_steps <= 0:
                raise ValueError("`speech.eval.steps` must be positive when validation uses a separate schedule.")
            eval_time_grid = None
        mixed_sampling_enabled = bool(getattr(rollout_conf, "mixed_sampling_enabled", False))
        if mixed_sampling_enabled and rollout_default_deterministic_mask is not None:
            raise ValueError(
                "Configure either progressive mixed sampling or a fixed rollout deterministic mask, not both."
            )
        if mixed_sampling_enabled:
            if rollout_solver != "flow":
                raise ValueError("Speech mixed SDE/ODE rollout currently supports only solver='flow'.")
            self.mixed_sampling_state = SpeechMixedSamplingState(
                steps=rollout_steps,
                group_size=int(getattr(rollout_conf, "mixed_group_size", 4)),
                strategy=str(getattr(rollout_conf, "mixed_strategy", "progressive")),
                overlap=bool(getattr(rollout_conf, "mixed_overlap", True)),
                overlap_step=int(getattr(rollout_conf, "mixed_overlap_step", 1)),
                update_interval=int(getattr(rollout_conf, "mixed_update_interval", 1)),
                roll_back=bool(getattr(rollout_conf, "mixed_roll_back", True)),
            )
        else:
            self.mixed_sampling_state = None

        self.train_rollout_engine = flowse_rollout.build_rollout_engine(
            old_model=self.snapshot_manager.old_model,
            vocoder=vocoder,
            reward_fn=self.reward_fn,
            cond_type=str(self.speech_conf.rollout.cond_type),
            input_sample_rate=int(self.speech_conf.data.sample_rate),
            vocoder_sample_rate=int(self.speech_conf.model.nnet_conf.mel_spec.target_sample_rate),
            output_sample_rate=int(self.speech_conf.data.sample_rate),
            steps=rollout_steps,
            cfg_strength=float(self.speech_conf.rollout.cfg_strength),
            base_seed=int(self.speech_conf.rollout.base_seed),
            solver=rollout_solver,
            deterministic=rollout_deterministic,
            noise_level=rollout_noise_level,
            sigma_min=rollout_sigma_min,
            sigma_max=rollout_sigma_max,
            time_grid=rollout_time_grid,
            default_deterministic_mask=rollout_default_deterministic_mask,
            nft_target_time=nft_target_time,
            collect_trajectory_similarity=bool(
                getattr(rollout_conf, "collect_trajectory_similarity", False)
            ),
        )
        self.eval_rollout_engine = flowse_rollout.build_rollout_engine(
            old_model=self.snapshot_manager.current_model,
            vocoder=vocoder,
            reward_fn=self.reward_fn,
            cond_type=str(self.speech_conf.rollout.cond_type),
            input_sample_rate=int(self.speech_conf.data.sample_rate),
            vocoder_sample_rate=int(self.speech_conf.model.nnet_conf.mel_spec.target_sample_rate),
            output_sample_rate=int(self.speech_conf.data.sample_rate),
            steps=eval_steps,
            cfg_strength=float(self.speech_conf.rollout.cfg_strength),
            base_seed=int(self.speech_conf.rollout.base_seed),
            solver=rollout_solver,
            deterministic=True,
            noise_level=rollout_noise_level,
            sigma_min=rollout_sigma_min,
            sigma_max=rollout_sigma_max,
            time_grid=eval_time_grid,
        )

        trainable_param_count, total_param_count = self._count_parameters(self.snapshot_manager.current_model)

        return SpeechModelBundle(
            current_model=self.snapshot_manager.current_model,
            old_model=self.snapshot_manager.old_model,
            ref_model=self.snapshot_manager.ref_model,
            vocoder=vocoder,
            lora_enabled=bool(lora_enabled),
            lora_strategy=lora_strategy,
            lora_target_modules=lora_targets,
            trainable_param_count=trainable_param_count,
            total_param_count=total_param_count,
        )

    def build_data(self, config) -> dict[str, Any]:
        del config
        model_conf = self._to_plain_dict(self.speech_conf.model.nnet_conf)
        data_conf = self.speech_conf.data

        distributed_flag = self.world_size > 1 and dist.is_available() and dist.is_initialized()

        train_sampler, train_loader = flowse_dataset.make_rl_loader(
            manifest_path=self._resolve_path(data_conf.train_manifest),
            data_root=getattr(data_conf, "data_root", None),
            batch_size=int(data_conf.train_batch_size),
            num_workers=int(data_conf.num_workers),
            shuffle=True,
            distributed=distributed_flag,
            sample_rate=int(data_conf.sample_rate),
            mel_spec_conf=model_conf["mel_spec"],
            pad_to_chunk=True,
            pin_memory=self.device.type == "cuda",
            persistent_workers=int(data_conf.num_workers) > 0,
            drop_last=bool(data_conf.drop_last),
        )
        val_sampler, val_loader = flowse_dataset.make_rl_loader(
            manifest_path=self._resolve_path(data_conf.val_manifest),
            data_root=getattr(data_conf, "data_root", None),
            batch_size=int(data_conf.eval_batch_size),
            num_workers=int(data_conf.num_workers),
            shuffle=False,
            distributed=distributed_flag,
            sample_rate=int(data_conf.sample_rate),
            mel_spec_conf=model_conf["mel_spec"],
            pad_to_chunk=True,
            pin_memory=self.device.type == "cuda",
            persistent_workers=int(data_conf.num_workers) > 0,
            drop_last=False,
        )

        return {
            "train_sampler": train_sampler,
            "train_loader": train_loader,
            "val_sampler": val_sampler,
            "val_loader": val_loader,
        }

    @staticmethod
    def _reduce_sum_count(sum_value: float, count_value: float, device: torch.device) -> tuple[float, float]:
        tensor = torch.tensor([sum_value, count_value], dtype=torch.float64, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor[0].item()), float(tensor[1].item())

    @staticmethod
    def _reduce_metric_dict(metric_sums: dict[str, float], count_value: float, device: torch.device) -> dict[str, float]:
        if not metric_sums:
            return {}
        keys = sorted(metric_sums)
        values = [metric_sums[key] for key in keys] + [count_value]
        tensor = torch.tensor(values, dtype=torch.float64, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        total_count = float(tensor[-1].item())
        if total_count <= 0:
            return {key: 0.0 for key in keys}
        return {key: float(tensor[idx].item() / total_count) for idx, key in enumerate(keys)}

    def _rollout_seed_offset(self, outer_epoch: int, batch_index: int) -> int:
        return int(outer_epoch * 100000 + batch_index * 100)

    def _next_train_deterministic_mask(self) -> list[bool] | None:
        if self.mixed_sampling_state is None:
            return None
        mask = self.mixed_sampling_state.get_current_deterministic_mask()
        self.mixed_sampling_state.update_iteration()
        return mask

    def _eval_deterministic_mask(self) -> list[bool]:
        return build_eval_deterministic_mask(
            int(self.eval_rollout_engine.steps),
            use_rollout_schedule=bool(getattr(self.speech_conf.eval, "use_rollout_schedule", True)),
            use_rollout_deterministic_mask=bool(
                getattr(self.speech_conf.eval, "use_rollout_deterministic_mask", False)
            ),
            rollout_deterministic_mask=getattr(self.speech_conf.rollout, "deterministic_mask", []),
        )

    def summarize_mixed_sampling(self, rollout_batches: list[RolloutBatch]) -> dict[str, float | int | bool]:
        masks = [
            batch.sde_timestep_mask.detach().cpu().to(torch.bool)
            for batch in rollout_batches
            if batch.sde_timestep_mask is not None
        ]
        if masks:
            stacked = torch.stack(masks, dim=0)
            sde_counts = stacked.sum(dim=1).to(torch.float32)
            last_indices = torch.nonzero(stacked[-1], as_tuple=False).reshape(-1)
            has_sde = bool(stacked.any().item())
            has_ode = bool((~stacked).any().item())
            return {
                "mixed_sampling_enabled": bool(has_sde and has_ode),
                "mixed_sde_step_count": float(sde_counts.mean().item()),
                "mixed_ode_step_count": float(stacked.shape[1] - sde_counts.mean().item()),
                "mixed_sde_timestep_start": int(last_indices[0].item()) if last_indices.numel() else -1,
                "mixed_sde_timestep_end": int(last_indices[-1].item()) if last_indices.numel() else -1,
            }

        if self.mixed_sampling_state is not None:
            return self.mixed_sampling_state.get_current_stats()

        return {
            "mixed_sampling_enabled": False,
            "mixed_sde_step_count": 0,
            "mixed_ode_step_count": int(self.speech_conf.rollout.steps),
            "mixed_sde_timestep_start": -1,
            "mixed_sde_timestep_end": -1,
        }

    def rollout(self, old_policy, batch: dict[str, Any], config, seed_offset: int) -> RolloutBatch:
        del config
        self.train_rollout_engine.model = old_policy.eval()
        deterministic_mask = self._next_train_deterministic_mask()
        rollout_result = self.train_rollout_engine.rollout_batch(
            batch,
            num_candidates=int(self.speech_conf.rollout.num_candidates),
            seed_offset=seed_offset,
            keep_waveforms=False,
            deterministic_mask=deterministic_mask,
            reward_metric_names=self.train_reward_metrics,
        )

        sample_count = int(rollout_result["reward"].shape[0])
        ode_time_grid = rollout_result.get("ode_time_grid")
        if ode_time_grid is None:
            raise KeyError("Rollout result is missing `ode_time_grid`; speech training requires rollout-aligned timesteps.")
        ode_time_grid = ode_time_grid.detach().cpu().to(torch.float32).reshape(-1)
        timesteps = ode_time_grid.unsqueeze(0).repeat(sample_count, 1)
        sde_timestep_mask = rollout_result.get("sde_timestep_mask")
        if sde_timestep_mask is not None:
            sde_timestep_mask = sde_timestep_mask.detach().cpu().to(torch.bool).reshape(-1)

        reward_dict = {
            "raw": {
                key: value.detach().cpu().to(torch.float32)
                for key, value in rollout_result["reward_breakdown"]["raw"].items()
            },
            "norm": {
                key: value.detach().cpu().to(torch.float32)
                for key, value in rollout_result["reward_breakdown"]["norm"].items()
            },
        }

        return RolloutBatch(
            x0_target=rollout_result["generated_mel"].detach().cpu().to(torch.float32),
            condition=rollout_result["noisy_mel"].detach().cpu().to(torch.float32),
            timesteps=timesteps,
            candidate_ids=list(rollout_result["candidate_id"]),
            group_ids=list(rollout_result["source_utt_id"]),
            reward_dict=reward_dict,
            reward_avg=rollout_result["reward"].detach().cpu().to(torch.float32),
            noisy_mel=rollout_result["noisy_mel"].detach().cpu().to(torch.float32),
            generated_mel=rollout_result["generated_mel"].detach().cpu().to(torch.float32),
            utt_ids=list(rollout_result["utt_id"]),
            source_utt_ids=list(rollout_result["source_utt_id"]),
            sde_timestep_mask=sde_timestep_mask,
            nft_target_time=float(rollout_result.get("nft_target_time", 1.0)),
            trajectory_final_similarity=(
                {
                    metric_name: values.detach().cpu().to(torch.float32)
                    for metric_name, values in rollout_result["trajectory_final_similarity"].items()
                }
                if "trajectory_final_similarity" in rollout_result
                else None
            ),
            trajectory_information_gain=(
                {
                    metric_name: values.detach().cpu().to(torch.float32)
                    for metric_name, values in rollout_result["trajectory_information_gain"].items()
                }
                if "trajectory_information_gain" in rollout_result
                else None
            ),
            trajectory_embedding_kinds=(
                dict(rollout_result["trajectory_embedding_kinds"])
                if "trajectory_embedding_kinds" in rollout_result
                else None
            ),
        )

    def _gather_object_payloads(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        if not (dist.is_available() and dist.is_initialized() and self.world_size > 1):
            return [payload]
        gathered = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, payload)
        return gathered

    def _global_masked_advantage_moments(
        self,
        advantages: np.ndarray,
        keep_mask: np.ndarray,
    ) -> np.ndarray:
        """Return global [count, sum, sum_sq] for retained advantages."""
        advantage_array = np.asarray(advantages, dtype=np.float64).reshape(-1)
        keep_array = np.asarray(keep_mask, dtype=bool).reshape(-1)
        if advantage_array.shape != keep_array.shape:
            raise ValueError("`advantages` and `keep_mask` must have the same shape.")

        retained = advantage_array[keep_array]
        moments = torch.tensor(
            [
                float(retained.size),
                float(np.sum(retained, dtype=np.float64)),
                float(np.sum(np.square(retained), dtype=np.float64)),
            ],
            dtype=torch.float64,
        )
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            collective_device = (
                self.device if str(dist.get_backend()).lower() == "nccl" else torch.device("cpu")
            )
            moments = moments.to(device=collective_device)
            dist.all_reduce(moments, op=dist.ReduceOp.SUM)
        return moments.cpu().numpy()

    def compute_advantages(self, rollout_batches: list[RolloutBatch], world_context: dict[str, Any] | None = None) -> dict[str, Any]:
        del world_context
        local_payload = {
            "candidate_ids": [],
            "group_ids": [],
            "reward_avg": [],
        }
        for batch in rollout_batches:
            local_payload["candidate_ids"].extend(batch.candidate_ids)
            local_payload["group_ids"].extend(batch.group_ids)
            local_payload["reward_avg"].extend(batch.reward_avg.tolist())

        gather_advantages = bool(getattr(self.speech_conf.train, "gather_advantages", False))
        if gather_advantages:
            gathered_payloads = self._gather_object_payloads(local_payload)
        else:
            # Speech rollout keeps all candidates for one source utterance on the same rank.
            # Local advantage computation avoids all_gather_object, which can hang on some NCCL setups.
            gathered_payloads = [local_payload]
        global_candidate_ids: list[str] = []
        global_group_ids: list[str] = []
        global_rewards: list[float] = []
        for payload in gathered_payloads:
            global_candidate_ids.extend(payload["candidate_ids"])
            global_group_ids.extend(payload["group_ids"])
            global_rewards.extend(payload["reward_avg"])

        tracker = PerConditionStatTracker(global_std=bool(self.speech_conf.train.global_advantage_std))
        global_advantages = tracker.update(global_group_ids, global_rewards)
        advantage_map = {
            candidate_id: float(advantage)
            for candidate_id, advantage in zip(global_candidate_ids, global_advantages.tolist(), strict=True)
        }
        stats = tracker.get_last_stats()

        filter_conf = getattr(self.speech_conf, "group_filter", None)
        filter_enabled = bool(getattr(filter_conf, "enabled", False)) if filter_conf is not None else False
        if filter_enabled:
            keep_mask, filter_stats = compute_low_std_group_keep_mask(
                global_group_ids,
                global_rewards,
                mean_threshold=float(getattr(filter_conf, "mean_threshold", 0.95)),
                std_threshold=float(getattr(filter_conf, "std_threshold", 0.01)),
            )
        else:
            keep_mask = np.ones(len(global_candidate_ids), dtype=bool)
            filter_stats = {
                "low_std_filter_enabled": False,
                "low_std_filter_mean_threshold": 0.0,
                "low_std_filter_std_threshold": 0.0,
                "low_std_filter_group_count": 0,
                "low_std_filter_group_ratio": 0.0,
                "low_std_filter_sample_count": 0,
                "low_std_filter_sample_ratio": 0.0,
                "low_std_filter_kept_sample_count": int(len(global_candidate_ids)),
                "low_std_filter_kept_sample_ratio": 1.0 if global_candidate_ids else 0.0,
                "low_std_filter_reward_mean_mean": 0.0,
                "low_std_filter_reward_std_mean": 0.0,
            }
        for batch in rollout_batches:
            batch.advantages = torch.tensor(
                [advantage_map[candidate_id] for candidate_id in batch.candidate_ids],
                dtype=torch.float32,
            )

        branch_reports: dict[str, dict[str, float]] = {}
        branch_score_maps: dict[str, dict[str, float]] = {}
        branch_advantage_maps: dict[str, dict[str, float]] = {}
        reward_branches = self._reward_branches_from_config(self.speech_conf.train)
        conflict_filter_conf = getattr(self.speech_conf.train, "reward_conflict_filter", None)
        conflict_filter_enabled = (
            bool(getattr(conflict_filter_conf, "enabled", False)) if conflict_filter_conf is not None else False
        )
        conflict_filter_stats: dict[str, float | int | bool] = {
            "reward_conflict_filter_enabled": False,
            "reward_conflict_filter_tau": float(
                getattr(conflict_filter_conf, "tau", 0.2) if conflict_filter_conf is not None else 0.2
            ),
            "reward_conflict_filter_sample_count": int(len(global_candidate_ids)),
            "reward_conflict_filter_kept_sample_count": int(len(global_candidate_ids)),
            "reward_conflict_filter_kept_sample_ratio": 1.0 if global_candidate_ids else 0.0,
            "reward_conflict_filter_filtered_sample_count": 0,
            "reward_conflict_filter_filtered_sample_ratio": 0.0,
        }
        if reward_branches:
            for branch in reward_branches:
                branch_name = str(branch["name"])
                section = str(branch["score_section"])
                metric_key = str(branch["metric_key"])
                score_scale = float(branch["score_scale"])

                branch_payload = {
                    "candidate_ids": [],
                    "group_ids": [],
                    "scores": [],
                }
                for batch in rollout_batches:
                    metric_values = batch.reward_dict.get(section, {}).get(metric_key)
                    if metric_values is None:
                        raise KeyError(
                            f"Reward branch {branch_name!r} requested {section}/{metric_key}, "
                            "but that metric is missing from rollout reward_dict."
                        )
                    branch_payload["candidate_ids"].extend(batch.candidate_ids)
                    branch_payload["group_ids"].extend(batch.group_ids)
                    branch_payload["scores"].extend((metric_values.to(torch.float32) * score_scale).tolist())

                if gather_advantages:
                    branch_gathered_payloads = self._gather_object_payloads(branch_payload)
                else:
                    branch_gathered_payloads = [branch_payload]

                branch_candidate_ids: list[str] = []
                branch_group_ids: list[str] = []
                branch_scores: list[float] = []
                for payload in branch_gathered_payloads:
                    branch_candidate_ids.extend(payload["candidate_ids"])
                    branch_group_ids.extend(payload["group_ids"])
                    branch_scores.extend(payload["scores"])

                branch_tracker = PerConditionStatTracker(global_std=bool(self.speech_conf.train.global_advantage_std))
                branch_advantages = branch_tracker.update(branch_group_ids, branch_scores)
                branch_score_maps[branch_name] = {
                    candidate_id: float(score)
                    for candidate_id, score in zip(branch_candidate_ids, branch_scores, strict=True)
                }
                branch_advantage_map = {
                    candidate_id: float(advantage)
                    for candidate_id, advantage in zip(branch_candidate_ids, branch_advantages.tolist(), strict=True)
                }
                branch_advantage_maps[branch_name] = branch_advantage_map
                branch_reports[branch_name] = branch_tracker.get_last_stats()

                for batch in rollout_batches:
                    if batch.reward_branch_advantages is None:
                        batch.reward_branch_advantages = {}
                    batch.reward_branch_advantages[branch_name] = torch.tensor(
                        [branch_advantage_map[candidate_id] for candidate_id in batch.candidate_ids],
                        dtype=torch.float32,
                    )
            branch_names = [str(branch["name"]) for branch in reward_branches]
            branch_weights = {
                str(branch["name"]): float(branch.get("loss_weight", 1.0))
                for branch in reward_branches
            }
            branch_advantage_vectors = {
                branch_name: [branch_advantage_maps[branch_name][candidate_id] for candidate_id in global_candidate_ids]
                for branch_name in branch_names
            }
            branch_score_vectors = {
                branch_name: [branch_score_maps[branch_name][candidate_id] for candidate_id in global_candidate_ids]
                for branch_name in branch_names
            }
            reward_update_mode = str(
                getattr(self.speech_conf.train, "multi_reward_update_mode", "branch")
            ).strip().lower()
            if reward_update_mode not in {"branch", "gd2po"}:
                raise ValueError(
                    "`speech.train.multi_reward_update_mode` must be either 'branch' or 'gd2po'."
                )
            gd2po_advantages = None
            if reward_update_mode == "gd2po":
                gd2po_advantages = aggregate_reward_advantages(
                    branch_advantage_vectors,
                    branch_names,
                    weights=branch_weights,
                )
                stats.update(
                    {
                        "gd2po_enabled": True,
                        "gd2po_advantage_mean": float(np.mean(gd2po_advantages)) if gd2po_advantages.size else 0.0,
                        "gd2po_advantage_std": float(np.std(gd2po_advantages)) if gd2po_advantages.size else 0.0,
                        "gd2po_advantage_pre_norm_mean": (
                            float(np.mean(gd2po_advantages)) if gd2po_advantages.size else 0.0
                        ),
                        "gd2po_advantage_pre_norm_std": (
                            float(np.std(gd2po_advantages)) if gd2po_advantages.size else 0.0
                        ),
                    }
                )
            tau = float(
                getattr(conflict_filter_conf, "tau", 0.2) if conflict_filter_conf is not None else 0.2
            )
            monitor_thresholds = tuple(dict.fromkeys((tau, 0.5, 0.8)))
            stats.update(
                compute_reward_advantage_conflict_stats(
                    branch_advantage_vectors,
                    branch_names,
                    weights=branch_weights,
                    snr_thresholds=monitor_thresholds,
                )
            )
            stats.update(
                compute_reward_advantage_pairwise_stats(
                    branch_advantage_vectors,
                    branch_names,
                    weights=branch_weights,
                )
            )
            if conflict_filter_enabled:
                conflict_keep_mask, conflict_filter_stats = compute_reward_advantage_snr_keep_mask(
                    branch_advantage_vectors,
                    branch_names,
                    tau=tau,
                    weights=branch_weights,
                    snr_eps=float(getattr(conflict_filter_conf, "snr_eps", 1e-8)),
                )
                stats.update(
                    compute_reward_conflict_filter_diagnostics(
                        branch_score_vectors,
                        branch_advantage_vectors,
                        branch_names,
                        conflict_keep_mask,
                        weights=branch_weights,
                        snr_eps=float(getattr(conflict_filter_conf, "snr_eps", 1e-8)),
                    )
                )
                keep_mask = keep_mask & conflict_keep_mask
        elif conflict_filter_enabled:
            raise ValueError("Reward conflict filtering requires configured multi-reward branches.")

        if reward_branches and reward_update_mode == "gd2po":
            if gd2po_advantages is None:
                raise RuntimeError("GD2PO advantages were not computed.")
            apply_group_keep_ratio = bool(
                getattr(conflict_filter_conf, "apply_group_keep_ratio", True)
                if conflict_filter_conf is not None
                else True
            )
            if apply_group_keep_ratio:
                group_keep_ratios, group_keep_ratio_stats = compute_gd2po_group_keep_ratios(
                    global_group_ids,
                    keep_mask,
                )
                # Match official GD2PO: query-level retained fraction is applied before masked whitening.
                gd2po_advantages = gd2po_advantages * group_keep_ratios
                stats.update(group_keep_ratio_stats)
            else:
                stats.update(
                    {
                        "gd2po_group_keep_ratio_enabled": False,
                        "gd2po_group_keep_ratio_mean": 1.0 if gd2po_advantages.size else 0.0,
                        "gd2po_group_keep_ratio_min": 1.0 if gd2po_advantages.size else 0.0,
                        "gd2po_group_keep_ratio_max": 1.0 if gd2po_advantages.size else 0.0,
                        "gd2po_group_keep_ratio_std": 0.0,
                    }
                )
            stats.update(
                {
                    "gd2po_advantage_pre_norm_mean": (
                        float(np.mean(gd2po_advantages)) if gd2po_advantages.size else 0.0
                    ),
                    "gd2po_advantage_pre_norm_std": (
                        float(np.std(gd2po_advantages)) if gd2po_advantages.size else 0.0
                    ),
                }
            )
            legacy_post_normalize_enabled = bool(
                getattr(conflict_filter_conf, "post_normalize", True)
                if conflict_filter_conf is not None
                else True
            )
            configured_post_normalize_mode = (
                getattr(conflict_filter_conf, "post_normalize_mode", None)
                if conflict_filter_conf is not None
                else None
            )
            if configured_post_normalize_mode is None:
                post_normalize_mode = "masked_whiten" if legacy_post_normalize_enabled else "none"
            else:
                post_normalize_mode = str(configured_post_normalize_mode).strip().lower()
            if post_normalize_mode not in {"masked_whiten", "rms_scale", "none"}:
                raise ValueError(
                    "`speech.train.reward_conflict_filter.post_normalize_mode` must be one of "
                    "{'masked_whiten', 'rms_scale', 'none'}."
                )
            post_normalize_enabled = post_normalize_mode != "none"
            post_normalize_eps = float(
                getattr(conflict_filter_conf, "post_normalize_eps", 1e-4)
                if conflict_filter_conf is not None
                else 1e-4
            )
            if post_normalize_enabled:
                global_moments = self._global_masked_advantage_moments(
                    gd2po_advantages,
                    keep_mask,
                )
            if post_normalize_mode == "masked_whiten":
                train_advantages = masked_normalize_aggregated_advantages(
                    gd2po_advantages,
                    keep_mask,
                    eps=post_normalize_eps,
                    moments=global_moments,
                )
            elif post_normalize_mode == "rms_scale":
                train_advantages = masked_rms_scale_aggregated_advantages(
                    gd2po_advantages,
                    keep_mask,
                    eps=post_normalize_eps,
                    moments=global_moments,
                )
            else:
                train_advantages = np.where(
                    keep_mask,
                    np.asarray(gd2po_advantages, dtype=np.float32),
                    0.0,
                ).astype(np.float32, copy=False)

            retained_train_advantages = train_advantages[keep_mask]
            sign_flip_counts = torch.tensor(
                compute_advantage_sign_flip_counts(
                    gd2po_advantages,
                    train_advantages,
                    keep_mask,
                ),
                dtype=torch.float64,
            )
            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                collective_device = (
                    self.device if str(dist.get_backend()).lower() == "nccl" else torch.device("cpu")
                )
                sign_flip_counts = sign_flip_counts.to(device=collective_device)
                dist.all_reduce(sign_flip_counts, op=dist.ReduceOp.SUM)
            sign_flip_stats = summarize_advantage_sign_flip_counts(sign_flip_counts.cpu().numpy())
            stats.update(
                {
                    "gd2po_post_normalization_enabled": post_normalize_enabled,
                    "gd2po_post_normalization_mode": post_normalize_mode,
                    "gd2po_advantage_post_norm_kept_mean": (
                        float(np.mean(retained_train_advantages)) if retained_train_advantages.size else 0.0
                    ),
                    "gd2po_advantage_post_norm_kept_std": (
                        float(np.std(retained_train_advantages)) if retained_train_advantages.size else 0.0
                    ),
                    **sign_flip_stats,
                }
            )
            gd2po_advantage_map = {
                candidate_id: float(advantage)
                for candidate_id, advantage in zip(global_candidate_ids, train_advantages.tolist(), strict=True)
            }
            for batch in rollout_batches:
                batch.advantages = torch.tensor(
                    [gd2po_advantage_map[candidate_id] for candidate_id in batch.candidate_ids],
                    dtype=torch.float32,
                )

        train_mask_map = {
            candidate_id: bool(keep)
            for candidate_id, keep in zip(global_candidate_ids, keep_mask.tolist(), strict=True)
        }
        for batch in rollout_batches:
            batch.train_mask = torch.tensor(
                [1.0 if train_mask_map[candidate_id] else 0.0 for candidate_id in batch.candidate_ids],
                dtype=torch.float32,
            )

        effective_kept_sample_count = int(np.count_nonzero(keep_mask))
        effective_sample_count = int(len(global_candidate_ids))
        stats.update(conflict_filter_stats)
        stats.update(
            {
                "train_mask_kept_sample_count": effective_kept_sample_count,
                "train_mask_kept_sample_ratio": (
                    float(effective_kept_sample_count / effective_sample_count) if effective_sample_count else 0.0
                ),
            }
        )
        stats.update(filter_stats)
        if branch_reports:
            stats["reward_branch_stats"] = branch_reports
        return stats

    def compute_rl_kl_loss(
        self,
        current,
        old,
        ref,
        train_batch: RolloutBatch,
        advantages: torch.Tensor,
        config,
        sampled_time: torch.Tensor | float | None = None,
        timestep_idx: int | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del config
        if self._lora_enabled and self._lora_strategy == "shared_adapter":
            if hasattr(self.snapshot_manager, "activate_current_adapter"):
                self.snapshot_manager.activate_current_adapter()
        batch_size = int(train_batch.x0_target.shape[0])
        if sampled_time is None:
            if timestep_idx is None:
                raise ValueError("`sampled_time` must be provided when `timestep_idx` is not set.")
            sampled_time_tensor = train_batch.timesteps[:, int(timestep_idx)].to(device=self.device, dtype=torch.float32)
        else:
            sampled_time_tensor = expand_sampled_time(
                sampled_time,
                batch_size=batch_size,
                device=self.device,
                dtype=torch.float32,
            )
        if torch.max(sampled_time_tensor).item() > 1.0 + 1e-6 or torch.min(sampled_time_tensor).item() < -1e-6:
            raise ValueError(
                f"Expected speech training timesteps in [0,1], got range "
                f"[{torch.min(sampled_time_tensor).item():.6f}, {torch.max(sampled_time_tensor).item():.6f}]"
            )
        cfm_time = sampled_time_tensor

        noisy_mel = train_batch.noisy_mel.to(device=self.device, dtype=torch.float32, non_blocking=True)
        generated_mel = train_batch.x0_target.to(device=self.device, dtype=torch.float32, non_blocking=True)

        return compute_pure_nft_terms(
            current_model=current,
            old_model=old,
            ref_model=ref,
            noisy_mel=noisy_mel,
            generated_mel=generated_mel,
            advantages=advantages,
            train_mask=train_batch.train_mask,
            train_mask_normalization=(
                "all"
                if str(getattr(self.speech_conf.train, "multi_reward_update_mode", "branch")).strip().lower()
                == "gd2po"
                else "kept"
            ),
            beta_mix=float(self.speech_conf.loss.beta_mix),
            adv_clip_max=float(self.speech_conf.loss.adv_clip_max),
            kl_coef=float(self.speech_conf.loss.kl_coef),
            time=cfm_time,
            noise=noise,
            cond_type=str(self.speech_conf.rollout.cond_type),
            adv_weight_mode=str(self.speech_conf.loss.adv_weight_mode),
            target_time=float(train_batch.nft_target_time),
        )

    def compute_multi_reward_rl_kl_loss(
        self,
        current,
        old,
        ref,
        train_batch: RolloutBatch,
        advantages_by_branch: dict[str, torch.Tensor],
        branch_weights: dict[str, float],
        config,
        sampled_time: torch.Tensor | float,
        noise: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Compute aligned reward branches while sharing current/old/ref forward passes."""
        del config
        if self._lora_enabled and self._lora_strategy == "shared_adapter":
            if hasattr(self.snapshot_manager, "activate_current_adapter"):
                self.snapshot_manager.activate_current_adapter()
        batch_size = int(train_batch.x0_target.shape[0])
        sampled_time_tensor = expand_sampled_time(
            sampled_time,
            batch_size=batch_size,
            device=self.device,
            dtype=torch.float32,
        )
        if torch.max(sampled_time_tensor).item() > 1.0 + 1e-6 or torch.min(sampled_time_tensor).item() < -1e-6:
            raise ValueError(
                f"Expected speech training timesteps in [0,1], got range "
                f"[{torch.min(sampled_time_tensor).item():.6f}, {torch.max(sampled_time_tensor).item():.6f}]"
            )

        return compute_multi_reward_pure_nft_terms(
            current_model=current,
            old_model=old,
            ref_model=ref,
            noisy_mel=train_batch.noisy_mel.to(device=self.device, dtype=torch.float32, non_blocking=True),
            generated_mel=train_batch.x0_target.to(device=self.device, dtype=torch.float32, non_blocking=True),
            advantages_by_branch=advantages_by_branch,
            branch_weights=branch_weights,
            train_mask=train_batch.train_mask,
            beta_mix=float(self.speech_conf.loss.beta_mix),
            adv_clip_max=float(self.speech_conf.loss.adv_clip_max),
            kl_coef=float(self.speech_conf.loss.kl_coef),
            time=sampled_time_tensor,
            noise=noise,
            cond_type=str(self.speech_conf.rollout.cond_type),
            adv_weight_mode=str(self.speech_conf.loss.adv_weight_mode),
            target_time=float(train_batch.nft_target_time),
        )

    def evaluate(self, current, ref, val_loader, config) -> dict[str, Any]:
        del ref, config
        limit_batches = int(self.speech_conf.eval.limit_batches)
        num_candidates = int(self.speech_conf.eval.num_candidates)

        current_model = current.module if hasattr(current, "module") else current
        current_model.eval()

        reward_sum = 0.0
        reward_count = 0.0
        raw_metric_sums: dict[str, float] = {}
        norm_metric_sums: dict[str, float] = {}

        # eval engine references current model already, but ensure eval mode
        self.eval_rollout_engine.model.eval()
        eval_deterministic_mask = self._eval_deterministic_mask()

        with torch.no_grad():
            for batch_index, batch in enumerate(val_loader):
                if limit_batches > 0 and batch_index >= limit_batches:
                    break
                rollout_result = self.eval_rollout_engine.rollout_batch(
                    batch,
                    num_candidates=num_candidates,
                    seed_offset=self._rollout_seed_offset(outer_epoch=0, batch_index=batch_index),
                    keep_waveforms=False,
                    deterministic_mask=eval_deterministic_mask,
                    reward_metric_names=self.eval_reward_metrics,
                )
                reward_tensor = rollout_result["reward"].to(torch.float32)
                reward_sum += float(reward_tensor.sum().item())
                reward_count += float(reward_tensor.numel())

                for key, value in rollout_result["reward_breakdown"]["raw"].items():
                    raw_metric_sums[key] = raw_metric_sums.get(key, 0.0) + float(value.detach().cpu().sum().item())
                for key, value in rollout_result["reward_breakdown"]["norm"].items():
                    norm_metric_sums[key] = norm_metric_sums.get(key, 0.0) + float(value.detach().cpu().sum().item())

        reduced_sum, reduced_count = self._reduce_sum_count(reward_sum, reward_count, self.device)
        reward_mean = 0.0 if reduced_count <= 0 else reduced_sum / reduced_count

        return {
            "reward_mean": float(reward_mean),
            "num_samples": int(reduced_count),
            "reward_breakdown_raw": self._reduce_metric_dict(raw_metric_sums, reward_count, self.device),
            "reward_breakdown_norm": self._reduce_metric_dict(norm_metric_sums, reward_count, self.device),
        }

    def update_old_policy(self, global_step: int) -> bool:
        return bool(self.snapshot_manager.update_old(global_step=global_step))

    def get_snapshot_distances(self) -> dict[str, dict[str, float]]:
        return self.snapshot_manager.distance_report()

    def ref_drift(self, initial_ref_state: dict[str, torch.Tensor]) -> float:
        return float(self.snapshot_manager.ref_max_abs_drift(initial_ref_state))

    def set_epoch(self, sampler, epoch: int) -> None:
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)


def set_seed(seed: int, rank: int = 0) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)
