from __future__ import annotations

import copy
import json
import os
import random
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from flow_grpo.speech_backend_adapter import SpeechBackendAdapter, set_seed
from flow_grpo.speech_resume import select_training_state, restore_training_state
from flow_grpo.speech_training_metrics import accumulate_loss_metrics_, reduce_training_metrics
from flow_grpo.speech_diagnostics import diagnostic_stage
from flow_grpo.speech_distributed import broadcast_save_decision, epoch_barrier
from flow_grpo.ema import EMAModuleWrapper
from flow_grpo.speech_nft_core import (
    build_training_timestep_grid,
    select_timestep_indices,
)


def _distributed_timeout(config=None) -> timedelta:
    timeout_minutes = os.environ.get("TORCH_DISTRIBUTED_TIMEOUT_MINUTES", "")
    if not timeout_minutes and config is not None:
        timeout_minutes = str(getattr(config, "distributed_timeout_minutes", "120"))
    if not timeout_minutes:
        timeout_minutes = "120"
    return timedelta(minutes=max(float(timeout_minutes), 1.0))


def setup_distributed(config=None) -> tuple[bool, int, int, int]:
    if dist.is_available() and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend, timeout=_distributed_timeout(config))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def cleanup_distributed(*, barrier: bool = False) -> None:
    if dist.is_available() and dist.is_initialized():
        if barrier:
            dist.barrier()
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def reduce_sum_count(sum_value: float, count_value: float, device: torch.device) -> tuple[float, float]:
    tensor = torch.tensor([sum_value, count_value], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor[0].item()), float(tensor[1].item())


def reduce_mean_from_sum_count(sum_value: float, count_value: float, device: torch.device) -> float:
    reduced_sum, reduced_count = reduce_sum_count(sum_value, count_value, device)
    if reduced_count <= 0:
        return 0.0
    return reduced_sum / reduced_count


def reduce_metric_dict(metric_sums: dict[str, float], count_value: float, device: torch.device) -> dict[str, float]:
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


def build_scheduler(optimizer: torch.optim.Optimizer, decay_steps: int) -> LambdaLR:
    decay_steps = int(decay_steps)

    def lr_lambda(step: int) -> float:
        if decay_steps <= 0:
            return 1.0
        remaining = max(decay_steps - int(step), 0)
        return float(remaining) / float(decay_steps)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


_WANDB_SCALAR_KEYS = (
    "outer_epoch",
    "global_step",
    "train_reward_mean",
    "val_reward_mean",
    "val_reward_delta_from_pretrain",
    "val_best_metric",
    "train_loss_total",
    "train_loss_rl",
    "train_loss_kl",
    "train_loss_kl_weighted",
    "timestep_grid_steps",
    "nft_target_time",
    "train_timestep_candidate_count_mean",
    "current_old_param_rmse",
    "old_ref_param_rmse",
    "ref_drift_max_abs",
    "adv_reward_std_mean",
    "adv_zero_std_ratio",
    "adv_num_groups",
    "adv_group_size_mean",
    "reward_adv_conflict_ratio",
    "reward_adv_snr_mean",
    "reward_conflict_filter_kept_sample_ratio",
    "gd2po_advantage_pre_norm_std",
    "gd2po_group_keep_ratio_mean",
    "gd2po_group_keep_ratio_std",
    "gd2po_post_norm_sign_flip_ratio",
    "gd2po_post_norm_positive_to_negative_ratio",
    "gd2po_post_norm_negative_to_positive_ratio",
    "lr",
    "grad_norm",
    "grad_norm_max",
    "old_policy_decay",
    "val_num_samples",
    "mixed_sampling_enabled",
    "mixed_sde_step_count",
    "mixed_ode_step_count",
    "mixed_sde_timestep_start",
    "mixed_sde_timestep_end",
)

_REPORT_DYNAMIC_SCALAR_PREFIXES = (
    "reward_adv_pair_valid_ratio_",
    "reward_adv_pair_agreement_",
    "reward_adv_pair_disagreement_",
    "reward_adv_pair_correlation_",
    "reward_adv_lone_dissent_ratio_",
    "reward_adv_abs_contribution_ratio_",
    "reward_filter_",
)

_WANDB_DYNAMIC_SCALAR_PREFIXES = ("reward_adv_pair_agreement_",)

_WANDB_MIXED_SAMPLING_SCALAR_PREFIXES = ("mixed_",)
_WANDB_GD2PO_CATEGORY_PREFIXES = (
    "gd2po_",
    "reward_adv_",
    "reward_conflict_filter_",
    "reward_filter_",
    "train_mask_",
)


def _as_wandb_scalar(value: Any) -> float | int | bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    return None


def _jsonable_config_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable_config_value(item) for item in value]
    if hasattr(value, "items"):
        return {str(key): _jsonable_config_value(item) for key, item in value.items()}
    return str(value)


def summarize_speech_reward_config(config: Any) -> dict[str, Any]:
    speech_conf = getattr(config, "speech", None)
    reward_conf = getattr(speech_conf, "reward", None)
    if reward_conf is None:
        return {}
    train_conf = getattr(speech_conf, "train", None)
    reward_registry = _jsonable_config_value(getattr(reward_conf, "registry", []))
    configured_eval_registry = _jsonable_config_value(getattr(reward_conf, "eval_registry", []))
    reward_eval_registry = configured_eval_registry or reward_registry
    return {
        "reward_registry": reward_registry,
        "reward_eval_registry": reward_eval_registry,
        "reward_weights": _jsonable_config_value(getattr(reward_conf, "weights", {})),
        "reward_primary_keys": _jsonable_config_value(getattr(reward_conf, "primary_keys", {})),
        "reward_normalization": _jsonable_config_value(getattr(reward_conf, "normalization", "")),
        "reward_raw_scales": _jsonable_config_value(getattr(reward_conf, "raw_scales", {})),
        "reward_branches": _jsonable_config_value(getattr(train_conf, "reward_branches", [])),
        "multi_reward_update_mode": _jsonable_config_value(
            getattr(train_conf, "multi_reward_update_mode", "branch")
        ),
        "reward_conflict_filter": _jsonable_config_value(getattr(train_conf, "reward_conflict_filter", {})),
    }


def _get_config_mapping(value: Any) -> dict[str, Any]:
    if value is None or not hasattr(value, "items"):
        return {}
    return {str(key): item for key, item in value.items()}


def compute_speech_best_metric(eval_metrics: dict[str, Any], config: Any) -> tuple[str, float]:
    best_metric = str(getattr(config, "best_metric", "val_reward_mean"))
    if best_metric in {"val_reward_mean", "reward_mean"}:
        return "val_reward_mean", float(eval_metrics.get("reward_mean", float("-inf")))

    if best_metric != "stable_multi_reward":
        raise ValueError(f"Unsupported speech best_metric: {best_metric!r}")

    speech_conf = getattr(config, "speech", None)
    reward_conf = getattr(speech_conf, "reward", None)
    if reward_conf is None:
        return "val_stable_multi_reward", float("-inf")

    registry = [str(metric).strip().lower() for metric in getattr(reward_conf, "registry", [])]
    weights = _get_config_mapping(getattr(reward_conf, "weights", {}))
    primary_keys = _get_config_mapping(getattr(reward_conf, "primary_keys", {}))
    norm_breakdown = eval_metrics.get("reward_breakdown_norm", {})
    if not isinstance(norm_breakdown, dict):
        return "val_stable_multi_reward", float("-inf")

    total = 0.0
    used_components = 0
    for metric_name in registry:
        weight = float(weights.get(metric_name, 0.0))
        if weight <= 0.0:
            continue
        primary_key = str(primary_keys.get(metric_name, metric_name))
        value = _as_wandb_scalar(norm_breakdown.get(primary_key))
        if value is None:
            continue
        total += weight * float(value)
        used_components += 1

    if used_components == 0:
        return "val_stable_multi_reward", float("-inf")
    return "val_stable_multi_reward", float(total)


def flatten_speech_wandb_metrics(
    payload: dict[str, Any],
    *,
    pretrain_val_reward_mean: float | None = None,
) -> dict[str, float | int | bool]:
    """Convert speech JSONL reports into flat WandB scalar metrics."""
    metrics: dict[str, float | int | bool] = {}
    event = str(payload.get("event", ""))
    has_eval = event == "pretrain_eval" or int(payload.get("val_num_samples", 0) or 0) > 0

    for key in _WANDB_SCALAR_KEYS:
        if key.startswith("val_") and not has_eval:
            continue
        if key.startswith(_WANDB_MIXED_SAMPLING_SCALAR_PREFIXES) and not bool(
            payload.get("mixed_sampling_enabled", False)
        ):
            continue
        if key.startswith(_WANDB_GD2PO_CATEGORY_PREFIXES) and not bool(payload.get("gd2po_enabled", False)):
            continue
        value = _as_wandb_scalar(payload.get(key))
        if value is not None:
            metric_key = f"GD2PO/{key}" if key.startswith(_WANDB_GD2PO_CATEGORY_PREFIXES) else key
            metrics[metric_key] = value

    for key, raw_value in payload.items():
        if not key.startswith(_WANDB_DYNAMIC_SCALAR_PREFIXES):
            continue
        if key.startswith(_WANDB_GD2PO_CATEGORY_PREFIXES) and not bool(payload.get("gd2po_enabled", False)):
            continue
        value = _as_wandb_scalar(raw_value)
        if value is not None:
            metric_key = f"GD2PO/{key}" if key.startswith(_WANDB_GD2PO_CATEGORY_PREFIXES) else key
            metrics[metric_key] = value

    if has_eval and pretrain_val_reward_mean is not None and "val_reward_mean" in metrics:
        metrics["val_reward_delta_from_pretrain"] = float(metrics["val_reward_mean"]) - float(pretrain_val_reward_mean)

    for prefix in ("train_reward_breakdown_raw",):
        values = payload.get(prefix)
        if not isinstance(values, dict):
            continue
        for name, value in values.items():
            scalar = _as_wandb_scalar(value)
            if scalar is None:
                continue
            metrics[f"{prefix}/{name}"] = scalar

    for prefix in ("val_reward_breakdown_raw",):
        values = payload.get(prefix)
        if not has_eval or not isinstance(values, dict):
            continue
        for name, value in values.items():
            scalar = _as_wandb_scalar(value)
            if scalar is None:
                continue
            metrics[f"{prefix}/{name}"] = scalar

    return metrics


def sum_rollout_reward_breakdown(rollout_batches: list[Any], section: str) -> dict[str, float]:
    """Sum raw/norm reward breakdown tensors over rollout batches."""
    metric_sums: dict[str, float] = {}
    for rollout_batch in rollout_batches:
        reward_dict = getattr(rollout_batch, "reward_dict", {})
        breakdown = reward_dict.get(section, {}) if isinstance(reward_dict, dict) else {}
        if not isinstance(breakdown, dict):
            continue
        for key, value in breakdown.items():
            tensor = torch.as_tensor(value, dtype=torch.float32)
            metric_sums[key] = metric_sums.get(key, 0.0) + float(tensor.sum().item())
    return metric_sums


def save_checkpoint(
    checkpoint_path: Path,
    *,
    current_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    snapshot_manager,
    outer_epoch: int,
    global_step: int,
    best_eval_reward: float,
    ema_state_dict: dict[str, Any] | None,
    config,
    training_states_by_rank=None,
    training_parameters=None,
) -> None:
    payload = {
        "checkpoint_version": 2,
        "training_parameters": training_parameters,
        "outer_epoch": int(outer_epoch),
        "global_step": int(global_step),
        "model_state_dict": current_model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "snapshot_state_dict": snapshot_manager.state_dict(),
        "best_eval_reward": float(best_eval_reward),
        "best_metric": str(getattr(config, "best_metric", "val_reward_mean")),
        "ema_state_dict": ema_state_dict,
        "config": config.to_dict() if hasattr(config, "to_dict") else {},
    }
    # Optional compatibility for callers with an existing runtime-state payload;
    # the training loop does not collect or save per-rank runtime states.
    if training_states_by_rank is not None:
        payload["training_states_by_rank"] = training_states_by_rank
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    *,
    current_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    snapshot_manager,
    device: torch.device,
    ema: EMAModuleWrapper | None = None,
    load_ema_state: bool = False,
    current_best_metric: str = "val_reward_mean",
    resume_state_out: dict | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[int, int, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if resume_state_out is not None:
        resume_state_out["training_state"] = select_training_state(checkpoint, rank, world_size)
    current_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optim_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    snapshot_state = checkpoint.get("snapshot_state_dict")
    if snapshot_state is not None:
        snapshot_manager.load_state_dict(snapshot_state)
    else:
        snapshot_manager.hard_sync_old()
        snapshot_manager.ref_model.load_state_dict(snapshot_manager.current_model.state_dict(), strict=True)
        snapshot_manager.freeze_auxiliary_models()
    ema_state = checkpoint.get("ema_state_dict")
    if load_ema_state and ema is not None and ema_state is not None:
        ema.load_state_dict(ema_state)
    # Evaluation may save EMA weights; restore the actual optimizer parameters for training.
    if checkpoint.get("training_parameters") is not None:
        with torch.no_grad():
            parameters = dict(current_model.named_parameters())
            for name, value in checkpoint["training_parameters"].items():
                parameters[name].copy_(value)
    outer_epoch = int(checkpoint.get("outer_epoch", -1)) + 1
    global_step = int(checkpoint.get("global_step", 0))
    saved_best_metric = str(checkpoint.get("best_metric", "val_reward_mean"))
    if saved_best_metric != str(current_best_metric):
        best_eval_reward = float("-inf")
    else:
        best_eval_reward = float(checkpoint.get("best_eval_reward", float("-inf")))
    return outer_epoch, global_step, best_eval_reward


class SpeechNFTOrchestrator:

    def __init__(self, config):
        self.config = config

    def run(self) -> None:
        distributed, rank, world_size, local_rank = setup_distributed(self.config)
        train_log_file = None
        wandb_run = None
        wandb_module = None
        run_failed = False
        try:
            if distributed and torch.cuda.is_available():
                device = torch.device("cuda", local_rank)
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            set_seed(int(self.config.seed), rank=rank)

            backend = SpeechBackendAdapter(self.config, device=device, rank=rank, world_size=world_size)
            model_bundle = backend.build_models_and_policies(self.config, device)

            if distributed:
                current_model = DDP(
                    model_bundle.current_model,
                    device_ids=[local_rank] if device.type == "cuda" else None,
                    output_device=local_rank if device.type == "cuda" else None,
                    find_unused_parameters=False,
                )
            else:
                current_model = model_bundle.current_model

            train_conf = self.config.speech.train
            trainable_parameters = [
                parameter for parameter in model_bundle.current_model.parameters() if parameter.requires_grad
            ]
            if not trainable_parameters:
                raise RuntimeError(
                    "No trainable parameters found on current speech model. "
                    "If LoRA is enabled, ensure target modules are matched."
                )
            optimizer = AdamW(
                trainable_parameters,
                lr=float(train_conf.learning_rate),
                betas=(float(train_conf.adam_beta1), float(train_conf.adam_beta2)),
                weight_decay=float(train_conf.adam_weight_decay),
                eps=float(train_conf.adam_epsilon),
            )
            lr_decay_steps = int(getattr(train_conf, "lr_decay_steps", getattr(train_conf, "warmup_steps", 2000)))
            scheduler = build_scheduler(optimizer, decay_steps=lr_decay_steps)

            use_ema = bool(getattr(train_conf, "ema", False))
            ema = None
            if use_ema:
                ema = EMAModuleWrapper(
                    trainable_parameters,
                    decay=float(getattr(train_conf, "ema_decay", 0.9)),
                    update_step_interval=int(getattr(train_conf, "ema_update_interval", 1)),
                    device=device,
                )
                ema.sync_with_model(trainable_parameters)

            use_amp = str(self.config.mixed_precision).lower() == "fp16" and device.type == "cuda"
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

            data_bundle = backend.build_data(self.config)
            train_sampler = data_bundle["train_sampler"]
            train_loader = data_bundle["train_loader"]
            val_sampler = data_bundle["val_sampler"]
            val_loader = data_bundle["val_loader"]

            save_dir = Path(str(self.config.save_dir)).expanduser().resolve()
            if is_main_process(rank):
                save_dir.mkdir(parents=True, exist_ok=True)
            save_best_only = bool(getattr(self.config, "save_best_only", False))
            best_ckpt_name = str(getattr(self.config, "best_ckpt_name", "checkpoint-best.pt"))
            train_log_name = str(getattr(self.config, "train_log_name", "train.log"))
            last_best_checkpoint_path: Path | None = None
            pretrain_val_reward_mean: float | None = None

            if is_main_process(rank) and bool(getattr(self.config, "wandb_enabled", True)):
                import wandb as wandb_module_local

                wandb_module = wandb_module_local
                wandb_kwargs = {
                    "project": str(getattr(self.config, "wandb_project", "flow-grpo")),
                    "name": str(getattr(self.config, "run_name", "speech_nft")),
                    "config": self.config.to_dict() if hasattr(self.config, "to_dict") else {},
                    "dir": str(save_dir),
                }
                wandb_entity = str(getattr(self.config, "wandb_entity", ""))
                wandb_mode = str(getattr(self.config, "wandb_mode", ""))
                if wandb_entity:
                    wandb_kwargs["entity"] = wandb_entity
                if wandb_mode:
                    wandb_kwargs["mode"] = wandb_mode
                wandb_run = wandb_module.init(**wandb_kwargs)

            if is_main_process(rank):
                train_log_path = save_dir / train_log_name
                train_log_path.parent.mkdir(parents=True, exist_ok=True)
                train_log_file = train_log_path.open("a", encoding="utf-8")

            def emit_train_log(payload: dict[str, Any]) -> None:
                line = json.dumps(payload, ensure_ascii=False)
                if is_main_process(rank):
                    print(line, flush=True)
                    if train_log_file is not None:
                        train_log_file.write(line + "\n")
                        train_log_file.flush()

            def emit_wandb_metrics(payload: dict[str, Any]) -> None:
                if wandb_module is None or wandb_run is None or not is_main_process(rank):
                    return
                metrics = flatten_speech_wandb_metrics(
                    payload,
                    pretrain_val_reward_mean=pretrain_val_reward_mean,
                )
                if not metrics:
                    return
                step = int(payload.get("global_step", 0) or 0)
                wandb_module.log(metrics, step=step)

            def train_sync_context(sync_gradients: bool):
                if sync_gradients or not hasattr(current_model, "no_sync"):
                    return nullcontext()
                return current_model.no_sync()

            resume_runtime_state = {}
            start_epoch = 0
            global_step = 0
            best_eval_reward = float("-inf")
            current_best_metric = str(getattr(self.config, "best_metric", "val_reward_mean"))
            resume_from = getattr(self.config, "resume_from", "")
            resume_from = "" if resume_from is None else str(resume_from)
            if resume_from and resume_from.lower() != "none":
                resume_ckpt_path = Path(resume_from).expanduser().resolve()
                resume_kind = "lora_adapter_dir" if resume_ckpt_path.is_dir() else "train_checkpoint"
                if resume_ckpt_path.is_dir():
                    if not bool(model_bundle.lora_enabled):
                        raise ValueError(
                            "Adapter-directory resume requires LoRA enabled, "
                            f"but lora_enabled={model_bundle.lora_enabled}."
                        )
                    start_epoch, global_step, best_eval_reward = backend.load_lora_resume(
                        resume_ckpt_path,
                        model_bundle,
                        current_best_metric=current_best_metric,
                    )
                else:
                    load_ema_state = bool(getattr(train_conf, "ema_resume_from_checkpoint", False))
                    start_epoch, global_step, best_eval_reward = load_checkpoint(
                        resume_ckpt_path,
                        current_model=model_bundle.current_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        snapshot_manager=backend.snapshot_manager,
                        device=device,
                        ema=ema,
                        load_ema_state=load_ema_state,
                        resume_state_out=resume_runtime_state,
                        rank=rank,
                        world_size=world_size,
                        current_best_metric=current_best_metric,
                    )
                    if save_best_only and resume_ckpt_path.parent == save_dir and resume_ckpt_path.exists():
                        last_best_checkpoint_path = resume_ckpt_path
                if is_main_process(rank):
                    emit_train_log(
                        {
                            "event": "resume_loaded",
                            "resume_from": str(resume_ckpt_path),
                            "resume_kind": resume_kind,
                            "outer_epoch": int(start_epoch),
                            "global_step": int(global_step),
                            "best_eval_reward": float(best_eval_reward),
                            "best_metric": current_best_metric,
                        }
                    )

            diagnostic_stage("pretrain_eval.begin")
            pretrain_eval_metrics = backend.evaluate(current_model, model_bundle.ref_model, val_loader, self.config)
            diagnostic_stage("pretrain_eval.end")
            pretrain_val_reward_mean = float(pretrain_eval_metrics.get("reward_mean", 0.0))
            pretrain_best_metric_name, pretrain_best_metric = compute_speech_best_metric(
                pretrain_eval_metrics,
                self.config,
            )
            reward_config_report = summarize_speech_reward_config(self.config)
            if is_main_process(rank):
                pretrain_report = {
                    "event": "pretrain_eval",
                    "outer_epoch": int(start_epoch) - 1,
                    "global_step": int(global_step),
                    "val_reward_mean": float(pretrain_val_reward_mean),
                    "val_reward_delta_from_pretrain": 0.0,
                    "val_best_metric_name": pretrain_best_metric_name,
                    "val_best_metric": float(pretrain_best_metric),
                    "val_num_samples": int(pretrain_eval_metrics.get("num_samples", 0)),
                }
                pretrain_report.update(reward_config_report)
                if "reward_breakdown_raw" in pretrain_eval_metrics:
                    pretrain_report["val_reward_breakdown_raw"] = pretrain_eval_metrics["reward_breakdown_raw"]
                    pretrain_report["val_reward_breakdown_norm"] = pretrain_eval_metrics["reward_breakdown_norm"]
                emit_train_log(pretrain_report)
                emit_wandb_metrics(pretrain_report)

            initial_ref_state = copy.deepcopy(backend.snapshot_manager.ref_model.state_dict())
            outer_epochs = int(self.config.num_epochs)

            # Restore after pretrain evaluation/setup, which can consume random numbers.
            restore_training_state(
                resume_runtime_state.get("training_state"), device,
                scaler=scaler, mixed_sampling_state=backend.mixed_sampling_state,
            )

            for outer_epoch in range(start_epoch, outer_epochs):
                diagnostic_stage("epoch.begin", epoch=outer_epoch + 1, global_step=global_step)
                backend.set_epoch(train_sampler, outer_epoch)
                backend.set_epoch(val_sampler, outer_epoch)

                rollout_buffer = []
                rollout_start = time.perf_counter()
                rollout_batches_limit = int(train_conf.rollout_batches_per_epoch)
                diagnostic_stage("rollout.loader.begin", expected_batches=rollout_batches_limit)
                for batch_index, batch in enumerate(train_loader):
                    if rollout_batches_limit > 0 and batch_index >= rollout_batches_limit:
                        break
                    seed_offset = int(outer_epoch * 100000 + batch_index * 100)
                    diagnostic_stage("rollout.batch.begin", batch=batch_index + 1)
                    rollout_batch = backend.rollout(
                        old_policy=model_bundle.old_model,
                        batch=batch,
                        config=self.config,
                        seed_offset=seed_offset,
                    )
                    rollout_buffer.append(rollout_batch)
                    diagnostic_stage("rollout.batch.end", batch=batch_index + 1)

                if not rollout_buffer:
                    raise RuntimeError("Rollout buffer is empty. Check manifests and dataloader settings.")

                diagnostic_stage("advantages.begin", batches=len(rollout_buffer))
                stat_report = backend.compute_advantages(rollout_buffer, world_context={"epoch": outer_epoch})
                diagnostic_stage("advantages.end")
                rollout_wall = time.perf_counter() - rollout_start

                current_model.train()
                model_bundle.old_model.eval()
                model_bundle.ref_model.eval()

                num_inner_epochs = int(train_conf.num_inner_epochs)
                max_grad_norm = float(train_conf.max_grad_norm)
                gradient_accumulation_steps = max(int(getattr(train_conf, "gradient_accumulation_steps", 1)), 1)
                timestep_fraction = float(getattr(train_conf, "timestep_fraction", 1.0))
                if not (0.0 < timestep_fraction <= 1.0):
                    raise ValueError("`speech.train.timestep_fraction` must be in (0, 1].")
                timestep_window = getattr(train_conf, "timestep_window", None)
                if timestep_window is not None:
                    timestep_window = [float(timestep_window[0]), float(timestep_window[1])]
                timesteps_per_batch = int(getattr(train_conf, "timesteps_per_batch", 0))
                if timesteps_per_batch < 0:
                    raise ValueError("`speech.train.timesteps_per_batch` must be non-negative.")
                timestep_grid_steps = int(getattr(train_conf, "timestep_grid_steps", 0))
                if timestep_grid_steps < 0:
                    raise ValueError("`speech.train.timestep_grid_steps` must be non-negative.")
                reward_branches = SpeechBackendAdapter._reward_branches_from_config(train_conf)
                configured_reward_branch_count = len(reward_branches)
                if reward_branches:
                    for branch in reward_branches:
                        branch_timestep_fraction = float(branch.get("timestep_fraction", 1.0))
                        if not (0.0 < branch_timestep_fraction <= 1.0):
                            raise ValueError(f"Reward branch {branch['name']!r} timestep_fraction must be in (0, 1].")
                        branch_window = branch.get("timestep_window", [0.0, 1.0])
                        branch["timestep_window"] = [float(branch_window[0]), float(branch_window[1])]
                        branch["timesteps_per_batch"] = int(branch.get("timesteps_per_batch", 0))
                        if int(branch["timesteps_per_batch"]) < 0:
                            raise ValueError(f"Reward branch {branch['name']!r} timesteps_per_batch must be non-negative.")
                        branch["timestep_grid_steps"] = int(branch.get("timestep_grid_steps", 0))
                        if int(branch["timestep_grid_steps"]) < 0:
                            raise ValueError(f"Reward branch {branch['name']!r} timestep_grid_steps must be non-negative.")
                        branch["timestep_fraction"] = branch_timestep_fraction
                        branch["loss_weight"] = float(branch.get("loss_weight", 1.0))
                        if float(branch["loss_weight"]) < 0.0:
                            raise ValueError(f"Reward branch {branch['name']!r} loss_weight must be non-negative.")
                        branch["use_default_advantage"] = False
                else:
                    reward_branches = [
                        {
                            "name": "reward_avg",
                            "loss_weight": 1.0,
                            "timestep_fraction": timestep_fraction,
                            "timestep_window": timestep_window,
                            "timesteps_per_batch": timesteps_per_batch,
                            "timestep_grid_steps": timestep_grid_steps,
                            "use_default_advantage": True,
                        }
                    ]
                multi_reward_update_mode = str(
                    getattr(train_conf, "multi_reward_update_mode", "branch")
                ).strip().lower()
                if multi_reward_update_mode not in {"branch", "gd2po"}:
                    raise ValueError(
                        "`speech.train.multi_reward_update_mode` must be either 'branch' or 'gd2po'."
                    )
                if multi_reward_update_mode == "gd2po":
                    if configured_reward_branch_count < 2:
                        raise ValueError("GD2PO requires at least two configured reward branches.")
                    gd2po_schedule_signatures = [
                        (
                            float(branch["timestep_fraction"]),
                            tuple(float(value) for value in (branch.get("timestep_window") or [0.0, 1.0])),
                            int(branch["timesteps_per_batch"]),
                            int(branch["timestep_grid_steps"]),
                        )
                        for branch in reward_branches
                    ]
                    if any(
                        signature != gd2po_schedule_signatures[0]
                        for signature in gd2po_schedule_signatures[1:]
                    ):
                        raise ValueError("GD2PO reward branches must share one timestep schedule.")
                    aggregate_schedule = reward_branches[0]
                    reward_branches = [
                        {
                            "name": "gd2po_aggregate",
                            "loss_weight": 1.0,
                            "timestep_fraction": float(aggregate_schedule["timestep_fraction"]),
                            "timestep_window": list(aggregate_schedule["timestep_window"]),
                            "timesteps_per_batch": int(aggregate_schedule["timesteps_per_batch"]),
                            "timestep_grid_steps": int(aggregate_schedule["timestep_grid_steps"]),
                            "use_default_advantage": True,
                        }
                    ]
                reward_branch_schedule_signatures = [
                    (
                        float(branch["timestep_fraction"]),
                        tuple(float(value) for value in (branch.get("timestep_window") or [0.0, 1.0])),
                        int(branch["timesteps_per_batch"]),
                        int(branch["timestep_grid_steps"]),
                    )
                    for branch in reward_branches
                ]
                shared_reward_branch_forward = (
                    len(reward_branches) > 1
                    and all(
                        signature == reward_branch_schedule_signatures[0]
                        for signature in reward_branch_schedule_signatures[1:]
                    )
                )


                # Float64 matches the previous Python-float accumulation precision.
                loss_sums = torch.zeros(5, dtype=torch.float64, device=device)
                total_batch_count = 0.0
                optimizer_step_count = 0.0
                total_selected_timesteps = 0.0
                total_available_timesteps = 0.0
                total_candidate_timesteps = 0.0
                grad_norm_sum = torch.zeros((), dtype=torch.float64, device=device)
                grad_norm_max = torch.zeros((), dtype=torch.float64, device=device)
                grad_norm_count = 0.0

                train_start = time.perf_counter()
                diagnostic_stage("train.begin")
                for _ in range(num_inner_epochs):
                    order = list(range(len(rollout_buffer)))
                    random.shuffle(order)
                    for chunk_start in range(0, len(order), gradient_accumulation_steps):
                        chunk_indices = order[chunk_start : chunk_start + gradient_accumulation_steps]
                        accum_size = len(chunk_indices)
                        optimizer.zero_grad(set_to_none=True)

                        for chunk_item_index, index in enumerate(chunk_indices):
                            diagnostic_stage("train.batch.begin", buffer_index=index, chunk_start=chunk_start)
                            rb = rollout_buffer[index]
                            if rb.advantages is None:
                                raise RuntimeError("Advantages are missing from rollout batch.")

                            total_timestep_count = int(rb.timesteps.shape[1])
                            batch_timestep_candidate_count = 0
                            batch_loss_sums = torch.zeros_like(loss_sums)
                            batch_reward_timestep_count = 0

                            if shared_reward_branch_forward:
                                shared_branch = reward_branches[0]
                                branch_advantages_by_name: dict[str, torch.Tensor] = {}
                                branch_weights_by_name: dict[str, float] = {}
                                for branch in reward_branches:
                                    branch_name = str(branch["name"])
                                    if bool(branch.get("use_default_advantage", False)):
                                        branch_advantages = rb.advantages
                                    else:
                                        if rb.reward_branch_advantages is None or branch_name not in rb.reward_branch_advantages:
                                            raise RuntimeError(f"Reward branch advantages are missing for branch {branch_name!r}.")
                                        branch_advantages = rb.reward_branch_advantages[branch_name]
                                    branch_advantages_by_name[branch_name] = branch_advantages.to(
                                        device=device,
                                        dtype=torch.float32,
                                        non_blocking=True,
                                    )
                                    branch_weights_by_name[branch_name] = float(branch["loss_weight"])

                                shared_timestep_values = build_training_timestep_grid(
                                    rb.timesteps[0],
                                    timestep_grid_steps=int(shared_branch.get("timestep_grid_steps", 0)),
                                )
                                timestep_indices = select_timestep_indices(
                                    shared_timestep_values,
                                    timestep_fraction=float(shared_branch["timestep_fraction"]),
                                    timestep_window=shared_branch.get("timestep_window"),
                                    timesteps_per_batch=int(shared_branch["timesteps_per_batch"]),
                                    rng=random,
                                )
                                branch_micro_steps = max(len(timestep_indices), 1)
                                branch_count = len(reward_branches)
                                batch_timestep_candidate_count += int(shared_timestep_values.numel()) * branch_count
                                batch_reward_timestep_count += len(timestep_indices) * branch_count

                                for timestep_position, timestep_idx in enumerate(timestep_indices):
                                    sampled_time = shared_timestep_values[int(timestep_idx)]
                                    sync_gradients = (
                                        chunk_item_index == accum_size - 1
                                        and timestep_position == len(timestep_indices) - 1
                                    )
                                    with train_sync_context(sync_gradients):
                                        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                                            terms = backend.compute_multi_reward_rl_kl_loss(
                                                current=current_model,
                                                old=model_bundle.old_model,
                                                ref=model_bundle.ref_model,
                                                train_batch=rb,
                                                advantages_by_branch=branch_advantages_by_name,
                                                branch_weights=branch_weights_by_name,
                                                sampled_time=sampled_time,
                                                config=self.config,
                                            )
                                            loss = terms["loss_total"] / float(branch_micro_steps * accum_size)

                                        diagnostic_stage("backward.begin", timestep_index=int(timestep_idx), sync_gradients=sync_gradients)
                                        if use_amp:
                                            scaler.scale(loss).backward()
                                        else:
                                            loss.backward()
                                        diagnostic_stage("backward.end", timestep_index=int(timestep_idx))

                                    accumulate_loss_metrics_(
                                        batch_loss_sums, terms, branch_micro_steps,
                                        shared_branch_count=branch_count,
                                    )
                                    del terms, loss
                            else:
                                for branch_position, branch in enumerate(reward_branches):
                                    branch_name = str(branch["name"])
                                    branch_weight = float(branch["loss_weight"])
                                    if bool(branch.get("use_default_advantage", False)):
                                        branch_advantages = rb.advantages
                                    else:
                                        if rb.reward_branch_advantages is None or branch_name not in rb.reward_branch_advantages:
                                            raise RuntimeError(f"Reward branch advantages are missing for branch {branch_name!r}.")
                                        branch_advantages = rb.reward_branch_advantages[branch_name]
                                    branch_advantages = branch_advantages.to(
                                        device=device,
                                        dtype=torch.float32,
                                        non_blocking=True,
                                    )

                                    branch_timestep_values = build_training_timestep_grid(
                                        rb.timesteps[0],
                                        timestep_grid_steps=int(branch.get("timestep_grid_steps", 0)),
                                    )
                                    batch_timestep_candidate_count += int(branch_timestep_values.numel())
                                    timestep_indices = select_timestep_indices(
                                        branch_timestep_values,
                                        timestep_fraction=float(branch["timestep_fraction"]),
                                        timestep_window=branch.get("timestep_window"),
                                        timesteps_per_batch=int(branch["timesteps_per_batch"]),
                                        rng=random,
                                    )
                                    branch_micro_steps = max(len(timestep_indices), 1)
                                    batch_reward_timestep_count += len(timestep_indices)
                                    for timestep_position, timestep_idx in enumerate(timestep_indices):
                                        sampled_time = branch_timestep_values[int(timestep_idx)]
                                        sync_gradients = (
                                            chunk_item_index == accum_size - 1
                                            and branch_position == len(reward_branches) - 1
                                            and timestep_position == len(timestep_indices) - 1
                                        )
                                        with train_sync_context(sync_gradients):
                                            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                                                terms = backend.compute_rl_kl_loss(
                                                    current=current_model,
                                                    old=model_bundle.old_model,
                                                    ref=model_bundle.ref_model,
                                                    train_batch=rb,
                                                    advantages=branch_advantages,
                                                    sampled_time=sampled_time,
                                                    config=self.config,
                                                )
                                                branch_loss = terms["loss_total"] * branch_weight
                                                loss = branch_loss / float(branch_micro_steps * accum_size)

                                            diagnostic_stage("backward.begin", timestep_index=int(timestep_idx), sync_gradients=sync_gradients)
                                            if use_amp:
                                                scaler.scale(loss).backward()
                                            else:
                                                loss.backward()
                                            diagnostic_stage("backward.end", timestep_index=int(timestep_idx))

                                        accumulate_loss_metrics_(
                                            batch_loss_sums, terms, branch_micro_steps,
                                            branch_weight=branch_weight,
                                            weighted_loss=branch_loss,
                                        )
                                        del terms, branch_loss, loss

                            batch_loss_sums[4].div_(float(max(len(reward_branches), 1)))
                            loss_sums.add_(batch_loss_sums)
                            total_batch_count += 1.0
                            total_selected_timesteps += float(batch_reward_timestep_count)
                            total_available_timesteps += float(total_timestep_count)
                            total_candidate_timesteps += float(batch_timestep_candidate_count)
                            diagnostic_stage("train.batch.end", buffer_index=index)

                        diagnostic_stage("optimizer.begin", global_step=global_step)
                        if use_amp:
                            scaler.unscale_(optimizer)
                            total_grad_norm = clip_grad_norm_(trainable_parameters, max_norm=max_grad_norm)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            total_grad_norm = clip_grad_norm_(trainable_parameters, max_norm=max_grad_norm)
                            optimizer.step()
                        scheduler.step()
                        diagnostic_stage("optimizer.end")
                        grad_norm_value = torch.as_tensor(
                            total_grad_norm, device=device, dtype=torch.float64,
                        ).detach()
                        grad_norm_sum.add_(grad_norm_value)
                        grad_norm_max = torch.where(grad_norm_value > grad_norm_max, grad_norm_value, grad_norm_max)
                        grad_norm_count += 1.0

                        global_step += 1
                        optimizer_step_count += 1.0
                        if ema is not None:
                            ema.step(trainable_parameters, global_step)

                # One reporting transfer per outer epoch also waits for queued GPU work,
                diagnostic_stage("train_metrics.reduce.begin")
                # keeping train_wall_time_sec meaningful after removing step-wise syncs.
                (
                    train_loss_total_mean, train_loss_rl_mean, train_loss_kl_mean,
                    train_loss_kl_weighted_mean, train_old_deviate_mean,
                    grad_norm_mean, grad_norm_max_global,
                ) = reduce_training_metrics(
                    loss_sums, total_batch_count, grad_norm_sum, grad_norm_count, grad_norm_max,
                )
                train_wall = time.perf_counter() - train_start
                diagnostic_stage("train_metrics.reduce.end")

                diagnostic_stage("snapshot.update.begin")
                backend.update_old_policy(global_step=global_step)
                distance_report = backend.get_snapshot_distances()
                ref_drift = backend.ref_drift(initial_ref_state)
                diagnostic_stage("snapshot.update.end")

                if ema is not None:
                    ema.copy_ema_to(trainable_parameters, store_temp=True)

                try:
                    eval_metrics = {"reward_mean": 0.0, "num_samples": 0}
                    evaluated_this_epoch = False
                    if int(self.config.eval_freq) > 0 and ((outer_epoch + 1) % int(self.config.eval_freq) == 0):
                        diagnostic_stage("eval.begin")
                        eval_metrics = backend.evaluate(current_model, model_bundle.ref_model, val_loader, self.config)
                        diagnostic_stage("eval.end")
                        evaluated_this_epoch = True
                    val_best_metric_name = ""
                    val_best_metric: float | None = None
                    if evaluated_this_epoch:
                        val_best_metric_name, val_best_metric = compute_speech_best_metric(eval_metrics, self.config)

                    train_reward_sum = sum(float(rb.reward_avg.sum().item()) for rb in rollout_buffer)
                    diagnostic_stage("reward_metrics.reduce.begin")
                    train_reward_count = sum(float(rb.reward_avg.numel()) for rb in rollout_buffer)
                    train_reward_mean = reduce_mean_from_sum_count(train_reward_sum, train_reward_count, device)
                    train_reward_breakdown_raw = reduce_metric_dict(
                        sum_rollout_reward_breakdown(rollout_buffer, "raw"),
                        train_reward_count,
                        device,
                    )
                    train_reward_breakdown_norm = reduce_metric_dict(
                        sum_rollout_reward_breakdown(rollout_buffer, "norm"),
                        train_reward_count,
                        device,
                    )
                    diagnostic_stage("reward_metrics.reduce.end")
                    if is_main_process(rank):
                        diagnostic_stage("logging.begin")
                        val_reward_mean = float(eval_metrics.get("reward_mean", 0.0))
                        mixed_sampling_report = backend.summarize_mixed_sampling(rollout_buffer)
                        report = {
                            "outer_epoch": int(outer_epoch),
                            "global_step": int(global_step),
                            "rollout_batches": int(len(rollout_buffer)),
                            "optimizer_steps": int(optimizer_step_count),
                            "gradient_accumulation_steps": int(gradient_accumulation_steps),
                            "timestep_fraction": float(timestep_fraction),
                            "timesteps_per_batch": int(timesteps_per_batch),
                            "timestep_grid_steps": int(timestep_grid_steps),
                            "nft_target_time": float(rollout_buffer[0].nft_target_time),
                            "timestep_window": list(timestep_window) if timestep_window is not None else None,
                            "train_timestep_count_mean": (
                                float(total_selected_timesteps / total_batch_count) if total_batch_count > 0 else 0.0
                            ),
                            "train_timestep_candidate_count_mean": (
                                float(total_candidate_timesteps / total_batch_count) if total_batch_count > 0 else 0.0
                            ),
                            "rollout_timestep_count_mean": (
                                float(total_available_timesteps / total_batch_count) if total_batch_count > 0 else 0.0
                            ),
                            "ema_enabled": bool(ema is not None),
                            "lora_enabled": bool(model_bundle.lora_enabled),
                            "lora_strategy": str(model_bundle.lora_strategy),
                            "rollout_wall_time_sec": float(rollout_wall),
                            "train_wall_time_sec": float(train_wall),
                            "train_reward_mean": float(train_reward_mean),
                            "train_loss_total": float(train_loss_total_mean),
                            "train_loss_rl": float(train_loss_rl_mean),
                            "train_loss_kl": float(train_loss_kl_mean),
                            "train_loss_kl_weighted": float(train_loss_kl_weighted_mean),
                            "train_old_deviate": float(train_old_deviate_mean),
                            "lr": float(optimizer.param_groups[0]["lr"]),
                            "grad_norm": float(grad_norm_mean),
                            "grad_norm_max": float(grad_norm_max_global),
                            "adv_num_groups": int(stat_report.get("num_groups", 0)),
                            "adv_group_size_mean": float(stat_report.get("group_size_mean", 0.0)),
                            "adv_zero_std_ratio": float(stat_report.get("zero_std_ratio", 0.0)),
                            "adv_reward_std_mean": float(stat_report.get("reward_std_mean", 0.0)),
                            "reward_adv_conflict_branch_count": int(
                                stat_report.get("reward_adv_conflict_branch_count", 0)
                            ),
                            "reward_adv_conflict_sample_count": int(
                                stat_report.get("reward_adv_conflict_sample_count", 0)
                            ),
                            "reward_adv_conflict_ratio": float(
                                stat_report.get("reward_adv_conflict_ratio", 0.0)
                            ),
                            "reward_adv_consensus_ratio": float(
                                stat_report.get("reward_adv_consensus_ratio", 0.0)
                            ),
                            "reward_adv_neutral_ratio": float(
                                stat_report.get("reward_adv_neutral_ratio", 0.0)
                            ),
                            "reward_adv_snr_mean": float(stat_report.get("reward_adv_snr_mean", 0.0)),
                            "reward_adv_snr_min": float(stat_report.get("reward_adv_snr_min", 0.0)),
                            "reward_adv_snr_retained_ratio_tau_0_2": float(
                                stat_report.get("reward_adv_snr_retained_ratio_tau_0_2", 0.0)
                            ),
                            "reward_adv_snr_retained_ratio_tau_0_5": float(
                                stat_report.get("reward_adv_snr_retained_ratio_tau_0_5", 0.0)
                            ),
                            "reward_adv_snr_retained_ratio_tau_0_8": float(
                                stat_report.get("reward_adv_snr_retained_ratio_tau_0_8", 0.0)
                            ),
                            "reward_conflict_filter_enabled": bool(
                                stat_report.get("reward_conflict_filter_enabled", False)
                            ),
                            "reward_conflict_filter_tau": float(
                                stat_report.get("reward_conflict_filter_tau", 0.0)
                            ),
                            "reward_conflict_filter_sample_count": int(
                                stat_report.get("reward_conflict_filter_sample_count", 0)
                            ),
                            "reward_conflict_filter_kept_sample_count": int(
                                stat_report.get("reward_conflict_filter_kept_sample_count", 0)
                            ),
                            "reward_conflict_filter_kept_sample_ratio": float(
                                stat_report.get("reward_conflict_filter_kept_sample_ratio", 0.0)
                            ),
                            "reward_conflict_filter_filtered_sample_count": int(
                                stat_report.get("reward_conflict_filter_filtered_sample_count", 0)
                            ),
                            "reward_conflict_filter_filtered_sample_ratio": float(
                                stat_report.get("reward_conflict_filter_filtered_sample_ratio", 0.0)
                            ),
                            "train_mask_kept_sample_count": int(
                                stat_report.get("train_mask_kept_sample_count", 0)
                            ),
                            "train_mask_kept_sample_ratio": float(
                                stat_report.get("train_mask_kept_sample_ratio", 0.0)
                            ),
                            "gd2po_enabled": bool(stat_report.get("gd2po_enabled", False)),
                            "gd2po_advantage_mean": float(stat_report.get("gd2po_advantage_mean", 0.0)),
                            "gd2po_advantage_std": float(stat_report.get("gd2po_advantage_std", 0.0)),
                            "gd2po_post_normalization_enabled": bool(
                                stat_report.get("gd2po_post_normalization_enabled", False)
                            ),
                            "gd2po_advantage_pre_norm_mean": float(
                                stat_report.get("gd2po_advantage_pre_norm_mean", 0.0)
                            ),
                            "gd2po_advantage_pre_norm_std": float(
                                stat_report.get("gd2po_advantage_pre_norm_std", 0.0)
                            ),
                            "gd2po_advantage_post_norm_kept_mean": float(
                                stat_report.get("gd2po_advantage_post_norm_kept_mean", 0.0)
                            ),
                            "gd2po_advantage_post_norm_kept_std": float(
                                stat_report.get("gd2po_advantage_post_norm_kept_std", 0.0)
                            ),
                            "gd2po_post_norm_sign_flip_ratio": float(
                                stat_report.get("gd2po_post_norm_sign_flip_ratio", 0.0)
                            ),
                            "gd2po_post_norm_positive_to_negative_ratio": float(
                                stat_report.get("gd2po_post_norm_positive_to_negative_ratio", 0.0)
                            ),
                            "gd2po_post_norm_negative_to_positive_ratio": float(
                                stat_report.get("gd2po_post_norm_negative_to_positive_ratio", 0.0)
                            ),
                            "low_std_filter_enabled": bool(stat_report.get("low_std_filter_enabled", False)),
                            "low_std_filter_group_count": int(stat_report.get("low_std_filter_group_count", 0)),
                            "low_std_filter_group_ratio": float(stat_report.get("low_std_filter_group_ratio", 0.0)),
                            "low_std_filter_sample_count": int(stat_report.get("low_std_filter_sample_count", 0)),
                            "low_std_filter_sample_ratio": float(stat_report.get("low_std_filter_sample_ratio", 0.0)),
                            "low_std_filter_kept_sample_count": int(
                                stat_report.get("low_std_filter_kept_sample_count", 0)
                            ),
                            "val_reward_mean": val_reward_mean,
                            "val_num_samples": int(eval_metrics.get("num_samples", 0)),
                            "old_policy_decay": float(backend.snapshot_manager.last_decay),
                            "current_old_param_rmse": float(distance_report["current_old"]["rmse"]),
                            "old_ref_param_rmse": float(distance_report["old_ref"]["rmse"]),
                            "ref_drift_max_abs": float(ref_drift),
                        }
                        for key, value in stat_report.items():
                            if key.startswith(_REPORT_DYNAMIC_SCALAR_PREFIXES):
                                report[key] = float(value)
                        if evaluated_this_epoch and val_best_metric is not None:
                            report["val_best_metric_name"] = val_best_metric_name
                            report["val_best_metric"] = float(val_best_metric)
                        report.update(reward_config_report)
                        report.update(mixed_sampling_report)
                        if train_reward_breakdown_raw:
                            report["train_reward_breakdown_raw"] = train_reward_breakdown_raw
                            report["train_reward_breakdown_norm"] = train_reward_breakdown_norm
                        if evaluated_this_epoch and pretrain_val_reward_mean is not None:
                            report["val_reward_delta_from_pretrain"] = val_reward_mean - float(pretrain_val_reward_mean)
                        if "reward_breakdown_raw" in eval_metrics:
                            report["val_reward_breakdown_raw"] = eval_metrics["reward_breakdown_raw"]
                            report["val_reward_breakdown_norm"] = eval_metrics["reward_breakdown_norm"]
                        emit_wandb_metrics(report)
                        diagnostic_stage("logging.wandb.end")
                        if evaluated_this_epoch:
                            eval_report = {
                                "event": "eval",
                                "outer_epoch": int(outer_epoch),
                                "global_step": int(global_step),
                                "val_reward_mean": val_reward_mean,
                                "val_num_samples": int(eval_metrics.get("num_samples", 0)),
                            }
                            if val_best_metric is not None:
                                eval_report["val_best_metric_name"] = val_best_metric_name
                                eval_report["val_best_metric"] = float(val_best_metric)
                            eval_report.update(reward_config_report)
                            if pretrain_val_reward_mean is not None:
                                eval_report["val_reward_delta_from_pretrain"] = (
                                    val_reward_mean - float(pretrain_val_reward_mean)
                                )
                            if "reward_breakdown_raw" in eval_metrics:
                                eval_report["val_reward_breakdown_raw"] = eval_metrics["reward_breakdown_raw"]
                                eval_report["val_reward_breakdown_norm"] = eval_metrics["reward_breakdown_norm"]
                            emit_train_log(eval_report)

                    if int(self.config.save_freq) > 0 and ((outer_epoch + 1) % int(self.config.save_freq) == 0):
                        if is_main_process(rank):
                            improved = False
                            if evaluated_this_epoch:
                                _, eval_reward = compute_speech_best_metric(eval_metrics, self.config)
                                if eval_reward > best_eval_reward:
                                    best_eval_reward = eval_reward
                                    improved = True

                            should_save = (not save_best_only) or improved
                        # Every rank participates, even when this epoch has no validation/save.
                        diagnostic_stage("save_decision.broadcast.begin", device=str(device))
                        should_save, best_eval_reward = broadcast_save_decision(
                            should_save if is_main_process(rank) else False,
                            best_eval_reward,
                            device,
                        )
                        diagnostic_stage("save_decision.broadcast.end", should_save=should_save)
                        if should_save:
                            # No per-rank runtime-state collective before writing.
                            if is_main_process(rank):
                                diagnostic_stage("checkpoint.write.begin")
                                if save_best_only:
                                    best_name = Path(best_ckpt_name)
                                    best_suffix = best_name.suffix if best_name.suffix else ".pt"
                                    best_stem = best_name.stem if best_name.suffix else best_ckpt_name
                                    checkpoint_path = save_dir / f"{best_stem}-epoch-{outer_epoch + 1:04d}{best_suffix}"
                                else:
                                    checkpoint_path = save_dir / f"checkpoint-{outer_epoch + 1:04d}.pt"
                                save_checkpoint(
                                    checkpoint_path,
                                    current_model=model_bundle.current_model,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    snapshot_manager=backend.snapshot_manager,
                                    outer_epoch=outer_epoch,
                                    global_step=global_step,
                                    best_eval_reward=best_eval_reward,
                                    ema_state_dict=(None if ema is None else ema.state_dict()),
                                    config=self.config,
                                    training_parameters=(None if ema is None else {
                                        name: value
                                        for (name, _), value in zip(
                                            ((name, parameter) for name, parameter in model_bundle.current_model.named_parameters()
                                             if parameter.requires_grad),
                                            ema.temp_stored_parameters,
                                        )
                                    }),
                                )
                                backend.save_lora_checkpoint_artifacts(
                                    checkpoint_path,
                                    model_bundle,
                                    outer_epoch=outer_epoch,
                                    global_step=global_step,
                                    best_eval_reward=best_eval_reward,
                                    best_metric=current_best_metric,
                                )
                                if save_best_only:
                                    if (
                                        last_best_checkpoint_path is not None
                                        and last_best_checkpoint_path != checkpoint_path
                                        and last_best_checkpoint_path.exists()
                                    ):
                                        try:
                                            last_best_checkpoint_path.unlink()
                                        except OSError:
                                            pass
                                        backend.cleanup_lora_checkpoint_artifacts(last_best_checkpoint_path)
                                    last_best_checkpoint_path = checkpoint_path
                                diagnostic_stage("checkpoint.write.end")
                finally:
                    if ema is not None:
                        ema.copy_temp_to(trainable_parameters)

                if dist.is_available() and dist.is_initialized():
                    diagnostic_stage("epoch.barrier.begin", device=str(device))
                    epoch_barrier(device)
                    diagnostic_stage("epoch.barrier.end")
                diagnostic_stage("epoch.end", epoch=outer_epoch + 1, global_step=global_step)

        except BaseException as exc:
            run_failed = True
            print(
                f"[rank{rank}] SpeechNFTOrchestrator failed with {type(exc).__name__}: {exc}",
                flush=True,
            )
            raise

        finally:
            if train_log_file is not None:
                train_log_file.close()
            cleanup_distributed()
            if wandb_run is not None:
                try:
                    wandb_run.finish(exit_code=1 if run_failed else 0)
                except Exception as wandb_exc:
                    print(
                        f"[rank{rank}] wandb finish failed with {type(wandb_exc).__name__}: {wandb_exc}",
                        flush=True,
                    )
