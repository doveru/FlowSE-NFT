from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

import torch


@dataclass
class PolicySnapshotConfig:
    decay_type: int | None = 1
    ema_decay: float | None = None
    ema_update_interval: int = 1
    track_buffers: bool = True


def _normalize_config(config: PolicySnapshotConfig | dict[str, Any] | None) -> PolicySnapshotConfig:
    if config is None:
        return PolicySnapshotConfig()
    if isinstance(config, PolicySnapshotConfig):
        return config
    if is_dataclass(config):
        return PolicySnapshotConfig(**asdict(config))
    if isinstance(config, dict):
        return PolicySnapshotConfig(**config)
    raise TypeError(f"Unsupported policy snapshot config type: {type(config)!r}")


def diffusion_nft_decay(step: int, decay_type: int) -> float:
    if decay_type == 0:
        flat = 0
        uprate = 0.0
        uphold = 0.0
    elif decay_type == 1:
        flat = 0
        uprate = 0.001
        uphold = 0.5
    elif decay_type == 2:
        flat = 75
        uprate = 0.0075
        uphold = 0.999
    else:
        raise ValueError(f"Unsupported decay_type: {decay_type!r}")

    if step < flat:
        return 0.0
    decay = (step - flat) * uprate
    return float(min(decay, uphold))


def _freeze_model(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _pair_distance(model_a: torch.nn.Module, model_b: torch.nn.Module) -> dict[str, float]:
    sum_sq = 0.0
    count = 0
    max_abs = 0.0
    for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
        diff = (param_a.detach() - param_b.detach()).to(torch.float32)
        if diff.numel() == 0:
            continue
        sum_sq += float(torch.sum(diff * diff).item())
        count += int(diff.numel())
        max_abs = max(max_abs, float(torch.max(torch.abs(diff)).item()))

    rmse = (sum_sq / count) ** 0.5 if count > 0 else 0.0
    return {
        "rmse": rmse,
        "max_abs": max_abs,
        "num_elements": count,
    }


def _max_abs_state_dict_diff(state_a: dict[str, torch.Tensor], state_b: dict[str, torch.Tensor]) -> float:
    max_abs = 0.0
    for key in state_a:
        tensor_a = state_a[key]
        tensor_b = state_b[key]
        if not isinstance(tensor_a, torch.Tensor) or not isinstance(tensor_b, torch.Tensor):
            continue
        diff = tensor_a.detach().to(torch.float32) - tensor_b.detach().to(torch.float32)
        if diff.numel() == 0:
            continue
        max_abs = max(max_abs, float(torch.max(torch.abs(diff)).item()))
    return max_abs


class PolicySnapshotManager:
    def __init__(
        self,
        current_model: torch.nn.Module,
        old_model: torch.nn.Module,
        ref_model: torch.nn.Module,
        config: PolicySnapshotConfig | dict[str, Any] | None = None,
    ):
        self.config = _normalize_config(config)
        if self.config.decay_type is not None and self.config.decay_type not in {0, 1, 2}:
            raise ValueError("`decay_type` must be one of {0, 1, 2} when provided.")
        if self.config.ema_decay is not None and not (0.0 <= self.config.ema_decay < 1.0):
            raise ValueError("`ema_decay` must be in [0, 1) when provided.")
        if self.config.decay_type is None and self.config.ema_decay is None:
            raise ValueError("Either `decay_type` or `ema_decay` must be configured.")
        if self.config.ema_update_interval <= 0:
            raise ValueError("`ema_update_interval` must be positive.")

        self.current_model = current_model
        self.old_model = _freeze_model(old_model)
        self.ref_model = _freeze_model(ref_model)
        self.num_ema_updates = 0
        self.last_decay = 0.0

    def freeze_auxiliary_models(self) -> None:
        _freeze_model(self.old_model)
        _freeze_model(self.ref_model)

    @torch.no_grad()
    def hard_sync_old(self) -> None:
        self.old_model.load_state_dict(self.current_model.state_dict(), strict=True)
        self.last_decay = 0.0
        self.freeze_auxiliary_models()

    @torch.no_grad()
    def resolve_decay(self, global_step: int | None = None, decay: float | None = None) -> float:
        if decay is not None:
            resolved_decay = float(decay)
        elif self.config.decay_type is not None:
            if global_step is None:
                raise ValueError("`global_step` is required when using DiffusionNFT-style decay.")
            resolved_decay = diffusion_nft_decay(int(global_step), int(self.config.decay_type))
        elif self.config.ema_decay is not None:
            resolved_decay = float(self.config.ema_decay)
        else:
            raise RuntimeError("No valid old-policy decay configuration found.")

        if not (0.0 <= resolved_decay < 1.0):
            raise ValueError(f"Resolved decay must be in [0, 1), got {resolved_decay}.")
        return resolved_decay

    @torch.no_grad()
    def update_old(self, global_step: int | None = None, decay: float | None = None) -> bool:
        if decay is None and self.config.decay_type is None and global_step is not None:
            if (global_step + 1) % self.config.ema_update_interval != 0:
                return False

        decay = self.resolve_decay(global_step=global_step, decay=decay)
        one_minus_decay = 1.0 - decay

        if one_minus_decay <= 0.0:
            return False

        for old_param, current_param in zip(
            self.old_model.parameters(),
            self.current_model.parameters(),
        ):
            if not bool(current_param.requires_grad):
                continue
            current_data = current_param.detach()
            if torch.is_floating_point(old_param):
                if old_param.device == current_data.device:
                    old_param.data.mul_(decay).add_(current_data.data, alpha=one_minus_decay)
                else:
                    current_copy = current_data.to(device=old_param.device, dtype=old_param.dtype)
                    old_param.data.mul_(decay).add_(current_copy, alpha=one_minus_decay)
            else:
                old_param.data.copy_(current_data.to(device=old_param.device))

        if self.config.track_buffers:
            for old_buffer, current_buffer in zip(
                self.old_model.buffers(),
                self.current_model.buffers(),
            ):
                old_buffer.data.copy_(current_buffer.detach().to(device=old_buffer.device, dtype=old_buffer.dtype))

        self.num_ema_updates += 1
        self.last_decay = float(decay)
        self.freeze_auxiliary_models()
        return True

    @torch.no_grad()
    def ema_update_old(self, global_step: int | None = None) -> bool:
        return self.update_old(global_step=global_step)

    def distance_report(self) -> dict[str, dict[str, float]]:
        return {
            "current_old": _pair_distance(self.current_model, self.old_model),
            "old_ref": _pair_distance(self.old_model, self.ref_model),
            "current_ref": _pair_distance(self.current_model, self.ref_model),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay_type": None if self.config.decay_type is None else int(self.config.decay_type),
            "ema_decay": None if self.config.ema_decay is None else float(self.config.ema_decay),
            "ema_update_interval": int(self.config.ema_update_interval),
            "track_buffers": bool(self.config.track_buffers),
            "num_ema_updates": int(self.num_ema_updates),
            "last_decay": float(self.last_decay),
            "old_model_state_dict": copy.deepcopy(self.old_model.state_dict()),
            "ref_model_state_dict": copy.deepcopy(self.ref_model.state_dict()),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "old_model_state_dict" not in state_dict or "ref_model_state_dict" not in state_dict:
            raise KeyError("Policy snapshot state dict must include old/ref model states.")

        if "decay_type" in state_dict:
            decay_type = state_dict["decay_type"]
            self.config.decay_type = None if decay_type is None else int(decay_type)
        if "ema_decay" in state_dict:
            ema_decay = state_dict["ema_decay"]
            self.config.ema_decay = None if ema_decay is None else float(ema_decay)
        if "ema_update_interval" in state_dict:
            self.config.ema_update_interval = int(state_dict["ema_update_interval"])
        if "track_buffers" in state_dict:
            self.config.track_buffers = bool(state_dict["track_buffers"])
        self.num_ema_updates = int(state_dict.get("num_ema_updates", 0))
        self.last_decay = float(state_dict.get("last_decay", 0.0))

        self.old_model.load_state_dict(state_dict["old_model_state_dict"], strict=True)
        self.ref_model.load_state_dict(state_dict["ref_model_state_dict"], strict=True)
        self.freeze_auxiliary_models()

    def ref_max_abs_drift(self, reference_state_dict: dict[str, torch.Tensor]) -> float:
        return _max_abs_state_dict_diff(reference_state_dict, self.ref_model.state_dict())


def build_policy_snapshot_manager(
    base_model: torch.nn.Module,
    config: PolicySnapshotConfig | dict[str, Any] | None = None,
) -> PolicySnapshotManager:
    current_model = base_model
    old_model = copy.deepcopy(base_model)
    ref_model = copy.deepcopy(base_model)
    manager = PolicySnapshotManager(
        current_model=current_model,
        old_model=old_model,
        ref_model=ref_model,
        config=config,
    )
    manager.hard_sync_old()
    manager.ref_model.load_state_dict(manager.current_model.state_dict(), strict=True)
    manager.freeze_auxiliary_models()
    return manager


class _AdapterRouteModel(torch.nn.Module):

    def __init__(self, base_model: torch.nn.Module, route: str):
        super().__init__()
        if route not in {"default", "old", "disable"}:
            raise ValueError(f"Unsupported adapter route: {route!r}")
        self.base_model = base_model
        self.route = route

    @staticmethod
    def _normalize_active_adapter(active_adapter: Any) -> Any:
        if isinstance(active_adapter, (list, tuple)):
            if len(active_adapter) == 1:
                return active_adapter[0]
            return list(active_adapter)
        return active_adapter

    @contextmanager
    def _adapter_scope(self):
        transformer = self.base_model.transformer
        if self.route == "disable":
            with transformer.disable_adapter():
                yield
            return

        previous = self._normalize_active_adapter(getattr(transformer, "active_adapter", None))
        transformer.set_adapter(self.route)
        try:
            yield
        finally:
            if previous is not None and previous != self.route:
                try:
                    transformer.set_adapter(previous)
                except Exception:
                    transformer.set_adapter("default")

    def train(self, mode: bool = True):
        self.training = bool(mode)
        return self

    def eval(self):
        self.training = False
        return self

    def parameters(self, recurse: bool = True):
        return self.base_model.parameters(recurse=recurse)

    @property
    def mel_spec(self):
        return self.base_model.mel_spec

    @property
    def num_channels(self):
        return self.base_model.num_channels

    def prepare_condition(self, *args, **kwargs):
        return self.base_model.prepare_condition(*args, **kwargs)

    def prepare_text(self, *args, **kwargs):
        return self.base_model.prepare_text(*args, **kwargs)

    def predict_flow(self, *args, **kwargs):
        with self._adapter_scope():
            return self.base_model.predict_flow(*args, **kwargs)

    def sample(self, *args, **kwargs):
        with self._adapter_scope():
            return self.base_model.sample(*args, **kwargs)

    def forward(self, *args, **kwargs):
        with self._adapter_scope():
            return self.base_model(*args, **kwargs)


class LoraAdapterPolicySnapshotManager:

    def __init__(
        self,
        current_model: torch.nn.Module,
        config: PolicySnapshotConfig | dict[str, Any] | None = None,
        *,
        current_adapter_name: str = "default",
        old_adapter_name: str = "old",
    ):
        self.config = _normalize_config(config)
        if self.config.decay_type is not None and self.config.decay_type not in {0, 1, 2}:
            raise ValueError("`decay_type` must be one of {0, 1, 2} when provided.")
        if self.config.ema_decay is not None and not (0.0 <= self.config.ema_decay < 1.0):
            raise ValueError("`ema_decay` must be in [0, 1) when provided.")
        if self.config.decay_type is None and self.config.ema_decay is None:
            raise ValueError("Either `decay_type` or `ema_decay` must be configured.")
        if self.config.ema_update_interval <= 0:
            raise ValueError("`ema_update_interval` must be positive.")

        self.current_model = current_model
        self.current_adapter_name = str(current_adapter_name)
        self.old_adapter_name = str(old_adapter_name)
        self.num_ema_updates = 0
        self.last_decay = 0.0

        transformer = self.current_model.transformer
        if not hasattr(transformer, "set_adapter"):
            raise TypeError("LoRA adapter snapshot manager requires a PEFT-wrapped transformer.")

        self._ensure_old_adapter()
        self.old_model = _AdapterRouteModel(self.current_model, route=self.old_adapter_name)
        self.ref_model = _AdapterRouteModel(self.current_model, route="disable")
        self.activate_current_adapter()
        self.hard_sync_old()

    def _ensure_old_adapter(self) -> None:
        transformer = self.current_model.transformer
        peft_config = getattr(transformer, "peft_config", None)
        if peft_config is None:
            raise TypeError("Transformer is missing `peft_config`; expected PEFT model.")

        if self.current_adapter_name not in peft_config:
            raise KeyError(
                f"Current adapter {self.current_adapter_name!r} not found in transformer.peft_config."
            )

        if self.old_adapter_name not in peft_config:
            old_adapter_cfg = copy.deepcopy(peft_config[self.current_adapter_name])
            transformer.add_adapter(self.old_adapter_name, old_adapter_cfg)

    def _set_trainable_adapter(self, adapter_name: str) -> None:
        token = f".{adapter_name}."
        for name, parameter in self.current_model.named_parameters():
            if "lora_" not in name:
                parameter.requires_grad_(False)
                continue
            parameter.requires_grad_(token in name)

    def _iter_named_adapter_params(self, adapter_name: str):
        token = f".{adapter_name}."
        for name, parameter in self.current_model.named_parameters():
            if "lora_" not in name:
                continue
            if token not in name:
                continue
            yield name, parameter

    def _iter_adapter_pairs(self, src_adapter: str, dst_adapter: str):
        src_token = f".{src_adapter}."
        dst_token = f".{dst_adapter}."
        named_parameters = dict(self.current_model.named_parameters())
        for src_name, src_parameter in named_parameters.items():
            if "lora_" not in src_name or src_token not in src_name:
                continue
            dst_name = src_name.replace(src_token, dst_token, 1)
            dst_parameter = named_parameters.get(dst_name)
            if dst_parameter is None:
                continue
            yield src_name, src_parameter, dst_name, dst_parameter

    @staticmethod
    def _copy_tensor(dst: torch.Tensor, src: torch.Tensor) -> None:
        if dst.device == src.device and dst.dtype == src.dtype:
            dst.data.copy_(src.data)
            return
        dst.data.copy_(src.data.to(device=dst.device, dtype=dst.dtype))

    def _copy_adapter_state(self, src_adapter: str, dst_adapter: str) -> int:
        copied = 0
        for _, src_param, _, dst_param in self._iter_adapter_pairs(src_adapter, dst_adapter):
            self._copy_tensor(dst_param, src_param.detach())
            copied += 1
        return copied

    def _adapter_distance(self, adapter_a: str, adapter_b: str) -> dict[str, float]:
        sum_sq = 0.0
        count = 0
        max_abs = 0.0
        for _, param_a, _, param_b in self._iter_adapter_pairs(adapter_a, adapter_b):
            diff = (param_a.detach() - param_b.detach()).to(torch.float32)
            if diff.numel() == 0:
                continue
            sum_sq += float(torch.sum(diff * diff).item())
            count += int(diff.numel())
            max_abs = max(max_abs, float(torch.max(torch.abs(diff)).item()))
        rmse = (sum_sq / count) ** 0.5 if count > 0 else 0.0
        return {"rmse": rmse, "max_abs": max_abs, "num_elements": count}

    def _adapter_vs_zero_distance(self, adapter_name: str) -> dict[str, float]:
        sum_sq = 0.0
        count = 0
        max_abs = 0.0
        for _, param in self._iter_named_adapter_params(adapter_name):
            value = param.detach().to(torch.float32)
            if value.numel() == 0:
                continue
            sum_sq += float(torch.sum(value * value).item())
            count += int(value.numel())
            max_abs = max(max_abs, float(torch.max(torch.abs(value)).item()))
        rmse = (sum_sq / count) ** 0.5 if count > 0 else 0.0
        return {"rmse": rmse, "max_abs": max_abs, "num_elements": count}

    def _collect_adapter_state(self, adapter_name: str) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self._iter_named_adapter_params(adapter_name)
        }

    def _load_adapter_state(self, adapter_name: str, state_dict: dict[str, torch.Tensor]) -> None:
        if not state_dict:
            return
        named_parameters = dict(self.current_model.named_parameters())
        loaded = 0
        for key, value in state_dict.items():
            if f".{adapter_name}." not in key:
                continue
            target = named_parameters.get(key)
            if target is None:
                continue
            self._copy_tensor(target, value.to(device=target.device, dtype=target.dtype))
            loaded += 1
        if loaded == 0:
            raise KeyError(f"No parameters were loaded for adapter {adapter_name!r}.")

    def activate_current_adapter(self) -> None:
        self.current_model.transformer.set_adapter(self.current_adapter_name)
        self._set_trainable_adapter(self.current_adapter_name)

    def freeze_auxiliary_models(self) -> None:
        return None

    @torch.no_grad()
    def hard_sync_old(self) -> None:
        copied = self._copy_adapter_state(self.current_adapter_name, self.old_adapter_name)
        if copied <= 0:
            raise RuntimeError("Failed to copy current adapter state into old adapter.")
        self.last_decay = 0.0
        self.activate_current_adapter()

    @torch.no_grad()
    def resolve_decay(self, global_step: int | None = None, decay: float | None = None) -> float:
        if decay is not None:
            resolved_decay = float(decay)
        elif self.config.decay_type is not None:
            if global_step is None:
                raise ValueError("`global_step` is required when using DiffusionNFT-style decay.")
            resolved_decay = diffusion_nft_decay(int(global_step), int(self.config.decay_type))
        elif self.config.ema_decay is not None:
            resolved_decay = float(self.config.ema_decay)
        else:
            raise RuntimeError("No valid old-policy decay configuration found.")

        if not (0.0 <= resolved_decay < 1.0):
            raise ValueError(f"Resolved decay must be in [0, 1), got {resolved_decay}.")
        return resolved_decay

    @torch.no_grad()
    def update_old(self, global_step: int | None = None, decay: float | None = None) -> bool:
        if decay is None and self.config.decay_type is None and global_step is not None:
            if (global_step + 1) % self.config.ema_update_interval != 0:
                return False

        decay = self.resolve_decay(global_step=global_step, decay=decay)
        one_minus_decay = 1.0 - decay
        if one_minus_decay <= 0.0:
            return False

        updated = 0
        for _, current_param, _, old_param in self._iter_adapter_pairs(
            self.current_adapter_name,
            self.old_adapter_name,
        ):
            if not bool(current_param.requires_grad):
                continue
            current_data = current_param.detach()
            if old_param.device == current_data.device:
                old_param.data.mul_(decay).add_(current_data.data, alpha=one_minus_decay)
            else:
                old_copy = current_data.to(device=old_param.device, dtype=old_param.dtype)
                old_param.data.mul_(decay).add_(old_copy.data, alpha=one_minus_decay)
            updated += 1

        if updated == 0:
            return False

        self.num_ema_updates += 1
        self.last_decay = float(decay)
        self.activate_current_adapter()
        return True

    @torch.no_grad()
    def ema_update_old(self, global_step: int | None = None) -> bool:
        return self.update_old(global_step=global_step)

    def distance_report(self) -> dict[str, dict[str, float]]:
        return {
            "current_old": self._adapter_distance(self.current_adapter_name, self.old_adapter_name),
            "old_ref": self._adapter_vs_zero_distance(self.old_adapter_name),
            "current_ref": self._adapter_vs_zero_distance(self.current_adapter_name),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "snapshot_kind": "adapter_lora",
            "decay_type": None if self.config.decay_type is None else int(self.config.decay_type),
            "ema_decay": None if self.config.ema_decay is None else float(self.config.ema_decay),
            "ema_update_interval": int(self.config.ema_update_interval),
            "track_buffers": bool(self.config.track_buffers),
            "num_ema_updates": int(self.num_ema_updates),
            "last_decay": float(self.last_decay),
            "current_adapter_name": self.current_adapter_name,
            "old_adapter_name": self.old_adapter_name,
            "old_adapter_state_dict": self._collect_adapter_state(self.old_adapter_name),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict.get("snapshot_kind") not in {None, "adapter_lora"}:
            raise ValueError(f"Unsupported snapshot kind for adapter manager: {state_dict.get('snapshot_kind')!r}")

        if "decay_type" in state_dict:
            decay_type = state_dict["decay_type"]
            self.config.decay_type = None if decay_type is None else int(decay_type)
        if "ema_decay" in state_dict:
            ema_decay = state_dict["ema_decay"]
            self.config.ema_decay = None if ema_decay is None else float(ema_decay)
        if "ema_update_interval" in state_dict:
            self.config.ema_update_interval = int(state_dict["ema_update_interval"])
        if "track_buffers" in state_dict:
            self.config.track_buffers = bool(state_dict["track_buffers"])
        self.num_ema_updates = int(state_dict.get("num_ema_updates", 0))
        self.last_decay = float(state_dict.get("last_decay", 0.0))

        loaded_old = state_dict.get("old_adapter_state_dict")
        if loaded_old is not None:
            self._load_adapter_state(self.old_adapter_name, loaded_old)
        else:
            self.hard_sync_old()
        self.activate_current_adapter()

    def ref_max_abs_drift(self, reference_state_dict: dict[str, torch.Tensor]) -> float:
        del reference_state_dict
        return 0.0

    def export_adapter_state(self) -> dict[str, Any]:
        return {
            "snapshot_kind": "adapter_lora",
            "current_adapter_name": self.current_adapter_name,
            "old_adapter_name": self.old_adapter_name,
            "current_adapter_state_dict": self._collect_adapter_state(self.current_adapter_name),
            "old_adapter_state_dict": self._collect_adapter_state(self.old_adapter_name),
        }

    def load_adapter_state(self, payload: dict[str, Any]) -> None:
        current_state = payload.get("current_adapter_state_dict", {})
        old_state = payload.get("old_adapter_state_dict", {})
        if current_state:
            self._load_adapter_state(self.current_adapter_name, current_state)
        if old_state:
            self._load_adapter_state(self.old_adapter_name, old_state)
        elif current_state:
            self.hard_sync_old()
        self.activate_current_adapter()


def build_lora_adapter_snapshot_manager(
    base_model: torch.nn.Module,
    config: PolicySnapshotConfig | dict[str, Any] | None = None,
    *,
    current_adapter_name: str = "default",
    old_adapter_name: str = "old",
) -> LoraAdapterPolicySnapshotManager:
    return LoraAdapterPolicySnapshotManager(
        current_model=base_model,
        config=config,
        current_adapter_name=current_adapter_name,
        old_adapter_name=old_adapter_name,
    )
