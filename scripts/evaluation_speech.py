from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch

from flow_grpo.speech_backend_adapter import SpeechBackendAdapter, set_seed
from flow_grpo.speech_orchestrator import cleanup_distributed, is_main_process, setup_distributed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Speech NFT checkpoint.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/nft.py:speech_wotext_multi_reward_rms_scale_tau03_weight211",
        help="Config in format path/to/config.py:config_name",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path from train_nft_speech.py")
    parser.add_argument("--output_json", type=str, default=None, help="Optional output json path")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_config(config_ref: str):
    if ":" not in config_ref:
        raise ValueError("`--config` must be in format path/to/config.py:config_name")
    config_path_str, config_name = config_ref.split(":", 1)
    config_path = Path(config_path_str).expanduser().resolve()
    spec = importlib.util.spec_from_file_location("nft_config_module", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load config module from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config(config_name)


def inspect_eval_checkpoint(
    checkpoint_ref: str | Path,
    *,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    checkpoint_path = Path(checkpoint_ref).expanduser().resolve()
    if checkpoint_path.is_dir():
        return {
            "path": str(checkpoint_path),
            "kind": "adapter_dir",
        }, None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(checkpoint)!r}")
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint is missing `model_state_dict`.")

    is_train_checkpoint = "optim_state_dict" in checkpoint and "scheduler_state_dict" in checkpoint
    if not is_train_checkpoint:
        return {
            "path": str(checkpoint_path),
            "kind": "base_model_checkpoint",
        }, checkpoint

    return {
        "path": str(checkpoint_path),
        "kind": "train_checkpoint",
        "outer_epoch": int(checkpoint.get("outer_epoch", -1)),
        "global_step": int(checkpoint.get("global_step", 0)),
        "best_eval_reward": float(checkpoint.get("best_eval_reward", float("-inf"))),
    }, checkpoint


def prepare_eval_config(config, checkpoint_info: dict[str, Any]) -> dict[str, Any]:
    if checkpoint_info["kind"] != "base_model_checkpoint":
        return checkpoint_info
    config.speech.model.init_checkpoint = checkpoint_info["path"]
    lora_conf = getattr(config.speech.model, "lora", None)
    lora_was_enabled = bool(getattr(lora_conf, "enabled", False)) if lora_conf is not None else False
    if lora_was_enabled:
        lora_conf.enabled = False
    result = dict(checkpoint_info)
    result["lora_disabled_for_eval"] = bool(lora_was_enabled)
    return result


def load_eval_checkpoint(
    checkpoint_info: dict[str, Any],
    checkpoint_payload: dict[str, Any] | None,
    *,
    backend: SpeechBackendAdapter,
    model_bundle,
) -> dict[str, Any]:
    if checkpoint_info["kind"] == "base_model_checkpoint":
        return checkpoint_info
    if checkpoint_info["kind"] == "adapter_dir":
        if not bool(model_bundle.lora_enabled):
            raise ValueError(
                "Adapter-directory evaluation requires LoRA enabled in config, "
                f"but lora_enabled={model_bundle.lora_enabled}."
            )
        start_epoch, global_step, best_eval_reward = backend.load_lora_resume(
            Path(checkpoint_info["path"]),
            model_bundle,
        )
        result = dict(checkpoint_info)
        result.update(
            {
                "resume_epoch": int(start_epoch),
                "global_step": int(global_step),
                "best_eval_reward": float(best_eval_reward),
            }
        )
        return result
    if checkpoint_payload is None:
        raise ValueError("Training checkpoint payload is required for evaluation load.")

    model_bundle.current_model.load_state_dict(checkpoint_payload["model_state_dict"], strict=True)
    snapshot_state = checkpoint_payload.get("snapshot_state_dict")
    if snapshot_state is not None:
        backend.snapshot_manager.load_state_dict(snapshot_state)
    return checkpoint_info


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    distributed, rank, world_size, local_rank = setup_distributed()
    try:
        if distributed and torch.cuda.is_available():
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        set_seed(int(args.seed), rank=rank)
        checkpoint_info, checkpoint_payload = inspect_eval_checkpoint(args.checkpoint, device=device)
        checkpoint_info = prepare_eval_config(config, checkpoint_info)

        backend = SpeechBackendAdapter(config, device=device, rank=rank, world_size=world_size)
        model_bundle = backend.build_models_and_policies(config, device)
        data_bundle = backend.build_data(config)
        val_loader = data_bundle["val_loader"]

        checkpoint_info = load_eval_checkpoint(
            checkpoint_info,
            checkpoint_payload,
            backend=backend,
            model_bundle=model_bundle,
        )

        metrics = backend.evaluate(model_bundle.current_model, model_bundle.ref_model, val_loader, config)
        eval_deterministic_mask = backend._eval_deterministic_mask()
        eval_time_grid = backend.eval_rollout_engine.time_grid

        report = {
            "config": args.config,
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "checkpoint_info": checkpoint_info,
            "device": str(device),
            "rank": int(rank),
            "world_size": int(world_size),
            "evaluation_sampling": {
                "steps": int(backend.eval_rollout_engine.steps),
                "time_grid": None if eval_time_grid is None else list(eval_time_grid),
                "deterministic_mask": list(eval_deterministic_mask),
                "num_sde_steps": int(sum(not value for value in eval_deterministic_mask)),
                "num_ode_steps": int(sum(eval_deterministic_mask)),
            },
            "metrics": metrics,
        }

        if is_main_process(rank):
            if args.output_json is not None:
                output_path = Path(args.output_json).expanduser().resolve()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with output_path.open("w", encoding="utf-8") as file:
                    json.dump(report, file, indent=2, ensure_ascii=False)
            print(json.dumps(report, indent=2, ensure_ascii=False))
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
