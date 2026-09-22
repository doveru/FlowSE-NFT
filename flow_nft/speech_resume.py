"""Per-rank state needed to resume at an outer-epoch boundary."""

from __future__ import annotations

import random
import warnings

import numpy as np
import torch
import torch.distributed as dist


def capture_training_state(device, scaler=None, mixed_sampling_state=None):
    numpy_state = np.random.get_state()
    return {
        "python_rng": random.getstate(),
        # Primitive types keep the checkpoint compatible with weights-only loading.
        "numpy_rng": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None,
        "scaler": None if scaler is None else scaler.state_dict(),
        "mixed_sampling": None if mixed_sampling_state is None else mixed_sampling_state.state_dict(),
    }


def collect_training_states(device, scaler=None, mixed_sampling_state=None):
    """All ranks must call this; only the writer needs to persist the result."""
    local_state = capture_training_state(device, scaler, mixed_sampling_state)
    if dist.is_available() and dist.is_initialized():
        states = [None] * dist.get_world_size()
        dist.all_gather_object(states, local_state)
        return states
    return [local_state]


def select_training_state(checkpoint, rank=0, world_size=1):
    states = checkpoint.get("training_states_by_rank")
    if states is None:
        warnings.warn(
            "Legacy checkpoint has no RNG/scaler/mixed-sampling state; "
            "training resumes with freshly initialized runtime state.",
            RuntimeWarning,
        )
        return None
    if len(states) != world_size or not 0 <= rank < len(states):
        raise ValueError(
            f"Resume world size must match the checkpoint: saved={len(states)}, current={world_size}, rank={rank}."
        )
    return states[rank]


def restore_training_state(state, device, scaler=None, mixed_sampling_state=None):
    if state is None:
        return
    saved_mixed = state["mixed_sampling"]
    if (saved_mixed is None) != (mixed_sampling_state is None):
        raise ValueError("Mixed sampling enabled/disabled setting differs from the checkpoint.")
    if mixed_sampling_state is not None:
        mixed_sampling_state.load_state_dict(saved_mixed)
    saved_scaler = state["scaler"]
    if scaler is not None and saved_scaler:
        if not scaler.is_enabled():
            raise ValueError("Checkpoint uses AMP scaling, but the resumed scaler is disabled.")
        scaler.load_state_dict(saved_scaler)
    elif scaler is not None and scaler.is_enabled():
        raise ValueError("Checkpoint has no enabled AMP scaler state, but AMP scaling is now enabled.")
    if (state["cuda_rng"] is not None) != (device.type == "cuda"):
        raise ValueError("Resume device type must match the checkpoint's CPU/CUDA random state.")
    random.setstate(state["python_rng"])
    name, keys, position, has_gauss, cached_gaussian = state["numpy_rng"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached_gaussian))
    torch.set_rng_state(state["torch_rng"].cpu())
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device)
