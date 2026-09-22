"""Small, device-explicit collectives for speech training control flow."""

import torch
import torch.distributed as dist


def broadcast_save_decision(should_save, best_eval_reward, device):
    """All ranks participate; rank 0 supplies both values, including +/-inf.

    Use a fixed-size tensor rather than pickle/object-size broadcasts. Reading
    it back on each rank completes the device work before subsequent decisions.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return bool(should_save), float(best_eval_reward)
    device = torch.device(device)
    if dist.get_backend() != "nccl":
        device = torch.device("cpu")
    values = [float(bool(should_save)), float(best_eval_reward)] if dist.get_rank() == 0 else [0.0, 0.0]
    decision = torch.tensor(values, dtype=torch.float64, device=device)
    dist.broadcast(decision, src=0)
    save_value, best_value = decision.cpu().tolist()
    return bool(save_value), float(best_value)


def epoch_barrier(device):
    """Use the local CUDA device explicitly for NCCL; retain CPU/Gloo support."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    device = torch.device(device)
    if dist.get_backend() == "nccl":
        if device.type != "cuda" or device.index is None:
            raise ValueError("NCCL epoch barrier requires an explicit local CUDA device index")
        dist.barrier(device_ids=[device.index])
    else:
        dist.barrier()
