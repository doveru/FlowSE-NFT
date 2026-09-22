"""Detached device-side statistics for the speech inner training loop."""

from __future__ import annotations

import torch
import torch.distributed as dist


@torch.no_grad()
def accumulate_loss_metrics_(
    accumulator: torch.Tensor,
    terms: dict[str, torch.Tensor],
    micro_steps: int,
    *,
    branch_weight: float = 1.0,
    shared_branch_count: int = 1,
    weighted_loss: torch.Tensor | None = None,
) -> None:
    """Keep the original total/RL/KL/weighted-KL/deviation averaging order."""
    values = torch.stack([
        terms["loss_total"] if weighted_loss is None else weighted_loss,
        terms["loss_rl"],
        terms["loss_kl"],
        terms["kl_coef"] * terms["loss_kl"],
        terms["old_deviate"],
    ]).detach().to(device=accumulator.device, dtype=accumulator.dtype)
    values[1:4].mul_(branch_weight)
    values[4].mul_(shared_branch_count)
    accumulator.add_(values / float(micro_steps))


@torch.no_grad()
def reduce_training_metrics(
    loss_sums: torch.Tensor,
    batch_count: float,
    grad_norm_sum: torch.Tensor,
    grad_norm_count: float,
    grad_norm_max: torch.Tensor,
) -> list[float]:
    """Reduce all ranks, then transfer the seven reported scalars together."""
    packed = torch.cat([
        loss_sums.detach(),
        grad_norm_sum.detach().reshape(1),
        loss_sums.new_tensor([batch_count, grad_norm_count]),
    ])
    maximum = grad_norm_max.detach().clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    loss_means = packed[:5] / packed[6].clamp_min(1.0)
    grad_mean = packed[5] / packed[7].clamp_min(1.0)
    return torch.cat([loss_means, grad_mean.reshape(1), maximum.reshape(1)]).cpu().tolist()
