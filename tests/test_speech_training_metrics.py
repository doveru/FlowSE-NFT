from __future__ import annotations

import pytest
import torch

from flow_grpo import speech_training_metrics as metrics


@pytest.mark.parametrize("weights,shared", [([1.0], False), ([2.0, 0.5, 1.0], False), ([1.0], True)])
def test_device_accumulation_matches_previous_logging_and_keeps_gradients(weights, shared, monkeypatch):
    sums = torch.zeros(5, dtype=torch.float64)
    expected = [0.0] * 5
    parameter = torch.tensor(0.37, requires_grad=True)
    expected_gradient = 0.0
    branch_count = 3 if shared else len(weights)
    for branch_index, weight in enumerate(weights):
        steps = branch_index + 2
        for step in range(steps):
            terms = {
                "loss_total": parameter * (step + 1),
                "loss_rl": parameter * 0.7,
                "loss_kl": parameter * 0.13,
                "kl_coef": torch.tensor(0.001),
                "old_deviate": parameter * 0.23,
            }
            weighted = terms["loss_total"] * weight
            old_values = [
                float(weighted.detach()),
                float(terms["loss_rl"].detach()) * weight,
                float(terms["loss_kl"].detach()) * weight,
                float((terms["kl_coef"] * terms["loss_kl"]).detach()) * weight,
                float(terms["old_deviate"].detach()) * (branch_count if shared else 1),
            ]
            expected = [a + b / steps for a, b in zip(expected, old_values)]

            def forbidden_transfer(*args, **kwargs):
                raise AssertionError("Per-timestep logging must not read a tensor on the host")

            with monkeypatch.context() as patch:
                for method in ("cpu", "item", "tolist", "__float__"):
                    patch.setattr(torch.Tensor, method, forbidden_transfer)
                metrics.accumulate_loss_metrics_(
                    sums, terms, steps, branch_weight=weight,
                    shared_branch_count=branch_count if shared else 1,
                    weighted_loss=weighted,
                )
            weighted.backward()
            expected_gradient += (step + 1) * weight
    sums[4] /= branch_count
    expected[4] /= branch_count
    assert not sums.requires_grad and sums.grad_fn is None
    assert sums.tolist() == pytest.approx(expected, abs=1e-12)
    assert float(parameter.grad) == pytest.approx(expected_gradient)
    actual = metrics.reduce_training_metrics(sums, 1, torch.tensor(6.0), 2, torch.tensor(4.0))
    assert actual == pytest.approx(expected + [3.0, 4.0], abs=1e-12)


def test_distributed_metrics_use_global_counts_and_maximum(monkeypatch):
    monkeypatch.setattr(metrics.dist, "is_available", lambda: True)
    monkeypatch.setattr(metrics.dist, "is_initialized", lambda: True)
    calls = []

    def all_reduce(tensor, op):
        calls.append(op)
        if op == metrics.dist.ReduceOp.SUM:
            # Other rank has a different batch count and optimizer-step count.
            tensor.add_(tensor.new_tensor([9, 6, 3, 1.5, 12, 15, 3, 3]))
        else:
            tensor.copy_(torch.maximum(tensor, tensor.new_tensor(7.0)))

    monkeypatch.setattr(metrics.dist, "all_reduce", all_reduce)
    sums = torch.tensor([3, 2, 1, 0.5, 4], dtype=torch.float64)
    result = metrics.reduce_training_metrics(sums, 1, sums.new_tensor(5), 1, sums.new_tensor(5))
    assert result == pytest.approx([3, 2, 1, 0.5, 4, 5, 7])
    assert calls == [metrics.dist.ReduceOp.SUM, metrics.dist.ReduceOp.MAX]
    assert sums.tolist() == [3, 2, 1, 0.5, 4]


def test_empty_statistics_are_zero():
    sums = torch.zeros(5, dtype=torch.float64)
    assert metrics.reduce_training_metrics(sums, 0, sums.new_zeros(()), 0, sums.new_zeros(())) == [0.0] * 7
