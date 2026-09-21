from __future__ import annotations

import pytest
import torch

from flow_grpo.diffusers_patch.solver import run_sampling


def _sigma_schedule() -> torch.Tensor:
    return torch.tensor([0.9, 0.6, 0.3], dtype=torch.float32)


def _constant_pred(z: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    del sigma
    return torch.ones_like(z)


def test_flow_all_ode_mask_matches_deterministic_update():
    z = torch.zeros(2, 3, dtype=torch.float32)

    masked, _, _ = run_sampling(
        _constant_pred,
        z.clone(),
        _sigma_schedule(),
        solver="flow",
        determistic=False,
        eta=0.5,
        deterministic_mask=[True, True],
    )
    deterministic, _, _ = run_sampling(
        _constant_pred,
        z.clone(),
        _sigma_schedule(),
        solver="flow",
        determistic=True,
        eta=0.5,
    )

    expected = torch.full_like(z, -0.6)
    assert torch.allclose(masked, expected)
    assert torch.allclose(masked, deterministic)


def test_flow_all_sde_mask_matches_existing_stochastic_path():
    z = torch.zeros(2, 3, dtype=torch.float32)

    torch.manual_seed(123)
    unmasked, _, _ = run_sampling(
        _constant_pred,
        z.clone(),
        _sigma_schedule(),
        solver="flow",
        determistic=False,
        eta=0.5,
    )
    torch.manual_seed(123)
    masked, _, _ = run_sampling(
        _constant_pred,
        z.clone(),
        _sigma_schedule(),
        solver="flow",
        determistic=True,
        eta=0.5,
        deterministic_mask=[False, False],
    )

    assert torch.allclose(masked, unmasked)


def test_flow_mixed_mask_returns_finite_outputs_and_log_probs():
    z = torch.zeros(2, 3, dtype=torch.float32)
    sigma_schedule = torch.tensor([0.9, 0.7, 0.5, 0.3], dtype=torch.float32)

    torch.manual_seed(7)
    latents, _, log_probs = run_sampling(
        _constant_pred,
        z,
        sigma_schedule,
        solver="flow",
        determistic=False,
        eta=0.5,
        deterministic_mask=[False, True, False],
    )

    assert torch.isfinite(latents).all()
    assert torch.isfinite(torch.stack(log_probs)).all()


def test_flow_deterministic_mask_length_is_validated():
    z = torch.zeros(2, 3, dtype=torch.float32)

    with pytest.raises(ValueError, match="deterministic_mask"):
        run_sampling(
            _constant_pred,
            z,
            _sigma_schedule(),
            solver="flow",
            deterministic_mask=[True],
        )


def test_mixed_mask_is_rejected_for_non_flow_solvers():
    z = torch.zeros(2, 3, dtype=torch.float32)

    with pytest.raises(ValueError, match="only supported for solver='flow'"):
        run_sampling(
            _constant_pred,
            z,
            _sigma_schedule(),
            solver="dance",
            deterministic_mask=[True, False],
        )


def test_flow_accepts_early16_then_deterministic_jump_schedule():
    z = torch.zeros(2, 3, dtype=torch.float32)
    time_grid = torch.tensor([index / 32.0 for index in range(17)] + [1.0])
    sigma_schedule = 1.0 - time_grid

    sampled, trajectory, log_probs = run_sampling(
        _constant_pred,
        z,
        sigma_schedule,
        solver="flow",
        determistic=False,
        eta=0.0,
        deterministic_mask=[False] * 16 + [True],
    )

    assert len(trajectory) == 18
    assert len(log_probs) == 17
    assert torch.allclose(sampled, torch.full_like(sampled, -1.0))
