import os
from pathlib import Path
import subprocess

import torch

from flow_grpo.speech_nft_core import build_training_timestep_grid, select_timestep_indices


def test_default_launcher_uses_paper_config():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, DRY_RUN="1")
    env.pop("CONFIG", None)
    result = subprocess.run(["bash", str(root / "train_flowse_nft.sh")], env=env,
                            capture_output=True, text=True, check=True)
    assert "config/nft.py:speech_wotext_multi_reward_rms_scale_tau03_weight211" in result.stdout


def test_quarter_window_selects_32_uniform_training_times():
    grid = build_training_timestep_grid([0.0, 1.0], timestep_grid_steps=128)
    indices = select_timestep_indices(grid, timestep_window=[0.0, 0.25],
                                     timestep_fraction=1.0, timesteps_per_batch=32)
    times = grid[sorted(indices)]
    assert len(times) == 32
    assert float(times[0]) == 0.0
    assert bool(torch.all(times < 0.25))
    assert torch.allclose(times.diff(), torch.full((31,), 1.0 / 127))
