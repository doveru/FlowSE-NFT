from __future__ import annotations

import copy
import importlib.util
import random
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from flow_grpo.speech_nft_core import SpeechMixedSamplingState
from flow_grpo import speech_resume as resume


class FakeScaler:
    def __init__(self, scale=65536.0):
        self.state = {"scale": scale, "_growth_tracker": 9}

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, state):
        self.state = dict(state)

    def is_enabled(self):
        return True


def draw():
    return random.random(), float(np.random.normal()), torch.randn(5)


def test_rng_scaler_and_mixed_progress_round_trip(tmp_path):
    state = SpeechMixedSamplingState(8, 3, update_interval=3)
    for _ in range(7):
        state.update_iteration()
    np.random.normal()  # Include the cached Gaussian component of NumPy state.
    saved = resume.capture_training_state(torch.device("cpu"), FakeScaler(), state)
    path = tmp_path / "runtime.pt"
    torch.save(saved, path)
    saved = torch.load(path, weights_only=True)
    expected = [draw() for _ in range(3)]
    scaler = FakeScaler(1.0)
    restored = SpeechMixedSamplingState(8, 3, update_interval=3)
    resume.restore_training_state(saved, torch.device("cpu"), scaler, restored)
    assert scaler.state == FakeScaler().state
    for expected_draw in expected:
        actual = draw()
        assert actual[:2] == expected_draw[:2]
        assert torch.equal(actual[2], expected_draw[2])
    for _ in range(25):
        assert restored.get_current_deterministic_mask() == state.get_current_deterministic_mask()
        restored.update_iteration()
        state.update_iteration()


def test_per_rank_selection_and_legacy_compatibility(monkeypatch):
    first = resume.capture_training_state(torch.device("cpu"))
    draw()
    second = resume.capture_training_state(torch.device("cpu"))
    monkeypatch.setattr(resume.dist, "is_available", lambda: True)
    monkeypatch.setattr(resume.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(resume.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(resume.dist, "all_gather_object", lambda output, local: output.__setitem__(slice(None), [first, second]))
    payload = {"training_states_by_rank": resume.collect_training_states(torch.device("cpu"))}
    assert resume.select_training_state(payload, 1, 2) is second
    assert resume.select_training_state(payload, 0, 2) is first
    with pytest.raises(ValueError, match="world size"):
        resume.select_training_state(payload, 0, 1)
    with pytest.warns(RuntimeWarning, match="Legacy checkpoint"):
        assert resume.select_training_state({}) is None
    resume.restore_training_state(None, torch.device("cpu"))


def test_mixed_sampling_configuration_mismatch_is_rejected():
    source = SpeechMixedSamplingState(8, 3, update_interval=2)
    target = SpeechMixedSamplingState(8, 4, update_interval=2)
    with pytest.raises(ValueError, match="configuration mismatch"):
        target.load_state_dict(source.state_dict())
    invalid = source.state_dict()
    invalid["cur_iter_in_interval"] = 2
    with pytest.raises(ValueError, match="Invalid mixed sampling progress"):
        source.load_state_dict(invalid)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_rng_and_real_amp_scaler_round_trip():
    device = torch.device("cuda", torch.cuda.current_device())
    scaler = torch.cuda.amp.GradScaler()
    parameter = torch.nn.Parameter(torch.ones(1, device=device))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    scaler.scale(parameter.square().sum()).backward()
    scaler.step(optimizer)
    scaler.update()
    state = resume.capture_training_state(device, scaler)
    expected = torch.randn(16, device=device)
    restored_scaler = torch.cuda.amp.GradScaler(init_scale=2.0)
    resume.restore_training_state(state, device, restored_scaler)
    assert torch.equal(torch.randn(16, device=device), expected)
    assert restored_scaler.state_dict() == scaler.state_dict()


def load_checkpoint_module():
    # Restore the import table immediately; do not contaminate other test modules.
    backend = ModuleType("flow_grpo.speech_backend_adapter")
    backend.SpeechBackendAdapter = object
    backend.set_seed = lambda *args, **kwargs: None
    ema = ModuleType("flow_grpo.ema")
    ema.EMAModuleWrapper = object
    path = Path(__file__).resolve().parents[1] / "flow_grpo/speech_orchestrator.py"
    spec = importlib.util.spec_from_file_location("_resume_test_orchestrator", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {backend.__name__: backend, ema.__name__: ema}):
        spec.loader.exec_module(module)
    return module


class Snapshot:
    def __init__(self, model):
        self.ref_model = copy.deepcopy(model)

    def state_dict(self):
        return self.ref_model.state_dict()

    def load_state_dict(self, state):
        self.ref_model.load_state_dict(state)


def test_full_checkpoint_resume_matches_uninterrupted_updates(tmp_path):
    module = load_checkpoint_module()
    device = torch.device("cpu")

    def create():
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
        return model, optimizer, scheduler, Snapshot(model), SpeechMixedSamplingState(8, 3, update_interval=2)

    def step(bundle):
        model, optimizer, scheduler, _, mixed = bundle
        optimizer.zero_grad()
        x = torch.randn(4, 3)
        target = random.random() + float(np.random.normal())
        loss = (model(x) - target).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        mask = mixed.get_current_deterministic_mask()
        mixed.update_iteration()
        return float(loss), mask

    bundle = create()
    for _ in range(3):
        step(bundle)
    model, optimizer, scheduler, snapshot, mixed = bundle
    path = tmp_path / "checkpoint.pt"
    module.save_checkpoint(
        path, current_model=model, optimizer=optimizer, scheduler=scheduler,
        snapshot_manager=snapshot, outer_epoch=2, global_step=3, best_eval_reward=0.7,
        ema_state_dict=None, config=SimpleNamespace(),
        training_states_by_rank=resume.collect_training_states(device, mixed_sampling_state=mixed),
    )
    expected = [step(bundle) for _ in range(4)]
    restored = create()
    new_model, new_optimizer, new_scheduler, new_snapshot, new_mixed = restored
    pending = {}
    assert module.load_checkpoint(
        path, current_model=new_model, optimizer=new_optimizer, scheduler=new_scheduler,
        snapshot_manager=new_snapshot, device=device, resume_state_out=pending,
    ) == (3, 3, 0.7)
    # A startup evaluation may consume randomness before the delayed restoration.
    for _ in range(5):
        draw()
    resume.restore_training_state(pending["training_state"], device, mixed_sampling_state=new_mixed)
    assert [step(restored) for _ in range(4)] == expected
    for a, b in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(a, b)
    assert new_scheduler.state_dict() == scheduler.state_dict()


def test_checkpoint_restores_training_parameters_instead_of_ema_weights(tmp_path):
    module = load_checkpoint_module()
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    training = {name: value.detach().clone() for name, value in model.named_parameters()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10)
    path = tmp_path / "ema.pt"
    module.save_checkpoint(
        path, current_model=model, optimizer=optimizer, scheduler=scheduler,
        snapshot_manager=Snapshot(model), outer_epoch=0, global_step=1, best_eval_reward=1,
        ema_state_dict=None, config=SimpleNamespace(), training_parameters=training,
    )
    module.load_checkpoint(
        path, current_model=model, optimizer=optimizer, scheduler=scheduler,
        snapshot_manager=Snapshot(model), device=torch.device("cpu"),
    )
    for name, value in model.named_parameters():
        assert torch.equal(value, training[name])


def test_checkpoint_without_runtime_states_can_resume(tmp_path):
    import ast
    module = load_checkpoint_module()
    # Guard against reintroducing the blocking collection in the training path.
    source = Path(module.__file__).read_text()
    assert not any(
        isinstance(node, ast.Name) and node.id == "collect_training_states"
        for node in ast.walk(ast.parse(source))
    )
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    path = tmp_path / "checkpoint.pt"
    module.save_checkpoint(
        path, current_model=model, optimizer=optimizer, scheduler=scheduler,
        snapshot_manager=Snapshot(model), outer_epoch=4, global_step=5,
        best_eval_reward=3.9, ema_state_dict=None, config=SimpleNamespace(),
    )
    payload = torch.load(path, weights_only=True)
    assert "training_states_by_rank" not in payload
    assert {"model_state_dict", "optim_state_dict", "scheduler_state_dict",
            "snapshot_state_dict", "ema_state_dict"} <= payload.keys()
    pending = {}
    with pytest.warns(RuntimeWarning, match="no RNG/scaler"):
        result = module.load_checkpoint(
            path, current_model=model, optimizer=optimizer, scheduler=scheduler,
            snapshot_manager=Snapshot(model), device=torch.device("cpu"),
            resume_state_out=pending,
        )
    assert result == (5, 5, 3.9)
    assert pending["training_state"] is None
