from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_FAKE_BACKEND_MODULE = ModuleType("flow_grpo.speech_backend_adapter")
_FAKE_BACKEND_MODULE.SpeechBackendAdapter = object
_FAKE_BACKEND_MODULE.set_seed = lambda *args, **kwargs: None
sys.modules[_FAKE_BACKEND_MODULE.__name__] = _FAKE_BACKEND_MODULE

_FAKE_ORCHESTRATOR_MODULE = ModuleType("flow_grpo.speech_orchestrator")
_FAKE_ORCHESTRATOR_MODULE.cleanup_distributed = lambda: None
_FAKE_ORCHESTRATOR_MODULE.is_main_process = lambda rank: True
_FAKE_ORCHESTRATOR_MODULE.setup_distributed = lambda: (False, 0, 1, 0)
sys.modules[_FAKE_ORCHESTRATOR_MODULE.__name__] = _FAKE_ORCHESTRATOR_MODULE

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluation_speech.py"
_SPEC = spec_from_file_location("evaluation_speech_module", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Unable to load evaluation_speech module for tests.")
_MODULE = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

inspect_eval_checkpoint = _MODULE.inspect_eval_checkpoint
load_eval_checkpoint = _MODULE.load_eval_checkpoint
prepare_eval_config = _MODULE.prepare_eval_config


class _FakeCurrentModel:
    def __init__(self):
        self.loaded = None

    def load_state_dict(self, state_dict, strict=True):
        self.loaded = {"state_dict": state_dict, "strict": bool(strict)}


class _FakeSnapshotManager:
    def __init__(self):
        self.loaded = None

    def load_state_dict(self, state_dict):
        self.loaded = state_dict


class _FakeBackend:
    def __init__(self):
        self.snapshot_manager = _FakeSnapshotManager()
        self.resume_calls = []

    def load_lora_resume(self, resume_dir, model_bundle):
        self.resume_calls.append((Path(resume_dir), bool(model_bundle.lora_enabled)))
        return 7, 123, 0.88


class _FakeModelBundle:
    def __init__(self, *, lora_enabled: bool):
        self.lora_enabled = bool(lora_enabled)
        self.current_model = _FakeCurrentModel()


def test_load_eval_checkpoint_supports_adapter_directory(tmp_path):
    adapter_dir = tmp_path / "best-epoch-0006.lora"
    adapter_dir.mkdir()
    backend = _FakeBackend()
    model_bundle = _FakeModelBundle(lora_enabled=True)
    checkpoint_info, checkpoint_payload = inspect_eval_checkpoint(adapter_dir, device=torch.device("cpu"))

    info = load_eval_checkpoint(
        checkpoint_info,
        checkpoint_payload,
        backend=backend,
        model_bundle=model_bundle,
    )

    assert info["kind"] == "adapter_dir"
    assert info["resume_epoch"] == 7
    assert info["global_step"] == 123
    assert backend.resume_calls == [(adapter_dir.resolve(), True)]


def test_load_eval_checkpoint_loads_training_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "checkpoint-0006.pt"
    payload = {
        "outer_epoch": 5,
        "global_step": 123,
        "best_eval_reward": 0.88,
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "optim_state_dict": {"state": {}, "param_groups": []},
        "scheduler_state_dict": {"base_lrs": [1e-4]},
        "snapshot_state_dict": {"snapshot_kind": "adapter_lora"},
    }
    torch.save(payload, checkpoint_path)
    backend = _FakeBackend()
    model_bundle = _FakeModelBundle(lora_enabled=True)
    checkpoint_info, checkpoint_payload = inspect_eval_checkpoint(checkpoint_path, device=torch.device("cpu"))

    info = load_eval_checkpoint(
        checkpoint_info,
        checkpoint_payload,
        backend=backend,
        model_bundle=model_bundle,
    )

    assert info["kind"] == "train_checkpoint"
    assert model_bundle.current_model.loaded == {
        "state_dict": payload["model_state_dict"],
        "strict": True,
    }
    assert backend.snapshot_manager.loaded == payload["snapshot_state_dict"]


def test_prepare_eval_config_supports_base_model_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "best.pt.tar"
    torch.save({"model_state_dict": {"weight": torch.tensor([1.0])}}, checkpoint_path)
    checkpoint_info, checkpoint_payload = inspect_eval_checkpoint(checkpoint_path, device=torch.device("cpu"))

    assert checkpoint_info == {
        "path": str(checkpoint_path.resolve()),
        "kind": "base_model_checkpoint",
    }
    assert checkpoint_payload is not None

    config = SimpleNamespace(
        speech=SimpleNamespace(
            model=SimpleNamespace(
                init_checkpoint="old-init.pt.tar",
                lora=SimpleNamespace(enabled=True),
            )
        )
    )
    updated_info = prepare_eval_config(config, checkpoint_info)

    assert config.speech.model.init_checkpoint == str(checkpoint_path.resolve())
    assert config.speech.model.lora.enabled is False
    assert updated_info["lora_disabled_for_eval"] is True


def test_load_eval_checkpoint_base_model_is_noop_after_prepare(tmp_path):
    checkpoint_path = tmp_path / "best.pt.tar"
    torch.save({"model_state_dict": {"weight": torch.tensor([1.0])}}, checkpoint_path)
    backend = _FakeBackend()
    model_bundle = _FakeModelBundle(lora_enabled=False)
    checkpoint_info, checkpoint_payload = inspect_eval_checkpoint(checkpoint_path, device=torch.device("cpu"))

    info = load_eval_checkpoint(
        checkpoint_info,
        checkpoint_payload,
        backend=backend,
        model_bundle=model_bundle,
    )

    assert info["kind"] == "base_model_checkpoint"
    assert model_bundle.current_model.loaded is None
    assert backend.snapshot_manager.loaded is None
