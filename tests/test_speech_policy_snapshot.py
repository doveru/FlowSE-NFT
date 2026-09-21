from __future__ import annotations

import copy
from contextlib import contextmanager
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import torch
from torch import nn

_POLICY_SNAPSHOT_PATH = Path(__file__).resolve().parents[1] / "flow_grpo" / "speech_flowse" / "policy_snapshot.py"
_SPEC = spec_from_file_location("speech_policy_snapshot", _POLICY_SNAPSHOT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Unable to load speech policy snapshot module for tests.")
_MODULE = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

PolicySnapshotConfig = _MODULE.PolicySnapshotConfig
PolicySnapshotManager = _MODULE.PolicySnapshotManager
build_lora_adapter_snapshot_manager = _MODULE.build_lora_adapter_snapshot_manager


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.trainable = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.frozen = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))


class _FakePeftTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_weight = nn.Parameter(torch.ones(2, 2, dtype=torch.float32))
        self.lora_A = nn.ModuleDict(
            {
                "default": nn.Linear(2, 2, bias=False),
                "old": nn.Linear(2, 2, bias=False),
            }
        )
        self.active_adapter = "default"
        self._adapter_disabled = False
        self.peft_config = {"default": {"kind": "fake"}, "old": {"kind": "fake"}}

    def set_adapter(self, adapter_name):
        self.active_adapter = adapter_name

    def add_adapter(self, adapter_name, adapter_config):
        self.lora_A[adapter_name] = nn.Linear(2, 2, bias=False)
        self.peft_config[adapter_name] = adapter_config

    @contextmanager
    def disable_adapter(self):
        prev = self._adapter_disabled
        self._adapter_disabled = True
        try:
            yield
        finally:
            self._adapter_disabled = prev


class _FakeCFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = _FakePeftTransformer()
        self.mel_spec = nn.Identity()
        self.num_channels = 2

    def prepare_condition(self, x):
        return x

    def prepare_text(self, text, batch: int, device=None):
        del batch, device
        return text

    def predict_flow(self, *, x, cond, text, time, drop_audio_cond=False, drop_text=False, mask=None):
        del cond, text, time, drop_audio_cond, drop_text, mask
        if self.transformer._adapter_disabled:
            return torch.zeros_like(x)
        adapter = self.transformer.active_adapter
        return self.transformer.lora_A[adapter](x)

    def sample(self, cond, text, **kwargs):
        del text, kwargs
        if self.transformer._adapter_disabled:
            return torch.zeros_like(cond), None, torch.tensor([0.0], dtype=cond.dtype)
        adapter = self.transformer.active_adapter
        return self.transformer.lora_A[adapter](cond), None, torch.tensor([0.0], dtype=cond.dtype)


def test_update_old_updates_only_trainable_parameters():
    current = _TinyModel()
    old = copy.deepcopy(current)
    ref = copy.deepcopy(current)

    current.trainable.data.fill_(4.0)
    current.frozen.data.fill_(9.0)
    current.frozen.requires_grad_(False)
    current.trainable.requires_grad_(True)

    old.trainable.data.fill_(2.0)
    old.frozen.data.fill_(3.0)

    manager = PolicySnapshotManager(
        current_model=current,
        old_model=old,
        ref_model=ref,
        config=PolicySnapshotConfig(decay_type=1),
    )
    manager.old_model.trainable.data.fill_(2.0)
    manager.old_model.frozen.data.fill_(3.0)

    updated = manager.update_old(decay=0.5)
    assert updated is True

    # trainable: old <- 0.5 * old + 0.5 * current
    assert torch.allclose(manager.old_model.trainable.detach(), torch.tensor([3.0]))
    # frozen stays unchanged (trainable-only update)
    assert torch.allclose(manager.old_model.frozen.detach(), torch.tensor([3.0]))


def test_lora_adapter_snapshot_manager_updates_old_adapter():
    model = _FakeCFM()
    with torch.no_grad():
        model.transformer.lora_A["default"].weight.fill_(2.0)
        model.transformer.lora_A["old"].weight.fill_(0.0)

    manager = build_lora_adapter_snapshot_manager(
        model,
        PolicySnapshotConfig(decay_type=1),
        current_adapter_name="default",
        old_adapter_name="old",
    )
    old_after_sync = model.transformer.lora_A["old"].weight.detach().clone()
    assert torch.allclose(old_after_sync, model.transformer.lora_A["default"].weight.detach())

    with torch.no_grad():
        model.transformer.lora_A["default"].weight.fill_(4.0)

    updated = manager.update_old(decay=0.5)
    assert updated is True
    expected = torch.full_like(model.transformer.lora_A["old"].weight.detach(), 3.0)
    assert torch.allclose(model.transformer.lora_A["old"].weight.detach(), expected)
    assert model.transformer.active_adapter == "default"


def test_lora_adapter_snapshot_manager_keeps_only_current_adapter_trainable():
    model = _FakeCFM()
    manager = build_lora_adapter_snapshot_manager(
        model,
        PolicySnapshotConfig(decay_type=1),
        current_adapter_name="default",
        old_adapter_name="old",
    )
    named_parameters = dict(model.named_parameters())

    assert named_parameters["transformer.base_weight"].requires_grad is False
    assert named_parameters["transformer.lora_A.default.weight"].requires_grad is True
    assert named_parameters["transformer.lora_A.old.weight"].requires_grad is False

    named_parameters["transformer.lora_A.old.weight"].requires_grad_(True)
    manager.activate_current_adapter()

    assert named_parameters["transformer.lora_A.default.weight"].requires_grad is True
    assert named_parameters["transformer.lora_A.old.weight"].requires_grad is False


def test_lora_adapter_route_disable_behaves_as_reference_branch():
    model = _FakeCFM()
    manager = build_lora_adapter_snapshot_manager(
        model,
        PolicySnapshotConfig(decay_type=1),
        current_adapter_name="default",
        old_adapter_name="old",
    )
    with torch.no_grad():
        model.transformer.lora_A["default"].weight.fill_(5.0)

    x = torch.randn(2, 2)
    kwargs = {
        "x": x,
        "cond": x,
        "text": [" ", " "],
        "time": torch.tensor([0.2, 0.8], dtype=torch.float32),
        "drop_audio_cond": False,
        "drop_text": True,
    }
    flow_ref = manager.ref_model.predict_flow(**kwargs)
    flow_old = manager.old_model.predict_flow(**kwargs)
    flow_cur = manager.current_model.predict_flow(**kwargs)

    assert torch.allclose(flow_ref, torch.zeros_like(flow_ref))
    assert not torch.allclose(flow_old, flow_cur)
    assert model.transformer._adapter_disabled is False
