from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from flow_grpo import speech_distributed as control


@pytest.mark.parametrize("save,best", [(False, float("-inf")), (True, 3.95), (False, 3.95)])
def test_single_process_passthrough(monkeypatch, save, best):
    monkeypatch.setattr(control.dist, "is_initialized", lambda: False)
    assert control.broadcast_save_decision(save, best, "cpu") == (save, best)
    control.epoch_barrier("cpu")


@pytest.mark.parametrize("rank", [0, 1])
def test_nccl_decision_uses_explicit_device_and_fixed_float64_tensor(monkeypatch, rank):
    monkeypatch.setattr(control.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(control.dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(control.dist, "get_rank", lambda: rank)
    original_tensor = torch.tensor
    allocated = []
    def make_tensor(values, *, dtype, device):
        allocated.append((dtype, device, values))
        return original_tensor(values, dtype=dtype)  # Simulate CUDA allocation on CPU.
    def broadcast(tensor, src):
        assert src == 0 and tuple(tensor.shape) == (2,)
        tensor.copy_(original_tensor([1.0, 3.95], dtype=torch.float64))
    def forbidden(*args, **kwargs):
        pytest.fail("save decision must not use object collectives")
    monkeypatch.setattr(control.torch, "tensor", make_tensor)
    monkeypatch.setattr(control.dist, "broadcast", broadcast)
    monkeypatch.setattr(control.dist, "broadcast_object_list", forbidden)
    assert control.broadcast_save_decision(True, 3.95, "cuda:4") == (True, 3.95)
    assert allocated[0][:2] == (torch.float64, torch.device("cuda:4"))
    assert allocated[0][2] == ([1.0, 3.95] if rank == 0 else [0.0, 0.0])


@pytest.mark.parametrize("backend,device,expected", [("nccl", "cuda:3", {"device_ids": [3]}), ("gloo", "cpu", {})])
def test_barrier_device_selection(monkeypatch, backend, device, expected):
    monkeypatch.setattr(control.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(control.dist, "get_backend", lambda: backend)
    calls = []
    monkeypatch.setattr(control.dist, "barrier", lambda **kwargs: calls.append(kwargs))
    control.epoch_barrier(device)
    assert calls == [expected]


def _gloo_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=20))
    try:
        # Four non-validation epochs, an improved checkpoint, then no improvement.
        decisions = [(False, float("-inf"))] * 4 + [(True, 3.95), (False, 3.95)]
        for expected in decisions:
            supplied = expected if rank == 0 else (True, -999.0)
            assert control.broadcast_save_decision(*supplied, device="cpu") == expected
            control.epoch_barrier("cpu")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable")
def test_two_rank_broadcast_and_epoch_barriers(tmp_path):
    mp.spawn(_gloo_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
