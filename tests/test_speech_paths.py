from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from flow_grpo.speech_paths import PROJECT_ROOT, resolve_audio_path, resolve_data_root, resolve_project_path
from scripts.build_speech_manifest import build_manifest


def load_dataset_module():
    # Isolate the optional model imports without replacing any production modules.
    spec = importlib.util.spec_from_file_location(
        "_speech_path_tests.rl_dataset", PROJECT_ROOT / "flow_grpo/speech_flowse/rl_dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resource_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_project_path("flow_grpo/speech_flowse/ckpts/best.pt.tar") == (
        PROJECT_ROOT / "flow_grpo/speech_flowse/ckpts/best.pt.tar"
    )
    assert resolve_project_path(tmp_path / "weights") == tmp_path / "weights"
    assert resolve_project_path("~/weights") == Path.home() / "weights"


def test_data_root_precedence_and_legacy_absolute_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECH_DATA_ROOT", str(tmp_path / "environment data"))
    assert resolve_audio_path("valid/noisy/0.wav") == tmp_path / "environment data/valid/noisy/0.wav"
    assert resolve_audio_path("train/clean/a.wav", tmp_path / "explicit") == tmp_path / "explicit/train/clean/a.wav"
    absolute = tmp_path / "old/absolute.wav"
    assert resolve_audio_path(absolute, tmp_path / "different") == absolute
    monkeypatch.setenv("SPEECH_DATA_ROOT", "relative-data")
    assert resolve_data_root() == PROJECT_ROOT / "relative-data"
    monkeypatch.delenv("SPEECH_DATA_ROOT")
    assert resolve_data_root() == PROJECT_ROOT.parent / "DNS_noreverb"


def test_builder_relative_paths_round_trip_after_moving_data(tmp_path, monkeypatch):
    data = tmp_path / "source data"
    for kind in ("noisy", "clean"):
        folder = data / "train" / kind
        folder.mkdir(parents=True)
        sf.write(folder / "sample.wav", np.zeros(3200, dtype=np.float32), 16000)
    args = SimpleNamespace(
        noisy_dir=data / "train/noisy", clean_dir=data / "train/clean",
        output_manifest=tmp_path / "manifest.jsonl", split="train", data_root=data,
        duration_tolerance_ms=50.0, sample_rate=16000, chunk_seconds=0.1,
    )
    relative, stats = build_manifest(args)
    args.data_root = None
    absolute, _ = build_manifest(args)
    assert stats["audio_path_base"] == "SPEECH_DATA_ROOT"
    for rel, old in zip(relative, absolute):
        for key in ("noisy_path", "clean_path"):
            assert data / rel[key] == Path(old[key])
        assert {k: v for k, v in rel.items() if k not in ("noisy_path", "clean_path")} == {
            k: v for k, v in old.items() if k not in ("noisy_path", "clean_path")
        }
    args.output_manifest.write_text("".join(json.dumps(r) + "\n" for r in relative))
    moved = tmp_path / "moved data"
    data.rename(moved)
    monkeypatch.chdir(tmp_path)
    module = load_dataset_module()
    entries = module.read_manifest(args.output_manifest, data_root=moved)
    assert len(entries) == len(relative) == 2
    for entry in entries:
        for key in ("noisy_path", "clean_path"):
            assert Path(entry[key]).is_file()
            assert module.load_first_channel_audio(entry[key], 16000).shape == (1, 3200)


def test_builder_rejects_files_outside_data_root(tmp_path):
    for kind in ("noisy", "clean"):
        (tmp_path / kind).mkdir()
        sf.write(tmp_path / kind / "a.wav", np.zeros(160), 16000)
    args = SimpleNamespace(
        noisy_dir=tmp_path / "noisy", clean_dir=tmp_path / "clean",
        output_manifest=tmp_path / "out.jsonl", split="train", data_root=tmp_path / "wrong",
        duration_tolerance_ms=50.0, sample_rate=16000, chunk_seconds=1.0,
    )
    with pytest.raises(ValueError, match="outside --data_root"):
        build_manifest(args)


def test_manifest_absolute_paths_and_invalid_entries(tmp_path):
    module = load_dataset_module()
    path = tmp_path / "manifest.jsonl"
    entry = {"noisy_path": str(tmp_path / "old/noisy.wav"), "clean_path": str(tmp_path / "old/clean.wav")}
    path.write_text(json.dumps(entry) + "\n")
    assert module.read_manifest(path, tmp_path / "new") == [entry]
    path.write_text(json.dumps({"noisy_path": ""}) + "\n")
    with pytest.raises(ValueError, match="noisy_path.*manifest.jsonl:1"):
        module.read_manifest(path)
