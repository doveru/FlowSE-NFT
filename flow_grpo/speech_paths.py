"""Stable path rules shared by speech configuration, runtime, and datasets."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_data_root(data_root: str | Path | None = None) -> Path:
    return resolve_project_path(data_root or os.environ.get("SPEECH_DATA_ROOT") or "../DNS_noreverb")


def resolve_audio_path(path: str | Path, data_root: str | Path | None = None) -> Path:
    """Absolute legacy entries remain valid; relative entries use the data root."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else (resolve_data_root(data_root) / path).resolve()
