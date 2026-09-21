import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from flow_grpo import speech_diagnostics as diagnostics


def test_disabled_has_no_files_or_timer(tmp_path, monkeypatch):
    monkeypatch.delenv("SPEECH_DIAGNOSTICS", raising=False)
    monkeypatch.setenv("SPEECH_DIAGNOSTICS_DIR", str(tmp_path / "unused"))
    def unexpected(*args, **kwargs):
        pytest.fail("disabled diagnostics must not arm a timer")
    monkeypatch.setattr(diagnostics.faulthandler, "dump_traceback_later", unexpected)
    with diagnostics.diagnostic_session():
        diagnostics.diagnostic_stage("unused")
    assert not list(tmp_path.iterdir())


def test_enabled_records_rank_and_cleans_up_on_error(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECH_DIAGNOSTICS", "1")
    monkeypatch.setenv("SPEECH_DIAGNOSTICS_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "3")
    armed, cancelled, dumped = [], [], []
    monkeypatch.setattr(diagnostics.faulthandler, "dump_traceback_later", lambda *a, **kw: armed.append((a, kw)))
    monkeypatch.setattr(diagnostics.faulthandler, "cancel_dump_traceback_later", lambda: cancelled.append(True))
    monkeypatch.setattr(diagnostics.faulthandler, "dump_traceback", lambda **kw: dumped.append(True))
    with pytest.raises(RuntimeError, match="test error"):
        with diagnostics.diagnostic_session():
            diagnostics.diagnostic_stage("backward.begin", timestep_index=4)
            raise RuntimeError("test error")
    rows = [json.loads(line) for line in next(tmp_path.glob("*.stages.jsonl")).read_text().splitlines()]
    assert [row["stage"] for row in rows] == ["session.start", "backward.begin", "session.error", "session.end"]
    assert all(row["rank"] == "3" and row["pid"] == os.getpid() for row in rows)
    assert armed[0][1]["exit"] is False
    assert armed[0][1]["repeat"] is True
    assert cancelled == [True] and dumped == [True]
    assert diagnostics._events is None


@pytest.mark.parametrize("interval", ["0", "-1", "nan", "inf", "bad"])
def test_invalid_interval_rejected(monkeypatch, interval):
    monkeypatch.setenv("SPEECH_DIAGNOSTICS", "1")
    monkeypatch.setenv("SPEECH_DIAGNOSTICS_INTERVAL", interval)
    with pytest.raises(ValueError):
        with diagnostics.diagnostic_session():
            pass


def test_real_stack_dump_in_subprocess(tmp_path):
    env = dict(os.environ, SPEECH_DIAGNOSTICS="1", SPEECH_DIAGNOSTICS_DIR=str(tmp_path), SPEECH_DIAGNOSTICS_INTERVAL="0.1")
    result = subprocess.run(
        [sys.executable, "-c", "from flow_grpo.speech_diagnostics import diagnostic_session; import time\nwith diagnostic_session():\n time.sleep(0.35)\nprint('finished')"],
        cwd=str(Path(__file__).resolve().parents[1]), env=env,
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "finished" in result.stdout
    stack = next(tmp_path.glob("*.stacks.log")).read_text()
    assert "Timeout" in stack and "<string>" in stack


def test_repeated_sessions_use_separate_files(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECH_DIAGNOSTICS", "1")
    monkeypatch.setenv("SPEECH_DIAGNOSTICS_DIR", str(tmp_path))
    for _ in range(2):
        with diagnostics.diagnostic_session():
            diagnostics.diagnostic_stage("test")
    assert len(list(tmp_path.glob("*.stages.jsonl"))) == 2
