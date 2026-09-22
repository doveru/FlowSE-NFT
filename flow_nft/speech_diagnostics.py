"""Opt-in, process-local hang diagnostics; no CUDA calls or collectives."""

from contextlib import contextmanager
from datetime import datetime, timezone
import faulthandler
import json
import math
import os
from pathlib import Path
import time


_events = None


def diagnostic_stage(stage, **fields):
    if _events is None:
        return
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "monotonic_sec": time.monotonic(),
        "rank": os.environ.get("RANK", "0"),
        "pid": os.getpid(),
        "stage": stage,
        **fields,
    }
    _events.write(json.dumps(record, ensure_ascii=False) + "\n")
    _events.flush()


@contextmanager
def diagnostic_session():
    """SPEECH_DIAGNOSTICS=1 enables periodic Python stacks and stage logs.

    Stage completion means the CPU returned, not that queued CUDA work finished.
    Periodic stacks are samples, not proof of a hang. No signals are registered.
    """
    global _events
    if os.environ.get("SPEECH_DIAGNOSTICS", "").lower() not in {"1", "true", "yes"}:
        yield
        return
    interval = float(os.environ.get("SPEECH_DIAGNOSTICS_INTERVAL", "120"))
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("SPEECH_DIAGNOSTICS_INTERVAL must be finite and positive")
    directory = Path(os.environ.get("SPEECH_DIAGNOSTICS_DIR", "logs/speech_diagnostics")).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    prefix = directory / f"{stamp}-rank{os.environ.get('RANK', '0')}-pid{os.getpid()}"
    events_path = Path(str(prefix) + ".stages.jsonl")
    stacks_path = Path(str(prefix) + ".stacks.log")
    with events_path.open("x", encoding="utf-8") as events, stacks_path.open("x", encoding="utf-8") as stacks:
        _events = events
        try:
            faulthandler.dump_traceback_later(interval, repeat=True, file=stacks, exit=False)
            diagnostic_stage("session.start", stack_interval_sec=interval)
            print(f"[diagnostics rank={os.environ.get('RANK', '0')}] stages={events_path} stacks={stacks_path}", flush=True)
            yield
        except BaseException as exc:
            diagnostic_stage("session.error", error_type=type(exc).__name__, error=str(exc))
            faulthandler.dump_traceback(file=stacks, all_threads=True)
            raise
        finally:
            faulthandler.cancel_dump_traceback_later()
            diagnostic_stage("session.end")
            _events = None
