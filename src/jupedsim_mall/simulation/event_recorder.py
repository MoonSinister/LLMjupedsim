"""Structured lifecycle event recording for simulation runs."""

from __future__ import annotations

import json
import pathlib
from datetime import datetime
from enum import Enum
from typing import Any


class RunState(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TerminalState(str, Enum):
    COMPLETED = "completed"
    TTL_REMOVED = "ttl_removed"
    SPAWN_FAILED = "failed"
    UNFINISHED = "unfinished"


class NullEventRecorder:
    def record(self, event_type: str, **payload: Any) -> None:
        del event_type, payload

    def close(self) -> None:
        return None


class EventRecorder:
    def __init__(self, path: str | pathlib.Path, *, run_id: str = "", simulation_dt: float = 0.01):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.simulation_dt = simulation_dt
        self._stream = self.path.open("a", encoding="utf-8", newline="\n")

    def record(self, event_type: str, **payload: Any) -> None:
        iteration = payload.get("iteration")
        event = {
            "schema_version": "1.0",
            "recorded_at": datetime.now().astimezone().isoformat(),
            "run_id": self.run_id,
            "event_type": event_type,
            **payload,
        }
        if iteration is not None and "simulation_time" not in event:
            event["simulation_time"] = round(iteration * self.simulation_dt, 6)
        self._stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        self._stream.flush()

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    def __enter__(self) -> "EventRecorder":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def create_event_recorder(path: str, *, run_id: str, simulation_dt: float):
    if not path:
        return NullEventRecorder()
    return EventRecorder(path, run_id=run_id, simulation_dt=simulation_dt)
