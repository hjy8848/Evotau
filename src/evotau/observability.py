"""Best-effort per-attempt telemetry; never an input to research decisions."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .alternating import _write_json_atomic

logger = logging.getLogger(__name__)


class EpisodeTelemetry:
    """Observe a native instance's steps without modifying messages or results.

    Each worker owns its attempt file. Atomic publication prevents torn snapshots.
    Observability failures must not change rollout/evaluation behavior.
    """

    def __init__(self, path: Path, *, manifest_sha256: str, attempt_id: str,
                 task_id: str, panel_name: str) -> None:
        self.path = path
        self.warned = False
        self.state = {"schema_version": 1, "manifest_sha256": manifest_sha256,
                      "attempt_id": attempt_id, "task_id": task_id, "panel_name": panel_name,
                      "turn_index": None, "role": None, "tool_name": None}
        self.publish("running", activity="preparing")

    def publish(self, status: str, **values: Any) -> None:
        self.state.update(status=status, updated_at=datetime.now(UTC).isoformat(), **values)
        try:
            _write_json_atomic(self.path, self.state)
        except Exception:  # noqa: BLE001 - telemetry cannot fail a native episode
            self.warn()

    def warn(self) -> None:
        if not self.warned:
            logger.warning("Live episode telemetry unavailable; native execution continues unchanged")
            self.warned = True

    def snapshot(self, orchestrator: Any, activity: str) -> None:
        try:
            messages = orchestrator.get_trajectory()
            message = messages[-1] if messages else None
            role = getattr(message, "role", None)
            tool_calls = getattr(message, "tool_calls", None) or []
            tool = getattr(tool_calls[0], "name", None) if tool_calls else getattr(message, "name", None)
            if role == "tool" and tool is None:
                tool = next((call.name for previous in reversed(messages[:-1])
                             for call in (getattr(previous, "tool_calls", None) or [])
                             if call.id == getattr(message, "id", None)), None)
            next_role = getattr(orchestrator, "to_role", None)
            self.publish("running", activity=activity,
                         turn_index=getattr(message, "turn_idx", None),
                         role=getattr(role, "value", role), tool_name=tool,
                         next_role=getattr(next_role, "value", next_role))
        except Exception:  # noqa: BLE001 - no diagnostic may alter native execution
            self.warn()

    @contextmanager
    def observe(self, orchestrator: Any):
        original = getattr(orchestrator, "step", None)
        if not callable(original):
            yield
            return
        had_override = "step" in vars(orchestrator)
        override = vars(orchestrator).get("step")

        def step():
            self.snapshot(orchestrator, "native_step_in_flight")
            try:
                return original()
            finally:
                self.snapshot(orchestrator, "native_step_recorded")

        orchestrator.step = step
        try:
            yield
        finally:
            if had_override:
                orchestrator.step = override
            else:
                del orchestrator.step
