"""Atomic run progress and checkpoint contract for resumable Qwen execution.

``status.json`` holds the latest machine-readable snapshot used for resume
decisions. ``events.jsonl`` appends one JSON object per line for the audit
trail. Status is always written via a sibling temporary file and
``os.replace`` so readers never observe partial JSON.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
STATUS_FILENAME = "status.json"
EVENTS_FILENAME = "events.jsonl"

VALID_STATES = frozenset({"pending", "running", "completed", "aborted", "failed"})
MUTABLE_STATES = frozenset({"pending", "running"})
TERMINAL_STATES = frozenset({"completed", "aborted"})

ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "pending": frozenset({"running", "aborted"}),
    "running": frozenset({"completed", "aborted", "failed"}),
    "completed": frozenset(),
    "aborted": frozenset(),
    "failed": frozenset({"running", "pending"}),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def validate_progress(*, completed_units: int, total_units: int) -> None:
    if completed_units < 0 or total_units < 0:
        raise ValueError("progress values must be non-negative")
    if completed_units > total_units:
        raise ValueError("completed_units exceeds total_units")


def validate_transition(current: str, target: str) -> None:
    if current not in VALID_STATES or target not in VALID_STATES:
        raise ValueError(f"unknown state: {current!r} -> {target!r}")
    if target not in ALLOWED_TRANSITIONS[current]:
        raise ValueError(
            f"invalid transition from {current!r} to {target!r}"
        )


class RunProgressStore:
    """Manage ``status.json`` and ``events.jsonl`` under an artifact directory."""

    def __init__(
        self,
        artifacts_dir: Path,
        *,
        profile: str,
        run_id: str,
    ) -> None:
        self.artifacts_dir = artifacts_dir
        self.profile = profile
        self.run_id = run_id
        self.status_path = artifacts_dir / STATUS_FILENAME
        self.events_path = artifacts_dir / EVENTS_FILENAME

    def _atomic_write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        directory = path.parent
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=directory,
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def _append_event(self, payload: dict[str, Any]) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        record = dict(payload)
        record["run_id"] = self.run_id
        record.setdefault("updated_at_utc", utc_now())
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def read_status(self) -> dict[str, Any] | None:
        if not self.status_path.is_file():
            return None
        return json.loads(self.status_path.read_text(encoding="utf-8"))

    def _write_status(self, payload: dict[str, Any]) -> None:
        self._atomic_write_json(self.status_path, payload)

    def _ensure_can_mutate(
        self,
        current: dict[str, Any],
        *,
        state: str | None,
        completed_units: int | None,
        total_units: int | None,
        message: str | None,
    ) -> None:
        current_state = current["state"]
        if current_state in TERMINAL_STATES:
            raise ValueError("status is terminal and cannot be mutated")
        changing_state = state is not None and state != current_state
        touching_fields = any(
            item is not None for item in (completed_units, total_units, message)
        )
        if current_state == "failed" and touching_fields and not changing_state:
            raise ValueError("status is terminal and cannot be mutated")
        if (
            current_state not in MUTABLE_STATES
            and touching_fields
            and not changing_state
        ):
            raise ValueError("status is terminal and cannot be mutated")

    def initialize(
        self,
        *,
        stage: str,
        total_units: int,
        message: str | None = None,
    ) -> dict[str, Any]:
        validate_progress(completed_units=0, total_units=total_units)
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "profile": self.profile,
            "run_id": self.run_id,
            "current_stage": stage,
            "state": "pending",
            "completed_units": 0,
            "total_units": total_units,
            "updated_at_utc": utc_now(),
        }
        if message is not None:
            payload["message"] = message
        self._write_status(payload)
        self._append_event(
            {
                "event": "initialized",
                "stage": stage,
                "state": "pending",
                "total_units": total_units,
            }
        )
        return payload

    def _mutate_status(
        self,
        *,
        state: str | None = None,
        completed_units: int | None = None,
        total_units: int | None = None,
        message: str | None = None,
        abort_gate: str | None = None,
        abort_reason: str | None = None,
        stage: str | None = None,
        event: str,
    ) -> dict[str, Any]:
        current = self.read_status()
        if current is None:
            raise ValueError("status is missing; call initialize first")
        self._ensure_can_mutate(
            current,
            state=state,
            completed_units=completed_units,
            total_units=total_units,
            message=message,
        )
        next_state = state if state is not None else current["state"]
        if state is not None:
            validate_transition(current["state"], next_state)
        next_completed = (
            completed_units
            if completed_units is not None
            else current["completed_units"]
        )
        next_total = (
            total_units if total_units is not None else current["total_units"]
        )
        validate_progress(completed_units=next_completed, total_units=next_total)
        payload = dict(current)
        payload["state"] = next_state
        payload["completed_units"] = next_completed
        payload["total_units"] = next_total
        payload["updated_at_utc"] = utc_now()
        if stage is not None:
            payload["current_stage"] = stage
        if message is not None:
            payload["message"] = message
        if abort_gate is not None:
            payload["abort_gate"] = abort_gate
        if abort_reason is not None:
            payload["abort_reason"] = abort_reason
        self._write_status(payload)
        event_record: dict[str, Any] = {
            "event": event,
            "state": next_state,
            "stage": payload["current_stage"],
            "completed_units": next_completed,
            "total_units": next_total,
        }
        if message is not None:
            event_record["message"] = message
        if abort_gate is not None:
            event_record["abort_gate"] = abort_gate
        if abort_reason is not None:
            event_record["abort_reason"] = abort_reason
        self._append_event(event_record)
        return payload

    def begin_running(self, message: str | None = None) -> dict[str, Any]:
        return self._mutate_status(
            state="running",
            message=message,
            event="state_changed",
        )

    def update_state(self, state: str) -> dict[str, Any]:
        return self._mutate_status(state=state, event="state_changed")

    def update_progress(
        self,
        *,
        completed_units: int,
        total_units: int | None = None,
        message: str | None = None,
    ) -> dict[str, Any]:
        return self._mutate_status(
            completed_units=completed_units,
            total_units=total_units,
            message=message,
            event="progress",
        )

    def complete_stage(self, message: str | None = None) -> dict[str, Any]:
        current = self.read_status()
        if current is None:
            raise ValueError("status is missing")
        return self._mutate_status(
            state="completed",
            completed_units=current["total_units"],
            message=message,
            event="stage_completed",
        )

    def abort(self, *, gate: str, reason: str) -> dict[str, Any]:
        return self._mutate_status(
            state="aborted",
            abort_gate=gate,
            abort_reason=reason,
            event="aborted",
        )

    def fail(self, message: str) -> dict[str, Any]:
        return self._mutate_status(
            state="failed",
            message=message,
            event="failed",
        )

    def should_skip_stage(self, stage: str) -> bool:
        status = self.read_status()
        if (
            status is not None
            and status.get("run_id") == self.run_id
            and status.get("current_stage") == stage
            and status.get("state") == "completed"
        ):
            return True
        if not self.events_path.is_file():
            return False
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if (
                record.get("run_id") == self.run_id
                and record.get("event") == "stage_completed"
                and record.get("stage") == stage
            ):
                return True
        return False