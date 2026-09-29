"""Append-only console event journal derived from committed Core artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from .artifact_reader import redact_secrets

_JOURNAL_LOCKS: dict[str, Lock] = {}


class EventJournalError(ValueError):
    """The console observation journal is unsafe or malformed."""


class EventJournal:
    """Persist UI-only observations without feeding them back to EvoTau Core."""

    def path_for(self, run: dict[str, Any]) -> Path:
        return Path(run["path"]) / "events.jsonl"

    def sync(self, run: dict[str, Any]) -> tuple[dict[str, Any], ...]:
        run_path = Path(run["path"])
        if not ((run_path / "manifest.json").is_file()
                or (run_path / "pilot-manifest.json").is_file()):
            return ()
        path = self.path_for(run)
        if path.is_symlink():
            raise EventJournalError("event journal cannot be a symlink")
        lock = _JOURNAL_LOCKS.setdefault(str(path.resolve()), Lock())
        with lock:
            existing = self.read(run)
            known = {row["event_id"] for row in existing}
            proposed = list(self._observations(run))
            missing = []
            for row in proposed:
                if row["event_id"] not in known:
                    missing.append(row)
                    known.add(row["event_id"])
            if missing:
                path.parent.mkdir(parents=True, exist_ok=True)
                flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                fd = os.open(path, flags, 0o600)
                try:
                    with os.fdopen(fd, "a", encoding="utf-8", newline="\n") as stream:
                        for event in missing:
                            stream.write(json.dumps(
                                event, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":"), allow_nan=False,
                            ) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                except Exception:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                    raise
            return self.read(run)

    def read(self, run: dict[str, Any]) -> tuple[dict[str, Any], ...]:
        path = self.path_for(run)
        if not path.exists():
            return ()
        if path.is_symlink() or not path.is_file():
            raise EventJournalError("event journal is not a regular file")
        events = []
        try:
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line:
                    continue
                event = json.loads(line)
                if (not isinstance(event, dict) or set(event) != {
                    "event_id", "timestamp", "run_id", "event_type", "generation", "episode_id", "payload",
                }):
                    raise ValueError("event fields do not match the schema")
                if event["run_id"] != run["run_id"] or not isinstance(event["event_id"], str):
                    raise ValueError("event identity does not match the run")
                events.append(redact_secrets(event))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise EventJournalError("event journal is malformed and has been hidden") from exc
        return tuple(events)

    def _observations(self, run: dict[str, Any]):
        run_id = run["run_id"]
        run_path = Path(run["path"])
        manifest_name = "pilot-manifest.json" if (run_path / "pilot-manifest.json").exists() else "manifest.json"
        manifest_path = run_path / manifest_name
        timestamp = _file_time(manifest_path)
        yield self._event(run_id, "run_started", f"{manifest_name}:started", timestamp, None, None, {
            "experiment_id": run["experiment_id"], "phase": run["phase"],
        })
        for commit in run.get("generation_commits", ()):
            generation = commit.get("generation")
            yield self._event(run_id, "generation_committed", f"generation:{generation}", timestamp,
                              generation, None, {
                                  "customer_id": commit.get("customer_id"),
                                  "service_id": commit.get("service_id"),
                                  "customer_evolved": commit.get("customer_evolved"),
                                  "service_evolved": commit.get("service_evolved"),
                                  "note": commit.get("note"),
                              })
            gate = ((commit.get("decision_record") or {}).get("service") or {}).get("gate")
            if isinstance(gate, dict):
                yield self._event(run_id, "service_repair_proposed", f"generation:{generation}:repair", timestamp,
                                  generation, None, redact_secrets(gate.get("proposal")))
                status = "inconclusive" if gate.get("inconclusive") else (
                    "accepted" if gate.get("accepted") else "rejected"
                )
                yield self._event(run_id, "service_gate_finished", f"generation:{generation}:gate", timestamp,
                                  generation, None, {"status": status, "reasons": gate.get("reasons", ())})
                yield self._event(run_id, f"service_repair_{status}", f"generation:{generation}:decision", timestamp,
                                  generation, None, {"target_failure_id": gate.get("target_failure_id")})
        active_episode = run.get("current_episode")
        if isinstance(active_episode, dict):
            attempt_id = active_episode.get("attempt_id")
            if attempt_id:
                yield self._event(
                    run_id, "episode_started", f"attempt:{attempt_id}:started",
                    active_episode.get("started_at", timestamp),
                    run.get("current_generation"), str(attempt_id),
                    {"state": "episode directory created; task details are not yet available"},
                )
        for episode in run.get("episodes", ()):
            episode_id = episode.get("episode_id")
            if episode_id is None:
                continue
            ep_time = episode.get("started_at") or timestamp
            yield self._event(run_id, "episode_started", f"episode:{episode_id}:started", ep_time,
                              episode.get("generation"), episode_id, {"task_id": episode.get("task_id")})
            yield self._event(run_id, "episode_finished", f"episode:{episode_id}:finished", ep_time,
                              episode.get("generation"), episode_id, {
                                  "task_id": episode.get("task_id"), "status": episode.get("status"),
                                  "native_reward": episode.get("native_reward"),
                              })
            for message in episode.get("messages", ()):
                turn = message.get("turn_index")
                kind = message.get("kind")
                if kind == "message":
                    event_type = "customer_message" if message.get("role") == "user" else "agent_message"
                    yield self._event(run_id, event_type, f"episode:{episode_id}:message:{turn}:{message.get('role')}",
                                      ep_time, episode.get("generation"), episode_id, {
                                          "turn_index": turn, "role": message.get("role"),
                                          "content": message.get("content", ""),
                                      })
                    for call_index, call in enumerate(message.get("tool_calls", ())):
                        yield self._event(run_id, "tool_call",
                                          f"episode:{episode_id}:call:{turn}:{call.get('id', call_index)}",
                                          ep_time, episode.get("generation"), episode_id, call)
                elif kind == "tool_call":
                    yield self._event(run_id, "tool_call",
                                      f"episode:{episode_id}:call:{turn}:{message.get('tool_id', turn)}",
                                      ep_time, episode.get("generation"), episode_id, message)
                elif kind == "tool_result":
                    yield self._event(run_id, "tool_result",
                                      f"episode:{episode_id}:tool-result:{turn}:{message.get('tool_id', turn)}",
                                      ep_time, episode.get("generation"), episode_id, message)
            if episode.get("task_success") is not None:
                yield self._event(run_id, "evaluation_finished", f"episode:{episode_id}:evaluation", ep_time,
                                  episode.get("generation"), episode_id, {
                                      "task_success": episode.get("task_success"),
                                      "native_reward": episode.get("native_reward"),
                                  })
            if episode.get("budget") is not None:
                yield self._event(run_id, "budget_updated", f"episode:{episode_id}:budget", ep_time,
                                  episode.get("generation"), episode_id, episode["budget"])
        for failure in run.get("verified_failures", ()):
            failure_id = failure.get("failure_id")
            yield self._event(run_id, "failure_verified", f"failure:{failure_id}", timestamp,
                              failure.get("generation"), failure.get("episode_id"), {
                                  "failure_id": failure_id,
                                  "task_id": failure.get("task_id"),
                                  "workflow_stage": (failure.get("signature") or {}).get("workflow_stage"),
                                  "policy_ref": failure.get("policy_ref"),
                              })
        if run.get("status") == "complete":
            yield self._event(run_id, "run_finished", "run:finished", timestamp, None, None, {
                "provider_attempts": run.get("budget", {}).get("attempts"),
            })
        elif run.get("status") == "failed":
            yield self._event(run_id, "run_failed", "run:failed", timestamp, None, None, {
                "status": "incomplete",
            })

    @staticmethod
    def _event(run_id, event_type, source_key, timestamp, generation, episode_id, payload):
        event_id = hashlib.sha256(f"{run_id}\0{source_key}\0{event_type}".encode()).hexdigest()
        return {
            "event_id": event_id,
            "timestamp": timestamp,
            "run_id": run_id,
            "event_type": event_type,
            "generation": generation,
            "episode_id": episode_id,
            "payload": redact_secrets(payload),
        }


def _file_time(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()
