"""Small append-only SQLite archive for verified failure signatures."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .manifest import sha256_json
from .records import EvidenceRef, FailureRecord, FailureSignature, customer_strategy_id
from .strategies import CustomerStrategy, ServiceStrategy

MAX_ACTIVE_FAILURE_REPRESENTATIVES = 32


class FailureArchive:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS failures (
                failure_id TEXT PRIMARY KEY,
                signature_key TEXT NOT NULL,
                task_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                payload TEXT NOT NULL,
                protocol_version INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""")
            columns = {row["name"] for row in db.execute("PRAGMA table_info(failures)")}
            if "protocol_version" not in columns:
                # Rows from the previous single-trace protocol stay in the
                # append-only log, but are not eligible for current replay.
                db.execute(
                    "ALTER TABLE failures ADD COLUMN protocol_version INTEGER NOT NULL DEFAULT 1"
                )
            db.execute("CREATE INDEX IF NOT EXISTS failures_signature ON failures(signature_key, generation DESC)")
            db.execute("CREATE INDEX IF NOT EXISTS failures_task ON failures(task_id, generation DESC)")
            db.execute("""CREATE TABLE IF NOT EXISTS replay_events (
                signature_key TEXT NOT NULL,
                generation INTEGER NOT NULL,
                failure_id TEXT NOT NULL,
                PRIMARY KEY(signature_key, generation)
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS customer_strategies (
                strategy_id TEXT PRIMARY KEY,
                parent_id TEXT,
                operator TEXT NOT NULL,
                generation INTEGER NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS service_strategies (
                strategy_id TEXT PRIMARY KEY,
                parent_id TEXT,
                operator TEXT NOT NULL,
                generation INTEGER NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""")

    def append_customer_strategy(self, strategy: CustomerStrategy | object, *, parent_id: str | None,
                                 operator: str, generation: int) -> bool:
        if generation < 0 or not operator:
            raise ValueError("Customer strategy archive requires a non-negative generation and operator")
        strategy_id = customer_strategy_id(strategy)
        payload = json.dumps(strategy.to_dict(), sort_keys=True, separators=(",", ":"))
        with self._connect() as db:
            cursor = db.execute(
                "INSERT OR IGNORE INTO customer_strategies(strategy_id, parent_id, operator, generation, payload) VALUES (?, ?, ?, ?, ?)",
                (strategy_id, parent_id, operator, generation, payload),
            )
            return cursor.rowcount == 1

    def customer_strategy_ids(self) -> frozenset[str]:
        with self._connect() as db:
            rows = db.execute("SELECT strategy_id FROM customer_strategies").fetchall()
        return frozenset(row["strategy_id"] for row in rows)

    def get_customer_strategy(self, strategy_id: str) -> CustomerStrategy | object | None:
        """Resolve one immutable Customer snapshot by its canonical strategy ID."""
        if not isinstance(strategy_id, str) or not strategy_id.strip():
            raise ValueError("Customer strategy lookup requires a non-empty strategy ID")
        with self._connect() as db:
            row = db.execute(
                "SELECT payload FROM customer_strategies WHERE strategy_id=?",
                (strategy_id,),
            ).fetchone()
        if row is None:
            return None
        value = json.loads(row["payload"])
        if value.get("schema_version") == 3:
            from .lifecycle import _customer_from_dict

            return _customer_from_dict(value)
        return CustomerStrategy(**value)

    def customer_strategies(self, limit: int = 100) -> tuple[dict, ...]:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        with self._connect() as db:
            rows = db.execute("SELECT payload FROM customer_strategies ORDER BY generation DESC, created_at DESC LIMIT ?", (limit,)).fetchall()
        return tuple(json.loads(row["payload"]) for row in rows)

    def append_service_strategy(self, strategy: ServiceStrategy, *, parent_id: str | None,
                                operator: str, generation: int) -> bool:
        if generation < 0 or not operator:
            raise ValueError("Service strategy archive requires a non-negative generation and operator")
        strategy_id = sha256_json(strategy.to_dict())[:16]
        payload = json.dumps(strategy.to_dict(), sort_keys=True, separators=(",", ":"))
        with self._connect() as db:
            cursor = db.execute(
                "INSERT OR IGNORE INTO service_strategies(strategy_id, parent_id, operator, generation, payload) VALUES (?, ?, ?, ?, ?)",
                (strategy_id, parent_id, operator, generation, payload),
            )
            return cursor.rowcount == 1

    def service_strategies(self, limit: int = 100) -> tuple[dict, ...]:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        with self._connect() as db:
            rows = db.execute("SELECT payload FROM service_strategies ORDER BY generation DESC, created_at DESC LIMIT ?", (limit,)).fetchall()
        return tuple(json.loads(row["payload"]) for row in rows)

    def append(self, failure: FailureRecord) -> bool:
        """Append once by immutable failure ID; return False for an exact replay."""
        payload = json.dumps(_failure_to_dict(failure), sort_keys=True, separators=(",", ":"))
        with self._connect() as db:
            cursor = db.execute(
                "INSERT OR IGNORE INTO failures(failure_id, signature_key, task_id, generation, payload, protocol_version) VALUES (?, ?, ?, ?, ?, 2)",
                (failure.failure_id, failure.signature.key, failure.task_id, failure.generation, payload),
            )
            return cursor.rowcount == 1

    def recent(self, limit: int = 100) -> tuple[FailureRecord, ...]:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        with self._connect() as db:
            rows = db.execute(
                "SELECT payload FROM failures WHERE protocol_version=2 ORDER BY generation DESC, created_at DESC, failure_id LIMIT ?",
                (limit,),
            ).fetchall()
        return tuple(_failure_from_dict(json.loads(row["payload"])) for row in rows)

    def representatives(self, limit: int = 100) -> tuple[FailureRecord, ...]:
        """Return newest representative of each recurring signature."""
        if limit < 0:
            raise ValueError("limit must be non-negative")
        with self._connect() as db:
            rows = db.execute(
                "SELECT payload FROM failures WHERE protocol_version=2 ORDER BY generation DESC, created_at DESC, failure_id"
            ).fetchall()
        latest_by_signature: dict[str, FailureRecord] = {}
        for row in rows:
            failure = _failure_from_dict(json.loads(row["payload"]))
            latest_by_signature.setdefault(failure.signature.key, failure)
        return tuple(latest_by_signature.values())[:limit]

    def active_representatives(
        self,
        limit: int = MAX_ACTIVE_FAILURE_REPRESENTATIVES,
        *,
        current_generation: int | None = None,
    ) -> tuple[FailureRecord, ...]:
        """Select a bounded replay set, preserving raw archive history.

        Untested signatures are favored, then recent recurrence, severity,
        total recurrence, and recency. The fixed tie order makes selections
        reproducible even when SQLite returns equivalent rows in another order.
        """
        if not 0 <= limit <= MAX_ACTIVE_FAILURE_REPRESENTATIVES:
            raise ValueError(f"active failure representatives are capped at {MAX_ACTIVE_FAILURE_REPRESENTATIVES}")
        all_failures = self.recent(limit=2**31 - 1)
        all_representatives = self.representatives(limit=2**31 - 1)
        if current_generation is None:
            current_generation = max((item.generation for item in all_representatives), default=0)
        if current_generation < 0:
            raise ValueError("current generation must be non-negative")
        with self._connect() as db:
            replayed_failure_ids = {
                row["failure_id"] for row in db.execute("SELECT DISTINCT failure_id FROM replay_events")
            }
        valid_ids = {failure.failure_id for failure in all_failures}
        signature_by_id = {failure.failure_id: failure.signature.key for failure in all_failures}
        tested = {
            signature_by_id[failure_id]
            for failure_id in replayed_failure_ids & valid_ids
        }
        counts: dict[str, tuple[int, int]] = {}
        for failure in all_failures:
            total, recent = counts.get(failure.signature.key, (0, 0))
            counts[failure.signature.key] = (
                total + 1,
                recent + int(failure.generation >= max(0, current_generation - 1)),
            )
        severity_rank = {"critical": 0, "high": 1, "material": 2, "low": 3}
        ordered = sorted(
            all_representatives,
            key=lambda failure: (
                -(counts[failure.signature.key][1]),
                severity_rank.get(failure.severity.lower(), 4),
                failure.signature.key in tested,
                -counts[failure.signature.key][0],
                -failure.generation,
                failure.failure_id,
            ),
        )
        return tuple(ordered[:limit])

    def mark_replayed(self, failure_id: str, *, generation: int) -> bool:
        """Record at most one replay-coverage event per signature and generation."""
        if generation < 0:
            raise ValueError("replay generation must be non-negative")
        with self._connect() as db:
            row = db.execute(
                "SELECT signature_key, payload, protocol_version FROM failures WHERE failure_id=?",
                (failure_id,),
            ).fetchone()
            if row is None or row["protocol_version"] != 2:
                raise ValueError("cannot mark an unknown or unreproduced failure as replayed")
            cursor = db.execute(
                "INSERT OR IGNORE INTO replay_events(signature_key, generation, failure_id) VALUES (?, ?, ?)",
                (row["signature_key"], generation, failure_id),
            )
            return cursor.rowcount == 1

    def active_replay_coverage(self, limit: int = MAX_ACTIVE_FAILURE_REPRESENTATIVES) -> dict[str, int | float]:
        """Report tested and untested signature representatives in the active set."""
        active = self.active_representatives(limit)
        signatures = {item.signature.key for item in active}
        with self._connect() as db:
            tested = {
                row["signature_key"] for row in db.execute("SELECT DISTINCT signature_key FROM replay_events")
            }
        covered = len(signatures & tested)
        total = len(signatures)
        return {
            "active_signatures": total,
            "replayed_signatures": covered,
            "uncovered_signatures": total - covered,
            "coverage_rate": covered / total if total else 1.0,
        }

    def recurrence_count(self, signature_key: str) -> int:
        return sum(failure.signature.key == signature_key for failure in self.recent(limit=2**31 - 1))


def _failure_to_dict(failure: FailureRecord) -> dict:
    return {
        "failure_id": failure.failure_id, "episode_id": failure.episode_id,
        "task_id": failure.task_id, "generation": failure.generation,
        "customer_strategy_id": failure.customer_strategy_id,
        "service_strategy_id": failure.service_strategy_id,
        "signature": failure.signature.to_dict(), "policy_ref": failure.policy_ref,
        "evidence": [ref.__dict__ if hasattr(ref, "__dict__") else {"turn_index": ref.turn_index, "source": ref.source, "summary": ref.summary} for ref in failure.evidence],
        "verification_ref": failure.verification_ref,
        "reproduction_episode_id": failure.reproduction_episode_id,
        "reproduction_verification_ref": failure.reproduction_verification_ref,
        "severity": failure.severity,
    }


def _failure_from_dict(value: dict) -> FailureRecord:
    return FailureRecord(
        failure_id=value["failure_id"], episode_id=value["episode_id"], task_id=value["task_id"],
        generation=value["generation"], customer_strategy_id=value["customer_strategy_id"],
        service_strategy_id=value["service_strategy_id"], signature=FailureSignature(**value["signature"]),
        policy_ref=value["policy_ref"], evidence=tuple(EvidenceRef(**ref) for ref in value["evidence"]),
        verification_ref=value["verification_ref"],
        reproduction_episode_id=value["reproduction_episode_id"],
        reproduction_verification_ref=value["reproduction_verification_ref"],
        severity=value["severity"],
    )
