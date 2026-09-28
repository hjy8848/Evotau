"""Small append-only SQLite archive for verified failure signatures."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .manifest import sha256_json
from .records import EvidenceRef, FailureRecord, FailureSignature
from .strategies import CustomerStrategy, ServiceStrategy


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
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS failures_signature ON failures(signature_key, generation DESC)")
            db.execute("CREATE INDEX IF NOT EXISTS failures_task ON failures(task_id, generation DESC)")
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

    def append_customer_strategy(self, strategy: CustomerStrategy, *, parent_id: str | None,
                                 operator: str, generation: int) -> bool:
        if generation < 0 or not operator:
            raise ValueError("Customer strategy archive requires a non-negative generation and operator")
        strategy_id = sha256_json(strategy.to_dict())[:16]
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
                "INSERT OR IGNORE INTO failures(failure_id, signature_key, task_id, generation, payload) VALUES (?, ?, ?, ?, ?)",
                (failure.failure_id, failure.signature.key, failure.task_id, failure.generation, payload),
            )
            return cursor.rowcount == 1

    def recent(self, limit: int = 100) -> tuple[FailureRecord, ...]:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        with self._connect() as db:
            rows = db.execute(
                "SELECT payload FROM failures ORDER BY generation DESC, created_at DESC, failure_id LIMIT ?",
                (limit,),
            ).fetchall()
        return tuple(_failure_from_dict(json.loads(row["payload"])) for row in rows)

    def representatives(self, limit: int = 100) -> tuple[FailureRecord, ...]:
        """Return newest representative of each recurring signature."""
        if limit < 0:
            raise ValueError("limit must be non-negative")
        with self._connect() as db:
            rows = db.execute("""SELECT payload FROM (
                SELECT payload, generation, ROW_NUMBER() OVER(PARTITION BY signature_key ORDER BY generation DESC, created_at DESC, failure_id) AS rank
                FROM failures
            ) WHERE rank = 1 ORDER BY generation DESC LIMIT ?""", (limit,)).fetchall()
        return tuple(_failure_from_dict(json.loads(row["payload"])) for row in rows)

    def recurrence_count(self, signature_key: str) -> int:
        with self._connect() as db:
            row = db.execute("SELECT COUNT(*) AS n FROM failures WHERE signature_key=?", (signature_key,)).fetchone()
            return int(row["n"])


def _failure_to_dict(failure: FailureRecord) -> dict:
    return {
        "failure_id": failure.failure_id, "episode_id": failure.episode_id,
        "task_id": failure.task_id, "generation": failure.generation,
        "customer_strategy_id": failure.customer_strategy_id,
        "service_strategy_id": failure.service_strategy_id,
        "signature": failure.signature.to_dict(), "policy_ref": failure.policy_ref,
        "evidence": [ref.__dict__ if hasattr(ref, "__dict__") else {"turn_index": ref.turn_index, "source": ref.source, "summary": ref.summary} for ref in failure.evidence],
        "verification_ref": failure.verification_ref, "severity": failure.severity,
    }


def _failure_from_dict(value: dict) -> FailureRecord:
    return FailureRecord(
        failure_id=value["failure_id"], episode_id=value["episode_id"], task_id=value["task_id"],
        generation=value["generation"], customer_strategy_id=value["customer_strategy_id"],
        service_strategy_id=value["service_strategy_id"], signature=FailureSignature(**value["signature"]),
        policy_ref=value["policy_ref"], evidence=tuple(EvidenceRef(**ref) for ref in value["evidence"]),
        verification_ref=value["verification_ref"], severity=value["severity"],
    )
