"""Small append-only SQLite archive for verified failure signatures."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .manifest import sha256_json
from .records import EvidenceRef, FailureRecord, FailureSignature
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
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""")
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

    def get_customer_strategy(self, strategy_id: str) -> CustomerStrategy | None:
        """Resolve one immutable Customer snapshot by its canonical strategy ID."""
        if not isinstance(strategy_id, str) or not strategy_id.strip():
            raise ValueError("Customer strategy lookup requires a non-empty strategy ID")
        with self._connect() as db:
            row = db.execute(
                "SELECT payload FROM customer_strategies WHERE strategy_id=?",
                (strategy_id,),
            ).fetchone()
        return None if row is None else CustomerStrategy(**json.loads(row["payload"]))

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
        all_representatives = self.representatives(limit=2**31 - 1)
        if current_generation is None:
            current_generation = max((item.generation for item in all_representatives), default=0)
        if current_generation < 0:
            raise ValueError("current generation must be non-negative")
        with self._connect() as db:
            tested = {
                row["signature_key"] for row in db.execute("SELECT DISTINCT signature_key FROM replay_events")
            }
            counts = {
                row["signature_key"]: (int(row["total_count"]), int(row["recent_count"]))
                for row in db.execute(
                    """SELECT signature_key, COUNT(*) AS total_count,
                       SUM(CASE WHEN generation >= ? THEN 1 ELSE 0 END) AS recent_count
                       FROM failures GROUP BY signature_key""",
                    (max(0, current_generation - 1),),
                )
            }
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
                "SELECT signature_key FROM failures WHERE failure_id=?", (failure_id,)
            ).fetchone()
            if row is None:
                raise ValueError("cannot mark an unknown failure as replayed")
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
