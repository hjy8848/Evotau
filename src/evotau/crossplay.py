"""Balanced cross-play summaries for frozen Customer/Service checkpoints."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from typing import Any

from .records import (
    EpisodeRecord,
    EpisodeStatus,
    FailureRecord,
    customer_strategy_id,
    service_strategy_id,
)
from .strategies import CustomerStrategy, ServiceStrategy


@dataclass(frozen=True, slots=True)
class CrossPlayCell:
    customer_strategy_id: str
    service_strategy_id: str
    attempted_episodes: int
    valid_episodes: int
    invalid_episodes: int
    infrastructure_episodes: int
    uncertain_episodes: int
    strategy_opportunities: int
    strategy_adherent_episodes: int
    strategy_not_applicable_episodes: int
    strategy_adherence_rate: float | None
    successful_episodes: int
    verified_failure_episodes: int
    verified_failure_rate: float | None
    unique_task_signature_failures: int
    unique_signatures: int
    native_success_rate: float | None = None
    policy_violation_episodes: int = 0
    policy_violation_rate: float | None = None
    recurrent_verified_failure_episodes: int = 0
    recurrent_verified_failure_rate: float | None = None
    verified_signature_keys: tuple[str, ...] = ()
    recurrent_signature_keys: tuple[str, ...] = ()
    verified_signature_episode_counts: tuple[tuple[str, int], ...] = ()
    repeated_write_audited_episodes: int = 0
    invalid_repeated_write_calls: int = 0
    repeated_write_audit_coverage: float | None = None


@dataclass(frozen=True, slots=True)
class CrossPlayMatrix:
    """A complete, balanced matrix on one frozen task/seed panel."""

    customer_strategy_ids: tuple[str, ...]
    service_strategy_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    cells: tuple[CrossPlayCell, ...]
    repaired_signature_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, values in (
            ("Customer strategy IDs", self.customer_strategy_ids),
            ("Service strategy IDs", self.service_strategy_ids),
            ("task IDs", self.task_ids),
            ("episode seeds", self.seeds),
        ):
            if not values or len(set(values)) != len(values):
                raise ValueError(f"cross-play {name} must be non-empty and unique")
        if any(not item for item in (*self.customer_strategy_ids, *self.service_strategy_ids,
                                     *self.task_ids)):
            raise ValueError("cross-play strategy IDs and task IDs must be non-empty")
        if any(type(seed) is not int or seed < 0 for seed in self.seeds):
            raise ValueError("cross-play seeds must be non-negative integers")
        expected = {
            (customer_id, service_id)
            for customer_id in self.customer_strategy_ids
            for service_id in self.service_strategy_ids
        }
        actual = {(cell.customer_strategy_id, cell.service_strategy_id) for cell in self.cells}
        if len(actual) != len(self.cells) or actual != expected:
            raise ValueError("cross-play matrix must contain exactly one cell per frozen strategy pair")
        expected_episodes = len(self.task_ids) * len(self.seeds)
        if any(cell.attempted_episodes != expected_episodes for cell in self.cells):
            raise ValueError("cross-play matrix cells must use the same complete task/seed panel")
        if (len(set(self.repaired_signature_keys)) != len(self.repaired_signature_keys)
                or any(not re.fullmatch(r"[0-9a-f]{16}", item)
                       for item in self.repaired_signature_keys)):
            raise ValueError("repaired signature keys must be unique 16-character lowercase hashes")
        if tuple(sorted(self.repaired_signature_keys)) != self.repaired_signature_keys:
            raise ValueError("repaired signature keys must be stored in canonical sorted order")
        repaired = set(self.repaired_signature_keys)
        for cell in self.cells:
            counts = dict(cell.verified_signature_episode_counts)
            if (len(counts) != len(cell.verified_signature_episode_counts)
                    or tuple(sorted(counts)) != cell.verified_signature_keys
                    or any(not re.fullmatch(r"[0-9a-f]{16}", key) for key in counts)
                    or any(type(count) is not int or count <= 0 for count in counts.values())
                    or sum(counts.values()) != cell.verified_failure_episodes
                    or len(counts) != cell.unique_signatures):
                raise ValueError("cross-play signature episode counts do not match verified failure totals")
            recurrent = {key: count for key, count in counts.items() if key in repaired}
            if (tuple(sorted(recurrent)) != cell.recurrent_signature_keys
                    or sum(recurrent.values()) != cell.recurrent_verified_failure_episodes):
                raise ValueError("cross-play recurrence counts do not match the frozen repaired signatures")
            _validate_cell(cell)

    def to_dict(self) -> dict[str, Any]:
        cells = []
        for cell in self.cells:
            value = asdict(cell)
            value["verified_signature_keys"] = list(cell.verified_signature_keys)
            value["recurrent_signature_keys"] = list(cell.recurrent_signature_keys)
            value["verified_signature_episode_counts"] = [
                list(item) for item in cell.verified_signature_episode_counts
            ]
            cells.append(value)
        return {
            "customer_strategy_ids": list(self.customer_strategy_ids),
            "service_strategy_ids": list(self.service_strategy_ids),
            "task_ids": list(self.task_ids),
            "seeds": list(self.seeds),
            "cells": cells,
            "repaired_signature_keys": list(self.repaired_signature_keys),
        }

    @classmethod
    def from_dict(cls, value: Any) -> CrossPlayMatrix:
        """Load and validate a complete serialized cross-play matrix."""

        required = {
            "customer_strategy_ids", "service_strategy_ids", "task_ids", "seeds",
            "cells", "repaired_signature_keys",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("cross-play matrix has missing or unknown fields")
        array_fields = (
            "customer_strategy_ids", "service_strategy_ids", "task_ids", "seeds",
            "cells", "repaired_signature_keys",
        )
        if any(not isinstance(value[name], list) for name in array_fields):
            raise TypeError("cross-play matrix fields must be JSON arrays")
        cell_fields = {item.name for item in fields(CrossPlayCell)}
        cells: list[CrossPlayCell] = []
        for index, row in enumerate(value["cells"]):
            if not isinstance(row, dict) or set(row) != cell_fields:
                raise ValueError(f"cross-play cell {index} has missing or unknown fields")
            for name in (
                "verified_signature_keys", "recurrent_signature_keys",
                "verified_signature_episode_counts",
            ):
                if not isinstance(row[name], list):
                    raise TypeError(f"cross-play cell {index} {name} must be a JSON array")
            try:
                cells.append(CrossPlayCell(
                    **{
                        **row,
                        "verified_signature_keys": tuple(row["verified_signature_keys"]),
                        "recurrent_signature_keys": tuple(row["recurrent_signature_keys"]),
                        "verified_signature_episode_counts": tuple(
                            tuple(item) for item in row["verified_signature_episode_counts"]
                        ),
                    }
                ))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid cross-play cell {index}: {exc}") from exc
        try:
            return cls(
                customer_strategy_ids=tuple(value["customer_strategy_ids"]),
                service_strategy_ids=tuple(value["service_strategy_ids"]),
                task_ids=tuple(value["task_ids"]),
                seeds=tuple(value["seeds"]),
                cells=tuple(cells),
                repaired_signature_keys=tuple(value["repaired_signature_keys"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid cross-play matrix: {exc}") from exc


def build_crossplay_matrix(
    episodes: Sequence[EpisodeRecord],
    verified_failures: Sequence[FailureRecord],
    *,
    customer_strategies: Sequence[CustomerStrategy | None],
    service_strategies: Sequence[ServiceStrategy],
    task_ids: Sequence[str],
    seeds: Sequence[int],
    repaired_signature_keys: Sequence[str] = (),
) -> CrossPlayMatrix:
    """Summarize saved paired runs without promoting unverified review signals.

    Every strategy pair must contain exactly one episode for every task/seed
    combination. Only verified failures linked to complete, valid, adherent
    episodes contribute to the failure rate.
    """

    customer_ids = tuple(customer_strategy_id(item) for item in customer_strategies)
    service_ids = tuple(service_strategy_id(item) for item in service_strategies)
    tasks, seed_values = tuple(str(item) for item in task_ids), tuple(seeds)
    if not customer_ids or not service_ids or not tasks or not seed_values:
        raise ValueError("cross-play requires strategies and a non-empty task/seed panel")
    if (len(set(customer_ids)) != len(customer_ids)
            or len(set(service_ids)) != len(service_ids)
            or len(set(tasks)) != len(tasks)
            or len(set(seed_values)) != len(seed_values)):
        raise ValueError("cross-play strategy IDs, tasks, and seeds must be unique")
    if any(seed < 0 for seed in seed_values):
        raise ValueError("cross-play seeds must be non-negative")
    repaired_keys = tuple(sorted(repaired_signature_keys))
    if (len(set(repaired_keys)) != len(repaired_keys)
            or any(not re.fullmatch(r"[0-9a-f]{16}", item) for item in repaired_keys)):
        raise ValueError("repaired signature keys must be unique 16-character lowercase hashes")

    pairs = {(customer_id, service_id) for customer_id in customer_ids for service_id in service_ids}
    expected_panel = {(task_id, seed) for task_id in tasks for seed in seed_values}
    grouped: dict[tuple[str, str], dict[tuple[str, int], EpisodeRecord]] = defaultdict(dict)
    by_episode_id: dict[str, EpisodeRecord] = {}
    for episode in episodes:
        pair = (episode.customer_strategy_id, episode.service_strategy_id)
        if pair not in pairs:
            raise ValueError("episode uses a Customer/Service strategy outside the frozen matrix")
        key = (episode.task_id, episode.seed)
        if key not in expected_panel:
            raise ValueError("episode falls outside the frozen cross-play task/seed panel")
        if key in grouped[pair]:
            raise ValueError("cross-play pair has duplicate episodes for a task/seed unit")
        if episode.episode_id in by_episode_id:
            raise ValueError("cross-play episodes must have unique episode IDs")
        grouped[pair][key] = episode
        by_episode_id[episode.episode_id] = episode
    for pair in pairs:
        if set(grouped[pair]) != expected_panel:
            missing = sorted(expected_panel - set(grouped[pair]))
            raise ValueError(f"cross-play pair is missing task/seed units: {missing}")

    failures_by_episode: dict[str, FailureRecord] = {}
    for failure in verified_failures:
        if failure.episode_id in failures_by_episode:
            raise ValueError("cross-play permits only the earliest verified failure per episode")
        episode = by_episode_id.get(failure.episode_id)
        if episode is None:
            raise ValueError("verified failure refers to an episode outside the cross-play matrix")
        if (
            not episode.has_attributable_failure_candidate
            or failure.task_id != episode.task_id
            or failure.customer_strategy_id != episode.customer_strategy_id
            or failure.service_strategy_id != episode.service_strategy_id
            or failure.policy_ref != episode.policy_rule_id
            or failure.signature.workflow_stage != episode.workflow_stage
            or failure.signature.mistake_type != episode.mistake_type
            or failure.evidence != episode.evidence
        ):
            raise ValueError("verified failure is not supported by its cross-play episode")
        failures_by_episode[failure.episode_id] = failure

    cells: list[CrossPlayCell] = []
    for customer_id in customer_ids:
        for service_id in service_ids:
            selected = tuple(grouped[(customer_id, service_id)].values())
            valid = tuple(
                item for item in selected
                if item.status in {EpisodeStatus.COMPLETE, EpisodeStatus.INVALID_STRATEGY}
                and item.customer_valid is True
            )
            invalid = tuple(
                item for item in selected
                if item.status == EpisodeStatus.INVALID_CUSTOMER or item.customer_valid is False
            )
            opportunities = tuple(item for item in valid if item.strategy_opportunity)
            adherent = tuple(item for item in opportunities if item.customer_strategy_adherent is True)
            not_applicable = tuple(item for item in valid if item.strategy_not_applicable)
            infrastructure = tuple(
                item for item in selected if item.status == EpisodeStatus.INFRASTRUCTURE_ERROR
            )
            classified = {item.episode_id for item in (*valid, *invalid, *infrastructure)}
            uncertain_count = len(selected) - len(classified)
            failure_ids = {
                item.episode_id for item in adherent if item.episode_id in failures_by_episode
            }
            unique_task_signature = {
                (failures_by_episode[episode_id].task_id, failures_by_episode[episode_id].signature.key)
                for episode_id in failure_ids
            }
            verified_signature_keys = tuple(sorted({
                failures_by_episode[episode_id].signature.key for episode_id in failure_ids
            }))
            signature_episode_counts = Counter(
                failures_by_episode[episode_id].signature.key for episode_id in failure_ids
            )
            recurrent_failure_ids = {
                episode_id for episode_id in failure_ids
                if failures_by_episode[episode_id].signature.key in repaired_keys
            }
            recurrent_signature_keys = tuple(sorted({
                failures_by_episode[episode_id].signature.key for episode_id in recurrent_failure_ids
            }))
            policy_violations = sum(item.policy_violation for item in valid)
            successful_episodes = sum(item.task_success is True for item in valid)
            repeated_write_audited = tuple(
                item for item in valid if item.invalid_repeated_write_calls is not None
            )
            repeated_write_count = sum(
                item.invalid_repeated_write_calls or 0 for item in repeated_write_audited
            )
            cells.append(CrossPlayCell(
                customer_strategy_id=customer_id,
                service_strategy_id=service_id,
                attempted_episodes=len(selected),
                valid_episodes=len(valid),
                invalid_episodes=len(invalid),
                infrastructure_episodes=len(infrastructure),
                uncertain_episodes=uncertain_count,
                strategy_opportunities=len(opportunities),
                strategy_adherent_episodes=len(adherent),
                strategy_not_applicable_episodes=len(not_applicable),
                strategy_adherence_rate=(len(adherent) / len(opportunities) if opportunities else None),
                successful_episodes=successful_episodes,
                verified_failure_episodes=len(failure_ids),
                verified_failure_rate=(len(failure_ids) / len(adherent) if adherent else None),
                unique_task_signature_failures=len(unique_task_signature),
                unique_signatures=len({
                    failures_by_episode[episode_id].signature.key for episode_id in failure_ids
                }),
                native_success_rate=(successful_episodes / len(valid) if valid else None),
                policy_violation_episodes=policy_violations,
                policy_violation_rate=(policy_violations / len(valid) if valid else None),
                recurrent_verified_failure_episodes=len(recurrent_failure_ids),
                recurrent_verified_failure_rate=(
                    len(recurrent_failure_ids) / len(adherent) if adherent else None
                ),
                verified_signature_keys=verified_signature_keys,
                recurrent_signature_keys=recurrent_signature_keys,
                verified_signature_episode_counts=tuple(sorted(signature_episode_counts.items())),
                repeated_write_audited_episodes=len(repeated_write_audited),
                invalid_repeated_write_calls=repeated_write_count,
                repeated_write_audit_coverage=(
                    len(repeated_write_audited) / len(valid) if valid else None
                ),
            ))
    return CrossPlayMatrix(
        customer_ids, service_ids, tasks, seed_values, tuple(cells), repaired_keys,
    )


def _validate_cell(cell: CrossPlayCell) -> None:
    count_fields = (
        "attempted_episodes", "valid_episodes", "invalid_episodes", "infrastructure_episodes",
        "uncertain_episodes", "strategy_opportunities", "strategy_adherent_episodes",
        "strategy_not_applicable_episodes", "successful_episodes", "verified_failure_episodes",
        "unique_task_signature_failures", "unique_signatures", "policy_violation_episodes",
        "recurrent_verified_failure_episodes",
        "repeated_write_audited_episodes", "invalid_repeated_write_calls",
    )
    if any(type(getattr(cell, field)) is not int or getattr(cell, field) < 0 for field in count_fields):
        raise ValueError("cross-play cell counts must be non-negative integers")
    if (cell.strategy_opportunities > cell.valid_episodes
            or cell.strategy_adherent_episodes > cell.strategy_opportunities
            or cell.strategy_not_applicable_episodes > cell.valid_episodes
            or cell.strategy_adherent_episodes + cell.strategy_not_applicable_episodes > cell.valid_episodes
            or cell.successful_episodes > cell.valid_episodes
            or cell.policy_violation_episodes > cell.valid_episodes
            or cell.repeated_write_audited_episodes > cell.valid_episodes
            or cell.verified_failure_episodes > cell.strategy_adherent_episodes
            or cell.recurrent_verified_failure_episodes > cell.verified_failure_episodes
            or cell.unique_task_signature_failures > cell.verified_failure_episodes):
        raise ValueError("cross-play cell counts exceed their valid episode or adherence denominators")
    _require_rate(cell.strategy_adherence_rate, cell.strategy_adherent_episodes,
                  cell.strategy_opportunities, "strategy adherence")
    _require_rate(cell.verified_failure_rate, cell.verified_failure_episodes,
                  cell.strategy_adherent_episodes, "attributable failure")
    _require_rate(cell.native_success_rate, cell.successful_episodes,
                  cell.valid_episodes, "native task success")
    _require_rate(cell.policy_violation_rate, cell.policy_violation_episodes,
                  cell.valid_episodes, "policy violation")
    _require_rate(cell.recurrent_verified_failure_rate,
                  cell.recurrent_verified_failure_episodes,
                  cell.strategy_adherent_episodes, "historical recurrence")
    _require_rate(cell.repeated_write_audit_coverage,
                  cell.repeated_write_audited_episodes,
                  cell.valid_episodes, "repeated-write audit coverage")


def _require_rate(rate: float | None, numerator: int, denominator: int, label: str) -> None:
    expected = numerator / denominator if denominator else None
    if expected is None:
        if rate is not None:
            raise ValueError(f"cross-play {label} rate requires a non-empty denominator")
    elif (rate is None or not math.isfinite(rate) or not 0 <= rate <= 1
          or not math.isclose(rate, expected, rel_tol=0, abs_tol=1e-12)):
        raise ValueError(f"cross-play {label} rate does not match its numerator and denominator")
