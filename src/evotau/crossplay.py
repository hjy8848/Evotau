"""Balanced cross-play summaries for frozen Customer/Service checkpoints."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
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
    successful_episodes: int
    verified_failure_episodes: int
    verified_failure_rate: float | None
    unique_task_signature_failures: int
    unique_signatures: int


@dataclass(frozen=True, slots=True)
class CrossPlayMatrix:
    """A complete, balanced matrix on one frozen task/seed panel."""

    customer_strategy_ids: tuple[str, ...]
    service_strategy_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    cells: tuple[CrossPlayCell, ...]

    def __post_init__(self) -> None:
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "customer_strategy_ids": list(self.customer_strategy_ids),
            "service_strategy_ids": list(self.service_strategy_ids),
            "task_ids": list(self.task_ids),
            "seeds": list(self.seeds),
            "cells": [asdict(cell) for cell in self.cells],
        }


def build_crossplay_matrix(
    episodes: Sequence[EpisodeRecord],
    verified_failures: Sequence[FailureRecord],
    *,
    customer_strategies: Sequence[CustomerStrategy],
    service_strategies: Sequence[ServiceStrategy],
    task_ids: Sequence[str],
    seeds: Sequence[int],
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
                if item.status == EpisodeStatus.COMPLETE
                and item.customer_valid is True
                and item.customer_strategy_adherent is True
            )
            invalid = tuple(
                item for item in selected
                if item.status in {EpisodeStatus.INVALID_CUSTOMER, EpisodeStatus.INVALID_STRATEGY}
                or item.customer_valid is False
                or item.customer_strategy_adherent is False
            )
            infrastructure = tuple(
                item for item in selected if item.status == EpisodeStatus.INFRASTRUCTURE_ERROR
            )
            classified = {item.episode_id for item in (*valid, *invalid, *infrastructure)}
            uncertain_count = len(selected) - len(classified)
            failure_ids = {
                item.episode_id for item in valid if item.episode_id in failures_by_episode
            }
            unique_task_signature = {
                (failures_by_episode[episode_id].task_id, failures_by_episode[episode_id].signature.key)
                for episode_id in failure_ids
            }
            cells.append(CrossPlayCell(
                customer_strategy_id=customer_id,
                service_strategy_id=service_id,
                attempted_episodes=len(selected),
                valid_episodes=len(valid),
                invalid_episodes=len(invalid),
                infrastructure_episodes=len(infrastructure),
                uncertain_episodes=uncertain_count,
                successful_episodes=sum(item.task_success is True for item in valid),
                verified_failure_episodes=len(failure_ids),
                verified_failure_rate=(len(failure_ids) / len(valid) if valid else None),
                unique_task_signature_failures=len(unique_task_signature),
                unique_signatures=len({
                    failures_by_episode[episode_id].signature.key for episode_id in failure_ids
                }),
            ))
    return CrossPlayMatrix(customer_ids, service_ids, tasks, seed_values, tuple(cells))
