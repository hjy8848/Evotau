"""Frozen Customer-control schedules for budget-matched RQ1 studies."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .manifest import sha256_json
from .mutation import CustomerCandidate, propose_customer_candidates
from .records import EpisodeRecord, customer_strategy_id
from .strategies import CustomerStrategy

_DISCLOSURES = {"minimal_on_request", "related_on_request"}
_REQUEST_ORDERS = {"scenario_order", "reverse_independent"}
_CHALLENGE_STYLES = {"none", "ask_reason", "rephrase_request"}
_CHALLENGE_BUDGETS = {0, 1, 2}


@dataclass(frozen=True, slots=True)
class StaticCustomerPortfolio:
    """Researcher-frozen static strategies covering each MVP field level.

    Portfolio composition is supplied by the study plan and fingerprinted; the
    library does not silently choose a weak hand-authored control.
    """

    portfolio_id: str
    strategies: tuple[CustomerStrategy, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.portfolio_id, str) or not self.portfolio_id.strip():
            raise ValueError("static Customer portfolio requires a non-empty ID")
        if not self.strategies or any(
            not isinstance(item, CustomerStrategy) for item in self.strategies
        ):
            raise ValueError("static Customer portfolio requires CustomerStrategy values")
        if len(self.strategies) > 20:
            raise ValueError("static portfolio exceeds the 20-strategy MVP search space")
        strategy_ids = [customer_strategy_id(item) for item in self.strategies]
        if len(strategy_ids) != len(set(strategy_ids)):
            raise ValueError("static Customer portfolio repeats a strategy")
        coverage = {
            "disclosure": {item.disclosure for item in self.strategies},
            "request_order": {item.request_order for item in self.strategies},
            "challenge_style": {item.challenge_style for item in self.strategies},
            "challenge_budget": {item.challenge_budget for item in self.strategies},
        }
        required = {
            "disclosure": _DISCLOSURES,
            "request_order": _REQUEST_ORDERS,
            "challenge_style": _CHALLENGE_STYLES,
            "challenge_budget": _CHALLENGE_BUDGETS,
        }
        missing = {
            field: sorted(levels - coverage[field])
            for field, levels in required.items()
            if levels - coverage[field]
        }
        if missing:
            raise ValueError(f"static Customer portfolio does not cover all field levels: {missing}")

    @property
    def strategy_ids(self) -> tuple[str, ...]:
        return tuple(customer_strategy_id(item) for item in self.strategies)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "portfolio_id": self.portfolio_id,
            "strategies": [item.to_dict() for item in self.strategies],
        }

    @classmethod
    def from_dict(cls, value: Any) -> StaticCustomerPortfolio:
        if not isinstance(value, dict) or set(value) != {
            "schema_version", "portfolio_id", "strategies",
        }:
            raise ValueError("static Customer portfolio has missing or unknown fields")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("unsupported static Customer portfolio schema_version")
        if not isinstance(value["strategies"], list):
            raise TypeError("static Customer strategies must be a JSON array")
        expected_strategy_fields = {
            "disclosure", "request_order", "challenge_style", "challenge_budget",
        }
        if any(
            not isinstance(item, dict) or set(item) != expected_strategy_fields
            for item in value["strategies"]
        ):
            raise ValueError("static Customer strategy has missing or unknown fields")
        try:
            return cls(
                portfolio_id=value["portfolio_id"],
                strategies=tuple(CustomerStrategy(**item) for item in value["strategies"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid static Customer portfolio: {exc}") from exc

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_dict())

    def schedule(
        self,
        *,
        task_ids: tuple[str, ...],
        seeds: tuple[int, ...],
    ) -> StaticCustomerSchedule:
        if not task_ids or len(set(task_ids)) != len(task_ids) or any(
            not isinstance(task_id, str) or not task_id.strip() for task_id in task_ids
        ):
            raise ValueError("static schedule requires unique ordered task IDs")
        if not seeds or len(set(seeds)) != len(seeds) or any(
            type(seed) is not int or seed < 0 for seed in seeds
        ):
            raise ValueError("static schedule requires unique non-negative episode seeds")
        pairs = tuple((task_id, seed) for task_id in task_ids for seed in seeds)
        if len(pairs) < len(self.strategies):
            raise ValueError("static schedule must execute every frozen portfolio strategy at least once")
        assignments = tuple(
            (task_id, seed, self.strategy_ids[index % len(self.strategies)])
            for index, (task_id, seed) in enumerate(pairs)
        )
        return StaticCustomerSchedule(
            portfolio_id=self.portfolio_id,
            portfolio_sha256=self.sha256,
            strategy_ids=self.strategy_ids,
            task_ids=task_ids,
            seeds=seeds,
            assignments=assignments,
        )

    def panel_schedule(
        self,
        *,
        task_ids: tuple[str, ...],
        seeds: tuple[int, ...],
        repeats_per_pair: int,
    ) -> StaticCustomerPanelSchedule:
        """Freeze balanced static assignments for repeated matched-panel slots."""

        return StaticCustomerPanelSchedule.create(
            portfolio=self, task_ids=task_ids, seeds=seeds,
            repeats_per_pair=repeats_per_pair,
        )


@dataclass(frozen=True, slots=True)
class StaticCustomerSchedule:
    """Immutable, balanced assignment of a frozen portfolio to task/seed pairs."""

    portfolio_id: str
    portfolio_sha256: str
    strategy_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    assignments: tuple[tuple[str, int, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.portfolio_id, str) or not self.portfolio_id.strip():
            raise ValueError("static schedule requires a portfolio ID")
        if not isinstance(self.portfolio_sha256, str) or len(self.portfolio_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.portfolio_sha256
        ):
            raise ValueError("static schedule portfolio hash must be a lowercase SHA-256")
        if (not self.strategy_ids or len(set(self.strategy_ids)) != len(self.strategy_ids)
                or any(len(item) != 16 or any(char not in "0123456789abcdef" for char in item)
                       for item in self.strategy_ids)):
            raise ValueError("static schedule requires unique strategy IDs")
        if (not self.task_ids or len(set(self.task_ids)) != len(self.task_ids)
                or any(not isinstance(item, str) or not item.strip() for item in self.task_ids)):
            raise ValueError("static schedule requires unique ordered task IDs")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or any(
            type(seed) is not int or seed < 0 for seed in self.seeds
        ):
            raise ValueError("static schedule requires unique non-negative episode seeds")
        expected_pairs = tuple((task_id, seed) for task_id in self.task_ids for seed in self.seeds)
        if len(expected_pairs) < len(self.strategy_ids):
            raise ValueError("static schedule must execute every frozen portfolio strategy at least once")
        actual_pairs = tuple((task_id, seed) for task_id, seed, _strategy_id in self.assignments)
        if not expected_pairs or actual_pairs != expected_pairs:
            raise ValueError("static schedule must cover the ordered task/seed Cartesian product")
        expected = tuple(
            (task_id, seed, self.strategy_ids[index % len(self.strategy_ids)])
            for index, (task_id, seed) in enumerate(expected_pairs)
        )
        if self.assignments != expected:
            raise ValueError("static schedule deviates from its frozen balanced rotation")
        if len(self.assignments) != len(expected_pairs):
            raise ValueError("static schedule repeats a task/seed pair")

    def strategy_for(self, task_id: str, seed: int) -> str:
        for scheduled_task, scheduled_seed, strategy_id in self.assignments:
            if (scheduled_task, scheduled_seed) == (task_id, seed):
                return strategy_id
        raise KeyError(f"task/seed pair {(task_id, seed)!r} is outside the frozen schedule")

    def validate_portfolio(self, portfolio: StaticCustomerPortfolio) -> None:
        if (
            portfolio.portfolio_id != self.portfolio_id
            or portfolio.sha256 != self.portfolio_sha256
            or portfolio.strategy_ids != self.strategy_ids
        ):
            raise ValueError("static schedule does not match its frozen Customer portfolio")

    def validate_records(self, episodes: tuple[EpisodeRecord, ...]) -> None:
        """Reject omissions, extra episodes, or deviations from the static assignment."""

        observed = tuple((item.task_id, item.seed) for item in episodes)
        expected = tuple((task_id, seed) for task_id, seed, _ in self.assignments)
        if observed != expected:
            raise ValueError("static baseline episodes do not match the frozen ordered task/seed schedule")
        for episode in episodes:
            if episode.customer_strategy_id != self.strategy_for(episode.task_id, episode.seed):
                raise ValueError("static baseline episode used a strategy outside its frozen assignment")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "portfolio_id": self.portfolio_id,
            "portfolio_sha256": self.portfolio_sha256,
            "strategy_ids": list(self.strategy_ids),
            "task_ids": list(self.task_ids),
            "seeds": list(self.seeds),
            "assignments": [
                {"task_id": task_id, "seed": seed, "strategy_id": strategy_id}
                for task_id, seed, strategy_id in self.assignments
            ],
        }

    @classmethod
    def from_dict(cls, value: Any) -> StaticCustomerSchedule:
        fields = {
            "schema_version", "portfolio_id", "portfolio_sha256", "strategy_ids",
            "task_ids", "seeds", "assignments",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("static Customer schedule has missing or unknown fields")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("unsupported static Customer schedule schema_version")
        if any(not isinstance(value[name], list) for name in ("strategy_ids", "task_ids", "seeds", "assignments")):
            raise TypeError("static Customer schedule fields must be JSON arrays")
        if any(not isinstance(item, dict) or set(item) != {"task_id", "seed", "strategy_id"}
               for item in value["assignments"]):
            raise ValueError("static Customer schedule assignment is malformed")
        try:
            return cls(
                portfolio_id=value["portfolio_id"],
                portfolio_sha256=value["portfolio_sha256"],
                strategy_ids=tuple(value["strategy_ids"]),
                task_ids=tuple(value["task_ids"]),
                seeds=tuple(value["seeds"]),
                assignments=tuple(
                    (item["task_id"], item["seed"], item["strategy_id"])
                    for item in value["assignments"]
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid static Customer schedule: {exc}") from exc

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class StaticCustomerPanelSchedule:
    """Balanced portfolio assignment for repeated task/seed evaluation slots."""

    portfolio_id: str
    portfolio_sha256: str
    strategy_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    repeats_per_pair: int
    assignments: tuple[tuple[str, int, int, str], ...]

    @classmethod
    def create(
        cls,
        *,
        portfolio: StaticCustomerPortfolio,
        task_ids: tuple[str, ...],
        seeds: tuple[int, ...],
        repeats_per_pair: int,
    ) -> StaticCustomerPanelSchedule:
        if type(repeats_per_pair) is not int or repeats_per_pair <= 0:
            raise ValueError("static panel schedule repeats_per_pair must be positive")
        if (not task_ids or any(not isinstance(item, str) or not item.strip() for item in task_ids)
                or len(set(task_ids)) != len(task_ids)):
            raise ValueError("static panel schedule requires unique ordered task IDs")
        if (not seeds or any(type(seed) is not int or seed < 0 for seed in seeds)
                or len(set(seeds)) != len(seeds)):
            raise ValueError("static panel schedule requires unique non-negative seeds")
        slots = tuple(
            (task_id, seed, repeat)
            for task_id in task_ids for seed in seeds
            for repeat in range(repeats_per_pair)
        )
        if len(slots) < len(portfolio.strategy_ids):
            raise ValueError("static panel schedule must use every frozen portfolio strategy")
        assignments = tuple(
            (*slot, portfolio.strategy_ids[index % len(portfolio.strategy_ids)])
            for index, slot in enumerate(slots)
        )
        return cls(
            portfolio.portfolio_id, portfolio.sha256, portfolio.strategy_ids,
            task_ids, seeds, repeats_per_pair, assignments,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.portfolio_id, str) or not self.portfolio_id.strip():
            raise ValueError("static panel schedule requires a unique frozen portfolio")
        if (not isinstance(self.portfolio_sha256, str) or len(self.portfolio_sha256) != 64
                or any(char not in "0123456789abcdef" for char in self.portfolio_sha256)):
            raise ValueError("static panel schedule portfolio hash must be a lowercase SHA-256")
        if (not self.strategy_ids
                or any(not isinstance(item, str) for item in self.strategy_ids)
                or len(self.strategy_ids) != len(set(self.strategy_ids))
                or any(len(item) != 16 or any(char not in "0123456789abcdef" for char in item)
                       for item in self.strategy_ids)):
            raise ValueError("static panel schedule requires unique strategy IDs")
        if (not self.task_ids
                or any(not isinstance(item, str) for item in self.task_ids)
                or len(set(self.task_ids)) != len(self.task_ids)
                or any(not item.strip() for item in self.task_ids)):
            raise ValueError("static panel schedule requires unique tasks and seeds")
        if (not self.seeds or any(type(seed) is not int or seed < 0 for seed in self.seeds)
                or len(set(self.seeds)) != len(self.seeds)):
            raise ValueError("static panel schedule requires unique tasks and seeds")
        if type(self.repeats_per_pair) is not int or self.repeats_per_pair <= 0:
            raise ValueError("static panel schedule repeats_per_pair must be positive")
        expected_slots = tuple(
            (task_id, seed, repeat)
            for task_id in self.task_ids for seed in self.seeds
            for repeat in range(self.repeats_per_pair)
        )
        if len(expected_slots) < len(self.strategy_ids):
            raise ValueError("static panel schedule omits a frozen portfolio strategy")
        if any(not isinstance(item, tuple) or len(item) != 4 for item in self.assignments):
            raise ValueError("static panel assignments must be task/seed/repeat/strategy tuples")
        actual_slots = tuple((task, seed, repeat) for task, seed, repeat, _ in self.assignments)
        expected = tuple(
            (*slot, self.strategy_ids[index % len(self.strategy_ids)])
            for index, slot in enumerate(expected_slots)
        )
        if actual_slots != expected_slots or self.assignments != expected:
            raise ValueError("static panel assignments differ from their balanced frozen rotation")

    def strategy_for(self, task_id: str, seed: int, repeat: int) -> str:
        for task, episode_seed, slot, strategy_id in self.assignments:
            if (task, episode_seed, slot) == (task_id, seed, repeat):
                return strategy_id
        raise KeyError(f"static schedule has no slot for {(task_id, seed, repeat)!r}")

    def validate_portfolio(self, portfolio: StaticCustomerPortfolio) -> None:
        if (portfolio.portfolio_id != self.portfolio_id
                or portfolio.sha256 != self.portfolio_sha256
                or portfolio.strategy_ids != self.strategy_ids):
            raise ValueError("static panel schedule differs from its frozen portfolio")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "portfolio_id": self.portfolio_id,
            "portfolio_sha256": self.portfolio_sha256,
            "strategy_ids": list(self.strategy_ids),
            "task_ids": list(self.task_ids),
            "seeds": list(self.seeds),
            "repeats_per_pair": self.repeats_per_pair,
            "assignments": [
                {
                    "task_id": task,
                    "seed": seed,
                    "repeat": repeat,
                    "strategy_id": strategy_id,
                }
                for task, seed, repeat, strategy_id in self.assignments
            ],
        }

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_dict())


def propose_random_mutation_candidates(
    incumbent: CustomerStrategy,
    count: int,
    *,
    seed: int,
    already_seen: tuple[str, ...] | frozenset[str] = (),
) -> tuple[CustomerCandidate, ...]:
    """Sample legal mutation operators without passing failure evidence."""

    candidates = propose_customer_candidates(
        incumbent, count, seed=seed, already_seen=already_seen,
    )
    return tuple(replace(item, rationale="random_mutation") for item in candidates)
