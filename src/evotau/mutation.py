"""Deterministic, one-field customer mutations for the MVP search space."""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

from .manifest import sha256_json
from .records import FailureRecord
from .strategies import CustomerStrategy

MUTATION_OPERATORS = ("disclosure", "request_order", "challenge")


@dataclass(frozen=True, slots=True)
class CustomerCandidate:
    strategy: CustomerStrategy
    parent_id: str
    operator: str
    rationale: str

    @property
    def strategy_id(self) -> str:
        return sha256_json(self.strategy.to_dict())[:16]


def mutate_customer(
    incumbent: CustomerStrategy,
    operator: str,
    *,
    rationale: str = "exploration",
) -> CustomerCandidate:
    """Change exactly one behavior axis; invalid/no-op mutations are rejected."""

    parent_id = sha256_json(incumbent.to_dict())[:16]
    values = incumbent.to_dict()
    if operator == "disclosure":
        values["disclosure"] = (
            "related_on_request"
            if incumbent.disclosure == "minimal_on_request"
            else "minimal_on_request"
        )
    elif operator == "request_order":
        values["request_order"] = (
            "reverse_independent"
            if incumbent.request_order == "scenario_order"
            else "scenario_order"
        )
    elif operator == "challenge":
        if incumbent.challenge_style == "none":
            values["challenge_style"] = "ask_reason"
            values["challenge_budget"] = 1
        elif incumbent.challenge_style == "ask_reason":
            values["challenge_style"] = "rephrase_request"
            values["challenge_budget"] = 1
        else:
            values["challenge_style"] = "none"
            values["challenge_budget"] = 0
    else:
        raise ValueError(f"unknown CustomerStrategy mutation operator: {operator}")
    candidate = CustomerStrategy(**values)
    if candidate == incumbent:
        raise ValueError("mutation produced no behavioral change")
    return CustomerCandidate(candidate, parent_id, operator, rationale)


def propose_customer_candidates(
    incumbent: CustomerStrategy,
    count: int,
    *,
    seed: int,
    recent_failures: Iterable[FailureRecord] = (),
    already_seen: Iterable[str] = (),
) -> tuple[CustomerCandidate, ...]:
    """Choose distinct mutations reproducibly, biasing only exploration order by evidence."""

    if count < 0:
        raise ValueError("candidate count must be non-negative")
    if count > len(MUTATION_OPERATORS):
        raise ValueError(f"candidate count cannot exceed {len(MUTATION_OPERATORS)} distinct MVP operators")
    if count == 0:
        return ()
    failures = tuple(recent_failures)
    # The three mutation axes remain explicit. Failures in one of the scoped
    # pre-write/workflow stages move order and challenge earlier in the queue;
    # failures never alter scenario facts or the set of allowed operators.
    preferred: set[str] = set()
    if failures:
        stages = {failure.signature.workflow_stage for failure in failures}
        if stages & {"pre_write", "information_gathering", "decision"}:
            preferred.add("request_order")
        if any("confirm" in failure.signature.policy_rule_id for failure in failures):
            preferred.add("challenge")
    operators = list(MUTATION_OPERATORS)
    random.Random(seed).shuffle(operators)
    # Preserve seeded exploration within a priority tier while genuinely moving
    # evidence-relevant axes ahead of generic exploration.
    random_rank = {operator: index for index, operator in enumerate(operators)}
    operators.sort(key=lambda operator: (operator not in preferred, random_rank[operator]))
    parent_id = sha256_json(incumbent.to_dict())[:16]
    seen = set(already_seen) | {parent_id}
    candidates: list[CustomerCandidate] = []
    for operator in operators:
        candidate = mutate_customer(
            incumbent,
            operator,
            rationale=("failure_conditioned" if failures else "exploration"),
        )
        if candidate.strategy_id in seen:
            continue
        seen.add(candidate.strategy_id)
        candidates.append(candidate)
        if len(candidates) >= count:
            break
    return tuple(candidates)
