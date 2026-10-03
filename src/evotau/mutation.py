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
    changed_fields: tuple[str, ...]
    expected_behavioral_effect: str
    supporting_failure_ids: tuple[str, ...] = ()
    proposal_context_sha256: str | None = None

    @property
    def strategy_id(self) -> str:
        return sha256_json(self.strategy.to_dict())[:16]


def mutate_customer(
    incumbent: CustomerStrategy,
    operator: str,
    *,
    rationale: str = "exploration",
    supporting_failure_ids: tuple[str, ...] = (),
) -> CustomerCandidate:
    """Change exactly one behavior axis; invalid/no-op mutations are rejected."""

    if incumbent.is_v2:
        raise ValueError("legacy mutation operators cannot mutate v2 Customer strategies")
    parent_id = sha256_json(incumbent.to_dict())[:16]
    values = incumbent.to_dict()
    if operator == "disclosure":
        values["disclosure"] = (
            "related_on_request"
            if incumbent.disclosure == "minimal_on_request"
            else "minimal_on_request"
        )
        changed_fields = ("disclosure",)
        expected_effect = "Test a different progressive disclosure policy while preserving all fixed scenario facts."
    elif operator == "request_order":
        values["request_order"] = (
            "reverse_independent"
            if incumbent.request_order == "scenario_order"
            else "scenario_order"
        )
        changed_fields = ("request_order",)
        expected_effect = "Test a different order for independent subrequests while preserving their dependencies."
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
        changed_fields = ("challenge_style", "challenge_budget")
        expected_effect = "Test a different bounded response to refusal, verification, or a raised policy limit."
    else:
        raise ValueError(f"unknown CustomerStrategy mutation operator: {operator}")
    candidate = CustomerStrategy(**values)
    if candidate == incumbent:
        raise ValueError("mutation produced no behavioral change")
    return CustomerCandidate(
        candidate, parent_id, operator, rationale, changed_fields, expected_effect,
        supporting_failure_ids,
    )


def propose_customer_candidates(
    incumbent: CustomerStrategy,
    count: int,
    *,
    seed: int,
    recent_failures: Iterable[FailureRecord] = (),
    already_seen: Iterable[str] = (),
) -> tuple[CustomerCandidate, ...]:
    """Choose distinct mutations reproducibly, biasing only exploration order by evidence."""

    if incumbent.is_v2:
        raise ValueError("legacy mutation operators cannot propose v2 Customer strategies")
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
        supporting = _supporting_failures(operator, failures)
        candidate = mutate_customer(
            incumbent,
            operator,
            rationale=("failure_conditioned" if supporting else "exploration"),
            supporting_failure_ids=tuple(item.failure_id for item in supporting),
        )
        if candidate.strategy_id in seen:
            continue
        seen.add(candidate.strategy_id)
        candidates.append(candidate)
        if len(candidates) >= count:
            break
    return tuple(candidates)


def _supporting_failures(
    operator: str,
    failures: tuple[FailureRecord, ...],
) -> tuple[FailureRecord, ...]:
    if operator == "request_order":
        return tuple(
            failure for failure in failures
            if failure.signature.workflow_stage in {"pre_write", "information_gathering", "decision"}
        )
    if operator == "challenge":
        return tuple(
            failure for failure in failures
            if "confirm" in failure.signature.policy_rule_id.lower()
            or "verification" in failure.signature.policy_rule_id.lower()
        )
    return ()
