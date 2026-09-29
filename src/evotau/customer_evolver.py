"""Bounded LLM ranking of legal Customer mutation operators."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .manifest import sha256_json
from .mutation import (
    MUTATION_OPERATORS,
    CustomerCandidate,
    propose_customer_candidates,
)
from .records import FailureRecord, FailureSignature, is_mvp_failure_signature
from .strategies import CustomerStrategy


@dataclass(frozen=True, slots=True)
class EvolutionFailureSignal:
    """Only the non-identifying, policy-taxonomy portion of an E failure."""

    failure_id: str
    workflow_stage: str
    policy_rule_id: str
    mistake_type: str

    def __post_init__(self) -> None:
        if not self.failure_id:
            raise ValueError("evolution feedback requires a failure reference")
        if not is_mvp_failure_signature(FailureSignature(
            "retail", self.workflow_stage, self.policy_rule_id, self.mistake_type,
        )):
            raise ValueError("evolution feedback must use a frozen MVP failure signature")

    def to_dict(self) -> dict[str, str]:
        return {
            "failure_id": self.failure_id,
            "workflow_stage": self.workflow_stage,
            "policy_rule_id": self.policy_rule_id,
            "mistake_type": self.mistake_type,
        }


@dataclass(frozen=True, slots=True)
class CustomerEvolutionProposalInput:
    generation: int
    incumbent: CustomerStrategy
    allowed_operators: tuple[str, ...]
    candidate_count: int
    proposal_seed: int
    failure_signals: tuple[EvolutionFailureSignal, ...]

    def __post_init__(self) -> None:
        if type(self.generation) is not int or not 0 <= self.generation < 8 or self.proposal_seed < 0:
            raise ValueError("Customer Evolver input has an invalid generation or seed")
        if (len(set(self.allowed_operators)) != len(self.allowed_operators)
                or not set(self.allowed_operators) <= set(MUTATION_OPERATORS)):
            raise ValueError("Customer Evolver input must contain unique legal mutation operators")
        if self.candidate_count != min(self.candidate_count, len(self.allowed_operators)):
            raise ValueError("Customer Evolver candidate count exceeds its allowed operator set")
        if self.candidate_count < 0:
            raise ValueError("Customer Evolver candidate count cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "generation": self.generation,
            "incumbent": self.incumbent.to_dict(),
            "allowed_operators": list(self.allowed_operators),
            "candidate_count": self.candidate_count,
            "proposal_seed": self.proposal_seed,
            "failure_signals": [item.to_dict() for item in self.failure_signals],
        }


OperatorSelector = Callable[[CustomerEvolutionProposalInput], Sequence[str]]


def propose_customer_candidates_with_selector(
    incumbent: CustomerStrategy,
    count: int,
    *,
    generation: int,
    seed: int,
    recent_failures: Sequence[FailureRecord],
    already_seen: Sequence[str],
    evolution_task_ids: Sequence[str],
    operator_selector: OperatorSelector,
) -> tuple[CustomerCandidate, ...]:
    """Let a bounded selector rank legal mutations using E-only failure signatures."""

    if not callable(operator_selector):
        raise TypeError("Customer Evolver requires an operator selector")
    if count < 0 or count > len(MUTATION_OPERATORS):
        raise ValueError(f"candidate count must be between 0 and {len(MUTATION_OPERATORS)}")
    task_ids = set(evolution_task_ids)
    if not task_ids:
        raise ValueError("Customer Evolver requires the frozen evolution task set")
    if any(failure.task_id not in task_ids for failure in recent_failures):
        raise ValueError("Customer Evolver may receive only failures from its frozen E tasks")

    pool = propose_customer_candidates(
        incumbent,
        len(MUTATION_OPERATORS),
        seed=seed,
        recent_failures=recent_failures,
        already_seen=already_seen,
    )
    target_count = min(count, len(pool))
    if target_count == 0:
        return ()
    signals = tuple(
        EvolutionFailureSignal(
            failure_id=item.failure_id,
            workflow_stage=item.signature.workflow_stage,
            policy_rule_id=item.signature.policy_rule_id,
            mistake_type=item.signature.mistake_type,
        )
        for item in sorted(recent_failures, key=lambda item: item.failure_id)
    )
    context = CustomerEvolutionProposalInput(
        generation=generation,
        incumbent=incumbent,
        allowed_operators=tuple(item.operator for item in pool),
        candidate_count=target_count,
        proposal_seed=seed,
        failure_signals=signals,
    )
    selected = tuple(operator_selector(context))
    if (len(selected) != target_count or len(set(selected)) != len(selected)
            or any(operator not in context.allowed_operators for operator in selected)):
        raise ValueError("Customer Evolver response must rank exactly the allowed candidate count")
    by_operator = {item.operator: item for item in pool}
    context_hash = sha256_json(context.to_dict())
    return tuple(
        replace(by_operator[operator], proposal_context_sha256=context_hash)
        for operator in selected
    )


@dataclass(frozen=True, slots=True)
class LLMCustomerEvolver:
    """Rank existing legal operators; the LLM cannot invent Customer fields."""

    model: str
    model_args: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Customer Evolver requires a frozen model ID")
        if "temperature" not in self.model_args:
            raise ValueError("Customer Evolver model arguments must freeze temperature")

    def __call__(self, context: CustomerEvolutionProposalInput) -> tuple[str, ...]:
        from tau2.data_model.message import SystemMessage, UserMessage
        from tau2.utils.llm_utils import generate

        system = (
            "You rank a small set of preconstructed Customer mutation operators for a fixed-fact "
            "benchmark. You do not create strategy text or facts. Use only the supplied legal "
            "operator names, rank each at most once, and return only JSON with an `operators` array. "
            "The failure signals are from the evolution panel only. They contain no customer, order, "
            "trajectory, reference-answer, validation, or heldout data. Prefer an operator only when "
            "the supplied verified policy signature is relevant; otherwise preserve seeded exploration."
        )
        message = generate(
            model=self.model,
            messages=[
                SystemMessage(role="system", content=system),
                UserMessage(
                    role="user",
                    content=json.dumps(context.to_dict(), ensure_ascii=False, sort_keys=True),
                ),
            ],
            call_name="evotau_customer_evolver",
            num_retries=0,
            **dict(self.model_args),
        )
        try:
            payload = json.loads(message.content or "")
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Customer Evolver returned invalid JSON") from exc
        if not isinstance(payload, dict) or set(payload) != {"operators"}:
            raise ValueError("Customer Evolver response must contain only the operators field")
        operators = payload["operators"]
        if (not isinstance(operators, list)
                or any(not isinstance(item, str) for item in operators)):
            raise ValueError("Customer Evolver operators must be a JSON string array")
        return tuple(operators)
