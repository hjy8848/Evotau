"""Versioned, direct Customer strategy proposals for activation experiments."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .customer_evolver import EvolutionFailureSignal
from .customer_skill_v3 import _assert_sanitized_context
from .manifest import sha256_json
from .mutation import CustomerCandidate
from .records import FailureRecord, customer_strategy_id
from .strategies import CustomerStrategy

STRATEGY_FIELDS = (
    "disclosure", "request_order", "request_decomposition", "preference_revision",
    "correction_behavior", "challenge_behavior", "challenge_budget",
)
CANDIDATE_FIELDS = frozenset({"strategy", "changed_fields", "hypothesis", "evidence_refs"})
_TASK_REFERENCE = re.compile(r"\btask\s*(?:id\s*)?#?\s*[\w-]*\d[\w-]*\b", re.IGNORECASE)
_EXPLORATION_TERMS = (
    "explor", "test", "probe", "assess", "examin", "evaluat", "investigat",
    "observ", "vary", "try", "探索", "测试", "考察", "评估", "观察", "检验",
)
_GUARANTEED_FAILURE = re.compile(
    r"\b(?:guaranteed?|certain(?:ly)?|will\s+fail|must\s+fail|ensure\s+(?:a\s+)?failure)\b|"
    r"(?:保证失败|必然失败|确保失败|导致失败)",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """You are the Customer Evolver in EvoTau.

Your job is to propose new Customer interaction strategies that can expose weaknesses in a fixed Service agent while preserving the benchmark task's factual truth, business goal, and policy constraints.

You are NOT allowed to invent facts, modify task facts, change the user's actual goal, impersonate another identity, provide knowingly false information, use hidden reference answers, access held-out data, or instruct the Customer to deceive the Service.

You evolve only the Customer's interaction procedure.

## Objective

Generate Customer strategies that create diverse but lawful interaction pressure on the Service. A useful strategy may change when information is disclosed, how requests are decomposed, the order of independent requests, responses to clarification, confirmation timing, truthful correction, a lawful preference before commitment, conditional requests, responses to refusal or verification, and coordination of multiple intents. The factual task itself must remain unchanged. Discover reproducible Service execution failures under valid, fact-preserving Customer behavior; do not make the task fail arbitrarily.

## Strategy schema

Each candidate strategy must contain exactly these fields and legal values:
- disclosure: minimal_on_request | progressive | related_on_request
- request_order: scenario_order | reverse_independent | dependency_first | high_risk_first
- request_decomposition: bundled | one_by_one | dependency_grouped
- preference_revision: fixed | revise_before_commit | narrow_after_options
- correction_behavior: accept_if_correct | correct_once | correct_and_restate_constraint
- challenge_behavior: none | ask_reason | ask_policy_boundary | rephrase_request
- challenge_budget: integer 0, 1, or 2; it must be 0 when challenge_behavior is none.

A preference revision may only change a preference the original task legitimately leaves flexible. It must never contradict a fixed task fact or replace the user's true goal.

## Mutation rules

Return exactly K candidate strategies. Each candidate should normally change 1 or 2 strategy fields relative to the incumbent. Change 3 fields only when supplied verified failure evidence clearly motivates the combination. Do not return the incumbent or any already_tested_strategies. Prefer meaningful behavioral diversity over superficial parameter changes.

Use only supplied abstract verified failure summaries to form hypotheses. Stale state after a preference change may motivate preference_revision or correction_behavior. Premature execution before confirmation may motivate request decomposition, correction, or a bounded challenge. Information-gathering failures may motivate disclosure, decomposition, or ordering. If there are no verified failures, explore novel legal regions. In this no-evidence case, change exactly 1 or 2 fields per candidate, never 3 or more; use empty evidence_refs; and begin each hypothesis with "Exploration:". Count challenge_budget as its own changed field. Before responding, count each candidate's changed fields against the incumbent and verify the list exactly matches. Do not invent failures.

## Hard safety and scientific constraints

Preserve every fixed scenario fact, business objective, identity, and account fact. Do not fabricate identifiers, prices, dates, payment methods, or any other facts. Do not deliberately provide incorrect information, impersonate, manipulate, threaten, or deceive. Do not request an explicitly prohibited action merely to cause failure. Do not use task IDs, reference answers, gold actions, validation tasks, held-out information, or task-specific answers. Every strategy must be reusable across tasks and may make interaction procedurally harder only while remaining factually valid. Optimize for reproducible Service execution weaknesses, not arbitrary task failure.

## Output

Return JSON only, with exactly this structure and no other fields or dialogue:
{"candidates":[{"strategy":{"disclosure":"...","request_order":"...","request_decomposition":"...","preference_revision":"...","correction_behavior":"...","challenge_behavior":"...","challenge_budget":0},"changed_fields":["..."],"hypothesis":"...","evidence_refs":["..."]}]}

The candidates array must have exactly K entries. changed_fields must list exactly the fields changed relative to the incumbent. A hypothesis describes an expected interaction effect, never a guaranteed failure. evidence_refs may contain only supplied verified failure IDs. Without supporting verified evidence, evidence_refs must be empty and the hypothesis must describe exploration. Do not add unsupported fields or natural-language Customer dialogue."""


REFLECTIVE_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "## Hard safety and scientific constraints",
    "The optional sanitized reflection_feedback contains aggregate applicability, adherence, Customer validity, and task-completion counts plus disputed review categories. It is not task truth and never affects fitness. Use it to avoid procedures associated with Customer invalidity, especially unsupported factual detail, without repeating or reconstructing any literal fact.\n\n## Hard safety and scientific constraints",
)


@dataclass(frozen=True, slots=True)
class CustomerStrategyProposalInput:
    generation: int
    incumbent: CustomerStrategy
    candidate_count: int
    proposal_seed: int
    already_tested_strategies: tuple[CustomerStrategy, ...]
    already_tested_strategy_ids: tuple[str, ...]
    failure_signals: tuple[EvolutionFailureSignal, ...]
    reflection_feedback: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.incumbent.is_v2 or any(not item.is_v2 for item in self.already_tested_strategies):
            raise ValueError("direct Customer proposals require v2 strategies")
        if (type(self.generation) is not int or self.generation < 0
                or type(self.candidate_count) is not int or self.candidate_count < 1
                or type(self.proposal_seed) is not int or self.proposal_seed < 0):
            raise ValueError("Customer proposal generation, K, and seed must be valid")
        if len(set(self.already_tested_strategy_ids)) != len(self.already_tested_strategy_ids):
            raise ValueError("already tested Customer strategy IDs must be unique")
        if len({item.failure_id for item in self.failure_signals}) != len(self.failure_signals):
            raise ValueError("verified failure references must be unique")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": 2,
            "generation": self.generation,
            "K": self.candidate_count,
            "proposal_seed": self.proposal_seed,
            "incumbent_strategy": self.incumbent.to_dict(),
            "already_tested_strategies": [item.to_dict() for item in self.already_tested_strategies],
            "already_tested_strategy_ids": list(self.already_tested_strategy_ids),
            "verified_failure_summaries": [item.to_dict() for item in self.failure_signals],
        }
        if self.reflection_feedback is not None:
            _assert_sanitized_context(self.reflection_feedback)
            payload["reflection_feedback"] = dict(self.reflection_feedback)
        return payload


StrategyProposalProvider = Callable[[CustomerStrategyProposalInput], Mapping[str, Any]]


class CustomerEvolverResponseError(ValueError):
    """A malformed provider response, retaining only its content fingerprint."""

    def __init__(self, message: str, *, response_sha256: str):
        super().__init__(message)
        self.response_sha256 = response_sha256


def propose_customer_strategies(
    incumbent: CustomerStrategy,
    count: int,
    *,
    generation: int,
    seed: int,
    recent_failures: Sequence[FailureRecord],
    already_seen: Sequence[str],
    already_tested_strategies: Sequence[CustomerStrategy],
    evolution_task_ids: Sequence[str],
    reflection_feedback: Mapping[str, Any] | None = None,
    proposal_provider: StrategyProposalProvider,
) -> tuple[CustomerCandidate, ...]:
    """Validate every model-supplied field before any candidate can run."""

    if not callable(proposal_provider):
        raise TypeError("Customer strategy proposal provider must be callable")
    if not incumbent.is_v2:
        raise ValueError("direct Customer proposals require a v2 incumbent")
    task_ids = set(evolution_task_ids)
    if recent_failures and (
        not task_ids or any(item.task_id not in task_ids for item in recent_failures)
    ):
        raise ValueError("Customer Evolver may receive only verified failures from frozen E tasks")
    signals = tuple(
        EvolutionFailureSignal(
            failure_id=item.failure_id,
            workflow_stage=item.signature.workflow_stage,
            policy_rule_id=item.signature.policy_rule_id,
            mistake_type=item.signature.mistake_type,
        )
        for item in sorted(recent_failures, key=lambda item: item.failure_id)
    )
    tested = tuple(sorted(
        (item for item in already_tested_strategies if item.is_v2),
        key=customer_strategy_id,
    ))
    context = CustomerStrategyProposalInput(
        generation=generation,
        incumbent=incumbent,
        candidate_count=count,
        proposal_seed=seed,
        already_tested_strategies=tested,
        already_tested_strategy_ids=tuple(sorted(set(already_seen))),
        failure_signals=signals,
        reflection_feedback=reflection_feedback,
    )
    result = proposal_provider(context)
    if not isinstance(result, Mapping) or set(result) != {"candidates"}:
        raise ValueError("Customer Evolver response must contain only candidates")
    rows = result["candidates"]
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("Customer Evolver must return exactly K candidates")
    forbidden_ids = set(already_seen) | {customer_strategy_id(incumbent)}
    forbidden_ids.update(customer_strategy_id(item) for item in tested)
    verified_ids = {item.failure_id for item in signals}
    parent_id = customer_strategy_id(incumbent)
    context_hash = sha256_json(context.to_dict())
    candidates: list[CustomerCandidate] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != CANDIDATE_FIELDS:
            raise ValueError(f"Customer candidate {index} has unsupported or missing fields")
        raw_strategy = row["strategy"]
        if not isinstance(raw_strategy, Mapping) or set(raw_strategy) != set(STRATEGY_FIELDS):
            raise ValueError(f"Customer candidate {index} has an invalid strategy schema")
        strategy = CustomerStrategy(**raw_strategy)
        if not strategy.is_v2:
            raise ValueError("Customer candidate must use the v2 strategy schema")
        strategy_id = customer_strategy_id(strategy)
        if strategy_id in forbidden_ids:
            raise ValueError("Customer Evolver returned an incumbent, tested, or duplicate strategy")
        forbidden_ids.add(strategy_id)
        actual_changes = tuple(
            field for field in STRATEGY_FIELDS
            if raw_strategy[field] != incumbent.to_dict()[field]
        )
        changed = row["changed_fields"]
        if (not isinstance(changed, list) or any(not isinstance(item, str) for item in changed)
                or len(changed) != len(set(changed)) or set(changed) != set(actual_changes)
                or not 1 <= len(actual_changes) <= 3):
            raise ValueError("Customer candidate changed_fields must exactly match 1–3 actual changes")
        refs = row["evidence_refs"]
        if (not isinstance(refs, list) or any(not isinstance(item, str) for item in refs)
                or len(refs) != len(set(refs)) or not set(refs) <= verified_ids):
            raise ValueError("Customer candidate evidence_refs must cite only supplied verified failures")
        if len(actual_changes) == 3 and not refs:
            raise ValueError("three-axis Customer mutation requires verified failure evidence")
        hypothesis = row["hypothesis"]
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise ValueError("Customer candidate hypothesis must describe an interaction effect")
        if _TASK_REFERENCE.search(hypothesis) or _GUARANTEED_FAILURE.search(hypothesis):
            raise ValueError("Customer candidate hypothesis cannot expose a task ID or guarantee failure")
        if not refs and not any(term in hypothesis.casefold() for term in _EXPLORATION_TERMS):
            raise ValueError("unsupported Customer candidate hypothesis must describe exploration")
        candidates.append(CustomerCandidate(
            strategy=strategy,
            parent_id=parent_id,
            operator="strategy_v2",
            rationale="failure_conditioned" if refs else "exploration",
            changed_fields=actual_changes,
            expected_behavioral_effect=hypothesis.strip(),
            supporting_failure_ids=tuple(refs),
            proposal_context_sha256=context_hash,
        ))
    return tuple(candidates)


@dataclass(frozen=True, slots=True)
class LLMCustomerStrategyEvolver:
    """Generate a complete v2 strategy JSON document from sanitized E-only data."""

    model: str
    model_args: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Customer Evolver requires a frozen model ID")
        if "temperature" not in self.model_args:
            raise ValueError("Customer Evolver model arguments must freeze temperature")

    def __call__(self, context: CustomerStrategyProposalInput) -> Mapping[str, Any]:
        from tau2.data_model.message import SystemMessage, UserMessage
        from tau2.utils.llm_utils import generate

        message = generate(
            model=self.model,
            messages=[
                SystemMessage(
                    role="system",
                    content=(REFLECTIVE_SYSTEM_PROMPT if context.reflection_feedback is not None else SYSTEM_PROMPT),
                ),
                UserMessage(role="user", content=json.dumps(
                    context.to_dict(), ensure_ascii=False, sort_keys=True,
                )),
            ],
            call_name="evotau_customer_strategy_evolver_v2",
            num_retries=0,
            **dict(self.model_args),
        )
        try:
            result = json.loads(message.content or "")
        except (TypeError, json.JSONDecodeError) as exc:
            content = message.content or ""
            raise CustomerEvolverResponseError(
                "Customer Evolver returned invalid JSON",
                response_sha256=sha256_json(content),
            ) from exc
        if not isinstance(result, dict):
            raise TypeError("Customer Evolver must return a JSON object")
        return result
