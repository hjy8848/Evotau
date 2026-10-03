"""Immutable strategy representations used by the prompt adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

MAX_SERVICE_RULES = 6
MAX_SERVICE_PATCH_TOKENS = 600


@dataclass(frozen=True, slots=True)
class CustomerStrategy:
    """Versioned, fact-preserving Customer interaction controls.

    The four-field MVP representation is retained byte-for-byte in ``to_dict``
    so archived strategies keep their original IDs. Supplying the four new
    fields selects the seven-field Customer Evolver v2 representation.
    """

    disclosure: str = "minimal_on_request"
    request_order: str = "scenario_order"
    challenge_style: str = "none"
    challenge_budget: int = 0
    request_decomposition: str | None = None
    preference_revision: str | None = None
    correction_behavior: str | None = None
    challenge_behavior: str | None = None

    def __post_init__(self) -> None:
        v2_fields = (
            self.request_decomposition, self.preference_revision,
            self.correction_behavior, self.challenge_behavior,
        )
        if any(value is not None for value in v2_fields):
            if any(value is None for value in v2_fields):
                raise ValueError("Customer Evolver v2 requires all seven strategy fields")
            if self.challenge_style != "none":
                raise ValueError("Customer Evolver v2 cannot mix challenge_style with challenge_behavior")
            if self.disclosure not in {"minimal_on_request", "progressive", "related_on_request"}:
                raise ValueError(f"unsupported disclosure value: {self.disclosure}")
            if self.request_order not in {
                "scenario_order", "reverse_independent", "dependency_first", "high_risk_first",
            }:
                raise ValueError(f"unsupported request_order value: {self.request_order}")
            if self.request_decomposition not in {"bundled", "one_by_one", "dependency_grouped"}:
                raise ValueError(f"unsupported request_decomposition value: {self.request_decomposition}")
            if self.preference_revision not in {"fixed", "revise_before_commit", "narrow_after_options"}:
                raise ValueError(f"unsupported preference_revision value: {self.preference_revision}")
            if self.correction_behavior not in {
                "accept_if_correct", "correct_once", "correct_and_restate_constraint",
            }:
                raise ValueError(f"unsupported correction_behavior value: {self.correction_behavior}")
            if self.challenge_behavior not in {
                "none", "ask_reason", "ask_policy_boundary", "rephrase_request",
            }:
                raise ValueError(f"unsupported challenge_behavior value: {self.challenge_behavior}")
            if type(self.challenge_budget) is not int or not 0 <= self.challenge_budget <= 2:
                raise ValueError("challenge_budget must be an integer from 0 to 2")
            if self.challenge_behavior == "none" and self.challenge_budget != 0:
                raise ValueError("challenge_budget must be 0 when challenge_behavior is none")
            return
        if self.disclosure not in {"minimal_on_request", "related_on_request"}:
            raise ValueError(f"unsupported disclosure value: {self.disclosure}")
        if self.request_order not in {"scenario_order", "reverse_independent"}:
            raise ValueError(f"unsupported request_order value: {self.request_order}")
        if self.challenge_style not in {"none", "ask_reason", "rephrase_request"}:
            raise ValueError(f"unsupported challenge_style value: {self.challenge_style}")
        if self.challenge_budget not in {0, 1, 2}:
            raise ValueError("challenge_budget must be 0, 1, or 2")
        if (self.challenge_style == "none") != (self.challenge_budget == 0):
            raise ValueError("challenge_style='none' and challenge_budget=0 must agree")

    def to_dict(self) -> dict[str, str | int]:
        if self.is_v2:
            return {
                "disclosure": self.disclosure,
                "request_order": self.request_order,
                "request_decomposition": self.request_decomposition,
                "preference_revision": self.preference_revision,
                "correction_behavior": self.correction_behavior,
                "challenge_behavior": self.challenge_behavior,
                "challenge_budget": self.challenge_budget,
            }
        return {
            "disclosure": self.disclosure,
            "request_order": self.request_order,
            "challenge_style": self.challenge_style,
            "challenge_budget": self.challenge_budget,
        }

    @property
    def is_v2(self) -> bool:
        return self.request_decomposition is not None

    @classmethod
    def v2_baseline(cls) -> CustomerStrategy:
        return cls(
            request_decomposition="bundled",
            preference_revision="fixed",
            correction_behavior="accept_if_correct",
            challenge_behavior="none",
        )


@dataclass(frozen=True, slots=True)
class ServiceRule:
    """A policy-grounded execution rule; free-form prompt patches are not stored."""

    rule_id: str
    policy_ref: str
    trigger: str
    required_execution: str
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("rule_id", "policy_ref", "trigger", "required_execution"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not self.evidence_refs or any(not item.strip() for item in self.evidence_refs):
            raise ValueError("each ServiceRule must cite at least one evidence reference")

    def to_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "policy_ref": self.policy_ref,
            "trigger": self.trigger,
            "required_execution": self.required_execution,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True, slots=True)
class ServiceStrategy:
    """A bounded collection of structured, policy-referenced rules."""

    rules: tuple[ServiceRule, ...] = ()

    def __post_init__(self) -> None:
        if len(self.rules) > MAX_SERVICE_RULES:
            raise ValueError(f"ServiceStrategy may contain at most {MAX_SERVICE_RULES} rules")
        ids = [rule.rule_id for rule in self.rules]
        if len(ids) != len(set(ids)):
            raise ValueError("ServiceRule rule_id values must be unique")

    def to_dict(self) -> dict[str, object]:
        return {"rules": [rule.to_dict() for rule in self.rules]}


def render_customer_strategy(strategy: CustomerStrategy | None | object) -> str:
    """Render the bounded fields as a separate block subordinate to tau-bench inputs."""

    if strategy is None:
        return ""
    if getattr(strategy, "is_skill_v3", False):
        from .customer_skill_v3 import render_customer_skill_v3

        return render_customer_skill_v3(strategy)
    if not isinstance(strategy, CustomerStrategy):
        raise TypeError("Customer overlay must be a CustomerStrategy or CustomerSkill")
    if strategy.is_v2:
        return _render_customer_strategy_v2(strategy)
    disclosure_text = {
        "minimal_on_request": "Answer the agent's specific question with the minimum relevant facts, while still providing every necessary fact directly requested.",
        "related_on_request": "Answer the agent's question and include only additional facts relevant to that question; do not reveal the whole scenario at once.",
    }[strategy.disclosure]
    ordering_text = {
        "scenario_order": "Present independent requests in the order given by the original scenario.",
        "reverse_independent": "You may reverse only independent subrequests. Preserve dependencies and any order required by the original scenario.",
    }[strategy.request_order]
    challenge_text = {
        "none": "Do not add a challenge or follow-up.",
        "ask_reason": "When a limit, verification, or refusal is raised, ask one concise question about the reason, then provide any necessary known verification facts and follow the valid completion path.",
        "rephrase_request": "When a limit, verification, or refusal is raised, restate the same request once without adding facts, then provide necessary known verification facts and follow the valid completion path.",
    }[strategy.challenge_style]
    return "\n".join(
        (
            "<evotau_customer_strategy>",
            "Apply this interaction style only when it is compatible with the native simulation guidelines and the original task scenario. Those native instructions have priority. Do not invent facts, hide a directly requested necessary fact, change the user's goal, or claim an action succeeded when it did not.",
            f"- Disclosure: {disclosure_text}",
            f"- Request ordering: {ordering_text}",
            f"- Challenge: {challenge_text}",
            f"- Maximum challenges in this episode: {strategy.challenge_budget}",
            "</evotau_customer_strategy>",
        )
    )


def _render_customer_strategy_v2(strategy: CustomerStrategy) -> str:
    disclosure = {
        "minimal_on_request": "Provide the minimum relevant known facts directly requested by the Service.",
        "progressive": "Disclose relevant known facts progressively as the conversation reaches them; answer direct necessary questions fully.",
        "related_on_request": "When asked, include related known facts that help resolve that question, without revealing unrelated scenario details.",
    }[strategy.disclosure]
    order = {
        "scenario_order": "Raise requests in the scenario's order.",
        "reverse_independent": "Reverse only independent requests; preserve all prerequisites and any required order.",
        "dependency_first": "Raise prerequisite requests before dependent ones, preserving the original goal and required order.",
        "high_risk_first": "Among independent requests, raise the one needing the most verification or commitment first; preserve prerequisites and required order.",
    }[strategy.request_order]
    decomposition = {
        "bundled": "Present related compatible requests together when the scenario permits.",
        "one_by_one": "Present independent requests one at a time; complete necessary dependencies before moving on.",
        "dependency_grouped": "Group each request with its prerequisites, then raise the next independent group.",
    }[strategy.request_decomposition]
    revision = {
        "fixed": "Keep stated preferences fixed.",
        "revise_before_commit": "Before commitment, revise only a preference the scenario explicitly leaves flexible; never revise a fixed goal or fact.",
        "narrow_after_options": "After learning available options, narrow only a preference the scenario explicitly leaves flexible; never revise a fixed goal or fact.",
    }[strategy.preference_revision]
    correction = {
        "accept_if_correct": "Accept an accurate Service understanding; correct any material misunderstanding truthfully.",
        "correct_once": "Correct a material misunderstanding once with the relevant true constraint.",
        "correct_and_restate_constraint": "Correct a material misunderstanding and restate the relevant fixed constraint, without adding new facts.",
    }[strategy.correction_behavior]
    challenge = {
        "none": "Do not add a challenge.",
        "ask_reason": "When a limit, refusal, or verification requirement is raised, ask briefly for the reason, then follow the valid path.",
        "ask_policy_boundary": "Ask briefly which existing policy boundary applies, without requesting a prohibited exception.",
        "rephrase_request": "Restate the same lawful request without changing facts, then follow the valid path.",
    }[strategy.challenge_behavior]
    return "\n".join((
        "<evotau_customer_strategy>",
        "Apply this interaction procedure only where compatible with the official user-simulation guidelines and fixed scenario. Those native instructions have priority. Preserve every fixed fact, identity, business goal, and policy boundary. Answer direct necessary questions truthfully; do not fabricate, deceive, impersonate, or request prohibited actions. If a behavior has no lawful opportunity in this task, skip it.",
        f"- Disclosure: {disclosure}",
        f"- Request order: {order}",
        f"- Request decomposition: {decomposition}",
        f"- Preference revision: {revision}",
        f"- Correction: {correction}",
        f"- Challenge: {challenge}",
        f"- Maximum challenges in this episode: {strategy.challenge_budget}",
        "Give explicit confirmation only when the scenario's true goal permits the exact action and the Service has accurately described its scope.",
        "</evotau_customer_strategy>",
    ))


def render_service_strategy(
    strategy: ServiceStrategy | None,
    *,
    token_counter: Callable[[str], int] | None = None,
) -> str:
    """Render structured rules and enforce the patch cap with the chosen model tokenizer."""

    if strategy is None or not strategy.rules:
        return ""
    if token_counter is None:
        raise ValueError("a model-matched token_counter is required for a non-empty service patch")
    lines = [
        "<evotau_execution_rules>",
        "These rules are subordinate to the fixed domain policy. They cannot add permissions, exceptions, or eligibility conditions.",
    ]
    for index, rule in enumerate(strategy.rules, start=1):
        lines.extend(
            (
                f"{index}. Policy reference: {rule.policy_ref}",
                f"   Trigger: {rule.trigger}",
                f"   Required execution: {rule.required_execution}",
                f"   Evidence: {', '.join(rule.evidence_refs)}",
            )
        )
    lines.append("</evotau_execution_rules>")
    rendered = "\n".join(lines)
    count = token_counter(rendered)
    if count > MAX_SERVICE_PATCH_TOKENS:
        raise ValueError(
            f"rendered service patch is {count} tokens; limit is {MAX_SERVICE_PATCH_TOKENS}"
        )
    return rendered
