"""Immutable strategy representations used by the prompt adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

MAX_SERVICE_RULES = 6
MAX_SERVICE_PATCH_TOKENS = 600


@dataclass(frozen=True, slots=True)
class CustomerStrategy:
    """The four bounded interaction controls in the MVP."""

    disclosure: str = "minimal_on_request"
    request_order: str = "scenario_order"
    challenge_style: str = "none"
    challenge_budget: int = 0

    def __post_init__(self) -> None:
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
        return {
            "disclosure": self.disclosure,
            "request_order": self.request_order,
            "challenge_style": self.challenge_style,
            "challenge_budget": self.challenge_budget,
        }


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


def render_customer_strategy(strategy: CustomerStrategy | None) -> str:
    """Render the bounded fields as a separate block subordinate to tau-bench inputs."""

    if strategy is None:
        return ""
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
