"""Structured, policy-bounded Service repair proposals and regression gates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .records import EpisodeRecord, EpisodeStatus, FailureRecord, service_strategy_id
from .strategies import (
    MAX_SERVICE_PATCH_TOKENS,
    MAX_SERVICE_RULES,
    ServiceRule,
    ServiceStrategy,
    render_service_strategy,
)


@dataclass(frozen=True, slots=True)
class RepairProposal:
    target_failure_id: str
    rule: ServiceRule


@dataclass(frozen=True, slots=True)
class RepairAudit:
    approved: bool
    verifier_ref: str
    approved_policy_refs: frozenset[str]
    permission_delta: bool = False
    task_specific_content: bool = False
    eligibility_change: bool = False
    reference_answer_exposure: bool = False
    rationale: str = ""


@dataclass(frozen=True, slots=True)
class GateUnit:
    key: str
    panel: str  # target, historical, clean, validation
    incumbent: EpisodeRecord
    candidate: EpisodeRecord
    target_failure_id: str | None = None


@dataclass(frozen=True, slots=True)
class GateReport:
    accepted: bool
    reasons: tuple[str, ...]
    candidate_strategy: ServiceStrategy | None
    unit_results: tuple[tuple[str, bool, str], ...]

    def __post_init__(self) -> None:
        if self.accepted and (self.reasons or self.candidate_strategy is None):
            raise ValueError("accepted gate report must have no rejection reasons and identify its candidate")
        if not self.accepted and self.candidate_strategy is not None:
            raise ValueError("rejected gate report cannot expose an accepted candidate strategy")


def build_repair_candidate(
    incumbent: ServiceStrategy,
    proposal: RepairProposal,
    failure: FailureRecord,
    audit: RepairAudit,
    *,
    token_counter: Callable[[str], int],
) -> ServiceStrategy:
    """Apply an independently audited repair, replacing repeated trigger/ref pairs."""
    if proposal.target_failure_id != failure.failure_id:
        raise ValueError("repair proposal must target the supplied verified failure")
    if not audit.approved or not audit.verifier_ref.strip():
        raise ValueError("repair proposal lacks independent policy audit approval")
    if audit.permission_delta:
        raise ValueError("Service repair cannot alter permissions")
    if audit.task_specific_content or audit.eligibility_change or audit.reference_answer_exposure:
        raise ValueError("repair audit found task-specific content, eligibility changes, or answer exposure")
    if proposal.rule.policy_ref != failure.policy_ref or proposal.rule.policy_ref not in audit.approved_policy_refs:
        raise ValueError("repair must cite the failure's audited fixed-policy reference")
    if not set(proposal.rule.evidence_refs) & set(failure.evidence_refs):
        raise ValueError("repair rule must cite evidence from its verified target failure")
    rules = list(incumbent.rules)
    match = next((i for i, old in enumerate(rules) if old.trigger == proposal.rule.trigger and old.policy_ref == proposal.rule.policy_ref), None)
    if match is None:
        if len(rules) >= MAX_SERVICE_RULES:
            raise ValueError("Service strategy rule limit reached")
        rules.append(proposal.rule)
    else:
        rules[match] = proposal.rule
    result = ServiceStrategy(tuple(rules))
    rendered = render_service_strategy(result, token_counter=token_counter)
    if not rendered:
        raise ValueError("rendered Service repair cannot be empty")
    if token_counter(rendered) > MAX_SERVICE_PATCH_TOKENS:
        raise ValueError("rendered Service repair exceeds the token cap")
    return result


def evaluate_repair_gate(
    incumbent: ServiceStrategy,
    candidate: ServiceStrategy,
    units: tuple[GateUnit, ...],
    *,
    target_failure_id: str,
    token_counter: Callable[[str], int],
) -> GateReport:
    """Apply target/replay/clean/validation checks to paired episode evidence."""
    reasons: list[str] = []
    results: list[tuple[str, bool, str]] = []
    try:
        render_service_strategy(candidate, token_counter=token_counter)
    except ValueError as exc:
        reasons.append(f"invalid candidate: {exc}")
    if not target_failure_id:
        reasons.append("repair gate requires a verified target failure ID")
    if candidate == incumbent:
        reasons.append("repair gate candidate must differ from the incumbent strategy")
    required_panels = {"target", "historical", "clean", "validation"}
    panels = {unit.panel for unit in units}
    if not required_panels <= panels:
        reasons.append(f"missing gate panels: {sorted(required_panels - panels)}")
    keys = [unit.key for unit in units]
    if len(keys) != len(set(keys)):
        reasons.append("gate unit keys must be unique")
    target_units = tuple(unit for unit in units if unit.panel == "target")
    if len(target_units) != 2:
        reasons.append("target error must be tested on exactly two pre-specified trials")
    elif (len({unit.incumbent.task_id for unit in target_units}) != 1
          or len({unit.incumbent.seed for unit in target_units}) != 2
          or {unit.target_failure_id for unit in target_units} != {target_failure_id}):
        reasons.append("target trials must replay one verified failure on two distinct seeds")
    elif len({(unit.incumbent.policy_rule_id, unit.incumbent.mistake_type,
               unit.incumbent.workflow_stage) for unit in target_units}) != 1:
        reasons.append("both target trials must reproduce the same failure signature")
    if sum(unit.panel == "historical" for unit in units) > 1:
        reasons.append("the MVP gate permits at most one historical replay unit")
    for unit in units:
        ok, reason = _check_unit(
            unit,
            incumbent_service_id=service_strategy_id(incumbent),
            candidate_service_id=service_strategy_id(candidate),
        )
        results.append((unit.key, ok, reason))
        if not ok:
            reasons.append(f"{unit.key}: {reason}")
    return GateReport(not reasons, tuple(reasons), candidate if not reasons else None, tuple(results))


def _check_unit(
    unit: GateUnit,
    *,
    incumbent_service_id: str,
    candidate_service_id: str,
) -> tuple[bool, str]:
    old, new = unit.incumbent, unit.candidate
    if (old.task_id, old.seed) != (new.task_id, new.seed):
        return False, "incumbent and candidate must use the same task and seed"
    if old.customer_strategy_id != new.customer_strategy_id:
        return False, "Service gate pairs must keep the Customer strategy fixed"
    if old.service_strategy_id != incumbent_service_id or new.service_strategy_id != candidate_service_id:
        return False, "paired episodes do not match the incumbent and candidate Service strategies"
    if old.status != EpisodeStatus.COMPLETE or new.status != EpisodeStatus.COMPLETE:
        return False, "incomplete or invalid episode evidence"
    if old.customer_valid is not True or new.customer_valid is not True:
        return False, "invalid Customer behavior cannot be used as gate evidence"
    if unit.panel == "clean":
        if old.strategy_applicable is not False or new.strategy_applicable is not False:
            return False, "clean gate must use the native Customer without an EvoTau strategy overlay"
    elif (old.strategy_applicable is not True or new.strategy_applicable is not True
          or old.customer_strategy_adherent is not True
          or new.customer_strategy_adherent is not True):
        return False, "adversarial Customer strategy must be applicable and adherent in both paired episodes"
    if unit.panel in {"target", "historical"}:
        if unit.panel == "target":
            if not unit.target_failure_id:
                return False, "target gate must reference its verified target failure"
            if old.task_success is not False or not old.has_attributable_failure_candidate:
                return False, "target baseline is not a valid attributable service failure"
            if new.task_success is not True:
                return False, "candidate did not resolve the target failure"
            if new.policy_violation:
                return False, "candidate retains the target policy violation"
        else:
            if old.task_success is not True:
                return False, "historical guard must use an incumbent-passing replay task"
            if new.task_success is not True:
                return False, "candidate regresses an incumbent-passing historical task"
            if new.policy_violation:
                return False, "candidate introduces a policy violation on historical replay"
        if new.tool_calls > 1.25 * old.tool_calls + 1:
            return False, "tool-call efficiency regression exceeds 25% plus one call"
    elif unit.panel in {"clean", "validation"}:
        if old.task_success is not True:
            return False, "clean/validation baseline is not a success"
        if new.task_success is not True:
            return False, "candidate regresses clean/validation task success"
        if unit.panel == "clean" and new.policy_violation:
            return False, "candidate introduces a policy violation on clean task"
        if unit.panel == "validation" and new.policy_violation:
            old_signature = (old.policy_rule_id, old.mistake_type, old.workflow_stage)
            new_signature = (new.policy_rule_id, new.mistake_type, new.workflow_stage)
            if not old.policy_violation or new_signature != old_signature:
                return False, "candidate introduces a new attributable adversarial validation failure"
        if new.tool_calls > 1.25 * old.tool_calls + 1:
            return False, "tool-call efficiency regression exceeds 25% plus one call"
    else:
        return False, f"unknown gate panel {unit.panel!r}"
    return True, "paired gate passed"
