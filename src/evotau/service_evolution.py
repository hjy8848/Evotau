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
    verification_hypothesis: str

    def __post_init__(self) -> None:
        if not isinstance(self.target_failure_id, str) or not self.target_failure_id.strip():
            raise ValueError("repair proposal requires a verified target failure ID")
        if (not isinstance(self.verification_hypothesis, str)
                or not self.verification_hypothesis.strip()):
            raise ValueError("repair proposal requires a falsifiable verification hypothesis")

    def to_dict(self) -> dict[str, object]:
        return {
            "target_failure_id": self.target_failure_id,
            "rule": self.rule.to_dict(),
            "verification_hypothesis": self.verification_hypothesis,
        }


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

    def __post_init__(self) -> None:
        if not isinstance(self.verifier_ref, str) or not self.verifier_ref.strip():
            raise ValueError("repair audit requires a verifier reference")
        if type(self.approved) is not bool:
            raise TypeError("repair audit approved must be an explicit boolean")
        for name in (
            "permission_delta", "task_specific_content", "eligibility_change",
            "reference_answer_exposure",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"repair audit {name} must be an explicit boolean")
        if any(not isinstance(item, str) or not item.strip() for item in self.approved_policy_refs):
            raise ValueError("approved policy references must be non-empty strings")
        if not isinstance(self.rationale, str) or not self.rationale.strip():
            raise ValueError("repair audit requires a rationale")

    def to_dict(self) -> dict[str, object]:
        return {
            "approved": self.approved,
            "verifier_ref": self.verifier_ref,
            "approved_policy_refs": sorted(self.approved_policy_refs),
            "permission_delta": self.permission_delta,
            "task_specific_content": self.task_specific_content,
            "eligibility_change": self.eligibility_change,
            "reference_answer_exposure": self.reference_answer_exposure,
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class GateUnit:
    key: str
    panel: str  # target, historical, clean, validation
    incumbent: EpisodeRecord
    candidate: EpisodeRecord
    target_failure_id: str | None = None
    initial_s0: EpisodeRecord | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("gate unit key must be non-empty")
        if self.panel not in {"target", "historical", "clean", "validation"}:
            raise ValueError(f"unsupported Service gate panel: {self.panel!r}")
        if self.initial_s0 is not None and self.panel != "clean":
            raise ValueError("initial S0 evidence is only valid for a clean gate unit")
        if self.initial_s0 is not None and not isinstance(self.initial_s0, EpisodeRecord):
            raise TypeError("initial S0 anchor must be an EpisodeRecord")


@dataclass(frozen=True, slots=True)
class GateReport:
    accepted: bool
    reasons: tuple[str, ...]
    candidate_strategy: ServiceStrategy | None
    unit_results: tuple[tuple[str, bool, str], ...]
    target_failure_id: str
    proposal: RepairProposal
    audit: RepairAudit
    evaluated_candidate: ServiceStrategy
    initial_service_strategy_id: str | None = None
    initial_s0_episode_refs: tuple[tuple[str, str], ...] = ()
    inconclusive: bool = False
    unit_episode_refs: tuple[tuple[str, str, str], ...] = ()
    partial_episode_refs: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if type(self.accepted) is not bool:
            raise TypeError("gate report accepted must be a boolean")
        if type(self.inconclusive) is not bool or (self.inconclusive and self.accepted):
            raise ValueError("an inconclusive repair gate cannot be accepted")
        if (not isinstance(self.target_failure_id, str) or not self.target_failure_id
                or self.proposal.target_failure_id != self.target_failure_id):
            raise ValueError("gate report must retain its exact verified target failure")
        if self.evaluated_candidate is None:
            raise ValueError("gate report must retain the evaluated Service candidate")
        if self.accepted and (
            self.reasons or self.candidate_strategy is None
            or self.candidate_strategy != self.evaluated_candidate or not self.audit.approved
        ):
            raise ValueError("accepted gate report must have no rejection reasons and identify its candidate")
        if self.accepted and (
            not self.initial_service_strategy_id or not self.initial_s0_episode_refs
        ):
            raise ValueError("accepted gate report must retain its frozen S0 checkpoint and clean anchors")
        if any(
            not key or not episode_id
            for key, episode_id in self.initial_s0_episode_refs
        ):
            raise ValueError("initial S0 anchor references must identify gate units and episodes")
        if len({key for key, _ in self.initial_s0_episode_refs}) != len(self.initial_s0_episode_refs):
            raise ValueError("each clean gate unit must have exactly one initial S0 anchor reference")
        if (len({key for key, _incumbent, _candidate in self.unit_episode_refs})
                != len(self.unit_episode_refs)
                or any(not key or not incumbent or not candidate
                       for key, incumbent, candidate in self.unit_episode_refs)):
            raise ValueError("gate episode references must identify each paired unit exactly once")
        if (len({key for key, _episode in self.partial_episode_refs})
                != len(self.partial_episode_refs)
                or any(not key or not episode for key, episode in self.partial_episode_refs)):
            raise ValueError("partial gate episode references must be unique and non-empty")
        if not self.accepted and self.candidate_strategy is not None:
            raise ValueError("rejected gate report cannot expose an accepted candidate strategy")

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "inconclusive": self.inconclusive,
            "unit_episode_refs": [list(item) for item in self.unit_episode_refs],
            "partial_episode_refs": [list(item) for item in self.partial_episode_refs],
            "reasons": list(self.reasons),
            "target_failure_id": self.target_failure_id,
            "proposal": self.proposal.to_dict(),
            "audit": self.audit.to_dict(),
            "evaluated_candidate": self.evaluated_candidate.to_dict(),
            "initial_service_strategy_id": self.initial_service_strategy_id,
            "initial_s0_episode_refs": [list(item) for item in self.initial_s0_episode_refs],
            "accepted_candidate": (
                None if self.candidate_strategy is None else self.candidate_strategy.to_dict()
            ),
            "unit_results": [list(item) for item in self.unit_results],
        }


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
    target_failure: FailureRecord,
    proposal: RepairProposal,
    audit: RepairAudit,
    initial_service_strategy_id: str,
    token_counter: Callable[[str], int],
    inconclusive: bool = False,
    partial_episode_refs: tuple[tuple[str, str], ...] = (),
) -> GateReport:
    """Apply target/replay/clean/validation checks, retaining the initial clean anchor."""
    reasons: list[str] = []
    results: list[tuple[str, bool, str]] = []
    target_failure_id = target_failure.failure_id
    if not isinstance(initial_service_strategy_id, str) or not initial_service_strategy_id.strip():
        reasons.append("repair gate requires the frozen initial S0 Service checkpoint ID")
    if proposal.target_failure_id != target_failure_id:
        reasons.append("repair proposal must identify the exact supplied target failure")
    try:
        expected_candidate = build_repair_candidate(
            incumbent, proposal, target_failure, audit, token_counter=token_counter,
        )
    except ValueError as exc:
        reasons.append(f"repair proposal failed static audit: {exc}")
    else:
        if candidate != expected_candidate:
            reasons.append("evaluated Service candidate differs from the audited repair proposal")
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
    if target_units and any(
        (unit.incumbent.policy_rule_id, unit.incumbent.mistake_type,
         unit.incumbent.workflow_stage)
        != (target_failure.signature.policy_rule_id, target_failure.signature.mistake_type,
            target_failure.signature.workflow_stage)
        for unit in target_units
    ):
        reasons.append("target baseline trials do not reproduce the verified target failure signature")
    if sum(unit.panel == "historical" for unit in units) > 1:
        reasons.append("the MVP gate permits at most one historical replay unit")
    for unit in units:
        ok, reason = _check_unit(
            unit,
            incumbent_service_id=service_strategy_id(incumbent),
            candidate_service_id=service_strategy_id(candidate),
            initial_service_strategy_id=initial_service_strategy_id,
        )
        results.append((unit.key, ok, reason))
        if not ok:
            reasons.append(f"{unit.key}: {reason}")
    return GateReport(
        accepted=not reasons,
        reasons=tuple(reasons),
        candidate_strategy=candidate if not reasons else None,
        unit_results=tuple(results),
        target_failure_id=target_failure_id,
        proposal=proposal,
        audit=audit,
        evaluated_candidate=candidate,
        initial_service_strategy_id=initial_service_strategy_id,
        initial_s0_episode_refs=tuple(
            (unit.key, unit.initial_s0.episode_id)
            for unit in units
            if unit.panel == "clean" and unit.initial_s0 is not None
        ),
        inconclusive=inconclusive,
        unit_episode_refs=tuple(
            (unit.key, unit.incumbent.episode_id, unit.candidate.episode_id)
            for unit in units
        ),
        partial_episode_refs=partial_episode_refs,
    )


def _check_unit(
    unit: GateUnit,
    *,
    incumbent_service_id: str,
    candidate_service_id: str,
    initial_service_strategy_id: str,
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
    if (old.invalid_repeated_write_calls is None
            or new.invalid_repeated_write_calls is None):
        return False, "independent audit is missing invalid repeated-write counts"
    if new.invalid_repeated_write_calls > old.invalid_repeated_write_calls:
        return False, "candidate introduces invalid repeated write calls"
    if unit.panel == "clean":
        if old.strategy_applicable is not False or new.strategy_applicable is not False:
            return False, "clean gate must use the native Customer without an EvoTau strategy overlay"
        initial = unit.initial_s0
        if initial is None:
            return False, "clean gate unit is missing its initial S0 success anchor"
        if (initial.status != EpisodeStatus.COMPLETE or initial.customer_valid is not True
                or initial.strategy_applicable is not False or initial.customer_strategy_adherent is not None
                or initial.invalid_repeated_write_calls is None):
            return False, "initial S0 anchor must be a complete, valid native-Customer episode"
        if (initial.task_id, initial.seed) != (old.task_id, old.seed):
            return False, "initial S0 anchor must match the clean task and seed"
        if (initial.customer_strategy_id != old.customer_strategy_id
                or initial.customer_strategy_id != new.customer_strategy_id):
            return False, "initial S0 anchor must use the same native Customer identity"
        if initial.service_strategy_id != initial_service_strategy_id:
            return False, "initial S0 anchor does not match the frozen Service checkpoint ID"
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
    elif unit.panel == "clean":
        initial = unit.initial_s0
        assert initial is not None  # checked above
        if old.task_success is True and new.task_success is not True:
            return False, "candidate regresses a clean task success of the incumbent"
        if initial.task_success is True and new.task_success is not True:
            return False, "candidate regresses a clean success anchored by initial S0"
        if new.policy_violation:
            return False, "candidate introduces a policy violation on clean task"
        if new.tool_calls > 1.25 * old.tool_calls + 1:
            return False, "tool-call efficiency regression exceeds 25% plus one call"
    elif unit.panel == "validation":
        if old.task_success is not True:
            return False, "validation baseline is not a success"
        if new.task_success is not True:
            return False, "candidate regresses validation task success"
        if new.policy_violation:
            old_signature = (old.policy_rule_id, old.mistake_type, old.workflow_stage)
            new_signature = (new.policy_rule_id, new.mistake_type, new.workflow_stage)
            if not old.policy_violation or new_signature != old_signature:
                return False, "candidate introduces a new attributable adversarial validation failure"
        if new.tool_calls > 1.25 * old.tool_calls + 1:
            return False, "tool-call efficiency regression exceeds 25% plus one call"
    else:
        return False, f"unknown gate panel {unit.panel!r}"
    return True, "paired gate passed"
