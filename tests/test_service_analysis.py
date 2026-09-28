from __future__ import annotations

from dataclasses import replace

import pytest

from evotau.crossplay import CrossPlayMatrix, build_crossplay_matrix
from evotau.records import (
    EpisodeRecord,
    EpisodeStatus,
    EvidenceRef,
    FailureRecord,
    customer_strategy_id,
    service_strategy_id,
)
from evotau.service_analysis import analyze_service_repair_crossplay
from evotau.service_evolution import GateReport, RepairAudit, RepairProposal
from evotau.strategies import CustomerStrategy, ServiceRule, ServiceStrategy


def make_episode(*, episode_id: str, task: str, seed: int, customer: CustomerStrategy | None,
                 service: ServiceStrategy, failure: bool) -> EpisodeRecord:
    policy = "retail.policy:explicit_confirmation"
    return EpisodeRecord(
        episode_id=episode_id,
        task_id=task,
        seed=seed,
        customer_strategy_id=customer_strategy_id(customer),
        service_strategy_id=service_strategy_id(service),
        status=EpisodeStatus.COMPLETE,
        task_success=not failure,
        customer_valid=True,
        strategy_applicable=customer is not None,
        customer_strategy_adherent=True if customer is not None else None,
        policy_violation=failure,
        invalid_repeated_write_calls=0,
        policy_rule_id=policy if failure else None,
        mistake_type="missing_explicit_confirmation" if failure else None,
        workflow_stage="pre_write" if failure else None,
        evidence=(EvidenceRef(2, "tool", "write was attempted before explicit confirmation"),)
        if failure else (),
        tool_calls=3 if failure else 2,
    )


def verified_failure(item: EpisodeRecord, generation: int = 0) -> FailureRecord:
    reproduction = replace(item, episode_id=f"{item.episode_id}:replay", seed=item.seed + 10_000)
    return FailureRecord.verify(
        item, reproduction_episode=reproduction, generation=generation,
        verifier=f"review:{item.episode_id}",
        reproduction_verifier=f"review:{reproduction.episode_id}",
    )


def matrix(*, task: str, seeds: tuple[int, ...], service: ServiceStrategy,
           attack: CustomerStrategy, failures: set[tuple[str, int]],
           verified_failures: tuple[FailureRecord, ...] = (),
           repaired_signature_keys: tuple[str, ...] = ()):
    customers = (None, attack)
    episodes = tuple(
        make_episode(
            episode_id=f"{service_strategy_id(service)[:6]}:{task}:{seed}:{customer_strategy_id(customer)[:6]}",
            task=task,
            seed=seed,
            customer=customer,
            service=service,
            failure=customer is not None and (task, seed) in failures,
        )
        for customer in customers for seed in seeds
    )
    return build_crossplay_matrix(
        episodes,
        verified_failures,
        customer_strategies=customers,
        service_strategies=(service,),
        task_ids=(task,),
        seeds=seeds,
        repaired_signature_keys=repaired_signature_keys,
    )


def gate_for(failure: FailureRecord, candidate: ServiceStrategy, *, accepted: bool) -> GateReport:
    rule = candidate.rules[0]
    proposal = RepairProposal(
        failure.failure_id,
        rule,
        "The same eligible write completes only after the required confirmation is obtained.",
    )
    audit = RepairAudit(
        approved=True,
        verifier_ref=f"audit:{failure.failure_id}",
        approved_policy_refs=frozenset({failure.policy_ref}),
        rationale="The repair preserves the fixed policy and changes no permissions.",
    )
    return GateReport(
        accepted=accepted,
        reasons=() if accepted else ("candidate failed one prespecified clean/replay gate",),
        candidate_strategy=candidate if accepted else None,
        unit_results=(("target-trial", accepted, "paired target result"),),
        target_failure_id=failure.failure_id,
        proposal=proposal,
        audit=audit,
        evaluated_candidate=candidate,
        initial_service_strategy_id=service_strategy_id(ServiceStrategy()),
        initial_s0_episode_refs=(("clean", "s0-clean-anchor"),),
    )


def test_rq2_reports_exact_target_effect_gate_acceptance_and_clean_preservation():
    attack = CustomerStrategy(challenge_style="ask_reason", challenge_budget=1)
    incumbent_service = ServiceStrategy()
    target_episodes = tuple(
        make_episode(
            episode_id=(
                f"{service_strategy_id(incumbent_service)[:6]}:target:{seed}:"
                f"{customer_strategy_id(attack)[:6]}"
            ),
            task="target", seed=seed,
            customer=attack, service=incumbent_service, failure=True,
        )
        for seed in (1, 2)
    )
    target_failures = tuple(verified_failure(item) for item in target_episodes)
    candidate_service = ServiceStrategy((ServiceRule(
        "confirm-before-write",
        "retail.policy:explicit_confirmation",
        "before a write action",
        "state the complete action details and obtain explicit confirmation before writing",
        target_failures[0].evidence_refs,
    ),))

    old_matrix = matrix(
        task="target", seeds=(1, 2), service=incumbent_service, attack=attack,
        failures={("target", 1), ("target", 2)}, verified_failures=target_failures,
    )
    new_matrix = matrix(
        task="target", seeds=(1, 2), service=candidate_service, attack=attack,
        failures=set(),
    )
    report = analyze_service_repair_crossplay(
        old_matrix,
        new_matrix,
        panel_scope="target",
        gate_reports=(gate_for(target_failures[0], candidate_service, accepted=True),
                      gate_for(target_failures[1], candidate_service, accepted=False)),
        target_failures=target_failures,
        request_budget_cap=100,
        provider_attempts=90,
    )

    assert report.proposed_repairs == 2
    assert report.accepted_repairs == 1 and report.rejected_repairs == 1
    assert report.repair_acceptance_rate == 0.5
    assert report.accepted_target_signature_count == 1
    assert report.target_failure_rate_reduction == 1.0
    assert report.attributable_failure_rate_change == -1.0
    assert report.task_success_rate_change == 0.5
    assert report.policy_violation_rate_change == -0.5
    assert report.invalid_repeated_write_call_rate_change == 0.0
    assert report.incumbent_repeated_write_audit_coverage == 1.0
    assert report.candidate_repeated_write_audit_coverage == 1.0
    assert report.clean_success_rate_change == 0.0
    assert report.clean_policy_violation_rate_change == 0.0
    assert len(report.gate_report_sha256) == 2
    assert report.to_dict()["panel_scope"] == "target"

    attack_id = customer_strategy_id(attack)
    partial_incumbent = replace(
        old_matrix,
        cells=tuple(
            replace(cell, repeated_write_audited_episodes=1,
                    repeated_write_audit_coverage=0.5)
            if cell.customer_strategy_id == attack_id else cell
            for cell in old_matrix.cells
        ),
    )
    partial_audit_report = analyze_service_repair_crossplay(
        partial_incumbent,
        new_matrix,
        panel_scope="target",
        gate_reports=(gate_for(target_failures[0], candidate_service, accepted=True),),
        target_failures=target_failures,
        request_budget_cap=100,
        provider_attempts=90,
    )
    assert partial_audit_report.incumbent_repeated_write_audit_coverage < 1.0
    assert partial_audit_report.invalid_repeated_write_call_rate_change is None

    partial_gate = replace(
        gate_for(target_failures[1], candidate_service, accepted=False),
        inconclusive=True,
    )
    incomplete = analyze_service_repair_crossplay(
        old_matrix,
        new_matrix,
        panel_scope="target",
        gate_reports=(gate_for(target_failures[0], candidate_service, accepted=True), partial_gate),
        target_failures=target_failures,
        request_budget_cap=100,
        provider_attempts=90,
    )
    assert incomplete.proposed_repairs == 2
    assert incomplete.accepted_repairs == 1 and incomplete.rejected_repairs == 0
    assert incomplete.inconclusive_repairs == 1
    assert incomplete.repair_acceptance_rate is None


def test_rq2_reports_recurrence_against_the_frozen_repaired_signature_set():
    attack = CustomerStrategy()
    old_service = ServiceStrategy()
    target_episode = make_episode(
        episode_id="source:target:1", task="source", seed=1, customer=attack,
        service=old_service, failure=True,
    )
    source_failure = verified_failure(target_episode)
    repaired_signature = source_failure.signature.key
    candidate_service = ServiceStrategy((ServiceRule(
        "historical-regression",
        source_failure.policy_ref,
        "before a write action",
        "verify complete action details and explicit confirmation before writing",
        source_failure.evidence_refs,
    ),))
    historical_failures = tuple(
        verified_failure(make_episode(
                episode_id=(
                    f"{service_strategy_id(candidate_service)[:6]}:history:{seed}:"
                    f"{customer_strategy_id(attack)[:6]}"
                ),
                task="history", seed=seed,
                customer=attack, service=candidate_service, failure=True,
            ), generation=1)
        for seed in (4, 5)
    )
    old_matrix = matrix(
        task="history", seeds=(4, 5), service=old_service, attack=attack,
        failures=set(), repaired_signature_keys=(repaired_signature,),
    )
    new_matrix = matrix(
        task="history", seeds=(4, 5), service=candidate_service, attack=attack,
        failures={("history", 4), ("history", 5)},
        verified_failures=historical_failures,
        repaired_signature_keys=(repaired_signature,),
    )

    report = analyze_service_repair_crossplay(
        old_matrix,
        new_matrix,
        panel_scope="historical",
        gate_reports=(),
        target_failures=(source_failure,),
        request_budget_cap=100,
        provider_attempts=90,
        repaired_signature_keys=(repaired_signature,),
    )

    assert report.repair_acceptance_rate is None
    assert report.historical_recurrence_episodes == 2
    assert report.historical_adherent_episodes == 2
    assert report.incumbent_historical_recurrence_episodes == 0
    assert report.incumbent_historical_adherent_episodes == 2
    assert report.incumbent_historical_recurrence_rate == 0.0
    assert report.historical_recurrence_rate == 1.0
    assert report.historical_recurrence_rate_change == 1.0
    assert report.incumbent_repaired_signature_coverage == 0.0
    assert report.repaired_signature_coverage == 1.0
    assert report.repaired_signature_count == 1


def test_rq2_requires_matching_frozen_panels_and_true_native_clean_control():
    attack = CustomerStrategy()
    old_service, new_service = ServiceStrategy(), ServiceStrategy((ServiceRule(
        "r", "retail.policy:explicit_confirmation", "before write", "confirm first",
        ("source#turn:2:tool",),
    ),))
    baseline = matrix(
        task="task", seeds=(1, 2), service=old_service, attack=attack, failures=set(),
    )
    changed_panel = matrix(
        task="other", seeds=(1, 2), service=new_service, attack=attack, failures=set(),
    )
    with pytest.raises(ValueError, match="same task/seed panel"):
        analyze_service_repair_crossplay(
            baseline, changed_panel, panel_scope="heldout", gate_reports=(), target_failures=(),
            request_budget_cap=100, provider_attempts=90,
        )

    clean_id = customer_strategy_id(None)
    base_candidate = matrix(
        task="task", seeds=(1, 2), service=new_service, attack=attack, failures=set(),
    )
    invalid_clean_cell_matrix = CrossPlayMatrix(
        base_candidate.customer_strategy_ids,
        base_candidate.service_strategy_ids,
        base_candidate.task_ids,
        base_candidate.seeds,
        tuple(
            replace(cell, strategy_opportunities=cell.attempted_episodes,
                    strategy_not_applicable_episodes=0, strategy_adherence_rate=0.0)
            if cell.customer_strategy_id == clean_id else cell
            for cell in base_candidate.cells
        ),
    )
    with pytest.raises(ValueError, match="native/no-overlay"):
        analyze_service_repair_crossplay(
            baseline,
            invalid_clean_cell_matrix,
            panel_scope="clean",
            gate_reports=(),
            target_failures=(),
            request_budget_cap=100,
            provider_attempts=90,
        )
