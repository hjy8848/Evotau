from __future__ import annotations

from dataclasses import replace

import pytest

from evotau.crossplay import build_crossplay_matrix
from evotau.records import (
    EpisodeRecord,
    EpisodeStatus,
    EvidenceRef,
    FailureRecord,
    customer_strategy_id,
    service_strategy_id,
)
from evotau.service_analysis import (
    ServiceRobustnessRun,
    analyze_rq2_service_robustness,
    analyze_service_repair_crossplay,
)
from evotau.service_evolution import GateReport, RepairAudit, RepairProposal
from evotau.strategies import CustomerStrategy, ServiceRule, ServiceStrategy


def make_episode(*, episode_id: str, task: str, seed: int, customer: CustomerStrategy | None,
                 service: ServiceStrategy, failure: bool) -> EpisodeRecord:
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
        policy_rule_id="retail.policy:explicit_confirmation" if failure else None,
        mistake_type="missing_explicit_confirmation" if failure else None,
        workflow_stage="pre_write" if failure else None,
        evidence=(EvidenceRef(2, "tool", "write preceded confirmation"),) if failure else (),
    )


def make_matrix(*, run_name: str, panel: str, service: ServiceStrategy,
                attack: CustomerStrategy, seeds: tuple[int, ...], failed_seeds: set[int],
                repaired_keys: tuple[str, ...] = ()):
    episodes = tuple(
        make_episode(
            episode_id=(
                f"{run_name}:{panel}:{service_strategy_id(service)[:5]}:{seed}:"
                f"{customer_strategy_id(customer)[:5]}"
            ),
            task=panel,
            seed=seed,
            customer=customer,
            service=service,
            failure=customer is not None and seed in failed_seeds,
        )
        for customer in (None, attack) for seed in seeds
    )
    failures = tuple(_verified_failure(episode) for episode in episodes
                     if episode.has_attributable_failure_candidate)
    matrix = build_crossplay_matrix(
        episodes,
        failures,
        customer_strategies=(None, attack),
        service_strategies=(service,),
        task_ids=(panel,),
        seeds=seeds,
        repaired_signature_keys=repaired_keys,
    )
    return matrix, failures


def _verified_failure(episode: EpisodeRecord) -> FailureRecord:
    reproduction = replace(
        episode, episode_id=f"{episode.episode_id}:replay", seed=episode.seed + 10_000,
    )
    return FailureRecord.verify(
        episode, reproduction_episode=reproduction, generation=0,
        verifier=f"audit:{episode.episode_id}",
        reproduction_verifier=f"audit:{reproduction.episode_id}",
    )


def gate(failure: FailureRecord, candidate: ServiceStrategy, *, accepted: bool) -> GateReport:
    proposal = RepairProposal(
        failure.failure_id,
        candidate.rules[0],
        "A successful write requires the policy-mandated identity and confirmation checks.",
    )
    audit = RepairAudit(
        approved=True,
        verifier_ref=f"audit:{failure.failure_id}",
        approved_policy_refs=frozenset({failure.policy_ref}),
        rationale="The structured rule preserves policy and adds no permission.",
    )
    return GateReport(
        accepted=accepted,
        reasons=() if accepted else ("pre-specified validation unit failed",),
        candidate_strategy=candidate if accepted else None,
        unit_results=(("target", accepted, "recorded gate result"),),
        target_failure_id=failure.failure_id,
        proposal=proposal,
        audit=audit,
        evaluated_candidate=candidate,
        initial_service_strategy_id=service_strategy_id(ServiceStrategy()),
        initial_s0_episode_refs=(("clean", "s0-clean-anchor"),),
    )


def service_run(run_number: int) -> ServiceRobustnessRun:
    run_name = f"run-{run_number}"
    evolution_seed = run_number * 10
    seeds = (evolution_seed, evolution_seed + 1)
    attack = CustomerStrategy(challenge_style="ask_reason", challenge_budget=1)
    incumbent = ServiceStrategy()
    target_matrix, target_failures = make_matrix(
        run_name=run_name, panel="E-target", service=incumbent, attack=attack,
        seeds=seeds, failed_seeds=set(seeds),
    )
    target = target_failures[0]
    candidate = ServiceStrategy((ServiceRule(
        "confirm-before-write",
        target.policy_ref,
        "before a write action",
        "confirm all action details and obtain explicit confirmation before writing",
        target.evidence_refs,
    ),))
    repaired_keys = (target.signature.key,)
    target_matrix, target_failures = make_matrix(
        run_name=run_name, panel="E-target", service=incumbent, attack=attack,
        seeds=seeds, failed_seeds=set(seeds), repaired_keys=repaired_keys,
    )
    gate_reports = (gate(target, candidate, accepted=True), gate(target_failures[1], candidate, accepted=False))

    panels = []
    for scope, task, baseline_failures, candidate_failures in (
        ("target", "E-target", set(seeds), set()),
        ("historical", "E-history", set(), set(seeds[:max(0, run_number - 1)])),
        ("clean", "V-clean", set(), set()),
        ("validation", "V-adversarial", set(seeds), set(seeds[:run_number % 2])),
        ("heldout", "H-final", set(seeds), set(seeds[:(run_number + 1) % 2])),
    ):
        if scope == "target":
            old_matrix = target_matrix
        else:
            old_matrix, _ = make_matrix(
                run_name=run_name, panel=task, service=incumbent, attack=attack,
                seeds=seeds, failed_seeds=baseline_failures, repaired_keys=repaired_keys,
            )
        new_matrix, _candidate_failures = make_matrix(
            run_name=run_name, panel=task, service=candidate, attack=attack,
            seeds=seeds, failed_seeds=candidate_failures, repaired_keys=repaired_keys,
        )
        # The repair audit target is a source artifact. Its full record and
        # gate hashes are retained in every paired panel report.
        panel = analyze_service_repair_crossplay(
            old_matrix,
            new_matrix,
            panel_scope=scope,
            gate_reports=gate_reports,
            target_failures=target_failures,
            request_budget_cap=100,
            provider_attempts=90 + run_number,
            repaired_signature_keys=repaired_keys,
        )
        panels.append(panel)
    return ServiceRobustnessRun(run_name, evolution_seed, tuple(panels))


def test_rq2_bootstraps_only_independent_runs_and_checks_evh_budgets_and_panels():
    runs = tuple(service_run(index) for index in (1, 2, 3))
    report = analyze_rq2_service_robustness(
        runs,
        evolution_task_ids=("E-target", "E-history"),
        validation_task_ids=("V-clean", "V-adversarial"),
        heldout_task_ids=("H-final",),
        bootstrap_seed=21,
        bootstrap_replicates=500,
    )
    repeated = analyze_rq2_service_robustness(
        runs,
        evolution_task_ids=("E-target", "E-history"),
        validation_task_ids=("V-clean", "V-adversarial"),
        heldout_task_ids=("H-final",),
        bootstrap_seed=21,
        bootstrap_replicates=500,
    )

    assert report.status == "descriptive"
    assert report.request_budget_cap == 100
    assert report.actual_provider_attempts == (("run-1", 91), ("run-2", 92), ("run-3", 93))
    assert len({item[2] for item in report.run_ids_and_input_sha256}) == 3
    metrics = {item.metric: item for item in report.metrics}
    assert metrics["target_failure_rate_reduction"].mean == 1.0
    assert metrics["repair_acceptance_rate"].mean == 0.5
    assert metrics["clean_success_rate_change"].mean == 0.0
    historical = metrics["historical_recurrence_rate"]
    assert historical.mean == pytest.approx(0.5)
    assert historical.bootstrap_95_percentile_interval == (
        {item.metric: item for item in repeated.metrics}["historical_recurrence_rate"]
        .bootstrap_95_percentile_interval
    )
    assert historical.directionally_favorable_runs == 1
    assert all(item.independent_runs == 3 for item in report.metrics)


def test_rq2_rejects_unmatched_h_partition_duplicate_seeds_and_budget_caps():
    runs = tuple(service_run(index) for index in (1, 2, 3))
    with pytest.raises(ValueError, match="sealed ordered H partition"):
        analyze_rq2_service_robustness(
            runs,
            evolution_task_ids=("E-target", "E-history"),
            validation_task_ids=("V-clean", "V-adversarial"),
            heldout_task_ids=("H-other",),
        )
    with pytest.raises(ValueError, match="distinct seeds"):
        analyze_rq2_service_robustness(
            (runs[0], replace(runs[1], evolution_seed=runs[0].evolution_seed), runs[2]),
            evolution_task_ids=("E-target", "E-history"),
            validation_task_ids=("V-clean", "V-adversarial"),
            heldout_task_ids=("H-final",),
        )
    changed_panels = tuple(
        replace(panel, request_budget_cap=99, provider_attempts=99)
        for panel in runs[2].panels
    )
    broken_run = ServiceRobustnessRun(
        runs[2].run_id,
        runs[2].evolution_seed,
        changed_panels,
    )
    with pytest.raises(ValueError, match="budget cap"):
        analyze_rq2_service_robustness(
            (runs[0], runs[1], broken_run),
            evolution_task_ids=("E-target", "E-history"),
            validation_task_ids=("V-clean", "V-adversarial"),
            heldout_task_ids=("H-final",),
        )


def test_rq2_reports_missing_denominators_without_silently_dropping_a_run():
    runs = tuple(service_run(index) for index in (1, 2, 3))
    target_index = next(
        index for index, panel in enumerate(runs[1].panels) if panel.panel_scope == "target"
    )
    missing = replace(runs[1].panels[target_index], repair_acceptance_rate=None)
    panels = list(runs[1].panels)
    panels[target_index] = missing
    runs = (runs[0], replace(runs[1], panels=tuple(panels)), runs[2])

    report = analyze_rq2_service_robustness(
        runs,
        evolution_task_ids=("E-target", "E-history"),
        validation_task_ids=("V-clean", "V-adversarial"),
        heldout_task_ids=("H-final",),
    )
    estimate = next(item for item in report.metrics if item.metric == "repair_acceptance_rate")
    assert report.status == "incomplete_denominators"
    assert estimate.status == "incomplete_denominator"
    assert estimate.complete_run_values == 2
    assert estimate.mean is None
    assert estimate.observations[1] == ("run-2", None)
