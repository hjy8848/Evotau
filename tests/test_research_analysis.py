from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from evotau.crossplay import CrossPlayCell, CrossPlayMatrix
from evotau.records import customer_strategy_id, service_strategy_id
from evotau.research_analysis import (
    AdaptationOutcome,
    EvolutionResponseRun,
    analyze_adaptation_response,
    analyze_rq3_longitudinal_response,
    load_rq3_document,
    main,
)
from evotau.strategies import CustomerStrategy, ServiceRule, ServiceStrategy


def matrix(customer, service, seeds, rates):
    customer_ids = tuple(customer_strategy_id(item) for item in customer)
    service_ids = tuple(service_strategy_id(item) for item in service)
    attempts = len(seeds)
    cells = []
    for customer_id in customer_ids:
        for service_id in service_ids:
            failures = round(rates[(customer_id, service_id)] * attempts)
            adherent = attempts if rates[(customer_id, service_id)] is not None else 0
            signature_keys = tuple(f"{index + 1:016x}" for index in range(failures))
            cells.append(CrossPlayCell(
                customer_strategy_id=customer_id,
                service_strategy_id=service_id,
                attempted_episodes=attempts,
                valid_episodes=attempts,
                invalid_episodes=0,
                infrastructure_episodes=0,
                uncertain_episodes=0,
                strategy_opportunities=attempts,
                strategy_adherent_episodes=adherent,
                strategy_not_applicable_episodes=0,
                strategy_adherence_rate=(adherent / attempts if attempts else None),
                successful_episodes=attempts - failures,
                verified_failure_episodes=failures,
                verified_failure_rate=(failures / adherent if adherent else None),
                unique_task_signature_failures=failures,
                unique_signatures=failures,
                native_success_rate=((attempts - failures) / attempts if attempts else None),
                policy_violation_episodes=failures,
                policy_violation_rate=(failures / attempts if attempts else None),
                repeated_write_audited_episodes=attempts,
                invalid_repeated_write_calls=0,
                repeated_write_audit_coverage=1.0 if attempts else None,
                recurrent_verified_failure_rate=(0.0 if adherent else None),
                verified_signature_keys=signature_keys,
                verified_signature_episode_counts=tuple((key, 1) for key in signature_keys),
            ))
    return CrossPlayMatrix(customer_ids, service_ids, ("task-1",), tuple(seeds), tuple(cells))


def setup():
    old_customer = CustomerStrategy()
    new_customer = CustomerStrategy(request_order="reverse_independent")
    old_service = ServiceStrategy()
    new_service = ServiceStrategy((ServiceRule(
        "confirmation", "retail.policy:explicit_confirmation", "before a write",
        "confirm action details before writing", ("episode-1#turn:2:tool",),
    ),))
    customers, services = (old_customer, new_customer), (old_service, new_service)
    return old_customer, new_customer, old_service, new_service, customers, services


def _service_version(version: int) -> ServiceStrategy:
    if version == 0:
        return ServiceStrategy()
    return ServiceStrategy((ServiceRule(
        f"repair-{version}", "retail.policy:explicit_confirmation",
        f"before write version {version}",
        "confirm all details before writing",
        ("fixture:verified-evidence",),
    ),))


def _rq3_transition(generation, old_customer, new_customer, old_service, new_service,
                    *, evolution_seed: int, observed: bool):
    customer_ids = tuple(customer_strategy_id(item) for item in (old_customer, new_customer))
    service_ids = tuple(service_strategy_id(item) for item in (old_service, new_service))
    distinct_services = (old_service,) if service_ids[0] == service_ids[1] else (old_service, new_service)
    distinct_service_ids = tuple(service_strategy_id(item) for item in distinct_services)
    panel = evolution_seed * 100 + generation * 4
    discovery_seeds = (panel, panel + 1)
    confirmation_seeds = (panel + 2, panel + 3)
    if observed:
        rates = {
            (customer_ids[0], distinct_service_ids[0]): 1.0,
            (customer_ids[1], distinct_service_ids[0]): 0.0,
        }
        if len(distinct_services) == 2:
            rates.update({
                (customer_ids[0], distinct_service_ids[1]): 0.0,
                (customer_ids[1], distinct_service_ids[1]): 1.0,
            })
    else:
        rates = {key: 0.5 for key in (
            (customer_id, service_id)
            for customer_id in customer_ids for service_id in distinct_service_ids
        )}
    discovery = matrix((old_customer, new_customer), distinct_services, discovery_seeds, rates)
    confirmation = matrix((old_customer, new_customer), distinct_services, confirmation_seeds, rates)
    return analyze_adaptation_response(
        discovery, confirmation,
        old_customer=old_customer, new_customer=new_customer,
        old_service=old_service, new_service=new_service,
        generation=generation,
    )


def _rq3_runs(blocks=(11, 22, 33)):
    old_customer = CustomerStrategy()
    middle_customer = CustomerStrategy(request_order="reverse_independent")
    new_customer = CustomerStrategy(disclosure="related_on_request")
    s0, s1, s2 = (_service_version(index) for index in range(3))
    runs = []
    for seed in blocks:
        adaptive = (
            _rq3_transition(0, old_customer, middle_customer, s0, s1,
                            evolution_seed=seed, observed=True),
            _rq3_transition(1, middle_customer, new_customer, s1, s2,
                            evolution_seed=seed, observed=True),
        )
        frozen = (
            _rq3_transition(0, old_customer, middle_customer, s0, s0,
                            evolution_seed=seed, observed=False),
            _rq3_transition(1, middle_customer, new_customer, s0, s0,
                            evolution_seed=seed, observed=False),
        )
        random_mutation = (
            _rq3_transition(0, old_customer, middle_customer, s0, s1,
                            evolution_seed=seed, observed=True),
            _rq3_transition(1, middle_customer, new_customer, s1, s2,
                            evolution_seed=seed, observed=False),
        )
        for condition, transitions in (
            ("adaptive_coevolution", adaptive),
            ("frozen_service", frozen),
            ("random_mutation", random_mutation),
        ):
            runs.append(EvolutionResponseRun(
                run_id=f"{seed}-{condition}",
                seed_block_id=f"block-{seed}",
                evolution_seed=seed,
                condition=condition,
                request_budget_cap=1800,
                provider_attempts=600,
                transitions=transitions,
            ))
    return tuple(runs)


def test_adaptation_response_requires_repair_and_fresh_customer_counter_adaptation():
    old_customer, new_customer, old_service, new_service, customers, services = setup()
    ids = tuple(customer_strategy_id(item) for item in customers)
    service_ids = tuple(service_strategy_id(item) for item in services)
    discovery = matrix(
        customers, services, (1, 2),
        {(ids[0], service_ids[0]): 1.0, (ids[0], service_ids[1]): 0.5,
         (ids[1], service_ids[0]): 0.5, (ids[1], service_ids[1]): 0.5},
    )
    confirmation = matrix(
        customers, services, (3, 4),
        {(ids[0], service_ids[0]): 1.0, (ids[0], service_ids[1]): 0.0,
         (ids[1], service_ids[0]): 0.5, (ids[1], service_ids[1]): 1.0},
    )

    report = analyze_adaptation_response(
        discovery, confirmation,
        old_customer=old_customer, new_customer=new_customer,
        old_service=old_service, new_service=new_service,
    )

    assert report.outcome == AdaptationOutcome.OBSERVED
    assert report.repair_reduced_failure_rate is True
    assert report.customer_raised_failure_rate_after_repair is True
    assert report.repair_reduction == 1.0
    assert report.counter_adaptation_increase == 1.0
    assert report.repair_specific_interaction == 1.5
    assert report.confirmation_seed_count == 2
    assert report.to_dict()["outcome"] == "observed"


def test_adaptation_response_does_not_claim_counter_adaptation_without_increase():
    old_customer, new_customer, old_service, new_service, customers, services = setup()
    ids = tuple(customer_strategy_id(item) for item in customers)
    service_ids = tuple(service_strategy_id(item) for item in services)
    rates = {(ids[0], service_ids[0]): 1.0, (ids[0], service_ids[1]): 0.5,
             (ids[1], service_ids[0]): 1.0, (ids[1], service_ids[1]): 0.0}
    discovery = matrix(customers, services, (1,), rates)
    confirmation = matrix(customers, services, (2,), rates)

    report = analyze_adaptation_response(
        discovery, confirmation,
        old_customer=old_customer, new_customer=new_customer,
        old_service=old_service, new_service=new_service,
    )

    assert report.outcome == AdaptationOutcome.NOT_OBSERVED
    assert report.repair_reduced_failure_rate is True
    assert report.customer_raised_failure_rate_after_repair is False


def test_adaptation_response_requires_a_fresh_panel_and_marks_missing_denominators():
    old_customer, new_customer, old_service, new_service, customers, services = setup()
    ids = tuple(customer_strategy_id(item) for item in customers)
    service_ids = tuple(service_strategy_id(item) for item in services)
    rates = {(ids[0], service_ids[0]): 1.0, (ids[0], service_ids[1]): 0.0,
             (ids[1], service_ids[0]): 0.5, (ids[1], service_ids[1]): 1.0}
    discovery = matrix(customers, services, (1, 2), rates)
    with pytest.raises(ValueError, match="not used for discovery"):
        analyze_adaptation_response(
            discovery, matrix(customers, services, (2,), rates),
            old_customer=old_customer, new_customer=new_customer,
            old_service=old_service, new_service=new_service,
        )

    no_customer_opportunity = matrix(customers, services, (3,), rates)
    no_customer_opportunity = CrossPlayMatrix(
        no_customer_opportunity.customer_strategy_ids,
        no_customer_opportunity.service_strategy_ids,
        no_customer_opportunity.task_ids,
        no_customer_opportunity.seeds,
        tuple(
            cell if cell.customer_strategy_id != ids[1]
            else replace(
                cell,
                strategy_opportunities=0,
                strategy_adherent_episodes=0,
                strategy_not_applicable_episodes=cell.attempted_episodes,
                strategy_adherence_rate=None,
                verified_failure_episodes=0,
                verified_failure_rate=None,
                unique_task_signature_failures=0,
                unique_signatures=0,
                verified_signature_keys=(),
                verified_signature_episode_counts=(),
                recurrent_verified_failure_rate=None,
            )
            for cell in no_customer_opportunity.cells
        ),
    )
    report = analyze_adaptation_response(
        discovery, no_customer_opportunity,
        old_customer=old_customer, new_customer=new_customer,
        old_service=old_service, new_service=new_service,
    )
    assert report.outcome == AdaptationOutcome.INCONCLUSIVE


def test_rq3_aggregates_two_consecutive_response_chains_at_seed_block_level():
    report = analyze_rq3_longitudinal_response(_rq3_runs(), bootstrap_replicates=200)

    assert report.status == "descriptive"
    assert report.independent_seed_blocks == 3
    adaptive_runs = [item for item in report.runs if item.condition == "adaptive_coevolution"]
    assert len(adaptive_runs) == 3
    assert all(item.confirmed_response_chains == 2 for item in adaptive_runs)
    assert all(item.sustained_two_chain_response == 1.0 for item in adaptive_runs)
    sustained_vs_frozen = next(
        item for item in report.paired_comparisons
        if item.baseline_condition == "frozen_service"
        and item.metric == "sustained_two_chain_response"
    )
    assert sustained_vs_frozen.mean_paired_difference == 1.0
    assert sustained_vs_frozen.adaptive_wins == 3
    assert sustained_vs_frozen.bootstrap_95_percentile_interval == (1.0, 1.0)
    sustained_vs_random = next(
        item for item in report.paired_comparisons
        if item.baseline_condition == "random_mutation"
        and item.metric == "sustained_two_chain_response"
    )
    assert sustained_vs_random.adaptive_wins == 3


def test_rq3_requires_three_complete_paired_conditions_and_fresh_lineage():
    two_blocks = _rq3_runs((11, 22))
    report = analyze_rq3_longitudinal_response(two_blocks, bootstrap_replicates=100)
    assert report.status == "insufficient_independent_runs"
    assert all(item.bootstrap_95_percentile_interval is None for item in report.paired_comparisons)

    runs = list(_rq3_runs())
    run = runs[0]
    broken_transition = replace(run.transitions[1], old_customer_strategy_id="not-the-committed-customer")
    with pytest.raises(ValueError, match="prior generation's committed strategies"):
        replace(run, transitions=(run.transitions[0], broken_transition))

    with pytest.raises(ValueError, match="at least three independent seed blocks"):
        analyze_rq3_longitudinal_response(_rq3_runs((11, 22)), minimum_independent_seed_blocks=2)


def test_rq3_frozen_service_control_cannot_change_service():
    runs = list(_rq3_runs())
    frozen_index = next(index for index, item in enumerate(runs) if item.condition == "frozen_service")
    frozen = runs[frozen_index]
    transition = frozen.transitions[0]
    mutated = replace(transition, new_service_strategy_id="changed-service")
    next_transition = replace(
        frozen.transitions[1], old_service_strategy_id="changed-service",
        new_service_strategy_id="changed-service",
    )
    with pytest.raises(ValueError, match="frozen_service control cannot change"):
        replace(frozen, transitions=(mutated, next_transition))


def test_rq3_inconclusive_transitions_are_not_scored_as_non_responses():
    runs = list(_rq3_runs())
    run = runs[0]
    incomplete = replace(
        run.transitions[1],
        outcome=AdaptationOutcome.INCONCLUSIVE,
        repair_reduction=None,
        counter_adaptation_increase=None,
        repair_specific_interaction=None,
        repair_reduced_failure_rate=None,
        customer_raised_failure_rate_after_repair=None,
    )
    runs[0] = replace(run, transitions=(run.transitions[0], incomplete))
    report = analyze_rq3_longitudinal_response(runs, bootstrap_replicates=100)

    adaptive = next(item for item in report.runs if item.run_id == run.run_id)
    assert adaptive.confirmed_response_chains == 1
    assert adaptive.sustained_two_chain_response is None
    comparison = next(
        item for item in report.paired_comparisons
        if item.baseline_condition == "frozen_service"
        and item.metric == "sustained_two_chain_response"
    )
    assert comparison.status == "incomplete_denominator"
    assert report.status == "incomplete_denominators"


def test_rq3_independent_seed_blocks_cannot_reuse_task_episode_seeds():
    runs = list(_rq3_runs())
    reference = {item.run_id: item for item in runs if item.seed_block_id == "block-11"}
    for index, run in enumerate(runs):
        if run.seed_block_id != "block-22":
            continue
        transitions = tuple(
            replace(
                report,
                discovery_seeds=reference[f"11-{run.condition}"].transitions[transition_index].discovery_seeds,
                confirmation_seeds=reference[f"11-{run.condition}"].transitions[transition_index].confirmation_seeds,
            )
            for transition_index, report in enumerate(run.transitions)
        )
        runs[index] = replace(run, transitions=transitions)

    with pytest.raises(ValueError, match="disjoint task/seed schedules"):
        analyze_rq3_longitudinal_response(runs, bootstrap_replicates=100)


def test_rq3_json_cli_binds_report_to_input_and_never_overwrites(tmp_path, capsys):
    document = {
        "schema_version": 1,
        "runs": [run.to_dict() for run in _rq3_runs()],
    }
    input_path = tmp_path / "rq3-runs.json"
    output_path = tmp_path / "rq3-report.json"
    input_bytes = (json.dumps(document, sort_keys=True, indent=2) + "\n").encode()
    input_path.write_bytes(input_bytes)

    assert len(load_rq3_document(document)) == 9
    assert main([
        "--input", str(input_path), "--output", str(output_path),
        "--bootstrap-replicates", "100",
    ]) == 0
    result = json.loads(output_path.read_text())
    assert result["input_sha256"] == hashlib.sha256(input_bytes).hexdigest()
    assert result["analysis"]["status"] == "descriptive"
    assert main([
        "--input", str(input_path), "--bootstrap-replicates", "100",
    ]) == 0
    stdout_result = json.loads(capsys.readouterr().out)
    assert stdout_result["input_sha256"] == result["input_sha256"]

    with pytest.raises(SystemExit) as error:
        main([
            "--input", str(input_path), "--output", str(output_path),
            "--bootstrap-replicates", "100",
        ])
    assert error.value.code == 2
    assert json.loads(output_path.read_text()) == result


def test_rq3_json_interchange_rejects_unknown_fields_and_bad_schema():
    document = {
        "schema_version": 1,
        "runs": [run.to_dict() for run in _rq3_runs()],
    }
    with pytest.raises(ValueError, match="schema_version and runs"):
        load_rq3_document({**document, "note": "unregistered metadata"})
    with pytest.raises(ValueError, match="unsupported RQ3 input schema_version"):
        load_rq3_document({**document, "schema_version": 2})


def test_rq3_transition_summary_is_recomputed_from_serialized_crossplay_matrices():
    document = json.loads(json.dumps({
        "schema_version": 1,
        "runs": [run.to_dict() for run in _rq3_runs()],
    }))
    transition = document["runs"][0]["transitions"][0]
    assert transition["discovery_matrix"]["cells"]
    transition["old_customer_old_service_rate"] = 0.123

    with pytest.raises(ValueError, match="does not match its source matrices"):
        load_rq3_document(document)
