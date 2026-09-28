from __future__ import annotations

from dataclasses import replace

import pytest

from evotau.crossplay import CrossPlayCell, CrossPlayMatrix
from evotau.records import customer_strategy_id, service_strategy_id
from evotau.research_analysis import AdaptationOutcome, analyze_adaptation_response
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
