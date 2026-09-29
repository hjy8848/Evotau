from __future__ import annotations

import json
from dataclasses import replace

import pytest

from evotau.baselines import (
    StaticCustomerPortfolio,
    StaticCustomerSchedule,
    propose_random_mutation_candidates,
)
from evotau.records import EpisodeRecord, EpisodeStatus
from evotau.strategies import CustomerStrategy


@pytest.fixture
def portfolio() -> StaticCustomerPortfolio:
    return StaticCustomerPortfolio(
        portfolio_id="rq1-static-customer-v1",
        strategies=(
            CustomerStrategy(),
            CustomerStrategy(disclosure="related_on_request"),
            CustomerStrategy(request_order="reverse_independent"),
            CustomerStrategy(challenge_style="ask_reason", challenge_budget=1),
            CustomerStrategy(challenge_style="rephrase_request", challenge_budget=1),
            CustomerStrategy(challenge_style="rephrase_request", challenge_budget=2),
        ),
    )


def test_static_portfolio_freezes_all_field_levels_and_round_trips(portfolio):
    payload = json.loads(json.dumps(portfolio.to_dict()))
    restored = StaticCustomerPortfolio.from_dict(payload)

    assert restored == portfolio
    assert restored.sha256 == portfolio.sha256
    assert len(set(restored.strategy_ids)) == 6
    with pytest.raises(ValueError, match="cover all field levels"):
        StaticCustomerPortfolio(
            portfolio_id="weak-control",
            strategies=(CustomerStrategy(),),
        )


def test_static_schedule_balances_every_strategy_and_verifies_episode_records(portfolio):
    schedule = portfolio.schedule(task_ids=("E1", "E2"), seeds=(11, 22, 33))
    restored = StaticCustomerSchedule.from_dict(json.loads(json.dumps(schedule.to_dict())))
    restored.validate_portfolio(portfolio)
    assert restored == schedule
    assert {item[2] for item in restored.assignments} == set(portfolio.strategy_ids)

    episodes = tuple(
        EpisodeRecord(
            episode_id=f"episode-{index}",
            task_id=task_id,
            seed=seed,
            customer_strategy_id=strategy_id,
            service_strategy_id="frozen-service",
            status=EpisodeStatus.COMPLETE,
            task_success=True,
            customer_valid=True,
            strategy_applicable=True,
            customer_strategy_adherent=True,
        )
        for index, (task_id, seed, strategy_id) in enumerate(restored.assignments)
    )
    restored.validate_records(episodes)
    with pytest.raises(ValueError, match="outside its frozen assignment"):
        restored.validate_records((replace(episodes[0], customer_strategy_id="0" * 16), *episodes[1:]))

    with pytest.raises(ValueError, match="every frozen portfolio strategy"):
        portfolio.schedule(task_ids=("E1",), seeds=(11,))


def test_static_schedule_is_bound_to_exact_order_and_rejects_tampering(portfolio):
    schedule = portfolio.schedule(task_ids=("E1", "E2"), seeds=(11, 22, 33))
    reversed_schedule = portfolio.schedule(task_ids=("E2", "E1"), seeds=(11, 22, 33))
    assert reversed_schedule.sha256 != schedule.sha256

    payload = schedule.to_dict()
    payload["assignments"][0]["strategy_id"] = portfolio.strategy_ids[-1]
    with pytest.raises(ValueError, match="balanced rotation"):
        StaticCustomerSchedule.from_dict(payload)


def test_static_panel_schedule_assigns_balanced_strategies_to_repeated_matched_slots(portfolio):
    schedule = portfolio.panel_schedule(
        task_ids=("E1", "E2"), seeds=(11, 22, 33), repeats_per_pair=3,
    )
    schedule.validate_portfolio(portfolio)
    assert len(schedule.assignments) == 18
    assert {item[3] for item in schedule.assignments} == set(portfolio.strategy_ids)
    assert schedule.strategy_for("E2", 22, 1) == next(
        strategy for task, seed, repeat, strategy in schedule.assignments
        if (task, seed, repeat) == ("E2", 22, 1)
    )
    with pytest.raises(ValueError, match="repeats_per_pair"):
        portfolio.panel_schedule(task_ids=("E1",), seeds=(11,), repeats_per_pair=0)


def test_random_mutation_baseline_has_deterministic_unconditioned_proposals():
    first = propose_random_mutation_candidates(CustomerStrategy(), 2, seed=421)
    second = propose_random_mutation_candidates(CustomerStrategy(), 2, seed=421)

    assert first == second
    assert len({item.strategy_id for item in first}) == 2
    assert all(item.rationale == "random_mutation" for item in first)
    assert all(item.supporting_failure_ids == () for item in first)
    assert all(len(item.changed_fields) in {1, 2} for item in first)
