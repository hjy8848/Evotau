"""Calibration regressions: objective separation, finite V, and task×seed protection."""

from copy import deepcopy

import pytest
from test_skill_evolution_v2 import FakeProviders, FakeRunner, record, run

from evotau.evolution_archive import EvolutionArchive
from evotau.evolution_gate import cheap_screen, evaluate_gate
from evotau.skill_evolution_config import DEFAULT_V2


def finite_policy():
    policy = deepcopy(DEFAULT_V2["statistical_gate"])
    policy.update(method="finite_panel_paired", bootstrap_samples=100)
    return policy


def test_current_improves_replayed_and_native_tie_is_accepted():
    policy = finite_policy()
    current = [record(str(t), t >= 25, s) for t in range(100) for s in (1, 2)]
    repair = [record(str(t), t >= 15, s) for t in range(100) for s in (1, 2)]
    history = [record(str(t), t >= 20, s) for t in range(100) for s in (1, 2)]
    native = [record(str(t), t >= 5, s) for t in range(100) for s in (1, 2)]
    results = [
        evaluate_gate(current, repair, policy, objective="superiority"),
        evaluate_gate(history, history, policy, objective="preservation"),
        evaluate_gate(native, native, policy, objective="preservation"),
    ]
    assert all(r["verdict"] == "ACCEPTED" for r in results)
    assert results[0]["paired_delta"] == 0.1
    assert all(
        r["inference_scope"] == "frozen_tasks_observed_seeds_only"
        and not r["population_risk_certified"]
        for r in results
    )


def test_V3_finite_panel_is_not_min_tasks_relaxation_or_population_claim():
    old = [record(str(t), True, s) for t in range(3) for s in (1, 2, 3, 4)]
    policy = finite_policy()
    result = evaluate_gate(old, old, policy, objective="preservation", looks=12)
    assert policy["min_tasks"] == 8 and result["task_count"] == 3
    assert result["verdict"] == "ACCEPTED" and result["harmfulness_ci"][1] > 0.05
    assert not result["population_risk_certified"]
    assert (
        evaluate_gate(
            old, old, DEFAULT_V2["statistical_gate"], objective="preservation"
        )["verdict"]
        == "INCONCLUSIVE"
    )
    assert (
        evaluate_gate(old, old, policy, objective="superiority")["verdict"]
        == "REJECTED"
    )


def test_preservation_rejects_loss_even_when_aggregate_improves():
    before = (
        [record("a", False, s) for s in (1, 2)]
        + [record("b", False, s) for s in (1, 2)]
        + [record("c", True, s) for s in (1, 2)]
    )
    after = (
        [record("a", True, s) for s in (1, 2)]
        + [record("b", True, s) for s in (1, 2)]
        + [record("c", False, s) for s in (1, 2)]
    )
    result = evaluate_gate(before, after, finite_policy(), objective="preservation")
    assert result["paired_delta"] > 0 and result["verdict"] == "REJECTED"
    assert result["regression_cells"] == [
        {"task_id": "c", "seed": 1},
        {"task_id": "c", "seed": 2},
    ]


def test_screen_protects_passing_seed_of_target_and_clean_cells():
    old = [record("22", False, 1), record("22", True, 2), record("clean", True, 1)]
    new = [record("22", True, 1), record("22", False, 2), record("clean", True, 1)]
    screen = cheap_screen(old, new, {"22"}, set())
    assert (
        not screen["passed"]
        and screen["fixed_cells"] == 1
        and screen["protected_regressions"] == 1
    )
    assert screen["broken_cells"] == [{"task_id": "22", "seed": 2}]
    assert {"task_id": "clean", "seed": 1} in screen["protected_cells"]


def test_finite_gate_unknown_and_single_seed_never_certify():
    assert (
        evaluate_gate(
            [record("1", True)],
            [record("1", None)],
            finite_policy(),
            objective="preservation",
        )["verdict"]
        == "INCONCLUSIVE"
    )
    assert (
        evaluate_gate([record("1", False)], [record("1", True)], finite_policy())[
            "verdict"
        ]
        == "INCONCLUSIVE"
    )
    with pytest.raises(ValueError):
        evaluate_gate(
            [record("1", True)],
            [record("1", True)],
            finite_policy(),
            objective="anything",
        )


def test_population_preservation_can_accept_tie_without_requiring_superiority():
    old = [record(str(t), True, s) for t in range(200) for s in (1, 2)]
    result = evaluate_gate(
        old, old, DEFAULT_V2["statistical_gate"], objective="preservation"
    )
    assert (
        result["verdict"] == "ACCEPTED"
        and result["paired_delta"] == 0
        and result["population_risk_certified"]
    )


def test_formal_orchestrator_requires_E_superiority_but_V_preservation(tmp_path):
    policy = deepcopy(DEFAULT_V2)
    policy["statistical_gate"].update(finite_policy())
    result, _, _ = run(tmp_path, policy=policy, validation=True)
    gate = result.generations[0]["service_phase"]["candidates"][0]["gate"]
    assert gate["repair_superiority"]["objective"] == "superiority"
    assert gate["repair_superiority"]["panel"] == "E"
    assert gate["verdict"] == "ACCEPTED" and result.service.skills
    assert all(
        g["objective"] == "preservation" and g["panel"] == "V"
        for g in gate["opponents"]
    )
    assert all(g["paired_delta"] == 0 for g in gate["opponents"])


def test_smoke_orchestrator_assigns_distinct_opponent_objectives(tmp_path):
    class Captured(FakeProviders):
        pass

    result, _, _ = run(tmp_path, provider=Captured(), runner=FakeRunner())
    gates = result.generations[0]["service_phase"]["candidates"][0]["gate"]["opponents"]
    assert (
        next(g for g in gates if g["opponent"] == "current")["objective"]
        == "superiority"
    )
    assert all(
        g["objective"] == "preservation" for g in gates if g["opponent"] != "current"
    )


def test_archive_never_pareto_prunes_full_E_against_screen_accuracy():
    # Higher screen accuracy is not evidence of domination of a full-E candidate.
    effect = {
        "candidate_accuracy": 0.5,
        "helpfulness": 0.5,
        "harmfulness": 0,
        "new_stuck_rate": 0,
        "token_delta": 0,
        "accepted": False,
        "fail_to_pass": ["1"],
    }
    full = {
        "mutation_id": "full",
        "generation": 0,
        "effect": effect,
        "comparison_key": "E",
        "evaluation_scope": "full_E",
    }
    screen = {
        "mutation_id": "screen",
        "generation": 1,
        "effect": {**effect, "candidate_accuracy": 1.0},
        "comparison_key": "screen",
        "evaluation_scope": "paired_screen_E",
    }
    archive = EvolutionArchive(max_candidates=1)
    archive.add(full)
    archive.add(screen)
    assert archive.entries[0]["mutation_id"] == "full"
