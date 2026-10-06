from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from evotau.alternating_manifest import AlternatingManifest
from evotau.budget import RequestBudget
from evotau.records import (
    EpisodeRecord,
    EpisodeStatus,
    customer_strategy_id,
    service_strategy_id,
)
from evotau.service_repair_run import (
    ARCHIVED_E_TASK_IDS,
    _validate_continuation_config,
    load_archived_c0_s0_evidence,
    load_service_repair_tasks,
    paired_outcome_transitions,
    run_service_repair_from_evidence,
)
from evotau.strategies import PromptStrategy

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "experiments/results/evotau-alternating-gpt61sol-e20-c1-mechanism-smoke-20261006-partial"
S0_FAILURE_TASK_IDS = {"29", "52", "67", "92"}


def _evidence(tmp_path: Path):
    archive = tmp_path / "archive"
    archive.mkdir()
    customer = PromptStrategy("")
    service = PromptStrategy("")
    records = []
    for index, task_id in enumerate(ARCHIVED_E_TASK_IDS):
        episode_id = f"saved-{task_id}"
        episode_dir = archive / "episodes" / episode_id
        episode_dir.mkdir(parents=True)
        simulation = {
            "id": episode_id,
            "task_id": task_id,
            "seed": 1,
            "termination_reason": "user_stop",
            "messages": [{"role": "assistant", "content": f"observed {task_id}"}],
        }
        (episode_dir / "native-simulation.json").write_text(json.dumps(simulation))
        records.append(EpisodeRecord(
            episode_id=episode_id,
            task_id=task_id,
            seed=1,
            customer_strategy_id=customer_strategy_id(customer),
            service_strategy_id=service_strategy_id(service),
            status=EpisodeStatus.COMPLETE,
            task_success=task_id not in S0_FAILURE_TASK_IDS,
            native_reward=0.0 if task_id in S0_FAILURE_TASK_IDS else 1.0,
            termination_reason="user_stop",
            trajectory_ref=f"episodes/{episode_id}/native-simulation.json",
        ))
    from evotau.service_repair_run import ArchivedServiceEvidence

    return ArchivedServiceEvidence(
        archive_root=archive,
        manifest_sha256="source-sha",
        task_ids=ARCHIVED_E_TASK_IDS,
        seed=1,
        customer=customer,
        service=service,
        accuracy=0.8,
        episodes=tuple(records),
    )


class _CandidateRunner:
    def __init__(self, successes: set[str] | None = None) -> None:
        self.calls: list[dict] = []
        self.successes = successes if successes is not None else set(ARCHIVED_E_TASK_IDS)

    def __call__(self, *, task_id, seed, customer, service, panel_name):
        self.calls.append({
            "task_id": task_id,
            "seed": seed,
            "customer": customer.text,
            "service": service.text,
            "panel_name": panel_name,
        })
        return EpisodeRecord(
            episode_id=f"new-{task_id}",
            task_id=task_id,
            seed=seed,
            customer_strategy_id=customer_strategy_id(customer),
            service_strategy_id=service_strategy_id(service),
            status=EpisodeStatus.COMPLETE,
            task_success=task_id in self.successes,
            native_reward=1.0 if task_id in self.successes else 0.0,
            termination_reason="user_stop",
            trajectory_ref=f"episodes/new-{task_id}/native-simulation.json",
        )


def _budgeted_proposal(budget: RequestBudget, *, strategy: str):
    def propose(context):
        assert context["selected_customer_accuracy"] == 0.8
        assert context["customer_strategy"] == ""
        assert context["current_service_strategy"] == ""
        assert len(context["task_interactions"]) == 20
        assert all("user_scenario" not in item["task"] for item in context["task_interactions"])
        response = budget.dispatch_external_call(
            model="gpt-6.1-sol",
            call_name="evotau_service_evolver",
            dispatch=lambda: {
                "analysis": "Repair failed service tool flow without weakening successful tasks.",
                "strategy": strategy,
                "usage": {"input_tokens": 123, "output_tokens": 17},
            },
        )
        return response

    return propose


def test_loads_only_frozen_c0_s0_episodes_from_pushed_archive() -> None:
    evidence = load_archived_c0_s0_evidence(ARCHIVE)

    assert evidence.task_ids == ARCHIVED_E_TASK_IDS
    assert len(evidence.episodes) == 20
    assert evidence.accuracy == 0.8
    assert all(item.seed == 1 for item in evidence.episodes)
    assert all(item.customer_strategy_id == item.service_strategy_id for item in evidence.episodes)
    assert all((evidence.archive_root / item.trajectory_ref).is_file() for item in evidence.episodes)


def test_continuation_config_uses_only_service_evolver_medium_fallback() -> None:
    raw = yaml.safe_load(
        (ROOT / "configs/service-repair-baseline-gpt61sol-e20-20261006.yaml")
        .read_text(encoding="utf-8"),
    )
    manifest = AlternatingManifest.from_mapping(raw)
    _validate_continuation_config(manifest)

    assert dict(manifest.role_models) == {
        "agent": "openai/deepseek-v4-flash",
        "customer": "openai/deepseek-v4-flash",
        "evaluator": "openai/deepseek-v4-flash",
        "evolver": "gpt-6.1-sol",
    }
    assert manifest.role_model_args_dict["evolver"]["reasoning_effort"] == "medium"
    assert manifest.max_parallel_episodes == 1
    assert manifest.evolution_task_ids == ARCHIVED_E_TASK_IDS
    assert manifest.evolution_fitness_seed == 1
    assert manifest.run_validation is False
    assert manifest.run_heldout is False


def test_continuation_loads_e_only_and_never_loads_v_or_h(monkeypatch) -> None:
    from evotau import alternating_run

    captured = {}

    def fake_loader(manifest, data_dir, *, include_validation, include_heldout):
        captured.update({
            "data_dir": data_dir,
            "include_validation": include_validation,
            "include_heldout": include_heldout,
        })
        return {task_id: object() for task_id in manifest.evolution_task_ids}

    monkeypatch.setattr(alternating_run, "load_alternating_tasks", fake_loader)
    manifest = SimpleNamespace(evolution_task_ids=("66", "92"))
    tasks = load_service_repair_tasks(manifest, "/pinned/tau2")

    assert set(tasks) == {"66", "92"}
    assert captured == {
        "data_dir": "/pinned/tau2",
        "include_validation": False,
        "include_heldout": False,
    }


def test_service_proposal_runs_one_frozen_c0_candidate_panel_and_writes_paired_transitions(
    tmp_path,
) -> None:
    evidence = _evidence(tmp_path)
    tasks = {
        task_id: SimpleNamespace(id=task_id, description=f"task {task_id}", user_tools=[], user_scenario="hidden")
        for task_id in evidence.task_ids
    }
    budget = RequestBudget(cap=None)
    runner = _CandidateRunner(successes=set(evidence.task_ids) - {"29", "92"})
    proposal_calls = 0
    propose_once = _budgeted_proposal(budget, strategy="S1 focused repair")

    def evolver(context):
        nonlocal proposal_calls
        proposal_calls += 1
        return propose_once(context)

    result = run_service_repair_from_evidence(
        evidence=evidence,
        tasks=tasks,
        runner=runner,
        domain_policy="pinned Retail policy",
        service_evolver=evolver,
        output_directory=tmp_path / "output",
        manifest_sha256="continuation-sha",
        model="gpt-6.1-sol",
        reasoning_effort="medium",
        api_base="https://inferaiapi.com/v1",
        request_budget=budget,
        max_parallel_episodes=1,
    )

    assert proposal_calls == 1
    assert len(runner.calls) == 20
    assert {call["panel_name"] for call in runner.calls} == {"service-repair-baseline-c0-x-s1"}
    assert {call["customer"] for call in runner.calls} == {""}
    assert {call["service"] for call in runner.calls} == {"S1 focused repair"}
    assert {call["seed"] for call in runner.calls} == {1}
    assert result["status"] == "complete"
    assert result["s0_accuracy"]["accuracy"] == 0.8
    assert result["s1_accuracy"]["accuracy"] == 0.9
    assert result["net_accuracy_delta"] == 0.1
    assert result["transition_counts"]["repaired_task_count"] == 2
    assert result["transition_counts"]["regressed_task_count"] == 0
    assert result["transition_counts"]["unchanged_pass_count"] == 16
    assert result["transition_counts"]["unchanged_fail_count"] == 2
    assert result["provider"]["attempt_count"] == 1
    assert result["provider"]["usage"] == {
        "status": "available", "prompt_tokens": 123, "completion_tokens": 17,
    }
    assert json.loads((tmp_path / "output/service-evolver-call.json").read_text())["attempt_count"] == 1
    input_artifact = json.loads((tmp_path / "output/service-evolver-input.json").read_text())
    assert "user_scenario" not in json.dumps(input_artifact["context"])


def test_paired_transition_diagnostics_cover_all_four_outcomes() -> None:
    def record(task_id: str, success: bool) -> EpisodeRecord:
        return EpisodeRecord(
            episode_id=f"{task_id}-{success}", task_id=task_id, seed=1,
            customer_strategy_id="c", service_strategy_id="s",
            status=EpisodeStatus.COMPLETE, task_success=success,
        )

    result = paired_outcome_transitions(
        [record("repair", False), record("regress", True), record("keep", True), record("remain", False)],
        [record("repair", True), record("regress", False), record("keep", True), record("remain", False)],
    )

    assert [row["transition"] for row in result["paired_outcomes"]] == [
        "fail → pass", "pass → fail", "pass → pass", "fail → fail",
    ]
    assert result["transition_counts"] == {
        "fail → pass": 1,
        "pass → fail": 1,
        "pass → pass": 1,
        "fail → fail": 1,
        "repaired_task_count": 1,
        "regressed_task_count": 1,
        "unchanged_pass_count": 1,
        "unchanged_fail_count": 1,
    }


def test_noop_service_proposal_does_not_dispatch_duplicate_e20(tmp_path) -> None:
    evidence = _evidence(tmp_path)
    tasks = {
        task_id: SimpleNamespace(id=task_id, description=f"task {task_id}", user_tools=[], user_scenario="hidden")
        for task_id in evidence.task_ids
    }
    budget = RequestBudget(cap=None)
    runner = _CandidateRunner()
    result = run_service_repair_from_evidence(
        evidence=evidence,
        tasks=tasks,
        runner=runner,
        domain_policy="pinned Retail policy",
        service_evolver=_budgeted_proposal(budget, strategy=evidence.service.text),
        output_directory=tmp_path / "noop",
        manifest_sha256="continuation-sha",
        model="gpt-6.1-sol",
        reasoning_effort="medium",
        api_base="https://inferaiapi.com/v1",
        request_budget=budget,
        max_parallel_episodes=1,
    )

    assert result["status"] == "no_op"
    assert result["s1_accuracy"] is None
    assert result["s1_episode_refs"] == []
    assert runner.calls == []


def test_provider_failure_is_saved_and_a_second_dispatch_is_refused(tmp_path) -> None:
    evidence = _evidence(tmp_path)
    tasks = {
        task_id: SimpleNamespace(id=task_id, description=f"task {task_id}", user_tools=[], user_scenario="hidden")
        for task_id in evidence.task_ids
    }
    output = tmp_path / "failed"
    budget = RequestBudget(cap=None)
    runner = _CandidateRunner()
    calls = 0

    def failed_provider(_context):
        nonlocal calls
        calls += 1

        def dispatch():
            raise RuntimeError("HTTP 504 Gateway Timeout")

        return budget.dispatch_external_call(
            model="gpt-6.1-sol",
            call_name="evotau_service_evolver",
            dispatch=dispatch,
        )

    kwargs = {
        "evidence": evidence,
        "tasks": tasks,
        "runner": runner,
        "domain_policy": "pinned Retail policy",
        "service_evolver": failed_provider,
        "output_directory": output,
        "manifest_sha256": "continuation-sha",
        "model": "gpt-6.1-sol",
        "reasoning_effort": "medium",
        "api_base": "https://inferaiapi.com/v1",
        "request_budget": budget,
        "max_parallel_episodes": 1,
    }
    import pytest

    with pytest.raises(RuntimeError, match="504"):
        run_service_repair_from_evidence(**kwargs)

    failure = json.loads((output / "service-repair-result.json").read_text())
    assert failure["status"] == "failed_partial"
    assert failure["s0_accuracy"]["accuracy"] == 0.8
    assert failure["s1_accuracy"] is None
    assert failure["provider"]["attempt_count"] == 1
    assert failure["provider"]["usage"]["status"] == "unavailable"
    assert runner.calls == []

    with pytest.raises(RuntimeError, match="refusing to dispatch a duplicate request"):
        run_service_repair_from_evidence(**kwargs)
    assert calls == 1
