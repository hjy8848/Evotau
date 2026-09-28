from __future__ import annotations

import json
import os
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from evotau.budget import RequestBudget
from evotau.checkpoint import manifest_fingerprint
from evotau.manifest import ExperimentManifest, MechanismManifest
from evotau.native_runner import (
    IndependentEpisodeAudit,
    TauBenchEpisodeRunner,
    _validate_phase0_parent,
    run_native_phase3,
)
from evotau.phase0 import load_config
from evotau.phase0_run import _load_pinned_task, execute_phase0
from evotau.records import EpisodeStatus, EvidenceRef, customer_strategy_id
from evotau.strategies import CustomerStrategy, ServiceStrategy

ROOT = Path(__file__).resolve().parents[1]


class FakeSimulation:
    id = "native-simulation-73"
    task_id = "73"

    def model_dump(self, *, mode: str) -> dict:
        assert mode == "json"
        return {
            "id": self.id,
            "task_id": self.task_id,
            "seed": 42,
            "termination_reason": "agent_stop",
            "reward_info": {"reward": 0.0},
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "tool", "content": "lookup"},
                {"role": "multi_tool", "tool_messages": [{}, {}]},
            ],
            "review": {"agent_errors": []},
            "auth_classification": {"result": "safe"},
        }


def configured_runner(tmp_path: Path, monkeypatch, *, audit_provider=None):
    monkeypatch.chdir(tmp_path)
    config = load_config(ROOT / "configs/phase3-mechanism.yaml")
    experiment = config["experiment"]
    experiment["real_provider_enabled"] = True
    experiment["models"] = {
        "agent": "mock-agent",
        "customer": "mock-customer",
        "reviewer": "mock-reviewer",
    }
    experiment["output_path"] = "runs/native-phase3-test"
    experiment["checkpoint_path"] = "checkpoints/native-phase3-test"
    manifest = MechanismManifest.from_mapping(config)
    budget = RequestBudget(manifest.request_budget_cap)

    def fake_builder(**kwargs):
        return SimpleNamespace(
            agent=SimpleNamespace(system_prompt="native policy"),
            user=SimpleNamespace(system_prompt="native guidelines"),
            task=kwargs["task"],
        )

    provider = SimpleNamespace(
        completion=lambda **_kwargs: SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3)
        ),
        DEFAULT_MAX_RETRIES=7,
    )

    def fake_run_with_budget(
        orchestrator, request_budget, *, reviewer_model, on_simulation, after_review
    ):
        assert reviewer_model == "mock-reviewer"
        simulation = FakeSimulation()
        on_simulation(simulation)
        with request_budget.instrument_tau_llm_utils(provider):
            provider.completion(model="mock-agent")
            if after_review is not None:
                after_review(simulation, orchestrator)
        return simulation

    monkeypatch.setattr(
        "evotau.native_runner._load_pinned_tasks",
        lambda *_args, **_kwargs: {"73": SimpleNamespace(id="73")},
    )
    monkeypatch.setattr("evotau.native_runner.build_phase0_orchestrator", fake_builder)
    monkeypatch.setattr("evotau.native_runner.run_with_budget", fake_run_with_budget)
    policy_path = tmp_path / "unused-pinned-data/tau2/domains/retail/policy.md"
    policy_path.parent.mkdir(parents=True)
    policy_path.write_text("fixture Retail policy", encoding="utf-8")
    runner = TauBenchEpisodeRunner(
        manifest=manifest,
        config=config,
        data_dir=tmp_path / "unused-pinned-data",
        request_budget=budget,
        audit_provider=audit_provider,
    )
    return runner, budget, provider


def test_native_runner_records_trajectory_review_independent_audit_and_shared_budget(
    tmp_path: Path, monkeypatch
):
    provider = None
    audit_attempts = []

    def audit(_simulation, _task, _customer, _service, panel_name):
        assert panel_name == "discovery"
        assert provider.DEFAULT_MAX_RETRIES == 0
        audit_attempts.append(True)
        provider.completion(model="mock-independent-auditor")
        return IndependentEpisodeAudit(
            verifier_ref="human-audit:review-17",
            customer_valid=True,
            strategy_applicable=True,
            customer_strategy_adherent=True,
            policy_violation=True,
            invalid_repeated_write_calls=0,
            policy_rule_id="retail.policy:explicit_confirmation",
            mistake_type="missing_explicit_confirmation",
            workflow_stage="pre_write",
            evidence=(EvidenceRef(2, "tool", "write occurred before confirmation"),),
        )

    runner, budget, provider = configured_runner(tmp_path, monkeypatch, audit_provider=audit)
    record = runner(
        task_id="73", seed=42, customer=CustomerStrategy(),
        service=ServiceStrategy(), panel_name="discovery",
    )
    assert record.status == EpisodeStatus.COMPLETE
    assert record.task_success is False
    assert record.has_attributable_failure_candidate
    assert record.audit_ref == "human-audit:review-17"
    assert record.tool_calls == 3
    assert record.invalid_repeated_write_calls == 0
    assert record.raw_review["native_review"] == {"agent_errors": []}
    assert audit_attempts == [True]
    assert budget.snapshot().attempts == 2
    assert budget.snapshot().prompt_tokens == 20
    assert budget.snapshot().completion_tokens == 6
    assert provider.DEFAULT_MAX_RETRIES == 7

    run_directory = runner.output_directory / Path(record.trajectory_ref).parent
    trajectory = json.loads((run_directory / "native-simulation.json").read_text(encoding="utf-8"))
    telemetry = json.loads((run_directory / "run-telemetry.json").read_text(encoding="utf-8"))
    stored_record = json.loads((run_directory / "episode-record.json").read_text(encoding="utf-8"))
    assert trajectory["id"] == record.episode_id
    assert telemetry["budget_delta"]["attempts"] == 2
    assert telemetry["independent_audit"]["verifier_ref"] == record.audit_ref
    assert telemetry["independent_audit"]["invalid_repeated_write_calls"] == 0
    assert stored_record["trajectory_ref"] == record.trajectory_ref
    loaded = runner.load_trajectory(record)
    assert loaded["id"] == record.episode_id
    assert loaded["task_id"] == record.task_id and loaded["seed"] == record.seed
    escaped = replace(record, trajectory_ref="../../outside/native-simulation.json")
    with pytest.raises(ValueError, match="escapes its run directory"):
        runner.load_trajectory(escaped)


def test_native_runner_without_independent_audit_is_uncertain_and_not_fit_eligible(
    tmp_path: Path, monkeypatch
):
    runner, budget, _provider = configured_runner(tmp_path, monkeypatch)
    record = runner(
        task_id="73", seed=42, customer=CustomerStrategy(),
        service=ServiceStrategy(), panel_name="discovery",
    )
    assert record.status == EpisodeStatus.UNCERTAIN
    assert record.customer_valid is None
    assert not record.has_attributable_failure_candidate
    assert budget.snapshot().attempts == 1


def test_native_runner_can_execute_clean_user_without_strategy_overlay(tmp_path: Path, monkeypatch):
    def audit(_simulation, _task, customer, _service, panel_name):
        assert customer is None
        assert panel_name == "clean"
        return IndependentEpisodeAudit(
            verifier_ref="human-audit:clean-user",
            customer_valid=True,
            strategy_applicable=False,
            customer_strategy_adherent=None,
            policy_violation=False,
            invalid_repeated_write_calls=0,
        )

    runner, _budget, _provider = configured_runner(tmp_path, monkeypatch, audit_provider=audit)
    record = runner(
        task_id="73", seed=42, customer=None, service=ServiceStrategy(), panel_name="clean",
    )
    assert record.status == EpisodeStatus.COMPLETE
    assert record.customer_strategy_id == customer_strategy_id(None)
    assert record.strategy_applicable is False
    assert record.customer_strategy_adherent is None
    assert record.invalid_repeated_write_calls == 0
    assert not record.has_attributable_failure_candidate


def test_native_runner_requires_explicit_provider_opt_in(tmp_path: Path, monkeypatch):
    config = load_config(ROOT / "configs/phase3-mechanism.yaml")
    manifest = MechanismManifest.from_mapping(config)
    with pytest.raises(RuntimeError, match="explicit real_provider_enabled opt-in"):
        TauBenchEpisodeRunner(
            manifest=manifest,
            config=config,
            data_dir=tmp_path,
            request_budget=RequestBudget(manifest.request_budget_cap),
        )


def test_native_phase3_entrypoint_stops_before_loading_or_running_disabled_manifest(tmp_path: Path):
    with pytest.raises(RuntimeError, match="disabled in the frozen manifest"):
        run_native_phase3(
            config_path=ROOT / "configs/phase3-mechanism.yaml",
            data_dir=tmp_path,
            phase0_result_path=tmp_path / "missing-phase0-result.json",
            audit_provider=lambda *_args: None,
            service_transition=lambda *_args: None,
        )


def test_native_phase3_requires_phase0_artifact_and_binds_its_budget(tmp_path: Path):
    config = load_config(ROOT / "configs/mvp.yaml")
    experiment = config["experiment"]
    experiment["real_provider_enabled"] = True
    experiment["models"] = {role: "fixture-model" for role in ("agent", "customer", "reviewer")}
    phase0_manifest = ExperimentManifest.from_mapping(config)
    phase0_dir = tmp_path / "phase0"
    phase0_dir.mkdir()
    (phase0_dir / "manifest.json").write_text(
        json.dumps(phase0_manifest.to_document()), encoding="utf-8"
    )
    simulation = {"id": "phase0-sim", "task_id": "73"}
    (phase0_dir / "native-simulation.json").write_text(json.dumps(simulation), encoding="utf-8")
    result = {
        "schema_version": 1,
        "status": "complete",
        "experiment_id": phase0_manifest.experiment_id,
        "manifest_sha256": phase0_manifest.sha256,
        "task_id": "73",
        "simulation_id": "phase0-sim",
        "simulation_file": "native-simulation.json",
        "provider_budget": {
            "cap": 70, "attempts": 3, "successes": 2, "failures": 1, "denied": 0,
            "prompt_tokens": 101, "completion_tokens": 47, "usage_responses": 2,
            "usage_unavailable": 1, "cache_hits": 0,
        },
    }
    result_path = phase0_dir / "phase0-result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    phase3_config = load_config(ROOT / "configs/phase3-mechanism.yaml")
    phase3_config["experiment"]["models"] = experiment["models"]
    phase3_manifest = MechanismManifest.from_mapping(phase3_config)

    phase0_budget, context = _validate_phase0_parent(result_path, phase3_manifest)

    assert phase0_budget.attempts == 3
    assert phase0_budget.prompt_tokens == 101
    assert context["phase0_simulation_id"] == "phase0-sim"
    assert len(context["phase0_result_sha256"]) == 64

    result["status"] = "incomplete"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="completed schema-version-1"):
        _validate_phase0_parent(result_path, phase3_manifest)


def test_native_runner_loads_pinned_e_and_v_tasks_without_provider_calls(
    tmp_path: Path, monkeypatch
):
    try:
        distribution("tau2")
    except PackageNotFoundError:
        pytest.skip("install the tau-bench optional extra to run native integration checks")
    data_dir = os.environ.get("EVOTAU_TAU2_DATA_DIR")
    if not data_dir:
        pytest.skip("set EVOTAU_TAU2_DATA_DIR to the pinned tau-bench data directory")
    monkeypatch.chdir(tmp_path)
    config = load_config(ROOT / "configs/phase3-mechanism.yaml")
    experiment = config["experiment"]
    experiment["real_provider_enabled"] = True
    experiment["models"] = {
        "agent": "offline-construction-check",
        "customer": "offline-construction-check",
        "reviewer": "offline-construction-check",
    }
    experiment["output_path"] = "runs/native-phase3-integration-test"
    experiment["checkpoint_path"] = "checkpoints/native-phase3-integration-test"
    manifest = MechanismManifest.from_mapping(config)
    budget = RequestBudget(manifest.request_budget_cap)
    runner = TauBenchEpisodeRunner(
        manifest=manifest,
        config=config,
        data_dir=data_dir,
        request_budget=budget,
    )
    assert set(runner.tasks) == {"73", "93"}
    assert {str(task.id) for task in runner.tasks.values()} == {"73", "93"}
    policy_path = Path(data_dir) / "tau2/domains/retail/policy.md"
    assert runner.service_policy_text == policy_path.read_text(encoding="utf-8")
    assert budget.snapshot().attempts == 0


def test_pinned_native_run_simulation_evaluation_and_review_without_provider_calls(
    tmp_path: Path, monkeypatch
):
    """Exercise the pinned runner end to end with a local deterministic completion stub."""
    try:
        distribution("tau2")
    except PackageNotFoundError:
        pytest.skip("install the tau-bench optional extra to run native integration checks")
    data_dir = os.environ.get("EVOTAU_TAU2_DATA_DIR")
    if not data_dir:
        pytest.skip("set EVOTAU_TAU2_DATA_DIR to the pinned tau-bench data directory")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TAU2_DATA_DIR", data_dir)
    from litellm import ModelResponse
    from tau2.utils import llm_utils

    config = load_config(ROOT / "configs/mvp.yaml")
    experiment = config["experiment"]
    experiment["real_provider_enabled"] = True
    experiment["models"] = {
        "agent": "evotau-offline-agent",
        "customer": "evotau-offline-customer",
        "reviewer": "evotau-offline-reviewer",
    }
    experiment["output_path"] = "runs/pinned-native-offline-e2e"
    manifest = ExperimentManifest.from_mapping(config)
    task = _load_pinned_task(
        manifest,
        data_dir=data_dir,
        task_selection=experiment["task_selection"],
    )
    replies = {"evotau-offline-customer": 0, "evotau-offline-agent": 0}
    calls: list[tuple[str, int]] = []

    def completion(*, model: str, messages: list[dict], **kwargs):
        """Mock only the HTTP-facing completion function, never tau2 runtime logic."""
        assert kwargs.get("num_retries") == 0
        calls.append((model, len(messages)))
        if model == "evotau-offline-customer":
            replies[model] += 1
            content = (
                "I would like help returning the items from my recent order. "
                "My email is fatima.wilson5721@example.com."
                if replies[model] == 1
                else "###STOP###"
            )
            message = {"role": "assistant", "content": content}
        elif model == "evotau-offline-agent":
            replies[model] += 1
            if replies[model] == 1:
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call-user-lookup",
                        "type": "function",
                        "function": {
                            "name": "find_user_id_by_email",
                            "arguments": json.dumps({"email": "fatima.wilson5721@example.com"}),
                        },
                    }],
                }
            else:
                message = {"role": "assistant", "content": "I found the account."}
        elif model == "evotau-offline-reviewer":
            system_prompt = messages[0]["content"]
            if "Classify the user authentication outcome" in system_prompt:
                content = json.dumps({"status": "succeeded", "reasoning": "Offline fixture."})
            else:
                content = json.dumps({"errors": [], "summary": "Offline fixture review."})
            message = {"role": "assistant", "content": content}
        else:
            # The pinned task's ALL evaluator may invoke its native NL judge.
            content = json.dumps({"results": [
                {
                    "expectedOutcome": assertion,
                    "reasoning": "Offline fixture; no model judgment was requested.",
                    "metExpectation": False,
                }
                for assertion in (task.evaluation_criteria.nl_assertions or ())
            ]})
            message = {"role": "assistant", "content": content}
        return ModelResponse(
            model=model,
            choices=[{
                "index": 0,
                "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                "message": message,
            }],
            usage={"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
        )

    monkeypatch.setattr(llm_utils, "completion", completion)
    monkeypatch.setattr(llm_utils, "get_response_cost", lambda _response: 0.0)
    result = execute_phase0(
        manifest=manifest,
        config=config,
        task=task,
    )

    output = tmp_path / experiment["output_path"]
    simulation = json.loads((output / "native-simulation.json").read_text(encoding="utf-8"))
    saved_result = json.loads((output / "phase0-result.json").read_text(encoding="utf-8"))
    assert result["status"] == "complete"
    assert result["simulation_id"] == simulation["id"]
    assert result["native_reward"] == 0.0  # The offline fixture looked up an account, not a return.
    assert result["review"]["summary"] == "Offline fixture review."
    assert result["auth_classification"]["status"] == "succeeded"
    assert set(result["rendered_prompt_sha256"]) == {"agent", "customer"}
    assert saved_result == result
    assert any(
        call.get("name") == "find_user_id_by_email"
        for item in simulation["messages"]
        for call in (item.get("tool_calls") or ())
    )
    assert any(
        item.get("role") == "tool" and item.get("requestor") == "assistant"
        for item in simulation["messages"]
    )
    assert result["provider_budget"]["attempts"] == len(calls)
    assert result["provider_budget"]["attempts"] == (
        result["provider_budget"]["successes"] + result["provider_budget"]["failures"]
    )
    assert result["provider_budget"]["attempts"] >= 6


def test_pinned_native_phase0_to_two_generation_phase3_without_provider_calls(
    tmp_path: Path, monkeypatch,
):
    """Exercise the full Phase 0 → native Phase 3 artifact and budget handoff offline."""
    try:
        distribution("tau2")
    except PackageNotFoundError:
        pytest.skip("install the tau-bench optional extra to run native integration checks")
    data_dir = os.environ.get("EVOTAU_TAU2_DATA_DIR")
    if not data_dir:
        pytest.skip("set EVOTAU_TAU2_DATA_DIR to the pinned tau-bench data directory")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TAU2_DATA_DIR", data_dir)
    from litellm import ModelResponse
    from tau2.utils import llm_utils

    models = {
        "agent": "evotau-full-offline-agent",
        "customer": "evotau-full-offline-customer",
        "reviewer": "evotau-full-offline-reviewer",
    }
    phase0_config = load_config(ROOT / "configs/mvp.yaml")
    phase0_config["experiment"]["real_provider_enabled"] = True
    phase0_config["experiment"]["models"] = models
    phase0_config["experiment"]["output_path"] = "runs/full-offline-phase0"
    phase0_manifest = ExperimentManifest.from_mapping(phase0_config)
    phase0_task = _load_pinned_task(
        phase0_manifest,
        data_dir=data_dir,
        task_selection=phase0_config["experiment"]["task_selection"],
    )

    calls: list[str] = []
    replies = {models["customer"]: 0, models["agent"]: 0}

    def completion(*, model: str, messages: list[dict], **kwargs):
        assert kwargs.get("num_retries") == 0
        calls.append(model)
        if model == models["customer"]:
            replies[model] += 1
            content = (
                "I want help returning items from my recent order. "
                "My email is fatima.wilson5721@example.com."
                if replies[model] % 2 == 1 else "###STOP###"
            )
            message = {"role": "assistant", "content": content}
        elif model == models["agent"]:
            replies[model] += 1
            if replies[model] % 2 == 1:
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": f"call-user-lookup-{replies[model]}",
                        "type": "function",
                        "function": {
                            "name": "find_user_id_by_email",
                            "arguments": json.dumps({"email": "fatima.wilson5721@example.com"}),
                        },
                    }],
                }
            else:
                message = {"role": "assistant", "content": "I found the account."}
        elif model == models["reviewer"]:
            system_prompt = messages[0]["content"]
            if "Classify the user authentication outcome" in system_prompt:
                content = json.dumps({"status": "succeeded", "reasoning": "Offline fixture."})
            else:
                content = json.dumps({"errors": [], "summary": "Offline fixture review."})
            message = {"role": "assistant", "content": content}
        else:
            task = phase0_task
            content = json.dumps({"results": [
                {
                    "expectedOutcome": assertion,
                    "reasoning": "Offline fixture; no model judgment was requested.",
                    "metExpectation": False,
                }
                for assertion in (task.evaluation_criteria.nl_assertions or ())
            ]})
            message = {"role": "assistant", "content": content}
        return ModelResponse(
            model=model,
            choices=[{
                "index": 0,
                "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                "message": message,
            }],
            usage={"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
        )

    monkeypatch.setattr(llm_utils, "completion", completion)
    monkeypatch.setattr(llm_utils, "get_response_cost", lambda _response: 0.0)
    phase0_result = execute_phase0(
        manifest=phase0_manifest,
        config=phase0_config,
        task=phase0_task,
    )

    phase3_config = load_config(ROOT / "configs/phase3-mechanism.yaml")
    phase3_config["experiment"]["real_provider_enabled"] = True
    phase3_config["experiment"]["models"] = models
    phase3_config["experiment"]["output_path"] = "runs/full-offline-phase3"
    phase3_config["experiment"]["checkpoint_path"] = "checkpoints/full-offline-phase3.json"
    phase3_path = tmp_path / "phase3-offline.yaml"
    phase3_path.write_text(yaml.safe_dump(phase3_config, sort_keys=False), encoding="utf-8")
    audit_calls: list[tuple[str, str, int, str]] = []

    def audit_provider(simulation, task, customer, _service, panel_name):
        audit_calls.append((str(simulation.id), str(task.id), int(simulation.seed), panel_name))
        return IndependentEpisodeAudit(
            verifier_ref=f"offline-audit:{simulation.id}",
            customer_valid=True,
            strategy_applicable=customer is not None,
            customer_strategy_adherent=True if customer is not None else None,
            policy_violation=False,
            invalid_repeated_write_calls=0,
        )

    def forbidden_service_transition(*_args):
        raise AssertionError("no-failure fixture should not request a Service repair")

    commits, budget = run_native_phase3(
        config_path=phase3_path,
        data_dir=data_dir,
        phase0_result_path=tmp_path / "runs/full-offline-phase0/phase0-result.json",
        audit_provider=audit_provider,
        service_transition=forbidden_service_transition,
    )

    assert phase0_result["status"] == "complete"
    assert len(commits) == 2
    assert all(not item.customer_evolved and not item.service_evolved for item in commits)
    assert audit_calls
    assert {task_id for _episode_id, task_id, _seed, _panel in audit_calls} == {"73"}
    assert len({episode_id for episode_id, *_rest in audit_calls}) == len(audit_calls)
    assert budget.attempts == len(calls)
    assert budget.attempts > phase0_result["provider_budget"]["attempts"]
    assert budget.attempts <= budget.cap == 1800

    run_context = json.loads(
        (tmp_path / "runs/full-offline-phase3/run-context.json").read_text(encoding="utf-8")
    )
    assert run_context["phase0_result_sha256"]
    checkpoint = json.loads(
        (tmp_path / "checkpoints/full-offline-phase3.json").read_text(encoding="utf-8")
    )
    phase3_manifest = MechanismManifest.from_mapping(phase3_config)
    assert checkpoint["manifest_hash"] == manifest_fingerprint({
        "manifest": phase3_manifest.to_payload(), "run_context": run_context,
    })
    assert checkpoint["state"]["request_budget"]["attempts"] == budget.attempts
