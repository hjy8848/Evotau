from __future__ import annotations

import json
import os
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from types import SimpleNamespace

import pytest

from evotau.budget import RequestBudget
from evotau.manifest import ExperimentManifest, MechanismManifest
from evotau.native_runner import (
    IndependentEpisodeAudit,
    TauBenchEpisodeRunner,
    _validate_phase0_parent,
    run_native_phase3,
)
from evotau.phase0 import load_config
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
    assert stored_record["trajectory_ref"] == record.trajectory_ref


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
        )

    runner, _budget, _provider = configured_runner(tmp_path, monkeypatch, audit_provider=audit)
    record = runner(
        task_id="73", seed=42, customer=None, service=ServiceStrategy(), panel_name="clean",
    )
    assert record.status == EpisodeStatus.COMPLETE
    assert record.customer_strategy_id == customer_strategy_id(None)
    assert record.strategy_applicable is False
    assert record.customer_strategy_adherent is None
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
    assert budget.snapshot().attempts == 0
