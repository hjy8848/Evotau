from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from evotau import alternating_run
from evotau.alternating import run_alternating_evolution
from evotau.alternating_manifest import AlternatingManifest
from evotau.records import EpisodeRecord, EpisodeStatus
from evotau.strategies import PromptStrategy
from evotau.tau_provenance import role_model_args_for_runtime

ROOT = Path(__file__).resolve().parents[1]


def _config(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_mechanism_smoke_config_freezes_seed_and_enables_eight_workers() -> None:
    config = _config("alternating-mechanism-smoke.yaml")
    manifest = AlternatingManifest.from_mapping(config)

    assert manifest.evolution_fitness_seed == 1
    assert manifest.max_parallel_episodes == 8
    assert manifest.run_validation is False
    assert manifest.run_heldout is False
    assert len(manifest.evolution_task_ids) == 5

    formal = AlternatingManifest.from_mapping(_config("alternating-evolution.yaml"))
    assert formal.evolution_fitness_seed == formal.seed
    assert formal.run_validation is True
    assert formal.run_heldout is True
    assert formal.max_parallel_episodes == 4

    config["experiment"]["max_parallel_episodes"] = 9
    with pytest.raises(ValueError, match="from 1 to 8"):
        AlternatingManifest.from_mapping(config)


def test_gpt_evolver_reasoning_effort_is_validated_and_forwarded_without_changing_roles() -> None:
    config = _config("alternating-mechanism-smoke-gpt61sol-e20-c1.yaml")
    manifest = AlternatingManifest.from_mapping(config)
    runtime_args = role_model_args_for_runtime(manifest.role_model_args)

    assert runtime_args["evolver"] == {
        "api_base": "https://inferaiapi.com/v1",
        "reasoning_effort": "high",
    }
    assert "temperature" not in runtime_args["evolver"]
    expected_deepseek_args = {
        "temperature": 0.0,
        "api_base": "https://inferaiapi.com/v1",
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    assert runtime_args["agent"] == expected_deepseek_args
    assert runtime_args["customer"] == expected_deepseek_args
    assert runtime_args["evaluator"] == expected_deepseek_args

    for invalid in ("none", "ultra", "", 1, None):
        invalid_config = _config("alternating-mechanism-smoke-gpt61sol-e20-c1.yaml")
        invalid_config["experiment"]["model_args"]["evolver"] = {
            "reasoning_effort": invalid,
        }
        with pytest.raises(ValueError, match="reasoning_effort"):
            AlternatingManifest.from_mapping(invalid_config)


def test_mechanism_smoke_accepts_strict_service_e_improvement_without_v() -> None:
    task = SimpleNamespace(id="e", description="E task", user_scenario="Goal.", user_tools=[])

    class FakeRunner:
        service_policy_text = "Fixed policy."

        def __init__(self) -> None:
            self.calls: list[dict] = []

        def __call__(self, *, task_id, seed, customer, service, panel_name):
            self.calls.append({
                "task_id": task_id,
                "seed": seed,
                "panel_name": panel_name,
                "service": service.text,
            })
            succeeded = service.text == "S1"
            return EpisodeRecord(
                episode_id=f"{panel_name}-{task_id}",
                task_id=task_id,
                seed=seed,
                customer_strategy_id="native" if customer is None else customer.text,
                service_strategy_id=service.text,
                status=EpisodeStatus.COMPLETE,
                task_success=succeeded,
                native_reward=1.0 if succeeded else 0.0,
                termination_reason="user_stop",
                trajectory_ref=None,
                raw_review={},
            )

    runner = FakeRunner()
    result = run_alternating_evolution(
        tasks={"e": task},
        evolution_task_ids=("e",),
        validation_task_ids=(),
        seed=1,
        evolution_fitness_seed=7,
        generations=1,
        customer_candidate_count=1,
        clean_panel_size=1,
        run_validation=False,
        initial_customer=PromptStrategy("C0"),
        initial_service=PromptStrategy("S0"),
        runner=runner,
        customer_evolver=lambda _context, _count: ["C0"],
        service_evolver=lambda _context: {"analysis": "E improved", "strategy": "S1"},
        domain_policy=runner.service_policy_text,
    )

    service_phase = result.generations[0]["service_phase"]
    assert service_phase["old_accuracy"] == 0.0
    assert service_phase["proposed_accuracy"] == 1.0
    assert service_phase["improved_on_E"] is True
    assert service_phase["validation_evaluated"] is False
    assert service_phase["acceptance_evaluated"] is True
    assert service_phase["accepted"] is True
    assert service_phase["acceptance_mode"] == "e_only_mechanism_smoke"
    reason = service_phase["selection"]["reason"].lower()
    assert "e-only" in reason
    assert "mechanism smoke" in reason
    assert "validation was intentionally disabled" in reason
    assert result.service.text == "S1"
    assert result.generations[0]["service_after"]["strategy"] == "S1"
    assert "seed" not in result.generations[0]
    assert result.generations[0]["evolution_fitness_seed"] == 7
    assert result.generations[0]["validation_seed"] is None
    assert {call["task_id"] for call in runner.calls} == {"e"}
    assert {call["seed"] for call in runner.calls} == {7}


def test_mechanism_smoke_keeps_old_service_when_e_accuracy_does_not_strictly_improve() -> None:
    task = SimpleNamespace(id="e", description="E task", user_scenario="Goal.", user_tools=[])

    class FakeRunner:
        service_policy_text = "Fixed policy."

        def __call__(self, *, task_id, seed, customer, service, panel_name):
            succeeded = service.text == "S0"
            return EpisodeRecord(
                episode_id=f"{panel_name}-{task_id}",
                task_id=task_id,
                seed=seed,
                customer_strategy_id="native" if customer is None else customer.text,
                service_strategy_id=service.text,
                status=EpisodeStatus.COMPLETE,
                task_success=succeeded,
                native_reward=1.0 if succeeded else 0.0,
                termination_reason="user_stop",
                trajectory_ref=None,
                raw_review={},
            )

    runner = FakeRunner()
    result = run_alternating_evolution(
        tasks={"e": task},
        evolution_task_ids=("e",),
        validation_task_ids=(),
        seed=1,
        generations=1,
        customer_candidate_count=1,
        clean_panel_size=1,
        run_validation=False,
        initial_customer=PromptStrategy("C0"),
        initial_service=PromptStrategy("S0"),
        runner=runner,
        customer_evolver=lambda _context, _count: ["C0"],
        service_evolver=lambda _context: {"analysis": "No strict gain", "strategy": "S1"},
        domain_policy=runner.service_policy_text,
    )

    phase = result.generations[0]["service_phase"]
    assert phase["old_accuracy"] == 1.0
    assert phase["proposed_accuracy"] == 0.0
    assert phase["improved_on_E"] is False
    assert phase["acceptance_mode"] == "e_only_mechanism_smoke"
    assert phase["acceptance_evaluated"] is True
    assert phase["accepted"] is False
    assert result.service.text == "S0"


def test_two_generation_smoke_chain_freezes_accepted_service_for_next_customer_phase() -> None:
    task = SimpleNamespace(id="e", description="E task", user_scenario="Goal.", user_tools=[])

    class FakeRunner:
        service_policy_text = "Fixed policy."

        def __init__(self) -> None:
            self.calls: list[dict] = []

        def __call__(self, *, task_id, seed, customer, service, panel_name):
            customer_text = None if customer is None else customer.text
            self.calls.append({
                "task_id": task_id,
                "seed": seed,
                "customer": customer_text,
                "service": service.text,
                "panel_name": panel_name,
            })
            successful_pair = {
                ("C0", "S0"),
                ("C1", "S1"),
                ("C2", "S2"),
            }
            succeeded = (customer_text, service.text) in successful_pair
            return EpisodeRecord(
                episode_id=f"{panel_name}-{task_id}",
                task_id=task_id,
                seed=seed,
                customer_strategy_id="native" if customer is None else customer.text,
                service_strategy_id=service.text,
                status=EpisodeStatus.COMPLETE,
                task_success=succeeded,
                native_reward=1.0 if succeeded else 0.0,
                termination_reason="user_stop",
                trajectory_ref=None,
                raw_review={},
            )

    runner = FakeRunner()

    result = run_alternating_evolution(
        tasks={"e": task},
        evolution_task_ids=("e",),
        validation_task_ids=(),
        seed=1,
        generations=2,
        customer_candidate_count=1,
        clean_panel_size=1,
        run_validation=False,
        initial_customer=PromptStrategy("C0"),
        initial_service=PromptStrategy("S0"),
        runner=runner,
        customer_evolver=lambda context, _count: (
            ["C1"] if context["generation"] == 0 else ["C2"]
        ),
        service_evolver=lambda context: {
            "analysis": "Strict E gain",
            "strategy": "S1" if context["generation"] == 0 else "S2",
        },
        domain_policy=runner.service_policy_text,
    )

    generation_zero, generation_one = result.generations
    assert generation_zero["service_phase"]["accepted"] is True
    assert generation_zero["service_after"]["strategy"] == "S1"
    assert generation_one["customer_phase"]["frozen_service"]["strategy"] == "S1"
    next_customer_calls = [
        call for call in runner.calls
        if call["panel_name"].startswith("generation-1-customer-")
    ]
    assert next_customer_calls
    assert {call["service"] for call in next_customer_calls} == {"S1"}
    assert generation_one["service_phase"]["accepted"] is True
    assert result.service.text == "S2"
    assert {episode.service_strategy_id for episode in result.final_evolution_episodes} == {"S2"}
    assert {episode.customer_strategy_id for episode in result.final_evolution_episodes} == {"C2"}


def test_run_from_config_writes_complete_mechanism_result_without_v_or_h(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config("alternating-mechanism-smoke.yaml")
    experiment = config["experiment"]
    experiment["id"] = "mechanism-smoke-test"
    experiment["output_path"] = "experiments/runs/mechanism-smoke-test"
    experiment["checkpoint_path"] = "experiments/checkpoints/mechanism-smoke-test.json"
    config_path = tmp_path / "project" / "configs" / "mechanism.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    loaded_panels: list[tuple[bool, bool]] = []
    evolution_tasks = {
        task_id: SimpleNamespace(id=task_id)
        for task_id in experiment["task_selection"]["evolution"]
    }

    def load_tasks(_manifest, _data_dir, *, include_validation=True, include_heldout):
        loaded_panels.append((include_validation, include_heldout))
        assert include_validation is False
        assert include_heldout is False
        return evolution_tasks

    class Runner:
        service_policy_text = "Fixed policy."

        def __init__(self, *, task_objects, **_kwargs):
            self.tasks = dict(task_objects)

    fake_generation = {
        "generation": 0,
        "service_phase": {"validation_evaluated": False},
        "timing": {"validation_wall_clock_seconds": 0.0},
    }
    fake_evolved = SimpleNamespace(
        customer=PromptStrategy("C1"),
        service=PromptStrategy("S0"),
        generations=(fake_generation,),
    )

    def evolve(**kwargs):
        assert kwargs["run_validation"] is False
        assert kwargs["validation_task_ids"] == tuple(
            experiment["task_selection"]["validation"]
        )
        assert set(kwargs["tasks"]) == set(experiment["task_selection"]["evolution"])
        return fake_evolved

    monkeypatch.setattr(alternating_run, "load_alternating_tasks", load_tasks)
    monkeypatch.setattr(alternating_run, "TauBenchEpisodeRunner", Runner)
    monkeypatch.setattr(
        alternating_run,
        "LLMAlternatingEvolvers",
        lambda **_kwargs: SimpleNamespace(customer_candidates=None, service_candidate=None),
    )
    monkeypatch.setattr(alternating_run, "run_alternating_evolution", evolve)
    monkeypatch.setattr(
        alternating_run,
        "propose_fresh_customer_challenge",
        lambda *_args, **_kwargs: pytest.fail("mechanism smoke generated an H-only Customer"),
    )
    monkeypatch.setattr(
        alternating_run,
        "run_final_endpoint_evaluation",
        lambda *_args, **_kwargs: pytest.fail("mechanism smoke ran held-out evaluation"),
    )

    output, result = alternating_run.run_from_config(config_path, tau2_data_dir=tmp_path)

    assert loaded_panels == [(False, False)]
    assert result["status"] == "complete"
    assert result["generations"] == [fake_generation]
    assert result["validation_enabled"] is False
    assert result["validation_evaluated"] is False
    assert result["heldout_evaluated"] is False
    assert result["heldout_endpoint_evaluation"] is None
    assert result["fresh_adaptive_customer"] is None
    assert result["episode_job_telemetry"]["configured_max_parallel_episodes"] == 8
    assert (output / "alternating-result.json").is_file()
