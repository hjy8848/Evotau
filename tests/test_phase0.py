from __future__ import annotations

import json
import os
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from evotau.budget import ProviderBudgetExceeded, RequestBudget
from evotau.eligibility import TaskEligibilityError, validate_smoke_selection
from evotau.manifest import ExperimentManifest, git_blob_sha1, write_manifest_once
from evotau.phase0_run import (
    Phase0ExecutionError,
    _load_pinned_task,
    execute_phase0,
    run_from_config,
)
from evotau.prompts import append_strategy_block
from evotau.strategies import (
    CustomerStrategy,
    ServiceRule,
    ServiceStrategy,
    render_customer_strategy,
    render_service_strategy,
)
from evotau.tau_adapter import (
    build_phase0_orchestrator,
    customer_user_class,
    run_phase0_episode,
    service_agent_class,
)

ROOT = Path(__file__).resolve().parents[1]


class MockTauUser:
    def __init__(self, prompt: str = "native user guidelines\n\n<scenario>fixed facts</scenario>"):
        self._prompt = prompt

    @property
    def system_prompt(self) -> str:
        return self._prompt


class MockTauAgent:
    def __init__(self, prompt: str = "<instructions>base</instructions>\n<policy>fixed</policy>"):
        self._prompt = prompt

    @property
    def system_prompt(self) -> str:
        return self._prompt


def _example_task(task_id: str, *, email: str, order: str, items: int) -> dict:
    item_ids = [f"item-{index}" for index in range(items)]
    return {
        "id": task_id,
        "user_scenario": {
            "instructions": {
                "task_instructions": "You are a customer.",
                "reason_for_call": "You want to return several items." if items > 1 else "You want an exchange.",
                "known_info": f"You name is Person {task_id} and your email is {email}.",
                "unknown_info": None,
            }
        },
        "evaluation_criteria": {
            "actions": [
                {
                    "name": "return_delivered_order_items" if items > 1 else "exchange_delivered_order_items",
                    "arguments": {
                        "order_id": order,
                        "item_ids": item_ids,
                        "new_item_ids": [] if items > 1 else ["replacement"],
                    },
                }
            ]
        },
    }


def test_no_strategy_keeps_native_prompts_byte_for_byte() -> None:
    native_user = MockTauUser()
    native_agent = MockTauAgent()
    adapted_user = customer_user_class(MockTauUser, None)(prompt=native_user.system_prompt)
    adapted_agent = service_agent_class(MockTauAgent, ServiceStrategy())(
        prompt=native_agent.system_prompt
    )
    assert adapted_user.system_prompt == native_user.system_prompt
    assert adapted_agent.system_prompt == native_agent.system_prompt
    assert append_strategy_block(native_user.system_prompt, "") == native_user.system_prompt


def test_customer_strategy_is_a_separate_subordinate_block() -> None:
    strategy = CustomerStrategy(
        disclosure="related_on_request",
        request_order="reverse_independent",
        challenge_style="ask_reason",
        challenge_budget=1,
    )
    native = MockTauUser()
    adapted = customer_user_class(MockTauUser, strategy)(prompt=native.system_prompt)
    assert adapted.system_prompt.startswith(native.system_prompt)
    assert "<evotau_customer_strategy>" in adapted.system_prompt
    assert "native simulation guidelines and the original task scenario" in adapted.system_prompt
    assert "include only additional facts relevant to that question" in adapted.system_prompt


def test_customer_strategy_rejects_inconsistent_challenge_budget() -> None:
    with pytest.raises(ValueError, match="must agree"):
        CustomerStrategy(challenge_style="ask_reason", challenge_budget=0)


def test_service_rules_require_policy_evidence_and_a_model_tokenizer() -> None:
    rule = ServiceRule(
        rule_id="r1",
        policy_ref="retail-policy#confirmation",
        trigger="before a write action",
        required_execution="summarize the complete requested change and obtain explicit confirmation",
        evidence_refs=("failure-1",),
    )
    strategy = ServiceStrategy((rule,))
    with pytest.raises(ValueError, match="token_counter"):
        render_service_strategy(strategy)
    patch = render_service_strategy(strategy, token_counter=lambda text: len(text.split()))
    assert "<evotau_execution_rules>" in patch
    assert "retail-policy#confirmation" in patch
    adapted = service_agent_class(
        MockTauAgent, strategy, token_counter=lambda text: len(text.split())
    )(prompt=MockTauAgent().system_prompt)
    assert adapted.system_prompt.startswith("<instructions>base</instructions>\n<policy>fixed</policy>")
    assert "service execution rules" not in adapted.system_prompt
    assert "<evotau_execution_rules>" in adapted.system_prompt


def test_service_patch_limit_is_enforced() -> None:
    rule = ServiceRule(
        rule_id="r1",
        policy_ref="p",
        trigger="t",
        required_execution="x",
        evidence_refs=("f",),
    )
    with pytest.raises(ValueError, match="limit is 600"):
        render_service_strategy(
            ServiceStrategy((rule,)),
            token_counter=lambda _text: 601,
        )


def test_manifest_hash_is_stable_and_writer_refuses_overwrite(tmp_path: Path) -> None:
    config = __import__("yaml").safe_load((ROOT / "configs/mvp.yaml").read_text(encoding="utf-8"))
    manifest = ExperimentManifest.from_mapping(config)
    assert manifest.sha256 == ExperimentManifest.from_mapping(config).sha256
    assert len(manifest.sha256) == 64
    target = write_manifest_once(tmp_path / "manifest.json", manifest)
    document = json.loads(target.read_text(encoding="utf-8"))
    assert document["manifest_sha256"] == manifest.sha256
    with pytest.raises(FileExistsError):
        write_manifest_once(target, manifest)


def test_saved_phase0_manifest_matches_the_canonical_config() -> None:
    config = __import__("yaml").safe_load((ROOT / "configs/mvp.yaml").read_text(encoding="utf-8"))
    manifest = ExperimentManifest.from_mapping(config)
    saved = json.loads(
        (ROOT / "experiments/manifests/evotau-phase0-retail-smoke.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved == manifest.to_document()


def test_git_blob_sha1_includes_the_git_object_header() -> None:
    assert git_blob_sha1(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


def test_manifest_rejects_a_test_split_task_and_nonzero_retries() -> None:
    config = __import__("yaml").safe_load((ROOT / "configs/mvp.yaml").read_text(encoding="utf-8"))
    config["experiment"]["task_selection"]["evolution"] = ["73"]
    config["experiment"]["task_selection"]["validation"] = ["73"]
    with pytest.raises(ValueError, match="disjoint"):
        ExperimentManifest.from_mapping(config)
    config = __import__("yaml").safe_load((ROOT / "configs/mvp.yaml").read_text(encoding="utf-8"))
    config["experiment"]["provider_retries"] = 1
    with pytest.raises(ValueError, match="retries"):
        ExperimentManifest.from_mapping(config)


def test_smoke_selection_requires_train_tasks_and_distinct_entities() -> None:
    evolution = _example_task("73", email="a@example.com", order="#W0000001", items=4)
    validation = _example_task("93", email="b@example.com", order="#W0000002", items=1)
    result = validate_smoke_selection(
        [evolution, validation],
        {"train": ["73", "93"], "test": ["5"]},
        evolution_task_id="73",
        validation_task_id="93",
    )
    assert result.evolution_task_id == "73"
    assert result.validation_task_id == "93"
    validation["user_scenario"]["instructions"]["known_info"] = (
        "You name is Person 73 and your email is a@example.com."
    )
    with pytest.raises(TaskEligibilityError, match="share a customer or order"):
        validate_smoke_selection(
            [evolution, validation],
            {"train": ["73", "93"], "test": ["5"]},
            evolution_task_id="73",
            validation_task_id="93",
        )


def test_smoke_selection_rejects_known_deception_and_test_ids() -> None:
    task = _example_task("46", email="a@example.com", order="#W0000001", items=4)
    task["user_scenario"]["instructions"]["reason_for_call"] = (
        "When asked for the order ID, provide the wrong number first."
    )
    clean = _example_task("93", email="b@example.com", order="#W0000002", items=1)
    with pytest.raises(TaskEligibilityError, match="explicit exclusion"):
        validate_smoke_selection(
            [task, clean],
            {"train": ["46", "93"], "test": []},
            evolution_task_id="46",
            validation_task_id="93",
        )
    test_task = _example_task("72", email="c@example.com", order="#W0000003", items=4)
    with pytest.raises(TaskEligibilityError, match="official train"):
        validate_smoke_selection(
            [test_task, clean],
            {"train": ["93"], "test": ["72"]},
            evolution_task_id="72",
            validation_task_id="93",
        )


def test_request_budget_blocks_before_dispatch_and_counts_failures() -> None:
    calls: list[str] = []

    def provider(*, model: str, fail: bool = False) -> str:
        calls.append(model)
        if fail:
            raise RuntimeError("mock transport error")
        return "ok"

    module = SimpleNamespace(completion=provider, DEFAULT_MAX_RETRIES=4)
    budget = RequestBudget(cap=2)
    with budget.instrument_tau_llm_utils(module):
        assert module.DEFAULT_MAX_RETRIES == 0
        assert module.completion(model="model-a") == "ok"
        with pytest.raises(RuntimeError, match="mock transport"):
            module.completion(model="model-b", fail=True)
        with pytest.raises(ProviderBudgetExceeded):
            module.completion(model="model-c")
    assert calls == ["model-a", "model-b"]
    assert module.DEFAULT_MAX_RETRIES == 4
    assert budget.snapshot().attempts == 2
    assert budget.snapshot().successes == 1
    assert budget.snapshot().failures == 1
    assert budget.snapshot().denied == 1
    assert budget.snapshot().usage_unavailable == 2
    assert budget.snapshot().remaining == 0


def test_request_budget_records_reported_tokens_and_cache_hits() -> None:
    usage = {
        "usage": {"prompt_tokens": 13, "completion_tokens": 4},
        "_hidden_params": {"cache_hit": True},
    }
    module = SimpleNamespace(completion=lambda **_kwargs: usage, DEFAULT_MAX_RETRIES=0)
    budget = RequestBudget(cap=1)
    with budget.instrument_tau_llm_utils(module):
        module.completion(model="model-a")
    snapshot = budget.snapshot()
    assert snapshot.prompt_tokens == 13
    assert snapshot.completion_tokens == 4
    assert snapshot.usage_responses == 1
    assert snapshot.cache_hits == 1


def test_live_episode_is_blocked_when_manifest_has_provider_disabled() -> None:
    config = __import__("yaml").safe_load((ROOT / "configs/mvp.yaml").read_text(encoding="utf-8"))
    manifest = ExperimentManifest.from_mapping(config)
    with pytest.raises(RuntimeError, match="real provider use is disabled"):
        run_phase0_episode(manifest=manifest, task=SimpleNamespace(user_scenario="scenario"))


def test_phase0_cli_requires_explicit_provider_enable_before_runtime_loading() -> None:
    with pytest.raises(RuntimeError, match="real provider use is disabled"):
        run_from_config(ROOT / "configs/mvp.yaml")


def test_phase0_run_record_is_immutable_and_contains_budget_and_native_result(
    tmp_path: Path, monkeypatch
) -> None:
    config = __import__("yaml").safe_load((ROOT / "configs/mvp.yaml").read_text(encoding="utf-8"))
    experiment = config["experiment"]
    experiment["real_provider_enabled"] = True
    experiment["models"] = {
        "agent": "mock-agent",
        "customer": "mock-customer",
        "reviewer": "mock-reviewer",
    }
    experiment["output_path"] = "runs/phase0-record-test"
    manifest = ExperimentManifest.from_mapping(config)
    monkeypatch.chdir(tmp_path)

    class FakeSimulation:
        id = "simulation-73"
        task_id = "73"

        def model_dump(self, *, mode):
            assert mode == "json"
            return {
                "id": self.id,
                "task_id": self.task_id,
                "termination_reason": "agent_stop",
                "reward_info": {"reward": 1.0},
            }

    def runner(**kwargs):
        budget = kwargs["request_budget"]
        kwargs["on_orchestrator"](
            SimpleNamespace(
                agent=SimpleNamespace(system_prompt="native agent prompt"),
                user=SimpleNamespace(system_prompt="native customer prompt"),
            )
        )
        provider = SimpleNamespace(completion=lambda **_kwargs: "mock", DEFAULT_MAX_RETRIES=0)
        with budget.instrument_tau_llm_utils(provider):
            provider.completion(model="mock-agent")
        return FakeSimulation(), budget

    result = execute_phase0(
        manifest=manifest,
        config=config,
        task=SimpleNamespace(id="73"),
        episode_runner=runner,
    )
    result_path = Path(experiment["output_path"]) / "phase0-result.json"
    saved = json.loads(result_path.read_text(encoding="utf-8"))
    assert saved == result
    assert result["native_reward"] == 1.0
    assert set(result["rendered_prompt_sha256"]) == {"agent", "customer"}
    assert result["provider_budget"]["attempts"] == 1
    assert result["provider_budget"]["usage_unavailable"] == 1
    assert (Path(experiment["output_path"]) / "manifest.json").exists()
    assert json.loads(
        (Path(experiment["output_path"]) / "native-simulation.json").read_text(encoding="utf-8")
    )["task_id"] == "73"
    with pytest.raises(FileExistsError, match="output already exists"):
        execute_phase0(
            manifest=manifest,
            config=config,
            task=SimpleNamespace(id="73"),
            episode_runner=runner,
        )

    experiment["output_path"] = "runs/phase0-failure-record-test"
    failed_manifest = ExperimentManifest.from_mapping(config)

    def failing_runner(**kwargs):
        budget = kwargs["request_budget"]
        kwargs["on_orchestrator"](
            SimpleNamespace(
                agent=SimpleNamespace(system_prompt="native agent prompt"),
                user=SimpleNamespace(system_prompt="native customer prompt"),
            )
        )
        kwargs["on_simulation"](FakeSimulation())

        def provider_error(**_kwargs):
            raise RuntimeError("credential=must-not-be-recorded")

        provider = SimpleNamespace(completion=provider_error, DEFAULT_MAX_RETRIES=0)
        with budget.instrument_tau_llm_utils(provider):
            provider.completion(model="mock-agent")

    with pytest.raises(Phase0ExecutionError, match="RuntimeError"):
        execute_phase0(
            manifest=failed_manifest,
            config=config,
            task=SimpleNamespace(id="73"),
            episode_runner=failing_runner,
        )
    failed_path = Path(experiment["output_path"]) / "phase0-result.json"
    failure_record = json.loads(failed_path.read_text(encoding="utf-8"))
    assert failure_record["status"] == "incomplete"
    assert failure_record["failure_type"] == "RuntimeError"
    assert failure_record["native_simulation_saved"]
    assert set(failure_record["rendered_prompt_sha256"]) == {"agent", "customer"}
    assert failure_record["provider_budget"]["failures"] == 1
    assert "must-not-be-recorded" not in failed_path.read_text(encoding="utf-8")
    assert (failed_path.parent / "native-simulation.json").exists()


def test_mock_native_runtime_assembles_scores_and_records_under_budget(monkeypatch) -> None:
    config = __import__("yaml").safe_load((ROOT / "configs/mvp.yaml").read_text(encoding="utf-8"))
    config["experiment"]["real_provider_enabled"] = True
    config["experiment"]["models"] = {
        "agent": "mock-agent",
        "customer": "mock-customer",
        "reviewer": "mock-reviewer",
    }
    manifest = ExperimentManifest.from_mapping(config)

    class FakeEnvironment:
        user_tools = None

        def get_tools(self):
            return ["retail-tools"]

        def get_policy(self):
            return "fixed retail policy"

        def get_user_tools(self, include=None):
            raise ValueError("User tools not available")

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        @property
        def system_prompt(self):
            return "native retail policy"

    class FakeUser:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.llm = kwargs["llm"]
            self.llm_args = kwargs["llm_args"]
            self.persona_config = None

        @property
        def system_prompt(self):
            return "native user guidelines and scenario"

        @property
        def global_simulation_guidelines(self):
            return "native global user guidelines"

    class FakeOrchestrator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            for key, value in kwargs.items():
                setattr(self, key, value)

    calls = []

    def add_module(name: str, *, package: bool = False) -> ModuleType:
        module = ModuleType(name)
        if package:
            module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        return module

    for package_name in (
        "tau2",
        "tau2.agent",
        "tau2.orchestrator",
        "tau2.runner",
        "tau2.user",
        "tau2.utils",
        "tau2.evaluator",
        "tau2.data_model",
    ):
        add_module(package_name, package=True)

    agent_module = add_module("tau2.agent.llm_agent")
    agent_module.LLMAgent = FakeAgent
    orchestrator_module = add_module("tau2.orchestrator.orchestrator")
    orchestrator_module.Orchestrator = FakeOrchestrator
    user_module = add_module("tau2.user.user_simulator")
    user_module.UserSimulator = FakeUser
    build_module = add_module("tau2.runner.build")
    environment = FakeEnvironment()
    build_module.build_environment = lambda domain: environment
    evaluation_module = add_module("tau2.evaluator.evaluator")
    evaluation_module.EvaluationType = SimpleNamespace(ALL=object())
    reviewer_module = add_module("tau2.evaluator.reviewer")
    reviewer_module.ReviewMode = SimpleNamespace(FULL=object())
    review_calls = []
    provider_calls = []

    def review_simulation(**kwargs):
        assert llm_utils.DEFAULT_MAX_RETRIES == 0
        review_calls.append(kwargs)
        llm_utils.completion(model="mock-reviewer")
        llm_utils.completion(model="mock-reviewer")
        return {"has_errors": False}, {"classification": "clean"}

    reviewer_module.review_simulation = review_simulation
    data_model_module = add_module("tau2.data_model.simulation")
    data_model_module.UserInfo = lambda **kwargs: SimpleNamespace(**kwargs)
    llm_utils = add_module("tau2.utils.llm_utils")
    llm_utils.DEFAULT_MAX_RETRIES = 5

    def completion(**kwargs):
        provider_calls.append(kwargs["model"])
        return {"model": kwargs["model"]}

    llm_utils.completion = completion
    sys.modules["tau2.utils"].llm_utils = llm_utils

    def run_simulation(orchestrator, *, evaluation_type):
        assert llm_utils.DEFAULT_MAX_RETRIES == 0
        calls.append((orchestrator, evaluation_type))
        llm_utils.completion(model="mock-agent")
        return SimpleNamespace(
            reward=1.0,
            task_id=orchestrator.kwargs["task"].id,
            recorded_trajectory=("user", "assistant", "tool"),
        )

    simulation_module = add_module("tau2.runner.simulation")
    simulation_module.run_simulation = run_simulation
    monkeypatch.setattr("evotau.tau_adapter.verify_tau2_installation", lambda: None)

    task = SimpleNamespace(user_scenario="fixed scenario", user_tools=(), id="73")
    result, budget = run_phase0_episode(manifest=manifest, task=task)

    orchestrator, evaluation_type = calls[0]
    assert evaluation_type is sys.modules["tau2.evaluator.evaluator"].EvaluationType.ALL
    assert orchestrator.kwargs["environment"] is environment
    assert orchestrator.kwargs["task"] is task
    assert orchestrator.kwargs["max_steps"] == 64
    assert orchestrator.kwargs["agent"].system_prompt == "native retail policy"
    assert orchestrator.kwargs["user"].system_prompt == "native user guidelines and scenario"
    assert orchestrator.kwargs["agent"].kwargs["llm_args"] == {"num_retries": 0}
    assert orchestrator.kwargs["user"].kwargs["llm_args"] == {"num_retries": 0}
    assert orchestrator.kwargs["user"].kwargs["tools"] is None
    assert result.reward == 1.0
    assert result.task_id == "73"
    assert result.recorded_trajectory == ("user", "assistant", "tool")
    assert result.review == {"has_errors": False}
    assert result.auth_classification == {"classification": "clean"}
    assert review_calls[0]["review_model"] == "mock-reviewer"
    assert review_calls[0]["mode"] is reviewer_module.ReviewMode.FULL
    assert review_calls[0]["user_info"].global_simulation_guidelines == (
        "native global user guidelines"
    )
    assert provider_calls == ["mock-agent", "mock-reviewer", "mock-reviewer"]
    assert budget.snapshot().attempts == 3
    assert budget.snapshot().successes == 3
    assert llm_utils.DEFAULT_MAX_RETRIES == 5


def test_pinned_tau_runtime_builds_adapters_without_provider_calls(monkeypatch) -> None:
    try:
        distribution("tau2")
    except PackageNotFoundError:
        pytest.skip("install the tau-bench optional extra to run this integration check")
    data_dir = os.environ.get("EVOTAU_TAU2_DATA_DIR")
    if not data_dir:
        pytest.skip("set EVOTAU_TAU2_DATA_DIR to the pinned tau-bench data directory")

    monkeypatch.setenv("TAU2_DATA_DIR", data_dir)
    from tau2.utils import llm_utils

    def forbidden_provider_call(*args, **kwargs):
        raise AssertionError("runtime construction attempted a provider request")

    monkeypatch.setattr(llm_utils, "completion", forbidden_provider_call)
    config = __import__("yaml").safe_load((ROOT / "configs/mvp.yaml").read_text(encoding="utf-8"))
    manifest = ExperimentManifest.from_mapping(config)
    task = _load_pinned_task(
        manifest,
        data_dir=data_dir,
        task_selection=config["experiment"]["task_selection"],
    )
    customer_strategy = CustomerStrategy(challenge_style="ask_reason", challenge_budget=1)
    service_strategy = ServiceStrategy(
        (
            ServiceRule(
                rule_id="evidence-check",
                policy_ref="retail-policy#return-eligibility",
                trigger="before a return",
                required_execution="check order and item eligibility",
                evidence_refs=("fixture:independent-audit",),
            ),
        )
    )
    orchestrator = build_phase0_orchestrator(
        task=task,
        agent_model="offline-inspection-only",
        customer_model="offline-inspection-only",
        seed=42,
        customer_strategy=customer_strategy,
        service_strategy=service_strategy,
        service_token_counter=lambda text: len(text.split()),
    )

    agent = orchestrator.agent
    user = orchestrator.user
    native_agent_prompt = type(agent).__mro__[1].system_prompt.fget(agent)
    native_user_prompt = type(user).__mro__[1].system_prompt.fget(user)
    service_block = render_service_strategy(
        service_strategy, token_counter=lambda text: len(text.split())
    )
    assert task.id == "73"
    assert agent.system_prompt == append_strategy_block(native_agent_prompt, service_block)
    assert user.system_prompt == append_strategy_block(
        native_user_prompt, render_customer_strategy(customer_strategy)
    )
