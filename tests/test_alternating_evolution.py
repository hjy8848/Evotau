from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Barrier, Lock
from time import sleep

from evotau.alternating import (
    LLMAlternatingEvolvers,
    _run_panel,
    propose_fresh_customer_challenge,
    run_alternating_evolution,
    run_final_endpoint_evaluation,
)
from evotau.records import EpisodeRecord, EpisodeStatus
from evotau.strategies import PromptStrategy


@dataclass
class FakeTask:
    id: str
    description: str
    user_scenario: str
    user_tools: tuple[str, ...] = ()


class FakeRunner:
    service_policy_text = "Fixed benchmark policy text."

    def __init__(self, success: Callable[[str, str, str], bool] | None = None) -> None:
        self.calls: list[dict] = []
        self.trajectories: dict[str, dict] = {}
        self.success = success or (lambda _task, _customer, _service: True)

    def __call__(self, *, task_id, seed, customer, service, panel_name):
        customer_text = "native" if customer is None else customer.text
        succeeded = self.success(str(task_id), customer_text, service.text)
        episode_id = f"episode-{len(self.calls)}"
        record = EpisodeRecord(
            episode_id=episode_id,
            task_id=str(task_id),
            seed=seed,
            customer_strategy_id=customer_text,
            service_strategy_id=service.text,
            status=EpisodeStatus.COMPLETE,
            task_success=succeeded,
            native_reward=1.0 if succeeded else 0.0,
            termination_reason="user_stop",
            trajectory_ref=f"{episode_id}.json",
            raw_review={},
        )
        self.calls.append({
            "task_id": str(task_id),
            "seed": seed,
            "customer": customer,
            "service": service,
            "panel_name": panel_name,
        })
        self.trajectories[record.trajectory_ref] = {
            "messages": [
                {"role": "user", "content": "Please help with my request."},
                {"role": "assistant", "content": "Checking the order."},
                {"role": "tool", "name": "lookup_order", "content": "order found"},
            ],
            "termination_reason": "user_stop",
            "reward_info": {"should_not_be_forwarded": True},
        }
        return record

    def load_trajectory(self, episode):
        return self.trajectories[episode.trajectory_ref]


def test_customer_prompt_is_task_grounded_and_reusable(monkeypatch) -> None:
    prompts = {}

    def fake_json_call(_model, _args, system_prompt, _context, *, call_name):
        prompts[call_name] = system_prompt
        return {"candidates": ["Ask one grounded follow-up question."]}

    monkeypatch.setattr(
        LLMAlternatingEvolvers, "_json_call", staticmethod(fake_json_call),
    )
    evolver = LLMAlternatingEvolvers(model="provider/model", model_args={})
    evolver.customer_candidates({"incumbent_accuracy": 0.5}, 1)
    prompt = prompts["evotau_customer_evolver"].lower()
    assert "reusable interaction skills" in prompt
    assert "avoid task-specific entities" in prompt
    assert "preserve the original user objective" in prompt
    assert "accuracy" in prompt


def test_accuracy_selects_hardest_customer_then_accepts_service_gain(tmp_path) -> None:
    tasks = {
        "e1": FakeTask("e1", "Evolution one", "Customer goal one."),
        "e2": FakeTask("e2", "Evolution two", "Customer goal two."),
        "v1": FakeTask("v1", "Validation one", "Native validation request."),
        "v2": FakeTask("v2", "Validation two", "Native validation request."),
    }

    def success(task_id: str, customer: str, service: str) -> bool:
        if task_id.startswith("v"):
            return True
        if service == "S0":
            return customer == "C0" or (customer == "C1" and task_id == "e1")
        if service == "S1":
            return customer == "C2" and task_id == "e1"
        return False

    runner = FakeRunner(success)
    customer_contexts = []
    service_contexts = []

    def customer_evolver(context, count):
        customer_contexts.append(context)
        assert count == 2
        return ["C1", "C2"]

    def service_evolver(context):
        service_contexts.append(context)
        assert context["selected_customer_accuracy"] == 0.0
        assert context["customer_strategy"] == "C2"
        assert "user_scenario" not in context["task_interactions"][0]["task"]
        return {"analysis": "Observed low E accuracy.", "strategy": "S1"}

    result = run_alternating_evolution(
        tasks=tasks,
        evolution_task_ids=("e1", "e2"),
        validation_task_ids=("v1", "v2"),
        seed=1,
        generations=1,
        customer_candidate_count=2,
        clean_panel_size=2,
        initial_customer=PromptStrategy("C0"),
        initial_service=PromptStrategy("S0"),
        runner=runner,
        customer_evolver=customer_evolver,
        service_evolver=service_evolver,
        domain_policy=runner.service_policy_text,
        output_directory=tmp_path,
        manifest_sha256="manifest",
    )

    generation = result.generations[0]
    customer_phase = generation["customer_phase"]
    service_phase = generation["service_phase"]
    assert customer_phase["incumbent_accuracy"] == 1.0
    assert customer_phase["candidate_accuracies"] == [0.5, 0.0]
    assert customer_phase["selected_customer"] == 1
    assert customer_phase["selected_accuracy"] == 0.0
    assert result.customer.text == "C2"
    assert service_phase["old_accuracy"] == 0.0
    assert service_phase["proposed_accuracy"] == 0.5
    assert service_phase["validation_old_accuracy"] == 1.0
    assert service_phase["validation_new_accuracy"] == 1.0
    assert service_phase["accepted"] is True
    assert result.service.text == "S1"
    assert all("user_scenario" in row["task"] for row in customer_contexts[0]["task_interactions"])
    assert service_contexts
    saved = __import__("json").loads((tmp_path / "generation-0000.json").read_text())
    assert saved["customer_phase"]["selected_accuracy"] == 0.0
    assert saved["service_phase"]["proposed_accuracy"] == 0.5
    assert (tmp_path / "checkpoint.json").is_file()


def test_customer_accuracy_tie_keeps_incumbent_without_selection_judge() -> None:
    tasks = {
        "e": FakeTask("e", "Evolution task", "Original customer goal."),
        "v": FakeTask("v", "Validation task", "Native validation request."),
    }
    runner = FakeRunner()
    result = run_alternating_evolution(
        tasks=tasks,
        evolution_task_ids=("e",),
        validation_task_ids=("v",),
        seed=2,
        generations=1,
        customer_candidate_count=1,
        clean_panel_size=1,
        initial_customer=PromptStrategy("incumbent"),
        initial_service=PromptStrategy("service"),
        runner=runner,
        customer_evolver=lambda _context, _count: ["candidate"],
        service_evolver=lambda _context: {"analysis": "no change", "strategy": "service"},
        domain_policy=runner.service_policy_text,
    )
    phase = result.generations[0]["customer_phase"]
    assert phase["incumbent_accuracy"] == phase["candidate_accuracies"][0] == 1.0
    assert phase["selected_customer"] == "incumbent"
    assert result.customer.text == "incumbent"
    assert len(runner.calls) == 2


def test_customer_selection_keeps_lowest_accuracy_when_last_candidate_is_easier() -> None:
    tasks = {
        "e": FakeTask("e", "Evolution task", "Original customer goal."),
        "v": FakeTask("v", "Validation task", "Native validation request."),
    }

    def success(_task_id: str, customer: str, _service: str) -> bool:
        return customer != "hard candidate"

    runner = FakeRunner(success)
    result = run_alternating_evolution(
        tasks=tasks,
        evolution_task_ids=("e",),
        validation_task_ids=("v",),
        seed=5,
        generations=1,
        customer_candidate_count=2,
        clean_panel_size=1,
        initial_customer=PromptStrategy("incumbent"),
        initial_service=PromptStrategy("service"),
        runner=runner,
        customer_evolver=lambda _context, _count: ["hard candidate", "easy candidate"],
        service_evolver=lambda _context: {"analysis": "keep", "strategy": "service"},
        domain_policy=runner.service_policy_text,
    )

    phase = result.generations[0]["customer_phase"]
    assert phase["candidate_accuracies"] == [0.0, 1.0]
    assert phase["selected_customer"] == 0
    assert phase["selected_accuracy"] == 0.0
    assert result.customer.text == "hard candidate"


def test_service_e_gain_is_rejected_when_native_validation_accuracy_decreases() -> None:
    tasks = {
        "e": FakeTask("e", "Evolution task", "Original customer goal."),
        "v": FakeTask("v", "Validation task", "Native validation request."),
    }

    def success(task_id: str, _customer: str, service: str) -> bool:
        return (task_id == "e" and service == "S1") or (task_id == "v" and service == "S0")

    runner = FakeRunner(success)
    result = run_alternating_evolution(
        tasks=tasks,
        evolution_task_ids=("e",),
        validation_task_ids=("v",),
        seed=4,
        generations=1,
        customer_candidate_count=1,
        clean_panel_size=1,
        initial_customer=PromptStrategy("C0"),
        initial_service=PromptStrategy("S0"),
        runner=runner,
        customer_evolver=lambda _context, _count: ["C1"],
        service_evolver=lambda _context: {"analysis": "improves E", "strategy": "S1"},
        domain_policy=runner.service_policy_text,
    )

    phase = result.generations[0]["service_phase"]
    assert phase["old_accuracy"] == 0.0
    assert phase["proposed_accuracy"] == 1.0
    assert phase["validation_old_accuracy"] == 1.0
    assert phase["validation_new_accuracy"] == 0.0
    assert phase["accepted"] is False
    assert result.service.text == "S0"


def test_fresh_customer_reuses_final_e_episodes_without_running_another_episode() -> None:
    tasks = {
        "e": FakeTask("e", "Evolution task", "The original goal."),
        "v": FakeTask("v", "Validation task", "Native validation goal."),
    }
    runner = FakeRunner()
    result = run_alternating_evolution(
        tasks=tasks,
        evolution_task_ids=("e",),
        validation_task_ids=("v",),
        seed=3,
        generations=1,
        customer_candidate_count=1,
        clean_panel_size=1,
        initial_customer=PromptStrategy("C0"),
        initial_service=PromptStrategy("S0"),
        runner=runner,
        customer_evolver=lambda _context, _count: ["C1"],
        service_evolver=lambda _context: {"analysis": "same", "strategy": "S0"},
        domain_policy=runner.service_policy_text,
    )
    calls_before = len(runner.calls)
    observed = []

    fresh = propose_fresh_customer_challenge(
        result,
        lambda context, count: observed.append((context, count)) or ["fresh skill"],
        tasks=tasks,
        runner=runner,
        domain_policy=runner.service_policy_text,
    )

    assert fresh.text == "fresh skill"
    assert len(runner.calls) == calls_before
    assert observed[0][1] == 1
    assert observed[0][0]["final_evolution_accuracy"] == 1.0
    assert observed[0][0]["task_interactions"][0]["trajectory"]["messages"]


def test_heldout_aliases_st_when_service_is_identical_to_s0() -> None:
    heldout = {"h": FakeTask("h", "Held-out task", "Held-out customer goal.")}
    runner = FakeRunner()
    evaluation = run_final_endpoint_evaluation(
        heldout_tasks=heldout,
        heldout_task_ids=("h",),
        seed=9,
        initial_service=PromptStrategy("same service"),
        final_service=PromptStrategy("same service"),
        fresh_customer=PromptStrategy("fresh challenge"),
        runner=runner,
    )
    assert len(runner.calls) == 2
    assert len(evaluation["cells"]) == 4
    for condition in ("native_customer", "fresh_adaptive_customer"):
        s0 = next(row for row in evaluation["cells"]
                  if row["customer_condition"] == condition and row["service_endpoint"] == "S0")
        st = next(row for row in evaluation["cells"]
                  if row["customer_condition"] == condition and row["service_endpoint"] == "ST")
        assert st["identical_to_S0"] is True
        assert st["episodes"] == s0["episodes"]
        assert st["accuracy"] == s0["accuracy"]


def test_heldout_runs_st_when_final_service_differs() -> None:
    heldout = {"h": FakeTask("h", "Held-out task", "Held-out customer goal.")}
    runner = FakeRunner()
    evaluation = run_final_endpoint_evaluation(
        heldout_tasks=heldout,
        heldout_task_ids=("h",),
        seed=10,
        initial_service=PromptStrategy("S0"),
        final_service=PromptStrategy("ST"),
        fresh_customer=PromptStrategy("fresh challenge"),
        runner=runner,
    )
    assert len(runner.calls) == 4
    assert evaluation["services_identical"] is False
    assert all(not cell["identical_to_S0"] for cell in evaluation["cells"])
    assert all("accuracy" in cell for cell in evaluation["cells"])


def test_run_panel_is_bounded_parallel_and_keeps_input_order() -> None:
    task_ids = ("t0", "t1", "t2", "t3", "t4")
    tasks = {task_id: object() for task_id in task_ids}
    first_wave = Barrier(3)
    lock = Lock()
    active = 0
    maximum_active = 0
    completion_order: list[str] = []

    def runner(*, task_id, seed, customer, service, panel_name):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            if task_id in task_ids[:3]:
                first_wave.wait(timeout=3)
            sleep({"t0": 0.15, "t1": 0.08, "t2": 0.01}.get(task_id, 0.01))
            record = EpisodeRecord(
                episode_id=f"episode-{task_id}",
                task_id=task_id,
                seed=seed,
                customer_strategy_id="native" if customer is None else customer.text,
                service_strategy_id=service.text,
                status=EpisodeStatus.COMPLETE,
                task_success=True,
                native_reward=1.0,
                termination_reason="user_stop",
                trajectory_ref=None,
                raw_review={},
            )
            with lock:
                completion_order.append(task_id)
            return record
        finally:
            with lock:
                active -= 1

    records = _run_panel(
        runner,
        task_ids=task_ids,
        tasks=tasks,
        seed=7,
        customer=PromptStrategy("customer"),
        service=PromptStrategy("service"),
        panel_name="parallel-test",
        max_parallel_episodes=3,
    )

    assert maximum_active == 3
    assert completion_order[0] == "t2"
    assert tuple(record.task_id for record in records) == task_ids


def test_run_panel_single_task_stays_single_and_accepts_default_cap() -> None:
    task_ids = ("only",)
    calls: list[str] = []

    def runner(*, task_id, seed, customer, service, panel_name):
        calls.append(task_id)
        return EpisodeRecord(
            episode_id="single",
            task_id=task_id,
            seed=seed,
            customer_strategy_id="native" if customer is None else customer.text,
            service_strategy_id=service.text,
            status=EpisodeStatus.COMPLETE,
            task_success=True,
            native_reward=1.0,
            termination_reason="user_stop",
            trajectory_ref=None,
            raw_review={},
        )

    episodes = _run_panel(
        runner,
        task_ids=task_ids,
        tasks={"only": object()},
        seed=1,
        customer=None,
        service=PromptStrategy("service"),
        panel_name="single-test",
    )

    assert calls == ["only"]
    assert len(episodes) == 1
    assert episodes[0].task_id == "only"
