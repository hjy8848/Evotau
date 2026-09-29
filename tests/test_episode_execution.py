from __future__ import annotations

from threading import Barrier, Event, Lock
from time import sleep
from types import SimpleNamespace

import pytest

from evotau.budget import ProviderBudgetExceeded, RequestBudget
from evotau.episode_execution import (
    EpisodeSpec,
    StopBeforeEpisodeDispatch,
    run_episode_batch,
)
from evotau.lifecycle import _evaluate_customer_panels, evaluate_customer_panel
from evotau.records import (
    EpisodeRecord,
    EpisodeStatus,
    customer_strategy_id,
    service_strategy_id,
)
from evotau.strategies import CustomerStrategy, ServiceStrategy

CUSTOMER = CustomerStrategy()
SERVICE = ServiceStrategy()


def _specs(count: int) -> tuple[EpisodeSpec, ...]:
    return tuple(
        EpisodeSpec(f"task-{index:02d}", index, CUSTOMER, SERVICE, "discovery", 0)
        for index in range(count)
    )


def _record(task_id: str, seed: int, customer: CustomerStrategy = CUSTOMER) -> EpisodeRecord:
    return EpisodeRecord(
        episode_id=f"{task_id}:{seed}",
        task_id=task_id,
        seed=seed,
        customer_strategy_id=customer_strategy_id(customer),
        service_strategy_id=service_strategy_id(SERVICE),
        status=EpisodeStatus.UNCERTAIN,
        task_success=None,
    )


def test_twenty_episode_stress_is_bounded_and_ordered() -> None:
    lock = Lock()
    active = 0
    max_active = 0
    completed: list[str] = []

    def runner(*, task_id, seed, customer, service, panel_name):
        nonlocal active, max_active
        index = int(task_id[-2:])
        with lock:
            active += 1
            max_active = max(max_active, active)
        sleep((5 - index % 6) * 0.002)
        with lock:
            active -= 1
            completed.append(task_id)
        return _record(task_id, seed)

    results = run_episode_batch(runner, _specs(20), max_concurrency=4)
    assert tuple(item.task_id for item in results) == tuple(f"task-{i:02d}" for i in range(20))
    assert len({item.episode_id for item in results}) == 20
    assert len(completed) == 20
    assert completed != [item.task_id for item in results]
    assert max_active == 4


def test_customer_panel_serial_and_parallel_aggregation_match() -> None:
    task_ids = ("72", "46", "93")

    def runner(*, task_id, seed, customer, service, panel_name):
        sleep((3 - task_ids.index(task_id)) * 0.003)
        return _record(task_id, seed)

    serial = evaluate_customer_panel(
        runner, task_ids=task_ids, seeds=(7, 8), strategy=CUSTOMER, service=SERVICE,
        panel_name="discovery", generation=0, max_concurrency=1,
    )
    parallel = evaluate_customer_panel(
        runner, task_ids=task_ids, seeds=(7, 8), strategy=CUSTOMER, service=SERVICE,
        panel_name="discovery", generation=0, max_concurrency=4,
    )
    assert parallel == serial
    assert [(item.task_id, item.seed) for item in parallel.episodes] == [
        (task_id, seed) for task_id in task_ids for seed in (7, 8)
    ]


def test_multiple_frozen_customer_panels_share_one_worker_limit() -> None:
    candidates = (
        CustomerStrategy(disclosure="related_on_request"),
        CustomerStrategy(request_order="reverse_independent"),
    )
    lock = Lock()
    active = 0
    max_active = 0

    def runner(*, task_id, seed, customer, service, panel_name):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        sleep(0.004)
        with lock:
            active -= 1
        return _record(task_id, seed, customer)

    evaluations = _evaluate_customer_panels(
        runner, task_ids=("a", "b", "c"), seeds=(1,), strategies=candidates,
        service=SERVICE, panel_name="discovery", generation=0, max_concurrency=2,
    )
    assert tuple(item.strategy_id for item in evaluations) == tuple(
        customer_strategy_id(item) for item in candidates
    )
    assert [[episode.task_id for episode in item.episodes] for item in evaluations] == [
        ["a", "b", "c"], ["a", "b", "c"],
    ]
    assert max_active == 2


def test_episode_cap_reservations_never_oversubscribe() -> None:
    lock = Lock()
    dispatched: list[str] = []
    reserved: set[str] = set()
    cap = 2

    def reserve(spec: EpisodeSpec) -> bool:
        with lock:
            if len(dispatched) + len(reserved) >= cap:
                return False
            reserved.add(spec.task_id)
            return True

    def release(spec: EpisodeSpec) -> None:
        with lock:
            reserved.discard(spec.task_id)

    def runner(*, task_id, seed, customer, service, panel_name):
        with lock:
            reserved.remove(task_id)
            dispatched.append(task_id)
        sleep(0.01)
        return _record(task_id, seed)

    with pytest.raises(RuntimeError, match="episode cap"):
        run_episode_batch(
            runner, _specs(8), max_concurrency=4,
            reserve_episode=reserve, release_episode=release,
        )
    assert len(dispatched) == 2
    assert not reserved


def test_pause_drains_running_workers_and_does_not_dispatch_queued_specs() -> None:
    started = Barrier(4)
    pause = Event()
    lock = Lock()
    calls: list[str] = []

    def runner(*, task_id, seed, customer, service, panel_name):
        with lock:
            calls.append(task_id)
        started.wait(timeout=2)
        if task_id == "task-03":
            pause.set()
        sleep(0.02)
        return _record(task_id, seed)

    with pytest.raises(StopBeforeEpisodeDispatch, match="active episodes finished"):
        run_episode_batch(
            runner, _specs(20), max_concurrency=4,
            should_pause=pause.is_set,
        )
    assert len(calls) == 4
    assert set(calls) == {f"task-{i:02d}" for i in range(4)}


def test_worker_error_drains_submitted_episodes_and_dispatches_no_replacements() -> None:
    started = Barrier(4)
    finished: list[str] = []
    lock = Lock()

    def runner(*, task_id, seed, customer, service, panel_name):
        started.wait(timeout=2)
        sleep(0.005)
        with lock:
            finished.append(task_id)
        if task_id == "task-00":
            raise ValueError("offline provider failure")
        return _record(task_id, seed)

    with pytest.raises(ValueError, match="offline provider failure"):
        run_episode_batch(runner, _specs(10), max_concurrency=4)
    assert len(finished) == 4
    assert set(finished) == {f"task-{i:02d}" for i in range(4)}


def test_concurrent_global_provider_instrumentation_is_counted_once_and_restored() -> None:
    budget = RequestBudget(cap=8)
    start = Barrier(4)
    calls: list[dict] = []
    lock = Lock()

    def completion(**kwargs):
        start.wait(timeout=2)
        sleep(0.002)
        with lock:
            calls.append(kwargs)
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2))

    litellm = SimpleNamespace(completion=lambda **_kwargs: None)
    original_litellm_completion = litellm.completion
    provider = SimpleNamespace(
        completion=completion, DEFAULT_MAX_RETRIES=9, litellm=litellm,
    )

    def runner(*, task_id, seed, customer, service, panel_name):
        with budget.instrument_tau_llm_utils(provider):
            provider.completion(model=f"role-{task_id}", num_retries=6)
        return _record(task_id, seed)

    results = run_episode_batch(
        runner, _specs(4), max_concurrency=4, request_budget=budget,
    )
    snapshot = budget.snapshot()
    assert len(results) == 4
    assert len(calls) == 4
    assert {item["model"] for item in calls} == {f"role-task-{i:02d}" for i in range(4)}
    assert all(item["num_retries"] == 0 for item in calls)
    assert snapshot.attempts == snapshot.successes + snapshot.failures == 4
    assert snapshot.prompt_tokens == 20 and snapshot.completion_tokens == 8
    assert {row.model_id for row in snapshot.model_usage} == {item["model"] for item in calls}
    assert provider.completion is completion
    assert provider.DEFAULT_MAX_RETRIES == 9
    assert litellm.completion is original_litellm_completion


def test_restored_budget_model_rows_accept_further_provider_calls() -> None:
    original = RequestBudget(cap=3)
    provider = SimpleNamespace(
        completion=lambda **_kwargs: SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
        ),
        DEFAULT_MAX_RETRIES=0,
    )
    with original.instrument_tau_llm_utils(provider):
        provider.completion(model="model-a")
    restored = RequestBudget(cap=3)
    restored.restore_usage(original.snapshot())
    with restored.instrument_tau_llm_utils(provider):
        provider.completion(model="model-a")
    assert restored.snapshot().attempts == 2
    assert restored.snapshot().model_usage[0].attempts == 2


def test_shared_provider_budget_hard_cap_holds_under_concurrent_followup_calls() -> None:
    budget = RequestBudget(cap=5)
    first_wave = Barrier(4)
    actual_calls = 0
    lock = Lock()
    provider = SimpleNamespace(DEFAULT_MAX_RETRIES=0)

    def completion(**_kwargs):
        nonlocal actual_calls
        with lock:
            actual_calls += 1
            call_number = actual_calls
        if call_number <= 4:
            first_wave.wait(timeout=2)
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))

    provider.completion = completion

    def runner(*, task_id, seed, customer, service, panel_name):
        with budget.instrument_tau_llm_utils(provider):
            provider.completion(model="shared")
            provider.completion(model="shared")
        return _record(task_id, seed)

    with pytest.raises(ProviderBudgetExceeded):
        run_episode_batch(
            runner, _specs(12), max_concurrency=4, request_budget=budget,
        )
    snapshot = budget.snapshot()
    assert actual_calls == snapshot.attempts == 5
    assert snapshot.attempts == snapshot.successes + snapshot.failures
    assert snapshot.denied >= 1
    assert snapshot.attempts + snapshot.in_flight + snapshot.reserved <= snapshot.cap
    assert provider.completion is completion
