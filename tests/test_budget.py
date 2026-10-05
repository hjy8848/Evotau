from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from types import SimpleNamespace

import pytest

from evotau.budget import (
    BudgetSnapshot,
    ModelUsageSnapshot,
    ProviderBudgetExceeded,
    RequestBudget,
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

    snapshot = budget.snapshot()
    assert calls == ["model-a", "model-b"]
    assert (snapshot.attempts, snapshot.successes, snapshot.failures, snapshot.denied) == (2, 1, 1, 1)
    assert snapshot.remaining == 0


def test_request_budget_records_reported_tokens_and_unbounded_runs() -> None:
    usage = {
        "usage": {"prompt_tokens": 13, "completion_tokens": 4},
        "_hidden_params": {"cache_hit": True},
    }
    module = SimpleNamespace(completion=lambda **_kwargs: usage, DEFAULT_MAX_RETRIES=0)
    budget = RequestBudget(cap=None)
    with budget.instrument_tau_llm_utils(module):
        module.completion(model="model-a")
    snapshot = budget.snapshot()

    assert snapshot.cap is None
    assert snapshot.attempts == 1
    assert snapshot.prompt_tokens == 13
    assert snapshot.completion_tokens == 4
    assert snapshot.cache_hits == 1
    assert snapshot.model_usage == (ModelUsageSnapshot(
        model_id="model-a", attempts=1, successes=1, prompt_tokens=13,
        completion_tokens=4, usage_responses=1, cache_hits=1,
    ),)
    assert BudgetSnapshot(**snapshot.to_dict()) == snapshot


def test_request_budget_records_call_name_success_failure_tokens_and_latency() -> None:
    def provider(*, model: str, fail: bool = False):
        if fail:
            raise RuntimeError("provider failed")
        return {"usage": {"prompt_tokens": 8, "completion_tokens": 3}}

    module = SimpleNamespace(completion=provider, DEFAULT_MAX_RETRIES=0)

    def generate(*, model: str, call_name: str, fail: bool = False):
        return module.completion(model=model, fail=fail)

    module.generate = generate
    budget = RequestBudget(cap=None)
    with budget.instrument_tau_llm_utils(module):
        module.generate(model="model-a", call_name="user_simulator_response")
        with pytest.raises(RuntimeError, match="provider failed"):
            module.generate(model="model-a", call_name="agent_response", fail=True)

    usage = budget.api_usage_by_call_name()
    assert usage["user_simulator_response"]["calls"] == 1
    assert usage["user_simulator_response"]["successes"] == 1
    assert usage["user_simulator_response"]["prompt_tokens"] == 8
    assert usage["user_simulator_response"]["completion_tokens"] == 3
    assert usage["user_simulator_response"]["total_elapsed_seconds"] >= 0
    assert usage["user_simulator_response"]["average_elapsed_seconds"] >= 0
    assert usage["agent_response"]["failures"] == 1
    assert usage["agent_response"]["usage_unavailable"] == 1


def test_live_api_usage_survives_process_resume_without_double_counting(tmp_path) -> None:
    usage_path = tmp_path / "api-usage-live.json"

    def provider(*, model: str):
        return {"usage": {"prompt_tokens": 8, "completion_tokens": 3}}

    module = SimpleNamespace(completion=provider, DEFAULT_MAX_RETRIES=0)

    def generate(*, model: str, call_name: str):
        return module.completion(model=model)

    module.generate = generate
    first_process = RequestBudget(cap=None)
    first_process.enable_live_usage(usage_path)
    with first_process.instrument_tau_llm_utils(module):
        module.generate(model="model-a", call_name="evotau_customer_evolver")

    resumed_process = RequestBudget(cap=None)
    resumed_process.enable_live_usage(usage_path)
    assert resumed_process.live_usage_restored is True
    assert resumed_process.snapshot().attempts == 1
    assert resumed_process.api_usage_by_call_name()["evotau_customer_evolver"]["calls"] == 1
    with resumed_process.instrument_tau_llm_utils(module):
        module.generate(model="model-a", call_name="agent_response")

    final_process = RequestBudget(cap=None)
    final_process.enable_live_usage(usage_path)
    assert final_process.snapshot().attempts == 2
    assert final_process.snapshot().prompt_tokens == 16
    assert final_process.snapshot().completion_tokens == 6
    usage = final_process.api_usage_by_call_name()
    assert usage["evotau_customer_evolver"]["calls"] == 1
    assert usage["agent_response"]["calls"] == 1
    assert sum(row["calls"] for row in usage.values()) == 2


def test_episode_reservations_prevent_parallel_budget_oversubscription() -> None:
    calls: list[str] = []
    calls_lock = Lock()
    provider_barrier = Barrier(2)

    def provider(*, model: str):
        with calls_lock:
            calls.append(model)
        provider_barrier.wait(timeout=3)
        return {"usage": {"prompt_tokens": 5, "completion_tokens": 2}}

    module = SimpleNamespace(completion=provider, DEFAULT_MAX_RETRIES=0)
    budget = RequestBudget(cap=2)

    def dispatch_episode() -> bool:
        reservation = budget.reserve_episode_dispatch()
        if reservation is None:
            return False
        with budget.use_episode_reservation(reservation):
            module.completion(model="model-a")
        return True

    with budget.instrument_tau_llm_utils(module), ThreadPoolExecutor(max_workers=8) as executor:
        dispatched = list(executor.map(lambda _index: dispatch_episode(), range(8)))

    snapshot = budget.snapshot()
    assert sum(dispatched) == len(calls) == 2
    assert (snapshot.attempts, snapshot.successes, snapshot.failures) == (2, 2, 0)
    assert (snapshot.in_flight, snapshot.reserved, snapshot.denied) == (0, 0, 0)
    assert snapshot.remaining == 0
