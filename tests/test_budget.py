from __future__ import annotations

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
