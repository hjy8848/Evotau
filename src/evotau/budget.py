"""Atomic provider-attempt accounting for one Phase 0 episode."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock, RLock
from types import ModuleType
from typing import Any


class ProviderBudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    cap: int
    attempts: int
    successes: int
    failures: int
    denied: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usage_responses: int = 0
    usage_unavailable: int = 0
    cache_hits: int = 0

    @property
    def remaining(self) -> int:
        return self.cap - self.attempts


class RequestBudget:
    """Count every call to the pinned tau-bench LiteLLM boundary before dispatch."""

    def __init__(self, cap: int):
        if cap < 1:
            raise ValueError("request budget cap must be positive")
        self._cap = cap
        self._attempts = 0
        self._successes = 0
        self._failures = 0
        self._denied = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._usage_responses = 0
        self._usage_unavailable = 0
        self._cache_hits = 0
        self._lock = Lock()
        self._patch_lock = RLock()

    def _begin(self, model: str | None) -> int:
        with self._lock:
            if self._attempts >= self._cap:
                self._denied += 1
                raise ProviderBudgetExceeded(
                    f"provider request denied before dispatch: cap {self._cap} reached"
                )
            self._attempts += 1
            return self._attempts

    @staticmethod
    def _field(value: Any, name: str) -> Any:
        if isinstance(value, Mapping):
            return value.get(name)
        return getattr(value, name, None)

    def _finish(self, succeeded: bool, response: Any = None) -> None:
        usage = self._field(response, "usage") if succeeded else None
        prompt_tokens = self._field(usage, "prompt_tokens")
        completion_tokens = self._field(usage, "completion_tokens")
        with self._lock:
            if succeeded:
                self._successes += 1
                if (
                    isinstance(prompt_tokens, int)
                    and prompt_tokens >= 0
                    and isinstance(completion_tokens, int)
                    and completion_tokens >= 0
                ):
                    self._prompt_tokens += prompt_tokens
                    self._completion_tokens += completion_tokens
                    self._usage_responses += 1
                else:
                    self._usage_unavailable += 1
                hidden_params = self._field(response, "_hidden_params") or {}
                if self._field(hidden_params, "cache_hit") is True:
                    self._cache_hits += 1
            else:
                self._failures += 1
                self._usage_unavailable += 1

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return BudgetSnapshot(
                cap=self._cap,
                attempts=self._attempts,
                successes=self._successes,
                failures=self._failures,
                denied=self._denied,
                prompt_tokens=self._prompt_tokens,
                completion_tokens=self._completion_tokens,
                usage_responses=self._usage_responses,
                usage_unavailable=self._usage_unavailable,
                cache_hits=self._cache_hits,
            )

    def restore_usage(self, snapshot: BudgetSnapshot) -> None:
        """Restore cumulative accounting before a resumed run dispatches work."""
        if snapshot.cap != self._cap:
            raise ValueError("restored provider budget cap does not match this budget")
        counters = (
            snapshot.attempts, snapshot.successes, snapshot.failures, snapshot.denied,
            snapshot.prompt_tokens, snapshot.completion_tokens,
            snapshot.usage_responses, snapshot.usage_unavailable, snapshot.cache_hits,
        )
        if any(value < 0 for value in counters):
            raise ValueError("provider budget counters must be non-negative")
        if snapshot.successes + snapshot.failures != snapshot.attempts or snapshot.attempts > snapshot.cap:
            raise ValueError("restored provider budget counters are inconsistent")
        if snapshot.usage_responses + snapshot.usage_unavailable > snapshot.attempts:
            raise ValueError("restored provider usage counters exceed request attempts")
        with self._lock:
            if self._attempts or self._successes or self._failures or self._denied:
                raise ValueError("provider budget can only be restored before it is used")
            self._attempts = snapshot.attempts
            self._successes = snapshot.successes
            self._failures = snapshot.failures
            self._denied = snapshot.denied
            self._prompt_tokens = snapshot.prompt_tokens
            self._completion_tokens = snapshot.completion_tokens
            self._usage_responses = snapshot.usage_responses
            self._usage_unavailable = snapshot.usage_unavailable
            self._cache_hits = snapshot.cache_hits

    @contextmanager
    def instrument_tau_llm_utils(self, llm_utils: ModuleType | Any) -> Iterator[None]:
        """Guard tau2.utils.llm_utils.completion and disable its default retries."""

        with self._patch_lock:
            original_completion = llm_utils.completion
            original_default_retries = llm_utils.DEFAULT_MAX_RETRIES

            def guarded_completion(*args: Any, **kwargs: Any) -> Any:
                model = kwargs.get("model")
                if model is None and args:
                    model = str(args[0])
                self._begin(None if model is None else str(model))
                try:
                    result = original_completion(*args, **kwargs)
                except BaseException:
                    self._finish(succeeded=False)
                    raise
                self._finish(succeeded=True, response=result)
                return result

            llm_utils.completion = guarded_completion
            llm_utils.DEFAULT_MAX_RETRIES = 0
            try:
                yield
            finally:
                llm_utils.completion = original_completion
                llm_utils.DEFAULT_MAX_RETRIES = original_default_retries
