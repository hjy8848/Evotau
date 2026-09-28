"""Atomic provider-attempt accounting for one Phase 0 episode."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock, RLock
from types import ModuleType
from typing import Any, Iterator


class ProviderBudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    cap: int
    attempts: int
    successes: int
    failures: int
    denied: int

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

    def _finish(self, succeeded: bool) -> None:
        with self._lock:
            if succeeded:
                self._successes += 1
            else:
                self._failures += 1

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return BudgetSnapshot(
                cap=self._cap,
                attempts=self._attempts,
                successes=self._successes,
                failures=self._failures,
                denied=self._denied,
            )

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
                self._finish(succeeded=True)
                return result

            llm_utils.completion = guarded_completion
            llm_utils.DEFAULT_MAX_RETRIES = 0
            try:
                yield
            finally:
                llm_utils.completion = original_completion
                llm_utils.DEFAULT_MAX_RETRIES = original_default_retries
