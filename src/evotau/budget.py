"""Atomic provider-attempt accounting for one Phase 0 episode."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from threading import Lock, RLock, local
from types import ModuleType
from typing import Any


class ProviderBudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ModelUsageSnapshot:
    """Provider-attempt and token counters for one requested model ID."""

    model_id: str
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    denied: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usage_responses: int = 0
    usage_unavailable: int = 0
    cache_hits: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model usage requires a non-empty model ID")
        counters = (
            self.attempts, self.successes, self.failures, self.denied,
            self.prompt_tokens, self.completion_tokens,
            self.usage_responses, self.usage_unavailable, self.cache_hits,
        )
        if any(type(value) is not int or value < 0 for value in counters):
            raise ValueError("model usage counters must be non-negative integers")
        if self.successes + self.failures != self.attempts:
            raise ValueError("model usage success/failure counters are inconsistent")
        if self.usage_responses + self.usage_unavailable > self.attempts:
            raise ValueError("model usage counters exceed provider attempts")


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
    model_usage: tuple[ModelUsageSnapshot, ...] = ()

    def __post_init__(self) -> None:
        rows = []
        for row in self.model_usage:
            if isinstance(row, ModelUsageSnapshot):
                rows.append(row)
            elif isinstance(row, Mapping):
                rows.append(ModelUsageSnapshot(**row))
            else:
                raise TypeError("model_usage entries must be snapshots or JSON objects")
        if len({row.model_id for row in rows}) != len(rows):
            raise ValueError("model usage snapshot repeats a model ID")
        object.__setattr__(self, "model_usage", tuple(sorted(rows, key=lambda row: row.model_id)))
        if rows:
            counter_names = (
                "attempts", "successes", "failures", "denied", "prompt_tokens",
                "completion_tokens", "usage_responses", "usage_unavailable", "cache_hits",
            )
            if any(
                sum(getattr(row, name) for row in rows) != getattr(self, name)
                for name in counter_names
            ):
                raise ValueError("per-model usage does not reconcile with the global budget snapshot")

    @property
    def remaining(self) -> int:
        return self.cap - self.attempts

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["model_usage"] = [asdict(item) for item in self.model_usage]
        return value


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
        self._models: dict[str, dict[str, int]] = {}
        self._lock = Lock()
        self._patch_lock = RLock()
        self._active_patches = local()

    def _begin(self, model: str | None) -> int:
        with self._lock:
            model_id = self._model_key(model)
            model_usage = self._model_counters(model_id)
            if self._attempts >= self._cap:
                self._denied += 1
                model_usage["denied"] += 1
                raise ProviderBudgetExceeded(
                    f"provider request denied before dispatch: cap {self._cap} reached"
                )
            self._attempts += 1
            model_usage["attempts"] += 1
            return self._attempts

    @staticmethod
    def _model_key(model: str | None) -> str:
        return model.strip() if isinstance(model, str) and model.strip() else "__unknown_model__"

    def _model_counters(self, model_id: str) -> dict[str, int]:
        return self._models.setdefault(model_id, {
            "attempts": 0, "successes": 0, "failures": 0, "denied": 0,
            "prompt_tokens": 0, "completion_tokens": 0,
            "usage_responses": 0, "usage_unavailable": 0, "cache_hits": 0,
        })

    @staticmethod
    def _field(value: Any, name: str) -> Any:
        if isinstance(value, Mapping):
            return value.get(name)
        return getattr(value, name, None)

    def _finish(
        self, succeeded: bool, response: Any = None, *, model: str | None = None,
    ) -> None:
        usage = self._field(response, "usage") if succeeded else None
        prompt_tokens = self._field(usage, "prompt_tokens")
        completion_tokens = self._field(usage, "completion_tokens")
        with self._lock:
            model_usage = self._model_counters(self._model_key(model))
            if succeeded:
                self._successes += 1
                model_usage["successes"] += 1
                if (
                    isinstance(prompt_tokens, int)
                    and prompt_tokens >= 0
                    and isinstance(completion_tokens, int)
                    and completion_tokens >= 0
                ):
                    self._prompt_tokens += prompt_tokens
                    self._completion_tokens += completion_tokens
                    self._usage_responses += 1
                    model_usage["prompt_tokens"] += prompt_tokens
                    model_usage["completion_tokens"] += completion_tokens
                    model_usage["usage_responses"] += 1
                else:
                    self._usage_unavailable += 1
                    model_usage["usage_unavailable"] += 1
                hidden_params = self._field(response, "_hidden_params") or {}
                if self._field(hidden_params, "cache_hit") is True:
                    self._cache_hits += 1
                    model_usage["cache_hits"] += 1
            else:
                self._failures += 1
                self._usage_unavailable += 1
                model_usage["failures"] += 1
                model_usage["usage_unavailable"] += 1

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
                model_usage=tuple(
                    ModelUsageSnapshot(model_id=model_id, **counters)
                    for model_id, counters in self._models.items()
                ),
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
            if snapshot.model_usage:
                self._models = {
                    row.model_id: {
                        name: getattr(row, name)
                        for name in (
                            "attempts", "successes", "failures", "denied", "prompt_tokens",
                            "completion_tokens", "usage_responses", "usage_unavailable", "cache_hits",
                        )
                    }
                    for row in snapshot.model_usage
                }
            elif any(counters):
                self._models = {"__unattributed__": {
                    "attempts": snapshot.attempts,
                    "successes": snapshot.successes,
                    "failures": snapshot.failures,
                    "denied": snapshot.denied,
                    "prompt_tokens": snapshot.prompt_tokens,
                    "completion_tokens": snapshot.completion_tokens,
                    "usage_responses": snapshot.usage_responses,
                    "usage_unavailable": snapshot.usage_unavailable,
                    "cache_hits": snapshot.cache_hits,
                }}

    def absorb_usage(self, snapshot: BudgetSnapshot) -> None:
        """Add attempts from a completed child phase to this run's global budget."""
        counters = (
            snapshot.attempts, snapshot.successes, snapshot.failures, snapshot.denied,
            snapshot.prompt_tokens, snapshot.completion_tokens,
            snapshot.usage_responses, snapshot.usage_unavailable, snapshot.cache_hits,
        )
        if any(value < 0 for value in counters):
            raise ValueError("absorbed provider budget counters must be non-negative")
        if snapshot.successes + snapshot.failures != snapshot.attempts:
            raise ValueError("absorbed provider budget success/failure counts are inconsistent")
        if snapshot.usage_responses + snapshot.usage_unavailable > snapshot.attempts:
            raise ValueError("absorbed usage counters exceed provider attempts")
        with self._lock:
            if self._attempts + snapshot.attempts > self._cap:
                raise ValueError("absorbed phase usage exceeds the global provider-attempt budget")
            self._attempts += snapshot.attempts
            self._successes += snapshot.successes
            self._failures += snapshot.failures
            self._denied += snapshot.denied
            self._prompt_tokens += snapshot.prompt_tokens
            self._completion_tokens += snapshot.completion_tokens
            self._usage_responses += snapshot.usage_responses
            self._usage_unavailable += snapshot.usage_unavailable
            self._cache_hits += snapshot.cache_hits
            if snapshot.model_usage:
                source_models = snapshot.model_usage
            elif any(counters):
                source_models = (ModelUsageSnapshot(
                    model_id="__unattributed__",
                    attempts=snapshot.attempts,
                    successes=snapshot.successes,
                    failures=snapshot.failures,
                    denied=snapshot.denied,
                    prompt_tokens=snapshot.prompt_tokens,
                    completion_tokens=snapshot.completion_tokens,
                    usage_responses=snapshot.usage_responses,
                    usage_unavailable=snapshot.usage_unavailable,
                    cache_hits=snapshot.cache_hits,
                ),)
            else:
                source_models = ()
            for row in source_models:
                target = self._model_counters(row.model_id)
                for name in target:
                    target[name] += getattr(row, name)

    @contextmanager
    def instrument_tau_llm_utils(self, llm_utils: ModuleType | Any) -> Iterator[None]:
        """Guard the τ-bench and LiteLLM completion boundaries with retries disabled."""

        with self._patch_lock:
            active = getattr(self._active_patches, "modules", None)
            if active is None:
                active = self._active_patches.modules = set()
            module_key = id(llm_utils)
            if module_key in active:
                yield
                return

            original_completion = llm_utils.completion
            original_default_retries = llm_utils.DEFAULT_MAX_RETRIES
            litellm_module = getattr(llm_utils, "litellm", None)
            original_litellm_completion = getattr(litellm_module, "completion", None)

            def guarded_completion(*args: Any, **kwargs: Any) -> Any:
                model = kwargs.get("model")
                if model is None and args:
                    model = str(args[0])
                # The manifest freezes transport retries to zero for every role.
                if litellm_module is not None:
                    kwargs["num_retries"] = 0
                model_id = None if model is None else str(model)
                self._begin(model_id)
                try:
                    result = original_completion(*args, **kwargs)
                except BaseException:
                    self._finish(succeeded=False, model=model_id)
                    raise
                self._finish(succeeded=True, response=result, model=model_id)
                return result

            llm_utils.completion = guarded_completion
            llm_utils.DEFAULT_MAX_RETRIES = 0
            if litellm_module is not None and callable(original_litellm_completion):
                litellm_module.completion = guarded_completion
            active.add(module_key)
            try:
                yield
            finally:
                llm_utils.completion = original_completion
                llm_utils.DEFAULT_MAX_RETRIES = original_default_retries
                if litellm_module is not None and callable(original_litellm_completion):
                    litellm_module.completion = original_litellm_completion
                active.remove(module_key)
