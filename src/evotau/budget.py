"""Atomic provider-attempt accounting for one Phase 0 episode."""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock, RLock
from time import perf_counter, sleep
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
    in_flight: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model usage requires a non-empty model ID")
        counters = (
            self.attempts, self.successes, self.failures, self.denied,
            self.prompt_tokens, self.completion_tokens,
            self.usage_responses, self.usage_unavailable, self.cache_hits, self.in_flight,
        )
        if any(type(value) is not int or value < 0 for value in counters):
            raise ValueError("model usage counters must be non-negative integers")
        if self.successes + self.failures != self.attempts:
            raise ValueError("model usage success/failure counters are inconsistent")
        if self.usage_responses + self.usage_unavailable > self.attempts:
            raise ValueError("model usage counters exceed provider attempts")


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    cap: int | None
    attempts: int
    successes: int
    failures: int
    denied: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usage_responses: int = 0
    usage_unavailable: int = 0
    cache_hits: int = 0
    in_flight: int = 0
    reserved: int = 0
    model_usage: tuple[ModelUsageSnapshot, ...] = ()

    def __post_init__(self) -> None:
        counters = (
            self.attempts, self.successes, self.failures, self.denied,
            self.prompt_tokens, self.completion_tokens, self.usage_responses,
            self.usage_unavailable, self.cache_hits, self.in_flight, self.reserved,
        )
        if any(type(value) is not int or value < 0 for value in counters):
            raise ValueError("provider budget counters must be non-negative integers")
        if self.cap is not None and (type(self.cap) is not int or self.cap < 1):
            raise ValueError("provider budget cap must be a positive integer or None")
        if (self.successes + self.failures != self.attempts
                or (self.cap is not None
                    and self.attempts + self.in_flight + self.reserved > self.cap)):
            raise ValueError("provider budget counters are inconsistent")
        if self.usage_responses + self.usage_unavailable > self.attempts:
            raise ValueError("provider usage counters exceed request attempts")
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
                "in_flight",
            )
            if any(
                sum(getattr(row, name) for row in rows) != getattr(self, name)
                for name in counter_names
            ):
                raise ValueError("per-model usage does not reconcile with the global budget snapshot")

    @property
    def remaining(self) -> int | None:
        if self.cap is None:
            return None
        return self.cap - self.attempts - self.in_flight - self.reserved

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["model_usage"] = [asdict(item) for item in self.model_usage]
        return value


class ProviderDispatchReservation:
    """One atomically reserved first provider request for a dispatched episode."""

    __slots__ = ("_budget", "_consumed", "_released")

    def __init__(self, budget: RequestBudget):
        self._budget = budget
        self._consumed = False
        self._released = False


class EpisodeUsageTracker:
    """Request/token totals attributed to one episode execution context."""

    _NAMES = (
        "attempts", "successes", "failures", "denied", "prompt_tokens",
        "completion_tokens", "usage_responses", "usage_unavailable", "cache_hits",
    )

    def __init__(self) -> None:
        self._values = {name: 0 for name in self._NAMES}
        self._models: dict[str, dict[str, int]] = {}

    @classmethod
    def _empty(cls) -> dict[str, int]:
        return {name: 0 for name in cls._NAMES}

    def denied(self, model_id: str) -> None:
        row = self._models.setdefault(model_id, self._empty())
        row["denied"] += 1
        self._values["denied"] += 1

    def completed(self, model_id: str, *, succeeded: bool, response: Any = None) -> None:
        row = self._models.setdefault(model_id, self._empty())
        row["attempts"] += 1
        self._values["attempts"] += 1
        if succeeded:
            row["successes"] += 1
            self._values["successes"] += 1
            usage = RequestBudget._field(response, "usage")
            prompt = RequestBudget._field(usage, "prompt_tokens")
            completion = RequestBudget._field(usage, "completion_tokens")
            if (isinstance(prompt, int) and prompt >= 0
                    and isinstance(completion, int) and completion >= 0):
                row["prompt_tokens"] += prompt
                row["completion_tokens"] += completion
                row["usage_responses"] += 1
                self._values["prompt_tokens"] += prompt
                self._values["completion_tokens"] += completion
                self._values["usage_responses"] += 1
            else:
                row["usage_unavailable"] += 1
                self._values["usage_unavailable"] += 1
            hidden = RequestBudget._field(response, "_hidden_params") or {}
            if RequestBudget._field(hidden, "cache_hit") is True:
                row["cache_hits"] += 1
                self._values["cache_hits"] += 1
        else:
            row["failures"] += 1
            row["usage_unavailable"] += 1
            self._values["failures"] += 1
            self._values["usage_unavailable"] += 1

    def snapshot(self, *, cap: int | None) -> BudgetSnapshot:
        return BudgetSnapshot(
            cap=cap,
            **self._values,
            model_usage=tuple(
                ModelUsageSnapshot(model_id=model_id, **values)
                for model_id, values in sorted(self._models.items())
            ),
        )


@dataclass(slots=True)
class _InstrumentationState:
    budget: RequestBudget
    llm_utils: Any
    litellm_module: Any
    original_completion: Any
    generate_bindings: list[tuple[Any, str, Any]]
    original_default_retries: Any
    original_litellm_completion: Any
    guarded_completion: Any
    guarded_generate: Any
    retry_empty_responses: bool = False
    references: int = 1


_INSTRUMENTATION_LOCK = RLock()
_INSTRUMENTATION_STATES: dict[int, _InstrumentationState] = {}


def _is_empty_completion(response: Any) -> bool:
    choices = RequestBudget._field(response, "choices")
    if not choices:
        return True
    message = RequestBudget._field(choices[0], "message")
    if message is None:
        return True
    content = RequestBudget._field(message, "content")
    tool_calls = RequestBudget._field(message, "tool_calls")
    return not content and not tool_calls


class RequestBudget:
    """Count every call to the pinned tau-bench LiteLLM boundary before dispatch."""

    def __init__(self, cap: int | None):
        if cap is not None and (type(cap) is not int or cap < 1):
            raise ValueError("request budget cap must be a positive integer or None")
        self._cap = cap
        self._attempts = 0
        self._in_flight = 0
        self._reserved_dispatches = 0
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
        self._reservation_context: ContextVar[ProviderDispatchReservation | None] = ContextVar(
            f"evotau_budget_reservation_{id(self)}", default=None,
        )
        self._usage_context: ContextVar[EpisodeUsageTracker | None] = ContextVar(
            f"evotau_episode_usage_{id(self)}", default=None,
        )
        self._call_name_context: ContextVar[str | None] = ContextVar(
            f"evotau_call_name_{id(self)}", default=None,
        )
        self._call_usage: dict[str, dict[str, int | float]] = {}
        self._live_usage_path: Path | None = None
        self._live_usage_restored = False
        self._live_usage_write_lock = Lock()

    @property
    def live_usage_restored(self) -> bool:
        """Whether this budget restored provider calls from a previous process."""
        return self._live_usage_restored

    def enable_live_usage(self, path: str | Path) -> None:
        """Restore and atomically persist cumulative provider telemetry after every call."""
        target = Path(path).expanduser().absolute()
        if self._live_usage_path is not None:
            if self._live_usage_path != target:
                raise ValueError("live API usage is already bound to another run directory")
            return
        self._live_usage_path = target
        if target.exists():
            payload = json.loads(target.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
                raise ValueError("live API usage artifact has an unsupported schema")
            provider_usage = payload.get("provider_usage")
            call_usage = payload.get("api_usage_by_call_name")
            if not isinstance(provider_usage, Mapping) or not isinstance(call_usage, Mapping):
                raise ValueError("live API usage artifact is incomplete")
            self.restore_usage(BudgetSnapshot(**dict(provider_usage)))
            restored: dict[str, dict[str, int | float]] = {}
            integer_fields = (
                "calls", "successes", "failures", "prompt_tokens", "completion_tokens",
                "usage_responses", "usage_unavailable",
            )
            for call_name, counters in call_usage.items():
                if not isinstance(call_name, str) or not isinstance(counters, Mapping):
                    raise TypeError("live API call usage rows are malformed")
                row: dict[str, int | float] = {}
                for name in integer_fields:
                    item = counters.get(name, 0)
                    if type(item) is not int or item < 0:
                        raise ValueError("live API usage counters must be non-negative integers")
                    row[name] = item
                elapsed = counters.get("total_elapsed_seconds", 0.0)
                if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                        or elapsed < 0):
                    raise ValueError("live API elapsed time must be non-negative")
                row["total_elapsed_seconds"] = float(elapsed)
                if row["successes"] + row["failures"] != row["calls"]:
                    raise ValueError("live API success/failure counts are inconsistent")
                restored[call_name] = row
            with self._lock:
                self._call_usage = restored
            self._live_usage_restored = True
        self._persist_live_usage()

    def _persist_live_usage(self) -> None:
        target = self._live_usage_path
        if target is None:
            return
        # Serialize writes and take the newest snapshot only after acquiring this
        # lock, so a slower thread cannot replace a newer snapshot with stale data.
        with self._live_usage_write_lock:
            payload = {
                "schema_version": 1,
                "provider_usage": self.snapshot().to_dict(),
                "api_usage_by_call_name": self.api_usage_by_call_name(),
            }
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                    json.dump(payload, stream, ensure_ascii=False, indent=2)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_name, target)
                directory_fd = os.open(target.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)

    def reserve_episode_dispatch(self) -> ProviderDispatchReservation | None:
        """Reserve one first request so parallel episodes cannot oversubscribe the cap."""
        with self._lock:
            if (self._cap is not None
                    and self._attempts + self._in_flight + self._reserved_dispatches >= self._cap):
                return None
            self._reserved_dispatches += 1
            return ProviderDispatchReservation(self)

    def release_episode_dispatch(
        self, reservation: ProviderDispatchReservation | None,
    ) -> None:
        if reservation is None or reservation._budget is not self:
            return
        with self._lock:
            if reservation._released or reservation._consumed:
                return
            reservation._released = True
            self._reserved_dispatches -= 1

    def has_current_episode_reservation(self) -> bool:
        reservation = self._reservation_context.get()
        return bool(
            reservation is not None
            and reservation._budget is self
            and not reservation._consumed
            and not reservation._released
        )

    @contextmanager
    def use_episode_reservation(
        self, reservation: ProviderDispatchReservation | None,
    ) -> Iterator[None]:
        if reservation is not None and reservation._budget is not self:
            raise ValueError("episode provider reservation belongs to a different RequestBudget")
        token: Token[ProviderDispatchReservation | None] = self._reservation_context.set(reservation)
        try:
            yield
        finally:
            self._reservation_context.reset(token)
            self.release_episode_dispatch(reservation)

    @contextmanager
    def track_episode_usage(
        self, tracker: EpisodeUsageTracker | None = None,
    ) -> Iterator[EpisodeUsageTracker]:
        local_usage = tracker or EpisodeUsageTracker()
        token: Token[EpisodeUsageTracker | None] = self._usage_context.set(local_usage)
        try:
            yield local_usage
        finally:
            self._usage_context.reset(token)

    def _begin(self, model: str | None) -> int:
        model_id = self._model_key(model)
        tracker = self._usage_context.get()
        reservation = self._reservation_context.get()
        with self._lock:
            model_usage = self._model_counters(model_id)
            reserved_slot = (
                reservation is not None
                and reservation._budget is self
                and not reservation._consumed
                and not reservation._released
            )
            if reserved_slot:
                reservation._consumed = True
                self._reserved_dispatches -= 1
            if (self._cap is not None and not reserved_slot and
                    self._attempts + self._in_flight + self._reserved_dispatches >= self._cap):
                self._denied += 1
                model_usage["denied"] += 1
                if tracker is not None:
                    tracker.denied(model_id)
                raise ProviderBudgetExceeded(
                    f"provider request denied before dispatch: cap {self._cap} reached"
                )
            self._in_flight += 1
            model_usage["in_flight"] += 1
            return self._attempts + self._in_flight

    @staticmethod
    def _model_key(model: str | None) -> str:
        return model.strip() if isinstance(model, str) and model.strip() else "__unknown_model__"

    def _model_counters(self, model_id: str) -> dict[str, int]:
        return self._models.setdefault(model_id, {
            "attempts": 0, "successes": 0, "failures": 0, "denied": 0,
            "prompt_tokens": 0, "completion_tokens": 0,
            "usage_responses": 0, "usage_unavailable": 0, "cache_hits": 0,
            "in_flight": 0,
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
        model_id = self._model_key(model)
        tracker = self._usage_context.get()
        with self._lock:
            model_usage = self._model_counters(model_id)
            self._in_flight -= 1
            model_usage["in_flight"] -= 1
            self._attempts += 1
            model_usage["attempts"] += 1
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
        if tracker is not None:
            tracker.completed(model_id, succeeded=succeeded, response=response)

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
                in_flight=self._in_flight,
                reserved=self._reserved_dispatches,
                model_usage=tuple(
                    ModelUsageSnapshot(model_id=model_id, **counters)
                    for model_id, counters in self._models.items()
                ),
            )

    def api_usage_by_call_name(self) -> dict[str, dict[str, int | float]]:
        """Return per-native-call_name attempt, token, and latency totals."""
        with self._lock:
            return {
                name: {
                    **{key: int(value) for key, value in counters.items()
                       if key not in {"total_elapsed_seconds"}},
                    "total_elapsed_seconds": round(
                        float(counters["total_elapsed_seconds"]), 6,
                    ),
                    "average_elapsed_seconds": round(
                        float(counters["total_elapsed_seconds"])
                        / max(int(counters["calls"]), 1),
                        6,
                    ),
                }
                for name, counters in sorted(self._call_usage.items())
            }

    def _record_api_call(
        self,
        call_name: str,
        *,
        succeeded: bool,
        response: Any,
        elapsed_seconds: float,
    ) -> None:
        usage = self._field(response, "usage") if succeeded else None
        prompt_tokens = self._field(usage, "prompt_tokens")
        completion_tokens = self._field(usage, "completion_tokens")
        has_usage = (
            isinstance(prompt_tokens, int) and prompt_tokens >= 0
            and isinstance(completion_tokens, int) and completion_tokens >= 0
        )
        with self._lock:
            row = self._call_usage.setdefault(call_name, {
                "calls": 0,
                "successes": 0,
                "failures": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "usage_responses": 0,
                "usage_unavailable": 0,
                "total_elapsed_seconds": 0.0,
            })
            row["calls"] += 1
            row["successes" if succeeded else "failures"] += 1
            row["total_elapsed_seconds"] += max(0.0, elapsed_seconds)
            if has_usage:
                row["prompt_tokens"] += prompt_tokens
                row["completion_tokens"] += completion_tokens
                row["usage_responses"] += 1
            else:
                row["usage_unavailable"] += 1
        self._persist_live_usage()

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
        if (snapshot.successes + snapshot.failures != snapshot.attempts
                or (snapshot.cap is not None
                    and snapshot.attempts + snapshot.in_flight + snapshot.reserved > snapshot.cap)
                or snapshot.in_flight or snapshot.reserved):
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
                            "in_flight",
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
                    "in_flight": 0,
                }}

    def absorb_usage(self, snapshot: BudgetSnapshot) -> None:
        """Add attempts from a completed child phase to this run's global budget."""
        if snapshot.in_flight or snapshot.reserved:
            raise ValueError("cannot absorb provider usage while requests are active or reserved")
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
            if (self._cap is not None
                    and self._attempts + self._in_flight + self._reserved_dispatches
                    + snapshot.attempts > self._cap):
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
    def instrument_tau_llm_utils(
        self,
        llm_utils: ModuleType | Any,
        *,
        retry_empty_responses: bool = False,
    ) -> Iterator[None]:
        """Share one process-wide wrapper while concurrent episode scopes are active.

        The registry lock protects only installation and restoration. It is never
        held while provider code runs, so independent episodes can overlap. When
        enabled, structurally empty successful completions are retried with the
        same request, and every provider call is separately accounted.
        """

        if type(retry_empty_responses) is not bool:
            raise TypeError("retry_empty_responses must be boolean")

        module_key = id(llm_utils)
        with _INSTRUMENTATION_LOCK:
            state = _INSTRUMENTATION_STATES.get(module_key)
            if state is not None:
                if state.llm_utils is not llm_utils:
                    raise RuntimeError("provider instrumentation module identity was reused while active")
                if state.budget is not self:
                    raise RuntimeError("concurrent τ-bench scopes must share one RequestBudget")
                if state.retry_empty_responses != retry_empty_responses:
                    raise RuntimeError("concurrent τ-bench scopes must share empty-response retry policy")
                state.references += 1
            else:
                original_completion = llm_utils.completion
                original_generate = getattr(llm_utils, "generate", None)
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
                    call_name = self._call_name_context.get() or "unknown"
                    empty_attempt = 0
                    while True:
                        self._begin(model_id)
                        started = perf_counter()
                        try:
                            result = original_completion(*args, **kwargs)
                        except BaseException:
                            self._finish(succeeded=False, model=model_id)
                            self._record_api_call(
                                call_name,
                                succeeded=False,
                                response=None,
                                elapsed_seconds=perf_counter() - started,
                            )
                            raise
                        self._finish(succeeded=True, response=result, model=model_id)
                        self._record_api_call(
                            call_name,
                            succeeded=True,
                            response=result,
                            elapsed_seconds=perf_counter() - started,
                        )
                        if not retry_empty_responses or not _is_empty_completion(result):
                            return result
                        empty_attempt += 1
                        logging.getLogger(__name__).warning(
                            "Provider returned an empty completion; retrying the frozen request "
                            "(model=%s, empty_response_retry=%d)",
                            model_id,
                            empty_attempt,
                        )
                        sleep(1.0)

                def guarded_generate(*args: Any, **kwargs: Any) -> Any:
                    call_name = kwargs.get("call_name")
                    if call_name is None and len(args) > 4:
                        call_name = args[4]
                    token = self._call_name_context.set(
                        str(call_name) if call_name else "unknown",
                    )
                    try:
                        return original_generate(*args, **kwargs)
                    finally:
                        self._call_name_context.reset(token)

                generate_bindings = []
                if callable(original_generate):
                    modules = [llm_utils]
                    modules.extend(
                        module for module in tuple(sys.modules.values())
                        if module is not None
                        and getattr(module, "__name__", "").startswith("tau2.")
                    )
                    seen_modules = set()
                    for module in modules:
                        if id(module) in seen_modules:
                            continue
                        seen_modules.add(id(module))
                        if getattr(module, "generate", None) is original_generate:
                            generate_bindings.append((module, "generate", original_generate))
                            module.generate = guarded_generate

                state = _InstrumentationState(
                    budget=self,
                    llm_utils=llm_utils,
                    litellm_module=litellm_module,
                    original_completion=original_completion,
                    generate_bindings=generate_bindings,
                    original_default_retries=original_default_retries,
                    original_litellm_completion=original_litellm_completion,
                    guarded_completion=guarded_completion,
                    guarded_generate=(guarded_generate if generate_bindings else None),
                    retry_empty_responses=retry_empty_responses,
                )
                _INSTRUMENTATION_STATES[module_key] = state
                llm_utils.completion = guarded_completion
                llm_utils.DEFAULT_MAX_RETRIES = 0
                if litellm_module is not None and callable(original_litellm_completion):
                    litellm_module.completion = guarded_completion
        try:
            yield
        finally:
            with _INSTRUMENTATION_LOCK:
                state = _INSTRUMENTATION_STATES.get(module_key)
                if state is None or state.budget is not self:
                    raise RuntimeError("provider instrumentation registry lost its active scope")
                state.references -= 1
                if state.references == 0:
                    if llm_utils.completion is state.guarded_completion:
                        llm_utils.completion = state.original_completion
                    for module, name, original in reversed(state.generate_bindings):
                        if getattr(module, name, None) is state.guarded_generate:
                            setattr(module, name, original)
                    llm_utils.DEFAULT_MAX_RETRIES = state.original_default_retries
                    if (state.litellm_module is not None
                            and callable(state.original_litellm_completion)
                            and state.litellm_module.completion is state.guarded_completion):
                        state.litellm_module.completion = state.original_litellm_completion
                    del _INSTRUMENTATION_STATES[module_key]
