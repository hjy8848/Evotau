"""Bounded execution of immutable episode panels."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any

from .budget import ProviderBudgetExceeded, RequestBudget
from .records import EpisodeRecord, customer_strategy_id, service_strategy_id
from .strategies import CustomerStrategy, ServiceStrategy


class StopBeforeEpisodeDispatch(RuntimeError):
    """A pause request stopped the coordinator before its next episode dispatch."""


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    """Frozen inputs for one episode in a panel evaluation batch."""

    task_id: str
    seed: int
    customer: CustomerStrategy | None
    service: ServiceStrategy
    panel_name: str
    generation: int

    def __post_init__(self) -> None:
        if not self.task_id or not self.panel_name:
            raise ValueError("episode specs require a task ID and panel name")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("episode specs require a non-negative seed")
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("episode specs require a non-negative generation")

    @property
    def customer_id(self) -> str:
        return customer_strategy_id(self.customer)

    @property
    def service_id(self) -> str:
        return service_strategy_id(self.service)

    def runner_kwargs(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "customer": self.customer,
            "service": self.service,
            "panel_name": self.panel_name,
        }


def run_episode_batch(
    runner: Callable[..., EpisodeRecord],
    specs: tuple[EpisodeSpec, ...],
    *,
    max_concurrency: int,
    request_budget: RequestBudget | None = None,
    reserve_episode: Callable[[EpisodeSpec], bool] | None = None,
    release_episode: Callable[[EpisodeSpec], None] | None = None,
    needs_provider_request: Callable[[EpisodeSpec], bool] | None = None,
    should_pause: Callable[[], bool] | None = None,
    episode_finished: Callable[[EpisodeSpec], None] | None = None,
    batch_finished: Callable[[], None] | None = None,
) -> tuple[EpisodeRecord, ...]:
    """Run a frozen panel with bounded workers and deterministic aggregation.

    At most ``max_concurrency`` episode specs are submitted. The coordinator
    stops dispatching after a pause, budget/episode-cap boundary, or worker
    error, drains already-dispatched workers, and only then returns or raises.
    """

    if type(max_concurrency) is not int or not 1 <= max_concurrency <= 4:
        raise ValueError("episode max_concurrency must be in the range 1..4")
    if not specs:
        return ()

    results: list[EpisodeRecord | None] = [None] * len(specs)
    errors: dict[int, BaseException] = {}
    stop_reason: str | None = None
    next_index = 0
    futures: dict[Future[EpisodeRecord], int] = {}

    def dispatch(index: int, executor: ThreadPoolExecutor | None) -> bool:
        nonlocal stop_reason
        spec = specs[index]
        if should_pause is not None and should_pause():
            stop_reason = "paused"
            return False
        if reserve_episode is not None and not reserve_episode(spec):
            stop_reason = "episode_cap"
            return False
        budget_reservation = None
        if (request_budget is not None
                and (needs_provider_request is None or needs_provider_request(spec))):
            budget_reservation = request_budget.reserve_episode_dispatch()
            if budget_reservation is None:
                if release_episode is not None:
                    release_episode(spec)
                stop_reason = "provider_budget"
                return False

        def invoke() -> EpisodeRecord:
            try:
                if request_budget is None:
                    return runner(**spec.runner_kwargs())
                with request_budget.use_episode_reservation(budget_reservation):
                    return runner(**spec.runner_kwargs())
            finally:
                if episode_finished is not None:
                    episode_finished(spec)

        try:
            if executor is None:
                results[index] = invoke()
            else:
                futures[executor.submit(invoke)] = index
        except BaseException:
            if request_budget is not None:
                request_budget.release_episode_dispatch(budget_reservation)
            if release_episode is not None:
                release_episode(spec)
            raise
        return True

    executor = None if max_concurrency == 1 else ThreadPoolExecutor(
        max_workers=max_concurrency, thread_name_prefix="evotau-episode",
    )
    try:
        if executor is None:
            while next_index < len(specs):
                if not dispatch(next_index, None):
                    break
                next_index += 1
        else:
            while next_index < len(specs) or futures:
                while (stop_reason is None and next_index < len(specs)
                       and len(futures) < max_concurrency):
                    if not dispatch(next_index, executor):
                        break
                    next_index += 1
                if not futures:
                    break
                completed, _pending = wait(tuple(futures), return_when=FIRST_COMPLETED)
                for future in sorted(completed, key=lambda item: futures[item]):
                    index = futures.pop(future)
                    try:
                        results[index] = future.result()
                    except StopBeforeEpisodeDispatch:
                        stop_reason = "paused"
                    except Exception as exc:  # noqa: BLE001 - record worker errors, then drain peers
                        errors[index] = exc
                        stop_reason = "worker_error"
                # No more specs are dispatched after any worker asks to pause or fails.
                if stop_reason in {"paused", "worker_error"}:
                    continue
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        if batch_finished is not None:
            batch_finished()

    if errors:
        raise errors[min(errors)]
    if stop_reason == "paused":
        raise StopBeforeEpisodeDispatch("paused after active episodes finished; no queued episode was dispatched")
    if stop_reason == "provider_budget":
        raise ProviderBudgetExceeded("provider budget exhausted before the full frozen panel was dispatched")
    if stop_reason == "episode_cap":
        raise RuntimeError("episode cap reached before the full frozen panel was dispatched")
    if any(item is None for item in results):
        raise RuntimeError("episode batch completed without a result for every frozen spec")
    return tuple(item for item in results if item is not None)
