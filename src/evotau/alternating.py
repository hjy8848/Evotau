"""Prompt-first alternating evolution on the native τ-bench runtime."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .budget import RequestBudget
from .records import EpisodeRecord, customer_strategy_id, service_strategy_id
from .strategies import PromptStrategy
from .tau_provenance import sha256_json

CustomerEvolver = Callable[[Mapping[str, Any], int], Sequence[str]]
ServiceEvolver = Callable[[Mapping[str, Any]], Mapping[str, str]]
EpisodeRunner = Callable[..., EpisodeRecord]


@dataclass(frozen=True, slots=True)
class AlternatingResult:
    initial_customer: PromptStrategy
    initial_service: PromptStrategy
    customer: PromptStrategy
    service: PromptStrategy
    generations: tuple[Mapping[str, Any], ...]
    final_evolution_episodes: tuple[EpisodeRecord, ...]


class LLMAlternatingEvolvers:
    """Two language-model calls propose free-form Customer and Service strategies."""

    def __init__(
        self,
        *,
        model: str,
        model_args: Mapping[str, Any],
    ) -> None:
        self.model = model
        self.model_args = dict(model_args)

    def customer_candidates(
        self, context: Mapping[str, Any], count: int,
    ) -> tuple[str, ...]:
        result = self._json_call(
            self.model,
            self.model_args,
            "You evolve the Customer as a task-grounded adaptive challenge generator. "
            "Your goal is to discover REUSABLE INTERACTION SKILLS that expose weaknesses in the "
            "current Service. A strategy should describe HOW the Customer interacts, not WHAT "
            "specific entity from this task it should mention. Good strategies probe a nearby boundary "
            "after a state-changing action and ask whether another action is possible; ask for a concrete "
            "policy or backend-state reason after a rejection and test whether it stays grounded; delay "
            "or reorder relevant information to test unresolved-goal tracking; or introduce a correction "
            "while preserving the original user goal. Bad strategies name a task-specific product, order, "
            "person, address, or request, or introduce a new product, order, identity, fact, or user "
            "objective not grounded in the supplied scenario. Make the interaction harder, more persistent, "
            "indirect, or demanding without primarily changing the original task objective or inventing "
            "facts. Preserve the original user objective. Avoid task-specific entities: do not copy "
            "task-specific entity names, IDs, "
            "product names, user names, addresses, or literal requests into the strategy. The strategy "
            "must be reusable across different tasks in the same domain. Do not assume the Customer "
            "must follow Service business policy. Use the supplied task, user "
            "scenario, current strategies, accuracy, and real conversations/tool results to infer a "
            "general interaction skill. Keep the underlying τ-bench task, "
            "policy, tools, backend, and evaluator fixed. Do not use reference answers or hidden data. "
            "Return only JSON: {\"candidates\": [\"reusable natural-language interaction strategy\", ...]}.",
            {"requested_candidates": count, **dict(context)},
            call_name="evotau_customer_evolver",
        )
        candidates = result["candidates"]
        if not isinstance(candidates, list) or any(not isinstance(item, str) for item in candidates):
            raise ValueError("Customer Evolver must return a JSON array of strategy strings")
        return tuple(candidates)

    def service_candidate(self, context: Mapping[str, Any]) -> Mapping[str, str]:
        result = self._json_call(
            self.model,
            self.model_args,
            "You evolve the Service from native τ-bench task outcomes. Study the fixed task policy, "
            "the current Service strategy, Customer strategy, E-panel accuracy, real interaction "
            "trajectories, tool results, and task-success results. Propose a reusable natural-language "
            "Service strategy that improves task accuracy without changing tasks or policy. "
            "Do not modify tasks, policy, tools, backend, or evaluator. "
            "Do not use reference answers or hidden data. Return only JSON with string fields "
            "`analysis` and `strategy`.",
            dict(context),
            call_name="evotau_service_evolver",
        )
        if not isinstance(result.get("analysis"), str) or not isinstance(result.get("strategy"), str):
            raise TypeError("Service Evolver must return string analysis and strategy fields")
        return {"analysis": result["analysis"], "strategy": result["strategy"]}

    @staticmethod
    def _json_call(
        model: str,
        model_args: Mapping[str, Any],
        system_prompt: str,
        context: Mapping[str, Any],
        *,
        call_name: str,
    ) -> dict[str, Any]:
        from tau2.data_model.message import SystemMessage, UserMessage
        from tau2.utils.llm_utils import generate

        message = generate(
            model=model,
            messages=[
                SystemMessage(role="system", content=system_prompt),
                UserMessage(
                    role="user",
                    content=json.dumps(context, ensure_ascii=False, sort_keys=True),
                ),
            ],
            call_name=call_name,
            num_retries=0,
            **dict(model_args),
        )
        try:
            value = json.loads(message.content or "")
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Evolver returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise TypeError("evolver responses must be JSON objects")
        return value


def run_alternating_evolution(
    *,
    tasks: Mapping[str, Any],
    evolution_task_ids: Sequence[str],
    validation_task_ids: Sequence[str],
    seed: int,
    generations: int,
    customer_candidate_count: int,
    clean_panel_size: int,
    initial_customer: PromptStrategy,
    initial_service: PromptStrategy,
    runner: EpisodeRunner,
    customer_evolver: CustomerEvolver,
    service_evolver: ServiceEvolver,
    domain_policy: str,
    request_budget: RequestBudget | None = None,
    output_directory: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    manifest_sha256: str | None = None,
) -> AlternatingResult:
    """Alternate Customer challenge and Service repair using native task accuracy."""

    e_tasks = tuple(str(item) for item in evolution_task_ids)
    v_tasks = tuple(str(item) for item in validation_task_ids[:clean_panel_size])
    if not e_tasks or not v_tasks or len(set(e_tasks)) != len(e_tasks):
        raise ValueError("alternating evolution needs unique E tasks and a small V clean panel")
    missing = (set(e_tasks) | set(v_tasks)) - set(tasks)
    if missing:
        raise ValueError(f"task loader is missing E/V tasks: {sorted(missing)}")
    if generations < 1 or customer_candidate_count < 1 or seed < 0:
        raise ValueError("seed, generation count, and candidate count must be non-negative")

    output_root = None if output_directory is None else Path(output_directory)
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_file = (
        Path(checkpoint_path) if checkpoint_path is not None
        else None if output_root is None else output_root / "checkpoint.json"
    )
    initial_customer_state = initial_customer
    initial_service_state = initial_service
    customer, service = initial_customer, initial_service
    history: list[dict[str, Any]] = []
    generation_documents: list[Mapping[str, Any]] = []
    final_evolution_episodes: tuple[EpisodeRecord, ...] = ()
    start_generation = 0
    if checkpoint_file is not None and checkpoint_file.exists():
        checkpoint = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        if checkpoint.get("manifest_sha256") != manifest_sha256:
            raise ValueError("alternating checkpoint belongs to a different frozen manifest")
        if checkpoint.get("initial_customer") != initial_customer_state.to_dict() or checkpoint.get(
            "initial_service",
        ) != initial_service_state.to_dict():
            raise ValueError("alternating checkpoint initial strategies differ from the current run")
        generation_documents = list(checkpoint.get("generations", ()))
        history = [dict(item) for item in checkpoint.get("history", ())]
        customer = PromptStrategy(checkpoint["customer"]["text"])
        service = PromptStrategy(checkpoint["service"]["text"])
        final_evolution_episodes = tuple(
            EpisodeRecord.from_dict(item)
            for item in checkpoint.get("final_evolution_episodes", ())
        )
        start_generation = int(checkpoint["completed_generation"]) + 1
        if not 0 <= start_generation <= generations:
            raise ValueError("alternating checkpoint generation is outside the frozen run")

    for generation in range(start_generation, generations):
        generation_seed = seed + generation
        customer_before = customer
        service_before = service
        budget_before = None if request_budget is None else request_budget.snapshot().to_dict()

        incumbent_runs = _run_panel(
            runner,
            task_ids=e_tasks,
            tasks=tasks,
            seed=generation_seed,
            customer=customer,
            service=service,
            panel_name=f"generation-{generation}-customer-incumbent",
        )
        incumbent_accuracy = _accuracy(incumbent_runs)
        proposal_context = {
            "generation": generation,
            "task_interactions": _context_episodes(incumbent_runs, runner, tasks),
            "incumbent_accuracy": incumbent_accuracy,
            "service_policy": domain_policy,
            "current_customer_strategy": customer.text,
            "current_service_strategy": service.text,
            "accuracy_history": history,
        }
        candidate_texts = tuple(_provider_call(
            request_budget, customer_evolver, proposal_context, customer_candidate_count,
        ))[:customer_candidate_count]
        candidates = tuple(PromptStrategy(text) for text in candidate_texts)
        candidate_runs = tuple(
            _run_panel(
                runner,
                task_ids=e_tasks,
                tasks=tasks,
                seed=generation_seed,
                customer=candidate,
                service=service,
                panel_name=f"generation-{generation}-customer-candidate-{index}",
            )
            for index, candidate in enumerate(candidates)
        )
        candidate_accuracies = [_accuracy(runs) for runs in candidate_runs]

        selected_customer_source: str | int = "incumbent"
        selected_runs = incumbent_runs
        selected_accuracy = incumbent_accuracy
        for index, (candidate, runs, accuracy) in enumerate(
            zip(candidates, candidate_runs, candidate_accuracies, strict=True),
        ):
            # Strict comparison makes an exact tie keep the incumbent. Candidate
            # ties are stable: the first candidate at the lowest score wins.
            if accuracy < selected_accuracy:
                customer = candidate
                selected_customer_source = index
                selected_runs = runs
                selected_accuracy = accuracy

        old_service = service
        old_accuracy = selected_accuracy
        service_context = {
            "generation": generation,
            "task_interactions": _service_context_episodes(selected_runs, runner, tasks),
            "selected_customer_accuracy": selected_accuracy,
            "customer_strategy": customer.text,
            "current_service_strategy": service.text,
            "service_policy": domain_policy,
            "accuracy_history": history,
        }
        service_proposal = _provider_call(request_budget, service_evolver, service_context)
        service_analysis = service_proposal.get("analysis", "")
        proposed_text = service_proposal.get("strategy", service.text)
        if not isinstance(service_analysis, str) or not isinstance(proposed_text, str):
            raise TypeError("Service Evolver must return natural-language analysis and strategy")
        proposed_service = PromptStrategy(proposed_text)

        proposed_runs: tuple[EpisodeRecord, ...] = ()
        proposed_accuracy = old_accuracy
        improved_on_e = False
        validation_old_accuracy: float | None = None
        validation_new_accuracy: float | None = None
        validation_old_runs: tuple[EpisodeRecord, ...] = ()
        validation_new_runs: tuple[EpisodeRecord, ...] = ()
        accepted = False
        service_reason = "The Service strategy did not change."
        if proposed_service.text != service.text:
            proposed_runs = _run_panel(
                runner,
                task_ids=e_tasks,
                tasks=tasks,
                seed=generation_seed,
                customer=customer,
                service=proposed_service,
                panel_name=f"generation-{generation}-service-candidate",
            )
            proposed_accuracy = _accuracy(proposed_runs)
            improved_on_e = proposed_accuracy > old_accuracy
            if not improved_on_e:
                service_reason = (
                    "Rejected by native E accuracy: proposed Service accuracy did not exceed "
                    "the old Service accuracy."
                )
            else:
                validation_old_runs = _run_panel(
                    runner,
                    task_ids=v_tasks,
                    tasks=tasks,
                    seed=seed,
                    customer=None,
                    service=old_service,
                    panel_name=f"generation-{generation}-validation-old-native-customer",
                )
                validation_new_runs = _run_panel(
                    runner,
                    task_ids=v_tasks,
                    tasks=tasks,
                    seed=seed,
                    customer=None,
                    service=proposed_service,
                    panel_name=f"generation-{generation}-validation-proposed-native-customer",
                )
                validation_old_accuracy = _accuracy(validation_old_runs)
                validation_new_accuracy = _accuracy(validation_new_runs)
                if validation_new_accuracy >= validation_old_accuracy:
                    service = proposed_service
                    accepted = True
                    service_reason = "Accepted: E accuracy improved and native V accuracy did not decrease."
                else:
                    service_reason = "Rejected: native V accuracy decreased."

        # For the final fresh challenge, keep evidence for the actual final
        # (Customer, Service) pair. If a proposed Service is rejected, use the
        # selected Customer's incumbent-Service episodes instead.
        final_evolution_episodes = proposed_runs if accepted else selected_runs

        customer_phase = {
            "frozen_service": _strategy_document("service", old_service),
            "evolver_input_sha256": sha256_json(proposal_context),
            "incumbent_accuracy": incumbent_accuracy,
            "candidate_accuracies": candidate_accuracies,
            "selected_accuracy": selected_accuracy,
            "selected_customer": selected_customer_source,
            "incumbent_episodes": [_episode_ref(item) for item in incumbent_runs],
            "candidates": [
                {
                    "strategy": candidate.to_dict(),
                    "accuracy": accuracy,
                    "episodes": [_episode_ref(item) for item in runs],
                }
                for candidate, accuracy, runs in zip(
                    candidates, candidate_accuracies, candidate_runs, strict=True,
                )
            ],
        }
        service_phase = {
            "frozen_customer": _strategy_document("customer", customer),
            "analysis": service_analysis,
            "evolver_input_sha256": sha256_json(service_context),
            "proposed_strategy": proposed_service.to_dict(),
            "old_accuracy": old_accuracy,
            "proposed_accuracy": proposed_accuracy,
            "improved_on_E": improved_on_e,
            "accepted": accepted,
            "validation_old_accuracy": validation_old_accuracy,
            "validation_new_accuracy": validation_new_accuracy,
            "challenge_episodes": [_episode_ref(item) for item in proposed_runs],
            "selection": {"reason": service_reason},
            "clean_panel": {
                "customer": "native τ-bench customer",
                "evaluated": validation_old_accuracy is not None,
                "reference_accuracy": validation_old_accuracy,
                "candidate_accuracy": validation_new_accuracy,
                "reference_episodes": [_episode_ref(item) for item in validation_old_runs],
                "candidate_episodes": [_episode_ref(item) for item in validation_new_runs],
            },
        }
        generation_doc = {
            "schema_version": 2,
            "generation": generation,
            "seed": generation_seed,
            "customer_before": _strategy_document("customer", customer_before),
            "service_before": _strategy_document("service", service_before),
            "customer_after": _strategy_document("customer", customer),
            "service_after": _strategy_document("service", service),
            "customer_phase": customer_phase,
            "service_phase": service_phase,
            "provider_budget": _budget_delta(
                budget_before,
                None if request_budget is None else request_budget.snapshot().to_dict(),
            ),
        }
        generation_documents.append(generation_doc)
        history.append({
            "generation": generation,
            "customer_incumbent_accuracy": incumbent_accuracy,
            "customer_candidate_accuracies": candidate_accuracies,
            "customer_selected_accuracy": selected_accuracy,
            "service_old_accuracy": old_accuracy,
            "service_proposed_accuracy": proposed_accuracy,
            "service_accepted": accepted,
        })
        if output_root is not None:
            _write_json_atomic(output_root / f"generation-{generation:04d}.json", generation_doc)
        if checkpoint_file is not None:
            _write_json_atomic(
                checkpoint_file,
                {
                    "schema_version": 2,
                    "manifest_sha256": manifest_sha256,
                    "completed_generation": generation,
                    "initial_customer": initial_customer_state.to_dict(),
                    "initial_service": initial_service_state.to_dict(),
                    "customer": customer.to_dict(),
                    "service": service.to_dict(),
                    "generations": generation_documents,
                    "history": history,
                    "final_evolution_episodes": [
                        episode.to_dict() for episode in final_evolution_episodes
                    ],
                },
            )

    return AlternatingResult(
        initial_customer_state,
        initial_service_state,
        customer,
        service,
        tuple(generation_documents),
        final_evolution_episodes,
    )


def propose_fresh_customer_challenge(
    result: AlternatingResult,
    customer_evolver: CustomerEvolver,
    *,
    tasks: Mapping[str, Any],
    runner: EpisodeRunner,
    domain_policy: str,
    request_budget: RequestBudget | None = None,
) -> PromptStrategy:
    """Propose one fresh Customer strategy from the final saved E episodes."""

    if not result.final_evolution_episodes:
        raise ValueError("final saved E episodes are required to propose the fresh Customer")
    context = {
        "generation": len(result.generations),
        "task_interactions": _context_episodes(
            result.final_evolution_episodes, runner, tasks,
        ),
        "purpose": "Create one reusable Customer challenge skill for held-out evaluation.",
        "final_evolution_accuracy": _accuracy(result.final_evolution_episodes),
        "current_customer_strategy": result.customer.text,
        "current_service_strategy": result.service.text,
        "service_policy": domain_policy,
        "accuracy_history": [dict(item) for item in result.generations],
    }
    candidates = tuple(_provider_call(request_budget, customer_evolver, context, 1))
    if not candidates:
        return result.customer
    return PromptStrategy(candidates[0])


def run_final_endpoint_evaluation(
    *,
    heldout_tasks: Mapping[str, Any],
    heldout_task_ids: Sequence[str],
    seed: int,
    initial_service: PromptStrategy,
    final_service: PromptStrategy,
    fresh_customer: PromptStrategy,
    runner: EpisodeRunner,
    output_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Compare S0 and ST endpoints without rerunning identical Service cells."""

    task_ids = tuple(str(item) for item in heldout_task_ids)
    if not task_ids or set(task_ids) - set(heldout_tasks):
        raise ValueError("final endpoint evaluation requires loaded H tasks")
    identical_services = final_service.text == initial_service.text
    customers = (
        ("native_customer", None),
        ("fresh_adaptive_customer", fresh_customer),
    )
    rows = []
    s0_episodes_by_customer: dict[str, tuple[EpisodeRecord, ...]] = {}
    for customer_label, customer in customers:
        episodes = _run_panel(
            runner,
            task_ids=task_ids,
            tasks=heldout_tasks,
            seed=seed,
            customer=customer,
            service=initial_service,
            panel_name=f"heldout-{customer_label}-S0",
        )
        s0_episodes_by_customer[customer_label] = episodes
        rows.append({
            "customer_condition": customer_label,
            "service_endpoint": "S0",
            "identical_to_S0": False,
            "episodes": [_episode_ref(item) for item in episodes],
        })
        if identical_services:
            rows.append({
                "customer_condition": customer_label,
                "service_endpoint": "ST",
                "identical_to_S0": True,
                "episodes": [_episode_ref(item) for item in episodes],
            })
        else:
            updated_episodes = _run_panel(
                runner,
                task_ids=task_ids,
                tasks=heldout_tasks,
                seed=seed,
                customer=customer,
                service=final_service,
                panel_name=f"heldout-{customer_label}-ST",
            )
            rows.append({
                "customer_condition": customer_label,
                "service_endpoint": "ST",
                "identical_to_S0": False,
                "episodes": [_episode_ref(item) for item in updated_episodes],
            })
    result = {
        "schema_version": 2,
        "heldout_task_ids": list(task_ids),
        "seed": seed,
        "initial_service": _strategy_document("service", initial_service),
        "final_service": _strategy_document("service", final_service),
        "services_identical": identical_services,
        "fresh_customer": fresh_customer.to_dict(),
        "cells": rows,
    }
    if output_path is not None:
        _write_json_atomic(Path(output_path), result)
    return result


def _run_panel(
    runner: EpisodeRunner,
    *,
    task_ids: Sequence[str],
    tasks: Mapping[str, Any],
    seed: int,
    customer: PromptStrategy | None,
    service: PromptStrategy,
    panel_name: str,
) -> tuple[EpisodeRecord, ...]:
    episodes = []
    for task_id in task_ids:
        if task_id not in tasks:
            raise ValueError(f"task {task_id!r} was not loaded for {panel_name}")
        episode = runner(
            task_id=str(task_id),
            seed=seed,
            customer=customer,
            service=service,
            panel_name=panel_name,
        )
        if not isinstance(episode, EpisodeRecord):
            raise TypeError("τ-bench runner must return an EpisodeRecord")
        if episode.task_id != str(task_id) or episode.seed != seed:
            raise ValueError("τ-bench runner returned a different task or seed")
        episodes.append(episode)
    return tuple(episodes)


def _context_episodes(
    episodes: Sequence[EpisodeRecord],
    runner: EpisodeRunner,
    tasks: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build Customer-side examples, including the Customer's source scenario."""

    return _project_context_episodes(episodes, runner, tasks, include_user_scenario=True)


def _service_context_episodes(
    episodes: Sequence[EpisodeRecord],
    runner: EpisodeRunner,
    tasks: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build Service-side examples from observed interactions, never hidden scenarios."""

    return _project_context_episodes(episodes, runner, tasks, include_user_scenario=False)


def _project_context_episodes(
    episodes: Sequence[EpisodeRecord],
    runner: EpisodeRunner,
    tasks: Mapping[str, Any],
    *,
    include_user_scenario: bool,
) -> list[dict[str, Any]]:
    loader = getattr(runner, "load_trajectory", None)
    rows = []
    for episode in episodes:
        trajectory = None if loader is None else loader(episode)
        rows.append({
            "task": _task_context(
                tasks[episode.task_id], include_user_scenario=include_user_scenario,
            ),
            "customer_strategy_id": episode.customer_strategy_id,
            "service_strategy_id": episode.service_strategy_id,
            "seed": episode.seed,
            "trajectory": _trajectory_context(trajectory),
            "tool_results": _tool_results(trajectory),
            "native_evaluation": {
                "task_success": episode.task_success,
                "reward": episode.native_reward,
                "termination_reason": episode.termination_reason,
            },
            "trajectory_ref": episode.trajectory_ref,
        })
    return rows


def _task_context(task: Any, *, include_user_scenario: bool) -> dict[str, Any]:
    """Project public task metadata, with the source scenario only for Customer-side use."""

    context = {
        "task_id": str(getattr(task, "id", "")),
        "description": str(getattr(task, "description", "")),
        "user_tools": list(getattr(task, "user_tools", ()) or ()),
    }
    if include_user_scenario:
        context["user_scenario"] = str(getattr(task, "user_scenario", ""))
    return context


def _trajectory_context(trajectory: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(trajectory, Mapping):
        return {"messages": [], "termination_reason": None}
    messages = trajectory.get("messages", ())
    projected = []
    for message in messages if isinstance(messages, Sequence) else ():
        if isinstance(message, Mapping):
            projected.append({
                key: message[key]
                for key in ("role", "content", "name", "tool_calls", "tool_call_id")
                if key in message
            })
        elif hasattr(message, "model_dump"):
            data = message.model_dump(mode="json")
            projected.append({
                key: data[key]
                for key in ("role", "content", "name", "tool_calls", "tool_call_id")
                if key in data
            })
    return {
        "messages": projected,
        "termination_reason": trajectory.get("termination_reason"),
    }


def _tool_results(trajectory: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(trajectory, Mapping):
        return []
    results = []
    for message in _trajectory_context(trajectory)["messages"]:
        if message.get("role") == "tool" or message.get("tool_call_id"):
            results.append(message)
    return results


def _accuracy(episodes: Sequence[EpisodeRecord]) -> float:
    if not episodes:
        raise ValueError("accuracy requires at least one task episode")
    return sum(episode.task_success is True for episode in episodes) / len(episodes)


def _episode_ref(episode: EpisodeRecord) -> dict[str, Any]:
    return {
        "episode_id": episode.episode_id,
        "task_id": episode.task_id,
        "seed": episode.seed,
        "customer_strategy_id": episode.customer_strategy_id,
        "service_strategy_id": episode.service_strategy_id,
        "status": episode.status.value,
        "task_success": episode.task_success,
        "native_reward": episode.native_reward,
        "trajectory_ref": episode.trajectory_ref,
    }


def _strategy_document(role: str, strategy: PromptStrategy) -> dict[str, str]:
    return {
        "role": role,
        "strategy_id": customer_strategy_id(strategy) if role == "customer"
        else service_strategy_id(strategy),
        "strategy": strategy.text,
    }


def _budget_delta(before: Mapping[str, Any] | None, after: Mapping[str, Any] | None) -> Any:
    if before is None or after is None:
        return None
    names = (
        "attempts", "successes", "failures", "denied", "prompt_tokens",
        "completion_tokens", "usage_responses", "usage_unavailable", "cache_hits",
    )
    old_models = {row["model_id"]: row for row in before.get("model_usage", ())}
    new_models = {row["model_id"]: row for row in after.get("model_usage", ())}
    return {
        "cap": after.get("cap"),
        **{name: int(after.get(name, 0)) - int(before.get(name, 0)) for name in names},
        "model_usage": [
            {
                "model_id": model_id,
                **{
                    name: int(new_models.get(model_id, {}).get(name, 0))
                    - int(old_models.get(model_id, {}).get(name, 0))
                    for name in names
                },
            }
            for model_id in sorted(set(old_models) | set(new_models))
        ],
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _provider_call(budget: RequestBudget | None, callback: Callable[..., Any], *args: Any) -> Any:
    if budget is None:
        return callback(*args)
    from tau2.utils import llm_utils

    with budget.instrument_tau_llm_utils(llm_utils):
        return callback(*args)
