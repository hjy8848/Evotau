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
CustomerJudge = Callable[[Mapping[str, Any]], Mapping[str, Any]]
ServiceJudge = Callable[[Mapping[str, Any]], Mapping[str, Any]]
EpisodeRunner = Callable[..., EpisodeRecord]


@dataclass(frozen=True, slots=True)
class AlternatingResult:
    initial_customer: PromptStrategy
    initial_service: PromptStrategy
    customer: PromptStrategy
    service: PromptStrategy
    generations: tuple[Mapping[str, Any], ...]


class LLMAlternatingEvolvers:
    """Four small language-model calls; strategies remain free-form text."""

    def __init__(
        self,
        *,
        model: str,
        model_args: Mapping[str, Any],
        judge_model: str,
        judge_model_args: Mapping[str, Any],
    ) -> None:
        self.model = model
        self.model_args = dict(model_args)
        self.judge_model = judge_model
        self.judge_model_args = dict(judge_model_args)

    def customer_candidates(
        self, context: Mapping[str, Any], count: int,
    ) -> tuple[str, ...]:
        result = self._json_call(
            self.model,
            self.model_args,
            "You evolve the Customer as a task-grounded adaptive challenge generator. "
            "Find natural interaction strategies that search the current Service's failure boundary. "
            "Use the supplied τ-bench task, user scenario, current strategies, real conversation, "
            "tool results, native evaluation, reviewer feedback, and history. Keep the underlying "
            "τ-bench task, scenario, policy, tools, backend, and evaluator fixed. Do not assume the "
            "Customer must follow Service business policy. Do not use reference answers or hidden data. "
            "Return only JSON: {\"candidates\": [natural-language strategy strings]}.",
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
            "You evolve the Service from actual failures. Study the fixed τ-bench task policy, "
            "the current Service strategy, real interaction trajectory, tool results, native "
            "evaluation, and reviewer feedback. Propose a natural-language Service strategy that "
            "addresses the observed weakness. Do not modify tasks, policy, tools, backend, or evaluator. "
            "Do not use reference answers or hidden data. Return only JSON with string fields "
            "`analysis` and `strategy`.",
            dict(context),
            call_name="evotau_service_evolver",
        )
        if not isinstance(result.get("analysis"), str) or not isinstance(result.get("strategy"), str):
            raise TypeError("Service Evolver must return string analysis and strategy fields")
        return {"analysis": result["analysis"], "strategy": result["strategy"]}

    def choose_customer(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._json_call(
            self.judge_model,
            self.judge_model_args,
            "Compare the incumbent Customer and its candidates using only their actual τ-bench "
            "trajectories, tool results, native evaluation, and reviewer feedback. Choose the one "
            "that best exposes a real weakness in the frozen Service while still engaging with the "
            "given task. Do not apply a fixed challenge taxonomy. Return JSON with `choice` (the "
            "string `incumbent` or a zero-based candidate index) and a short `reason`.",
            dict(context),
            call_name="evotau_customer_selection",
            judge=True,
        )

    def service_is_better(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._json_call(
            self.judge_model,
            self.judge_model_args,
            "Compare the old and proposed Service on the same evolved Customer challenge. Decide "
            "whether the proposal handles the observed challenge better under the fixed τ-bench policy, "
            "using the real trajectories, tool results, native evaluation, and reviewer feedback. "
            "Do not treat a Customer's policy conflict as a reason to make the Customer obey Service "
            "policy. Return JSON with boolean `improved` and a short `reason`.",
            dict(context),
            call_name="evotau_service_selection",
            judge=True,
        )

    @staticmethod
    def _json_call(
        model: str,
        model_args: Mapping[str, Any],
        system_prompt: str,
        context: Mapping[str, Any],
        *,
        call_name: str,
        judge: bool = False,
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
            role = "judge" if judge else "evolver"
            raise ValueError(f"{role} returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise TypeError("evolver and judge responses must be JSON objects")
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
    customer_judge: CustomerJudge,
    service_judge: ServiceJudge,
    domain_policy: str,
    request_budget: RequestBudget | None = None,
    output_directory: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    manifest_sha256: str | None = None,
    prior_feedback: Sequence[Mapping[str, Any]] = (),
) -> AlternatingResult:
    """Run complete ``C_t → C_(t+1) → S_(t+1)`` generations in order."""

    e_tasks = tuple(str(item) for item in evolution_task_ids)
    v_tasks = tuple(str(item) for item in validation_task_ids[:clean_panel_size])
    if not e_tasks or not v_tasks or len(set(e_tasks)) != len(e_tasks):
        raise ValueError("alternating evolution needs unique E tasks and a small V clean panel")
    missing = (set(e_tasks) | set(v_tasks)) - set(tasks)
    if missing:
        raise ValueError(f"task loader is missing E/V tasks: {sorted(missing)}")
    if generations < 1 or customer_candidate_count < 1 or seed < 0:
        raise ValueError("seed, generations, and customer candidate count must be non-negative")

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
    history = [dict(item) for item in prior_feedback]
    generation_documents: list[Mapping[str, Any]] = []
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
        start_generation = int(checkpoint["completed_generation"]) + 1
        if not 0 <= start_generation <= generations:
            raise ValueError("alternating checkpoint generation is outside the frozen run")

    for generation in range(start_generation, generations):
        generation_seed = seed + generation
        customer_before = customer
        service_before = service
        budget_before = None if request_budget is None else request_budget.snapshot().to_dict()

        # Customer phase: the current Service value is passed to every E run
        # and is not changed until all Customer selection is complete.
        incumbent_runs = _run_panel(
            runner,
            task_ids=e_tasks,
            tasks=tasks,
            seed=generation_seed,
            customer=customer,
            service=service,
            panel_name=f"generation-{generation}-customer-incumbent",
        )
        incumbent_context = _context_episodes(incumbent_runs, runner, tasks)
        proposal_context = {
            "generation": generation,
            "task_interactions": incumbent_context,
            "service_policy": domain_policy,
            "current_customer_strategy": customer.text,
            "current_service_strategy": service.text,
            "history": history,
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
        customer_decision: Mapping[str, Any]
        if candidates:
            selection_context = {
                "generation": generation,
                "task_interactions": incumbent_context,
                "incumbent_strategy": customer.text,
                "incumbent_episodes": incumbent_context,
                "candidates": [
                    {
                        "strategy": candidate.text,
                        "episodes": _context_episodes(runs, runner, tasks),
                    }
                    for candidate, runs in zip(candidates, candidate_runs, strict=True)
                ],
                "service_strategy": service.text,
                "service_policy": domain_policy,
                "history": history,
            }
            customer_decision = _provider_call(request_budget, customer_judge, selection_context)
            choice = customer_decision.get("choice", "incumbent")
            if isinstance(choice, bool):
                raise ValueError("Customer judge choice must be `incumbent` or a candidate index")
            if isinstance(choice, int):
                if choice < 0 or choice >= len(candidates):
                    raise ValueError("Customer judge selected a candidate that was not run")
                customer = candidates[choice]
                selected_runs = candidate_runs[choice]
            elif choice == "incumbent":
                selected_runs = incumbent_runs
            else:
                raise ValueError("Customer judge choice must be `incumbent` or a candidate index")
        else:
            customer_decision = {"choice": "incumbent", "reason": "No candidate was returned."}
            selected_runs = incumbent_runs

        # Service phase: the selected Customer is fixed for both old and new
        # Service runs, including the small τ-bench native-Customer clean panel.
        service_context = {
            "generation": generation,
            "task_interactions": _service_context_episodes(selected_runs, runner, tasks),
            "customer_strategy": customer.text,
            "current_service_strategy": service.text,
            "service_policy": domain_policy,
            "history": history,
        }
        service_proposal = _provider_call(request_budget, service_evolver, service_context)
        service_analysis = service_proposal.get("analysis", "")
        proposed_text = service_proposal.get("strategy", service.text)
        if not isinstance(service_analysis, str) or not isinstance(proposed_text, str):
            raise TypeError("Service Evolver must return natural-language analysis and strategy")
        proposed_service = PromptStrategy(proposed_text)
        service_decision: Mapping[str, Any] = {
            "improved": False,
            "reason": "The Service strategy did not change.",
        }
        clean_regression: bool | None = None
        target_runs: tuple[EpisodeRecord, ...] = ()
        clean_reference_runs: tuple[EpisodeRecord, ...] = ()
        clean_candidate_runs: tuple[EpisodeRecord, ...] = ()
        old_service = service
        if proposed_service.text != service.text:
            target_runs = _run_panel(
                runner,
                task_ids=e_tasks,
                tasks=tasks,
                seed=generation_seed,
                customer=customer,
                service=proposed_service,
                panel_name=f"generation-{generation}-service-candidate",
            )
            service_selection_context = {
                "generation": generation,
                "customer_strategy": customer.text,
                "service_policy": domain_policy,
                "old_service_strategy": service.text,
                "proposed_service_strategy": proposed_service.text,
                "old_service_episodes": _service_context_episodes(selected_runs, runner, tasks),
                "proposed_service_episodes": _service_context_episodes(target_runs, runner, tasks),
            }
            service_decision = _provider_call(
                request_budget, service_judge, service_selection_context,
            )
            if service_decision.get("improved") is True:
                clean_reference_runs = _run_panel(
                    runner,
                    task_ids=v_tasks,
                    tasks=tasks,
                    seed=seed,
                    customer=None,
                    service=old_service,
                    panel_name=f"generation-{generation}-clean-reference-native-customer",
                )
                clean_candidate_runs = _run_panel(
                    runner,
                    task_ids=v_tasks,
                    tasks=tasks,
                    seed=seed,
                    customer=None,
                    service=proposed_service,
                    panel_name=f"generation-{generation}-clean-candidate-native-customer",
                )
                clean_regression = _has_clean_regression(
                    clean_reference_runs, clean_candidate_runs,
                )
                if not clean_regression:
                    service = proposed_service

        generation_doc = {
            "schema_version": 1,
            "generation": generation,
            "seed": generation_seed,
            "customer_before": _strategy_document("customer", customer_before),
            "service_before": _strategy_document("service", service_before),
            "customer_after": _strategy_document("customer", customer),
            "service_after": _strategy_document("service", service),
            "customer_phase": {
                "frozen_service": _strategy_document("service", old_service),
                "evolver_input_sha256": sha256_json(proposal_context),
                "selection_input_sha256": (
                    None if not candidates else sha256_json(selection_context)
                ),
                "incumbent_episodes": [_episode_ref(item) for item in incumbent_runs],
                "candidates": [
                    {
                        "strategy": candidate.to_dict(),
                        "episodes": [_episode_ref(item) for item in runs],
                    }
                    for candidate, runs in zip(candidates, candidate_runs, strict=True)
                ],
                "selection": dict(customer_decision),
            },
            "service_phase": {
                "frozen_customer": _strategy_document("customer", customer),
                "analysis": service_analysis,
                "evolver_input_sha256": sha256_json(service_context),
                "proposed_strategy": proposed_service.to_dict(),
                "challenge_episodes": [_episode_ref(item) for item in target_runs],
                "selection": dict(service_decision),
                "selection_input_sha256": (
                    None if not target_runs else sha256_json(service_selection_context)
                ),
                "clean_panel": {
                    "customer": "native τ-bench customer",
                    "evaluated": clean_regression is not None,
                    "reference_episodes": [_episode_ref(item) for item in clean_reference_runs],
                    "candidate_episodes": [_episode_ref(item) for item in clean_candidate_runs],
                    "catastrophic_regression": clean_regression,
                },
                "accepted": service == proposed_service and service != old_service,
            },
            "provider_budget": _budget_delta(
                budget_before,
                None if request_budget is None else request_budget.snapshot().to_dict(),
            ),
        }
        generation_documents.append(generation_doc)
        summary = {
            "generation": generation,
            "customer_id": customer_strategy_id(customer),
            "service_id": service_strategy_id(service),
            "customer_selection": customer_decision.get("reason", ""),
            "service_selection": service_decision.get("reason", ""),
        }
        history.append(summary)
        if output_root is not None:
            _write_json_atomic(output_root / f"generation-{generation:04d}.json", generation_doc)
        if checkpoint_file is not None:
            _write_json_atomic(
                checkpoint_file,
                {
                    "schema_version": 1,
                    "manifest_sha256": manifest_sha256,
                    "completed_generation": generation,
                    "initial_customer": initial_customer_state.to_dict(),
                    "initial_service": initial_service_state.to_dict(),
                    "customer": customer.to_dict(),
                    "service": service.to_dict(),
                    "generations": generation_documents,
                    "history": history,
                },
            )

    return AlternatingResult(
        initial_customer_state,
        initial_service_state,
        customer,
        service,
        tuple(generation_documents),
    )


def propose_fresh_customer_challenge(
    result: AlternatingResult,
    customer_evolver: CustomerEvolver,
    *,
    tasks: Mapping[str, Any],
    evolution_task_ids: Sequence[str],
    runner: EpisodeRunner,
    domain_policy: str,
    seed: int,
    request_budget: RequestBudget | None = None,
) -> PromptStrategy:
    """Observe the final pair on E and propose a fresh challenge before H is loaded."""

    observed_runs = _run_panel(
        runner,
        task_ids=evolution_task_ids,
        tasks=tasks,
        seed=seed,
        customer=result.customer,
        service=result.service,
        panel_name="fresh-challenge-generation",
    )
    context = {
        "generation": len(result.generations),
        "task_interactions": _context_episodes(observed_runs, runner, tasks),
        "purpose": "Create one fresh adaptive Customer strategy for final held-out evaluation.",
        "current_customer_strategy": result.customer.text,
        "current_service_strategy": result.service.text,
        "service_policy": domain_policy,
        "history": [dict(item) for item in result.generations],
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
    """Compare S0 and ST on native τ-bench Customers and one fresh adaptive Customer."""

    task_ids = tuple(str(item) for item in heldout_task_ids)
    if not task_ids or set(task_ids) - set(heldout_tasks):
        raise ValueError("final endpoint evaluation requires loaded H tasks")
    rows = []
    for customer_label, customer in (
        ("native_customer", None),
        ("fresh_adaptive_customer", fresh_customer),
    ):
        for service_label, service in (
            ("S0", initial_service),
            ("ST", final_service),
        ):
            episodes = _run_panel(
                runner,
                task_ids=task_ids,
                tasks=heldout_tasks,
                seed=seed,
                customer=customer,
                service=service,
                panel_name=f"heldout-{customer_label}-{service_label}",
            )
            rows.append({
                "customer_condition": customer_label,
                "service_endpoint": service_label,
                "episodes": [_episode_ref(item) for item in episodes],
            })
    result = {
        "schema_version": 1,
        "heldout_task_ids": list(task_ids),
        "seed": seed,
        "initial_service": _strategy_document("service", initial_service),
        "final_service": _strategy_document("service", final_service),
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
            "reviewer_feedback": dict(episode.raw_review),
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


def _has_clean_regression(
    reference: Sequence[EpisodeRecord], candidate: Sequence[EpisodeRecord],
) -> bool:
    by_task = {item.task_id: item for item in candidate}
    for baseline in reference:
        updated = by_task.get(baseline.task_id)
        if updated is None:
            return True
        if _clean_success(baseline) and not _clean_success(updated):
            return True
    return False


def _clean_success(episode: EpisodeRecord) -> bool:
    if episode.task_success is not True:
        return False
    review = episode.raw_review.get("native_review", {})
    return not isinstance(review, Mapping) or review.get("agent_error") is not True


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
        "reviewer_feedback": dict(episode.raw_review),
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
