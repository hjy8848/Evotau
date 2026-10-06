"""Prompt-first alternating evolution on the native τ-bench runtime."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

from .alternating_manifest import (
    DEFAULT_MAX_PARALLEL_EPISODES,
    MAX_PARALLEL_EPISODES,
)
from .budget import RequestBudget
from .inferai_responses import generate_text as inferai_responses_generate_text
from .records import EpisodeRecord, customer_strategy_id, service_strategy_id
from .strategies import PromptStrategy
from .tau_provenance import sha256_json

CustomerEvolver = Callable[[Mapping[str, Any], int], Sequence[str]]
ServiceEvolver = Callable[[Mapping[str, Any]], Mapping[str, str]]
EpisodeRunner = Callable[..., EpisodeRecord]
_MAX_EVOLVER_RAW_RESPONSE_CHARS = 16_000


@dataclass(frozen=True, slots=True)
class AlternatingResult:
    initial_customer: PromptStrategy
    initial_service: PromptStrategy
    customer: PromptStrategy
    service: PromptStrategy
    generations: tuple[Mapping[str, Any], ...]
    final_evolution_episodes: tuple[EpisodeRecord, ...]


class EpisodeJobTelemetry:
    """Thread-safe counts for bounded episode jobs, including cache reuse."""

    def __init__(self, configured_max_parallel_episodes: int) -> None:
        self.configured_max_parallel_episodes = configured_max_parallel_episodes
        self._lock = Lock()
        self._active = 0
        self._total = 0
        self._cache_hits = 0
        self._peak = 0

    def start(self, *, cache_hit: bool) -> None:
        with self._lock:
            self._total += 1
            self._cache_hits += int(cache_hit)
            self._active += 1
            self._peak = max(self._peak, self._active)

    def finish(self) -> None:
        with self._lock:
            self._active -= 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "episode_jobs_total": self._total,
                "episode_jobs_cache_hits": self._cache_hits,
                "configured_max_parallel_episodes": self.configured_max_parallel_episodes,
                "peak_parallel_episodes": self._peak,
            }


class EvolverJSONError(ValueError):
    """Strict JSON parse failure with bounded provider output for diagnostics."""

    def __init__(self, *, call_name: str, raw_response: str, parse_error: str) -> None:
        super().__init__(f"{call_name} returned invalid JSON: {parse_error}")
        self.call_name = call_name
        self.raw_response = raw_response[:_MAX_EVOLVER_RAW_RESPONSE_CHARS]
        self.raw_response_truncated = len(raw_response) > _MAX_EVOLVER_RAW_RESPONSE_CHARS
        self.parse_error = parse_error


class LLMAlternatingEvolvers:
    """Two language-model calls propose free-form Customer and Service strategies."""

    def __init__(
        self,
        *,
        model: str,
        model_args: Mapping[str, Any],
        request_budget: RequestBudget | None = None,
    ) -> None:
        self.model = model
        self.model_args = dict(model_args)
        self.request_budget = request_budget

    def customer_candidates(
        self, context: Mapping[str, Any], count: int,
    ) -> tuple[str, ...]:
        result = self._provider_json_call(
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
        result = self._provider_json_call(
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

    def _provider_json_call(
        self,
        model: str,
        model_args: Mapping[str, Any],
        system_prompt: str,
        context: Mapping[str, Any],
        *,
        call_name: str,
    ) -> dict[str, Any]:
        if model_args.get("api_protocol") != "responses":
            return self._json_call(
                model,
                model_args,
                system_prompt,
                context,
                call_name=call_name,
            )
        content = inferai_responses_generate_text(
            model=model,
            api_base=str(model_args["api_base"]),
            api_key_env=str(model_args.get("api_key_env", "INFERAI_API_KEY")),
            reasoning_effort=str(model_args["reasoning_effort"]),
            system_prompt=system_prompt,
            user_prompt=json.dumps(context, ensure_ascii=False, sort_keys=True),
            call_name=call_name,
            request_budget=self.request_budget,
        )
        value = _parse_evolver_json(content, call_name=call_name)
        if not isinstance(value, dict):
            raise TypeError("evolver responses must be JSON objects")
        return value

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
        value = _parse_evolver_json(message.content or "", call_name=call_name)
        if not isinstance(value, dict):
            raise TypeError("evolver responses must be JSON objects")
        return value


def _strip_json_markdown_fence(content: str) -> str:
    """Remove only a complete, standard JSON markdown fence; leave other text intact."""
    if not isinstance(content, str):
        return content
    stripped = content.strip()
    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[0].strip().lower() in {"```", "```json"} and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return content


def _parse_evolver_json(raw_content: str, *, call_name: str) -> Any:
    try:
        return json.loads(raw_content)
    except (TypeError, json.JSONDecodeError) as first_error:
        cleaned = _strip_json_markdown_fence(raw_content)
        if cleaned == raw_content:
            raise EvolverJSONError(
                call_name=call_name,
                raw_response=str(raw_content),
                parse_error=str(first_error),
            ) from first_error
        try:
            return json.loads(cleaned)
        except (TypeError, json.JSONDecodeError) as second_error:
            raise EvolverJSONError(
                call_name=call_name,
                raw_response=str(raw_content),
                parse_error=str(second_error),
            ) from second_error


def run_alternating_evolution(
    *,
    tasks: Mapping[str, Any],
    evolution_task_ids: Sequence[str],
    validation_task_ids: Sequence[str],
    seed: int,
    generations: int,
    customer_candidate_count: int,
    clean_panel_size: int,
    max_parallel_episodes: int = DEFAULT_MAX_PARALLEL_EPISODES,
    evolution_fitness_seed: int | None = None,
    run_validation: bool = True,
    episode_job_telemetry: EpisodeJobTelemetry | None = None,
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
    v_tasks = (
        tuple(str(item) for item in validation_task_ids[:clean_panel_size])
        if run_validation else ()
    )
    if not e_tasks or len(set(e_tasks)) != len(e_tasks) or (run_validation and not v_tasks):
        raise ValueError("alternating evolution needs unique E tasks and a V panel when enabled")
    missing = (set(e_tasks) | set(v_tasks)) - set(tasks)
    if missing:
        raise ValueError(f"task loader is missing E/V tasks: {sorted(missing)}")
    if type(run_validation) is not bool:
        raise ValueError("run_validation must be boolean")
    fitness_seed = seed if evolution_fitness_seed is None else evolution_fitness_seed
    if (generations < 1 or customer_candidate_count < 1 or seed < 0
            or type(fitness_seed) is not int or fitness_seed < 0):
        raise ValueError("seed, generation count, and candidate count must be non-negative")
    _validate_max_parallel_episodes(max_parallel_episodes)
    job_telemetry = episode_job_telemetry or EpisodeJobTelemetry(max_parallel_episodes)

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

    stage_order = (
        "customer_incumbent_complete", "customer_proposals_ready",
        "customer_candidates_complete", "customer_selected",
        "service_proposal_ready", "service_E_complete", "validation_complete",
        "generation_complete",
    )
    stage_rank = {name: index for index, name in enumerate(stage_order)}

    for generation in range(start_generation, generations):
        generation_started = perf_counter()
        customer_phase_started = generation_started
        validation_elapsed = 0.0
        evolution_seed = fitness_seed
        telemetry_before = job_telemetry.snapshot()
        customer_before = customer
        service_before = service
        old_service = service
        budget_before = None if request_budget is None else request_budget.snapshot().to_dict()
        generation_prefix = f"generation-{generation:04d}"
        stage_file = None if output_root is None else output_root / f"{generation_prefix}-stage.json"
        stage_state = _read_json_if_exists(stage_file)
        if (stage_state is not None and (
                stage_state.get("manifest_sha256") != manifest_sha256
                or stage_state.get("generation") != generation
                or stage_state.get("customer_before") != customer_before.to_dict()
                or stage_state.get("service_before") != service_before.to_dict()
                or stage_state.get("stage") not in stage_rank)):
            raise ValueError("generation stage checkpoint does not match its frozen input")
        stage_state_box = {"value": stage_state}

        def record_stage(
            stage: str, *, _file: Path | None = stage_file,
            _generation: int = generation,
            _customer: PromptStrategy = customer_before,
            _service: PromptStrategy = service_before,
            _state_box: dict[str, Any] = stage_state_box,
            **values: Any,
        ) -> None:
            if _file is None:
                return
            saved_stage = _state_box["value"]
            current_rank = -1 if saved_stage is None else stage_rank[saved_stage["stage"]]
            if stage_rank[stage] < current_rank:
                return
            payload = {
                "schema_version": 1,
                "manifest_sha256": manifest_sha256,
                "generation": _generation,
                "stage": stage,
                "customer_before": _customer.to_dict(),
                "service_before": _service.to_dict(),
                **values,
            }
            _write_json_atomic(_file, payload)
            _state_box["value"] = payload

        incumbent_runs = _run_panel(
            runner, task_ids=e_tasks, tasks=tasks, seed=evolution_seed,
            customer=customer_before, service=service_before,
            panel_name=f"generation-{generation}-customer-incumbent",
            max_parallel_episodes=max_parallel_episodes,
            telemetry=job_telemetry,
        )
        incumbent_accuracy = _accuracy(incumbent_runs)
        record_stage(
            "customer_incumbent_complete", incumbent_accuracy=incumbent_accuracy,
            incumbent_episodes=[_episode_ref(item) for item in incumbent_runs],
        )
        proposal_context = {
            "generation": generation,
            "task_interactions": _context_episodes(incumbent_runs, runner, tasks),
            "incumbent_accuracy": incumbent_accuracy,
            "service_policy": domain_policy,
            "current_customer_strategy": customer_before.text,
            "current_service_strategy": service_before.text,
            "accuracy_history": history,
        }
        customer_input_sha = sha256_json(proposal_context)
        customer_proposal_path = (
            None if output_root is None else output_root / f"{generation_prefix}-customer-proposals.json"
        )
        if customer_proposal_path is not None and customer_proposal_path.exists():
            customer_proposal_doc = json.loads(customer_proposal_path.read_text(encoding="utf-8"))
            _validate_customer_proposal_document(
                customer_proposal_doc, generation=generation,
                manifest_sha256=manifest_sha256, customer=customer_before,
                service=service_before, input_sha256=customer_input_sha,
            )
        else:
            try:
                candidate_texts = tuple(_provider_call(
                    request_budget, customer_evolver, proposal_context, customer_candidate_count,
                ))[:customer_candidate_count]
            except EvolverJSONError as exc:
                _append_evolver_failure(output_root, {
                    "generation": generation,
                    "stage": "customer_proposals",
                    "call_name": exc.call_name,
                    "raw_response": exc.raw_response,
                    "raw_response_truncated": exc.raw_response_truncated,
                    "parse_error": exc.parse_error,
                })
                raise
            candidates = tuple(PromptStrategy(text) for text in candidate_texts)
            customer_proposal_doc = {
                "schema_version": 1,
                "manifest_sha256": manifest_sha256,
                "generation": generation,
                "stage": "customer_proposals_ready",
                "incumbent_customer": _strategy_document("customer", customer_before),
                "frozen_service": _strategy_document("service", service_before),
                "evolver_input_sha256": customer_input_sha,
                "customer_candidates": [
                    {
                        "index": index,
                        "strategy": candidate.text,
                        "strategy_id": customer_strategy_id(candidate),
                    }
                    for index, candidate in enumerate(candidates)
                ],
            }
            if customer_proposal_path is not None:
                _write_json_once(customer_proposal_path, customer_proposal_doc)
        _validate_customer_proposal_document(
            customer_proposal_doc, generation=generation,
            manifest_sha256=manifest_sha256, customer=customer_before,
            service=service_before, input_sha256=customer_input_sha,
        )
        candidates = tuple(
            PromptStrategy(item["strategy"])
            for item in customer_proposal_doc["customer_candidates"]
        )
        record_stage(
            "customer_proposals_ready",
            customer_proposals_path=(
                None if customer_proposal_path is None else customer_proposal_path.name
            ),
            customer_candidate_ids=[customer_strategy_id(item) for item in candidates],
        )

        candidate_runs = _run_panels(
            runner,
            task_ids=e_tasks,
            tasks=tasks,
            seed=evolution_seed,
            panels=tuple(
                (
                    f"generation-{generation}-customer-candidate-{index}",
                    candidate,
                    service_before,
                )
                for index, candidate in enumerate(candidates)
            ),
            max_parallel_episodes=max_parallel_episodes,
            telemetry=job_telemetry,
        )
        candidate_accuracies = [_accuracy(runs) for runs in candidate_runs]
        record_stage(
            "customer_candidates_complete",
            candidate_accuracies=candidate_accuracies,
            candidate_episodes=[[_episode_ref(item) for item in runs] for runs in candidate_runs],
        )

        selected_customer_source: str | int = "incumbent"
        customer = customer_before
        selected_runs = incumbent_runs
        selected_accuracy = incumbent_accuracy
        for index, (candidate, runs, accuracy) in enumerate(
            zip(candidates, candidate_runs, candidate_accuracies, strict=True),
        ):
            # Strict comparison keeps incumbent on ties and preserves first-minimum order.
            if accuracy < selected_accuracy:
                customer = candidate
                selected_customer_source = index
                selected_runs = runs
                selected_accuracy = accuracy
        record_stage(
            "customer_selected", selected_customer=selected_customer_source,
            selected_customer_id=customer_strategy_id(customer),
            selected_accuracy=selected_accuracy,
            selected_episodes=[_episode_ref(item) for item in selected_runs],
        )
        customer_phase_elapsed = perf_counter() - customer_phase_started
        old_accuracy = selected_accuracy
        service_phase_started = perf_counter()
        service_context = {
            "generation": generation,
            "task_interactions": _service_context_episodes(selected_runs, runner, tasks),
            "selected_customer_accuracy": selected_accuracy,
            "customer_strategy": customer.text,
            "current_service_strategy": service_before.text,
            "service_policy": domain_policy,
            "accuracy_history": history,
        }
        service_input_sha = sha256_json(service_context)
        service_proposal_path = (
            None if output_root is None else output_root / f"{generation_prefix}-service-proposal.json"
        )
        if service_proposal_path is not None and service_proposal_path.exists():
            service_proposal_doc = json.loads(service_proposal_path.read_text(encoding="utf-8"))
            _validate_service_proposal_document(
                service_proposal_doc, generation=generation,
                manifest_sha256=manifest_sha256, customer=customer,
                service=service_before, input_sha256=service_input_sha,
            )
        else:
            try:
                service_proposal = _provider_call(request_budget, service_evolver, service_context)
            except EvolverJSONError as exc:
                _append_evolver_failure(output_root, {
                    "generation": generation,
                    "stage": "service_proposal",
                    "call_name": exc.call_name,
                    "raw_response": exc.raw_response,
                    "raw_response_truncated": exc.raw_response_truncated,
                    "parse_error": exc.parse_error,
                })
                raise
            service_analysis = service_proposal.get("analysis", "")
            proposed_text = service_proposal.get("strategy", service_before.text)
            if not isinstance(service_analysis, str) or not isinstance(proposed_text, str):
                raise TypeError("Service Evolver must return natural-language analysis and strategy")
            proposed_service = PromptStrategy(proposed_text)
            service_proposal_doc = {
                "schema_version": 1,
                "manifest_sha256": manifest_sha256,
                "generation": generation,
                "stage": "service_proposal_ready",
                "frozen_customer": _strategy_document("customer", customer),
                "frozen_service": _strategy_document("service", service_before),
                "evolver_input_sha256": service_input_sha,
                "analysis": service_analysis,
                "strategy": proposed_service.text,
                "strategy_id": service_strategy_id(proposed_service),
            }
            if service_proposal_path is not None:
                _write_json_once(service_proposal_path, service_proposal_doc)
        _validate_service_proposal_document(
            service_proposal_doc, generation=generation,
            manifest_sha256=manifest_sha256, customer=customer,
            service=service_before, input_sha256=service_input_sha,
        )
        service_analysis = service_proposal_doc["analysis"]
        proposed_service = PromptStrategy(service_proposal_doc["strategy"])
        record_stage(
            "service_proposal_ready",
            service_proposal_path=(
                None if service_proposal_path is None else service_proposal_path.name
            ),
            proposed_service_id=service_strategy_id(proposed_service),
        )

        proposed_runs: tuple[EpisodeRecord, ...] = ()
        proposed_accuracy = old_accuracy
        improved_on_e = False
        validation_old_accuracy: float | None = None
        validation_new_accuracy: float | None = None
        validation_old_runs: tuple[EpisodeRecord, ...] = ()
        validation_new_runs: tuple[EpisodeRecord, ...] = ()
        accepted = False
        service_reason = "The Service strategy did not change."
        if proposed_service.text != service_before.text:
            proposed_runs = _run_panel(
                runner, task_ids=e_tasks, tasks=tasks, seed=evolution_seed,
                customer=customer, service=proposed_service,
                panel_name=f"generation-{generation}-service-candidate",
                max_parallel_episodes=max_parallel_episodes,
                telemetry=job_telemetry,
            )
            proposed_accuracy = _accuracy(proposed_runs)
            improved_on_e = proposed_accuracy > old_accuracy
            if not improved_on_e:
                service_reason = (
                    "Rejected by native E accuracy: proposed Service accuracy did not exceed "
                    "the old Service accuracy."
                )
        record_stage(
            "service_E_complete", proposed_accuracy=proposed_accuracy,
            improved_on_E=improved_on_e,
            service_candidate_episodes=[_episode_ref(item) for item in proposed_runs],
        )

        acceptance_mode = (
            "validation_gated" if run_validation else "e_only_mechanism_smoke"
        )
        if proposed_service.text == service_before.text:
            service_reason = "The Service strategy did not change."
        elif improved_on_e and run_validation:
            validation_started = perf_counter()
            validation_old_runs = _run_panel(
                runner, task_ids=v_tasks, tasks=tasks, seed=seed,
                customer=None, service=service_before,
                panel_name=f"generation-{generation}-validation-old-native-customer",
                max_parallel_episodes=max_parallel_episodes,
                telemetry=job_telemetry,
            )
            validation_new_runs = _run_panel(
                runner, task_ids=v_tasks, tasks=tasks, seed=seed,
                customer=None, service=proposed_service,
                panel_name=f"generation-{generation}-validation-proposed-native-customer",
                max_parallel_episodes=max_parallel_episodes,
                telemetry=job_telemetry,
            )
            validation_elapsed = perf_counter() - validation_started
            validation_old_accuracy = _accuracy(validation_old_runs)
            validation_new_accuracy = _accuracy(validation_new_runs)
            if validation_new_accuracy >= validation_old_accuracy:
                service = proposed_service
                accepted = True
                service_reason = "Accepted: E accuracy improved and native V accuracy did not decrease."
            else:
                service_reason = "Rejected: native V accuracy decreased."
        elif improved_on_e:
            service = proposed_service
            accepted = True
            service_reason = (
                "Accepted for mechanism smoke based on strict E-only accuracy improvement; "
                "native V validation was intentionally disabled."
            )
        acceptance_evaluated = True
        record_stage(
            "validation_complete", validation_evaluated=validation_old_accuracy is not None,
            validation_skipped=not run_validation,
            validation_old_accuracy=validation_old_accuracy,
            validation_new_accuracy=validation_new_accuracy,
            validation_old_episodes=[_episode_ref(item) for item in validation_old_runs],
            validation_new_episodes=[_episode_ref(item) for item in validation_new_runs],
            accepted=accepted,
        )

        # For the final fresh challenge, keep evidence for the actual final pair.
        final_evolution_episodes = proposed_runs if accepted else selected_runs
        customer_phase = {
            "frozen_service": _strategy_document("service", old_service),
            "evolver_input_sha256": customer_input_sha,
            "incumbent_accuracy": incumbent_accuracy,
            "candidate_accuracies": candidate_accuracies,
            "selected_accuracy": selected_accuracy,
            "selected_customer": selected_customer_source,
            "incumbent_episodes": [_episode_ref(item) for item in incumbent_runs],
            "candidates": [
                {
                    "strategy": candidate.to_dict(),
                    "strategy_id": customer_strategy_id(candidate),
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
            "evolver_input_sha256": service_input_sha,
            "proposed_strategy": proposed_service.to_dict(),
            "old_accuracy": old_accuracy,
            "proposed_accuracy": proposed_accuracy,
            "improved_on_E": improved_on_e,
            "accepted": accepted,
            "acceptance_mode": acceptance_mode,
            "acceptance_evaluated": acceptance_evaluated,
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
            "validation_evaluated": validation_old_accuracy is not None,
        }
        generation_doc = {
            "schema_version": 2,
            "generation": generation,
            "evolution_fitness_seed": fitness_seed,
            "validation_seed": seed if run_validation else None,
            "customer_before": _strategy_document("customer", customer_before),
            "service_before": _strategy_document("service", service_before),
            "customer_after": _strategy_document("customer", customer),
            "service_after": _strategy_document("service", service),
            "customer_phase": customer_phase,
            "service_phase": service_phase,
            "timing": {
                "customer_phase_wall_clock_seconds": round(customer_phase_elapsed, 6),
                "service_phase_wall_clock_seconds": round(
                    perf_counter() - service_phase_started, 6,
                ),
                "validation_wall_clock_seconds": round(validation_elapsed, 6),
                "generation_wall_clock_seconds": round(perf_counter() - generation_started, 6),
                "episode_jobs_total": (
                    job_telemetry.snapshot()["episode_jobs_total"]
                    - telemetry_before["episode_jobs_total"]
                ),
                "episode_jobs_cache_hits": (
                    job_telemetry.snapshot()["episode_jobs_cache_hits"]
                    - telemetry_before["episode_jobs_cache_hits"]
                ),
                "configured_max_parallel_episodes": max_parallel_episodes,
                "peak_parallel_episodes": job_telemetry.snapshot()["peak_parallel_episodes"],
            },
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
        record_stage("generation_complete", generation_result=generation_doc)

    return AlternatingResult(
        initial_customer_state,
        initial_service_state,
        customer,
        service,
        tuple(generation_documents),
        final_evolution_episodes,
    )


def _read_json_if_exists(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"checkpoint {path.name} must contain a JSON object")
    return value


def _validate_customer_proposal_document(
    value: Mapping[str, Any], *, generation: int, manifest_sha256: str | None,
    customer: PromptStrategy, service: PromptStrategy, input_sha256: str,
) -> None:
    if (value.get("schema_version") != 1
            or value.get("manifest_sha256") != manifest_sha256
            or value.get("generation") != generation
            or value.get("stage") != "customer_proposals_ready"
            or value.get("incumbent_customer") != _strategy_document("customer", customer)
            or value.get("frozen_service") != _strategy_document("service", service)
            or value.get("evolver_input_sha256") != input_sha256):
        raise ValueError("frozen Customer proposal does not match its generation input")
    proposals = value.get("customer_candidates")
    if not isinstance(proposals, list):
        raise TypeError("frozen Customer proposals must be a list")
    for index, proposal in enumerate(proposals):
        if (not isinstance(proposal, Mapping) or proposal.get("index") != index
                or not isinstance(proposal.get("strategy"), str)
                or proposal.get("strategy_id") != customer_strategy_id(
                    PromptStrategy(proposal["strategy"]),
                )):
            raise ValueError("frozen Customer proposal identity is invalid")


def _validate_service_proposal_document(
    value: Mapping[str, Any], *, generation: int, manifest_sha256: str | None,
    customer: PromptStrategy, service: PromptStrategy, input_sha256: str,
) -> None:
    if (value.get("schema_version") != 1
            or value.get("manifest_sha256") != manifest_sha256
            or value.get("generation") != generation
            or value.get("stage") != "service_proposal_ready"
            or value.get("frozen_customer") != _strategy_document("customer", customer)
            or value.get("frozen_service") != _strategy_document("service", service)
            or value.get("evolver_input_sha256") != input_sha256
            or not isinstance(value.get("analysis"), str)
            or not isinstance(value.get("strategy"), str)):
        raise ValueError("frozen Service proposal does not match its generation input")
    proposed = PromptStrategy(value["strategy"])
    if value.get("strategy_id") != service_strategy_id(proposed):
        raise ValueError("frozen Service proposal strategy identity is invalid")


def _append_evolver_failure(output_root: Path | None, value: Mapping[str, Any]) -> None:
    if output_root is None:
        return
    path = output_root / "evolver-failures.json"
    previous = _read_json_if_exists(path)
    failures = [] if previous is None else previous.get("failures")
    if not isinstance(failures, list):
        raise TypeError("evolver failure artifact is malformed")
    failures.append(dict(value))
    _write_json_atomic(path, {"schema_version": 1, "failures": failures})


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    """Durably publish one immutable JSON proposal without exposing partial contents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp_name, path)
        except FileExistsError as exc:
            raise FileExistsError(f"immutable proposal already exists: {path}") from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def propose_fresh_customer_challenge(
    result: AlternatingResult,
    customer_evolver: CustomerEvolver,
    *,
    tasks: Mapping[str, Any],
    runner: EpisodeRunner,
    domain_policy: str,
    request_budget: RequestBudget | None = None,
    proposal_path: str | Path | None = None,
    manifest_sha256: str | None = None,
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
    input_sha256 = sha256_json(context)
    path = None if proposal_path is None else Path(proposal_path)
    if path is not None and path.exists():
        proposal = json.loads(path.read_text(encoding="utf-8"))
        if (proposal.get("schema_version") != 1
                or proposal.get("stage") != "fresh_customer_proposal_ready"
                or proposal.get("manifest_sha256") != manifest_sha256
                or proposal.get("evolver_input_sha256") != input_sha256):
            raise ValueError("frozen fresh Customer proposal does not match its final E input")
        text = proposal.get("strategy")
        if not isinstance(text, str):
            raise ValueError("frozen fresh Customer proposal has no strategy text")
        strategy = PromptStrategy(text)
        if proposal.get("strategy_id") != customer_strategy_id(strategy):
            raise ValueError("frozen fresh Customer proposal identity is invalid")
        return strategy
    try:
        candidates = tuple(_provider_call(request_budget, customer_evolver, context, 1))
    except EvolverJSONError as exc:
        _append_evolver_failure(None if path is None else path.parent, {
            "stage": "fresh_customer_proposal",
            "call_name": exc.call_name,
            "raw_response": exc.raw_response,
            "raw_response_truncated": exc.raw_response_truncated,
            "parse_error": exc.parse_error,
        })
        raise
    if not candidates:
        selected = result.customer
    else:
        selected = PromptStrategy(candidates[0])
    if path is not None:
        _write_json_once(path, {
            "schema_version": 1,
            "manifest_sha256": manifest_sha256,
            "stage": "fresh_customer_proposal_ready",
            "evolver_input_sha256": input_sha256,
            "strategy": selected.text,
            "strategy_id": customer_strategy_id(selected),
        })
    return selected


def run_final_endpoint_evaluation(
    *,
    heldout_tasks: Mapping[str, Any],
    heldout_task_ids: Sequence[str],
    seed: int,
    initial_service: PromptStrategy,
    final_service: PromptStrategy,
    fresh_customer: PromptStrategy,
    runner: EpisodeRunner,
    max_parallel_episodes: int = DEFAULT_MAX_PARALLEL_EPISODES,
    telemetry: EpisodeJobTelemetry | None = None,
    output_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Compare S0 and ST endpoints without rerunning identical Service cells."""

    task_ids = tuple(str(item) for item in heldout_task_ids)
    _validate_max_parallel_episodes(max_parallel_episodes)
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
            max_parallel_episodes=max_parallel_episodes,
            telemetry=telemetry,
        )
        s0_episodes_by_customer[customer_label] = episodes
        rows.append({
            "customer_condition": customer_label,
            "service_endpoint": "S0",
            "identical_to_S0": False,
            "accuracy": _accuracy(episodes),
            "episodes": [_episode_ref(item) for item in episodes],
        })
        if identical_services:
            rows.append({
                "customer_condition": customer_label,
                "service_endpoint": "ST",
                "identical_to_S0": True,
                "accuracy": _accuracy(episodes),
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
                max_parallel_episodes=max_parallel_episodes,
                telemetry=telemetry,
            )
            rows.append({
                "customer_condition": customer_label,
                "service_endpoint": "ST",
                "identical_to_S0": False,
                "accuracy": _accuracy(updated_episodes),
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
    max_parallel_episodes: int = DEFAULT_MAX_PARALLEL_EPISODES,
    telemetry: EpisodeJobTelemetry | None = None,
) -> tuple[EpisodeRecord, ...]:
    result = _run_panels(
        runner,
        task_ids=task_ids,
        tasks=tasks,
        seed=seed,
        panels=((panel_name, customer, service),),
        max_parallel_episodes=max_parallel_episodes,
        telemetry=telemetry,
    )
    return result[0]


def _run_panels(
    runner: EpisodeRunner,
    *,
    task_ids: Sequence[str],
    tasks: Mapping[str, Any],
    seed: int,
    panels: Sequence[tuple[str, PromptStrategy | None, PromptStrategy]],
    max_parallel_episodes: int = DEFAULT_MAX_PARALLEL_EPISODES,
    telemetry: EpisodeJobTelemetry | None = None,
) -> tuple[tuple[EpisodeRecord, ...], ...]:
    _validate_max_parallel_episodes(max_parallel_episodes)
    ordered_task_ids = tuple(str(task_id) for task_id in task_ids)
    if not ordered_task_ids or len(set(ordered_task_ids)) != len(ordered_task_ids):
        raise ValueError("episode panel task IDs must be non-empty and unique")
    if not panels:
        return ()
    job_telemetry = telemetry or EpisodeJobTelemetry(max_parallel_episodes)

    def run_one(panel_index: int, task_id: str) -> EpisodeRecord:
        panel_name, customer, service = panels[panel_index]
        if task_id not in tasks:
            raise ValueError(f"task {task_id!r} was not loaded for {panel_name}")
        cache_checker = getattr(runner, "has_completed_episode", None)
        cache_hit = bool(cache_checker(
            task_id=task_id,
            seed=seed,
            customer=customer,
            service=service,
            panel_name=panel_name,
        )) if callable(cache_checker) else False
        job_telemetry.start(cache_hit=cache_hit)
        try:
            episode = runner(
                task_id=task_id,
                seed=seed,
                customer=customer,
                service=service,
                panel_name=panel_name,
            )
            if not isinstance(episode, EpisodeRecord):
                raise TypeError("τ-bench runner must return an EpisodeRecord")
            if episode.task_id != task_id or episode.seed != seed:
                raise ValueError("τ-bench runner returned a different task or seed")
            return episode
        finally:
            job_telemetry.finish()

    completed: list[list[EpisodeRecord | None]] = [
        [None] * len(ordered_task_ids) for _ in panels
    ]
    # Round-robin candidate jobs so the bounded queue admits all candidates early.
    jobs = [
        (panel_index, task_index, task_id)
        for task_index, task_id in enumerate(ordered_task_ids)
        for panel_index in range(len(panels))
    ]
    if len(jobs) <= 1 or max_parallel_episodes == 1:
        for panel_index, task_index, task_id in jobs:
            completed[panel_index][task_index] = run_one(panel_index, task_id)
    else:
        executor = ThreadPoolExecutor(
            max_workers=min(max_parallel_episodes, len(jobs)),
            thread_name_prefix="evotau-episode",
        )
        futures = {
            executor.submit(run_one, panel_index, task_id): (
                panel_index, task_index,
            )
            for panel_index, task_index, task_id in jobs
        }
        try:
            for future in as_completed(futures):
                panel_index, task_index = futures[future]
                completed[panel_index][task_index] = future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    if any(episode is None for panel in completed for episode in panel):
        raise RuntimeError("bounded episode queue did not return every scheduled job")
    return tuple(
        tuple(episode for episode in panel if episode is not None) for panel in completed
    )


def _validate_max_parallel_episodes(value: int) -> None:
    if type(value) is not int or not 1 <= value <= MAX_PARALLEL_EPISODES:
        raise ValueError(
            f"max_parallel_episodes must be an integer from 1 to {MAX_PARALLEL_EPISODES}"
        )


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
