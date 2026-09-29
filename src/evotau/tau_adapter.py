"""Thin, optional adapters around the pinned tau-bench text runtime."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, distribution
from typing import Any
from uuid import uuid4

from .budget import RequestBudget
from .manifest import (
    TAU2_PACKAGE_VERSION,
    TAU_BENCH_COMMIT,
    role_model_args_for_runtime,
)
from .prompts import append_strategy_block
from .strategies import (
    CustomerStrategy,
    ServiceStrategy,
    render_customer_strategy,
    render_service_strategy,
)


class TauBenchPinError(RuntimeError):
    pass


def verify_tau2_installation() -> None:
    """Require installation metadata that proves the exact source commit."""

    try:
        installed = distribution("tau2")
    except PackageNotFoundError as exc:
        raise TauBenchPinError(
            'tau-bench is not installed; use the project optional extra "tau-bench"'
        ) from exc
    if installed.version != TAU2_PACKAGE_VERSION:
        raise TauBenchPinError(
            f"tau2 package version is {installed.version}; expected {TAU2_PACKAGE_VERSION}"
        )
    direct_url = installed.read_text("direct_url.json")
    if not direct_url:
        raise TauBenchPinError("tau2 direct_url.json is missing; cannot verify its Git commit")
    try:
        source = json.loads(direct_url)
        actual_commit = source["vcs_info"]["commit_id"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise TauBenchPinError("tau2 was not installed from a verifiable Git revision") from exc
    if actual_commit != TAU_BENCH_COMMIT:
        raise TauBenchPinError(
            f"tau2 Git commit is {actual_commit}; expected {TAU_BENCH_COMMIT}"
        )


def customer_user_class(
    base_user_class: type,
    strategy: CustomerStrategy | None,
) -> type:
    """Bind an immutable customer strategy to a UserSimulator subclass."""

    block = render_customer_strategy(strategy)

    class EvoTauCustomerUser(base_user_class):
        @property
        def system_prompt(self) -> str:
            base_prompt = super().system_prompt
            return append_strategy_block(base_prompt, block)

    EvoTauCustomerUser.__name__ = "EvoTauCustomerUser"
    return EvoTauCustomerUser


def service_agent_class(
    base_agent_class: type,
    strategy: ServiceStrategy | None,
    *,
    token_counter: Callable[[str], int] | None = None,
) -> type:
    """Bind structured policy-grounded rules to an LLMAgent subclass."""

    block = render_service_strategy(strategy, token_counter=token_counter)

    class EvoTauServiceAgent(base_agent_class):
        @property
        def system_prompt(self) -> str:
            base_prompt = super().system_prompt
            return append_strategy_block(base_prompt, block)

    EvoTauServiceAgent.__name__ = "EvoTauServiceAgent"
    return EvoTauServiceAgent


def build_phase0_orchestrator(
    *,
    task: Any,
    agent_model: str,
    customer_model: str,
    agent_model_args: dict[str, Any] | None = None,
    customer_model_args: dict[str, Any] | None = None,
    seed: int,
    max_steps: int = 64,
    customer_strategy: CustomerStrategy | None = None,
    service_strategy: ServiceStrategy | None = None,
    service_token_counter: Callable[[str], int] | None = None,
    enforce_communication_protocol: bool = False,
) -> Any:
    """Construct native Retail text components directly, without changing the runner."""

    if type(enforce_communication_protocol) is not bool:
        raise TypeError("enforce_communication_protocol must be boolean")
    verify_tau2_installation()
    from tau2.agent.llm_agent import LLMAgent
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.runner.build import build_environment
    from tau2.user.user_simulator import UserSimulator

    environment = build_environment("retail")
    agent_type = service_agent_class(
        LLMAgent, service_strategy, token_counter=service_token_counter
    )
    customer_type = customer_user_class(UserSimulator, customer_strategy)
    agent = agent_type(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        llm=agent_model,
        llm_args={**(agent_model_args or {}), "num_retries": 0},
    )
    # Match tau-bench's own build_user behavior: some domains (including the
    # pinned Retail environment) do not expose user tools, and the upstream
    # builder treats that as an empty tool set for the simulator.
    try:
        user_tools = environment.get_user_tools(include=task.user_tools) or None
    except ValueError:
        if getattr(environment, "user_tools", None) is not None:
            raise
        user_tools = None
    customer = customer_type(
        tools=user_tools,
        instructions=str(task.user_scenario),
        llm=customer_model,
        llm_args={**(customer_model_args or {}), "num_retries": 0},
    )
    return Orchestrator(
        domain="retail",
        agent=agent,
        user=customer,
        environment=environment,
        task=task,
        max_steps=max_steps,
        max_errors=10,
        seed=seed,
        solo_mode=False,
        simulation_id=f"evotau-phase0-{uuid4()}",
        validate_communication=enforce_communication_protocol,
    )


@contextmanager
def _freeze_tau_evaluator_settings(
    *,
    evaluator_model: str,
    evaluator_model_args: dict[str, Any],
    reviewer_model: str,
    reviewer_model_args: dict[str, Any],
):
    """Override upstream module defaults so every evaluation request is manifest-bound."""

    from tau2.evaluator import (
        auth_classifier,
        evaluator_nl_assertions,
        review_llm_judge,
    )

    evaluator_attributes = (
        (evaluator_nl_assertions, "DEFAULT_LLM_NL_ASSERTIONS", evaluator_model),
        (evaluator_nl_assertions, "DEFAULT_LLM_NL_ASSERTIONS_ARGS", dict(evaluator_model_args)),
    )
    reviewer_modules = (auth_classifier, review_llm_judge)
    originals: list[tuple[Any, str, Any]] = []
    try:
        for module, name, value in evaluator_attributes:
            originals.append((module, name, getattr(module, name)))
            setattr(module, name, value)
        for module in reviewer_modules:
            original = module.generate

            def frozen_generate(*args: Any, _original=original, **kwargs: Any) -> Any:
                # These two upstream modules import `generate` directly, so changing
                # llm_utils defaults alone would leave their model choice implicit.
                kwargs = dict(kwargs)
                kwargs["model"] = reviewer_model
                kwargs.update(reviewer_model_args)
                return _original(*args, **kwargs)

            originals.append((module, "generate", original))
            module.generate = frozen_generate
        yield
    finally:
        for module, name, original in reversed(originals):
            setattr(module, name, original)


def run_with_budget(
    orchestrator: Any,
    budget: RequestBudget,
    *,
    reviewer_model: str,
    reviewer_model_args: dict[str, Any],
    evaluator_model: str,
    evaluator_model_args: dict[str, Any],
    on_simulation: Callable[[Any], None] | None = None,
    after_review: Callable[[Any, Any], None] | None = None,
) -> Any:
    """Run, score, and review natively under one provider-request budget."""

    verify_tau2_installation()
    from tau2.data_model.simulation import UserInfo
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.evaluator.reviewer import ReviewMode, review_simulation
    from tau2.runner.simulation import run_simulation
    from tau2.utils import llm_utils

    if getattr(llm_utils, "LLM_CACHE_ENABLED", False):
        raise RuntimeError("Phase 0 requires τ-bench/LiteLLM response caching to be disabled")
    with _freeze_tau_evaluator_settings(
        evaluator_model=evaluator_model,
        evaluator_model_args=evaluator_model_args,
        reviewer_model=reviewer_model,
        reviewer_model_args=reviewer_model_args,
    ), budget.instrument_tau_llm_utils(llm_utils):
        result = run_simulation(orchestrator, evaluation_type=EvaluationType.ALL)
        if on_simulation is not None:
            on_simulation(result)
        user = orchestrator.user
        review, auth_classification = review_simulation(
            simulation=result,
            task=orchestrator.task,
            mode=ReviewMode.FULL,
            user_info=UserInfo(
                implementation="llm",
                llm=user.llm,
                llm_args=user.llm_args,
                global_simulation_guidelines=user.global_simulation_guidelines,
                persona_config=user.persona_config,
            ),
            policy=orchestrator.environment.get_policy(),
            review_model=reviewer_model,
        )
        result.review = review
        result.auth_classification = auth_classification
        if after_review is not None:
            after_review(result, orchestrator)
        return result


def run_phase0_episode(
    *,
    manifest: Any,
    task: Any,
    customer_strategy: CustomerStrategy | None = None,
    service_strategy: ServiceStrategy | None = None,
    service_token_counter: Callable[[str], int] | None = None,
    request_budget: RequestBudget | None = None,
    on_orchestrator: Callable[[Any], None] | None = None,
    on_simulation: Callable[[Any], None] | None = None,
) -> tuple[Any, RequestBudget]:
    """Run the sole live Phase 0 episode only when its budget was explicitly enabled."""

    if not manifest.real_provider_enabled:
        raise RuntimeError(
            "real provider use is disabled in the Phase 0 manifest; "
            "freeze the role models and explicitly enable the real request budget first"
        )
    budget = request_budget or RequestBudget(manifest.request_budget_cap)
    snapshot = budget.snapshot()
    if snapshot.cap != manifest.request_budget_cap:
        raise ValueError("request budget cap must match the frozen Phase 0 manifest")
    if any(
        (
            snapshot.attempts,
            snapshot.successes,
            snapshot.failures,
            snapshot.denied,
            snapshot.prompt_tokens,
            snapshot.completion_tokens,
            snapshot.usage_responses,
            snapshot.usage_unavailable,
            snapshot.cache_hits,
        )
    ):
        raise ValueError("Phase 0 must start with an unused request budget")
    models = dict(manifest.role_models)
    model_args = role_model_args_for_runtime(manifest.role_model_args)
    orchestrator = build_phase0_orchestrator(
        task=task,
        agent_model=models["agent"],
        customer_model=models["customer"],
        agent_model_args=model_args["agent"],
        customer_model_args=model_args["customer"],
        seed=manifest.seed,
        max_steps=manifest.max_steps,
        customer_strategy=customer_strategy,
        service_strategy=service_strategy,
        service_token_counter=service_token_counter,
        enforce_communication_protocol=manifest.enforce_communication_protocol,
    )
    if on_orchestrator is not None:
        on_orchestrator(orchestrator)
    result = run_with_budget(
        orchestrator,
        budget,
        reviewer_model=models["reviewer"],
        reviewer_model_args=model_args["reviewer"],
        evaluator_model=models["evaluator"],
        evaluator_model_args=model_args["evaluator"],
        on_simulation=on_simulation,
    )
    return result, budget
