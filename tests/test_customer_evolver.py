from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from evotau.customer_evolver import (
    CustomerEvolutionProposalInput,
    EvolutionFailureSignal,
    LLMCustomerEvolver,
    propose_customer_candidates_with_selector,
)
from evotau.manifest import role_model_args_for_runtime
from evotau.records import (
    EvidenceRef,
    FailureRecord,
    FailureSignature,
    customer_strategy_id,
    service_strategy_id,
)
from evotau.strategies import CustomerStrategy, ServiceStrategy


def verified_failure(task_id: str = "73") -> FailureRecord:
    policy_ref = "retail.policy:explicit_confirmation"
    return FailureRecord(
        failure_id="failure-safe-id",
        episode_id="episode-e",
        task_id=task_id,
        generation=0,
        customer_strategy_id=customer_strategy_id(CustomerStrategy()),
        service_strategy_id=service_strategy_id(ServiceStrategy()),
        signature=FailureSignature(
            "retail", "pre_write", policy_ref, "missing_explicit_confirmation",
        ),
        policy_ref=policy_ref,
        evidence=(EvidenceRef(3, "assistant", "policy sequence was not followed"),),
        verification_ref="review:source",
        reproduction_episode_id="episode-e-confirmed",
        reproduction_verification_ref="review:reproduction",
    )


def test_selector_sees_only_evolution_failure_signatures_and_ranks_legal_mutations() -> None:
    seen = []

    def selector(context):
        seen.append(context)
        serialized = json.dumps(context.to_dict())
        assert "73" not in serialized
        assert "episode-e" not in serialized
        assert "review:source" not in serialized
        assert "trajectory" not in serialized
        assert context.failure_signals == (
            EvolutionFailureSignal(
                "failure-safe-id", "pre_write",
                "retail.policy:explicit_confirmation", "missing_explicit_confirmation",
            ),
        )
        return ("request_order", "challenge")

    proposals = propose_customer_candidates_with_selector(
        CustomerStrategy(), 2, generation=0, seed=101,
        recent_failures=(verified_failure(),), already_seen=(),
        evolution_task_ids=("73",), operator_selector=selector,
    )

    assert tuple(item.operator for item in proposals) == ("request_order", "challenge")
    assert all(item.proposal_context_sha256 and len(item.proposal_context_sha256) == 64
               for item in proposals)
    assert proposals[0].supporting_failure_ids == ("failure-safe-id",)
    assert proposals[1].supporting_failure_ids == ("failure-safe-id",)
    assert set(seen[0].allowed_operators) == {"disclosure", "request_order", "challenge"}


def test_customer_evolver_rejects_validation_feedback_and_invented_operators() -> None:
    with pytest.raises(ValueError, match="frozen E tasks"):
        propose_customer_candidates_with_selector(
            CustomerStrategy(), 2, generation=0, seed=1,
            recent_failures=(verified_failure("93"),), already_seen=(),
            evolution_task_ids=("73",), operator_selector=lambda _context: (
                "request_order", "challenge",
            ),
        )

    with pytest.raises(ValueError, match="rank exactly"):
        propose_customer_candidates_with_selector(
            CustomerStrategy(), 2, generation=0, seed=1,
            recent_failures=(), already_seen=(), evolution_task_ids=("73",),
            operator_selector=lambda _context: ("invented_operator", "challenge"),
        )


def test_llm_customer_evolver_uses_frozen_model_args_and_strict_json(monkeypatch) -> None:
    calls = []

    def add_module(name: str, *, package: bool = False) -> ModuleType:
        module = ModuleType(name)
        if package:
            module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        return module

    for name in ("tau2", "tau2.data_model", "tau2.utils"):
        add_module(name, package=True)
    message_module = add_module("tau2.data_model.message")
    message_module.SystemMessage = lambda **kwargs: SimpleNamespace(**kwargs)
    message_module.UserMessage = lambda **kwargs: SimpleNamespace(**kwargs)
    llm_utils = add_module("tau2.utils.llm_utils")

    def generate(*, model, messages, call_name, **kwargs):
        calls.append((model, call_name, kwargs, messages))
        payload = json.loads(messages[-1].content)
        return SimpleNamespace(content=json.dumps({
            "operators": payload["allowed_operators"][:payload["candidate_count"]],
        }))

    llm_utils.generate = generate
    context = CustomerEvolutionProposalInput(
        generation=1,
        incumbent=CustomerStrategy(),
        allowed_operators=("disclosure", "request_order", "challenge"),
        candidate_count=2,
        proposal_seed=202,
        failure_signals=(),
    )
    evolver_args = role_model_args_for_runtime((
        ("evolver", (
            ("api_base", "https://inferaiapi.com/v1"),
            ("temperature", 0.0),
            ("thinking_mode", "disabled"),
        )),
    ))["evolver"]
    evolver = LLMCustomerEvolver(
        model="provider/evolver-model", model_args=evolver_args,
    )

    assert evolver(context) == ("disclosure", "request_order")
    model, call_name, kwargs, messages = calls[0]
    assert model == "provider/evolver-model"
    assert call_name == "evotau_customer_evolver"
    assert kwargs == {
        "num_retries": 0,
        "temperature": 0.0,
        "api_base": "https://inferaiapi.com/v1",
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    prompt_context = json.loads(messages[-1].content)
    assert prompt_context["failure_signals"] == []
    assert set(prompt_context) == {
        "schema_version", "generation", "incumbent", "allowed_operators",
        "candidate_count", "proposal_seed", "failure_signals",
    }
