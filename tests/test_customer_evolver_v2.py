from __future__ import annotations

import json
import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest

from evotau.customer_evolver_v2 import (
    CustomerStrategyProposalInput,
    LLMCustomerStrategyEvolver,
    propose_customer_strategies,
)
from evotau.lifecycle import TwoGenerationSmoke, run_customer_round
from evotau.mutation import mutate_customer, propose_customer_candidates
from evotau.records import (
    EpisodeRecord,
    EpisodeStatus,
    EvidenceRef,
    FailureRecord,
    FailureSignature,
    customer_strategy_id,
    service_strategy_id,
)
from evotau.strategies import (
    CustomerStrategy,
    ServiceStrategy,
    render_customer_strategy,
)


def _candidate(incumbent: CustomerStrategy, strategy: CustomerStrategy) -> dict:
    old = incumbent.to_dict()
    new = strategy.to_dict()
    return {
        "strategy": new,
        "changed_fields": [key for key in old if old[key] != new[key]],
        "hypothesis": "Exploration of a different interaction sequence before commitment.",
        "evidence_refs": [],
    }


def _proposals(incumbent: CustomerStrategy) -> dict:
    return {"candidates": [
        _candidate(incumbent, replace(incumbent, request_decomposition="one_by_one")),
        _candidate(incumbent, replace(incumbent, correction_behavior="correct_once")),
    ]}


def test_v2_strategy_has_exact_fields_and_preserves_legacy_strategy_identity() -> None:
    legacy = CustomerStrategy()
    assert legacy.to_dict() == {
        "disclosure": "minimal_on_request", "request_order": "scenario_order",
        "challenge_style": "none", "challenge_budget": 0,
    }
    new = CustomerStrategy.v2_baseline()
    assert set(new.to_dict()) == {
        "disclosure", "request_order", "request_decomposition", "preference_revision",
        "correction_behavior", "challenge_behavior", "challenge_budget",
    }
    assert CustomerStrategy(**new.to_dict()) == new
    assert customer_strategy_id(new) != customer_strategy_id(legacy)
    rendered = render_customer_strategy(replace(new, preference_revision="revise_before_commit"))
    assert "only a preference the scenario explicitly leaves flexible" in rendered
    assert "official user-simulation guidelines" in rendered

    with pytest.raises(ValueError, match="all seven"):
        CustomerStrategy(challenge_behavior="ask_reason", challenge_budget=1)
    with pytest.raises(ValueError, match="must be 0"):
        replace(new, challenge_budget=1)


def test_v2_evolver_proposes_exact_k_without_task_truth_or_duplicate_strategies() -> None:
    incumbent = CustomerStrategy.v2_baseline()
    seen = []

    def provider(context: CustomerStrategyProposalInput) -> dict:
        seen.append(context.to_dict())
        return _proposals(incumbent)

    proposals = propose_customer_strategies(
        incumbent, 2, generation=0, seed=1,
        recent_failures=(), already_seen=(), already_tested_strategies=(),
        evolution_task_ids=("73",), proposal_provider=provider,
    )
    assert len(proposals) == 2
    assert tuple(item.changed_fields for item in proposals) == (
        ("request_decomposition",), ("correction_behavior",),
    )
    assert all(item.operator == "strategy_v2" and item.rationale == "exploration" for item in proposals)
    assert all(item.proposal_context_sha256 == proposals[0].proposal_context_sha256 for item in proposals)
    context_text = json.dumps(seen[0])
    assert '"K": 2' in context_text
    assert '"73"' not in context_text
    assert "task_id" not in context_text
    assert "reference_answer" not in context_text


def test_v2_evolver_rejects_unreviewed_or_invalid_output_before_episode_dispatch() -> None:
    incumbent = CustomerStrategy.v2_baseline()
    tested = replace(incumbent, request_decomposition="one_by_one")

    def call(response: dict, *, tested_strategies=()) -> tuple:
        return propose_customer_strategies(
            incumbent, 2, generation=0, seed=1,
            recent_failures=(), already_seen=(), already_tested_strategies=tested_strategies,
            evolution_task_ids=("73",), proposal_provider=lambda _context: response,
        )

    with pytest.raises(ValueError, match="exactly K"):
        call({"candidates": _proposals(incumbent)["candidates"][:1]})
    with pytest.raises(ValueError, match="tested, or duplicate"):
        call(_proposals(incumbent), tested_strategies=(tested,))
    wrong_fields = _proposals(incumbent)
    wrong_fields["candidates"][0]["changed_fields"] = ["disclosure"]
    with pytest.raises(ValueError, match="changed_fields"):
        call(wrong_fields)
    false_evidence = _proposals(incumbent)
    false_evidence["candidates"][0]["evidence_refs"] = ["invented-failure"]
    with pytest.raises(ValueError, match="supplied verified"):
        call(false_evidence)
    three_axes = _proposals(incumbent)
    three_axes["candidates"][0] = _candidate(
        incumbent, replace(
            incumbent, disclosure="progressive", request_order="dependency_first",
            request_decomposition="one_by_one",
        ),
    )
    with pytest.raises(ValueError, match="three-axis"):
        call(three_axes)

    leaked_task_id = _proposals(incumbent)
    leaked_task_id["candidates"][0]["hypothesis"] = "Exploration of Task ID 73 behavior."
    with pytest.raises(ValueError, match="task ID"):
        call(leaked_task_id)
    guaranteed_failure = _proposals(incumbent)
    guaranteed_failure["candidates"][0]["hypothesis"] = "Exploration will guarantee failure."
    with pytest.raises(ValueError, match="guarantee failure"):
        call(guaranteed_failure)


def test_legacy_mutation_operators_fail_closed_for_v2_strategies() -> None:
    incumbent = CustomerStrategy.v2_baseline()
    with pytest.raises(ValueError, match="legacy mutation operators"):
        mutate_customer(incumbent, "disclosure")
    with pytest.raises(ValueError, match="legacy mutation operators"):
        propose_customer_candidates(incumbent, 1, seed=1)


def test_v2_verified_evidence_is_e_only_and_can_support_a_three_axis_hypothesis() -> None:
    incumbent = CustomerStrategy.v2_baseline()
    failure = FailureRecord(
        failure_id="verified-safe-id",
        episode_id="private-episode-e",
        task_id="73",
        generation=0,
        customer_strategy_id=customer_strategy_id(incumbent),
        service_strategy_id=service_strategy_id(ServiceStrategy()),
        signature=FailureSignature(
            "retail", "pre_write", "retail.policy:explicit_confirmation",
            "missing_explicit_confirmation",
        ),
        policy_ref="retail.policy:explicit_confirmation",
        evidence=(EvidenceRef(2, "assistant", "private trajectory detail"),),
        verification_ref="review:source",
        reproduction_episode_id="private-episode-replay",
        reproduction_verification_ref="review:replay",
    )
    first = _candidate(incumbent, replace(
        incumbent, request_decomposition="one_by_one",
        preference_revision="revise_before_commit",
        correction_behavior="correct_once",
    ))
    first["hypothesis"] = "May expose stale execution state after a lawful preference change and correction."
    first["evidence_refs"] = [failure.failure_id]
    second = _candidate(incumbent, replace(incumbent, disclosure="progressive"))
    contexts = []

    def provider(context):
        contexts.append(context.to_dict())
        return {"candidates": [first, second]}

    result = propose_customer_strategies(
        incumbent, 2, generation=1, seed=11, recent_failures=(failure,),
        already_seen=(), already_tested_strategies=(),
        evolution_task_ids=("73",), proposal_provider=provider,
    )
    assert result[0].supporting_failure_ids == ("verified-safe-id",)
    assert result[0].rationale == "failure_conditioned"
    context_text = json.dumps(contexts[0])
    assert "verified-safe-id" in context_text
    assert "private-episode" not in context_text
    assert "private trajectory detail" not in context_text
    assert '"73"' not in context_text

    with pytest.raises(ValueError, match="frozen E tasks"):
        propose_customer_strategies(
            incumbent, 2, generation=1, seed=11, recent_failures=(failure,),
            already_seen=(), already_tested_strategies=(),
            evolution_task_ids=("93",), proposal_provider=provider,
        )


def test_v2_round_evaluates_model_proposals_on_the_frozen_e_panel() -> None:
    incumbent = CustomerStrategy.v2_baseline()
    service = ServiceStrategy()
    calls = []

    def runner(*, task_id, seed, customer, service, panel_name):
        calls.append((task_id, customer_strategy_id(customer), panel_name))
        return EpisodeRecord(
            episode_id=f"episode-{len(calls)}", task_id=task_id, seed=seed,
            customer_strategy_id=customer_strategy_id(customer),
            service_strategy_id=service_strategy_id(service),
            status=EpisodeStatus.COMPLETE, task_success=True, native_reward=1.0,
            customer_valid=True, strategy_applicable=True, customer_strategy_adherent=True,
            invalid_repeated_write_calls=0,
        )

    round_result = run_customer_round(
        runner, incumbent=incumbent, service=service, task_ids=("E-only",),
        seeds=(1,), generation=0, proposal_seed=1, count=2,
        proposal_provider=lambda _context: _proposals(incumbent),
        proposal_mode="strategy_v2",
    )
    assert len(calls) == 3
    assert {item[0] for item in calls} == {"E-only"}
    assert len(round_result.proposals) == 2
    assert not round_result.selection.evolved


def test_v2_resume_reuses_frozen_proposals_after_interrupted_episode(tmp_path) -> None:
    incumbent = CustomerStrategy.v2_baseline()
    service = ServiceStrategy()
    calls = []
    proposal_calls = []
    interrupt = True

    def runner(*, task_id, seed, customer, service, panel_name):
        nonlocal interrupt
        calls.append(customer_strategy_id(customer))
        if interrupt and len(calls) == 2:
            interrupt = False
            raise RuntimeError("synthetic interruption after proposal checkpoint")
        return EpisodeRecord(
            episode_id=f"episode-{len(calls)}", task_id=task_id, seed=seed,
            customer_strategy_id=customer_strategy_id(customer),
            service_strategy_id=service_strategy_id(service),
            status=EpisodeStatus.COMPLETE, task_success=True, native_reward=1.0,
            customer_valid=True, strategy_applicable=True, customer_strategy_adherent=True,
            invalid_repeated_write_calls=0,
        )

    manifest = {
        "phase": "3-multitask-service-repair-activation",
        "customer_evolver_schema": "strategy_v2",
        "generations": 1, "max_episodes": 10, "max_concurrency": 1,
    }
    checkpoint = tmp_path / "checkpoint.json"

    def controller() -> TwoGenerationSmoke:
        created = TwoGenerationSmoke(
            manifest=manifest, checkpoint_path=str(checkpoint), runner=runner,
            task_ids=("E", "V"), evolution_task_ids=("E",), seed=1,
            generations=1, max_episodes=10,
        )
        created.episode_attempts = 0
        return created

    def provider(_context):
        proposal_calls.append(1)
        return _proposals(incumbent)

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        controller().run(
            incumbent, service, customer_proposal_provider=provider,
            customer_proposal_mode="strategy_v2",
        )
    assert len(proposal_calls) == 1
    assert checkpoint.is_file()
    assert len(json.loads(checkpoint.read_text())["state"]["progress"]["customer_proposals"]) == 2

    commits = controller().run(
        incumbent, service,
        customer_proposal_provider=lambda _context: pytest.fail("proposal was regenerated"),
        customer_proposal_mode="strategy_v2",
    )
    assert len(commits) == 1
    assert len(proposal_calls) == 1


def test_llm_v2_evolver_uses_frozen_args_and_json_only(monkeypatch) -> None:
    def add_module(name: str, *, package: bool = False) -> ModuleType:
        module = ModuleType(name)
        if package:
            module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        return module

    for name in ("tau2", "tau2.data_model", "tau2.utils"):
        add_module(name, package=True)
    messages_module = add_module("tau2.data_model.message")
    messages_module.SystemMessage = lambda **kwargs: SimpleNamespace(**kwargs)
    messages_module.UserMessage = lambda **kwargs: SimpleNamespace(**kwargs)
    llm_utils = add_module("tau2.utils.llm_utils")
    calls = []

    def generate(*, model, messages, call_name, **kwargs):
        calls.append((model, messages, call_name, kwargs))
        incumbent = CustomerStrategy.v2_baseline()
        return SimpleNamespace(content=json.dumps(_proposals(incumbent)))

    llm_utils.generate = generate
    incumbent = CustomerStrategy.v2_baseline()
    context = CustomerStrategyProposalInput(0, incumbent, 2, 1, (), (), ())
    evolver = LLMCustomerStrategyEvolver(
        "provider/evolver", {"temperature": 0.0, "thinking_mode": "disabled"},
    )
    assert len(evolver(context)["candidates"]) == 2
    assert calls[0][2] == "evotau_customer_strategy_evolver_v2"
    assert calls[0][3]["num_retries"] == 0
    assert "You are the Customer Evolver in EvoTau" in calls[0][1][0].content
    assert json.loads(calls[0][1][1].content)["incumbent_strategy"] == incumbent.to_dict()

    llm_utils.generate = lambda **_kwargs: SimpleNamespace(content="```json\n{}\n```")
    with pytest.raises(ValueError, match="invalid JSON"):
        evolver(context)
