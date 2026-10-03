from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest

from evotau.archive import FailureArchive
from evotau.customer_skill_v3 import (
    MAX_TESTED_SKILL_MEMORY,
    CustomerSkillProposalInput,
    CustomerSkillProviderResponse,
    LLMCustomerSkillEvolver,
    extract_forbidden_literals,
    propose_customer_skills,
    render_customer_skill_v3,
    sanitized_reflection_context,
    validate_reflection_seed_signals,
    validate_skill,
)
from evotau.lifecycle import TwoGenerationSmoke
from evotau.manifest import sha256_json
from evotau.prompts import append_strategy_block
from evotau.records import (
    CandidateEvaluation,
    EpisodeRecord,
    EpisodeStatus,
    customer_strategy_id,
    service_strategy_id,
)
from evotau.strategies import (
    CustomerStrategy,
    ServiceStrategy,
    render_customer_strategy,
)
from evotau.tau_adapter import customer_user_class


def _skill(name: str = "Clarify in sequence") -> dict:
    return {
        "name": name,
        "trigger": "When the agent asks about several independent requests at once.",
        "procedure": [
            "Answer the current question with only the relevant known facts.",
            "After that issue is resolved, restate the next original request without adding facts.",
        ],
        "stop_conditions": "Stop when the agent has a clear next action or needs no further information.",
        "hypothesis": "Exploration: this sequence will improve adherence to independent requests.",
        "evidence_refs": [],
    }


def _response() -> dict:
    first = _skill("Clarify in sequence")
    second = _skill("Confirm scope at closure")
    second["trigger"] = "When the agent summarizes a proposed multi-part resolution."
    second["procedure"] = [
        "Compare the summary with the original goal and any fixed constraints.",
        "Correct only a stated mismatch, then ask for a truthful completion status.",
    ]
    second["stop_conditions"] = "Stop after the scope is confirmed or the agent explains a limitation."
    second["hypothesis"] = "Exploration: scope checking will reveal whether the agent preserves the stated goal."
    return {"candidates": [
        {"operation": "create", "skill": first},
        {"operation": "create", "skill": second},
    ]}


def _propose(provider=None, *, forbidden_literals=()):
    return propose_customer_skills(
        CustomerStrategy.v2_baseline(), 2, generation=0, seed=1,
        recent_failures=(), already_seen=(), already_tested_skills=(),
        reflection_feedback={"schema_version": 1, "aggregate": {"episode_count": 0}},
        forbidden_literals=forbidden_literals,
        proposal_provider=provider or (lambda _context: _response()),
    )


def test_tested_skill_memory_is_bounded_and_contains_only_compact_outcomes() -> None:
    memory = []
    for index in range(MAX_TESTED_SKILL_MEMORY + 1):
        historical = validate_skill(
            {**_skill(f"Historical procedure {index}"),
             "trigger": f"When a reusable request pattern {index} appears."},
            incumbent=CustomerStrategy.v2_baseline(), operation="create", verified_failure_ids=(),
        )
        memory.append({
            "skill_id": historical.strategy_id,
            "skill": historical.to_dict()["skill"],
            "parent_id": customer_strategy_id(CustomerStrategy.v2_baseline()),
            "generation": index,
            "applicability_count": 2,
            "adherence_count": 1,
            "customer_valid_count": 2,
            "provisional_failure_count": 0,
            "verified_failure_refs": [],
            "invalidity_categories": {"unsupported_factual_detail": 1},
        })
    captured = []

    def proposal_provider(context):
        captured.append(context.to_dict())
        return _response()

    propose_customer_skills(
        CustomerStrategy.v2_baseline(), 2, generation=0, seed=1,
        recent_failures=(), already_seen=(), already_tested_skills=(),
        tested_skill_memory=tuple(memory),
        reflection_feedback={"schema_version": 1, "aggregate": {"episode_count": 0}},
        forbidden_literals=(), proposal_provider=proposal_provider,
    )
    assert len(captured[0]["tested_skill_memory"]) == MAX_TESTED_SKILL_MEMORY
    assert "transcript" not in json.dumps(captured[0])


def test_llm_v3_provider_uses_frozen_model_args_and_hashes_raw_json(monkeypatch) -> None:
    calls = []

    def add_module(name: str, *, package: bool = False) -> ModuleType:
        module = ModuleType(name)
        if package:
            module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        return module

    for name in ("tau2", "tau2.data_model", "tau2.utils"):
        add_module(name, package=True)
    messages = add_module("tau2.data_model.message")
    messages.SystemMessage = lambda **kwargs: SimpleNamespace(**kwargs)
    messages.UserMessage = lambda **kwargs: SimpleNamespace(**kwargs)
    utils = add_module("tau2.utils.llm_utils")
    content = json.dumps(_response())

    def generate(*, model, messages, call_name, **kwargs):
        calls.append((model, messages, call_name, kwargs))
        return SimpleNamespace(content=content)

    utils.generate = generate
    context = CustomerSkillProposalInput(
        generation=0, incumbent=CustomerStrategy.v2_baseline(), candidate_count=2,
        proposal_seed=1, tested_skill_fingerprints=(), tested_skills=(),
        tested_skill_memory=(), verified_failures=(),
        reflection_feedback={"schema_version": 1, "aggregate": {"episode_count": 0}},
    )
    evolver = LLMCustomerSkillEvolver(
        model="openai/deepseek-v4-flash",
        model_args={"temperature": 0.0, "thinking_mode": "disabled"},
    )
    response = evolver(context)
    assert isinstance(response, CustomerSkillProviderResponse)
    assert response.response_sha256 == hashlib.sha256(content.encode()).hexdigest()
    assert calls[0][2] == "evotau_customer_skill_evolver_v3"
    assert calls[0][3]["num_retries"] == 0
    assert "max_tokens" not in calls[0][3]
    assert calls[0][3]["thinking_mode"] == "disabled"
    assert "response_schema" in json.loads(calls[0][1][1].content)


def test_skill_v3_schema_canonical_id_and_deterministic_renderer() -> None:
    skill = validate_skill(
        _skill(), incumbent=CustomerStrategy.v2_baseline(), operation="create",
        verified_failure_ids=(),
    )
    normalized = validate_skill(
        {**_skill(), "name": "CLARIFY IN SEQUENCE",
         "procedure": ["answer the current question with only the relevant known facts.",
                       _skill()["procedure"][1]]},
        incumbent=CustomerStrategy.v2_baseline(), operation="create", verified_failure_ids=(),
    )
    assert skill.strategy_id == normalized.strategy_id
    assert len(skill.strategy_id) == 64
    rendered = render_customer_skill_v3(skill)
    assert rendered == render_customer_skill_v3(skill)
    assert rendered.startswith("<evotau_customer_skill>\nFixed EvoTau Customer invariants")
    assert rendered.endswith("</evotau_customer_skill>")
    assert "evidence_refs" not in rendered
    assert "Exploration:" not in rendered


@pytest.mark.parametrize("mutate, message", [
    (lambda row: row.update(extra="no"), "exactly the six"),
    (lambda row: row.update(procedure=["only one step"]), "2..5"),
    (lambda row: row.update(evidence_refs=["unverified"]), "only supplied verified"),
    (lambda row: row.update(hypothesis="Known Service failure"), "begin with"),
    (lambda row: row.update(procedure=["Task 22 should be handled first", *row["procedure"][1:]]),
     "task or scenario identifier"),
    (lambda row: row.update(procedure=["Tell the agent to lie about completion", *row["procedure"][1:]]),
     "fixed truthfulness"),
    (lambda row: row.update(stop_conditions="Guarantee a failure and then stop."), "guarantee a failure"),
    (lambda row: row.update(trigger="Never answer a necessary fact."), "withhold"),
])
def test_skill_v3_fails_closed_on_invalid_or_unsafe_text(mutate, message) -> None:
    row = _skill()
    mutate(row)
    with pytest.raises(ValueError, match=message):
        validate_skill(
            row, incumbent=CustomerStrategy.v2_baseline(), operation="create",
            verified_failure_ids=(),
        )


def test_task_literals_and_benchmark_identifiers_are_rejected_locally() -> None:
    with pytest.raises(ValueError, match="frozen benchmark scenario"):
        validate_skill(
            {**_skill(), "procedure": ["Mention order 22 only if asked.", _skill()["procedure"][1]]},
            incumbent=CustomerStrategy.v2_baseline(), operation="create",
            verified_failure_ids=(), forbidden_literals=("22",),
        )
    with pytest.raises(ValueError, match="benchmark identifier"):
        validate_skill(
            {**_skill(), "trigger": "When the tau-bench task asks for several items."},
            incumbent=CustomerStrategy.v2_baseline(), operation="create", verified_failure_ids=(),
        )
    tasks = {"22": type("Task", (), {"user_scenario": {"instructions": "Email me at a@b.com, order W1234567."}})()}
    found = extract_forbidden_literals(tasks)
    assert "22" in found
    assert "a@b.com" in found
    assert "W1234567" in found


def test_skill_v3_proposal_context_is_sanitized_and_lineage_is_hashed() -> None:
    packet = {"schema_version": 1, "signals": [{
        "category": "unsupported_factual_detail", "adjudication": "disputed",
        "fitness_eligible": False, "source_fingerprint": "a" * 64,
    }]}
    seen = []
    proposals = propose_customer_skills(
        CustomerStrategy.v2_baseline(), 2, generation=0, seed=7,
        recent_failures=(), already_seen=(), already_tested_skills=(),
        reflection_feedback=sanitized_reflection_context(
            CandidateEvaluation(customer_strategy_id(CustomerStrategy.v2_baseline()), ()),
            prior_signals=packet,
        ), forbidden_literals=(), proposal_provider=lambda context: seen.append(context.to_dict()) or _response(),
    )
    text = json.dumps(seen[0], ensure_ascii=False)
    for forbidden in ("task_id", "user_scenario", "transcript", "reference_answer", "gold_actions", "Task 22", "a@b.com"):
        assert forbidden not in text
    assert "unsupported_factual_detail" in text
    assert '"fitness_eligible": false' in text
    assert len(proposals) == 2
    assert all(item.parent_id == customer_strategy_id(CustomerStrategy.v2_baseline()) for item in proposals)
    assert all(len(item.proposal_context_sha256) == len(item.proposal_response_sha256) == 64 for item in proposals)
    assert all(item.rendered_skill_sha256 == hashlib.sha256(
        render_customer_skill_v3(item.strategy).encode("utf-8")
    ).hexdigest() for item in proposals)


def test_skill_v3_rejects_duplicate_candidate_and_incumbent_repeats() -> None:
    duplicate = {"candidates": [
        {"operation": "create", "skill": _skill("One")},
        {"operation": "create", "skill": _skill("One revised")},
    ]}
    with pytest.raises(ValueError, match="duplicate or nearly identical"):
        _propose(lambda _context: duplicate)
    incumbent_skill = validate_skill(
        _skill(), incumbent=CustomerStrategy.v2_baseline(), operation="create", verified_failure_ids=(),
    )
    with pytest.raises(ValueError, match="duplicates or nearly matches"):
        validate_skill(
            incumbent_skill.to_dict()["skill"], incumbent=incumbent_skill,
            operation="refine", verified_failure_ids=(),
        )


def test_reflection_seed_is_exact_schema_and_never_fitness_eligible() -> None:
    signal = {"schema_version": 1, "signals": [{
        "category": "unsupported_factual_detail", "adjudication": "disputed",
        "fitness_eligible": False, "source_fingerprint": "b" * 64,
    }]}
    assert validate_reflection_seed_signals(signal) == signal
    with pytest.raises(ValueError, match="never be fitness"):
        validate_reflection_seed_signals({
            "schema_version": 1,
            "signals": [{**signal["signals"][0], "fitness_eligible": True}],
        })
    with pytest.raises(ValueError, match="JSON object"):
        validate_reflection_seed_signals({**signal, "task_id": "22"})


def test_native_user_prompt_and_task_scenario_are_preserved() -> None:
    skill = validate_skill(
        _skill(), incumbent=CustomerStrategy.v2_baseline(), operation="create", verified_failure_ids=(),
    )

    class NativeUser:
        def __init__(self) -> None:
            self.instructions = "frozen official task scenario"

        @property
        def system_prompt(self) -> str:
            return f"native τ-bench prompt: {self.instructions}"

    Wrapped = customer_user_class(NativeUser, skill)
    user = Wrapped()
    block = render_customer_strategy(skill)
    assert user.instructions == "frozen official task scenario"
    assert user.system_prompt == append_strategy_block("native τ-bench prompt: frozen official task scenario", block)


def test_customer_invalid_episode_cannot_be_fitness_even_when_native_task_failed() -> None:
    skill = validate_skill(
        _skill(), incumbent=CustomerStrategy.v2_baseline(), operation="create", verified_failure_ids=(),
    )
    episode = EpisodeRecord(
        episode_id="invalid-customer-episode", task_id="22", seed=1,
        customer_strategy_id=skill.strategy_id,
        service_strategy_id=service_strategy_id(ServiceStrategy()),
        status=EpisodeStatus.INVALID_CUSTOMER, task_success=False, native_reward=0.0,
        customer_valid=False, customer_invalidity_category="unsupported_factual_detail",
        strategy_applicable=True, customer_strategy_adherent=True,
        policy_violation=True, policy_rule_id="policy-rule", mistake_type="wrong_action",
        workflow_stage="completion", evidence=(),
    )
    evaluation = CandidateEvaluation(skill.strategy_id, (episode,))
    assert evaluation.invalid_episode_count == 1
    assert evaluation.fitness == 0
    assert not episode.has_attributable_failure_candidate


def test_v1_v2_identity_and_strict_zero_fitness_selection_are_unchanged() -> None:
    legacy = CustomerStrategy()
    baseline = CustomerStrategy.v2_baseline()
    assert customer_strategy_id(legacy) == sha256_json(legacy.to_dict())[:16]
    assert customer_strategy_id(baseline) == sha256_json(baseline.to_dict())[:16]
    episode = EpisodeRecord(
        episode_id="same-panel", task_id="E", seed=1,
        customer_strategy_id=customer_strategy_id(legacy),
        service_strategy_id=service_strategy_id(ServiceStrategy()),
        status=EpisodeStatus.COMPLETE, task_success=True, native_reward=1.0,
        customer_valid=True, strategy_applicable=True, customer_strategy_adherent=True,
    )
    incumbent = CandidateEvaluation(customer_strategy_id(legacy), (episode,))
    candidate_episode = replace(episode, customer_strategy_id=customer_strategy_id(baseline))
    challenger = CandidateEvaluation(customer_strategy_id(baseline), (candidate_episode,))
    from evotau.selection import select_customer

    decision = select_customer(incumbent, (challenger,))
    assert not decision.evolved
    assert decision.selected_id == incumbent.strategy_id


def test_skill_v3_archive_preserves_canonical_ids_and_roundtrips(tmp_path) -> None:
    skill = validate_skill(
        _skill(), incumbent=CustomerStrategy.v2_baseline(), operation="create", verified_failure_ids=(),
    )
    archive = FailureArchive(tmp_path / "archive.sqlite")
    assert archive.append_customer_strategy(skill, parent_id=customer_strategy_id(CustomerStrategy.v2_baseline()),
                                            operator="create", generation=0)
    assert archive.customer_strategy_ids() == {skill.strategy_id}
    assert archive.get_customer_strategy(skill.strategy_id) == skill
    assert archive.customer_strategies() == (skill.to_dict(),)


def test_skill_v3_frozen_proposals_are_reused_after_resume(tmp_path) -> None:
    incumbent = CustomerStrategy.v2_baseline()
    service = ServiceStrategy()
    provider_calls = []
    runner_calls = []
    should_interrupt = True

    def provider(context):
        provider_calls.append(context.to_dict())
        return _response()

    def runner(*, task_id, seed, customer, service, panel_name):
        nonlocal should_interrupt
        runner_calls.append(customer_strategy_id(customer))
        if should_interrupt and len(runner_calls) == 2:
            should_interrupt = False
            raise RuntimeError("interrupted after frozen skill proposal")
        return EpisodeRecord(
            episode_id=f"episode-{len(runner_calls)}", task_id=task_id, seed=seed,
            customer_strategy_id=customer_strategy_id(customer),
            service_strategy_id=service_strategy_id(service),
            status=EpisodeStatus.COMPLETE, task_success=True, native_reward=1.0,
            customer_valid=True, strategy_applicable=True, customer_strategy_adherent=True,
            invalid_repeated_write_calls=0,
        )

    manifest = {
        "phase": "3-multitask-service-repair-activation",
        "customer_evolver_schema": "skill_v3",
        "generations": 1, "max_episodes": 100, "max_concurrency": 1,
    }
    checkpoint = tmp_path / "checkpoint.json"

    def controller():
        value = TwoGenerationSmoke(
            manifest=manifest, checkpoint_path=str(checkpoint), runner=runner,
            task_ids=("E", "V"), evolution_task_ids=("E",), seed=1,
            generations=1, max_episodes=100,
        )
        value.episode_attempts = 0
        return value

    with pytest.raises(RuntimeError, match="interrupted after frozen skill proposal"):
        controller().run(
            incumbent, service, customer_proposal_provider=provider,
            customer_proposal_mode="skill_v3", allow_frozen_service=True,
        )
    assert len(provider_calls) == 1
    frozen = json.loads(checkpoint.read_text())["state"]["progress"]["customer_proposals"]
    assert len(frozen) == 2
    commits = controller().run(
        incumbent, service,
        customer_proposal_provider=lambda _context: pytest.fail("V3 proposal regenerated on resume"),
        customer_proposal_mode="skill_v3", allow_frozen_service=True,
    )
    assert len(commits) == 1
    assert len(provider_calls) == 1
    assert len(commits[0].decision_record["customer"]["proposals"]) == 2
