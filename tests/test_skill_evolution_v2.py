import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from evotau.evolution_archive import (
    EvolutionArchive,
    crossover_eligible,
    semantic_duplicate,
)
from evotau.evolution_failures import paired_effect, paired_failure_matrix
from evotau.evolution_gate import evaluate_gate
from evotau.records import (
    EpisodeRecord,
    EpisodeStatus,
    customer_strategy_id,
    service_strategy_id,
)
from evotau.service_skills import (
    ServiceSkillMemoryV2,
    apply_v2_mutation,
    render_selected_service_skills,
)
from evotau.skill_activation import SkillActivator, activating_service_agent_class
from evotau.skill_evolution import run_skill_evolution_v2
from evotau.skill_evolution_config import DEFAULT_V2
from evotau.strategies import PromptStrategy


def draft(
    guidance="Execute the confirmed operation promptly.",
    positive="The user adds scope immediately before confirmation.",
):
    return {
        "trigger": "Late scope expansion before a one-shot action.",
        "guidance": guidance,
        "activation_signature": {
            "positive_conditions": [positive],
            "negative_conditions": ["Ordinary unchanged cancellation."],
            "interaction_phase": ["before_confirmation"],
        },
    }


def mutation(
    guidance="Execute the confirmed operation promptly.",
    family="late_scope",
    operation="add",
    target=None,
):
    return {
        "analysis": "Fix count 1; risk count 0.",
        "semantic_family": family,
        "target_cluster_id": "cluster-1",
        "operation": operation,
        "target_skill_id": target,
        "skill": draft(guidance),
        "expected_fixes": ["1"],
        "protected_cases_at_risk": [],
        "substantive_delta_from_prior": "",
    }


def record(task, success, seed=1, **kwargs):
    return EpisodeRecord(
        f"ep-{task}-{seed}",
        str(task),
        seed,
        "c",
        "s",
        EpisodeStatus.COMPLETE if success is not None else EpisodeStatus.UNCERTAIN,
        success,
        total_steps=5,
        **kwargs,
    )


class FakeProviders:
    def __init__(self):
        self.calls = []

    def customers(self, context, count):
        self.calls.append("customers")
        return {
            "candidates": [
                {
                    "strategy": "Ask for a grounded explanation.",
                    "semantic_family": "explanation",
                    "target_weakness_family": "scope",
                    "substantive_delta_from_prior": "",
                }
            ]
            * count
        }

    def validate_customer(self, context):
        self.calls.append("customer_validator")
        return dict.fromkeys(
            (
                "preserves_facts",
                "preserves_objective",
                "interaction_only",
                "no_benchmark_leakage",
            ),
            True,
        )

    def diagnose(self, context):
        self.calls.append("diagnose")
        assert "secret-hidden" not in json.dumps(context)
        return {
            "clusters": [
                {
                    "cluster_id": "cluster-1",
                    "root_cause": "Late scope handling.",
                    "evidence_task_ids": ["1"],
                    "protected_success_task_ids": ["2"],
                    "recommended_surface": "skill",
                    "recommended_mutation_types": ["add"],
                    "risk": "Ordinary cancellation.",
                }
            ]
        }

    def mutate(self, context):
        self.calls.append("mutate")
        bias = context["proposal_bias"]
        if bias == "minimal_behavior":
            return mutation("Break protected passing behavior.", family="broad")
        if bias == "structural_decomposition":
            return mutation(operation="invented")
        return mutation()

    def validate_skill(self, context):
        self.calls.append("skill_validator")
        return {"reusable": True, "policy_subordinate": True, "no_task_entities": True}

    def crossover(self, context):
        self.calls.append("crossover")
        return mutation(family="complementary")


class FakeRunner:
    def __init__(self, fail_once=None):
        self.cache, self.calls = {}, []
        self.fail_once = fail_once
        self.failed = False

    def __call__(self, *, task_id, seed, customer, service, panel_name):
        key = (
            task_id,
            seed,
            customer_strategy_id(customer),
            service_strategy_id(service),
        )
        if key in self.cache:
            return self.cache[key]
        if self.fail_once and self.fail_once in panel_name and not self.failed:
            self.failed = True
            raise RuntimeError("episode interruption")
        self.calls.append(key)
        helpful = bool(service.skills)
        bad = helpful and "Break" in service.skills[0].guidance
        success = task_id != "1" or helpful
        if bad and task_id == "2":
            success = False
        result = EpisodeRecord(
            f"ep-{len(self.cache)}",
            task_id,
            seed,
            key[2],
            key[3],
            EpisodeStatus.COMPLETE,
            success,
            native_reward=float(success),
            termination_reason="user_stop",
            total_steps=5,
            trajectory_ref=f"{len(self.cache)}.json",
        )
        self.cache[key] = result
        return result

    def load_trajectory(self, episode):
        return {
            "messages": [
                {"role": "user", "content": "Please help."},
                {"role": "assistant", "content": "Checking."},
            ],
            "termination_reason": "user_stop",
        }


def run(
    tmp_path,
    *,
    provider=None,
    runner=None,
    interrupt=None,
    generations=1,
    policy=None,
    validation=False,
):
    providers = provider or FakeProviders()
    native = runner or FakeRunner()
    settings = deepcopy(policy or DEFAULT_V2)
    settings["service_evolution"]["crossover"] = False
    tasks = {
        t: SimpleNamespace(
            id=t,
            user_scenario="secret-hidden-" + t,
            description="secret-hidden",
            user_tools=[],
        )
        for t in (("1", "2", "3", "4") if validation else ("1", "2", "3"))
    }
    result = run_skill_evolution_v2(
        tasks=tasks,
        evolution_task_ids=("1", "2", "3"),
        validation_task_ids=("4",),
        seed=1,
        generations=generations,
        customer_candidate_count=1,
        max_parallel_episodes=1,
        evolution_fitness_seed=1,
        run_validation=validation,
        initial_customer=PromptStrategy(""),
        initial_service=ServiceSkillMemoryV2(),
        runner=native,
        providers=providers,
        domain_policy="Do not change native policy.",
        output_directory=tmp_path,
        checkpoint_path=tmp_path / "checkpoint.json",
        manifest_sha256="manifest",
        policy=settings,
        interrupt_after_stage=interrupt,
    )
    return result, providers, native


def test_multi_candidate_rejected_lineage_and_runtime_isolation(tmp_path):
    result, provider, runner = run(tmp_path)
    board = result.generations[0]["service_phase"]["candidates"]
    assert len(board) == 3
    assert board[0]["runtime_deployed"] and len(result.service.skills) == 1
    assert (
        board[1]["effect"]["pass_to_fail"] == ["2"] and not board[1]["runtime_deployed"]
    )
    assert board[2]["pre_rollout_rejection"].startswith("invalid_mutation")
    assert provider.calls.count("mutate") == 3
    assert len(runner.calls) == len(set(runner.calls))
    prompt = render_selected_service_skills(result.service, ["skill-0001"])
    assert (
        "mutation_id" not in prompt
        and "secret-hidden" not in prompt
        and "Break" not in prompt
    )


@pytest.mark.parametrize(
    "stage",
    [
        "service_diagnosis",
        "service_proposals-g0000-cluster-1-0",
        "service_screen-g0000-cluster-1-0-decision",
        "service_full_gate-g0000-cluster-1-0-current-decision",
        "generation_complete",
    ],
)
def test_resume_frozen_stages_and_no_double_episode_dispatch(tmp_path, stage):
    provider, runner = FakeProviders(), FakeRunner()
    with pytest.raises(RuntimeError):
        run(tmp_path, provider=provider, runner=runner, interrupt=stage)
    calls = list(provider.calls)
    result, _, _ = run(tmp_path, provider=provider, runner=runner)
    assert len(result.service.skills) == 1
    assert len(runner.calls) == len(set(runner.calls))
    if stage == "generation_complete":
        assert provider.calls == calls
    assert (
        provider.calls.count("diagnose") == 1 and provider.calls.count("customers") == 1
    )
    run(tmp_path, provider=provider, runner=runner)
    assert provider.calls.count("mutate") == 3


def test_tampered_artifacts_and_checkpoint_fail_closed(tmp_path):
    run(tmp_path)
    p = tmp_path / "evolution-v2/g0000-service_diagnosis.json"
    value = json.loads(p.read_text())
    value["payload"]["clusters"][0]["root_cause"] = "tampered"
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="digest"):
        run(tmp_path)


def test_unknown_outcomes_are_not_failures():
    before, after = [record("1", True)], [record("1", None)]
    assert (
        paired_failure_matrix(before, after, generation=0, candidate_id="a")[0][
            "status"
        ]
        == "UNCERTAIN"
    )
    result = evaluate_gate(before, after, DEFAULT_V2["statistical_gate"])
    assert result["verdict"] == "INCONCLUSIVE"


def test_task_block_gate_accepts_clear_positive_and_corrects_looks():
    before = [record(str(t), t >= 20, s) for t in range(100) for s in (1, 2, 3, 4)]
    after = [record(str(t), True, s) for t in range(100) for s in (1, 2, 3, 4)]
    result = evaluate_gate(before, after, DEFAULT_V2["statistical_gate"], looks=1)
    assert result["verdict"] == "ACCEPTED" and result["task_count"] == 100
    assert result["success_ci"][0] > 0
    corrected = evaluate_gate(before, after, DEFAULT_V2["statistical_gate"], looks=10)
    assert corrected["adjusted_confidence"] > result["adjusted_confidence"]
    tie = evaluate_gate(before, before, DEFAULT_V2["statistical_gate"])
    assert tie["verdict"] == "INCONCLUSIVE"


def test_small_panel_cannot_claim_zero_harm_risk():
    before = [record("1", False), record("2", True)]
    after = [record("1", True), record("2", True)]
    result = evaluate_gate(before, after, DEFAULT_V2["statistical_gate"])
    assert result["verdict"] == "INCONCLUSIVE" and result["harmfulness_ci"][1] > 0.05


def test_structural_edits_and_budgets():
    m = apply_v2_mutation(ServiceSkillMemoryV2(), mutation(), next_skill_id_number=1)
    edit = mutation(operation="narrow_trigger", target="skill-0001")
    edit["skill"]["guidance"] = "changed"
    with pytest.raises(ValueError):
        apply_v2_mutation(m, edit, next_skill_id_number=2)
    split = mutation(operation="split", target="skill-0001")
    split["skill"] = None
    split["children"] = [draft(), draft("Other local mechanism.")]
    children = apply_v2_mutation(m, split, next_skill_id_number=2)
    assert [s.skill_id for s in children.skills] == ["skill-0002", "skill-0003"]
    deletion = mutation(operation="delete", target="skill-0001")
    deletion["skill"] = None
    assert not apply_v2_mutation(m, deletion, next_skill_id_number=2).skills


def test_semantic_dedup_and_real_narrowing():
    prior = {
        "mutation_id": "prior",
        "mutation": mutation(),
        "effect": {"accepted": False},
    }
    para = mutation("Execute confirmed operations promptly.")
    assert semantic_duplicate(para, [prior]) == "prior"
    narrow = mutation(operation="narrow_trigger", target="skill-0001")
    narrow["substantive_delta_from_prior"] = (
        "Only before confirmation; exclude ordinary cancellation."
    )
    assert semantic_duplicate(narrow, [prior]) is None


def test_archive_preserves_local_loser_and_crossover_conflicts():
    a, b = mutation(family="a"), mutation(family="b")
    old = [record("1", False), record("2", True), record("3", False)]
    candidate = [record("1", True), record("2", False), record("3", False)]
    effect = paired_effect(
        old,
        candidate,
        a,
        generation=0,
        mutation_id="a",
        parent_memory_id="old",
        proposed_memory_id="new",
    ).to_dict()
    entry = {"mutation_id": "a", "mutation": a, "effect": effect}
    archive = EvolutionArchive()
    archive.add(entry)
    assert archive.entries and not archive.to_dict()["runtime_deployed"]
    other = {
        "mutation_id": "b",
        "mutation": b,
        "effect": {**effect, "fail_to_pass": ["3"], "pass_to_fail": []},
    }
    assert crossover_eligible(entry, other)
    other["effect"]["pass_to_fail"] = ["1"]
    assert not crossover_eligible(entry, other)


def test_activator_context_and_topk_fail_closed():
    memory = apply_v2_mutation(
        ServiceSkillMemoryV2(), mutation(), next_skill_id_number=1
    )
    seen = []

    def select(prompt, context):
        seen.append(context)
        late = "also include" in (context["messages"][-1].get("content") or "")
        negative = "ordinary unchanged" in context["messages"][-1].get("content", "")
        return {
            "active_skill_ids": ["skill-0001"] if late and not negative else [],
            "reason": "specific condition",
            "confidence": 0.9,
        }

    activator = SkillActivator(model="fake", model_args={}, json_call=select)
    for text, expected in [
        ("Cancel please.", ()),
        ("exchange", ()),
        ("Please also include another item.", ("skill-0001",)),
        ("ordinary unchanged cancellation; also include", ()),
    ]:
        decision, _ = activator.select(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": text,
                        "user_scenario": "hidden",
                        "task_id": "H",
                    }
                ]
            },
            memory.skills,
        )
        assert decision.active_skill_ids == expected
    assert "hidden" not in json.dumps(seen) and "guidance" not in json.dumps(seen)
    with pytest.raises(ValueError):
        activator.select({"messages": [], "task_success": True}, memory.skills)
    invalid = SkillActivator(
        model="fake",
        model_args={},
        json_call=lambda *_: {"active_skill_ids": ["missing"]},
    )
    with pytest.raises(ValueError):
        invalid.select({"messages": []}, memory.skills)
    empty, _ = activator.select({"messages": []}, (), 2)
    assert empty.active_skill_ids == ()


def test_native_agent_turn_prompt_does_not_accumulate_unselected_guidance(monkeypatch):
    import os
    if 'TAU2_DATA_DIR' not in os.environ:
        monkeypatch.setenv('TAU2_DATA_DIR', '/Users/spring/.cache/evotau/tau2-data-b7ea9074')
    from tau2.data_model.message import UserMessage

    memory = apply_v2_mutation(
        ServiceSkillMemoryV2(), mutation(), next_skill_id_number=1
    )
    outputs = iter([{"active_skill_ids": ["skill-0001"]}, {"active_skill_ids": []}])
    activator = SkillActivator(
        model="fake", model_args={}, json_call=lambda *_: next(outputs)
    )
    prompts = []
    sidecars = []

    class Base:
        @property
        def system_prompt(self):
            return "Native policy unchanged."

        def generate_next_message(self, message, state):
            prompts.append(state.system_messages[0].content)
            state.messages.append(message)
            return message, state

    cls = activating_service_agent_class(
        Base, memory, activator=activator, max_active_skills=2, sink=sidecars.append
    )
    state = SimpleNamespace(messages=[], system_messages=[])
    agent = cls()
    agent.generate_next_message(UserMessage(role="user", content="Add scope."), state)
    agent.generate_next_message(
        UserMessage(role="user", content="Different request."), state
    )
    assert (
        memory.skills[0].guidance in prompts[0]
        and memory.skills[0].guidance not in prompts[1]
    )
    assert prompts[1] == "Native policy unchanged." and len(sidecars) == 2
