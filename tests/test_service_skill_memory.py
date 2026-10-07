from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from evotau import alternating
from evotau.alternating import run_alternating_evolution
from evotau.alternating_manifest import AlternatingManifest
from evotau.records import EpisodeRecord, EpisodeStatus, service_strategy_id
from evotau.service_skills import (
    ServiceSkill,
    ServiceSkillMemory,
    ServiceSkillMutation,
    ServiceSkillProvenance,
    apply_skill_mutation,
    render_service_skill_memory,
)
from evotau.strategies import PromptStrategy
from evotau.tau_adapter import service_agent_class
from evotau.web.view_models import (
    generation_view,
    service_strategy_view,
    strategy_diff_rows,
)


def _mutation(
    operation: str,
    *,
    analysis: str = "Evidence-based local repair.",
    target: str | None = None,
    trigger: str | None = None,
    guidance: str | None = None,
) -> dict:
    return {
        "analysis": analysis,
        "operation": operation,
        "target_skill_id": target,
        "skill": (
            None
            if trigger is None and guidance is None
            else {"trigger": trigger, "guidance": guidance}
        ),
    }


def test_empty_skill_memory_and_empty_rendering_are_supported() -> None:
    memory = ServiceSkillMemory()
    assert memory.skills == ()
    assert memory.to_dict() == {"skills": []}
    assert render_service_skill_memory(memory) == ""


@pytest.mark.parametrize(
    ("trigger", "guidance"),
    [("", "Do a useful thing."), ("When relevant.", "  ")],
)
def test_skill_requires_non_empty_trigger_and_guidance(
    trigger: str, guidance: str
) -> None:
    with pytest.raises(ValueError):
        ServiceSkill("skill-0001", trigger, guidance)


def test_memory_serialization_and_strategy_identity_are_canonical() -> None:
    first = ServiceSkill("skill-0001", "When A.", "Do A.")
    second = ServiceSkill("skill-0002", "When B.", "Do B.")
    memory_a = ServiceSkillMemory((second, first))
    memory_b = ServiceSkillMemory((first, second))
    assert memory_a.to_dict() == memory_b.to_dict()
    assert service_strategy_id(memory_a) == service_strategy_id(memory_b)
    assert [item.skill_id for item in memory_a.skills] == ["skill-0001", "skill-0002"]


def test_provenance_never_changes_runtime_skill_memory_identity_or_rendering() -> None:
    memory = ServiceSkillMemory((ServiceSkill("skill-0001", "When A.", "Do A."),))
    provenance_a = ServiceSkillProvenance("skill-0001", 0, 0, ("task-a",))
    provenance_b = ServiceSkillProvenance("skill-0001", 0, 2, ("task-a", "task-b"))
    artifact_a = {"memory": memory.to_dict(), "provenance": [provenance_a.to_dict()]}
    artifact_b = {"memory": memory.to_dict(), "provenance": [provenance_b.to_dict()]}
    assert artifact_a != artifact_b
    assert service_strategy_id(memory) == service_strategy_id(
        ServiceSkillMemory.from_mapping(artifact_b["memory"]),
    )
    rendered = render_service_skill_memory(memory)
    assert "task-a" not in rendered and "generation" not in rendered


def test_skill_memory_rendering_is_deterministic_includes_all_skills_and_policy_precedence() -> (
    None
):
    memory = ServiceSkillMemory(
        (
            ServiceSkill("skill-0002", "When B.", "Do B."),
            ServiceSkill("skill-0001", "When A.", "Do A."),
        )
    )
    rendered = render_service_skill_memory(memory)
    assert rendered == render_service_skill_memory(memory)
    assert rendered.index("## skill-0001") < rendered.index("## skill-0002")
    assert "## skill-0001" in rendered and "## skill-0002" in rendered
    assert "native task policy" in rendered
    assert "tool/backend evidence" in rendered


def test_add_assigns_next_stable_id_and_does_not_change_existing_skills() -> None:
    original = ServiceSkill("skill-0007", "Existing trigger.", "Existing guidance.")
    memory = ServiceSkillMemory((original,))
    changed = apply_skill_mutation(
        memory,
        ServiceSkillMutation.from_mapping(
            _mutation(
                "add",
                trigger="When another capability is needed.",
                guidance="Perform it carefully.",
            )
        ),
    )
    assert [item.skill_id for item in changed.skills] == ["skill-0007", "skill-0008"]
    assert changed.skills[0] is original
    assert service_strategy_id(changed) != service_strategy_id(memory)


def test_add_from_empty_memory_always_assigns_skill_0001() -> None:
    mutation = ServiceSkillMutation.from_mapping(
        _mutation(
            "add",
            trigger="When A.",
            guidance="Do A.",
        )
    )
    first = apply_skill_mutation(ServiceSkillMemory(), mutation)
    second = apply_skill_mutation(ServiceSkillMemory(), mutation)
    assert first.to_dict() == second.to_dict()
    assert first.skills[0].skill_id == "skill-0001"


def test_update_preserves_id_and_changes_only_the_target_skill() -> None:
    first = ServiceSkill("skill-0001", "Broad trigger.", "Original guidance.")
    second = ServiceSkill("skill-0002", "Other trigger.", "Other guidance.")
    memory = ServiceSkillMemory((first, second))
    changed = apply_skill_mutation(
        memory,
        ServiceSkillMutation.from_mapping(
            _mutation(
                "update",
                target="skill-0001",
                trigger="Narrow trigger.",
                guidance="Refined guidance.",
            )
        ),
    )
    updated = next(item for item in changed.skills if item.skill_id == "skill-0001")
    unchanged = next(item for item in changed.skills if item.skill_id == "skill-0002")
    assert updated.skill_id == "skill-0001"
    assert updated.trigger == "Narrow trigger."
    assert updated.guidance == "Refined guidance."
    assert unchanged is second
    assert service_strategy_id(changed) != service_strategy_id(memory)


def test_update_of_unknown_target_is_rejected() -> None:
    mutation = ServiceSkillMutation.from_mapping(
        _mutation(
            "update",
            target="skill-0002",
            trigger="When.",
            guidance="Do.",
        )
    )
    with pytest.raises(ValueError, match="does not exist"):
        apply_skill_mutation(ServiceSkillMemory(), mutation)


def test_no_op_returns_identical_memory_and_strategy_id() -> None:
    memory = ServiceSkillMemory((ServiceSkill("skill-0001", "When.", "Do."),))
    changed = apply_skill_mutation(
        memory, ServiceSkillMutation.from_mapping(_mutation("no_op"))
    )
    assert changed is memory
    assert service_strategy_id(changed) == service_strategy_id(memory)


@pytest.mark.parametrize(
    "payload",
    [
        _mutation("delete"),
        _mutation("add", target="skill-0001", trigger="When.", guidance="Do."),
        _mutation("add", trigger="When.", guidance="Do.", analysis=""),
        _mutation("update", trigger="When.", guidance="Do."),
        _mutation("no_op", trigger="When.", guidance="Do."),
        {**_mutation("add", trigger="When.", guidance="Do."), "extra": True},
    ],
)
def test_mutation_schema_rejects_invalid_operation_shapes(payload: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        ServiceSkillMutation.from_mapping(payload)


def test_evolver_output_rejects_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        alternating, "inferai_responses_generate_text", lambda **_kwargs: "not JSON"
    )
    evolver = alternating.LLMAlternatingEvolvers(
        model="fake",
        model_args={
            "api_protocol": "responses",
            "api_base": "https://example.test",
            "reasoning_effort": "high",
        },
    )
    with pytest.raises(alternating.EvolverJSONError):
        evolver.service_skill_mutation({})


def test_evolver_output_rejects_unknown_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = json.dumps(_mutation("rewrite_all"))
    monkeypatch.setattr(
        alternating, "inferai_responses_generate_text", lambda **_kwargs: content
    )
    evolver = alternating.LLMAlternatingEvolvers(
        model="fake",
        model_args={
            "api_protocol": "responses",
            "api_base": "https://example.test",
            "reasoning_effort": "high",
        },
    )
    with pytest.raises(ValueError, match="add, update, or no_op"):
        evolver.service_skill_mutation({})


def test_all_active_skills_are_injected_after_native_policy_without_provenance() -> (
    None
):
    class NativeAgent:
        def __init__(
            self, prompt: str = "Native Retail policy and tool instructions."
        ) -> None:
            self.prompt = prompt

        @property
        def system_prompt(self) -> str:
            return self.prompt

    memory = ServiceSkillMemory(
        tuple(
            ServiceSkill(
                f"skill-{index:04d}", f"Trigger {index}.", f"Guidance {index}."
            )
            for index in range(1, 4)
        )
    )
    adapted = service_agent_class(NativeAgent, memory)(
        prompt="Native Retail policy and tool instructions."
    )
    prompt = adapted.system_prompt
    assert prompt.startswith("Native Retail policy and tool instructions.")
    assert prompt.count("## skill-") == 3
    assert (
        prompt.index("## skill-0001")
        < prompt.index("## skill-0002")
        < prompt.index("## skill-0003")
    )
    assert "created_generation" not in prompt and "source_task_ids" not in prompt


class _SkillRunner:
    service_policy_text = "Fixed Retail policy."

    def __init__(self) -> None:
        self.tasks = {
            task_id: SimpleNamespace(
                id=task_id,
                description=f"Public description {task_id}",
                user_scenario=f"SECRET-SCENARIO-{task_id}",
                user_tools=[],
            )
            for task_id in ("a", "b")
        }
        self.calls: list[dict] = []

    def __call__(self, *, task_id, seed, customer, service, panel_name):
        customer_text = "" if customer is None else customer.text
        strategy_id = service_strategy_id(service)
        self.calls.append(
            {
                "task_id": task_id,
                "seed": seed,
                "customer": customer_text,
                "service": service.to_dict(),
                "service_id": strategy_id,
                "panel_name": panel_name,
            }
        )
        skills = getattr(service, "skills", ())
        guidance = "" if not skills else skills[0].guidance
        if customer_text == "C0":
            succeeded = guidance in {"C0 repair v1", "C0 repair v2"}
        elif customer_text == "C1" and guidance == "C0 repair v1":
            succeeded = task_id == "a"
        elif customer_text == "C1" and guidance == "C0 repair v2":
            succeeded = True
        else:
            succeeded = False
        return EpisodeRecord(
            episode_id=f"{panel_name}-{task_id}",
            task_id=task_id,
            seed=seed,
            customer_strategy_id="native" if customer is None else customer.text,
            service_strategy_id=strategy_id,
            status=EpisodeStatus.COMPLETE,
            task_success=succeeded,
            native_reward=float(succeeded),
            termination_reason="user_stop",
            trajectory_ref=None,
            raw_review={},
        )


def test_mocked_two_generation_add_then_update_is_accepted_and_versioned(
    tmp_path: Path,
) -> None:
    runner = _SkillRunner()
    evolver_contexts: list[dict] = []

    def service_evolver(context: dict) -> dict:
        evolver_contexts.append(context)
        if context["generation"] == 0:
            return _mutation(
                "add",
                trigger="When C0's task goal needs repair.",
                guidance="C0 repair v1",
            )
        return _mutation(
            "update",
            target="skill-0001",
            trigger="When C1 exposes a remaining gap.",
            guidance="C0 repair v2",
        )

    output = tmp_path / "skill-evolution"
    checkpoint = tmp_path / "checkpoint.json"
    result = run_alternating_evolution(
        tasks=runner.tasks,
        evolution_task_ids=("a", "b"),
        validation_task_ids=(),
        seed=1,
        evolution_fitness_seed=7,
        generations=2,
        customer_candidate_count=1,
        clean_panel_size=1,
        run_validation=False,
        initial_customer=PromptStrategy("C0"),
        initial_service=ServiceSkillMemory(),
        service_carrier="skill_memory_v1",
        runner=runner,
        customer_evolver=lambda context, _count: [
            "C0" if context["generation"] == 0 else "C1"
        ],
        service_evolver=service_evolver,
        domain_policy=runner.service_policy_text,
        output_directory=output,
        checkpoint_path=checkpoint,
        manifest_sha256="frozen-skill-run",
        service_evolver_model="mock-model",
        service_evolver_reasoning_effort="high",
        service_evolver_provider="mock",
    )

    generation_zero, generation_one = result.generations
    assert generation_zero["service_phase"]["accepted"] is True
    assert generation_zero["service_phase"]["operation"] == "add"
    assert generation_one["customer_phase"]["selected_customer"] == 0
    assert generation_one["service_phase"]["accepted"] is True
    assert generation_one["service_phase"]["operation"] == "update"
    assert generation_one["service_phase"]["skill_count_before"] == 1
    assert generation_one["service_phase"]["skill_count_after"] == 1
    assert result.service.skills[0].skill_id == "skill-0001"
    assert result.service.skills[0].guidance == "C0 repair v2"
    assert "user_scenario" not in evolver_contexts[0]
    assert all(
        "user_scenario" not in row["task"]
        for row in evolver_contexts[0]["task_interactions"]
    )
    assert "SECRET-SCENARIO" not in json.dumps(evolver_contexts[0])
    assert "reference_answer" not in json.dumps(evolver_contexts[0]).lower()
    assert "provenance" not in json.dumps(evolver_contexts[0])

    artifacts = output / "service-memory"
    g0 = json.loads((artifacts / "generation-0000-accepted.json").read_text())
    g0_version = json.loads((output / "generation-0000.json").read_text())
    g1_input = json.loads((artifacts / "generation-0001-input.json").read_text())
    g1_proposed = json.loads((artifacts / "generation-0001-proposed.json").read_text())
    g1_accepted = json.loads((artifacts / "generation-0001-accepted.json").read_text())
    assert g0["memory"]["skills"][0]["guidance"] == "C0 repair v1"
    assert g1_input["memory"]["skills"][0]["skill_id"] == "skill-0001"
    assert g1_proposed["old_skill"]["guidance"] == "C0 repair v1"
    assert g1_proposed["new_skill"]["guidance"] == "C0 repair v2"
    assert g1_proposed["evolver_model"] == "mock-model"
    assert g1_proposed["input_memory_id"] == g1_input["memory_id"]
    assert g1_accepted["memory"]["skills"][0]["guidance"] == "C0 repair v2"
    assert (
        g0_version["service_after"]["strategy"]["skills"][0]["guidance"]
        == "C0 repair v1"
    )
    assert generation_one["service_phase"]["regression_diagnostics"]["counts"] == {
        "fail_to_pass": 1,
        "pass_to_fail": 0,
        "pass_to_pass": 1,
        "fail_to_fail": 0,
    }
    assert set(
        generation_one["service_phase"]["regression_diagnostics"]["task_ids"][
            "fail_to_pass"
        ]
    ) == {"b"}

    before_resume_calls = len(runner.calls)
    resumed = run_alternating_evolution(
        tasks=runner.tasks,
        evolution_task_ids=("a", "b"),
        validation_task_ids=(),
        seed=1,
        evolution_fitness_seed=7,
        generations=2,
        customer_candidate_count=1,
        clean_panel_size=1,
        run_validation=False,
        initial_customer=PromptStrategy("C0"),
        initial_service=ServiceSkillMemory(),
        service_carrier="skill_memory_v1",
        runner=runner,
        customer_evolver=lambda *_args: pytest.fail(
            "completed resume called Customer Evolver"
        ),
        service_evolver=lambda *_args: pytest.fail(
            "completed resume called Service Evolver"
        ),
        domain_policy=runner.service_policy_text,
        output_directory=output,
        checkpoint_path=checkpoint,
        manifest_sha256="frozen-skill-run",
    )
    assert len(runner.calls) == before_resume_calls
    assert resumed.service.to_dict() == result.service.to_dict()
    assert resumed.generations == result.generations


def test_rejected_update_does_not_replace_active_memory_and_no_op_skips_e(
    tmp_path: Path,
) -> None:
    original = ServiceSkillMemory(
        (ServiceSkill("skill-0001", "When a task is active.", "stable guidance"),)
    )
    provenance = (ServiceSkillProvenance("skill-0001", 0, 0, ("a", "b")),)
    runner = _SkillRunner()
    contexts: list[dict] = []
    evolver_operations: list[str] = []

    def service_evolver(context: dict) -> dict:
        contexts.append(context)
        if context["generation"] == 0:
            evolver_operations.append("update")
            return _mutation(
                "update",
                target="skill-0001",
                trigger="Always.",
                guidance="regression",
            )
        evolver_operations.append("no_op")
        return _mutation("no_op")

    # For this test, the existing rule works and the rejected update makes every task fail.
    def runner_call(*, task_id, seed, customer, service, panel_name):
        runner.calls.append(
            {"task_id": task_id, "panel_name": panel_name, "service": service.to_dict()}
        )
        succeeded = service.skills[0].guidance == "stable guidance"
        return EpisodeRecord(
            episode_id=f"{panel_name}-{task_id}",
            task_id=task_id,
            seed=seed,
            customer_strategy_id=customer.text,
            service_strategy_id=service_strategy_id(service),
            status=EpisodeStatus.COMPLETE,
            task_success=succeeded,
            native_reward=float(succeeded),
            termination_reason="user_stop",
            raw_review={},
        )

    runner.__call__ = runner_call

    # Special methods are resolved on the class, so use a tiny callable wrapper.
    class CallableRunner:
        service_policy_text = runner.service_policy_text
        tasks = runner.tasks
        calls = runner.calls

        def __call__(self, **kwargs):
            return runner_call(**kwargs)

    output = tmp_path / "rejected"
    result = run_alternating_evolution(
        tasks=runner.tasks,
        evolution_task_ids=("a", "b"),
        validation_task_ids=(),
        seed=1,
        generations=2,
        customer_candidate_count=1,
        clean_panel_size=1,
        run_validation=False,
        initial_customer=PromptStrategy("C0"),
        initial_service=original,
        initial_service_provenance=provenance,
        service_carrier="skill_memory_v1",
        runner=CallableRunner(),
        customer_evolver=lambda _context, _count: ["C0"],
        service_evolver=service_evolver,
        domain_policy="Fixed policy.",
        output_directory=output,
        checkpoint_path=tmp_path / "rejected-checkpoint.json",
        manifest_sha256="frozen-rejection-run",
    )
    assert result.generations[0]["service_phase"]["accepted"] is False
    assert (
        result.generations[0]["service_after"]["strategy"]["skills"][0]["guidance"]
        == "stable guidance"
    )
    assert (
        result.generations[1]["service_before"]["strategy"]["skills"][0]["guidance"]
        == "stable guidance"
    )
    assert result.generations[1]["service_phase"]["operation"] == "no_op"
    assert result.generations[1]["service_phase"]["challenge_episodes"] == []
    assert evolver_operations == ["update", "no_op"]
    assert contexts[1]["current_service_skill_memory"] == original.to_dict()
    service_candidate_calls = [
        item
        for item in runner.calls
        if item["panel_name"].endswith("service-candidate")
    ]
    assert len(service_candidate_calls) == 2
    artifacts = output / "service-memory"
    assert (artifacts / "generation-0000-proposed.json").is_file()
    assert not (artifacts / "generation-0000-accepted.json").exists()
    assert (
        json.loads((artifacts / "generation-0000-active.json").read_text())["memory"]
        == original.to_dict()
    )
    assert (
        json.loads((artifacts / "generation-0001-active.json").read_text())["memory"]
        == original.to_dict()
    )


def test_rejected_add_reservation_survives_resume_without_reusing_skill_id(
    tmp_path: Path,
) -> None:
    tasks = {
        task_id: SimpleNamespace(
            id=task_id,
            description=f"Public {task_id}",
            user_scenario=f"private {task_id}",
            user_tools=[],
        )
        for task_id in ("a", "b", "c")
    }

    class Runner:
        service_policy_text = "Fixed policy."

        def __call__(self, *, task_id, seed, customer, service, panel_name):
            guidance = "" if not service.skills else service.skills[0].guidance
            succeeded = (
                task_id == "a"
                if not service.skills
                else guidance == "good reusable rule"
            )
            return EpisodeRecord(
                episode_id=f"{panel_name}-{task_id}",
                task_id=task_id,
                seed=seed,
                customer_strategy_id=customer.text,
                service_strategy_id=service_strategy_id(service),
                status=EpisodeStatus.COMPLETE,
                task_success=succeeded,
                native_reward=float(succeeded),
                termination_reason="user_stop",
                raw_review={},
            )

    runner = Runner()
    output = tmp_path / "rejected-add-resume"
    checkpoint = tmp_path / "rejected-add-checkpoint.json"

    def service_evolver(context):
        return _mutation(
            "add",
            trigger=f"Reusable trigger generation {context['generation']}.",
            guidance="bad one-off rule"
            if context["generation"] == 0
            else "good reusable rule",
        )

    first = run_alternating_evolution(
        tasks=tasks,
        evolution_task_ids=("a", "b", "c"),
        validation_task_ids=(),
        seed=1,
        generations=1,
        customer_candidate_count=1,
        clean_panel_size=1,
        run_validation=False,
        initial_customer=PromptStrategy("C0"),
        initial_service=ServiceSkillMemory(),
        service_carrier="skill_memory_v1",
        runner=runner,
        customer_evolver=lambda *_args: ["C0"],
        service_evolver=service_evolver,
        domain_policy=runner.service_policy_text,
        output_directory=output,
        checkpoint_path=checkpoint,
        manifest_sha256="same-frozen-config",
    )
    assert first.generations[0]["service_phase"]["accepted"] is False
    assert first.service.skills == ()
    saved_checkpoint = json.loads(checkpoint.read_text())
    assert saved_checkpoint["skill_id_high_watermark"] == 1

    resumed = run_alternating_evolution(
        tasks=tasks,
        evolution_task_ids=("a", "b", "c"),
        validation_task_ids=(),
        seed=1,
        generations=2,
        customer_candidate_count=1,
        clean_panel_size=1,
        run_validation=False,
        initial_customer=PromptStrategy("C0"),
        initial_service=ServiceSkillMemory(),
        service_carrier="skill_memory_v1",
        runner=runner,
        customer_evolver=lambda *_args: ["C0"],
        service_evolver=service_evolver,
        domain_policy=runner.service_policy_text,
        output_directory=output,
        checkpoint_path=checkpoint,
        manifest_sha256="same-frozen-config",
    )
    assert resumed.generations[1]["service_phase"]["accepted"] is True
    assert resumed.service.skills[0].skill_id == "skill-0002"
    assert (
        json.loads((output / "generation-0000.json").read_text())
        == first.generations[0]
    )
    assert (
        json.loads(
            (output / "service-memory/generation-0001-proposed.json").read_text(),
        )["new_skill"]["skill_id"]
        == "skill-0002"
    )


def test_skill_memory_smoke_config_freezes_carrier_and_disables_v_h() -> None:
    project = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (
            project / "configs/alternating-skill-memory-v1-gpt61sol-e20-c1.yaml"
        ).read_text(encoding="utf-8"),
    )
    experiment = config["experiment"]
    manifest = AlternatingManifest.from_mapping(config)
    assert len(manifest.evolution_task_ids) == 20
    assert manifest.service_carrier == "skill_memory_v1"
    assert manifest.customer_carrier == "prompt_strategy"
    assert manifest.service_skill_runtime == "inject_all"
    assert manifest.service_mutation_ops == ("add", "update", "no_op")
    assert manifest.max_service_mutations_per_generation == 1
    assert experiment["generations"] == 2 and experiment["customer_candidates"] == 1
    assert experiment["evolution_fitness_seed"] == 1
    assert experiment["run_validation"] is False and experiment["run_heldout"] is False
    assert experiment["real_provider_enabled"] is True
    assert not (Path(experiment["output_path"]).exists())


def test_web_strategy_view_and_diff_render_service_skills_by_stable_id() -> None:
    before = {
        "strategy_id": "s0",
        "carrier": "skill_memory_v1",
        "strategy": {"skills": []},
    }
    after = {
        "strategy_id": "s1",
        "carrier": "skill_memory_v1",
        "strategy": {
            "skills": [
                {
                    "skill_id": "skill-0001",
                    "trigger": "When a clarification interrupts resolved work.",
                    "guidance": "Preserve the resolved parameters.",
                }
            ],
        },
    }
    view = service_strategy_view(after)
    assert view["carrier"] == "skill_memory_v1"
    assert view["skills"][0]["skill_id"] == "skill-0001"
    assert "Preserve the resolved parameters" in view["text"]
    diff = strategy_diff_rows(before, after, "service")
    assert diff["added"] == ("skill-0001",)
    assert diff["rows"][0]["changed"] is True

    generation = generation_view(
        {
            "generation": 0,
            "customer_before": {"strategy_id": "c0", "strategy": "C0"},
            "customer_after": {"strategy_id": "c0", "strategy": "C0"},
            "service_before": before,
            "service_after": after,
            "customer_phase": {"selection": {}},
            "service_phase": {
                "accepted": True,
                "selection": {"reason": "E improved."},
                "proposed_strategy": after["strategy"],
            },
        }
    )
    assert "skill-0001" in generation["proposed_service_text"]
    assert generation["service_diff"]["added"] == ("skill-0001",)


def test_skill_proposal_partial_evaluation_resume_reuses_immutable_artifacts(tmp_path):
    tasks = {key: SimpleNamespace(id=key, description=key, user_scenario=key, user_tools=[])
             for key in ('a', 'b')}
    cache = {}
    completed = []
    interrupted = False
    evolver_calls = []

    def runner(*, task_id, seed, customer, service, panel_name):
        nonlocal interrupted
        key = (task_id, seed, customer.text, service_strategy_id(service))
        if key in cache:
            return cache[key]
        if service.skills and task_id == 'b' and not interrupted:
            interrupted = True
            raise RuntimeError('provider disconnected')
        success = bool(service.skills) or task_id == 'a'
        record = EpisodeRecord(
            episode_id=str(len(completed)), task_id=task_id, seed=seed,
            customer_strategy_id=customer.text,
            service_strategy_id=service_strategy_id(service),
            status=EpisodeStatus.COMPLETE, task_success=success,
            native_reward=float(success), termination_reason='user_stop', raw_review={},
        )
        completed.append(key)
        cache[key] = record
        return record

    def customer_evolver(*_args):
        evolver_calls.append('customer')
        return ['C0']

    def service_evolver(*_args):
        evolver_calls.append('service')
        return _mutation('add', trigger='Reusable trigger.', guidance='Reusable guidance.')

    kwargs = {
        "tasks": tasks, "evolution_task_ids": ('a', 'b'), "validation_task_ids": (), "seed": 1,
        "generations": 1, "customer_candidate_count": 1, "clean_panel_size": 1,
        "max_parallel_episodes": 1, "run_validation": False,
        "initial_customer": PromptStrategy('C0'), "initial_service": ServiceSkillMemory(),
        "service_carrier": 'skill_memory_v1', "runner": runner,
        "customer_evolver": customer_evolver, "service_evolver": service_evolver,
        "domain_policy": 'Fixed policy.', "output_directory": tmp_path,
        "checkpoint_path": tmp_path / 'checkpoint.json', "manifest_sha256": 'frozen',
    }
    with pytest.raises(RuntimeError, match='provider disconnected'):
        run_alternating_evolution(**kwargs)
    paths = [tmp_path / 'service-memory/generation-0000-input.json',
             tmp_path / 'service-memory/generation-0000-proposed.json',
             tmp_path / 'generation-0000-service-proposal.json']
    before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths]
    assert len(completed) == 3
    result = run_alternating_evolution(**kwargs)
    assert len(completed) == 4
    assert evolver_calls == ['customer', 'service']
    assert result.generations[0]['service_phase']['accepted'] is True
    assert [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths] == before


def test_immutable_artifact_same_bytes_reused_but_changes_rejected(tmp_path):
    path = tmp_path / 'artifact.json'
    alternating._write_json_once(path, {'value': 1})
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    alternating._write_json_once(path, {'value': 1})
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    with pytest.raises(FileExistsError, match='content differs'):
        alternating._write_json_once(path, {'value': True})
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
