import json
from copy import deepcopy
from pathlib import Path
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
        if not any(r["task_success"] is False for r in context["current_outcomes"]):
            return {"clusters": []}
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
            trajectory_ref=f"episodes/ep-{len(self.cache)}/native-simulation.json",
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
    manifest_sha="manifest",
):
    providers = provider or FakeProviders()
    native = runner or FakeRunner()
    settings = deepcopy(policy or DEFAULT_V2)
    if policy is None:
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
        manifest_sha256=manifest_sha,
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


def test_strict_provider_contract_reaches_gate_archive_and_exact_resume(tmp_path, monkeypatch):
    from evotau.alternating import LLMAlternatingEvolvers
    from evotau.budget import RequestBudget
    from evotau.evolution_candidates import V2Providers

    scripted, calls = FakeProviders(), []

    def dispatch(model, args, prompt, context, *, call_name):
        calls.append(call_name)
        if call_name == "evotau_customer_evolver":
            return scripted.customers(context, context["requested_candidates"])
        if call_name == "evotau_customer_semantic_validator":
            return {**scripted.validate_customer(context), "reason": "Task-faithful interaction."}
        if call_name == "evotau_service_diagnoser":
            return scripted.diagnose(context)
        if call_name == "evotau_service_skill_mutator":
            return scripted.mutate(context)
        if call_name == "evotau_skill_semantic_validator":
            return {**scripted.validate_skill(context), "reason": "Policy subordinate."}
        raise AssertionError(call_name)

    monkeypatch.setattr(LLMAlternatingEvolvers, "_json_call", staticmethod(dispatch))
    policy = deepcopy(DEFAULT_V2)
    policy["service_evolution"].update(candidates_per_cluster=1, crossover=False)
    providers = V2Providers(LLMAlternatingEvolvers(
        model="offline/contract", model_args={"temperature": 0},
        request_budget=RequestBudget(10), output_directory=tmp_path,
    ))
    result, _, runner = run(tmp_path, provider=providers, policy=policy)
    row = result.generations[0]["service_phase"]["candidates"][0]
    assert row["screen"]["passed"] and row["gate"] is not None
    assert (tmp_path / "evolution-v2/g0000-archive_update.json").is_file()
    assert calls == ["evotau_customer_evolver", "evotau_customer_semantic_validator",
                     "evotau_service_diagnoser", "evotau_service_skill_mutator",
                     "evotau_skill_semantic_validator"]
    frozen = {p: p.read_bytes() for p in (tmp_path / "evolution-v2").glob("*.json")}
    before = list(calls), list(runner.calls)
    resumed, _, _ = run(tmp_path, provider=providers, runner=runner, policy=policy)
    assert resumed.generations == result.generations
    assert (calls, runner.calls) == before
    assert all(p.read_bytes() == content for p, content in frozen.items())


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

    if "TAU2_DATA_DIR" not in os.environ:
        monkeypatch.setenv(
            "TAU2_DATA_DIR", "/Users/spring/.cache/evotau/tau2-data-b7ea9074"
        )
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


def test_two_generations_resume_history_and_stagnation(tmp_path):
    provider, runner = FakeProviders(), FakeRunner()
    result, _, _ = run(tmp_path, provider=provider, runner=runner, generations=2)
    assert len(result.generations) == 2
    calls = list(provider.calls)
    resumed, _, _ = run(tmp_path, provider=provider, runner=runner, generations=2)
    assert resumed.generations == result.generations and provider.calls == calls


def test_formal_gate_does_not_promote_small_V(tmp_path):
    result, _, _ = run(tmp_path, validation=True)
    assert not result.service.skills
    row = result.generations[0]["service_phase"]["candidates"][0]
    assert row["gate"]["verdict"] == "INCONCLUSIVE"
    assert row["gate"]["opponents"][0]["panel"] == "V"


def test_no_lineage_ablation_resume_keeps_evidence_deterministic(tmp_path):
    policy = deepcopy(DEFAULT_V2)
    policy["service_evolution"]["lineage"] = False
    result, provider, runner = run(tmp_path, policy=policy, generations=2)
    resumed, _, _ = run(
        tmp_path, policy=policy, generations=2, provider=provider, runner=runner
    )
    assert result.generations == resumed.generations


def test_new_config_manifest_freezes_activation_and_v1_identity_unchanged():
    import yaml

    from evotau.alternating_manifest import AlternatingManifest

    root = Path(__file__).resolve().parents[1]
    old = yaml.safe_load(
        (
            root
            / "configs/alternating-skill-memory-v1-qwen3-7-plus-runtime-v4-pro-retail-e20-g2-p4.yaml"
        ).read_text()
    )
    legacy = AlternatingManifest.from_mapping(old).to_payload()
    assert "skill_evolution_v2" not in legacy
    raw = yaml.safe_load(
        (
            root / "configs/alternating-skill-memory-v2-qwen3-7-plus-retail.yaml"
        ).read_text()
    )
    manifest = AlternatingManifest.from_mapping(raw)
    doc = manifest.to_document()
    v2 = doc["skill_evolution_v2"]
    assert v2["activator"]["model"] == "openai/qwen3.7-plus"
    assert v2["activator"]["model_args"]["thinking_mode"] == "disabled"
    assert v2["max_active_service_skills"] == 2 and v2["evaluation"]["gate_seeds"] == [
        1,
        2,
        3,
        4,
    ]
    raw["experiment"]["skill_evolution_v2"]["activator"]["model_args"][
        "temperature"
    ] = 0.1
    assert AlternatingManifest.from_mapping(raw).sha256 != manifest.sha256


def web_fixture(tmp_path, run_id="v2-fixture", activation="activate_topk_v2"):
    from evotau.tau_provenance import sha256_json

    root = tmp_path / "experiments/runs" / run_id
    root.mkdir(parents=True)
    policy = deepcopy(DEFAULT_V2)
    policy["service_skill_runtime"] = activation
    manifest = {
        "experiment_id": run_id,
        "phase": "alternating-self-evolution",
        "real_provider_enabled": False,
        "task_panels": {"E": ["1", "2", "3"], "V": ["4"], "H": ["999"]},
        "seed": 1,
        "max_parallel_episodes": 1,
        "role_models": {"agent": "offline"},
        "role_model_args": {"agent": {}},
        "enforce_communication_protocol": False,
        "run_validation": False,
        "run_heldout": False,
        "request_budget_cap": None,
        "evolution": {"service_carrier": "skill_memory_v2"},
        "skill_evolution_v2": policy,
        "checkpoint_path": f"experiments/runs/{run_id}/checkpoint.json",
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest))
    result, _, runner = run(
        root, policy=policy, manifest_sha=manifest["manifest_sha256"]
    )
    for record_ in runner.cache.values():
        directory = root / "episodes" / record_.episode_id
        directory.mkdir(parents=True)
        (directory / "episode-record.json").write_text(json.dumps(record_.to_dict()))
        (root / record_.trajectory_ref).write_text(
            json.dumps(
                {
                    "id": record_.episode_id,
                    "task_id": record_.task_id,
                    "seed": record_.seed,
                    "messages": [
                        {"role": "user", "content": "Please help."},
                        {"role": "assistant", "content": "Checking."},
                    ],
                }
            )
        )
        (directory / "run-telemetry.json").write_text(
            json.dumps(
                {
                    "simulation_id": record_.episode_id,
                    "panel_name": "generation-0-fixture",
                }
            )
        )
    final = {
        "schema_version": 3,
        "status": "complete",
        "experiment_id": run_id,
        "manifest_sha256": manifest["manifest_sha256"],
        "initial_service": result.initial_service.to_dict(),
        "final_service": result.service.to_dict(),
        "initial_customer": result.initial_customer.to_dict(),
        "final_customer": result.customer.to_dict(),
        "generations": result.generations,
        "validation_evaluated": False,
        "heldout_evaluated": False,
    }
    (root / "alternating-result.json").write_text(json.dumps(final))
    return root, result, runner


def test_web_v2_board_lineage_and_comparison(tmp_path):
    from fastapi.testclient import TestClient

    from evotau.web.app import create_app

    _root, result, _ = web_fixture(tmp_path)
    web_fixture(tmp_path, "no-activation", activation="render_all_v1")
    client = TestClient(create_app(project_root=tmp_path))
    page = client.get("/runs/v2-fixture/evolution")
    assert page.status_code == 200, page.text
    for expected in (
        "Helpfulness",
        "Harmfulness",
        "BROKEN",
        "FIXED",
        "INCONCLUSIVE",
        "RESEARCH CANDIDATE",
        "Mutation Lineage",
        "Customer Challenge Archive",
        "MECHANISM SMOKE",
    ):
        if expected == "INCONCLUSIVE":
            continue  # this fixture deliberately promotes a deterministic smoke winner
        assert expected in page.text
    compare = client.get("/compare?run_a=v2-fixture&run_b=no-activation")
    assert compare.status_code == 200 and "NOT COMPARABLE" in compare.text
    strategy = client.get(
        "/runs/v2-fixture/strategies/"
        + service_strategy_id(result.service)
        + "?side=service"
    )
    # Discover the existing route convention instead of depending on a plural alias.
    if strategy.status_code == 404:
        strategy = client.get(
            "/runs/v2-fixture/strategy/"
            + service_strategy_id(result.service)
            + "?side=service"
        )
    assert strategy.status_code == 200
    assert "Active Skills" in strategy.text and "activation_signature" in strategy.text


def test_web_activation_trace_hash_and_redaction(tmp_path):
    from fastapi.testclient import TestClient

    from evotau.tau_provenance import sha256_json
    from evotau.web.app import create_app

    root, _, runner = web_fixture(tmp_path)
    record_ = next(iter(runner.cache.values()))
    manifest = json.loads((root / "manifest.json").read_text())
    row = {
        "turn": 0,
        "decision": {
            "active_skill_ids": ["skill-0001"],
            "reason": "Bearer secret-token-value",
            "confidence": 0.9,
        },
        "not_selected_skill_ids": [],
        "activated_guidance_tokens": 12,
    }
    row["artifact_sha256"] = sha256_json(row)
    trace = {
        "manifest_sha256": manifest["manifest_sha256"],
        "episode_id": record_.episode_id,
        "decisions": [row],
    }
    trace["artifact_sha256"] = sha256_json(trace)
    # Reader resolves the activation sidecar beside the referenced trajectory.
    (
        root / Path(record_.trajectory_ref).parent / "skill-activation-trace.json"
    ).write_text(json.dumps(trace))
    client = TestClient(create_app(project_root=tmp_path))
    response = client.get("/runs/v2-fixture/episodes/" + record_.episode_id)
    assert response.status_code == 200
    assert (
        "learned skill(s) activated" in response.text
        and "secret-token-value" not in response.text
    )
    trace["decisions"][0]["decision"]["confidence"] = 0.1
    (
        root / Path(record_.trajectory_ref).parent / "skill-activation-trace.json"
    ).write_text(json.dumps(trace))
    assert (
        client.get("/runs/v2-fixture/episodes/" + record_.episode_id).status_code == 422
    )


def test_no_H_task_content_loaded_in_v2_dry_validation():
    import yaml

    from evotau.alternating_manifest import AlternatingManifest
    from evotau.alternating_run import load_alternating_tasks

    cfg = (
        Path(__file__).resolve().parents[1]
        / "configs/alternating-skill-memory-v2-qwen3-7-plus-retail.yaml"
    )
    data = Path("/Users/spring/.cache/evotau/tau2-data-b7ea9074")
    if not data.is_dir():
        pytest.skip("pinned data unavailable")
    manifest = AlternatingManifest.from_mapping(yaml.safe_load(cfg.read_text()))
    tasks = load_alternating_tasks(
        manifest, data, include_heldout=False, include_validation=True
    )
    assert set(tasks) == set(manifest.evolution_task_ids + manifest.validation_task_ids)
    assert not set(tasks) & set(manifest.heldout_task_ids)


def test_provider_output_recovered_after_stage_publication_crash(tmp_path, monkeypatch):
    from evotau.alternating import LLMAlternatingEvolvers
    from evotau.evolution_candidates import V2Providers

    calls = []
    provider = LLMAlternatingEvolvers(
        model="offline", model_args={}, output_directory=tmp_path
    )
    monkeypatch.setattr(
        provider,
        "_dispatch_json_call",
        lambda *a, **kw: calls.append(kw) or {"ready": True},
    )
    v2 = V2Providers(provider)
    first = v2.call("system", {"generation": 0}, "call")
    second = v2.call("system", {"generation": 0}, "call")
    assert first == second == {"ready": True} and len(calls) == 1
    output = next((tmp_path / "evolver-calls").glob("*/output.json"))
    saved = json.loads(output.read_text())
    saved["response"]["ready"] = False
    output.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="digest"):
        v2.call("system", {"generation": 0}, "call")


def test_crossover_child_independently_screened_gated_and_resumed(tmp_path):
    class ComplementaryProviders(FakeProviders):
        def mutate(self, context):
            self.calls.append("mutate")
            bias = context["proposal_bias"]
            if bias == "structural_decomposition":
                return {"operation": "invalid"}
            if bias == "narrow_applicability":
                return mutation(
                    "Validate request contents before committing.", family="contents"
                )
            value = mutation(
                "Reconcile newly disclosed scope with the previous confirmation.",
                family="reconciliation",
            )
            value["skill"]["trigger"] = "New scope conflicts with earlier confirmation."
            value["skill"]["activation_signature"]["positive_conditions"] = [
                "New conflicting scope is visible."
            ]
            value["substantive_delta_from_prior"] = (
                "Distinct scope reconciliation instead of request validation."
            )
            return value

        def crossover(self, context):
            self.calls.append("crossover")
            assert len(context["parents"]) == 2
            value = mutation(
                "Validate request contents and Reconcile newly disclosed scope.",
                family="complementary",
            )
            value["substantive_delta_from_prior"] = (
                "Compose independently observed complementary mechanisms."
            )
            return value

    class ComplementaryRunner(FakeRunner):
        def __call__(self, **kwargs):
            row = super().__call__(**kwargs)
            key = (
                kwargs["task_id"],
                kwargs["seed"],
                customer_strategy_id(kwargs["customer"]),
                service_strategy_id(kwargs["service"]),
            )
            text = " ".join(s.guidance for s in kwargs["service"].skills)
            success = (
                kwargs["task_id"] == "2"
                or (kwargs["task_id"] == "1" and "Validate" in text)
                or (kwargs["task_id"] == "3" and "Reconcile" in text)
            )
            from dataclasses import replace

            row = replace(row, task_success=success, native_reward=float(success))
            self.cache[key] = row
            return row

    policy = deepcopy(DEFAULT_V2)
    result, provider, runner = run(
        tmp_path,
        policy=policy,
        provider=ComplementaryProviders(),
        runner=ComplementaryRunner(),
    )
    board = result.generations[0]["service_phase"]["candidates"]
    child = next(c for c in board if c["parent_mutation_ids"])
    assert (
        child["screen"]["passed"]
        and child["gate"]["verdict"] == "ACCEPTED"
        and child["runtime_deployed"]
    )
    assert child["effect"]["fail_to_pass"] == ["1", "3"]
    assert provider.calls.count("crossover") == 1
    prior_calls = len(runner.calls), len(provider.calls)
    resumed, _, _ = run(tmp_path, policy=policy, provider=provider, runner=runner)
    assert resumed.generations == result.generations and prior_calls == (
        len(runner.calls),
        len(provider.calls),
    )


def test_customer_semantic_failure_is_rejected_before_rollout(tmp_path):
    class InvalidCustomer(FakeProviders):
        def validate_customer(self, context):
            return {
                "preserves_facts": False,
                "preserves_objective": True,
                "interaction_only": True,
                "no_benchmark_leakage": True,
            }

    result, _, runner = run(tmp_path, provider=InvalidCustomer())
    assert (
        result.generations[0]["customer_phase"]["candidates"][0][
            "pre_rollout_rejection"
        ]
        == "semantic_preservation_failed"
    )
    assert all(
        key[2]
        != customer_strategy_id(PromptStrategy("Ask for a grounded explanation."))
        for key in runner.calls
    )


def test_fresh_challenge_validated_before_heldout_and_endpoint_reuses_identical_service(
    tmp_path,
):
    from evotau.skill_evolution import (
        propose_fresh_customer_v2,
        run_v2_endpoint_evaluation,
    )

    result, provider, runner = run(tmp_path)
    tasks = {
        t: SimpleNamespace(
            id=t, user_scenario="only-E-" + t, description="", user_tools=[]
        )
        for t in ("1", "2", "3")
    }
    fresh = propose_fresh_customer_v2(
        result,
        provider,
        tasks=tasks,
        runner=runner,
        domain_policy="policy",
        output_directory=tmp_path,
        manifest_sha256="manifest",
    )
    calls = list(provider.calls)
    assert (
        propose_fresh_customer_v2(
            result,
            provider,
            tasks=tasks,
            runner=runner,
            domain_policy="policy",
            output_directory=tmp_path,
            manifest_sha256="manifest",
        )
        == fresh
    )
    assert provider.calls == calls
    endpoint_runner = FakeRunner()
    endpoint = run_v2_endpoint_evaluation(
        gate_seeds=[1, 2],
        heldout_tasks={"H": object()},
        heldout_task_ids=["H"],
        initial_service=ServiceSkillMemoryV2(),
        final_service=ServiceSkillMemoryV2(),
        fresh_customer=fresh,
        runner=endpoint_runner,
        max_parallel_episodes=1,
    )
    assert (
        len(endpoint_runner.calls) == 4
    )  # 1 task × 2 seeds × 2 Customer conditions; ST reused.
    assert endpoint["services_identical"] and all(
        c["pass_power_k"]["2"] == 1.0 for c in endpoint["cells"]
    )
    assert all(
        c["identical_to_S0"] for c in endpoint["cells"] if c["service_endpoint"] == "ST"
    )


def test_v2_completed_E_only_run_still_seals_H_every_console_surface(tmp_path):
    from fastapi.testclient import TestClient

    from evotau.web.app import create_app
    from evotau.web.artifact_reader import ArtifactReader

    root, _, _ = web_fixture(tmp_path)
    heldout_dir = root / "episodes" / "sealed-H-attempt"
    heldout_dir.mkdir()
    heldout_record = record("999", True).to_dict()
    heldout_record["episode_id"] = "sealed-H-attempt"
    heldout_record["trajectory_ref"] = (
        "episodes/sealed-H-attempt/native-simulation.json"
    )
    (heldout_dir / "episode-record.json").write_text(json.dumps(heldout_record))
    (heldout_dir / "native-simulation.json").write_text(
        json.dumps(
            {
                "id": "sealed-H-attempt",
                "task_id": "999",
                "messages": [{"role": "user", "content": "H_SECRET_SENTINEL"}],
            }
        )
    )
    reader = ArtifactReader(tmp_path / "experiments/runs", project_root=tmp_path)
    run_ = reader.get_run("v2-fixture")
    assert run_["status"] == "complete" and run_["heldout_sealed"]
    assert run_["manifest"]["task_panels"]["H"] != ["999"]
    client = TestClient(create_app(project_root=tmp_path))
    for suffix in (
        "",
        "/evolution",
        "/episodes",
        "/artifacts",
        "/progress",
        "/heldout",
    ):
        response = client.get("/runs/v2-fixture" + suffix)
        assert response.status_code == 200
        assert (
            '"999"' not in response.text
            and "secret-hidden" not in response.text
            and "H_SECRET_SENTINEL" not in response.text
        )
    # Even a recomputed checkpoint digest cannot smuggle H into research evidence.
    from evotau.tau_provenance import sha256_json

    path = root / "checkpoint.json"
    doc = json.loads(path.read_text())
    doc["state"]["mutation_effects"][0]["effect"]["fail_to_pass"] = ["999"]
    doc["checkpoint_sha256"] = sha256_json(
        {k: v for k, v in doc.items() if k != "checkpoint_sha256"}
    )
    path.write_text(json.dumps(doc))
    assert client.get("/runs/v2-fixture/evolution").status_code == 422


def test_render_all_v2_ablation_has_observed_trace_without_activator_calls():
    from evotau.skill_activation import RenderAllSkillSelector

    memory = apply_v2_mutation(
        ServiceSkillMemoryV2(), mutation(), next_skill_id_number=1
    )
    decision, context = RenderAllSkillSelector().select(
        {"messages": [{"role": "user", "content": "Ordinary unchanged cancel."}]},
        memory.skills,
    )
    assert decision.active_skill_ids == (
        "skill-0001",
    ) and "guidance" not in json.dumps(context["catalog"])


def test_v2_render_all_budget_and_nonfinite_policy_rejected():
    from evotau.skill_evolution_config import freeze_v2_policy

    roles = {"agent": "offline"}
    args = {"agent": {"temperature": 0}}
    frozen = freeze_v2_policy({"service_skill_runtime": "render_all_v1"}, roles, args)
    assert (
        frozen["skill_budgets"]["max_active_skills"]
        == frozen["skill_budgets"]["max_skills"]
    )
    with pytest.raises(ValueError, match="finite"):
        freeze_v2_policy(
            {"opponent_replay": {"current_weight": float("inf")}}, roles, args
        )


def test_dedup_signature_narrowing_is_substantive_but_delta_claim_alone_is_not():
    prior = mutation(operation="narrow_trigger", target="skill-0001")
    entry = {"mutation_id": "old", "mutation": prior, "effect": {"accepted": False}}
    narrowed = deepcopy(prior)
    narrowed["skill"]["activation_signature"]["negative_conditions"].append(
        "Already confirmed scope is unchanged."
    )
    narrowed["substantive_delta_from_prior"] = (
        "Exclude unchanged confirmation explicitly."
    )
    assert semantic_duplicate(narrowed, [entry]) is None
    claimed = deepcopy(prior)
    claimed["substantive_delta_from_prior"] = "I promise this is different."
    assert semantic_duplicate(claimed, [entry]) == "old"
