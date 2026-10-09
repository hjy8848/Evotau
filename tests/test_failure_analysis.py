from copy import deepcopy
from itertools import combinations

import pytest
from test_evolution_context import row
from test_skill_evolution_v2 import FakeProviders, bind_evidence, mutation, run

from evotau.evolution_context import build_service_mutation_evidence
from evotau.failure_analysis import (
    ANALYST_VERSION,
    analysis_outcome,
    select_distinct_hypotheses,
    validate_analysis,
    validate_assigned_mutation,
)
from evotau.skill_evolution_config import DEFAULT_V2


def context():
    return {
        "requested_hypotheses": 3,
        "task_interactions": build_service_mutation_evidence(
            [row("1", False), row("2", True)], case_chars=None
        ),
    }


def hypothesis(ctx, ident="mechanism-1", change="Preserve latest scope"):
    value = bind_evidence(mutation(), ctx)
    return {
        "mechanism_id": ident,
        "target_task_ids": ["1"],
        "protected_success_task_ids": [],
        "observed_deviation": "Submitted stale scope",
        "root_cause_hypothesis": f"Observable hypothesis: {change}",
        "evidence_refs": value["evidence_refs"],
        "expected_behavior_change": change,
        "alternative_explanations": ["Stochastic execution"],
        "regression_risk": "Additional steps",
        "repairability": "skill",
    }


def review(hs, same=False):
    return {
        "comparisons": [
            {
                "left": a["mechanism_id"],
                "right": b["mechanism_id"],
                "same_mechanism": same,
                "reason": "Explicit semantic judgment",
            }
            for a, b in combinations(hs, 2)
        ]
    }


def test_distinct_and_paraphrased_mechanisms():
    hs = [hypothesis(context(), str(i), f"Distinct intervention {i}") for i in range(3)]
    validate_analysis({"hypotheses": hs, "insufficient_evidence_reason": ""}, context())
    assert (
        len(select_distinct_hypotheses(hs, review(hs), 3)["assigned_hypotheses"]) == 3
    )
    result = select_distinct_hypotheses(hs, review(hs, True), 3)
    assert len(result["assigned_hypotheses"]) == 1
    assert len(result["excluded_hypotheses"]) == 2
    uncertain = select_distinct_hypotheses(hs, review(hs, None), 3)
    assert len(uncertain["assigned_hypotheses"]) == 1
    assert uncertain["excluded_hypotheses"][0]["status"] == "uncertain_deferred"
    assert uncertain["semantic_distinctness_proven"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("task_id", "H-private"),
        ("seed", 999),
        ("projected_message_index", 999),
        ("message_sha256", "fake"),
    ],
)
def test_bad_refs_rejected(field, value):
    ctx = context()
    h = hypothesis(ctx)
    h["evidence_refs"][0][field] = value
    with pytest.raises(ValueError):
        validate_analysis({"hypotheses": [h], "insufficient_evidence_reason": ""}, ctx)


def test_wrong_labels_payload_duplicates_and_noop():
    ctx = context()
    h = hypothesis(ctx)
    for edit in (
        {"target_task_ids": ["2"]},
        {"skill": {}},
        {"protected_success_task_ids": ["1"]},
    ):
        with pytest.raises(ValueError):
            validate_analysis(
                {"hypotheses": [{**h, **edit}], "insufficient_evidence_reason": ""}, ctx
            )
    with pytest.raises(ValueError, match="Duplicate"):
        validate_analysis(
            {"hypotheses": [h, h], "insufficient_evidence_reason": ""}, ctx
        )
    result = analysis_outcome(
        lambda: {
            "hypotheses": [],
            "insufficient_evidence_reason": "No supported repair",
        },
        ctx,
    )
    assert result["status"] == "VALID"
    with pytest.raises(TimeoutError):
        analysis_outcome(lambda: (_ for _ in ()).throw(TimeoutError()), ctx)


def test_mutator_cannot_replace_assignment():
    ctx = context()
    h = hypothesis(ctx)
    ctx["assigned_hypothesis"] = h
    m = bind_evidence(mutation(), ctx)
    m["target_cluster_id"] = h["mechanism_id"]
    m["root_cause_hypothesis"] = h["root_cause_hypothesis"]
    validate_assigned_mutation(m, ctx)
    m["root_cause_hypothesis"] = "Another root cause"
    with pytest.raises(ValueError, match="assigned"):
        validate_assigned_mutation(m, ctx)
    validate_assigned_mutation(mutation(operation="no_op"), ctx)


class AnalystProviders(FakeProviders):
    def __init__(self, count=1, same=False, invalid=False):
        super().__init__()
        self.count = count
        self.same = same
        self.invalid = invalid
        self.contexts = []

    def analyze_service_failures(self, ctx):
        self.calls.append("analyst")
        self.contexts.append(deepcopy(ctx))
        hs = [
            hypothesis(ctx, f"mechanism-{i}", f"Specific intervention {i}")
            for i in range(self.count)
        ]
        if self.invalid:
            hs[0]["target_task_ids"] = ["4"]
        return {
            "hypotheses": hs,
            "insufficient_evidence_reason": "Insufficient evidence" if not hs else "",
        }

    def deduplicate_failure_hypotheses(self, ctx):
        self.calls.append("diversity")
        return review(ctx["hypotheses"], self.same)

    def propose_skill_mutation(self, ctx):
        self.calls.append("mutate")
        self.contexts.append(deepcopy(ctx))
        assert "proposal_bias" not in ctx
        h = ctx["assigned_hypothesis"]
        # NO_OP exercises assignment flow without pretending every hypothesis deserves a Skill.
        m = bind_evidence(mutation(operation="no_op"), ctx)
        m["target_cluster_id"] = h["mechanism_id"]
        return m


def settings():
    p = deepcopy(DEFAULT_V2)
    p["algorithm_version"] = ANALYST_VERSION
    p["evaluation"]["promotion_protocol"] = "v_primary"
    p["statistical_gate"]["method"] = "task_block_bootstrap"
    p["service_evolution"]["candidate_error_policy"] = "reject_candidate"
    return p


@pytest.mark.parametrize(
    "count,same,expected", [(3, False, 3), (3, True, 1), (1, False, 1), (0, False, 0)]
)
def test_generation_assignment_count_and_e_only(tmp_path, count, same, expected):
    p = AnalystProviders(count, same)
    result, _, _ = run(
        tmp_path, provider=p, policy=settings(), validation=True, generations=2
    )
    assert p.calls.count("analyst") == 2
    assert p.calls.count("mutate") == expected * 2
    assert not result.service.skills
    for ctx in p.contexts:
        assert {r["task"]["task_id"] for r in ctx["task_interactions"]} == {
            "1",
            "2",
            "3",
        }
        assert "secret-hidden" not in str(ctx)
    for g in result.generations:
        phase = g["service_phase"]
        assert len(phase["hypothesis_selection"]["assigned_hypotheses"]) == expected
        assert len(phase["candidates"]) == expected
        assert not phase["validation_evaluated"]


def test_invalid_analyst_safe_exit_without_blind_mutation(tmp_path):
    p = AnalystProviders(invalid=True)
    result, _, _ = run(tmp_path, provider=p, policy=settings(), validation=True)
    assert "mutate" not in p.calls
    assert (
        result.generations[0]["service_phase"]["failure_analysis"]["status"]
        == "REJECTED"
    )


def test_resume_reuses_frozen_analyst_and_mutator(tmp_path):
    p = AnalystProviders()
    run(tmp_path, provider=p, policy=settings(), validation=True, generations=2)
    before = list(p.calls)
    run(tmp_path, provider=p, policy=settings(), validation=True, generations=2)
    assert p.calls == before
    with pytest.raises(ValueError):
        run(
            tmp_path,
            provider=p,
            policy=settings(),
            validation=True,
            manifest_sha="other",
        )


class RepairProviders(AnalystProviders):
    def propose_skill_mutation(self, ctx):
        self.calls.append("mutate")
        self.contexts.append(deepcopy(ctx))
        h = ctx["assigned_hypothesis"]
        m = bind_evidence(
            mutation(
                f"Use public cancel_reservation only under policy; repair {h['mechanism_id']}",
                family=h["mechanism_id"],
            ),
            ctx,
        )
        m.update(
            target_cluster_id=h["mechanism_id"],
            root_cause_hypothesis=h["root_cause_hypothesis"],
        )
        m["substantive_delta_from_prior"] = h["expected_behavior_change"]
        return m


def test_three_repairs_reach_unchanged_screen_full_e_and_v(tmp_path):
    p = RepairProviders(count=3)
    result, _, _runner = run(tmp_path, provider=p, policy=settings(), validation=True)
    board = result.generations[0]["service_phase"]["candidates"]
    assert len(board) == 3 and p.calls.count("skill_validator") == 3
    assert all(r["screen"] and r["effect"] and r["gate"] for r in board)
    assert any(g["panel"] == "V" for r in board for g in r["gate"]["opponents"])
    assert (
        not result.service.skills
    )  # Small V evidence remains inconclusive, never forced promotion.


def test_versioned_config_preserves_panels_thresholds_and_runtime():
    from pathlib import Path

    import yaml

    from evotau.alternating_manifest import AlternatingManifest

    root = Path(__file__).resolve().parents[1]
    old = yaml.safe_load(
        (
            root / "configs/airline-gateway-qwen37plus-unbounded-resume-p2.yaml"
        ).read_text()
    )
    new = yaml.safe_load(
        (
            root / "configs/airline-failure-analyst-qwen37plus-official-dsflash-p2.yaml"
        ).read_text()
    )
    a, b = old["experiment"], new["experiment"]
    for k in (
        "task_selection",
        "models",
        "model_args",
        "max_steps",
        "max_parallel_episodes",
        "evolution_fitness_seed",
        "customer_candidates",
    ):
        assert a[k] == b[k]
    for k in (
        "evaluation",
        "statistical_gate",
        "service_evolution",
        "activator",
        "mutation_context",
    ):
        assert a["skill_evolution_v2"][k] == b["skill_evolution_v2"][k]
    x, y = AlternatingManifest.from_mapping(old), AlternatingManifest.from_mapping(new)
    assert x.sha256 != y.sha256
    assert x.output_path != y.output_path


@pytest.mark.parametrize(
    "stage_name",
    [
        "service_failure_analysis",
        "service_hypothesis_assignment",
        "service_proposals-g0000-direct-0",
    ],
)
def test_interrupted_stage_resume_preserves_completed_requests(tmp_path, stage_name):
    p = AnalystProviders()
    policy = settings()
    from test_skill_evolution_v2 import FakeRunner

    runner = FakeRunner()
    with pytest.raises(RuntimeError, match="deterministic interruption"):
        run(
            tmp_path,
            provider=p,
            runner=runner,
            policy=policy,
            validation=True,
            interrupt=stage_name,
        )
    previous_analyst = p.calls.count("analyst")
    previous_mutator = p.calls.count("mutate")
    run(tmp_path, provider=p, runner=runner, policy=policy, validation=True)
    assert p.calls.count("analyst") == previous_analyst == 1
    assert p.calls.count("mutate") == 1
    if previous_mutator:
        assert p.calls.count("mutate") == previous_mutator


def test_actual_provider_chain_offline_budget_journal_g2(tmp_path, monkeypatch):
    from evotau.alternating import LLMAlternatingEvolvers
    from evotau.budget import RequestBudget
    from evotau.evolution_candidates import V2Providers

    scripted = RepairProviders(count=1)
    calls = []

    def dispatch(model, args, prompt, ctx, *, call_name):
        calls.append(call_name)
        if call_name == "evotau_customer_evolver":
            return scripted.customers(ctx, ctx["requested_candidates"])
        if call_name == "evotau_customer_semantic_validator":
            return {**scripted.validate_customer(ctx), "reason": "Task faithful"}
        if call_name == "evotau_service_failure_analyst":
            return scripted.analyze_service_failures(ctx)
        if call_name == "evotau_service_skill_mutator":
            return scripted.propose_skill_mutation(ctx)
        if call_name == "evotau_skill_semantic_validator":
            assert (
                "Domain-specific Airline policies and public tool names are allowed"
                in prompt
            )
            return {**scripted.validate_skill(ctx), "reason": "Public operational rule"}
        raise AssertionError(call_name)

    budget = RequestBudget(20)

    def counted_dispatch(model, args, prompt, ctx, *, call_name):
        return budget.dispatch_external_call(
            model=model,
            call_name=call_name,
            dispatch=lambda: dispatch(model, args, prompt, ctx, call_name=call_name),
        )

    monkeypatch.setattr(
        LLMAlternatingEvolvers, "_json_call", staticmethod(counted_dispatch)
    )
    providers = V2Providers(
        LLMAlternatingEvolvers(
            model="offline/mock",
            model_args={},
            request_budget=budget,
            output_directory=tmp_path,
        )
    )
    policy = settings()
    policy["service_evolution"]["crossover"] = False
    result, _, runner = run(
        tmp_path, provider=providers, policy=policy, validation=True, generations=2
    )
    assert calls.count("evotau_service_failure_analyst") == 2
    assert calls.count("evotau_service_skill_mutator") == 2
    assert budget.snapshot().attempts == len(calls)
    before = list(calls), budget.snapshot().attempts
    run(
        tmp_path,
        provider=providers,
        runner=runner,
        policy=policy,
        validation=True,
        generations=2,
    )
    assert before == (calls, budget.snapshot().attempts)
    assert all(g["service_phase"]["candidates"] for g in result.generations)


def test_public_tool_validator_and_fixture_rejection_use_same_flags(tmp_path):
    from types import SimpleNamespace

    from evotau.evolution_candidates import V2Providers

    p = V2Providers(SimpleNamespace())
    captured = []

    def call(prompt, ctx, name):
        captured.append(prompt)
        valid = "fixture-reservation" not in str(ctx)
        return {
            "reusable": valid,
            "policy_subordinate": True,
            "no_task_entities": valid,
            "reason": "fixture leakage" if not valid else "public tool",
        }

    p.call = call
    valid = p.validate_skill(
        {
            "algorithm_version": ANALYST_VERSION,
            "guidance": "Use public cancel_reservation under policy",
        }
    )
    invalid = p.validate_skill(
        {
            "algorithm_version": ANALYST_VERSION,
            "guidance": "Cancel fixture-reservation exactly",
        }
    )
    assert valid["reusable"] and not invalid["no_task_entities"]
    assert "never effectiveness" in captured[0]


def test_replay_default_is_offline_and_execution_resumes(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    script = (
        Path(__file__).resolve().parents[1]
        / "experiments/execution/replay-airline-skill-generation.py"
    )
    spec = importlib.util.spec_from_file_location("analyst_replay", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ctx = {**context(), "generation": 0}
    from evotau.tau_provenance import sha256_json

    bundle = {
        "context": ctx,
        "context_sha256": sha256_json(ctx),
        "e_task_ids": ["1", "2"],
        "model": "offline/mock",
        "request_args": {},
    }
    calls = []

    class Provider:
        def __init__(self, unused):
            pass

        def propose_skill_mutation(self, ctx):
            calls.append("mutator")
            return bind_evidence(mutation(operation="no_op"), ctx)

        def analyze_service_failures(self, ctx):
            calls.append("analyst")
            return {
                "hypotheses": [
                    hypothesis(ctx, str(i), f"Intervention {i}") for i in range(3)
                ],
                "insufficient_evidence_reason": "",
            }

        def deduplicate_failure_hypotheses(self, ctx):
            calls.append("diversity")
            return review(ctx["hypotheses"])

    monkeypatch.setattr(module, "V2Providers", Provider)
    plan = module.replay(bundle, tmp_path)
    assert not plan["real_requests_started"] and not calls
    result = module.replay(bundle, tmp_path, True)
    assert len(result["old"]) == len(result["new"]) == 3
    assert calls.count("mutator") == 6 and calls.count("analyst") == 1
    before = list(calls)
    assert module.replay(bundle, tmp_path, True) == result
    assert calls == before


def test_legacy_frozen_prompt_bytes_preserved():
    from evotau.evolution_candidates import DIRECT_MUTATOR_PROMPT
    from evotau.tau_provenance import sha256_json

    assert (
        sha256_json(DIRECT_MUTATOR_PROMPT)
        == "edb6ad7c2db025882f495d1aa2b3e1a259781d04c6a420b5f050bb11630cb554"
    )


def test_single_seed_config_only_changes_repetition_and_identity():
    from pathlib import Path

    import yaml

    from evotau.alternating_manifest import AlternatingManifest

    root = Path(__file__).resolve().parents[1]
    base = yaml.safe_load((root / 'configs/airline-failure-analyst-qwen37plus-official-dsflash-p2.yaml').read_text())
    single = yaml.safe_load((root / 'configs/airline-failure-analyst-single-seed-p2.yaml').read_text())
    old, new = deepcopy(base['experiment']), deepcopy(single['experiment'])
    assert new['skill_evolution_v2']['evaluation']['screen_seeds'] == [1]
    assert new['skill_evolution_v2']['evaluation']['gate_seeds'] == [1]
    assert new['evolution_fitness_seed'] == 1
    a, b = AlternatingManifest.from_mapping(base), AlternatingManifest.from_mapping(single)
    assert a.sha256 != b.sha256 and a.output_path != b.output_path
    for k in ('id', 'output_path', 'checkpoint_path', 'provider_provenance'):
        old.pop(k); new.pop(k)
    for k in ('screen_seeds', 'gate_seeds'):
        old['skill_evolution_v2']['evaluation'].pop(k)
        new['skill_evolution_v2']['evaluation'].pop(k)
    assert old == new  # All models, tasks, thresholds, native runtime and search intensity unchanged.
