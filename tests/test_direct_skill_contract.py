"""Direct evolution contract, explicit budgets and absence of a diagnosis veto; offline."""

from copy import deepcopy

import pytest
from test_evolution_context import row
from test_skill_evolution_v2 import (
    FakeProviders,
    FakeRunner,
    bind_evidence,
    mutation,
    run,
)

from evotau.evolution_candidates import (
    V2Providers,
    validate_direct_evidence,
    validate_mutation,
)
from evotau.evolution_context import (
    build_service_mutation_evidence,
    enforce_mutation_context_budget,
)
from evotau.skill_evolution_config import DEFAULT_V2, freeze_v2_policy


def context():
    return {
        "task_interactions": build_service_mutation_evidence(
            [row("1", False), row("2", True)]
        )
    }


def test_direct_candidate_without_diagnosis_and_protected_original_evidence():
    ctx = context()
    value = mutation()
    value["protected_success_task_ids"] = ["2"]
    value = bind_evidence(value, ctx)
    validate_mutation(value)
    assert validate_direct_evidence(value, ctx) == value
    assert not hasattr(V2Providers, "diagnose")
    assert not hasattr(V2Providers, "mutate")


@pytest.mark.parametrize(
    "kind",
    [
        "outside_E",
        "wrong_failure",
        "wrong_success",
        "forged_hash",
        "forged_index",
        "forged_seed",
        "forged_path",
        "missing_ref",
        "extra_ref_field",
        "invented_operation",
    ],
)
def test_bad_candidate_is_rejected_without_rewriting(kind):
    ctx = context()
    value = bind_evidence(mutation(), ctx)
    if kind == "outside_E":
        value["evidence_task_ids"] = ["H99"]
    elif kind == "wrong_failure":
        value["evidence_task_ids"] = ["2"]
    elif kind == "wrong_success":
        value["protected_success_task_ids"] = ["1"]
    elif kind == "missing_ref":
        value["evidence_refs"] = []
    elif kind == "invented_operation":
        value["operation"] = "fix_runtime_protocol"
    else:
        field, change = {
            "forged_hash": ("message_sha256", "invented"),
            "forged_index": ("projected_message_index", 999),
            "forged_seed": ("seed", 999),
            "forged_path": ("trajectory_ref", "different.json"),
            "extra_ref_field": ("hidden", "forbidden"),
        }[kind]
        value["evidence_refs"][0][field] = change
    before = deepcopy(value)
    with pytest.raises((ValueError, TypeError)):
        validate_mutation(value)
        validate_direct_evidence(value, ctx)
    assert value == before


def test_failures_can_legitimately_yield_only_no_ops(tmp_path):
    class Insufficient(FakeProviders):
        def propose_skill_mutation(self, ctx):
            self.calls.append("direct")
            return bind_evidence(mutation(operation="no_op"), ctx)

    result, provider, runner = run(tmp_path, provider=Insufficient(), generations=2)
    assert provider.calls.count("direct") == 6
    assert all(not g["service_phase"]["accepted"] for g in result.generations)
    assert all(
        c["pre_rollout_rejection"] == "no_op" and c["gate"] is None
        for g in result.generations
        for c in g["service_phase"]["candidates"]
    )
    assert not any("service_screen" in key for key in runner.calls)
    assert not result.generations[-1]["service_after"]["skill_count"]


def test_legacy_budget_is_explicit_product_and_algorithm_remains_read_only():
    roles = {"agent": "offline"}
    args = {"agent": {"temperature": 0.0}}
    frozen = freeze_v2_policy(
        {
            "service_evolution": {
                "candidates_per_cluster": 3,
                "max_clusters_per_generation": 2,
            }
        },
        roles,
        args,
    )
    assert frozen["algorithm_version"] == "diagnoser_v2"
    assert frozen["service_evolution"]["candidates_per_generation"] == 6
    assert frozen["legacy_candidate_budget_migration"]["candidates_per_generation"] == 6
    with pytest.raises(ValueError, match="mix"):
        freeze_v2_policy(
            {
                "service_evolution": {
                    "candidates_per_cluster": 3,
                    "candidates_per_generation": 2,
                }
            },
            roles,
            args,
        )


def test_legacy_algorithm_cannot_dispatch(tmp_path):
    policy = deepcopy(DEFAULT_V2)
    policy["algorithm_version"] = "diagnoser_v2"
    provider = FakeProviders()
    runner = FakeRunner()
    with pytest.raises(ValueError, match="read-only"):
        run(tmp_path, policy=policy, provider=provider, runner=runner)
    assert not provider.calls and not runner.calls


def test_evidence_expansion_is_controlled_and_budget_errors_do_not_drop_facts():
    rows = [row(str(i), i >= 7) for i in range(20)]
    source = deepcopy(rows)
    expanded = build_service_mutation_evidence(rows, representative_cases=8)
    assert len(expanded) == 20 and sum(r["representative_case"] for r in expanded) == 8
    ctx = {"task_interactions": expanded}
    before = deepcopy(ctx)
    with pytest.raises(ValueError, match="allowance"):
        enforce_mutation_context_budget(ctx, "direct", 10)
    assert rows == source and ctx == before


@pytest.mark.parametrize("mode", ["fail_fast", "full_audit"])
def test_V_rejection_stops_only_remaining_opponents_and_resume_is_exact(tmp_path, mode):
    class VRegression(FakeRunner):
        def __call__(self, **kwargs):
            from dataclasses import replace

            value = super().__call__(**kwargs)
            if kwargs["task_id"] == "4" and kwargs["service"].skills:
                value = replace(value, task_success=False, native_reward=0.0)
            return value

    policy = deepcopy(DEFAULT_V2)
    policy["service_evolution"].update(candidates_per_generation=1, crossover=False)
    policy["evaluation"]["v_gate_mode"] = mode
    result, provider, runner = run(
        tmp_path, validation=True, policy=policy, runner=VRegression()
    )
    candidate = result.generations[0]["service_phase"]["candidates"][0]
    gate = candidate["gate"]
    assert gate["verdict"] == "REJECTED"
    assert gate["opponents"][0]["verdict"] == "REJECTED"
    skipped = [r for r in gate["opponents"] if r["verdict"] == "NOT_EVALUATED"]
    assert bool(skipped) == (mode == "fail_fast")
    assert gate["risk_profile_complete"] == (mode == "full_audit")
    phase = result.generations[0]["service_phase"]
    assert phase["opponents_replayed"] == (
        ["current"] if mode == "fail_fast" else phase["opponents_planned"]
    )
    assert gate["gate_looks"] == len(gate["opponents"]) + 1
    assert not result.generations[0]["service_after"]["skill_count"]
    count = len(provider.calls), len(runner.calls)
    resumed, _, _ = run(
        tmp_path, validation=True, policy=policy, provider=provider, runner=runner
    )
    assert resumed.generations == result.generations
    assert count == (len(provider.calls), len(runner.calls))


def test_new_prompt_does_not_reuse_legacy_evolver_cache(tmp_path, monkeypatch):
    import json

    from evotau.alternating import LLMAlternatingEvolvers
    from evotau.tau_provenance import sha256_json

    ctx = context()
    legacy = tmp_path / "evolver-calls" / "old-mutator"
    legacy.mkdir(parents=True)
    response = {"legacy_output": True}
    (legacy / "input.json").write_text(
        json.dumps(
            {
                "call_name": "evotau_service_skill_mutator",
                "model": "offline",
                "system_prompt": "Legacy mutator requiring root_cause_cluster",
                "request_args": {},
                "input_sha256": sha256_json(ctx),
                "context": ctx,
            }
        )
    )
    (legacy / "output.json").write_text(
        json.dumps(
            {
                "response": response,
                "response_sha256": sha256_json(response),
            }
        )
    )
    frozen = {p.name: p.read_bytes() for p in legacy.iterdir()}
    calls = []
    value = bind_evidence(mutation(operation="no_op"), ctx)

    def local(*args, **kwargs):
        calls.append(kwargs["call_name"])
        return value

    monkeypatch.setattr(LLMAlternatingEvolvers, "_json_call", staticmethod(local))
    provider = V2Providers(
        LLMAlternatingEvolvers(
            model="offline", model_args={}, output_directory=tmp_path
        )
    )
    assert provider.propose_skill_mutation(ctx) == value
    assert provider.propose_skill_mutation(ctx) == value
    assert calls == ["evotau_service_skill_mutator"]
    assert all((legacy / key).read_bytes() == before for key, before in frozen.items())


def test_archived_diagnoser_console_remains_readable_without_provider_calls(
    monkeypatch,
):
    from pathlib import Path

    from fastapi.testclient import TestClient

    from evotau.alternating import LLMAlternatingEvolvers
    from evotau.web.app import create_app

    def forbidden(*args, **kwargs):
        raise AssertionError("Console must never invoke an Evolver")

    monkeypatch.setattr(LLMAlternatingEvolvers, "_json_call", staticmethod(forbidden))
    project = (
        Path(__file__).resolve().parents[1]
        / "experiments/results/skill-evolution-v2-offline-native-20261007-final"
    )
    client = TestClient(create_app(project_root=project))
    page = client.get("/runs/offline-v2-native/evolution")
    assert page.status_code == 200, page.text
    assert "Legacy Failure Diagnosis" in page.text
    assert "Mutation Lineage" in page.text


def test_new_config_has_explicit_direct_version_and_disjoint_fixed_panels():
    from pathlib import Path

    import yaml

    from evotau.alternating_manifest import AlternatingManifest

    path = (
        Path(__file__).resolve().parents[1]
        / "configs/v2-direct-skill-qwen37plus-inferai-v4pro-e20-v3-h5-g2-p1.yaml"
    )
    manifest = AlternatingManifest.from_mapping(yaml.safe_load(path.read_text()))
    assert len(manifest.evolution_task_ids) == 20
    assert len(manifest.validation_task_ids) == 3
    assert len(manifest.heldout_task_ids) == 5
    assert manifest.generations == 2 and manifest.max_parallel_episodes == 1
    assert manifest.request_budget_cap == 100000
    import json

    policy = json.loads(manifest.skill_evolution_v2_json)
    assert policy["algorithm_version"] == "direct_skill_evolution_v1"
    assert policy["service_evolution"]["candidates_per_generation"] == 3
    assert policy["evaluation"]["v_gate_mode"] == "fail_fast"
    assert "legacy_candidate_budget_migration" not in policy


def test_failure_reference_cannot_be_taken_from_passing_seed_of_same_task():
    failed = row("1", False)
    passed = row("1", True)
    passed["seed"] = 2
    ctx = {"task_interactions": build_service_mutation_evidence([failed, passed])}
    value = mutation()
    source = ctx["task_interactions"][1]
    value["evidence_refs"] = [
        {
            "task_id": "1",
            "seed": 2,
            "trajectory_ref": source["trajectory_ref"],
            **source["trajectory"]["messages"][0]["evidence_ref"],
        }
    ]
    with pytest.raises(ValueError, match="cell outcome"):
        validate_direct_evidence(value, ctx)
