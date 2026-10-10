"""Task-faithful Customer protocol: deterministic checks and offline G2 integration."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from test_skill_evolution_v2 import (
    FakeProviders,
    FakeRunner,
    bind_evidence,
    mutation,
    record,
    run,
)

from evotau.alternating import _context_episodes
from evotau.alternating_manifest import AlternatingManifest
from evotau.customer_diagnostic import run_customer_diagnostic
from evotau.customer_evolution import prepare_candidate
from evotau.customer_skills import PROTOCOL, CustomerSkill, validate_customer_policy
from evotau.customer_trajectory_validity import (
    assess_trajectories,
    paired_customer_feedback,
    validate_trajectory_report,
)
from evotau.evolution_candidates import EvolverSchemaError, V2Providers
from evotau.evolution_context import build_customer_evidence
from evotau.service_skills import ServiceSkillMemoryV2
from evotau.skill_evolution import propose_fresh_customer_v2
from evotau.skill_evolution_config import DEFAULT_V2
from evotau.tau_provenance import sha256_json

ROOT = Path(__file__).resolve().parents[1]


def skill(g=0):
    return {
        "schema_version": 1,
        "mechanism": "dependency clarification " + str(g),
        "trigger": "An existing multi-step request has an unresolved dependency "
        + str(g),
        "procedure": [
            "Describe the dependency between existing requests truthfully.",
            "Ask for the cost explanation before conditional confirmation.",
        ],
        "intensity": "medium",
        "stop_conditions": [
            "The original request is resolved.",
            "The scenario does not support the trigger.",
        ],
        "hypothesis": "The Service may lose a stated condition.",
        "evidence_refs": [],
    }


def policy():
    value = deepcopy(DEFAULT_V2)
    value["customer_evolution"] = validate_customer_policy({})
    value["service_evolution"].update(crossover=False)
    return value


def tasks():
    return {
        t: SimpleNamespace(
            id=t, user_scenario="secret-hidden-" + t, description="", user_tools=[]
        )
        for t in ("1", "2", "3")
    }


class SkillProviders(FakeProviders):
    def __init__(self, status="valid", static=True):
        super().__init__()
        self.status, self.static = status, static
        self.contexts = []

    def configure_customer(self, value):
        self.customer_policy = value["customer_evolution"]

    def customers(self, context, count):
        self.calls.append("customers")
        self.contexts.append(deepcopy(context))
        assert {r["task"]["task_id"] for r in context["task_interactions"]} == {
            "1",
            "2",
            "3",
        }
        assert "gold_target" not in json.dumps(context)
        return {
            "candidates": [
                skill(context.get("generation", 99) + i) for i in range(count)
            ]
        }

    def validate_customer(self, context):
        self.calls.append("customer_validator")
        return {
            **dict.fromkeys(
                (
                    "preserves_facts",
                    "preserves_objective",
                    "interaction_only",
                    "no_benchmark_leakage",
                ),
                self.static,
            ),
            "reason": "checked original scenario",
        }

    def validate_customer_trajectory(self, context):
        self.calls.append("trajectory_validator")
        return {
            "cells": [
                {
                    "task_id": c["task_id"],
                    "seed": c["seed"],
                    "status": self.status,
                    "reason": "Original conditions preserved."
                    if self.status == "valid"
                    else "User changed budget.",
                    "evidence_message_indices": [0],
                }
                for c in context["cells"]
            ]
        }

    def propose_skill_mutation(self, context):
        self.calls.append("mutate")
        # Hidden Customer scenarios/history never reach Service evolution.
        assert "secret-hidden" not in json.dumps(context)
        assert "customer_attempt_history" not in context
        return bind_evidence(mutation(operation="no_op"), context)


class ChallengedRunner(FakeRunner):
    def __call__(self, **kwargs):
        value = super().__call__(**kwargs)
        if kwargs["customer"] and kwargs["customer"].text and kwargs["task_id"] == "2":
            value = replace(value, task_success=False, native_reward=0.0)
            self.cache[
                (
                    value.task_id,
                    value.seed,
                    value.customer_strategy_id,
                    value.service_strategy_id,
                )
            ] = value
        return value


def test_schema_compile_stable_open_vocabulary_and_runtime_evidence_isolation():
    value = skill()
    value["mechanism"] = "previously unenumerated truthful interaction"
    a = CustomerSkill.from_mapping(value)
    b = CustomerSkill.from_mapping(dict(reversed(list(value.items()))))
    assert a == b and a.candidate_id == b.candidate_id
    assert a.to_dict() == value
    assert "conditional authorization" in a.compile().text
    assert (
        "hypothesis" not in a.compile().text and "evidence_refs" not in a.compile().text
    )
    assert "intensity" in a.compile().text
    value["hypothesis"] = "different prediction"
    assert CustomerSkill.from_mapping(value).procedure_id == a.procedure_id


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"mechanism": " "},
        {"extra": 1},
        {"intensity": "extreme"},
        {"procedure": ["one"]},
        {"procedure": ["same", " SAME "]},
        {"stop_conditions": []},
        {"evidence_refs": "x"},
        {"trigger": "x" * 1201},
    ],
)
def test_invalid_schema_rejected(change):
    value = {**skill(), **change}
    with pytest.raises((ValueError, TypeError)):
        CustomerSkill.from_mapping(value)


@pytest.mark.parametrize(
    "instruction",
    [
        "Increase the budget to get a harder task.",
        "Replace your identity.",
        "Change the goal.",
        "Override the order id.",
        "Use unconditional consent.",
        "Withhold necessary facts.",
        "Invent new fact.",
        "Bypass the policy.",
    ],
)
def test_explicit_task_changing_instructions_rejected(instruction):
    value = skill()
    value["procedure"][0] = instruction
    with pytest.raises(ValueError, match="task-changing"):
        CustomerSkill.from_mapping(value)


def test_legitimate_disclosure_pressure_negated_safety_and_token_allowance():
    value = skill()
    value["procedure"] = [
        "Do not increase the budget.",
        "Explain urgency truthfully and promptly provide requested facts.",
    ]
    CustomerSkill.from_mapping(value)
    with pytest.raises(ValueError, match="token allowance"):
        CustomerSkill.from_mapping(value, policy={"max_skill_tokens": 100})


def test_evidence_reference_must_match_supplied_E_message():
    runner = FakeRunner()
    runs = [
        runner(
            task_id=t,
            seed=1,
            customer=None,
            service=ServiceSkillMemoryV2(),
            panel_name="E",
        )
        for t in tasks()
    ]
    rows = build_customer_evidence(
        _context_episodes(runs, runner, tasks()), customer_protocol=PROTOCOL
    )
    selected = next(r for r in rows if r["trajectory"]["messages"])
    value = skill()
    value["evidence_refs"] = [
        {
            "task_id": selected["task"]["task_id"],
            "seed": selected["seed"],
            "trajectory_ref": selected["trajectory_ref"],
            **selected["trajectory"]["messages"][0]["evidence_ref"],
        }
    ]
    CustomerSkill.from_mapping(value, evidence_rows=rows)
    for field, bad in (("task_id", "H"), ("seed", 2), ("message_sha256", "forged")):
        other = deepcopy(value)
        other["evidence_refs"][0][field] = bad
        row, parsed = prepare_candidate(
            other, {"task_interactions": rows}, validate_customer_policy({})
        )
        assert parsed is None and row["candidate_validity"] == "invalid"


def test_success_complexity_priority_is_deterministic_and_retains_failed_control():
    rows = []
    for i in range(6):
        messages = [{"role": "user", "content": "Existing truthful request."}]
        messages += [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": str(n), "name": "update_reservation", "arguments": {}}
                ],
            }
            for n in range(i)
        ]
        rows.append(
            {
                "task": {"task_id": str(i), "user_scenario": "fixed", "user_tools": []},
                "seed": 1,
                "trajectory_ref": "original/" + str(i),
                "trajectory": {"messages": messages},
                "native_evaluation": {"task_success": i != 0},
            }
        )
    before = deepcopy(rows)
    built = build_customer_evidence(
        rows, representative_cases=4, customer_protocol=PROTOCOL
    )
    chosen = {r["task"]["task_id"] for r in built if r["representative_case"]}
    assert chosen == {"0", "3", "4", "5"}
    assert rows == before and len(built) == 6
    assert (
        build_customer_evidence(
            rows, representative_cases=4, customer_protocol=PROTOCOL
        )
        == built
    )
    # Explicit frozen evidence limits fail, never silently discard decisive messages.
    with pytest.raises(ValueError, match="evidence exceeds"):
        build_customer_evidence(
            rows, representative_cases=4, case_chars=1000, customer_protocol=PROTOCOL
        )


def review_context():
    return {
        "cells": [
            {
                "task_id": "1",
                "seed": 1,
                "user_scenario": "Fixed budget",
                "trajectory": {
                    "messages": [
                        {"role": "user", "content": "Truthful facts"},
                        {"role": "assistant", "content": "Checking"},
                    ]
                },
            }
        ]
    }


def report(status="valid"):
    return {
        "cells": [
            {
                "task_id": "1",
                "seed": 1,
                "status": status,
                "reason": "Evidence checked",
                "evidence_message_indices": [0],
            }
        ]
    }


@pytest.mark.parametrize("status", ["valid", "invalid", "uncertain"])
def test_trajectory_verdicts_and_original_customer_evidence(status):
    assert (
        assess_trajectories(review_context(), lambda ctx: report(status))["status"]
        == status
    )
    invalid_ref = report(status)
    invalid_ref["cells"][0]["evidence_message_indices"] = [1]
    with pytest.raises(ValueError, match="Customer messages"):
        validate_trajectory_report(invalid_ref, review_context())
    with pytest.raises(ValueError, match="omitted"):
        validate_trajectory_report({"cells": []}, review_context())


def test_missing_scenario_is_uncertain_without_calling_provider():
    context = review_context()
    context["cells"][0]["user_scenario"] = ""

    def unexpected(_):
        pytest.fail("No provider call for missing ground truth")

    assert assess_trajectories(context, unexpected)["status"] == "uncertain"


def test_paired_statistics_and_complete_validity_service_identity_requirements():
    before = [record("1", True), record("2", False), record("3", True)]
    after = [record("3", False), record("1", False), record("2", True)]
    validity = {
        "status": "valid",
        "cells": [
            {"task_id": t, "seed": 1, "status": "valid"} for t in ("1", "2", "3")
        ],
    }
    result = paired_customer_feedback(before, after, validity=validity)
    assert result["new_failures"] == [
        {"task_id": "1", "seed": 1},
        {"task_id": "3", "seed": 1},
    ]
    assert result["recovered_cells"] == [{"task_id": "2", "seed": 1}]
    assert result["paired_accuracy_delta"] == pytest.approx(-1 / 3)
    assert not result["causal_service_failure_confirmed"]
    with pytest.raises(ValueError, match="identical Service"):
        paired_customer_feedback(
            before,
            [replace(after[0], service_strategy_id="different"), *after[1:]],
            validity=validity,
        )
    with pytest.raises(ValueError, match="complete unique"):
        paired_customer_feedback(before, after[:-1], validity=validity)
    with pytest.raises(ValueError, match="validity must cover"):
        paired_customer_feedback(
            before, after, validity={"status": "valid", "cells": []}
        )


@pytest.mark.parametrize("status", ["invalid", "uncertain"])
def test_invalid_or_uncertain_rollout_blocks_whole_candidate_and_feedback_reaches_G1(
    tmp_path, status
):
    provider = SkillProviders(status)
    result, _, _ = run(
        tmp_path,
        provider=provider,
        runner=ChallengedRunner(),
        generations=2,
        policy=policy(),
    )
    for gen in result.generations:
        candidate = gen["customer_phase"]["candidates"][0]
        assert candidate["accuracy"] < gen["customer_phase"]["incumbent_accuracy"]
        assert candidate["candidate_validity"] == status
        assert not candidate["selected"] and not candidate["eligible_for_selection"]
        assert len(candidate["trajectory_validity"]["cells"]) == 3
    assert result.customer.text == "" and result.service.skills == ()
    history = provider.contexts[1]["customer_attempt_history"]
    assert history[0]["candidate_validity"] == status
    assert history[0]["trajectory_reasons"][0]["reason"] == "User changed budget."


def test_legal_drop_selected_tie_retains_incumbent_and_G2_resumes_without_calls(
    tmp_path,
):
    provider, runner = SkillProviders(), ChallengedRunner()
    result, _, _ = run(
        tmp_path, provider=provider, runner=runner, generations=2, policy=policy()
    )
    a, b = result.generations
    assert a["customer_phase"]["selected_accuracy"] == pytest.approx(1 / 3)
    assert a["customer_phase"]["candidates"][0]["selected"]
    assert b["customer_phase"]["selected_customer"] == "incumbent"
    assert b["customer_phase"]["candidates"][0]["candidate_validity"] == "valid"
    calls = provider.calls[:], runner.calls[:]
    repeated, _, _ = run(
        tmp_path, provider=provider, runner=runner, generations=2, policy=policy()
    )
    assert (
        repeated.customer == result.customer and (provider.calls, runner.calls) == calls
    )
    saved = json.loads((tmp_path / "checkpoint.json").read_text())
    assert saved["state"]["customer_protocol"] == PROTOCOL


def test_interrupted_trajectory_review_and_checkpoint_protocol_isolation(tmp_path):
    provider, runner = SkillProviders(), ChallengedRunner()
    stage = "customer-0-trajectory-validator-" + sha256_json(["1", 1])[:12]
    with pytest.raises(RuntimeError, match="deterministic interruption"):
        run(
            tmp_path, provider=provider, runner=runner, policy=policy(), interrupt=stage
        )
    assert provider.calls.count("trajectory_validator") == 1
    run(tmp_path, provider=provider, runner=runner, policy=policy())
    assert (
        provider.calls.count("customers") == 1
        and provider.calls.count("trajectory_validator") == 3
    )
    with pytest.raises(ValueError, match="checkpoint Customer protocol"):
        run(tmp_path, policy=DEFAULT_V2)


def test_static_semantic_rejection_prevents_rollouts_even_for_paraphrased_changes(
    tmp_path,
):
    provider = SkillProviders(static=False)
    result, _, runner = run(
        tmp_path, provider=provider, runner=ChallengedRunner(), policy=policy()
    )
    candidate = result.generations[0]["customer_phase"]["candidates"][0]
    assert candidate["candidate_validity"] == "invalid" and not candidate["episodes"]
    assert "trajectory_validator" not in provider.calls and len(runner.calls) == 3


def test_schema_error_generation_is_recorded_not_a_fabricated_score(tmp_path):
    provider = SkillProviders()

    def malformed(ctx, count):
        raise EvolverSchemaError("invalid candidate envelope")

    provider.customers = malformed
    result, _, _ = run(tmp_path, provider=provider, policy=policy())
    row = result.generations[0]["customer_phase"]["candidates"][0]
    assert row["candidate_validity"] == "invalid" and row.get("accuracy") is None
    assert result.customer.text == ""


def test_fresh_customer_new_protocol_E_only_validated_and_cached(tmp_path):
    provider, runner = SkillProviders(), ChallengedRunner()
    result, _, _ = run(tmp_path, provider=provider, runner=runner, policy=policy())
    args = {
        "tasks": tasks(),
        "runner": runner,
        "domain_policy": "policy",
        "output_directory": tmp_path,
        "manifest_sha256": "manifest",
        "customer_policy": policy()["customer_evolution"],
    }
    fresh = propose_fresh_customer_v2(result, provider, **args)
    assert fresh is not None and fresh != result.customer
    calls = provider.calls[:]
    assert (
        propose_fresh_customer_v2(result, provider, **args) == fresh
        and provider.calls == calls
    )
    doc = json.loads((tmp_path / "fresh-customer-proposal.json").read_text())
    assert doc["candidate_validity"] == "valid" and len(doc["episodes"]) == 3
    assert all(r["task_id"] in tasks() for r in doc["episodes"])


def test_fresh_uncertain_not_mislabeled_as_available(tmp_path):
    provider = SkillProviders("uncertain")
    result, _, runner = run(tmp_path, provider=provider, policy=policy())
    assert (
        propose_fresh_customer_v2(
            result,
            provider,
            tasks=tasks(),
            runner=runner,
            domain_policy="policy",
            output_directory=tmp_path,
            manifest_sha256="manifest",
            customer_policy=policy()["customer_evolution"],
        )
        is None
    )
    assert (
        json.loads((tmp_path / "fresh-customer-proposal.json").read_text())["status"]
        == "unavailable"
    )


def test_new_policy_and_manifest_opt_in_legacy_identity_preserved():
    raw = yaml.safe_load(
        (ROOT / "configs/airline-customer-skill-v1-e10-diagnostic.yaml").read_text()
    )
    new = AlternatingManifest.from_mapping(raw)
    assert (
        not new.run_validation
        and not new.run_heldout
        and len(new.evolution_task_ids) == 10
    )
    assert (
        json.loads(new.skill_evolution_v2_json)["customer_evolution"][
            "protocol_version"
        ]
        == PROTOCOL
    )
    raw["experiment"]["skill_evolution_v2"].pop("customer_evolution")
    legacy = AlternatingManifest.from_mapping(raw)
    assert "customer_evolution" not in json.loads(legacy.skill_evolution_v2_json)
    assert new.sha256 != legacy.sha256
    assert (
        new.role_models == legacy.role_models
        and new.role_model_args == legacy.role_model_args
    )


def test_customer_only_G2_frozen_service_and_resume_is_idempotent(tmp_path):
    provider, runner = SkillProviders(), ChallengedRunner()
    args = {
        "tasks": tasks(),
        "task_ids": tuple(tasks()),
        "runner": runner,
        "providers": provider,
        "service": ServiceSkillMemoryV2(),
        "policy": policy()["customer_evolution"],
        "generations": 2,
        "count": 1,
        "seed": 1,
        "concurrency": 2,
        "domain_policy": "policy",
        "output": tmp_path,
        "manifest_sha": "manifest",
    }
    result = run_customer_diagnostic(**args)
    assert result["status"] == "complete" and len(result["generations"]) == 2
    assert (
        "mutate" not in provider.calls
        and result["frozen_service"] == ServiceSkillMemoryV2().to_dict()
    )
    calls = provider.calls[:], runner.calls[:]
    assert (
        run_customer_diagnostic(**args) == result
        and (provider.calls, runner.calls) == calls
    )
    with pytest.raises(ValueError, match="frozen stage input changed"):
        run_customer_diagnostic(**{**args, "seed": 2})


def test_real_native_user_simulator_compiles_overlay_without_changing_task():
    import os

    from evotau.alternating_run import load_alternating_tasks
    from evotau.tau_adapter import build_phase0_orchestrator

    data = Path(
        os.environ.get(
            "TAU2_DATA_DIR", "/Users/spring/.cache/evotau/tau2-data-b7ea9074"
        )
    )
    if not (data / "tau2/domains/airline/tasks.json").exists():
        pytest.skip("pinned data unavailable")
    raw = yaml.safe_load(
        (ROOT / "configs/airline-customer-skill-v1-e10-diagnostic.yaml").read_text()
    )
    manifest = AlternatingManifest.from_mapping(raw)
    loaded = load_alternating_tasks(
        manifest, data, include_validation=False, include_heldout=False
    )
    task = loaded[manifest.evolution_task_ids[0]]
    original = task.model_dump(mode="json")
    orchestrator = build_phase0_orchestrator(
        task=task,
        domain="airline",
        agent_model="offline",
        customer_model="offline",
        seed=1,
        customer_strategy=CustomerSkill.from_mapping(skill()).compile(),
    )
    assert str(task.user_scenario) in orchestrator.user.system_prompt
    assert "Customer procedure" in orchestrator.user.system_prompt
    assert task.model_dump(mode="json") == original
    assert set(loaded) == set(manifest.evolution_task_ids)


def test_provider_structured_customer_and_trajectory_call_contract_without_network():
    provider = V2Providers(SimpleNamespace())
    provider.configure_customer({"customer_evolution": validate_customer_policy({})})
    calls = []

    def capture(prompt, context, name):
        calls.append((prompt, context, name))
        return {"candidates": [skill()]} if "evolver" in name else report()

    provider.call = capture
    assert provider.customers({"task_interactions": []}, 1) == {"candidates": [skill()]}
    provider.validate_customer_trajectory(review_context())
    assert [c[2] for c in calls] == [
        "evotau_customer_skill_evolver_v1",
        "evotau_customer_trajectory_validator_v1",
    ]
    assert "2–5" in calls[0][0] and "USER message indices" in calls[1][0]


def test_new_customer_does_not_bypass_service_screen_full_E_V_or_replay(tmp_path):
    provider = SkillProviders()
    provider.propose_skill_mutation = FakeProviders.propose_skill_mutation.__get__(
        provider
    )
    result, _, _ = run(
        tmp_path,
        provider=provider,
        runner=ChallengedRunner(),
        policy=policy(),
        validation=True,
    )
    board = result.generations[0]["service_phase"]["candidates"]
    winner = board[0]
    assert winner["screen"]["passed"]
    assert winner["effect"]["fail_to_pass"] == ["1"]
    assert any(g["panel"] == "V" for g in winner["gate"]["opponents"])
    assert set(result.generations[0]["service_phase"]["opponents_replayed"]) >= {
        "current",
        "native",
    }
    assert (
        winner["decision"] == "INCONCLUSIVE"
    )  # V has one task in this fixture, not eight.
    assert not result.service.skills


def test_exact_procedure_duplicates_rejected_and_history_bounded(tmp_path):
    provider = SkillProviders()

    def same(context, count):
        value = skill()
        value["hypothesis"] += str(context["generation"])
        return {"candidates": [value]}

    provider.customers = same
    settings = policy()
    settings["customer_evolution"]["max_history"] = 1
    result, _, _ = run(tmp_path, provider=provider, policy=settings, generations=3)
    for g in result.generations[1:]:
        assert g["customer_phase"]["candidates"][0]["pre_rollout_rejection"].startswith(
            "duplicate"
        )
        assert len(g["customer_attempt_history"]) == 1


def test_transport_error_not_scored_or_swallowed(tmp_path):
    provider = SkillProviders()

    def failed(context):
        raise TimeoutError("provider unavailable")

    provider.validate_customer_trajectory = failed
    with pytest.raises(TimeoutError, match="unavailable"):
        run(tmp_path, provider=provider, policy=policy())
    assert not (tmp_path / "generation-0000.json").exists()
    assert list((tmp_path / "evolution-v2").glob("*customer-candidate-0.json"))


@pytest.mark.parametrize(
    "fault", [TimeoutError("submitted timeout"), KeyboardInterrupt()]
)
def test_new_customer_unresolved_provider_request_is_not_blindly_resent(
    tmp_path, monkeypatch, fault
):
    from test_evolver_recovery import provider as recorded_provider

    from evotau.evolver_recovery import UnknownRequestState

    p, budget, calls = recorded_provider(tmp_path, monkeypatch, [fault])
    p.recovery = None  # Customer-only launcher does not enable Service format recovery.
    p.configure_customer({"customer_evolution": validate_customer_policy({})})
    with pytest.raises(type(fault)):
        p.customers({"task_interactions": []}, 1)
    with pytest.raises(UnknownRequestState):
        p.customers({"task_interactions": []}, 1)
    assert len(calls) == budget.snapshot().attempts == 1


def test_recorded_customer_response_reused_before_stage_publication(
    tmp_path, monkeypatch
):
    from test_evolver_recovery import provider as recorded_provider

    p, budget, calls = recorded_provider(
        tmp_path, monkeypatch, [{"candidates": [skill()]}]
    )
    p.recovery = None
    p.configure_customer({"customer_evolution": validate_customer_policy({})})
    context = {"task_interactions": []}
    assert p.customers(context, 1) == {"candidates": [skill()]}
    assert p.customers(context, 1) == {"candidates": [skill()]}
    assert len(calls) == budget.snapshot().attempts == 1
    # Frozen protocol/input/prompt changes cannot import this old Evolver result.
    p.call = lambda prompt, context, name: {"candidates": []}
    with pytest.raises(ValueError, match="candidate count"):
        p.customers(context, 1)


def test_static_or_trajectory_schema_error_is_uncertain_not_reward(tmp_path):
    provider = SkillProviders()

    def malformed(ctx):
        raise EvolverSchemaError("returned illegal status")

    provider.validate_customer_trajectory = malformed
    result, _, _ = run(
        tmp_path, provider=provider, runner=ChallengedRunner(), policy=policy()
    )
    row = result.generations[0]["customer_phase"]["candidates"][0]
    assert row["candidate_validity"] == "uncertain" and not row["selected"]
    assert all(
        "schema error" in c["reason"] for c in row["trajectory_validity"]["cells"]
    )


def test_structurally_invalid_empty_strategy_is_never_marked_selected(tmp_path):
    provider = SkillProviders()
    provider.customers = lambda ctx, count: {
        "candidates": [{**skill(), "intensity": "illegal"}]
    }
    result, _, _ = run(tmp_path, provider=provider, policy=policy())
    row = result.generations[0]["customer_phase"]["candidates"][0]
    assert row["candidate_validity"] == "invalid" and not row["selected"]
    assert row["accuracy"] is None and not row["episodes"]


def test_diagnostic_dry_run_cannot_retrieve_keys_or_start_provider(monkeypatch, capsys):
    import importlib.util
    import os
    import sys

    data = Path(
        os.environ.get(
            "TAU2_DATA_DIR", "/Users/spring/.cache/evotau/tau2-data-b7ea9074"
        )
    )
    if not (data / "tau2/domains/airline/tasks.json").exists():
        pytest.skip("pinned data unavailable")
    script = ROOT / "experiments/execution/run-customer-skill-diagnostic.py"
    spec = importlib.util.spec_from_file_location("customer_diagnostic_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def forbidden(*args, **kwargs):
        pytest.fail("dry-run reached execute/credential/provider path")

    monkeypatch.setattr(module, "execute", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--config",
            str(ROOT / "configs/airline-customer-skill-v1-e10-diagnostic.yaml"),
            "--tau2-data-dir",
            str(data),
        ],
    )
    module.main()
    doc = json.loads(capsys.readouterr().out)
    assert (
        not doc["real_requests_started"]
        and len(doc["E"]) == 10
        and doc["service_frozen"]
    )
