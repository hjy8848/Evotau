"""Frozen-release fault injection; no network requests or H content read."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from test_skill_evolution_v2 import FakeProviders, FakeRunner, mutation, run

from evotau.alternating import LLMAlternatingEvolvers
from evotau.evolution_candidates import EvolverSchemaError, V2Providers
from evotau.evolution_history import stagnation_state
from evotau.release_recovery import (
    authorize_schema_retry,
    frozen_run_lock,
    reconcile_interrupted_requests,
)
from evotau.skill_evolution import (
    _evolution_parent,
    propose_fresh_customer_v2,
    run_v2_endpoint_evaluation,
)
from evotau.tau_provenance import sha256_json


def test_gen1_with_screen_rejection_has_nullable_gate(tmp_path):
    class Rejected(FakeProviders):
        def mutate(self, context):
            return mutation("Break protected passing behavior.")

    result, _, _ = run(tmp_path, provider=Rejected(), generations=2)
    assert all(not g["service_phase"]["accepted"] for g in result.generations)
    assert (
        result.generations[1]["service_phase"]["exploration"][
            "no_promotion_generations"
        ]
        == 1
    )
    rows = [
        {
            "effect": {"fail_to_fail": ["1"]},
            "mutation": {"semantic_family": family},
            "gate": None,
        }
        for family in ("x", "y")
    ]
    assert stagnation_state([], rows, 2)["uncertain_improvement"] is False


def test_archive_parent_keeps_E_effects_and_strips_V_cells():
    entry = {
        "mutation_id": "m",
        "mutation": mutation(),
        "effect": {"fail_to_pass": ["66"]},
        "full_gate_episodes": [{"task_id": "44", "secret": "V"}],
        "gate": {"V": "secret"},
    }
    parent = _evolution_parent(entry)
    assert "secret" not in json.dumps(parent)
    assert parent["effect"]["fail_to_pass"] == ["66"]
    parent["effect"]["fail_to_pass"].append("98")
    assert entry["effect"]["fail_to_pass"] == ["66"]


def test_schema_failure_requires_explicit_bound_retry_and_keeps_raw_output(
    tmp_path, monkeypatch
):
    (tmp_path / "manifest.json").write_text(json.dumps({"manifest_sha256": "manifest"}))
    responses = [{"clusters": [{"invented": True}]}, {"clusters": []}]
    calls = []

    def reply(*args, **kwargs):
        calls.append(1)
        return responses.pop(0)

    monkeypatch.setattr(LLMAlternatingEvolvers, "_json_call", staticmethod(reply))
    provider = V2Providers(
        LLMAlternatingEvolvers(
            model="offline", model_args={}, output_directory=tmp_path
        )
    )
    context = {"task_interactions": []}
    with pytest.raises(EvolverSchemaError) as raised:
        provider.diagnose(context)
    assert raised.value.diagnostics_ref.endswith("schema-error.json")
    directory = provider.last_call_directory
    frozen = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(EvolverSchemaError):
        provider.diagnose(context)
    assert len(calls) == 1
    authorize_schema_retry(
        tmp_path,
        directory.name,
        "operator reviewed invalid schema; resend unchanged input once",
    )
    assert provider.diagnose(context) == {"clusters": []}
    assert provider.diagnose(context) == {"clusters": []}
    assert len(calls) == 2
    assert all(
        (directory / name).read_bytes() == value for name, value in frozen.items()
    )
    with pytest.raises(FileExistsError):
        authorize_schema_retry(tmp_path, directory.name, "duplicate authorization")


def test_wrong_diagnosis_labels_fail_before_stage_freeze(tmp_path, monkeypatch):
    from evotau.evolution_candidates import DIAGNOSER_PROMPT

    row = {
        "cluster_id": "scope",
        "root_cause": "scope",
        "risk": "risk",
        "evidence_task_ids": ["66"],
        "protected_success_task_ids": [],
        "recommended_surface": "skill",
        "recommended_mutation_types": ["add"],
    }
    monkeypatch.setattr(
        LLMAlternatingEvolvers,
        "_json_call",
        staticmethod(lambda *a, **kw: {"clusters": [row]}),
    )
    providers = V2Providers(
        LLMAlternatingEvolvers(
            model="offline", model_args={}, output_directory=tmp_path
        )
    )
    with pytest.raises(EvolverSchemaError, match="labels disagree"):
        providers.diagnose(
            {
                "task_interactions": [{"task": {"task_id": "66"}}],
                "current_outcomes": [{"task_id": "66", "task_success": True}],
            }
        )
    assert DIAGNOSER_PROMPT  # Real contract supplied to the recorded call.


@pytest.mark.parametrize("kind", ["rejected", "duplicate", "valid"])
def test_fresh_outcome_finalizes_native_H_without_fake_fresh_and_replays(
    tmp_path, kind
):
    result, _, runner = run(tmp_path)

    class Fresh(FakeProviders):
        def customers(self, ctx, count):
            value = super().customers(ctx, count)
            if kind != "duplicate":
                value["candidates"][0]["strategy"] = (
                    "Request a concise grounded correction after clarification."
                )
            return value

        def validate_customer(self, ctx):
            value = super().validate_customer(ctx)
            value["preserves_facts"] = kind != "rejected"
            return value

    provider = Fresh()
    kwargs = {
        "tasks": {
            t: SimpleNamespace(
                id=t, user_scenario="E-only", description="", user_tools=[]
            )
            for t in ("1", "2", "3")
        },
        "runner": runner,
        "domain_policy": "policy",
        "output_directory": tmp_path,
        "manifest_sha256": "manifest",
    }
    fresh = propose_fresh_customer_v2(result, provider, **kwargs)
    calls = list(provider.calls)
    assert propose_fresh_customer_v2(result, provider, **kwargs) == fresh
    assert calls == provider.calls
    assert (fresh is None) == (kind != "valid")
    native = FakeRunner()
    args = {
        "gate_seeds": [1, 2],
        "heldout_tasks": {"H": object()},
        "heldout_task_ids": ["H"],
        "initial_service": result.initial_service,
        "final_service": result.initial_service,
        "fresh_customer": fresh,
        "runner": native,
        "max_parallel_episodes": 1,
    }
    score = run_v2_endpoint_evaluation(**args)
    before = list(native.calls)
    assert run_v2_endpoint_evaluation(**args) == score and native.calls == before
    assert len(score["cells"]) == (4 if fresh else 2)
    assert all(
        c["identical_to_S0"] for c in score["cells"] if c["service_endpoint"] == "ST"
    )
    if not fresh:
        assert score["fresh_adaptive_status"] == "unavailable"
        assert all(c["customer_condition"] == "native_customer" for c in score["cells"])


def test_exclusive_writer_lock(tmp_path):
    with (
        frozen_run_lock(tmp_path / "checkpoint.json"),
        pytest.raises(RuntimeError, match="active writer"),
        frozen_run_lock(tmp_path / "checkpoint.json"),
    ):
        pytest.fail("second writer must not acquire")
    with frozen_run_lock(tmp_path / "checkpoint.json"):
        pass


def test_interrupted_request_charge_never_imputes_tokens_or_benchmark_failure(tmp_path):
    from evotau.budget import RequestBudget

    path = tmp_path / "api-usage-live.json"
    budget = RequestBudget(2)
    budget.enable_live_usage(path)
    budget._begin(
        "offline"
    )  # Hard kill before response; test never dispatches a request.
    before = path.read_bytes()
    (tmp_path / "manifest.json").write_text(json.dumps({"manifest_sha256": "frozen"}))
    charged = reconcile_interrupted_requests(
        tmp_path, "process confirmed dead; conservatively charge unknown"
    )
    restored = RequestBudget(2)
    restored.enable_live_usage(path)
    assert restored.snapshot().attempts == restored.snapshot().failures == 1
    assert (
        restored.snapshot().prompt_tokens == restored.snapshot().completion_tokens == 0
    )
    assert restored.snapshot().usage_unavailable == 1
    evidence = json.loads(Path(charged["audit"]).read_text())
    assert evidence["before"] == json.loads(before)
    assert not list(tmp_path.glob("episodes/*/episode-record.json"))


def test_episode_commit_republishes_exactly_without_provider_or_evaluator(tmp_path):
    from evotau.tau_episode_runner import _publish_episode_completion

    payload = {
        "manifest_sha256": "m",
        "record": {"task_success": False},
        "telemetry": {"episode_key": "frozen"},
        "activation_trace": {"decisions": []},
    }
    payload["completion_sha256"] = sha256_json(payload)
    (tmp_path / "episode-completion.json").write_text(json.dumps(payload))
    _publish_episode_completion(tmp_path, "m")
    records = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    _publish_episode_completion(tmp_path, "m")
    assert all(
        (tmp_path / name).read_bytes() == value for name, value in records.items()
    )
    with pytest.raises(ValueError, match="manifest/digest"):
        _publish_episode_completion(tmp_path, "other")


@pytest.mark.parametrize("interrupt", [None, "gen1", "H"])
@pytest.mark.parametrize("fresh_accepted", [True, False])
def test_full_G2_V_replay_fresh_H_finalization_and_resume(
    tmp_path, monkeypatch, interrupt, fresh_accepted
):
    """Use the actual config/manifest/orchestrator, with explicitly synthetic outcomes."""
    from evotau import alternating_run
    from evotau.alternating_manifest import AlternatingManifest
    from evotau.evolution_candidates import V2Providers
    from evotau.records import (
        EpisodeRecord,
        EpisodeStatus,
        customer_strategy_id,
        service_strategy_id,
    )
    from evotau.service_skills import ServiceSkillMemoryV2

    raw = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / "configs/v2-release-qwen37plus-gpt61sol-e20-v3-h5-g2-p1.yaml"
        ).read_text()
    )
    directory = tmp_path / "configs"
    directory.mkdir()
    file = directory / "formal.yaml"
    file.write_text(yaml.safe_dump(raw))
    exp = raw["experiment"]
    output = tmp_path / exp["output_path"]
    ids = exp["task_selection"]
    native = FakeRunner()
    native.service_policy_text = "Immutable native policy."
    native.tasks = {}
    failed_once = False
    loads = []

    def load(manifest, data, *, include_validation, include_heldout):
        loads.append(include_heldout)
        if include_heldout:
            decision = json.loads((output / "fresh-customer-proposal.json").read_text())
            assert decision["status"] == (
                "available" if fresh_accepted else "unavailable"
            )
            assert (output / "evolution-v2/g0001-generation_complete.json").exists()
        keys = (
            ids["evolution"]
            + ids["validation"]
            + (ids["heldout"] if include_heldout else [])
        )
        return {
            t: SimpleNamespace(
                id=t, user_scenario="hidden-" + t, description="hidden", user_tools=[]
            )
            for t in keys
        }

    def build_runner(**kw):
        native.tasks = kw["task_objects"]
        return native

    monkeypatch.setattr(alternating_run, "load_alternating_tasks", load)
    monkeypatch.setattr(alternating_run, "TauBenchEpisodeRunner", build_runner)

    def rollout(*, task_id, seed, customer, service, panel_name):
        nonlocal failed_once
        key = (
            task_id,
            seed,
            customer_strategy_id(customer),
            service_strategy_id(service),
        )
        if key in native.cache:
            return native.cache[key]
        if interrupt == "H" and task_id == ids["heldout"][1] and not failed_once:
            failed_once = True
            raise TimeoutError("offline H provider timeout")
        success = task_id != "66" or bool(service.skills)
        value = EpisodeRecord(
            "ep-" + str(len(native.cache)),
            task_id,
            seed,
            key[2],
            key[3],
            EpisodeStatus.COMPLETE,
            success,
            native_reward=float(success),
            total_steps=5,
            termination_reason="user_stop",
        )
        native.calls.append(key)
        native.cache[key] = value
        return value

    monkeypatch.setattr(FakeRunner, "__call__", lambda self, **kw: rollout(**kw))

    def customers(self, ctx, count):
        return {
            "candidates": [
                {
                    "strategy": (
                        "Fresh grounded correction."
                        if "purpose" in ctx
                        else "Ask a grounded explanation."
                    ),
                    "semantic_family": "grounding",
                    "target_weakness_family": "scope",
                    "substantive_delta_from_prior": "",
                }
            ]
        }

    def validator(self, ctx):
        return {
            "preserves_facts": fresh_accepted
            if ctx["strategy"] == "Fresh grounded correction."
            else True,
            "preserves_objective": True,
            "interaction_only": True,
            "no_benchmark_leakage": True,
            "reason": "offline verdict",
        }

    def diagnose(self, ctx):
        nonlocal failed_once
        assert "hidden" not in json.dumps(ctx)
        if interrupt == "gen1" and ctx["generation"] == 1 and not failed_once:
            failed_once = True
            raise TimeoutError("offline Gen1 diagnosis timeout")
        if all(row["task_success"] for row in ctx["current_outcomes"]):
            return {"clusters": []}
        return {
            "clusters": [
                {
                    "cluster_id": "scope",
                    "root_cause": "Scope handling",
                    "evidence_task_ids": ["66"],
                    "protected_success_task_ids": ["92"],
                    "recommended_surface": "skill",
                    "recommended_mutation_types": ["add"],
                    "risk": "Confirmation",
                }
            ]
        }

    def mutate(self, ctx):
        value = mutation()
        value.update(target_cluster_id="scope", expected_fixes=["66"])
        return value

    monkeypatch.setattr(V2Providers, "customers", customers)
    monkeypatch.setattr(V2Providers, "validate_customer", validator)
    monkeypatch.setattr(V2Providers, "diagnose", diagnose)
    monkeypatch.setattr(V2Providers, "mutate", mutate)
    monkeypatch.setattr(
        V2Providers,
        "validate_skill",
        lambda *a: {
            "reusable": True,
            "policy_subordinate": True,
            "no_task_entities": True,
            "reason": "offline",
        },
    )
    if interrupt:
        with pytest.raises(TimeoutError):
            alternating_run.run_from_config(file, tau2_data_dir=tmp_path)
        frozen = {p: p.read_bytes() for p in (output / "evolution-v2").glob("*.json")}
    else:
        frozen = {}
    _, result = alternating_run.run_from_config(file, tau2_data_dir=tmp_path)
    assert result["status"] == "complete" and len(result["generations"]) == 2
    assert result["generations"][0]["service_phase"]["accepted"] is True
    assert result["generations"][1]["service_phase"]["accepted"] is False
    assert result["validation_evaluated"] and result["heldout_evaluated"]
    assert (result["fresh_adaptive_customer"] is not None) == fresh_accepted
    assert len(result["heldout_endpoint_evaluation"]["cells"]) == (
        4 if fresh_accepted else 2
    )
    assert all(p.read_bytes() == content for p, content in frozen.items())
    assert len(native.calls) == len(set(native.calls))
    calls = list(native.calls)
    _, completed_replay = alternating_run.run_from_config(file, tau2_data_dir=tmp_path)
    assert len(completed_replay["generations"]) == 2 and calls == native.calls
    assert (
        completed_replay["provider_usage"]["attempts"] == 0
    )  # All outcomes explicitly synthetic.
    assert native.cache and isinstance(ServiceSkillMemoryV2(), ServiceSkillMemoryV2)
    assert len(AlternatingManifest.from_mapping(raw).evolution_task_ids) == 20


def test_repair_E_rejection_does_not_claim_V_was_evaluated(tmp_path):
    from copy import deepcopy
    from dataclasses import replace

    from evotau.skill_evolution_config import DEFAULT_V2

    class LateSeedRegression(FakeRunner):
        def __call__(self, **kw):
            result = super().__call__(**kw)
            if kw["task_id"] == "2" and kw["seed"] == 3 and kw["service"].skills:
                return replace(result, task_success=False)
            return result

    policy = deepcopy(DEFAULT_V2)
    policy["service_evolution"].update(candidates_per_cluster=1, crossover=False)
    policy["statistical_gate"]["method"] = "finite_panel_paired"
    result, _, _ = run(
        tmp_path, runner=LateSeedRegression(), policy=policy, validation=True
    )
    phase = result.generations[0]["service_phase"]
    assert phase["candidates"][0]["gate"]["repair_superiority"]["verdict"] == "REJECTED"
    assert phase["candidates"][0]["gate"]["opponents"] == []
    assert phase["validation_evaluated"] is False
    assert phase["acceptance_mode"] == "finite_panel_validation_gated"


def test_native_cache_never_opens_H_completion_bundle_before_H_unseal(tmp_path):
    from threading import Lock

    from evotau.tau_episode_runner import TauBenchEpisodeRunner

    directory = tmp_path / "episodes" / "heldout"
    directory.mkdir(parents=True)
    (directory / "completion-key.json").write_text(
        json.dumps(
            {"task_id": "74", "manifest_sha256": "m", "episode_key_sha256": "key"}
        )
    )
    # If decoded at all, it fails. H trajectory/activation material is never opened.
    (directory / "episode-completion.json").write_text("FORBIDDEN H CONTENT")
    runner = object.__new__(TauBenchEpisodeRunner)
    runner.output_directory = tmp_path
    runner.tasks = {"66": object()}
    runner.manifest = SimpleNamespace(sha256="m")
    runner._completed_episode_cache = {}
    runner._completed_episode_cache_lock = Lock()
    runner._load_completed_episode_cache()
    assert runner._completed_episode_cache == {}


def test_committed_generation_reconstructs_missing_publication_without_calls(tmp_path):
    result, providers, runner = run(tmp_path, generations=2)
    (
        tmp_path / "generation-0000.json"
    ).unlink()  # Simulate kill between commit and publication.
    calls = list(providers.calls), list(runner.calls)
    replay, _, _ = run(tmp_path, generations=2, provider=providers, runner=runner)
    assert (tmp_path / "generation-0000.json").exists()
    assert replay.generations == result.generations
    assert calls == (providers.calls, runner.calls)


def test_evolver_pause_file_prevents_next_request(tmp_path, monkeypatch):
    from evotau.episode_execution import StopBeforeEpisodeDispatch

    signal = tmp_path / "PAUSE"
    signal.touch()
    dispatch = LLMAlternatingEvolvers(
        model="offline", model_args={}, output_directory=tmp_path
    )
    dispatch.stop_before_next_episode_file = signal
    monkeypatch.setattr(
        LLMAlternatingEvolvers,
        "_json_call",
        staticmethod(lambda *a, **kw: pytest.fail("must not dispatch while paused")),
    )
    with pytest.raises(StopBeforeEpisodeDispatch):
        V2Providers(dispatch).diagnose({"task_interactions": []})
    assert not list(tmp_path.glob("evolver-calls/*/input.json"))
