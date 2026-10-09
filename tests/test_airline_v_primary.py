"""Native Airline integration, V-only promotion and independent A/A contracts."""

import json
import os
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from test_skill_evolution_v2 import FakeProviders, FakeRunner, record, run

from evotau.aa_diagnostic import compare_trajectories, run_independent_repetitions
from evotau.alternating_manifest import AlternatingManifest
from evotau.alternating_run import load_alternating_tasks
from evotau.evolution_gate import cheap_screen, evaluate_gate, pass_power_k
from evotau.skill_evolution_config import DEFAULT_V2
from evotau.tau_adapter import build_phase0_orchestrator

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(
    os.environ.get("TAU2_DATA_DIR", "/Users/spring/.cache/evotau/tau2-data-b7ea9074")
)


def config():
    return yaml.safe_load(
        (ROOT / "configs/airline-direct-skill-v-primary-e10-v20-h20.yaml").read_text()
    )


def test_airline_pinned_split_loader_policy_tools_and_domain(monkeypatch):
    if not (DATA / "tau2/domains/airline/tasks.json").exists():
        pytest.skip("pinned Airline data unavailable")
    raw = config()
    manifest = AlternatingManifest.from_mapping(raw)
    assert len(manifest.evolution_task_ids) == 10
    assert len(manifest.validation_task_ids) == len(manifest.heldout_task_ids) == 20
    tasks = load_alternating_tasks(
        manifest, DATA, include_validation=False, include_heldout=False
    )
    assert set(tasks) == set(manifest.evolution_task_ids)
    # H bodies never enter streaming Task validation.
    import evotau.alternating_run as module

    original = module._load_selected_retail_task_records
    seen = []

    def capture(path, selected):
        seen.extend(selected)
        assert not selected & set(manifest.heldout_task_ids)
        return original(path, selected)

    monkeypatch.setattr(module, "_load_selected_retail_task_records", capture)
    loaded = load_alternating_tasks(
        manifest, DATA, include_validation=True, include_heldout=False
    )
    assert len(loaded) == 30 and len(seen) == 30
    orchestrator = build_phase0_orchestrator(
        task=tasks[manifest.evolution_task_ids[0]],
        domain="airline",
        agent_model="offline-agent",
        customer_model="offline-customer",
        seed=1,
    )
    assert orchestrator.domain == "airline"
    assert (
        orchestrator.environment.get_policy()
        == (DATA / "tau2/domains/airline/policy.md").read_text()
    )
    tool_names = {t.name for t in orchestrator.environment.get_tools()}
    assert "book_reservation" in tool_names and "cancel_reservation" in tool_names
    assert "cancel_pending_order" not in tool_names


def test_airline_manifest_rejects_retail_fingerprints_and_unsupported_domain():
    raw = config()
    raw["experiment"]["source_blob_sha1"] = yaml.safe_load(
        (
            ROOT
            / "configs/v2-direct-skill-reject-invalid-qwen37plus-official-dsflash-e20-v3-h5-g2-p1.yaml"
        ).read_text()
    )["experiment"]["source_blob_sha1"]
    with pytest.raises(ValueError, match="fingerprints"):
        AlternatingManifest.from_mapping(raw)
    raw = config()
    raw["experiment"]["domain"] = "unknown"
    with pytest.raises(ValueError):
        AlternatingManifest.from_mapping(raw)


def gate_policy():
    policy = deepcopy(DEFAULT_V2["statistical_gate"])
    policy.update(
        bootstrap_samples=300,
        max_harmfulness=0.5,
        max_stuck_delta=0.1,
        min_tasks=20,
        min_success_gain=0.02,
        min_positive_seed_fraction=0.5,
    )
    return policy


def test_v_gate_allows_limited_regression_with_strong_task_block_gain():
    old = [record(str(t), t >= 50, s) for t in range(100) for s in (1, 2, 3, 4)]
    new = [
        record(str(t), t >= 20 and t != 90, s) for t in range(100) for s in (1, 2, 3, 4)
    ]
    result = evaluate_gate(old, new, gate_policy())
    assert result["verdict"] == "ACCEPTED"
    assert result["pass_to_fail_count"] == 4 and result["fail_to_pass_count"] == 120
    assert result["task_count"] == 100 and result["positive_seed_fraction"] == 1


@pytest.mark.parametrize("kind", ["worse", "hard", "stuck", "small", "tie"])
def test_v_rejected_and_inconclusive_paths(kind):
    old = [record(str(t), t >= 50, s) for t in range(100) for s in (1, 2, 3, 4)]
    new = [record(str(t), t >= 20, s) for t in range(100) for s in (1, 2, 3, 4)]
    if kind == "worse":
        new = [record(str(t), t >= 60, s) for t in range(100) for s in (1, 2, 3, 4)]
    if kind == "hard":
        new[0] = record("0", True, 1, hard_policy_protocol_violations=1)
    if kind == "stuck":
        new = [
            record(str(t), t >= 20, s, termination_reason="max_steps")
            for t in range(100)
            for s in (1, 2, 3, 4)
        ]
    if kind == "small":
        old, new = old[:8], new[:8]
    if kind == "tie":
        new = old
    result = evaluate_gate(old, new, gate_policy())
    assert result["verdict"] == (
        "INCONCLUSIVE" if kind in ("small", "tie") else "REJECTED"
    )


def test_screen_retains_cell_evidence_but_allows_configured_noise():
    old = [
        record("a", False, 1),
        record("b", True, 1),
        record("c", True, 1),
        record("d", True, 1),
        record("e", True, 1),
    ]
    new = [record("a", True, 1), record("b", False, 1), *old[2:]]
    assert not cheap_screen(old, new, {"a"}, set())["passed"]
    out = cheap_screen(old, new, {"a"}, set(), max_regression_rate=0.25)
    assert out["passed"] and out["protected_regressions"] == 1
    new[0] = record("a", True, 1, hard_policy_protocol_violations=1)
    assert not cheap_screen(old, new, {"a"}, set(), max_regression_rate=0.25)["passed"]


def test_v_primary_g2_no_e_veto_and_resume_without_model_requests(tmp_path):
    policy = deepcopy(DEFAULT_V2)
    policy["algorithm_version"] = "direct_skill_v_validation_v2"
    policy["service_evolution"].update(crossover=False, candidates_per_generation=1)
    policy["evaluation"].update(
        promotion_protocol="v_primary", screen_max_regression_rate=0.5
    )
    policy["statistical_gate"].update(
        min_tasks=1, max_harmfulness=1, max_stuck_delta=1, bootstrap_samples=100
    )

    class Runner(FakeRunner):
        def __call__(self, **kw):
            r = super().__call__(**kw)
            if kw["task_id"] == "4":
                r = record("4", bool(kw["service"].skills), kw["seed"])
            return r

    class Providers(FakeProviders):
        def propose_skill_mutation(self, context):
            text = json.dumps(context)
            assert "current-repair-E:" not in text
            assert not any(
                row["task"]["task_id"] == "4" for row in context["task_interactions"]
            )
            return super().propose_skill_mutation(context)

    native, providers = Runner(), Providers()
    result, _, _ = run(
        tmp_path,
        policy=policy,
        validation=True,
        generations=2,
        runner=native,
        provider=providers,
    )
    first = result.generations[0]["service_phase"]["candidates"][0]
    assert first["gate"]["repair_superiority"] is None
    assert first["gate"]["opponents"][0]["panel"] == "V"
    assert first["gate"]["opponents"][0]["objective"] == "superiority"
    assert first["decision"] == "ACCEPTED"
    calls = list(providers.calls)
    resumed, _, _ = run(
        tmp_path,
        policy=policy,
        validation=True,
        generations=2,
        runner=native,
        provider=providers,
    )
    assert resumed.service == result.service and providers.calls == calls


def test_aa_independent_cache_namespaces_and_exact_comparison(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    created = []
    executions = []

    class Runner:
        def __init__(self, **kw):
            self.root = Path(kw["output_directory"])
            self.root.mkdir(parents=True)
            self.cache = {}
            created.append((kw["manifest"].experiment_id, self.root))

        def __call__(self, **kw):
            key = (kw["task_id"], kw["seed"])
            if key not in self.cache:
                executions.append((self.root, key))
                payload = {
                    "messages": [{"role": "user", "content": str(self.root)}],
                    "reward_info": {"reward": 1},
                    "termination_reason": "user_stop",
                }
                (self.root / "native.json").write_text(json.dumps(payload))
                self.cache[key] = record(
                    *key[:1], True, key[1], trajectory_ref="native.json"
                )
            return self.cache[key]

    report = run_independent_repetitions(
        config(),
        data_dir="offline",
        output="diagnostic",
        tasks=["1"],
        seeds=[1],
        repetitions=2,
        approved_cap=200000,
        runner_factory=Runner,
        task_loader=lambda *a, **kw: {"1": object()},
    )
    assert len(executions) == 2 and len(set(created)) == 2
    assert not report["comparisons"][0]["first_customer_message_equal"]
    assert report["comparisons"][0]["first_divergence_native_message_index"] == 0
    assert not report["comparisons"][0]["causal_skill_effect_identified"]
    with pytest.raises(ValueError, match="frozen"):
        run_independent_repetitions(
            config(),
            data_dir="offline",
            output="diagnostic",
            tasks=["1"],
            seeds=[2],
            repetitions=2,
            approved_cap=200000,
            runner_factory=Runner,
            task_loader=lambda *a, **k: {"1": object()},
        )
    assert pass_power_k([record("1", True, s) for s in (1, 2, 3, 4)], 4) == 1
    assert (
        compare_trajectories({"messages": []}, {"messages": []})[
            "first_divergence_native_message_index"
        ]
        is None
    )


def test_native_airline_episode_and_resume_use_native_evaluator(tmp_path, monkeypatch):
    if not (DATA / "tau2/domains/airline/tasks.json").exists():
        pytest.skip("pinned Airline data unavailable")
    import tau2.utils.llm_utils as llm
    from litellm import ModelResponse

    from evotau.budget import RequestBudget
    from evotau.service_skills import ServiceSkillMemoryV2
    from evotau.strategies import PromptStrategy
    from evotau.tau_episode_runner import TauBenchEpisodeRunner

    raw = config()
    exp = raw["experiment"]
    exp["models"] = {
        r: "offline-" + r for r in ("agent", "customer", "evaluator", "evolver")
    }
    exp["model_args"] = {r: {"temperature": 0} for r in exp["models"]}
    exp["skill_evolution_v2"]["activator"] = {
        "model": "offline-activator",
        "model_args": {"temperature": 0},
    }
    exp["request_budget_cap"] = 100
    manifest = AlternatingManifest.from_mapping(raw)
    tasks = load_alternating_tasks(
        manifest, DATA, include_validation=False, include_heldout=False
    )
    calls = []

    def completion(*, model, messages, **kwargs):
        calls.append(model)
        assert kwargs.get("num_retries") == 0
        return ModelResponse(
            model=model,
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "###STOP###"
                        if model == "offline-customer"
                        else '{"results":[]}',
                    },
                }
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        )

    monkeypatch.setattr(llm, "completion", completion)
    budget = RequestBudget(100)
    kwargs = {
        "manifest": manifest,
        "config": raw,
        "data_dir": DATA,
        "request_budget": budget,
        "output_directory": tmp_path / "native",
        "task_objects": tasks,
    }
    runner = TauBenchEpisodeRunner(**kwargs)
    args = {
        "task_id": manifest.evolution_task_ids[0],
        "seed": 1,
        "customer": PromptStrategy(""),
        "service": ServiceSkillMemoryV2(),
        "panel_name": "airline-offline-native",
    }
    record1 = runner(**args)
    native = json.loads((tmp_path / "native" / record1.trajectory_ref).read_text())
    assert "reward_info" in native and record1.status.value == "complete"
    before = list(calls)
    kwargs["request_budget"] = RequestBudget(100)
    assert TauBenchEpisodeRunner(**kwargs)(**args).episode_id == record1.episode_id
    assert calls == before

    monkeypatch.chdir(tmp_path)
    before_count = len(calls)
    aa_kwargs = {
        "data_dir": DATA,
        "output": "native-aa",
        "tasks": [manifest.evolution_task_ids[0]],
        "seeds": [1],
        "repetitions": 2,
        "approved_cap": 100,
    }
    report = run_independent_repetitions(raw, **aa_kwargs)
    assert report["episode_count"] == 2 and len(calls) >= before_count + 2
    completed_calls = list(calls)
    again = run_independent_repetitions(raw, **aa_kwargs)
    assert again == report and calls == completed_calls


def test_adaptive_v20_observed_risk_does_not_claim_population_certificate():
    policy = gate_policy()
    policy.update(
        adaptive_validation=True, risk_scope="observed_panel", max_harmfulness=0.15
    )
    old = [record(str(t), t >= 10, s) for t in range(20) for s in (1, 2, 3, 4)]
    new = [
        record(str(t), t >= 2 and not (t == 19 and s == 1), s)
        for t in range(20)
        for s in (1, 2, 3, 4)
    ]
    result = evaluate_gate(old, new, policy, looks=32)
    assert result["verdict"] == "ACCEPTED"
    assert result["pass_to_fail_count"] == 1
    assert result["harmfulness_ci"][1] > 0.15
    assert not result["population_risk_certified"]
    assert result["inference_scope"] == "adaptive_validation_panel_task_block_evidence"


def test_uncalibrated_formal_launch_refused_before_provider(tmp_path):
    from evotau.alternating_run import run_from_config

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config()))
    with pytest.raises(ValueError, match="A/A calibration"):
        run_from_config(path, tau2_data_dir=DATA)
    assert not (tmp_path / "experiments/runs").exists()


def test_airline_booking_result_is_not_retail_order_projection():
    from evotau.evolution_context import _action_result

    payload = {
        "reservation_id": "example",
        "status": "confirmed",
        "passengers": [{"name": "visible-user"}],
        "flights": [
            {"flight_number": "visible-flight", "cabin": "economy", "price": 120}
        ],
        "payments": [{"amount": 120}],
        "baggages": {"count": 2},
        "extra": "x" * 2000,
    }
    message = {"role": "tool", "content": json.dumps(payload)}
    value, omission = _action_result(message)
    assert value == message and omission is None


def test_aa_excludes_generated_call_ids_from_behavioral_divergence():
    a = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "random-a",
                        "name": "get_user_details",
                        "arguments": {"user_id": "u"},
                    }
                ],
            }
        ]
    }
    b = deepcopy(a)
    b["messages"][0]["tool_calls"][0]["id"] = "random-b"
    result = compare_trajectories(a, b)
    assert result["first_divergence_native_message_index"] == 0
    assert result["first_behavioral_divergence_native_message_index"] is None


def test_v_rejects_absolute_stuck_and_existing_severe_violations():
    policy = gate_policy()
    policy.update(max_stuck_rate=0.2, require_zero_hard_violations=True)
    old = [
        record(str(t), t >= 50, s, termination_reason="max_steps")
        for t in range(100)
        for s in (1, 2)
    ]
    new = [
        record(str(t), t >= 20, s, termination_reason="max_steps")
        for t in range(100)
        for s in (1, 2)
    ]
    assert evaluate_gate(old, new, policy)["verdict"] == "REJECTED"
    old = [
        record(str(t), t >= 50, s, hard_policy_protocol_violations=1)
        for t in range(100)
        for s in (1, 2)
    ]
    new = [
        record(str(t), t >= 20, s, hard_policy_protocol_violations=1)
        for t in range(100)
        for s in (1, 2)
    ]
    assert evaluate_gate(old, new, policy)["verdict"] == "REJECTED"


def test_single_e_reversal_does_not_automatically_veto_v_primary_screen():
    old = [record("a", False), record("b", True)]
    new = [record("a", True), record("b", False)]
    assert not cheap_screen(old, new, {"a"}, {"b"})["passed"]
    assert cheap_screen(old, new, {"a"}, {"b"}, regression_allowance=1)["passed"]


def test_gateway_thinking_flag_is_frozen_and_translated():
    from evotau.provider_diagnostics import safe_request_args
    from evotau.tau_provenance import (
        freeze_role_model_args,
        role_model_args_for_runtime,
    )

    args = freeze_role_model_args({'agent': {'enable_thinking': False}}, roles=('agent',))
    runtime = role_model_args_for_runtime(args)['agent']
    assert runtime == {'extra_body': {'enable_thinking': False}}
    assert safe_request_args(runtime)['extra_body'] == {'enable_thinking': False}
    for invalid in ('false', 0, None):
        with pytest.raises(ValueError, match='boolean'):
            freeze_role_model_args({'agent': {'enable_thinking': invalid}}, roles=('agent',))
    with pytest.raises(ValueError, match='coexist'):
        freeze_role_model_args({'agent': {'enable_thinking': False, 'thinking_mode': 'disabled'}}, roles=('agent',))


def test_aa_bounded_parallelism_and_result_order(tmp_path, monkeypatch):
    from threading import Barrier, Lock
    from time import sleep

    monkeypatch.chdir(tmp_path)
    raw = config()
    raw['experiment']['max_parallel_episodes'] = 2
    barrier, lock = Barrier(2), Lock()
    active, peak = 0, 0

    class Runner:
        def __init__(self, **kw):
            self.root = Path(kw['output_directory'])
            self.root.mkdir(parents=True, exist_ok=True)

        def __call__(self, **kw):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            barrier.wait(timeout=5)
            if kw['task_id'] == '1':
                sleep(0.02)
            path = f"{kw['task_id']}-{kw['seed']}.json"
            (self.root / path).write_text(json.dumps({'messages': [], 'reward_info': {}}))
            with lock:
                active -= 1
            return record(kw['task_id'], True, kw['seed'], trajectory_ref=path)

    result = run_independent_repetitions(raw, data_dir='offline', output='parallel-aa',
                                        tasks=['1', '0'], seeds=[1], repetitions=2,
                                        approved_cap=200000, runner_factory=Runner,
                                        task_loader=lambda *a, **kw: {'1': object(), '0': object()})
    assert peak == 2
    assert [r['task_id'] for r in result['comparisons']] == ['1', '0']


def test_explicit_uncalibrated_launch_keeps_thresholds_and_calibration_status():
    raw = config()
    original = AlternatingManifest.from_mapping(raw)
    raw['experiment']['skill_evolution_v2']['evaluation']['allow_uncalibrated_launch'] = True
    manifest = AlternatingManifest.from_mapping(raw)
    before = json.loads(original.skill_evolution_v2_json)
    after = json.loads(manifest.skill_evolution_v2_json)
    assert after['evaluation']['allow_uncalibrated_launch'] is True
    assert after['evaluation']['calibration_confirmed'] is False
    assert after['statistical_gate'] == before['statistical_gate']
    assert manifest.skill_evolution_v2_json != original.skill_evolution_v2_json
    raw['experiment']['skill_evolution_v2']['evaluation']['allow_uncalibrated_launch'] = 'yes'
    with pytest.raises(ValueError, match='boolean'):
        AlternatingManifest.from_mapping(raw)
