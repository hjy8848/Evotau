"""Offline reproductions of the 2026-10-06 reliability audit findings."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace

import pytest
import yaml

from evotau import alternating_run, tau_provenance
from evotau.alternating import (
    LLMAlternatingEvolvers,
    _accuracy,
    _run_panel,
    _service_context_episodes,
    _trajectory_context,
)
from evotau.alternating_manifest import AlternatingManifest
from evotau.budget import RequestBudget
from evotau.episode_execution import StopBeforeEpisodeDispatch
from evotau.records import EpisodeRecord, EpisodeStatus
from evotau.strategies import PromptStrategy
from evotau.tau_adapter import service_agent_class
from evotau.tau_episode_runner import NativeEpisodeRunError, TauBenchEpisodeRunner

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def native_fixture(tmp_path, monkeypatch):
    config = yaml.safe_load((ROOT / "configs/real-alternating-deepseek-v4-flash-5e3v5h.yaml").read_text())
    config["experiment"]["request_budget_cap"] = 1
    manifest = AlternatingManifest.from_mapping(config)
    data = tmp_path / "data"
    policy = data / "tau2/domains/retail/policy.md"
    policy.parent.mkdir(parents=True)
    policy.write_text("Native policy")
    tasks = {"29": SimpleNamespace(id="29", user_scenario="PRIVATE_SCENARIO")}
    output = tmp_path / "run"
    built = []

    class NativeAgent:
        @property
        def system_prompt(self):
            return "Native policy"

    def build(**kwargs):
        built.append(kwargs)
        return SimpleNamespace(
            agent=service_agent_class(NativeAgent, kwargs["service_strategy"])(),
            user=SimpleNamespace(system_prompt="Native user prompt"),
            simulation_id="simulation-id", trajectory=[{"role": "user", "content": "hello"}],
        )

    def simulation(reward=1):
        payload = {"id": "simulation-id", "task_id": "29", "seed": 1,
                   "messages": [{"role": "user", "content": "hello"}],
                   "reward_info": {"reward": reward}}
        return SimpleNamespace(model_dump=lambda **_: payload)

    def run(orchestrator, budget, **kwargs):
        assert kwargs["run_reviewer"] is False
        budget.dispatch_external_call(
            model="offline/model", call_name="agent_response", dispatch=lambda: {"output_text": "ok"},
        )
        value = simulation()
        kwargs["on_simulation"](value)
        return value

    monkeypatch.setattr("evotau.tau_episode_runner.build_phase0_orchestrator", build)
    monkeypatch.setattr("evotau.tau_episode_runner.run_with_budget", run)

    def make(budget=None, frozen=manifest):
        return TauBenchEpisodeRunner(
            manifest=frozen, config=config, data_dir=data, request_budget=budget or RequestBudget(1),
            output_directory=output, task_objects=tasks,
        )

    def invoke(runner, panel="generation-0-customer-incumbent", seed=1, service="S0"):
        return runner(task_id="29", seed=seed, customer=None, service=PromptStrategy(service), panel_name=panel)

    return SimpleNamespace(config=config, manifest=manifest, output=output, built=built,
                           make=make, invoke=invoke, simulation=simulation, data=data, tasks=tasks)


def test_runner_model_args_reach_actual_tau_generate_and_http_payload(native_fixture, monkeypatch):
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")
    llm_utils = pytest.importorskip("tau2.utils.llm_utils")
    from tau2.data_model.message import SystemMessage, UserMessage

    sent = []

    def respond(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "offline-response", "object": "chat.completion", "created": 1,
            "model": "deepseek-v4-flash",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        })

    client = openai.OpenAI(api_key="offline-placeholder", http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    monkeypatch.setattr(llm_utils, "LLM_CACHE_ENABLED", False)
    fixture = native_fixture
    fixture.config["experiment"]["request_budget_cap"] = None
    fixture.manifest = AlternatingManifest.from_mapping(fixture.config)
    runner = TauBenchEpisodeRunner(
        manifest=fixture.manifest, config=fixture.config, data_dir=fixture.data,
        request_budget=RequestBudget(None), output_directory=fixture.output, task_objects=fixture.tasks,
    )

    def run(orchestrator, budget, **kwargs):
        call_names = {"agent": "agent_response", "customer": "user_simulator_response", "evaluator": "nl_assertions_eval"}
        actual = {"agent": fixture.built[0]["agent_model_args"],
                  "customer": fixture.built[0]["customer_model_args"],
                  "evaluator": kwargs["evaluator_model_args"]}
        with budget.instrument_tau_llm_utils(llm_utils):
            for role, args in actual.items():
                llm_utils.generate(
                    model=runner.models[role], messages=[SystemMessage(role="system", content="test"),
                                                       UserMessage(role="user", content="ok")],
                    call_name=call_names[role], client=client, num_retries=0, **args,
                )
        value = fixture.simulation()
        kwargs["on_simulation"](value)
        return value

    monkeypatch.setattr("evotau.tau_episode_runner.run_with_budget", run)
    fixture.invoke(runner)
    client.close()
    assert len(sent) == 3
    for payload in sent:
        assert payload["thinking"] == {"type": "disabled"}
        assert "thinking_mode" not in payload
        assert payload["temperature"] == 0.0
    calls = [json.loads(line) for line in next(fixture.output.glob("episodes/*/provider-calls.jsonl")).read_text().splitlines()]
    assert {row["call_name"] for row in calls} == {
        "agent_response", "user_simulator_response", "nl_assertions_eval",
    }
    assert all(row["response"]["finish_reason"] == "stop" for row in calls)
    assert runner.request_budget.snapshot().attempts == 3


@pytest.mark.parametrize("reward", [None, True, "1", float("nan"), float("inf")])
def test_invalid_reward_is_incomplete_and_cannot_be_cached(native_fixture, monkeypatch, reward):
    fixture = native_fixture

    def run(_, __, **kwargs):
        value = fixture.simulation(reward)
        kwargs["on_simulation"](value)
        return value

    monkeypatch.setattr("evotau.tau_episode_runner.run_with_budget", run)
    runner = fixture.make()
    with pytest.raises(NativeEpisodeRunError, match="unscored"):
        fixture.invoke(runner)
    assert not list(fixture.output.glob("episodes/*/episode-record.json"))
    assert len(list(fixture.output.glob("episodes/*/incomplete-run.json"))) == 1
    assert not runner.has_completed_episode(task_id="29", seed=1, customer=None, service=PromptStrategy("S0"), panel_name="other")


def test_uncertain_score_and_missing_trajectory_cannot_enter_fitness():
    unknown = EpisodeRecord(episode_id="unknown", task_id="29", seed=1,
                            customer_strategy_id="c", service_strategy_id="s",
                            status=EpisodeStatus.UNCERTAIN, task_success=None)
    with pytest.raises(ValueError, match="complete native"):
        _accuracy([unknown])
    with pytest.raises(ValueError, match="complete native"):
        _run_panel(lambda **_: unknown, task_ids=["29"], tasks={"29": {}}, seed=1,
                   customer=None, service=PromptStrategy("S"), panel_name="test")
    with pytest.raises(TypeError, match="trajectory is missing"):
        _service_context_episodes([replace(unknown, status=EpisodeStatus.COMPLETE, task_success=True)],
                                  SimpleNamespace(load_trajectory=lambda _: None), {"29": {}})


def test_multitool_results_are_visible_without_raw_private_fields():
    projected = _trajectory_context({"messages": [
        {"role": "assistant", "tool_calls": [{"id": "a", "name": "lookup", "arguments": {}}]},
        {"role": "multi_tool", "raw_data": "PRIVATE", "tool_messages": [
            {"role": "tool", "name": "lookup", "tool_call_id": "a", "content": "state=updated", "raw_data": "PRIVATE"},
            {"role": "tool", "name": "check", "tool_call_id": "b", "content": "allowed=false"},
        ]},
    ]})["messages"]
    assert [item["role"] for item in projected] == ["assistant", "tool", "tool"]
    assert projected[1]["content"] == "state=updated"
    assert projected[2]["tool_call_id"] == "b"
    assert "PRIVATE" not in json.dumps(projected)


def test_resume_preserves_original_manifest_but_rejects_changed_execution_inputs(native_fixture):
    old = native_fixture.manifest
    changed_git = replace(old, evotau_git_commit="b" * 40, evotau_working_tree_clean=True)
    assert changed_git.bind_saved_provenance(old.to_document()).to_document() == old.to_document()
    for changed in (replace(changed_git, max_steps=old.max_steps + 1),
                    replace(changed_git, evotau_source_sha256="c" * 64),
                    replace(changed_git, config_sha256="d" * 64)):
        with pytest.raises(ValueError, match="execution inputs"):
            changed.bind_saved_provenance(old.to_document())
    tampered = old.to_document()
    tampered["seed"] += 1
    with pytest.raises(ValueError, match="fingerprint"):
        old.bind_saved_provenance(tampered)
    runner = native_fixture.make()
    record = native_fixture.invoke(runner)
    resumed = native_fixture.make(frozen=changed_git)
    assert resumed.manifest.sha256 == old.sha256
    assert native_fixture.invoke(resumed, panel="generation-1-customer-incumbent") == record
    assert len(native_fixture.built) == 1
    assert resumed.request_budget.snapshot().attempts == 1


def test_unrelated_configs_do_not_change_source_fingerprint(tmp_path, monkeypatch):
    source = tmp_path / "src/evotau"
    source.mkdir(parents=True)
    module = source / "tau_provenance.py"
    module.write_text("source v1")
    (tmp_path / "pyproject.toml").write_text("name = 'test'")
    monkeypatch.setattr(tau_provenance, "__file__", str(module))
    before = tau_provenance.capture_code_provenance().source_sha256
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "unrelated.yaml").write_text("irrelevant: 1")
    assert tau_provenance.capture_code_provenance().source_sha256 == before
    module.write_text("source v2")
    assert tau_provenance.capture_code_provenance().source_sha256 != before


def test_identical_concurrent_conditions_dispatch_once_and_cache_needs_no_reservation(native_fixture, monkeypatch):
    fixture = native_fixture
    entered, release = Event(), Event()

    def run(_, budget, **kwargs):
        entered.set()
        assert release.wait(3)
        budget.dispatch_external_call(model="offline/model", call_name="agent_response", dispatch=dict)
        value = fixture.simulation()
        kwargs["on_simulation"](value)
        return value

    monkeypatch.setattr("evotau.tau_episode_runner.run_with_budget", run)
    budget = RequestBudget(1)
    runner = fixture.make(budget)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(fixture.invoke, runner, "generation-0-customer-incumbent")
        assert entered.wait(3)
        second = executor.submit(fixture.invoke, runner, "generation-1-customer-incumbent")
        release.set()
        assert first.result() == second.result()
    assert len(fixture.built) == 1
    assert budget.snapshot().attempts == 1
    assert budget.snapshot().denied == 0
    assert budget.snapshot().reserved == 0
    assert fixture.invoke(runner, "generation-2-customer-incumbent") == first.result()
    assert budget.snapshot().denied == 0
    refs = json.loads((fixture.output / "episode-panel-references.json").read_text())["references"].values()
    assert len(refs) == 3
    assert sum(row["reused"] for row in refs) == 2
    assert len(list(fixture.output.glob("episodes/*/episode-record.json"))) == 1


def test_failed_reply_saves_partial_trace_and_response_diagnostics(native_fixture, monkeypatch):
    fixture = native_fixture

    def run(_, budget, **kwargs):
        budget.dispatch_external_call(model="offline/model", call_name="user_simulator_response", dispatch=lambda: {
            "id": "empty-reply", "output_text": "", "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "usage": {"input_tokens": 100, "output_tokens": 20, "output_tokens_details": {"reasoning_tokens": 20}},
        }, request_args={"model": "offline/model", "reasoning_effort": "high", "api_key": "offline-secret", "headers": {"Authorization": "secret"}})
        raise ValueError("UserMessage has no visible content")

    monkeypatch.setattr("evotau.tau_episode_runner.run_with_budget", run)
    with pytest.raises(NativeEpisodeRunError) as raised:
        fixture.invoke(fixture.make())
    incomplete = json.loads((fixture.output / raised.value.diagnostics_ref).read_text())
    partial = json.loads((fixture.output / incomplete["partial_trajectory_ref"]).read_text())
    assert partial["messages"] == [{"role": "user", "content": "hello"}]
    raw = (fixture.output / incomplete["provider_calls_ref"]).read_text()
    row = json.loads(raw)
    assert row["api_success"] is True
    assert row["response"]["output_state"] == "empty"
    assert row["response"]["incomplete_reason"] == "max_output_tokens"
    assert row["response"]["reasoning_tokens"] == 20
    assert row["request_args"]["reasoning_effort"] == "high"
    assert "secret" not in raw


@pytest.mark.parametrize("exception,status", [(RuntimeError("native load failed"), "failed"),
                                              (StopBeforeEpisodeDispatch("operator pause"), "paused")])
def test_cli_failure_and_pause_finalize_state_and_keep_resume_provenance(tmp_path, monkeypatch, exception, status):
    config = yaml.safe_load((ROOT / "configs/alternating-mechanism-smoke.yaml").read_text())
    config["experiment"]["real_provider_enabled"] = True
    configs = tmp_path / "configs"
    configs.mkdir()
    path = configs / "run.yaml"
    path.write_text(yaml.safe_dump(config))

    def fail(*_, **__):
        raise exception

    monkeypatch.setattr(alternating_run, "load_alternating_tasks", fail)
    with pytest.raises(type(exception)):
        alternating_run.run_from_config(path, tau2_data_dir=tmp_path)
    output = tmp_path / config["experiment"]["output_path"]
    state = json.loads((output / "run-execution-state.json").read_text())
    assert state["status"] == status
    assert state["failure"]["stage"] == "task_loading"
    assert state["failure"]["failure_message"] == str(exception)
    assert state["attempts"][0]["evotau"]["source_sha256"]
    # Commit/dirty metadata changes can resume, with a new invocation audit.
    original = alternating_run.AlternatingManifest.from_mapping(config)
    monkeypatch.setattr(alternating_run.AlternatingManifest, "from_mapping", lambda _: replace(original, evotau_git_commit="b" * 40))
    with pytest.raises(type(exception)):
        alternating_run.run_from_config(path, tau2_data_dir=tmp_path)
    state = json.loads((output / "run-execution-state.json").read_text())
    assert state["resume_count"] == 1
    assert len(state["attempts"]) == 2
    assert state["total_wall_clock_seconds"] == round(sum(x["wall_clock_seconds"] for x in state["attempts"]), 6)


def test_evolver_persists_actual_input_and_failed_call_metadata(tmp_path, monkeypatch):
    from evotau import inferai_responses

    monkeypatch.setenv("OFFLINE_TEST_KEY", "offline-only")
    received = []

    def fail(url, payload, api_key, **kwargs):
        received.append(payload)
        raise RuntimeError("HTTP 504")

    monkeypatch.setattr(inferai_responses, "_post_responses", fail)
    provider = LLMAlternatingEvolvers(model="gpt-6.1-sol", model_args={
        "api_protocol": "responses", "api_base": "https://inferaiapi.com/v1",
        "api_key_env": "OFFLINE_TEST_KEY", "reasoning_effort": "high",
    }, request_budget=RequestBudget(None), output_directory=tmp_path)
    with pytest.raises(RuntimeError, match="504") as raised:
        provider.customer_candidates({"generation": 0, "task_interactions": []}, 1)
    failure = tmp_path / raised.value.diagnostics_ref
    actual = json.loads((failure.parent / "input.json").read_text())
    assert json.loads(received[0]["input"][1]["content"]) == actual["context"]
    calls = [json.loads(row) for row in (failure.parent / "provider-calls.jsonl").read_text().splitlines()]
    assert len(calls) == 1
    assert calls[0]["error_type"] == "RuntimeError"
    assert calls[0]["request_args"]["reasoning"] == {"effort": "high"}
    assert "offline-only" not in (failure.parent / "provider-calls.jsonl").read_text()


def test_chat_empty_response_keeps_finish_reason_and_hides_reasoning_and_credentials(tmp_path):
    response = {
        "id": "last-response", "model": "offline/model",
        "choices": [{"finish_reason": "length", "message": {
            "content": "", "reasoning_content": "PRIVATE_REASONING", "tool_calls": [],
        }}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20,
                  "completion_tokens_details": {"reasoning_tokens": 20}},
    }
    module = SimpleNamespace(completion=lambda **_: response, DEFAULT_MAX_RETRIES=0)
    budget = RequestBudget(1)
    path = tmp_path / "calls.jsonl"
    with budget.record_provider_calls(path), budget.instrument_tau_llm_utils(module):
        assert module.completion(model="offline/model", api_key="PRIVATE_KEY", messages=[{"content": "PRIVATE_PROMPT"}]) is response
    raw = path.read_text()
    data = json.loads(raw)
    assert data["api_success"] is True
    assert data["response"]["finish_reason"] == "length"
    assert data["response"]["reasoning_content_chars"] == len("PRIVATE_REASONING")
    assert data["response"]["reasoning_tokens"] == 20
    assert data["response"]["output_state"] == "empty"
    assert "PRIVATE" not in raw
    assert budget.snapshot().attempts == 1


def test_console_shows_failed_partial_conversation(native_fixture, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from evotau.web.app import create_app
    from evotau.web.artifact_reader import ArtifactReader

    fixture = native_fixture

    def fail(*_, **__):
        raise ValueError("no visible reply")

    monkeypatch.setattr("evotau.tau_episode_runner.run_with_budget", fail)
    with pytest.raises(NativeEpisodeRunError):
        fixture.invoke(fixture.make())
    (fixture.output / "run-execution-state.json").write_text(json.dumps({
        "manifest_sha256": fixture.manifest.sha256, "status": "failed",
        "failure": {"stage": "evolution", "failure_message": "no visible reply"},
    }))
    root = fixture.output.parent
    reader = ArtifactReader(root, project_root=root)
    run = reader.get_run("run")
    assert run["status"] == "failed"
    episode = run["episodes"][0]
    assert episode["task_success"] is None
    assert episode["messages"][0]["content"] == "hello"
    client = TestClient(create_app(project_root=root, runs_root=root))
    overview = client.get("/runs/run")
    assert overview.status_code == 200
    assert "no visible reply" in overview.text
    page = client.get(f"/runs/run/episodes/{episode['episode_id']}")
    assert page.status_code == 200
    assert "hello" in page.text and "尚未评分" in page.text


def test_console_accepts_v2_result_and_filters_reused_panel_references(native_fixture):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from evotau.web.app import create_app
    from evotau.web.artifact_reader import ArtifactReader

    fixture = native_fixture
    runner = fixture.make()
    record = fixture.invoke(runner)
    fixture.invoke(runner, panel="generation-1-customer-incumbent")
    (fixture.output / "alternating-result.json").write_text(json.dumps({
        "schema_version": 2, "status": "complete", "manifest_sha256": fixture.manifest.sha256,
        "experiment_id": fixture.manifest.experiment_id, "generations": [],
        "initial_service": {"text": "S0"}, "final_service": {"text": "S0"},
        "initial_customer": {"text": ""}, "final_customer": {"text": ""},
    }))
    root = fixture.output.parent
    reader = ArtifactReader(root, project_root=root)
    run = reader.get_run("run")
    assert run["status"] == "complete"
    assert len(run["episodes"]) == 1
    assert run["episodes"][0]["generations"] == [0, 1]
    assert run["max_concurrency"] == fixture.manifest.max_parallel_episodes
    client = TestClient(create_app(project_root=root, runs_root=root))
    page = client.get("/runs/run/episodes?generation=1&panel=generation-1-customer-incumbent")
    assert page.status_code == 200
    assert record.episode_id in page.text
    assert "No episodes match" not in page.text


def test_concurrent_provider_logs_keep_episode_context_and_cap(tmp_path):
    barrier = Barrier(4)
    budget = RequestBudget(4)

    def request(index):
        def dispatch():
            barrier.wait(timeout=3)
            return {"id": f"reply-{index}", "output_text": "ok", "usage": {"input_tokens": 10, "output_tokens": 2}}
        with budget.record_provider_calls(tmp_path / f"episode-{index}.jsonl"):
            return budget.dispatch_external_call(model=f"offline/{index}", call_name="agent_response", dispatch=dispatch)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(request, range(4)))
    for index in range(4):
        rows = (tmp_path / f"episode-{index}.jsonl").read_text().splitlines()
        assert len(rows) == 1
        assert json.loads(rows[0])["response"]["response_id"] == f"reply-{index}"
    usage = budget.snapshot()
    assert (usage.attempts, usage.prompt_tokens, usage.completion_tokens) == (4, 40, 8)
    assert usage.denied == usage.in_flight == usage.reserved == 0
    assert budget.api_usage_by_call_name()["agent_response"]["calls"] == 4


def test_evolution_resume_does_not_read_saved_heldout_conversation(native_fixture, monkeypatch):
    fixture = native_fixture
    runner = fixture.make()
    record = fixture.invoke(runner)
    heldout_id = fixture.manifest.heldout_task_ids[0]
    directory = fixture.output / "episodes/heldout"
    directory.mkdir()
    heldout = replace(record, task_id=heldout_id, episode_id="H-id", trajectory_ref="episodes/heldout/native-simulation.json")
    (directory / "episode-record.json").write_text(json.dumps(heldout.to_dict()))
    telemetry = json.loads(next(fixture.output.glob("episodes/*/run-telemetry.json")).read_text())
    telemetry["episode_key"]["task_id"] = heldout_id
    telemetry["episode_key_sha256"] = tau_provenance.sha256_json(telemetry["episode_key"])
    telemetry["simulation_id"] = "H-id"
    (directory / "run-telemetry.json").write_text(json.dumps(telemetry))
    (directory / "native-simulation.json").write_text(json.dumps({
        "id": "H-id", "task_id": heldout_id, "seed": 1, "messages": [{"content": "H_PRIVATE_CONVERSATION"}],
    }))
    original = TauBenchEpisodeRunner.load_trajectory
    loaded = []

    def checked(self, episode):
        loaded.append(episode.task_id)
        assert episode.task_id != heldout_id
        return original(self, episode)

    monkeypatch.setattr(TauBenchEpisodeRunner, "load_trajectory", checked)
    resumed = fixture.make()
    assert loaded == ["29"]
    assert fixture.invoke(resumed, panel="next-generation") == record


def test_keyboard_interrupt_saves_partial_attempt_without_becoming_task_failure(native_fixture, monkeypatch):
    def interrupted(*_, **__):
        raise KeyboardInterrupt()
    monkeypatch.setattr("evotau.tau_episode_runner.run_with_budget", interrupted)
    with pytest.raises(KeyboardInterrupt):
        native_fixture.invoke(native_fixture.make())
    artifact = json.loads(next(native_fixture.output.glob("episodes/*/incomplete-run.json")).read_text())
    assert artifact["failure_type"] == "KeyboardInterrupt"
    assert artifact["partial_trajectory_ref"]
    assert not list(native_fixture.output.glob("episodes/*/episode-record.json"))


def test_console_preview_can_resume_same_source_after_git_metadata_changes(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from evotau.web.run_manager import RunManager, RunManagerError

    config = yaml.safe_load((ROOT / "configs/alternating-mechanism-smoke.yaml").read_text())
    configs = tmp_path / "configs"
    configs.mkdir()
    path = configs / "smoke.yaml"
    path.write_text(yaml.safe_dump(config))
    saved = AlternatingManifest.from_mapping(config)
    output = tmp_path / saved.output_path
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(json.dumps(saved.to_document()))
    current = replace(saved, evotau_git_commit="b" * 40, evotau_working_tree_clean=True)
    monkeypatch.setattr(AlternatingManifest, "from_mapping", lambda _: current)
    manager = RunManager(project_root=tmp_path, runs_root=tmp_path / "experiments/runs")
    preview = manager.preview(phase="alternating-self-evolution", config_path=path)
    assert preview.manifest_document == saved.to_document()
    assert preview.max_concurrency == saved.max_parallel_episodes
    manager._check_existing_output(preview)
    monkeypatch.setattr(AlternatingManifest, "from_mapping", lambda _: replace(current, evotau_source_sha256="c" * 64))
    with pytest.raises(RunManagerError, match="不兼容"):
        manager.preview(phase="alternating-self-evolution", config_path=path)
