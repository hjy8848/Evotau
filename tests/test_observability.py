from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest
import yaml

from evotau import tau_provenance
from evotau.alternating_manifest import AlternatingManifest
from evotau.observability import EpisodeTelemetry


def native_orchestrator():
    from tau2.data_model.message import (
        AssistantMessage,
        ToolCall,
        ToolMessage,
        UserMessage,
    )
    from tau2.orchestrator.orchestrator import Orchestrator, Role

    orchestrator = Orchestrator.__new__(Orchestrator)
    user = UserMessage(role="user", content="PRIVATE_CUSTOMER_TEXT", timestamp="2026-10-07T09:00:01")
    agent = AssistantMessage(role="assistant", content=None, timestamp="2026-10-07T09:00:02",
                             tool_calls=[ToolCall(id="call", name="exchange_delivered_order_items", arguments={"PRIVATE_ARGUMENT": 1})])
    tool = ToolMessage(role="tool", id="call", content="PRIVATE_TOOL_RESULT", timestamp="2026-10-07T09:00:03")
    orchestrator.done = False; orchestrator.step_count = 0; orchestrator.solo_mode = False
    orchestrator.validate_communication = False
    orchestrator.trajectory = []
    orchestrator.from_role = Role.AGENT; orchestrator.to_role = Role.USER
    orchestrator.message = AssistantMessage.text("hello")
    orchestrator.user = SimpleNamespace(generate_next_message=lambda *_: (user, None))
    orchestrator.agent = SimpleNamespace(generate_next_message=lambda *_: (agent, None), is_stop=lambda _: False)
    orchestrator.user_state = orchestrator.agent_state = None
    orchestrator.environment = SimpleNamespace(sync_tools=lambda: None)
    orchestrator._update_voice_metadata = lambda _: None
    orchestrator._execute_tool_calls = lambda _: [tool]
    return orchestrator


def observer(tmp_path, attempt="attempt"):
    return EpisodeTelemetry(tmp_path / attempt / "active-episode.json", manifest_sha256="digest",
                            attempt_id=attempt, task_id="29", panel_name="generation-0-customer-incumbent")


def test_actual_pinned_native_steps_are_observed_without_changing_trajectory(tmp_path):
    original, observed = native_orchestrator(), native_orchestrator()
    live = observer(tmp_path)
    with live.observe(observed):
        for turn, role in enumerate(("user", "assistant", "tool")):
            assert observed.step() == original.step()
            assert observed.get_trajectory() == original.get_trajectory()
            value = json.loads(live.path.read_text())
            assert value["turn_index"] == turn and value["role"] == role
            assert value["task_id"] == "29" and value["panel_name"] == "generation-0-customer-incumbent"
            assert "PRIVATE" not in live.path.read_text()
            if role != "user":
                assert value["tool_name"] == "exchange_delivered_order_items"
    assert "step" not in vars(observed)  # Restore the native method after observation.
    assert all(message.turn_idx is None for message in observed.trajectory)


def test_observer_preserves_original_exception_and_ignores_telemetry_io_errors(tmp_path, monkeypatch):
    orchestrator = native_orchestrator(); live = observer(tmp_path)
    failure = RuntimeError("native provider failure")
    def fail(*_):
        raise failure
    orchestrator.user.generate_next_message = fail
    monkeypatch.setattr("evotau.observability._write_json_atomic", fail)
    with live.observe(orchestrator), pytest.raises(RuntimeError) as raised:
        orchestrator.step()
    assert raised.value is failure and "step" not in vars(orchestrator)


def test_parallel_native_instances_publish_separate_atomic_attempts(tmp_path):
    barrier = Barrier(2)
    def worker(attempt):
        orch = native_orchestrator(); live = observer(tmp_path, attempt)
        with live.observe(orch):
            barrier.wait(timeout=5)
            for _ in range(3):
                orch.step()
        live.publish("complete")
        return json.loads(live.path.read_text())
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(worker, ("first", "second")))
    assert {row["attempt_id"] for row in rows} == {"first", "second"}
    assert all(row["status"] == "complete" and row["turn_index"] == 2 for row in rows)


def test_runtime_console_fingerprints_are_separate_and_resume_ignores_console_changes(tmp_path, monkeypatch):
    source = tmp_path / "src/evotau"; source.mkdir(parents=True)
    module = source / "tau_provenance.py"; module.write_text("runtime v1")
    (tmp_path / "pyproject.toml").write_text("name = 'test'")
    web = source / "web"; web.mkdir()
    page = web / "page.html"; page.write_text("console v1")
    console_py = web / "app.py"; console_py.write_text("console v1")
    telemetry = source / "observability.py"; telemetry.write_text("observer v1")
    monkeypatch.setattr(tau_provenance, "__file__", str(module))
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/real-alternating-deepseek-v4-flash-5e3v5h.yaml").read_text())
    frozen = AlternatingManifest.from_mapping(config).to_document()
    before = tau_provenance.capture_code_provenance(runtime_only=True)
    legacy = tau_provenance.capture_code_provenance().source_sha256
    for path in (page, console_py, telemetry):
        path.write_text("console v2")
    after = tau_provenance.capture_code_provenance(runtime_only=True)
    assert before.source_sha256 == after.source_sha256
    assert before.console_source_sha256 != after.console_source_sha256
    assert legacy != tau_provenance.capture_code_provenance().source_sha256  # Legacy rules retained.
    assert frozen["evotau"]["source_scope"] == "runtime-v2"
    assert frozen["evotau"]["runtime_source_sha256"] == before.source_sha256
    assert "console_source_sha256" not in frozen["evotau"]  # Console cannot change execution identity.
    assert AlternatingManifest.from_mapping(config).bind_saved_provenance(frozen).to_document() == frozen
    module.write_text("runtime v2")
    assert tau_provenance.capture_code_provenance(runtime_only=True).source_sha256 != before.source_sha256
    with pytest.raises(ValueError, match="different frozen execution inputs"):
        AlternatingManifest.from_mapping(config).bind_saved_provenance(frozen)
