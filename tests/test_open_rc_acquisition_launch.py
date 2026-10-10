"""No paid requests: acquisition isolation and durable preflight replay rules."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from evotau.budget import RequestBudget
from evotau.tau_provenance import freeze_role_model_args, role_model_args_for_runtime


@pytest.fixture
def launcher():
    path = Path(__file__).parents[1] / "experiments/execution/run-open-rc-acquisition.py"
    spec = importlib.util.spec_from_file_location("open_rc_acquisition", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_acquisition_config_has_independent_native_EV_no_H(launcher):
    config = yaml.safe_load((Path(__file__).parents[1] /
                            "configs/airline-open-rc-pair-acquisition-g1-c1-p2.yaml").read_text())
    manifest = launcher.check_config(config)
    assert len(manifest.evolution_task_ids) == 10
    assert len(manifest.validation_task_ids) == 20
    assert manifest.request_budget_cap is None
    assert manifest.run_heldout is False
    assert manifest.role_models == tuple(sorted(config["experiment"]["models"].items()))
    for change in ("heldout", "authorization", "search"):
        other = deepcopy(config)
        if change == "heldout":
            other["experiment"]["run_heldout"] = True
        elif change == "authorization":
            other["launch_readiness"]["authorization"] = "unapproved"
        else:
            other["experiment"]["skill_evolution_v2"]["discovery_handoff"] = {
                "protocol_version": "repair_discovery_handoff_v1"}
        with pytest.raises(ValueError):
            launcher.check_config(other)


def test_preflight_success_reuses_completed_response_only(launcher, tmp_path):
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(content='{"ok":true}', model_dump=lambda **_: {"content": '{"ok":true}'})

    for _ in range(2):
        launcher.recorded_probe(tmp_path, "agent", "frozen-model", {}, RequestBudget(None), generate, [])
    assert len(calls) == 1
    assert calls[0]["num_retries"] == 0
    assert "tools" not in calls[0]
    with pytest.raises(ValueError, match="identity"):
        launcher.recorded_probe(tmp_path, "agent", "foreign-model", {}, RequestBudget(None), generate, [])


@pytest.mark.parametrize("content", ['{"ok":', '{"ok":false}', '{"ok":true,"extra":1}'])
def test_preflight_bad_JSON_is_preserved_and_never_blindly_replayed(launcher, tmp_path, content):
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(content=content, model_dump=lambda **_: {"content": content})

    with pytest.raises(ValueError):
        launcher.recorded_probe(tmp_path, "agent", "frozen-model", {}, RequestBudget(None), generate, [])
    assert json.loads((tmp_path / "agent-visible.json").read_text())["content"] == content
    with pytest.raises(RuntimeError, match="Unresolved"):
        launcher.recorded_probe(tmp_path, "agent", "frozen-model", {}, RequestBudget(None), generate, [])
    assert len(calls) == 1


def test_preflight_timeout_keeps_intent_without_fabricating_response(launcher, tmp_path):
    def generate(**kwargs):
        raise TimeoutError("offline injected")

    with pytest.raises(TimeoutError):
        launcher.recorded_probe(tmp_path, "evolver", "frozen-model", {}, RequestBudget(None), generate, [])
    assert (tmp_path / "evolver-intent.json").exists()
    assert not (tmp_path / "evolver-response.json").exists()


def test_wire_thinking_and_effort_must_match_frozen_config(launcher):
    body = {"model": "deepseek-flash", "thinking": {"type": "enabled"}, "reasoning_effort": "high"}
    launcher.verify_official_evolver_wire(body)
    for key, value in (("thinking", {"type": "disabled"}), ("reasoning_effort", "low"),
                       ("model", "deepseek-v4-pro"), ("tools", [{"type": "function"}])):
        with pytest.raises(ValueError, match="wire args"):
            launcher.verify_official_evolver_wire({**body, key: value})


def test_official_reasoning_high_survives_actual_tau_litellm_wire(launcher, monkeypatch):
    import httpx
    from tau2.data_model.message import SystemMessage, UserMessage
    from tau2.utils import llm_utils

    seen = []
    monkeypatch.setenv("OPENAI_API_KEY", "offline-placeholder")

    def send(client, request, *args, **kwargs):
        body = json.loads(request.content)
        launcher.verify_official_evolver_wire(body)
        seen.append(body)
        return httpx.Response(200, request=request, json={
            "id": "offline", "object": "chat.completion", "created": 1, "model": "deepseek-flash",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": '{"ok":true}'}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}})

    monkeypatch.setattr(httpx.Client, "send", send)
    frozen = freeze_role_model_args({"evolver": {"api_base": "https://api.deepseek.com/v1",
                                     "thinking_mode": "enabled", "reasoning_effort": "high"}}, roles=("evolver",))
    llm_utils.generate(model="openai/deepseek-flash", messages=[
        SystemMessage(role="system", content="JSON"), UserMessage(role="user", content="ok")],
        num_retries=0, **role_model_args_for_runtime(frozen)["evolver"])
    assert len(seen) == 1
    assert seen[0]["thinking"] == {"type": "enabled"}
    assert seen[0]["reasoning_effort"] == "high"


def test_wire_bridge_does_not_change_gateway_or_other_provider_args():
    original = {"agent": {"api_base": "http://10.130.138.46:8010/v1", "enable_thinking": False,
                           "temperature": 0.0},
                "evolver": {"api_base": "https://inferaiapi.com/v1", "reasoning_effort": "high"}}
    actual = role_model_args_for_runtime(freeze_role_model_args(original, roles=("agent", "evolver")))
    assert actual["agent"] == {"api_base": "http://10.130.138.46:8010/v1", "temperature": 0.0,
                                "extra_body": {"enable_thinking": False}}
    assert actual["evolver"] == original["evolver"]


def test_acquisition_consumes_native_result_tuple_without_reading_directory():
    import ast
    source = (Path(__file__).parents[1] /
              "experiments/execution/run-open-rc-acquisition.py").read_text()
    tree = ast.parse(source)
    assignment = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                      and isinstance(node.value.func, ast.Name)
                      and node.value.func.id == "run_from_config")
    assert [elt.id for elt in assignment.targets[0].elts] == ["output_directory", "result"]
    assert "result_path.read_text()" not in source
