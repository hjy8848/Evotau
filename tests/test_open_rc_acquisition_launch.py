"""No paid requests: acquisition isolation and durable preflight replay rules."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from evotau.budget import RequestBudget


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
