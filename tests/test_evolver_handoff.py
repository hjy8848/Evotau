"""Prevent an Evolver handoff from disguising a changed native treatment."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from evotau.alternating_manifest import AlternatingManifest
from evotau.tau_provenance import sha256_json

ROOT = Path(__file__).resolve().parents[1]


def helper():
    path = ROOT / "experiments/execution/import-v2-evolver-handoff.py"
    spec = importlib.util.spec_from_file_location("handoff", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def documents():
    return tuple(
        AlternatingManifest.from_mapping(
            yaml.safe_load((ROOT / "configs" / name).read_text())
        ).to_document()
        for name in (
            "alternating-skill-memory-v2-qwen37plus-v4pro-e20-g2-live.yaml",
            "alternating-skill-memory-v2-qwen37plus-gpt61sol-e20-g2-handoff.yaml",
        )
    )


@pytest.mark.parametrize(
    "change",
    ["agent_model", "customer_thinking", "gate_method", "seed", "max_steps", "panel"],
)
def test_handoff_rejects_changed_native_or_selection_conditions(change):
    module = helper()
    parent, child = documents()
    module.validate_contract(parent, child)
    child = deepcopy(child)
    if change == "agent_model":
        child["role_models"]["agent"] = "openai/deepseek-v4-flash"
    elif change == "customer_thinking":
        child["role_model_args"]["customer"]["thinking_mode"] = "enabled"
    elif change == "gate_method":
        child["skill_evolution_v2"]["statistical_gate"]["method"] = (
            "task_block_bootstrap"
        )
    elif change == "seed":
        child["evolution_fitness_seed"] = 2
    elif change == "max_steps":
        child["max_steps"] = 64
    else:
        child["task_panels"]["E"] = child["task_panels"]["E"][:-1]
    with pytest.raises(ValueError, match="preserve every native/search/gate input"):
        module.validate_contract(parent, child)


def test_handoff_preserves_native_payload_and_rehashes_only_binding():
    module = helper()
    payload = {
        "episodes": [
            {"task_id": "66", "seed": 1, "task_success": False, "native_reward": 0.0}
        ],
        "observed_text": "old-manifest-is-data",
    }
    doc = {
        "manifest_sha256": "old",
        "payload": payload,
        "payload_sha256": sha256_json(payload),
        "input_sha256": "same-input",
    }
    doc["envelope_sha256"] = sha256_json(doc)
    original = deepcopy(doc)
    new = module.rebind(doc, "old", "new")
    assert doc == original and new["payload"] == payload
    assert (
        new["input_sha256"] == doc["input_sha256"]
        and new["payload_sha256"] == doc["payload_sha256"]
    )
    assert new["manifest_sha256"] == "new"
    assert new["envelope_sha256"] == sha256_json(
        {k: v for k, v in new.items() if k != "envelope_sha256"}
    )
    decision = {
        "manifest_sha256": "old",
        "decision": {"active_skill_ids": []},
        "native_prompt_sha256": "unchanged",
    }
    decision["artifact_sha256"] = sha256_json(decision)
    trace = {"manifest_sha256": "old", "decisions": [decision]}
    trace["artifact_sha256"] = sha256_json(trace)
    rebound = module.rebind(trace, "old", "new")
    assert rebound["decisions"][0]["decision"] == decision["decision"]
    assert rebound["decisions"][0]["native_prompt_sha256"] == "unchanged"
    assert rebound["decisions"][0]["artifact_sha256"] == sha256_json(
        {k: v for k, v in rebound["decisions"][0].items() if k != "artifact_sha256"}
    )


def test_handoff_rejects_corrupt_source_manifest_and_wrong_fallback():
    module = helper()
    parent, child = documents()
    parent["max_steps"] = 500
    with pytest.raises(ValueError, match="parent manifest digest"):
        module.validate_contract(parent, child)
    parent, child = documents()
    child["role_models"]["evolver"] = "openai/gpt-6-sol"
    with pytest.raises(ValueError, match="explicitly authorized GPT"):
        module.validate_contract(parent, child)
