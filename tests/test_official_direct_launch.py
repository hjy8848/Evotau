"""Native-only import and credential isolation tests. No network."""

import importlib.util
import json
import os
from copy import deepcopy
from pathlib import Path

import httpx
import pytest
import yaml

from evotau.alternating_manifest import AlternatingManifest
from evotau.tau_provenance import sha256_json

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    path = ROOT / "experiments/execution" / name
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_official_auth_never_overwrites_runtime_key(tmp_path, monkeypatch):
    launcher = module("run-v2-official-deepseek-evolver.py")
    read = Path.read_text

    def read_binding(path, *args, **kwargs):
        if str(path) == "/tmp/evotau-v2-live-process.json":
            return json.dumps({"pid": os.getpid(), "output": str(tmp_path)})
        return read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_binding)
    observed = []

    def transport(client, request, *args, **kwargs):
        observed.append((request.url.host, request.headers["Authorization"]))
        return httpx.Response(200, request=request, json={"choices": []})

    send = launcher.install_official_transport("official-test-secret", transport)
    native = httpx.Request(
        "POST",
        "https://inferaiapi.com/v1/chat/completions",
        headers={"Authorization": "Bearer native-test-secret"},
        json={"model": "qwen3.7-plus"},
    )
    official = httpx.Request(
        "POST",
        "https://api.deepseek.com/v1/chat/completions",
        headers={"Authorization": "Bearer native-test-secret"},
        json={"model": "deepseek-flash", "thinking": {"type": "enabled"}},
    )
    send(None, native)
    send(None, official)
    assert observed == [
        ("inferaiapi.com", "Bearer native-test-secret"),
        ("api.deepseek.com", "Bearer official-test-secret"),
    ]
    assert "test-secret" not in (tmp_path / "actual-provider-http.jsonl").read_text()
    invalid = httpx.Request(
        "POST",
        "https://api.deepseek.com/v1/chat/completions",
        json={"model": "invented"},
    )
    with pytest.raises(ValueError, match="scoped"):
        send(None, invalid)
    assert len(observed) == 2


def test_native_import_contract_ignores_only_external_search_changes():
    importer = module("import-direct-native-baseline.py")
    raw = yaml.safe_load(
        (
            ROOT
            / "configs/v2-direct-skill-qwen37plus-official-dsflash-e20-v3-h5-g2-p1.yaml"
        ).read_text()
    )
    doc = AlternatingManifest.from_mapping(raw).to_document()
    changed = deepcopy(doc)
    changed["role_models"]["evolver"] = "historical-model"
    changed["skill_evolution_v2"]["algorithm_version"] = "diagnoser_v2"
    assert importer.native_condition(doc) == importer.native_condition(changed)
    for key, value in [("max_steps", 31), ("evolution_fitness_seed", 2)]:
        changed = deepcopy(doc)
        changed[key] = value
        assert importer.native_condition(doc) != importer.native_condition(changed)
    changed = deepcopy(doc)
    changed["role_model_args"]["agent"]["temperature"] = 0.1
    assert importer.native_condition(doc) != importer.native_condition(changed)


def test_completion_rebind_preserves_native_record_and_recomputes_binding_digest():
    importer = module("import-direct-native-baseline.py")
    value = {
        "manifest_sha256": "parent",
        "record": {"task_id": "66", "task_success": False},
        "activation_trace": {"manifest_sha256": "parent"},
    }
    value["activation_trace"]["artifact_sha256"] = sha256_json(
        value["activation_trace"]
    )
    value["completion_sha256"] = sha256_json(value)
    before = deepcopy(value)
    rebound = importer.rebind(value, "parent", "child")
    assert value == before and rebound["record"] == value["record"]
    assert rebound["manifest_sha256"] == "child"
    assert rebound["completion_sha256"] == sha256_json(
        {k: v for k, v in rebound.items() if k != "completion_sha256"}
    )
