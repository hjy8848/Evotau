"""The controlled launcher freezes treatments and resumes without new rollouts."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from evotau.records import EpisodeRecord, EpisodeStatus


def test_frozen_morphology_treatments_resume_and_reject_changes(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    directory = root / "experiments/execution"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "morphology", directory / "run-v2-activation-morphology-smoke.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = yaml.safe_load(
        (
            root / "configs/skill-evolution-v2-activation-morphology-smoke.yaml"
        ).read_text()
    )
    historic = root / raw["measurement_protocol"]["historical_source"]
    source = tmp_path / "source.json"
    source.write_bytes(historic.read_bytes())
    raw["measurement_protocol"]["historical_source"] = "source.json"
    raw["measurement_protocol"]["historical_sha256"] = hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(
        module,
        "load_alternating_tasks",
        lambda manifest, *args, **kwargs: {
            task: {} for task in manifest.evolution_task_ids
        },
    )
    monkeypatch.setattr(module, "install_transport", lambda output: None)
    # Never access real Keychain/network in this test.
    actual_subprocess = module.subprocess.run
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: (
            type("Credential", (), {"stdout": "test-secret"})()
            if args[0][0] == "security"
            else actual_subprocess(*args, **kwargs)
        ),
    )
    rollouts, calls, seen = [], [], {}

    class Provider:
        def __init__(self, unused):
            pass

        def call(self, prompt, context, name):
            calls.append(name)
            if name == "evotau_provider_preflight":
                return {"ok": True}
            return {
                "activation_signature": {
                    "positive_conditions": ["The user adds scope before confirmation."],
                    "negative_conditions": [
                        "The unchanged action is already confirmed."
                    ],
                    "interaction_phase": ["before_confirmation"],
                },
                "reason": "Narrowed observable boundary.",
            }

        def validate_skill(self, context):
            calls.append("validate")
            return dict.fromkeys(
                ("reusable", "policy_subordinate", "no_task_entities"), True
            )

    class Runner:
        service_policy_text = "Native policy."

        def __init__(self, **kwargs):
            self.manifest = kwargs["manifest"]
            seen[self.manifest.experiment_id] = self.manifest.sha256

        def __call__(self, *, task_id, seed, customer, service, panel_name):
            rollouts.append((panel_name, task_id, seed, service.to_dict()))
            return EpisodeRecord(
                f"{panel_name}-{task_id}-{seed}",
                task_id,
                seed,
                "c",
                "s",
                EpisodeStatus.COMPLETE,
                task_id != "80",
                total_steps=5,
            )

        def load_trajectory(self, record):
            return {
                "messages": [{"role": "user", "content": "Please explain the policy."}]
            }

    monkeypatch.setattr(module, "V2Providers", Provider)
    monkeypatch.setattr(module, "TauBenchEpisodeRunner", Runner)
    monkeypatch.setattr(
        sys,
        "argv",
        ["smoke", "--config", str(config), "--tau2-data-dir", str(tmp_path)],
    )
    # Simulate interruption after baseline publication, before signature completion.
    original = Provider.call

    def interrupt(self, prompt, context, name):
        if name == "evotau_service_skill_mutator":
            raise RuntimeError("simulated interruption")
        return original(self, prompt, context, name)

    monkeypatch.setattr(Provider, "call", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        module.main()
    assert len(rollouts) == 12
    assert len(list((tmp_path / "experiments/runs").glob("*/manifest.json"))) == 3
    monkeypatch.setattr(Provider, "call", original)
    module.main()
    assert len(rollouts) == 36  # The complete baseline was not rerun.
    assert len(set(seen.values())) == 3
    empty = [r for r in rollouts if r[0] == "morphology-empty"]
    render = [r for r in rollouts if r[0] == "morphology-render-all"]
    activation = [r for r in rollouts if r[0] == "morphology-activation"]
    assert all(not r[3]["skills"] for r in empty)
    assert render[0][3] == activation[0][3]
    result = json.loads(
        (
            tmp_path / raw["experiment"]["output_path"] / "mechanism-smoke-result.json"
        ).read_text()
    )
    assert result["completed_episodes"] == 36
    assert not result["V_evaluated"] and not result["H_evaluated"]
    before = list(calls)
    module.main()
    assert len(rollouts) == 36 and calls == before
    raw["experiment"]["max_steps"] = 30
    config.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="different frozen execution inputs"):
        module.main()
