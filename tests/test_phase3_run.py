from __future__ import annotations

import sys
from types import ModuleType

import pytest

from evotau.manifest import MechanismManifest
from evotau.phase3_run import ProviderPluginError, load_provider_bundle, run_from_config


def manifest() -> MechanismManifest:
    from pathlib import Path

    from evotau.phase0 import load_config

    config = load_config(Path(__file__).parents[1] / "configs/phase3-mechanism.yaml")
    experiment = config["experiment"]
    experiment["models"] = {
        "agent": "provider/agent",
        "customer": "provider/customer",
        "reviewer": "provider/reviewer",
        "evaluator": "provider/evaluator",
        "evolver": "provider/evolver",
    }
    experiment["real_provider_enabled"] = True
    return MechanismManifest.from_mapping(config)


def register_plugin(monkeypatch, factory):
    module = ModuleType("evotau_test_provider_plugin")
    module.build_providers = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module.__name__ + ":build_providers"


def test_provider_bundle_accepts_independent_audit_and_separate_repair_callbacks(monkeypatch) -> None:
    callbacks = {
        "audit_provider": lambda *_args: None,
        "service_proposal_provider": lambda *_args: None,
        "service_repair_audit_provider": lambda *_args: None,
    }
    received = []

    def factory(*, config, manifest):
        received.append((config, manifest))
        return callbacks

    specification = register_plugin(monkeypatch, factory)
    config = {"experiment": {"id": "fixture"}}
    frozen_manifest = manifest()
    result = load_provider_bundle(
        specification,
        config=config,
        manifest=frozen_manifest,
    )

    assert result == callbacks
    assert received == [(config, frozen_manifest)]


def test_provider_bundle_rejects_missing_audit_and_incomplete_repair_route(monkeypatch) -> None:
    specification = register_plugin(
        monkeypatch,
        lambda **_kwargs: {"service_transition": lambda *_args: None},
    )
    with pytest.raises(ProviderPluginError, match="audit_provider"):
        load_provider_bundle(specification, config={}, manifest=manifest())

    specification = register_plugin(
        monkeypatch,
        lambda **_kwargs: {
            "audit_provider": lambda *_args: None,
            "service_proposal_provider": lambda *_args: None,
        },
    )
    with pytest.raises(ProviderPluginError, match="both Service repair callbacks"):
        load_provider_bundle(specification, config={}, manifest=manifest())


def test_provider_bundle_rejects_conflicting_routes_and_unknown_callbacks(monkeypatch) -> None:
    base = {"audit_provider": lambda *_args: None}
    specification = register_plugin(
        monkeypatch,
        lambda **_kwargs: {
            **base,
            "service_transition": lambda *_args: None,
            "service_proposal_provider": lambda *_args: None,
            "service_repair_audit_provider": lambda *_args: None,
        },
    )
    with pytest.raises(ProviderPluginError, match="either service_transition"):
        load_provider_bundle(specification, config={}, manifest=manifest())

    specification = register_plugin(
        monkeypatch,
        lambda **_kwargs: {**base, "unreviewed_callback": lambda: None},
    )
    with pytest.raises(ProviderPluginError, match="unsupported keys"):
        load_provider_bundle(specification, config={}, manifest=manifest())


def test_phase3_cli_runner_refuses_provider_disabled_config_before_loading_plugin(tmp_path) -> None:
    import shutil
    from pathlib import Path

    source = Path(__file__).parents[1] / "configs/phase3-mechanism.yaml"
    config = tmp_path / "phase3-disabled.yaml"
    shutil.copyfile(source, config)

    with pytest.raises(RuntimeError, match="disabled"):
        run_from_config(
            config,
            phase0_result_path=tmp_path / "missing-phase0-result.json",
            tau2_data_dir=None,
            provider_plugin="does_not_import:factory",
        )

