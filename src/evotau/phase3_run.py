"""Explicit, plugin-backed CLI for a manifest-bound native Phase 3 smoke."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml

from .manifest import MechanismManifest
from .native_runner import _validate_phase0_parent, run_native_phase3
from .phase0 import load_config
from .phase0_run import _load_pinned_tasks

_PROVIDER_KEYS = frozenset({
    "audit_provider",
    "customer_proposal_provider",
    "service_transition",
    "service_proposal_provider",
    "service_repair_audit_provider",
})


class ProviderPluginError(ValueError):
    """The explicitly selected provider plugin does not meet the callback contract."""


def load_provider_bundle(
    specification: str,
    *,
    config: Mapping[str, Any],
    manifest: MechanismManifest,
) -> dict[str, Callable[..., Any]]:
    """Load a caller-selected `module:factory` returning native runner callbacks."""

    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name.strip() or not attribute.strip():
        raise ProviderPluginError("provider plugin must use the module:factory form")
    try:
        module = importlib.import_module(module_name.strip())
    except Exception as exc:
        raise ProviderPluginError("provider plugin module could not be imported") from exc
    factory = getattr(module, attribute.strip(), None)
    if not callable(factory):
        raise ProviderPluginError("provider plugin factory is not callable")
    try:
        callbacks = factory(config=config, manifest=manifest)
    except Exception as exc:
        raise ProviderPluginError(
            f"provider plugin factory failed ({type(exc).__name__})"
        ) from exc
    if not isinstance(callbacks, Mapping):
        raise ProviderPluginError("provider plugin factory must return a callback mapping")
    if any(not isinstance(key, str) for key in callbacks):
        raise ProviderPluginError("provider plugin callback keys must be strings")
    unknown = set(callbacks) - _PROVIDER_KEYS
    if unknown:
        raise ProviderPluginError(f"provider plugin returned unsupported keys: {sorted(unknown)}")
    bundle = {key: value for key, value in callbacks.items() if value is not None}
    invalid = sorted(key for key, value in bundle.items() if not callable(value))
    if invalid:
        raise ProviderPluginError(f"provider callbacks must be callable: {invalid}")
    if "audit_provider" not in bundle:
        raise ProviderPluginError("provider plugin must return an independent audit_provider")

    has_transition = "service_transition" in bundle
    repair_keys = {"service_proposal_provider", "service_repair_audit_provider"}
    has_repair_providers = repair_keys <= set(bundle)
    if bool(set(bundle) & repair_keys) and not has_repair_providers:
        raise ProviderPluginError("both Service repair callbacks must be provided together")
    if has_transition and set(bundle) & repair_keys:
        raise ProviderPluginError(
            "provide either service_transition or both repair callbacks, not both"
        )
    if not has_transition and not has_repair_providers:
        raise ProviderPluginError(
            "provider plugin must return service_transition or both Service repair callbacks"
        )
    return bundle


def run_from_config(
    config_path: str | Path,
    *,
    phase0_result_path: str | Path,
    tau2_data_dir: str | Path | None,
    provider_plugin: str,
) -> tuple[tuple[Any, ...], Any, MechanismManifest]:
    """Validate all frozen inputs before loading callbacks or dispatching episodes."""

    config = load_config(config_path)
    manifest = MechanismManifest.from_mapping(config)
    if not manifest.real_provider_enabled:
        raise RuntimeError("native Phase 3 is disabled in the frozen manifest")
    data_dir = tau2_data_dir or os.environ.get("TAU2_DATA_DIR")
    if data_dir is None:
        raise ValueError("provide --tau2-data-dir or set TAU2_DATA_DIR to the pinned data directory")

    selection = config["experiment"].get("task_selection", {})
    _load_pinned_tasks(
        manifest,
        data_dir=data_dir,
        task_selection=selection,
        task_ids=(manifest.evolution_task_id, manifest.validation_task_id),
    )
    _validate_phase0_parent(phase0_result_path, manifest)
    providers = load_provider_bundle(
        provider_plugin,
        config=config,
        manifest=manifest,
    )
    commits, budget = run_native_phase3(
        config_path=config_path,
        data_dir=data_dir,
        phase0_result_path=phase0_result_path,
        **providers,
    )
    return commits, budget, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase3-mechanism.yaml"),
    )
    parser.add_argument("--phase0-result", type=Path, required=True)
    parser.add_argument(
        "--tau2-data-dir",
        type=Path,
        help="path to data/ from the pinned tau-bench checkout (or set TAU2_DATA_DIR)",
    )
    parser.add_argument(
        "--provider-plugin",
        required=True,
        help="Python module:factory returning independent audit and Service callbacks",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        commits, budget, manifest = run_from_config(
            args.config,
            phase0_result_path=args.phase0_result,
            tau2_data_dir=args.tau2_data_dir,
            provider_plugin=args.provider_plugin,
        )
    except (OSError, ValueError, RuntimeError, KeyError, yaml.YAMLError) as exc:
        print(f"Phase 3 run failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - provider SDK errors must not expose secrets
        print(
            f"Phase 3 run failed ({type(exc).__name__}); inspect the immutable run artifacts",
            file=sys.stderr,
        )
        return 2
    result_path = Path(manifest.output_path) / "phase3-result.json"
    print(json.dumps({
        "status": "complete",
        "experiment_id": manifest.experiment_id,
        "manifest_sha256": manifest.sha256,
        "generation_count": len(commits),
        "provider_budget": budget.to_dict(),
        "result_path": str(result_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
