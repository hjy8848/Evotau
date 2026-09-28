"""Plugin-backed native executor for a frozen Phase 4 Pilot condition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .manifest import PilotManifest
from .native_runner import run_native_pilot
from .phase0 import load_config
from .phase3_run import load_provider_bundle


def run_from_config(
    config_path: str | Path,
    *,
    tau2_data_dir: str | Path,
    provider_plugin: str,
) -> dict[str, Any]:
    config = load_config(config_path)
    manifest = PilotManifest.from_mapping(config)

    def callback_factory(_config, _manifest, seed):
        bundle = load_provider_bundle(
            provider_plugin,
            config=config,
            manifest=manifest,
            seed=seed,
        )
        return {**bundle.callbacks, "provider_provenance": bundle.provenance}

    return run_native_pilot(
        config_path=config_path,
        data_dir=tau2_data_dir,
        callback_factory=callback_factory,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--tau2-data-dir", type=Path,
        help="path to data/ from the pinned τ-bench checkout; may use TAU2_DATA_DIR",
    )
    parser.add_argument(
        "--provider-plugin", required=True,
        help="module:factory returning per-seed independent audit and Service callbacks",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        data_dir = args.tau2_data_dir
        if data_dir is None:
            import os

            data_dir = os.environ.get("TAU2_DATA_DIR")
        if data_dir is None:
            raise ValueError("provide --tau2-data-dir or set TAU2_DATA_DIR")
        result = run_from_config(
            args.config,
            tau2_data_dir=data_dir,
            provider_plugin=args.provider_plugin,
        )
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Pilot run failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - avoid exposing provider credentials
        print(
            f"Pilot run failed ({type(exc).__name__}); inspect immutable seed-block artifacts",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
