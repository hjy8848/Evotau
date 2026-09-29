"""Plugin-backed runner for the independent one-generation activation smoke."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .manifest import ActivationSmokeManifest
from .native_runner import StopBeforeEpisodeDispatch, run_native_activation_smoke
from .phase0 import load_config
from .phase0_run import _load_pinned_tasks
from .phase3_run import load_provider_bundle


def _project_root(config_path: Path) -> Path:
    resolved = config_path.expanduser().resolve()
    return resolved.parent.parent if resolved.parent.name == "configs" else resolved.parent


def run_from_config(
    config_path: str | Path,
    *,
    tau2_data_dir: str | Path,
    provider_plugin: str,
    stop_before_next_episode_file: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the human-reviewed E/V panel and pinned data before providers load."""

    config_file = Path(config_path).expanduser().resolve()
    project_root = _project_root(config_file)
    config = load_config(config_file)
    review_relative = str(config["experiment"]["task_review_path"])
    unresolved_review_path = project_root / review_relative
    current = project_root
    for part in Path(review_relative).parts:
        if part in {"", "."}:
            continue
        current = current / part
        if current.is_symlink():
            raise ValueError("activation task-review path cannot traverse symlinks")
    review_path = unresolved_review_path.resolve()
    if not review_path.is_relative_to(project_root):
        raise ValueError("activation task-review artifact must be a regular project file")
    if not review_path.is_file():
        raise FileNotFoundError(
            "activation requires the existing task-review workflow to freeze ten human-approved E tasks"
        )
    task_review_document = json.loads(review_path.read_text(encoding="utf-8"))
    manifest = ActivationSmokeManifest.from_mapping(
        config, task_review_document=task_review_document,
    )
    if not manifest.real_provider_enabled:
        raise RuntimeError("activation smoke is disabled in the frozen config")

    data_root = Path(tau2_data_dir).expanduser().resolve()
    _load_pinned_tasks(
        manifest,
        data_dir=data_root,
        task_selection=config["experiment"]["task_selection"],
        task_ids=manifest.evolution_task_ids + manifest.validation_task_ids,
    )
    bundle = load_provider_bundle(
        provider_plugin,
        config=config,
        manifest=manifest,
    )
    run_native_activation_smoke(
        config_path=config_file,
        data_dir=data_root,
        task_review_document=task_review_document,
        provider_provenance=bundle.provenance,
        stop_before_next_episode_file=stop_before_next_episode_file,
        **bundle.callbacks,
    )
    result_path = project_root / manifest.output_path / "activation-smoke-result.json"
    return json.loads(result_path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/activation-smoke-inferai-deepseek-v4-flash.yaml"),
    )
    parser.add_argument(
        "--tau2-data-dir",
        type=Path,
        help="path to data/ from the pinned τ-bench checkout; may use TAU2_DATA_DIR",
    )
    parser.add_argument(
        "--provider-plugin",
        required=True,
        help="module:factory returning the independent audit and Service repair providers",
    )
    parser.add_argument(
        "--stop-before-next-episode-file",
        type=Path,
        help="optional Console control file checked before native episode dispatch",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        data_dir = args.tau2_data_dir or os.environ.get("TAU2_DATA_DIR")
        if data_dir is None:
            raise ValueError("provide --tau2-data-dir or set TAU2_DATA_DIR")
        result = run_from_config(
            args.config,
            tau2_data_dir=data_dir,
            provider_plugin=args.provider_plugin,
            stop_before_next_episode_file=args.stop_before_next_episode_file,
        )
    except StopBeforeEpisodeDispatch:
        print("Activation smoke paused before the next episode dispatch.")
        return 75
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Activation smoke run failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - avoid exposing provider credentials
        print(
            f"Activation smoke run failed ({type(exc).__name__}); inspect immutable run artifacts",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
