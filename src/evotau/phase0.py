"""Offline Phase 0 manifest and eligibility preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from .manifest import ExperimentManifest
from .tau_provenance import verify_git_blob_sha1, write_manifest_once


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError("configuration root must be a mapping")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mvp.yaml"),
        help="Phase 0 YAML configuration",
    )
    parser.add_argument("--tasks-json", type=Path)
    parser.add_argument("--split-json", type=Path)
    parser.add_argument(
        "--manifest-out",
        type=Path,
        help="write the manifest as a new immutable file; existing paths are rejected",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        manifest = ExperimentManifest.from_mapping(config)
        if bool(args.tasks_json) != bool(args.split_json):
            raise ValueError("--tasks-json and --split-json must be supplied together")
        if args.tasks_json and args.split_json:
            source_blobs = dict(manifest.source_blob_sha1)
            verify_git_blob_sha1(
                args.tasks_json,
                source_blobs["data/tau2/domains/retail/tasks.json"],
            )
            verify_git_blob_sha1(
                args.split_json,
                source_blobs["data/tau2/domains/retail/split_tasks.json"],
            )
            with args.tasks_json.open("r", encoding="utf-8") as handle:
                tasks = json.load(handle)
            with args.split_json.open("r", encoding="utf-8") as handle:
                splits = json.load(handle)
            selection = config["experiment"]["task_selection"]
            selected = {
                str(task_id)
                for task_id in (*selection["evolution"], *selection["validation"])
            }
            train_ids = {str(task_id) for task_id in splits.get(manifest.split_name, ())}
            task_ids = {
                str(task.get("id"))
                for task in tasks
                if isinstance(task, dict) and task.get("id") is not None
            }
            expected_count = len(manifest.evolution_task_ids) + len(manifest.validation_task_ids)
            if len(selected) != expected_count:
                raise ValueError("E and V task IDs must be distinct")
            if selected - train_ids or selected - task_ids:
                raise ValueError("Phase 0 tasks must exist in the pinned official train split")
            selection_result = {
                "evolution_task_ids": list(manifest.evolution_task_ids),
                "validation_task_ids": list(manifest.validation_task_ids),
                "official_train_split": True,
            }
        else:
            selection_result = "not checked: provide the pinned task and split files"
        if args.manifest_out:
            write_manifest_once(args.manifest_out, manifest)
        print(
            json.dumps(
                {
                    "manifest_sha256": manifest.sha256,
                    "experiment_id": manifest.experiment_id,
                    "upstream_commit": manifest.upstream_commit,
                    "evaluation_type": manifest.evaluation_type,
                    "real_provider_enabled": manifest.real_provider_enabled,
                    "selection_validation": selection_result,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
        print(f"Phase 0 preflight failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
