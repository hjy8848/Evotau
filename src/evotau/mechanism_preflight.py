"""Offline preflight for the frozen two-generation mechanism manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from .eligibility import validate_smoke_selection
from .manifest import MechanismManifest, verify_git_blob_sha1, write_manifest_once
from .phase0 import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase3-mechanism.yaml"))
    parser.add_argument("--tasks-json", type=Path)
    parser.add_argument("--split-json", type=Path)
    parser.add_argument("--manifest-out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        manifest = MechanismManifest.from_mapping(config)
        if bool(args.tasks_json) != bool(args.split_json):
            raise ValueError("--tasks-json and --split-json must be supplied together")
        selection_validation: str | dict[str, object] = "not checked: provide pinned task and split files"
        if args.tasks_json and args.split_json:
            sources = dict(manifest.source_blob_sha1)
            verify_git_blob_sha1(args.tasks_json, sources["data/tau2/domains/retail/tasks.json"])
            verify_git_blob_sha1(args.split_json, sources["data/tau2/domains/retail/split_tasks.json"])
            tasks = json.loads(args.tasks_json.read_text(encoding="utf-8"))
            splits = json.loads(args.split_json.read_text(encoding="utf-8"))
            eligible = validate_smoke_selection(
                tasks, splits, evolution_task_id=manifest.evolution_task_id,
                validation_task_id=manifest.validation_task_id,
                excluded_task_ids=config["experiment"]["task_selection"].get("excluded", ()),
            )
            selection_validation = {
                "evolution_task_id": eligible.evolution_task_id,
                "validation_task_id": eligible.validation_task_id,
                "evolution_entity_keys": list(eligible.evolution_entity_keys),
                "validation_entity_keys": list(eligible.validation_entity_keys),
            }
        if args.manifest_out:
            write_manifest_once(args.manifest_out, manifest)
        print(json.dumps({
            "manifest_sha256": manifest.sha256,
            "experiment_id": manifest.experiment_id,
            "upstream_commit": manifest.upstream_commit,
            "max_episodes": manifest.max_episodes,
            "request_budget_cap": manifest.request_budget_cap,
            "real_provider_enabled": manifest.real_provider_enabled,
            "selection_validation": selection_validation,
        }, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
        print(f"Mechanism preflight failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
