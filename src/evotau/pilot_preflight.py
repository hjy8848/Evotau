"""Offline validation for a frozen multi-task, multi-seed Phase 4 Pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .eligibility import validate_generalization_selection
from .manifest import PilotManifest, verify_git_blob_sha1
from .phase0 import load_config
from .phase0_run import _write_json_once


def validate_pilot_config(
    config: dict[str, Any], *, data_dir: str | Path | None = None,
) -> dict[str, Any]:
    manifest = PilotManifest.from_mapping(config)
    selection: dict[str, Any] | str = "not_checked_provide_pinned_tau2_data_dir"
    if data_dir is not None:
        data_root = Path(data_dir).expanduser().resolve()
        task_path = data_root / "tau2/domains/retail/tasks.json"
        split_path = data_root / "tau2/domains/retail/split_tasks.json"
        blobs = dict(manifest.source_blob_sha1)
        verify_git_blob_sha1(task_path, blobs["data/tau2/domains/retail/tasks.json"])
        verify_git_blob_sha1(split_path, blobs["data/tau2/domains/retail/split_tasks.json"])
        tasks = json.loads(task_path.read_text(encoding="utf-8"))
        split = json.loads(split_path.read_text(encoding="utf-8"))
        selected = validate_generalization_selection(
            tasks, split,
            evolution_task_ids=manifest.evolution_task_ids,
            validation_task_ids=manifest.validation_task_ids,
            heldout_task_ids=manifest.heldout_task_ids,
            excluded_task_ids=manifest.excluded_task_ids,
        )
        review_document = json.loads(manifest.task_semantic_review_json)
        from .task_review import verify_reviewed_task_hashes

        verify_reviewed_task_hashes(review_document, tasks)
        selection = {
            "status": "eligible_partition_validated",
            "evolution_task_ids": list(selected.evolution_task_ids),
            "validation_task_ids": list(selected.validation_task_ids),
            "heldout_task_ids": list(selected.heldout_task_ids),
            "evolution_entity_keys": list(selected.evolution_entity_keys),
            "validation_entity_keys": list(selected.validation_entity_keys),
            "heldout_entity_keys": list(selected.heldout_entity_keys),
            "task_file_sha1": blobs["data/tau2/domains/retail/tasks.json"],
            "split_file_sha1": blobs["data/tau2/domains/retail/split_tasks.json"],
        }
    return {
        "schema_version": 1,
        "status": "manifest_valid",
        "experiment_id": manifest.experiment_id,
        "condition": manifest.condition,
        "manifest_sha256": manifest.sha256,
        "evolution_seeds": list(manifest.evolution_seeds),
        "generation_count": manifest.generations,
        "max_episodes_per_seed": manifest.max_episodes,
        "request_budget_cap_per_seed": manifest.request_budget_cap,
        "real_provider_enabled": manifest.real_provider_enabled,
        "task_semantic_review": {
            "status": (
                "pinned_task_hashes_verified" if data_dir is not None
                else "manifest_bound_not_checked_against_pinned_tasks"
            ),
            "review_id": json.loads(manifest.task_semantic_review_json)["review_id"],
            "sha256": manifest.task_semantic_review_sha256,
            "task_count": len(json.loads(manifest.task_semantic_review_json)["task_reviews"]),
            "pairwise_review_count": len(
                json.loads(manifest.task_semantic_review_json)["pairwise_reviews"]
            ),
        },
        "task_selection": selection,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tau2-data-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = validate_pilot_config(load_config(args.config), data_dir=args.tau2_data_dir)
        if args.output is None:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            _write_json_once(args.output, result)
            print(json.dumps({"status": result["status"], "output": str(args.output)}))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Pilot preflight failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
