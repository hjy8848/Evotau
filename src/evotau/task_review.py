"""Strict, task-bound records for the Pilot's required ex-ante human review."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

TASK_REVIEW_FIELDS = {
    "task_id", "panel", "task_sha256", "no_deception_required",
    "policy_tool_compatible", "satisfiable", "rationale",
}
PAIR_REVIEW_FIELDS = {"left_task_id", "right_task_id", "distinct_scenario", "rationale"}


def validate_task_semantic_review(
    value: Any,
    *,
    expected_panels: tuple[tuple[str, str], ...],
) -> tuple[dict[str, Any], str]:
    """Validate and fingerprint a human review bound to the exact E/V/H design."""

    if not isinstance(value, dict) or set(value) != {
        "schema_version", "review_id", "reviewer_id", "reviewed_at",
        "task_reviews", "pairwise_reviews",
    }:
        raise ValueError("task semantic review has missing or unknown fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported task semantic review schema_version")
    review_id = _nonempty_text(value["review_id"], "review_id")
    reviewer_id = _nonempty_text(value["reviewer_id"], "reviewer_id")
    reviewed_at = _validate_utc_timestamp(value["reviewed_at"])
    if not isinstance(value["task_reviews"], list):
        raise TypeError("task_reviews must be a JSON array")
    if not isinstance(value["pairwise_reviews"], list):
        raise TypeError("pairwise_reviews must be a JSON array")

    expected_panels = _normalize_expected_panels(expected_panels)
    expected_task_ids = tuple(task_id for task_id, _panel in expected_panels)
    if not expected_task_ids or len(set(expected_task_ids)) != len(expected_task_ids):
        raise ValueError("expected task panels must contain unique non-empty Pilot task IDs")
    if len(value["task_reviews"]) != len(expected_panels):
        raise ValueError("task semantic review must cover every selected E/V/H task exactly once")

    normalized_task_reviews = []
    for row, (expected_task_id, expected_panel) in zip(
        value["task_reviews"], expected_panels, strict=True,
    ):
        if not isinstance(row, dict) or set(row) != TASK_REVIEW_FIELDS:
            raise ValueError("task review row has missing or unknown fields")
        task_id = _nonempty_text(row["task_id"], "task_id")
        panel = row["panel"]
        if (task_id, panel) != (expected_task_id, expected_panel):
            raise ValueError("task review order or panel assignment differs from the frozen E/V/H design")
        task_sha256 = row["task_sha256"]
        if not isinstance(task_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", task_sha256):
            raise ValueError("task review task_sha256 must be a lowercase SHA-256 digest")
        for field in ("no_deception_required", "policy_tool_compatible", "satisfiable"):
            if type(row[field]) is not bool:
                raise TypeError(f"task review {field} must be boolean")
            if row[field] is not True:
                raise ValueError(f"task {task_id} failed required ex-ante review: {field}")
        normalized_task_reviews.append({
            "task_id": task_id,
            "panel": panel,
            "task_sha256": task_sha256,
            "no_deception_required": True,
            "policy_tool_compatible": True,
            "satisfiable": True,
            "rationale": _review_rationale(row["rationale"], "task rationale"),
        })

    expected_pairs = tuple(combinations(expected_task_ids, 2))
    if len(value["pairwise_reviews"]) != len(expected_pairs):
        raise ValueError("task semantic review must assess every selected task pair")
    normalized_pairwise_reviews = []
    for row, (expected_left, expected_right) in zip(
        value["pairwise_reviews"], expected_pairs, strict=True,
    ):
        if not isinstance(row, dict) or set(row) != PAIR_REVIEW_FIELDS:
            raise ValueError("pairwise task review row has missing or unknown fields")
        pair = (
            _nonempty_text(row["left_task_id"], "left_task_id"),
            _nonempty_text(row["right_task_id"], "right_task_id"),
        )
        if pair != (expected_left, expected_right):
            raise ValueError("pairwise task review order differs from the frozen panel order")
        if type(row["distinct_scenario"]) is not bool:
            raise TypeError("pairwise distinct_scenario must be boolean")
        if row["distinct_scenario"] is not True:
            raise ValueError(
                f"selected tasks {expected_left} and {expected_right} are semantically overlapping"
            )
        normalized_pairwise_reviews.append({
            "left_task_id": pair[0],
            "right_task_id": pair[1],
            "distinct_scenario": True,
            "rationale": _review_rationale(row["rationale"], "pairwise rationale"),
        })

    normalized = {
        "schema_version": 1,
        "review_id": review_id,
        "reviewer_id": reviewer_id,
        "reviewed_at": reviewed_at,
        "task_reviews": normalized_task_reviews,
        "pairwise_reviews": normalized_pairwise_reviews,
    }
    from .manifest import canonical_json

    digest = hashlib.sha256(canonical_json(normalized).encode("utf-8")).hexdigest()
    return normalized, digest


def build_task_semantic_review_template(
    tasks: list[dict[str, Any]],
    *,
    expected_panels: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    """Build an explicitly unapproved review form for a frozen E/V/H selection."""

    expected_panels = _normalize_expected_panels(expected_panels)
    task_map: dict[str, dict[str, Any]] = {}
    for task in tasks:
        if not isinstance(task, dict):
            raise TypeError("pinned task data must contain task objects")
        task_id = task.get("id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("pinned task data contains a missing or invalid task ID")
        if task_id in task_map:
            raise ValueError("pinned task data contains duplicate task IDs")
        task_map[task_id] = task
    missing = [task_id for task_id, _panel in expected_panels if task_id not in task_map]
    if missing:
        raise ValueError(f"pinned task data is missing selected tasks: {', '.join(missing)}")

    from .manifest import sha256_json

    task_ids = tuple(task_id for task_id, _panel in expected_panels)
    return {
        "schema_version": 1,
        "review_id": "REPLACE_WITH_REVIEW_ID",
        "reviewer_id": "REPLACE_WITH_REVIEWER_ID",
        "reviewed_at": "REPLACE_WITH_UTC_TIMESTAMP",
        "task_reviews": [
            {
                "task_id": task_id,
                "panel": panel,
                "task_sha256": sha256_json(task_map[task_id]),
                "no_deception_required": False,
                "policy_tool_compatible": False,
                "satisfiable": False,
                "rationale": "",
            }
            for task_id, panel in expected_panels
        ],
        "pairwise_reviews": [
            {
                "left_task_id": left,
                "right_task_id": right,
                "distinct_scenario": False,
                "rationale": "",
            }
            for left, right in combinations(task_ids, 2)
        ],
    }


def verify_reviewed_task_hashes(
    review: dict[str, Any],
    tasks: list[dict[str, Any]],
) -> None:
    """Refuse a review whose task IDs or per-task digests differ from pinned data."""

    from .manifest import sha256_json

    if not isinstance(tasks, list):
        raise TypeError("pinned task data must be a JSON array")
    task_map: dict[str, dict[str, Any]] = {}
    for task in tasks:
        if not isinstance(task, dict):
            raise TypeError("pinned task data must contain task objects")
        task_id = task.get("id")
        if not isinstance(task_id, str):
            raise TypeError("pinned task data contains a missing or invalid task ID")
        if not task_id.strip():
            raise ValueError("pinned task data contains an empty task ID")
        if task_id in task_map:
            raise ValueError("pinned task data contains duplicate task IDs")
        task_map[task_id] = task
    if not isinstance(review, dict) or not isinstance(review.get("task_reviews"), list):
        raise TypeError("task semantic review has no valid task_reviews array")
    for row in review["task_reviews"]:
        if not isinstance(row, dict) or not isinstance(row.get("task_id"), str):
            raise TypeError("task semantic review contains an invalid task review row")
        task_id = row["task_id"]
        task = task_map.get(task_id)
        if task is None or sha256_json(task) != row["task_sha256"]:
            raise ValueError(f"task semantic review is not bound to pinned task {task_id}")


def _normalize_expected_panels(
    expected_panels: Any,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(expected_panels, (list, tuple)):
        raise TypeError("expected task panels must be an ordered array")
    normalized = []
    for item in expected_panels:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("expected task panels must contain task/panel pairs")
        task_id, panel = item
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("expected task panels must contain unique non-empty Pilot task IDs")
        if not isinstance(panel, str) or panel not in {"evolution", "validation", "heldout"}:
            raise ValueError("expected task panels contain an unknown panel name")
        normalized.append((task_id, panel))
    task_ids = tuple(task_id for task_id, _panel in normalized)
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("expected task panels must contain unique non-empty Pilot task IDs")
    return tuple(normalized)


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"task review {label} must be non-empty text")
    return value.strip()


def _review_rationale(value: Any, label: str) -> str:
    rationale = _nonempty_text(value, label)
    if len(rationale) > 1000:
        raise ValueError(f"task review {label} exceeds the 1000-character limit")
    return rationale


def _validate_utc_timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("task review reviewed_at must be an ISO-8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("task review reviewed_at must be an ISO-8601 UTC timestamp ending in Z") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError("task review reviewed_at must be an ISO-8601 UTC timestamp ending in Z")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tau2-data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from .eligibility import validate_generalization_selection
        from .manifest import verify_git_blob_sha1
        from .phase0 import load_config

        config = load_config(args.config)
        experiment = config.get("experiment")
        if not isinstance(experiment, dict) or experiment.get("phase") != "4-pilot":
            raise ValueError("task review template requires phase='4-pilot'")
        if experiment.get("domain") != "retail":
            raise ValueError("task review template currently supports the frozen Retail domain only")
        selection = experiment.get("task_selection")
        if not isinstance(selection, dict):
            raise TypeError("Pilot task_selection must be a mapping")
        if selection.get("source_split") != "train" or selection.get("heldout_split") != "test":
            raise ValueError("Pilot task_selection must declare train E/V and test H sources")
        panels = ("evolution", "validation", "heldout")
        if any(not isinstance(selection.get(panel), list) for panel in panels):
            raise TypeError("Pilot E/V/H task selections must be JSON arrays")
        expected_panels = tuple(
            (task_id, panel)
            for panel in panels
            for task_id in selection[panel]
        )
        expected_panels = _normalize_expected_panels(expected_panels)

        data_root = args.tau2_data_dir.expanduser().resolve()
        task_path = data_root / "tau2/domains/retail/tasks.json"
        split_path = data_root / "tau2/domains/retail/split_tasks.json"
        source_blobs = experiment.get("source_blob_sha1")
        if not isinstance(source_blobs, dict):
            raise TypeError("Pilot source_blob_sha1 must be a mapping")
        verify_git_blob_sha1(
            task_path, source_blobs["data/tau2/domains/retail/tasks.json"],
        )
        verify_git_blob_sha1(
            split_path, source_blobs["data/tau2/domains/retail/split_tasks.json"],
        )
        tasks = json.loads(task_path.read_text(encoding="utf-8"))
        split = json.loads(split_path.read_text(encoding="utf-8"))
        validate_generalization_selection(
            tasks,
            split,
            evolution_task_ids=selection["evolution"],
            validation_task_ids=selection["validation"],
            heldout_task_ids=selection["heldout"],
            excluded_task_ids=selection.get("excluded", ()),
        )
        template = build_task_semantic_review_template(
            tasks, expected_panels=expected_panels,
        )
        target = args.output
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(template, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(json.dumps({
            "status": "unapproved_template_written",
            "output": str(target),
            "task_count": len(template["task_reviews"]),
            "pairwise_review_count": len(template["pairwise_reviews"]),
        }))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"Task review template generation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
