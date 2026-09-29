"""Strict, task-bound records for the Pilot's required ex-ante human review."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from itertools import combinations
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

    if not isinstance(expected_panels, (list, tuple)):
        raise TypeError("expected task panels must be an ordered array")
    normalized_expected_panels = []
    for item in expected_panels:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("expected task panels must contain task/panel pairs")
        task_id, panel = item
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("expected task panels must contain unique non-empty Pilot task IDs")
        if not isinstance(panel, str) or panel not in {"evolution", "validation", "heldout"}:
            raise ValueError("expected task panels contain an unknown panel name")
        normalized_expected_panels.append((task_id, panel))
    expected_panels = tuple(normalized_expected_panels)
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
