"""Descriptive cost profiling from immutable per-episode provider telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from .budget import BudgetSnapshot

_INPUT_FIELDS = {
    "schema_version", "profile_id", "currency", "pricing_effective_at_utc",
    "pricing_source_sha256", "rates", "runs",
}
_RATE_FIELDS = {
    "model_id", "prompt_cost_per_million_tokens", "completion_cost_per_million_tokens",
}
_RUN_FIELDS = {
    "run_id", "condition", "evolution_seed", "valid_episode_count",
    "verified_failure_count", "episodes",
}
_EPISODE_FIELDS = {
    "episode_id", "valid", "verified_failure", "budget_before", "budget_after",
}
_EPISODE_FIELDS_WITH_LOCAL_USAGE = _EPISODE_FIELDS | {"episode_budget_delta"}
_COUNTER_FIELDS = (
    "attempts", "successes", "failures", "denied", "prompt_tokens",
    "completion_tokens", "usage_responses", "usage_unavailable", "cache_hits",
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def analyze_cost_profile(value: Any, *, input_sha256: str) -> dict[str, Any]:
    """Summarize requests, reported tokens, and price-schedule estimates.

    The calculation uses only the explicitly supplied frozen prices. Any
    unpriced model, missing token report, legacy unattributed model, or cache
    hit makes cost coverage incomplete; covered spend is retained as a lower
    bound and is never presented as a complete invoice total.
    """

    _require_sha256(input_sha256, "input_sha256")
    if not isinstance(value, dict) or set(value) != _INPUT_FIELDS:
        raise ValueError("cost-profile input has missing or unknown fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported cost-profile schema_version")
    profile_id = _nonempty(value["profile_id"], "profile_id")
    currency = _nonempty(value["currency"], "currency")
    if not re.fullmatch(r"[A-Z]{3}", currency):
        raise ValueError("currency must be an uppercase three-letter code")
    _parse_utc(value["pricing_effective_at_utc"], "pricing_effective_at_utc")
    pricing_digest = _require_sha256(value["pricing_source_sha256"], "pricing_source_sha256")
    rates = _parse_rates(value["rates"])
    runs = value["runs"]
    if not isinstance(runs, list) or not runs:
        raise ValueError("cost-profile input requires at least one run")

    run_ids: set[str] = set()
    seeds_by_condition: dict[str, set[int]] = defaultdict(set)
    episode_ids: set[str] = set()
    episode_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for run in runs:
        if not isinstance(run, dict) or set(run) != _RUN_FIELDS:
            raise ValueError("cost-profile run has missing or unknown fields")
        run_id = _nonempty(run["run_id"], "run_id")
        condition = _nonempty(run["condition"], "condition")
        seed = run["evolution_seed"]
        if run_id in run_ids or type(seed) is not int or seed < 0:
            raise ValueError("cost-profile run IDs and evolution seeds must be valid and unique")
        run_ids.add(run_id)
        if seed in seeds_by_condition[condition]:
            raise ValueError("a cost-profile condition repeats an evolution seed")
        seeds_by_condition[condition].add(seed)
        rows = run["episodes"]
        if not isinstance(rows, list) or not rows:
            raise ValueError("each cost-profile run requires episode telemetry")
        valid_count = _nonnegative_int(run["valid_episode_count"], "valid_episode_count")
        failure_count = _nonnegative_int(run["verified_failure_count"], "verified_failure_count")
        if valid_count > len(rows) or failure_count > valid_count:
            raise ValueError("valid-episode and verified-failure counts exceed their denominators")
        run_episodes: list[dict[str, Any]] = []
        for episode in rows:
            if (not isinstance(episode, dict)
                    or (set(episode) != _EPISODE_FIELDS
                        and set(episode) != _EPISODE_FIELDS_WITH_LOCAL_USAGE)):
                raise ValueError("episode telemetry has missing or unknown fields")
            episode_id = _nonempty(episode["episode_id"], "episode_id")
            if episode_id in episode_ids:
                raise ValueError("episode IDs must be unique across the input")
            episode_ids.add(episode_id)
            valid = _boolean(episode["valid"], "episode.valid")
            verified_failure = _boolean(episode["verified_failure"], "episode.verified_failure")
            if verified_failure and not valid:
                raise ValueError("an invalid episode cannot count as a verified failure")
            before = _budget_snapshot(episode["budget_before"], "budget_before")
            after = _budget_snapshot(episode["budget_after"], "budget_after")
            usage = (
                _usage_from_snapshot(_budget_snapshot(
                    episode["episode_budget_delta"], "episode_budget_delta",
                ))
                if "episode_budget_delta" in episode
                else _episode_delta(before, after)
            )
            cost = _episode_cost(usage, rates)
            output_episode = {
                "episode_id": episode_id,
                "valid": valid,
                "verified_failure": verified_failure,
                "provider_attempts": usage["attempts"],
                "reported_prompt_tokens": usage["prompt_tokens"],
                "reported_completion_tokens": usage["completion_tokens"],
                "reported_total_tokens": usage["prompt_tokens"] + usage["completion_tokens"],
                "usage_responses": usage["usage_responses"],
                "usage_unavailable": usage["usage_unavailable"],
                "cache_hits": usage["cache_hits"],
                "estimated_cost_lower_bound": cost["lower_bound"],
                "cost_coverage_complete": cost["complete"],
                "cost_coverage_reasons": list(cost["reasons"]),
                "model_usage": usage["models"],
            }
            run_episodes.append(output_episode)
            episode_rows.append({
                "condition": condition,
                **output_episode,
            })
        if sum(item["valid"] for item in run_episodes) != valid_count:
            raise ValueError("run valid_episode_count does not match per-episode validity")
        if sum(item["verified_failure"] for item in run_episodes) != failure_count:
            raise ValueError("run verified_failure_count does not match per-episode evidence")
        run_attempts = sum(item["provider_attempts"] for item in run_episodes)
        run_tokens = sum(item["reported_total_tokens"] for item in run_episodes)
        run_lower_bound = round(sum(item["estimated_cost_lower_bound"] for item in run_episodes), 12)
        run_complete = all(item["cost_coverage_complete"] for item in run_episodes)
        run_rows.append({
            "run_id": run_id,
            "condition": condition,
            "evolution_seed": seed,
            "episode_count": len(run_episodes),
            "valid_episode_count": valid_count,
            "verified_failure_count": failure_count,
            "provider_attempts": run_attempts,
            "reported_tokens": run_tokens,
            "estimated_cost_lower_bound": run_lower_bound,
            "cost_coverage_complete": run_complete,
        })

    conditions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in episode_rows:
        conditions[row["condition"]].append(row)
    runs_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        runs_by_condition[row["condition"]].append(row)

    summaries = []
    for condition in sorted(conditions):
        rows = conditions[condition]
        condition_runs = runs_by_condition[condition]
        attempts = [row["provider_attempts"] for row in rows]
        prompt = [row["reported_prompt_tokens"] for row in rows]
        completion = [row["reported_completion_tokens"] for row in rows]
        total = [row["reported_total_tokens"] for row in rows]
        known_cost = [row["estimated_cost_lower_bound"] for row in rows]
        valid_episodes = sum(run["valid_episode_count"] for run in condition_runs)
        total_episodes = len(rows)
        verified_failures = sum(run["verified_failure_count"] for run in condition_runs)
        coverage_complete = all(row["cost_coverage_complete"] for row in rows)
        summaries.append({
            "condition": condition,
            "independent_runs": len(condition_runs),
            "episodes": total_episodes,
            "valid_episode_rate": valid_episodes / total_episodes,
            "verified_failures": verified_failures,
            "verified_failure_yield_per_valid_episode": (
                verified_failures / valid_episodes if valid_episodes else None
            ),
            "provider_attempts_per_episode": _summary(attempts),
            "reported_prompt_tokens_per_episode": _summary(prompt),
            "reported_completion_tokens_per_episode": _summary(completion),
            "reported_total_tokens_per_episode": _summary(total),
            "estimated_cost_lower_bound_per_episode": _summary(known_cost),
            "estimated_cost_lower_bound_total": round(sum(known_cost), 12),
            "cost_coverage_complete": coverage_complete,
            "cost_coverage_reasons": sorted({
                reason for row in rows for reason in row["cost_coverage_reasons"]
            }),
        })

    complete = all(item["cost_coverage_complete"] for item in summaries)
    return {
        "schema_version": 1,
        "status": "complete" if complete else "incomplete_cost_coverage",
        "profile_id": profile_id,
        "source_input_sha256": input_sha256,
        "pricing_source_sha256": pricing_digest,
        "pricing_effective_at_utc": value["pricing_effective_at_utc"],
        "currency": currency,
        "condition_summaries": summaries,
        "runs": run_rows,
        "episodes": episode_rows,
    }


def _parse_rates(value: Any) -> dict[str, tuple[float, float]]:
    if not isinstance(value, list) or not value:
        raise ValueError("rates must be a non-empty JSON array")
    rates: dict[str, tuple[float, float]] = {}
    for item in value:
        if not isinstance(item, dict) or set(item) != _RATE_FIELDS:
            raise ValueError("price-rate row has missing or unknown fields")
        model = _nonempty(item["model_id"], "rate.model_id")
        if model in rates:
            raise ValueError("price schedule repeats a model ID")
        rates[model] = (
            _nonnegative_number(item["prompt_cost_per_million_tokens"], "prompt rate"),
            _nonnegative_number(item["completion_cost_per_million_tokens"], "completion rate"),
        )
    return rates


def _budget_snapshot(value: Any, name: str) -> BudgetSnapshot:
    legacy_fields = {
        "cap", "attempts", "successes", "failures", "denied", "prompt_tokens",
        "completion_tokens", "usage_responses", "usage_unavailable", "cache_hits",
        "model_usage",
    }
    current_fields = legacy_fields | {"in_flight", "reserved"}
    if (not isinstance(value, dict)
            or (set(value) != legacy_fields and set(value) != current_fields)):
        raise ValueError(f"{name} must contain a complete provider-budget snapshot")
    try:
        snapshot = BudgetSnapshot(**value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} provider-budget snapshot is malformed") from exc
    if type(snapshot.cap) is not int or snapshot.cap < 1:
        raise ValueError(f"{name} cap must be a positive integer")
    for field in _COUNTER_FIELDS:
        _nonnegative_int(getattr(snapshot, field), f"{name}.{field}")
    if snapshot.successes + snapshot.failures != snapshot.attempts:
        raise ValueError(f"{name} attempts do not equal successes plus failures")
    if snapshot.attempts > snapshot.cap:
        raise ValueError(f"{name} attempts exceed the provider budget cap")
    return snapshot


def _usage_from_snapshot(snapshot: BudgetSnapshot) -> dict[str, Any]:
    if snapshot.in_flight or snapshot.reserved:
        raise ValueError("episode budget delta cannot include active or reserved provider work")
    usage = {name: getattr(snapshot, name) for name in _COUNTER_FIELDS}
    if snapshot.model_usage:
        usage["models"] = [
            {"model_id": row.model_id, **{
                name: getattr(row, name) for name in _COUNTER_FIELDS
            }}
            for row in snapshot.model_usage
        ]
        for name in _COUNTER_FIELDS:
            if sum(item[name] for item in usage["models"]) != usage[name]:
                raise ValueError("per-model episode usage does not reconcile")
    elif any(usage.values()):
        usage["models"] = [{"model_id": "__unattributed__", **usage.copy()}]
    else:
        usage["models"] = []
    return usage


def _episode_delta(before: BudgetSnapshot, after: BudgetSnapshot) -> dict[str, Any]:
    if before.cap != after.cap:
        raise ValueError("episode budget cap changed during the episode")
    delta = {name: getattr(after, name) - getattr(before, name) for name in _COUNTER_FIELDS}
    if any(value < 0 for value in delta.values()):
        raise ValueError("episode provider-budget counters must be monotonic")
    before_models = {row.model_id: row for row in before.model_usage}
    after_models = {row.model_id: row for row in after.model_usage}
    if not before_models and not after_models:
        if any(delta.values()):
            delta["models"] = [{"model_id": "__unattributed__", **delta.copy()}]
        else:
            delta["models"] = []
    else:
        if before_models and not after_models:
            raise ValueError("per-model provider telemetry disappeared during an episode")
        model_rows = []
        for model in sorted(set(before_models) | set(after_models)):
            old, new = before_models.get(model), after_models.get(model)
            row_delta = {
                name: (getattr(new, name) if new else 0) - (getattr(old, name) if old else 0)
                for name in _COUNTER_FIELDS
            }
            if any(value < 0 for value in row_delta.values()):
                raise ValueError("per-model provider counters must be monotonic")
            if any(row_delta.values()):
                model_rows.append({"model_id": model, **row_delta})
        for name in _COUNTER_FIELDS:
            if sum(row[name] for row in model_rows) != delta[name]:
                raise ValueError("per-model provider deltas do not reconcile with the episode totals")
        delta["models"] = model_rows
    return delta


def _episode_cost(usage: dict[str, Any], rates: dict[str, tuple[float, float]]) -> dict[str, Any]:
    reasons: set[str] = set()
    if usage["usage_unavailable"]:
        reasons.add("provider token usage is missing for one or more attempts")
    if usage["cache_hits"]:
        reasons.add("cache hits require a separate frozen cache-price schedule")
    lower_bound = 0.0
    for row in usage["models"]:
        model = row["model_id"]
        if row["cache_hits"]:
            # Aggregate telemetry cannot identify cached-token quantities or
            # cache pricing, so this model's token subtotal is not safely priced.
            reasons.add(f"cache-hit token quantities are unavailable for model {model}")
            continue
        if model.startswith("__") or model not in rates:
            reasons.add(f"no frozen price rate is available for model {model}")
            continue
        prompt_rate, completion_rate = rates[model]
        lower_bound += (
            row["prompt_tokens"] * prompt_rate
            + row["completion_tokens"] * completion_rate
        ) / 1_000_000
    return {
        "lower_bound": round(lower_bound, 12),
        "complete": not reasons,
        "reasons": tuple(sorted(reasons)),
    }


def _summary(values: list[int | float]) -> dict[str, float | int]:
    return {"mean": mean(values), "p95": _nearest_rank(values, 0.95)}


def _nearest_rank(values: list[int | float], probability: float) -> int | float:
    if not values:
        raise ValueError("cannot summarize an empty value set")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _nonnegative_number(value: Any, name: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _parse_utc(value: Any, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{name} must be a UTC ISO-8601 timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid UTC ISO-8601 timestamp") from exc
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{name} must be UTC")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a descriptive provider-attempt, token, and cost profile."
    )
    parser.add_argument("--input", required=True, type=Path, help="version 1 cost-profile input JSON")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        raw = args.input.read_bytes()
        document = json.loads(raw.decode("utf-8"))
        report = analyze_cost_profile(document, input_sha256=hashlib.sha256(raw).hexdigest())
        rendered = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError,
            ValueError, ArithmeticError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
