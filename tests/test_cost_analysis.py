from __future__ import annotations

import hashlib
import json

import pytest

from evotau.cost_analysis import analyze_cost_profile, main


def _model(model_id: str, *, attempts: int = 0, successes: int = 0,
           failures: int = 0, prompt: int = 0, completion: int = 0,
           usage_responses: int = 0, usage_unavailable: int = 0,
           cache_hits: int = 0) -> dict:
    return {
        "model_id": model_id,
        "attempts": attempts,
        "successes": successes,
        "failures": failures,
        "denied": 0,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "usage_responses": usage_responses,
        "usage_unavailable": usage_unavailable,
        "cache_hits": cache_hits,
    }


def _snapshot(*, attempts: int = 0, successes: int = 0, failures: int = 0,
              prompt: int = 0, completion: int = 0, usage_responses: int = 0,
              usage_unavailable: int = 0, cache_hits: int = 0,
              models: list[dict] | None = None) -> dict:
    return {
        "cap": 100,
        "attempts": attempts,
        "successes": successes,
        "failures": failures,
        "denied": 0,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "usage_responses": usage_responses,
        "usage_unavailable": usage_unavailable,
        "cache_hits": cache_hits,
        "model_usage": [] if models is None else models,
    }


def _input() -> dict:
    zero = _snapshot()
    one = _snapshot(
        attempts=1, successes=1, prompt=100, completion=20, usage_responses=1,
        models=[_model("model-a", attempts=1, successes=1, prompt=100,
                       completion=20, usage_responses=1)],
    )
    two = _snapshot(
        attempts=2, successes=2, prompt=300, completion=30, usage_responses=2,
        cache_hits=1,
        models=[_model("model-a", attempts=2, successes=2, prompt=300,
                       completion=30, usage_responses=2, cache_hits=1)],
    )
    return {
        "schema_version": 1,
        "profile_id": "pilot-cost-v1",
        "currency": "USD",
        "pricing_effective_at_utc": "2026-09-29T00:00:00Z",
        "pricing_source_sha256": "a" * 64,
        "rates": [{
            "model_id": "model-a",
            "prompt_cost_per_million_tokens": 1.0,
            "completion_cost_per_million_tokens": 2.0,
        }],
        "runs": [{
            "run_id": "adaptive-seed-11",
            "condition": "adaptive",
            "evolution_seed": 11,
            "valid_episode_count": 1,
            "verified_failure_count": 1,
            "episodes": [
                {"episode_id": "ep-1", "valid": True, "verified_failure": True,
                 "budget_before": zero, "budget_after": one},
                {"episode_id": "ep-2", "valid": False, "verified_failure": False,
                 "budget_before": one, "budget_after": two},
            ],
        }],
    }


def test_cost_profile_reports_descriptive_stats_and_conservative_incomplete_cost():
    source = _input()
    report = analyze_cost_profile(source, input_sha256="b" * 64)

    assert report["status"] == "incomplete_cost_coverage"
    assert report["source_input_sha256"] == "b" * 64
    assert report["runs"][0]["valid_episode_count"] == 1
    summary = report["condition_summaries"][0]
    assert summary["valid_episode_rate"] == 0.5
    assert summary["verified_failure_yield_per_valid_episode"] == 1.0
    assert summary["provider_attempts_per_episode"] == {"mean": 1, "p95": 1}
    assert summary["reported_total_tokens_per_episode"] == {"mean": 165, "p95": 210}
    # The cached episode's aggregate model tokens are omitted because the
    # telemetry cannot distinguish cached from non-cached token quantities.
    assert report["episodes"][0]["estimated_cost_lower_bound"] == pytest.approx(0.00014)
    assert report["episodes"][1]["estimated_cost_lower_bound"] == 0
    assert report["episodes"][1]["cost_coverage_complete"] is False
    assert any("cache-hit" in item for item in report["episodes"][1]["cost_coverage_reasons"])


def test_cost_profile_uses_episode_local_budget_delta_when_runs_overlap():
    source = _input()
    episode = source["runs"][0]["episodes"][0]
    # A second episode completed between this episode's global snapshots.
    episode["budget_after"] = _snapshot(
        attempts=2, successes=2, prompt=300, completion=30, usage_responses=2,
        models=[_model("model-a", attempts=2, successes=2, prompt=300,
                       completion=30, usage_responses=2)],
    )
    episode["episode_budget_delta"] = episode["budget_before"] | {
        "attempts": 1, "successes": 1, "prompt_tokens": 100,
        "completion_tokens": 20, "usage_responses": 1,
        "model_usage": [_model("model-a", attempts=1, successes=1, prompt=100,
                                completion=20, usage_responses=1)],
    }

    report = analyze_cost_profile(source, input_sha256="c" * 64)
    assert report["episodes"][0]["provider_attempts"] == 1
    assert report["episodes"][0]["reported_total_tokens"] == 120


def test_cost_profile_requires_episode_level_denominator_evidence():
    source = _input()
    source["runs"][0]["episodes"][1]["valid"] = True
    with pytest.raises(ValueError, match="valid_episode_count"):
        analyze_cost_profile(source, input_sha256="b" * 64)

    source = _input()
    source["runs"][0]["episodes"][1]["verified_failure"] = True
    with pytest.raises(ValueError, match="invalid episode"):
        analyze_cost_profile(source, input_sha256="b" * 64)


def test_cost_profile_rejects_non_monotonic_or_mismatched_model_snapshots():
    source = _input()
    after = source["runs"][0]["episodes"][1]["budget_after"]
    after.update({"prompt_tokens": 99, "completion_tokens": 20})
    after["model_usage"][0]["prompt_tokens"] = 99
    after["model_usage"][0]["completion_tokens"] = 20
    with pytest.raises(ValueError, match="monotonic"):
        analyze_cost_profile(source, input_sha256="b" * 64)

    source = _input()
    source["runs"][0]["episodes"][0]["budget_after"]["model_usage"][0]["prompt_tokens"] = 101
    with pytest.raises(ValueError, match="provider-budget snapshot is malformed"):
        analyze_cost_profile(source, input_sha256="b" * 64)


def test_cost_profile_cli_hashes_exact_input_and_never_overwrites(tmp_path):
    input_path = tmp_path / "cost-input.json"
    output_path = tmp_path / "cost-report.json"
    raw = json.dumps(_input(), separators=(",", ":")).encode()
    input_path.write_bytes(raw)

    assert main(["--input", str(input_path), "--output", str(output_path)]) == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["source_input_sha256"] == hashlib.sha256(raw).hexdigest()
    with pytest.raises(SystemExit):
        main(["--input", str(input_path), "--output", str(output_path)])
