from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from evotau.records import EpisodeRecord, EpisodeStatus, EvidenceRef, FailureRecord
from evotau.study_analysis import (
    StudyRun,
    analyze_rq1_customer_advantage,
    load_rq1_document,
    main,
)


def episode(*, episode_id: str, task_id: str, seed: int, failure: bool) -> EpisodeRecord:
    if task_id == "task-a":
        policy, mistake, stage = (
            "retail.policy:identity_verification",
            "missing_identity_verification",
            "identity_verification",
        )
    else:
        policy, mistake, stage = (
            "retail.policy:complete_change_scope",
            "incomplete_write_scope",
            "pre_write",
        )
    return EpisodeRecord(
        episode_id=episode_id,
        task_id=task_id,
        seed=seed,
        customer_strategy_id="customer-strategy",
        service_strategy_id="service-strategy",
        status=EpisodeStatus.COMPLETE,
        task_success=not failure,
        customer_valid=True,
        strategy_applicable=True,
        customer_strategy_adherent=True,
        policy_violation=failure,
        policy_rule_id=policy if failure else None,
        mistake_type=mistake if failure else None,
        workflow_stage=stage if failure else None,
        evidence=(EvidenceRef(2, "tool", f"audited failure for {task_id}"),) if failure else (),
    )


def run(*, block: str, evolution_seed: int, condition: str, yield_count: int,
        budget: int = 100, tasks: tuple[str, ...] = ("task-a", "task-b")) -> StudyRun:
    episodes = tuple(
        episode(
            episode_id=f"{block}-{condition}-{task_id}",
            task_id=task_id,
            seed=evolution_seed + index,
            failure=index < yield_count,
        )
        for index, task_id in enumerate(tasks)
    )
    failures = tuple(_verified_failure(item) for item in episodes
                     if item.has_attributable_failure_candidate)
    reproductions = tuple(_reproduction_episode(item) for item in episodes
                          if item.has_attributable_failure_candidate)
    return StudyRun(
        run_id=f"{block}-{condition}",
        seed_block_id=block,
        evolution_seed=evolution_seed,
        condition=condition,
        task_ids=tasks,
        request_budget_cap=budget,
        provider_attempts=budget,
        episodes=episodes,
        verified_failures=failures,
        reproduction_episodes=reproductions,
    )


def complete_blocks(blocks: tuple[tuple[str, int, int, int, int], ...]) -> tuple[StudyRun, ...]:
    runs = []
    for block, seed, adaptive, static, random in blocks:
        runs.extend((
            run(block=block, evolution_seed=seed, condition="adaptive_customer", yield_count=adaptive),
            run(block=block, evolution_seed=seed, condition="static_customer", yield_count=static),
            run(block=block, evolution_seed=seed, condition="random_mutation", yield_count=random),
        ))
    return tuple(runs)


def _verified_failure(item: EpisodeRecord) -> FailureRecord:
    reproduction = _reproduction_episode(item)
    return FailureRecord.verify(
        item, reproduction_episode=reproduction, generation=0,
        verifier=f"audit:{item.episode_id}",
        reproduction_verifier=f"audit:{reproduction.episode_id}",
    )


def _reproduction_episode(item: EpisodeRecord) -> EpisodeRecord:
    return replace(item, episode_id=f"{item.episode_id}:replay", seed=item.seed + 10_000)


def test_rq1_uses_paired_independent_seed_blocks_and_verified_run_level_yield():
    runs = complete_blocks((
        ("seed-block-11", 11, 2, 1, 0),
        ("seed-block-22", 22, 2, 1, 1),
        ("seed-block-33", 33, 1, 1, 0),
    ))

    report = analyze_rq1_customer_advantage(runs, bootstrap_seed=9, bootstrap_replicates=500)
    repeated = analyze_rq1_customer_advantage(runs, bootstrap_seed=9, bootstrap_replicates=500)

    static = report.comparisons[0]
    random = report.comparisons[1]
    first_run = runs[0]
    assert first_run.metrics().attempted_episodes == len(first_run.episodes) + len(
        first_run.reproduction_episodes
    )
    assert first_run.metrics().reproduction_episodes == len(first_run.reproduction_episodes)
    with pytest.raises(ValueError, match="missing its strategy-level reproduction"):
        replace(first_run, reproduction_episodes=())
    assert report.status == "descriptive"
    assert static.baseline_condition == "static_customer"
    assert static.independent_seed_blocks == 3
    assert static.adaptive_wins == 2 and static.ties == 1 and static.adaptive_losses == 0
    assert static.mean_paired_yield_difference == pytest.approx(2 / 3)
    assert static.bootstrap_95_percentile_interval == repeated.comparisons[0].bootstrap_95_percentile_interval
    assert static.outcome == "higher_adaptive_yield_observed"
    assert random.baseline_condition == "random_mutation"
    assert random.mean_paired_yield_difference == pytest.approx(4 / 3)
    assert report.adaptive_runs[0].valid_episodes == 2
    assert report.adaptive_runs[0].verified_task_signature_yield == 2
    assert report.adaptive_runs[0].attributable_failure_rate == 1.0
    assert report.adaptive_runs[0].discovery_yield_per_100_requests == 2.0
    assert len(report.adaptive_runs[0].input_sha256) == 64


def test_rq1_does_not_bootstrap_episodes_or_claim_pilot_readiness_with_two_runs():
    runs = complete_blocks((
        ("seed-block-11", 11, 2, 1, 0),
        ("seed-block-22", 22, 2, 1, 1),
    ))

    report = analyze_rq1_customer_advantage(runs)

    assert report.status == "insufficient_independent_runs"
    assert report.comparisons[0].bootstrap_95_percentile_interval is None
    assert report.comparisons[0].outcome == "insufficient_independent_runs"
    assert "only 2 independent seed blocks" in report.reasons[0]


def test_rq1_rejects_unmatched_task_panels_budgets_and_reused_evolution_seeds():
    base = complete_blocks((
        ("seed-block-11", 11, 2, 1, 0),
        ("seed-block-22", 22, 2, 1, 0),
        ("seed-block-33", 33, 2, 1, 0),
    ))
    mismatched_panel = replace(
        base[1], task_ids=("task-a", "task-c"), episodes=(
            base[1].episodes[0],
            replace(base[1].episodes[1], task_id="task-c"),
        ), verified_failures=(),
    )
    with pytest.raises(ValueError, match="same ordered task panel"):
        analyze_rq1_customer_advantage((base[0], mismatched_panel, *base[2:]))

    mismatched_budget = replace(base[1], request_budget_cap=99, provider_attempts=99)
    with pytest.raises(ValueError, match="same request budget cap"):
        analyze_rq1_customer_advantage((base[0], mismatched_budget, *base[2:]))

    mismatched_schedule = replace(
        base[1], episodes=(base[1].episodes[0], replace(base[1].episodes[1], seed=999)),
    )
    with pytest.raises(ValueError, match="same task/seed schedule"):
        analyze_rq1_customer_advantage((base[0], mismatched_schedule, *base[2:]))

    reused_seed = tuple(replace(item, evolution_seed=11) if item.seed_block_id == "seed-block-22" else item
                        for item in base)
    with pytest.raises(ValueError, match="distinct evolution seeds"):
        analyze_rq1_customer_advantage(reused_seed)


def test_rq1_rejects_incomplete_condition_blocks_and_failure_evidence_mismatch():
    runs = complete_blocks((
        ("seed-block-11", 11, 2, 1, 0),
        ("seed-block-22", 22, 2, 1, 0),
        ("seed-block-33", 33, 2, 1, 0),
    ))
    with pytest.raises(ValueError, match="must contain all RQ1 conditions"):
        analyze_rq1_customer_advantage(runs[:-1])

    with pytest.raises(ValueError, match="not backed by a complete valid adherent episode"):
        StudyRun(
            run_id=runs[0].run_id,
            seed_block_id=runs[0].seed_block_id,
            evolution_seed=runs[0].evolution_seed,
            condition=runs[0].condition,
            task_ids=runs[0].task_ids,
            request_budget_cap=runs[0].request_budget_cap,
            provider_attempts=runs[0].provider_attempts,
            episodes=(replace(runs[0].episodes[0], customer_strategy_adherent=False),
                      runs[0].episodes[1]),
            verified_failures=runs[0].verified_failures,
        )


def test_rq1_json_cli_is_strict_hashes_input_and_never_overwrites_report(tmp_path):
    runs = complete_blocks((
        ("seed-block-11", 11, 2, 1, 0),
        ("seed-block-22", 22, 2, 1, 1),
        ("seed-block-33", 33, 1, 1, 0),
    ))
    document = {"schema_version": 2, "runs": [item.to_dict() for item in runs]}
    parsed = load_rq1_document(document)
    assert parsed == runs

    input_path = tmp_path / "study-runs.json"
    raw = json.dumps(document, sort_keys=True).encode("utf-8")
    input_path.write_bytes(raw)
    output_path = tmp_path / "rq1-report.json"
    assert main([
        "--input", str(input_path), "--output", str(output_path),
        "--bootstrap-seed", "4", "--bootstrap-replicates", "200",
    ]) == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["input_sha256"] == hashlib.sha256(raw).hexdigest()
    assert report["analysis"]["status"] == "descriptive"
    assert len(report["analysis"]["comparisons"]) == 2
    saved = output_path.read_bytes()
    with pytest.raises(SystemExit):
        main(["--input", str(input_path), "--output", str(output_path)])
    assert output_path.read_bytes() == saved

    with pytest.raises(ValueError, match="missing or unknown fields"):
        load_rq1_document({"schema_version": 2, "runs": [runs[0].to_dict() | {"extra": 1}]})
    with pytest.raises(ValueError, match="unsupported RQ1 input schema_version"):
        load_rq1_document({"schema_version": 1, "runs": [item.to_dict() for item in runs]})


def test_rq1_input_hash_covers_all_reproduction_audit_fields():
    run = complete_blocks((
        ("seed-block-11", 11, 2, 1, 0),
        ("seed-block-22", 22, 2, 1, 0),
        ("seed-block-33", 33, 2, 1, 0),
    ))[0]
    reproduction = run.reproduction_episodes[0]
    changed = replace(reproduction, notes="additional audited replay note")
    changed_run = replace(run, reproduction_episodes=(changed, *run.reproduction_episodes[1:]))
    assert changed_run.input_sha256 != run.input_sha256
