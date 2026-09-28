from __future__ import annotations

import hashlib
import json

import pytest
from test_preregistration import formal_plan, power_result, sha

from evotau.formal_analysis import (
    _holm_adjust,
    _student_t_cdf,
    analyze_formal_reports,
    main,
)


def report_bundle(plan):
    seeds = plan["independent_evolution_seeds"]
    cap = plan["condition_budget_caps"]["adaptive_customer"]
    rq1_runs = {name: [] for name in (
        "adaptive_customer", "static_customer", "random_mutation",
    )}
    for index, seed in enumerate(seeds):
        for condition, base in (
            ("adaptive_customer", 8), ("static_customer", 2), ("random_mutation", 3),
        ):
            rq1_runs[condition].append({
                "run_id": f"{condition}-{seed}",
                "seed_block_id": f"block-{seed}",
                "evolution_seed": seed,
                "condition": condition,
                "input_sha256": sha(f"rq1-{condition}-{seed}"),
                "request_budget_cap": cap,
                "provider_attempts": cap - index,
                "verified_task_signature_yield": base + index,
            })
    rq1 = {
        "status": "descriptive",
        "task_ids": plan["task_panels"]["E"],
        "request_budget_cap": cap,
        "adaptive_runs": rq1_runs["adaptive_customer"],
        "static_runs": rq1_runs["static_customer"],
        "random_runs": rq1_runs["random_mutation"],
    }

    rq2_run_ids = [f"rq2-{seed}" for seed in seeds]
    rq2 = {
        "status": "descriptive",
        "evolution_task_ids": plan["task_panels"]["E"],
        "validation_task_ids": plan["task_panels"]["V"],
        "heldout_task_ids": plan["task_panels"]["H"],
        "request_budget_cap": cap,
        "run_ids_and_input_sha256": [
            [run_id, seed, sha(f"rq2-{seed}")]
            for run_id, seed in zip(rq2_run_ids, seeds, strict=True)
        ],
        "actual_provider_attempts": [
            [run_id, cap - index]
            for index, run_id in enumerate(rq2_run_ids)
        ],
        "metrics": [],
    }
    endpoint_values = {
        "target_failure_rate_reduction": [0.2 + index * 0.01 for index in range(len(seeds))],
        "clean_success_rate_change": [0.0] * len(seeds),
        "heldout_success_rate_change": [0.0] * len(seeds),
    }
    rq2["metrics"] = [
        {
            "metric": name,
            "status": "descriptive",
            "independent_runs": len(seeds),
            "complete_run_values": len(seeds),
            "observations": [
                [run_id, value]
                for run_id, value in zip(rq2_run_ids, values, strict=True)
            ],
        }
        for name, values in endpoint_values.items()
    ]

    rq3_runs = []
    for seed in seeds:
        for condition, result in (
            ("adaptive_coevolution", 1.0),
            ("frozen_service", 0.0),
            ("random_mutation", 0.0),
        ):
            rq3_runs.append({
                "run_id": f"rq3-{condition}-{seed}",
                "seed_block_id": f"block-{seed}",
                "evolution_seed": seed,
                "condition": condition,
                "input_sha256": sha(f"rq3-{condition}-{seed}"),
                "request_budget_cap": cap,
                "provider_attempts": cap - 1,
                "sustained_two_chain_response": result,
            })
    rq3 = {
        "status": "descriptive",
        "task_ids": plan["task_panels"]["E"],
        "runs": rq3_runs,
    }
    return {
        name: {"input_sha256": sha(f"{name}-source"), "analysis": analysis}
        for name, analysis in (("rq1", rq1), ("rq2", rq2), ("rq3", rq3))
    }


def analyze(plan, reports):
    plan_bytes = json.dumps(plan, sort_keys=True).encode()
    return analyze_formal_reports(
        plan=plan,
        plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
        rq1_report=reports["rq1"], rq1_report_sha256=sha("rq1-report"),
        rq2_report=reports["rq2"], rq2_report_sha256=sha("rq2-report"),
        rq3_report=reports["rq3"], rq3_report_sha256=sha("rq3-report"),
    )


def test_formal_inference_pairs_seed_blocks_and_adjusts_registered_family():
    plan = formal_plan()
    report = analyze(plan, report_bundle(plan))
    assert report.status == "formal_inference_computed"
    assert report.multiple_comparison_method == "holm"
    assert len(report.tests) == 7
    assert all(item.independent_seed_blocks == 6 for item in report.tests)
    assert all(item.reject_at_familywise_alpha for item in report.tests)
    assert report.tests[0].mean_paired_difference == 6.0
    assert report.tests[0].paired_differences == tuple(
        (seed, 6.0) for seed in plan["independent_evolution_seeds"]
    )
    json.dumps(report.to_dict(), allow_nan=False)


def test_registered_sign_flip_uses_exact_enumeration_and_preserves_registration():
    plan = formal_plan()
    hypothesis = plan["hypotheses"][0]
    hypothesis.update({
        "statistical_method": "paired_sign_flip",
        "permutation_seed": 712,
        "permutation_replicates": 9_999,
    })
    test = analyze(plan, report_bundle(plan)).tests[0]
    assert test.method == "paired_sign_flip"
    assert test.permutation_seed == 712
    assert test.permutation_replicates_registered == 9_999
    assert test.permutation_mode == "exact"
    assert test.permutation_replicates_used == 2 ** 6
    assert test.raw_p_value == pytest.approx(1 / 64)


def test_student_t_cdf_matches_reference_values():
    assert _student_t_cdf(1.0, 1) == pytest.approx(0.75, abs=1e-12)
    assert _student_t_cdf(1.0, 2) == pytest.approx(0.7886751345948129, abs=1e-12)
    assert _student_t_cdf(-1.0, 2) == pytest.approx(0.2113248654051871, abs=1e-12)


def test_holm_adjustment_is_monotone_and_clamped_within_family():
    intermediate = [
        ({"hypothesis_id": "h1", "multiplicity_family": "primary"}, [], None),
        ({"hypothesis_id": "h2", "multiplicity_family": "primary"}, [], None),
        ({"hypothesis_id": "h3", "multiplicity_family": "primary"}, [], None),
        ({"hypothesis_id": "h4", "multiplicity_family": "secondary"}, [], None),
    ]
    raw = [(1.0, 0.01, None, 0), (1.0, 0.02, None, 0),
           (1.0, 0.04, None, 0), (1.0, 0.9, None, 0)]
    assert _holm_adjust(intermediate, raw) == {
        "h1": pytest.approx(0.03),
        "h2": pytest.approx(0.04),
        "h3": pytest.approx(0.04),
        "h4": pytest.approx(0.9),
    }


def test_formal_inference_rejects_incomplete_seed_blocks_and_hashes():
    plan = formal_plan()
    reports = report_bundle(plan)
    reports["rq1"]["analysis"]["static_runs"].pop()
    with pytest.raises(ValueError, match="missing a preregistered evolution seed"):
        analyze(plan, reports)

    reports = report_bundle(plan)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        analyze_formal_reports(
            plan=plan, plan_sha256=sha("plan"),
            rq1_report=reports["rq1"], rq1_report_sha256="bad",
            rq2_report=reports["rq2"], rq2_report_sha256=sha("rq2-report"),
            rq3_report=reports["rq3"], rq3_report_sha256=sha("rq3-report"),
        )


def _write_reference(root, reference, content):
    path = root / reference["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    reference["sha256"] = hashlib.sha256(content).hexdigest()


def test_formal_analysis_cli_verifies_artifact_hashes_and_writes_once(tmp_path):
    plan = formal_plan()
    _write_reference(tmp_path, plan["shared_manifest_artifact"], b"manifest")
    _write_reference(tmp_path, plan["eligibility_review_artifact"], b"eligibility")
    for name, reference in plan["pilot_artifacts"].items():
        _write_reference(tmp_path, reference, name.encode())
    for item in plan["power_calculations"]:
        calculator = f"calculator-{item['hypothesis_id']}".encode()
        _write_reference(tmp_path, item["calculator_artifact"], calculator)
        rq = next(
            hypothesis["research_question"] for hypothesis in plan["hypotheses"]
            if hypothesis["hypothesis_id"] == item["hypothesis_id"]
        )
        pilot_name = {"RQ1": "rq1_report", "RQ2": "rq2_report", "RQ3": "rq3_report"}[rq]
        item["pilot_artifact_sha256"] = plan["pilot_artifacts"][pilot_name]["sha256"]
        calculation = json.dumps(power_result(plan, item), sort_keys=True).encode()
        _write_reference(tmp_path, item["calculation_artifact"], calculation)

    plan_path = tmp_path / "formal-plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True, indent=2), encoding="utf-8")
    reports = report_bundle(plan)
    report_paths = {}
    for name, document in reports.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(document, sort_keys=True, indent=2), encoding="utf-8")
        report_paths[name] = path
    output_path = tmp_path / "formal-analysis.json"
    args = [
        "--plan", str(plan_path), "--rq1-report", str(report_paths["rq1"]),
        "--rq2-report", str(report_paths["rq2"]), "--rq3-report", str(report_paths["rq3"]),
        "--output", str(output_path),
    ]
    assert main(args) == 0
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["status"] == "formal_inference_computed"
    assert result["registry_reference_unverified"] == plan["registry_url"]
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
