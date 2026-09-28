from __future__ import annotations

import hashlib
import json

import pytest

from evotau.calibration import (
    AttributionCalibrationCase,
    CalibrationStatus,
    CalibrationVerdict,
    calibrate_attribution,
    load_calibration_document,
    main,
)


def case(index, a, b, *, dispute=False):
    return AttributionCalibrationCase(
        case_id=f"cal-{index}",
        failure_id=f"failure-{index}",
        reviewer_a_id="reviewer-a",
        reviewer_a_ref=f"reviewer-a:{index}",
        reviewer_a_verdict=a,
        reviewer_b_id="reviewer-b",
        reviewer_b_ref=f"reviewer-b:{index}",
        reviewer_b_verdict=b,
        critical_fact_dispute=dispute,
    )


def test_attribution_calibration_waits_for_thirty_distinct_candidate_positives():
    cases = [
        case(index, CalibrationVerdict.CONFIRMED, CalibrationVerdict.CONFIRMED)
        for index in range(29)
    ]
    report = calibrate_attribution(cases)
    assert report.status == CalibrationStatus.INSUFFICIENT_EVIDENCE
    assert report.reviewed_candidate_positives == 29
    assert report.conservative_precision == 1.0


def test_calibration_uses_conservative_precision_and_frozen_boundary_values():
    cases = [
        *(case(index, CalibrationVerdict.CONFIRMED, CalibrationVerdict.CONFIRMED,
               dispute=index < 3) for index in range(27)),
        *(case(index, CalibrationVerdict.REJECTED, CalibrationVerdict.REJECTED)
          for index in range(27, 30)),
    ]
    report = calibrate_attribution(cases)
    assert report.status == CalibrationStatus.AUTOMATION_READY
    assert report.conservative_precision == 0.9
    assert report.critical_fact_dispute_rate == 0.1
    assert report.to_dict()["status"] == "automation_ready"


def test_calibration_pauses_below_precision_threshold_and_counts_uncertain_cases():
    cases = [
        *(case(index, CalibrationVerdict.CONFIRMED, CalibrationVerdict.CONFIRMED)
          for index in range(26)),
        *(case(index, CalibrationVerdict.REJECTED, CalibrationVerdict.REJECTED)
          for index in range(26, 29)),
        case(29, CalibrationVerdict.CONFIRMED, CalibrationVerdict.UNCERTAIN),
    ]
    report = calibrate_attribution(cases)
    assert report.status == CalibrationStatus.PAUSE_AUTOMATION
    assert report.confirmed_by_both == 26
    assert report.rejected_by_at_least_one == 3
    assert report.unresolved == 1
    assert report.conservative_precision == 26 / 30
    assert len(report.reasons) == 1


def test_calibration_pauses_when_critical_fact_disputes_exceed_ten_percent():
    cases = [
        case(index, CalibrationVerdict.CONFIRMED, CalibrationVerdict.CONFIRMED,
             dispute=index < 4)
        for index in range(30)
    ]
    report = calibrate_attribution(cases)
    assert report.status == CalibrationStatus.PAUSE_AUTOMATION
    assert report.conservative_precision == 1.0
    assert report.critical_fact_dispute_rate == 4 / 30
    assert "critical-fact dispute rate" in report.reasons[0]


def test_calibration_requires_two_distinct_reviewer_refs_and_unique_failures():
    with pytest.raises(ValueError, match="two distinct reviewer IDs"):
        AttributionCalibrationCase(
            case_id="cal", failure_id="failure",
            reviewer_a_id="same-reviewer", reviewer_a_ref="review-a",
            reviewer_a_verdict=CalibrationVerdict.CONFIRMED,
            reviewer_b_id="same-reviewer", reviewer_b_ref="review-b",
            reviewer_b_verdict=CalibrationVerdict.CONFIRMED, critical_fact_dispute=False,
        )
    with pytest.raises(ValueError, match="two distinct reviewer references"):
        AttributionCalibrationCase(
            case_id="cal", failure_id="failure",
            reviewer_a_id="reviewer-a", reviewer_a_ref="same",
            reviewer_a_verdict=CalibrationVerdict.CONFIRMED,
            reviewer_b_id="reviewer-b", reviewer_b_ref="same",
            reviewer_b_verdict=CalibrationVerdict.CONFIRMED, critical_fact_dispute=False,
        )
    duplicate_failure = case(0, CalibrationVerdict.CONFIRMED, CalibrationVerdict.CONFIRMED)
    duplicate_case = AttributionCalibrationCase(
        case_id="cal-1", failure_id=duplicate_failure.failure_id,
        reviewer_a_id="reviewer-a",
        reviewer_a_ref="reviewer-a:1", reviewer_a_verdict=CalibrationVerdict.CONFIRMED,
        reviewer_b_id="reviewer-b",
        reviewer_b_ref="reviewer-b:1", reviewer_b_verdict=CalibrationVerdict.CONFIRMED,
        critical_fact_dispute=False,
    )
    with pytest.raises(ValueError, match="each candidate failure may appear only once"):
        calibrate_attribution([duplicate_failure, duplicate_case])


def test_calibration_cli_reads_strict_json_and_binds_report_to_input_hash(tmp_path, capsys):
    cases = [
        case(index, CalibrationVerdict.CONFIRMED, CalibrationVerdict.CONFIRMED).to_dict()
        for index in range(30)
    ]
    source = tmp_path / "human-audit.json"
    source.write_text(json.dumps({"schema_version": 1, "cases": cases}), encoding="utf-8")
    raw = source.read_bytes()

    assert main(["--input", str(source)]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["input_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["report"]["status"] == "automation_ready"
    with pytest.raises(ValueError, match="unknown fields"):
        load_calibration_document({
            "schema_version": 1,
            "cases": [{**cases[0], "private_free_text": "must not enter the schema"}],
        })
