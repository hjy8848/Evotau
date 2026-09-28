"""Conservative readiness checks for independent failure-attribution audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

MIN_CALIBRATION_CANDIDATES = 30
MIN_ATTRIBUTION_PRECISION = 0.90
MAX_CRITICAL_FACT_DISPUTE_RATE = 0.10


class CalibrationVerdict(StrEnum):
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"


class CalibrationStatus(StrEnum):
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    AUTOMATION_READY = "automation_ready"
    PAUSE_AUTOMATION = "pause_automation"


@dataclass(frozen=True, slots=True)
class AttributionCalibrationCase:
    """Two independent blinded judgments of one proposed attributable failure.

    The candidate is an internally promoted positive. A positive enters the
    conservative precision numerator only when both independent reviewers
    confirm it. Uncertain and disagreeing cases remain visible in the
    denominator instead of silently improving precision.
    """

    case_id: str
    failure_id: str
    reviewer_a_id: str
    reviewer_a_ref: str
    reviewer_a_verdict: CalibrationVerdict
    reviewer_b_id: str
    reviewer_b_ref: str
    reviewer_b_verdict: CalibrationVerdict
    critical_fact_dispute: bool

    def __post_init__(self) -> None:
        if (not isinstance(self.case_id, str) or not self.case_id.strip()
                or not isinstance(self.failure_id, str) or not self.failure_id.strip()):
            raise ValueError("calibration case and failure IDs must be non-empty")
        if (not isinstance(self.reviewer_a_id, str) or not self.reviewer_a_id.strip()
                or not isinstance(self.reviewer_b_id, str) or not self.reviewer_b_id.strip()
                or not isinstance(self.reviewer_a_ref, str) or not self.reviewer_a_ref.strip()
                or not isinstance(self.reviewer_b_ref, str) or not self.reviewer_b_ref.strip()):
            raise ValueError("both independent reviewer references are required")
        if self.reviewer_a_id == self.reviewer_b_id:
            raise ValueError("calibration requires two distinct reviewer IDs")
        if self.reviewer_a_ref == self.reviewer_b_ref:
            raise ValueError("calibration requires two distinct reviewer references")
        if not isinstance(self.reviewer_a_verdict, CalibrationVerdict):
            raise TypeError("reviewer A verdict must be a CalibrationVerdict")
        if not isinstance(self.reviewer_b_verdict, CalibrationVerdict):
            raise TypeError("reviewer B verdict must be a CalibrationVerdict")
        if type(self.critical_fact_dispute) is not bool:
            raise ValueError("critical_fact_dispute must be an explicit boolean")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AttributionCalibrationReport:
    status: CalibrationStatus
    reviewed_candidate_positives: int
    confirmed_by_both: int
    rejected_by_at_least_one: int
    unresolved: int
    reviewer_disagreements: int
    critical_fact_disputes: int
    conservative_precision: float | None
    critical_fact_dispute_rate: float | None
    minimum_required_candidates: int
    required_precision: float
    maximum_critical_fact_dispute_rate: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        result["reasons"] = list(self.reasons)
        return result


def calibrate_attribution(
    cases: tuple[AttributionCalibrationCase, ...] | list[AttributionCalibrationCase],
) -> AttributionCalibrationReport:
    """Apply the frozen 30-case attribution readiness rule from the plan.

    Calibration is intentionally conservative: both reviewers must confirm a
    candidate for it to count as a true positive, while every reviewed
    candidate remains in the precision denominator. The function reports
    readiness only; it never changes prior FailureRecords or fitness values.
    """

    sample = tuple(cases)
    if any(not isinstance(case, AttributionCalibrationCase) for case in sample):
        raise TypeError("calibration accepts only AttributionCalibrationCase records")
    case_ids = [case.case_id for case in sample]
    failure_ids = [case.failure_id for case in sample]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("calibration case IDs must be unique")
    if len(set(failure_ids)) != len(failure_ids):
        raise ValueError("each candidate failure may appear only once in calibration")

    confirmed = sum(
        case.reviewer_a_verdict == CalibrationVerdict.CONFIRMED
        and case.reviewer_b_verdict == CalibrationVerdict.CONFIRMED
        for case in sample
    )
    rejected = sum(
        case.reviewer_a_verdict == CalibrationVerdict.REJECTED
        or case.reviewer_b_verdict == CalibrationVerdict.REJECTED
        for case in sample
    )
    unresolved = len(sample) - confirmed - rejected
    disagreements = sum(case.reviewer_a_verdict != case.reviewer_b_verdict for case in sample)
    disputes = sum(case.critical_fact_dispute for case in sample)
    precision = confirmed / len(sample) if sample else None
    dispute_rate = disputes / len(sample) if sample else None

    reasons: list[str] = []
    if len(sample) < MIN_CALIBRATION_CANDIDATES:
        status = CalibrationStatus.INSUFFICIENT_EVIDENCE
        reasons.append(
            f"requires at least {MIN_CALIBRATION_CANDIDATES} independently reviewed candidate positives"
        )
    else:
        if precision is not None and precision < MIN_ATTRIBUTION_PRECISION:
            reasons.append(
                f"conservative attribution precision {precision:.3f} is below "
                f"{MIN_ATTRIBUTION_PRECISION:.2f}"
            )
        if dispute_rate is not None and dispute_rate > MAX_CRITICAL_FACT_DISPUTE_RATE:
            reasons.append(
                f"critical-fact dispute rate {dispute_rate:.3f} exceeds "
                f"{MAX_CRITICAL_FACT_DISPUTE_RATE:.2f}"
            )
        status = CalibrationStatus.PAUSE_AUTOMATION if reasons else CalibrationStatus.AUTOMATION_READY

    return AttributionCalibrationReport(
        status=status,
        reviewed_candidate_positives=len(sample),
        confirmed_by_both=confirmed,
        rejected_by_at_least_one=rejected,
        unresolved=unresolved,
        reviewer_disagreements=disagreements,
        critical_fact_disputes=disputes,
        conservative_precision=precision,
        critical_fact_dispute_rate=dispute_rate,
        minimum_required_candidates=MIN_CALIBRATION_CANDIDATES,
        required_precision=MIN_ATTRIBUTION_PRECISION,
        maximum_critical_fact_dispute_rate=MAX_CRITICAL_FACT_DISPUTE_RATE,
        reasons=tuple(reasons),
    )


def load_calibration_document(value: Any) -> tuple[AttributionCalibrationCase, ...]:
    """Parse the versioned JSON hand-review interchange format strictly."""

    if not isinstance(value, dict) or set(value) != {"schema_version", "cases"}:
        raise ValueError("calibration input must contain only schema_version and cases")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported attribution calibration schema_version")
    rows = value["cases"]
    if not isinstance(rows, list):
        raise TypeError("calibration cases must be a JSON array")
    required = {
        "case_id", "failure_id", "reviewer_a_ref", "reviewer_a_verdict",
        "reviewer_a_id", "reviewer_b_id", "reviewer_b_ref", "reviewer_b_verdict",
        "critical_fact_dispute",
    }
    cases: list[AttributionCalibrationCase] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError(f"calibration case {index} has missing or unknown fields")
        try:
            cases.append(AttributionCalibrationCase(
                case_id=row["case_id"],
                failure_id=row["failure_id"],
                reviewer_a_id=row["reviewer_a_id"],
                reviewer_a_ref=row["reviewer_a_ref"],
                reviewer_a_verdict=CalibrationVerdict(row["reviewer_a_verdict"]),
                reviewer_b_id=row["reviewer_b_id"],
                reviewer_b_ref=row["reviewer_b_ref"],
                reviewer_b_verdict=CalibrationVerdict(row["reviewer_b_verdict"]),
                critical_fact_dispute=row["critical_fact_dispute"],
            ))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid calibration case {index}: {exc}") from exc
    # Reuse the aggregation validator here so malformed/duplicated input fails
    # before a report is emitted.
    calibrate_attribution(cases)
    return tuple(cases)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate blinded double-review calibration for candidate failures."
    )
    parser.add_argument("--input", required=True, type=Path, help="version 1 calibration JSON")
    args = parser.parse_args(argv)
    try:
        raw = args.input.read_bytes()
        document = json.loads(raw.decode("utf-8"))
        cases = load_calibration_document(document)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    payload = {
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "report": calibrate_attribution(cases).to_dict(),
    }
    sys.stdout.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
