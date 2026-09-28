"""Fail-closed validation for a pilot-informed Formal study preregistration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .manifest import MECHANISM_ROLE_NAMES, freeze_role_model_args

REQUIRED_CONDITIONS = frozenset({
    "adaptive_customer",
    "static_customer",
    "random_mutation",
    "adaptive_coevolution",
    "frozen_service",
    "frozen_customer",
    "no_historical_replay",
    "one_shot_repair",
})
REQUIRED_ABLATIONS = frozenset({
    "frozen_customer",
    "frozen_service",
    "no_historical_replay",
    "random_instead_of_failure_conditioned_mutation",
})
REQUIRED_PILOT_ARTIFACTS = frozenset({
    "cost_profile", "rq1_report", "rq2_report", "rq3_report",
})
REQUIRED_MODEL_ROLES = frozenset(MECHANISM_ROLE_NAMES)
PRIMARY_CONTRASTS = {
    ("RQ1", "verified_task_signature_yield", "static_customer", "greater"),
    ("RQ1", "verified_task_signature_yield", "random_mutation", "greater"),
    ("RQ2", "target_failure_rate_reduction", "incumbent_service", "greater"),
    ("RQ2", "clean_success_rate_change", "incumbent_service", "non_inferior"),
    ("RQ2", "heldout_success_rate_change", "incumbent_service", "non_inferior"),
    ("RQ3", "sustained_two_chain_response", "frozen_service", "greater"),
    ("RQ3", "sustained_two_chain_response", "random_mutation", "greater"),
}
_POWER_INPUT_FIELDS = {
    "schema_version", "study_id", "hypothesis_id", "pilot_artifact_sha256",
    "pilot_seed_blocks", "alternative", "noninferiority_margin",
    "minimum_relevant_effect", "familywise_alpha", "primary_family_size",
    "target_power", "maximum_seed_blocks", "simulation_replicates", "simulation_seed",
    "statistical_method", "permutation_seed", "permutation_replicates",
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


@dataclass(frozen=True, slots=True)
class FormalPlanSummary:
    study_id: str
    registry_url: str
    shared_manifest_sha256: str
    eligibility_review_sha256: str
    independent_seed_blocks: int
    task_counts: tuple[tuple[str, int], ...]
    budget_caps: tuple[tuple[str, int], ...]
    primary_hypotheses: int
    pilot_artifacts: tuple[tuple[str, str], ...]
    plan_sha256: str
    status: str = "structurally_valid_registry_reference_unverified"
    local_artifacts_verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "registry_url": self.registry_url,
            "shared_manifest_sha256": self.shared_manifest_sha256,
            "eligibility_review_sha256": self.eligibility_review_sha256,
            "independent_seed_blocks": self.independent_seed_blocks,
            "task_counts": {key: value for key, value in self.task_counts},
            "budget_caps": {key: value for key, value in self.budget_caps},
            "primary_hypotheses": self.primary_hypotheses,
            "pilot_artifacts": {key: value for key, value in self.pilot_artifacts},
            "plan_sha256": self.plan_sha256,
            "status": self.status,
            "local_artifacts_verified": self.local_artifacts_verified,
        }


def validate_formal_preregistration(
    value: Any,
    *,
    exact_input_sha256: str,
    artifact_root: Path | None = None,
) -> FormalPlanSummary:
    """Validate the design freeze required before a Formal run.

    This checks that pilot-derived decisions and external registration evidence
    are present. It does not choose hypotheses, margins, sample sizes, or claim
    that the registry URL has been independently verified.
    """

    required = {
        "schema_version", "study_id", "registry_url", "registered_at_utc",
        "evo_tau_commit", "tau2_commit", "shared_manifest_artifact",
        "pilot_artifacts", "model_ids", "sampling_parameters", "task_panels",
        "eligibility_review_artifact", "independent_evolution_seeds",
        "condition_budget_caps", "hypotheses", "power_calculations",
        "familywise_alpha", "multiple_comparison_method", "noninferiority_margins",
        "ablations", "human_review_plan", "exclusion_rules", "missing_data_rule",
        "stopping_rule", "blinded_before_outcome_access",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Formal plan has missing or unknown top-level fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported Formal preregistration schema_version")
    _nonempty_string(value["study_id"], "study_id")
    _nonempty_string(value["registry_url"], "registry_url")
    parsed_url = urlparse(value["registry_url"])
    if parsed_url.scheme != "https" or not parsed_url.netloc:
        raise ValueError("registry_url must be an HTTPS registration record")
    registered_at = _parse_utc_timestamp(value["registered_at_utc"])
    if registered_at is None:
        raise ValueError("registered_at_utc is required")
    for name in ("evo_tau_commit", "tau2_commit"):
        if not isinstance(value[name], str) or not _GIT_COMMIT.fullmatch(value[name]):
            raise ValueError(f"{name} must be a full lowercase Git commit SHA")
    _require_sha256(exact_input_sha256, "exact_input_sha256")
    shared_manifest_hash = _verify_artifact_reference(
        value["shared_manifest_artifact"], "shared_manifest_artifact", artifact_root,
    )

    pilot = _mapping(value["pilot_artifacts"], "pilot_artifacts")
    if set(pilot) != REQUIRED_PILOT_ARTIFACTS:
        raise ValueError(f"pilot_artifacts must contain exactly {sorted(REQUIRED_PILOT_ARTIFACTS)}")
    pilot_hashes = {
        key: _verify_artifact_reference(item, f"pilot_artifacts.{key}", artifact_root)
        for key, item in pilot.items()
    }

    models = _mapping(value["model_ids"], "model_ids")
    if set(models) != REQUIRED_MODEL_ROLES:
        raise ValueError(f"model_ids must contain exactly {sorted(REQUIRED_MODEL_ROLES)}")
    for role, model_id in models.items():
        _nonempty_string(model_id, f"model_ids.{role}")
        if any(word in model_id.lower() for word in ("api_key", "token=", "password")):
            raise ValueError("model_ids must not contain credentials")
    sampling = _mapping(value["sampling_parameters"], "sampling_parameters")
    if set(sampling) != REQUIRED_MODEL_ROLES:
        raise ValueError("sampling_parameters must freeze each role's model settings")
    for role, params in sampling.items():
        params = _mapping(params, f"sampling_parameters.{role}")
        if not params:
            raise ValueError(f"sampling_parameters.{role} must not be empty")
    try:
        freeze_role_model_args(sampling, roles=MECHANISM_ROLE_NAMES)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sampling_parameters are invalid: {exc}") from exc

    panels = _mapping(value["task_panels"], "task_panels")
    if set(panels) != {"E", "V", "H"}:
        raise ValueError("Formal task_panels must contain exactly E, V, and H")
    normalized_panels: dict[str, tuple[str, ...]] = {}
    for name, raw_tasks in panels.items():
        tasks = _string_list(raw_tasks, f"task_panels.{name}")
        if not tasks or len(set(tasks)) != len(tasks):
            raise ValueError(f"task_panels.{name} must be non-empty and unique")
        normalized_panels[name] = tuple(tasks)
    all_tasks = tuple(task for tasks in normalized_panels.values() for task in tasks)
    if len(set(all_tasks)) != len(all_tasks):
        raise ValueError("Formal E, V, and H task panels must be disjoint")
    eligibility_hash = _verify_artifact_reference(
        value["eligibility_review_artifact"], "eligibility_review_artifact", artifact_root,
    )

    seeds = value["independent_evolution_seeds"]
    if (not isinstance(seeds, list) or len(seeds) < 3
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Formal study needs at least three unique non-negative evolution seeds")

    budget_caps = _mapping(value["condition_budget_caps"], "condition_budget_caps")
    if not REQUIRED_CONDITIONS <= set(budget_caps):
        missing = sorted(REQUIRED_CONDITIONS - set(budget_caps))
        raise ValueError(f"condition_budget_caps is missing planned conditions: {missing}")
    normalized_budgets: dict[str, int] = {}
    for condition, cap in budget_caps.items():
        _nonempty_string(condition, "condition_budget_caps key")
        if type(cap) is not int or cap <= 0:
            raise ValueError("every condition budget cap must be a positive integer")
        normalized_budgets[condition] = cap
    if len(set(normalized_budgets.values())) != 1:
        raise ValueError("Formal conditions must use the same frozen request-attempt cap")

    hypotheses = value["hypotheses"]
    if not isinstance(hypotheses, list) or not hypotheses:
        raise ValueError("hypotheses must be a non-empty JSON array")
    parsed_hypotheses: dict[str, tuple[tuple[str, str, str, str], bool, str, str, int | None, int | None]] = {}
    hypothesis_fields = {
        "hypothesis_id", "research_question", "endpoint", "contrast",
        "alternative", "primary", "multiplicity_family", "statistical_method",
        "permutation_seed", "permutation_replicates",
    }
    for index, item in enumerate(hypotheses):
        if not isinstance(item, dict) or set(item) != hypothesis_fields:
            raise ValueError(f"hypotheses[{index}] has missing or unknown fields")
        identifier = _nonempty_string(item["hypothesis_id"], f"hypotheses[{index}].hypothesis_id")
        if identifier in parsed_hypotheses:
            raise ValueError("hypothesis_id values must be unique")
        rq = item["research_question"]
        if rq not in {"RQ1", "RQ2", "RQ3"}:
            raise ValueError(f"hypotheses[{index}].research_question must be RQ1, RQ2, or RQ3")
        endpoint = _nonempty_string(item["endpoint"], f"hypotheses[{index}].endpoint")
        contrast = _nonempty_string(item["contrast"], f"hypotheses[{index}].contrast")
        alternative = item["alternative"]
        if alternative not in {"greater", "less", "non_inferior"}:
            raise ValueError("hypothesis alternative must be greater, less, or non_inferior")
        if type(item["primary"]) is not bool:
            raise ValueError("hypothesis primary must be an explicit boolean")
        family = _nonempty_string(item["multiplicity_family"], "multiplicity_family")
        method = item["statistical_method"]
        if method not in {"paired_t", "paired_sign_flip"}:
            raise ValueError("statistical_method must be paired_t or paired_sign_flip")
        seed, replicates = item["permutation_seed"], item["permutation_replicates"]
        if method == "paired_t":
            if seed is not None or replicates is not None:
                raise ValueError("paired_t hypotheses cannot include permutation parameters")
        elif (type(seed) is not int or seed < 0
              or type(replicates) is not int or replicates < 9_999):
            raise ValueError("paired_sign_flip requires a seed and at least 9,999 replicates")
        parsed_hypotheses[identifier] = (
            (rq, endpoint, contrast, alternative), item["primary"], family,
            method, seed, replicates,
        )

    primary_signatures = [
        signature for signature, primary, *_ in parsed_hypotheses.values() if primary
    ]
    primary_tuples = set(primary_signatures)
    if primary_tuples != PRIMARY_CONTRASTS or len(primary_signatures) != len(PRIMARY_CONTRASTS):
        missing = sorted(PRIMARY_CONTRASTS - primary_tuples)
        unexpected = sorted(primary_tuples - PRIMARY_CONTRASTS)
        raise ValueError(
            "primary hypotheses must contain each planned RQ contrast exactly once; "
            f"missing={missing}, unexpected={unexpected}"
        )
    primary_families = {
        family for _, primary, family, *_ in parsed_hypotheses.values() if primary
    }
    if len(primary_families) != 1:
        raise ValueError("all preregistered primary hypotheses must share one multiplicity family")
    if value["multiple_comparison_method"] != "holm":
        raise ValueError("Formal primary hypotheses must use the preregistered Holm adjustment")
    alpha = value["familywise_alpha"]
    if type(alpha) not in {int, float} or not 0 < alpha < 1:
        raise ValueError("familywise_alpha must be between 0 and 1")

    margins = _mapping(value["noninferiority_margins"], "noninferiority_margins")
    for endpoint in ("clean_success_rate_change", "heldout_success_rate_change"):
        margin = margins.get(endpoint)
        if type(margin) not in {int, float} or not 0 < margin <= 1:
            raise ValueError(f"noninferiority_margins must freeze a positive {endpoint} margin")
    if any(item[0][3] == "non_inferior" and item[0][1] not in margins
           for item in parsed_hypotheses.values()):
        raise ValueError("every non-inferiority hypothesis requires a predeclared margin")

    calculations = value["power_calculations"]
    if not isinstance(calculations, list):
        raise TypeError("power_calculations must be a JSON array")
    primary_ids = {
        identifier for identifier, (_, primary, *_) in parsed_hypotheses.items() if primary
    }
    family_size = len(primary_ids)
    calculation_fields = {
        "hypothesis_id", "pilot_artifact_sha256", "pilot_power_input_artifact",
        "calculator_artifact", "calculation_artifact", "target_power",
        "minimum_relevant_effect", "maximum_seed_blocks", "planned_seed_blocks",
    }
    seen_power: set[str] = set()
    required_power: dict[str, int] = {}
    for index, item in enumerate(calculations):
        if not isinstance(item, dict) or set(item) != calculation_fields:
            raise ValueError(f"power_calculations[{index}] has missing or unknown fields")
        identifier = item["hypothesis_id"]
        if identifier not in primary_ids:
            raise ValueError("power calculations must refer to a primary hypothesis")
        if identifier in seen_power:
            raise ValueError("each primary hypothesis requires exactly one power calculation")
        seen_power.add(identifier)
        hypothesis = next(row for row in hypotheses if row["hypothesis_id"] == identifier)
        pilot_key = f"{hypothesis['research_question'].lower()}_report"
        if item["pilot_artifact_sha256"] != pilot_hashes[pilot_key]:
            raise ValueError("power calculation must cite the Pilot report for its hypothesis")
        input_digest = _verify_artifact_reference(
            item["pilot_power_input_artifact"],
            f"power_calculations[{index}].pilot_power_input_artifact", artifact_root,
        )
        calculator_digest = _verify_artifact_reference(
            item["calculator_artifact"], f"power_calculations[{index}].calculator_artifact",
            artifact_root,
        )
        _verify_artifact_reference(
            item["calculation_artifact"], f"power_calculations[{index}].calculation_artifact",
            artifact_root,
        )
        target = item["target_power"]
        if type(target) not in {int, float} or not 0.5 < target < 1:
            raise ValueError("target_power must be between 0.5 and 1")
        planned = item["planned_seed_blocks"]
        if type(planned) is not int or planned < 3:
            raise ValueError("planned_seed_blocks must be at least three")
        minimum_effect = item["minimum_relevant_effect"]
        if type(minimum_effect) not in {int, float} or not math.isfinite(minimum_effect) or minimum_effect <= 0:
            raise ValueError("minimum_relevant_effect must be positive and finite")
        maximum_blocks = item["maximum_seed_blocks"]
        if type(maximum_blocks) is not int or maximum_blocks < planned:
            raise ValueError("maximum_seed_blocks must cover planned_seed_blocks")
        if artifact_root is not None:
            pilot_report = _read_artifact_json(
                value["pilot_artifacts"][pilot_key], artifact_root,
                f"pilot_artifacts.{pilot_key}",
            )
            from .formal_analysis import paired_differences_from_report

            pilot_differences = paired_differences_from_report(value, hypothesis, pilot_report)
            _verify_power_input_artifact(
                item["pilot_power_input_artifact"], artifact_root,
                expected={
                    "schema_version": 1,
                    "study_id": value["study_id"],
                    "hypothesis_id": identifier,
                    "pilot_artifact_sha256": item["pilot_artifact_sha256"],
                    "alternative": hypothesis["alternative"],
                    "noninferiority_margin": (
                        margins.get(hypothesis["endpoint"])
                        if hypothesis["alternative"] == "non_inferior" else None
                    ),
                    "minimum_relevant_effect": minimum_effect,
                    "familywise_alpha": alpha,
                    "primary_family_size": family_size,
                    "target_power": target,
                    "maximum_seed_blocks": maximum_blocks,
                    "statistical_method": hypothesis["statistical_method"],
                    "permutation_seed": hypothesis["permutation_seed"],
                    "permutation_replicates": hypothesis["permutation_replicates"],
                },
                expected_pilot_differences=pilot_differences,
            )
            _verify_power_calculation_result(
                item["calculation_artifact"], artifact_root,
                expected={
                    "study_id": value["study_id"],
                    "hypothesis_id": identifier,
                    "pilot_artifact_sha256": item["pilot_artifact_sha256"],
                    "power_input_sha256": input_digest,
                    "calculator_source_sha256": calculator_digest,
                    "statistical_method": hypothesis["statistical_method"],
                    "permutation_seed": hypothesis["permutation_seed"],
                    "permutation_replicates": hypothesis["permutation_replicates"],
                    "alternative": hypothesis["alternative"],
                    "noninferiority_margin": (
                        margins.get(hypothesis["endpoint"])
                        if hypothesis["alternative"] == "non_inferior" else None
                    ),
                    "target_power": target,
                    "minimum_relevant_effect_null_adjusted": minimum_effect,
                    "maximum_seed_blocks": maximum_blocks,
                    "planned_seed_blocks": planned,
                    "familywise_alpha": alpha,
                    "primary_family_size": family_size,
                    "conservative_per_hypothesis_alpha": alpha / family_size,
                },
            )
        required_power[identifier] = planned
    if seen_power != primary_ids:
        raise ValueError("every primary hypothesis needs one pilot-backed power calculation")
    planned_blocks = max(required_power.values())
    if len(seeds) < planned_blocks:
        raise ValueError("independent_evolution_seeds does not cover the largest preregistered sample size")

    ablations = _string_list(value["ablations"], "ablations")
    if not REQUIRED_ABLATIONS <= set(ablations):
        raise ValueError(f"ablations must include {sorted(REQUIRED_ABLATIONS)}")
    audit = _mapping(value["human_review_plan"], "human_review_plan")
    if set(audit) != {"reviewer_count", "calibration_candidate_count", "sample_seed", "blind_conditions"}:
        raise ValueError("human_review_plan has missing or unknown fields")
    if type(audit["reviewer_count"]) is not int or audit["reviewer_count"] < 2:
        raise ValueError("human review requires at least two independent reviewers")
    if type(audit["calibration_candidate_count"]) is not int or audit["calibration_candidate_count"] < 30:
        raise ValueError("human review plan must include at least 30 calibration candidates")
    if type(audit["sample_seed"]) is not int or audit["sample_seed"] < 0:
        raise ValueError("human review sample_seed must be non-negative")
    if type(audit["blind_conditions"]) is not bool or not audit["blind_conditions"]:
        raise ValueError("human review must blind condition labels")

    exclusions = _string_list(value["exclusion_rules"], "exclusion_rules")
    if not exclusions:
        raise ValueError("exclusion_rules must be frozen before Formal outcomes are accessed")
    for name in ("missing_data_rule", "stopping_rule"):
        _nonempty_string(value[name], name)
    if type(value["blinded_before_outcome_access"]) is not bool or not value["blinded_before_outcome_access"]:
        raise ValueError("blinded_before_outcome_access: preregistration must precede outcome access")

    return FormalPlanSummary(
        study_id=value["study_id"],
        registry_url=value["registry_url"],
        shared_manifest_sha256=shared_manifest_hash,
        eligibility_review_sha256=eligibility_hash,
        independent_seed_blocks=len(seeds),
        task_counts=tuple(sorted((name, len(tasks)) for name, tasks in normalized_panels.items())),
        budget_caps=tuple(sorted(normalized_budgets.items())),
        primary_hypotheses=len(primary_ids),
        pilot_artifacts=tuple(sorted(pilot_hashes.items())),
        plan_sha256=exact_input_sha256,
        local_artifacts_verified=artifact_root is not None,
    )


def _parse_utc_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError:
        return None
    if timestamp.tzinfo is None or timestamp.utcoffset() != UTC.utcoffset(timestamp):
        return None
    return timestamp


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a JSON object with string keys")
    return value


def _string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise TypeError(f"{name} must be an array of non-empty strings")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _verify_artifact_reference(value: Any, name: str, root: Path | None) -> str:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{name} must contain exactly path and sha256")
    path_value = _nonempty_string(value["path"], f"{name}.path")
    relative = Path(path_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{name}.path must stay within the preregistration artifact directory")
    digest = value["sha256"]
    _require_sha256(digest, f"{name}.sha256")
    if root is not None:
        resolved_root = root.resolve()
        resolved_path = (resolved_root / relative).resolve(strict=True)
        if resolved_root not in resolved_path.parents:
            raise ValueError(f"{name}.path resolves outside the preregistration artifact directory")
        actual = hashlib.sha256(resolved_path.read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"{name} artifact SHA-256 does not match the referenced file")
    return digest


def _verify_power_calculation_result(
    reference: dict[str, Any], root: Path, *, expected: dict[str, Any],
) -> None:
    relative = Path(reference["path"])
    path = (root.resolve() / relative).resolve(strict=True)
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("power calculation artifact must be a readable JSON result") from exc
    if not isinstance(result, dict):
        raise TypeError("power calculation artifact must be a JSON object")
    required = {
        "schema_version", "status", "power_input_sha256", "planned_power_lower_bound",
        *expected.keys(),
    }
    if not required <= set(result):
        raise ValueError("power calculation result is missing required preregistration fields")
    if type(result["schema_version"]) is not int or result["schema_version"] != 1:
        raise ValueError("unsupported power calculation result schema_version")
    if result["status"] != "target_power_reached":
        raise ValueError("power calculation did not reach its registered target power")
    _require_sha256(result["power_input_sha256"], "power_input_sha256")
    lower_bound = result["planned_power_lower_bound"]
    if type(lower_bound) not in {int, float} or not math.isfinite(lower_bound):
        raise ValueError("power calculation lower confidence bound must be finite")
    if lower_bound < expected["target_power"] or lower_bound > 1:
        raise ValueError("power calculation lower confidence bound is below target power")
    for name, required_value in expected.items():
        actual = result[name]
        if name == "conservative_per_hypothesis_alpha":
            if (type(actual) not in {int, float}
                    or not math.isclose(actual, required_value, rel_tol=1e-12, abs_tol=1e-15)):
                raise ValueError("power calculation multiplicity threshold differs from the registered family")
        elif name in {"target_power", "familywise_alpha"}:
            if (type(actual) not in {int, float}
                    or not math.isclose(actual, required_value, rel_tol=1e-12, abs_tol=1e-15)):
                raise ValueError(f"power calculation {name} differs from preregistration")
        elif name == "noninferiority_margin":
            if required_value is None:
                if actual is not None:
                    raise ValueError("power calculation includes an unregistered non-inferiority margin")
            elif (type(actual) not in {int, float}
                  or not math.isclose(actual, required_value, rel_tol=1e-12, abs_tol=1e-15)):
                raise ValueError("power calculation margin differs from preregistration")
        elif actual != required_value:
            raise ValueError(f"power calculation {name} differs from preregistration")


def _read_artifact_json(reference: dict[str, Any], root: Path, name: str) -> dict[str, Any]:
    resolved_root = root.resolve()
    path = (resolved_root / Path(reference["path"])).resolve(strict=True)
    if resolved_root not in path.parents:
        raise ValueError(f"{name}.path resolves outside the preregistration artifact directory")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be a readable JSON artifact") from exc
    if not isinstance(document, dict):
        raise TypeError(f"{name} must be a JSON object")
    return document


def _verify_power_input_artifact(
    reference: dict[str, Any],
    root: Path,
    *,
    expected: dict[str, Any],
    expected_pilot_differences: tuple[tuple[int, float], ...],
) -> None:
    value = _read_artifact_json(reference, root, "pilot_power_input_artifact")
    if set(value) != _POWER_INPUT_FIELDS:
        raise ValueError("Pilot power input has missing or unknown fields")
    for name, required_value in expected.items():
        actual = value[name]
        if name in {
            "minimum_relevant_effect", "familywise_alpha", "target_power",
            "noninferiority_margin",
        }:
            if required_value is None:
                if actual is not None:
                    raise ValueError(f"power input {name} differs from preregistration")
            elif (type(actual) not in {int, float} or not math.isfinite(actual)
                  or not math.isclose(actual, required_value, rel_tol=1e-12, abs_tol=1e-15)):
                raise ValueError(f"power input {name} differs from preregistration")
        elif actual != required_value:
            raise ValueError(f"power input {name} differs from preregistration")
    if type(value["primary_family_size"]) is not int or value["primary_family_size"] < 1:
        raise ValueError("power input primary_family_size must be a positive integer")
    if type(value["simulation_replicates"]) is not int or value["simulation_replicates"] < 1_000:
        raise ValueError("power input simulation_replicates must be at least 1,000")
    if type(value["simulation_seed"]) is not int or value["simulation_seed"] < 0:
        raise ValueError("power input simulation_seed must be a non-negative integer")
    rows = value["pilot_seed_blocks"]
    if not isinstance(rows, list) or len(rows) != len(expected_pilot_differences):
        raise ValueError("power input Pilot seed blocks do not match the source report denominator")
    if value["maximum_seed_blocks"] < len(rows):
        raise ValueError("power input maximum_seed_blocks must cover the Pilot denominator")
    for row, (expected_seed, expected_difference) in zip(rows, expected_pilot_differences, strict=True):
        if not isinstance(row, dict) or set(row) != {"evolution_seed", "difference"}:
            raise ValueError("power input Pilot seed-block row has missing or unknown fields")
        if row["evolution_seed"] != expected_seed or type(row["evolution_seed"]) is not int:
            raise ValueError("power input Pilot evolution seeds differ from the source report")
        difference = row["difference"]
        if (type(difference) not in {int, float} or not math.isfinite(difference)
                or not math.isclose(difference, expected_difference, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError("power input Pilot differences differ from the source report")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate pilot-informed Formal study freeze and preregistration evidence."
    )
    parser.add_argument("--input", required=True, type=Path, help="version 1 Formal plan JSON")
    parser.add_argument("--output", type=Path, help="write a new report; stdout by default")
    args = parser.parse_args(argv)
    try:
        raw = args.input.read_bytes()
        document = json.loads(raw.decode("utf-8"))
        summary = validate_formal_preregistration(
            document, exact_input_sha256=hashlib.sha256(raw).hexdigest(),
            artifact_root=args.input.resolve().parent,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    payload = json.dumps(summary.to_dict(), sort_keys=True, indent=2) + "\n"
    if args.output:
        try:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(payload)
        except OSError as exc:
            parser.error(str(exc))
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
